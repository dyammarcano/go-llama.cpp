# VRAM layer-fit estimator — design

Date: 2026-05-29
Status: approved
Repo: `github.com/go-skynet/go-llama.cpp`
Roadmap: feature #2 of `docs/OLLAMA-PORTABLE-FEATURES.md` (consumes the `gguf` reader shipped as feature #1)

## Goal

Add a **pure-Go** estimator to the `gguf` package that, given a GGUF file + a VRAM
budget + context/batch parameters, computes how many transformer layers fit on
the GPU (`n_gpu_layers`) plus a memory breakdown — so callers stop guessing
`SetGPULayers(99)` and OOMing on small cards (e.g. a 4 GB GTX 1650).

Faithful port of Ollama's estimation model (`llm/server.go` `createLayout`/
`buildLayout`/`greedyFit` + `fs/ggml/ggml.go` `GraphSize`), adapted to consume
this repo's pure-Go `gguf` reader. Single-GPU.

## Scope

**Faithful-core (this spec):**
- Per-layer **weights** from `blk.N` tensor groups; output layer from
  `output` / `output_norm` / `token_embd`.
- **KV-cache** per layer (dense attention).
- **Graph/compute buffer** using the **Llama-family** `graphFullOffload` /
  `graphPartialOffload` formulas.
- **Single-GPU fit loop** with Ollama's reserve composition.

**Deferred (documented TODOs / fallbacks, NOT in this spec):**
- Per-architecture `GraphSize` beyond Llama-family → fall back to the Llama
  formula and set an `Approximate bool` flag on the result.
- SSM / recurrent (non-attention) layers → return a clear error.
- Multi-GPU balancing + binary-search backoff (single-GPU reduces to
  reserve-then-fill).
- Flash-attention graph deltas → accept the flag, document that it does not yet
  alter the formula.

## Non-goals

- No cgo, no build tags, no GPU querying. The caller supplies `FreeVRAM`.
- No llmark edits (llmark wiring is a separate later feature).
- No changes to the existing `gguf` reader public API (the estimator reads what
  it needs via `File.KeyValue` / `File.TensorInfos`).

## Package shape

New files in the existing `gguf` package (pure-Go, no build tags):
- `gguf/estimate.go` — `EstimateOptions`, `Estimate`, `EstimateLayers`, the fit loop, KV + weights math.
- `gguf/graph.go` — the Llama-family graph-size formula (isolated so more architectures can be added later).
- `gguf/estimate_test.go` — table tests.
- Fixture writer (`gguf/writer_test.go`) extended with a multi-block Llama GGUF builder.

### Public API

```go
// EstimateOptions parameterizes a layer-fit estimate.
type EstimateOptions struct {
    NumCtx        int    // context length in tokens (default 2048 if 0)
    BatchSize     int    // tokens per batch (default 512 if 0)
    NumParallel   int    // concurrent sequences (default 1 if 0)
    KVCacheType   string // "f16" (default), "q8_0", "q4_0", "f32"
    FreeVRAM      uint64 // caller-provided budget in bytes (required, >0)
    OverheadBytes uint64 // extra reserve added on top of the minimum reserve
    FlashAttention bool  // accepted; does not yet change the formula
}

// Estimate is the result of a layer-fit computation.
type Estimate struct {
    Layers         int      // recommended n_gpu_layers (repeating blocks offloaded)
    FullyOffloaded bool     // true if all blocks + the output layer fit
    TotalVRAM      uint64   // weights+kv+graph for the fully-offloaded case
    Weights        uint64   // sum of offloaded block weights (+ output if FullyOffloaded)
    KVCache        uint64   // KV bytes for the offloaded blocks
    Graph          uint64   // compute buffer (full- or partial-offload value)
    PerLayerBytes  []uint64 // weights+kv per block, index 0..BlockCount-1 (diagnostics)
    Approximate    bool     // true when a non-Llama arch used the fallback formula
}

// EstimateLayers opens path, reads metadata + tensor sizes, computes the
// estimate, and closes the file.
func EstimateLayers(path string, opts EstimateOptions) (*Estimate, error)
```

## Algorithm (faithful core)

All byte math uses `uint64`; intermediate quant scaling uses `float64` then
truncates (matching Ollama).

### 1. Weights per layer — `groupLayers`

Iterate `File.TensorInfos()`; bucket by the prefix before the second `.`:
- Tensors named `blk.<i>.*` → block group `i`; `blockWeights[i] = Σ TensorInfo.NumBytes()`.
- `output_norm.*`, `output.*`, else `token_embd.*` → `outputWeights` (the
  non-repeating head layer, offloaded last).
- Other non-repeating tensors (e.g. `token_embd` when also used as output) are
  counted once toward `outputWeights`, never per-block.

`BlockCount` comes from `<arch>.block_count`.

### 2. KV cache per layer

Per dense block:
```
ctx        = NumCtx * NumParallel
embd_k     = key_length   (<arch>.attention.key_length)  else embedding_length/head_count
embd_v     = value_length (<arch>.attention.value_length) else embedding_length/head_count
head_kv    = head_count_kv (<arch>.attention.head_count_kv) else head_count
bytesPerEl = f16:2  q8_0:1  q4_0:0.5  f32:4   (default 2)
kvPerLayer = uint64(float64(ctx*(embd_k+embd_v)*head_kv) * bytesPerEl)
```
(Per-layer head arrays — for models with per-layer head counts — are out of
scope; use the scalar metadata values.)

### 3. Graph / compute buffer — Llama-family (`graph.go`)

With `embedding`, `heads`, `embeddingHeads` (head dim), `headsKV`, `context =
NumCtx*NumParallel`, `batch = BatchSize`, `vocab`:
```
graphFullOffload = max(
    4*batch*(1 + 4*embedding + context*(1 + heads)),
    4*batch*(embedding + vocab),
)
graphPartialOffload = 4*batch*embedding + max(
    4*batch*(1 + embedding + max(context, embedding)) +
        embedding*embedding*9/16 +
        4*context*(batch*heads + embeddingHeads*headsKV),
    4*batch*(embedding + vocab) + embedding*vocab*105/128,
)
```
`vocab` from `len(tokenizer.ggml.tokens)`; `embeddingHeads` = `embedding/heads`
when `key_length` absent. Unknown architecture → use these formulas and set
`Estimate.Approximate = true`.

### 4. Single-GPU fit loop

```
minReserve = 457 MiB                         // Ollama DeviceInfo.MinimumMemory (non-Metal)
reserve    = minReserve + OverheadBytes
graph      = graphPartialOffload             // assume partial until proven full
available  = FreeVRAM - reserve - graph      // clamp at 0; if <=0 -> Layers=0

// fill repeating blocks while they fit.
// Dense-model blocks are near-identical in size, so fill order only matters
// under heterogeneity; we fill largest-cost-first to bias conservative (fewer
// fit). This differs from Ollama's reverse-by-index fill but yields the same
// count for uniform blocks.
order = block indices sorted by (blockWeights[i]+kv[i]) desc
used  = 0; n = 0
for i in order:
    cost = blockWeights[i] + kv[i]
    if used+cost <= available: used += cost; n++
    else: break
Layers = n

// all blocks fit -> try to also offload the output layer, and switch to full-offload graph
if n == BlockCount:
    if FreeVRAM - reserve - graphFullOffload - used - outputWeights >= 0:
        FullyOffloaded = true
        Graph = graphFullOffload
    else:
        Graph = graphPartialOffload
else:
    Graph = graphPartialOffload
```
Populate `Weights`, `KVCache`, `TotalVRAM`, `PerLayerBytes` accordingly.

Note: Ollama also subtracts a "first-layer buffer" (`blk.0 + kv[0]`) before
filling and runs a binary-search across GPUs for balance. For a single GPU the
reserve already covers the safety margin; we keep the simpler reserve-then-fill
and document the deviation. Bias is conservative (toward fewer layers).

## Metadata keys read (via `File.KeyValue`)

`general.architecture`, `<arch>.block_count`, `<arch>.embedding_length`,
`<arch>.attention.head_count`, `<arch>.attention.head_count_kv`,
`<arch>.attention.key_length`, `<arch>.attention.value_length`,
`len(tokenizer.ggml.tokens)`. Missing optional keys fall back as specified;
missing required keys (`block_count`, `embedding_length`) → error.

## Error handling

- `FreeVRAM == 0` → error (`budget required`).
- Recurrent/SSM architecture detected (no attention head metadata, SSM keys
  present) → error (`unsupported: recurrent architecture`).
- Unknown architecture with attention metadata → proceed with Llama formula,
  `Approximate = true` (not an error).
- `Open`/read errors propagated.

## Testing

- Extend `gguf/writer_test.go` with a builder for a multi-block Llama-arch GGUF
  (N `blk.i.*` tensors of known sizes + `output`/`token_embd` + the metadata
  keys above + a `tokenizer.ggml.tokens` array for vocab).
- `gguf/estimate_test.go` (table tests, `package gguf`):
  - exact `Weights`, `KVCache`, `Graph` for a known fixture (hand-computed).
  - `Layers` monotonic in `FreeVRAM`: tiny budget → 0; mid → partial; huge →
    `BlockCount` + `FullyOffloaded == true`.
  - KV scales with `NumCtx`, `NumParallel`, and `KVCacheType` (q8_0 = half f16).
  - `OverheadBytes` reduces `Layers`.
  - unknown arch → `Approximate == true`, still returns a sane estimate.
  - `FreeVRAM == 0` → error.
- Optional real-model check gated on `LLMARK_TEST_GGUF` (skipped if unset):
  assert `0 < Layers <= BlockCount` for a 4 GB-ish budget and log the breakdown.
- Run: `go test ./gguf/`. Pure-Go, no GPU, no build tags, no new deps.
- Must remain golangci-lint `default:all` clean (repo policy).

## Risks

- Graph formula is Llama-specific; other dense arches get an approximate value
  (flagged). Acceptable — the reserve margin absorbs error and bias is toward
  fewer layers (no OOM).
- Dropping Ollama's binary-search/first-layer-buffer slightly changes the exact
  layer count vs Ollama, but stays conservative. Documented.
