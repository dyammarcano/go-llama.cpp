# VRAM Layer-Fit Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-Go `gguf.EstimateLayers` that, given a GGUF + a VRAM budget + context params, computes how many transformer layers fit on the GPU plus a memory breakdown — a faithful Llama-family port of Ollama's estimator.

**Architecture:** Two new files in the existing `gguf` package: `graph.go` (Llama-family compute-buffer formula, isolated for future per-arch extension) and `estimate.go` (`EstimateOptions`/`Estimate`, `groupLayers`, KV math, single-GPU reserve-then-fill loop). Consumes the existing pure-Go reader (`File.TensorInfos`, `TensorInfo.NumBytes`, `File.KeyValue` arch-prefixing). No cgo, no build tags, no new deps.

**Tech Stack:** Go 1.25 stdlib only (`fmt`, `sort`, `strconv`, `strings`). Tests are `package gguf` (internal) reusing the `writeGGUF` fixture helper. Must stay golangci-lint `default:all` clean (repo policy).

**Spec:** `docs/superpowers/specs/2026-05-29-vram-estimator-design.md`
**Reference (read-only):** `C:\Users\dyamm\My Drive\acer\public_repos\ollama\fs\ggml\ggml.go` (`GraphSize`), `ollama\llm\server.go` (`buildLayout`/`greedyFit`).

---

## File Structure

```
gguf/
  graph.go          # NEW — llamaGraphSize(...) (full, partial) uint64
  graph_test.go     # NEW — exact literals for the Llama formula
  estimate.go       # NEW — EstimateOptions, Estimate, kvBytesPerElement, groupLayers, EstimateLayers
  estimate_test.go  # NEW — fit-loop + KV + error table tests
  writer_test.go    # MODIFY — add sampleLlamaModel(tb, n) fixture builder (reuses existing writeGGUF)
```

Attribution header on the two new non-test source files:
```go
// Derived from github.com/ollama/ollama (MIT License) memory-estimation logic.
// Adapted for github.com/go-skynet/go-llama.cpp.
```

Work from: `C:/Users/dyamm/My Drive/acer/public_repos/go-llama.cpp`. NO AI attribution in commits. Do NOT touch files outside `gguf/`. Do NOT run repo-wide gofmt/go fix.

---

## Task 1: Llama-family graph formula (`graph.go`)

Pure arithmetic, no GGUF dependency — easy to pin with exact literals.

**Files:**
- Create: `gguf/graph.go`
- Test: `gguf/graph_test.go`

- [ ] **Step 1: Write the failing test**

Create `gguf/graph_test.go`:
```go
package gguf

import "testing"

func TestLlamaGraphSize(t *testing.T) {
	// tiny dims chosen so the result is hand-computable:
	// embedding=64 heads=8 embeddingHeads=8 headsKV=8 context=16 batch=8 vocab=128
	full, partial := llamaGraphSize(64, 8, 8, 8, 16, 8, 128)

	// full = max(4*8*(1+4*64+16*(1+8)), 4*8*(64+128))
	//      = max(32*401, 32*192) = max(12832, 6144) = 12832
	if full != 12832 {
		t.Errorf("full = %d, want 12832", full)
	}
	// partial = 4*8*64 + max(
	//   4*8*(1+64+max(16,64)) + 64*64*9/16 + 4*16*(8*8 + 8*8),
	//   4*8*(64+128) + 64*128*105/128)
	// = 2048 + max(32*129 + 2304 + 64*128, 6144 + 6720)
	// = 2048 + max(4128+2304+8192, 12864) = 2048 + max(14624,12864) = 2048+14624 = 16672
	if partial != 16672 {
		t.Errorf("partial = %d, want 16672", partial)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Users/dyamm/My Drive/acer/public_repos/go-llama.cpp" && go test ./gguf/ -run TestLlamaGraphSize -v`
Expected: FAIL — `undefined: llamaGraphSize`.

- [ ] **Step 3: Write `graph.go`**

Create `gguf/graph.go`:
```go
// Derived from github.com/ollama/ollama (MIT License) memory-estimation logic.
// Adapted for github.com/go-skynet/go-llama.cpp.

package gguf

// llamaGraphSize returns the Llama-family compute-buffer sizes in bytes for the
// full-offload and partial-offload cases, following ollama/fs/ggml.go GraphSize.
//
// embedding = n_embd, heads = n_head, embeddingHeads = attention head dim,
// headsKV = n_head_kv, context = NumCtx*NumParallel, batch = BatchSize,
// vocab = vocabulary size. All values are token/element counts; the result is
// bytes (the 4* factors are f32 activation bytes, matching upstream).
func llamaGraphSize(embedding, heads, embeddingHeads, headsKV, context, batch, vocab uint64) (full, partial uint64) {
	full = max(
		4*batch*(1+4*embedding+context*(1+heads)),
		4*batch*(embedding+vocab),
	)

	partial = 4*batch*embedding + max(
		4*batch*(1+embedding+max(context, embedding))+
			embedding*embedding*9/16+
			4*context*(batch*heads+embeddingHeads*headsKV),
		4*batch*(embedding+vocab)+embedding*vocab*105/128,
	)
	return full, partial
}
```
(Go 1.21+ builtin `max` works on `uint64`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./gguf/ -run TestLlamaGraphSize -v`
Expected: PASS. If `full`/`partial` differ, re-derive the two literals from the formula above and update the test (the formula is the source of truth) — do NOT change `graph.go` to match a typo'd literal.

- [ ] **Step 5: Commit**
```bash
git add gguf/graph.go gguf/graph_test.go
git commit -m "feat(gguf): Llama-family compute-buffer (graph) size formula"
```

---

## Task 2: Multi-block Llama fixture builder (`writer_test.go`)

Adds a fixture so later tasks have a realistic GGUF. Uses the "large shape, empty data" trick: `TensorInfo.NumBytes()` derives from `Shape`, and the estimator never reads tensor *data*, so declaring big shapes with empty `Data` keeps the temp file tiny.

**Files:**
- Modify: `gguf/writer_test.go` (append; do not change existing `writeGGUF`/`sampleModel`)

- [ ] **Step 1: Add the builder + a smoke test**

Append to `gguf/writer_test.go`:
```go
// sampleLlamaModel builds an n-block llama-arch GGUF fixture with known sizes.
// Each block has one F32 weight tensor of shape [64,16] => 64*16*4 = 4096 bytes.
// The output tensor is [64,16] => 4096 bytes. token_embd is present (tied).
// Metadata uses tiny dims so graph/KV are hand-computable in estimate_test.go:
//   embedding=64 head_count=8 head_count_kv=8 key_length=8 value_length=8 vocab=128
// Tensor Data is left empty: NumBytes() comes from Shape, and the estimator
// never reads tensor data, so the file stays tiny.
func sampleLlamaModel(tb testing.TB, n int) string {
	tb.Helper()
	kvs := []kvPair{
		{"general.architecture", "llama"},
		{"general.name", "tiny-llama-fixture"},
		{"llama.block_count", uint32(n)},
		{"llama.embedding_length", uint32(64)},
		{"llama.attention.head_count", uint32(8)},
		{"llama.attention.head_count_kv", uint32(8)},
		{"llama.attention.key_length", uint32(8)},
		{"llama.attention.value_length", uint32(8)},
		{"llama.vocab_size", uint32(128)},
	}
	var tensors []testTensor
	for i := 0; i < n; i++ {
		tensors = append(tensors, testTensor{
			Name:  "blk." + strconv.Itoa(i) + ".attn.weight",
			Type:  0, // F32
			Shape: []uint64{64, 16},
			Data:  nil, // shape-only; estimator reads NumBytes() from Shape
		})
	}
	tensors = append(tensors,
		testTensor{Name: "token_embd.weight", Type: 0, Shape: []uint64{128, 64}, Data: nil},
		testTensor{Name: "output.weight", Type: 0, Shape: []uint64{64, 16}, Data: nil},
	)
	return writeGGUF(tb, kvs, tensors)
}

func TestSampleLlamaModelOpens(t *testing.T) {
	f, err := Open(sampleLlamaModel(t, 4))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()
	if f.NumTensors() != 6 { // 4 blocks + token_embd + output
		t.Errorf("NumTensors() = %d, want 6", f.NumTensors())
	}
	if got := f.KeyValue("block_count").Uint(); got != 4 {
		t.Errorf("block_count = %d, want 4", got)
	}
}
```
`writer_test.go` already imports `strconv`? It does not — it imports `bytes`, `encoding/binary`, `os`, `path/filepath`, `testing`. Add `"strconv"` to its import block.

- [ ] **Step 2: Run the smoke test**

Run: `go test ./gguf/ -run TestSampleLlamaModelOpens -v`
Expected: PASS (6 tensors, block_count 4). If `writeGGUF` rejects `Data: nil`, change those to `Data: []byte{}` — empty is fine since offsets are computed from `len(Data)`.

- [ ] **Step 3: Commit**
```bash
git add gguf/writer_test.go
git commit -m "test(gguf): multi-block llama fixture builder for estimator tests"
```

---

## Task 3: KV math + layer grouping (`estimate.go` part 1)

**Files:**
- Create: `gguf/estimate.go`
- Test: `gguf/estimate_test.go`

- [ ] **Step 1: Write failing tests for grouping + KV**

Create `gguf/estimate_test.go`:
```go
package gguf

import (
	"errors"
	"testing"
)

func TestGroupLayers(t *testing.T) {
	f, err := Open(sampleLlamaModel(t, 4))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	blocks, output := groupLayers(f)
	if len(blocks) != 4 {
		t.Fatalf("blocks = %d, want 4", len(blocks))
	}
	for i := 0; i < 4; i++ {
		if blocks[i] != 4096 { // [64,16] F32
			t.Errorf("blocks[%d] = %d, want 4096", i, blocks[i])
		}
	}
	if output != 4096 { // output.weight [64,16] F32
		t.Errorf("output = %d, want 4096", output)
	}
}

func TestKVBytesPerElement(t *testing.T) {
	cases := map[string]float64{"f16": 2, "": 2, "q8_0": 1, "q4_0": 0.5, "f32": 4}
	for in, want := range cases {
		if got := kvBytesPerElement(in); got != want {
			t.Errorf("kvBytesPerElement(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestEstimateRequiresBudget(t *testing.T) {
	_, err := EstimateLayers(sampleLlamaModel(t, 4), EstimateOptions{FreeVRAM: 0})
	if !errors.Is(err, ErrUnsupported) {
		t.Errorf("err = %v, want ErrUnsupported", err)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `go test ./gguf/ -run 'TestGroupLayers|TestKVBytesPerElement|TestEstimateRequiresBudget' -v`
Expected: FAIL — `undefined: groupLayers / kvBytesPerElement / EstimateLayers / EstimateOptions`.

- [ ] **Step 3: Write `estimate.go` (types, helpers, grouping)**

Create `gguf/estimate.go`:
```go
// Derived from github.com/ollama/ollama (MIT License) memory-estimation logic.
// Adapted for github.com/go-skynet/go-llama.cpp.

package gguf

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// DefaultMinReserveBytes mirrors Ollama's non-Metal DeviceInfo.MinimumMemory.
const DefaultMinReserveBytes uint64 = 457 << 20

// EstimateOptions parameterizes a layer-fit estimate.
type EstimateOptions struct {
	NumCtx          int    // context length in tokens (0 => 2048)
	BatchSize       int    // tokens per batch (0 => 512)
	NumParallel     int    // concurrent sequences (0 => 1)
	KVCacheType     string // "f16" (default), "q8_0", "q4_0", "f32"
	FreeVRAM        uint64 // caller-provided budget in bytes (required, > 0)
	OverheadBytes   uint64 // extra reserve on top of the minimum reserve
	MinReserveBytes uint64 // minimum reserve floor; 0 => DefaultMinReserveBytes
	FlashAttention  bool   // accepted; does not yet alter the formula
}

// Estimate is the result of a layer-fit computation.
type Estimate struct {
	Layers         int      // recommended n_gpu_layers (repeating blocks offloaded)
	FullyOffloaded bool     // true if all blocks + the output layer fit
	TotalVRAM      uint64   // weights+kv+graph actually placed on the GPU
	Weights        uint64   // offloaded block weights (+ output if FullyOffloaded)
	KVCache        uint64   // KV bytes for the offloaded blocks
	Graph          uint64   // compute buffer (full- or partial-offload value)
	PerLayerBytes  []uint64 // weights+kv per block, index 0..BlockCount-1
	Approximate    bool     // true when a non-llama arch used the fallback formula
}

// kvBytesPerElement returns the per-element KV-cache byte size for a cache type.
func kvBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "q8_0":
		return 1
	case "q4_0":
		return 0.5
	case "f32":
		return 4
	default: // "f16" and unknown
		return 2
	}
}

// groupLayers sums repeating blk.<i>.* tensor bytes per block and the
// non-repeating output layer (output[_norm], else tied token_embd).
func groupLayers(f *File) (blocks map[int]uint64, output uint64) {
	blocks = make(map[int]uint64)
	var outputW, tokenEmbd uint64
	for _, ti := range f.TensorInfos() {
		name := ti.Name
		switch {
		case strings.HasPrefix(name, "blk."):
			rest := name[len("blk."):]
			dot := strings.IndexByte(rest, '.')
			if dot < 0 {
				continue
			}
			idx, err := strconv.Atoi(rest[:dot])
			if err != nil {
				continue
			}
			blocks[idx] += uint64(ti.NumBytes())
		case strings.HasPrefix(name, "output_norm"), strings.HasPrefix(name, "output."):
			outputW += uint64(ti.NumBytes())
		case strings.HasPrefix(name, "token_embd"):
			tokenEmbd += uint64(ti.NumBytes())
		}
	}
	if outputW == 0 {
		outputW = tokenEmbd // tied embeddings
	}
	return blocks, outputW
}

func orDefault(v, def int) int {
	if v <= 0 {
		return def
	}
	return v
}
```

- [ ] **Step 4: Add a minimal `EstimateLayers` stub so the budget test compiles & passes**

Append to `gguf/estimate.go`:
```go
// EstimateLayers opens path, reads metadata + tensor sizes, computes the
// estimate, and closes the file. (Full fit logic added in the next task.)
func EstimateLayers(path string, opts EstimateOptions) (*Estimate, error) {
	if opts.FreeVRAM == 0 {
		return nil, fmt.Errorf("%w: FreeVRAM budget required", ErrUnsupported)
	}
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return &Estimate{}, nil
}
```

- [ ] **Step 5: Run the three tests**

Run: `go test ./gguf/ -run 'TestGroupLayers|TestKVBytesPerElement|TestEstimateRequiresBudget' -v`
Expected: PASS. `sort` is imported but not yet used → it will fail to compile. Remove `"sort"` from the import block for now; it is re-added in Task 4.

- [ ] **Step 6: Commit**
```bash
git add gguf/estimate.go gguf/estimate_test.go
git commit -m "feat(gguf): estimate types, KV-per-element, layer grouping"
```

---

## Task 4: Full `EstimateLayers` fit loop (`estimate.go` part 2)

**Files:**
- Modify: `gguf/estimate.go` (replace the stub `EstimateLayers`, re-add `"sort"`)
- Test: `gguf/estimate_test.go` (append fit tests)

- [ ] **Step 1: Write failing fit tests**

Append to `gguf/estimate_test.go`:
```go
// helper: compute the graph + reserve the estimator will use for the fixture,
// so fit assertions don't hard-code the graph literal.
func fixtureGraph(t *testing.T, n int, opts EstimateOptions) (partial, full, kvPerLayer, blockWeight uint64) {
	t.Helper()
	ctx := uint64(orDefault(opts.NumCtx, 2048)) * uint64(orDefault(opts.NumParallel, 1))
	batch := uint64(orDefault(opts.BatchSize, 512))
	// fixture dims: embedding=64 heads=8 headsKV=8 key=val=8 vocab=128
	full, partial = llamaGraphSize(64, 8, 8, 8, ctx, batch, 128)
	kvPerLayer = uint64(float64(ctx*(8+8)*8) * kvBytesPerElement(opts.KVCacheType))
	blockWeight = 4096
	return partial, full, kvPerLayer, blockWeight
}

func TestEstimateFullOffload(t *testing.T) {
	opts := EstimateOptions{FreeVRAM: 1 << 40, MinReserveBytes: 4096} // 1 TiB
	est, err := EstimateLayers(sampleLlamaModel(t, 4), opts)
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if est.Layers != 4 {
		t.Errorf("Layers = %d, want 4", est.Layers)
	}
	if !est.FullyOffloaded {
		t.Error("FullyOffloaded = false, want true")
	}
	if est.Approximate {
		t.Error("Approximate = true, want false (llama arch)")
	}
}

func TestEstimateNoFit(t *testing.T) {
	// budget below reserve+graph => 0 layers
	est, err := EstimateLayers(sampleLlamaModel(t, 4), EstimateOptions{FreeVRAM: 1024, MinReserveBytes: 4096})
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if est.Layers != 0 || est.FullyOffloaded {
		t.Errorf("Layers=%d FullyOffloaded=%v, want 0/false", est.Layers, est.FullyOffloaded)
	}
}

func TestEstimatePartialFit(t *testing.T) {
	opts := EstimateOptions{NumCtx: 2048, MinReserveBytes: 4096}
	partial, _, kv, w := fixtureGraph(t, 4, opts)
	perLayer := kv + w
	// budget = reserve + partial graph + exactly 2 layers + a little slack
	opts.FreeVRAM = 4096 + partial + 2*perLayer + 8
	est, err := EstimateLayers(sampleLlamaModel(t, 4), opts)
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if est.Layers != 2 {
		t.Errorf("Layers = %d, want 2 (perLayer=%d partial=%d)", est.Layers, perLayer, partial)
	}
	if est.FullyOffloaded {
		t.Error("FullyOffloaded = true, want false (partial)")
	}
	if est.KVCache != 2*kv {
		t.Errorf("KVCache = %d, want %d", est.KVCache, 2*kv)
	}
}

func TestEstimateKVQuantHalvesF16(t *testing.T) {
	base := EstimateOptions{NumCtx: 2048, MinReserveBytes: 4096, FreeVRAM: 1 << 40}
	_, _, kvF16, _ := fixtureGraph(t, 4, base)
	q := base
	q.KVCacheType = "q8_0"
	_, _, kvQ8, _ := fixtureGraph(t, 4, q)
	if kvQ8 != kvF16/2 {
		t.Errorf("q8_0 kv = %d, want half of f16 %d", kvQ8, kvF16)
	}
}

func TestEstimateApproximateForUnknownArch(t *testing.T) {
	// build a fixture with a non-llama architecture but llama-shaped metadata
	path := writeGGUF(t,
		[]kvPair{
			{"general.architecture", "mistral"},
			{"mistral.block_count", uint32(2)},
			{"mistral.embedding_length", uint32(64)},
			{"mistral.attention.head_count", uint32(8)},
			{"mistral.attention.head_count_kv", uint32(8)},
			{"mistral.vocab_size", uint32(128)},
		},
		[]testTensor{
			{Name: "blk.0.attn.weight", Type: 0, Shape: []uint64{64, 16}},
			{Name: "blk.1.attn.weight", Type: 0, Shape: []uint64{64, 16}},
			{Name: "output.weight", Type: 0, Shape: []uint64{64, 16}},
		},
	)
	est, err := EstimateLayers(path, EstimateOptions{FreeVRAM: 1 << 40, MinReserveBytes: 4096})
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if !est.Approximate {
		t.Error("Approximate = false, want true (non-llama arch)")
	}
	if est.Layers != 2 {
		t.Errorf("Layers = %d, want 2", est.Layers)
	}
}

func TestEstimateRecurrentUnsupported(t *testing.T) {
	// no attention.head_count => treated as recurrent/unsupported
	path := writeGGUF(t,
		[]kvPair{
			{"general.architecture", "mamba"},
			{"mamba.block_count", uint32(2)},
			{"mamba.embedding_length", uint32(64)},
		},
		[]testTensor{{Name: "blk.0.x.weight", Type: 0, Shape: []uint64{64, 16}}},
	)
	_, err := EstimateLayers(path, EstimateOptions{FreeVRAM: 1 << 40})
	if !errors.Is(err, ErrUnsupported) {
		t.Errorf("err = %v, want ErrUnsupported", err)
	}
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `go test ./gguf/ -run TestEstimate -v`
Expected: FAIL — the stub returns an empty `Estimate{}` (Layers 0, never FullyOffloaded), and the recurrent case currently returns no error.

- [ ] **Step 3: Replace the stub `EstimateLayers` with the full implementation**

In `gguf/estimate.go`, re-add `"sort"` to the imports, and replace the stub `EstimateLayers` with:
```go
// EstimateLayers opens path, reads metadata + tensor sizes, computes how many
// transformer blocks fit in opts.FreeVRAM, and closes the file. Single-GPU,
// Llama-family graph model; non-llama dense architectures get an approximate
// estimate (Estimate.Approximate). Recurrent/SSM architectures are unsupported.
func EstimateLayers(path string, opts EstimateOptions) (*Estimate, error) {
	if opts.FreeVRAM == 0 {
		return nil, fmt.Errorf("%w: FreeVRAM budget required", ErrUnsupported)
	}
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	arch := f.KeyValue("general.architecture").String()
	blockCount := int(f.KeyValue("block_count").Uint())
	embedding := f.KeyValue("embedding_length").Uint()
	if blockCount == 0 || embedding == 0 {
		return nil, fmt.Errorf("%w: missing block_count/embedding_length", ErrUnsupported)
	}
	heads := f.KeyValue("attention.head_count").Uint()
	if heads == 0 {
		return nil, fmt.Errorf("%w: recurrent or unsupported architecture (no attention heads)", ErrUnsupported)
	}
	headsKV := f.KeyValue("attention.head_count_kv").Uint()
	if headsKV == 0 {
		headsKV = heads
	}
	keyLen := f.KeyValue("attention.key_length").Uint()
	if keyLen == 0 {
		keyLen = embedding / heads
	}
	valLen := f.KeyValue("attention.value_length").Uint()
	if valLen == 0 {
		valLen = embedding / heads
	}
	vocab := f.KeyValue("vocab_size").Uint()
	if vocab == 0 {
		vocab = uint64(len(f.KeyValue("tokenizer.ggml.tokens").Strings()))
	}

	ctx := uint64(orDefault(opts.NumCtx, 2048)) * uint64(orDefault(opts.NumParallel, 1))
	batch := uint64(orDefault(opts.BatchSize, 512))

	blocks, output := groupLayers(f)
	kvPerLayer := uint64(float64(ctx*(keyLen+valLen)*headsKV) * kvBytesPerElement(opts.KVCacheType))
	full, partial := llamaGraphSize(embedding, heads, keyLen, headsKV, ctx, batch, vocab)

	perLayer := make([]uint64, blockCount)
	for i := 0; i < blockCount; i++ {
		perLayer[i] = blocks[i] + kvPerLayer
	}

	minReserve := opts.MinReserveBytes
	if minReserve == 0 {
		minReserve = DefaultMinReserveBytes
	}
	reserve := minReserve + opts.OverheadBytes

	est := &Estimate{PerLayerBytes: perLayer, Approximate: arch != "llama"}

	// fill repeating blocks largest-cost-first while they fit under the
	// partial-offload graph budget. (Dense blocks are equal-sized, so order
	// only matters under heterogeneity; largest-first biases conservative.)
	order := make([]int, blockCount)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return perLayer[order[a]] > perLayer[order[b]] })

	avail := int64(opts.FreeVRAM) - int64(reserve) - int64(partial)
	var weightsUsed, kvUsed uint64
	n := 0
	if avail > 0 {
		for _, i := range order {
			if int64(weightsUsed+kvUsed+perLayer[i]) <= avail {
				weightsUsed += blocks[i]
				kvUsed += kvPerLayer
				n++
			} else {
				break
			}
		}
	}

	est.Layers = n
	est.Graph = partial

	if n == blockCount {
		// all blocks fit: try to also offload the output layer under the
		// (larger) full-offload graph budget.
		if int64(opts.FreeVRAM)-int64(reserve)-int64(full)-int64(weightsUsed+kvUsed+output) >= 0 {
			est.FullyOffloaded = true
			est.Graph = full
			weightsUsed += output
		}
	}

	est.Weights = weightsUsed
	est.KVCache = kvUsed
	est.TotalVRAM = est.Weights + est.KVCache + est.Graph
	return est, nil
}
```

- [ ] **Step 4: Run the fit tests**

Run: `go test ./gguf/ -run TestEstimate -v`
Expected: PASS for all `TestEstimate*`. If `TestEstimatePartialFit` is off by one, confirm the `+8` slack in the test exceeds rounding; the helper computes the same `partial`/`kv` the code uses, so the count must match.

- [ ] **Step 5: Run the whole package + vet**

Run: `go test ./gguf/ -count=1 && go vet ./gguf/`
Expected: all PASS, vet clean.

- [ ] **Step 6: Commit**
```bash
git add gguf/estimate.go gguf/estimate_test.go
git commit -m "feat(gguf): single-GPU layer-fit estimator (EstimateLayers)"
```

---

## Task 5: Optional real-model test, docs, lint, final verify

**Files:**
- Modify: `gguf/estimate_test.go` (append real-model test)
- Modify: `README.md`

- [ ] **Step 1: Append the optional real-model test**

Append to `gguf/estimate_test.go` (it already imports `os`? No — add `"os"` to its import block):
```go
func TestEstimateRealModel(t *testing.T) {
	path := os.Getenv("LLMARK_TEST_GGUF")
	if path == "" {
		t.Skip("set LLMARK_TEST_GGUF to a .gguf file to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("LLMARK_TEST_GGUF not readable: %v", err)
	}
	// ~3.5 GiB usable on a 4 GB card
	est, err := EstimateLayers(path, EstimateOptions{NumCtx: 4096, FreeVRAM: 3500 << 20})
	if err != nil {
		t.Fatalf("EstimateLayers(%s): %v", path, err)
	}
	if est.Layers < 0 || est.Layers > len(est.PerLayerBytes) {
		t.Errorf("Layers = %d out of range 0..%d", est.Layers, len(est.PerLayerBytes))
	}
	t.Logf("layers=%d fullyOffloaded=%v weights=%dMiB kv=%dMiB graph=%dMiB approx=%v",
		est.Layers, est.FullyOffloaded, est.Weights>>20, est.KVCache>>20, est.Graph>>20, est.Approximate)
}
```
The `os` import: `estimate_test.go` currently imports only `errors` and `testing` — change to a grouped block adding `"os"`.

- [ ] **Step 2: Run full suite (real-model skipped)**

Run: `go test ./gguf/ -v -count=1`
Expected: all PASS; `TestEstimateRealModel` SKIPPED. Optional: `LLMARK_TEST_GGUF="<path>.gguf" go test ./gguf/ -run TestEstimateRealModel -v` → PASS with a logged breakdown.

- [ ] **Step 3: Lint (repo policy: golangci-lint default:all must be clean)**

Run: `golangci-lint run ./gguf/...`
Expected: 0 issues. Fix any findings in the new files (`graph.go`, `estimate.go`, `*_test.go`) — e.g. `wsl_v5` blank-line rules, `godoclint` on exported `EstimateOptions`/`Estimate`/`EstimateLayers`/`DefaultMinReserveBytes` (ensure each exported symbol has a doc comment starting with its name), `errcheck`. Do NOT modify the previously-shipped lifted files.

- [ ] **Step 4: Document in README**

In `README.md`, immediately after the "## Reading GGUF metadata (no model load)" subsection, add:
```markdown
## Estimating GPU layers (no model load)

`gguf.EstimateLayers` computes how many transformer layers fit in a VRAM budget
— pure Go, no cgo, no GPU calls (the caller supplies the budget):

```go
import "github.com/go-skynet/go-llama.cpp/gguf"

est, err := gguf.EstimateLayers("model.gguf", gguf.EstimateOptions{
    NumCtx:   4096,
    FreeVRAM: 3500 << 20, // ~3.5 GiB
})
// est.Layers (n_gpu_layers), est.FullyOffloaded, est.Weights/KVCache/Graph
```

It faithfully ports Ollama's Llama-family memory model (single-GPU). Non-Llama
dense architectures get an approximate estimate (`est.Approximate == true`);
recurrent/SSM architectures are not yet supported.
```
(Mind the nested ```go fence — ensure fences close correctly.)

- [ ] **Step 5: Final commit**
```bash
git add gguf/estimate_test.go README.md
git commit -m "test(gguf): optional real-model estimate test; document EstimateLayers"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- Package shape `gguf/estimate.go` + `gguf/graph.go` → Tasks 1,3,4. ✓
- `EstimateOptions`/`Estimate` exact fields → Task 3 (matches spec; **adds `MinReserveBytes`** — a testability refinement, 0 ⇒ 457 MiB default, flagged below). ✓
- Weights via `blk.N` grouping + output/token_embd → Task 3 `groupLayers`. ✓
- KV formula (key/value_length fallback, kv-type bytes) → Task 4. ✓
- Llama graph full/partial formulas → Task 1. ✓
- Single-GPU reserve-then-fill, "all fit → offload output + full graph" → Task 4. ✓
- Errors: FreeVRAM=0, recurrent (no heads), missing required keys → Tasks 3,4. ✓
- Unknown arch → `Approximate` not error → Task 4 + `TestEstimateApproximateForUnknownArch`. ✓
- Fixture builder, monotonic/threshold/quant tests, optional real-model → Tasks 2,4,5. ✓
- Pure-Go, no deps, lint-clean → enforced Task 5. ✓

**Deviation from spec (intentional):** `MinReserveBytes` option added so tests can bypass the 457 MiB floor and exercise the fit math on tiny fixtures; production callers leave it 0 and get the spec's 457 MiB. Behaviour for production is unchanged.

**Placeholder scan:** none — every code step shows complete code; graph literals are derived in-comment.

**Type consistency:** `EstimateOptions`, `Estimate`, `EstimateLayers`, `groupLayers`, `kvBytesPerElement`, `llamaGraphSize`, `orDefault`, `DefaultMinReserveBytes`, fixture helpers (`sampleLlamaModel`, `fixtureGraph`, `writeGGUF`, `kvPair`, `testTensor`) are used consistently across tasks. `llamaGraphSize` is called with `keyLen` as the `embeddingHeads` arg in both Task 4 and the test helper. ✓
