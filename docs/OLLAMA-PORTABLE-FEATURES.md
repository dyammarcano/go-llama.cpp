# What from Ollama is worth incorporating into go-llama.cpp

go-llama.cpp is a **thin cgo binding**: the smarts live in llama.cpp's C/C++.
Ollama is a **full engine**: most of its code re-wraps things llama.cpp already
does (sampling, GPU enum, decode loop) plus a server/scheduler the binding
should never grow. So the only Ollama pieces worth porting are the ones it does
**in pure Go** that the binding currently **can't do or does badly** — not the
ones that merely duplicate the C API.

Verdict legend: ✅ port · 🟡 build-from (use as reference) · 🔧 wire the C that's
already stubbed · ⛔ skip.

---

## ✅ 1. Pure-Go GGUF metadata reader — `fs/gguf` + `fs/ggml`
Pure Go, zero cgo. Reads the GGUF header, KV metadata (architecture,
`*.context_length`, `n_embd`, `n_layer`, quantization), the embedded chat
template, and the **per-tensor shapes/sizes** — all **without loading the model
into VRAM**.

**Why it's the #1 win.** Today the binding can tell you *nothing* about a model
until it's loaded. This unblocks: pre-flight validation, choosing `n_ctx` from
the trained context length, reading the template name, and — critically —
feeding the layer-fit estimator (#2). For llmark, `models discover-gguf`
currently parses quant/arch from *filenames*; this gives it real metadata.

Effort: low (lift the package, swap Ollama's API types for plain structs).

## 🟡 2. VRAM / layer-fit estimation — reference `llm/memory.go`
The killer gap. Users currently guess `SetGPULayers(99)` and OOM on a 4GB card.
Ollama's estimator is *partly* cgo-backed so it isn't a clean lift — **but the
heuristic is reproducible in pure Go** from #1's data:

```
per_layer_bytes = sum(tensor bytes for that block)
kv_cache_bytes  = 2 * n_layer * n_ctx * n_embd_kv * sizeof(kv_type)
overhead        = compute buffer + output tensors
fit_layers      = max k s.t. (k*per_layer + kv + overhead) <= free_vram
```

Pair with #1 for inputs and llmark's `internal/env` (or a one-shot
`ggml_backend_dev` query) for free-VRAM. Directly solves the GTX-1650 guessing.

Effort: medium. Highest value-per-line after #1.

## ✅ 3. Stop-sequence + incomplete-UTF-8 buffering — `runner/common/stop.go`
Tiny, pure Go: `FindStop`, `TruncateStop`, `IncompleteUnicode`. Fixes a **real
correctness bug** in the current binding (GAP #4): tokens are streamed raw, so
multi-byte UTF-8 tokens can be split mid-character, and stop-word matching is a
fragile C++ suffix-compare on the whole accumulated string.

**Insertion point: the Go callback path, not C.** `SetTokenCallback` already
streams pieces to Go — wrap it with Ollama's unicode buffer + stop detector.
No C++ changes needed.

Effort: very low.

**Wired (2026-06-01):** `streamfilter` is now connected to the binding's
token-callback path; the C++ antiprompt suffix-compare in `binding.cpp` was
removed so Go owns stop detection. See
`docs/superpowers/specs/2026-06-01-streamfilter-wiring-design.md`.

## ✅ 4. Wire the sampler features already stubbed in `binding.cpp`
`min_p`, `typical_p`, `mirostat` (v1 & v2), and `logit_bias` are now fully wired
into `make_sampler()` and connected to the Go API (`SetMinP`, `SetTypicalP`,
`SetMirostat`, `SetLogitBias`). The new cgo-free `logitbias` subpackage parses
logit bias strings (e.g. `"<id>:-100"`) into parallel C arrays. `make_sampler()`
redesigned to `make_sampler(bp, vocab)` with the sampler chain order:
`logit_bias → penalties → [mirostat terminal | greedy | top_k → typical → top_p → min_p → temp → dist]`,
all opt-in so defaults reproduce prior behavior. `tfs_z` and `penalize_nl` were
dropped as documented no-ops (removed upstream llama.cpp).

**GBNF grammar wiring is deferred** as a follow-up — see the design specs:
`docs/superpowers/specs/2026-05-30-sampler-wiring-design.md` and
`docs/superpowers/plans/2026-05-30-sampler-wiring.md`.

---

## ⛔ Skip — duplicates the C API or is server-coupled

- **Go-side standard samplers (`sample/`)** — top-k/p/min-p/temp already run in
  llama.cpp's chain. Doing them in Go means copying the full vocab-sized logit
  vector across cgo **every token** (~32k–150k floats). Pure perf loss. Only
  justified for custom logit processors the C chain can't express — not the case
  here. Prefer #4.
- **Continuous batching / parallel slots (`runner/llamarunner`)** — welded to
  Ollama's HTTP server + scheduler. A binding needs no server. (The *idea* — one
  context serving N prompts — is a real benchmarking win, but it belongs *above*
  the binding, e.g. in llmark's runner, not inside go-llama.cpp.)
- **GPU discovery (`discover/`)** — cgo-bound to the ggml backend; llmark already
  has `internal/env` for GPU/VRAM probing. Don't duplicate.
- **Chat template engine (`template/`)** — excellent pure Go, but exists because
  Ollama supports non-embedded templates + fuzzy name-matching. The binding is
  tied to one GGUF and already uses the model's embedded template via
  `llama_chat_apply_template`. The C path is simpler and correct here.

---

## Recommended order
1. `fs/gguf` reader (unblocks everything else)
2. Stop/UTF-8 buffer in the Go callback (cheap correctness fix)
3. VRAM layer-fit estimator (auto `n_gpu_layers`)
4. Wire GBNF grammar + min-p/typical/mirostat/logit-bias in `make_sampler()`

Everything else stays in Ollama / llmark — pulling it into the binding would
either duplicate the C API or drag a server into a library.
