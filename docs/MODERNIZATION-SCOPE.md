# go-llama.cpp Modernization Scope

Porting the binding from early-2024 llama.cpp to current `master` (pinned `19e92c33`, 2026-05-28).

## Guiding constraint — keep the C ABI stable
`binding.h` defines 13 exported functions, all using `void*` + primitives. **None of their
signatures need to change.** If we reimplement only the *bodies* in `binding.cpp`, then
`llama.go`, `options.go`, and `llama_*.go` stay **untouched**. That is the whole strategy:
freeze `binding.h`, rewrite `binding.cpp`, replace the Makefile.

## API migration map (old → new)

| Old (binding.cpp) | New (llama.cpp master) | Notes |
|---|---|---|
| `gpt_params` | `common_params` (`common/common.h`) | field-by-field re-map |
| `llama_init_from_gpt_params` / `llama_context_params_from_gpt_params` | `common_init_from_params(params)` → `common_init_result_ptr` (`.model()`, `.context()`) | now returns a smart-ptr wrapper |
| `llama_load_model_from_file` | `llama_model_load_from_file` | rename |
| `llama_new_context_with_model` | `llama_init_from_model` | old name deprecated |
| `llama_free_model` | `llama_model_free` | rename |
| `llama_eval(ctx, toks, n, n_past)` | `llama_decode(ctx, batch)` + `llama_batch_get_one` | KV/seq managed by batch; track positions |
| `llama_backend_init(numa)` | `llama_backend_init()` + `llama_numa_init(strategy)` | split into two calls |
| `llama_n_vocab(ctx)` | `llama_vocab_n_tokens(vocab)` | need `vocab = llama_model_get_vocab(model)` |
| `llama_token_bos/eos(ctx)` | `llama_vocab_bos/eos(vocab)` | vocab-based |
| `llama_tokenize(ctx, …)` | `llama_tokenize(vocab, text, len, out, max, add_special, parse_special)` | sig changed → vocab |
| `llama_token_to_piece(ctx, …)` | `llama_token_to_piece(vocab, tok, buf, len, lstrip, special)` | sig changed → vocab |
| `llama_get_logits` / `llama_n_ctx` | unchanged | ✅ |
| `llama_load_session_file` / `…save…` | `llama_state_load_file` / `llama_state_save_file` | rename |
| **all `llama_sample_*`** (`repetition_penalty`, `frequency_and_presence_penalties`, `temperature`, `token_mirostat`, `token_greedy`, `grammar`, `token`) | **`common_sampler`** (`common/sampling.h`): `common_sampler_init(model, params.sampling)` → `common_sampler_sample(smpl, ctx, idx)` → `common_sampler_accept(smpl, tok, accept_grammar)` | **collapses ~200 lines into 3 calls**; chain built from `common_params_sampling` |
| `llama_grammar_init` / `_accept_token` + `grammar-parser.h` | folded into the sampler (`common_params_sampling.grammar`) | `grammar-parser.h` removed |
| `llama_sample_classifier_free_guidance` (negative prompt / CFG) | **removed upstream** | drop or stub `negative_prompt*` args |

## Functions to rewrite in `binding.cpp` (1205 LOC)

| Fn (line) | Effort | Action |
|---|---|---|
| `eval` (92) | 🟢 easy | `llama_decode` + batch |
| `llama_tokenize_string` (817) | 🟢 easy | vocab tokenize |
| `load_state` / `save_state` (840/866) | 🟢 easy | `llama_state_*_file` rename |
| `llama_free_params` / `llama_binding_free_model` (812) | 🟢 easy | `llama_model_free` |
| `llama_allocate_params` (881) | 🟢 easy | pack into `common_params(_sampling)` |
| `load_model` (957) | 🟡 moderate | `common_params` + `llama_model_load_from_file` + `llama_init_from_model` |
| `get_embeddings` / `get_token_embeddings` (37/76) | 🟡 moderate | `ctx_params.embeddings=true`, `llama_get_embeddings_seq`, pooling |
| `llama_predict` (119–562, ~440 LOC) | 🔴 **hard** | full rewrite: tokenize → decode loop → `common_sampler` → `token_to_piece` → `tokenCallback`. The core. |
| `speculative_sampling` (563–811, ~250 LOC) | 🔴 hard / **defer** | rewrite on `common/speculative.h` or **stub** initially |

## Build system (replace the Makefile)
Old Makefile links hand-built objects (`ggml.o`, `k_quants.o`, `grammar-parser.o`, `llama.o`,
`common.o`) — **none exist anymore**. New flow:

1. Build llama.cpp via CMake, static:
   ```
   cmake -S llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
   cmake --build llama.cpp/build --config Release -j
   ```
   Produces `libllama.a`, `libggml.a` + `libggml-base.a` + `libggml-cpu.a` (+ `libggml-cuda.a` etc.), `libcommon.a`.
2. Compile `binding.cpp` → archive into `libbinding.a` alongside those.
3. Update cgo in `llama.go`:
   - `CXXFLAGS`: `-I${SRCDIR}/llama.cpp/include -I${SRCDIR}/llama.cpp/common -I${SRCDIR}/llama.cpp/ggml/include`
   - `LDFLAGS`: `-L…/build/... -lbinding -lcommon -lllama -lggml -lggml-base -lggml-cpu -lstdc++ -lm`
   - GPU tags (`llama_cublas.go`→`-DGGML_CUDA=ON` + `-lggml-cuda`; metal/openblas similarly)

### ⚠️ Static-link gotcha
GGML device backends self-register via constructors. With **static** libs the linker drops them
unless forced. Either:
- link backend archives with `-Wl,--whole-archive … -Wl,--no-whole-archive`, **or**
- build GGML shared (`BUILD_SHARED_LIBS=ON`) + call `ggml_backend_load_all()` at startup (ship the `.dll/.so`).
This is the single biggest build risk, especially on **Windows** (needs MinGW/clang for cgo, not MSVC).

## Effort estimate
- 🟢 easy batch (eval, tokenize, state, frees, allocate_params): ~0.5 day
- 🟡 load_model + embeddings: ~1 day
- 🔴 `llama_predict` rewrite (the core): ~1–1.5 days
- Build: CMake integration + cgo flags + CPU+one GPU backend + Windows toolchain: ~1–1.5 days
- `speculative_sampling`: **deferred/stubbed** (≈1 day if pursued later)

**~4–5 days** to a working CPU(+CUDA) build with `New`/`Predict`/`Embeddings`/streaming;
speculative + CFG dropped initially.

## Suggested sequencing
1. Freeze `binding.h`; stub `binding.cpp` (all fns return error) → get **CMake+cgo link** green first (de-risks the hardest unknown).
2. `load_model` → `eval` → `llama_tokenize_string` (smoke-test load + tokenize).
3. `llama_predict` with `common_sampler` (greedy first, then full sampling) + streaming callback.
4. `embeddings`, `state` save/load.
5. Defer `speculative_sampling`; drop CFG/negative-prompt.
6. Add GPU build tags last.

## Net
`binding.h` ABI stays fixed → **`llama.go` untouched**. The work is ~1 hard function
(`llama_predict`) + a handful of easy renames + a CMake-based build. `common_sampler` and
`common_init_from_params` do most of the heavy lifting that the old binding did by hand.
