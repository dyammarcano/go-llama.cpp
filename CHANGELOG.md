# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the API is
pre-1.0 (`0.x`), minor versions may include breaking changes.

## [0.1.0] - 2026-07-20

First tagged release of the modernized fork. The module path is now
`github.com/dyammarcano/go-llama.cpp`, so the package is importable directly.

### Added
- Pure-Go `gguf` subpackage: GGUF metadata/tensor reader and a VRAM layer estimator (no cgo).
- Pure-Go `streamfilter` subpackage: stop-sequence + UTF-8-safe streaming filter, wired into `Predict`/`PredictResult`.
- GBNF grammar-constrained sampling via `WithGrammar`.
- Exported sentinel errors (`ErrModelLoad`, `ErrStateLoad`, `ErrInference`, `ErrOutOfMemory`, `ErrEmbeddingsDisabled`, `ErrNotImplemented`) wrapped with `%w` for `errors.Is`.
- godoc on the core public API (`LLama`, `New`, `Free`, `Eval`, `Predict`, and the embedding/state/speculative calls).

### Changed
- Module path renamed from `github.com/go-skynet/go-llama.cpp` to `github.com/dyammarcano/go-llama.cpp`.
- Rebuilt onto the current `llama.cpp` C API (`llama_sampler` chain, chat templates); pinned to upstream `b10069`.
- Build system migrated from `make` to [Task](https://taskfile.dev); CPU static libs build into `llama.cpp/build-cpu`.

### Fixed
- CPU cgo link pointed at a stale `llama.cpp/build` tree instead of `build-cpu`, causing undefined-symbol link failures.
- Freed leaked C strings in `Predict` and `SpeculativeSampling`.

### Known limitations
- `Embeddings`, `TokenEmbeddings`, `LoadState`, `SaveState`, and `SpeculativeSampling` are exported but not yet implemented; they return `ErrNotImplemented`.
- CUDA and Vulkan build targets exist but are unverified against the current pin.
- Verified on Windows/MinGW only; Linux and macOS are not yet CI-covered.
- The cgo binding requires native static libs (`task build:cpu`) before a consumer can compile; `go get` alone is not sufficient.

[0.1.0]: https://github.com/dyammarcano/go-llama.cpp/releases/tag/v0.1.0
