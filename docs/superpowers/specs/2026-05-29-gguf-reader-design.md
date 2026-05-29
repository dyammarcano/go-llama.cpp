# Pure-Go GGUF reader — design

Date: 2026-05-29
Status: approved
Repo: `github.com/go-skynet/go-llama.cpp`

## Goal

Add a **pure-Go GGUF metadata reader** to go-llama.cpp so callers can introspect
a model — architecture, trained context length, embedding/layer counts, chat
template, quantization, and per-tensor sizes — **without loading it into VRAM or
touching llama.cpp / cgo**. This unblocks two downstream features (VRAM layer-fit
estimation; better model discovery in llmark) and gives the binding a model-info
surface it currently lacks entirely.

Ported (faithful lift) from Ollama's `fs/gguf` package (MIT). See
`docs/OLLAMA-PORTABLE-FEATURES.md` for the broader rationale.

## Non-goals

- No cgo, no llama.cpp dependency, no build tags — this package compiles and
  tests standalone.
- No VRAM estimator (separate feature #2 — will *consume* this package's tensor
  sizing).
- No stop-sequence / UTF-8 buffer (separate feature #3).
- No tensor *data* decoding beyond exposing `TensorReader` section readers.

## Package layout

New subpackage at repo root:

```
gguf/
  gguf.go        # Open, File, lazy KV/tensor readers, accessors
  keyvalue.go    # KeyValue, Value + typed accessors
  tensor.go      # TensorInfo, TensorType, NumBytes() sizing math
  lazy.go        # iter.Pull-based lazy loader
  reader.go      # offset-tracking buffered reader
  metadata.go    # NEW: Info(path) convenience layer
  writer_test.go     # NEW: tiny pure-stdlib GGUF fixture writer (test-only)
  gguf_test.go       # adapted (uses writer_test.go fixtures)
  keyvalue_test.go   # ported
  metadata_test.go   # NEW
```

Import path: `github.com/go-skynet/go-llama.cpp/gguf`.

## Public API

### Low-level (faithful port of Ollama's surface)

```go
func Open(path string) (*File, error)

func (f *File) Close() error
func (f *File) KeyValue(key string) KeyValue          // smart arch.-prefixing
func (f *File) KeyValues() iter.Seq2[int, KeyValue]
func (f *File) NumKeyValues() int
func (f *File) TensorInfo(name string) TensorInfo
func (f *File) TensorInfos() iter.Seq2[int, TensorInfo]
func (f *File) NumTensors() int
func (f *File) TensorReader(name string) (TensorInfo, io.Reader, error)

type KeyValue struct { Key string; Value }
type Value struct { /* ... */ }
func (v Value) Int() int64        // + Uint/Float/Bool/String + slice variants
type TensorInfo struct { Name string; Offset uint64; Shape []uint64; Type TensorType }
func (ti TensorInfo) NumValues() int64
func (ti TensorInfo) NumBytes() int64
type TensorType uint32
func (tt TensorType) NumBytes() float64   // typeSize/blockSize quant math
func (tt TensorType) String() string
```

### Convenience layer (new — `metadata.go`)

```go
// Info holds the commonly-needed model facts read without loading the model.
type Info struct {
    Architecture    string  // general.architecture
    Name            string  // general.name
    FileType        uint32  // general.file_type (quant scheme enum)
    Quantization    string  // human label derived from FileType / dominant tensor type
    ContextLength   uint64  // <arch>.context_length
    EmbeddingLength uint64  // <arch>.embedding_length
    BlockCount      uint64  // <arch>.block_count (layers)
    HeadCount       uint64  // <arch>.attention.head_count
    HeadCountKV     uint64  // <arch>.attention.head_count_kv
    ChatTemplate    string  // tokenizer.chat_template ("" if none)
    NumTensors      int
}

// Info opens path, reads common metadata, and closes the file.
func Info(path string) (*Info, error)
```

`Info` uses `File.KeyValue`'s arch-prefixing so `"context_length"` resolves to
`<arch>.context_length` automatically. Missing keys yield zero values, not
errors (GGUFs vary by architecture).

## Adaptations to the verbatim copy

1. **Magic validation fix.** Ollama's `Open` only rejects lowercase `"gguf"` and
   never asserts the magic *is* `"GGUF"`, so non-GGUF input passes. The port
   asserts `Magic == "GGUF"` (bytes `0x47 0x47 0x55 0x46`) and returns
   `fmt.Errorf("%w: bad magic %q", ErrUnsupported, magic)` otherwise.
2. **Attribution header.** Each ported file carries a header noting it derives
   from `github.com/ollama/ollama/fs/gguf` (MIT) — both projects are MIT.
3. **Package path** in tests updated to the new import.

## Error handling

- `Open` returns wrapped errors for: file open failure, bad magic, unsupported
  version (`< 2`), unsupported KV/array types (`ErrUnsupported`).
- `Info` propagates `Open`/read errors; absent keys are not errors.
- `TensorReader` returns an error when the named tensor is absent.

## Testing

Ollama's `gguf_test.go` builds fixtures via its `fs/ggml` *writer* package. We do
NOT port `fs/ggml` (scope creep, defeats the zero-dep goal). Instead:

- `writer_test.go` (test-only, `package gguf_test`): a ~60-line GGUF serializer
  using `encoding/binary` — writes magic, version, the KV block, and tensor info
  + padded data for a given `map[string]any` + tensor list. Enough to exercise
  `Open`, `KeyValue`, `TensorInfo`, `TensorReader`, and `Info`.
- `gguf_test.go`: adapted from Ollama's assertions but built on `writer_test.go`
  fixtures instead of `fs/ggml`.
- `keyvalue_test.go`: ported (value-accessor unit tests, no fixture dependency).
- `metadata_test.go`: assert `Info()` fields against a written fixture; plus an
  optional test against a real local model (`testing.Short()`-skippable, skipped
  if the file is absent) to confirm real-world parsing.
- Assertions use stdlib (`reflect.DeepEqual`) to avoid promoting `go-cmp` to a
  direct dependency.
- Run: `go test ./gguf/`. No build tags, no GPU, no llama.cpp, no new go.mod deps.

## Risks

- **Quantization label** (`Info.Quantization`) is a derived nicety; if the
  `general.file_type` → label mapping is incomplete it falls back to the
  dominant tensor `Type.String()`. Low risk, cosmetic.
- Ollama's `iter.Pull` lazy reader requires `Close()` to release the pull
  iterator goroutine; `Info()` always closes. Documented on `File.Close`.
