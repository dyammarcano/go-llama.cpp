# Pure-Go GGUF Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-Go `gguf` package to go-llama.cpp that reads GGUF model metadata and tensor info without cgo/llama.cpp, plus a `Stat()` convenience accessor.

**Architecture:** Faithful lift of Ollama's `fs/gguf` (MIT) into `gguf/` at the repo root — a lazy (`iter.Pull`) reader over the GGUF header, KV block, and tensor-info block. Adds one new file (`metadata.go`) with a `Stat(path) (*Info, error)` convenience layer, and a test-only stdlib GGUF fixture writer so tests carry no dependency on Ollama's `fs/ggml`.

**Tech Stack:** Go 1.25 (uses `iter`, range-over-int). Stdlib only — `encoding/binary`, `bufio`, `io`, `os`, `iter`, `reflect`, `slices`, `cmp`, `log/slog`. No new `go.mod` deps, no build tags, no cgo.

**Source of truth for lifted files (read these from disk during the lift):**
- `C:/Users/dyamm/My Drive/acer/public_repos/ollama/fs/gguf/gguf.go`
- `.../keyvalue.go`, `.../lazy.go`, `.../reader.go`, `.../tensor.go`

**Destination repo:** `C:/Users/dyamm/My Drive/acer/public_repos/go-llama.cpp`

---

## File Structure

```
gguf/
  reader.go      # LIFT verbatim — offset-tracking buffered reader (unexported)
  lazy.go        # LIFT verbatim — iter.Pull lazy loader (unexported)
  keyvalue.go    # LIFT verbatim — KeyValue/Value + typed accessors
  tensor.go      # LIFT verbatim — TensorInfo/TensorType + NumBytes sizing
  gguf.go        # LIFT + magic-byte fix — Open, File, accessors
  metadata.go    # NEW — Info struct + Stat(path) + quantLabel
  keyvalue_test.go   # NEW — value-accessor unit tests (no fixture)
  writer_test.go     # NEW — stdlib GGUF fixture writer (test-only)
  gguf_test.go       # NEW — Open/KeyValue/TensorInfo/TensorReader tests
  metadata_test.go   # NEW — Stat() field tests + optional real-model smoke
```

Every lifted/new `.go` file starts with this header comment (both repos are MIT):

```go
// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.
```

---

## Task 1: Lift the leaf files (keyvalue.go, tensor.go) + unit tests

These two files are pure value/sizing logic with no fixture dependency. TDD them first.

**Files:**
- Create: `gguf/keyvalue.go` (lift)
- Create: `gguf/tensor.go` (lift)
- Test: `gguf/keyvalue_test.go` (new)

- [ ] **Step 1: Write the failing tests**

Create `gguf/keyvalue_test.go`:

```go
// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.
package gguf

import (
	"reflect"
	"testing"
)

func TestValueScalars(t *testing.T) {
	if got := (Value{int32(7)}).Int(); got != 7 {
		t.Errorf("Int() = %d, want 7", got)
	}
	if got := (Value{uint16(9)}).Uint(); got != 9 {
		t.Errorf("Uint() = %d, want 9", got)
	}
	if got := (Value{float32(1.5)}).Float(); got != 1.5 {
		t.Errorf("Float() = %v, want 1.5", got)
	}
	if got := (Value{true}).Bool(); got != true {
		t.Errorf("Bool() = %v, want true", got)
	}
	if got := (Value{"hi"}).String(); got != "hi" {
		t.Errorf("String() = %q, want hi", got)
	}
	// wrong-kind access returns zero, not panic
	if got := (Value{"hi"}).Int(); got != 0 {
		t.Errorf("Int() on string = %d, want 0", got)
	}
}

func TestValueSlices(t *testing.T) {
	if got := (Value{[]int32{1, 2, 3}}).Ints(); !reflect.DeepEqual(got, []int64{1, 2, 3}) {
		t.Errorf("Ints() = %v, want [1 2 3]", got)
	}
	if got := (Value{[]string{"a", "b"}}).Strings(); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Errorf("Strings() = %v, want [a b]", got)
	}
}

func TestTensorNumBytes(t *testing.T) {
	// F32, shape 2x3 = 6 values * 4 bytes = 24
	ti := TensorInfo{Name: "x", Shape: []uint64{2, 3}, Type: TensorTypeF32}
	if got := ti.NumValues(); got != 6 {
		t.Errorf("NumValues() = %d, want 6", got)
	}
	if got := ti.NumBytes(); got != 24 {
		t.Errorf("NumBytes() = %d, want 24", got)
	}
	// Q4_0: typeSize=2+32/2=18, blockSize=32 -> 0.5625 bytes/value; 32 values -> 18
	q := TensorInfo{Name: "y", Shape: []uint64{32}, Type: TensorTypeQ4_0}
	if got := q.NumBytes(); got != 18 {
		t.Errorf("Q4_0 NumBytes() = %d, want 18", got)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/dyamm/My Drive/acer/public_repos/go-llama.cpp" && go test ./gguf/ -run 'TestValue|TestTensorNumBytes' -v`
Expected: FAIL — `package gguf` has no `Value`/`TensorInfo` (build error / undefined).

- [ ] **Step 3: Lift keyvalue.go and tensor.go**

Copy `ollama/fs/gguf/keyvalue.go` → `gguf/keyvalue.go` verbatim, then prepend the attribution header (keep `package gguf`). It defines `KeyValue`, `Value`, and the `Int/Uint/Float/Bool/String` (+ `Ints/Uints/Floats/Bools/Strings`) accessors via the generic `value`/`values` reflect helpers.

Copy `ollama/fs/gguf/tensor.go` → `gguf/tensor.go` verbatim, then prepend the attribution header (keep `package gguf`). It defines `TensorInfo` (`Name/Offset/Shape/Type`, `NumValues`, `NumBytes`, `Valid`, `LogValue`) and `TensorType` (the full quant enum, `typeSize`, `blockSize`, `NumBytes`, `String`, `LogValue`).

Make NO logic changes to either file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./gguf/ -run 'TestValue|TestTensorNumBytes' -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/dyamm/My Drive/acer/public_repos/go-llama.cpp"
git add gguf/keyvalue.go gguf/tensor.go gguf/keyvalue_test.go
git commit -m "feat(gguf): lift KeyValue/Value accessors and TensorType sizing"
```

---

## Task 2: Lift the reader (reader.go, lazy.go, gguf.go) + fixture writer + Open tests

`reader.go` and `lazy.go` are unexported infra exercised through `Open`. `gguf.go` gets the one logic change: magic-byte validation. The fixture writer lets tests build GGUF blobs with stdlib only.

**Files:**
- Create: `gguf/reader.go` (lift), `gguf/lazy.go` (lift), `gguf/gguf.go` (lift + magic fix)
- Create: `gguf/writer_test.go` (new), `gguf/gguf_test.go` (new)

- [ ] **Step 1: Write the fixture writer + failing tests**

Create `gguf/writer_test.go`:

```go
package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// GGUF metadata value type tags (subset emitted by fixtures).
const (
	wUint32  uint32 = 4
	wInt32   uint32 = 5
	wFloat32 uint32 = 6
	wBool    uint32 = 7
	wString  uint32 = 8
	wArray   uint32 = 9
	wUint64  uint32 = 10
)

type kvPair struct {
	Key string
	Val any
}

type testTensor struct {
	Name  string
	Type  uint32 // TensorType value (0 = F32)
	Shape []uint64
	Data  []byte
}

func wString(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func wValue(tb testing.TB, b *bytes.Buffer, v any) {
	le := binary.LittleEndian
	switch x := v.(type) {
	case uint32:
		_ = binary.Write(b, le, wUint32)
		_ = binary.Write(b, le, x)
	case int32:
		_ = binary.Write(b, le, wInt32)
		_ = binary.Write(b, le, x)
	case uint64:
		_ = binary.Write(b, le, wUint64)
		_ = binary.Write(b, le, x)
	case float32:
		_ = binary.Write(b, le, wFloat32)
		_ = binary.Write(b, le, x)
	case bool:
		_ = binary.Write(b, le, wBool)
		_ = binary.Write(b, le, x)
	case string:
		_ = binary.Write(b, le, wString)
		wString(b, x)
	case []string:
		_ = binary.Write(b, le, wArray)
		_ = binary.Write(b, le, wString)
		_ = binary.Write(b, le, uint64(len(x)))
		for _, s := range x {
			wString(b, s)
		}
	case []int32:
		_ = binary.Write(b, le, wArray)
		_ = binary.Write(b, le, wInt32)
		_ = binary.Write(b, le, uint64(len(x)))
		for _, e := range x {
			_ = binary.Write(b, le, e)
		}
	default:
		tb.Fatalf("wValue: unsupported type %T", v)
	}
}

func alignUp(n, a uint64) uint64 { return (n + a - 1) / a * a }

// writeGGUF serializes a minimal valid GGUF v3 file and returns its path.
// Layout: magic, version, tensor_count, kv_count, KV block, tensor-info block,
// pad-to-alignment, tensor data. Alignment is the default 32.
func writeGGUF(tb testing.TB, kvs []kvPair, tensors []testTensor) string {
	tb.Helper()
	le := binary.LittleEndian
	const alignment = 32

	b := new(bytes.Buffer)
	b.WriteString("GGUF")
	_ = binary.Write(b, le, uint32(3))            // version
	_ = binary.Write(b, le, uint64(len(tensors))) // tensor count
	_ = binary.Write(b, le, uint64(len(kvs)))     // kv count
	for _, kv := range kvs {
		wString(b, kv.Key)
		wValue(tb, b, kv.Val)
	}

	// assign each tensor an aligned offset within the data section
	offsets := make([]uint64, len(tensors))
	var dataLen uint64
	for i, t := range tensors {
		dataLen = alignUp(dataLen, alignment)
		offsets[i] = dataLen
		dataLen += uint64(len(t.Data))
	}

	for i, t := range tensors {
		wString(b, t.Name)
		_ = binary.Write(b, le, uint32(len(t.Shape)))
		for _, d := range t.Shape {
			_ = binary.Write(b, le, d)
		}
		_ = binary.Write(b, le, t.Type)
		_ = binary.Write(b, le, offsets[i])
	}

	// pad from start-of-file so the data section starts on an alignment boundary
	if pad := (alignment - uint64(b.Len())%alignment) % alignment; pad > 0 {
		b.Write(make([]byte, pad))
	}

	data := make([]byte, dataLen)
	for i, t := range tensors {
		copy(data[offsets[i]:], t.Data)
	}
	b.Write(data)

	path := filepath.Join(tb.TempDir(), "model.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		tb.Fatal(err)
	}
	return path
}

// sampleModel returns a small llama-arch fixture used across tests.
func sampleModel(tb testing.TB) string {
	tb.Helper()
	return writeGGUF(tb,
		[]kvPair{
			{"general.architecture", "llama"},
			{"general.name", "tiny-test"},
			{"general.file_type", uint32(7)}, // Q8_0
			{"llama.context_length", uint32(4096)},
			{"llama.embedding_length", uint32(3)},
			{"llama.block_count", uint32(2)},
			{"llama.attention.head_count", uint32(2)},
			{"llama.attention.head_count_kv", uint32(1)},
			{"tokenizer.chat_template", "{{ .Prompt }}"},
		},
		[]testTensor{
			// F32 2x3 = 24 bytes, with a recognizable data pattern
			{Name: "token_embd.weight", Type: 0, Shape: []uint64{2, 3}, Data: bytes.Repeat([]byte{0xAB}, 24)},
		},
	)
}
```

Create `gguf/gguf_test.go`:

```go
package gguf

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenReadsHeaderAndKV(t *testing.T) {
	f, err := Open(sampleModel(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	if f.NumTensors() != 1 {
		t.Errorf("NumTensors() = %d, want 1", f.NumTensors())
	}
	if got := f.KeyValue("general.architecture").String(); got != "llama" {
		t.Errorf("architecture = %q, want llama", got)
	}
	// arch-prefixed lookup: "context_length" -> "llama.context_length"
	if got := f.KeyValue("context_length").Uint(); got != 4096 {
		t.Errorf("context_length = %d, want 4096", got)
	}
	if got := f.KeyValue("block_count").Uint(); got != 2 {
		t.Errorf("block_count = %d, want 2", got)
	}
}

func TestTensorInfoAndReader(t *testing.T) {
	f, err := Open(sampleModel(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	ti := f.TensorInfo("token_embd.weight")
	if !ti.Valid() {
		t.Fatal("token_embd.weight not found")
	}
	if ti.NumBytes() != 24 {
		t.Errorf("NumBytes() = %d, want 24", ti.NumBytes())
	}

	_, r, err := f.TensorReader("token_embd.weight")
	if err != nil {
		t.Fatalf("TensorReader: %v", err)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(data) != 24 || data[0] != 0xAB || data[23] != 0xAB {
		t.Errorf("tensor data = %v (len %d), want 24x 0xAB", data[:min(4, len(data))], len(data))
	}
}

func TestOpenRejectsBadMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.gguf")
	if err := os.WriteFile(path, []byte("NOPExxxxxxxxxxxx"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Open(path)
	if !errors.Is(err, ErrUnsupported) {
		t.Errorf("Open(bad magic) err = %v, want ErrUnsupported", err)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./gguf/ -run TestOpen -v`
Expected: FAIL — `Open`, `File`, `ErrUnsupported` undefined (build error).

- [ ] **Step 3: Lift reader.go and lazy.go verbatim**

Copy `ollama/fs/gguf/reader.go` → `gguf/reader.go` and `ollama/fs/gguf/lazy.go` → `gguf/lazy.go`, each verbatim with the attribution header prepended (keep `package gguf`). No logic changes. `reader.go` defines `bufferedReader` (offset-tracking `bufio.Reader`); `lazy.go` defines the generic `lazy[T]` with `newLazy`, `Values`, `All`, `rest`, and `successFunc`.

- [ ] **Step 4: Lift gguf.go with the magic-byte fix**

Copy `ollama/fs/gguf/gguf.go` → `gguf/gguf.go` verbatim with the attribution header prepended (keep `package gguf`). It defines the type tag consts, `ErrUnsupported`, `File`, `Open`, the `read*` helpers, `Close`, `KeyValue`, `KeyValues`, `NumKeyValues`, `TensorInfo`, `TensorInfos`, `NumTensors`, `TensorReader`.

Then apply ONE change. Replace the original magic guard in `Open`:

```go
	if bytes.Equal(f.Magic[:], []byte("gguf")) {
		return nil, fmt.Errorf("%w file type %v", ErrUnsupported, f.Magic)
	}
```

with proper validation (assert the magic IS uppercase "GGUF"):

```go
	if !bytes.Equal(f.Magic[:], []byte("GGUF")) {
		return nil, fmt.Errorf("%w: bad magic %q", ErrUnsupported, f.Magic[:])
	}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `go test ./gguf/ -run TestOpen -v && go test ./gguf/ -run TestTensorInfoAndReader -v`
Expected: PASS. If `TestOpenReadsHeaderAndKV` fails on offsets, recheck the writer's header field order (magic, version, tensor_count, kv_count) against `Open` — it reads `tensors.count` before `keyValues.count`.

- [ ] **Step 6: Commit**

```bash
git add gguf/reader.go gguf/lazy.go gguf/gguf.go gguf/writer_test.go gguf/gguf_test.go
git commit -m "feat(gguf): lift Open/File reader with GGUF magic validation fix"
```

---

## Task 3: metadata.go — Info struct + Stat() convenience layer

**Files:**
- Create: `gguf/metadata.go` (new)
- Test: `gguf/metadata_test.go` (new)

- [ ] **Step 1: Write the failing test**

Create `gguf/metadata_test.go`:

```go
package gguf

import "testing"

func TestStatReadsCommonFields(t *testing.T) {
	info, err := Stat(sampleModel(t))
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.Architecture != "llama" {
		t.Errorf("Architecture = %q, want llama", info.Architecture)
	}
	if info.Name != "tiny-test" {
		t.Errorf("Name = %q, want tiny-test", info.Name)
	}
	if info.ContextLength != 4096 {
		t.Errorf("ContextLength = %d, want 4096", info.ContextLength)
	}
	if info.EmbeddingLength != 3 {
		t.Errorf("EmbeddingLength = %d, want 3", info.EmbeddingLength)
	}
	if info.BlockCount != 2 {
		t.Errorf("BlockCount = %d, want 2", info.BlockCount)
	}
	if info.HeadCount != 2 || info.HeadCountKV != 1 {
		t.Errorf("HeadCount/KV = %d/%d, want 2/1", info.HeadCount, info.HeadCountKV)
	}
	if info.ChatTemplate != "{{ .Prompt }}" {
		t.Errorf("ChatTemplate = %q", info.ChatTemplate)
	}
	if info.NumTensors != 1 {
		t.Errorf("NumTensors = %d, want 1", info.NumTensors)
	}
	// general.file_type = 7 -> Q8_0
	if info.Quantization != "Q8_0" {
		t.Errorf("Quantization = %q, want Q8_0", info.Quantization)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./gguf/ -run TestStat -v`
Expected: FAIL — `Stat` / `Info` undefined.

- [ ] **Step 3: Write metadata.go**

Create `gguf/metadata.go`:

```go
// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.
package gguf

import "fmt"

// Info holds commonly-needed model facts, read without loading the model.
type Info struct {
	Architecture    string
	Name            string
	FileType        uint32 // general.file_type (quant scheme enum)
	Quantization    string // human label derived from FileType (fallback: dominant tensor type)
	ContextLength   uint64
	EmbeddingLength uint64
	BlockCount      uint64 // number of transformer blocks (layers)
	HeadCount       uint64
	HeadCountKV     uint64
	ChatTemplate    string // "" if the model embeds none
	NumTensors      int
}

// Stat opens path, reads common metadata, and closes the file.
// Missing keys yield zero values, not errors — GGUF keys vary by architecture.
func Stat(path string) (*Info, error) {
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	info := &Info{
		Architecture:    f.KeyValue("general.architecture").String(),
		Name:            f.KeyValue("general.name").String(),
		FileType:        uint32(f.KeyValue("general.file_type").Uint()),
		ContextLength:   f.KeyValue("context_length").Uint(),
		EmbeddingLength: f.KeyValue("embedding_length").Uint(),
		BlockCount:      f.KeyValue("block_count").Uint(),
		HeadCount:       f.KeyValue("attention.head_count").Uint(),
		HeadCountKV:     f.KeyValue("attention.head_count_kv").Uint(),
		ChatTemplate:    f.KeyValue("tokenizer.chat_template").String(),
		NumTensors:      f.NumTensors(),
	}
	info.Quantization = quantLabel(info.FileType, f)
	return info, nil
}

// fileTypeNames maps the well-known llama_ftype enum values to labels.
var fileTypeNames = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	7:  "Q8_0",
	8:  "Q5_0",
	9:  "Q5_1",
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
}

// quantLabel returns a human quantization label for a file_type enum value.
// Unknown enums fall back to the dominant tensor type, then to "ftype_N".
func quantLabel(ft uint32, f *File) string {
	if name, ok := fileTypeNames[ft]; ok {
		return name
	}
	counts := map[TensorType]int{}
	for _, ti := range f.TensorInfos() {
		counts[ti.Type]++
	}
	best, bestN := TensorType(0), -1
	for tt, n := range counts {
		if n > bestN {
			best, bestN = tt, n
		}
	}
	if bestN <= 0 {
		return fmt.Sprintf("ftype_%d", ft)
	}
	return best.String()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./gguf/ -run TestStat -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gguf/metadata.go gguf/metadata_test.go
git commit -m "feat(gguf): add Stat() metadata convenience layer"
```

---

## Task 4: Real-model smoke test, docs, and final verification

**Files:**
- Modify: `gguf/metadata_test.go` (add optional real-model test)
- Modify: `README.md` (mention the new package)

- [ ] **Step 1: Add an optional real-model smoke test**

Append to `gguf/metadata_test.go`:

```go
import (
	"os"
	"testing"
)

// TestStatRealModel parses an actual GGUF if one is available locally.
// Set LLMARK_TEST_GGUF to a .gguf path to enable; skipped otherwise.
func TestStatRealModel(t *testing.T) {
	path := os.Getenv("LLMARK_TEST_GGUF")
	if path == "" {
		t.Skip("set LLMARK_TEST_GGUF to a .gguf file to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("LLMARK_TEST_GGUF not readable: %v", err)
	}

	info, err := Stat(path)
	if err != nil {
		t.Fatalf("Stat(%s): %v", path, err)
	}
	if info.Architecture == "" {
		t.Error("Architecture empty — metadata parse likely failed")
	}
	if info.NumTensors == 0 {
		t.Error("NumTensors = 0 — tensor block parse likely failed")
	}
	t.Logf("arch=%s name=%q ctx=%d embd=%d blocks=%d quant=%s tensors=%d",
		info.Architecture, info.Name, info.ContextLength, info.EmbeddingLength,
		info.BlockCount, info.Quantization, info.NumTensors)
}
```

Note: the existing `metadata_test.go` already imports `testing`; merge the `os` import into a single import block (do not duplicate the `import "testing"` line).

- [ ] **Step 2: Run the full package test suite**

Run: `go test ./gguf/ -v`
Expected: PASS for all `TestValue*`, `TestTensor*`, `TestOpen*`, `TestStat*`; `TestStatRealModel` SKIPPED (no env var).

Optionally verify against a real model:
Run: `LLMARK_TEST_GGUF="<path to any local .gguf>" go test ./gguf/ -run TestStatRealModel -v`
Expected: PASS with a logged metadata line.

- [ ] **Step 3: Vet and lint**

Run: `go vet ./gguf/`
Expected: no output (clean).

Run: `golangci-lint run ./gguf/...`
Expected: clean. If it flags the lifted files' `LogValue`/unexported-enum-unused patterns, those are intentional from the upstream lift — only fix issues in the NEW files (`metadata.go`, `*_test.go`).

- [ ] **Step 4: Document the package in README**

In `README.md`, add a short subsection under Usage (after the build section, before "## Acceleration"):

```markdown
## Reading GGUF metadata (no model load)

The `gguf` subpackage reads model metadata and tensor info in pure Go — no cgo,
no llama.cpp, no GPU:

```go
import "github.com/go-skynet/go-llama.cpp/gguf"

info, err := gguf.Stat("model.gguf")
// info.Architecture, info.ContextLength, info.BlockCount,
// info.Quantization, info.ChatTemplate, info.NumTensors, ...
```

For lower-level access (per-tensor shapes/types, raw key-values), use
`gguf.Open` and the `*gguf.File` accessors.
```

- [ ] **Step 5: Final commit**

```bash
git add gguf/metadata_test.go README.md
git commit -m "test(gguf): optional real-model smoke test; document gguf package"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- Package layout (`gguf/` + 6 source files) → Tasks 1–3. ✓
- Low-level API (Open/File/KeyValue/TensorInfo/TensorReader/Value/TensorType) → lifted in Tasks 1–2. ✓
- `Stat()`/`Info` convenience layer → Task 3. ✓ (spec's `func Info` corrected to `func Stat` — name collided with the `Info` type.)
- Magic-byte fix → Task 2 Step 4. ✓
- MIT attribution headers → header block applied to every file. ✓
- Test fixture writer (no `fs/ggml` dep) → Task 2 `writer_test.go`. ✓
- Optional real-model test → Task 4. ✓
- Pure-Go, no new deps, no build tags → enforced; `go test ./gguf/` needs nothing external. ✓

**Placeholder scan:** none — every code step shows complete code; the lift steps name exact source/dest paths and the single diff. ✓

**Type consistency:** `Info`, `Stat`, `quantLabel`, `fileTypeNames`, `TensorType`, `TensorInfo`, `Value`, `File`, `ErrUnsupported`, and the test helpers (`writeGGUF`, `sampleModel`, `kvPair`, `testTensor`) are used consistently across tasks. Fixture writer type tags (`wUint32`…) are local to `writer_test.go`. ✓
