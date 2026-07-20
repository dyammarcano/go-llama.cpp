// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/dyammarcano/go-llama.cpp.

package gguf

import (
	"bytes"
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

	defer func() { _ = f.Close() }()

	if f.NumTensors() != 1 {
		t.Errorf("NumTensors() = %d, want 1", f.NumTensors())
	}

	if got := f.KeyValue("general.architecture").String(); got != "llama" {
		t.Errorf("architecture = %q, want llama", got)
	}

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

	defer func() { _ = f.Close() }()

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
		t.Errorf("tensor data len %d, want 24x 0xAB", len(data))
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

func TestTensorReaderMultiOffset(t *testing.T) {
	// Two F32 tensors with distinct sentinel bytes. The second tensor's data
	// offset must be the first tensor's size (24) aligned up to 32 -> 32.
	path := writeGGUF(t,
		[]kvPair{{"general.architecture", "llama"}},
		[]testTensor{
			{Name: "a", Type: 0, Shape: []uint64{2, 3}, Data: bytes.Repeat([]byte{0x11}, 24)},
			{Name: "b", Type: 0, Shape: []uint64{1, 4}, Data: bytes.Repeat([]byte{0x22}, 16)},
		},
	)

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	defer func() { _ = f.Close() }()

	bi := f.TensorInfo("b")
	if bi.Offset != 32 {
		t.Errorf("tensor b Offset = %d, want 32", bi.Offset)
	}

	_, rb, err := f.TensorReader("b")
	if err != nil {
		t.Fatalf("TensorReader(b): %v", err)
	}

	db, err := io.ReadAll(rb)
	if err != nil {
		t.Fatalf("ReadAll(b): %v", err)
	}

	if len(db) != 16 || db[0] != 0x22 || db[15] != 0x22 {
		t.Errorf("tensor b data wrong: len=%d first=%#x last=%#x", len(db), db[0], db[len(db)-1])
	}

	_, ra, err := f.TensorReader("a")
	if err != nil {
		t.Fatalf("TensorReader(a): %v", err)
	}

	da, err := io.ReadAll(ra)
	if err != nil {
		t.Fatalf("ReadAll(a): %v", err)
	}

	if len(da) != 24 || da[0] != 0x11 || da[23] != 0x11 {
		t.Errorf("tensor a data wrong: len=%d", len(da))
	}
}
