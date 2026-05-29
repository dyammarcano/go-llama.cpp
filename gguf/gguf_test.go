// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.

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
