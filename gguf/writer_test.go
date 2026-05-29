// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.

package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// GGUF metadata value type tags (subset emitted by fixtures).
const (
	wUint32  uint32 = 4
	wInt32   uint32 = 5
	wFloat32 uint32 = 6
	wBool    uint32 = 7
	wStr     uint32 = 8
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

func writeStr(b *bytes.Buffer, s string) {
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
		_ = binary.Write(b, le, wStr)
		writeStr(b, x)
	case []string:
		_ = binary.Write(b, le, wArray)
		_ = binary.Write(b, le, wStr)

		_ = binary.Write(b, le, uint64(len(x)))
		for _, s := range x {
			writeStr(b, s)
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
func writeGGUF(tb testing.TB, kvs []kvPair, tensors []testTensor) string {
	tb.Helper()

	le := binary.LittleEndian

	const alignment = 32

	b := new(bytes.Buffer)
	b.WriteString("GGUF")
	_ = binary.Write(b, le, uint32(3))            // version
	_ = binary.Write(b, le, uint64(len(tensors))) // tensor count

	_ = binary.Write(b, le, uint64(len(kvs))) // kv count
	for _, kv := range kvs {
		writeStr(b, kv.Key)
		wValue(tb, b, kv.Val)
	}

	offsets := make([]uint64, len(tensors))

	var dataLen uint64
	for i, t := range tensors {
		dataLen = alignUp(dataLen, alignment)
		offsets[i] = dataLen
		dataLen += uint64(len(t.Data))
	}

	for i, t := range tensors {
		writeStr(b, t.Name)

		_ = binary.Write(b, le, uint32(len(t.Shape)))
		for _, d := range t.Shape {
			_ = binary.Write(b, le, d)
		}

		_ = binary.Write(b, le, t.Type)
		_ = binary.Write(b, le, offsets[i])
	}

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

// sampleLlamaModel builds an n-block llama-arch GGUF fixture with known sizes.
// Each block has one F32 weight tensor of shape [64,16] => 64*16*4 = 4096 bytes.
// The output tensor is [64,16] => 4096 bytes. token_embd is present (tied).
// Metadata uses tiny dims so graph/KV are hand-computable in estimate_test.go:
//
//	embedding=64 head_count=8 head_count_kv=8 key_length=8 value_length=8 vocab=128
//
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
	tensors := make([]testTensor, 0, n+2)

	for i := range n {
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

	defer func() { _ = f.Close() }()

	if f.NumTensors() != 6 { // 4 blocks + token_embd + output
		t.Errorf("NumTensors() = %d, want 6", f.NumTensors())
	}

	if got := f.KeyValue("block_count").Uint(); got != 4 {
		t.Errorf("block_count = %d, want 4", got)
	}
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
			{Name: "token_embd.weight", Type: 0, Shape: []uint64{2, 3}, Data: bytes.Repeat([]byte{0xAB}, 24)},
		},
	)
}
