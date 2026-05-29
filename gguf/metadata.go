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
