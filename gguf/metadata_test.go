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
	if info.Quantization != "Q8_0" {
		t.Errorf("Quantization = %q, want Q8_0", info.Quantization)
	}
}

func TestStatQuantFallbackUppercase(t *testing.T) {
	// file_type 99 is not in fileTypeNames, forcing the dominant-tensor fallback.
	path := writeGGUF(t,
		[]kvPair{
			{"general.architecture", "llama"},
			{"general.file_type", uint32(99)},
			{"llama.block_count", uint32(1)},
		},
		[]testTensor{
			{Name: "blk.0.attn_q.weight", Type: 2, Shape: []uint64{32}, Data: make([]byte, 18)}, // Q4_0
			{Name: "blk.0.attn_k.weight", Type: 2, Shape: []uint64{32}, Data: make([]byte, 18)}, // Q4_0
		},
	)
	info, err := Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.Quantization != "Q4_0" {
		t.Errorf("Quantization = %q, want Q4_0 (uppercase, from dominant tensor fallback)", info.Quantization)
	}
}
