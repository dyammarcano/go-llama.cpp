package gguf

import (
	"errors"
	"testing"
)

func TestGroupLayers(t *testing.T) {
	f, err := Open(sampleLlamaModel(t, 4))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	blocks, output := groupLayers(f)
	if len(blocks) != 4 {
		t.Fatalf("blocks = %d, want 4", len(blocks))
	}
	for i := 0; i < 4; i++ {
		if blocks[i] != 4096 { // [64,16] F32
			t.Errorf("blocks[%d] = %d, want 4096", i, blocks[i])
		}
	}
	if output != 4096 { // output.weight [64,16] F32
		t.Errorf("output = %d, want 4096", output)
	}
}

func TestKVBytesPerElement(t *testing.T) {
	cases := map[string]float64{"f16": 2, "": 2, "q8_0": 1, "q4_0": 0.5, "f32": 4}
	for in, want := range cases {
		if got := kvBytesPerElement(in); got != want {
			t.Errorf("kvBytesPerElement(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestEstimateRequiresBudget(t *testing.T) {
	_, err := EstimateLayers(sampleLlamaModel(t, 4), EstimateOptions{FreeVRAM: 0})
	if !errors.Is(err, ErrUnsupported) {
		t.Errorf("err = %v, want ErrUnsupported", err)
	}
}
