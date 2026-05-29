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

// fixtureGraph computes the graph + KV + per-block-weight the estimator will use
// for the sampleLlamaModel fixture, so fit assertions don't hard-code the graph
// literal. Dims must match sampleLlamaModel: embedding=64 heads=8 headsKV=8
// key=val=8 vocab=128.
func fixtureGraph(t *testing.T, opts EstimateOptions) (partial, full, kvPerLayer, blockWeight uint64) {
	t.Helper()
	ctx := uint64(orDefault(opts.NumCtx, 2048)) * uint64(orDefault(opts.NumParallel, 1))
	batch := uint64(orDefault(opts.BatchSize, 512))
	full, partial = llamaGraphSize(64, 8, 8, 8, ctx, batch, 128)
	kvPerLayer = uint64(float64(ctx*(8+8)*8) * kvBytesPerElement(opts.KVCacheType))
	blockWeight = 4096
	return partial, full, kvPerLayer, blockWeight
}

func TestEstimateFullOffload(t *testing.T) {
	opts := EstimateOptions{FreeVRAM: 1 << 40, MinReserveBytes: 4096} // 1 TiB
	est, err := EstimateLayers(sampleLlamaModel(t, 4), opts)
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if est.Layers != 4 {
		t.Errorf("Layers = %d, want 4", est.Layers)
	}
	if !est.FullyOffloaded {
		t.Error("FullyOffloaded = false, want true")
	}
	if est.Approximate {
		t.Error("Approximate = true, want false (llama arch)")
	}
}

func TestEstimateNoFit(t *testing.T) {
	est, err := EstimateLayers(sampleLlamaModel(t, 4), EstimateOptions{FreeVRAM: 1024, MinReserveBytes: 4096})
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if est.Layers != 0 || est.FullyOffloaded {
		t.Errorf("Layers=%d FullyOffloaded=%v, want 0/false", est.Layers, est.FullyOffloaded)
	}
}

func TestEstimatePartialFit(t *testing.T) {
	opts := EstimateOptions{NumCtx: 2048, MinReserveBytes: 4096}
	partial, _, kv, w := fixtureGraph(t, opts)
	perLayer := kv + w
	opts.FreeVRAM = 4096 + partial + 2*perLayer + 8 // reserve + graph + exactly 2 layers + slack
	est, err := EstimateLayers(sampleLlamaModel(t, 4), opts)
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if est.Layers != 2 {
		t.Errorf("Layers = %d, want 2 (perLayer=%d partial=%d)", est.Layers, perLayer, partial)
	}
	if est.FullyOffloaded {
		t.Error("FullyOffloaded = true, want false (partial)")
	}
	if est.KVCache != 2*kv {
		t.Errorf("KVCache = %d, want %d", est.KVCache, 2*kv)
	}
}

func TestEstimateKVQuantHalvesF16(t *testing.T) {
	base := EstimateOptions{NumCtx: 2048, MinReserveBytes: 4096, FreeVRAM: 1 << 40}
	_, _, kvF16, _ := fixtureGraph(t, base)
	q := base
	q.KVCacheType = "q8_0"
	_, _, kvQ8, _ := fixtureGraph(t, q)
	if kvQ8 != kvF16/2 {
		t.Errorf("q8_0 kv = %d, want half of f16 %d", kvQ8, kvF16)
	}
}

func TestEstimateApproximateForUnknownArch(t *testing.T) {
	path := writeGGUF(t,
		[]kvPair{
			{"general.architecture", "mistral"},
			{"mistral.block_count", uint32(2)},
			{"mistral.embedding_length", uint32(64)},
			{"mistral.attention.head_count", uint32(8)},
			{"mistral.attention.head_count_kv", uint32(8)},
			{"mistral.vocab_size", uint32(128)},
		},
		[]testTensor{
			{Name: "blk.0.attn.weight", Type: 0, Shape: []uint64{64, 16}},
			{Name: "blk.1.attn.weight", Type: 0, Shape: []uint64{64, 16}},
			{Name: "output.weight", Type: 0, Shape: []uint64{64, 16}},
		},
	)
	est, err := EstimateLayers(path, EstimateOptions{FreeVRAM: 1 << 40, MinReserveBytes: 4096})
	if err != nil {
		t.Fatalf("EstimateLayers: %v", err)
	}
	if !est.Approximate {
		t.Error("Approximate = false, want true (non-llama arch)")
	}
	if est.Layers != 2 {
		t.Errorf("Layers = %d, want 2", est.Layers)
	}
}

func TestEstimateRecurrentUnsupported(t *testing.T) {
	path := writeGGUF(t,
		[]kvPair{
			{"general.architecture", "mamba"},
			{"mamba.block_count", uint32(2)},
			{"mamba.embedding_length", uint32(64)},
		},
		[]testTensor{{Name: "blk.0.x.weight", Type: 0, Shape: []uint64{64, 16}}},
	)
	_, err := EstimateLayers(path, EstimateOptions{FreeVRAM: 1 << 40})
	if !errors.Is(err, ErrUnsupported) {
		t.Errorf("err = %v, want ErrUnsupported", err)
	}
}
