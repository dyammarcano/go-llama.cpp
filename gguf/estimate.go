// Derived from github.com/ollama/ollama (MIT License) memory-estimation logic.
// Adapted for github.com/go-skynet/go-llama.cpp.

package gguf

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// DefaultMinReserveBytes mirrors Ollama's non-Metal DeviceInfo.MinimumMemory.
const DefaultMinReserveBytes uint64 = 457 << 20

// EstimateOptions parameterizes a layer-fit estimate.
type EstimateOptions struct {
	NumCtx          int    // context length in tokens (0 => 2048)
	BatchSize       int    // tokens per batch (0 => 512)
	NumParallel     int    // concurrent sequences (0 => 1)
	KVCacheType     string // "f16" (default), "q8_0", "q4_0", "f32"
	FreeVRAM        uint64 // caller-provided budget in bytes (required, > 0)
	OverheadBytes   uint64 // extra reserve on top of the minimum reserve
	MinReserveBytes uint64 // minimum reserve floor; 0 => DefaultMinReserveBytes
	FlashAttention  bool   // accepted; does not yet alter the formula
}

// Estimate is the result of a layer-fit computation.
type Estimate struct {
	Layers         int      // recommended n_gpu_layers (repeating blocks offloaded)
	FullyOffloaded bool     // true if all blocks + the output layer fit
	TotalVRAM      uint64   // weights+kv+graph actually placed on the GPU
	Weights        uint64   // offloaded block weights (+ output if FullyOffloaded)
	KVCache        uint64   // KV bytes for the offloaded blocks
	Graph          uint64   // compute buffer (full- or partial-offload value)
	PerLayerBytes  []uint64 // weights+kv per block, index 0..BlockCount-1
	Approximate    bool     // true when a non-llama arch used the fallback formula
}

// kvBytesPerElement returns the per-element KV-cache byte size for a cache type.
func kvBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "q8_0":
		return 1
	case "q4_0":
		return 0.5
	case "f32":
		return 4
	default: // "f16" and unknown
		return 2
	}
}

// groupLayers sums repeating blk.<i>.* tensor bytes per block and the
// non-repeating output layer (output[_norm], else tied token_embd).
func groupLayers(f *File) (blocks map[int]uint64, output uint64) {
	blocks = make(map[int]uint64)

	var outputW, tokenEmbd uint64

	for _, ti := range f.TensorInfos() {
		name := ti.Name

		switch {
		case strings.HasPrefix(name, "blk."):
			idxStr, _, ok := strings.Cut(name[len("blk."):], ".")
			if !ok {
				continue
			}

			idx, err := strconv.Atoi(idxStr)
			if err != nil {
				continue
			}

			blocks[idx] += uint64(ti.NumBytes())
		case strings.HasPrefix(name, "output_norm"), strings.HasPrefix(name, "output."):
			outputW += uint64(ti.NumBytes())
		case strings.HasPrefix(name, "token_embd"):
			tokenEmbd += uint64(ti.NumBytes())
		}
	}

	if outputW == 0 {
		outputW = tokenEmbd // tied embeddings
	}

	return blocks, outputW
}

func orDefault(v, def int) int {
	if v <= 0 {
		return def
	}

	return v
}

// EstimateLayers opens path, reads metadata + tensor sizes, computes how many
// transformer blocks fit in opts.FreeVRAM, and closes the file. Single-GPU,
// Llama-family graph model; non-llama dense architectures get an approximate
// estimate (Estimate.Approximate). Recurrent/SSM architectures are unsupported.
func EstimateLayers(path string, opts EstimateOptions) (*Estimate, error) {
	if opts.FreeVRAM == 0 {
		return nil, fmt.Errorf("%w: FreeVRAM budget required", ErrUnsupported)
	}

	f, err := Open(path)
	if err != nil {
		return nil, err
	}

	defer func() { _ = f.Close() }()

	// KeyValue auto-prefixes the architecture for keys not starting with
	// "general."/"tokenizer.", so "attention.head_count" resolves to
	// "<arch>.attention.head_count".
	arch := f.KeyValue("general.architecture").String()
	blockCount := int(f.KeyValue("block_count").Uint())

	embedding := f.KeyValue("embedding_length").Uint()
	if blockCount == 0 || embedding == 0 {
		return nil, fmt.Errorf("%w: missing block_count/embedding_length", ErrUnsupported)
	}

	heads := f.KeyValue("attention.head_count").Uint()
	if heads == 0 {
		return nil, fmt.Errorf("%w: recurrent or unsupported architecture (no attention heads)", ErrUnsupported)
	}

	headsKV := f.KeyValue("attention.head_count_kv").Uint()
	if headsKV == 0 {
		headsKV = heads
	}

	keyLen := f.KeyValue("attention.key_length").Uint()
	if keyLen == 0 {
		keyLen = embedding / heads
	}

	valLen := f.KeyValue("attention.value_length").Uint()
	if valLen == 0 {
		valLen = embedding / heads
	}

	vocab := f.KeyValue("vocab_size").Uint()
	if vocab == 0 {
		vocab = uint64(len(f.KeyValue("tokenizer.ggml.tokens").Strings()))
	}

	ctx := uint64(orDefault(opts.NumCtx, 2048)) * uint64(orDefault(opts.NumParallel, 1))
	batch := uint64(orDefault(opts.BatchSize, 512))

	blocks, output := groupLayers(f)
	kvPerLayer := uint64(float64(ctx*(keyLen+valLen)*headsKV) * kvBytesPerElement(opts.KVCacheType))
	full, partial := llamaGraphSize(embedding, heads, keyLen, headsKV, ctx, batch, vocab)

	perLayer := make([]uint64, blockCount)

	for i := range blockCount {
		perLayer[i] = blocks[i] + kvPerLayer
	}

	minReserve := opts.MinReserveBytes
	if minReserve == 0 {
		minReserve = DefaultMinReserveBytes
	}

	reserve := minReserve + opts.OverheadBytes

	est := &Estimate{PerLayerBytes: perLayer, Approximate: arch != "llama"}

	// fill repeating blocks largest-cost-first while they fit under the
	// partial-offload graph budget. Dense blocks are equal-sized, so order only
	// matters under heterogeneity; largest-first biases conservative.
	order := make([]int, blockCount)
	for i := range order {
		order[i] = i
	}

	sort.SliceStable(order, func(a, b int) bool { return perLayer[order[a]] > perLayer[order[b]] })

	avail := int64(opts.FreeVRAM) - int64(reserve) - int64(partial)

	var weightsUsed, kvUsed uint64

	n := 0

	if avail > 0 {
		for _, i := range order {
			if int64(weightsUsed+kvUsed+perLayer[i]) <= avail {
				weightsUsed += blocks[i]
				kvUsed += kvPerLayer
				n++
			} else {
				break
			}
		}
	}

	est.Layers = n
	est.Graph = partial

	if n == blockCount {
		// all blocks fit: try to also offload the output layer under the
		// (larger) full-offload graph budget.
		if int64(opts.FreeVRAM)-int64(reserve)-int64(full)-int64(weightsUsed+kvUsed+output) >= 0 {
			est.FullyOffloaded = true
			est.Graph = full
			weightsUsed += output
		}
	}

	est.Weights = weightsUsed
	est.KVCache = kvUsed
	est.TotalVRAM = est.Weights + est.KVCache + est.Graph

	return est, nil
}
