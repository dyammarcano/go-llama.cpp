// Derived from github.com/ollama/ollama (MIT License) memory-estimation logic.
// Adapted for github.com/go-skynet/go-llama.cpp.

package gguf

import (
	"fmt"
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
			rest := name[len("blk."):]
			dot := strings.IndexByte(rest, '.')
			if dot < 0 {
				continue
			}
			idx, err := strconv.Atoi(rest[:dot])
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

// EstimateLayers opens path, reads metadata + tensor sizes, computes the
// estimate, and closes the file. (Full fit logic added in the next task.)
func EstimateLayers(path string, opts EstimateOptions) (*Estimate, error) {
	if opts.FreeVRAM == 0 {
		return nil, fmt.Errorf("%w: FreeVRAM budget required", ErrUnsupported)
	}
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return &Estimate{}, nil
}
