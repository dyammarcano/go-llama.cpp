package llama

// #cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/llama.cpp/include -I${SRCDIR}/llama.cpp/common -I${SRCDIR}/llama.cpp/ggml/include
// #cgo CFLAGS: -I${SRCDIR}/llama.cpp/include
// #cgo LDFLAGS: -L${SRCDIR}/ -lbinding
// #cgo linux LDFLAGS: -fopenmp -lstdc++ -lm
// #cgo darwin LDFLAGS: -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
// #cgo darwin CXXFLAGS: -std=c++17
// #include "binding.h"
// #include <stdlib.h>
import "C"
import (
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"unsafe"

	"github.com/go-skynet/go-llama.cpp/logitbias"
	"github.com/go-skynet/go-llama.cpp/streamfilter"
)

// cstr allocates a C string and returns both the pointer and a free function.
// Use: cs, free := cstr(s); defer free()
func cstr(s string) (*C.char, func()) {
	p := C.CString(s)
	return p, func() { C.free(unsafe.Pointer(p)) }
}

// cLogitBias parses a "id:bias,..." spec into C arrays for llama_allocate_params.
// On empty input or a parse error (logged and skipped — a malformed bias must
// never abort generation) it returns nil pointers, 0, and a no-op free.
func cLogitBias(spec string) (toks *C.int32_t, vals *C.float, count C.int, cleanup func()) {
	noop := func() {}
	entries, err := logitbias.Parse(spec)
	if err != nil {
		slog.Warn("ignoring malformed logit_bias", "spec", spec, "err", err)
		return nil, nil, 0, noop
	}
	n := len(entries)
	if n == 0 {
		return nil, nil, 0, noop
	}
	const maxLogitBias = 1 << 20 // generous ceiling; real vocabularies are far smaller
	if n > maxLogitBias {
		slog.Warn("logit_bias list too large; ignoring", "count", n, "max", maxLogitBias)
		return nil, nil, 0, noop
	}
	tokMem := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int32_t(0))))
	valMem := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.float(0))))
	tokSlice := (*[1 << 28]C.int32_t)(tokMem)[:n:n]
	valSlice := (*[1 << 28]C.float)(valMem)[:n:n]
	for i, e := range entries {
		tokSlice[i] = C.int32_t(e.Token)
		valSlice[i] = C.float(e.Bias)
	}
	return (*C.int32_t)(tokMem), (*C.float)(valMem), C.int(n), func() {
		C.free(tokMem)
		C.free(valMem)
	}
}

type LLama struct {
	state       unsafe.Pointer
	embeddings  bool
	contextSize int
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)
	modelPath := C.CString(model)
	defer C.free(unsafe.Pointer(modelPath))
	loraBase := C.CString(mo.LoraBase)
	defer C.free(unsafe.Pointer(loraBase))
	loraAdapter := C.CString(mo.LoraAdapter)
	defer C.free(unsafe.Pointer(loraAdapter))

	MulMatQ := true

	if mo.MulMatQ != nil {
		MulMatQ = *mo.MulMatQ
	}

	result := C.load_model(modelPath,
		C.int(mo.ContextSize), C.int(mo.Seed),
		C.bool(mo.F16Memory), C.bool(mo.MLock), C.bool(mo.Embeddings), C.bool(mo.MMap), C.bool(mo.LowVRAM),
		C.int(mo.NGPULayers), C.int(mo.NBatch), C.CString(mo.MainGPU), C.CString(mo.TensorSplit), C.bool(mo.NUMA),
		C.float(mo.FreqRopeBase), C.float(mo.FreqRopeScale),
		C.bool(MulMatQ), loraAdapter, loraBase, C.bool(mo.Perplexity),
	)

	if result == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	ll := &LLama{state: result, contextSize: mo.ContextSize, embeddings: mo.Embeddings}
	return ll, nil
}

func (l *LLama) Free() {
	C.llama_binding_free_model(l.state)
}

func (l *LLama) LoadState(state string) error {
	d := C.CString(state)
	w := C.CString("rb")
	result := C.load_state(l.state, d, w)

	defer C.free(unsafe.Pointer(d)) // free allocated C string
	defer C.free(unsafe.Pointer(w)) // free allocated C string

	if result != 0 {
		return fmt.Errorf("error while loading state")
	}

	return nil
}

func (l *LLama) SaveState(dst string) error {
	d := C.CString(dst)
	w := C.CString("wb")

	C.save_state(l.state, d, w)

	defer C.free(unsafe.Pointer(d)) // free allocated C string
	defer C.free(unsafe.Pointer(w)) // free allocated C string

	_, err := os.Stat(dst)
	return err
}

// Token Embeddings
func (l *LLama) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	outSize := po.Tokens
	if po.Tokens == 0 {
		outSize = 9999999
	}

	floats := make([]float32, outSize)

	myArray := (*C.int)(C.malloc(C.size_t(len(tokens)) * C.sizeof_int))

	// Copy the values from the Go slice to the C array
	for i, v := range tokens {
		(*[1<<31 - 1]int32)(unsafe.Pointer(myArray))[i] = int32(v)
	}
	// llama_allocate_params marshals PredictOptions into a C binding_params; see
	// binding.h for the authoritative signature. The trailing params are
	// min_p followed by the parsed logit_bias token/value arrays + count (the
	// legacy const char *logit_bias string param was removed in feature #4).
	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
	params := C.llama_allocate_params(C.CString(""), C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), nil, C.int(0),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)
	ret := C.get_token_embeddings(params, l.state, myArray, C.int(len(tokens)), (*C.float)(&floats[0]))
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}
	return floats, nil
}

// Embeddings
func (l *LLama) Embeddings(text string, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	floats := make([]float32, po.Tokens)
	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)

	ret := C.get_embeddings(params, l.state, (*C.float)(&floats[0]))
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed")
	}

	return floats, nil
}

func (l *LLama) Eval(text string, opts ...PredictOption) error {
	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)
	ret := C.eval(params, l.state, input)
	if ret != 0 {
		return fmt.Errorf("inference failed")
	}

	C.llama_free_params(params)

	return nil
}

func (l *LLama) SpeculativeSampling(ll *LLama, text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(l.state, po.TokenCallback)
	}

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)
	ret := C.speculative_sampling(params, l.state, ll.state, (*C.char)(unsafe.Pointer(&out[0])), C.bool(po.DebugMode))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")

	for _, s := range po.StopPrompts {
		res = strings.TrimRight(res, s)
	}

	C.llama_free_params(params)

	if po.TokenCallback != nil {
		setCallback(l.state, nil)
	}

	return res, nil
}

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	// Go owns stop detection + UTF-8 hold-back. Register a filtering sink that
	// wraps the optional user callback; restore any persistent callback
	// (SetTokenCallback) on return instead of clearing it.
	prev := getCallback(l.state)
	user := po.TokenCallback
	if user == nil {
		user = prev
	}
	sink := streamfilter.NewSink(po.StopPrompts, user)
	setCallback(l.state, sink.OnToken)
	defer setCallback(l.state, prev)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	// C-heap result buffer: it is written by C while the cgo token callback may
	// trigger a Go GC, so it must not live on the (movable) Go heap. Its content
	// is ignored — Go assembles the result from the sink — but llama_predict has
	// no size argument, so the buffer is sized to po.Tokens to avoid an overrun.
	outBuf := C.malloc(C.size_t(po.Tokens))
	if outBuf == nil {
		return "", fmt.Errorf("inference: out of memory allocating %d bytes", po.Tokens)
	}
	defer C.free(outBuf)

	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), nil, C.int(0),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)
	ret := C.llama_predict(params, l.state, (*C.char)(outBuf), C.bool(po.DebugMode))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := sink.Finish()

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")

	C.llama_free_params(params)

	return res, nil
}

// PredictResult generates a completion and returns the full generated text plus
// the number of tokens generated. Unlike Predict, it does not cap the output to
// the token count: it sizes the result buffer to the full length, growing and
// retrying if the first buffer was too small.
func (l *LLama) PredictResult(text string, opts ...PredictOption) (string, int, error) {
	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(l.state, po.TokenCallback)
		defer setCallback(l.state, nil)
	}

	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		cs := C.CString(s)
		reversePrompt[i] = cs
		pass = &reversePrompt[0]
	}

	// Free C strings that cannot be bound to a named variable before the call.
	pathPromptCache, freePathPromptCache := cstr(po.PathPromptCache)
	defer freePathPromptCache()
	mainGPU, freeMainGPU := cstr(po.MainGPU)
	defer freeMainGPU()
	tensorSplit, freeTensorSplit := cstr(po.TensorSplit)
	defer freeTensorSplit()
	grammar, freeGrammar := cstr(po.Grammar)
	defer freeGrammar()
	negativePrompt, freeNegativePrompt := cstr(po.NegativePrompt)
	defer freeNegativePrompt()

	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		pathPromptCache, C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		mainGPU, tensorSplit,
		C.bool(po.PromptCacheRO),
		grammar,
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), negativePrompt,
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)
	defer C.llama_free_params(params)

	// Allocate the result buffer with C.malloc so it lives outside the Go heap.
	// This is required because llama_predict_full calls back into Go (tokenCallback)
	// during generation; a cgo callback re-entering Go can trigger a GC cycle that
	// may move Go heap objects — corrupting the buffer pointer held by C code.
	// A C-heap buffer is never moved by the Go GC.
	var nTok C.int
	size := 32768
	for {
		cbuf := C.malloc(C.size_t(size))
		if cbuf == nil {
			return "", 0, fmt.Errorf("inference: out of memory allocating %d bytes", size)
		}
		full := int(C.llama_predict_full(params, l.state, (*C.char)(cbuf), C.int(size), &nTok, C.bool(po.DebugMode)))
		if full < 0 {
			C.free(cbuf)
			return "", 0, fmt.Errorf("inference failed")
		}
		if full < size {
			// Copy the C buffer into a Go string before freeing.
			result := C.GoStringN((*C.char)(cbuf), C.int(full))
			C.free(cbuf)
			return result, int(nTok), nil
		}
		C.free(cbuf)
		size = full + 1 // output was truncated; grow and retry
	}
}

// ApplyChatTemplate formats a (system, user) pair using the model's embedded
// GGUF chat template. It returns ("", nil) when the model has no template, so
// callers can fall back to raw concatenation.
func (l *LLama) ApplyChatTemplate(system, user string) (string, error) {
	csys := C.CString(system)
	defer C.free(unsafe.Pointer(csys))
	cusr := C.CString(user)
	defer C.free(unsafe.Pointer(cusr))

	size := 8192
	for {
		buf := make([]byte, size)
		n := int(C.apply_chat_template(l.state, csys, cusr, (*C.char)(unsafe.Pointer(&buf[0])), C.int(size)))
		switch {
		case n < 0:
			return "", fmt.Errorf("apply_chat_template failed (%d)", n)
		case n == 0:
			return "", nil // no embedded template
		case n <= size:
			return string(buf[:n]), nil
		default:
			size = n + 1 // buffer too small; grow and retry
		}
	}
}

// tokenize has an interesting return property: negative lengths (potentially) have meaning.
// Therefore, return the length seperate from the slice and error - all three can be used together
func (l *LLama) TokenizeString(text string, opts ...PredictOption) (int32, []int32, error) {
	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 4096 // ???
	}
	out := make([]C.int, po.Tokens)

	var fakeDblPtr **C.char

	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), fakeDblPtr, C.int(0),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)

	tokRet := C.llama_tokenize_string(params, l.state, (*C.int)(unsafe.Pointer(&out[0]))) //, C.int(po.Tokens), true)

	if tokRet < 0 {
		return int32(tokRet), []int32{}, fmt.Errorf("llama_tokenize_string returned negative count %d", tokRet)
	}

	// TODO: Is this loop still required to unbox cgo to go?
	gTokRet := int32(tokRet)

	gLenOut := min(len(out), int(gTokRet))

	goSlice := make([]int32, gLenOut)
	for i := 0; i < gLenOut; i++ {
		goSlice[i] = int32(out[i])
	}

	return gTokRet, goSlice, nil
}

// CGo only allows us to use static calls from C to Go, we can't just dynamically pass in func's.
// This is the next best thing, we register the callbacks in this map and call tokenCallback from
// the C code. We also attach a finalizer to LLama, so it will unregister the callback when the
// garbage collection frees it.

// SetTokenCallback registers a callback for the individual tokens created when running Predict. It
// will be called once for each token. The callback shall return true as long as the model should
// continue predicting the next token. When the callback returns false the predictor will return.
// The tokens are just converted into Go strings, they are not trimmed or otherwise changed. Also
// the tokens may not be valid UTF-8.
// Pass in nil to remove a callback.
//
// It is save to call this method while a prediction is running.
func (l *LLama) SetTokenCallback(callback func(token string) bool) {
	setCallback(l.state, callback)
}

var (
	m         sync.RWMutex
	callbacks = map[uintptr]func(string) bool{}
)

//export tokenCallback
func tokenCallback(statePtr unsafe.Pointer, token *C.char) bool {
	m.RLock()
	defer m.RUnlock()

	if callback, ok := callbacks[uintptr(statePtr)]; ok {
		return callback(C.GoString(token))
	}

	return true
}

// setCallback can be used to register a token callback for LLama. Pass in a nil callback to
// remove the callback.
func setCallback(statePtr unsafe.Pointer, callback func(string) bool) {
	m.Lock()
	defer m.Unlock()

	if callback == nil {
		delete(callbacks, uintptr(statePtr))
	} else {
		callbacks[uintptr(statePtr)] = callback
	}
}

// getCallback returns the token callback currently registered for statePtr, or
// nil if none. Used by the prediction methods to preserve a persistent callback
// (registered via SetTokenCallback) while a per-call filtering sink is active.
func getCallback(statePtr unsafe.Pointer) func(string) bool {
	m.RLock()
	defer m.RUnlock()

	return callbacks[uintptr(statePtr)]
}
