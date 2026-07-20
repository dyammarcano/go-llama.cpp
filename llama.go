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

// LLama is a handle to a loaded llama.cpp model and its context. It is created
// with New and must be released with Free when no longer needed. A LLama is not
// safe for concurrent use by multiple goroutines.
type LLama struct {
	state       unsafe.Pointer
	embeddings  bool
	contextSize int
}

// New loads the GGUF model at the given path and returns a ready LLama handle.
// Configure loading with ModelOption values (context size, mmap, GPU layers,
// LoRA, and so on). The returned handle must be released with Free.
// It returns ErrModelLoad if the model cannot be loaded.
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
		return nil, fmt.Errorf("%w: %s", ErrModelLoad, model)
	}

	ll := &LLama{state: result, contextSize: mo.ContextSize, embeddings: mo.Embeddings}
	return ll, nil
}

// Free releases the model and its context. It must be called exactly once when
// the handle is no longer needed; the handle must not be used afterwards.
func (l *LLama) Free() {
	C.llama_binding_free_model(l.state)
}

// LoadState restores a previously saved context state from the given file.
//
// Not yet implemented: it returns ErrNotImplemented. The underlying binding is
// a stub, so this call is a loud no-op rather than a silent one.
func (l *LLama) LoadState(state string) error {
	return fmt.Errorf("%w: LoadState", ErrNotImplemented)
}

// SaveState writes the current context state to the given file.
//
// Not yet implemented: it returns ErrNotImplemented. The underlying binding is
// a stub, so this call is a loud no-op rather than a silent one.
func (l *LLama) SaveState(dst string) error {
	return fmt.Errorf("%w: SaveState", ErrNotImplemented)
}

// TokenEmbeddings returns the embedding vector for the given token IDs.
//
// Not yet implemented: it returns ErrNotImplemented (or ErrEmbeddingsDisabled
// if the model was loaded without EnableEmbeddings). The underlying binding is
// a stub, so this fails loudly instead of returning an empty vector.
func (l *LLama) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, ErrEmbeddingsDisabled
	}
	return []float32{}, fmt.Errorf("%w: TokenEmbeddings", ErrNotImplemented)
}

// Embeddings returns the embedding vector for the given text.
//
// Not yet implemented: it returns ErrNotImplemented (or ErrEmbeddingsDisabled
// if the model was loaded without EnableEmbeddings). The underlying binding is
// a stub, so this fails loudly instead of returning an empty vector.
func (l *LLama) Embeddings(text string, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, ErrEmbeddingsDisabled
	}
	return []float32{}, fmt.Errorf("%w: Embeddings", ErrNotImplemented)
}

// Eval runs a forward pass over the given text without returning generated
// output. It is used to warm the context / KV cache. Configure it with
// PredictOption values. It returns ErrInference if the forward pass fails.
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
		return ErrInference
	}

	C.llama_free_params(params)

	return nil
}

// SpeculativeSampling generates text from the target model (the receiver) using
// the given draft model to propose tokens.
//
// Not yet implemented: it returns ErrNotImplemented. The underlying binding is a
// stub, so this fails loudly instead of returning an empty string.
func (l *LLama) SpeculativeSampling(ll *LLama, text string, opts ...PredictOption) (string, error) {
	return "", fmt.Errorf("%w: SpeculativeSampling", ErrNotImplemented)
}

// Predict generates text from the given prompt and returns the completed
// output. Configure sampling and stopping with PredictOption values; use
// SetTokenCallback to stream tokens as they are produced. It returns
// ErrOutOfMemory if the result buffer cannot be allocated, or ErrInference if
// generation fails.
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
	defer C.free(unsafe.Pointer(input))
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	// C-heap result buffer: it is written by C while the cgo token callback may
	// trigger a Go GC, so it must not live on the (movable) Go heap. Its content
	// is ignored — Go assembles the result from the sink — but llama_predict has
	// no size argument, so the buffer is sized to po.Tokens to avoid an overrun.
	outBuf := C.malloc(C.size_t(po.Tokens))
	if outBuf == nil {
		return "", fmt.Errorf("%w: %d bytes", ErrOutOfMemory, po.Tokens)
	}
	defer C.free(outBuf)

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
		C.int(po.Batch), C.int(po.NKeep), nil, C.int(0),
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
	ret := C.llama_predict(params, l.state, (*C.char)(outBuf), C.bool(po.DebugMode))
	if ret != 0 {
		return "", ErrInference
	}
	res := sink.Finish()

	res = strings.TrimPrefix(res, " ")
	res = strings.TrimPrefix(res, text)
	res = strings.TrimPrefix(res, "\n")

	return res, nil
}

// PredictResult generates a completion and returns the full generated text plus
// the number of tokens generated. The text is assembled in Go from the filtering
// sink (stop sequences trimmed, UTF-8 made whole), so it is not capped by any C
// buffer size. n is the number of tokens C produced (including any whose text
// the filter trimmed as part of a stop sequence).
func (l *LLama) PredictResult(text string, opts ...PredictOption) (string, int, error) {
	po := NewPredictOptions(opts...)

	prev := getCallback(l.state)
	user := po.TokenCallback
	if user == nil {
		user = prev
	}
	sink := streamfilter.NewSink(po.StopPrompts, user)
	setCallback(l.state, sink.OnToken)
	defer setCallback(l.state, prev)

	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	if po.Tokens == 0 {
		po.Tokens = 99999999
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
		C.int(po.Batch), C.int(po.NKeep), nil, C.int(0),
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

	// The result text is assembled in Go from the sink, so the C result buffer's
	// content is ignored; only n_tokens is read back. A fixed C-heap buffer is
	// enough: llama_predict_full is size-bounded and never overruns it. The C
	// heap is used (not a Go slice) because the cgo callback can trigger a GC.
	var nTok C.int
	const scratch = 32768
	cbuf := C.malloc(C.size_t(scratch))
	if cbuf == nil {
		return "", 0, fmt.Errorf("%w: %d bytes", ErrOutOfMemory, scratch)
	}
	defer C.free(cbuf)
	if C.llama_predict_full(params, l.state, (*C.char)(cbuf), C.int(scratch), &nTok, C.bool(po.DebugMode)) < 0 {
		return "", 0, ErrInference
	}

	return sink.Finish(), int(nTok), nil
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
