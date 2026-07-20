package llama

import "errors"

// Sentinel errors returned by the binding. Callers can match them with
// errors.Is so failure handling does not depend on message text.
var (
	// ErrModelLoad is returned when a model file cannot be loaded.
	ErrModelLoad = errors.New("llama: failed to load model")

	// ErrStateLoad is returned when a saved context state cannot be restored.
	ErrStateLoad = errors.New("llama: failed to load state")

	// ErrInference is returned when a decode/generation call fails.
	ErrInference = errors.New("llama: inference failed")

	// ErrOutOfMemory is returned when a generation buffer cannot be allocated.
	ErrOutOfMemory = errors.New("llama: out of memory allocating result buffer")

	// ErrEmbeddingsDisabled is returned by embedding calls on a model that was
	// loaded without EnableEmbeddings.
	ErrEmbeddingsDisabled = errors.New("llama: model loaded without embeddings")

	// ErrNotImplemented is returned by API surface that is exported but not yet
	// backed by a real implementation (embeddings, state save/load, speculative
	// sampling). It exists so these calls fail loudly instead of silently
	// returning empty results.
	ErrNotImplemented = errors.New("llama: not implemented")
)
