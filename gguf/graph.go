// Derived from github.com/ollama/ollama (MIT License) memory-estimation logic.
// Adapted for github.com/go-skynet/go-llama.cpp.

package gguf

// llamaGraphSize returns the Llama-family compute-buffer sizes in bytes for the
// full-offload and partial-offload cases, following ollama/fs/ggml.go GraphSize.
//
// embedding = n_embd, heads = n_head, embeddingHeads = attention head dim,
// headsKV = n_head_kv, context = NumCtx*NumParallel, batch = BatchSize,
// vocab = vocabulary size. All inputs are token/element counts; the result is
// bytes (the 4* factors are f32 activation bytes, matching upstream).
func llamaGraphSize(embedding, heads, embeddingHeads, headsKV, context, batch, vocab uint64) (full, partial uint64) {
	full = max(
		4*batch*(1+4*embedding+context*(1+heads)),
		4*batch*(embedding+vocab),
	)

	partial = 4*batch*embedding + max(
		4*batch*(1+embedding+max(context, embedding))+
			embedding*embedding*9/16+
			4*context*(batch*heads+embeddingHeads*headsKV),
		4*batch*(embedding+vocab)+embedding*vocab*105/128,
	)

	return full, partial
}
