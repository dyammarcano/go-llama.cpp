// go-llama.cpp binding — MODERNIZATION STEP 1 (stub).
//
// Purpose: prove that binding.cpp compiles and LINKS against the current
// llama.cpp (submodule pinned at 19e92c33) and the cgo/MinGW toolchain,
// WITHOUT changing the C ABI declared in binding.h (so llama.go is untouched).
//
// Only `load_model`/`llama_binding_free_model` call real llama.cpp symbols
// (enough to force the linker to resolve libllama + libggml). Every other
// entry point is a typed stub returning an "unimplemented" code. Real logic
// is ported in later steps — see docs/MODERNIZATION-SCOPE.md.

#include "binding.h"
#include "llama.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace {
// Opaque handle returned by load_model and threaded back through state_pr.
// This is the shape the real binding will grow into (model + ctx + vocab).
struct binding_state {
    llama_model       *model = nullptr;
    llama_context     *ctx   = nullptr;
    const llama_vocab *vocab = nullptr;
};
} // namespace

extern "C" {

void *load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16,
                 bool mlock, bool embeddings, bool mmap, bool low_vram,
                 int n_gpu, int n_batch, const char *maingpu, const char *tensorsplit,
                 bool numa, float rope_freq_base, float rope_freq_scale,
                 bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity) {
    (void)n_seed; (void)memory_f16; (void)mlock; (void)mmap; (void)low_vram;
    (void)maingpu; (void)tensorsplit; (void)mul_mat_q; (void)lora; (void)lora_base;
    (void)perplexity; (void)rope_freq_base; (void)rope_freq_scale;

    llama_backend_init();
    llama_numa_init(numa ? GGML_NUMA_STRATEGY_DISTRIBUTE : GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;

    llama_model *model = llama_model_load_from_file(fname, mparams);
    if (model == nullptr) {
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    if (n_ctx > 0)   { cparams.n_ctx   = (uint32_t)n_ctx; }
    if (n_batch > 0) { cparams.n_batch = (uint32_t)n_batch; }
    cparams.embeddings = embeddings;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        return nullptr;
    }

    binding_state *st = new binding_state();
    st->model = model;
    st->ctx   = ctx;
    st->vocab = llama_model_get_vocab(model);
    return st;
}

void llama_binding_free_model(void *state) {
    binding_state *st = static_cast<binding_state *>(state);
    if (st == nullptr) {
        return;
    }
    if (st->ctx != nullptr)   { llama_free(st->ctx); }
    if (st->model != nullptr) { llama_model_free(st->model); }
    delete st;
}

int llama_predict(void *params_ptr, void *state_pr, char *result, bool debug) {
    (void)params_ptr; (void)debug;
    binding_state *st = static_cast<binding_state *>(state_pr);
    if (st != nullptr && st->ctx != nullptr) {
        (void)llama_n_ctx(st->ctx); // touch a real symbol
    }
    if (result != nullptr) { result[0] = '\0'; }
    return 1; // TODO(step 3): decode loop + common_sampler
}

int eval(void *params_ptr, void *state_pr, char *text) {
    (void)params_ptr; (void)state_pr; (void)text;
    return 1; // TODO(step 2): llama_decode
}

int get_embeddings(void *params_ptr, void *state_pr, float *res_embeddings) {
    (void)params_ptr; (void)state_pr; (void)res_embeddings;
    return 1; // TODO(step 4)
}

int get_token_embeddings(void *params_ptr, void *state_pr, int *tokens, int tokenSize,
                         float *res_embeddings) {
    (void)params_ptr; (void)state_pr; (void)tokens; (void)tokenSize; (void)res_embeddings;
    return 1; // TODO(step 4)
}

int llama_tokenize_string(void *params_ptr, void *state_pr, int *result) {
    (void)params_ptr; (void)state_pr; (void)result;
    return 0; // TODO(step 2): vocab-based llama_tokenize
}

int speculative_sampling(void *params_ptr, void *target_model, void *draft_model,
                         char *result, bool debug) {
    (void)params_ptr; (void)target_model; (void)draft_model; (void)debug;
    if (result != nullptr) { result[0] = '\0'; }
    return 1; // TODO(deferred): common/speculative.h
}

int load_state(void *ctx, char *statefile, char *modes) {
    (void)ctx; (void)statefile; (void)modes;
    return 1; // TODO(step 4): llama_state_load_file
}

void save_state(void *ctx, char *dst, char *modes) {
    (void)ctx; (void)dst; (void)modes;
    // TODO(step 4): llama_state_save_file
}

void *llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty,
                            int repeat_last_n, bool ignore_eos, bool memory_f16,
                            int n_batch, int n_keep, const char **antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty,
                            float presence_penalty, int mirostat, float mirostat_eta,
                            float mirostat_tau, bool penalize_nl, const char *logit_bias,
                            const char *session_file, bool prompt_cache_all, bool mlock, bool mmap,
                            const char *maingpu, const char *tensorsplit, bool prompt_cache_ro,
                            const char *grammar, float rope_freq_base, float rope_freq_scale,
                            float negative_prompt_scale, const char *negative_prompt, int n_draft) {
    (void)prompt; (void)seed; (void)threads; (void)tokens; (void)top_k; (void)top_p; (void)temp;
    (void)repeat_penalty; (void)repeat_last_n; (void)ignore_eos; (void)memory_f16; (void)n_batch;
    (void)n_keep; (void)antiprompt; (void)antiprompt_count; (void)tfs_z; (void)typical_p;
    (void)frequency_penalty; (void)presence_penalty; (void)mirostat; (void)mirostat_eta;
    (void)mirostat_tau; (void)penalize_nl; (void)logit_bias; (void)session_file;
    (void)prompt_cache_all; (void)mlock; (void)mmap; (void)maingpu; (void)tensorsplit;
    (void)prompt_cache_ro; (void)grammar; (void)rope_freq_base; (void)rope_freq_scale;
    (void)negative_prompt_scale; (void)negative_prompt; (void)n_draft;
    // step 1: opaque non-null placeholder; real impl packs common_params (step 2/3).
    return std::calloc(1, sizeof(int));
}

void llama_free_params(void *params_ptr) {
    std::free(params_ptr);
}

} // extern "C"

std::vector<std::string> create_vector(const char **strings, int count) {
    std::vector<std::string> v;
    if (count > 0) {
        v.reserve((size_t)count);
        for (int i = 0; i < count; i++) {
            v.push_back(std::string(strings[i]));
        }
    }
    return v;
}

void delete_vector(std::vector<std::string> *vec) {
    delete vec;
}
