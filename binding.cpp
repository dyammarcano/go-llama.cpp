// go-llama.cpp binding — MODERNIZATION STEPS 2-3.
//
// Real implementation against current llama.cpp (submodule 19e92c33). The C ABI
// in binding.h is unchanged, so llama.go is untouched. Sampling is delegated to
// common_sampler (common/sampling.h); generation uses llama_decode + the
// llama_batch_get_one helper. Embeddings, state save/load and speculative
// sampling remain stubs (step 4 / deferred) — see docs/MODERNIZATION-SCOPE.md.

#include "binding.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Opaque handle returned by load_model and threaded back through state_pr.
struct binding_state {
    llama_model       *model  = nullptr;
    llama_context     *ctx    = nullptr;
    const llama_vocab *vocab  = nullptr;
    int                n_past = 0;
};

// Opaque handle returned by llama_allocate_params.
struct binding_params {
    std::string prompt;
    int         n_predict = 128;
    int         n_keep    = 0;
    int         n_batch   = 512;
    int         n_threads = 0;
    std::vector<std::string> antiprompt;
    common_params_sampling   sparams;
};

// Decode a token span in n_batch chunks. Position is tracked automatically by
// llama_decode. Returns 0 on success.
int decode_tokens(llama_context *ctx, const std::vector<llama_token> &toks, int n_batch) {
    if (n_batch <= 0) {
        n_batch = 512;
    }
    for (size_t i = 0; i < toks.size(); i += (size_t)n_batch) {
        const int n = (int)std::min((size_t)n_batch, toks.size() - i);
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data() + i), n);
        if (llama_decode(ctx, b) != 0) {
            return 1;
        }
    }
    return 0;
}

} // namespace

extern "C" {

void *load_model(const char *fname, int n_ctx, int n_seed, bool memory_f16,
                 bool mlock, bool embeddings, bool mmap, bool low_vram,
                 int n_gpu, int n_batch, const char *maingpu, const char *tensorsplit,
                 bool numa, float rope_freq_base, float rope_freq_scale,
                 bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity) {
    (void)n_seed; (void)memory_f16; (void)mlock; (void)mmap; (void)low_vram;
    (void)maingpu; (void)tensorsplit; (void)mul_mat_q; (void)lora; (void)lora_base;
    (void)perplexity;

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
    if (rope_freq_base  > 0.0f) { cparams.rope_freq_base  = rope_freq_base; }
    if (rope_freq_scale > 0.0f) { cparams.rope_freq_scale = rope_freq_scale; }

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        return nullptr;
    }

    binding_state *st = new binding_state();
    st->model = model;
    st->ctx   = ctx;
    st->vocab = llama_model_get_vocab(model);
    st->n_past = 0;
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
    (void)memory_f16; (void)penalize_nl; (void)logit_bias; (void)session_file;
    (void)prompt_cache_all; (void)mlock; (void)mmap; (void)maingpu; (void)tensorsplit;
    (void)prompt_cache_ro; (void)rope_freq_base; (void)rope_freq_scale;
    (void)negative_prompt_scale; (void)negative_prompt; (void)n_draft; (void)tfs_z;

    binding_params *p = new binding_params();
    p->prompt    = prompt ? std::string(prompt) : std::string();
    p->n_threads = threads;
    p->n_predict = tokens;
    p->n_batch   = n_batch > 0 ? n_batch : 512;
    p->n_keep    = n_keep;

    if (seed >= 0) {
        p->sparams.seed = (uint32_t)seed;
    }
    p->sparams.top_k           = top_k;
    p->sparams.top_p           = top_p;
    p->sparams.typ_p           = typical_p;
    p->sparams.temp            = temp;
    p->sparams.penalty_last_n  = repeat_last_n;
    p->sparams.penalty_repeat  = repeat_penalty;
    p->sparams.penalty_freq    = frequency_penalty;
    p->sparams.penalty_present = presence_penalty;
    p->sparams.mirostat        = mirostat;
    p->sparams.mirostat_eta    = mirostat_eta;
    p->sparams.mirostat_tau    = mirostat_tau;
    p->sparams.ignore_eos      = ignore_eos;
    if (grammar != nullptr && grammar[0] != '\0') {
        p->sparams.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, std::string(grammar));
    }

    if (antiprompt != nullptr && antiprompt_count > 0) {
        for (int i = 0; i < antiprompt_count; i++) {
            if (antiprompt[i] != nullptr) {
                p->antiprompt.push_back(std::string(antiprompt[i]));
            }
        }
    }
    return p;
}

void llama_free_params(void *params_ptr) {
    delete static_cast<binding_params *>(params_ptr);
}

int eval(void *params_ptr, void *state_pr, char *text) {
    binding_params *bp = static_cast<binding_params *>(params_ptr);
    binding_state  *st = static_cast<binding_state *>(state_pr);
    if (st == nullptr || st->ctx == nullptr || text == nullptr) {
        return 1;
    }
    if (bp != nullptr && bp->n_threads > 0) {
        llama_set_n_threads(st->ctx, bp->n_threads, bp->n_threads);
    }
    std::vector<llama_token> toks =
        common_tokenize(st->ctx, std::string(text), /*add_special*/ false, /*parse_special*/ true);
    const int rc = decode_tokens(st->ctx, toks, bp ? bp->n_batch : 512);
    if (rc == 0) {
        st->n_past += (int)toks.size();
    }
    return rc;
}

int llama_predict(void *params_ptr, void *state_pr, char *result, bool debug) {
    binding_params *bp = static_cast<binding_params *>(params_ptr);
    binding_state  *st = static_cast<binding_state *>(state_pr);
    if (bp == nullptr || st == nullptr || st->ctx == nullptr || result == nullptr) {
        return 1;
    }

    if (bp->n_threads > 0) {
        llama_set_n_threads(st->ctx, bp->n_threads, bp->n_threads);
    }

    // Fresh context for this prediction.
    llama_memory_clear(llama_get_memory(st->ctx), true);
    st->n_past = 0;

    std::vector<llama_token> toks =
        common_tokenize(st->ctx, bp->prompt, /*add_special*/ true, /*parse_special*/ true);

    const int n_ctx = (int)llama_n_ctx(st->ctx);
    if ((int)toks.size() >= n_ctx) {
        result[0] = '\0';
        return 1;
    }
    if (decode_tokens(st->ctx, toks, bp->n_batch) != 0) {
        result[0] = '\0';
        return 1;
    }
    st->n_past = (int)toks.size();

    common_sampler *smpl = common_sampler_init(st->model, bp->sparams);
    if (smpl == nullptr) {
        result[0] = '\0';
        return 1;
    }

    std::string out;
    const int n_predict = bp->n_predict > 0 ? bp->n_predict : 128;
    bool stop = false;

    for (int i = 0; i < n_predict && st->n_past < n_ctx && !stop; i++) {
        llama_token id = common_sampler_sample(smpl, st->ctx, -1);
        common_sampler_accept(smpl, id, /*is_generated*/ true);

        if (llama_vocab_is_eog(st->vocab, id)) {
            break;
        }

        const std::string piece = common_token_to_piece(st->ctx, id, /*special*/ false);
        out += piece;
        if (debug) {
            fprintf(stderr, "%s", piece.c_str());
            fflush(stderr);
        }

        // Per-token callback (registered on the Go side, keyed by state pointer).
        if (!piece.empty()) {
            std::vector<char> buf(piece.begin(), piece.end());
            buf.push_back('\0');
            if (tokenCallback(st, buf.data()) == 0) {
                stop = true;
            }
        }

        // Stop on any anti-prompt suffix.
        for (const std::string &a : bp->antiprompt) {
            if (!a.empty() && out.size() >= a.size() &&
                out.compare(out.size() - a.size(), a.size(), a) == 0) {
                stop = true;
                break;
            }
        }
        if (stop) {
            break;
        }

        // Feed the sampled token back into the context.
        llama_batch b = llama_batch_get_one(&id, 1);
        if (llama_decode(st->ctx, b) != 0) {
            break;
        }
        st->n_past++;
    }

    common_sampler_free(smpl);

    // Inherited ABI: caller (Go) owns a result buffer sized to po.Tokens bytes.
    std::strcpy(result, out.c_str());
    return 0;
}

int llama_tokenize_string(void *params_ptr, void *state_pr, int *result) {
    binding_params *bp = static_cast<binding_params *>(params_ptr);
    binding_state  *st = static_cast<binding_state *>(state_pr);
    if (bp == nullptr || st == nullptr || st->ctx == nullptr || result == nullptr) {
        return -1;
    }
    std::vector<llama_token> toks =
        common_tokenize(st->ctx, bp->prompt, /*add_special*/ true, /*parse_special*/ true);
    for (size_t i = 0; i < toks.size(); i++) {
        result[i] = (int)toks[i];
    }
    return (int)toks.size();
}

// ---- step 4 / deferred stubs ----

int get_embeddings(void *params_ptr, void *state_pr, float *res_embeddings) {
    (void)params_ptr; (void)state_pr; (void)res_embeddings;
    return 1; // TODO(step 4): llama_get_embeddings_seq + pooling
}

int get_token_embeddings(void *params_ptr, void *state_pr, int *tokens, int tokenSize,
                         float *res_embeddings) {
    (void)params_ptr; (void)state_pr; (void)tokens; (void)tokenSize; (void)res_embeddings;
    return 1; // TODO(step 4)
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
