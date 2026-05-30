# Wire stubbed C samplers (min-p / typical / mirostat / logit-bias) — design

Date: 2026-05-30
Status: draft (awaiting user review)
Repo: `github.com/go-skynet/go-llama.cpp`
Roadmap: feature #4 of `docs/OLLAMA-PORTABLE-FEATURES.md`

## Goal

Make the binding's `make_sampler` actually honor the sampler knobs the Go API
already exposes but the C layer currently ignores. Today only `top_k`, `top_p`,
`temp`, the four `penalties`, and `seed` reach the llama.cpp sampler chain. The
options `TypicalP`, `Mirostat`/`MirostatETA`/`MirostatTAU`, and `LogitBias` are
plumbed Go→C and then `(void)`-discarded in `llama_allocate_params`; `min_p` is
not plumbed at all even though it is the single most impactful missing knob
(LM Studio's default sampling that closed our accuracy gap relies on it).

This feature wires four samplers into the chain — **min_p, typical, mirostat
(v1/v2), logit_bias** — with opt-in defaults so existing generation behavior is
byte-for-byte unchanged when the new knobs are left at their defaults.

## Scope

**Ships now:**
- `min_p`: full end-to-end plumbing — new `PredictOptions.MinP`, `SetMinP`
  option, a new ABI argument on `llama_allocate_params`, a new `binding_params`
  field, and a chain entry in `make_sampler`.
- `typical_p`, `mirostat`/`mirostat_eta`/`mirostat_tau`: C-only wiring — already
  received by `llama_allocate_params` (currently discarded). Add
  `binding_params` fields, store them, add chain entries.
- `logit_bias`: a **tested Go parser** (`parseLogitBias`) that turns the
  user-facing `"id:bias,..."` string into parallel arrays passed across the ABI;
  the C side builds a `std::vector<llama_logit_bias>` and adds the chain entry.
- `make_sampler` redesign: new signature `make_sampler(bp, vocab)`, opt-in
  chain ordering (below).
- Headless Go tests for `parseLogitBias`; compile-verify of the C binding.

**Deferred (NOT in this feature):**
- GBNF grammar wiring (`llama_sampler_init_grammar`) — the user chose
  "chain samplers now, grammar follow-up". `grammar` stays `(void)`-discarded.
- The streamfilter cgo wiring (feature #3's deferred contract) — independent.

**Dropped (cannot be wired — removed from upstream llama.cpp sampler API):**
- `tfs_z` (tail-free sampling) — no `llama_sampler_init_tfs` exists. Stays
  `(void)`-discarded; `SetTailFreeSamplingZ` becomes a documented no-op.
- `penalize_nl` — folded into `penalties` upstream; no standalone sampler. Stays
  `(void)`-discarded; `SetPenalizeNL` becomes a documented no-op.

## Non-goals

- No new sampling *algorithms* (no DRY, no XTC) — only wire what the Go API
  already advertises.
- No change to the streaming/stop path, KV handling, or model loading.
- No llmark changes. (llmark's gguf provider hardcodes temp/top_p/top_k today;
  consuming min_p is a separate llmark follow-up.)

## Current state (verified 2026-05-30)

- `binding.cpp:28-45` `struct binding_params`: fields are `prompt, n_predict,
  n_keep, n_batch, n_threads, seed, top_k=40, top_p=0.95, temp=0.80,
  penalty_last_n=64, penalty_repeat=1.0, penalty_freq=0.0, penalty_present=0.0,
  antiprompt`. No min_p / typical_p / mirostat / logit_bias.
- `binding.cpp:90-110` `make_sampler(const binding_params *bp)`: penalties (if
  configured) → `temp<=0 ? greedy : (top_k>0? top_k) → (top_p<1? top_p) → temp →
  dist(seed)`.
- `binding.cpp:133` call site: `llama_sampler *smpl = make_sampler(bp);` —
  `st->vocab` (a `const llama_vocab*`) is in scope here.
- `binding.cpp:230-271` `llama_allocate_params(...)`: receives `tfs_z, typical_p,
  mirostat, mirostat_eta, mirostat_tau, penalize_nl, logit_bias, grammar, ...`
  and `(void)`-casts them at lines 241-245; stores only the fields listed above.
  Does **not** have a `min_p` parameter.
- `llama.go`: 7 identical `C.llama_allocate_params(...)` call sites at lines
  117, 159, 198, 243, 301, 364, 434 (28 args each).
- `options.go:24-58` `PredictOptions`: has `TypicalP, Mirostat, MirostatETA,
  MirostatTAU, PenalizeNL, LogitBias`; no `MinP`. Setters exist for all of those
  (`SetTypicalP`, `SetMirostat`, …) but not `SetMinP`.
- `llama.cpp/include/llama.h` signatures (verbatim):
  - `llama_sampler_init_min_p(float p, size_t min_keep)`
  - `llama_sampler_init_typical(float p, size_t min_keep)`
  - `llama_sampler_init_mirostat(int32_t n_vocab, uint32_t seed, float tau, float eta, int32_t m)`
  - `llama_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta)`
  - `llama_sampler_init_penalties(int32_t last_n, float repeat, float freq, float present)`
  - `llama_sampler_init_logit_bias(int32_t n_vocab, int32_t n_logit_bias, const llama_logit_bias* arr)`
  - `typedef struct llama_logit_bias { llama_token token; float bias; } llama_logit_bias;`
  - `int32_t llama_vocab_n_tokens(const llama_vocab* vocab);`

## Design

### 1. `min_p` end-to-end plumbing

- **Go option** (`options.go`): add `MinP float32` to `PredictOptions`
  (zero-value `0` = disabled — no constructor/default change needed) and:
  ```go
  // SetMinP sets the min-p sampling cutoff (0 = disabled, llama.cpp default).
  func SetMinP(minp float32) PredictOption {
      return func(po *PredictOptions) { po.MinP = minp }
  }
  ```
- **ABI** (`binding.h` + `binding.cpp` `llama_allocate_params`): append a new
  trailing argument `float min_p` (append-only keeps the change easy to read).
  Store `p->min_p = min_p;`.
- **cgo** (`llama.go`): add `C.float(po.MinP)` as the matching trailing argument
  at all **7** call sites.
- **struct** (`binding.cpp`): add `float min_p = 0.0f;` to `binding_params`.

### 2. `typical_p` / `mirostat` C-only wiring

These already arrive at `llama_allocate_params`; stop discarding them.

- Add to `binding_params`:
  ```cpp
  float   min_p        = 0.0f;   // (from §1)
  float   typical_p    = 1.0f;   // 1.0 = disabled
  int32_t mirostat     = 0;      // 0 = disabled, 1 = v1, 2 = v2
  float   mirostat_eta = 0.10f;
  float   mirostat_tau = 5.00f;
  ```
- In `llama_allocate_params`: remove the `(void)` casts for `typical_p,
  mirostat, mirostat_eta, mirostat_tau` (keep `tfs_z`, `penalize_nl`, `grammar`,
  `negative_prompt*`, `rope_*` discarded) and store:
  ```cpp
  p->typical_p    = typical_p;
  p->mirostat     = mirostat;
  p->mirostat_eta = mirostat_eta;
  p->mirostat_tau = mirostat_tau;
  ```

### 3. `logit_bias` — Go parser + parallel-array ABI

The user-facing format stays `"<tokenID>:<bias>[,<tokenID>:<bias>...]"`
(unchanged from today's `SetLogitBias(string)`), but Go now owns parsing so it is
headlessly testable and the C side stays trivial.

- **Go** (`options.go` or new `logitbias.go`):
  ```go
  // LogitBiasEntry is one parsed token-bias pair.
  type LogitBiasEntry struct {
      Token int32
      Bias  float32
  }

  // parseLogitBias parses "id:bias,id:bias" into entries. Whitespace around
  // ids/biases is tolerated; empty input yields nil, nil. A malformed pair
  // returns a non-nil error naming the offending segment. Later duplicates of
  // the same token override earlier ones (last-wins), matching llama.cpp.
  func parseLogitBias(s string) ([]LogitBiasEntry, error)
  ```
  `SetLogitBias` keeps its `(string)` signature (option setters cannot return
  errors); the string is parsed in `llama.go` just before the C call. On parse
  error the bias is skipped and a one-line `log` warning is emitted (generation
  still proceeds) — malformed bias must never abort a benchmark run.
- **ABI**: replace the discarded `const char *logit_bias` parameter of
  `llama_allocate_params` with three parameters
  `const int32_t *logit_bias_tokens, const float *logit_bias_values,
  int logit_bias_count` (same slot, repurposed — the fork owns both sides).
- **struct**: `std::vector<llama_logit_bias> logit_bias;` in `binding_params`;
  `llama_allocate_params` fills it from the arrays:
  ```cpp
  for (int i = 0; i < logit_bias_count; i++) {
      p->logit_bias.push_back({ (llama_token)logit_bias_tokens[i],
                                logit_bias_values[i] });
  }
  ```
- **cgo** (`llama.go`): at each of the 7 sites, build the arrays from
  `parseLogitBias(po.LogitBias)` and pass `(*C.int32_t)(tokensPtr),
  (*C.float)(valuesPtr), C.int(len)`. Empty → pass `nil, nil, 0`. Pin the
  slices for the duration of the call (they are consumed synchronously inside
  `llama_allocate_params`, which copies into the vector, so no long-term pinning
  is required).

### 4. `make_sampler` redesign

New signature and chain. `st->vocab` is passed so mirostat-v1 and logit_bias can
read `n_vocab`.

```cpp
llama_sampler *make_sampler(const binding_params *bp, const llama_vocab *vocab) {
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // logit_bias first — bias raw logits before any truncation.
    if (!bp->logit_bias.empty()) {
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
            n_vocab, (int32_t)bp->logit_bias.size(), bp->logit_bias.data()));
    }

    // penalties (unchanged condition).
    if (bp->penalty_last_n != 0 &&
        (bp->penalty_repeat != 1.0f || bp->penalty_freq != 0.0f || bp->penalty_present != 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            bp->penalty_last_n, bp->penalty_repeat, bp->penalty_freq, bp->penalty_present));
    }

    // mirostat is terminal: it performs its own temperature+selection, so it
    // replaces the truncation+dist tail entirely.
    if (bp->mirostat == 1) {
        const float eta = bp->mirostat_eta > 0.0f ? bp->mirostat_eta : 0.10f;
        const float tau = bp->mirostat_tau > 0.0f ? bp->mirostat_tau : 5.00f;
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(n_vocab, bp->seed, tau, eta, 100));
        return smpl;
    }
    if (bp->mirostat == 2) {
        const float eta = bp->mirostat_eta > 0.0f ? bp->mirostat_eta : 0.10f;
        const float tau = bp->mirostat_tau > 0.0f ? bp->mirostat_tau : 5.00f;
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(bp->seed, tau, eta));
        return smpl;
    }

    if (bp->temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        return smpl;
    }

    // truncation tail (each entry opt-in; defaults reproduce today's chain).
    if (bp->top_k > 0)        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(bp->top_k));
    if (bp->typical_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_typical(bp->typical_p, 1));
    if (bp->top_p < 1.0f)     llama_sampler_chain_add(smpl, llama_sampler_init_top_p(bp->top_p, 1));
    if (bp->min_p > 0.0f)     llama_sampler_chain_add(smpl, llama_sampler_init_min_p(bp->min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(bp->temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(bp->seed));
    return smpl;
}
```

Call site `binding.cpp:133` becomes `make_sampler(bp, st->vocab)`.

**Behavior-preservation check:** with defaults (`min_p=0`, `typical_p=1`,
`mirostat=0`, `logit_bias` empty), the chain reduces to penalties →
`top_k → top_p → temp → dist` — identical ordering and conditions to the current
`make_sampler`. No regression for existing callers.

## Error handling

- `parseLogitBias` returns an error for malformed segments; the binding logs and
  skips (never aborts generation).
- `make_sampler` is total — every knob is guarded; unknown `mirostat` values
  (anything not 1/2) fall through to the normal tail.
- Mirostat with unset eta/tau falls back to llama.cpp defaults (0.10 / 5.00).

## Testing

**Headless Go (no model, runs in CI):**
- `parseLogitBias` table tests: single pair, multiple pairs, surrounding
  whitespace, empty string (→ nil,nil), negative bias, malformed (`"abc"`,
  `"5:"`, `"5:x"`, `":1"`) → error, duplicate token last-wins.
- Compile/vet of the package with the new option present (`SetMinP` reachable).

**Compile-verify of the C binding (developer machine, not CI):**
- `bash .scripts/build-binding.sh` (CPU) must compile and link with the new
  `make_sampler` signature, the repurposed ABI, and the new struct fields.
- `go build ./...` of the fork.

**User-run, env-gated real-model smoke test (documented, not automated):**
- Greedy (`temp=0`) output is unchanged vs. pre-change (behavior preservation).
- `SetMinP(0.05)` with `temp>0` produces coherent text.
- `SetMirostat(2)` produces coherent text.
- `SetLogitBias("<id>:-100")` for a token that otherwise appears → that token
  never appears in the output (the load-bearing logit_bias assertion).

Lint: golangci-lint `default:all` clean for the Go changes
(`options.go`, the new parser + its test).

## Risks

- **ABI churn across 7 cgo call sites.** Mechanical but must be applied
  uniformly; a missed site fails to compile (caught immediately), so low risk of
  silent breakage.
- **No automated C-side test.** The fork has no C++ test harness; the logit_bias
  and chain wiring are validated only by compile + the user smoke test. The Go
  parser carries the testable logic; the C consumption is a trivial loop.
- **mirostat-v1's `m` parameter** is hardcoded to 100 (llama.cpp's conventional
  value); not exposed as a knob (YAGNI).
- Repurposing the `logit_bias` ABI slot from `const char*` to three array params
  is an internal-only change (the fork owns both sides); no external consumer
  depends on `llama_allocate_params`'s C signature.
