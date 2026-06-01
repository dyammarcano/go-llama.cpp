# GBNF grammar wiring — design

Feature #4's deferred follow-up. Makes the existing `po.Grammar` option actually
constrain generation by adding a GBNF grammar sampler to `make_sampler()` in
`binding.cpp`.

Parent spec: `docs/superpowers/specs/2026-05-30-sampler-wiring-design.md`, which
wired `min_p`/`typical_p`/`mirostat`/`logit_bias` and explicitly deferred grammar
(`grammar` stays `(void)`-discarded; "chain samplers now, grammar follow-up").

## Goal

A user-supplied GBNF grammar string (`WithGrammar(...)` → `po.Grammar`) currently
crosses the cgo ABI but is discarded on the C side (`(void)grammar;` in
`llama_allocate_params`, `binding.cpp:278`). Wire it through so generation is
constrained to the grammar — enabling structured output (e.g. JSON) — using
upstream `llama_sampler_init_grammar`.

## Scope

**Ships now (`binding.cpp` only + `libbinding.a` rebuild):**
- Add a `std::string grammar;` field to `binding_params`.
- In `llama_allocate_params`, store the incoming `grammar` C-string into
  `bp->grammar` (replacing the `(void)grammar;` discard).
- In `make_sampler`, when `bp->grammar` is non-empty, add a grammar sampler to
  the chain (positioned so all sampling paths respect it). On a parse failure
  (NULL return) free the chain and return `nullptr`.
- Rebuild `libbinding.a` via `scripts/binding.sh`.
- Document the manual GGUF smoke test.

**No changes to:** `binding.h` (the `grammar` parameter already exists — ABI
frozen), `llama.go`, `options.go`, or any Go-exported signature. `WithGrammar`
and `po.Grammar` already exist and already marshal the string to C.

**Out of scope / deferred:**
- The `llama_sampler_init_grammar_lazy` / `_lazy_patterns` variants.
- A configurable grammar root symbol (would need a new ABI parameter) — the root
  is hardcoded to `"root"` (GBNF / llama.cpp CLI convention).
- Any Go-side grammar validation or parsing (the grammar string passes straight
  through to llama.cpp, which is the single source of GBNF parsing).

## Upstream API

From `llama.cpp/include/llama.h:1366`:

```c
LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);
```

- Returns an **empty** grammar sampler (non-NULL) if `grammar_str` is empty.
- Returns **NULL** if parsing of `grammar_str` fails.

We only call it when `bp->grammar` is non-empty, so a NULL return unambiguously
means a parse failure.

## Components

### 1. `binding_params` field (`binding.cpp`)

Add alongside the other sampler config fields (after the mirostat fields, before
the `antiprompt` vector):

```cpp
    std::string grammar;        // GBNF; "" = unconstrained
```

### 2. `llama_allocate_params` (`binding.cpp`)

Replace the `(void)grammar;` discard with a store:

```cpp
    p->grammar = grammar ? std::string(grammar) : std::string();
```

(Remove `grammar` from the `(void)...;` discard list.)

### 3. `make_sampler` insert (`binding.cpp`)

Insert immediately **after** the repetition-penalties block and **before** the
mirostat check, so the grammar masks the logits on every downstream path
(mirostat v1/v2, greedy, and the top_k/typical/top_p/min_p/temp/dist tail):

```cpp
    // grammar constraint — masks tokens that violate the GBNF before any
    // terminal sampler. NULL means the grammar failed to parse: fail loudly.
    if (!bp->grammar.empty()) {
        llama_sampler *gr = llama_sampler_init_grammar(vocab, bp->grammar.c_str(), "root");
        if (gr == nullptr) {
            llama_sampler_free(smpl);
            return nullptr;
        }
        llama_sampler_chain_add(smpl, gr);
    }
```

`make_sampler` already returns `llama_sampler *`; `generate()` already checks
`if (smpl == nullptr) { return 1; }`, so a NULL propagates to a generation error
with no further changes.

## Data flow

```
Go: WithGrammar(s) → po.Grammar
      │  C.CString(po.Grammar)  (already marshalled at all 3 call sites)
      ▼
C: llama_allocate_params(..., grammar, ...) → bp->grammar = grammar
      ▼
C: make_sampler(bp, vocab)
      ├─ grammar empty?  → no grammar sampler (unconstrained, as today)
      └─ non-empty → llama_sampler_init_grammar(vocab, grammar, "root")
                       ├─ NULL (parse fail) → free chain, return nullptr
                       └─ ok → chain_add(grammar) before mirostat/greedy/tail
      ▼
C: generate() loop → llama_sampler_sample(smpl, ctx, -1)
      • apply: grammar masks invalid tokens to -inf
      • accept: grammar state advances on the chosen token
      ▼
generate() returns 1 if make_sampler returned nullptr → Go "inference failed"
```

## Sampler-chain position (why early)

Placing the grammar sampler before the terminal branches guarantees:
- **mirostat / greedy** (which early-return) still sample from grammar-masked
  logits.
- the **truncation tail** (top_k/top_p/min_p) operates on grammar-valid tokens,
  so it cannot first discard every grammar-valid token and leave truncation with
  only invalid ones.

Placing it after truncation (the rejected alternative) risks an all-`-inf`
distribution when truncation and the grammar disagree.

## Error handling

- **Empty grammar** (`po.Grammar == ""`): no grammar sampler is added; behavior
  is identical to today (unconstrained). This is the default for all existing
  callers.
- **Invalid grammar** (non-empty, parse fails → NULL): `make_sampler` frees the
  partially-built chain and returns `nullptr`; `generate()` returns 1; the Go
  caller (`Predict`/`PredictResult`/`SpeculativeSampling`) returns
  `"inference failed"`. Chosen over silent skip so a broken constraint never
  yields silently-unconstrained output.
- No memory leak on the failure path: the chain is freed before returning.

## Testing

There is no new pure-Go logic (the grammar string passes straight through to
llama.cpp), so automated verification is **compile + existing tests**:

- `go build ./... && go vet .` clean after the rebuild.
- `go test ./streamfilter/ ./gguf/ ./logitbias/` still green (unaffected).

**Manual GGUF smoke test** (maintainer; requires a built binary + a model),
documented in `docs/grammar-smoke-test.md`:
1. A simple GBNF (e.g. a JSON-object grammar) constrains output to valid JSON.
2. A boolean grammar (`root ::= "true" | "false"`) yields only `true`/`false`.
3. An **invalid** grammar makes `Predict` return an `"inference failed"` error.
4. Empty grammar (`WithGrammar` not set) produces unchanged output.

## Build / integration

- Rebuild after the `binding.cpp` edit: `bash scripts/binding.sh` (recompiles
  `binding.cpp` into `libbinding.a`; does not rebuild llama.cpp).
- `binding.h` untouched → cgo signatures unaffected.

## Risks

- **Rebuild toolchain** (MinGW): mitigated — `libbinding.a` already builds in this
  checkout.
- **Model-dependent validation**: grammar actually constraining output can only
  be confirmed with a real model — the maintainer's manual smoke test, not CI.
- **Grammar performance**: grammar sampling adds per-token masking cost; inherent
  to the feature, not a defect.
