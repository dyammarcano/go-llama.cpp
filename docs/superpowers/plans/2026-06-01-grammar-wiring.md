# GBNF Grammar Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `po.Grammar` (`WithGrammar`) option actually constrain generation by adding a GBNF grammar sampler to `make_sampler()` in `binding.cpp`.

**Architecture:** `binding.cpp`-only change: store the already-marshalled `grammar` C-string into a new `binding_params.grammar` field, then add `llama_sampler_init_grammar(vocab, grammar, "root")` to the sampler chain — placed before the terminal mirostat/greedy/truncation branches so every path is constrained. An unparseable grammar makes `make_sampler` return `nullptr`, which `generate()` already turns into an error. Then rebuild `libbinding.a`.

**Tech Stack:** C++17 (`binding.cpp`), upstream `llama_sampler_init_grammar` (`llama.cpp/include/llama.h`), `bash scripts/binding.sh` (MinGW) for the cgo rebuild, Go 1.26 for build/vet/test verification.

**Spec:** `docs/superpowers/specs/2026-06-01-grammar-wiring-design.md`

---

## File Structure

- **Modify** `binding.cpp` — three edits in one logical change: add `binding_params.grammar`; store it in `llama_allocate_params`; add the grammar sampler in `make_sampler`.
- **Create** `docs/grammar-smoke-test.md` — the maintainer's manual GGUF smoke-test procedure.
- **Modify** `docs/OLLAMA-PORTABLE-FEATURES.md` — mark grammar wired in the feature #4 section.

No `binding.h`, `llama.go`, or `options.go` changes (the `grammar` ABI param, `po.Grammar`, and `WithGrammar` already exist). The rebuilt `libbinding.a` is gitignored and not committed. There is no new pure-Go logic, so there are no headless unit tests — verification is compile + existing tests + the manual GGUF smoke test.

---

## Task 1: Wire grammar through `binding.cpp`

**Files:**
- Modify: `binding.cpp` — `binding_params` struct, `llama_allocate_params`, `make_sampler`.

- [ ] **Step 1: Add the `grammar` field to `binding_params`.**

Find the end of the `binding_params` struct. It currently ends:

```cpp
    float    mirostat_eta    = 0.10f;
    float    mirostat_tau    = 5.00f;

    std::vector<std::string>      antiprompt;
    std::vector<llama_logit_bias> logit_bias;
};
```

Change it to (add the `grammar` line):

```cpp
    float    mirostat_eta    = 0.10f;
    float    mirostat_tau    = 5.00f;

    std::string grammar;            // GBNF; "" = unconstrained

    std::vector<std::string>      antiprompt;
    std::vector<llama_logit_bias> logit_bias;
};
```

- [ ] **Step 2: Store the grammar string in `llama_allocate_params`.**

First, remove `(void)grammar;` from the discard statement. The current line (around line 278) is:

```cpp
    (void)tensorsplit; (void)prompt_cache_ro; (void)grammar; (void)rope_freq_base;
```

Change it to (drop `(void)grammar;`):

```cpp
    (void)tensorsplit; (void)prompt_cache_ro; (void)rope_freq_base;
```

Then store the value. Find this line (around line 301):

```cpp
    p->min_p = min_p;
```

Add the grammar assignment immediately after it:

```cpp
    p->min_p = min_p;
    p->grammar = grammar ? std::string(grammar) : std::string();
```

- [ ] **Step 3: Add the grammar sampler in `make_sampler`.**

Find the repetition-penalties block followed by the mirostat comment in `make_sampler`:

```cpp
    // repetition penalties (unchanged condition).
    if (bp->penalty_last_n != 0 &&
        (bp->penalty_repeat != 1.0f || bp->penalty_freq != 0.0f || bp->penalty_present != 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            bp->penalty_last_n, bp->penalty_repeat, bp->penalty_freq, bp->penalty_present));
    }

    // mirostat is terminal: it performs its own temperature + selection, so it
    // replaces the truncation + dist tail entirely.
```

Insert the grammar block between the penalties block's closing `}` and the `// mirostat` comment:

```cpp
    // repetition penalties (unchanged condition).
    if (bp->penalty_last_n != 0 &&
        (bp->penalty_repeat != 1.0f || bp->penalty_freq != 0.0f || bp->penalty_present != 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            bp->penalty_last_n, bp->penalty_repeat, bp->penalty_freq, bp->penalty_present));
    }

    // grammar constraint — masks tokens that violate the GBNF before any
    // terminal sampler, so mirostat / greedy / the truncation tail all sample
    // from grammar-valid tokens. NULL means the grammar failed to parse: free
    // the chain and fail (generate() returns an error on a nullptr sampler).
    if (!bp->grammar.empty()) {
        llama_sampler *gr = llama_sampler_init_grammar(vocab, bp->grammar.c_str(), "root");
        if (gr == nullptr) {
            llama_sampler_free(smpl);
            return nullptr;
        }
        llama_sampler_chain_add(smpl, gr);
    }

    // mirostat is terminal: it performs its own temperature + selection, so it
    // replaces the truncation + dist tail entirely.
```

- [ ] **Step 4: Rebuild the binding.**

Run: `bash scripts/binding.sh`
Expected: recompiles `binding.cpp` into `libbinding.a` with no errors (1-3 min; does not rebuild llama.cpp).

- [ ] **Step 5: Verify the module builds + existing tests pass.**

Run: `go build ./... && go vet . && go test ./streamfilter/ ./gguf/ ./logitbias/ -count=1`
Expected: clean build/link; all pure-Go tests PASS (these packages are unaffected, but confirm no regression).

- [ ] **Step 6: Commit (source only — not the rebuilt artifact).**

```bash
git add binding.cpp
git commit -m "feat(binding): wire GBNF grammar into make_sampler (feature #4 follow-up)"
```

Before committing, run `git status` and confirm `libbinding.a` is NOT staged (it is a gitignored build artifact). Stage ONLY `binding.cpp`.

---

## Task 2: Docs — smoke-test procedure + feature marker

**Files:**
- Create: `docs/grammar-smoke-test.md`
- Modify: `docs/OLLAMA-PORTABLE-FEATURES.md`

- [ ] **Step 1: Create `docs/grammar-smoke-test.md`** with exactly this content:

```markdown
# GBNF grammar wiring — manual smoke test

Requires a built cgo binary (`bash scripts/binding.sh`) and a local GGUF model.
Use `WithGrammar(<gbnf>)` as a predict option. Run from the repo root.

## 1. JSON-object grammar constrains output

Use a GBNF that only admits a JSON object, e.g.:

```
root   ::= "{" ws "\"ok\"" ws ":" ws ("true" | "false") ws "}"
ws     ::= [ \t\n]*
```

Run a `Predict`/`PredictResult` with this grammar. Confirm the output is exactly
a matching JSON object (e.g. `{"ok": true}`) and nothing else.

## 2. Enum grammar

Grammar: `root ::= "true" | "false"`. Confirm the output is only `true` or
`false` — no other tokens.

## 3. Invalid grammar fails loudly

Pass a syntactically broken grammar (e.g. `root ::= "unterminated`). Confirm
`Predict` returns an error (`"inference failed"`) rather than unconstrained text.

## 4. Empty grammar is unchanged

With no `WithGrammar` option (empty `po.Grammar`), confirm generation behaves
exactly as before this feature (unconstrained).
```

- [ ] **Step 2: Mark grammar wired in the feature doc.**

In `docs/OLLAMA-PORTABLE-FEATURES.md`, find the feature #4 section. It contains a line like:

> **GBNF grammar wiring is deferred** as a follow-up — see the design specs:
> `docs/superpowers/specs/2026-05-30-sampler-wiring-design.md` and
> `docs/superpowers/plans/2026-05-30-sampler-wiring.md`.

Immediately after that paragraph (before the next `##` heading), append:

```markdown

**GBNF grammar wired (2026-06-01):** `make_sampler()` now adds
`llama_sampler_init_grammar(vocab, po.Grammar, "root")` to the chain when a
grammar is set, so `WithGrammar(...)` constrains output; an unparseable grammar
fails generation loudly. See
`docs/superpowers/specs/2026-06-01-grammar-wiring-design.md` and
`docs/grammar-smoke-test.md`.
```

(Leave the original "deferred" sentence in place as history; the new note records the change.)

- [ ] **Step 3: Verify nothing broke.**

Run: `go build ./...`
Expected: clean (docs-only change).

- [ ] **Step 4: Commit.**

```bash
git add docs/grammar-smoke-test.md docs/OLLAMA-PORTABLE-FEATURES.md
git commit -m "docs: mark GBNF grammar wired; document smoke test"
```

---

## Self-Review

**Spec coverage:**
- `binding_params.grammar` field → Task 1 Step 1. ✓
- Store grammar in `llama_allocate_params` (remove `(void)grammar`) → Task 1 Step 2. ✓
- Grammar sampler in `make_sampler` after penalties / before mirostat, NULL→`nullptr` → Task 1 Step 3. ✓
- Root hardcoded `"root"` → Task 1 Step 3. ✓
- Rebuild `libbinding.a` → Task 1 Step 4. ✓
- Compile + existing-tests verification → Task 1 Step 5. ✓
- `docs/grammar-smoke-test.md` (JSON / enum / invalid / empty cases) → Task 2 Step 1. ✓
- Feature-doc wired note → Task 2 Step 2. ✓
- No `binding.h`/`llama.go`/`options.go`/Go-API change → respected (no task touches them). ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** the field is `bp->grammar` (a `std::string`) in `make_sampler` and `p->grammar` in `llama_allocate_params` — both name the same `binding_params::grammar` member (`bp`/`p` are just the local pointer names in each function, matching the existing code). `llama_sampler_init_grammar(vocab, bp->grammar.c_str(), "root")` matches the header signature `(const llama_vocab*, const char*, const char*)`. `make_sampler` already returns `llama_sampler*` and `generate()` already handles a `nullptr` return — no new wiring needed.

**Verification reality check:** the runtime behavior (grammar actually constrains output; invalid grammar fails) can only be confirmed with a model — that is the maintainer's manual smoke test (`docs/grammar-smoke-test.md`), not CI.
