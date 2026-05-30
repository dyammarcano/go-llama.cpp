# Sampler Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the binding's `make_sampler` honor `min_p`, `typical_p`, `mirostat` (v1/v2), and `logit_bias` — knobs the Go API exposes but the C layer currently discards — with opt-in defaults so existing generation is unchanged.

**Architecture:** A new cgo-free `logitbias/` subpackage owns the `"id:bias,..."` parser (headless-tested, like the existing `gguf/` and `streamfilter/` packages). The C binding gains `binding_params` fields, a redesigned `make_sampler(bp, vocab)` chain, and a widened `llama_allocate_params` ABI; `llama.go` marshals parsed biases into C arrays at its 7 call sites. `tfs_z`/`penalize_nl` are dropped (removed upstream); GBNF grammar is deferred.

**Tech Stack:** Go 1.25 (cgo, MinGW on Windows), C++17 binding over llama.cpp, golangci-lint v2 (`default: all`).

**Spec:** `docs/superpowers/specs/2026-05-30-sampler-wiring-design.md`

---

## File Structure

- **Create** `logitbias/logitbias.go` — `Entry` type + `Parse(string) ([]Entry, error)`. Pure Go, no cgo, stdlib only (`strconv`, `strings`, `fmt`). One responsibility: the bias-string grammar.
- **Create** `logitbias/logitbias_test.go` — table tests for `Parse`.
- **Modify** `binding.cpp` — add `binding_params` fields; store typical/mirostat (Task 2) and min_p/logit_bias (Task 3) in `llama_allocate_params`; redesign `make_sampler`.
- **Modify** `binding.h` — widen the `llama_allocate_params` ABI (Task 3); ensure `<stdint.h>`.
- **Modify** `options.go` — add `MinP` field + `SetMinP`; document `SetTailFreeSamplingZ`/`SetPenalizeNL` as no-ops.
- **Modify** `llama.go` — add `cLogitBias` marshal helper; update the 7 `C.llama_allocate_params` call sites.
- **Modify** `docs/OLLAMA-PORTABLE-FEATURES.md`, `README.md` — mark feature #4 done; document the manual smoke test.

**Build/verify commands (developer machine — cgo, not CI):**
- Headless Go test: `go test ./logitbias/` (no C libs needed).
- C binding rebuild: `bash .scripts/build-binding.sh` (recompiles `libbinding.a` from `binding.cpp`; assumes `bash .scripts/build-llamacpp.sh` already ran and the llama.cpp static libs exist).
- Whole-module compile: `go build ./...` (default CPU static link path).
- Lint: `golangci-lint run ./...` (the root cgo package lints only after `libbinding.a` exists).

> **Build-order constraint:** the root `package llama` is cgo — it cannot be `go build`/`go vet`'d without `libbinding.a`. Tasks 2 and 3 each END in a green `build-binding.sh` + `go build`. Task 1 is fully independent (cgo-free subpackage).

---

## Task 1: `logitbias` parser subpackage (headless TDD)

**Files:**
- Create: `logitbias/logitbias.go`
- Test: `logitbias/logitbias_test.go`

- [ ] **Step 1: Write the failing test**

Create `logitbias/logitbias_test.go`:

```go
package logitbias

import "testing"

func TestParse(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []Entry
		wantErr bool
	}{
		{"empty", "", nil, false},
		{"whitespace only", "   ", nil, false},
		{"single", "100:-1.5", []Entry{{Token: 100, Bias: -1.5}}, false},
		{"multiple", "1:2,3:-4", []Entry{{Token: 1, Bias: 2}, {Token: 3, Bias: -4}}, false},
		{"surrounding spaces", " 5 : 0.25 , 6 : -0.5 ", []Entry{{Token: 5, Bias: 0.25}, {Token: 6, Bias: -0.5}}, false},
		{"trailing comma", "7:1,", []Entry{{Token: 7, Bias: 1}}, false},
		{"duplicate last wins", "9:1,9:-100", []Entry{{Token: 9, Bias: -100}}, false},
		{"bad pair no colon", "abc", nil, true},
		{"bad token", "x:1", nil, true},
		{"bad bias", "5:y", nil, true},
		{"empty bias", "5:", nil, true},
		{"empty token", ":1", nil, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Parse(tc.in)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Parse(%q) err = %v, wantErr %v", tc.in, err, tc.wantErr)
			}
			if tc.wantErr {
				return
			}
			if len(got) != len(tc.want) {
				t.Fatalf("Parse(%q) = %v, want %v", tc.in, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("Parse(%q)[%d] = %v, want %v", tc.in, i, got[i], tc.want[i])
				}
			}
		})
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./logitbias/`
Expected: FAIL — `undefined: Entry`, `undefined: Parse`.

- [ ] **Step 3: Write the implementation**

Create `logitbias/logitbias.go`:

```go
// Package logitbias parses the user-facing logit-bias specification string
// ("<tokenID>:<bias>[,<tokenID>:<bias>...]") into token/bias pairs. It is
// cgo-free so it can be unit-tested without building the C binding.
package logitbias

import (
	"fmt"
	"strconv"
	"strings"
)

// Entry is one parsed token-bias pair.
type Entry struct {
	Token int32
	Bias  float32
}

// Parse converts "id:bias,id:bias" into entries. Whitespace around ids and
// biases is tolerated and empty segments (e.g. a trailing comma) are skipped.
// Empty or whitespace-only input yields (nil, nil). A malformed pair returns a
// non-nil error naming the offending segment. When the same token appears more
// than once the later value wins (matching llama.cpp's logit-bias semantics).
func Parse(s string) ([]Entry, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}

	order := make([]int32, 0)
	index := make(map[int32]int)
	entries := make([]Entry, 0)

	for _, seg := range strings.Split(s, ",") {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		tokStr, biasStr, ok := strings.Cut(seg, ":")
		if !ok {
			return nil, fmt.Errorf("logit_bias: segment %q is not <token>:<bias>", seg)
		}
		tok, err := strconv.ParseInt(strings.TrimSpace(tokStr), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("logit_bias: bad token in %q: %w", seg, err)
		}
		bias, err := strconv.ParseFloat(strings.TrimSpace(biasStr), 32)
		if err != nil {
			return nil, fmt.Errorf("logit_bias: bad bias in %q: %w", seg, err)
		}
		e := Entry{Token: int32(tok), Bias: float32(bias)}
		if pos, seen := index[e.Token]; seen {
			entries[pos] = e
			continue
		}
		index[e.Token] = len(entries)
		order = append(order, e.Token)
		entries = append(entries, e)
	}

	if len(entries) == 0 {
		return nil, nil
	}
	_ = order
	return entries, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./logitbias/`
Expected: PASS (all subtests).

- [ ] **Step 5: Remove the dead `order` slice**

The `order`/`_ = order` lines above are scaffolding. Delete the `order := make([]int32, 0)` line, the `order = append(order, e.Token)` line, and the `_ = order` line. Re-run `go test ./logitbias/` — Expected: PASS.

- [ ] **Step 6: Lint**

Run: `golangci-lint run ./logitbias/`
Expected: 0 issues. Fix any reported (likely none).

- [ ] **Step 7: Commit**

```bash
git add logitbias/logitbias.go logitbias/logitbias_test.go
git commit -m "feat(logitbias): pure-Go logit-bias spec parser (feature #4)"
```

---

## Task 2: C-side sampler wiring — typical_p, mirostat, make_sampler (no ABI change)

This task changes only `binding.cpp`. The ABI is untouched, so `llama.go` still
compiles. New `binding_params` fields for `min_p`/`logit_bias` are added now but
stay at their disabled defaults until Task 3 populates them (`make_sampler`
guards skip them), so behavior with `typical_p`/`mirostat` defaults is unchanged.

**Files:**
- Modify: `binding.cpp` (struct ~28-45, `make_sampler` 90-110, call site 133, `llama_allocate_params` 230-271)

- [ ] **Step 1: Add fields to `binding_params`**

In `binding.cpp`, the struct currently ends:

```cpp
    float    penalty_present = 0.00f;

    std::vector<std::string> antiprompt;
};
```

Replace with:

```cpp
    float    penalty_present = 0.00f;

    float    min_p           = 0.00f;   // 0 = disabled
    float    typical_p       = 1.00f;   // 1.0 = disabled
    int32_t  mirostat        = 0;       // 0 = off, 1 = v1, 2 = v2
    float    mirostat_eta    = 0.10f;
    float    mirostat_tau    = 5.00f;

    std::vector<std::string>      antiprompt;
    std::vector<llama_logit_bias> logit_bias;
};
```

(`llama_logit_bias` and `std::vector` are already available via `binding.h`'s
includes — `llama.h` and `<vector>`.)

- [ ] **Step 2: Store typical_p / mirostat in `llama_allocate_params`**

Find the `(void)` cast block (lines ~241-242):

```cpp
    (void)ignore_eos; (void)memory_f16; (void)tfs_z; (void)typical_p; (void)mirostat;
    (void)mirostat_eta; (void)mirostat_tau; (void)penalize_nl; (void)logit_bias;
```

Replace with (drop the four now-used casts; keep tfs_z, penalize_nl, logit_bias discarded for now):

```cpp
    (void)ignore_eos; (void)memory_f16; (void)tfs_z; (void)penalize_nl; (void)logit_bias;
```

Then find the existing store block ending:

```cpp
    p->penalty_present = presence_penalty;
```

Insert immediately after it:

```cpp
    p->typical_p    = typical_p;
    p->mirostat     = mirostat;
    p->mirostat_eta = mirostat_eta;
    p->mirostat_tau = mirostat_tau;
```

- [ ] **Step 3: Redesign `make_sampler`**

Replace the entire current function (`binding.cpp:90-110`):

```cpp
llama_sampler *make_sampler(const binding_params *bp) {
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (bp->penalty_last_n != 0 &&
        (bp->penalty_repeat != 1.0f || bp->penalty_freq != 0.0f || bp->penalty_present != 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            bp->penalty_last_n, bp->penalty_repeat, bp->penalty_freq, bp->penalty_present));
    }
    if (bp->temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (bp->top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(bp->top_k));
        }
        if (bp->top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(bp->top_p, 1));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(bp->temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(bp->seed));
    }
    return smpl;
}
```

with:

```cpp
llama_sampler *make_sampler(const binding_params *bp, const llama_vocab *vocab) {
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // logit_bias first — bias raw logits before any truncation.
    if (!bp->logit_bias.empty()) {
        llama_sampler_chain_add(smpl, llama_sampler_init_logit_bias(
            n_vocab, (int32_t)bp->logit_bias.size(), bp->logit_bias.data()));
    }

    // repetition penalties (unchanged condition).
    if (bp->penalty_last_n != 0 &&
        (bp->penalty_repeat != 1.0f || bp->penalty_freq != 0.0f || bp->penalty_present != 0.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            bp->penalty_last_n, bp->penalty_repeat, bp->penalty_freq, bp->penalty_present));
    }

    // mirostat is terminal: it performs its own temperature + selection, so it
    // replaces the truncation + dist tail entirely.
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

    // truncation tail — each entry opt-in; defaults reproduce the original chain.
    if (bp->top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(bp->top_k));
    }
    if (bp->typical_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_typical(bp->typical_p, 1));
    }
    if (bp->top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(bp->top_p, 1));
    }
    if (bp->min_p > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(bp->min_p, 1));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(bp->temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(bp->seed));
    return smpl;
}
```

- [ ] **Step 4: Update the `make_sampler` call site**

`binding.cpp:133` currently:

```cpp
    llama_sampler *smpl = make_sampler(bp);
```

Replace with (the surrounding `generate`/predict body has `st->vocab` in scope):

```cpp
    llama_sampler *smpl = make_sampler(bp, st->vocab);
```

- [ ] **Step 5: Rebuild the binding and compile the module**

Run: `bash .scripts/build-binding.sh && go build ./...`
Expected: both succeed (no errors). The ABI is unchanged, so `llama.go` builds
as-is. `make_sampler` now references `min_p`/`logit_bias` fields (still at
defaults → skipped).

- [ ] **Step 6: Commit**

```bash
git add binding.cpp
git commit -m "feat(binding): wire typical_p + mirostat; redesign make_sampler chain (feature #4)"
```

---

## Task 3: min_p + logit_bias ABI plumbing (C + Go together)

This is the ABI-breaking change; the C signature and the 7 Go call sites land in
one commit so the tree stays buildable. The legacy `const char *logit_bias`
parameter (currently discarded, and leaking a `C.CString` on every call) is
**removed**; the real data now crosses as parsed parallel arrays, and `min_p` is
appended.

**Files:**
- Modify: `options.go` (struct 24-58, new setter near line 458)
- Modify: `binding.h` (`llama_allocate_params` decl 39-48; includes)
- Modify: `binding.cpp` (`llama_allocate_params` signature 230-240 + body)
- Modify: `llama.go` (new helper; 7 call sites at 117, 159, 198, 243, 301, 364, 434)

- [ ] **Step 1: Add `MinP` to `PredictOptions`**

In `options.go`, the struct block (lines 33-41) currently:

```go
	TailFreeSamplingZ float32
	TypicalP          float32
	FrequencyPenalty  float32
	PresencePenalty   float32
	Mirostat          int
	MirostatETA       float32
	MirostatTAU       float32
	PenalizeNL        bool
	LogitBias         string
```

Insert `MinP float32` after `TypicalP`:

```go
	TailFreeSamplingZ float32
	TypicalP          float32
	MinP              float32
	FrequencyPenalty  float32
	PresencePenalty   float32
	Mirostat          int
	MirostatETA       float32
	MirostatTAU       float32
	PenalizeNL        bool
	LogitBias         string
```

- [ ] **Step 2: Add `SetMinP` and document the dropped no-ops**

In `options.go`, find `SetTypicalP` (line ~409):

```go
func SetTypicalP(tp float32) PredictOption {
	return func(p *PredictOptions) {
		p.TypicalP = tp
	}
}
```

Add immediately after it:

```go
// SetMinP sets the min-p sampling cutoff: tokens below this fraction of the top
// token's probability are dropped. 0 (the default) disables it.
func SetMinP(minp float32) PredictOption {
	return func(p *PredictOptions) {
		p.MinP = minp
	}
}
```

Then update the doc comments on the two dropped setters. Find `SetTailFreeSamplingZ` (line ~402) and `SetPenalizeNL` (line ~451) and prepend a deprecation note to each, e.g. for `SetTailFreeSamplingZ`:

```go
// SetTailFreeSamplingZ is a no-op: tail-free sampling was removed from
// upstream llama.cpp's sampler API and is no longer wired into the chain.
func SetTailFreeSamplingZ(tfz float32) PredictOption {
```

and for `SetPenalizeNL`:

```go
// SetPenalizeNL is a no-op: newline penalization was folded into the unified
// penalties sampler upstream and is no longer wired as a standalone knob.
func SetPenalizeNL(pnl bool) PredictOption {
```

(Keep the existing function bodies unchanged — the fields stay so callers still compile.)

- [ ] **Step 3: Widen the C ABI in `binding.h`**

Ensure `<stdint.h>` is included near the top of `binding.h` (add `#include <stdint.h>` after the existing `#include <stdbool.h>` if not present).

The current declaration (`binding.h:39-48`):

```c
void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty,
                            int repeat_last_n, bool ignore_eos, bool memory_f16,
                            int n_batch, int n_keep, const char **antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty,
                            float presence_penalty, int mirostat, float mirostat_eta,
                            float mirostat_tau, bool penalize_nl, const char *logit_bias,
                            const char *session_file, bool prompt_cache_all, bool mlock, bool mmap,
                            const char *maingpu, const char *tensorsplit, bool prompt_cache_ro,
                            const char *grammar, float rope_freq_base, float rope_freq_scale,
                            float negative_prompt_scale, const char *negative_prompt, int n_draft);
```

Replace the `bool penalize_nl, const char *logit_bias,` line with `bool penalize_nl,` (drop the char* param) and replace the final `int n_draft);` with the widened tail:

```c
void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty,
                            int repeat_last_n, bool ignore_eos, bool memory_f16,
                            int n_batch, int n_keep, const char **antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty,
                            float presence_penalty, int mirostat, float mirostat_eta,
                            float mirostat_tau, bool penalize_nl,
                            const char *session_file, bool prompt_cache_all, bool mlock, bool mmap,
                            const char *maingpu, const char *tensorsplit, bool prompt_cache_ro,
                            const char *grammar, float rope_freq_base, float rope_freq_scale,
                            float negative_prompt_scale, const char *negative_prompt, int n_draft,
                            float min_p,
                            const int32_t *logit_bias_tokens, const float *logit_bias_values,
                            int logit_bias_count);
```

- [ ] **Step 4: Mirror the signature + store the values in `binding.cpp`**

Update the definition header (`binding.cpp:230-240`) to match Step 3 exactly: remove `const char *logit_bias` from the `bool penalize_nl, const char *logit_bias,` line, and append `, float min_p, const int32_t *logit_bias_tokens, const float *logit_bias_values, int logit_bias_count` after `int n_draft`.

Then in the body, remove `(void)logit_bias;` from the cast line (now reads `(void)ignore_eos; (void)memory_f16; (void)tfs_z; (void)penalize_nl;`).

Find the store block you extended in Task 2 ending with `p->mirostat_tau = mirostat_tau;` and insert after it:

```cpp
    p->min_p = min_p;
    for (int i = 0; i < logit_bias_count; i++) {
        p->logit_bias.push_back(llama_logit_bias{
            (llama_token)logit_bias_tokens[i], logit_bias_values[i] });
    }
```

- [ ] **Step 5: Add the `cLogitBias` marshal helper to `llama.go`**

Add `"log/slog"` to the import block (lines ~12+) if not already present.

Add this helper near the top of `llama.go` (after the imports, before the first method). It parses the spec once per call and returns C arrays plus a cleanup closure:

```go
// cLogitBias parses a "id:bias,..." spec into C arrays for llama_allocate_params.
// On empty input or a parse error (logged and skipped — a malformed bias must
// never abort generation) it returns nil pointers, 0, and a no-op free.
func cLogitBias(spec string) (toks *C.int32_t, vals *C.float, count C.int, free func()) {
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
```

Add the import for the new subpackage to the import block:

```go
	"github.com/go-skynet/go-llama.cpp/logitbias"
```

- [ ] **Step 6: Update all 7 `C.llama_allocate_params` call sites**

Each of the 7 calls (lines 117, 159, 198, 243, 301, 364, 434) shares an identical
logit-bias line and trailing `n_draft` line. Apply the SAME two edits to each.

First, immediately ABOVE each `params := C.llama_allocate_params(` line, insert the marshal + deferred free:

```go
	lbTok, lbVal, lbCnt, lbFree := cLogitBias(po.LogitBias)
	defer lbFree()
```

Second, in each call's argument list:
- The line `C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), C.CString(po.LogitBias),` becomes `C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),` (drop the trailing `C.CString(po.LogitBias),`).
- The final argument line `C.int(po.NDraft),` becomes `C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,`.

Because these two lines are byte-identical across all 7 sites, you may apply each with a replace-all, then visually confirm 7 occurrences changed. After editing, the call tail reads:

```go
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL),
		C.CString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		C.CString(po.MainGPU), C.CString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		C.CString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), C.CString(po.NegativePrompt),
		C.int(po.NDraft), C.float(po.MinP), lbTok, lbVal, lbCnt,
	)
```

> Note: one call site (line ~117, in the embeddings path) uses `defer` already; placing `defer lbFree()` there is fine — it runs at function return after the synchronous C call has copied the arrays.

- [ ] **Step 7: Rebuild and compile**

Run: `bash .scripts/build-binding.sh && go build ./...`
Expected: both succeed. If `go build` reports an argument-count mismatch, a call
site was missed — grep `C.llama_allocate_params` and confirm all 7 end with
`lbCnt,`.

- [ ] **Step 8: Commit**

```bash
git add options.go binding.h binding.cpp llama.go
git commit -m "feat(binding): plumb min_p + logit_bias end-to-end via parsed C arrays (feature #4)"
```

---

## Task 4: Docs — feature status, no-op notes, smoke test

**Files:**
- Modify: `docs/OLLAMA-PORTABLE-FEATURES.md`
- Modify: `README.md`

- [ ] **Step 1: Mark feature #4 done in the roadmap**

In `docs/OLLAMA-PORTABLE-FEATURES.md`, update the feature #4 entry to DONE, noting:
min_p/typical_p/mirostat (v1/v2)/logit_bias wired into `make_sampler`; logit_bias
parsed by the cgo-free `logitbias` package; tfs_z and penalize_nl dropped
(removed upstream); GBNF grammar still deferred as a follow-up.

- [ ] **Step 2: Document the manual smoke test in README**

Add a short "Sampler smoke test (manual)" subsection to `README.md` listing the
four checks from the spec:

```markdown
### Sampler smoke test (manual, requires a built cgo binary + a GGUF model)

1. Greedy (`SetTemperature(0)`) output is identical to before this change.
2. `SetMinP(0.05)` with `SetTemperature(0.8)` produces coherent text.
3. `SetMirostat(2)` produces coherent text.
4. `SetLogitBias("<id>:-100")` for a token that otherwise appears makes that
   token never appear in the output.
```

- [ ] **Step 3: Commit**

```bash
git add docs/OLLAMA-PORTABLE-FEATURES.md README.md
git commit -m "docs: mark sampler wiring (feature #4) done; document smoke test"
```

---

## Task 5: Final lint sweep

**Files:** none (verification only)

- [ ] **Step 1: Lint the Go surface**

Run: `golangci-lint run ./...`
Expected: 0 issues. The root `package llama` lints only because `libbinding.a`
exists from Tasks 2-3. If `unparam`/`wsl`/`godoclint` flag the new helper or
setter, fix inline (match the formatting of the surrounding code; see how the
`gguf` and `streamfilter` packages were made `default:all` clean).

- [ ] **Step 2: Re-run the headless test**

Run: `go test ./logitbias/`
Expected: PASS.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "style: lint clean for sampler wiring (feature #4)"
```

(If Step 1 reported nothing, skip this commit.)

---

## Self-Review

**Spec coverage:**
- min_p end-to-end (option, ABI, struct, chain) → Task 3 (Steps 1,3,4,6) + Task 2 (Step 3 chain guard). ✓
- typical_p / mirostat C-only wiring → Task 2 (Steps 1-3). ✓
- logit_bias Go parser + parallel-array ABI → Task 1 (parser) + Task 3 (Steps 3-6). ✓
- make_sampler redesign + `(bp, vocab)` signature + call site → Task 2 (Steps 3-4). ✓
- Drop tfs_z / penalize_nl as documented no-ops → Task 3 (Step 2). ✓
- Behavior preservation (defaults reproduce old chain) → Task 2 Step 3 (guards) + verified by smoke test #1 (Task 4). ✓
- Headless parser tests + compile-verify + manual smoke test → Task 1, Task 2/3 Step "rebuild", Task 4 Step 2. ✓
- Grammar deferred → not in any task (correct). ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `logitbias.Parse`/`logitbias.Entry{Token int32, Bias float32}` used identically in Task 1 and Task 3's `cLogitBias`. `PredictOptions.MinP float32` defined (Task 3 Step 1) and consumed as `C.float(po.MinP)` (Step 6). `binding_params.min_p`/`typical_p`/`mirostat`/`logit_bias` defined (Task 2 Step 1) and consumed in `make_sampler` (Task 2 Step 3) + `llama_allocate_params` (Task 2 Step 2, Task 3 Step 4). C signature in `binding.h` (Task 3 Step 3) and `binding.cpp` (Task 3 Step 4) match. ✓

One refinement vs. spec: the spec described "repurposing the logit_bias slot"; this plan instead **removes** the dead `const char *logit_bias` param and **appends** the three array params + `min_p` (functionally identical, smaller/cleaner call-site diff, and it removes the pre-existing `C.CString` leak). Noted here so the reviewer expects an append-style diff.
