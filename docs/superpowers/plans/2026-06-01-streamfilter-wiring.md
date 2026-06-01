# Streamfilter → binding callback wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Go own stop-sequence detection and incomplete-UTF-8 hold-back by routing the cgo token callback through `streamfilter`, and remove the fragile C++ antiprompt suffix-compare.

**Architecture:** A new cgo-free `streamfilter.Sink` wraps a per-generation `*streamfilter.Filter` + the optional user callback + a result accumulator. `Predict`, `PredictResult`, and `SpeculativeSampling` register `sink.OnToken` as the bridge callback (restoring any persistent callback on return) and return `sink.Finish()`. `binding.cpp`'s `generate()` loses its antiprompt block and relies on the `tokenCallback`-returns-0 halt. `Predict`/`SpeculativeSampling` move their result buffers to the C heap for GC safety; the buffer content is now ignored because Go assembles the text.

**Tech Stack:** Go 1.25+ (cgo), C++17 (`binding.cpp`), the existing pure-Go `streamfilter` package, `go test`, `golangci-lint`, `bash scripts/binding.sh` (MinGW) for the cgo rebuild.

**Spec:** `docs/superpowers/specs/2026-06-01-streamfilter-wiring-design.md`

---

## File Structure

- **Create** `streamfilter/sink.go` — `Sink` type (`NewSink`, `OnToken`, `Finish`). Pure Go, no cgo. Owns `Filter` + user callback + `strings.Builder`.
- **Create** `streamfilter/sink_test.go` — headless table tests for `Sink` (no cgo, no model).
- **Modify** `llama.go` — add `getCallback`; import `streamfilter`; rewire `Predict`, `PredictResult`, `SpeculativeSampling`.
- **Modify** `binding.cpp` — delete the antiprompt suffix-compare block in `generate()`.
- **Modify** `README.md` — mark the streamfilter wiring done; document the manual smoke test.
- **Modify** `docs/OLLAMA-PORTABLE-FEATURES.md` — note the #3 deferred follow-up is now wired.
- **Create** `docs/streamfilter-smoke-test.md` — the maintainer's manual GGUF smoke-test procedure.

Rebuild artifact `libbinding.a` is **not** committed (build output).

---

## Task 1: `streamfilter.Sink` (headless, TDD)

**Files:**
- Test: `streamfilter/sink_test.go`
- Create: `streamfilter/sink.go`

- [ ] **Step 1: Write the failing tests**

Create `streamfilter/sink_test.go`:

```go
package streamfilter

import (
	"testing"
	"unicode/utf8"
)

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestSinkPassthrough(t *testing.T) {
	var got []string
	s := NewSink(nil, func(e string) bool { got = append(got, e); return true })
	for _, p := range []string{"Hello", ", ", "world"} {
		if !s.OnToken(p) {
			t.Fatalf("OnToken(%q) = false, want true", p)
		}
	}
	if res := s.Finish(); res != "Hello, world" {
		t.Fatalf("Finish() = %q, want %q", res, "Hello, world")
	}
	if want := []string{"Hello", ", ", "world"}; !equal(got, want) {
		t.Fatalf("emitted = %v, want %v", got, want)
	}
}

func TestSinkStopWithinOnePiece(t *testing.T) {
	s := NewSink([]string{"<end>"}, nil)
	if s.OnToken("hi <end> there") {
		t.Fatalf("OnToken with stop = true, want false (halt)")
	}
	if res := s.Finish(); res != "hi " {
		t.Fatalf("Finish() = %q, want %q", res, "hi ")
	}
}

func TestSinkStopSplitAcrossPieces(t *testing.T) {
	s := NewSink([]string{"<end>"}, nil)
	if !s.OnToken("answer<") {
		t.Fatalf("first OnToken = false, want true (hold)")
	}
	if s.OnToken("end> more") {
		t.Fatalf("second OnToken = true, want false (stop)")
	}
	if res := s.Finish(); res != "answer" {
		t.Fatalf("Finish() = %q, want %q", res, "answer")
	}
}

func TestSinkPartialStopResolvesToText(t *testing.T) {
	s := NewSink([]string{"<end>"}, nil)
	if !s.OnToken("a<") {
		t.Fatalf("first OnToken = false, want true (hold ambiguous tail)")
	}
	if !s.OnToken("x") {
		t.Fatalf("second OnToken = false, want true (disambiguated, not a stop)")
	}
	if res := s.Finish(); res != "a<x" {
		t.Fatalf("Finish() = %q, want %q", res, "a<x")
	}
}

func TestSinkSplitUTF8(t *testing.T) {
	// "é" is bytes 0xC3 0xA9, split across two pieces.
	s := NewSink(nil, nil)
	if !s.OnToken("caf\xc3") {
		t.Fatalf("first OnToken = false, want true (hold incomplete UTF-8)")
	}
	if !s.OnToken("\xa9") {
		t.Fatalf("second OnToken = false, want true")
	}
	res := s.Finish()
	if res != "café" {
		t.Fatalf("Finish() = %q, want %q", res, "café")
	}
	if !utf8.ValidString(res) {
		t.Fatalf("Finish() = %q is not valid UTF-8", res)
	}
}

func TestSinkUserHalt(t *testing.T) {
	s := NewSink(nil, func(e string) bool { return false })
	if s.OnToken("stop me") {
		t.Fatalf("OnToken = true, want false (user halt)")
	}
	if res := s.Finish(); res != "stop me" {
		t.Fatalf("Finish() = %q, want %q", res, "stop me")
	}
}

func TestSinkFlushRemainder(t *testing.T) {
	// A trailing incomplete-UTF-8 byte is held, then surfaced as-is by Finish
	// (matching Filter.Flush at true end-of-generation).
	s := NewSink(nil, nil)
	if !s.OnToken("hi\xc3") {
		t.Fatalf("OnToken = false, want true (hold)")
	}
	if res := s.Finish(); res != "hi\xc3" {
		t.Fatalf("Finish() = %q, want %q", res, "hi\xc3")
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./streamfilter/ -run TestSink -v`
Expected: FAIL — `undefined: NewSink`.

- [ ] **Step 3: Implement `streamfilter/sink.go`**

Create `streamfilter/sink.go`:

```go
package streamfilter

import "strings"

// Sink routes each decoded token piece through a Filter: it forwards only
// safe-to-emit text to an optional user callback, accumulates the full filtered
// text for the caller's return value, and reports when generation should halt.
//
// Sink has no cgo dependency; the in-process llama binding uses it as the glue
// for its token-callback path, but it is exercised entirely with headless tests.
type Sink struct {
	f    *Filter
	user func(string) bool // optional; may be nil
	buf  strings.Builder
}

// NewSink returns a Sink that filters against stops and forwards emitted text to
// user (which may be nil). Empty/nil stops means no stop sequences — only
// incomplete-UTF-8 hold-back applies.
func NewSink(stops []string, user func(string) bool) *Sink {
	return &Sink{f: New(stops), user: user}
}

// OnToken feeds one decoded piece. It accumulates and forwards the safe-to-emit
// text, and returns false when generation should halt — either because a stop
// sequence was reached or because the user callback returned false.
func (s *Sink) OnToken(piece string) bool {
	emit, stop := s.f.Push(piece)
	if emit != "" {
		s.buf.WriteString(emit)
		if s.user != nil && !s.user(emit) {
			return false
		}
	}
	return !stop
}

// Finish flushes any text held at end-of-generation, forwards it to the user
// callback, and returns the full accumulated filtered text.
func (s *Sink) Finish() string {
	if rem := s.f.Flush(); rem != "" {
		s.buf.WriteString(rem)
		if s.user != nil {
			s.user(rem)
		}
	}
	return s.buf.String()
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./streamfilter/ -v`
Expected: PASS (all `TestSink*` plus the existing filter/stop tests).

- [ ] **Step 5: Vet + lint**

Run: `go vet ./streamfilter/ && golangci-lint run ./streamfilter/`
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add streamfilter/sink.go streamfilter/sink_test.go
git commit -m "feat(streamfilter): Sink wrapper (Filter + user callback + accumulator)"
```

---

## Task 2: `getCallback` accessor

**Files:**
- Modify: `llama.go` (after `setCallback`, around line 590)

- [ ] **Step 1: Add the accessor**

In `llama.go`, immediately after the `setCallback` function (currently ending at line 590), add:

```go
// getCallback returns the token callback currently registered for statePtr, or
// nil if none. Used by the prediction methods to preserve a persistent callback
// (registered via SetTokenCallback) while a per-call filtering sink is active.
func getCallback(statePtr unsafe.Pointer) func(string) bool {
	m.RLock()
	defer m.RUnlock()

	return callbacks[uintptr(statePtr)]
}
```

(The registry mutex is the package-level `var m sync.RWMutex` defined alongside `callbacks` at `llama.go:562-565`.)

- [ ] **Step 2: Verify it compiles**

Run: `go build . && go vet .`
Expected: clean (no output).

- [ ] **Step 3: Commit**

```bash
git add llama.go
git commit -m "feat(binding): getCallback accessor for the token-callback registry"
```

---

## Task 3: Wire `Predict`

**Files:**
- Modify: `llama.go` — the `Predict` method (currently lines 327-385) and the import block (around lines 10-21).

- [ ] **Step 1: Add the `streamfilter` import**

In `llama.go`, add the import to the existing block (it already imports `logitbias`):

```go
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
```

- [ ] **Step 2: Replace the `Predict` body**

Replace the entire `Predict` function (lines 327-385) with:

```go
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
```

Key changes vs. the original: callback wiring via `prev`/`sink`/`defer`; antiprompt C args are now `nil, C.int(0)` (the `reversePrompt` construction is gone); result buffer is `C.malloc` not `make([]byte,...)`; the return text is `sink.Finish()`; the buggy `for ... strings.TrimRight(res, s)` loop is removed (the filter trims stops correctly). The three `TrimPrefix` lines are preserved.

- [ ] **Step 3: Verify it compiles + vets**

Run: `go build . && go vet . && go build ./examples/`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add llama.go
git commit -m "feat(binding): wire streamfilter Sink into Predict (Go owns stop detection)"
```

---

## Task 4: Wire `PredictResult`

**Files:**
- Modify: `llama.go` — the `PredictResult` method (currently lines 387-469).

- [ ] **Step 1: Replace the `PredictResult` body (and doc comment)**

Replace lines 387-469 with:

```go
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
		return "", 0, fmt.Errorf("inference: out of memory allocating %d bytes", scratch)
	}
	defer C.free(cbuf)
	if C.llama_predict_full(params, l.state, (*C.char)(cbuf), C.int(scratch), &nTok, C.bool(po.DebugMode)) < 0 {
		return "", 0, fmt.Errorf("inference failed")
	}

	return sink.Finish(), int(nTok), nil
}
```

Key changes: callback wiring via `prev`/`sink`/`defer`; antiprompt C args → `nil, C.int(0)`; the grow-and-retry buffer loop is replaced by a single fixed C-heap scratch buffer whose content is ignored (Go owns the text via `sink.Finish()`). The C-heap GC safety from commit `272b743` is preserved (still `C.malloc`/`C.free`).

- [ ] **Step 2: Verify it compiles + vets**

Run: `go build . && go vet . && go build ./examples/`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add llama.go
git commit -m "feat(binding): wire streamfilter Sink into PredictResult"
```

---

## Task 5: Wire `SpeculativeSampling`

**Files:**
- Modify: `llama.go` — the `SpeculativeSampling` method (currently lines 267-325).

- [ ] **Step 1: Replace the `SpeculativeSampling` body**

Replace lines 267-325 with:

```go
func (l *LLama) SpeculativeSampling(ll *LLama, text string, opts ...PredictOption) (string, error) {
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
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	// C-heap result buffer (GC-safe across the cgo callback); content ignored —
	// the result text is assembled in Go from the sink.
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
	ret := C.speculative_sampling(params, l.state, ll.state, (*C.char)(outBuf), C.bool(po.DebugMode))
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
```

Key changes mirror `Predict`: sink wiring, antiprompt args `nil, C.int(0)`, C-heap buffer, `sink.Finish()` return, removed `TrimRight` stop loop, kept the `TrimPrefix` cleanup.

- [ ] **Step 2: Verify it compiles + vets**

Run: `go build . && go vet . && go build ./examples/`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add llama.go
git commit -m "feat(binding): wire streamfilter Sink into SpeculativeSampling"
```

---

## Task 6: Remove the C antiprompt block + rebuild

**Files:**
- Modify: `binding.cpp` — `generate()` (currently lines 195-202).

- [ ] **Step 1: Delete the antiprompt suffix-compare block**

In `binding.cpp`, inside `generate()`, delete these lines (currently 195-202):

```cpp
        for (size_t a = 0; a < bp->antiprompt.size(); a++) {
            const std::string &ap = bp->antiprompt[a];
            if (!ap.empty() && out.size() >= ap.size() &&
                out.compare(out.size() - ap.size(), ap.size(), ap) == 0) {
                stop = true;
                break;
            }
        }
```

After deletion, the loop body around the callback reads (for reference — do not duplicate):

```cpp
        if (!piece.empty()) {
            std::vector<char> buf(piece.begin(), piece.end());
            buf.push_back('\0');
            if (tokenCallback(st, buf.data()) == 0) {
                stop = true;
            }
        }
        if (stop) {
            break;
        }
        llama_batch b = llama_batch_get_one(&id, 1);
```

The EOG check (`llama_vocab_is_eog`), the `out += piece` accumulation, `n_tokens++`, and the `tokenCallback`-returns-0 → `stop` → `break` path are all kept. `bp->antiprompt` is now unused by `generate()` (Go passes `nil, 0`); leave the field and its population code in place — the `binding.h` ABI is unchanged.

- [ ] **Step 2: Rebuild the binding**

Run: `bash scripts/binding.sh`
Expected: recompiles `binding.cpp` into `libbinding.a` with no errors. (Does not rebuild llama.cpp.)

- [ ] **Step 3: Verify the whole module builds + pure-Go tests pass**

Run: `go build ./... && go vet . && go test ./streamfilter/ ./gguf/ ./logitbias/`
Expected: clean build/link; all pure-Go tests PASS.

- [ ] **Step 4: Commit (source only — not the rebuilt artifact)**

```bash
git add binding.cpp
git commit -m "feat(binding): remove C++ antiprompt suffix-compare; Go owns stops"
```

Confirm `libbinding.a` is not staged (`git status` should not list it as staged; it is a build artifact).

---

## Task 7: Docs + manual smoke-test procedure

**Files:**
- Modify: `README.md` (the streamfilter section's follow-up note).
- Modify: `docs/OLLAMA-PORTABLE-FEATURES.md` (the #3 entry).
- Create: `docs/streamfilter-smoke-test.md`.

- [ ] **Step 1: Update the README streamfilter note**

In `README.md`, replace the deferred-follow-up sentence in the "Streaming output filter" section:

> It is a faithful port of Ollama's stop-sequence handling. Wiring it into the
> in-process binding callback is tracked as a follow-up (see
> `docs/superpowers/specs/2026-05-29-streamfilter-design.md`).

with:

```markdown
It is a faithful port of Ollama's stop-sequence handling, and it is now wired
into the in-process binding: `Predict`, `PredictResult`, and
`SpeculativeSampling` route every decoded piece through it, so stop sequences are
trimmed (even when split across token pieces) and multibyte UTF-8 is never
broken. See `docs/streamfilter-smoke-test.md` for the manual model smoke test.
```

- [ ] **Step 2: Update the feature doc**

In `docs/OLLAMA-PORTABLE-FEATURES.md`, in the "## ✅ 3. Stop-sequence + incomplete-UTF-8 buffering" section, append:

```markdown

**Wired (2026-06-01):** `streamfilter` is now connected to the binding's
token-callback path; the C++ antiprompt suffix-compare in `binding.cpp` was
removed so Go owns stop detection. See
`docs/superpowers/specs/2026-06-01-streamfilter-wiring-design.md`.
```

- [ ] **Step 3: Write the smoke-test procedure**

Create `docs/streamfilter-smoke-test.md`:

```markdown
# Streamfilter wiring — manual smoke test

Requires a built cgo binary (`bash scripts/binding.sh`) and a local GGUF model.
Run from the repo root, substituting your model path.

## 1. Greedy output unchanged (no stop hit)

```bash
go run ./examples -m /path/to/model.gguf -t 8
```
Expect coherent text identical to before this change when no stop word is set.

## 2. Stop split across token pieces is trimmed

Use a prompt + stop word likely to be emitted across two pieces (e.g. a stop of
`"<end>"`). Confirm:
- The streamed output stops at (and excludes) the stop word.
- The returned string from Predict/PredictResult excludes the stop word and
  everything after it — even when the model emitted `"<"` and `"end>"` as
  separate tokens.

## 3. Multibyte UTF-8 is never broken

Prompt for output containing accented or non-Latin characters (e.g. "Write a
sentence in French about café culture"). Confirm the streamed pieces and the
returned string contain no replacement characters (�) or broken bytes — i.e.
`utf8.ValidString(result)` is true.

## 4. Persistent SetTokenCallback survives across calls

Call `SetTokenCallback(cb)` once, then run two `Predict` calls without a per-call
`SetTokenCallback` option. Confirm `cb` fires (with filtered pieces) on both.
```

- [ ] **Step 4: Verify docs build is unaffected**

Run: `go build ./...`
Expected: clean (docs-only change).

- [ ] **Step 5: Commit**

```bash
git add README.md docs/OLLAMA-PORTABLE-FEATURES.md docs/streamfilter-smoke-test.md
git commit -m "docs: mark streamfilter wiring done; document smoke test"
```

---

## Self-Review

**Spec coverage:**
- Sink (Filter + user cb + accumulator) → Task 1. ✓
- `getCallback` + persistent-callback restore → Task 2 + Tasks 3-5. ✓
- Wire Predict / PredictResult / SpeculativeSampling → Tasks 3 / 4 / 5. ✓
- Pass `nil,0` antiprompt args + drop `reversePrompt` → Tasks 3-5. ✓
- C-heap buffers for Predict & SpeculativeSampling → Tasks 3, 5. ✓
- Remove `strings.TrimRight` stop loop; keep prefix trims; PredictResult no prefix trims → Tasks 3-5. ✓
- Delete C antiprompt block + rebuild → Task 6. ✓
- Headless Sink test → Task 1. ✓
- README / feature doc / smoke-test procedure → Task 7. ✓

**Deviation from spec (intentional):** the spec placed the sink type in `llama.go`; the plan places it in the `streamfilter` package as `streamfilter.Sink`. Reason: package `llama` is a cgo package, so a test there is not headless — contradicting the spec's "no cgo, no model" Sink test. `streamfilter` is the natural, cgo-free owner. The spec's Components section is updated to match.

**Placeholder scan:** none — every code/command step has concrete content.

**Type consistency:** `NewSink(stops []string, user func(string) bool) *Sink`, methods `OnToken(string) bool` and `Finish() string`, used identically in Tasks 1, 3, 4, 5. `getCallback(unsafe.Pointer) func(string) bool` defined in Task 2, used in Tasks 3-5. ✓

**Known pre-existing issues left untouched (per spec non-goals):** the global `callbacks` map keyed by `statePtr` (concurrent same-`LLama` generations collide); `Predict`/`SpeculativeSampling` inline `C.CString(...)` args that are never freed; `po.Tokens=99999999` default sizing. These are out of scope to keep the diff focused.
