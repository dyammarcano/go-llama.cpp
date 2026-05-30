# Streaming Stop-Sequence + UTF-8 Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-Go `streamfilter` package that does streaming stop-sequence detection and incomplete-UTF-8 hold-back over decoded token pieces — a faithful port of Ollama's `runner/common/stop.go` plus a stateful `Filter` wrapper.

**Architecture:** New subpackage `streamfilter/` at the fork root (pure-Go, no cgo/build-tags, like `gguf/`). `stop.go` lifts Ollama's four primitives verbatim; `filter.go` adds a stateful `Filter` (`Push`/`Flush`) that distills the runner's per-sequence decode loop. Package-only this feature; the `llama.go`/`binding.cpp` wiring is a documented follow-up.

**Tech Stack:** Go 1.25 stdlib only (`strings`). Tests are `package streamfilter` (internal). Must stay golangci-lint `default:all` clean (repo policy).

**Spec:** `docs/superpowers/specs/2026-05-29-streamfilter-design.md`
**Reference (read-only):** `C:\Users\dyamm\My Drive\acer\public_repos\ollama\runner\common\stop.go`

Work from: `C:/Users/dyamm/My Drive/acer/public_repos/go-llama.cpp`. NO AI attribution in commits. Touch only files under `streamfilter/` (+ `README.md` in Task 3). No repo-wide gofmt/go fix.

---

## File Structure

```
streamfilter/
  stop.go        # NEW — verbatim lift of the 4 Ollama primitives (+ doc comments)
  stop_test.go   # NEW — primitive unit tests
  filter.go      # NEW — Filter streaming wrapper
  filter_test.go # NEW — streaming scenario tests
```

Attribution header on `stop.go` and `filter.go`:
```go
// Derived from github.com/ollama/ollama/runner/common/stop.go (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.
```

---

## Task 1: Lift the four stop primitives (`stop.go`)

**Files:** Create `streamfilter/stop.go`, `streamfilter/stop_test.go`.

- [ ] **Step 1: Write the failing tests.** Create `streamfilter/stop_test.go`:
```go
package streamfilter

import (
	"reflect"
	"testing"
)

func TestFindStop(t *testing.T) {
	stops := []string{"</s>", "STOP"}
	if ok, m := FindStop("hello</s>", stops); !ok || m != "</s>" {
		t.Errorf("FindStop hit = %v,%q want true,</s>", ok, m)
	}
	if ok, _ := FindStop("hello world", stops); ok {
		t.Error("FindStop should miss")
	}
	if ok, _ := FindStop("anything", nil); ok {
		t.Error("FindStop with no stops should be false")
	}
}

func TestContainsStopSuffix(t *testing.T) {
	stops := []string{"<end>"}
	if !ContainsStopSuffix("foo<", stops) {
		t.Error("tail '<' is a prefix of '<end>' -> true")
	}
	if !ContainsStopSuffix("foo<en", stops) {
		t.Error("tail '<en' is a prefix of '<end>' -> true")
	}
	if ContainsStopSuffix("foobar", stops) {
		t.Error("no prefix match -> false")
	}
	if ContainsStopSuffix("foo", nil) {
		t.Error("no stops -> false")
	}
}

func TestTruncateStop(t *testing.T) {
	got, trunc := TruncateStop([]string{"ab", "cd"}, "bc")
	if !reflect.DeepEqual(got, []string{"a"}) || !trunc {
		t.Errorf("TruncateStop split = %v,%v want [a],true", got, trunc)
	}
	got, trunc = TruncateStop([]string{"ab", "cd"}, "cd")
	if !reflect.DeepEqual(got, []string{"ab"}) || trunc {
		t.Errorf("TruncateStop boundary = %v,%v want [ab],false", got, trunc)
	}
	got, trunc = TruncateStop([]string{"ab", "cd"}, "zz")
	if !reflect.DeepEqual(got, []string{"ab", "cd"}) || trunc {
		t.Errorf("TruncateStop absent = %v,%v want [ab cd],false", got, trunc)
	}
}

func TestIncompleteUnicode(t *testing.T) {
	if IncompleteUnicode("hello") {
		t.Error("ASCII is complete")
	}
	if IncompleteUnicode("héllo") {
		t.Error("complete multibyte is complete")
	}
	if !IncompleteUnicode("h" + string([]byte{0xC3})) {
		t.Error("lone 0xC3 lead byte -> incomplete")
	}
	if !IncompleteUnicode(string([]byte{0xE2, 0x82})) {
		t.Error("truncated 3-byte sequence -> incomplete")
	}
}
```

- [ ] **Step 2: Run, expect FAIL** (undefined funcs):
`cd "C:/Users/dyamm/My Drive/acer/public_repos/go-llama.cpp" && go test ./streamfilter/ -run 'TestFindStop|TestContainsStopSuffix|TestTruncateStop|TestIncompleteUnicode' -v`

- [ ] **Step 3: Write `streamfilter/stop.go`** (verbatim logic from Ollama; only the package name, attribution header, and doc comments are added):
```go
// Derived from github.com/ollama/ollama/runner/common/stop.go (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.

package streamfilter

import "strings"

// FindStop reports whether any stop sequence is a substring of sequence,
// returning the first match.
func FindStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
			return true, stop
		}
	}

	return false, ""
}

// ContainsStopSuffix reports whether the tail of sequence equals a prefix of any
// stop sequence — i.e. sequence may be partway through a stop that completes in
// a later piece.
func ContainsStopSuffix(sequence string, stops []string) bool {
	for _, stop := range stops {
		for i := 1; i <= len(stop); i++ {
			if strings.HasSuffix(sequence, stop[:i]) {
				return true
			}
		}
	}

	return false
}

// TruncateStop removes the provided stop string from pieces, returning the
// partial pieces with stop removed, including truncating the last piece if
// required (and signalling if this was the case).
func TruncateStop(pieces []string, stop string) ([]string, bool) {
	joined := strings.Join(pieces, "")

	index := strings.Index(joined, stop)
	if index == -1 {
		return pieces, false
	}

	joined = joined[:index]

	// Split the truncated string back into pieces of their original lengths.
	lengths := make([]int, len(pieces))
	for i, piece := range pieces {
		lengths[i] = len(piece)
	}

	var result []string

	tokenTruncated := false
	start := 0

	for _, length := range lengths {
		if start >= len(joined) {
			break
		}

		end := start + length
		if end > len(joined) {
			end = len(joined)
			tokenTruncated = true
		}

		result = append(result, joined[start:end])
		start = end
	}

	return result, tokenTruncated
}

// IncompleteUnicode reports whether the trailing bytes of token form an
// incomplete (truncated) multibyte UTF-8 character.
func IncompleteUnicode(token string) bool {
	incomplete := false

	// check if there is an incomplete UTF-8 character at the end
	for i := 1; i < 5 && i <= len(token); i++ {
		c := token[len(token)-i]

		if (c & 0xc0) == 0x80 {
			// continuation byte: 10xxxxxx
			continue
		}

		//nolint:gocritic // faithful lift; the trailing break must target the
		// for loop, so this cannot become a switch (break would exit the switch).
		if (c & 0xe0) == 0xc0 {
			// 2-byte character: 110xxxxx ...
			incomplete = i < 2
		} else if (c & 0xf0) == 0xe0 {
			// 3-byte character: 1110xxxx ...
			incomplete = i < 3
		} else if (c & 0xf8) == 0xf0 {
			// 4-byte character: 11110xxx ...
			incomplete = i < 4
		}

		// else 1-byte character or invalid byte
		break
	}

	return incomplete
}
```

- [ ] **Step 4: Run, expect PASS:** `go test ./streamfilter/ -run 'TestFindStop|TestContainsStopSuffix|TestTruncateStop|TestIncompleteUnicode' -v`. Then `go vet ./streamfilter/`.

- [ ] **Step 5: Commit:**
```bash
git add streamfilter/stop.go streamfilter/stop_test.go
git commit -m "feat(streamfilter): lift Ollama stop-sequence + UTF-8 primitives"
```

---

## Task 2: `Filter` streaming wrapper (`filter.go`)

**Files:** Create `streamfilter/filter.go`, `streamfilter/filter_test.go`.

- [ ] **Step 1: Write the failing tests.** Create `streamfilter/filter_test.go`:
```go
package streamfilter

import "testing"

// collect drives a Filter through pieces, concatenating emitted text. It stops
// early if a stop is hit, otherwise appends Flush() output.
func collect(f *Filter, pieces ...string) (emitted string, stopped bool) {
	for _, p := range pieces {
		e, s := f.Push(p)
		emitted += e

		if s {
			return emitted, true
		}
	}

	emitted += f.Flush()

	return emitted, false
}

func TestFilterNoStops(t *testing.T) {
	got, stopped := collect(New(nil), "hello ", "world")
	if got != "hello world" || stopped {
		t.Errorf("got %q,%v want 'hello world',false", got, stopped)
	}
}

func TestFilterStopWithinPiece(t *testing.T) {
	got, stopped := collect(New([]string{"<end>"}), "keep<end>drop")
	if got != "keep" || !stopped {
		t.Errorf("got %q,%v want 'keep',true", got, stopped)
	}
}

func TestFilterStopSplitAcrossPieces(t *testing.T) {
	f := New([]string{"<end>"})

	e1, s1 := f.Push("keep<")
	if e1 != "" || s1 {
		t.Errorf("push1 = %q,%v want '',false (held)", e1, s1)
	}

	e2, s2 := f.Push("end>drop")
	if e2 != "keep" || !s2 {
		t.Errorf("push2 = %q,%v want 'keep',true", e2, s2)
	}
}

func TestFilterPartialThatIsNotStop(t *testing.T) {
	f := New([]string{"<end>"})

	e1, _ := f.Push("a<") // held: '<' is a prefix of '<end>'
	e2, _ := f.Push("b")  // '<b' is not a stop prefix -> flush all

	if e1 != "" || e2 != "a<b" {
		t.Errorf("emits = %q,%q want '','a<b'", e1, e2)
	}
}

func TestFilterMultibyteSplit(t *testing.T) {
	f := New(nil)

	e1, _ := f.Push("x" + string([]byte{0xC3})) // first byte of 'é'
	if e1 != "" {
		t.Errorf("push1 = %q want '' (incomplete utf8 held)", e1)
	}

	e2, _ := f.Push(string([]byte{0xA9}) + "y") // second byte + more
	if e2 != "xéy" {
		t.Errorf("push2 = %q want 'xéy'", e2)
	}
}

func TestFilterFlushRemainder(t *testing.T) {
	f := New([]string{"<end>"})

	f.Push("ab<") // held as a possible stop prefix
	if got := f.Flush(); got != "ab<" {
		t.Errorf("Flush = %q want 'ab<'", got)
	}
}
```

- [ ] **Step 2: Run, expect FAIL** (undefined `New`/`Filter`):
`go test ./streamfilter/ -run TestFilter -v`

- [ ] **Step 3: Write `streamfilter/filter.go`:**
```go
// Derived from github.com/ollama/ollama/runner/common/stop.go (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.

package streamfilter

import "strings"

// Filter incrementally filters a stream of decoded token pieces: it holds back
// text that might be part of a stop sequence or an incomplete UTF-8 character,
// emits only text that is safe to show, and reports when a stop sequence is hit.
type Filter struct {
	stops   []string
	pending []string
}

// New returns a Filter for the given stop sequences (nil or empty means no
// stops — the Filter then only guards against split UTF-8 characters).
func New(stops []string) *Filter {
	return &Filter{stops: stops}
}

// Push feeds one decoded piece and returns the text now safe to emit (possibly
// empty) plus whether a stop sequence was reached. When stop is true, the
// matched stop sequence and everything after it are dropped.
func (f *Filter) Push(piece string) (emit string, stop bool) {
	f.pending = append(f.pending, piece)
	seq := strings.Join(f.pending, "")

	if found, matched := FindStop(seq, f.stops); found {
		truncated, _ := TruncateStop(f.pending, matched)
		f.pending = nil

		return strings.Join(truncated, ""), true
	}

	if ContainsStopSuffix(seq, f.stops) || IncompleteUnicode(seq) {
		return "", false
	}

	f.pending = nil

	return seq, false
}

// Flush returns any buffered remainder at end-of-generation and clears the
// buffer.
func (f *Filter) Flush() string {
	out := strings.Join(f.pending, "")
	f.pending = nil

	return out
}
```

- [ ] **Step 4: Run, expect PASS:** `go test ./streamfilter/ -run TestFilter -v`. Then the whole package + vet: `go test ./streamfilter/ -count=1 && go vet ./streamfilter/`.

- [ ] **Step 5: Commit:**
```bash
git add streamfilter/filter.go streamfilter/filter_test.go
git commit -m "feat(streamfilter): stateful streaming Filter (Push/Flush)"
```

---

## Task 3: README docs, lint-to-zero, final verify

**Files:** Modify `README.md`.

- [ ] **Step 1: Document in README.** In `README.md`, immediately after the "## Estimating GPU layers (no model load)" subsection and before the next `##`, add:
```markdown
## Streaming output filter (stop sequences + UTF-8)

`streamfilter.Filter` filters a stream of decoded token pieces — pure Go, no
cgo: it holds back text that might be part of a stop sequence or an incomplete
multibyte UTF-8 character, and reports when a stop is hit:

```go
import "github.com/go-skynet/go-llama.cpp/streamfilter"

f := streamfilter.New([]string{"</s>", "User:"})
emit, stop := f.Push(piece) // forward emit to your callback; halt when stop
// ... at end of generation:
emit = f.Flush()
```

It is a faithful port of Ollama's stop-sequence handling. Wiring it into the
in-process binding callback is tracked as a follow-up (see
`docs/superpowers/specs/2026-05-29-streamfilter-design.md`).
```
(Mind the nested ```go fence — ensure it closes so the outer markdown isn't broken.)

- [ ] **Step 2: Full-package test + vet:**
`go test ./streamfilter/ -v -count=1` → all pass. `go vet ./streamfilter/` → clean.

- [ ] **Step 3: Lint to ZERO (repo gate).**
`golangci-lint run ./streamfilter/...` → must be 0 issues. Fix findings in the new files. Expected categories: `wsl_v5` blank-line rules (add blank lines where pointed), `godoclint` (every exported symbol — `FindStop`, `ContainsStopSuffix`, `TruncateStop`, `IncompleteUnicode`, `Filter`, `New`, `Push`, `Flush` — must have a name-prefixed doc comment; they do, verify wording), `gocritic` on `IncompleteUnicode` (already suppressed with an explained `//nolint:gocritic` — do NOT remove it; the trailing `break` must target the `for` loop, so the if/else-if cannot become a `switch`). Prefer real fixes elsewhere; the only sanctioned `//nolint` is the one in `IncompleteUnicode`.

- [ ] **Step 4: Confirm scope** — `git status --short` shows only `README.md` modified (the streamfilter files were committed in Tasks 1–2). If lint fixes touched the streamfilter sources, include them.

- [ ] **Step 5: Commit:**
```bash
git add README.md streamfilter/
git commit -m "docs(streamfilter): document Filter; lint clean"
```

---

## Self-Review (completed during planning)

**Spec coverage:**
- Package `streamfilter/` with `stop.go` + `filter.go` → Tasks 1,2. ✓
- Four verbatim primitives (`FindStop`/`ContainsStopSuffix`/`TruncateStop`/`IncompleteUnicode`) → Task 1. ✓
- `Filter`/`New`/`Push`/`Flush` with the exact Push ordering (FindStop → hold-on-suffix-or-incomplete → flush) → Task 2. ✓
- Tests: primitives + streaming scenarios (stop within piece, stop split across pieces, partial-not-stop, multibyte split, flush remainder, no stops) → Tasks 1,2. ✓
- Pure-Go, stdlib only, no build tags, no new deps, lint-clean → enforced Task 3. ✓
- Integration contract for wiring → already in the spec (deferred); README points to it (Task 3). ✓

**Placeholder scan:** none — all steps show complete code; the single `//nolint` is explained.

**Type consistency:** `FindStop`, `ContainsStopSuffix`, `TruncateStop`, `IncompleteUnicode`, `Filter`, `New`, `Push`, `Flush`, and the test helper `collect` are used consistently across tasks. `Filter.Push` uses `matched` (not a shadowed `stop`) for the `FindStop` result. ✓
