# Streaming stop-sequence + UTF-8 filter — design

Date: 2026-05-29
Status: approved
Repo: `github.com/go-skynet/go-llama.cpp`
Roadmap: feature #3 of `docs/OLLAMA-PORTABLE-FEATURES.md`

## Goal

Add a **pure-Go** `streamfilter` package that does robust, streaming
stop-sequence detection and incomplete-UTF-8 hold-back over decoded token
pieces. This replaces the fragile C++ antiprompt suffix-compare in `binding.cpp`
and fixes two real correctness bugs in the current binding:

1. A stop sequence split across two token pieces is missed by the C++
   suffix-compare unless it happens to align to a single piece.
2. A multibyte UTF-8 character split across two token pieces is streamed as
   broken bytes (the binding emits raw pieces).

Faithful port of Ollama's `runner/common/stop.go` (MIT), wrapped in a stateful
streaming `Filter` that distills the runner's per-sequence decode-loop logic.

## Scope

**This spec (ships now):**
- A new pure-Go subpackage `streamfilter` — the 4 Ollama primitives (verbatim
  lift) + a stateful `Filter` streaming wrapper + full headless table tests.
- No cgo, no build tags, no new `go.mod` deps (stdlib `strings` only).

**Deferred (documented integration contract below, NOT implemented now):**
- Wiring `Filter` into `llama.go`'s token-callback path.
- Removing the antiprompt block from `binding.cpp` so Go owns stop detection.
- These require the MinGW/CUDA cgo rebuild + a real-model smoke test and are a
  separate follow-up feature.

## Non-goals

- No changes to `llama.go`, `binding.cpp`, `binding.h` in this feature.
- No llmark changes.
- Not a general text-streaming framework — just stop + UTF-8 hold-back.

## Package layout

```
streamfilter/
  stop.go        # verbatim lift of ollama/runner/common/stop.go (4 funcs)
  stop_test.go   # unit tests for the 4 primitives
  filter.go      # NEW: Filter streaming wrapper
  filter_test.go # NEW: streaming scenario table tests
```

Import path: `github.com/go-skynet/go-llama.cpp/streamfilter`.

Attribution header on `stop.go`:
```go
// Derived from github.com/ollama/ollama/runner/common/stop.go (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.
```

## The four primitives (verbatim from Ollama, package `streamfilter`)

```go
func FindStop(sequence string, stops []string) (bool, string)
// true + matched stop if any stop is a substring of sequence (strings.Contains).

func ContainsStopSuffix(sequence string, stops []string) bool
// true if the tail of sequence equals a prefix stop[:i] of any stop
// (for i := 1..len(stop), strings.HasSuffix(sequence, stop[:i])).
// This is the cross-token hold-back primitive.

func TruncateStop(pieces []string, stop string) ([]string, bool)
// join pieces, cut at strings.Index(joined, stop), re-split into pieces of the
// original byte-lengths; second return = whether the last kept piece was
// truncated mid-piece.

func IncompleteUnicode(token string) bool
// scan up to the last 4 bytes; true if they form an incomplete 2/3/4-byte
// UTF-8 sequence (using the 0xc0/0xe0/0xf0/0xf8 lead-byte masks).
```

These are lifted byte-for-byte (logic unchanged) from the source; only the
package name and attribution header differ.

## The `Filter` streaming wrapper (`filter.go`)

```go
// Filter incrementally filters a stream of decoded token pieces: it holds back
// text that might be part of a stop sequence or an incomplete UTF-8 character,
// emits only text that is safe to show, and reports when a stop sequence is hit.
type Filter struct {
    stops   []string
    pending []string
}

// New returns a Filter for the given stop sequences (nil/empty = no stops).
func New(stops []string) *Filter { return &Filter{stops: stops} }

// Push feeds one decoded piece. It returns the text that is now safe to emit
// (possibly "") and whether a stop sequence was reached (caller should halt
// generation). When stop is true, the matched stop sequence and everything
// after it are dropped from the returned text.
func (f *Filter) Push(piece string) (emit string, stop bool) {
    f.pending = append(f.pending, piece)
    seq := strings.Join(f.pending, "")

    if found, matched := FindStop(seq, f.stops); found {
        truncated, _ := TruncateStop(f.pending, matched)
        f.pending = nil
        return strings.Join(truncated, ""), true
    }

    if ContainsStopSuffix(seq, f.stops) || IncompleteUnicode(seq) {
        return "", false // hold the whole pending buffer until disambiguated
    }

    f.pending = nil
    return seq, false
}

// Flush returns any buffered remainder at end-of-generation (EOG/natural stop)
// and clears the buffer.
func (f *Filter) Flush() string {
    out := strings.Join(f.pending, "")
    f.pending = nil
    return out
}
```

Design notes:
- `pending` is kept as `[]string` (not a single string) specifically so
  `TruncateStop` can re-split on original piece boundaries, matching Ollama.
- The hold policy is coarse (hold the entire pending buffer when the tail is
  ambiguous), exactly as Ollama's runner does — not a byte-precise split. Once
  the next piece disambiguates, the whole buffer flushes.
- `IncompleteUnicode` is applied to the joined `seq`; it only inspects trailing
  bytes, so this is equivalent to Ollama applying it to the last piece.
- `Flush` emits the remainder as-is (Ollama does not drop a trailing incomplete
  char at true EOG; neither do we).

## Error handling

No errors — `Filter` is total. Empty `stops` ⇒ `FindStop`/`ContainsStopSuffix`
always false ⇒ pure pass-through except for UTF-8 hold-back. `nil` Filter
methods are not supported (construct via `New`).

## Testing (pure-Go, headless — no cgo, no model)

`stop_test.go` — the 4 primitives:
- `FindStop`: hit, miss, multiple stops, empty stops, stop at start/middle/end.
- `ContainsStopSuffix`: full-prefix tail, partial-prefix tail, no match, empty.
- `TruncateStop`: stop mid-piece (tokenTruncated true), stop on boundary, absent.
- `IncompleteUnicode`: complete ASCII, complete multibyte, truncated 2/3/4-byte
  sequence, trailing continuation byte.
(Port Ollama's own `stop_test.go` cases if that file exists, for fidelity.)

`filter_test.go` — `Filter` streaming scenarios:
- no stops, ASCII pieces → emits each piece immediately, `Flush` empty.
- stop within one piece → emits text before stop, `stop==true`, remainder dropped.
- **stop split across pieces** (`"<"`, `"end>"`, stop `"<end>"`) → first Push
  emits `""` (held), second returns text-before-stop + `stop==true`.
- partial-stop tail that turns out NOT to be a stop (`"<"` then `"x"`, stop
  `"<end>"`) → second Push emits `"<x"` (buffer flushes once disambiguated).
- **multibyte char split across pieces** (e.g. `"e"`, then the two bytes of `é`
  split) → incomplete half held, completed on next Push, both emitted intact.
- multiple stops, earliest wins.
- `Flush` returns the buffered remainder when generation ends with text still held.

Run: `go test ./streamfilter/`. golangci-lint `default:all` must be clean.

## Integration contract (for the DEFERRED wiring feature)

When the follow-up wires this in (needs the cgo rebuild + model smoke test):

1. In `llama.go`, per `Predict`/`PredictResult` call, build
   `f := streamfilter.New(po.Antiprompt)` (the existing stop-words option).
2. In the token-callback bridge, route each decoded piece through
   `emit, stop := f.Push(piece)`; forward `emit` to the user's `TokenCallback`
   only when non-empty; if `stop`, return `false` from the callback to halt the
   C decode loop.
3. After the loop returns, call `f.Flush()` and forward any remainder; for
   `PredictResult`, the accumulated returned text should be the concatenation of
   emitted segments (so the stop sequence is already trimmed and UTF-8 is valid).
4. In `binding.cpp`, remove the antiprompt suffix-compare block from `generate()`
   (Go now owns stop detection); keep the EOG (`llama_vocab_is_eog`) check and
   the `tokenCallback`-returns-false halt path.

This contract is documented here so the wiring feature can be planned without
re-deriving it.

## Risks

- Coarse whole-buffer hold can delay emission of a few tokens until a partial
  stop/UTF-8 tail disambiguates — identical to Ollama's behaviour; acceptable.
- `TruncateStop`'s second return (tokenTruncated) is unused by `Filter` (we emit
  strings, not token-aligned pieces). Kept for faithful-lift parity; `Filter`
  discards it with `_`.
