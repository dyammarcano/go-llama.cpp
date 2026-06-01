# Streamfilter → binding callback wiring — design

Feature #3 follow-up. Wires the existing pure-Go `streamfilter` package into the
in-process cgo token-callback path so **Go owns stop detection and UTF-8
hold-back**, and removes the fragile C++ antiprompt suffix-compare from
`binding.cpp`.

Parent spec: `docs/superpowers/specs/2026-05-29-streamfilter-design.md`
("Integration contract (for the DEFERRED wiring feature)"). This spec supersedes
that contract and **corrects one stale name in it**: the stop-words option is
`po.StopPrompts` (set via `SetStopWords`), not `po.Antiprompt` (no such field
exists).

## Goal

Fix the two real correctness bugs the parent spec identified (GAP #4), now at the
binding boundary instead of only in the standalone package:

1. **Split stop sequence** — a stop word spread across two token pieces (e.g.
   `"<"` then `"end>"` for stop `"<end>"`) is missed by the C++ suffix-compare.
2. **Split multibyte UTF-8** — a character whose bytes straddle two pieces is
   streamed as broken bytes.

After this feature, both are handled by `streamfilter.Filter` in pure Go, and the
returned text from `Predict` / `PredictResult` / `SpeculativeSampling` is
stop-trimmed and UTF-8-valid.

## Scope

**Ships now:**
- A small `streamfilter.Sink` type in the (cgo-free) `streamfilter` package that
  wraps a per-generation `*streamfilter.Filter` + the optional user callback + a
  result accumulator. It lives outside the cgo `llama` package so it can be
  unit-tested headlessly (no cgo, no model).
- A `getCallback` accessor next to the existing `setCallback`, so the wiring can
  read and restore the persistent callback registered via `SetTokenCallback`.
- Wiring in **all three** generation entry points: `Predict`, `PredictResult`,
  `SpeculativeSampling`.
- `binding.cpp`: delete the antiprompt suffix-compare block in `generate()`
  (Go now owns stop detection). Keep the EOG check and the
  `tokenCallback`-returns-0 halt path.
- `Predict` & `SpeculativeSampling`: switch their result buffers from Go-heap
  (`make([]byte, ...)`) to **C-heap** (`C.malloc`), matching the GC-safe pattern
  `PredictResult` already uses (commit `272b743`). Their C buffer content becomes
  unused — the return text now comes from the Go accumulator.
- Remove the now-dead per-call `reversePrompt` construction in the three
  functions and pass `nil, 0` for the C antiprompt params (the `binding.h` ABI is
  unchanged; the params simply go unused on the C side).
- Replace the buggy `strings.TrimRight(res, stop)` stop trimming (it treats the
  stop as a character *cutset*, not a suffix) with the filter's correct
  `TruncateStop` behavior.
- Rebuild `libbinding.a` via `scripts/binding.sh`.

**Deferred / out of scope (documented, not done here):**
- GBNF grammar wiring (feature #4 follow-up; `grammar` stays discarded).
- The pre-existing concurrency limitation: the global `callbacks` map keyed by
  `statePtr` means two concurrent generations on the *same* `LLama` collide. This
  is true today and is not introduced or fixed here.
- `Predict`'s default `po.Tokens = 99999999` over-allocation (now a `C.malloc`
  instead of a Go allocation, so it no longer pressures the Go heap, but the size
  is unchanged). Switching `Predict` to the `llama_predict_full` grow-retry path
  is a possible future cleanup.
- No `binding.h` signature changes (ABI stays frozen). No llmark changes.

## Components

### 1. `streamfilter.Sink` (new, in the `streamfilter` package)

Exported from the cgo-free `streamfilter` package (`streamfilter/sink.go`) so it
can be unit-tested without building the cgo `llama` package or loading a model.

```go
// Sink routes each decoded token piece through a Filter: it forwards only
// safe-to-emit text to an optional user callback, accumulates the full filtered
// text for the caller's return value, and reports when generation should halt.
type Sink struct {
    f    *Filter
    user func(string) bool // optional; may be nil
    buf  strings.Builder
}

func NewSink(stops []string, user func(string) bool) *Sink {
    return &Sink{f: New(stops), user: user}
}

// OnToken is registered as the bridge callback. Returning false halts the C
// decode loop (tokenCallback-returns-0 path).
func (s *Sink) OnToken(piece string) bool {
    emit, stop := s.f.Push(piece)
    if emit != "" {
        s.buf.WriteString(emit)
        if s.user != nil && !s.user(emit) {
            return false // user requested halt
        }
    }
    return !stop
}

// Finish flushes any held remainder at end-of-generation and returns the full
// filtered text.
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

### 2. `getCallback` accessor (new, in `llama.go`)

A read counterpart to `setCallback`, used to preserve the persistent callback:

```go
func getCallback(statePtr unsafe.Pointer) func(string) bool {
    callbacksMu.Lock() // reuse the existing lock guarding `callbacks`
    defer callbacksMu.Unlock()
    return callbacks[uintptr(statePtr)]
}
```

(If the existing `setCallback` uses a different mutex name, reuse it verbatim.)

### 3. Per-entry-point wiring (`Predict`, `PredictResult`, `SpeculativeSampling`)

Replace the current `if po.TokenCallback != nil { setCallback(...) }` blocks and
the `reversePrompt` construction with:

```go
prev := getCallback(l.state)
user := po.TokenCallback
if user == nil {
    user = prev // honor a persistent SetTokenCallback as the streaming sink
}
sink := streamfilter.NewSink(po.StopPrompts, user)
setCallback(l.state, sink.OnToken)
defer setCallback(l.state, prev) // restore (not nil) so persistent cb survives
```

- Pass `nil, 0` for the C antiprompt args in the `llama_allocate_params` call.
- After the C call returns, the function's text result is `sink.finish()`:
  - `Predict`  → `return applyPrefixTrims(sink.Finish(), text), nil`
  - `PredictResult` → `return sink.Finish(), int(nTok), nil`
  - `SpeculativeSampling` → `return applyPrefixTrims(sink.Finish(), text), nil`
- The C result buffer is still passed (the ABI requires it; `llama_predict` has
  no size param and writes unbounded, so `Predict`/`SpeculativeSampling` keep a
  C-heap buffer sized to `po.Tokens`) but its content is ignored.

`applyPrefixTrims` captures each function's **existing** prefix cleanup so output
contracts stay stable: `TrimPrefix(res, " ")`, `TrimPrefix(res, text)`,
`TrimPrefix(res, "\n")`. `PredictResult` keeps its current behavior of applying
**no** prefix trims.

### 4. `binding.cpp` `generate()` edit

Delete the antiprompt suffix-compare loop (current lines ~195-202):

```cpp
for (size_t a = 0; a < bp->antiprompt.size(); a++) {
    const std::string &ap = bp->antiprompt[a];
    if (/* out ends with ap */) { stop = true; }
}
```

Keep: the `llama_vocab_is_eog` check, the `tokenCallback(st, buf.data()) == 0 →
stop = true` halt, the loop bound (`n_predict`, `n_ctx`), and `out` accumulation
(the C return buffer is still produced; Go just ignores its content).

## Data flow

```
C generate() loop
  └─ sample token id
     ├─ llama_vocab_is_eog? ─ yes ─► break (natural end)
     └─ decode piece ─► tokenCallback(state, piece)   [C → Go bridge]
                          └─ sink.OnToken(piece)
                               └─ Filter.Push(piece) → (emit, stop)
                                    ├─ emit != "" → buf += emit; user(emit)
                                    └─ return !stop   (false halts C loop)
  (loop ends: EOG, token limit, n_ctx, or OnToken==false)
        │
        ▼
  Go: result = sink.Finish()           // Filter.Flush() → buf += remainder
        │                              // buf is stop-trimmed + UTF-8-valid
        ▼
  apply per-function prefix trims → return
```

## Behavior changes & compatibility

- **Stops are now correctly trimmed**, including when split across pieces — this
  is the fix. `Predict` and `SpeculativeSampling` previously used a buggy
  `TrimRight` cutset; `PredictResult` previously returned text that could include
  the trailing stop sequence (it did no trimming). After this change all three
  return text with the matched stop and everything after it removed.
- **UTF-8 is never split** mid-character in streamed pieces or the return value.
- **Streaming can be bursty**: when the pending tail is an ambiguous stop-prefix
  or an incomplete multibyte char, the `Filter` holds the whole pending buffer
  until the next piece disambiguates, then flushes it. This matches Ollama's
  runner and is inherent to correct detection. Token-by-token consumers see the
  same total text, just occasionally coalesced.
- **Persistent `SetTokenCallback` is preserved** across calls (restored on defer
  rather than nil-ed).
- No API/ABI changes: `binding.h`, `options.go`, and all exported Go signatures
  are unchanged.

## Stop semantics

- `streamfilter.New(po.StopPrompts)` — empty/nil stops ⇒ pure pass-through except
  for UTF-8 hold-back (matches the package's documented `Filter` contract).
- Earliest stop wins; the matched stop and everything after it are dropped.

## Error handling

- `Filter` is total (no errors). `streamSink.onToken`/`finish` cannot fail.
- C error returns (`ret != 0`, `full < 0`, OOM) are handled exactly as today and
  short-circuit before `sink.finish()`.
- A `nil` user callback is supported (accumulate-only).

## Testing

**Headless (no cgo, no model) — added:**
- `streamfilter.Sink` unit test driving `OnToken`/`Finish` directly with
  scripted pieces (no C, no model):
  - no stops, ASCII → each piece emitted, return == concatenation.
  - stop within one piece → emit text before stop, `onToken` returns false,
    remainder dropped, return text excludes the stop.
  - **stop split across pieces** (`"<"`, `"end>"`, stop `"<end>"`) → first
    `onToken` emits nothing and returns true; second returns false; return text
    is everything before `<end>`.
  - **multibyte split across pieces** → held then emitted intact; return text is
    valid UTF-8.
  - user callback returns false → `onToken` returns false (user halt) and the
    accumulator still reflects emitted text.
  - `finish` forwards and appends the buffered remainder at natural end.
- Existing `streamfilter` primitive + `Filter` tests are unchanged and remain the
  source of truth for the filter logic itself.

**End-to-end (manual, requires a built cgo binary + a GGUF model) — scripted &
documented for the maintainer to run:**
1. Greedy (`SetTemperature(0)`) output is byte-identical to before this change
   when no stop is hit.
2. Stop split across two pieces is detected and trimmed from both the streamed
   output and the returned string.
3. A response containing a multibyte character at a piece boundary streams and
   returns valid UTF-8 (no replacement characters / broken bytes).
4. A persistent `SetTokenCallback` still fires across successive `Predict` calls.

`go test ./...` (pure-Go packages) and `go build ./...` must stay green;
`golangci-lint` clean.

## Build / integration

- Rebuild the binding after the `binding.cpp` edit: `bash scripts/binding.sh`
  (recompiles `binding.cpp` into `libbinding.a`; does **not** rebuild llama.cpp).
- `binding.h` is untouched, so `llama.go`'s cgo signatures are unaffected.

## Risks

- **Rebuild toolchain** (MinGW g++): mitigated — `libbinding.a` is already built
  in this checkout, so the toolchain is present. Only `scripts/binding.sh` runs.
- **Return-text behavior change** for `PredictResult` (trailing stop now
  trimmed): intended and consistent with `Predict`; called out above and covered
  by the smoke test.
- **Model-dependent validation**: the split-stop / split-UTF-8 end-to-end checks
  can only be confirmed with a real model — done by the maintainer (manual smoke
  test), not in CI.
