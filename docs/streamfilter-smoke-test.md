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
returned string contain no replacement characters (U+FFFD) or broken bytes — i.e.
`utf8.ValidString(result)` is true.

## 4. Persistent SetTokenCallback survives across calls

Call `SetTokenCallback(cb)` once, then run two `Predict` calls without a per-call
`SetTokenCallback` option. Confirm `cb` fires (with filtered pieces) on both.
