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
