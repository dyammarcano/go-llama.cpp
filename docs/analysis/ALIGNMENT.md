<!-- align head=18b05d3 date=2026-07-20 branch=main -->
# go-llama.cpp — Alignment Brief

**Headline:** T4 Library · KEEP GOING · 8 above-the-line essentials (~8–11 sessions) · CPU libs build clean (`task build:cpu` exit 0) but `go build ./examples` fails on 28 undefined symbols and cgo tests can't build.

## Verified state
- HEAD `18b05d3` on `main`, in sync with origin — _proof:_ `git rev-list --left-right --count origin/main...HEAD` → `0 0`
- Working tree carries uncommitted work — _proof:_ `git status --short` → ` M llama.cpp`, `?? docs/analysis/`
- Uncommitted submodule pin bump: gitlink `19e92c33` → checkout `178a6c44` (upstream tag **b10069**); 667 commits, 1369 files, +196510/−55833 — _proof:_ submodule diff probe
- Submodule dirty at `ggml/src/ggml-cpu/ggml-cpu.c` **by design** — the `GGML_NO_THREAD_POWER_THROTTLING` MinGW workaround is re-applied every build — _proof:_ `scripts/llamacpp.sh:19`
- Static analysis clean — _proof:_ `go vet ./...` exit 0
- CPU backend builds — _proof:_ `task build:cpu` exit 0, 232/232 ninja targets, 7 static libs → `llama.cpp/build-cpu/`
- Binding compiles against the new pin — _proof:_ `g++ -std=c++17` compile of `binding.cpp` vs b10069 headers, exit 0
- Examples do **not** link — _proof:_ `go build ./examples` exit 1, 28 undefined refs (`__emutls_v._ZSt11__once_call`, `std::codecvt<wchar_t,char,int>`, `__codecvt_utf8_utf16_base<wchar_t>::do_*`)
- Tests: cgo root package fails to build; pure-Go packages pass — _proof:_ `go test ./...` exit 1; PASS: `gguf`, `logitbias`, `streamfilter`
- llama.h delta `19e92c33..b10069` is mostly additive (`llama_ftype_name`, `llama_model_ftype`, `llama_model_n_layer_nextn`, `LLAMA_FTYPE_MOSTLY_Q2_0`, fields `n_outputs_max`, `ctx_other`); two breaks — `llama_set_warmup` DEPRECATED and 6 `gguf_get_tensor_*` signatures changed — neither touches this repo — _proof:_ grep of all non-`llama.cpp` sources → no matches
- Build-dir freshness: `build/` 7 `.a` @2026-05-29 (STALE), `build-cpu/` 7 `.a` @2026-07-20 (FRESH), `build-vulkan/` 0 `.a`, `build-cuda/` 0 `.a` — _proof:_ directory listing with mtimes
- Toolchain: go1.26.5, MinGW g++ 16.1.0, cmake 4.4.0, ninja 1.13.2 (shim was broken; repaired via `scoop install ninja` this session) — _proof:_ version probes
- Repo posture: authed `dyammarcano`; no open PRs; **issues DISABLED**; 0 stars, 0 forks — _proof:_ `gh repo view` / `gh pr list`
- Rollback cut exists only locally: annotated tag `llama.cpp-19e92c3` on `main@18b05d3f`, **never pushed**, on one machine — _proof:_ `git tag -l` + no origin ref
- License is genuine MIT and agrees with the README claim — SATISFIED, not a gap — _proof:_ `LICENSE:1-3` ("Copyright (c) 2023 go-skynet authors"). An earlier scout pass wrongly reported LICENSE absent; corrected.

## The line
From `/project:horizon` this session — see `docs/analysis/COMPLETION.md`.

**Tier: T4 LIBRARY**, user-confirmed (they intend others to import this). Honesty note: every observable signal — 0 stars, 0 forks, issues disabled, no releases, upstream module path — reads **T1-Personal**. The user overrode to T4 and the bar was held at T4. That override is precisely why the distance is large rather than nearly-closed.

**Verdict: KEEP GOING.** Distance: **8 above-the-line essentials, ~8–11 sessions**; 3 blocked; 8 items below the line. This is a **baseline run** — there is no prior `COMPLETION.md`, so **no burndown delta exists**.

Above the line, ordered:
1. Repoint the 7 LDFLAGS paths at `link_static_windows.go:10` → `build-cpu/`, and delete the stale `llama.cpp/build/` tree
2. Full `go test ./...` against b10069
3. Fix CI so it runs at all
4. Measure coverage + settle a `TEST_MODEL` fixture strategy
5. Resolve the 5 silent stubs
6. Document the 8 core exports, fix `options.go:144`, settle `Set*`/`With*` naming before the first tag
7. Sentinel / typed errors
8. Truthful README for this fork + `doc.go` + CHANGELOG

**The key is item 1** — roughly half a session, and it gates four of the eight.

## Drift reconciled
Mode: **fix**. This supersedes the earlier `check`-mode brief written today. Fix mode mutates **docs only**; source was deliberately left untouched.

**Applied to `README.md` (4 edits):**
- CPU/CUDA/Vulkan support claim softened to "build targets exist"; CUDA and Vulkan explicitly qualified as unverified against the current pin (their build dirs hold 0 `.a`)
- Added a "Known issue (2026-07-20)" block before the `go run ./examples` snippet, flagging the link failure and naming the cause
- "## Acceleration" → "## Acceleration (legacy — pre-Task, non-functional)" with a caveat that its `make libbinding.a` blocks cannot run (no Makefile exists)
- "## GPU offloading" → same legacy treatment
- Doc link `blob/master/...` → `blob/main/...`

**Deliberately not changed:** the README MIT line — license claims are the owner's call. Subsequently verified correct against `LICENSE:1-3`.

**Not applied (out of scope — source):** `link_static_windows.go:10` (7 paths); `link_static_windows.go:6` wrong script attribution (credits `.scripts/build-llamacpp.sh`; the live path is `Taskfile.yml:24-25` → `scripts/llamacpp.sh` + `scripts/binding.sh`); `go.mod:1` module path; the CI workflow files.

**CONTRADICTIONS (recommendations — source fixes still owed):**
1. **Stale-artifact trap.** `link_static_windows.go:10` reads `llama.cpp/build/*.a` (7 paths, incl. `vendor/cpp-httplib/libcpp-httplib.a`) while `scripts/llamacpp.sh:12` writes `build-$BACKEND` (= `build-cpu`) and `:25` `rm -rf`s it. A populated stale `build/` from 2026-05-29 still exists, so the link resolves **files** but fails on **28 symbols** against MinGW 16.1. Pre-existing; independent of the pin bump.
2. **CI HAS NEVER RUN.** `.github/workflows/test.yaml:33,54,75` and `test-gpu.yaml:49` all invoke `make test`, but no Makefile exists (replaced by `Taskfile.yml:21`/`:44`). Additionally `test.yaml:4-6` and `test-gpu.yaml:6-8` push-gate on `branches: [master]` while the branch is `main`. This resolves an Open Question from the earlier check pass: `gh run list --status failure` was empty because there are **no runs**, not because CI is green. Windows — the only platform actually developed on — has no CI job at all (matrix is ubuntu / macOS / macOS-metal plus a self-hosted GPU runner, `test-gpu.yaml:18`).
3. **Effective coverage ≈2%.** `llama_test.go:13` gates every meaningful spec on `os.Getenv("TEST_MODEL")`; `:38,:51,:83,:109` are `Skip()`; `TEST_MODEL` is set in neither workflow. Even on a green CI exactly one assertion runs (`llama_test.go:16-20`). ~61 exported symbols in the cgo package, 1 covered.
4. **Five exported ABI functions are `(void)`-cast stubs returning 1:** `binding.cpp:458` `get_embeddings`, `:463` `get_token_embeddings`, `:476` `load_state`, `:481` `save_state`, `:469` `SpeculativeSampling` (exported at `llama.go:268`, exercised by `llama_test.go:73` **against the stub**). Embeddings and state save/load were scoped as DELIVERED (`MODERNIZATION-SCOPE.md:38,42`), not deferred. No error signal distinguishes "stub" from "failed".
5. **Module identity is unusable.** `go.mod:1` = `github.com/go-skynet/go-llama.cpp` but the repo is `dyammarcano/go-llama.cpp` → no consumer can `go get` this fork. `README.md:1,24` still advertise the upstream badge and clone URL; the import snippets at `README.md:24,64,80,100` resolve to upstream, which has no `gguf` or `streamfilter` package.

**CHECKOFF (doc drift, zero real work):** 132 of 184 checkboxes across 6 superpowers plans are unticked while the code is merged and green. Marked N/A-at-tier in `COMPLETION.md` — counting them would inflate the burndown.

## Honest next move
1. Repoint the 7 LDFLAGS paths (`link_static_windows.go:10`) to `build-cpu/` **and delete the stale `llama.cpp/build/` tree**. The deletion matters on its own: it is what makes this present as 28 missing symbols instead of an honest "file not found", and it will mislead the next person identically. Then re-run `go build ./examples` and `go test ./...`.
2. **Only after those pass**, commit the b10069 pin. Do not commit an unverified pin.
3. Free and independent — do it now: push the `llama.cpp-19e92c3` rollback tag. One command, retires a single-machine-only risk.

## Open questions / Unverified
- Will linking against a fresh `build-cpu/` actually succeed? **Unproven.** The stale-lib attribution is inferred from mtimes and path inspection; no real link against `build-cpu/` has been performed.
- Does b10069 work at **runtime**? Zero tests have executed against it. 667 commits / 1369 files unprobed. A silent cgo struct-layout or parameter drift would surface only in a consumer's runtime.
- Is the `build/` path in `link_static_windows.go` intentional (some workflow populating it)? Asked twice this session; never answered.
- **Operator decision pending:** module identity — rename `go.mod` to `dyammarcano`, or document a `replace`-directive consumption model? Either choice breaks every existing importer.
- **Operator decision pending:** cgo distribution model — prebuilt per-platform artifacts as release assets, vs a documented `task build:cpu` prerequisite? This gates what a v0.1.0 release *means*, not just its mechanics.
- Linux (`llama.go:6`) and Darwin (`llama.go:7`) cgo paths: claimed, never verified here — needs machines not available; delegate to CI.
- CUDA and Vulkan: build dirs empty, entirely unverified against b10069.
- Whether the self-hosted GPU runner (`test-gpu.yaml:18`) still exists.
- Should the `Set*`/`With*` option-naming split be resolved before the first tag? After a semver tag it becomes a deprecation cycle.
- Should issues be re-enabled before any T4 announcement? The only inbound support channel is currently closed.
