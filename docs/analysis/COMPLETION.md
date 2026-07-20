# COMPLETION — go-llama.cpp

**Project:** go-llama.cpp (fork at `dyammarcano/go-llama.cpp`, HEAD `18b05d3`, branch `main`)
**Confirmed tier:** **T4 — LIBRARY** (user-confirmed; others are intended to import this)
**Verdict:** **KEEP GOING**
**Distance to done:** **8 above-the-line essentials (~8–11 sessions)** · 3 blocked-not-done · 8 consciously deferred
**Date:** 2026-07-20

> **Tier honesty note.** Every observable signal — 0 stars, 0 forks, issues disabled, zero
> releases, zero semver tags, `go.mod:1` still on the upstream module path — reads **T1
> Personal**. The operator explicitly overrode that to **T4 Library**. That override is
> respected here and the bar is *not* lowered. The consequence is that the gap to done is
> **large**, not nearly-closed. This is a "keep going, and here is exactly how far" map.

> **Progress since baseline (2026-07-20, same day).** The DoD table below is the original
> snapshot at HEAD `18b05d3` and is left intact for the record; the next `/project:align`
> will regenerate it. Shipped since:
> - **D1 Builds green → CLOSED.** `cbbe526` repointed the 7 CPU LDFLAGS paths to
>   `build-cpu/` and deleted the stale `build/` tree. `go build ./examples` links (10.9 MB).
> - **D2 Tests pass on the pin → ADVANCED.** `go test ./...` exit 0; b10069 verified at
>   runtime — real inference across 4 models (dense Q8_0/Q4_K_M + MoE IQ1_M). Pin committed
>   `6b690ba`. (Coverage breadth still per D8.)
> - **D5 Stubs implemented → HONESTY-CLOSED.** `d4e9a83` makes all 5 stubs fail loudly with
>   `ErrNotImplemented` instead of silently returning empty. (Real implementations still owed.)
> - **D3 API documented → ADVANCED.** `d4e9a83` documents the 8 core exports and fixes the
>   `SetMMap` doc bug. (`Set*`/`With*` naming split still open — B-list decision.)
> - **D4 Typed errors → CLOSED.** `d4e9a83` adds `errors.go` sentinels wrapped with `%w`.
> - **D15 Rollback safety → CLOSED.** Tag `llama.cpp-19e92c3` pushed to origin.
> - **Still above the line:** D6 module identity (decision), D7 CI, D8 coverage/fixture,
>   D9 onboarding docs, D10 release; B-list: naming split, distribution model.

---

## 1. Burndown delta

**No prior `docs/analysis/COMPLETION.md` exists. This is the first run — baseline.**
There is nothing to delta against; the numbers above are the starting burndown, not a trend.
The next run measures movement from 8 essentials / ~8–11 sessions.

---

## 2. Definition-of-Done checklist (T4 Library)

| # | DoD item | Status | Signal |
|---|---|---|---|
| D1 | Builds green from a clean checkout | **GAP** | `link_static_windows.go:10` points at `llama.cpp/build/`; `scripts/llamacpp.sh:12` writes `build-cpu/` and `:25` `rm -rf`s it. A stale `build/` (7 `.a`, dated 2026-05-29) satisfies the *file* lookup but fails **28 symbols** under MinGW 16.1 → `go build ./examples` exit 1 |
| D2 | Test suite passes on the shipped native pin | **GAP** | `go test ./...` exit 1 (root cgo pkg fails to build). `gguf`/`logitbias`/`streamfilter` PASS. **Zero tests have ever run against b10069** — the uncommitted pin bump is 667 commits / 1369 files / +196510/-55833 |
| D3 | Exported API documented + stable-shaped | **GAP** | 8 core exports undocumented: `llama.go:64,70,101,105,120,226,268,335`. `Free` (`:101`) is the sole release call with no "must be called"/double-Free semantics. `options.go:144` copy-paste doc bug ("SetContext sets the context size." above `SetMMap`). Incoherent `Set*`/`With*` split in one option family |
| D4 | Typed/sentinel errors for `errors.Is`/`As` | **GAP** | 15 `fmt.Errorf`, **zero** exported `Err*` or typed errors in root pkg (`llama.go:94,114`) |
| D5 | Advertised API is actually implemented | **GAP** | 5 exported ABI functions are `(void)`-cast **stubs returning 1**: `get_embeddings` (`binding.cpp:458`), `get_token_embeddings` (`:463`), `load_state` (`:476`), `save_state` (`:481`), `SpeculativeSampling` (`:469`, exported `llama.go:268`, and `llama_test.go:73` asserts against the stub). Scoped as **delivered**, not deferred (`MODERNIZATION-SCOPE.md:38,42`, sequencing `:84` step 4). Go layer gives no signal distinguishing "stub" from "failed" |
| D6 | Importable by a third party | **GAP** | `go.mod:1` = `github.com/go-skynet/go-llama.cpp`, repo is `dyammarcano/go-llama.cpp` → `go get` fails on module-path mismatch. **No consumer can import this fork today.** Depends on operator decision B1 |
| D7 | CI verifies the matrix | **GAP** | Three independent breaks: (a) `.github/workflows/test.yaml:33,54,75` and `test-gpu.yaml:49` run `make test` — **no Makefile exists** (replaced by `Taskfile.yml:21`/`:44`); (b) `test.yaml:4-6`, `test-gpu.yaml:6-8` gate on `branches: [master]`, branch is `main` → **workflows never trigger** (this is why `gh run list --status failure` is empty — no runs, not green runs); (c) no Windows job — the only platform actually developed on. The submodule pin bump is gated by nothing |
| D8 | Meaningful coverage of the reason-to-exist | **GAP** | `llama_test.go:13` gates every real spec on `os.Getenv("TEST_MODEL")`; `:38,51,83,109` are `Skip()`. `TEST_MODEL` appears in **neither** workflow. Even on hypothetically-green CI exactly **one** assertion runs (`llama_test.go:16-20`, "fails with no model"). ~61 exported symbols (llama.go 13 + options.go 48), 1 covered ≈ **2%**. Coverage never measured. The three pure-Go pkgs are genuinely well tested — the hole is confined to the cgo binding |
| D9 | Onboarding docs a stranger can follow | **GAP** | `README.md:24,64,80,100` clone/import snippets all resolve to **upstream go-skynet**, which has no `gguf` or `streamfilter` package — the quickstart works for nobody but the owner. `README.md:1` still shows the upstream badge. No `doc.go` in any package. No CHANGELOG, no CONTRIBUTING. Native-prebuild requirement is normal for cgo bindings and is *not* the gap — it being undocumented for this fork is |
| D10 | A consumable release exists | **GAP** | Zero semver tags, zero releases, no CHANGELOG. Legitimately **last** — needs D2 green, D6 importable, B2 answered |
| D11 | License is clear and consistent | **SATISFIED** | `LICENSE:1-3` genuine MIT, "Copyright (c) 2023 go-skynet authors", agrees with the README MIT claim. **Not a gap** |
| D12 | No leaked cgo types across the API boundary | **SATISFIED** | `LLama.state` unexported; build tags well documented |
| D13 | Inbound support channel | **BLOCKED** | GitHub Issues **disabled**. One operator toggle, not an engineering task |
| D14 | Linux / Darwin verified | **BLOCKED** | Machines not available on this host; delegate to CI (rides on D7) |
| D15 | Rollback safety | **GAP (cheap)** | Annotated tag `llama.cpp-19e92c3` on `main@18b05d3f` exists **locally only, never pushed, single machine**. Pushing it is the cheapest risk retirement available |
| D16 | Plan-doc checkoff hygiene | **N/A at T4** | 132 of 184 checkboxes unticked across 6 superpowers plans (gguf-reader 22, vram-estimator 26, streamfilter 16, streamfilter-wiring 29, sampler-wiring 28, grammar-wiring 11) while the code is merged and green. Pure doc drift — **zero real work**. Consumers never see it |

---

## 3. The line

### ABOVE THE LINE — finite essentials, critical-path ordered

Eight items. Nothing here is optional for a T4 library; nothing here is polish.

1. **Repoint the 7 stale LDFLAGS paths in `link_static_windows.go:10` → `build-cpu/`, and DELETE the stale `llama.cpp/build/` tree.** (S — ~0.5 session)
   *Both halves are required.* Repointing alone leaves the trap: the stale tree is precisely
   what makes this present as "28 missing symbols" instead of an honest "file not found," and
   it will mislead the next person identically. Note the count is **7** paths, not 6 — the
   seventh is `vendor/cpp-httplib/libcpp-httplib.a`.
   **Gates 2, 3, 4, 8. Nothing downstream is measurable until this is green.**
2. **Run a full `go test ./...` against b10069.** (M — ~1–2 sessions incl. fallout)
   First real signal on 667 commits of upstream drift. `go vet ./...` is already exit 0 and
   `task build:cpu` produces 232/232 + 7 static libs, so the native side is healthy — the
   unknown is the Go↔cgo boundary.
3. **Fix CI so it actually runs.** (M — ~1 session)
   `master` → `main` on both push gates; `make test` → `task build:cpu` + `task test`; add a
   Windows job. A pin bump of this size gated by nothing is the single largest silent-breakage
   surface. Can run **in parallel** with 1–2.
4. **Measure coverage and decide the model-fixture strategy.** (M — ~1–2 sessions)
   2% on the cgo binding is the dominant risk: a silent struct-layout or param drift across
   667 commits surfaces only in a downstream consumer's runtime. Either wire `TEST_MODEL` with
   a small fixture in CI, or accept a documented lower bar — but *decide*, don't leave it
   implicitly skipped.
5. **Resolve the 5 stubs: implement or fail loudly.** (M — ~1–2 sessions)
   `get_embeddings`, `get_token_embeddings`, `load_state`, `save_state`, `SpeculativeSampling`
   silently return 1. At T4 a stub that returns success is worse than a missing function.
   Minimum acceptable: a typed error from the Go layer saying "not implemented on this build."
   These were scoped as *delivered*, so this is closing a promise, not new work.
6. **Document the 8 core exports; fix the `options.go:144` doc bug; settle `Set*` vs `With*`.** (M — ~1 session)
   The naming split **must** be resolved *before the first semver tag* or it becomes a
   deprecation cycle. `Free` needs explicit must-call and double-Free semantics.
7. **Add sentinel/typed errors so callers can `errors.Is`/`As`.** (S — ~0.5 session)
   Same window as 6 — post-tag this is a breaking change.
8. **Make the README truthful for *this fork*.** (S — ~0.5 session)
   Clone URL, import paths, badge. Add `doc.go` per package and a CHANGELOG. Document the
   `task build:cpu` native prerequisite explicitly — that requirement is normal, its absence
   from the docs is not.

**Immediate freebie, not counted as an essential:** push the rollback tag
`llama.cpp-19e92c3` now. It costs one command and retires the largest single-machine risk.

### BLOCKED-NOT-DONE — real, but not your next action

The project is **not done** while these are open, but they need an operator decision or a
machine that does not exist here.

- **B1 — Module identity: `go-skynet` vs `dyammarcano`.** Mechanical once decided (~0.5
  session), but the decision breaks every existing importer either way. This gates D6 and
  therefore D10. *Decide early — it can run in parallel with essentials 1–2.*
- **B2 — cgo distribution model.** Prebuilt release artifacts vs a documented `task build:cpu`
  prerequisite. This is a real gate on **what a release means**, not just its mechanics. Gates D10.
- **B3 — Linux/Darwin verification (D14) and enabling Issues (D13).** B3 rides on essential 3;
  enabling Issues is a single repo toggle.

### BELOW THE LINE — consciously deferred, not hidden

Listed so the deferral is deliberate rather than blind. None of these block T4-done.

- CUDA / Vulkan verification (README already qualifies these as unverified this session).
- Metal / OpenBLAS legacy sections (README already marks Acceleration + GPU-offloading
  "legacy — pre-Task, non-functional").
- The 132 unticked plan checkboxes across the 6 superpowers plans — pure doc drift, zero real work.
- `MODERNIZATION-SCOPE.md` still reading forward-looking after the rewrite + CMake migration landed.
- `ROADMAP.md` / `BACKLOG.md` / `ISSUES.md` / `ARCHITECTURE.md` scaffolding — global convention
  docs, **not** T4 consumer requirements. Deliberately not counted as gaps.
- Coverage beyond a sane baseline (chasing a percentage past "the binding is exercised").
- Prebuilt binary artifacts per platform (contingent on B2; not required if the prerequisite
  is documented instead).
- The self-hosted GPU runner in `test-gpu.yaml:18`, which likely no longer exists — either
  delete the workflow or leave it dormant; it is not on the path to T4.

---

## 4. Critical path

```
[push rollback tag]                       <- do this first, costs one command
        |
 1. repoint 7 LDFLAGS + DELETE stale build/     (gates everything below)
        |
 2. go test ./... vs b10069                     (first real signal, 667 commits)
        |
 3. CI actually runs + builds natively   ---- can start in parallel with 1-2
        |
 4. coverage measured + fixture decision
        |
 5. stubs: implement or error loudly
        |
 6. + 7. API docs + naming + typed errors       (MUST land before any semver tag)
        |
 8. README truthful for this fork
        |
 [B1 module rename] ---- decide in parallel, apply here
        |
 [B2 answered] -> tag v0.1.0                    (legitimately last)
```

## 5. The key

**Repoint the 7 LDFLAGS paths in `link_static_windows.go:10` to `build-cpu/` and delete the
stale `llama.cpp/build/` tree.**

It is half a session, it gates four of the eight essentials, and until it is green **every
other assessment in this document is unmeasured**. The stale tree is actively lying to you:
it turns "the linker cannot find the libraries" into "28 mysterious missing symbols." Delete
it, and the next failure you see will be an honest one.

## 6. Verdict

**KEEP GOING** — the native build is healthy (`go vet` exit 0, `task build:cpu` 232/232,
`binding.cpp` compiles clean against b10069 headers) and the pure-Go packages are genuinely
good work, but at the confirmed **T4 Library** bar the fork is not importable, not CI-verified,
not documented, ~2% covered on its reason to exist, and ships 5 exported functions that
silently return success while doing nothing.
