# CI vetting for ddrs — design

**Date:** 2026-09-03
**Status:** approved (brainstormed 2026-09-02/03)
**Goal:** every change reaching `master` is vetted by the routing acceptance
gates — the DDR sandbox parity test, the full default test suite, and the
Juniata end-to-end metric floors — with no dependency on any developer
machine being online.

## Decisions already made

- **Posture: hybrid.** CI runs on both PRs and `master` pushes. Branch
  protection requires the checks for PR merges, but admin direct pushes stay
  possible; a direct push is vetted immediately after by the push-triggered
  run (detection, not prevention, for that path).
- **Runners: GitHub-hosted only** (`ubuntu-latest`; free for this public
  repo). No self-hosted runner — a public repo executing PR code on the
  workstation is a security hole, and CI must not depend on the workstation
  being up. GPU (`--features cuda`) and real-data tests are out of CI scope.
- **The release acceptance job runs on every PR and every master push**,
  not master-only. Catching a metric regression at PR time is the point.
- **Local hook: pre-push, not pre-commit.** The meaningful gates need
  compilation; a multi-minute pre-commit punishes small commits, and
  "vetted pushes" is a push-time property.

## Non-goals

- No `cargo fmt --check` or clippy gate. The tree is not fmt-clean today
  (e.g. `examples/benchmark_hydrograph.rs`) and reformatting it is not this
  project's scope. Revisit only as its own change.
- No GPU coverage, no `/mnt/ssd1` data-dependent coverage, no self-hosted
  runner, no performance/timing gates.
- No change to which tests exist. CI runs the gates the repo already has
  (including the 2026-09-02 acceptance tests `ddr_sandbox_match` and
  `juniata_acceptance`).

## Component 1: `.github/workflows/ci.yml`

### Triggers

```yaml
on:
  pull_request:
  push:
    branches: [master]
  workflow_dispatch:
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

**No path filters.** A required status check skipped by a path filter leaves
the PR stuck on "expected — waiting for status". Doc-only PRs are rare and
warm-cached runs are cheap; always-run is the correct trade for required
checks.

### Shared job setup (both jobs)

1. `actions/checkout@v4`
2. `dtolnay/rust-toolchain@stable` (matches `docs.yml` convention)
3. CUDA toolkit install — see below
4. `Swatinem/rust-cache@v2` with per-job `shared-key` (`ci-debug` /
   `ci-release`) so the two target trees cache independently
5. `timeout-minutes: 90`

`cmake` is preinstalled on `ubuntu-latest` (needed by the static
netcdf/HDF5 build). No apt packages expected beyond CUDA.

### The CUDA toolkit step

`burn-cuda` and `cudarc` are non-optional dependencies
(`use burn_cuda::Cuda;` is unconditional in five modules), so **compilation
requires a CUDA toolkit even though CI never executes GPU code**. Install
CUDA 12.x via `Jimver/cuda-toolkit` (pinned to a release tag) with a minimal
sub-package set (compiler/headers/runtime stubs; exact set determined during
bring-up — `cudarc` uses `cuda-version-from-build-system`, so the toolkit
must be discoverable via the usual env/paths the action sets).

This step is the design's main risk: it cannot be verified locally. Plan for
one or two iteration commits on the bring-up PR before first green.
Fallback if the action fights us: NVIDIA's apt repo with
`cuda-toolkit-12-x` metapackage (slower, well-documented).

### Job `test` — full default suite, debug

```bash
cargo test --features fixtures --no-fail-fast
```

What this vets, by existing repo design (no test changes needed):

- `ddr_sandbox_match` — machine-enforced invariant 1 (fixtures are
  git-tracked; `fixtures/sandbox/` predates the `/fixtures/` ignore rule and
  is in the index, so a clean clone has it).
- The 4-test KAN parity sweep (`tests/fixtures/` is tracked; enabled by
  `--features fixtures`).
- Everything else in the default suite. Data-dependent tests
  (`training_verification`, real-zarr loaders, …) self-skip on missing
  paths; `--features cuda` tests are compiled out; `juniata_acceptance`
  self-skips under `debug_assertions`.

`--no-fail-fast` so one failure still reports the rest of the suite.

### Job `acceptance` — release gates

One release build (LTO thin) serving two sequential steps:

```bash
mkdir -p output
cargo run --release --example compare_ddr_sandbox      # exits 1 unless ABSOLUTE MATCH
cargo test --release --test juniata_acceptance -- --nocapture
```

The second step is the end-to-end floor: train-and-test on the committed
Juniata bundle, asserting routed NSE ≥ 0.75, KGE ≥ 0.80, beats the summed-Q'
baseline, and baseline NSE ∈ [0.67, 0.72] (reference run: NSE 0.7903 /
KGE 0.8810 / baseline 0.6947 in 18.3 s locally).

The jobs are independent (no `needs:`) so debug results arrive without
waiting on the release build.

### Expected timings

First uncached run: 30–60 min per job on 2-core runners (the
burn/cubecl/netcdf-static tree dominates). Warm-cached: roughly 5–15 min.
If the release target tree blows the runner disk (~14 GB usable), add a
disk-cleanup step (`jlumbroso/free-disk-space` or manual `rm` of
preinstalled SDKs) to the `acceptance` job only.

## Component 2: branch protection (applied after first green run)

Required checks must exist before GitHub lets them be required, so this is
step two. Applied via `gh api` (documented in the workflow's header comment
so it is reproducible):

```bash
gh api -X PUT repos/taddyb/ddrs/branches/master/protection --input - <<'EOF'
{
  "required_status_checks": {
    "strict": false,
    "checks": [{"context": "test"}, {"context": "acceptance"}]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
EOF
```

(JSON body rather than `-f`/`-F` flags: the endpoint requires literal JSON
`null` for the review/restriction fields, which flag syntax cannot express.)

Semantics: PRs cannot merge with red checks; PR review is not required;
`enforce_admins=false` preserves the direct-push escape hatch (vetted
post-hoc by the push run). `strict=false` (no "branch up to date" rule) —
this is a single-developer repo and the rule adds rebase churn for little
safety.

## Component 3: local pre-push hook

Committed at `.githooks/pre-push`:

```bash
#!/usr/bin/env bash
# Fast local gate before any push. Bypass: git push --no-verify
set -euo pipefail
cargo test --test ddr_sandbox_match
```

- Enabled per-clone (opt-in, never forced):
  `git config core.hooksPath .githooks`
- Scope is deliberately minimal — the sandbox parity test runs in seconds on
  a warm build. The full suite and release acceptance belong to CI, not the
  push path.

## Documentation updates (same change)

- `CLAUDE.md` Commands section: one short block — CI runs `test` +
  `acceptance` on every PR and master push; hook setup one-liner.
- `.claude/skills/ddrs-dev/SKILL.md` + `references/testing.md`: a CI section
  stating what CI does and does not vet (no GPU, no real-data tests), so
  nobody mistakes green CI for a full Tier-A/B pass on data-touching
  changes.
- README: badge + a sentence, optional.

## Rollout plan

1. Branch + PR carrying `ci.yml` and the hook — the bring-up PR exercises
   the PR path itself. Iterate on the CUDA step until both jobs are green.
2. Merge; confirm the push-triggered run goes green on `master`.
3. Apply branch protection (Component 2).
4. Docs/skill updates and hook setup note (can ride the same PR).

## Risks and open items

- **CUDA install on runners** — main risk, untestable locally; mitigated by
  iterating on the bring-up PR. Fallback documented above.
- **A "skipping" test that actually panics without data** — the first CI run
  is the audit that finds any such test; the fix is the standard
  path-check-and-return guard, applied to the offender only.
- **Runner disk** — watched on the bring-up PR; cleanup step if needed.
- **Git-dependency availability** — all burn/cubecl forks resolve from
  `github.com/taddyb/*` branches and `rskan` from a tag; if a fork branch is
  ever deleted, CI breaks loudly (that is a feature — the public build broke
  too).
- **Fork PRs** run with read-only tokens on GitHub-hosted runners; no
  secrets are used anywhere in the workflow, so there is nothing to leak.

## Acceptance criteria

- A PR with a deliberately broken routing change (or metric floor) shows
  red `test`/`acceptance` and cannot be merged.
- A doc-only PR still runs and passes both checks (no "expected — waiting"
  hang).
- Direct push to `master` triggers the same two jobs.
- `git config core.hooksPath .githooks` + a push runs the sandbox gate
  locally; `--no-verify` bypasses it.
