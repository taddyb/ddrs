# ddrs CLI UX cleanup + README — Design Spec

**Date:** 2026-06-01
**Scope:** Eliminate the first-run init/plan ping-pong, move `workflow` into `ddrs.yaml`, add a CPU-fallback smoke test, and document the setup flow in `README.md`.
**Out of scope:** The new `ddrs run --workflow evaluate` (auto-plots) feature. That gets its own spec.

---

## Background

After `cargo install --path .`, a first-time user hits this sequence:

```
ddrs init                       → "no ddrs.yaml found — run `ddrs plan` to bootstrap"
ddrs plan                       → "neither --workflow nor `workflow:` key set"
ddrs plan --workflow train-and-test
                                → "workspace not initialized at ./.ddrs; run `ddrs init`"
ddrs init                       → succeeds
ddrs plan --workflow train-and-test
                                → succeeds
ddrs run --workflow train-and-test
                                → succeeds
```

Six commands to reach the first run. Two root causes:

1. **`workflow` is a required CLI flag, not a config key.** The error message even claims `workflow:` in YAML is supported, but the code never reads it.
2. **`init` and `plan` each refuse to work until the other has run.** `init` waits for `plan` to bootstrap `ddrs.yaml`, then `plan` waits for `init` to create the workspace.

The README has no setup section, so users discover this dance by trial and error.

---

## Goals

1. **Three-command happy path:** `ddrs init → ddrs plan → ddrs run`.
2. **`workflow` lives in `ddrs.yaml`** as a top-level key, with `--workflow` as an optional CLI override.
3. **Smoke test runs on CPU** when no CUDA device is detected, so `init` works on laptops and CI.
4. **README documents the full setup**, including the `~/.cargo/bin` PATH gotcha.

Non-goals: changing what `plan`/`run`/`show`/`status`/`gc` already do, refactoring `plan_bootstrap.rs` beyond moving its call site.

---

## Design

### 1. `workflow:` as a top-level YAML key

**Config change** (`src/config.rs`):

Add `pub workflow: Option<Workflow>` to `Config`. `Workflow` already derives `Deserialize` with kebab-case in `src/cli/types.rs`. To avoid a layering inversion (config depending on cli), move `Workflow` (and only `Workflow`) from `src/cli/types.rs` to `src/config.rs`, then re-export it from `cli::types` so existing imports keep working.

**Resolution order** in `cli::plan::plan(...)` and `cli::run::run(...)`:

```
workflow = cli_flag_override
    .or(config.workflow)
    .ok_or(ConfigInvalid {
        path: config_path,
        source: "no `workflow:` key in ddrs.yaml; add `workflow: train-and-test` (or `train` / `eval`), or pass `--workflow <name>`."
    })?
```

**Cross-validation:** if both `mode:` and `workflow:` are set and disagree (e.g. `mode: training` + `workflow: eval`), error with:

> `ddrs.yaml: conflicting top-level keys — mode: training but workflow: eval. Pick one or align them (mode=training implies workflow ∈ {train, train-and-test}).`

The bootstrap template written by `plan_bootstrap.rs` (now called from `init`, see §2) gets a commented `workflow:` line so new users see the option immediately.

### 2. `ddrs init` as a one-shot first-run command

`init` consolidates everything needed before `plan` works. The phases:

| Phase | Always runs? | What it does |
|---|---|---|
| **A. Workspace** | yes | Creates `./.ddrs/` directory tree if missing. |
| **B. System probe** | yes | Runs `system::probe()`, writes `.ddrs/system.json`. Records GPU/driver/CUDA info. |
| **C. Smoke test** | yes (unless `--skip-smoke` or cached verdict reusable) | Runs the 5-reach sandbox parity check on CUDA if available, else NdArray (CPU). See §3. |
| **D. YAML bootstrap** | only if `ddrs.yaml` missing | Opens `$EDITOR` on a template (existing `plan_bootstrap::bootstrap_yaml` logic, moved). Skips with a clear error if stdin is not a TTY. |
| **E. Lock sources** | only if `ddrs.yaml` exists after Phase D | Reads `data_sources:`, fingerprints each path, writes `.ddrs/sources.lock`. |

If yaml already exists, Phase D is skipped. Re-running `init` is idempotent — Phase E re-fingerprints and rewrites the lockfile only if mtime/size changed.

Removing the chicken-and-egg means `plan` and `run` can assume the workspace, lockfile, and `ddrs.yaml` all exist. Their existing "not initialized" errors become genuinely actionable (they point at a single missing thing, not a loop).

### 3. Backend-aware smoke test

Current `cli::init::run_smoke()` hardcodes `burn_cuda::Cuda`. Replace with:

```rust
fn run_smoke(probe: &SystemProbe) -> Result<(bool, &'static str), CliError> {
    if probe.gpu.is_empty() {
        eprintln!("no CUDA detected — running CPU smoke (slower but functionally equivalent)");
        type I = burn::backend::NdArray<f32>;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
        Ok((r.passed, "cpu"))
    } else {
        type I = burn_cuda::Cuda<f32, i32>;
        let device = burn_cuda::CudaDevice::default();
        let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
        Ok((r.passed, "cuda"))
    }
}
```

**Smoke verdict cache key** (`system::smoke_key`) gains a `backend` term so a machine that loses its CUDA driver doesn't reuse a stale cuda-verdict for the cpu run, and vice versa:

```
driver=<v>;cuda=<v>;ddrs=<v>;sm=<v>;backend=<cuda|cpu>
```

`SmokeTestRecord` in `system.json` gains a `backend: "cuda" | "cpu"` field. Existing records without this field invalidate and trigger a re-run.

**GPU-required check moves out of init.** Currently `init` errors with *"no CUDA device detected … or build with --features cpu"* if `probe.gpu` is empty. Remove that check from init. CPU users get a working init. The check belongs in `run` for workflows that actually require GPU (i.e. all current ones — Training without GPU is infeasible on CONUS scale); add a clear pre-flight there:

> `run: workflow `train-and-test` requires a CUDA GPU; system probe found none. Smoke verified the routing core works on CPU, but production training does not.`

`plan` has no GPU requirement — it validates config and drift without touching CUDA.

### 4. Error message rewrites

| Current message | Replacement |
|---|---|
| `error: config invalid at <path>: neither --workflow nor `workflow:` key set` | `error: no `workflow:` key in <path>. Add `workflow: train-and-test` (or `train` / `eval`), or pass `--workflow <name>`.` |
| `error: workspace not initialized at ./.ddrs; run `ddrs init`` (from plan/run) | Unchanged in wording, but the user will no longer hit this after running `init` once. |
| `no ddrs.yaml found — run `ddrs plan` to bootstrap one, then re-run `ddrs init` to lock data sources.` | Removed. `init` now bootstraps `ddrs.yaml` itself. |
| `no CUDA device detected; install nvidia-driver ≥ 530 or build with --features cpu` | Removed from init; replaced by the smoke log line and the run pre-flight in §3. |

### 5. `README.md` setup section

Add a new section near the top (after the existing project description), structured as:

```markdown
## Getting started

### Install

```bash
cargo install --path .
```

This puts `ddrs` in `~/.cargo/bin/`. If that directory isn't on your PATH:

```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### First-time setup

From your project root:

```bash
ddrs init      # creates ./.ddrs/, probes GPU, runs smoke test, opens $EDITOR on ddrs.yaml, locks data sources
ddrs plan      # validates ddrs.yaml against locked sources, prints workflow summary
ddrs run       # executes the workflow, writes manifest + outputs under .ddrs/runs/<id>/
```

### What lives where

| Path | Written by | Purpose |
|---|---|---|
| `ddrs.yaml` | `ddrs init` (via $EDITOR) | Workflow + experiment config (mirrors DDR's `merit_training_config.yaml`) |
| `.ddrs/system.json` | `ddrs init` | GPU/driver/smoke-test record |
| `.ddrs/sources.lock` | `ddrs init` | Fingerprints of data_sources paths |
| `.ddrs/runs/<id>/manifest.json` | `ddrs run` | Per-run manifest (config + sources + git SHA + outputs) |
| `output/predictions_latest.zarr` | `ddrs run --workflow eval` or `train-and-test` Phase 2 | Predictions for plotting |
| `output/saved_models_*/epoch_*_mb_*.mpk` | `ddrs run --workflow train` or `train-and-test` Phase 1 | KAN checkpoints |
```

The README does **not** document advanced commands (`show`, `status`, `gc`) — `ddrs <cmd> --help` already covers them. Keep the README narrow: install + happy path.

---

## Architecture impact (blast radius)

**Files modified:**

- `src/config.rs` — add `workflow` field to `Config`, host `Workflow` enum (moved from cli/types.rs)
- `src/cli/types.rs` — re-export `Workflow` from config
- `src/cli/plan.rs` — change workflow resolution to `flag.or(config.workflow)`, update error message
- `src/cli/run.rs` — same workflow resolution change
- `src/cli/init.rs` — consolidate phases A-E, replace `run_smoke()` with backend-aware version, drop GPU-required error
- `src/cli/system.rs` — extend `smoke_key()` and `SmokeTestRecord` with backend field
- `src/cli/plan_bootstrap.rs` — no logic change, but called from init instead of plan
- `src/cli/run.rs` — add GPU pre-flight that errors when probe.gpu is empty
- `README.md` — new "Getting started" section
- `config/merit_training.yaml` — add `workflow: train-and-test` near the top so the canonical example is current

**Files removed/deprecated:** none.

**Public API changes:** `cli::plan_bootstrap::bootstrap_yaml` is no longer called from `cli::plan`. If any test or external caller depends on `plan` doing the bootstrap, those break. (Internal-only — confirmed by grep.)

**Tests affected:**

- `tests/cli_plan.rs` — update for new error message wording.
- `tests/cli_plan_bootstrap.rs` — bootstrap-from-plan tests need to move to init or be split.
- `tests/cli_init.rs` — add a CPU-smoke variant (force no-CUDA by stubbing the probe, assert smoke still runs).
- `tests/cli_workspace_uninit.rs` — should still pass; the workspace-uninitialized error from `plan`/`run` is unchanged.

---

## Concerns and risks

1. **Moving `Workflow` out of `cli/types.rs` breaks downstream imports.** Mitigation: re-export with `pub use crate::config::Workflow;` in `cli/types.rs`. All existing `use crate::cli::types::Workflow;` keep working.

2. **`mode:` ↔ `workflow:` cross-validation is new conditional logic.** If the rule is wrong (e.g. `mode: training` is actually permitted with `workflow: eval` for some edge case I don't know about), the new error blocks legitimate configs. Mitigation: explicit grep through DDR's own configs to confirm no such case exists; if found, weaken the cross-check to a warning.

3. **CPU smoke is functionally equivalent but slower.** Sub-second on a 5-reach sandbox in practice. If someone runs `init` repeatedly during dev, the verdict cache absorbs the cost — only changes to driver/cuda/ddrs versions trigger a re-run.

4. **The GPU pre-flight in `run` is a behavior change.** Today, a CPU-only user could (theoretically) try `ddrs run` and get a cryptic burn-side panic. The new pre-flight is a clear early error. Risk: if anyone is running tiny training jobs on CPU (none currently), they'd be blocked. Mitigation: the pre-flight only blocks `train` / `train-and-test` workflows; `eval` could in principle work on CPU, so allow it through with a slow-path warning. (Decision: implement strict blocking for v1; relax later if anyone asks.)

5. **`$EDITOR` opening unexpectedly during `init`.** Users running `ddrs init` in a script or CI would block on the editor. Mitigation: detect non-TTY stdin (already done in `plan_bootstrap.rs`) and error with:
   > `init: no ddrs.yaml found and stdin is not a TTY. Write ddrs.yaml manually, then re-run `ddrs init`.`

---

## Assumptions

- The existing sandbox smoke test (`crate::sandbox::smoke`) works on the NdArray backend without modification. It's parametric over `I: Backend` already, so this should hold; verify in the implementation plan with a quick test.
- `$EDITOR` bootstrap is acceptable interactive behavior for the target audience (hydrology researchers running locally) — matches `git commit`, `kubectl edit`, `crontab -e`.
- No existing user has scripted around `ddrs init` failing without `ddrs.yaml`. Reasonable: the CLI shipped less than a day ago.
- The `--workflow` CLI flag stays available as an override. Scripts that explicitly pass `--workflow X` continue to work unchanged.

---

## Acceptance criteria

A fresh checkout passes when:

1. `cargo install --path . && ddrs init && ddrs plan && ddrs run --max-mini-batches 1` succeeds end-to-end on a GPU machine.
2. The same sequence succeeds on a CPU-only machine through `ddrs plan`. `ddrs run` errors with a clear "GPU required for this workflow" message.
3. `ddrs.yaml` with `workflow: train-and-test` and no `--workflow` flag works for `plan` and `run`.
4. `ddrs.yaml` with conflicting `mode: training` + `workflow: eval` errors with the cross-validation message.
5. `README.md` "Getting started" section, when followed verbatim by a new user, reaches a passing `ddrs plan` without consulting other docs.
6. All existing tests in `tests/cli_*.rs` pass after updates for new error wording.
