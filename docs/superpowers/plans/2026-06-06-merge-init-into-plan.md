# Merge `ddrs init` into `ddrs plan` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `ddrs init` and `ddrs plan` into a single idempotent `ddrs plan` command (probe → smoke → bootstrap → lock → validate → caches), with auto-relock drift semantics, an interactive bootstrap source prompt, a help-text overhaul, and a plain-language rewrite of the config template comments.

**Architecture:** `init` Phase A (probe/smoke/skeleton) moves to `system::ensure_system_ready()`; `plan()` calls it first, then bootstraps the config if missing, then fingerprints → diffs → relocks (strict callers abort before the relock). `run` keeps calling `plan()` as a library. `src/cli/init.rs` is deleted; a hidden clap stub redirects with exit 2.

**Tech Stack:** Rust, clap 4 (derive), serde/serde_json, tempfile (tests). Spec: `docs/superpowers/specs/2026-06-06-merge-init-into-plan-design.md`.

**Verification baseline:** before Task 1, `cargo test` must be green. Run it once and note any pre-existing failures so they aren't attributed to this work.

---

### Task 1: Move init Phase A into `system::ensure_system_ready`

`run_init` lines 44–80 of `src/cli/init.rs` (probe, GPU-memory warning, workspace skeleton, smoke-cache logic, `system.json` write) become a reusable function in `src/cli/system.rs`. `run_smoke` moves with it. `init.rs` delegates so everything still compiles and behaves identically — its deletion happens in Task 4.

**Files:**
- Modify: `src/cli/system.rs`
- Modify: `src/cli/init.rs:44-80` (Phase A body → delegate) and `src/cli/init.rs:177-202` (`run_smoke` → forwarder)
- Test: `tests/cli_system_ready.rs` (new)

- [x] **Step 1: Write the failing test**

Create `tests/cli_system_ready.rs`:

```rust
//! `system::ensure_system_ready` — the former `init` Phase A as a library
//! call: workspace skeleton + GPU probe + cached smoke test.

use ddrs::cli::manifest::SystemProbe;
use ddrs::cli::system::ensure_system_ready;
use ddrs::cli::workspace::Workspace;

#[test]
fn creates_skeleton_and_system_json() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let r = ensure_system_ready(&ws, false, 0.0, true).unwrap();
    assert!(r.smoke_passed, "skip_smoke=true reports passed");
    assert!(ws.root().join("version").is_file());
    assert!(ws.system_json().is_file());
    assert!(ws.runs_dir().is_dir());
}

#[test]
fn second_call_reuses_smoke_verdict() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    // First call runs the real (CPU on CI) smoke and records a verdict.
    let first = ensure_system_ready(&ws, false, 0.0, false).unwrap();
    assert!(first.smoke_passed);
    assert!(!first.smoke_reused, "first run must execute the smoke");
    let passed_at_1 = SystemProbe::read(&ws.system_json())
        .unwrap().smoke_test.unwrap().passed_at;
    // Second call reuses the cached verdict — passed_at unchanged.
    let second = ensure_system_ready(&ws, false, 0.0, false).unwrap();
    assert!(second.smoke_reused, "second run must reuse the cache");
    let passed_at_2 = SystemProbe::read(&ws.system_json())
        .unwrap().smoke_test.unwrap().passed_at;
    assert_eq!(passed_at_1, passed_at_2);
}

#[test]
fn force_reruns_smoke_without_touching_runs_dir() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    ensure_system_ready(&ws, false, 0.0, false).unwrap();
    // Plant a fake run dir; --force must NOT delete it (old init --force
    // nuked the whole workspace — that behavior is dropped).
    let fake_run = ws.runs_dir().join("2026-01-01T00-00-00Z-train");
    std::fs::create_dir_all(&fake_run).unwrap();
    let r = ensure_system_ready(&ws, true, 0.0, false).unwrap();
    assert!(!r.smoke_reused, "force must re-run the smoke");
    assert!(fake_run.is_dir(), "force must never touch .ddrs/runs/");
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `cargo test --test cli_system_ready 2>&1 | tail -20`
Expected: compile error — `ensure_system_ready` not found in `ddrs::cli::system`.

- [x] **Step 3: Implement `ensure_system_ready` in `src/cli/system.rs`**

Append to `src/cli/system.rs` (and add the new imports at the top of the file):

```rust
// add to the existing imports at the top:
use std::fs;
use std::path::Path;
use crate::cli::workspace::Workspace;
```

```rust
/// Result of [`ensure_system_ready`].
pub struct SystemReadiness {
    pub probe: SystemProbe,
    pub smoke_passed: bool,
    pub smoke_reused: bool,
}

/// Ensure the workspace skeleton exists and the GPU probe + smoke test are
/// recorded in `.ddrs/system.json`. Idempotent: a cached smoke verdict
/// (keyed by [`smoke_key`]) is reused unless `force` is set. This is the
/// former `ddrs init` Phase A, now the first step of `ddrs plan`.
pub fn ensure_system_ready(
    ws: &Workspace,
    force: bool,
    min_free_gpu_gb: f32,
    skip_smoke: bool,
) -> Result<SystemReadiness, CliError> {
    let mut probe = probe()?.unwrap_or_default();
    if probe.free_gpu_gb_at_probe < min_free_gpu_gb && probe.free_gpu_gb_at_probe > 0.0 {
        eprintln!(
            "warning: free GPU memory {:.1} GB is below floor {} GB",
            probe.free_gpu_gb_at_probe, min_free_gpu_gb
        );
    }
    fs::create_dir_all(ws.runs_dir())?;
    fs::write(ws.version_file(), env!("CARGO_PKG_VERSION"))?;

    // Pick backend up-front so the cache key matches the work we'd do.
    let backend = if probe.gpu.is_empty() { "cpu" } else { "cuda" };
    let key = smoke_key(&probe, backend);
    let cached_passing = SystemProbe::read(&ws.system_json())
        .ok()
        .and_then(|p| p.smoke_test)
        .map(|s| s.key == key)
        .unwrap_or(false);
    let (smoke_passed, smoke_reused) = if skip_smoke {
        // Don't claim "reused" if there's no prior record — just "passed".
        (true, cached_passing)
    } else if cached_passing && !force {
        (true, true)
    } else {
        let (ok, _b) = run_smoke(&probe)?;
        (ok, false)
    };
    if smoke_passed && !smoke_reused {
        record_smoke(&mut probe, key, backend);
    } else if smoke_reused {
        // Preserve the prior smoke_test record.
        if let Ok(prior) = SystemProbe::read(&ws.system_json()) {
            probe.smoke_test = prior.smoke_test;
        }
    }
    probe.write_atomic(&ws.system_json())?;
    Ok(SystemReadiness { probe, smoke_passed, smoke_reused })
}

fn run_smoke(probe: &SystemProbe) -> Result<(bool, &'static str), CliError> {
    let inputs = crate::sandbox::load_embedded()
        .or_else(|_| crate::sandbox::load_from_dir(Path::new("fixtures/sandbox")))?;
    if probe.gpu.is_empty() {
        eprintln!("no CUDA detected — running CPU smoke (slower but functionally equivalent)");
        type I = burn::backend::NdArray<f32>;
        let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
        let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
        Ok((r.passed, "cpu"))
    } else {
        type I = burn_cuda::Cuda<f32, i32>;
        let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
        let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
        Ok((r.passed, "cuda"))
    }
}

/// Test-only re-export so integration tests can drive the backend selection.
#[doc(hidden)]
pub fn run_smoke_for_test(probe: &SystemProbe) -> Result<(bool, &'static str), CliError> {
    run_smoke(probe)
}
```

Note: `run_smoke` and `run_smoke_for_test` are copied verbatim from `src/cli/init.rs:177-202` (only the `crate::cli::manifest::SystemProbe` path shortens to the already-imported `SystemProbe`).

- [x] **Step 4: Delegate from `init.rs`**

In `src/cli/init.rs`, replace lines 44–80 (from `// ── Phase A` through `probe.write_atomic(&ws.system_json())?;`) with:

```rust
    // ── Phase A: install-level probes (no config required) ─────────────
    let ready = system::ensure_system_ready(
        &ws, input.force, input.min_free_gpu_gb, input.skip_smoke,
    )?;
    let (smoke_passed, smoke_reused) = (ready.smoke_passed, ready.smoke_reused);
```

Delete the `run_smoke` function body (lines 177–194) and change `run_smoke_for_test` to forward:

```rust
/// Test-only forwarder (the implementation moved to `cli::system`).
#[doc(hidden)]
pub fn run_smoke_for_test(probe: &crate::cli::manifest::SystemProbe)
    -> Result<(bool, &'static str), CliError>
{
    crate::cli::system::run_smoke_for_test(probe)
}
```

Remove now-unused imports from `init.rs` (`SystemProbe` from the `use crate::cli::{...}` block if the compiler flags it — keep `system` itself).

- [x] **Step 5: Run the tests**

Run: `cargo test --test cli_system_ready --test cli_init 2>&1 | tail -10`
Expected: all PASS (cli_init still passes because behavior is unchanged).

- [x] **Step 6: Commit**

```bash
git add src/cli/system.rs src/cli/init.rs tests/cli_system_ready.rs
git commit -m "refactor(cli): extract init Phase A into system::ensure_system_ready"
```

---

### Task 2: Interactive bootstrap source prompt

`pick_source` (`src/cli/plan_bootstrap.rs:58-63`) silently prefers the last successful run's config. Make it prompt when `interactive: true`; keep the last-run preference when `interactive: false` (tests, and the documented historical behavior).

**Files:**
- Modify: `src/cli/plan_bootstrap.rs`
- Test: `tests/cli_plan_bootstrap.rs`

- [x] **Step 1: Write the failing tests**

Append to `tests/cli_plan_bootstrap.rs`:

```rust
use std::io::Cursor;
use std::path::PathBuf;

#[test]
fn choose_source_picks_template_on_2() {
    let mut input = Cursor::new(b"2\n".to_vec());
    let chosen = ddrs::cli::plan_bootstrap::choose_source(
        &mut input,
        PathBuf::from("/x/.ddrs/runs/2026-01-01T00-00-00Z-train/config.yaml"),
    )
    .unwrap();
    assert!(matches!(chosen, BootstrapSource::Template));
}

#[test]
fn choose_source_defaults_to_last_run_on_empty_and_1() {
    for text in [&b"\n"[..], &b"1\n"[..]] {
        let mut input = Cursor::new(text.to_vec());
        let chosen = ddrs::cli::plan_bootstrap::choose_source(
            &mut input,
            PathBuf::from("/x/.ddrs/runs/r/config.yaml"),
        )
        .unwrap();
        assert!(matches!(chosen, BootstrapSource::LastSuccessful(_)));
    }
}

#[test]
fn choose_source_reprompts_on_garbage() {
    let mut input = Cursor::new(b"bananas\n2\n".to_vec());
    let chosen = ddrs::cli::plan_bootstrap::choose_source(
        &mut input,
        PathBuf::from("/x/.ddrs/runs/r/config.yaml"),
    )
    .unwrap();
    assert!(matches!(chosen, BootstrapSource::Template));
}
```

- [x] **Step 2: Run to verify failure**

Run: `cargo test --test cli_plan_bootstrap 2>&1 | tail -10`
Expected: compile error — `choose_source` not found.

- [x] **Step 3: Implement the prompt**

In `src/cli/plan_bootstrap.rs`, change the imports line to include `BufRead` and `Write`:

```rust
use std::io::{BufRead, IsTerminal, Write};
```

Replace `pick_source` (lines 58–63) with:

```rust
fn pick_source(input: &BootstrapInput) -> Result<BootstrapSource, CliError> {
    match latest_successful_run(&input.runs_dir)? {
        Some(p) => {
            if input.interactive {
                let stdin = std::io::stdin();
                let mut lock = stdin.lock();
                choose_source(&mut lock, p)
            } else {
                // Non-interactive callers (tests) keep the historical
                // last-run preference.
                Ok(BootstrapSource::LastSuccessful(p))
            }
        }
        None => Ok(BootstrapSource::Template),
    }
}

/// Interactive source selection, split out with an injectable reader so
/// tests can drive it without a TTY. Empty input defaults to [1] (last run).
pub fn choose_source(
    reader: &mut impl BufRead,
    last_run_config: PathBuf,
) -> Result<BootstrapSource, CliError> {
    let run_id = last_run_config
        .parent()
        .and_then(|d| d.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_string();
    loop {
        eprintln!("No ddrs.yaml found. Start from:");
        eprintln!("  [1] config of last successful run ({run_id})");
        eprintln!("  [2] clean template (config/merit_training.yaml)");
        eprint!("> ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            // EOF mid-prompt — fall back to the historical default.
            return Ok(BootstrapSource::LastSuccessful(last_run_config));
        }
        match line.trim() {
            "" | "1" => return Ok(BootstrapSource::LastSuccessful(last_run_config)),
            "2" => return Ok(BootstrapSource::Template),
            other => eprintln!("unrecognized choice {other:?} — enter 1 or 2"),
        }
    }
}
```

- [x] **Step 4: Run the tests**

Run: `cargo test --test cli_plan_bootstrap 2>&1 | tail -10`
Expected: all 5 tests PASS (the 2 pre-existing ones use `interactive: false` and are unaffected).

- [x] **Step 5: Commit**

```bash
git add src/cli/plan_bootstrap.rs tests/cli_plan_bootstrap.rs
git commit -m "feat(cli): prompt for bootstrap source instead of silently preferring last run"
```

---

### Task 3: Merge the pipeline into `plan()`

`plan()` gains a `PlanInput` struct and absorbs init's remaining responsibilities: system readiness first, config bootstrap when no path is given, and fingerprint → diff → (strict? abort : warn) → relock instead of read-lock-or-die. All callers (`src/bin/ddrs.rs`, `src/cli/run.rs`) and all tests that call `plan()` are updated in this task so the tree compiles and stays green.

**Files:**
- Modify: `src/cli/plan.rs`
- Modify: `src/cli/run.rs:39-66`
- Modify: `src/bin/ddrs.rs:29-32,83-119` (Plan variant + arm; Init untouched until Task 4)
- Modify: `tests/cli_plan.rs`, `tests/cli_runtime_failure.rs`, `tests/cli_run_drift.rs`
- Rewrite: `tests/cli_workspace_uninit.rs` → `tests/cli_plan_fresh.rs`
- Rewrite: `tests/cli_first_run_e2e.rs`

- [x] **Step 1: Write the failing lifecycle tests**

`git mv tests/cli_workspace_uninit.rs tests/cli_plan_fresh.rs`, then replace its content with:

```rust
//! Merged `plan` initializes a fresh workspace itself — no separate `init`.
//! Covers spec §9 tests 1 (fresh dir) and the smoke/lock artifacts.

use ddrs::cli::plan::{plan, PlanInput};
use ddrs::cli::workspace::Workspace;
use std::fs;
use std::path::{Path, PathBuf};

/// Minimal valid config in `dir`: real (1-byte) data files + zarr store
/// skeletons that pass plan's up-front layout validation. Mirrors the
/// fixture in cli_first_run_e2e.rs.
pub fn write_fixture_config(dir: &Path) -> PathBuf {
    let cfg_path = dir.join("ddrs.yaml");
    let mut yaml = String::from(
        "mode: training\nworkflow: train-and-test\ngeodataset: merit\nseed: 1\nnp_seed: 1\ndata_sources:\n",
    );
    for name in ["attributes", "streamflow", "observations", "gages"] {
        let p = dir.join(format!("{name}.bin"));
        fs::write(&p, b"x").unwrap();
        yaml.push_str(&format!("  {name}: {}\n", p.display()));
    }
    let conus = dir.join("conus.zarr");
    fs::create_dir_all(&conus).unwrap();
    fs::write(conus.join("zarr.json"), "{}").unwrap();
    for array in ["order", "length_m", "slope", "indices_0", "indices_1"] {
        fs::create_dir_all(conus.join(array)).unwrap();
        fs::write(conus.join(array).join("zarr.json"), "{}").unwrap();
    }
    let gages = dir.join("gages_adj.zarr");
    fs::create_dir_all(&gages).unwrap();
    fs::write(gages.join("zarr.json"), "{}").unwrap();
    yaml.push_str(&format!("  conus_adjacency: {}\n", conus.display()));
    yaml.push_str(&format!("  gages_adjacency: {}\n", gages.display()));
    yaml.push_str(
        "experiment:\n  batch_size: 1\n  start_time: \"2000-01-01\"\n  end_time: \"2000-01-02\"\n  epochs: 1\n  warmup: 1\n",
    );
    fs::write(&cfg_path, yaml).unwrap();
    cfg_path
}

#[test]
fn plan_initializes_fresh_workspace() {
    let d = tempfile::tempdir().unwrap();
    let cfg = write_fixture_config(d.path());
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let pr = plan(
        PlanInput { config_path: Some(cfg), skip_smoke: true, ..Default::default() },
        &ws,
    )
    .expect("plan must initialize a fresh workspace and succeed");
    assert!(ws.root().join("version").is_file());
    assert!(ws.system_json().is_file());
    assert!(ws.lockfile().is_file(), "first plan writes the lock");
    assert!(pr.drift.is_empty(), "no prior lock → no drift");
}
```

Replace the content of `tests/cli_first_run_e2e.rs` with:

```rust
//! Smokes the documented lifecycle: `plan → plan (idempotent) → drift →
//! relock → strict`. Covers spec §9 tests 2 (idempotency), 3 (drift +
//! relock), and 4 (strict preserves the lock).
//!
//! `ddrs run` is NOT exercised end-to-end here — it needs real CONUS data.

use ddrs::cli::plan::{plan, PlanInput};
use ddrs::cli::workspace::Workspace;
use std::fs;
use std::path::{Path, PathBuf};

// Same fixture as tests/cli_plan_fresh.rs (integration tests are separate
// crates; the ~30 lines are duplicated rather than reshaping tests/common.rs,
// which is routing-focused).
fn write_fixture_config(dir: &Path) -> PathBuf {
    let cfg_path = dir.join("ddrs.yaml");
    let mut yaml = String::from(
        "mode: training\nworkflow: train-and-test\ngeodataset: merit\nseed: 1\nnp_seed: 1\ndata_sources:\n",
    );
    for name in ["attributes", "streamflow", "observations", "gages"] {
        let p = dir.join(format!("{name}.bin"));
        fs::write(&p, b"x").unwrap();
        yaml.push_str(&format!("  {name}: {}\n", p.display()));
    }
    let conus = dir.join("conus.zarr");
    fs::create_dir_all(&conus).unwrap();
    fs::write(conus.join("zarr.json"), "{}").unwrap();
    for array in ["order", "length_m", "slope", "indices_0", "indices_1"] {
        fs::create_dir_all(conus.join(array)).unwrap();
        fs::write(conus.join(array).join("zarr.json"), "{}").unwrap();
    }
    let gages = dir.join("gages_adj.zarr");
    fs::create_dir_all(&gages).unwrap();
    fs::write(gages.join("zarr.json"), "{}").unwrap();
    yaml.push_str(&format!("  conus_adjacency: {}\n", conus.display()));
    yaml.push_str(&format!("  gages_adjacency: {}\n", gages.display()));
    yaml.push_str(
        "experiment:\n  batch_size: 1\n  start_time: \"2000-01-01\"\n  end_time: \"2000-01-02\"\n  epochs: 1\n  warmup: 1\n",
    );
    fs::write(&cfg_path, yaml).unwrap();
    cfg_path
}

fn plan_input(cfg: &Path) -> PlanInput {
    PlanInput {
        config_path: Some(cfg.to_path_buf()),
        skip_smoke: true,
        ..Default::default()
    }
}

#[test]
fn plan_lifecycle_idempotent_drift_relock_strict() {
    let d = tempfile::tempdir().unwrap();
    let cfg = write_fixture_config(d.path());
    let ws = Workspace::with_root(d.path().join(".ddrs"));

    // 1. Fresh plan: initializes + locks.
    let pr = plan(plan_input(&cfg), &ws).expect("fresh plan succeeds");
    assert_eq!(pr.workflow, ddrs::cli::Workflow::TrainAndTest);
    assert!(pr.drift.is_empty());
    let lock_1 = fs::read_to_string(ws.lockfile()).unwrap();

    // 2. Idempotency: second plan → no drift, lock byte-identical.
    let pr2 = plan(plan_input(&cfg), &ws).expect("second plan succeeds");
    assert!(pr2.drift.is_empty());
    let lock_2 = fs::read_to_string(ws.lockfile()).unwrap();
    assert_eq!(lock_1, lock_2, "unchanged sources must not rewrite the lock");

    // 3. Drift + auto-relock: mutate a source, plan reports + relocks.
    fs::write(d.path().join("gages.bin"), b"yy").unwrap();
    let pr3 = plan(plan_input(&cfg), &ws).expect("drifted plan still succeeds");
    assert_eq!(pr3.drift, vec!["gages".to_string()]);
    let lock_3 = fs::read_to_string(ws.lockfile()).unwrap();
    assert_ne!(lock_2, lock_3, "drift must refresh the lock");

    // 4. Post-relock: drift is gone.
    let pr4 = plan(plan_input(&cfg), &ws).expect("post-relock plan succeeds");
    assert!(pr4.drift.is_empty());

    // 5. Strict aborts BEFORE relocking (evidence preserved).
    fs::write(d.path().join("gages.bin"), b"zzz").unwrap();
    let err = plan(
        PlanInput { strict: true, ..plan_input(&cfg) },
        &ws,
    )
    .unwrap_err();
    assert!(
        matches!(err, ddrs::cli::CliError::LockDrift { .. }),
        "expected LockDrift, got: {err:?}"
    );
    let lock_5 = fs::read_to_string(ws.lockfile()).unwrap();
    assert_eq!(lock_3, lock_5, "strict abort must leave the lock untouched");
}
```

- [x] **Step 2: Run to verify failure**

Run: `cargo test --test cli_plan_fresh --test cli_first_run_e2e 2>&1 | tail -10`
Expected: compile error — `PlanInput` not found.

- [x] **Step 3: Restructure `plan()`**

In `src/cli/plan.rs`:

3a. Add `PlanInput` above `pub fn plan` and replace the function signature + steps 4–5 (the lockfile read at lines 124–129 and the drift line 168). The full new top of the function:

```rust
pub struct PlanInput {
    /// Explicit config path. `None` → bootstrap `./ddrs.yaml` interactively.
    pub config_path: Option<PathBuf>,
    pub workflow: Option<Workflow>,
    /// Re-run the GPU smoke test even if a cached verdict exists.
    pub force: bool,
    pub min_free_gpu_gb: f32,
    /// Skip the smoke test (CI/tests).
    pub skip_smoke: bool,
    /// Abort with `LockDrift` on drift instead of warning + relocking.
    /// `run --strict` passes true.
    pub strict: bool,
}

impl Default for PlanInput {
    fn default() -> Self {
        Self {
            config_path: None,
            workflow: None,
            force: false,
            min_free_gpu_gb: 8.0,
            skip_smoke: false,
            strict: false,
        }
    }
}

pub fn plan(input: PlanInput, workspace: &Workspace) -> Result<PlanResult, CliError> {
    // Step 0: workspace skeleton + GPU probe + cached smoke test (the
    // former `init` Phase A). Idempotent and cheap after the first call.
    let ready = crate::cli::system::ensure_system_ready(
        workspace,
        input.force,
        input.min_free_gpu_gb,
        input.skip_smoke,
    )?;
    if !ready.smoke_passed {
        return Err(CliError::Runtime(
            "smoke test failed: the routing core does not run on this system. \
             See .ddrs/system.json for the probe record."
                .into(),
        ));
    }

    // Step 1: locate or bootstrap ddrs.yaml (interactive, TTY only).
    let config_path = match input.config_path {
        Some(p) => p,
        None => bootstrap_config(workspace)?,
    };
    let config_path = config_path.as_path();
    let workflow_override = input.workflow;
```

The existing steps 1–3 (config preview at line 91, workflow resolution at line 98, mode re-parse at line 110) follow unchanged — they already use the local names `config_path` / `workflow_override`.

3b. Replace step 4 (the lockfile read-or-die, lines 124–129) with:

```rust
    // Step 4: read the prior lock if one exists. First-ever plan: none.
    let lock_path = workspace.lockfile();
    let prior_lock = if lock_path.is_file() {
        Some(Lockfile::read(&lock_path)?)
    } else {
        None
    };
```

3c. In the fingerprint loop (line 154), change the lock lookup to go through `prior_lock`:

```rust
        let live = match prior_lock.as_ref().and_then(|l| l.sources.get(&key)) {
```

3d. Replace the drift line (`let drift = diff_against_live(&lock, &sources);`) with the drift + relock policy:

```rust
    let drift = prior_lock
        .as_ref()
        .map(|l| diff_against_live(l, &sources))
        .unwrap_or_default();

    // Drift policy + auto-relock. Strict callers (run --strict) abort
    // BEFORE the lock is refreshed so the drift evidence survives.
    if !drift.is_empty() {
        if input.strict {
            return Err(CliError::LockDrift { fields: drift });
        }
        eprintln!("warning: data source drift since last plan: {drift:?} — relocking");
    }
    // Rewrite only when something actually changed (mtime/size/fp), so an
    // unchanged re-plan leaves the lock byte-identical.
    let needs_write = prior_lock
        .as_ref()
        .map(|l| l.sources != sources)
        .unwrap_or(true);
    if needs_write {
        Lockfile {
            ddrs_version: env!("CARGO_PKG_VERSION").into(),
            created_at: chrono::Utc::now()
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            sources: sources.clone(),
        }
        .write_atomic(&lock_path)?;
    }
```

3e. Add `bootstrap_config` at the bottom of the file (moved from `init.rs:83-116`, message updated):

```rust
/// Bootstrap `./ddrs.yaml` via $EDITOR when no config was found. TTY only —
/// non-interactive callers get an actionable ConfigInvalid.
fn bootstrap_config(workspace: &Workspace) -> Result<PathBuf, CliError> {
    let target = std::env::current_dir()
        .map_err(CliError::from)?
        .join("ddrs.yaml");
    let bundled = PathBuf::from("config/merit_training.yaml");
    crate::cli::plan_bootstrap::bootstrap(crate::cli::plan_bootstrap::BootstrapInput {
        target: target.clone(),
        runs_dir: workspace.runs_dir(),
        bundled_template: bundled,
        editor_cmd: None,
        interactive: true,
    })
    .map_err(|e| {
        let msg = format!("{e}");
        if msg.contains("not a TTY") || msg.contains("run interactively") {
            CliError::ConfigInvalid {
                path: target.clone(),
                source: "no ddrs.yaml found and stdin is not a TTY. \
                         Pass --config or write ddrs.yaml manually, then \
                         re-run `ddrs plan`."
                    .into(),
            }
        } else {
            e
        }
    })?;
    Ok(target)
}
```

3f. The `import` line for `Lockfile` stays (`diff_against_live` too). `chrono` is already a crate dependency (used by `init.rs`/`run.rs`); the fully-qualified `chrono::Utc::now()` needs no new import.

- [x] **Step 4: Update `run()`**

In `src/cli/run.rs`, replace step 1 (line 41) and delete the drift-policy block (lines 60–66):

```rust
    // 1. Plan as a library call (reused — not re-parsed in run). Handles
    //    workspace init, smoke caching, drift policy (strict aborts before
    //    the relock), and adjacency/baseline caches.
    let pr: PlanResult = plan(
        crate::cli::plan::PlanInput {
            config_path: Some(input.config_path.clone()),
            workflow: input.workflow,
            strict: input.strict,
            ..crate::cli::plan::PlanInput::default()
        },
        &input.workspace,
    )?;
```

(The old step 2 `if !pr.drift.is_empty() { ... }` block is now inside `plan` — delete it. `pr.drift` is still used by the manifest's `source_lock` further down; that stays.)

- [x] **Step 5: Update the binary's Plan variant and arm**

In `src/bin/ddrs.rs`, the `Plan` variant gains init's flags:

```rust
    Plan {
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        #[arg(long)] json: bool,
        #[arg(long)] force: bool,
        #[arg(long, default_value_t = 8.0)] min_free_gpu_gb: f32,
    },
```

And the `Cmd::Plan` arm drops the config-required error (plan bootstraps when `None`):

```rust
        Cmd::Plan { workflow, json, force, min_free_gpu_gb } => {
            let pr = ddrs::cli::plan::plan(
                ddrs::cli::plan::PlanInput {
                    config_path: cfg_path,
                    workflow,
                    force,
                    min_free_gpu_gb,
                    skip_smoke: false,
                    strict: false,
                },
                &ws,
            )?;
            if json {
```

(the printing body from line 90 down is unchanged).

- [x] **Step 6: Update remaining test callers**

`tests/cli_plan.rs` — all three `plan(...)` call sites change to the struct form, and the pre-seeded empty lockfiles are no longer needed (plan writes its own):

In `workflow_resolved_from_yaml_key` (lines 33–42), delete the `Lockfile` creation block and replace the call:

```rust
    let ws_root = tmp.path().join(".ddrs");
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let res = ddrs::cli::plan::plan(
        ddrs::cli::plan::PlanInput {
            config_path: Some(cfg_path.clone()),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    );
```

In `run_inherits_yaml_workflow_resolution` (lines 71–78), likewise delete the lockfile-seeding block (keep the `RunInput` call as-is — `run` now self-initializes).

In `plan_succeeds_on_repo_config` (line 11):

```rust
    let _ = plan(
        ddrs::cli::plan::PlanInput {
            config_path: Some(cfg.to_path_buf()),
            workflow: Some(Workflow::Train),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    );
```

In `no_workflow_anywhere_gives_actionable_error` (line 110):

```rust
    let err = ddrs::cli::plan::plan(
        ddrs::cli::plan::PlanInput {
            config_path: Some(cfg_path.clone()),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    )
    .unwrap_err();
```

Add the non-TTY bootstrap test to `tests/cli_plan.rs` (replaces the deleted `cli_init.rs` coverage; chdir is process-global so it takes a lock):

```rust
use std::sync::Mutex;
static CHDIR_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn plan_errors_clearly_when_no_yaml_and_no_tty() {
    let _g = CHDIR_LOCK.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(tmp.path().join(".ddrs"));
    let original = std::env::current_dir().unwrap();
    std::env::set_current_dir(tmp.path()).unwrap();
    let res = ddrs::cli::plan::plan(
        ddrs::cli::plan::PlanInput { skip_smoke: true, ..Default::default() },
        &ws,
    );
    std::env::set_current_dir(&original).unwrap();
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("no ddrs.yaml found") && msg.contains("not a TTY"),
        "expected non-interactive bootstrap error, got: {msg}"
    );
}
```

`tests/cli_runtime_failure.rs` — replace the `run_init` block (lines 1, 24–37):

```rust
use ddrs::cli::plan::{plan, PlanInput};
```

```rust
    // `run` self-initializes via its internal plan() call, but probe data
    // reachability first so the test can skip on hosts without merit data.
    let plan_result = plan(
        PlanInput {
            config_path: Some(cfg.clone()),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    );
    if let Err(e) = plan_result {
        eprintln!("skipping: data sources not reachable ({e})");
        return;
    }
```

`tests/cli_run_drift.rs` (ignored test) — replace the `run_init` import (line 8) and call (lines 27–34):

```rust
use ddrs::cli::plan::{plan, PlanInput};
```

```rust
    plan(
        PlanInput {
            config_path: Some(cfg.clone()),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    )
    .expect("plan must succeed when merit data is reachable");
```

- [x] **Step 7: Run the full CLI test suite**

Run: `cargo test --test cli_plan_fresh --test cli_first_run_e2e --test cli_plan --test cli_runtime_failure --test cli_system_ready --test cli_plan_bootstrap --test cli_init 2>&1 | tail -15`
Expected: all PASS. (`cli_init` still passes — `init.rs` exists until Task 4 and its Phase B path is untouched.)

- [x] **Step 8: Run the whole suite to catch stragglers**

Run: `cargo test 2>&1 | tail -15`
Expected: green (modulo any failures noted in the pre-task baseline). If another test calls `cli::plan::plan` with the old signature, the compiler will name it — update it to the `PlanInput` struct form exactly as in Step 6.

- [x] **Step 9: Commit**

```bash
git add -A src/cli/plan.rs src/cli/run.rs src/bin/ddrs.rs tests/
git commit -m "feat(cli): merge init pipeline into plan with auto-relock drift policy"
```

---

### Task 4: Delete `init`, add the redirect stub

**Files:**
- Delete: `src/cli/init.rs`, `tests/cli_init.rs`
- Modify: `src/cli/mod.rs:6`, `src/bin/ddrs.rs` (Init variant + arm, run-arm message), `src/error.rs:20`, `src/sandbox.rs:5`
- Create: `tests/cli_init_stub.rs`, `tests/cli_smoke.rs`

- [x] **Step 1: Write the failing stub test**

Create `tests/cli_init_stub.rs`:

```rust
//! `ddrs init` is a hidden stub (removed in 0.4): prints a redirect and
//! exits 2 so muscle-memory scripts fail loudly.

use std::process::Command;

#[test]
fn init_stub_redirects_to_plan_with_exit_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_ddrs"))
        .arg("init")
        .output()
        .expect("ddrs binary should run");
    assert_eq!(out.status.code(), Some(2), "stub must exit 2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("merged into ddrs plan"),
        "stub must redirect to plan, got: {stderr}"
    );
}

#[test]
fn init_does_not_appear_in_help() {
    let out = Command::new(env!("CARGO_BIN_EXE_ddrs"))
        .arg("--help")
        .output()
        .expect("ddrs binary should run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("init"), "init must be hidden from --help");
}
```

Create `tests/cli_smoke.rs` (preserves the CPU-smoke coverage from the doomed `cli_init.rs:57-65`):

```rust
use ddrs::cli::manifest::SystemProbe;

#[test]
fn run_smoke_returns_cpu_when_no_cuda() {
    let mut probe = SystemProbe::default();
    probe.gpu = String::new();
    let (passed, backend) = ddrs::cli::system::run_smoke_for_test(&probe).unwrap();
    assert!(passed, "CPU smoke must pass on the bundled sandbox fixture");
    assert_eq!(backend, "cpu");
}
```

- [x] **Step 2: Run to verify failure**

Run: `cargo test --test cli_init_stub --test cli_smoke 2>&1 | tail -10`
Expected: `init_stub_redirects_to_plan_with_exit_2` FAILS (real init runs and errors differently or succeeds); `cli_smoke` PASSES already (function exists since Task 1). The `init_does_not_appear_in_help` test FAILS (init is visible).

- [x] **Step 3: Replace init with the stub**

```bash
git rm src/cli/init.rs tests/cli_init.rs
```

`src/cli/mod.rs`: delete the `pub mod init;` line.

`src/bin/ddrs.rs` — replace the `Init` variant (lines 25–28) with the hidden stub (old flags still parse so stale scripts get the redirect, not a clap error):

```rust
    /// Deprecated: merged into `ddrs plan`. Stub removed in 0.4.
    #[command(hide = true)]
    Init {
        #[arg(long, hide = true)] force: bool,
        #[arg(long, default_value_t = 8.0, hide = true)] min_free_gpu_gb: f32,
    },
```

Replace the `Cmd::Init` arm (lines 74–82) with:

```rust
        Cmd::Init { .. } => {
            eprintln!("ddrs init has been merged into ddrs plan — run `ddrs plan`");
            ExitCode::ConfigInvalid.exit();
        }
```

Update the run-arm message (line 124): `"no ddrs.yaml found in current directory. Run \`ddrs plan\` first."`

`src/error.rs:20` — the `WorkspaceNotInitialized` message (now only reachable from `status`/`show`/`gc`):

```rust
    #[error("workspace not initialized at {path}; run `ddrs plan` first")]
    WorkspaceNotInitialized { path: PathBuf },
```

`src/sandbox.rs:5` — update the doc comment: `//!   - \`cli::system\` smoke test (does routing work on this machine?)`

- [x] **Step 4: Run the tests**

Run: `cargo test --test cli_init_stub --test cli_smoke 2>&1 | tail -10` then `cargo test 2>&1 | tail -10`
Expected: all PASS; nothing references `cli::init` anymore (`grep -rn "cli::init\|run_init" src/ tests/` returns nothing).

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cli)!: remove ddrs init (merged into plan); hidden stub exits 2"
```

---

### Task 5: Help-text overhaul

**Files:**
- Modify: `src/bin/ddrs.rs` (top-level command attrs + every subcommand/flag doc comment)
- Modify: `src/config.rs:34-38` (Workflow variant docs)

- [x] **Step 1: Top-level help**

Replace the `#[command(...)]` attribute on `struct Cli` (`src/bin/ddrs.rs:13`) with:

```rust
#[command(
    name = "ddrs",
    about = "Differentiable Distributed Routing — train and evaluate a \
             Muskingum-Cunge routing model with a KAN parameter head",
    after_help = "\
LIFECYCLE:
    ddrs plan    Prepare + preview: probes the GPU (first run only), bootstraps
                 ddrs.yaml if missing, locks data sources, validates the config,
                 and builds adjacency/baseline caches. Idempotent — run anytime.
    ddrs run     Execute a workflow. Re-plans internally, then trains/evaluates.
    ddrs show    Inspect a past run's manifest.
    ddrs status  Workspace summary + disk usage.
    ddrs gc      Prune old runs from .ddrs/runs/.

WORKFLOWS (--workflow flag, or the `workflow:` key in ddrs.yaml):
    train           Train the KAN head            (needs `mode: training`)
    eval            Evaluate a checkpoint         (needs `mode: testing`)
    train-and-test  Train, evaluate, and compare vs. the summed-Q' baseline

STARTING FRESH:
    rm ddrs.yaml && ddrs plan — you'll be asked whether to start from your
    last successful run's config or the clean bundled template."
)]
```

And document the two global flags:

```rust
    /// Path to the experiment config (default: discover ddrs.yaml upward
    /// from the current directory, stopping at the first .git ancestor).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Workspace directory (default: .ddrs/ beside the config).
    #[arg(long, global = true)]
    workspace: Option<PathBuf>,
```

- [x] **Step 2: Subcommand + flag doc comments**

Apply doc comments to every variant and flag in `enum Cmd` (the Init stub keeps only its Task 4 comment):

```rust
    /// Prepare the workspace and preview the workflow: GPU probe + cached
    /// smoke test, ddrs.yaml bootstrap, data-source locking, config
    /// validation, adjacency/baseline cache builds. Idempotent.
    Plan {
        /// Override the `workflow:` key in ddrs.yaml for this invocation.
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        /// Print the plan result as JSON instead of human-readable text.
        #[arg(long)] json: bool,
        /// Re-run the GPU smoke test even if a cached verdict exists.
        #[arg(long)] force: bool,
        /// Warn when free GPU memory at probe time is below this many GB.
        #[arg(long, default_value_t = 8.0)] min_free_gpu_gb: f32,
    },
    /// Execute a workflow: re-plans, then trains and/or evaluates, writing
    /// checkpoints + manifest to .ddrs/runs/<id>/.
    Run {
        /// Override the `workflow:` key in ddrs.yaml for this invocation.
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        /// After a successful run, dump per-COMID KAN parameters to
        /// plot/kan_parameters.nc (NetCDF).
        #[arg(long)] plot: bool,
        /// Exit with code 4 if data sources changed since the last plan,
        /// instead of warning and relocking.
        #[arg(long)] strict: bool,
        /// Stop each training epoch after this many mini-batches (debugging).
        #[arg(long)] max_mini_batches: Option<usize>,
        /// Replay a captured mini-batch order from JSON (matched-batch parity
        /// experiment). When set, overrides the default per-epoch shuffle.
        /// JSON schema: array of {"epoch": int, "mb": int, "staids": [str, ...]}.
        #[arg(long, value_name = "PATH")] batch_order_from: Option<PathBuf>,
        /// Print the run result as JSON instead of human-readable text.
        #[arg(long)] json: bool,
    },
    /// Inspect a past run's manifest.
    Show {
        /// Run ID under .ddrs/runs/ (list them with `ddrs status`).
        run_id: String,
        /// Print the manifest as JSON.
        #[arg(long)] json: bool,
    },
    /// Summarize the workspace: runs, lockfile state, disk usage.
    Status {
        /// Print the summary as JSON.
        #[arg(long)] json: bool,
    },
    /// Delete old run directories from .ddrs/runs/.
    Gc {
        /// Keep the N most recent runs.
        #[arg(long)] keep: Option<usize>,
        /// Never delete successful runs.
        #[arg(long)] keep_successful: bool,
        /// Only delete runs older than this duration (e.g. "30d", "12h").
        #[arg(long)] older_than: Option<String>,
        /// List what would be deleted without deleting anything.
        #[arg(long)] dry_run: bool,
    },
```

- [x] **Step 3: Workflow value help**

In `src/config.rs:34-38`, add variant doc comments (clap's derive turns these into `--workflow` value help):

```rust
pub enum Workflow {
    /// Train the KAN head (requires `mode: training`).
    Train,
    /// Evaluate a trained checkpoint over the testing window (requires `mode: testing`).
    Eval,
    /// Train, then evaluate, then compare against the summed-Q' baseline.
    TrainAndTest,
}
```

- [x] **Step 4: Verify the rendered help**

Run: `cargo run --bin ddrs -- --help` and `cargo run --bin ddrs -- plan --help`
Expected: top-level shows the LIFECYCLE / WORKFLOWS / STARTING FRESH sections and no `init` row; `plan --help` documents all four flags. Run `cargo run --bin ddrs -- run --help` and confirm `--workflow` lists the three values with their descriptions.

- [x] **Step 5: Run the suite and commit**

Run: `cargo test --test cli_init_stub 2>&1 | tail -5` (the `init_does_not_appear_in_help` guard still passes), then:

```bash
git add src/bin/ddrs.rs src/config.rs
git commit -m "docs(cli): lifecycle help text; document every subcommand and flag"
```

---

### Task 6: Rewrite `config/merit_training.yaml` comments

Comments only — the parsed YAML must be value-identical. The new comments explain each field in plain language; benchmarking jargon (SP-10, V7a) and internal test references go.

**Files:**
- Modify: `config/merit_training.yaml` (full rewrite below)

- [x] **Step 1: Replace the file content**

```yaml
# ddrs experiment config. `ddrs plan` copies this template to ./ddrs.yaml on
# first run (when you pick "clean template" at the bootstrap prompt).
#
# Hyperparameter VALUES mirror DDR's reference config
# (~/projects/ddr/config/merit_training_config.yaml) — keep them in sync when
# comparing against the Python implementation.

# Run mode. `training` activates the experiment: section; `testing` applies
# the testing: overlay at the bottom instead. Must agree with `workflow:`
# (training ↔ train / train-and-test, testing ↔ eval) — contradictions are
# rejected at load time.
mode: training
# Default workflow for `ddrs plan` / `ddrs run`; override per-invocation
# with --workflow.
workflow: train-and-test
# Dataset family. `merit` (MERIT-Hydro river network) is the only supported
# value today.
geodataset: merit
# CUDA device ordinal. On multi-GPU hosts, pick a non-display GPU.
device: 0
# RNG seeds: `seed` drives model initialization, `np_seed` drives the
# per-epoch gauge shuffle.
seed: 42
np_seed: 42

# Where ddrs reads its inputs. Everything is read in place — no export step.
#
# Adjacency strategy: with `geospatial_fabric` set, the first `ddrs plan`
# builds the CONUS + per-gauge adjacency zarr stores from the fabric's .dbf
# attribute table into .ddrs/adjacency/<key>/ (~10 s, content-addressed,
# reused afterwards). To use pre-built stores instead, drop
# `geospatial_fabric` and set both:
#   conus_adjacency: /path/to/merit_conus_adjacency.zarr
#   gages_adjacency: /path/to/merit_gages_conus_adjacency.zarr
data_sources:
  # Per-reach catchment attributes (NetCDF): soils, climate, topography.
  # The kan_head.input_var_names below are columns of this file.
  attributes: /home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc
  # MERIT-Hydro flowlines shapefile. Only the sibling .dbf attribute table
  # is read — geometry is never opened.
  geospatial_fabric: /projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp
  # Daily lateral inflow (Q') per reach from the dHBV2 retrospective
  # (icechunk store). Interpolated to hourly internally.
  streamflow: /mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic
  # USGS daily observed discharge (icechunk store) — the training targets.
  observations: /mnt/ssd1/data/icechunk/usgs_daily_observations
  # Gauge table (CSV): station IDs and the MERIT COMID each gauge sits on.
  gages: /home/tbindas/projects/ddr/references/gage_info/gages_3000.csv

# Training-loop settings (active when mode: training).
experiment:
  # Gauges per mini-batch.
  batch_size: 64
  # Training window (water years 1982–1995).
  start_time: 1981/10/01
  end_time: 1995/09/30
  epochs: 5
  # Sequence length in DAYS sampled per mini-batch.
  rho: 90
  # Re-shuffle gauge order each epoch (seeded by np_seed).
  shuffle: true
  # Days excluded from the loss at the start of each sequence, so routing
  # state can spin up from the cold-start estimate.
  warmup: 5
  # Learning-rate schedule: epoch → LR, applying from that epoch onward.
  learning_rate:
    1: 0.001
    3: 0.0005
  # Global gradient-norm clip.
  grad_clip_max_norm: 1.0

# The KAN head mapping catchment attributes → routing parameters:
# Linear(F,H) → KanLayer(H,H) × num_hidden_layers → Linear(H,P) → Sigmoid.
kan_head:
  hidden_size: 21
  num_hidden_layers: 2
  # B-spline grid intervals per KAN edge (pykan's `num`).
  grid: 50
  # B-spline order (pykan's `k`). DDR overrides pykan's default of 3 to 2
  # in production; keep 2 for parity.
  k: 2
  # Catchment attributes fed to the head (column names in `attributes`).
  input_var_names:
    - SoilGrids1km_clay
    - aridity
    - meanelevation
    - meanP
    - NDVI
    - meanslope
    - log10_uparea
    - SoilGrids1km_sand
    - ETPOT_Hargr
    - Porosity
  # Routing parameters the head predicts (denormalized through
  # params.parameter_ranges below).
  learnable_parameters:
    - n
    - q_spatial
    - p_spatial

# Routing-engine settings.
params:
  # Physical [min, max] each sigmoid-normalized KAN output maps onto.
  parameter_ranges:
    # Manning's roughness coefficient.
    n: [0.015, 0.25]
    # Leopold & Maddock width–depth exponent (top_width = p · depth^q).
    q_spatial: [0.0, 1.0]
    # Leopold & Maddock width coefficient (same power law).
    p_spatial: [1.0, 200.0]
  # Floors applied for numerical stability (SI units).
  attribute_minimums:
    discharge: 1.0e-4    # m³/s
    slope: 1.0e-3        # m/m
    velocity: 0.01       # m/s
    depth: 0.01          # m
    bottom_width: 0.01   # m
  # Fixed values for parameters NOT listed in learnable_parameters.
  defaults:
    p_spatial: 21.0
  # Parameters denormalized in log space (their range spans decades).
  log_space_parameters:
    - p_spatial
  # Sparse triangular solve on the GPU via cuSPARSE (requires the CUDA
  # backend; remove or set to cpu to fall back to the CPU solver).
  sparse_solver: cuda
  # Capture the routing forward pass as a CUDA graph and replay it each
  # timestep — faster, CUDA backend only.
  use_cuda_graphs: true

# Testing-mode overlay: when mode: testing (workflow: eval), these keys
# REPLACE the matching experiment: keys; absent keys inherit.
#
# CAUTION — batch_size changes meaning between modes:
#   experiment.batch_size = GAUGES per mini-batch (training)
#   testing.batch_size    = DAYS per evaluation chunk
testing:
  # Evaluation window (water years 1996–2010).
  start_time: 1995/10/01
  end_time: 2010/09/30
  # DAYS per evaluation chunk, not gauges.
  batch_size: 15
  # Sequence sampling disabled in test mode.
  rho: null
```

- [x] **Step 2: Verify values are byte-for-byte unchanged after parsing**

Run:

```bash
uv run --with pyyaml python3 -c "
import subprocess, yaml
old = yaml.safe_load(subprocess.check_output(['git', 'show', 'HEAD:config/merit_training.yaml']))
new = yaml.safe_load(open('config/merit_training.yaml'))
assert old == new, f'values changed!\nonly-in-old: {set(map(str,old))-set(map(str,new))}\nonly-in-new: {set(map(str,new))-set(map(str,old))}'
print('parsed values identical')
"
```

Expected: `parsed values identical`. If it fails, diff the two parsed dicts key-by-key and fix the template — values must not change.

- [x] **Step 3: Confirm the Rust parser still accepts it**

Run: `cargo test --test cli_plan 2>&1 | tail -5` and `cargo run --bin ddrs -- --config config/merit_training.yaml plan --workflow train 2>&1 | head -5`
Expected: tests pass; plan starts (it may fail later on unreachable data sources on this host — `error: data source unreachable` is acceptable here, a YAML parse error is not).

- [x] **Step 4: Commit**

```bash
git add config/merit_training.yaml
git commit -m "docs(config): plain-language template comments; values unchanged"
```

---

### Task 7: Update README, CLAUDE.md, old spec pointer; final verification

**Files:**
- Modify: `README.md` (First-time setup, what-lives-where table, workflow-agreement note)
- Modify: `CLAUDE.md` (CLI section, workspace table, bootstrap gotcha)
- Modify: `docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md` (supersession note)

- [x] **Step 1: README**

In `README.md` "First-time setup", replace the three-command block and the paragraph below it with:

````markdown
```bash
ddrs plan      # probes GPU + smoke test (first run), opens $EDITOR on
               # ddrs.yaml if missing, locks data sources, validates,
               # builds adjacency/baseline caches, prints the plan
ddrs run       # executes the workflow, writes manifest + outputs
```

The first `ddrs plan` runs a 5-reach RAPID sandbox parity check on CUDA when
available and falls back to CPU otherwise — so the install path works on
laptops and CI. The verdict is cached; later plans are fast. When no
`ddrs.yaml` exists, `plan` asks whether to start from your last successful
run's config or the clean bundled template (`config/merit_training.yaml`).
To start fresh at any time: `rm ddrs.yaml && ddrs plan`.
````

In the "What lives where" table, change the "Written by" cells: `ddrs.yaml` → `` `ddrs plan` (via `$EDITOR`) ``; `.ddrs/system.json` → `` `ddrs plan` ``; `.ddrs/sources.lock` → `` `ddrs plan` ``.

In "Override workflow on the command line", change the last sentence of the agreement note from "`ddrs init` will reject contradictions at load time." to "`ddrs plan` will reject contradictions at load time."

- [x] **Step 2: CLAUDE.md**

In `CLAUDE.md` § "### `ddrs` CLI (preferred entrypoint)", replace the first-time-flow block:

````markdown
First-time flow `plan → run`:

```bash
ddrs plan                                      # GPU probe + smoke (cached) + bootstraps
                                               #   ./ddrs.yaml (opens $EDITOR) + locks
                                               #   data_sources + validates + computes
                                               #   summed-Q' baseline (cached)
ddrs run --workflow train-and-test             # full sweep: train, eval, write manifest
````

(keep the remaining `ddrs run/show/status/gc` lines as they are; delete the `ddrs init` line).

Replace the "**The bootstrap-from-last-run gotcha**" paragraph with:

```markdown
**Bootstrap source prompt** (`src/cli/plan_bootstrap.rs`): when `ddrs plan`
materializes a missing `ddrs.yaml` and a previous successful run exists, it
asks whether to start from that run's `config.yaml` snapshot or the bundled
`config/merit_training.yaml` template. Non-TTY callers must pass `--config`.
Lock semantics: `plan` reports drift against `.ddrs/sources.lock`, then
refreshes the lock ("sources as of my last plan"); `ddrs run --strict` aborts
with exit 4 *before* relocking, preserving the evidence.
```

In the workspace-layout table, change the "Written by" cells for `ddrs.yaml`, `.ddrs/system.json`, `.ddrs/sources.lock` from `ddrs init` to `ddrs plan`. In the same section, fix any remaining `ddrs init` references (`grep -n "ddrs init" CLAUDE.md` and update each to `ddrs plan`).

- [x] **Step 3: Mark the old spec superseded**

In `docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`, insert directly under the title:

```markdown
> **Partially superseded (2026-06-06):** `ddrs init` and `ddrs plan` were
> merged into a single `ddrs plan` command — see
> `2026-06-06-merge-init-into-plan-design.md`. The run/show/status/gc,
> manifest, and exit-code sections below still stand.
```

- [x] **Step 4: Sweep for stale references**

Run: `grep -rn "ddrs init" README.md CLAUDE.md src/ tests/ config/ | grep -v "merged into"`
Expected: no output (docs under `docs/superpowers/` keep their historical references).

- [x] **Step 5: Full verification**

Run: `cargo test 2>&1 | tail -15`
Expected: green (modulo the pre-task baseline).

Insurance (spec §4 — routing untouched, but cheap to confirm):

Run: `cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3`
Expected: `ABSOLUTE MATCH`.

- [x] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md
git commit -m "docs: plan→run lifecycle in README/CLAUDE.md; mark old CLI spec superseded"
```
