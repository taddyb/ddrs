# ddrs CLI UX cleanup + README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the first-run init/plan/init/plan ping-pong into `ddrs init → ddrs plan → ddrs run`, with `workflow:` as a top-level YAML key and a CPU-fallback smoke test.

**Architecture:** Move `Workflow` from `cli/types.rs` into `config.rs` so `Config` can deserialize it from YAML; resolve workflow as `cli_flag.or(config.workflow)`. Consolidate `init` to also bootstrap `ddrs.yaml` (Phase D) and lock sources (Phase E). Make the smoke test backend-aware (CUDA when probed, NdArray CPU otherwise). Move the GPU-required check from `init` to a `run` pre-flight.

**Tech Stack:** Rust, BURN 0.21 (CUDA + NdArray backends), serde_yaml, clap derive.

**Spec:** `docs/superpowers/specs/2026-06-01-ddrs-cli-ux-and-readme-design.md`

---

## File map

**Modified:**

| Path | Why |
|---|---|
| `src/config.rs` | Host `Workflow` enum, add `workflow: Option<Workflow>` to `Config`, cross-validate against `mode` |
| `src/cli/types.rs` | Re-export `Workflow` from config (preserves all existing imports) |
| `src/cli/plan.rs` | Resolve workflow via `cli.or(config.workflow)`, rewrite error message |
| `src/cli/run.rs` | Same workflow resolution + GPU pre-flight for train/train-and-test |
| `src/cli/init.rs` | Backend-aware smoke, drop GPU-required error, add Phase D (yaml bootstrap) before Phase E (lock) |
| `src/cli/system.rs` | Extend `smoke_key` with `backend` term, add `record_smoke_with_backend` |
| `src/cli/manifest.rs` | Add `backend: Option<String>` to `SmokeTestRecord` (Option = backward-compat with old records) |
| `src/bin/ddrs.rs` | Remove `plan`'s bootstrap branch (it's now in init); pass `--skip-smoke` flag through |
| `config/merit_training.yaml` | Add `workflow: train-and-test` near the top |
| `README.md` | New "Getting started" section |

**Tests touched:**

| Path | Why |
|---|---|
| `tests/cli_plan.rs` | Update for new no-workflow error wording |
| `tests/cli_plan_bootstrap.rs` | Bootstrap is now invoked from init; rename/move covering tests |
| `tests/cli_init.rs` | Add CPU-smoke variant; assert phase ordering when yaml is/isn't present |
| `tests/cli_workspace_uninit.rs` | Unchanged behavior, but verify message |
| `tests/cli_first_run_e2e.rs` | **NEW** — verify the 3-command happy path in a tmpdir |

---

## Task 1: Move `Workflow` enum from `cli/types.rs` to `config.rs`

This removes the layering inversion that prevents `Config` from owning a workflow field. `Workflow` keeps its `ValueEnum + Serialize + Deserialize` derives. Re-export from the old location so every `use crate::cli::types::Workflow;` keeps compiling.

**Files:**
- Modify: `src/config.rs` — add `Workflow` enum
- Modify: `src/cli/types.rs` — remove enum body, replace with re-export
- Test: existing tests under `tests/cli_*.rs` are the regression guard

- [ ] **Step 1: Add `Workflow` to `src/config.rs`**

Open `src/config.rs` and insert this block after the `ConfigMode` enum (~line 24):

```rust
// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum,
    serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum Workflow {
    Train,
    Eval,
    TrainAndTest,
}
```

The `clap::ValueEnum` derive needs `clap` to be available to `config.rs`. `clap` is already a dependency for the binary; add `clap = { workspace = true, optional = false }` to the lib's deps if not present.

- [ ] **Step 2: Confirm `clap` is reachable from the lib crate**

Run:
```bash
grep -A 2 "^clap" /home/tbindas/projects/ddrs/Cargo.toml
```
Expected: see `clap = { version = "4", features = ["derive"] }` listed in `[dependencies]`. If it's only in `[dev-dependencies]` or behind a feature, promote it to `[dependencies]`.

- [ ] **Step 3: Replace `Workflow` body in `src/cli/types.rs` with a re-export**

Open `src/cli/types.rs`. Change the file's top to:

```rust
use serde::{Deserialize, Serialize};

pub use crate::config::Workflow;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Ok,
    Failed,
    Interrupted,
}
```

(Keep `ExitCode` and any other types below unchanged. Only `Workflow` is moved.)

Also remove the now-unused `use clap::ValueEnum;` line at the top if no other enum in this file uses it.

- [ ] **Step 4: Build and run all tests**

Run:
```bash
cargo build
cargo test --lib
cargo test --tests
```
Expected: all green. Any compile error means a path that used `crate::cli::types::Workflow` resolved through the re-export incorrectly — verify the re-export line is `pub use`, not `use`.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/cli/types.rs Cargo.toml
git commit -m "$(cat <<'EOF'
refactor: move Workflow enum to config.rs, re-export from cli::types

Lets Config own the workflow field without a cli→config layering
inversion. All existing imports keep working via re-export.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `workflow: Option<Workflow>` to `Config`

**Files:**
- Modify: `src/config.rs` — add field to `Config` and `ConfigRaw`; thread through `From<ConfigRaw>`
- Test: `src/config.rs` — add unit test that reads workflow from YAML

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests { … }` block at the bottom of `src/config.rs`:

```rust
#[test]
fn loads_workflow_from_yaml() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train-and-test
"#;
    let path = std::env::temp_dir().join("ddrs_config_workflow_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("load yaml");
    assert_eq!(cfg.workflow, Some(Workflow::TrainAndTest));
}

#[test]
fn workflow_absent_is_none() {
    let yaml = "mode: training\ngeodataset: merit\nseed: 1\nnp_seed: 1\n";
    let path = std::env::temp_dir().join("ddrs_config_no_workflow_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("load yaml");
    assert_eq!(cfg.workflow, None);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --lib loads_workflow_from_yaml
```
Expected: FAIL — "no field `workflow` on type `Config`".

- [ ] **Step 3: Add the field to `Config` and `ConfigRaw`**

Edit `src/config.rs`. In `pub struct Config { … }` (~line 235), add a field:

```rust
pub struct Config {
    pub params: Params,
    pub data_sources: Option<DataSources>,
    pub experiment: Option<Experiment>,
    pub kan_head: Option<KanHeadConfigSection>,
    pub mode: String,
    pub geodataset: String,
    pub seed: u64,
    pub np_seed: u64,
    pub workflow: Option<Workflow>,
}
```

In `struct ConfigRaw { … }` (~line 276), add:

```rust
struct ConfigRaw {
    mode: Option<String>,
    geodataset: Option<String>,
    seed: Option<u64>,
    np_seed: Option<u64>,
    workflow: Option<Workflow>,
    params: ParamsRaw,
    data_sources: Option<DataSources>,
    experiment: Option<Experiment>,
    #[serde(alias = "mlp")]
    kan_head: Option<KanHeadConfigSection>,
    testing: TestingOverridesRaw,
}
```

In `impl From<ConfigRaw> for Config { … }` (~line 291), thread the field:

```rust
impl From<ConfigRaw> for Config {
    fn from(r: ConfigRaw) -> Self {
        Self {
            params: r.params.into(),
            data_sources: r.data_sources,
            experiment: r.experiment,
            kan_head: r.kan_head,
            mode: r.mode.unwrap_or_else(|| "training".to_string()),
            geodataset: r.geodataset.unwrap_or_else(|| "merit".to_string()),
            seed: r.seed.unwrap_or(42),
            np_seed: r.np_seed.unwrap_or(42),
            workflow: r.workflow,
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run:
```bash
cargo test --lib loads_workflow_from_yaml workflow_absent_is_none
```
Expected: both PASS.

Also rerun the existing yaml test:
```bash
cargo test --lib loads_merit_training_yaml
```
Expected: PASS (workflow defaults to `None` for the current merit yaml — fine).

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "$(cat <<'EOF'
feat: deserialize top-level workflow key from ddrs.yaml

Config now has workflow: Option<Workflow>. Unset means
"caller must supply --workflow"; set means the CLI flag is
optional. Cross-validation against mode lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Cross-validate `mode` ↔ `workflow` consistency

`mode: training` with `workflow: eval` is contradictory. Reject at load time with a clear message.

**Files:**
- Modify: `src/config.rs` — validation inside `from_yaml_file_with_mode`
- Test: `src/config.rs` — unit test for both conflict directions

- [ ] **Step 1: Write the failing test**

Append to `#[cfg(test)] mod tests { … }`:

```rust
#[test]
fn mode_workflow_conflict_rejected() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: eval
"#;
    let path = std::env::temp_dir().join("ddrs_config_conflict_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let err = Config::from_yaml_file(&path).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("conflicting") && msg.contains("mode: training") && msg.contains("workflow: eval"),
        "expected conflict message, got: {msg}"
    );
}

#[test]
fn mode_testing_with_train_workflow_rejected() {
    let yaml = r#"
mode: testing
geodataset: merit
seed: 1
np_seed: 1
workflow: train
"#;
    let path = std::env::temp_dir().join("ddrs_config_conflict2_test.yaml");
    std::fs::write(&path, yaml).unwrap();
    let err = Config::from_yaml_file(&path).unwrap_err();
    assert!(format!("{}", err).contains("conflicting"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --lib mode_workflow_conflict_rejected mode_testing_with_train_workflow_rejected
```
Expected: FAIL — config loads without error.

- [ ] **Step 3: Add the validation**

Add to `src/data/error.rs` if missing, otherwise reuse the existing `DataError::Other` variant. To keep this surgical, raise a `DataError` with an inline message (find the existing pattern in the file — most variants take a `String`).

Then in `src/config.rs`, modify `from_yaml_file_with_mode` (~line 315) to add a check after `let mut cfg: Self = raw.into();`:

```rust
        let mut cfg: Self = raw.into();
        validate_mode_workflow(&cfg).map_err(|msg| DataError::Yaml {
            path: path.to_path_buf(),
            source: serde_yaml::Error::custom(msg),
        })?;
        if mode == ConfigMode::Testing {
            apply_testing_overlay(&mut cfg, testing_raw);
        }
        Ok(cfg)
    }
}

fn validate_mode_workflow(cfg: &Config) -> std::result::Result<(), String> {
    use Workflow::*;
    let Some(wf) = cfg.workflow else { return Ok(()); };
    let ok = match (cfg.mode.as_str(), wf) {
        ("training", Train | TrainAndTest) => true,
        ("testing", Eval) => true,
        _ => false,
    };
    if !ok {
        return Err(format!(
            "conflicting top-level keys — mode: {} but workflow: {} \
             (mode=training implies workflow ∈ {{train, train-and-test}}; \
              mode=testing implies workflow=eval)",
            cfg.mode,
            match wf { Train => "train", Eval => "eval", TrainAndTest => "train-and-test" },
        ));
    }
    Ok(())
}
```

At the top of `src/config.rs`, add the import needed for `serde_yaml::Error::custom`:

```rust
use serde::de::Error as _;
```

- [ ] **Step 4: Run tests**

Run:
```bash
cargo test --lib mode_workflow_conflict_rejected mode_testing_with_train_workflow_rejected loads_workflow_from_yaml
```
Expected: all PASS.

Run the full lib suite to check nothing else broke:
```bash
cargo test --lib
```
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "$(cat <<'EOF'
feat: cross-validate mode and workflow in ddrs.yaml

mode: training with workflow: eval (or testing+train) errors at
load time. Single canonical message names both conflicting keys.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Resolve workflow in `cli::plan` from flag-or-config

**Files:**
- Modify: `src/cli/plan.rs` — change resolution + error message
- Test: `tests/cli_plan.rs` — update existing test and add new yaml-key test

- [ ] **Step 1: Write the failing test**

Add to `tests/cli_plan.rs` (or create the file if it doesn't yet have this case):

```rust
#[test]
fn workflow_resolved_from_yaml_key() {
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train-and-test
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    fs::create_dir_all(ws_root.join("runs")).unwrap();
    // Seed an empty lockfile so plan doesn't bail on workspace check.
    let lock = ddrs::cli::lockfile::Lockfile {
        ddrs_version: "test".into(),
        created_at: "0".into(),
        sources: std::collections::BTreeMap::new(),
    };
    lock.write_atomic(&ws_root.join("sources.lock")).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    // No --workflow flag.
    let err = ddrs::cli::plan::plan(&cfg_path, None, &ws);
    // We expect this to FAIL on fingerprinting /dev/null (not on missing workflow).
    // The success criterion: the error is NOT "no workflow" — it's about the path.
    match err {
        Err(e) => {
            let msg = format!("{e}");
            assert!(!msg.contains("workflow"), "expected non-workflow error, got: {msg}");
        }
        Ok(_) => {} // tolerated
    }
}

#[test]
fn no_workflow_anywhere_gives_actionable_error() {
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    std::fs::create_dir_all(&ws_root).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let err = ddrs::cli::plan::plan(&cfg_path, None, &ws).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no `workflow:` key") && msg.contains("--workflow"),
        "got: {msg}"
    );
}
```

If `tempfile` is not already a dev-dep, add it:
```bash
grep -A 1 "^\[dev-dependencies\]" Cargo.toml | grep tempfile || \
  cargo add --dev tempfile
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --test cli_plan workflow_resolved_from_yaml_key no_workflow_anywhere_gives_actionable_error
```
Expected: FAIL — current code rejects with "neither --workflow nor `workflow:` key set" even when workflow is in YAML, because nothing reads the key.

- [ ] **Step 3: Update `src/cli/plan.rs`**

In `pub fn plan(…)` (~line 38), the current first block is:

```rust
    // Step 1: resolve workflow.
    let workflow = workflow_override.ok_or_else(|| CliError::ConfigInvalid {
        path: config_path.into(),
        source: "neither --workflow nor `workflow:` key set".into(),
    })?;

    // Step 2: parse config in the appropriate mode.
    let mode = match workflow {
        Workflow::Train | Workflow::TrainAndTest => ConfigMode::Training,
        Workflow::Eval => ConfigMode::Testing,
    };
    let config = Config::from_yaml_file_with_mode(config_path, mode)
        .map_err(|e| CliError::ConfigInvalid {
            path: config_path.into(),
            source: Box::new(e),
        })?;
```

Replace with:

```rust
    // Step 1: load config once (in Training mode initially — we'll re-parse
    // if the resolved workflow says testing).
    let preview = Config::from_yaml_file_with_mode(config_path, ConfigMode::Training)
        .map_err(|e| CliError::ConfigInvalid {
            path: config_path.into(),
            source: Box::new(e),
        })?;

    // Step 2: resolve workflow — CLI flag wins, then YAML, then error.
    let workflow = workflow_override.or(preview.workflow).ok_or_else(|| {
        CliError::ConfigInvalid {
            path: config_path.into(),
            source: format!(
                "no `workflow:` key in {}. Add `workflow: train-and-test` \
                 (or `train` / `eval`), or pass `--workflow <name>`.",
                config_path.display()
            ).into(),
        }
    })?;

    // Step 3: re-parse if the resolved workflow needs Testing overlay.
    let mode = match workflow {
        Workflow::Train | Workflow::TrainAndTest => ConfigMode::Training,
        Workflow::Eval => ConfigMode::Testing,
    };
    let config = if mode == ConfigMode::Training {
        preview
    } else {
        Config::from_yaml_file_with_mode(config_path, mode)
            .map_err(|e| CliError::ConfigInvalid {
                path: config_path.into(),
                source: Box::new(e),
            })?
    };
```

- [ ] **Step 4: Run tests**

Run:
```bash
cargo test --test cli_plan
```
Expected: both new tests PASS. Any other test in `cli_plan.rs` that asserted the *old* error message ("neither --workflow nor `workflow:` key set") needs its assertion updated to match the new wording. Find them with:

```bash
grep -n "neither --workflow" tests/cli_plan.rs
```
Update any matches to assert the new message: `no \`workflow:\` key`.

- [ ] **Step 5: Commit**

```bash
git add src/cli/plan.rs tests/cli_plan.rs Cargo.toml
git commit -m "$(cat <<'EOF'
feat: plan resolves workflow as flag-or-yaml-key

ddrs plan now reads `workflow:` from ddrs.yaml when the CLI
flag is absent. Error message names the file and the fix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Same workflow resolution in `cli::run`

`run` delegates to `plan` internally (it calls `plan(…)` at the top), so the resolution is already inherited from Task 4. This task verifies that flow end-to-end so future refactors don't accidentally re-introduce a separate resolution path.

**Files:**
- Test: `tests/cli_plan.rs` — add a `run`-side resolution test that asserts run accepts a yaml-only workflow

- [ ] **Step 1: Write the failing test**

This test confirms that `run`'s `RunInput { workflow: None, … }` succeeds resolution when `ddrs.yaml` carries a `workflow:` key (it should fail later, on missing data sources, not earlier on workflow).

Add to `tests/cli_plan.rs`:

```rust
#[test]
fn run_inherits_yaml_workflow_resolution() {
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    fs::create_dir_all(ws_root.join("runs")).unwrap();
    let lock = ddrs::cli::lockfile::Lockfile {
        ddrs_version: "test".into(),
        created_at: "0".into(),
        sources: std::collections::BTreeMap::new(),
    };
    lock.write_atomic(&ws_root.join("sources.lock")).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let res = ddrs::cli::run::run(ddrs::cli::run::RunInput {
        workspace: ws,
        config_path: cfg_path,
        workflow: None, // resolution must come from YAML
        plot: false,
        strict: false,
        max_mini_batches: Some(1),
    });
    // Expected: fails downstream (sandbox / data source / GPU), NOT at workflow.
    if let Err(e) = res {
        let msg = format!("{e}");
        assert!(!msg.contains("workflow:"), "got premature workflow error: {msg}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails OR passes**

Run:
```bash
cargo test --test cli_plan run_inherits_yaml_workflow_resolution -- --nocapture
```
Expected: PASS already (because Task 4's change to `plan` is reused by `run`). If it FAILS with a workflow error, the chain is broken — re-inspect `src/cli/run.rs:38` to confirm it calls `plan(&input.config_path, input.workflow, &input.workspace)`.

- [ ] **Step 3: No implementation change needed**

This task is a regression guard. If Step 2 already passes, proceed to commit.

- [ ] **Step 4: Commit**

```bash
git add tests/cli_plan.rs
git commit -m "$(cat <<'EOF'
test: regression guard for run inheriting yaml workflow

Locks the contract that `ddrs run` resolves workflow through
plan, not a separate path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add `backend` field to `SmokeTestRecord`

`Option<String>` keeps existing `.ddrs/system.json` files readable (deserializes to `None`). Old records without the field will be invalidated by Task 7's `smoke_key` change.

**Files:**
- Modify: `src/cli/manifest.rs:31-35` — `SmokeTestRecord` struct

- [ ] **Step 1: Write the failing test**

Add to a new test file `tests/cli_system_backend.rs`:

```rust
use ddrs::cli::manifest::{SmokeTestRecord, SystemProbe};

#[test]
fn smoke_record_with_backend_roundtrips() {
    let r = SmokeTestRecord {
        key: "x".into(),
        passed_at: "2026-06-01T00:00:00Z".into(),
        backend: Some("cuda".into()),
    };
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.contains("\"backend\":\"cuda\""));
    let r2: SmokeTestRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(r2, r);
}

#[test]
fn smoke_record_old_format_deserializes_with_none_backend() {
    // Old records (pre-backend field) must still load.
    let json = r#"{"key":"x","passed_at":"2026-06-01T00:00:00Z"}"#;
    let r: SmokeTestRecord = serde_json::from_str(json).unwrap();
    assert_eq!(r.backend, None);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --test cli_system_backend
```
Expected: FAIL — `SmokeTestRecord` has no `backend` field; cannot construct.

- [ ] **Step 3: Add the field**

Edit `src/cli/manifest.rs:31-35`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmokeTestRecord {
    pub key: String,
    pub passed_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}
```

- [ ] **Step 4: Run tests**

Run:
```bash
cargo test --test cli_system_backend
cargo test --lib
```
Expected: both green. The library smoke test usage (in `system::record_smoke`) needs an update — that's Task 7.

- [ ] **Step 5: Commit**

```bash
git add src/cli/manifest.rs tests/cli_system_backend.rs
git commit -m "$(cat <<'EOF'
feat: SmokeTestRecord carries a backend tag

Backward-compatible Option<String> so existing system.json
files keep loading.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Extend `smoke_key` and `record_smoke` with backend

**Files:**
- Modify: `src/cli/system.rs:80-93` — `smoke_key` and `record_smoke`

- [ ] **Step 1: Write the failing test**

Append to `tests/cli_system_backend.rs`:

```rust
#[test]
fn smoke_key_includes_backend() {
    let probe = SystemProbe {
        ddrs_version: "1".into(),
        probed_at: "t".into(),
        gpu: "g".into(),
        cuda_runtime: "12.4".into(),
        driver: "530".into(),
        sm: "8.0".into(),
        free_gpu_gb_at_probe: 1.0,
        smoke_test: None,
    };
    let k_cuda = ddrs::cli::system::smoke_key_for(&probe, "cuda");
    let k_cpu  = ddrs::cli::system::smoke_key_for(&probe, "cpu");
    assert_ne!(k_cuda, k_cpu);
    assert!(k_cuda.contains("backend=cuda"));
    assert!(k_cpu.contains("backend=cpu"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --test cli_system_backend smoke_key_includes_backend
```
Expected: FAIL — `smoke_key_for` doesn't exist.

- [ ] **Step 3: Modify `src/cli/system.rs`**

Replace lines 78-93 with:

```rust
/// Stable key used to decide whether a cached smoke-test verdict is still
/// valid. Re-run when this string changes. Backend-aware: switching between
/// CUDA and CPU invalidates the cache.
pub fn smoke_key(probe: &SystemProbe, backend: &str) -> String {
    format!(
        "driver={};cuda={};ddrs={};sm={};backend={}",
        probe.driver, probe.cuda_runtime, probe.ddrs_version, probe.sm, backend
    )
}

/// Shim kept for the regression test only.
pub fn smoke_key_for(probe: &SystemProbe, backend: &str) -> String {
    smoke_key(probe, backend)
}

pub fn record_smoke(probe: &mut SystemProbe, key: String, backend: &str) {
    probe.smoke_test = Some(SmokeTestRecord {
        key,
        passed_at: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        backend: Some(backend.to_string()),
    });
}
```

(The `smoke_key_for` alias keeps the test independent of in-progress refactors; remove it after Task 8 if you want.)

- [ ] **Step 4: Update callers in `src/cli/init.rs`**

`src/cli/init.rs:61` currently calls `system::smoke_key(&probe)` (single arg). Find and update both call sites in that file — they'll be re-touched in Task 8 too. For now, leave a temporary `"cuda"` arg so the file compiles:

```rust
    let key = system::smoke_key(&probe, "cuda");
    // ...
    if smoke_passed && !smoke_reused {
        system::record_smoke(&mut probe, key, "cuda");
    }
```

- [ ] **Step 5: Run tests**

Run:
```bash
cargo build
cargo test --test cli_system_backend
cargo test --lib
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/cli/system.rs src/cli/init.rs tests/cli_system_backend.rs
git commit -m "$(cat <<'EOF'
feat: smoke_key + record_smoke take a backend tag

Switching between CUDA and CPU now invalidates the smoke cache.
Init still passes 'cuda' literally — the CPU-fallback wiring
lands in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Backend-aware `run_smoke` (CPU fallback)

**Files:**
- Modify: `src/cli/init.rs:144-151` — `run_smoke` plus its caller

- [ ] **Step 1: Write the failing test**

Add to `tests/cli_init.rs`:

```rust
#[test]
fn run_smoke_returns_cpu_when_no_cuda() {
    use ddrs::cli::manifest::SystemProbe;
    // We can't easily fake an absent CUDA, but we *can* call the new
    // public helper that picks backend from a SystemProbe. Build a probe
    // with gpu == "" to force the CPU path.
    let mut probe = SystemProbe::default();
    probe.gpu = String::new();
    let (passed, backend) = ddrs::cli::init::run_smoke_for_test(&probe).unwrap();
    assert!(passed, "CPU smoke must pass on the bundled sandbox fixture");
    assert_eq!(backend, "cpu");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --test cli_init run_smoke_returns_cpu_when_no_cuda
```
Expected: FAIL — `run_smoke_for_test` doesn't exist; current `run_smoke` returns only `bool` and hardcodes CUDA.

- [ ] **Step 3: Replace `run_smoke` and add test helper**

In `src/cli/init.rs`, replace the bottom `fn run_smoke()` and update its caller (~line 73).

Replace lines 144-151 with:

```rust
fn run_smoke(probe: &crate::cli::manifest::SystemProbe)
    -> Result<(bool, &'static str), CliError>
{
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
pub fn run_smoke_for_test(probe: &crate::cli::manifest::SystemProbe)
    -> Result<(bool, &'static str), CliError>
{
    run_smoke(probe)
}
```

Now update the call site at lines 61-82. Replace:

```rust
    let key = system::smoke_key(&probe, "cuda");
    let cached_passing = SystemProbe::read(&ws.system_json())
        .ok()
        .and_then(|p| p.smoke_test)
        .map(|s| s.key == key)
        .unwrap_or(false);
    let (smoke_passed, smoke_reused) = if input.skip_smoke {
        (true, cached_passing)
    } else if cached_passing && !input.force {
        (true, true)
    } else {
        (run_smoke()?, false)
    };
    if smoke_passed && !smoke_reused {
        system::record_smoke(&mut probe, key, "cuda");
    } else if smoke_reused {
        if let Ok(prior) = SystemProbe::read(&ws.system_json()) {
            probe.smoke_test = prior.smoke_test;
        }
    }
```

with:

```rust
    // Pick backend up-front so the cache key matches the work we'd do.
    let backend = if probe.gpu.is_empty() { "cpu" } else { "cuda" };
    let key = system::smoke_key(&probe, backend);
    let cached_passing = SystemProbe::read(&ws.system_json())
        .ok()
        .and_then(|p| p.smoke_test)
        .map(|s| s.key == key)
        .unwrap_or(false);
    let (smoke_passed, smoke_reused) = if input.skip_smoke {
        (true, cached_passing)
    } else if cached_passing && !input.force {
        (true, true)
    } else {
        let (ok, _b) = run_smoke(&probe)?;
        (ok, false)
    };
    if smoke_passed && !smoke_reused {
        system::record_smoke(&mut probe, key, backend);
    } else if smoke_reused {
        if let Ok(prior) = SystemProbe::read(&ws.system_json()) {
            probe.smoke_test = prior.smoke_test;
        }
    }
```

Also remove the old GPU-required warning at lines 46-51 (the smoke log line now signals CPU mode). Replace lines 44-57 with:

```rust
    // ── Phase A: install-level probes (no config required) ─────────────
    let mut probe = system::probe()?.unwrap_or_default();
    if probe.free_gpu_gb_at_probe < input.min_free_gpu_gb && probe.free_gpu_gb_at_probe > 0.0 {
        eprintln!(
            "warning: free GPU memory {:.1} GB is below floor {} GB",
            probe.free_gpu_gb_at_probe, input.min_free_gpu_gb
        );
    }
```

- [ ] **Step 4: Run tests**

Run:
```bash
cargo build
cargo test --test cli_init run_smoke_returns_cpu_when_no_cuda
cargo test --lib
cargo test --tests
```
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/cli/init.rs
git commit -m "$(cat <<'EOF'
feat: init smoke falls back to CPU when no CUDA is detected

The 5-reach RAPID sandbox is tiny; CPU smoke runs sub-second.
Removes the GPU-required error so laptop and CI runs of
`ddrs init` succeed. Run pre-flight (next commit) takes over
the GPU requirement for workflows that actually need it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: GPU pre-flight in `cli::run` for training workflows

`run` errors early with a clear message when a GPU is required but absent. `eval` is allowed through (it can run on CPU if anyone needs it).

**Files:**
- Modify: `src/cli/run.rs` — add pre-flight after Step 1 (plan)

- [ ] **Step 1: Write the failing test**

Add to a new file `tests/cli_run_preflight.rs`:

```rust
#[test]
fn run_train_requires_gpu_when_none_probed() {
    // We can't reliably suppress CUDA on the test machine, so this test
    // is "soft": if a GPU IS present, the test skips. Otherwise it
    // asserts the pre-flight fires with a clear message.
    if ddrs::cli::system::probe().ok().flatten()
        .map(|p| !p.gpu.is_empty()).unwrap_or(false)
    {
        eprintln!("skipping — GPU present on this host");
        return;
    }
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let cfg_path = tmp.path().join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
workflow: train
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
"#).unwrap();
    let ws_root = tmp.path().join(".ddrs");
    fs::create_dir_all(ws_root.join("runs")).unwrap();
    let lock = ddrs::cli::lockfile::Lockfile {
        ddrs_version: "test".into(),
        created_at: "0".into(),
        sources: std::collections::BTreeMap::new(),
    };
    lock.write_atomic(&ws_root.join("sources.lock")).unwrap();
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let err = ddrs::cli::run::run(ddrs::cli::run::RunInput {
        workspace: ws,
        config_path: cfg_path,
        workflow: None,
        plot: false,
        strict: false,
        max_mini_batches: Some(1),
    }).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("requires a CUDA GPU") && msg.contains("train"),
        "expected GPU pre-flight, got: {msg}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails (or skip on GPU host)**

Run:
```bash
cargo test --test cli_run_preflight -- --nocapture
```
Expected on a CPU-only host: FAIL — `run` proceeds past the missing GPU and dies later with a cryptic burn error. On a GPU host the test prints `skipping`.

- [ ] **Step 3: Add the pre-flight to `src/cli/run.rs`**

Insert this block in `pub fn run(input: RunInput) -> …` right after the `plan(…)` call (~line 38), before the drift check:

```rust
    // 1b. GPU pre-flight for workflows that need training kernels.
    if matches!(pr.workflow, Workflow::Train | Workflow::TrainAndTest) {
        let has_gpu = crate::cli::system::probe()
            .ok()
            .flatten()
            .map(|p| !p.gpu.is_empty())
            .unwrap_or(false);
        if !has_gpu {
            return Err(CliError::Runtime(format!(
                "run: workflow `{}` requires a CUDA GPU; system probe found none. \
                 Smoke verified the routing core works on CPU, but production \
                 training does not.",
                match pr.workflow {
                    Workflow::Train => "train",
                    Workflow::TrainAndTest => "train-and-test",
                    Workflow::Eval => "eval",
                }
            )));
        }
    }
```

- [ ] **Step 4: Run tests**

Run:
```bash
cargo test --test cli_run_preflight -- --nocapture
cargo test --tests
```
Expected: pre-flight test PASS (or skipped on GPU host). All other tests still green.

- [ ] **Step 5: Commit**

```bash
git add src/cli/run.rs tests/cli_run_preflight.rs
git commit -m "$(cat <<'EOF'
feat: run pre-flight errors when train workflow lacks a GPU

Clear up-front error replaces the cryptic burn-side panic
that a CPU-only host would have hit deep in training.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Move `ddrs.yaml` bootstrap from `plan` into `init`

`init` becomes the single first-run command. If `ddrs.yaml` is missing, init bootstraps it via `$EDITOR` before Phase E (lock).

**Files:**
- Modify: `src/cli/init.rs` — insert Phase D before Phase E
- Modify: `src/bin/ddrs.rs` — remove bootstrap from the `Plan` dispatch branch
- Test: `tests/cli_init.rs` — add a test that asserts init's behavior when yaml is missing in non-interactive mode

- [ ] **Step 1: Write the failing test**

Add to `tests/cli_init.rs`:

```rust
#[test]
fn init_errors_clearly_when_no_yaml_and_no_tty() {
    // In CI/test contexts stdin is never a TTY, so we exercise the
    // non-interactive guard.
    let tmp = tempfile::tempdir().unwrap();
    let ws_root = tmp.path().join(".ddrs");
    let res = ddrs::cli::init::run_init(ddrs::cli::init::InitInput {
        workspace: ws_root,
        config_path: None,
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true, // keep test fast
    });
    let err = res.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no ddrs.yaml found") && msg.contains("not a TTY"),
        "expected non-interactive bootstrap error, got: {msg}"
    );
}
```

(Note: this changes prior behavior, which returned `Ok(InitOutput { phase_b_skipped: true, … })`. The new contract: init *must* produce a usable workspace, and a missing yaml in non-interactive mode is an error.)

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --test cli_init init_errors_clearly_when_no_yaml_and_no_tty
```
Expected: FAIL — init currently returns `Ok` with a "run plan to bootstrap" message instead of erroring.

- [ ] **Step 3: Modify `src/cli/init.rs` to bootstrap inline**

Replace lines 85-95 (the current Phase B skip block) with:

```rust
    // ── Phase D: bootstrap ddrs.yaml if missing (interactive) ─────────
    let config_path = input.config_path.or_else(|| {
        crate::cli::workspace::discover_config(Path::new("."))
    });
    let cfg_path = match config_path {
        Some(p) => p,
        None => {
            // Compose the bootstrap target. Default to ./ddrs.yaml.
            let target = std::env::current_dir()
                .map_err(CliError::from)?
                .join("ddrs.yaml");
            let bundled = PathBuf::from("config/merit_training.yaml");
            crate::cli::plan_bootstrap::bootstrap(
                crate::cli::plan_bootstrap::BootstrapInput {
                    target: target.clone(),
                    runs_dir: ws.runs_dir(),
                    bundled_template: bundled,
                    editor_cmd: None,
                    interactive: true,
                },
            ).map_err(|e| {
                // Rewrite the bootstrap TTY error to the init-specific form.
                let msg = format!("{e}");
                if msg.contains("not a TTY") || msg.contains("run interactively") {
                    CliError::ConfigInvalid {
                        path: target.clone(),
                        source: "no ddrs.yaml found and stdin is not a TTY. \
                                 Write ddrs.yaml manually, then re-run `ddrs init`."
                                .into(),
                    }
                } else {
                    e
                }
            })?;
            target
        }
    };

    // ── Phase E: lock data sources from the (now-present) yaml ─────────
    let cfg = Config::from_yaml_file_with_mode(&cfg_path, ConfigMode::Training)
        .map_err(|e| CliError::ConfigInvalid { path: cfg_path.clone(), source: Box::new(e) })?;
```

(Keep the rest of Phase E — the `ds = cfg.data_sources…` through `lock.write_atomic(…)` — unchanged.)

Also remove `phase_b_skipped` from the success path's `Ok(InitOutput { … })` if no longer reachable; if the field is referenced by other tests, leave the field on the struct but set it to `false`. To check:

```bash
grep -rn "phase_b_skipped" src tests
```
If only init.rs references it, the field can be removed entirely. Otherwise, leave it.

- [ ] **Step 4: Remove the bootstrap branch from `src/bin/ddrs.rs`**

In `src/bin/ddrs.rs`, the `Cmd::Plan` arm (~lines 79-107) currently bootstraps yaml when `cfg_path` is `None`. That responsibility now lives in `init`. Replace the `Plan` arm with:

```rust
        Cmd::Plan { workflow, json } => {
            let cfg_path = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
                path: ".".into(),
                source: "no ddrs.yaml found in current directory. \
                         Run `ddrs init` first.".into(),
            })?;
            let pr = ddrs::cli::plan::plan(&cfg_path, workflow, &ws)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&pr)
                    .map_err(|e| CliError::Other(Box::new(e)))?);
            } else {
                println!("workflow {:?}", pr.workflow);
                println!("drift    {:?}", pr.drift);
            }
            Ok(())
        }
```

- [ ] **Step 5: Run tests**

Run:
```bash
cargo build
cargo test --test cli_init init_errors_clearly_when_no_yaml_and_no_tty
cargo test --tests
```
Expected: green. If `tests/cli_plan_bootstrap.rs` had tests asserting bootstrap happened from `plan`, update them to call into init instead — or delete them, since Task 10's new init test covers the equivalent behavior.

Find such tests:
```bash
grep -rn "plan_bootstrap\|bootstrap" tests/
```
Update any caller of `plan()` that previously relied on its bootstrap side-effect to either pre-create `ddrs.yaml` or call `run_init` first.

- [ ] **Step 6: Commit**

```bash
git add src/cli/init.rs src/bin/ddrs.rs tests/cli_init.rs tests/cli_plan_bootstrap.rs
git commit -m "$(cat <<'EOF'
feat: init bootstraps ddrs.yaml inline; plan no longer does

First-run flow collapses to: ddrs init → ddrs plan → ddrs run.
Non-TTY init without ddrs.yaml errors with a clear remediation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Add `workflow: train-and-test` to the bundled config

The bundled `config/merit_training.yaml` is what `init` copies as the template. Adding `workflow:` to it means new users see the key immediately and can `ddrs plan` with no flag.

**Files:**
- Modify: `config/merit_training.yaml` — add `workflow: train-and-test` near the top

- [ ] **Step 1: Inspect the current top of the file**

Run:
```bash
head -20 config/merit_training.yaml
```

You should see `mode: training`, `geodataset: merit`, etc. Insert `workflow:` right below `mode:`.

- [ ] **Step 2: Edit the file**

Open `config/merit_training.yaml` and add line 11 (right after `mode: training`):

```yaml
mode: training
workflow: train-and-test    # ddrs plan/run picks this up; override with --workflow X
geodataset: merit
```

- [ ] **Step 3: Run the yaml loader test**

Run:
```bash
cargo test --lib loads_merit_training_yaml
```
Expected: PASS. The test doesn't currently assert workflow, so this just confirms the file still parses.

Optionally tighten the test by adding one assertion:

```rust
assert_eq!(cfg.workflow, Some(Workflow::TrainAndTest));
```

at the bottom of `fn loads_merit_training_yaml()`.

- [ ] **Step 4: Commit**

```bash
git add config/merit_training.yaml src/config.rs
git commit -m "$(cat <<'EOF'
chore: bundled config sets workflow: train-and-test by default

New users running `ddrs init` get a yaml with workflow already
set; `ddrs plan` works with no flag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Write `README.md` "Getting started" section

**Files:**
- Modify: `README.md` — replace the current 1-line file with a real intro + setup

- [ ] **Step 1: Write the README**

The current `README.md` contains only `# ddrs` (no newline). Overwrite with:

```markdown
# ddrs

Differentiable distributed routing. A BURN-based Rust port of the
Muskingum-Cunge routing solver from [DDR](https://github.com/) (Python/PyTorch),
gradient-exact against the reference at single precision.

## Getting started

### Install

```bash
cargo install --path .
```

This puts the `ddrs` binary in `~/.cargo/bin/`. If that directory isn't on
your `PATH`:

```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### First-time setup

From your project root:

```bash
ddrs init      # creates ./.ddrs/, probes GPU, runs smoke test,
               # opens $EDITOR on ddrs.yaml, locks data sources
ddrs plan      # validates ddrs.yaml against locked sources, prints summary
ddrs run       # executes the workflow, writes manifest + outputs
```

`init` runs a 5-reach RAPID sandbox parity check on CUDA when available and
falls back to CPU otherwise — so the install path works on laptops and CI.
The bundled `config/merit_training.yaml` is the editor template; the
`workflow:` key is already set to `train-and-test`.

### What lives where

| Path | Written by | Purpose |
|---|---|---|
| `ddrs.yaml` | `ddrs init` (via `$EDITOR`) | Workflow + experiment config |
| `.ddrs/system.json` | `ddrs init` | GPU/driver/smoke-test record |
| `.ddrs/sources.lock` | `ddrs init` | Fingerprints of `data_sources` paths |
| `.ddrs/runs/<id>/manifest.json` | `ddrs run` | Per-run manifest (config + sources + git SHA + outputs) |
| `output/predictions_latest.zarr` | `ddrs run --workflow eval` / `train-and-test` Phase 2 | Predictions for plotting |
| `output/saved_models_*/epoch_*_mb_*.mpk` | `ddrs run --workflow train` / `train-and-test` Phase 1 | KAN checkpoints |

### Override workflow on the command line

The `workflow:` key in `ddrs.yaml` is what `plan`/`run` use by default. To
override for a single invocation:

```bash
ddrs plan --workflow eval
ddrs run --workflow train
```

`mode:` and `workflow:` must agree (`mode: training` ↔ `workflow ∈ {train, train-and-test}`; `mode: testing` ↔ `workflow: eval`). `ddrs init` will reject contradictions at load time.

### Advanced

- `ddrs show <run_id>` — inspect a past run's manifest
- `ddrs status` — list runs
- `ddrs gc` — clean up old run directories
- `ddrs <cmd> --help` for full flag list
```

- [ ] **Step 2: Render-check**

Run:
```bash
head -40 README.md
```
Expected: clean Markdown, no template artefacts.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: README Getting started section

Walks new users through install, the three-command first-run
flow, where outputs land, and how to override workflow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: End-to-end happy-path integration test

Verify the documented `init → plan → run` sequence works in a tmpdir. Uses the bundled merit yaml verbatim — the only data sources that exist are the bundled fixture, so we substitute a minimal yaml.

**Files:**
- Test: `tests/cli_first_run_e2e.rs` — NEW

- [ ] **Step 1: Write the test**

Create `tests/cli_first_run_e2e.rs`:

```rust
//! Smokes the documented first-run flow:
//!   1. write ddrs.yaml (substitute for $EDITOR step)
//!   2. ddrs::cli::init::run_init (skip_smoke=false, but allow cpu fallback)
//!   3. ddrs::cli::plan::plan (no --workflow flag — must resolve from yaml)
//!
//! `ddrs run` is NOT exercised end-to-end here — it needs real CONUS data.
//! The pre-flight test in cli_run_preflight covers the workflow=train branch.

use std::fs;

#[test]
fn first_run_flow_init_then_plan() {
    let tmp = tempfile::tempdir().unwrap();
    let proj = tmp.path();
    let cfg_path = proj.join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
workflow: train-and-test
geodataset: merit
seed: 1
np_seed: 1
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
experiment:
  batch_size: 1
  start_time: "2000-01-01"
  end_time: "2000-01-02"
  epochs: 1
  warmup: 1
"#).unwrap();
    let ws_root = proj.join(".ddrs");

    // Step 2: init. /dev/null paths will fail fingerprinting → init errors.
    // For a happy-path init we point at real files; tmp files are fine.
    for name in ["attributes", "conus_adjacency", "gages_adjacency",
                 "streamflow", "observations", "gages"] {
        let p = proj.join(format!("{name}.bin"));
        fs::write(&p, b"x").unwrap();
        // Patch yaml to point at it.
        let s = fs::read_to_string(&cfg_path).unwrap();
        let s = s.replace(&format!("{name}: /dev/null"),
                          &format!("{name}: {}", p.display()));
        fs::write(&cfg_path, s).unwrap();
    }

    let init_out = ddrs::cli::init::run_init(ddrs::cli::init::InitInput {
        workspace: ws_root.clone(),
        config_path: Some(cfg_path.clone()),
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true, // keep test fast; smoke is exercised in cli_init.rs
    }).expect("init succeeds");
    assert!(init_out.smoke_passed);
    assert!(ws_root.join("sources.lock").is_file());

    // Step 3: plan with no --workflow — must resolve from yaml.
    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let pr = ddrs::cli::plan::plan(&cfg_path, None, &ws)
        .expect("plan resolves workflow from yaml");
    assert_eq!(pr.workflow, ddrs::cli::Workflow::TrainAndTest);
    assert!(pr.drift.is_empty(), "no drift expected on fresh init");
}
```

- [ ] **Step 2: Run the test**

Run:
```bash
cargo test --test cli_first_run_e2e -- --nocapture
```
Expected: PASS. If it fails at `init_out.smoke_passed`, you may have `skip_smoke: false` accidentally — the test forces `true`. If plan errors with "no workflow", check that Task 4 actually landed.

- [ ] **Step 3: Commit**

```bash
git add tests/cli_first_run_e2e.rs
git commit -m "$(cat <<'EOF'
test: end-to-end first-run flow (init then plan)

Locks the three-command happy path documented in README:
init bootstraps the workspace + lock, plan resolves workflow
from the yaml with no CLI flag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Manual smoke on real working directory

Plan-level changes verified by tests — this step verifies the user-visible flow in the actual repo so the spec's acceptance criteria 5 (README-followable) is satisfied.

**Files:** none modified — verification only.

- [ ] **Step 1: Rebuild and reinstall the binary**

```bash
cargo install --path . --force
```

Expected: build succeeds; `ddrs` rewritten in `~/.cargo/bin/`.

- [ ] **Step 2: Move the existing ddrs.yaml aside, dry-run init**

```bash
mv ddrs.yaml ddrs.yaml.bak 2>/dev/null
rm -rf .ddrs
EDITOR=true ddrs init   # `true` exits 0 immediately — skips actual edit
ls .ddrs/
cat .ddrs/system.json | head -20
```

Expected: `.ddrs/` contains `system.json`, `sources.lock`, `version`, `runs/`. `system.json` shows a `smoke_test` block with `backend: "cuda"` or `"cpu"` matching the host.

(`EDITOR=true` lets init proceed without opening an editor on a freshly-bootstrapped yaml. The user would normally edit interactively.)

- [ ] **Step 3: Verify plan works with no flag**

Restore the real yaml (with `workflow: train-and-test` from Task 11):

```bash
mv ddrs.yaml.bak ddrs.yaml
ddrs plan
```

Expected: prints `workflow TrainAndTest` and `drift    []` (or a clear data-drift message if your data paths changed). No "neither --workflow" error.

- [ ] **Step 4: Verify the GPU pre-flight (only meaningful on a CPU-only host)**

If you're on a GPU machine, skip. Otherwise:

```bash
ddrs run --max-mini-batches 1
```

Expected: clear error message containing "requires a CUDA GPU".

- [ ] **Step 5: No commit needed**

This task is a manual verification gate. If any step fails, return to the relevant prior task.

---

## Self-review notes

**Spec coverage check:**

| Spec §  | Requirement | Task |
|---|---|---|
| 1 | `workflow:` top-level YAML key | 2 |
| 1 | Workflow enum lives in config.rs | 1 |
| 1 | Resolution: cli.or(yaml).or(error) | 4, 5 |
| 1 | mode↔workflow cross-validation | 3 |
| 1 | Bootstrap template carries `workflow:` line | 11 |
| 2 | init does workspace + probe + smoke + bootstrap + lock | 8, 10 |
| 2 | Re-running init is idempotent | (existing — Phase E rewrites lockfile unconditionally) |
| 3 | Backend-aware smoke (CUDA / NdArray) | 8 |
| 3 | smoke_key includes backend term | 7 |
| 3 | SmokeTestRecord has backend field | 6 |
| 3 | GPU-required check moves to run pre-flight | 8 (removal), 9 (re-add) |
| 4 | Error message rewrites | 4 (plan), 10 (init bootstrap removal) |
| 5 | README "Getting started" section | 12 |

All spec sections covered. Idempotency in §2 isn't a new task — current Phase E already rewrites the lockfile every call, so re-running init naturally re-fingerprints.

**Type / API consistency:**

- `Workflow` is moved to `config.rs` in Task 1 and re-exported. Every later task imports it from `crate::cli::types::Workflow` or `crate::cli::Workflow` (both work via re-export) or from `crate::config::Workflow` directly. All three forms work.
- `smoke_key(probe, backend)` — 2-arg signature consistent across Tasks 7, 8.
- `record_smoke(probe, key, backend)` — 3-arg signature consistent across Tasks 7, 8.
- `run_smoke(probe)` — takes a `&SystemProbe`, returns `(bool, &'static str)`. Used in init.rs (Task 8) and `run_smoke_for_test` (Task 8).
- `Config.workflow: Option<Workflow>` — set in Task 2, read in Task 4, written by the merit yaml in Task 11.

**Placeholder check:** none. Every code step has full code. Every command has expected output. No "similar to" backreferences.

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-06-01-ddrs-cli-ux-and-readme.md`.

Tasks are tightly coupled (each depends on earlier tasks' types and field shapes) but otherwise independent enough for subagent dispatch — the controller threads the type signatures through prompts. Each task has self-contained tests so subagent dispatch with two-stage review is the recommended approach.
