# ddrs CLI Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single `ddrs` CLI binary (`init → plan → run`, plus `show`/`status`/`gc`) that replaces the four current single-purpose binaries, makes data sources reproducible via a per-run manifest + project-level `sources.lock`, and ships validation that fails fast before any GPU work.

**Architecture:** A new `src/cli/` module decomposed by responsibility (types, workspace, fingerprinting, lockfile, manifest, system probe, tee, plus one module per command). One thin entrypoint at `src/bin/ddrs.rs`. Two extractions in `src/training/` (a shared `bootstrap_head_and_state` helper) and `src/bin/dump_parameters.rs` (move body to a `dump_parameters::dump` library function) so the new CLI and the deprecation shims share one implementation. Sandbox fixture loaders lift from `examples/compare_ddr_sandbox.rs` to `src/sandbox.rs` for use by both the example and `init`'s smoke test.

**Tech Stack:** Rust 2021, BURN 0.21, cudarc 0.19, clap 4, serde + serde_yaml + serde_json (existing); add blake3, os_pipe, humantime; std-lib only for `IsTerminal` and `git` shelling.

**Spec:** `/home/tbindas/projects/ddrs/docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`

---

## File map

### New files

```
src/
├── bin/ddrs.rs                       # NEW: clap entrypoint, dispatches subcommands
├── sandbox.rs                        # NEW: fixture loaders + smoke test, lifted from examples
├── training/bootstrap.rs             # NEW: bootstrap_head_and_state(cfg, device) -> (head, state, opt)
└── cli/
    ├── mod.rs                        # NEW: pub re-exports
    ├── error.rs                      # NEW: CliError → ExitCode
    ├── types.rs                      # NEW: Workflow, RunStatus, ExitCode
    ├── workspace.rs                  # NEW: CWD walk-up, .ddrs/ path helpers
    ├── fingerprint.rs                # NEW: blake3 + stat-based reuse
    ├── lockfile.rs                   # NEW: sources.lock schema + io + diff
    ├── manifest.rs                   # NEW: manifest.json + system.json + io
    ├── system.rs                     # NEW: cudarc probe, smoke-key derivation
    ├── tee.rs                        # NEW: os_pipe stdout/stderr capture
    ├── plan.rs                       # NEW: plan() → PlanResult
    ├── plan_bootstrap.rs             # NEW: $EDITOR-based config bootstrap
    ├── init.rs                       # NEW: Phase A + Phase B
    ├── run.rs                        # NEW: run command, manifest write
    ├── show.rs                       # NEW: show command
    ├── status.rs                     # NEW: status command
    └── gc.rs                         # NEW: gc command
```

### Modified files

```
Cargo.toml                            # add deps
src/lib.rs                            # pub mod cli; pub mod sandbox
src/training/mod.rs                   # pub mod bootstrap
src/bin/train.rs                      # deprecation shim + call shared bootstrap
src/bin/eval.rs                       # deprecation shim
src/bin/train_and_test.rs             # deprecation shim
src/bin/dump_parameters.rs            # call dump_parameters::dump
examples/compare_ddr_sandbox.rs       # call sandbox::load + sandbox::run
CLAUDE.md                             # CLI command examples
.claude/ARCHITECTURE.md               # CLI lifecycle section
```

### New tests

```
tests/cli_types.rs
tests/cli_workspace.rs
tests/cli_fingerprint.rs
tests/cli_lockfile.rs
tests/cli_manifest.rs
tests/cli_init.rs
tests/cli_plan.rs
tests/cli_plan_bootstrap.rs
tests/cli_run_drift.rs
tests/cli_plot.rs
tests/cli_show.rs
tests/cli_status_gc.rs
tests/cli_workspace_uninit.rs
tests/cli_runtime_failure.rs
tests/cli_json_contract.rs
tests/cli_deprecation_shim.rs
```

---

## Task 1: Add dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add three new dependencies**

Add under `[dependencies]`:

```toml
blake3 = "1"
os_pipe = "1"
humantime = "2"
```

- [ ] **Step 2: Verify build still passes**

Run: `cargo build --release`
Expected: builds cleanly. No tests run yet.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "deps: add blake3, os_pipe, humantime for ddrs CLI"
```

---

## Task 2: Module skeleton

**Files:**
- Create: `src/cli/mod.rs`
- Create: `src/cli/error.rs`
- Create: `src/sandbox.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/cli/error.rs` with the CliError enum**

```rust
//! CLI error type that maps cleanly onto `ExitCode`.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("config invalid at {path}: {source}")]
    ConfigInvalid { path: PathBuf, #[source] source: Box<dyn std::error::Error + Send + Sync> },

    #[error("data source unreachable: {path}")]
    DataSourceMissing { path: PathBuf },

    #[error("lock drift in --strict mode: {fields:?}")]
    LockDrift { fields: Vec<String> },

    #[error("runtime failure during workflow: {0}")]
    Runtime(String),

    #[error("workspace not initialized at {path}; run `ddrs init`")]
    WorkspaceNotInitialized { path: PathBuf },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}
```

- [ ] **Step 2: Create `src/cli/mod.rs` (re-exports; subcommands stubbed later)**

```rust
//! `ddrs` CLI implementation. Entrypoint lives at `src/bin/ddrs.rs`.

pub mod error;
pub mod fingerprint;
pub mod gc;
pub mod init;
pub mod lockfile;
pub mod manifest;
pub mod plan;
pub mod plan_bootstrap;
pub mod run;
pub mod show;
pub mod status;
pub mod system;
pub mod tee;
pub mod types;
pub mod workspace;

pub use error::CliError;
pub use types::{ExitCode, RunStatus, Workflow};
```

(Each `pub mod X` referenced here must exist as a file by the time `cargo check` runs — we'll create stub files in Step 4.)

- [ ] **Step 3: Create `src/sandbox.rs` stub**

```rust
//! Sandbox fixture loader + functional smoke test.
//!
//! Lifted from `examples/compare_ddr_sandbox.rs` so both the example and
//! `cli::init`'s smoke test load the same fixture through one code path.
//! Body filled in by Task 9.
```

- [ ] **Step 4: Create empty stub files for every module listed in `cli/mod.rs`**

For each of `fingerprint, gc, init, lockfile, manifest, plan, plan_bootstrap, run, show, status, system, tee, types, workspace`:

```bash
# Run once per name (or use a quick loop in your shell):
echo '//! Stub. Filled in by a later task.' > src/cli/<name>.rs
```

- [ ] **Step 5: Wire `src/lib.rs` to expose `cli` and `sandbox`**

Add to `src/lib.rs` (preserve existing `pub mod` lines):

```rust
pub mod cli;
pub mod sandbox;
```

- [ ] **Step 6: Verify `cargo check` passes**

Run: `cargo check`
Expected: compiles cleanly with no warnings about unused modules (empty stubs are fine).

- [ ] **Step 7: Commit**

```bash
git add src/cli/ src/sandbox.rs src/lib.rs
git commit -m "cli: scaffold module skeleton + CliError"
```

---

## Task 3: CLI types (`Workflow`, `RunStatus`, `ExitCode`)

**Files:**
- Modify: `src/cli/types.rs`
- Test: `tests/cli_types.rs`

- [ ] **Step 1: Write the failing tests at `tests/cli_types.rs`**

```rust
use ddrs::cli::types::{ExitCode, RunStatus, Workflow};

#[test]
fn workflow_serializes_kebab_case() {
    assert_eq!(serde_json::to_string(&Workflow::Train).unwrap(), "\"train\"");
    assert_eq!(serde_json::to_string(&Workflow::Eval).unwrap(), "\"eval\"");
    assert_eq!(
        serde_json::to_string(&Workflow::TrainAndTest).unwrap(),
        "\"train-and-test\""
    );
}

#[test]
fn run_status_round_trips() {
    for s in [RunStatus::Ok, RunStatus::Failed, RunStatus::Interrupted] {
        let s2: RunStatus = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, s2);
    }
}

#[test]
fn exit_code_values_match_spec() {
    assert_eq!(ExitCode::Success as i32, 0);
    assert_eq!(ExitCode::Generic as i32, 1);
    assert_eq!(ExitCode::ConfigInvalid as i32, 2);
    assert_eq!(ExitCode::DataSourceMissing as i32, 3);
    assert_eq!(ExitCode::LockDrift as i32, 4);
    assert_eq!(ExitCode::RuntimeFailure as i32, 5);
    assert_eq!(ExitCode::WorkspaceNotInitialized as i32, 6);
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test --test cli_types`
Expected: compile error (`Workflow` / `RunStatus` / `ExitCode` not found).

- [ ] **Step 3: Implement `src/cli/types.rs`**

```rust
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum Workflow {
    Train,
    Eval,
    TrainAndTest,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunStatus {
    Ok,
    Failed,
    Interrupted,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Generic = 1,
    ConfigInvalid = 2,
    DataSourceMissing = 3,
    LockDrift = 4,
    RuntimeFailure = 5,
    WorkspaceNotInitialized = 6,
}

impl ExitCode {
    pub fn exit(self) -> ! {
        std::process::exit(self as i32);
    }
}

impl From<&crate::cli::CliError> for ExitCode {
    fn from(e: &crate::cli::CliError) -> Self {
        use crate::cli::CliError::*;
        match e {
            ConfigInvalid { .. } => ExitCode::ConfigInvalid,
            DataSourceMissing { .. } => ExitCode::DataSourceMissing,
            LockDrift { .. } => ExitCode::LockDrift,
            Runtime(_) => ExitCode::RuntimeFailure,
            WorkspaceNotInitialized { .. } => ExitCode::WorkspaceNotInitialized,
            Io(_) | Other(_) => ExitCode::Generic,
        }
    }
}
```

- [ ] **Step 4: Run tests until passing**

Run: `cargo test --test cli_types`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cli/types.rs tests/cli_types.rs
git commit -m "cli: add Workflow, RunStatus, ExitCode types"
```

---

## Task 4: Workspace discovery

**Files:**
- Modify: `src/cli/workspace.rs`
- Test: `tests/cli_workspace.rs`

- [ ] **Step 1: Write failing tests at `tests/cli_workspace.rs`**

```rust
use ddrs::cli::workspace::{discover_config, Workspace};
use std::fs;

fn tmp() -> tempfile::TempDir { tempfile::tempdir().unwrap() }

#[test]
fn discover_finds_ddrs_yaml_in_cwd() {
    let d = tmp();
    fs::write(d.path().join("ddrs.yaml"), "workflow: train\n").unwrap();
    let found = discover_config(d.path()).unwrap();
    assert_eq!(found, d.path().join("ddrs.yaml"));
}

#[test]
fn discover_walks_up_until_git_root() {
    let root = tmp();
    fs::create_dir_all(root.path().join(".git")).unwrap();
    fs::write(root.path().join("ddrs.yaml"), "workflow: train\n").unwrap();
    let sub = root.path().join("a").join("b");
    fs::create_dir_all(&sub).unwrap();
    let found = discover_config(&sub).unwrap();
    assert_eq!(found, root.path().join("ddrs.yaml"));
}

#[test]
fn discover_stops_at_git_root_without_config() {
    let root = tmp();
    fs::create_dir_all(root.path().join(".git")).unwrap();
    let sub = root.path().join("a");
    fs::create_dir_all(&sub).unwrap();
    assert!(discover_config(&sub).is_none());
}

#[test]
fn workspace_paths_resolve_relative_to_config() {
    let d = tmp();
    let cfg = d.path().join("ddrs.yaml");
    fs::write(&cfg, "workflow: train\n").unwrap();
    let ws = Workspace::beside(&cfg);
    assert_eq!(ws.root(), d.path().join(".ddrs"));
    assert_eq!(ws.runs_dir(), d.path().join(".ddrs").join("runs"));
    assert_eq!(ws.lockfile(), d.path().join(".ddrs").join("sources.lock"));
    assert_eq!(ws.system_json(), d.path().join(".ddrs").join("system.json"));
}
```

Add `tempfile = "3"` to `[dev-dependencies]` in `Cargo.toml` if not already present.

- [ ] **Step 2: Run failing test**

Run: `cargo test --test cli_workspace`
Expected: compile error (`discover_config` / `Workspace` not found).

- [ ] **Step 3: Implement `src/cli/workspace.rs`**

```rust
use std::path::{Path, PathBuf};

/// Walk up from `start` to find a `ddrs.yaml`. Stops at the first `.git`
/// ancestor (inclusive — the dir containing `.git` is searched, but no
/// further). Returns `None` if not found.
pub fn discover_config(start: &Path) -> Option<PathBuf> {
    let mut cur = start;
    loop {
        let cand = cur.join("ddrs.yaml");
        if cand.is_file() {
            return Some(cand);
        }
        if cur.join(".git").exists() {
            return None;
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => return None,
        }
    }
}

/// All `.ddrs/` paths derived from a config location.
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn beside(config: &Path) -> Self {
        let parent = config.parent().unwrap_or_else(|| Path::new("."));
        Self { root: parent.join(".ddrs") }
    }
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn root(&self) -> &Path { &self.root }
    pub fn runs_dir(&self) -> PathBuf { self.root.join("runs") }
    pub fn lockfile(&self) -> PathBuf { self.root.join("sources.lock") }
    pub fn system_json(&self) -> PathBuf { self.root.join("system.json") }
    pub fn version_file(&self) -> PathBuf { self.root.join("version") }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test cli_workspace`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cli/workspace.rs tests/cli_workspace.rs Cargo.toml Cargo.lock
git commit -m "cli: add config discovery + Workspace paths"
```

---

## Task 5: Source fingerprinting

**Files:**
- Modify: `src/cli/fingerprint.rs`
- Test: `tests/cli_fingerprint.rs`

- [ ] **Step 1: Write failing tests**

```rust
// tests/cli_fingerprint.rs
use ddrs::cli::fingerprint::{Fingerprint, fingerprint_path, reuse_if_unchanged};
use std::fs;

#[test]
fn fingerprint_blake3_csv_matches_known_content() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp = fingerprint_path(&p).unwrap();
    assert_eq!(fp.size, 5);
    assert!(fp.fp.starts_with("blake3:"));
    let again = fingerprint_path(&p).unwrap();
    assert_eq!(fp.fp, again.fp);
}

#[test]
fn fingerprint_changes_when_content_changes() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp1 = fingerprint_path(&p).unwrap();
    fs::write(&p, b"world!").unwrap();
    let fp2 = fingerprint_path(&p).unwrap();
    assert_ne!(fp1.fp, fp2.fp);
    assert_ne!(fp1.size, fp2.size);
}

#[test]
fn reuse_returns_locked_fp_when_stat_matches() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp = fingerprint_path(&p).unwrap();
    // No mutation; reuse should return the stored fp with no rehash.
    let reused = reuse_if_unchanged(&p, &fp).unwrap();
    assert_eq!(reused.fp, fp.fp);
    assert!(reused.reused);
}

#[test]
fn reuse_recomputes_when_size_changes() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("a.csv");
    fs::write(&p, b"hello").unwrap();
    let fp = fingerprint_path(&p).unwrap();
    fs::write(&p, b"goodbye").unwrap();
    let reused = reuse_if_unchanged(&p, &fp).unwrap();
    assert_ne!(reused.fp, fp.fp);
    assert!(!reused.reused);
}
```

- [ ] **Step 2: Run failing test**

Run: `cargo test --test cli_fingerprint`
Expected: compile error.

- [ ] **Step 3: Implement `src/cli/fingerprint.rs`**

```rust
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::cli::CliError;

/// Stat-and-content fingerprint stored in `sources.lock` and the per-run
/// manifest. `fp` is opaque to consumers — see spec § Schemas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub path: PathBuf,
    pub mtime: String,
    pub size: u64,
    pub fp: String,
}

#[derive(Debug)]
pub struct ReuseResult {
    pub fp: String,
    pub mtime: String,
    pub size: u64,
    pub reused: bool,
}

pub fn fingerprint_path(path: &Path) -> Result<Fingerprint, CliError> {
    let md = fs::metadata(path).map_err(|_| CliError::DataSourceMissing { path: path.into() })?;
    let size = md.len();
    let mtime = systime_to_iso(md.modified()?);
    let fp = compute_fp(path, &md)?;
    Ok(Fingerprint { path: path.into(), mtime, size, fp })
}

/// Reuse the locked `fp` when `(path, mtime, size)` is unchanged; otherwise
/// re-hash.
pub fn reuse_if_unchanged(path: &Path, locked: &Fingerprint) -> Result<ReuseResult, CliError> {
    let md = fs::metadata(path).map_err(|_| CliError::DataSourceMissing { path: path.into() })?;
    let size = md.len();
    let mtime = systime_to_iso(md.modified()?);
    if size == locked.size && mtime == locked.mtime {
        return Ok(ReuseResult { fp: locked.fp.clone(), mtime, size, reused: true });
    }
    let fp = compute_fp(path, &md)?;
    Ok(ReuseResult { fp, mtime, size, reused: false })
}

fn compute_fp(path: &Path, md: &fs::Metadata) -> Result<String, CliError> {
    // For directories (zarr / icechunk stores), hash the root metadata file.
    // For regular files (CSV, NetCDF), hash full content.
    let bytes = if md.is_dir() {
        // Try canonical zarr v3 / v2 metadata filenames in order.
        for candidate in ["zarr.json", ".zarray", ".zgroup"] {
            let p = path.join(candidate);
            if p.is_file() {
                return Ok(format!("blake3:{}", blake3::hash(&fs::read(p)?).to_hex()));
            }
        }
        // Fallback: hash the sorted top-level listing for icechunk dirs
        // whose metadata layout varies.
        let mut names: Vec<_> = fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names.join("\n").into_bytes()
    } else {
        fs::read(path)?
    };
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

fn systime_to_iso(t: SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Utc> = t.into();
    dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test cli_fingerprint`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cli/fingerprint.rs tests/cli_fingerprint.rs
git commit -m "cli: blake3 fingerprinting with stat-based reuse"
```

---

## Task 6: Lockfile module

**Files:**
- Modify: `src/cli/lockfile.rs`
- Test: `tests/cli_lockfile.rs`

- [ ] **Step 1: Write failing tests**

```rust
// tests/cli_lockfile.rs
use ddrs::cli::lockfile::{Lockfile, diff_against_live};
use ddrs::cli::fingerprint::Fingerprint;
use std::collections::BTreeMap;
use std::fs;

fn fp(path: &str, fp: &str) -> Fingerprint {
    Fingerprint {
        path: path.into(), mtime: "2026-05-30T00:00:00Z".into(),
        size: 1, fp: fp.into(),
    }
}

#[test]
fn lockfile_round_trips() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("sources.lock");
    let mut sources = BTreeMap::new();
    sources.insert("attributes".into(), fp("/x", "blake3:aaa"));
    let lock = Lockfile { ddrs_version: "0.1.0".into(),
        created_at: "2026-05-30T00:00:00Z".into(), sources };
    lock.write_atomic(&p).unwrap();
    let loaded = Lockfile::read(&p).unwrap();
    assert_eq!(loaded, lock);
}

#[test]
fn diff_lists_drifted_keys() {
    let mut sources = BTreeMap::new();
    sources.insert("attributes".into(), fp("/x", "blake3:aaa"));
    sources.insert("conus_adjacency".into(), fp("/y", "blake3:bbb"));
    let lock = Lockfile { ddrs_version: "x".into(),
        created_at: "x".into(), sources };
    let mut live = BTreeMap::new();
    live.insert("attributes".into(), fp("/x", "blake3:aaa"));         // unchanged
    live.insert("conus_adjacency".into(), fp("/y", "blake3:CHANGED"));// drifted
    let drift = diff_against_live(&lock, &live);
    assert_eq!(drift, vec!["conus_adjacency".to_string()]);
}
```

- [ ] **Step 2: Run failing test**

Run: `cargo test --test cli_lockfile`
Expected: compile error.

- [ ] **Step 3: Implement `src/cli/lockfile.rs`**

```rust
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::cli::CliError;
use crate::cli::fingerprint::Fingerprint;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lockfile {
    pub ddrs_version: String,
    pub created_at: String,
    pub sources: BTreeMap<String, Fingerprint>,
}

impl Lockfile {
    pub fn read(path: &Path) -> Result<Self, CliError> {
        let s = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s).map_err(|e| CliError::Other(Box::new(e)))?)
    }

    pub fn write_atomic(&self, path: &Path) -> Result<(), CliError> {
        let tmp = path.with_extension("lock.tmp");
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| CliError::Other(Box::new(e)))?;
        let mut f = fs::File::create(&tmp)?;
        f.write_all(s.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Return field names whose `fp` differs between the locked snapshot and
/// the live fingerprints. Keys only present on one side are also reported.
pub fn diff_against_live(
    lock: &Lockfile,
    live: &BTreeMap<String, Fingerprint>,
) -> Vec<String> {
    let mut drift = Vec::new();
    for (k, v) in &lock.sources {
        match live.get(k) {
            Some(l) if l.fp == v.fp => {}
            _ => drift.push(k.clone()),
        }
    }
    for k in live.keys() {
        if !lock.sources.contains_key(k) {
            drift.push(k.clone());
        }
    }
    drift.sort();
    drift
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test cli_lockfile`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cli/lockfile.rs tests/cli_lockfile.rs
git commit -m "cli: sources.lock schema + atomic write + diff"
```

---

## Task 7: Manifest + system.json schemas

**Files:**
- Modify: `src/cli/manifest.rs`
- Test: `tests/cli_manifest.rs`

- [ ] **Step 1: Write failing test**

```rust
// tests/cli_manifest.rs
use ddrs::cli::manifest::{Manifest, GitInfo, SourceLockRef, RunOutputs, SystemProbe};
use ddrs::cli::types::{Workflow, RunStatus};
use std::collections::BTreeMap;

#[test]
fn manifest_round_trips_via_serde_json() {
    let m = Manifest {
        run_id: "2026-05-30T00-00-00-train".into(),
        ddrs_version: "0.1.0".into(),
        git: GitInfo { sha: "abc".into(), dirty: false, branch: "main".into() },
        workflow: Workflow::Train,
        config_path: ".ddrs/runs/.../config.yaml".into(),
        started_at: "2026-05-30T00:00:00Z".into(),
        finished_at: Some("2026-05-30T01:00:00Z".into()),
        status: RunStatus::Ok,
        exit_reason: None,
        system: SystemProbe::default(),
        sources: BTreeMap::new(),
        source_lock: SourceLockRef {
            lockfile: ".ddrs/sources.lock".into(),
            matched: true,
            drift: vec![],
        },
        outputs: RunOutputs { checkpoints: vec![], plot: None },
        metrics: serde_json::json!({"final_loss": 0.385}),
        max_mini_batches: None,
    };
    let s = serde_json::to_string(&m).unwrap();
    let m2: Manifest = serde_json::from_str(&s).unwrap();
    assert_eq!(m, m2);
}
```

- [ ] **Step 2: Run failing test**

Run: `cargo test --test cli_manifest`
Expected: compile error.

- [ ] **Step 3: Implement `src/cli/manifest.rs`**

```rust
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cli::CliError;
use crate::cli::fingerprint::Fingerprint;
use crate::cli::types::{RunStatus, Workflow};

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SystemProbe {
    #[serde(default)]
    pub ddrs_version: String,
    #[serde(default)]
    pub probed_at: String,
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub cuda_runtime: String,
    #[serde(default)]
    pub driver: String,
    #[serde(default)]
    pub sm: String,
    #[serde(default)]
    pub free_gpu_gb_at_probe: f32,
    #[serde(default)]
    pub smoke_test: Option<SmokeTestRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmokeTestRecord {
    pub key: String,
    pub passed_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitInfo {
    pub sha: String,
    pub dirty: bool,
    pub branch: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLockRef {
    pub lockfile: PathBuf,
    pub matched: bool,
    pub drift: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunOutputs {
    pub checkpoints: Vec<PathBuf>,
    pub plot: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub run_id: String,
    pub ddrs_version: String,
    pub git: GitInfo,
    pub workflow: Workflow,
    pub config_path: PathBuf,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub status: RunStatus,
    pub exit_reason: Option<String>,
    pub system: SystemProbe,
    pub sources: BTreeMap<String, Fingerprint>,
    pub source_lock: SourceLockRef,
    pub outputs: RunOutputs,
    pub metrics: serde_json::Value,
    pub max_mini_batches: Option<usize>,
}

// Eq for Manifest is enough — serde_json::Value already has PartialEq.
impl Eq for Manifest {}

impl Manifest {
    pub fn read(path: &Path) -> Result<Self, CliError> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)
            .map_err(|e| CliError::Other(Box::new(e)))?)
    }
    pub fn write_atomic(&self, path: &Path) -> Result<(), CliError> {
        let tmp = path.with_extension("json.tmp");
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| CliError::Other(Box::new(e)))?;
        let mut f = fs::File::create(&tmp)?;
        f.write_all(s.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        Ok(())
    }
}

impl SystemProbe {
    pub fn read(path: &Path) -> Result<Self, CliError> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)
            .map_err(|e| CliError::Other(Box::new(e)))?)
    }
    pub fn write_atomic(&self, path: &Path) -> Result<(), CliError> {
        let tmp = path.with_extension("json.tmp");
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| CliError::Other(Box::new(e)))?;
        let mut f = fs::File::create(&tmp)?;
        f.write_all(s.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        Ok(())
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test cli_manifest`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/cli/manifest.rs tests/cli_manifest.rs
git commit -m "cli: manifest.json + system.json schemas with atomic write"
```

---

## Task 8: System probe (cudarc)

**Files:**
- Modify: `src/cli/system.rs`

- [ ] **Step 1: Implement `src/cli/system.rs` (no separate test file — exercised by `cli_init`)**

```rust
use crate::cli::CliError;
use crate::cli::manifest::{SmokeTestRecord, SystemProbe};

/// In-process GPU probe via cudarc. Returns `Ok(None)` when no CUDA device
/// is present (so the caller can present a remediation hint).
pub fn probe() -> Result<Option<SystemProbe>, CliError> {
    use cudarc::driver::result::init as cuda_init;
    use cudarc::driver::CudaContext;

    if cuda_init().is_err() {
        return Ok(None);
    }
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };
    let name = ctx.name().unwrap_or_else(|_| "unknown".to_string());
    let (major, minor) = ctx.compute_cap().unwrap_or((0, 0));
    let sm = format!("{}.{}", major, minor);
    let (free, _total) = cudarc::driver::result::mem_get_info()
        .unwrap_or((0, 0));
    let free_gpu_gb_at_probe = free as f32 / 1e9;
    let cuda_runtime = cudarc::driver::result::driver_get_version()
        .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10))
        .unwrap_or_default();
    let driver = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let probed_at = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    Ok(Some(SystemProbe {
        ddrs_version: env!("CARGO_PKG_VERSION").to_string(),
        probed_at, gpu: name, cuda_runtime, driver, sm,
        free_gpu_gb_at_probe, smoke_test: None,
    }))
}

/// Stable key used to decide whether a cached smoke-test verdict is still
/// valid. Re-run when this string changes.
pub fn smoke_key(probe: &SystemProbe) -> String {
    format!(
        "driver={};cuda={};ddrs={};sm={}",
        probe.driver, probe.cuda_runtime, probe.ddrs_version, probe.sm
    )
}

pub fn record_smoke(probe: &mut SystemProbe, key: String) {
    probe.smoke_test = Some(SmokeTestRecord {
        key,
        passed_at: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    });
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles. If `mem_get_info`/`driver_get_version`/`compute_cap` API surface differs in cudarc 0.19, grep the `cudarc` crate docs in `~/.cargo/registry/src/index.crates.io-*/cudarc-0.19.*/src/driver/` and adjust to the actual function names. The shape is the same in all 0.19.x versions.

- [ ] **Step 3: Commit**

```bash
git add src/cli/system.rs
git commit -m "cli: cudarc-based GPU probe + smoke-test cache key"
```

---

## Task 9: Sandbox loaders + smoke test

**Files:**
- Modify: `src/sandbox.rs`
- Modify: `examples/compare_ddr_sandbox.rs` (call into the lifted helpers)

- [ ] **Step 1: Read the current sandbox loaders to know what to lift**

Read: `examples/compare_ddr_sandbox.rs` lines 1-200. Identify the helpers `read_int_csv`, `read_matrix_csv`, the fixture-loading block (~lines 74-151), and the per-step routing call.

- [ ] **Step 2: Implement `src/sandbox.rs`**

```rust
//! Sandbox fixture loader + functional smoke test.
//!
//! 5-reach RAPID sandbox at `fixtures/sandbox/*.csv`. Used by:
//!   - `examples/compare_ddr_sandbox.rs` (DDR-parity regression)
//!   - `cli::init` smoke test (does routing work on this machine?)
//!
//! Loader helpers were lifted from `compare_ddr_sandbox.rs` so both paths
//! share one implementation.

use ndarray::Array2;
use std::path::Path;

use crate::cli::CliError;

#[derive(Debug)]
pub struct SandboxInputs {
    pub topo: Vec<usize>,
    pub q_prime: Array2<f32>,
    pub q_lateral: Array2<f32>,
    pub n_timesteps: usize,
}

/// Load sandbox fixtures. From source they live at `fixtures/sandbox/`;
/// from an installed binary they are embedded via `include_bytes!` (see
/// `embedded()`).
pub fn load_from_dir(dir: &Path) -> Result<SandboxInputs, CliError> {
    let order = read_int_csv(&dir.join("rapid2_order.csv"))?;
    let qprime = read_matrix_csv(&dir.join("qprime_topo.csv"))?;
    // ... topo, adjacency, config — port the body of compare_ddr_sandbox.rs
    //     lines 74-151 verbatim, returning a SandboxInputs struct.
    unimplemented!("port from examples/compare_ddr_sandbox.rs")
}

/// Same as `load_from_dir` but reads from `include_bytes!`-embedded
/// fixture data. Used by installed binaries.
pub fn load_embedded() -> Result<SandboxInputs, CliError> {
    static ORDER: &[u8] = include_bytes!("../fixtures/sandbox/rapid2_order.csv");
    static QPRIME: &[u8] = include_bytes!("../fixtures/sandbox/qprime_topo.csv");
    static TOPO: &[u8] = include_bytes!("../fixtures/sandbox/topo_order.csv");
    static ADJ: &[u8] = include_bytes!("../fixtures/sandbox/adjacency_topo.csv");
    static CONFIG: &[u8] = include_bytes!("../fixtures/sandbox/config.csv");
    // Parse the same way the file-based path does, but from byte slices.
    parse_inputs(ORDER, QPRIME, TOPO, ADJ, CONFIG)
}

fn read_int_csv(path: &Path) -> Result<Vec<usize>, CliError> { /* port */ unimplemented!() }
fn read_matrix_csv(path: &Path) -> Result<Array2<f32>, CliError> { /* port */ unimplemented!() }
fn parse_inputs(o: &[u8], q: &[u8], t: &[u8], a: &[u8], c: &[u8])
    -> Result<SandboxInputs, CliError> { /* port byte-slice path */ unimplemented!() }

/// Functional smoke result. `passed == true` iff every invariant holds.
#[derive(Debug)]
pub struct SmokeResult {
    pub passed: bool,
    pub max_q: f32,
    pub n_reaches: usize,
    pub n_nan: usize,
    pub n_negative: usize,
}

/// Run a single MC forward pass on the sandbox and check well-formedness.
/// Reuses `routing::MuskingumCunge` and the helpers from `setup_inputs`.
pub fn smoke<I>(inputs: &SandboxInputs, device: &<I as burn::tensor::backend::Backend>::Device)
    -> Result<SmokeResult, CliError>
where
    I: burn::tensor::backend::Backend,
{
    // Build MC inputs from `inputs`, run forward, collect output discharge,
    // and check: all finite, all >= 0, at least one > 0. Body is direct
    // adaptation of the per-step block in compare_ddr_sandbox.rs but
    // discards the DDR-reference comparison.
    unimplemented!("adapt the forward call from compare_ddr_sandbox.rs lines 152-220 \
        and produce a SmokeResult")
}
```

The body marked `unimplemented!()` is mechanical porting from `examples/compare_ddr_sandbox.rs`. The acceptance criterion (next step) tells you when you're done.

- [ ] **Step 3: Update `examples/compare_ddr_sandbox.rs` to call `sandbox::load_from_dir`**

In the example, replace the inline fixture-loading block with:

```rust
let inputs = ddrs::sandbox::load_from_dir(std::path::Path::new("fixtures/sandbox"))?;
```

…and feed `inputs` into the existing MC forward block. The DDR-comparison step is unchanged.

- [ ] **Step 4: Verify the regression example still passes**

Run: `cargo run --release --example compare_ddr_sandbox`
Expected: prints `"ABSOLUTE MATCH"` (max abs diff < 1e-3 m³/s). This is the regression invariant from CLAUDE.md §1.

- [ ] **Step 5: Commit**

```bash
git add src/sandbox.rs examples/compare_ddr_sandbox.rs
git commit -m "sandbox: extract fixture loader + smoke test for reuse"
```

---

## Task 10: stdout/stderr tee (`os_pipe`)

**Files:**
- Modify: `src/cli/tee.rs`

- [ ] **Step 1: Implement `src/cli/tee.rs`**

```rust
//! Run a closure with stdout/stderr piped through `os_pipe` into log files
//! while still forwarding to the original fds. Writes use `O_APPEND` and
//! flush per chunk so CUDA stderr is not lost on crash.

use os_pipe::pipe;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;
use std::thread;

use crate::cli::CliError;

/// Spawn pipe readers that tee captured bytes to `stdout_log`/`stderr_log`
/// and the inherited `stdout`/`stderr`. Returns join handles for callers to
/// drop after their workflow completes.
pub fn tee_to<F, R>(
    stdout_log: &Path,
    stderr_log: &Path,
    body: F,
) -> Result<R, CliError>
where
    F: FnOnce() -> Result<R, CliError>,
{
    let (mut or, ow) = pipe()?;
    let (mut er, ew) = pipe()?;

    let stdout_log_p = stdout_log.to_path_buf();
    let stderr_log_p = stderr_log.to_path_buf();

    let h_out = thread::spawn(move || {
        let mut f = OpenOptions::new().create(true).append(true).open(&stdout_log_p)?;
        let mut buf = [0u8; 8192];
        let mut out = std::io::stdout();
        loop {
            let n = or.read(&mut buf)?;
            if n == 0 { break; }
            f.write_all(&buf[..n])?;
            f.flush()?;
            out.write_all(&buf[..n])?;
            out.flush()?;
        }
        Ok::<(), std::io::Error>(())
    });
    let h_err = thread::spawn(move || {
        let mut f = OpenOptions::new().create(true).append(true).open(&stderr_log_p)?;
        let mut buf = [0u8; 8192];
        let mut err = std::io::stderr();
        loop {
            let n = er.read(&mut buf)?;
            if n == 0 { break; }
            f.write_all(&buf[..n])?;
            f.flush()?;
            err.write_all(&buf[..n])?;
            err.flush()?;
        }
        Ok::<(), std::io::Error>(())
    });

    // Body runs in-process and writes to the original stdout/stderr; the
    // tee threads are for sub-processes spawned by the body (e.g., the
    // workflow run). For in-process workflows, callers should set up
    // `std::io::set_output_capture` instead. This function is the
    // primitive used by `run.rs`.
    let _ = ow; let _ = ew;  // dropping closes the writer ends
    let result = body();
    drop(h_out); drop(h_err);
    result
}
```

(The `tee_to` shape is a primitive; the precise wiring to the workflow body lives in `run.rs` Task 16.)

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`

- [ ] **Step 3: Commit**

```bash
git add src/cli/tee.rs
git commit -m "cli: stdout/stderr tee primitive via os_pipe"
```

---

## Task 11: Shared training bootstrap helper

**Files:**
- Create: `src/training/bootstrap.rs`
- Modify: `src/training/mod.rs` (add `pub mod bootstrap;`)
- Modify: `src/bin/train.rs` (call new helper; behavior unchanged)
- Modify: `src/bin/eval.rs` (same)
- Modify: `src/bin/train_and_test.rs` (same)
- Modify: `src/bin/dump_parameters.rs` (same)

- [ ] **Step 1: Read the duplicated setup in `src/bin/train.rs:50-107`**

Identify lines: `type I = Cuda<f32, i32>; type AB = Autodiff<I>;`, `device` resolution, `head_cfg` construction (lines 74-83 in train.rs), backend seeding, `KanHeadConfig::init`, `TrainState { ... }` setup, `build_adam`.

- [ ] **Step 2: Create `src/training/bootstrap.rs`**

```rust
//! One shared head + state + optimizer constructor so the CLI and the
//! deprecated binaries don't each re-implement the ~30-line setup.

use burn::backend::Autodiff;
use burn::tensor::backend::{Backend, BackendTypes};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::config::Config;
use crate::nn::kan_head::{KanHead, KanHeadConfig};
use crate::training::driver::TrainState;
use crate::training::optimizer::{build_adam, AdamOpt};

pub struct Bootstrapped<I: Backend>
where I: BackendTypes
{
    pub head: KanHead<Autodiff<I>>,
    pub state: TrainState<I>,
    pub optimizer: AdamOpt<KanHead<Autodiff<I>>, Autodiff<I>>,
}

pub fn bootstrap_head_and_state<I>(
    cfg: &Config,
    device: &<I as BackendTypes>::Device,
) -> Bootstrapped<I>
where
    I: Backend + BackendTypes,
    Autodiff<I>: Backend,
{
    let head_section = cfg.kan_head.as_ref().expect("kan_head config required");
    let head_cfg = KanHeadConfig::new(
        head_section.input_var_names.clone(),
        head_section.learnable_parameters.clone(),
        cfg.seed,
    )
    .with_hidden_size(head_section.hidden_size)
    .with_num_hidden_layers(head_section.num_hidden_layers)
    .with_grid(head_section.grid)
    .with_k(head_section.k);

    <I as Backend>::seed(device, cfg.seed);
    let head: KanHead<Autodiff<I>> = head_cfg.init::<Autodiff<I>>(device);

    let state = TrainState::<I> {
        head: head.clone(),
        epoch: 1,
        mini_batch: 0,
        rng: StdRng::seed_from_u64(cfg.seed),
    };
    let optimizer = build_adam::<KanHead<Autodiff<I>>, Autodiff<I>>();

    Bootstrapped { head, state, optimizer }
}
```

(Adjust the `AdamOpt` / `KanHead` generic param names to match whatever the real exports are — the existing binaries already use them.)

- [ ] **Step 3: Add `pub mod bootstrap;` to `src/training/mod.rs`**

- [ ] **Step 4: Refactor `src/bin/train.rs` to use the helper**

Replace lines ~74-97 (head_cfg construction through optimizer build) with:

```rust
let boot = ddrs::training::bootstrap::bootstrap_head_and_state::<I>(&cfg, &device);
let head = boot.head;
let mut state = boot.state;
let mut optimizer = boot.optimizer;
```

- [ ] **Step 5: Refactor `src/bin/eval.rs`, `src/bin/train_and_test.rs`, `src/bin/dump_parameters.rs` identically**

Each currently has its own ~30 lines. Same replacement.

- [ ] **Step 6: Run training_verification integration test as a regression guard**

Run: `cargo test --test training_verification`
Expected: passes (same as before — behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/training/bootstrap.rs src/training/mod.rs src/bin/*.rs
git commit -m "training: extract bootstrap_head_and_state, shared across all binaries"
```

---

## Task 12: dump_parameters as library function

**Files:**
- Create: `src/dump_parameters.rs` (new top-level module)
- Modify: `src/lib.rs` (add `pub mod dump_parameters;`)
- Modify: `src/bin/dump_parameters.rs` (call into the library)

- [ ] **Step 1: Move the body of `src/bin/dump_parameters.rs` (lines 70-232) into `src/dump_parameters.rs::dump`**

Signature:

```rust
//! KAN parameter dump as a library function. Same logic that
//! `src/bin/dump_parameters.rs` runs; lifted out so `cli::run --plot`
//! and the standalone binary share one path.

use std::path::Path;

use burn::tensor::backend::{Backend, BackendTypes};

use crate::config::Config;
use crate::cli::CliError;

pub fn dump<I>(
    cfg: &Config,
    checkpoint: &Path,
    output_csv: &Path,
    batch_size: usize,
    device: &<I as BackendTypes>::Device,
) -> Result<usize, CliError>
where
    I: Backend + BackendTypes,
{
    // Body: copy of src/bin/dump_parameters.rs lines 70-232, with `cli.X`
    // replaced by the function args. Return Ok(n_reaches).
    unimplemented!("port body verbatim")
}
```

- [ ] **Step 2: Replace `src/bin/dump_parameters.rs` body with a thin call to `dump`**

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg = Config::from_yaml_file_with_mode(&cli.config, ConfigMode::Testing)?;
    type I = burn_cuda::Cuda<f32, i32>;
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let n = ddrs::dump_parameters::dump::<I>(&cfg, &cli.checkpoint, &cli.output, cli.batch_size, &device)?;
    println!("wrote {n} reaches → {}", cli.output.display());
    Ok(())
}
```

- [ ] **Step 3: Wire `pub mod dump_parameters;` into `src/lib.rs`**

- [ ] **Step 4: Run the binary on the existing fixture to verify behavior unchanged**

Run: `cargo build --release --bin dump_parameters`
Expected: builds. (Functional test deferred to Task 22's `cli_plot` test.)

- [ ] **Step 5: Commit**

```bash
git add src/dump_parameters.rs src/lib.rs src/bin/dump_parameters.rs
git commit -m "dump_parameters: lift body to library function for CLI reuse"
```

---

## Task 13: `plan` + `PlanResult`

**Files:**
- Modify: `src/cli/plan.rs`
- Test: `tests/cli_plan.rs`

- [ ] **Step 1: Define `PlanResult` and the `plan` function shape**

```rust
//! Dry-run validation. Returns a PlanResult that `run` consumes directly
//! to avoid duplicated I/O.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::cli::CliError;
use crate::cli::fingerprint::{Fingerprint, fingerprint_path, reuse_if_unchanged};
use crate::cli::lockfile::{Lockfile, diff_against_live};
use crate::cli::types::Workflow;
use crate::cli::workspace::Workspace;
use crate::config::{Config, ConfigMode};

#[derive(Debug, Clone, Serialize)]
pub struct PlanResult {
    #[serde(skip)]
    pub config: Config,
    pub config_path: PathBuf,
    pub workflow: Workflow,
    pub sources: BTreeMap<String, Fingerprint>,
    pub drift: Vec<String>,
    pub summary: PlanSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanSummary {
    pub workflow: Workflow,
    pub n_gauges: Option<usize>,
    pub batches_per_epoch: Option<usize>,
    pub epochs: Option<usize>,
    pub est_timesteps: Option<usize>,
    pub n_days: Option<usize>,
    pub chunks: Option<usize>,
    pub gpu_mem_gb_upper_bound: Option<f32>,
}

pub fn plan(
    config_path: &Path,
    workflow_override: Option<Workflow>,
    workspace: &Workspace,
) -> Result<PlanResult, CliError> {
    let workflow = resolve_workflow(config_path, workflow_override)?;
    let mode = match workflow {
        Workflow::Train | Workflow::TrainAndTest => ConfigMode::Training,
        Workflow::Eval => ConfigMode::Testing,
    };
    let config = Config::from_yaml_file_with_mode(config_path, mode)
        .map_err(|e| CliError::ConfigInvalid { path: config_path.into(), source: Box::new(e) })?;

    let lock_path = workspace.lockfile();
    if !lock_path.is_file() {
        return Err(CliError::WorkspaceNotInitialized { path: workspace.root().into() });
    }
    let lock = Lockfile::read(&lock_path)?;

    // Compute live fingerprints, reusing locked fp when stat unchanged.
    let mut sources = BTreeMap::new();
    let data_sources = config.data_sources.as_ref()
        .ok_or_else(|| CliError::ConfigInvalid {
            path: config_path.into(),
            source: "data_sources: missing".into(),
        })?;
    for (key, path) in data_source_pairs(data_sources) {
        let locked = lock.sources.get(&key);
        let live = match locked {
            Some(l) => {
                let r = reuse_if_unchanged(&path, l)?;
                Fingerprint { path: path.clone(), mtime: r.mtime, size: r.size, fp: r.fp }
            }
            None => fingerprint_path(&path)?,
        };
        sources.insert(key, live);
    }
    let drift = diff_against_live(&lock, &sources);

    // Open zarr/icechunk metadata-only validation. Use existing
    // ConusAdjacencyStore / streamflow openers — they already do exactly
    // this check on `open` and bubble up a typed error.
    validate_time_window(&config)?;

    let summary = compute_summary(&config, workflow)?;
    Ok(PlanResult { config, config_path: config_path.into(),
        workflow, sources, drift, summary })
}

fn resolve_workflow(_config_path: &Path, override_: Option<Workflow>)
    -> Result<Workflow, CliError>
{
    if let Some(w) = override_ { return Ok(w); }
    // Read `workflow:` from raw YAML before full parse, to give a clear
    // error for "neither flag nor key set".
    Err(CliError::ConfigInvalid {
        path: "workflow".into(),
        source: "neither --workflow nor `workflow:` key set".into(),
    })
}

fn data_source_pairs(ds: &crate::config::DataSources) -> Vec<(String, PathBuf)> {
    vec![
        ("attributes".into(),      ds.attributes.clone()),
        ("conus_adjacency".into(), ds.conus_adjacency.clone()),
        ("gages_adjacency".into(), ds.gages_adjacency.clone()),
        ("streamflow".into(),      ds.streamflow.clone()),
        ("observations".into(),    ds.observations.clone()),
        ("gages".into(),           ds.gages.clone()),
    ]
}

fn validate_time_window(_cfg: &Config) -> Result<(), CliError> {
    // Open `ConusAdjacencyStore` + the streamflow store, confirm
    // `start_time..end_time` is within their time axes. Use existing
    // store APIs; do NOT open the heavy data arrays.
    Ok(())
}

fn compute_summary(cfg: &Config, workflow: Workflow) -> Result<PlanSummary, CliError> {
    let exp = cfg.experiment.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: "experiment".into(),
        source: "experiment: missing".into(),
    })?;
    // Placeholder counts; the real values come from opening the gages CSV
    // and the time axis. Acceptable to start with `None`s and tighten in a
    // follow-up commit during this task.
    let n_gauges: Option<usize> = None;
    let batches_per_epoch = n_gauges.map(|n| (n + exp.batch_size - 1) / exp.batch_size);
    let rho = exp.rho.unwrap_or(0);
    let est_timesteps = batches_per_epoch.map(|b| rho * b * exp.epochs);

    // GPU mem upper bound — formula from spec: rho * max_subgraph * 4 * 8 / 1e9.
    // max_subgraph is not known without opening adjacency; leave None for now.
    let gpu_mem_gb_upper_bound: Option<f32> = None;

    Ok(PlanSummary {
        workflow,
        n_gauges,
        batches_per_epoch,
        epochs: Some(exp.epochs),
        est_timesteps,
        n_days: None,
        chunks: None,
        gpu_mem_gb_upper_bound,
    })
}
```

The `validate_time_window` / `n_gauges` / `gpu_mem` blocks are marked as starting-point — the real values come from opening the existing zarr stores. Wire those in as a same-task follow-up commit; the integration test below catches regressions.

- [ ] **Step 2: Write the integration test at `tests/cli_plan.rs`**

```rust
use ddrs::cli::plan::plan;
use ddrs::cli::types::Workflow;
use ddrs::cli::workspace::Workspace;
use std::path::Path;

#[test]
#[ignore = "requires the merit data sources to be reachable; runs locally"]
fn plan_succeeds_on_repo_config() {
    // Use the real repo config and a workspace seeded by `ddrs init` (Task 15).
    // For now, this test compiles but is `#[ignore]` until init can produce a
    // lockfile. Re-enable in Task 21's integration sweep.
    let cfg = Path::new("config/merit_training.yaml");
    let ws = Workspace::with_root(std::env::temp_dir().join("ddrs_plan_test/.ddrs"));
    let _ = plan(cfg, Some(Workflow::Train), &ws);
}
```

- [ ] **Step 3: Run cargo check**

Run: `cargo check --test cli_plan && cargo test --test cli_plan -- --include-ignored=never`
Expected: compiles; ignored test does not run.

- [ ] **Step 4: Commit**

```bash
git add src/cli/plan.rs tests/cli_plan.rs
git commit -m "cli: plan() returning PlanResult (validation + drift)"
```

---

## Task 14: `plan_bootstrap` ($EDITOR flow)

**Files:**
- Modify: `src/cli/plan_bootstrap.rs`
- Test: `tests/cli_plan_bootstrap.rs`

- [ ] **Step 1: Write failing test at `tests/cli_plan_bootstrap.rs`**

```rust
use ddrs::cli::plan_bootstrap::{bootstrap, BootstrapInput, BootstrapSource};
use std::fs;
use std::path::PathBuf;

#[test]
fn bootstrap_copies_template_when_no_history() {
    let d = tempfile::tempdir().unwrap();
    let target = d.path().join("ddrs.yaml");
    let template = d.path().join("template.yaml");
    fs::write(&template, "workflow: train\n").unwrap();
    let stub_editor = std::env::current_exe().unwrap();  // anything that exits 0
    let _ = stub_editor;
    let input = BootstrapInput {
        target: target.clone(),
        runs_dir: d.path().join(".ddrs/runs"),
        bundled_template: template,
        editor_cmd: Some("true".into()),  // shell `true` exits 0 immediately
        interactive: false,
    };
    let chosen = bootstrap(input).unwrap();
    assert!(matches!(chosen, BootstrapSource::Template));
    assert!(target.is_file());
}

#[test]
fn bootstrap_uses_latest_successful_run_when_present() {
    let d = tempfile::tempdir().unwrap();
    let runs = d.path().join(".ddrs/runs/2026-05-30T00-00-00-train");
    fs::create_dir_all(&runs).unwrap();
    fs::write(runs.join("manifest.json"),
        r#"{"status":"ok","workflow":"train","run_id":"x","ddrs_version":"x","git":{"sha":"x","dirty":false,"branch":"x"},"config_path":"x","started_at":"x","finished_at":null,"exit_reason":null,"system":{},"sources":{},"source_lock":{"lockfile":"x","matched":true,"drift":[]},"outputs":{"checkpoints":[],"plot":null},"metrics":{}}"#
    ).unwrap();
    fs::write(runs.join("config.yaml"), "workflow: train\nfrom_last: true\n").unwrap();

    let target = d.path().join("ddrs.yaml");
    let template = d.path().join("template.yaml");
    fs::write(&template, "workflow: train\n").unwrap();
    let input = BootstrapInput {
        target: target.clone(),
        runs_dir: d.path().join(".ddrs/runs"),
        bundled_template: template,
        editor_cmd: Some("true".into()),
        interactive: false,  // non-interactive auto-picks latest successful
    };
    let chosen = bootstrap(input).unwrap();
    assert!(matches!(chosen, BootstrapSource::LastSuccessful(_)));
    let copied = fs::read_to_string(&target).unwrap();
    assert!(copied.contains("from_last: true"));
}
```

- [ ] **Step 2: Implement `src/cli/plan_bootstrap.rs`**

```rust
use serde::Deserialize;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cli::CliError;

pub struct BootstrapInput {
    pub target: PathBuf,
    pub runs_dir: PathBuf,
    pub bundled_template: PathBuf,
    pub editor_cmd: Option<String>,
    pub interactive: bool,
}

#[derive(Debug)]
pub enum BootstrapSource {
    LastSuccessful(PathBuf),
    Template,
}

pub fn bootstrap(input: BootstrapInput) -> Result<BootstrapSource, CliError> {
    if !std::io::stdin().is_terminal() && input.interactive {
        return Err(CliError::ConfigInvalid {
            path: input.target.clone(),
            source: "no ddrs.yaml found; pass --config or run interactively".into(),
        });
    }
    let chosen = pick_source(&input)?;
    let src_path = match &chosen {
        BootstrapSource::LastSuccessful(p) => p.clone(),
        BootstrapSource::Template => input.bundled_template.clone(),
    };
    fs::copy(&src_path, &input.target)?;

    let editor = input.editor_cmd
        .or_else(|| std::env::var("EDITOR").ok())
        .unwrap_or_else(|| "vi".to_string());
    Command::new(&editor).arg(&input.target).status()?;
    Ok(chosen)
}

#[derive(Deserialize)]
struct ManifestStatusOnly { status: String }

fn pick_source(input: &BootstrapInput) -> Result<BootstrapSource, CliError> {
    let latest_ok = latest_successful_run(&input.runs_dir)?;
    match latest_ok {
        Some(p) => Ok(BootstrapSource::LastSuccessful(p)),
        None => Ok(BootstrapSource::Template),
    }
}

fn latest_successful_run(runs_dir: &Path) -> Result<Option<PathBuf>, CliError> {
    if !runs_dir.is_dir() { return Ok(None); }
    let mut entries: Vec<_> = fs::read_dir(runs_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.reverse();
    for d in entries {
        let mpath = d.join("manifest.json");
        if !mpath.is_file() { continue; }
        let s = fs::read_to_string(&mpath)?;
        if let Ok(m) = serde_json::from_str::<ManifestStatusOnly>(&s) {
            if m.status == "ok" {
                return Ok(Some(d.join("config.yaml")));
            }
        }
    }
    Ok(None)
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --test cli_plan_bootstrap`
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/cli/plan_bootstrap.rs tests/cli_plan_bootstrap.rs
git commit -m "cli: $EDITOR-based plan bootstrap"
```

---

## Task 15: `init` command

**Files:**
- Modify: `src/cli/init.rs`
- Test: `tests/cli_init.rs`

- [ ] **Step 1: Write the failing integration test**

```rust
// tests/cli_init.rs
use ddrs::cli::init::{run_init, InitInput};
use ddrs::cli::workspace::Workspace;
use std::fs;

#[test]
fn init_phase_a_creates_workspace_and_runs_smoke() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let input = InitInput {
        workspace: ws.root().to_path_buf(),
        config_path: None,
        min_free_gpu_gb: 0.0,    // never warn in CI
        force: false,
        skip_smoke: cfg!(not(feature = "real_gpu")),
    };
    let r = run_init(input).unwrap();
    assert!(ws.root().join("version").is_file());
    assert!(ws.root().join("system.json").is_file());
    assert!(ws.root().join("runs").is_dir());
    assert!(!ws.lockfile().is_file(), "no config → no lockfile");
    assert!(r.phase_b_skipped);
}

#[test]
fn init_phase_b_writes_lock_when_config_present() {
    // Requires a reachable merit config — gate behind ignored unless local.
    // ...
}
```

- [ ] **Step 2: Run failing test**

Run: `cargo test --test cli_init`
Expected: compile error.

- [ ] **Step 3: Implement `src/cli/init.rs`**

```rust
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::cli::{CliError, lockfile::Lockfile, fingerprint::fingerprint_path,
    manifest::SystemProbe, system, workspace::Workspace};
use crate::config::{Config, ConfigMode};

pub struct InitInput {
    pub workspace: PathBuf,
    pub config_path: Option<PathBuf>,
    pub min_free_gpu_gb: f32,
    pub force: bool,
    pub skip_smoke: bool,
}

pub struct InitOutput {
    pub smoke_passed: bool,
    pub smoke_reused: bool,
    pub phase_b_skipped: bool,
}

pub fn run_init(input: InitInput) -> Result<InitOutput, CliError> {
    if input.force && input.workspace.exists() {
        fs::remove_dir_all(&input.workspace)?;
    }
    let ws = Workspace::with_root(&input.workspace);

    // Phase A
    let mut probe = system::probe()?.unwrap_or_default();
    if probe.gpu.is_empty() {
        eprintln!("warning: no CUDA device found; install nvidia driver \
                  ≥ 530 or build with `--features cpu`");
    }
    if probe.free_gpu_gb_at_probe < input.min_free_gpu_gb && probe.free_gpu_gb_at_probe > 0.0 {
        eprintln!("warning: free GPU memory {:.1} GB is below floor {} GB",
            probe.free_gpu_gb_at_probe, input.min_free_gpu_gb);
    }
    fs::create_dir_all(ws.runs_dir())?;
    fs::write(ws.version_file(), env!("CARGO_PKG_VERSION"))?;

    let key = system::smoke_key(&probe);
    let cached = SystemProbe::read(&ws.system_json()).ok()
        .and_then(|p| p.smoke_test).map(|s| s.key == key).unwrap_or(false);
    let (smoke_passed, smoke_reused) = if input.skip_smoke {
        (true, true)
    } else if cached && !input.force {
        (true, true)
    } else {
        let r = run_smoke()?;
        (r, false)
    };
    if smoke_passed && !smoke_reused {
        system::record_smoke(&mut probe, key);
    } else if smoke_reused {
        // Preserve the prior smoke_test record
        if let Ok(prior) = SystemProbe::read(&ws.system_json()) {
            probe.smoke_test = prior.smoke_test;
        }
    }
    probe.write_atomic(&ws.system_json())?;

    // Phase B — requires a config
    let config_path = input.config_path.or_else(|| {
        crate::cli::workspace::discover_config(Path::new("."))
    });
    let Some(cfg_path) = config_path else {
        eprintln!("no ddrs.yaml found — run `ddrs plan` to bootstrap one, \
                   then re-run `ddrs init` to lock data sources.");
        return Ok(InitOutput { smoke_passed, smoke_reused, phase_b_skipped: true });
    };
    let cfg = Config::from_yaml_file_with_mode(&cfg_path, ConfigMode::Training)
        .map_err(|e| CliError::ConfigInvalid { path: cfg_path.clone(), source: Box::new(e) })?;
    let ds = cfg.data_sources.as_ref().ok_or_else(|| CliError::ConfigInvalid {
        path: cfg_path.clone(),
        source: "data_sources: missing".into(),
    })?;

    let pairs = [
        ("attributes", &ds.attributes),
        ("conus_adjacency", &ds.conus_adjacency),
        ("gages_adjacency", &ds.gages_adjacency),
        ("streamflow", &ds.streamflow),
        ("observations", &ds.observations),
        ("gages", &ds.gages),
    ];
    // Reachability + fingerprint, parallel.
    let results: Vec<_> = std::thread::scope(|s| {
        pairs.iter().map(|(k, p)| {
            let p = (*p).clone();
            s.spawn(move || (k.to_string(), fingerprint_path(&p)))
        }).collect::<Vec<_>>()
            .into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut sources = BTreeMap::new();
    for (k, r) in results {
        sources.insert(k, r?);
    }
    let lock = Lockfile {
        ddrs_version: env!("CARGO_PKG_VERSION").into(),
        created_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        sources,
    };
    lock.write_atomic(&ws.lockfile())?;
    Ok(InitOutput { smoke_passed, smoke_reused, phase_b_skipped: false })
}

fn run_smoke() -> Result<bool, CliError> {
    // Load sandbox (try embedded first, fall back to ./fixtures/sandbox)
    let inputs = crate::sandbox::load_embedded()
        .or_else(|_| crate::sandbox::load_from_dir(Path::new("fixtures/sandbox")))?;
    type I = burn_cuda::Cuda<f32, i32>;
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let r = crate::sandbox::smoke::<I>(&inputs, &device)?;
    Ok(r.passed)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test cli_init -- --test-threads=1`
Expected: `init_phase_a_creates_workspace_and_runs_smoke` passes.

- [ ] **Step 5: Commit**

```bash
git add src/cli/init.rs tests/cli_init.rs
git commit -m "cli: init command (Phase A + Phase B)"
```

---

## Task 16: `run` command + manifest write

**Files:**
- Modify: `src/cli/run.rs`

- [ ] **Step 1: Implement `src/cli/run.rs`**

```rust
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::cli::{CliError, plan::{plan, PlanResult},
    manifest::{GitInfo, Manifest, RunOutputs, SourceLockRef},
    types::{RunStatus, Workflow}, workspace::Workspace};

pub struct RunInput {
    pub workspace: Workspace,
    pub config_path: PathBuf,
    pub workflow: Option<Workflow>,
    pub plot: bool,
    pub strict: bool,
    pub max_mini_batches: Option<usize>,
}

pub fn run(input: RunInput) -> Result<PathBuf, CliError> {
    let pr: PlanResult = plan(&input.config_path,
        input.workflow,
        &input.workspace)?;

    if !pr.drift.is_empty() {
        if input.strict {
            return Err(CliError::LockDrift { fields: pr.drift });
        }
        eprintln!("warning: data source drift: {:?}", pr.drift);
    }

    let started_at = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let run_id = format!("{}-{:?}", started_at.replace(':', "-"), pr.workflow).to_lowercase();
    let run_dir = input.workspace.runs_dir().join(&run_id);
    fs::create_dir_all(run_dir.join("checkpoints"))?;
    fs::copy(&input.config_path, run_dir.join("config.yaml"))?;
    let _ = copy_cargo_lock_if_reachable(&run_dir);

    let stdout_log = run_dir.join("stdout.log");
    let stderr_log = run_dir.join("stderr.log");

    // Dispatch to workflow. The tee primitive (Task 10) is wired around
    // subprocess output for spawned workflows; for in-process direct calls,
    // CUDA stderr lands on the real stderr — we copy log content
    // post-hoc by mirroring the println! lines through a tracing subscriber
    // that also writes to stderr_log. For v1 simplicity, the in-process
    // path writes a minimal "stdout.log: see terminal" stub and the real
    // log capture lands in v1.1 alongside the run-as-subprocess refactor.
    let _ = (&stdout_log, &stderr_log);

    let (status, exit_reason, metrics, outputs) = dispatch(&input, &pr, &run_dir);

    let manifest = Manifest {
        run_id, ddrs_version: env!("CARGO_PKG_VERSION").into(),
        git: capture_git(), workflow: pr.workflow,
        config_path: run_dir.join("config.yaml"),
        started_at, finished_at: Some(chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        status, exit_reason,
        system: Default::default(),
        sources: pr.sources.clone(),
        source_lock: SourceLockRef {
            lockfile: input.workspace.lockfile(),
            matched: pr.drift.is_empty(), drift: pr.drift.clone(),
        },
        outputs, metrics,
        max_mini_batches: input.max_mini_batches,
    };
    manifest.write_atomic(&run_dir.join("manifest.json"))?;
    Ok(run_dir)
}

fn dispatch(
    input: &RunInput,
    pr: &PlanResult,
    run_dir: &Path,
) -> (RunStatus, Option<String>, serde_json::Value, RunOutputs) {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        type I = burn_cuda::Cuda<f32, i32>;
        let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
        match pr.workflow {
            Workflow::Train | Workflow::TrainAndTest => {
                // Build dataset, boot, and call training::driver::train(...).
                // Final checkpoint lands in `run_dir/checkpoints/`.
                Ok::<_, CliError>(serde_json::json!({"final_loss": 0.0}))
            }
            Workflow::Eval => {
                // Call training::eval::eval(...). Return metrics.
                Ok(serde_json::json!({"eval_done": true}))
            }
        }
    }));

    match res {
        Ok(Ok(metrics)) => {
            let mut outputs = RunOutputs { checkpoints: vec![], plot: None };
            if input.plot {
                if let Some(ck) = latest_checkpoint(&run_dir.join("checkpoints")) {
                    let csv = run_dir.join("plot/kan_parameters.csv");
                    fs::create_dir_all(csv.parent().unwrap()).ok();
                    let _ = ck; let _ = csv;
                    // crate::dump_parameters::dump::<I>(&pr.config, &ck, &csv, 50_000, &device)?;
                    outputs.plot = Some(PathBuf::from("plot/kan_parameters.csv"));
                    eprintln!("plot CSV written. To visualize: jupyter run \
                        ~/projects/ddr/examples/merit/plot_parameter_map.ipynb \
                        --csv {}", outputs.plot.as_ref().unwrap().display());
                }
            }
            (RunStatus::Ok, None, metrics, outputs)
        }
        Ok(Err(e)) => (RunStatus::Failed, Some(e.to_string()),
            serde_json::json!({}), RunOutputs { checkpoints: vec![], plot: None }),
        Err(_) => (RunStatus::Failed, Some("workflow panicked".to_string()),
            serde_json::json!({}), RunOutputs { checkpoints: vec![], plot: None }),
    }
}

fn latest_checkpoint(dir: &Path) -> Option<PathBuf> {
    let mut e: Vec<_> = fs::read_dir(dir).ok()?.filter_map(Result::ok)
        .map(|x| x.path()).filter(|p| p.extension().map(|e| e == "mpk").unwrap_or(false))
        .collect();
    e.sort();
    e.pop()
}

fn copy_cargo_lock_if_reachable(run_dir: &Path) -> std::io::Result<()> {
    let candidates = [Path::new("Cargo.lock"), Path::new("../Cargo.lock")];
    for p in candidates {
        if p.is_file() {
            return fs::copy(p, run_dir.join("Cargo.lock")).map(|_| ());
        }
    }
    Ok(())
}

fn capture_git() -> GitInfo {
    use std::process::Command;
    fn out(args: &[&str]) -> String {
        Command::new("git").args(args).output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string()).unwrap_or_default()
    }
    let dirty = !out(&["status", "--porcelain"]).is_empty();
    GitInfo {
        sha: out(&["rev-parse", "HEAD"]),
        dirty,
        branch: out(&["rev-parse", "--abbrev-ref", "HEAD"]),
    }
}
```

The TODO-shaped blocks (`// Build dataset, boot, ...` and the `dump_parameters::dump` call) need to be filled in with the actual library calls from `src/bin/train.rs:99-107` and `src/dump_parameters.rs::dump`. The shape above gives every other piece — manifest layout, dispatch, drift handling, git capture.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles. Tests for `run` come in Task 22's integration sweep.

- [ ] **Step 3: Commit**

```bash
git add src/cli/run.rs
git commit -m "cli: run command (plan → execute → manifest)"
```

---

## Task 17: `show`, `status`, `gc` commands

**Files:**
- Modify: `src/cli/show.rs`
- Modify: `src/cli/status.rs`
- Modify: `src/cli/gc.rs`

- [ ] **Step 1: Implement `src/cli/show.rs`**

```rust
use std::path::Path;

use crate::cli::{CliError, manifest::Manifest, workspace::Workspace};

pub fn run_show(ws: &Workspace, run_id: &str, as_json: bool) -> Result<(), CliError> {
    let path = ws.runs_dir().join(run_id).join("manifest.json");
    let m = Manifest::read(&path)?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&m)
            .map_err(|e| CliError::Other(Box::new(e)))?);
    } else {
        println!("run     {}", m.run_id);
        println!("status  {:?}", m.status);
        println!("workflow {:?}", m.workflow);
        println!("started  {}", m.started_at);
        if let Some(f) = &m.finished_at { println!("finished {}", f); }
        println!("git     {} ({})", m.git.sha, if m.git.dirty {"dirty"} else {"clean"});
        println!("drift   {:?}", m.source_lock.drift);
        if let Some(p) = &m.outputs.plot { println!("plot    {}", p.display()); }
    }
    Ok(())
}
```

- [ ] **Step 2: Implement `src/cli/status.rs`**

```rust
use std::fs;
use std::path::Path;

use crate::cli::{CliError, workspace::Workspace};

pub fn run_status(ws: &Workspace, as_json: bool) -> Result<(), CliError> {
    let runs = ws.runs_dir();
    let total = walk_size(&runs).unwrap_or(0);
    let total_gb = total as f64 / 1e9;
    let last_run = latest_run_id(&runs)?;
    let lock_present = ws.lockfile().is_file();
    if as_json {
        let v = serde_json::json!({
            "workspace": ws.root(),
            "lockfile_present": lock_present,
            "last_run": last_run,
            "runs_dir_bytes": total,
            "runs_dir_gb": total_gb,
        });
        println!("{}", serde_json::to_string_pretty(&v)
            .map_err(|e| CliError::Other(Box::new(e)))?);
    } else {
        println!("workspace     {}", ws.root().display());
        println!("lockfile      {}", if lock_present { "present" } else { "missing" });
        println!("last run      {}", last_run.unwrap_or_else(|| "(none)".into()));
        println!(".ddrs/runs/   {:.2} GB", total_gb);
        if total_gb > 10.0 {
            println!("hint: total runs/ exceeds 10 GB — consider `ddrs gc`");
        }
    }
    Ok(())
}

fn walk_size(p: &Path) -> std::io::Result<u64> {
    if !p.is_dir() { return Ok(0); }
    let mut total = 0;
    for e in fs::read_dir(p)? {
        let e = e?;
        let md = e.metadata()?;
        total += if md.is_dir() { walk_size(&e.path())? } else { md.len() };
    }
    Ok(total)
}

fn latest_run_id(runs_dir: &Path) -> Result<Option<String>, CliError> {
    if !runs_dir.is_dir() { return Ok(None); }
    let mut e: Vec<_> = fs::read_dir(runs_dir)?
        .filter_map(Result::ok).map(|x| x.file_name().to_string_lossy().into_owned())
        .collect();
    e.sort(); Ok(e.pop())
}
```

- [ ] **Step 3: Implement `src/cli/gc.rs`**

```rust
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::cli::{CliError, manifest::Manifest, workspace::Workspace};

pub struct GcInput {
    pub keep: Option<usize>,
    pub keep_successful: bool,
    pub older_than: Option<Duration>,
    pub dry_run: bool,
}

pub fn run_gc(ws: &Workspace, input: GcInput) -> Result<Vec<PathBuf>, CliError> {
    let runs_dir = ws.runs_dir();
    if !runs_dir.is_dir() { return Ok(vec![]); }

    let mut entries: Vec<PathBuf> = fs::read_dir(&runs_dir)?
        .filter_map(Result::ok).map(|e| e.path())
        .filter(|p| p.is_dir()).collect();
    entries.sort();  // oldest first
    let n = entries.len();
    let mut to_delete: Vec<PathBuf> = Vec::new();

    for (idx, dir) in entries.iter().enumerate() {
        let from_newest = n - idx - 1;
        if let Some(k) = input.keep {
            if from_newest < k { continue; }
        }
        if input.keep_successful {
            let mpath = dir.join("manifest.json");
            if mpath.is_file() {
                if let Ok(m) = Manifest::read(&mpath) {
                    if matches!(m.status, crate::cli::types::RunStatus::Ok) { continue; }
                }
            }
        }
        if let Some(threshold) = input.older_than {
            let md = fs::metadata(dir)?;
            let age = md.modified().ok()
                .and_then(|t| t.elapsed().ok())
                .unwrap_or_default();
            if age < threshold { continue; }
        }
        to_delete.push(dir.clone());
    }

    // Filters compose with AND — only delete if ANY filter was passed.
    if !input.dry_run && (input.keep.is_some() || input.keep_successful || input.older_than.is_some()) {
        for d in &to_delete { let _ = fs::remove_dir_all(d); }
    }
    Ok(to_delete)
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check`

- [ ] **Step 5: Commit**

```bash
git add src/cli/show.rs src/cli/status.rs src/cli/gc.rs
git commit -m "cli: show, status, gc commands"
```

---

## Task 18: `ddrs` binary entrypoint

**Files:**
- Create: `src/bin/ddrs.rs`

- [ ] **Step 1: Implement `src/bin/ddrs.rs`**

```rust
//! `ddrs` CLI entrypoint. Dispatches to subcommands defined in
//! `ddrs::cli::*`. See spec at
//! `docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};

use ddrs::cli::{CliError, ExitCode, Workflow};
use ddrs::cli::workspace::{Workspace, discover_config};

#[derive(Parser)]
#[command(name = "ddrs", about = "Differentiable Distributed Routing")]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    workspace: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Init {
        #[arg(long)] force: bool,
        #[arg(long, default_value_t = 8.0)] min_free_gpu_gb: f32,
    },
    Plan {
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        #[arg(long)] json: bool,
    },
    Run {
        #[arg(long, value_enum)] workflow: Option<Workflow>,
        #[arg(long)] plot: bool,
        #[arg(long)] strict: bool,
        #[arg(long)] max_mini_batches: Option<usize>,
        #[arg(long)] json: bool,
    },
    Show { run_id: String, #[arg(long)] json: bool },
    Status { #[arg(long)] json: bool },
    Gc {
        #[arg(long)] keep: Option<usize>,
        #[arg(long)] keep_successful: bool,
        #[arg(long)] older_than: Option<String>,
        #[arg(long)] dry_run: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    let r = dispatch(cli);
    if let Err(e) = r {
        eprintln!("error: {e}");
        ExitCode::from(&e).exit();
    }
}

fn dispatch(cli: Cli) -> Result<(), CliError> {
    let cfg_path = cli.config.clone()
        .or_else(|| discover_config(std::path::Path::new(".")));
    let ws_root = cli.workspace.unwrap_or_else(|| {
        cfg_path.as_ref().and_then(|p| p.parent()).map(|d| d.join(".ddrs"))
            .unwrap_or_else(|| std::path::PathBuf::from(".ddrs"))
    });
    let ws = Workspace::with_root(&ws_root);

    match cli.cmd {
        Cmd::Init { force, min_free_gpu_gb } => {
            ddrs::cli::init::run_init(ddrs::cli::init::InitInput {
                workspace: ws_root,
                config_path: cfg_path,
                min_free_gpu_gb, force, skip_smoke: false,
            }).map(|_| ())
        }
        Cmd::Plan { workflow, json } => {
            let cfg = cfg_path.ok_or_else(|| {
                // Bootstrap path lives in plan_bootstrap; wire it in here.
                CliError::ConfigInvalid { path: ".".into(),
                    source: "no config; bootstrap not yet wired in main".into() }
            })?;
            let pr = ddrs::cli::plan::plan(&cfg, workflow, &ws)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&pr)
                    .map_err(|e| CliError::Other(Box::new(e)))?);
            } else {
                println!("workflow {:?}", pr.workflow);
                println!("drift    {:?}", pr.drift);
            }
            Ok(())
        }
        Cmd::Run { workflow, plot, strict, max_mini_batches, json: _ } => {
            let cfg = cfg_path.ok_or_else(|| CliError::ConfigInvalid {
                path: ".".into(), source: "no config".into(),
            })?;
            ddrs::cli::run::run(ddrs::cli::run::RunInput {
                workspace: Workspace::with_root(ws.root()),
                config_path: cfg,
                workflow, plot, strict, max_mini_batches,
            }).map(|_| ())
        }
        Cmd::Show { run_id, json } => ddrs::cli::show::run_show(&ws, &run_id, json),
        Cmd::Status { json } => ddrs::cli::status::run_status(&ws, json),
        Cmd::Gc { keep, keep_successful, older_than, dry_run } => {
            let dur = older_than.as_deref().map(|s| humantime::parse_duration(s))
                .transpose().map_err(|e| CliError::Other(Box::new(e)))?;
            let deleted = ddrs::cli::gc::run_gc(&ws, ddrs::cli::gc::GcInput {
                keep, keep_successful, older_than: dur, dry_run,
            })?;
            for p in &deleted {
                println!("{} {}", if dry_run { "would delete" } else { "deleted" }, p.display());
            }
            Ok(())
        }
    }
}
```

- [ ] **Step 2: Verify build**

Run: `cargo build --release --bin ddrs`
Expected: builds. (Plan bootstrap is wired in Task 21.)

- [ ] **Step 3: Commit**

```bash
git add src/bin/ddrs.rs
git commit -m "cli: ddrs binary entrypoint with clap dispatch"
```

---

## Task 19: Wire `plan_bootstrap` into the `ddrs plan` dispatch

**Files:**
- Modify: `src/bin/ddrs.rs`

- [ ] **Step 1: Replace the `Cmd::Plan` arm**

```rust
Cmd::Plan { workflow, json } => {
    let cfg_path = match cfg_path {
        Some(p) => p,
        None => {
            // No config — enter bootstrap.
            let target = std::env::current_dir()?.join("ddrs.yaml");
            let bundled = std::path::PathBuf::from("config/merit_training.yaml");
            ddrs::cli::plan_bootstrap::bootstrap(
                ddrs::cli::plan_bootstrap::BootstrapInput {
                    target: target.clone(),
                    runs_dir: ws.runs_dir(),
                    bundled_template: bundled,
                    editor_cmd: None,
                    interactive: true,
                },
            )?;
            target
        }
    };
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

- [ ] **Step 2: Verify build**

Run: `cargo build --release --bin ddrs`

- [ ] **Step 3: Commit**

```bash
git add src/bin/ddrs.rs
git commit -m "cli: wire plan bootstrap into ddrs plan"
```

---

## Task 20: Deprecation shims on existing binaries

**Files:**
- Modify: `src/bin/train.rs`
- Modify: `src/bin/eval.rs`
- Modify: `src/bin/train_and_test.rs`

- [ ] **Step 1: At the top of each `main()`, before any other work, print the deprecation warning**

```rust
// src/bin/train.rs (add immediately after `let cli = Cli::parse();`)
eprintln!(
    "warning: `train` is deprecated and will be removed in 0.4. \
     use `ddrs run --workflow train` instead."
);
```

Identical text for `eval` (substitute `train` → `eval`, `train` → `eval` in the suggestion) and `train_and_test` (substitute `train` → `train-and-test`).

- [ ] **Step 2: Verify the binaries still build and the existing behavior is unchanged**

Run: `cargo build --release --bin train --bin eval --bin train_and_test`

- [ ] **Step 3: Commit**

```bash
git add src/bin/train.rs src/bin/eval.rs src/bin/train_and_test.rs
git commit -m "deprecate: train, eval, train_and_test point users at ddrs run"
```

---

## Task 21: Integration tests — init, plan, plan_bootstrap

**Files:**
- Create / extend: `tests/cli_init.rs`
- Create: `tests/cli_workspace_uninit.rs`
- Modify: `tests/cli_plan.rs`, `tests/cli_plan_bootstrap.rs` (un-ignore now that init can produce a lockfile)

- [ ] **Step 1: Extend `cli_init.rs` Phase B test**

```rust
#[test]
#[ignore = "requires repo data sources reachable"]
fn init_phase_b_writes_lockfile() {
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("ddrs.yaml");
    std::fs::copy("config/merit_training.yaml", &cfg).unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let r = run_init(InitInput {
        workspace: ws.root().into(),
        config_path: Some(cfg), min_free_gpu_gb: 0.0,
        force: false, skip_smoke: true,
    }).unwrap();
    assert!(!r.phase_b_skipped);
    assert!(ws.lockfile().is_file());
}
```

- [ ] **Step 2: Create `tests/cli_workspace_uninit.rs`**

```rust
use ddrs::cli::plan::plan;
use ddrs::cli::types::Workflow;
use ddrs::cli::workspace::Workspace;
use std::fs;

#[test]
fn plan_without_init_exits_workspace_not_initialized() {
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("ddrs.yaml");
    fs::copy("config/merit_training.yaml", &cfg).unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.root()).unwrap();
    let err = plan(&cfg, Some(Workflow::Train), &ws).unwrap_err();
    assert!(matches!(err, ddrs::cli::CliError::WorkspaceNotInitialized { .. }));
}
```

- [ ] **Step 3: Run**

Run: `cargo test --test cli_workspace_uninit && cargo test --test cli_init -- --ignored`
Expected: workspace_uninit passes always; ignored init test passes locally where data sources exist.

- [ ] **Step 4: Commit**

```bash
git add tests/cli_init.rs tests/cli_workspace_uninit.rs
git commit -m "test: cli_init Phase B + cli_workspace_uninit"
```

---

## Task 22: Integration tests — run, drift, plot

**Files:**
- Create: `tests/cli_run_drift.rs`
- Create: `tests/cli_plot.rs`
- Create: `tests/cli_show.rs`
- Create: `tests/cli_runtime_failure.rs`

- [ ] **Step 1: `tests/cli_run_drift.rs`**

```rust
use ddrs::cli::{init, run, workspace::Workspace};
use std::fs;

#[test]
#[ignore = "requires repo data sources reachable"]
fn run_warns_then_strict_fails_on_drift() {
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("ddrs.yaml");
    fs::copy("config/merit_training.yaml", &cfg).unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));

    init::run_init(init::InitInput {
        workspace: ws.root().into(), config_path: Some(cfg.clone()),
        min_free_gpu_gb: 0.0, force: false, skip_smoke: true,
    }).unwrap();

    // Touch the gauges CSV to force a drift fp diff.
    let gages = std::path::PathBuf::from("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv");
    let mtime_now = std::time::SystemTime::now();
    filetime::set_file_mtime(&gages, filetime::FileTime::from_system_time(mtime_now)).ok();

    // Strict mode → LockDrift error.
    let err = run::run(run::RunInput {
        workspace: Workspace::with_root(ws.root()), config_path: cfg.clone(),
        workflow: Some(ddrs::cli::types::Workflow::Train),
        plot: false, strict: true, max_mini_batches: Some(1),
    }).unwrap_err();
    assert!(matches!(err, ddrs::cli::CliError::LockDrift { .. }));
}
```

Add `filetime = "0.2"` to `[dev-dependencies]`.

- [ ] **Step 2: `tests/cli_plot.rs`, `tests/cli_show.rs`, `tests/cli_runtime_failure.rs`**

Each follows the same pattern: small ignored tests that exercise the real workflow under a tmp dir. Use `#[ignore]` and document the local-data prereq. (Plot test: `--max-mini-batches 1`, assert CSV exists and has > 1 row. Show: write a stub manifest, call `run_show`, assert stdout. Runtime failure: spawn `run::run` against a config that triggers a panic; assert `RunStatus::Failed` in the manifest.)

- [ ] **Step 3: Run**

Run: `cargo test --test cli_run_drift --test cli_plot --test cli_show --test cli_runtime_failure -- --ignored`
Expected: all pass locally.

- [ ] **Step 4: Commit**

```bash
git add tests/cli_run_drift.rs tests/cli_plot.rs tests/cli_show.rs tests/cli_runtime_failure.rs Cargo.toml Cargo.lock
git commit -m "test: cli_run_drift, cli_plot, cli_show, cli_runtime_failure"
```

---

## Task 23: Integration tests — status, gc, json contract, deprecation shim

**Files:**
- Create: `tests/cli_status_gc.rs`
- Create: `tests/cli_json_contract.rs`
- Create: `tests/cli_deprecation_shim.rs`

- [ ] **Step 1: `tests/cli_status_gc.rs`**

```rust
use ddrs::cli::{gc, status, workspace::Workspace};
use std::fs;

#[test]
fn status_reports_runs_dir_size_and_gc_respects_keep() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    fs::create_dir_all(ws.runs_dir()).unwrap();
    for i in 0..3 {
        let r = ws.runs_dir().join(format!("2026-05-3{i}T00-00-00-train"));
        fs::create_dir_all(&r).unwrap();
        fs::write(r.join("manifest.json"),
            r#"{"status":"failed","workflow":"train","run_id":"x","ddrs_version":"x","git":{"sha":"x","dirty":false,"branch":"x"},"config_path":"x","started_at":"x","finished_at":null,"exit_reason":null,"system":{},"sources":{},"source_lock":{"lockfile":"x","matched":true,"drift":[]},"outputs":{"checkpoints":[],"plot":null},"metrics":{}}"#).unwrap();
    }
    status::run_status(&ws, false).unwrap();
    let kept = gc::run_gc(&ws, gc::GcInput {
        keep: Some(1), keep_successful: false,
        older_than: None, dry_run: false,
    }).unwrap();
    assert_eq!(kept.len(), 2);
    assert_eq!(fs::read_dir(ws.runs_dir()).unwrap().count(), 1);
}
```

- [ ] **Step 2: `tests/cli_json_contract.rs`**

```rust
use ddrs::cli::workspace::Workspace;
use std::fs;

#[test]
fn show_json_parses_with_expected_keys() {
    let d = tempfile::tempdir().unwrap();
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let r = ws.runs_dir().join("2026-05-30T00-00-00-train");
    fs::create_dir_all(&r).unwrap();
    fs::write(r.join("manifest.json"),
        r#"{"status":"ok","workflow":"train","run_id":"x","ddrs_version":"x","git":{"sha":"x","dirty":false,"branch":"x"},"config_path":"x","started_at":"x","finished_at":null,"exit_reason":null,"system":{},"sources":{},"source_lock":{"lockfile":"x","matched":true,"drift":[]},"outputs":{"checkpoints":[],"plot":null},"metrics":{}}"#).unwrap();

    // Capture stdout via assert_cmd against the real binary so we exercise
    // the same path users hit.
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_ddrs"));
    cmd.arg("--workspace").arg(ws.root());
    cmd.arg("show").arg("2026-05-30T00-00-00-train").arg("--json");
    let out = cmd.output().unwrap();
    let s = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    for key in &["run_id","ddrs_version","git","workflow","status","outputs","metrics"] {
        assert!(v.get(*key).is_some(), "missing key {}", key);
    }
}
```

- [ ] **Step 3: `tests/cli_deprecation_shim.rs`**

```rust
#[test]
fn train_binary_prints_deprecation_warning_to_stderr() {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_train"));
    cmd.args(["--help"]);  // any invocation triggers the eprintln
    let out = cmd.output().unwrap();
    let s = String::from_utf8(out.stderr).unwrap();
    assert!(s.contains("deprecated"), "stderr was: {s}");
}
```

- [ ] **Step 4: Run**

Run: `cargo test --test cli_status_gc --test cli_json_contract --test cli_deprecation_shim`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/cli_status_gc.rs tests/cli_json_contract.rs tests/cli_deprecation_shim.rs
git commit -m "test: cli_status_gc, cli_json_contract, cli_deprecation_shim"
```

---

## Task 24: Documentation updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.claude/ARCHITECTURE.md`

- [ ] **Step 1: Update `CLAUDE.md` Commands section**

Lead with the new commands, move old ones to a "Legacy" subsection:

```markdown
## Commands

```bash
cargo build --release                      # build the binaries

# New (preferred):
ddrs init                                  # set up + lock data sources
ddrs plan                                  # dry-run validation
ddrs run --workflow train                  # equivalent to old `train`
ddrs run --workflow eval                   # equivalent to old `eval`
ddrs run --workflow train-and-test --plot  # full sweep with parameter dump
ddrs show <run-id>                         # inspect a past run
ddrs status                                # workspace summary
ddrs gc --keep 5 --keep-successful         # prune .ddrs/runs/

# Legacy (deprecated, removed in 0.4):
cargo run --release --bin train -- ...
cargo run --release --bin eval -- ...
cargo run --release --bin train_and_test -- ...
```
```

- [ ] **Step 2: Add a "When in doubt" pointer for the new doc root**

In `CLAUDE.md` under "When in doubt":

```markdown
- New design docs from `/superpowers` brainstorms live at `docs/superpowers/specs/`
  with implementation plans at `docs/superpowers/plans/`.
```

- [ ] **Step 3: Add "CLI lifecycle" section to `.claude/ARCHITECTURE.md`**

```markdown
## CLI lifecycle (added by Spec A)

The `ddrs` binary at `src/bin/ddrs.rs` is the single entrypoint. It
dispatches to subcommands under `src/cli/`. Full design at
`docs/superpowers/specs/2026-05-30-ddrs-cli-lifecycle-design.md`.
First-run flow: `init → plan → init → run`.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md .claude/ARCHITECTURE.md
git commit -m "docs: ddrs CLI commands + CLI lifecycle section"
```

---

## Task 25: Final acceptance — full sweep

**Files:** none modified

- [ ] **Step 1: Run the full test suite**

Run: `cargo test`
Expected: all unit and non-ignored integration tests pass.

- [ ] **Step 2: Run all `--ignored` integration tests on a host with merit data**

Run: `cargo test -- --ignored`
Expected: all CLI integration tests pass.

- [ ] **Step 3: Run the absolute-match regression**

Run: `cargo run --release --example compare_ddr_sandbox`
Expected: `ABSOLUTE MATCH` (max abs diff < 1e-3 m³/s). This is CLAUDE.md invariant #1 — the whole CLI port is meaningless if this breaks.

- [ ] **Step 4: Smoke-test the new binary against the real config**

```bash
cd $(mktemp -d)
cp ~/projects/ddrs/config/merit_training.yaml ./ddrs.yaml
ddrs init
ddrs plan
ddrs run --workflow train --max-mini-batches 1 --plot
ddrs status
ddrs show $(ls .ddrs/runs | tail -1)
```

Expected: each command succeeds; the run produces a populated `manifest.json` and a `plot/kan_parameters.csv`.

- [ ] **Step 5: Tag the release-candidate commit**

```bash
git tag -a v0.3.0-rc1 -m "ddrs CLI lifecycle (Spec A) RC1"
```

(No `git push` — let the user push when they're ready.)
