# Summed Q' baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `~/projects/ddr/scripts/summed_q_prime.py` into ddrs as a baseline that runs at `ddrs plan` time, prints a DDR-style metrics table to console, and lands a zarr + JSON pair under `<run_dir>/baseline/` when `ddrs run` is invoked. The baseline is a non-routing reference: for each gauge, sum the per-divide `Qr` over its upstream subgraph and compare against USGS daily observations. If the trained KAN doesn't beat this, the routing isn't earning its keep.

**Architecture:** New `src/baseline/` module with two files:
- `summed_q_prime.rs` — pure compute: takes a testing-mode `Config` + a `Workspace`, returns a `SummedQPrime { predictions, observations, metrics, ... }`.
- `cache.rs` — blake3-keyed content-addressed cache under `.ddrs/baselines/<key>/` so `ddrs plan` and `ddrs run` share the result without recomputation.

`ddrs plan` will call `compute_or_load_cached`, print the metrics table, and exit. `ddrs run --workflow train-and-test` will load (or compute) the cached baseline and hardlink/copy its files into `<run_dir>/baseline/`. The hash key covers `(streamflow.path ∥ observations.path ∥ gages.path ∥ gages_adjacency.path ∥ testing.start_time ∥ testing.end_time)`. Drift on training-only fields (seed, KAN config, learning rate) does **not** invalidate.

**Tech Stack:** Rust, BURN 0.21 (CUDA backend for `Tensor` sums; ndarray for everything else), zarrs (write baseline zarr), blake3 (cache key), serde_json (cache metadata).

**Spec source:** This plan was derived from an in-conversation design discussion on 2026-06-02. The two confirmed decisions:
- Plan/run share the baseline via `.ddrs/baselines/<key>/` cache (option b, not recompute).
- Extend `Metrics` with bias / FHV / FLV for full DDR parity (already committed as `c02b2b3` — prerequisite, do **not** redo).

---

## File map

**New:**

| Path | Why |
|---|---|
| `src/baseline/mod.rs` | Module entry; re-exports `SummedQPrime`, `compute_or_load_cached` |
| `src/baseline/summed_q_prime.rs` | Pure compute pipeline (upstream union → bulk read → per-gauge sum → metrics) |
| `src/baseline/cache.rs` | Blake3 cache key, write/load `<key>/predictions.zarr` + `metrics_summary.json` + `cache.json` |
| `src/baseline/print.rs` | DDR-style metrics table renderer (mirrors `summed_q_prime.py:85-110`) |
| `tests/baseline_compute.rs` | Tiny 3-gauge / 5-day synthetic scenario, verify per-gauge sums and metric parity |
| `tests/baseline_cache.rs` | Round-trip cache write→load, verify metric equality |
| `tests/baseline_plan_e2e.rs` | End-to-end `ddrs plan` invocation in a tmpdir with stubbed stores |

**Modified:**

| Path | Why |
|---|---|
| `src/data/store/icechunk.rs` | Add `StreamflowStore::read_window_daily` and `UsgsObservationsStore::read_window_daily` to skip the hourly-expand (eval is 15 yr, hourly would be ~8.5 GB; daily is ~370 MB) |
| `src/data/store/zarr.rs` | Add `GageSubgraph::upstream_comids(&ConusAdjacencyStore) -> Vec<Comid>` helper |
| `src/cli/plan.rs` | After `compute_summary`, call `baseline::compute_or_load_cached` and print the metrics table |
| `src/cli/run.rs` | In `TrainAndTest` branch, load cached baseline and copy/hardlink into `<run_dir>/baseline/` |
| `src/cli/manifest.rs` | Add `baseline_zarr: Option<PathBuf>` and `baseline_metrics_json: Option<PathBuf>` to `RunOutputs` |
| `src/lib.rs` | `pub mod baseline;` |
| `Cargo.toml` | Add `blake3 = "1"` to `[dependencies]` |
| `CLAUDE.md` | Mention the baseline under "Data sources" and the `.ddrs/baselines/` cache layout |

**Not touched (intentionally):**

| Path | Why |
|---|---|
| `src/training/metrics.rs` | Already extended in commit `c02b2b3` (bias / FHV / FLV). Do not modify. |
| `src/training/driver.rs`, `src/training/eval.rs` | Baseline is independent of training. Don't entangle. |

---

## Concerns

1. **`ddrs plan` is no longer side-effect-free.** Today plan is config-validation only; after this change it (a) reads ~370 MB from icechunk and (b) writes to `.ddrs/baselines/<key>/`. The trade-off was accepted in the design discussion: cheaper than recomputing in `run`, and the cache key isolates blast radius. But: document this in CLAUDE.md so future agents don't assume plan is pure.

2. **Cache invalidation is path-based, not content-based.** If the user updates the icechunk store in place (e.g. `merit_dhbv2_UH_retrospective.ic` evolves) without renaming, the cache key won't change and stale results will be served. Mitigation: include the icechunk store's `latest_snapshot_id` in the hash if `icechunk::Repository::snapshot_id()` is reachable; otherwise document the limitation.

3. **No-GPU case.** `ddrs plan` runs on CPU. The Python script uses cupy; ddrs's Rust port should use ndarray (no need for tensors — these are mat-mul-style reductions, fine on CPU). Make sure the eval-window memory budget (~370 MB) doesn't surprise a low-RAM machine.

4. **FHV/FLV require ≥ 50 finite pairs per gauge** to be non-NaN (round(0.98 × N) must be < N). For the 15-year daily eval window (~5479 days), this is always satisfied — but flag NaN counts in the summary so an unexpectedly short window is loud.

5. **Memory peak during compute.** Holding `qr_daily: (n_upstream_unique, n_days) ≈ (17k × 5479) f32 = 370 MB` plus `preds: (n_gauges, n_days) ≈ (2365 × 5479) f32 = 50 MB` simultaneously is fine on most machines. But if the per-gauge upstream-divide union grows (>50k divides), watch out. Add a one-line log of the planned read size before the read.

## Assumptions

1. **Eval window comes from the testing-mode config.** `Config::from_yaml_file_with_mode(.., ConfigMode::Testing)` returns the testing-overlay config whose `experiment.start_time`/`end_time` define the baseline window. Same source the trained model is evaluated against.

2. **`StreamflowStore` and `UsgsObservationsStore` both store daily data.** Confirmed by `config/merit_training.yaml:5` ("Forcing is daily; the loader interpolates to hourly..."). The new `read_window_daily` exits before the daily-to-hourly expand at `daily_to_hourly_trim`.

3. **Per-gauge upstream-divide set = union of `GageSubgraph.indices_0 ∪ indices_1`.** These are CONUS-position indices; map through `ConusAdjacencyStore.order: Vec<Comid>` to get COMIDs. This mirrors how the training dataloader assembles the per-gauge network.

4. **The cache survives across config edits that don't affect the baseline.** A user re-running `ddrs plan` after tweaking `seed:` or `kan_head.hidden_size:` should get instant-load (cached). Only changing `data_sources.*` or `testing.{start,end}_time` should invalidate.

5. **Manifest schema additions are optional fields.** `baseline_zarr` and `baseline_metrics_json` are `Option<PathBuf>` so old manifest readers (e.g. `ddrs show`) don't break.

---

## Task 1: Add `read_window_daily` to streamflow + observations stores

The existing `read_window` returns hourly-shaped output. For a 15-yr eval × ~17k upstream divides, hourly is ~8.5 GB transient; daily is ~370 MB. Add a sister method that exits before `daily_to_hourly_trim`.

**Files:**
- Modify: `src/data/store/icechunk.rs` — extract the daily-Array2 body of `read_window` into a private `read_daily_inner`, expose `read_window_daily` as a thin wrapper that returns it directly. Repeat for `UsgsObservationsStore`.
- Test: `tests/baseline_compute.rs` (Task 3) covers integration; no dedicated unit test needed.

- [ ] **Step 1: Refactor `StreamflowStore::read_window`** into `read_window_daily` + `read_window`

In `src/data/store/icechunk.rs` around line 291:

```rust
pub fn read_window_daily(
    &self,
    window_start: NaiveDate,
    n_days: usize,
    comids: &[Comid],
) -> Result<Array2<f32>> {
    // (Steps 1-4 of the existing read_window body, parameterized by
    // window_start + n_days instead of &RhoWindow.)
    // Returns (n_days, comids.len()) — same daily shape used internally.
}

pub fn read_window(&self, window: &RhoWindow, comids: &[Comid]) -> Result<Array2<f32>> {
    let daily = self.read_window_daily(window.window_start, window.rho_days, comids)?;
    Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
}
```

The existing `read_window` body has the daily Array2 explicit at line 324, so the refactor is a literal extraction.

- [ ] **Step 2: Add `UsgsObservationsStore::read_window_daily`**

Same pattern for observations (`src/data/store/icechunk.rs:493`+). Note: the obs store's axes may differ from streamflow's — check whether obs stores `(time, gage)` or `(gage, time)` and align the returned shape to `(n_days, n_gauges)` for consistency with `read_window_daily`.

- [ ] **Step 3: Build + run existing tests**

```bash
cargo build
cargo test --lib --tests
```

Expected: all existing tests still green. The refactor is internal — `read_window`'s public behavior is unchanged.

- [ ] **Step 4: Commit**

```
data/store/icechunk: extract read_window_daily helpers

The existing read_window always converts daily-to-hourly. The
upcoming summed Q' baseline needs daily output over a 15-year window
where the hourly form would consume ~8.5 GB transient. Pull the
daily Array2 body into read_window_daily and have read_window wrap
it.
```

---

## Task 2: Add `GageSubgraph::upstream_comids` helper

`summed_q_prime.py:198` reads `gages_adjacency[gauge]["order"][:]` to get the COMID list per gauge. The Rust loader stores only COO indices in CONUS-position space; derive the COMID list from `indices_0 ∪ indices_1` mapped through `ConusAdjacencyStore.order`.

**Files:**
- Modify: `src/data/store/zarr.rs`
- Test: unit test in the same file

- [ ] **Step 1: Add the helper to `GageSubgraph`**

In `src/data/store/zarr.rs`:

```rust
impl GageSubgraph {
    /// Returns the unique COMIDs in this gauge's upstream subgraph,
    /// sorted by CONUS position (stable across runs).
    /// Mirrors `gages_adjacency[gauge]["order"][:]` from
    /// `~/projects/ddr/scripts/summed_q_prime.py:198`.
    pub fn upstream_comids(&self, conus: &ConusAdjacencyStore) -> Vec<Comid> {
        let mut seen = std::collections::BTreeSet::new();
        seen.extend(self.indices_0.iter().copied());
        seen.extend(self.indices_1.iter().copied());
        seen.iter().map(|&pos| conus.order[pos as usize]).collect()
    }
}
```

- [ ] **Step 2: Add a unit test**

Append to `src/data/store/zarr.rs::tests`:

```rust
#[test]
fn upstream_comids_dedupes_and_orders() {
    let conus = ConusAdjacencyStore {
        order: vec![Comid(100), Comid(200), Comid(300), Comid(400)],
        ..Default::default()  // adapt to whatever the struct needs
    };
    let sg = GageSubgraph {
        staid: Staid("test".into()),
        gage_idx: 3,
        gage_catchment: String::new(),
        indices_0: vec![3, 2, 1],
        indices_1: vec![2, 1, 0],
    };
    let comids = sg.upstream_comids(&conus);
    assert_eq!(comids, vec![Comid(100), Comid(200), Comid(300), Comid(400)]);
}
```

If `ConusAdjacencyStore` lacks `Default`, construct it manually with the minimum fields.

- [ ] **Step 3: Build + test**

```bash
cargo test --lib zarr::tests::upstream_comids
```

- [ ] **Step 4: Commit**

```
data/store/zarr: GageSubgraph::upstream_comids helper

Derives the unique COMIDs in a gauge's upstream subgraph from its
COO indices via the CONUS order array. Replaces the Python
gages_adjacency[gauge]["order"][:] lookup for the upcoming Q' baseline.
```

---

## Task 3: Pure compute — `src/baseline/summed_q_prime.rs`

The algorithm, faithfully ported from `summed_q_prime.py:155-293`:

```
fn compute(test_cfg, ws) -> Result<SummedQPrime>:
    1. Parse eval window from test_cfg.experiment.{start,end}_time
    2. Open: ConusAdjacencyStore, GagesAdjacencyStore (over gauges_csv),
             StreamflowStore, UsgsObservationsStore
    3. For each gauge:
         upstream = subgraph.upstream_comids(&conus)
         gauge_basins.insert(staid, upstream)
         all_needed.extend(upstream)
    4. all_needed_sorted: Vec<Comid> = sorted dedup
       qr_daily: Array2<f32> = streamflow.read_window_daily(start, n_days, &all_needed_sorted)
         → shape (n_days, n_needed)
    5. preds: Array2<f32>::zeros((n_gauges, n_days))
       For each gauge i:
         indices = positions of gauge_basins[staid_i] within all_needed_sorted
         preds.row_mut(i) = qr_daily.select(Axis(1), &indices).map_axis(Axis(1), nansum)
       (Note Python uses cp.nansum across axis 0 since shape is (divides, days);
        Rust here uses (days, divides) so axis differs.)
    6. obs = observations.read_window_daily(start, n_days, &gauges)
       → shape (n_days, n_gauges); transpose to (n_gauges, n_days)
    7. metrics = Metrics::compute(&preds.t().to_owned(), &obs.t().to_owned())
       (Metrics::compute takes (gauges, time) shape.)
    8. Return SummedQPrime { preds, obs, gage_ids, time_range_daily, metrics }
```

**Files:**
- New: `src/baseline/mod.rs`, `src/baseline/summed_q_prime.rs`
- Modify: `src/lib.rs` to add `pub mod baseline;`
- Test: `tests/baseline_compute.rs`

- [ ] **Step 1: Scaffold `src/baseline/mod.rs`**

```rust
//! Non-routing baselines for sanity-checking trained KAN performance.
//!
//! `summed_q_prime`: sum each gauge's upstream divide Qr time series,
//! compare against USGS daily observations. Mirrors
//! ~/projects/ddr/scripts/summed_q_prime.py.

pub mod summed_q_prime;

pub use summed_q_prime::{compute, SummedQPrime};
```

- [ ] **Step 2: Implement `summed_q_prime::compute`**

Define the struct:

```rust
use ndarray::Array2;
use chrono::NaiveDate;
use crate::data::ids::Staid;
use crate::training::metrics::Metrics;

pub struct SummedQPrime {
    pub predictions: Array2<f32>,  // (n_gauges, n_days)
    pub observations: Array2<f32>, // (n_gauges, n_days)
    pub gage_ids: Vec<Staid>,
    pub time_range_daily: Vec<NaiveDate>,
    pub metrics: Metrics,
}
```

Implement the pipeline steps 1-8 above. Use `ndarray::s!` macro for slicing, `Axis(1)` for the time axis. NaN-safe sum: `iter().filter(|v| v.is_finite()).sum::<f32>()`.

- [ ] **Step 3: Add `pub mod baseline;` to `src/lib.rs`**

- [ ] **Step 4: Write `tests/baseline_compute.rs`**

A synthetic scenario with 3 gauges, 3 upstream divides each (overlapping), 5 days. Hand-compute expected per-gauge sums and metrics, assert agreement to 1e-5.

If full integration against real icechunk stores is too heavy for a unit test, factor `compute` into a `compute_from_arrays(qr_daily, obs_daily, gauge_basins, gage_ids, time_range)` inner function and test that — leave the icechunk-opening shell untested. Real-data integration goes in Task 6's e2e test.

- [ ] **Step 5: Build + test**

```bash
cargo test --test baseline_compute
```

- [ ] **Step 6: Commit**

```
baseline: add summed_q_prime::compute

Per-gauge upstream Qr summation pipeline ported from
~/projects/ddr/scripts/summed_q_prime.py. Returns predictions,
observations, and DDR-parity Metrics (NSE/KGE/RMSE/bias/FHV/FLV).
Independent of routing or trained parameters.
```

---

## Task 4: Cache layer — `src/baseline/cache.rs`

**Files:**
- New: `src/baseline/cache.rs`, modify `src/baseline/mod.rs` to expose `compute_or_load_cached`
- Modify: `Cargo.toml` (add blake3)
- Test: `tests/baseline_cache.rs`

- [ ] **Step 1: Add `blake3 = "1"` to `Cargo.toml [dependencies]`**

- [ ] **Step 2: Define the cache key**

In `src/baseline/cache.rs`:

```rust
use blake3::Hasher;
use std::path::Path;

pub fn cache_key(test_cfg: &Config) -> String {
    let mut h = Hasher::new();
    let ds = test_cfg.data_sources.as_ref().expect("data_sources required");
    h.update(canonicalize_or_raw(&ds.streamflow).as_bytes());
    h.update(b"\n");
    h.update(canonicalize_or_raw(&ds.observations).as_bytes());
    h.update(b"\n");
    h.update(canonicalize_or_raw(&ds.gages).as_bytes());
    h.update(b"\n");
    h.update(canonicalize_or_raw(&ds.gages_adjacency).as_bytes());
    h.update(b"\n");
    let exp = test_cfg.experiment.as_ref().expect("experiment required");
    h.update(exp.start_time.as_bytes());
    h.update(b"\n");
    h.update(exp.end_time.as_bytes());
    let hex = h.finalize().to_hex();
    hex[..16].to_string()  // 16 chars = 64 bits, comfortably collision-free at our scale
}

fn canonicalize_or_raw(p: &Path) -> String {
    p.canonicalize()
        .map(|c| c.display().to_string())
        .unwrap_or_else(|_| p.display().to_string())
}
```

- [ ] **Step 3: Write/load helpers**

```rust
pub fn load_cached(workspace_root: &Path, key: &str) -> Option<SummedQPrime> {
    let dir = workspace_root.join("baselines").join(key);
    if !dir.is_dir() { return None; }
    // Read predictions.zarr + metrics_summary.json + cache.json
    // Return Some only if all three exist and parse cleanly.
}

pub fn save_cached(workspace_root: &Path, key: &str, q: &SummedQPrime) -> Result<()> {
    let dir = workspace_root.join("baselines").join(key);
    std::fs::create_dir_all(&dir)?;
    write_predictions_zarr(&dir.join("predictions.zarr"), q, ...)?;
    write_metrics_summary_json(&dir.join("metrics_summary.json"), &q.metrics)?;
    write_cache_metadata(&dir.join("cache.json"), key, ...)?;
    Ok(())
}
```

Reuse `crate::training::zarr_io::write_predictions_zarr` if the schema matches; otherwise write a focused baseline zarr writer.

- [ ] **Step 4: Expose `compute_or_load_cached`**

```rust
pub fn compute_or_load_cached(
    test_cfg: &Config,
    workspace_root: &Path,
) -> Result<SummedQPrime> {
    let key = cache_key(test_cfg);
    if let Some(cached) = load_cached(workspace_root, &key) {
        return Ok(cached);
    }
    let q = compute(test_cfg)?;
    save_cached(workspace_root, &key, &q)?;
    Ok(q)
}
```

- [ ] **Step 5: Round-trip test in `tests/baseline_cache.rs`**

Build a `SummedQPrime` by hand (small ndarray shapes), `save_cached(tmpdir, key, &q)`, then `load_cached(tmpdir, key).unwrap()`. Assert predictions/observations equal element-wise; assert all six metric vectors are finite and match.

- [ ] **Step 6: Commit**

```
baseline: cache layer keyed by data-source paths + eval window

Content-addressed under .ddrs/baselines/<key>/ so plan and run share
the result. Key = blake3(streamflow ∥ observations ∥ gages ∥
gages_adjacency ∥ start_time ∥ end_time). Drift on training-only
fields (seed, kan_head, lr) does not invalidate.
```

---

## Task 5: Print metrics table — `src/baseline/print.rs`

Mirror `summed_q_prime.py:85-110` so the table is recognizable to anyone familiar with the DDR script.

**Files:**
- New: `src/baseline/print.rs`, re-export from `src/baseline/mod.rs`
- Test: snapshot or assertion-based test on a hand-built `Metrics`

- [ ] **Step 1: Implement `print_metrics_summary`**

```rust
pub fn print_metrics_summary(m: &Metrics, total_gauges: usize) {
    println!();
    println!("{}", "=".repeat(80));
    println!("{:^80}", "SUMMED Q' METRICS SUMMARY");
    println!("{}", "=".repeat(80));
    println!("Total Gauges Evaluated: {total_gauges}");
    println!("{}", "-".repeat(80));
    println!("{:<12} {:>9} {:>9} {:>9} {:>9} {:>7}",
             "METRIC", "MEDIAN", "MEAN", "Q25", "Q75", "VALID");
    println!("{}", "-".repeat(80));
    for (name, xs, decimals) in [
        ("Bias",    &m.bias, 3usize),
        ("FLV (%)", &m.flv,  2),
        ("FHV (%)", &m.fhv,  2),
        ("KGE",     &m.kge,  3),
        ("NSE",     &m.nse,  3),
    ] {
        let s = stats(xs);
        println!(
            "{:<12} {:>9.*} {:>9.*} {:>9.*} {:>9.*} {:>7}",
            name,
            decimals, s.median, decimals, s.mean,
            decimals, s.q25,    decimals, s.q75,
            s.valid,
        );
    }
    println!("{}", "=".repeat(80));
}

struct Stats { median: f32, mean: f32, q25: f32, q75: f32, valid: usize }
fn stats(xs: &[f32]) -> Stats { /* nan-skipping percentile/mean */ }
```

- [ ] **Step 2: Test the renderer**

Capture stdout (e.g. via `gag` crate or by refactoring `print_metrics_summary` to take a `&mut impl Write`). Build a `Metrics` with all-finite values, render, assert key substrings are present (`"SUMMED Q' METRICS SUMMARY"`, `"Bias"`, etc.).

- [ ] **Step 3: Commit**

```
baseline: DDR-parity metrics table renderer

Mirrors ~/projects/ddr/scripts/summed_q_prime.py:85-110 so the
console output is recognizable. Prints median/mean/Q25/Q75 + valid
count for Bias/FLV/FHV/KGE/NSE.
```

---

## Task 6: Wire `ddrs plan` to compute + print

**Files:**
- Modify: `src/cli/plan.rs`
- Test: `tests/baseline_plan_e2e.rs`

- [ ] **Step 1: Add the call in `plan::plan`**

After `compute_summary` (around `src/cli/plan.rs:107`):

```rust
// Compute or load cached baseline. Prints DDR-style summary to stdout.
let test_cfg = Config::from_yaml_file_with_mode(config_path, ConfigMode::Testing)
    .map_err(|e| CliError::Other(Box::new(e)))?;
let baseline = crate::baseline::compute_or_load_cached(&test_cfg, workspace.root())
    .map_err(|e| CliError::Other(Box::new(e)))?;
crate::baseline::print_metrics_summary(&baseline.metrics, baseline.gage_ids.len());
eprintln!(
    "baseline cached → {}",
    workspace.root().join("baselines")
        .join(crate::baseline::cache_key(&test_cfg)).display()
);
```

The print goes to stdout (so `ddrs plan --json` users can still parse JSON; baseline summary stays human-readable in plain text). The path log goes to stderr.

- [ ] **Step 2: Decide JSON-mode behavior**

If `cli.json` is set, skip the print and inject `baseline_metrics` into the JSON output instead. Wire this via a `with_baseline: bool` flag on `PlanResult`, or via a separate return value the bin layer composes.

- [ ] **Step 3: E2E test**

In `tests/baseline_plan_e2e.rs`, build a tmpdir with a minimal ddrs.yaml pointing at synthetic stores (constructed via the same factories used by other store tests). Invoke `plan::plan` and assert:
- The cache dir is created
- The Metrics fields are populated
- `compute_or_load_cached` returns the cached result on second call (verify by checking file mtime didn't change)

- [ ] **Step 4: Commit**

```
cli/plan: run summed Q' baseline; print DDR-style table

Plan now also opens icechunk streamflow + observations, computes
per-gauge upstream Qr sums, and prints the metrics table. Result is
cached under .ddrs/baselines/<key>/ so repeat plans (and the
subsequent run) share the work.
```

---

## Task 7: Wire `ddrs run --workflow train-and-test` to record baseline

**Files:**
- Modify: `src/cli/run.rs`, `src/cli/manifest.rs`

- [ ] **Step 1: Extend `RunOutputs`**

```rust
pub struct RunOutputs {
    pub checkpoints: Vec<PathBuf>,
    pub plot: Option<PathBuf>,
    pub eval_zarr: Option<PathBuf>,
    pub baseline_zarr: Option<PathBuf>,         // NEW
    pub baseline_metrics_json: Option<PathBuf>, // NEW
}
```

Both new fields are `Option` so existing serialized manifests (`Default::default()`-via-serde) deserialize cleanly.

- [ ] **Step 2: Copy baseline into run dir in `TrainAndTest` branch**

After Phase 2 (testing, around `src/cli/run.rs:312` post-evaluate):

```rust
// Copy the cached baseline into the run dir so artifacts travel
// with the manifest. compute_or_load_cached is idempotent — if plan
// already ran, this is just a load.
let baseline = crate::baseline::compute_or_load_cached(&test_cfg, input.workspace.root())
    .map_err(|e| CliError::Other(Box::new(e)))?;
let run_baseline_dir = run_dir.join("baseline");
fs::create_dir_all(&run_baseline_dir)?;
let cache_dir = input.workspace.root()
    .join("baselines")
    .join(crate::baseline::cache_key(&test_cfg));
copy_dir_recursive(
    &cache_dir.join("predictions.zarr"),
    &run_baseline_dir.join("summed_q_prime.zarr"),
)?;
fs::copy(
    cache_dir.join("metrics_summary.json"),
    run_baseline_dir.join("metrics_summary.json"),
)?;
outputs.baseline_zarr = Some(PathBuf::from("baseline/summed_q_prime.zarr"));
outputs.baseline_metrics_json = Some(PathBuf::from("baseline/metrics_summary.json"));
eprintln!("baseline → {}", run_baseline_dir.display());
```

Add `fn copy_dir_recursive(src, dst) -> std::io::Result<()>` as a private helper at the bottom of `run.rs`. For zarr (just a directory of small files), `fs::copy` per file is fine — no need for `cp -r` shellout. Or hardlink (`fs::hard_link`) to save space if same filesystem.

- [ ] **Step 3: Build + run integration tests**

```bash
cargo test --test cli_run
cargo test --test baseline_plan_e2e
```

- [ ] **Step 4: Commit**

```
cli/run: copy cached baseline into run_dir for train-and-test

After Phase 2 evaluation, copy .ddrs/baselines/<key>/ into
<run_dir>/baseline/ and record the paths in manifest.outputs. Cache
ensures we don't recompute when plan already ran.
```

---

## Task 8: Documentation

- [ ] **Step 1: Update CLAUDE.md**

Add a "Baseline" subsection under "Data sources" (around line 124):

```markdown
## Baselines

`ddrs plan` and `ddrs run --workflow train-and-test` compute a summed Q'
baseline — per-gauge sum of upstream divide Qr over the testing eval
window. Cached at `.ddrs/baselines/<key>/`, keyed by hash of data-source
paths + eval window. Run output ends up at `<run_dir>/baseline/`.

If your trained KAN's median NSE doesn't beat this baseline, the
routing isn't earning its keep — check training loss curves and the
KAN head's gradient stats first, not the solver.
```

Also update the "When in doubt" section to mention `.claude/skills/ddrs-architecture.md`-style canonical docs need a `baseline.md` entry when this lands.

- [ ] **Step 2: Update the docs mdBook**

If `regenerate-docs` is wired up (per `.claude/skills/regenerate-docs.md`), add a stub `.claude/skills/ddrs-baseline.md` with frontmatter:

```yaml
---
name: ddrs-baseline
description: Summed Q' baseline — non-routing per-gauge sum of upstream Qr for sanity-checking trained models.
output: reference/baseline.md
sources:
  - src/baseline/summed_q_prime.rs
  - src/baseline/cache.rs
  - src/cli/plan.rs
---
```

Run `/regenerate-docs` to publish.

- [ ] **Step 3: Commit**

```
docs: baseline section in CLAUDE.md + canonical skill

Document the summed Q' cache layout and how to read the metrics
table in CLAUDE.md. Add a new canonical skill feeding
docs/reference/baseline.md.
```

---

## Verification checklist (final)

- [ ] `cargo build --release` — clean compile
- [ ] `cargo test --lib` — all green
- [ ] `cargo test --tests` — all green (including 3 new test files)
- [ ] `ddrs plan` on `config/merit_training.yaml` prints the metrics table
- [ ] `ddrs plan` runs a second time in well under a minute (cache hit)
- [ ] `ddrs run --workflow train-and-test --max-mini-batches 1` produces `<run_dir>/baseline/summed_q_prime.zarr` and `metrics_summary.json`
- [ ] `manifest.json` records both paths under `outputs`
- [ ] Editing `seed:` in the YAML does **not** invalidate the cache (verify by file mtime)
- [ ] Editing `testing.start_time` **does** invalidate the cache (different `<key>` dir is created)

## Out of scope (do not implement)

- Cache GC. `ddrs gc` already prunes runs; if `.ddrs/baselines/` grows unboundedly we'll address it then.
- Baseline integration with `--workflow train` or `--workflow eval` standalone. Only `train-and-test` records it for now.
- A "compare-to-baseline" metric in the train loop (e.g. "you're +0.15 NSE above baseline"). Tempting but speculative — wait until the data is in front of someone.
