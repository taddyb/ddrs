# SP-2 Async-Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two icechunk-backed readers — `StreamflowStore` (`Qr` ×
daily→hourly) and `UsgsObservationsStore` (daily streamflow) — so SP-3 can
assemble a `RoutingBatch` with q' inputs and target observations.

**Architecture:** One module (`src/data/store/icechunk.rs`) holding both
stores. Each owns a `tokio::runtime::Runtime` and exposes sync
`read_window(&RhoWindow, ids)` methods. `block_on` happens at the icechunk
boundary; callers stay sync. Adapter strategy (zarrs-over-icechunk vs
native icechunk reads) is determined at implementation time — pick
whichever compiles first against the v2 crate.

**Tech Stack:** `icechunk` v2 (already pinned), `tokio` (already pinned),
`zarrs` (already in use for `ConusAdjacencyStore`), `chrono::NaiveDate`,
`ndarray`.

**Spec:** `.claude/specs/2026-05-17-sp2-async-data-design.md`

**Parent spec:**
`.claude/specs/2026-05-17-train_and_test-replication-design.md`

**DDR reference (read-only, cite line numbers in comments):**

- `~/projects/ddr/src/ddr/io/readers.py::read_ic` (~lines 376-403)
- `~/projects/ddr/src/ddr/io/readers.py::StreamflowReader.forward`
  (~lines 405-468)
- `~/projects/ddr/src/ddr/io/readers.py::IcechunkUSGSReader.read_data`
  (~lines 478-510)

**Production stores (read-only, used by integration tests):**

- `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` — streamflow
- `/mnt/ssd1/data/icechunk/usgs_daily_observations` — observations

Tests skip-if-absent so a clean machine still passes.

**Schema (verified at design time):**

- Streamflow: dims `(time=14976, divide_id=197088)`, var `Qr`, time
  origin 1980-01-01, daily.
- Observations: dims `(time=14610, gage_id=9067)`, var `streamflow`, time
  origin 1980-01-01, daily; `gage_id` is a string coord (8-char STAID).

---

## File Structure

**Created:**

- `src/data/store/icechunk.rs` — both stores in one module
- `tests/data_async.rs` — integration tests against live stores

**Modified:**

- `src/data/store/mod.rs` — `pub mod icechunk;` + re-exports
- `src/data/mod.rs` — re-export `StreamflowStore`, `UsgsObservationsStore`

**No new dependencies.** `icechunk = "2"` and `tokio = "1"` are already in
`Cargo.toml`.

---

## Implementation strategy notes (read once before starting Task 1)

This sub-project is the highest-unknown of the four reader projects. Two
parts of the design are best resolved at the keyboard, not in this plan:

1. **Repository open syntax.** The `icechunk` crate v2 likely exposes
   `Repository::open_existing(local_filesystem_storage(path))` (async). It
   may also have helpers like `Repository::open_existing_local(path)`.
   Try whichever compiles. If the crate's API has shifted entirely, the
   implementer reports BLOCKED with the actual `docs.rs/icechunk/2` page
   contents.

2. **Read adapter.** Two options, in preference order:
   - **Option A (preferred):** session exposes something implementing
     `zarrs::storage::ReadableStorage`. Then post-open reads are
     synchronous through `zarrs::array::Array::open` + `retrieve_array_subset`,
     exactly like the existing `ConusAdjacencyStore`.
   - **Option B (fallback):** read chunks via icechunk's native async
     `get` API; we `runtime.block_on(...)` each read.

   Pick A if it compiles. Otherwise B. Document the choice in the module
   doc-comment.

If both options take more than ~30 minutes of API archaeology per Task to
debug, stop and report BLOCKED with the compiler errors. We'll then either
spike the API in a sibling agent or fall back to a Python-side pre-export
of the time-series slices — but spending hours auto-fighting an API is
out of scope.

---

## Conventions for this plan

- Module doc-comments cite DDR source line numbers (see existing
  `src/data/store/zarr.rs` for style).
- Error variants: `DataError::IceChunk { path, source: Box<dyn ...> }`,
  `DataError::Malformed { path, message }`, `DataError::MissingIds`,
  `DataError::Io`. Do **not** add new variants.
- Tests against production stores live in `tests/data_async.rs`; skip
  cleanly with `eprintln!` if files are absent.
- Run `cargo test` (not `--release`) during the TDD cycle.
- After every passing cycle: commit. Style follows `git log --oneline`.

---

### Task 1: Module skeleton + helper function for repo open

**Files:**
- Create: `src/data/store/icechunk.rs`
- Modify: `src/data/store/mod.rs`
- Modify: `src/data/mod.rs`

The first task lays out the module structure and writes one private
helper that opens an icechunk repo + creates a read-only session on
`main`. This isolates the "which API call works" question so it's
resolved once.

- [ ] **Step 1: Create the module skeleton**

```rust
//! Icechunk-backed time-series readers.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/readers.py::read_ic` (lines ~376-403),
//! `StreamflowReader` (~lines 405-468), and `IcechunkUSGSReader`
//! (~lines 478-510). Both production stores live on a local filesystem
//! under `/mnt/ssd1/data/icechunk/`; S3 access is out of scope.
//!
//! Each store owns a `tokio::runtime::Runtime` and exposes a sync
//! `read_window(&RhoWindow, ids)` API — `block_on` happens at the icechunk
//! boundary. The dataset (SP-3) may later consolidate to a shared runtime
//! if profiling demands it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::NaiveDate;
use ndarray::Array2;
use tokio::runtime::Runtime;

use crate::data::dates::RhoWindow;
use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, IdIndex, Staid};

/// Shared session handle returned by the repo-open helper.
///
/// The exact inner types depend on which icechunk-vs-zarrs adapter
/// strategy compiles (Option A or B in the implementation strategy
/// notes). Keep this struct local to the module — it never escapes.
struct IcSession {
    runtime: Arc<Runtime>,
    /// Whatever the adapter handed us. Used by the per-var Array opens.
    // TODO(implementer): pick one of the two strategies and fill this in.
    _placeholder: (),
}

/// Open an icechunk repo at `path` and start a read-only session on the
/// `main` branch. Returns a shared session handle plus an owned runtime.
fn open_session(path: &Path) -> Result<IcSession> {
    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| DataError::Io {
                path: path.to_path_buf(),
                source: e,
            })?,
    );

    // ---- This is the load-bearing icechunk API call. Adapt as needed. ----
    let _ = runtime.block_on(async {
        // Replace this body with the real open. Likely shapes (in v2):
        //
        //   let storage = icechunk::storage::local_filesystem(path).await?;
        //   let repo = icechunk::Repository::open(storage).await?;
        //   let session = repo.readonly_session("main").await?;
        //   session
        //
        // The compile errors will tell you what method names actually exist.
        Result::<()>::Ok(())
    });
    // ----------------------------------------------------------------------

    Ok(IcSession {
        runtime,
        _placeholder: (),
    })
}
```

- [ ] **Step 2: Wire the module into `src/data/store/mod.rs`**

Update `src/data/store/mod.rs` to:

```rust
//! Per-source store modules. Each is a small focused reader over one of the
//! DDR data sources, returning `ndarray` buffers + domain-typed metadata.
//! Backend types (`zarrs::Array`, `netcdf::Variable`, `icechunk::Session`)
//! never escape the modules — callers see only `ndarray` and `data::ids`
//! types.
//!
//! Per the design notes in `src/data/mod.rs`: no `trait Store`, no
//! `Box<dyn Store>` — premature unification across three different I/O
//! models. Composition over abstraction at this layer.

pub mod gage_csv;
pub mod icechunk;
pub mod netcdf;
pub mod zarr;

pub use gage_csv::{GageMetadata, GageRow};
pub use icechunk::{StreamflowStore, UsgsObservationsStore};
pub use netcdf::AttributesStore;
pub use zarr::{ConusAdjacencyStore, GageSubgraph, GagesAdjacencyStore};
```

**Note:** the `pub use icechunk::...` line will fail until Task 2/4 define
those types. That's fine — we add it in Task 1 to lock the wire-up.
Actually — to keep the build green per task, **skip the `pub use icechunk`
line for now**, add it in Task 2 once `StreamflowStore` exists. Update
this step to only add `pub mod icechunk;` here:

```rust
pub mod gage_csv;
pub mod icechunk;     // <-- new
pub mod netcdf;
pub mod zarr;

pub use gage_csv::{GageMetadata, GageRow};
pub use netcdf::AttributesStore;
pub use zarr::{ConusAdjacencyStore, GageSubgraph, GagesAdjacencyStore};
```

- [ ] **Step 3: Build and verify a clean compile**

```
cargo build 2>&1 | tail -10
```

Expected: clean. The skeleton compiles; `open_session` is dead code at
this point (only used by Task 2+). If clippy fires on dead-code, add
`#[allow(dead_code)]` to `open_session` and `IcSession` — they're
referenced by upcoming tasks.

- [ ] **Step 4: Adapt the icechunk open call until it compiles**

Iterate on the body of `open_session`. The signature stays as written;
only the inner `block_on` body changes. Common forms to try in order:

1. `icechunk::Repository::open(icechunk::storage::local_filesystem(path).await?).await?`
2. `icechunk::Repository::open_existing(...)`
3. `icechunk::open_local_repository(path).await?`
4. Inspect `cargo doc --package icechunk --open` (if it builds).

When it compiles, populate `IcSession` with the concrete types you ended
up with (e.g. `session: icechunk::Session`, etc.) and remove
`_placeholder`. Update the doc-comment on `IcSession` to document the
strategy you picked.

If after 3 honest attempts it still doesn't compile, **stop and report
BLOCKED** with the compiler errors and the icechunk v2 docs.rs URL.

- [ ] **Step 5: Commit**

```
git add src/data/store/icechunk.rs src/data/store/mod.rs
git commit -m "Stub icechunk module with read-only session helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(`src/data/mod.rs` re-exports come in Task 2 when `StreamflowStore` exists.)

---

### Task 2: `StreamflowStore::open` + time/divide coord reads

**Files:**
- Modify: `src/data/store/icechunk.rs`

This task adds the public `StreamflowStore` struct and its `open` method.
At end of this task, the store can open the repo and remember the time
axis + divide-id index — but `read_window` doesn't exist yet.

- [ ] **Step 1: Append the `StreamflowStore` struct + open**

Append to `src/data/store/icechunk.rs`:

```rust
/// `Qr` reader over `merit_dhbv2_UH_retrospective.ic`-style icechunk repos.
pub struct StreamflowStore {
    pub path: PathBuf,
    /// Maps each COMID present in the store to its position in the
    /// `divide_id` coord.
    pub index: IdIndex<Comid>,
    /// Calendar date of the store's first time entry. Reads use
    /// `(window.window_start - time_start).num_days()` as the offset.
    pub time_start: NaiveDate,
    pub n_time: usize,
    /// Owned runtime. Reads `block_on` here.
    runtime: Arc<Runtime>,
    /// Whatever `Qr`-handle the adapter strategy gave us. Used by
    /// `read_window`.
    qr: QrHandle,
}

/// Strategy-specific `Qr` array handle. Either a `zarrs::array::Array` (Option A)
/// or an icechunk-native chunk-fetcher (Option B). Wrap whatever
/// concrete type Task 1's `open_session` returned.
struct QrHandle {
    // TODO(implementer): one of:
    //   zarrs_array: zarrs::array::Array<...>,
    //   ic_array: icechunk::Array,
}

impl StreamflowStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let session = open_session(&path)?;

        // 1. Read the `time` coord. icechunk's time is datetime64[ns];
        //    we cast to a Vec<i64> (nanoseconds since 1970-01-01) and
        //    convert the first entry to NaiveDate.
        let time_i64 = read_coord_i64(&session, "time", &path)?;
        let n_time = time_i64.len();
        let time_start = nanos_to_naive_date(time_i64[0]).ok_or_else(|| {
            DataError::Malformed {
                path: path.clone(),
                message: format!("unparseable time[0]: {}", time_i64[0]),
            }
        })?;

        // 2. Read the `divide_id` coord (int64) into a Vec<Comid> and
        //    build the IdIndex.
        let divide_i64 = read_coord_i64(&session, "divide_id", &path)?;
        let divide_comids: Vec<Comid> =
            divide_i64.into_iter().map(Comid).collect();
        let index = IdIndex::new(divide_comids);

        // 3. Open the `Qr` data var handle.
        let qr = open_var_qr(&session, &path)?;

        Ok(Self {
            path,
            index,
            time_start,
            n_time,
            runtime: session.runtime,
            qr,
        })
    }
}

// -------------------------------- helpers --------------------------------

fn read_coord_i64(_session: &IcSession, _name: &str, _path: &Path) -> Result<Vec<i64>> {
    // TODO(implementer): use whichever adapter you chose in Task 1.
    // For Option A (zarrs over icechunk), this is the same pattern as
    // ConusAdjacencyStore's `read_array_i32` in src/data/store/zarr.rs,
    // returning a Vec<i64> instead of Vec<i32>.
    unimplemented!("fill in Task 2 step 2")
}

fn open_var_qr(_session: &IcSession, _path: &Path) -> Result<QrHandle> {
    unimplemented!("fill in Task 2 step 2")
}

/// Convert a `datetime64[ns]` nanoseconds-since-epoch into a `NaiveDate`.
/// Returns `None` if the value is out of `chrono`'s representable range
/// (won't happen for the production stores).
fn nanos_to_naive_date(ns: i64) -> Option<NaiveDate> {
    let dt = chrono::DateTime::from_timestamp_nanos(ns);
    Some(dt.naive_utc().date())
}
```

- [ ] **Step 2: Implement `read_coord_i64` and `open_var_qr`**

Fill in the two `unimplemented!` helpers using whatever adapter strategy
Task 1 settled on. For **Option A (zarrs over icechunk session)** the
pattern mirrors `src/data/store/zarr.rs:178-191`:

```rust
fn read_coord_i64(session: &IcSession, name: &str, path: &Path) -> Result<Vec<i64>> {
    let arr = zarrs::array::Array::open(session.storage.clone(), &format!("/{name}"))
        .map_err(|e| ic_err(path, e))?;
    let subset = arr.subset_all();
    arr.retrieve_array_subset::<Vec<i64>>(&subset)
        .map_err(|e| ic_err(path, e))
}

fn open_var_qr(session: &IcSession, path: &Path) -> Result<QrHandle> {
    let arr = zarrs::array::Array::open(session.storage.clone(), "/Qr")
        .map_err(|e| ic_err(path, e))?;
    Ok(QrHandle { array: arr })
}

fn ic_err<E: std::error::Error + Send + Sync + 'static>(path: &Path, e: E) -> DataError {
    DataError::IceChunk {
        path: path.to_path_buf(),
        source: Box::new(e),
    }
}
```

For Option B (icechunk native), replace `zarrs::array::Array::open` with
the icechunk-native array handle and `runtime.block_on(...)` the read.
Concrete shape determined at the keyboard.

- [ ] **Step 3: Build clean**

```
cargo build 2>&1 | tail -5
```

Expected: clean. `StreamflowStore::read_window` doesn't exist yet — that's
fine; we add it in Task 3.

- [ ] **Step 4: Re-export from `src/data/mod.rs`**

In `src/data/store/mod.rs`, replace the existing `pub use icechunk::` line
or add it:

```rust
pub use icechunk::StreamflowStore;
```

(`UsgsObservationsStore` joins in Task 4.)

Update `src/data/mod.rs`'s `pub use store::{...}` to include
`StreamflowStore`:

```rust
pub use store::{
    AttributesStore, ConusAdjacencyStore, GageMetadata, GageRow, GageSubgraph,
    GagesAdjacencyStore, StreamflowStore,
};
```

- [ ] **Step 5: Commit**

```
git add src/data/store/icechunk.rs src/data/store/mod.rs src/data/mod.rs
git commit -m "Add StreamflowStore::open over icechunk

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `StreamflowStore::read_window` + `daily_to_hourly_trim` helper

**Files:**
- Modify: `src/data/store/icechunk.rs`

- [ ] **Step 1: Add a failing TDD unit test for `daily_to_hourly_trim`**

Append to `src/data/store/icechunk.rs` (inside a new `#[cfg(test)] mod
tests` block at the bottom of the file):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn daily_to_hourly_trim_pads_and_truncates() {
        // 3 days × 2 divides → 72 hours; trim to 71 (which is what
        // (rho_days - 1) * 24 produces for rho_days=3).
        let daily: Array2<f32> = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        )
        .unwrap();
        let hourly = daily_to_hourly_trim(&daily, 47);
        assert_eq!(hourly.shape(), &[47, 2]);
        // First 24 hours: day 0 values.
        for h in 0..24 {
            assert_eq!(hourly[(h, 0)], 1.0);
            assert_eq!(hourly[(h, 1)], 10.0);
        }
        // Next 23 hours: day 1 values (n_hourly=47 → we cut off at 47, so
        // hours 24..47 are still day 1).
        for h in 24..47 {
            assert_eq!(hourly[(h, 0)], 2.0);
            assert_eq!(hourly[(h, 1)], 20.0);
        }
    }
}
```

- [ ] **Step 2: Verify the test fails**

```
cargo test --lib data::store::icechunk::tests
```

Expected: compile error — `daily_to_hourly_trim` doesn't exist.

- [ ] **Step 3: Implement `daily_to_hourly_trim` and `read_window`**

Insert into `src/data/store/icechunk.rs`, before the `#[cfg(test)]` block:

```rust
/// Repeat a `(rho_days, N)` daily slab to `(n_hourly, N)` by replicating
/// each row 24 times along the time axis, then trim to `n_hourly` rows.
/// Mirrors `np.repeat(daily, 24, axis=1)[:, :n_hourly].T` in
/// `readers.py:447-454` (DDR transposes after — we yield time-major
/// directly).
fn daily_to_hourly_trim(daily: &Array2<f32>, n_hourly: usize) -> Array2<f32> {
    let (rho_days, n_div) = daily.dim();
    debug_assert!(
        n_hourly <= rho_days * 24,
        "n_hourly={n_hourly} exceeds rho_days*24={}",
        rho_days * 24
    );
    let mut hourly = Array2::<f32>::zeros((n_hourly, n_div));
    for h in 0..n_hourly {
        let d = h / 24;
        for j in 0..n_div {
            hourly[(h, j)] = daily[(d, j)];
        }
    }
    hourly
}

impl StreamflowStore {
    /// Read `Qr` for `window` and `comids`, returning a `(n_hourly, N)` f32
    /// matrix. Missing COMIDs (not in the store) get filled with the
    /// discharge minimum 0.001 — matches DDR's
    /// `torch.full(..., fill_value=0.001)` in `readers.py:464-468`.
    pub fn read_window(
        &self,
        window: &RhoWindow,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        // 1. Compute store-local time indices.
        let store_start_day_i64 =
            (window.window_start - self.time_start).num_days();
        if store_start_day_i64 < 0 {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window starts {} before store start {}",
                    window.window_start, self.time_start
                ),
            });
        }
        let store_start_day = store_start_day_i64 as usize;
        let end_day = store_start_day + window.rho_days;
        if end_day > self.n_time {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window extends to store day {end_day} but n_time={}",
                    self.n_time
                ),
            });
        }

        // 2. Resolve COMIDs → positions; track misses so we can fill them.
        let (positions, missing_indices) = self.index.positions_of(comids);
        // `positions` is len = (comids.len() - missing). Build a
        // companion list of "for each requested column, which store row
        // do we read?", with `None` for missing.
        let mut requested_to_store: Vec<Option<usize>> =
            vec![None; comids.len()];
        let mut next_pos = 0usize;
        for (i, _) in comids.iter().enumerate() {
            if missing_indices.binary_search(&i).is_ok() {
                requested_to_store[i] = None;
            } else {
                requested_to_store[i] = Some(positions[next_pos]);
                next_pos += 1;
            }
        }

        // 3. Read the slab. Two strategies (matching Task 1's choice):
        //    Option A (zarrs over icechunk): build an Extents covering
        //    rows [store_start_day .. end_day] and the unique COMID
        //    positions; read into Vec<f32>; reshape; scatter into output.
        //    Option B (icechunk native): block_on a chunk fetch per
        //    needed chunk; assemble the slab in Rust.
        //
        // For simplicity, read the full `(rho_days, N_active_in_store)`
        // slab for just the present COMIDs.
        let n_div = comids.len();
        let mut daily = Array2::<f32>::from_elem((window.rho_days, n_div), 0.001);
        let present_positions: Vec<usize> = requested_to_store
            .iter()
            .filter_map(|p| *p)
            .collect();
        if !present_positions.is_empty() {
            let raw = self.read_qr_slab(
                store_start_day,
                window.rho_days,
                &present_positions,
            )?;
            // raw shape: (rho_days, present_positions.len()).
            let mut present_col_idx = 0usize;
            for (out_col, mapping) in requested_to_store.iter().enumerate() {
                if mapping.is_some() {
                    for d in 0..window.rho_days {
                        daily[(d, out_col)] = raw[(d, present_col_idx)];
                    }
                    present_col_idx += 1;
                }
            }
        }

        // 4. Daily → hourly repeat-and-trim.
        Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
    }

    /// Read `Qr[store_start_day .. store_start_day + rho_days,
    /// present_positions]` as f32. Cast from f64 at the boundary.
    fn read_qr_slab(
        &self,
        _store_start_day: usize,
        _rho_days: usize,
        _present_positions: &[usize],
    ) -> Result<Array2<f32>> {
        // TODO(implementer): the actual zarrs `retrieve_array_subset` or
        // icechunk-native call. Pattern for Option A:
        //
        //   let subset = ArraySubset::new_with_ranges(&[
        //       store_start_day..store_start_day + rho_days,
        //       /* a contiguous range covering present_positions */,
        //   ]);
        //   let raw_f64: Vec<f64> = self.qr.array
        //       .retrieve_array_subset(&subset)
        //       .map_err(...)?;
        //   let arr_f32: Array2<f32> = ... reshape + cast + select cols ...;
        //
        // If `present_positions` is non-contiguous, the simple path is:
        //   - find min/max position
        //   - read the contiguous range [min..=max]
        //   - select the wanted columns into the output
        //
        // For SP-2 verification this is fast enough. SP-3 may introduce
        // gather-style reads if profiling demands.
        unimplemented!("fill in Task 3 step 3")
    }
}
```

Fill in `read_qr_slab` for the chosen adapter strategy. For Option A
(zarrs over icechunk), the call shape is the same as
`zarr.rs:178-191`'s `retrieve_array_subset` but for a 2D range.

- [ ] **Step 4: Run all unit tests**

```
cargo test --lib data::store::icechunk
```

Expected: 1 test passes (the `daily_to_hourly_trim` test). `read_window`
isn't unit-tested at this layer; it's integration-tested in Task 6.

- [ ] **Step 5: Commit**

```
git add src/data/store/icechunk.rs
git commit -m "Add StreamflowStore::read_window with daily-to-hourly transform

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `UsgsObservationsStore` (open + read_window)

**Files:**
- Modify: `src/data/store/icechunk.rs`
- Modify: `src/data/store/mod.rs`
- Modify: `src/data/mod.rs`

The observations store mirrors `StreamflowStore` but with two
differences: gauge IDs are strings, not int64; and missing gauges produce
a hard error (not a 0.001 fill).

- [ ] **Step 1: Append `UsgsObservationsStore`**

Append to `src/data/store/icechunk.rs` (above the `#[cfg(test)]` block):

```rust
/// `streamflow` reader over `usgs_daily_observations`-style icechunk repos.
pub struct UsgsObservationsStore {
    pub path: PathBuf,
    /// Maps each STAID present in the store to its position in the
    /// `gage_id` coord.
    pub index: IdIndex<Staid>,
    pub time_start: NaiveDate,
    pub n_time: usize,
    runtime: Arc<Runtime>,
    streamflow: StreamflowVarHandle,
}

/// `streamflow` var handle (analogous to `QrHandle` for streamflow).
struct StreamflowVarHandle {
    // TODO(implementer): zarrs::array::Array or icechunk-native handle.
}

impl UsgsObservationsStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let session = open_session(&path)?;

        // 1. time coord.
        let time_i64 = read_coord_i64(&session, "time", &path)?;
        let n_time = time_i64.len();
        let time_start = nanos_to_naive_date(time_i64[0]).ok_or_else(|| {
            DataError::Malformed {
                path: path.clone(),
                message: format!("unparseable time[0]: {}", time_i64[0]),
            }
        })?;

        // 2. gage_id coord — strings.
        let gage_strings = read_coord_strings(&session, "gage_id", &path)?;
        let staids: Vec<Staid> = gage_strings.iter().map(|s| Staid::new(s)).collect();
        let index = IdIndex::new(staids);

        // 3. `streamflow` data var.
        let streamflow = open_var_streamflow(&session, &path)?;

        Ok(Self {
            path,
            index,
            time_start,
            n_time,
            runtime: session.runtime,
            streamflow,
        })
    }

    /// Read observations for `window` and `staids`. Returns `(rho_days, G)`.
    /// Errors with `MissingIds` if any STAID is absent from the store —
    /// matches DDR's `.sel(gage_id=...)` KeyError behavior.
    pub fn read_window(
        &self,
        window: &RhoWindow,
        staids: &[Staid],
    ) -> Result<Array2<f32>> {
        // 1. Time-window validation.
        let store_start_day_i64 =
            (window.window_start - self.time_start).num_days();
        if store_start_day_i64 < 0 {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window starts {} before store start {}",
                    window.window_start, self.time_start
                ),
            });
        }
        let store_start_day = store_start_day_i64 as usize;
        let end_day = store_start_day + window.rho_days;
        if end_day > self.n_time {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window extends to store day {end_day} but n_time={}",
                    self.n_time
                ),
            });
        }

        // 2. Resolve STAIDs → positions; bail hard on misses.
        let (positions, missing_indices) = self.index.positions_of(staids);
        if !missing_indices.is_empty() {
            return Err(DataError::MissingIds {
                path: self.path.clone(),
                kind: "gage_id",
                missing: missing_indices.len(),
                total: staids.len(),
            });
        }
        debug_assert_eq!(positions.len(), staids.len());

        // 3. Read the slab.
        let raw = self.read_streamflow_slab(store_start_day, window.rho_days, &positions)?;
        // raw shape: (rho_days, G). Already in the layout we need; just
        // hand it back.
        Ok(raw)
    }

    fn read_streamflow_slab(
        &self,
        _store_start_day: usize,
        _rho_days: usize,
        _positions: &[usize],
    ) -> Result<Array2<f32>> {
        // TODO(implementer): same pattern as read_qr_slab. The variable
        // is `/streamflow` instead of `/Qr`; storage shape is
        // (time, gage_id). NaN values pass through unmodified.
        unimplemented!("fill in Task 4 step 2")
    }
}

fn read_coord_strings(_session: &IcSession, _name: &str, _path: &Path) -> Result<Vec<String>> {
    // TODO(implementer): the string-coord read is the riskiest part of
    // SP-2. Two paths:
    //   1. zarrs supports object/string arrays directly — read as
    //      Vec<String>.
    //   2. icechunk-native: read the raw chunk and decode the Python
    //      object-array format (a length-prefixed UTF-8 blob per entry).
    // If neither works, the spec's fallback is a one-shot Python pre-export
    // of just the gage_id list. Report BLOCKED with details if you reach
    // that point — we'll handle the fallback wiring then.
    unimplemented!("fill in Task 4 step 2")
}

fn open_var_streamflow(_session: &IcSession, _path: &Path) -> Result<StreamflowVarHandle> {
    unimplemented!("fill in Task 4 step 2")
}
```

- [ ] **Step 2: Fill in the three unimplemented helpers**

Implement `read_streamflow_slab`, `read_coord_strings`, and
`open_var_streamflow` for the chosen adapter strategy.

If `read_coord_strings` is genuinely blocked by string-array support,
**don't spend more than 30 minutes** trying to make it work. Stop and
report BLOCKED with the error. The fallback (Python-side coord dump)
will be wired in separately.

- [ ] **Step 3: Build**

```
cargo build 2>&1 | tail -5
```

- [ ] **Step 4: Update re-exports**

In `src/data/store/mod.rs`:

```rust
pub use icechunk::{StreamflowStore, UsgsObservationsStore};
```

In `src/data/mod.rs`:

```rust
pub use store::{
    AttributesStore, ConusAdjacencyStore, GageMetadata, GageRow, GageSubgraph,
    GagesAdjacencyStore, StreamflowStore, UsgsObservationsStore,
};
```

- [ ] **Step 5: Commit**

```
git add src/data/store/icechunk.rs src/data/store/mod.rs src/data/mod.rs
git commit -m "Add UsgsObservationsStore over icechunk

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Integration test — streamflow read against live store

**Files:**
- Create: `tests/data_async.rs`

- [ ] **Step 1: Create the integration test file**

```rust
//! Integration tests for SP-2 icechunk readers, exercised against the
//! production stores under `/mnt/ssd1/data/icechunk/`.
//!
//! Tests skip with `eprintln!` if the production stores are absent so a
//! clean machine still passes.

use std::path::Path;

use chrono::NaiveDate;

use ddrs::data::dates::{RhoWindow, TimeAxis};
use ddrs::data::ids::Comid;
use ddrs::data::{ConusAdjacencyStore, StreamflowStore};

const STREAMFLOW_IC: &str =
    "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic";
const CONUS_ADJ: &str =
    "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr";

#[test]
fn streamflow_store_reads_known_window() {
    if !Path::new(STREAMFLOW_IC).exists() || !Path::new(CONUS_ADJ).exists() {
        eprintln!("skipping: production streamflow / adjacency files absent");
        return;
    }
    let conus = ConusAdjacencyStore::open(CONUS_ADJ).expect("conus");
    let comids: Vec<Comid> = conus.order.iter().take(50).copied().collect();

    let store = StreamflowStore::open(STREAMFLOW_IC).expect("open streamflow");

    // 1981-10-01 is the MERIT training-period start (per
    // config/merit_training.yaml).
    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    use rand::SeedableRng;
    let window = axis.sample_rho_window(&mut rng, 90);

    let q_prime = store.read_window(&window, &comids).expect("read window");

    assert_eq!(q_prime.shape(), &[window.n_hourly(), 50]);
    // All values finite (either real q' or the 0.001 fill).
    for &v in q_prime.iter() {
        assert!(v.is_finite(), "got non-finite q': {v}");
    }
    // The 0.001 fill should not dominate — at least 80% of the first
    // 50 COMIDs from CONUS adjacency have DHBv2 coverage.
    let nonfill = q_prime.iter().filter(|&&v| (v - 0.001).abs() > 1e-9).count();
    assert!(
        nonfill > q_prime.len() / 2,
        "too many fill values: nonfill={nonfill}, total={}",
        q_prime.len()
    );
}

#[test]
fn streamflow_missing_divides_get_filled() {
    if !Path::new(STREAMFLOW_IC).exists() {
        eprintln!("skipping: streamflow not present");
        return;
    }
    let store = StreamflowStore::open(STREAMFLOW_IC).expect("open");

    // A real COMID from the design-time probe and a fake one.
    let comids = vec![Comid(71024425), Comid(-1)];
    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    use rand::SeedableRng;
    let window = axis.sample_rho_window(&mut rng, 90);

    let q = store.read_window(&window, &comids).expect("read");
    // Column 1 (Comid(-1)) is entirely the 0.001 fill.
    for h in 0..window.n_hourly() {
        assert!((q[(h, 1)] - 0.001).abs() < 1e-9, "fill at ({h},1) = {}", q[(h, 1)]);
    }
}
```

- [ ] **Step 2: Run the tests**

```
cargo test --test data_async streamflow_ 2>&1 | tail -10
```

Expected: 2 tests pass. If the icechunk read fails at runtime (e.g. a
chunk-fetch error), that surfaces here — debug at this layer, not by
hacking the store.

- [ ] **Step 3: Commit**

```
git add tests/data_async.rs
git commit -m "Add streamflow icechunk integration tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Integration test — observations read against live store

**Files:**
- Modify: `tests/data_async.rs`

- [ ] **Step 1: Append the observations integration tests**

Append to `tests/data_async.rs`:

```rust
use ddrs::data::ids::Staid;
use ddrs::data::{GageMetadata, UsgsObservationsStore};

const OBS_IC: &str = "/mnt/ssd1/data/icechunk/usgs_daily_observations";
const GAGES_CSV: &str =
    "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";

#[test]
fn observations_store_reads_known_window() {
    if !Path::new(OBS_IC).exists() || !Path::new(GAGES_CSV).exists() {
        eprintln!("skipping: observations / gages files absent");
        return;
    }
    let gages = GageMetadata::open(GAGES_CSV).expect("gages");
    let staids: Vec<Staid> = gages.staids().into_iter().take(10).collect();

    let store = UsgsObservationsStore::open(OBS_IC).expect("open obs");

    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    use rand::SeedableRng;
    let window = axis.sample_rho_window(&mut rng, 90);

    let obs = store.read_window(&window, &staids).expect("read obs");

    assert_eq!(obs.shape(), &[90, 10]);
    // At least one finite value across all 10 gauges × 90 days.
    let finite_count = obs.iter().filter(|v| v.is_finite()).count();
    assert!(finite_count > 0, "no finite obs values in window");
}

#[test]
fn observations_missing_gauges_errors() {
    if !Path::new(OBS_IC).exists() {
        eprintln!("skipping: observations not present");
        return;
    }
    let store = UsgsObservationsStore::open(OBS_IC).expect("open");

    let axis = TimeAxis::new(
        NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    use rand::SeedableRng;
    let window = axis.sample_rho_window(&mut rng, 90);

    let bogus = vec![Staid::new("99999999")];
    let err = store.read_window(&window, &bogus).unwrap_err();
    match err {
        ddrs::data::error::DataError::MissingIds { kind, missing, total, .. } => {
            assert_eq!(kind, "gage_id");
            assert_eq!(missing, 1);
            assert_eq!(total, 1);
        }
        other => panic!("expected MissingIds, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run the tests**

```
cargo test --test data_async 2>&1 | tail -10
```

Expected: 4 tests pass (2 streamflow + 2 observations).

- [ ] **Step 3: Commit**

```
git add tests/data_async.rs
git commit -m "Add observations icechunk integration tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final clippy / regression sweep

**Files:** none modified — verification only.

- [ ] **Step 1: Full test suite**

```
cargo test 2>&1 | grep "test result" | tail -20
```

Expected: every per-file summary shows `ok`; total ~58+ tests pass.

- [ ] **Step 2: Clippy on SP-2 code**

```
cargo clippy --all-targets -- -D warnings 2>&1 | grep -E "error|warning" | grep "icechunk\|data_async"
```

Expected: nothing matches (SP-2 code is clean). Pre-existing clippy
warnings elsewhere are out of scope per SP-1's precedent.

- [ ] **Step 3: Regression benchmark**

```
cargo run --release --example compare_ddr_sandbox 2>&1 | grep -i "absolute match\|verdict"
```

Expected: `ABSOLUTE MATCH`. SP-2 doesn't touch the routing core, but the
invariant must hold.

- [ ] **Step 4: Note any out-of-scope concerns**

If clippy finds new warnings in SP-2 code, fix them inline. If it finds
pre-existing warnings in routing-core code, leave them — note in the
final commit message that they're inherited from SP-1's housekeeping
pass.

No new commit needed for this task unless clippy required SP-2 fixes.

---

## Self-Review

### Spec coverage check

| Spec section | Covered by |
|---|---|
| `StreamflowStore::open` | Task 2 |
| `StreamflowStore::read_window` (incl. daily→hourly + missing fill) | Task 3 |
| `UsgsObservationsStore::open` | Task 4 |
| `UsgsObservationsStore::read_window` (incl. MissingIds error) | Task 4 |
| `daily_to_hourly_trim` helper | Task 3 |
| `read_coord_i64` / `read_coord_strings` / `open_var_*` helpers | Tasks 2, 4 |
| Module wiring (`pub mod icechunk`, re-exports) | Tasks 1, 2, 4 |
| Integration test — streamflow read | Task 5 |
| Integration test — observations read | Task 6 |
| Integration test — missing divide / missing gauge | Tasks 5, 6 |
| Regression invariant | Task 7 |

### Placeholder scan

The plan deliberately leaves the icechunk-vs-zarrs adapter call as an
`unimplemented!` in Tasks 1-4 because the v2 crate's exact API surface
is the highest-uncertainty piece of the design. Each `TODO(implementer)`
block points to a concrete fix-it-at-the-keyboard step and lists fallback
strategies. This is not a plan defect — it is the *only* honest way to
write the plan without doing a live spike here.

If the adapter resists 30+ minutes of attempts at any task, the
implementer reports BLOCKED, the controller spikes the API, and the
plan resumes. That contingency is called out explicitly in Tasks 1 and
4 and is preferable to a plan that lies about the API shape.

### Type consistency

- `StreamflowStore`, `UsgsObservationsStore` exposed publicly; `IcSession`,
  `QrHandle`, `StreamflowVarHandle` private to the module.
- `RhoWindow`, `TimeAxis`, `Comid`, `Staid`, `IdIndex` come from
  pre-existing modules and are used consistently.
- Error variants: `DataError::IceChunk`, `DataError::Malformed`,
  `DataError::MissingIds` — all pre-existing.
- Output shapes: streamflow `(n_hourly, N)`, observations `(rho_days, G)` —
  consistent across spec, plan, and tests.

No drift detected.

---

## Execution choice

Plan complete and saved to `.claude/specs/2026-05-17-sp2-async-data-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review
   between tasks, fast iteration. Same pattern as SP-1.
2. **Inline Execution** — execute in this session via executing-plans,
   batch with checkpoints.

Which approach?
