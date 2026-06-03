# SP-2 design: async-data layer (icechunk streamflow + observations)

**Status:** Draft, pending user review
**Parent:** [`2026-05-17-train_and_test-replication-design.md`](./2026-05-17-train_and_test-replication-design.md)
**Mirrors:** `ddr/io/readers.py::read_ic` (~lines 376-403),
`ddr/io/readers.py::StreamflowReader` (~lines 405-468),
`ddr/io/readers.py::IcechunkUSGSReader` (~lines 478-510).

## Why this sub-project

SP-2 reads the two icechunk-backed time-series stores that feed every
training batch:

- `merit_dhbv2_UH_retrospective.ic` — modeled lateral inflow `q'` (var `Qr`)
- `usgs_daily_observations` — USGS daily streamflow (target labels)

Without SP-2, SP-3's `MeritGagesDataset` cannot assemble a `RoutingBatch`.
SP-2 is the highest unknown-unknown in the project (the `icechunk` v2 Rust
crate is less battle-tested than its Python counterpart), which is why it
gets its own focused sub-project.

## Schemas (verified at design time against the production stores)

### Streamflow (`merit_dhbv2_UH_retrospective.ic`)
- coords: `time` (14976 days, 1980-01-01 → 2020-12-31), `divide_id` (197088, int64 COMIDs)
- data_var: `Qr` (the only var; dims `(time, divide_id)`)
- time resolution: **daily**; the MC engine consumes hourly, so SP-2 does the
  `repeat(24)` + trim-to-`n_hourly` transform at the read boundary
  (mirrors `StreamflowReader.forward`, `readers.py:447-454`)

### Observations (`usgs_daily_observations`)
- coords: `time` (14610 days, 1980-01-01 → 2019-12-31), `gage_id` (9067, string)
- data_var: `streamflow` (the only var; dims `(time, gage_id)`)
- gage_id dtype is `object` in xarray (Python strings); 8-char zero-padded
- daily-native; consumer is the loss function which also operates on daily
  resolution — no temporal resampling needed

**Both stores have the same time origin: 1980-01-01.** The streamflow store
extends one year beyond the observations store. We never need to read past
the observations end date during training because the loss is computed
against observations.

## Scope

In scope:

1. `StreamflowStore` — opens the icechunk repo, reads `Qr` for
   `(RhoWindow, &[Comid])`, returns `Array2<f32>` of shape `(n_hourly, N)`
   with the daily→hourly transform baked in.
2. `UsgsObservationsStore` — opens the icechunk repo, reads `streamflow` for
   `(RhoWindow, &[Staid])`, returns `Array2<f32>` of shape `(rho_days, G)`.
3. Single `tokio::runtime::Runtime` ownership pattern: each store owns its
   own `Arc<Runtime>` for SP-2 standalone. SP-3 will introduce a dataset-
   level shared runtime if profiling shows that's worth it.
4. Sync public API. `block_on` happens at the icechunk boundary, hidden
   inside each store.

Out of scope:

- S3-backed stores. DDR has an S3 fallback in `read_ic`; we go local-FS only.
  Re-add later when training in the cloud.
- icechunk write paths. We open read-only sessions on the `main` branch.
- Hourly-native streamflow stores. The MERIT config is daily; the hourly
  branch in DDR (`is_hourly: true`) is for a different dataset variant.
- A `trait TimeSeriesStore`. Per the data-layer convention, no premature
  unification.
- Background pre-fetching, chunk caching, async batch I/O. Sync reads are
  fine until profiling says otherwise.

## Architecture

```
                      RhoWindow            &[Comid]
                          │                    │
                          ▼                    ▼
            ┌─────────────────────────────────────────┐
            │  StreamflowStore::read_window           │
            │   ├─ map (window_start, rho_days) →     │
            │   │     store-local time indices        │
            │   ├─ resolve Comids → divide positions  │
            │   ├─ block_on(icechunk read slab)       │
            │   ├─ cast f64 → f32                     │
            │   └─ repeat(24) + trim to n_hourly      │
            └─────────────────────────────────────────┘
                          │
                          ▼
                  Array2<f32> (n_hourly, N)
```

Same shape for `UsgsObservationsStore::read_window`, minus the repeat-and-
trim. Output `(rho_days, G)`.

## Components

### 1. `src/data/store/icechunk.rs` — both stores in one module

Two structs, one shared helper for repo-open. Single module keeps the
icechunk dependency contained.

```rust
pub struct StreamflowStore {
    pub path: PathBuf,
    /// `Comid → position in the store's divide_id coord`.
    pub index: IdIndex<Comid>,
    /// Calendar date of the store's first time entry (1980-01-01 for the
    /// MERIT retrospective). Subsequent entries are at daily cadence.
    pub time_start: NaiveDate,
    pub n_time: usize,
    /// Owned tokio runtime; reads happen on it via `block_on`.
    runtime: Arc<tokio::runtime::Runtime>,
    /// The zarrs Array handle for `Qr`, with the icechunk session-backed
    /// store as `ReadableStorage`.
    qr: zarrs::array::Array<...>,
}

impl StreamflowStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self>;

    /// Read q' for the given window and divides. Returns `(n_hourly, N)`,
    /// where `n_hourly == window.n_hourly()` and `N == comids.len()`.
    /// Missing COMIDs are filled with `0.001` (matches DDR's
    /// `torch.full(..., fill_value=0.001)` in `readers.py:466`).
    pub fn read_window(
        &self,
        window: &RhoWindow,
        comids: &[Comid],
    ) -> Result<Array2<f32>>;
}

pub struct UsgsObservationsStore {
    pub path: PathBuf,
    pub index: IdIndex<Staid>,
    pub time_start: NaiveDate,
    pub n_time: usize,
    runtime: Arc<tokio::runtime::Runtime>,
    streamflow: zarrs::array::Array<...>,
}

impl UsgsObservationsStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self>;

    /// Read observations for the given window and gauges. Returns
    /// `(rho_days, G)`. NaN values are preserved (downstream filters them
    /// per-gauge via `nan_mask`).
    pub fn read_window(
        &self,
        window: &RhoWindow,
        staids: &[Staid],
    ) -> Result<Array2<f32>>;
}
```

### How the icechunk boundary works (the key unknown)

The `icechunk` crate v2 (already pinned in `Cargo.toml`) exposes:

- `icechunk::Repository::open(storage) -> Result<Repository, _>` — async
- `Repository::readonly_session(branch_or_tag) -> Result<Session, _>` — async
- `Session::store() -> impl ObjectStore` or similar — async or sync,
  depending on the version

We expect to be able to wrap the session's storage interface in `zarrs`'s
`ReadableStorage` trait (same trait the existing `ConusAdjacencyStore` uses
over `FilesystemStore`). If the adapter exists, post-open reads are
synchronous via `zarrs::Array::retrieve_array_subset`. If it does not, we
do `runtime.block_on(...)` per read.

**The exact adapter call is verified at implementation time, not in this
spec.** Per the project verification philosophy ("trust the readers,
validate alignment"), we don't pre-validate the icechunk-vs-zarrs glue —
the integration test against the live store either reads sane numbers or
it doesn't.

### 2. Module wire-up

```rust
// src/data/store/mod.rs
pub mod icechunk;
pub use icechunk::{StreamflowStore, UsgsObservationsStore};

// src/data/mod.rs
pub use store::{StreamflowStore, UsgsObservationsStore /* + existing */};
```

## Time-axis alignment

DDR's pattern uses a 1980-01-01 epoch + offset arithmetic
(`readers.py:438-446`). Both production stores happen to start at
1980-01-01, so the offset is always zero. We **do not replicate the epoch
arithmetic** — it's a fragile invariant.

Instead:

1. At `open`, read the store's `time` coord, parse the first value as a
   `NaiveDate`, store as `time_start`. Store `n_time = time.len()`.
2. On `read_window(rho_window, ids)`:
   - `store_start_day = (rho_window.window_start - self.time_start).num_days() as usize`
   - Assert `store_start_day + rho_window.rho_days <= self.n_time` —
     panics with a clear message if the caller asked for an out-of-range
     window.
3. Read the slab `[store_start_day .. store_start_day + rho_days, ...]`.

This is robust to stores whose start dates differ from 1980-01-01 (e.g.
a smaller-coverage retrospective), without DDR's epoch-offset gymnastics.

## Daily → hourly transform (streamflow only)

DDR's pattern, `readers.py:447-454`:

```python
n_hourly = len(routing_dataclass.dates.batch_hourly_time_range)
streamflow_data = np.repeat(_ds.compute().values.astype(np.float32), 24, axis=1)[
    :, :n_hourly
].T
```

In Rust this is a tight loop over the daily array:

```rust
fn daily_to_hourly_trim(daily: &Array2<f32>, n_hourly: usize) -> Array2<f32> {
    // daily: shape (rho_days, N).
    // hourly: shape (rho_days * 24, N), then truncated to (n_hourly, N).
    let (rho_days, n_div) = daily.dim();
    debug_assert!(n_hourly <= rho_days * 24);
    let mut hourly = Array2::<f32>::zeros((n_hourly, n_div));
    for h in 0..n_hourly {
        let d = h / 24;
        for j in 0..n_div {
            hourly[(h, j)] = daily[(d, j)];
        }
    }
    hourly
}
```

`n_hourly = (rho_days - 1) * 24` per `RhoWindow::n_hourly`. DDR's
`pd.date_range(... inclusive="left")` gives the same `(rho-1)*24` count.

## Missing-ID handling

**Streamflow:** DDR fills missing divides with `0.001` m³/s (the discharge
minimum from `attribute_minimums`). We mirror this. Caller does **not**
need to filter the COMID list before passing in — the store handles
missing IDs by leaving those columns at 0.001.

**Observations:** DDR uses `.sel(gage_id=...)` which raises `KeyError` if
any STAID is missing. We bail with `DataError::MissingIds { path, kind:
"gage_id", missing, total }` — matches existing convention.

(Distinct treatment: streamflow misses are common — not every COMID has
DHBv2 coverage. Observation misses are configuration bugs — gauges in
the training set must all have observation data.)

## Verification protocol

Per the project verification philosophy:

- **No fixture export.** No `scripts/export_streamflow_fixture.py`.
- **Trust the icechunk crate** to read the same bytes Python's `xarray.open_zarr`
  reads.
- Tests just sanity-check shape + a few sampled known values, against the
  live production stores. Skip-if-absent for CI.

### Unit tests

- `daily_to_hourly_trim`: round-trip property tests on small synthetic
  inputs (e.g. 3 days × 2 divides → 48 hours, trim to 47 if `n_hourly=47`).

### Integration tests against the live stores

`tests/data_async.rs` (new):

1. `streamflow_store_reads_known_window`:
   - Open `merit_dhbv2_UH_retrospective.ic` (skip-if-absent).
   - Take the first 50 COMIDs from `ConusAdjacencyStore.order`.
   - Build a `RhoWindow` over 1981-10-01 (start of MERIT training period) +
     90 days.
   - Read; assert shape `(89*24, 50)`, all values finite or 0.001-filled.

2. `observations_store_reads_known_window`:
   - Open `usgs_daily_observations`.
   - Take the first 10 STAIDs from `GageMetadata::open("...gages_3000.csv").staids()`.
   - Build a `RhoWindow` over 1981-10-01 + 90 days.
   - Read; assert shape `(90, 10)`. NaN-tolerant — observations have gaps.
   - Spot-check: assert at least one value is finite (the gauges aren't all
     NaN over a 90-day window in 1981).

3. `streamflow_missing_divides_get_filled`:
   - Pass a bogus `Comid(-1)` mixed into a list of real COMIDs.
   - Assert the column for `Comid(-1)` is all `0.001`.

4. `observations_missing_gauges_errors`:
   - Pass a `Staid("99999999")` (not in the store).
   - Assert `Err(DataError::MissingIds { kind: "gage_id", .. })`.

## Concerns

1. **icechunk-rs API for sessions over a local repo.** Need to confirm:
   - `Repository::open(local_filesystem_storage(path))` syntax in v2.
   - That a read-only session on `main` is achievable without a write
     handle.
   - That the session exposes something `zarrs` can adapt.

   Mitigation: if `zarrs` adapter doesn't exist, we use `block_on(...)`
   around an `icechunk::store`-native read API for each chunk fetch. Less
   ergonomic but always works.

2. **Streamflow store has 197K divides, observations has 9K gauges.**
   Building a `HashMap<Comid, usize>` over 197K entries at open is fine
   (~10 MB of HashMap). At-startup cost only.

3. **Time coord might be stored as `datetime64[ns]` (i64 nanoseconds).**
   Need to convert to `NaiveDate` correctly. `chrono::NaiveDateTime::from_timestamp_nanos`
   handles this. Verify with the production store at implementation time.

4. **`gage_id` is a string coord.** zarrs over icechunk reading string
   arrays — this is the lowest-confidence path. If `gage_id` is stored as
   a numpy `object` dtype and zarrs can't decode it, we fall back to
   reading via icechunk's native API (which knows about Python-style
   strings) or to a one-shot Python pre-export of the gage_id list.

   The "fall back to a Python pre-export of just the coord list" is
   acceptable — coords don't change.

5. **Test fixture pinning.** Spot-check assertions ("at least one finite
   value over 90 days") are loose by design. If we want a tighter check
   (e.g. "the 14190500 gauge has X non-NaN days in this window"), we'd
   need a small dump — which violates the verification policy. We accept
   the looser bar.

## Assumptions

1. icechunk-rs v2 can open the production stores read-only without
   surprises. (Will validate at first compile.)
2. Both stores will keep their 1980-01-01 start date. Switching the
   store's time origin would only require regenerating `time_start` at
   open — the API doesn't change.
3. Streamflow's `Qr` and observations' `streamflow` remain the sole
   data_vars. If new vars are added, they're ignored.
4. Numerical NaN propagation through the f64→f32 cast is correct
   (`f64::NAN as f32 == f32::NAN`).

## Module layout summary

```
src/data/
├── store/
│   ├── mod.rs                       (+) re-export StreamflowStore, UsgsObservationsStore
│   ├── icechunk.rs                  (new) both stores in one focused module
│   ├── netcdf.rs                    (unchanged from SP-1)
│   ├── gage_csv.rs                  (unchanged from SP-1)
│   └── zarr.rs                      (unchanged)
└── ...

tests/
└── data_async.rs                    (new) 4 integration tests against live icechunk stores

Cargo.toml                           (no changes — icechunk + tokio already declared)
```

No new top-level dependencies. `icechunk = "2"` and `tokio` with
`rt-multi-thread + macros` are already in `Cargo.toml`.

## Risks summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| icechunk-rs Repository::open API differs from sketch | High | Implementer adapts at compile time, like SP-1 Task 6. |
| zarrs over icechunk session doesn't work | Medium | Fall back to icechunk's native chunk-fetch API. |
| String coord (`gage_id`) reads fail | Medium | One-shot Python pre-export of coord list, or icechunk-native read. |
| `block_on` deadlocks in nested-runtime scenarios | Low | Each store owns its own runtime; no nesting. |
| Time coord units differ from `datetime64[ns]` | Low | Detect at open; refuse with a clear `Malformed` error. |
| Tokio runtime overhead for small reads | Low | Profile-driven; not optimizing in SP-2. |

## What I'm NOT going to do

- No `trait TimeSeriesStore` unifying the two readers. They have different
  ID types (Comid vs Staid), different data_var names (`Qr` vs
  `streamflow`), and different missing-ID semantics (fill vs error). A
  trait would force convergence on the wrong axes.
- No S3 support.
- No caching layer. The dataset (SP-3) can add per-batch caching if
  profiling demands it.
- No async public API. Sync everywhere, `block_on` at the boundary.

## Open questions for review

None — small enough to spec in full. The implementation plan will turn
this into ~8 ordered tasks.

## Next step after approval

Invoke writing-plans → implementation plan → subagent-driven execution
(same workflow as SP-1).
