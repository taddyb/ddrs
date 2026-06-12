---
name: ddrs-reading-inputs
description: How ddrs reads the live training data — zarr adjacency stores (CONUS + per-gauge subgraphs), netcdf catchment attributes, icechunk streamflow forcing and USGS observations, plus the Comid/Staid newtypes and TimeAxis sampler.
output: usage/inputs-reading.md
sources:
  - src/data/store/zarr.rs
  - src/data/store/netcdf.rs
  - src/data/store/icechunk.rs
  - src/data/ids.rs
  - src/data/dates.rs
  - src/data/error.rs
  - src/data/mod.rs
---

# ddrs-reading-inputs

> Canonical agent-readable skill. Published chapter at `docs/usage/inputs-reading.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

ddrs reads DDR's training data **in place** — there is no export or
conversion step. Five sources back the dataloader, each with a focused
module under `src/data/store/`. Reads return `ndarray::Array` buffers
keyed by `Comid` / `Staid` newtypes; no `trait Store` unifies them
(see Gotchas).

| Source | Path | Crate |
|---|---|---|
| MERIT adjacency | `~/projects/ddr/data/merit_conus_adjacency.zarr` | `zarrs` |
| Per-gauge subgraphs | `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` | `zarrs` |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` | `netcdf` |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` | `icechunk` |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` | `icechunk` |

CONUS MERIT is **346,321 reaches × 338,814 edges** — not millions; consumer
GPUs handle it. `src/data/mod.rs` owns the public re-exports; the dataset
(`src/data/dataset.rs`) owns a single `tokio::runtime::Runtime` and calls
`block_on(...)` at the icechunk boundary so the rest of ddrs stays sync.

## Zarr adjacency stores

Both adjacency targets are zarr v3 binsparse-COO with int32/uint8 arrays
and `bytes` + `zstd` codecs (written by `ddr/engine/src/ddr_engine/core/zarr_io.py`).
ddrs reads them via the `zarrs` crate and never exposes `zarrs::Array`
to callers — reads return `Vec<T>` or `ndarray::Array1` with the foreign
types contained.

### `ConusAdjacencyStore`

The full CONUS-wide graph + per-reach geometry. Loaded **once** at dataset
construction, eager (~30 MB zstd-compressed at 346K reaches).

```rust
pub struct ConusAdjacencyStore {
    pub path: PathBuf,
    pub order: Vec<Comid>,           // topological order
    pub index: IdIndex<Comid>,       // COMID -> position
    pub length_m: Array1<f32>,       // per-reach channel length [m]
    pub slope: Array1<f32>,          // per-reach channel slope [-]
    pub indices_0: Vec<i32>,         // COO rows (downstream)
    pub indices_1: Vec<i32>,         // COO cols (upstream)
    pub n: usize,                    // reach count
    pub nnz: usize,                  // edge count
}
```

`element i of order` is the COMID at zarr position `i`; downstream stores
(attributes, forcing) reuse this position-space via `IdIndex`.

### `GagesAdjacencyStore`

Per-STAID subgraph COOs keyed by gauge. Eager-loaded for the chosen-gauge
set only (a few MB). Each batch picks a gauge; the subgraph's `n_active`
varies — see Gotchas.

### Construction

Each store has an `open(path)` constructor that returns `Result<Self>`
with a `DataError::Zarr { path, source }` on failure. Path context is
preserved end-to-end.

## Netcdf catchment attributes

`AttributesStore` (`src/data/store/netcdf.rs`) reads the static catchment
attributes via the `netcdf` crate (`DataError::NetCdf` variant), mirroring
DDR's `AttributesReader`. `open(path, &attr_names, &comids)` materializes a
dense `(F, N_present)` f32 matrix — `F` requested attributes × the COMIDs
present in the file:

```rust
pub struct AttributesStore {
    pub path: PathBuf,
    pub attr_names: Vec<String>,
    pub attrs: Array2<f32>,        // (F, N_present), f32
    pub index: IdIndex<Comid>,     // present COMIDs -> column
    pub row_means: Array1<f32>,    // per-attribute nan/inf-safe mean
}
```

Each attribute column is read in full once (~24 MB at 2.94M f64), cast to
f32, reduced to a NaN/Inf-safe mean (`row_means`, via `naninfmean`), then
sliced to the present COMID subset. A missing `COMID` coord or attribute
variable yields `DataError::Malformed`.

## Icechunk forcing + USGS

`StreamflowStore` and `UsgsObservationsStore` (`src/data/store/icechunk.rs`)
read the two time-series sources from local icechunk repos. Targets:

- Streamflow forcing: `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic`
- USGS observations: `/mnt/ssd1/data/icechunk/usgs_daily_observations`

Because the `icechunk` crate has no `zarrs` dependency, the module wraps an
`icechunk::Store` behind an `IcZarrStorage` adapter implementing zarrs's
`ReadableStorageTraits`; each store opens a read-only `main`-branch session
and owns a `tokio::runtime::Runtime`, calling `block_on(...)` at the
icechunk boundary so the rest of ddrs stays sync. `DataError::IceChunk
{ path, source }` carries the path on failure.

Both `open(path)` parse a CF-convention `time` coord ("days since
YYYY-MM-DD") into `time_start` + `n_time`, then expose windowed reads:

```rust
pub struct StreamflowStore {        // keyed by Comid
    pub path: PathBuf,
    pub index: IdIndex<Comid>,
    pub time_start: NaiveDate,
    pub n_time: usize,
    // ...
}

let forcing = qr.read_window(&window, &comids)?;                  // (n_hourly, N)
let daily   = qr.read_window_daily(window_start, n_days, &comids)?; // (n_days, N)
```

`StreamflowStore::read_window` repeats daily → hourly (`(rho-1)*24` rows);
`read_window_daily` / `read_test_window` are the daily/eval variants.
Missing COMIDs fill with `0.001` (discharge minimum).
`UsgsObservationsStore` keys on `Staid`, stays daily (no hourly transform),
and hard-errors with `DataError::MissingIds` on absent STAIDs.

## Newtype IDs

DDR's Python uses raw `int` for COMIDs and raw `str` for STAIDs, which
has been a recurring bug surface (forgot-to-zfill, COMID-vs-divide_id
mixups). Newtypes in `src/data/ids.rs` let the compiler catch those:

```rust
pub struct Comid(pub i64);          // MERIT catchment id
pub struct Staid(String);           // USGS gauge id, zero-padded to 8 chars
```

`Staid::new("1563500")` zero-pads to `"01563500"` to match DDR's
canonical form (`base_geodataset.py:35`, `readers.py:131`). Never pass
raw ints or strings across the data layer — always wrap.

### `IdIndex<T>`

Every store builds one of these at `open` time; every read consumes one
to map domain IDs to integer array positions:

```rust
pub struct IdIndex<Id: Eq + Hash + Clone + Debug> { /* ... */ }

impl<Id> IdIndex<Id> {
    pub fn position(&self, id: &Id) -> Option<usize>;
    pub fn positions_of(&self, ids: &[Id]) -> (Vec<usize>, Vec<usize>);
    // returns (positions, missing_indices_into_input)
}
```

`positions_of` returns both the resolved positions and the indices of
the requested IDs that were missing — callers decide whether to warn,
error, or fill with sentinels.

## Time axes + rho-window sampler

`TimeAxis` mirrors DDR's `Dates` class (`geodatazoo/dataclasses.py`),
covering the bits the loader actually uses:

```rust
pub struct TimeAxis {
    pub start: NaiveDate,
    pub end: NaiveDate,       // inclusive
    pub num_days: usize,
}

impl TimeAxis {
    pub fn new(start: NaiveDate, end: NaiveDate) -> Self;
    pub fn sample_rho_window<R: Rng>(&self, rng: &mut R, rho_days: usize) -> RhoWindow;
    pub fn day_index(&self, date: NaiveDate) -> Option<usize>;
}
```

`sample_rho_window` picks a contiguous `rho`-day window uniformly at
random (`random_start ~ U[0, num_days - rho)`), mirroring DDR's
`Dates.calculate_time_period` (`dataclasses.py:160-167`).

**Daily ↔ hourly invariant:** when `rho` daily steps are selected, the
corresponding hourly range has `(rho - 1) * 24` entries — DDR's
`StreamflowReader.forward` relies on this when it does
`np.repeat(daily, 24)[:, :n_hourly]`. Don't break that semantic.

## DataError convention

Every variant of `DataError` (`src/data/error.rs`) carries a `PathBuf`
so error context survives wrapping:

```rust
pub enum DataError {
    Zarr      { path: PathBuf, source: Box<dyn Error + Send + Sync> },
    NetCdf    { path: PathBuf, source: netcdf::Error },
    IceChunk  { path: PathBuf, source: Box<dyn Error + Send + Sync> },
    Io        { path: PathBuf, source: std::io::Error },
    MissingIds{ path: PathBuf, kind: &'static str, missing: usize, total: usize },
    Malformed { path: PathBuf, message: String },
    Yaml      { path: PathBuf, source: serde_yaml::Error },
    Csv       { path: PathBuf, source: csv::Error },
}
```

DDR's stack traces (`KeyError: 'gage_id'` from a wrapped pandas read) are
notoriously hard to debug — paying the extra field once here means
callers don't have to wrap every read with their own context.

## Gotchas

- **Zarr stores opened lazily, slice on demand.** `ConusAdjacencyStore`
  is eager (load once, ~30 MB), but per-batch slices into
  attributes/forcing/observations are read on demand. Don't pre-materialize
  the full attribute matrix — it doesn't fit cleanly into the training loop.
- **No `Box<dyn Store>` / no `trait Store`.** Premature unification was
  explicitly rejected (`src/data/mod.rs`): the five sources have different
  I/O models (sync zarr/netcdf vs async icechunk) and the call sites
  diverge too much. Each store is a focused module returning typed
  `ndarray::Array` buffers.
- **Gauge subgraphs differ per batch.** `n_active` varies with the gauge
  pick; downstream code can't cache shapes across batches. The static
  CONUS state lives in `ConusAdjacencyStore`; per-batch state lives in
  whatever `GagesAdjacencyStore` returns for the chosen gauge.
- **MERIT CONUS scale.** 346,321 reaches × 338,814 edges. Not millions —
  the port targets consumer GPUs (24 GB VRAM is comfortable). Don't
  assume a "production HPC" footprint when planning memory budgets.
- **Adjacency is topologically ordered, lower-triangular.** `rows[k] >=
  cols[k]` holds for every COO edge. The forward-substitution sparse
  solver in `src/sparse.rs` assumes this. The regression test
  `data_zarr_store::conus_adjacency_loads_real_merit_zarr` asserts it
  against the on-disk zarr.

## Verification

| Path | Covered by |
|---|---|
| Zarr adjacency loads + topo-order invariant | `cargo test --test data_zarr_store conus_adjacency_loads_real_merit_zarr` |
| `Staid` zero-pad | `cargo test --lib data::ids::tests::staid_zfill_8` |
| `IdIndex` roundtrip | `cargo test --lib data::ids::tests::id_index_roundtrip` |
| `TimeAxis` + rho sampler | `cargo test --lib data::dates::` |

The CONUS-zarr test is the cross-cutting one: it both verifies the
reader and locks the lower-triangular invariant that the routing core
depends on.
