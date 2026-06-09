# Reading inputs

ddrs reads DDR's training data **in place** — there is no export or
conversion step. Five on-disk sources back the dataloader, each with a
focused module under `src/data/store/`. Reads return `ndarray::Array`
buffers keyed by `Comid` / `Staid` newtypes; there is deliberately no
`trait Store` unifying them. This chapter walks through each reader, the
newtype IDs that index them, the `TimeAxis` / `RhoWindow` sampler, and
the `DataError` convention that gives every failure a source path.

## What it is

The data layer (`src/data/`) is the boundary between DDR's heterogeneous
on-disk formats and ddrs's sync routing core. Each source has a different
I/O model — sync zarr, sync netCDF, async-first icechunk — so each gets
its own small module rather than a shared abstraction. The five sources,
their canonical paths (from `config/merit_training.yaml`), and the reader
that opens them:

| Source | Path | Reader |
|---|---|---|
| MERIT adjacency | `~/projects/ddr/data/merit_conus_adjacency.zarr` | `zarrs` — `ConusAdjacencyStore` (`src/data/store/zarr.rs`) |
| Per-gauge subgraphs | `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` | `zarrs` — `GagesAdjacencyStore` (`src/data/store/zarr.rs`) |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` | `netcdf` — `AttributesStore` (`src/data/store/netcdf.rs`) |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` | `icechunk` — `StreamflowStore` (`src/data/store/icechunk.rs`) |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` | `icechunk` — `UsgsObservationsStore` (`src/data/store/icechunk.rs`) |

All five readers are implemented; `src/data/store/mod.rs` re-exports
`ConusAdjacencyStore`, `GagesAdjacencyStore`, `AttributesStore`,
`StreamflowStore`, and `UsgsObservationsStore`. CONUS MERIT is
**346,321 reaches × 338,814 edges** — not millions; consumer GPUs (24 GB
VRAM is comfortable) handle it. Backend types (`zarrs::Array`,
`netcdf::Variable`, `icechunk::Store`) never escape their modules —
callers see only `ndarray` and `data::ids` types.

## Zarr adjacency stores

Both adjacency targets are zarr v3 binsparse-COO with int32/uint8 arrays
and `bytes` + `zstd` codecs (written by DDR's
`ddr_engine/core/zarr_io.py`). ddrs reads them via the `zarrs` crate and
never exposes `zarrs::Array` to callers — reads return `Vec<T>` or
`ndarray::Array1` with the foreign types contained.

### `ConusAdjacencyStore`

The full CONUS-wide graph plus per-reach geometry. Loaded **once** at
dataset construction, eager (~30 MB zstd-compressed at 346K reaches):

```rust
pub struct ConusAdjacencyStore {
    pub path: PathBuf,
    pub order: Vec<Comid>,           // COMIDs in topological order
    pub index: IdIndex<Comid>,       // COMID -> position
    pub length_m: Array1<f32>,       // per-reach channel length [m]
    pub slope: Array1<f32>,          // per-reach channel slope [-]
    pub indices_0: Vec<i32>,         // COO rows (downstream)
    pub indices_1: Vec<i32>,         // COO cols (upstream)
    pub n: usize,                    // reach count (== order.len())
    pub nnz: usize,                  // edge count
}
```

`order[i]` is the COMID at zarr position `i`; downstream stores
(attributes, forcing) reuse this position-space via `IdIndex`. The COO
pair `(indices_0, indices_1)` describes the sparse routing graph in
lower-triangular form — every edge `(rows[k], cols[k])` has
`rows[k] >= cols[k]` after the topological sort. `open` validates that
`order`, `length_m`, and `slope` agree in length and that `indices_0`
and `indices_1` are the same length, returning `DataError::Malformed`
otherwise:

```rust
let store = ConusAdjacencyStore::open(path)?;  // -> Result<Self, DataError>
```

A `DataError::Zarr { path, source }` carries the store path on any
`zarrs` failure.

### `GagesAdjacencyStore`

Per-STAID subgraph COOs keyed by gauge. `open(path, staids)` eager-loads
only the requested gauge set (a few MB); STAIDs whose subgroup is missing
are silently dropped (mirroring DDR's `valid_gauges_mask` in
`_collate_gages`):

```rust
pub struct GagesAdjacencyStore {
    pub path: PathBuf,
    pub subgraphs: HashMap<Staid, GageSubgraph>,
}

pub struct GageSubgraph {
    pub staid: Staid,
    pub gage_idx: usize,         // outlet position in the CONUS array
    pub gage_catchment: String,  // MERIT COMID of the outlet (attr)
    pub indices_0: Vec<i32>,     // COO rows in CONUS position space
    pub indices_1: Vec<i32>,     // COO cols in CONUS position space
}
```

A subgraph's COO indices reference **CONUS** positions, not compressed
positions — the dataset compresses at batch time when it unions multiple
gauges' subgraphs. `GageSubgraph::upstream_comids(&conus)` returns the
unique COMIDs in a gauge's upstream subgraph, sorted by CONUS position
(stable across runs). Each batch picks a gauge, so the active node count
varies between batches — see [Gotchas](#gotchas).

## NetCDF catchment attributes

`AttributesStore` (`src/data/store/netcdf.rs`) reads the static catchment
attributes via the `netcdf` crate, mirroring DDR's `AttributesReader`. At
`open` it materializes a dense `(F, N)` f32 matrix where `F` is the number
of requested attributes and `N` is the count of requested COMIDs present
in the file:

```rust
pub struct AttributesStore {
    pub path: PathBuf,
    pub attr_names: Vec<String>,
    pub attrs: Array2<f32>,        // (F, N_present), f32
    pub index: IdIndex<Comid>,     // present COMIDs -> column
    pub row_means: Array1<f32>,    // per-attribute nan/inf-safe mean
}

let store = AttributesStore::open(path, &attr_names, &comids)?;
```

The file stores 1D variables on a `COMID` dimension. Each requested
attribute column is read in full once (~24 MB at 2.94M f64), cast to f32,
reduced to a NaN/Inf-safe mean (`row_means`, via `naninfmean`), then
sliced down to the present COMID subset — fancy indexing is unnecessary
and the peak transient is bounded by `F × 24 MB`. A missing `COMID`
coordinate or a missing attribute variable yields `DataError::Malformed`;
a netCDF-level failure yields `DataError::NetCdf { path, source }`. The
attribute names that feed the routing head come from
`config/merit_training.yaml` (see [Formatting inputs](inputs-formatting.md)).

## Icechunk forcing + USGS observations

`StreamflowStore` and `UsgsObservationsStore` (`src/data/store/icechunk.rs`)
read the two time-series sources from local icechunk repositories. Because
the `icechunk` crate has no `zarrs` dependency, the module wraps an
`icechunk::Store` behind an `IcZarrStorage` adapter implementing zarrs's
`ReadableStorageTraits`; each store opens a read-only session on the `main`
branch and owns a `tokio::runtime::Runtime`, calling `block_on(...)` at the
icechunk boundary so the rest of ddrs stays sync:

```rust
pub struct StreamflowStore {
    pub path: PathBuf,
    pub index: IdIndex<Comid>,     // COMID -> column
    pub time_start: NaiveDate,
    pub n_time: usize,
    // ...
}

pub struct UsgsObservationsStore {
    pub path: PathBuf,
    pub index: IdIndex<Staid>,     // STAID -> column
    pub time_start: NaiveDate,
    pub n_time: usize,
    // ...
}
```

Both parse a CF-convention `time` coordinate ("days since YYYY-MM-DD")
into `time_start` + `n_time` at `open`, then expose windowed reads:

```rust
let qr  = StreamflowStore::open(streamflow_path)?;
let obs = UsgsObservationsStore::open(observations_path)?;

let forcing: Array2<f32> = qr.read_window(&window, &comids)?;     // (n_reach, T)
let daily:   Array2<f32> = qr.read_window_daily(&window, &comids)?;
```

`read_window` takes a `&RhoWindow` (below) plus the IDs to slice, returning
an `ndarray::Array2<f32>`. `StreamflowStore` keys on `Comid`;
`UsgsObservationsStore` keys on `Staid`. Both also expose
`read_test_window` / test-mode variants for contiguous evaluation chunks.
An icechunk-level failure yields `DataError::IceChunk { path, source }`.

## Newtype IDs

DDR's Python uses raw `int` for COMIDs and raw `str` for STAIDs, which has
been a recurring bug surface (forgot-to-zfill mistakes, COMID-vs-divide_id
mixups). Newtypes in `src/data/ids.rs` let the compiler catch those:

```rust
/// MERIT catchment identifier.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Comid(pub i64);

/// USGS gauge identifier — zero-padded to 8 characters at construction.
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Staid(String);

impl Staid {
    pub fn new(s: &str) -> Self {
        let mut padded = s.to_string();
        while padded.len() < 8 {
            padded.insert(0, '0');
        }
        Self(padded)
    }
    pub fn as_str(&self) -> &str { &self.0 }
}
```

`Staid::new("1563500")` zero-pads to `"01563500"` to match DDR's canonical
form (`base_geodataset.py:35`, `readers.py:131`); a string already 8+
characters is left untouched. The unit test locks the contract:

```rust
#[test]
fn staid_zfill_8() {
    assert_eq!(Staid::new("1563500").as_str(), "01563500");
    assert_eq!(Staid::new("01563500").as_str(), "01563500");
    assert_eq!(Staid::new("123456789").as_str(), "123456789"); // longer untouched
}
```

The convention everywhere in `ddrs`: use these types, never raw
`i64`/`String` across the data layer.

### `IdIndex<T>`

Every store builds one of these at `open` time; every read consumes one to
map domain IDs to integer array positions:

```rust
pub struct IdIndex<Id: Eq + Hash + Clone + Debug> { /* ... */ }

impl<Id> IdIndex<Id> {
    pub fn new(ids: Vec<Id>) -> Self;
    pub fn position(&self, id: &Id) -> Option<usize>;
    pub fn contains(&self, id: &Id) -> bool;
    pub fn positions_of(&self, ids: &[Id]) -> (Vec<usize>, Vec<usize>);
    // returns (positions, missing_indices_into_input)
    pub fn len(&self) -> usize;
    pub fn id_at(&self, pos: usize) -> Option<&Id>;
    pub fn ids(&self) -> &[Id];
}
```

`positions_of` is the workhorse — it returns both the resolved positions
and the indices of the requested IDs that were missing, so callers decide
whether to warn, error, or fill with sentinels
(`positions.len() + missing.len() == ids.len()`). The roundtrip test:

```rust
#[test]
fn id_index_roundtrip() {
    let idx = IdIndex::new(vec![Comid(10), Comid(20), Comid(30)]);
    assert_eq!(idx.position(&Comid(20)), Some(1));
    assert_eq!(idx.position(&Comid(99)), None);
    let (positions, missing) =
        idx.positions_of(&[Comid(30), Comid(99), Comid(10), Comid(42)]);
    assert_eq!(positions, vec![2, 0]);
    assert_eq!(missing, vec![1, 3]);
}
```

## Time axes + rho-window sampler

`TimeAxis` (`src/data/dates.rs`) mirrors DDR's `Dates` class
(`geodatazoo/dataclasses.py`), covering the bits the loader actually uses:

```rust
pub struct TimeAxis {
    pub start: NaiveDate,
    pub end: NaiveDate,       // inclusive
    pub num_days: usize,
}

impl TimeAxis {
    pub fn new(start: NaiveDate, end: NaiveDate) -> Self;
    pub fn sample_rho_window<R: Rng + ?Sized>(&self, rng: &mut R, rho_days: usize) -> RhoWindow;
    pub fn day_index(&self, date: NaiveDate) -> Option<usize>;
}
```

`new` builds an axis inclusive of both endpoints (`num_days =
(end - start) + 1`). `sample_rho_window` picks a contiguous `rho`-day
window uniformly at random (`random_start ~ U[0, num_days - rho)`),
mirroring DDR's `Dates.calculate_time_period` (`dataclasses.py:160-167`).
The returned `RhoWindow` carries the start day index, the rho count, and
the calendar date of the first day — enough state to slice the streamflow
and observation arrays along both daily and hourly axes.

**Daily ↔ hourly invariant:** when `rho` daily steps are selected, the
corresponding hourly range has `(rho - 1) * 24` entries — DDR's
`StreamflowReader.forward` relies on this when it does
`np.repeat(daily, 24)[:, :n_hourly]`. The Rust mirror is on `RhoWindow`:

```rust
impl RhoWindow {
    pub fn daily_range(&self) -> Range<usize> {
        self.start_day_idx..self.start_day_idx + self.rho_days
    }

    pub fn n_hourly(&self) -> usize {
        (self.rho_days.saturating_sub(1)) * 24
    }

    pub fn hourly_range(&self) -> Range<usize>;  // start_day_idx*24 .. + n_hourly
}
```

The test in `src/data/dates.rs` locks both halves of the contract:

```rust
#[test]
fn rho_window_n_hourly_is_rho_minus_1_times_24() {
    let w = RhoWindow {
        start_day_idx: 0,
        rho_days: 90,
        window_start: NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
    };
    assert_eq!(w.n_hourly(), 89 * 24);
    assert_eq!(w.daily_range(), 0..90);
}
```

Don't break that semantic — if you change `rho_days` accounting, both the
daily-resolution observation reader and the hourly-resolution forcing
reader silently misalign. Seeded sampling is reproducible: two RNGs with
the same seed draw the same window.

## DataError convention

Every variant of `DataError` (`src/data/error.rs`) carries a `PathBuf` so
error context survives wrapping:

```rust
#[derive(thiserror::Error, Debug)]
pub enum DataError {
    #[error("zarr read failed at {path}: {source}")]
    Zarr     { path: PathBuf, source: Box<dyn Error + Send + Sync> },
    #[error("netcdf read failed at {path}: {source}")]
    NetCdf   { path: PathBuf, source: netcdf::Error },
    #[error("icechunk read failed at {path}: {source}")]
    IceChunk { path: PathBuf, source: Box<dyn Error + Send + Sync> },
    #[error("io error at {path}: {source}")]
    Io       { path: PathBuf, source: std::io::Error },
    #[error("missing {missing}/{total} {kind} in store at {path}")]
    MissingIds { path: PathBuf, kind: &'static str, missing: usize, total: usize },
    #[error("malformed store at {path}: {message}")]
    Malformed { path: PathBuf, message: String },
    #[error("yaml parse error at {path}: {source}")]
    Yaml     { path: PathBuf, source: serde_yaml::Error },
    #[error("csv parse error at {path}: {source}")]
    Csv      { path: PathBuf, source: csv::Error },
}

pub type Result<T> = std::result::Result<T, DataError>;
```

DDR's stack traces (`KeyError: 'gage_id'` from a wrapped pandas read) are
notoriously hard to debug — paying the extra `PathBuf` field once here
means callers don't have to wrap every read with their own context.

## Gotchas

- **Zarr adjacency is eager; per-batch time-series reads are windowed.**
  `ConusAdjacencyStore` loads once (~30 MB), but `StreamflowStore` /
  `UsgsObservationsStore` slice a `RhoWindow` on demand. Don't
  pre-materialize the full attribute or forcing matrix — it doesn't fit
  cleanly into the training loop.
- **No `Box<dyn Store>` / no `trait Store`.** Premature unification was
  explicitly rejected (`src/data/mod.rs`): the five sources have different
  I/O models (sync zarr/netcdf vs async icechunk) and the call sites
  diverge too much. Each store is a focused module returning typed
  `ndarray::Array` buffers — composition over abstraction.
- **Gauge subgraphs differ per batch.** The active node count varies with
  the gauge pick; downstream code can't cache shapes across batches. The
  static CONUS state lives in `ConusAdjacencyStore`; per-batch state lives
  in whatever `GagesAdjacencyStore` returns for the chosen gauge.
- **Subgraph indices are CONUS-relative, not compressed.** A
  `GageSubgraph`'s `indices_0`/`indices_1` reference CONUS positions; the
  dataset compresses them at batch time when unioning subgraphs.
- **MERIT CONUS scale is small enough for consumer GPUs.** 346,321 reaches
  × 338,814 edges. Not millions — the port targets consumer GPUs. Don't
  assume a "production HPC" footprint when planning memory budgets.
- **Adjacency is topologically ordered, lower-triangular.** `rows[k] >=
  cols[k]` holds for every COO edge. The forward-substitution sparse
  solver assumes this. The regression test
  `data_zarr_store::conus_adjacency_loads_real_merit_zarr` asserts it
  against the on-disk zarr.

## Reference

Tests that lock the data-layer contracts:

| Path | Covered by |
|---|---|
| Zarr adjacency loads + topo-order invariant | `cargo test --test data_zarr_store conus_adjacency_loads_real_merit_zarr` |
| `GageSubgraph::upstream_comids` dedup/order | `cargo test --lib data::store::zarr::tests` |
| `Staid` zero-pad | `cargo test --lib data::ids::tests::staid_zfill_8` |
| `IdIndex` roundtrip | `cargo test --lib data::ids::tests::id_index_roundtrip` |
| `TimeAxis` + rho sampler | `cargo test --lib data::dates::` |

The CONUS-zarr test is the cross-cutting one: it both verifies the reader
and locks the lower-triangular invariant that the routing core depends on.

Source modules, all under `src/data/`:

- `store/zarr.rs` — `ConusAdjacencyStore`, `GagesAdjacencyStore`, `GageSubgraph`
- `store/netcdf.rs` — `AttributesStore`
- `store/icechunk.rs` — `StreamflowStore`, `UsgsObservationsStore`
- `ids.rs` — `Comid`, `Staid`, `IdIndex<T>`
- `dates.rs` — `TimeAxis`, `RhoWindow`, `Frequency`
- `error.rs` — `DataError`, `Result<T>`
- `mod.rs` — public re-exports and the anti-`trait Store` design notes

## See also

- [Setup](../setup.md) — the on-disk paths these readers resolve against.
- [Formatting inputs](inputs-formatting.md) — the YAML `data_sources:`
  block that wires these paths into the dataloader, plus the attribute
  variable names.
- [Graph objects](graph-objects.md) — how `ConusAdjacencyStore`'s COO
  triplets become the sparse routing pattern consumed by `MuskingumCunge`.
- [Architecture](../architecture.md) — module map showing where the data
  layer sits relative to the routing core.
- [Baseline](../reference/baseline.md) — the summed-Q′ reference that
  reads streamflow + observations through these same stores.
