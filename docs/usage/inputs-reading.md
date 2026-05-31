# Reading inputs

ddrs reads DDR's training data **in place** — there is no export or
conversion step. Five sources back the dataloader, each with a focused
module under `src/data/store/`. Reads return `ndarray::Array` buffers
keyed by `Comid` / `Staid` newtypes; no `trait Store` unifies them.
This chapter walks through each source, the newtype IDs, the
`TimeAxis` / `RhoWindow` sampler, and the unified `DataError`
convention that gives every failure a source path.

## The five data sources

| Source | Path | Crate |
|---|---|---|
| MERIT adjacency | `~/projects/ddr/data/merit_conus_adjacency.zarr` | `zarrs` |
| Per-gauge subgraphs | `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` | `zarrs` |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` | `netcdf` (TODO) |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` | `icechunk` (TODO) |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` | `icechunk` (TODO) |

CONUS MERIT is **346,321 reaches × 338,814 edges** — not millions;
consumer GPUs (24 GB VRAM is comfortable) handle it. `src/data/mod.rs`
owns the public re-exports; the dataset (`src/data/dataset.rs`) owns a
single `tokio::runtime::Runtime` and calls `block_on(...)` at the
icechunk boundary so the rest of ddrs stays sync.

## Zarr adjacency stores

Both adjacency targets are zarr v3 binsparse-COO with int32/uint8
arrays and `bytes` + `zstd` codecs (written by DDR's
`ddr_engine/core/zarr_io.py`). ddrs reads them via the `zarrs` crate
and never exposes `zarrs::Array` to callers — reads return `Vec<T>` or
`ndarray::Array1` with the foreign types contained.

### `ConusAdjacencyStore`

The full CONUS-wide graph + per-reach geometry. Loaded **once** at
dataset construction, eager (~30 MB zstd-compressed at 346K reaches).

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

`element i of order` is the COMID at zarr position `i`; downstream
stores (attributes, forcing) reuse this position-space via `IdIndex`.
The COO triplets `(indices_0, indices_1, adj_values)` describe the
sparse routing graph in lower-triangular form — every edge `(rows[k],
cols[k])` has `rows[k] >= cols[k]` after the topological sort.

Construction is a single fallible call:

```rust
let store = ConusAdjacencyStore::open(path)?;
```

which returns `Result<Self, DataError>` with a `DataError::Zarr {
path, source }` on failure. Path context is preserved end-to-end.

### `GagesAdjacencyStore`

Per-STAID subgraph COOs keyed by gauge. Eager-loaded for the
chosen-gauge set only (a few MB). Each batch picks a gauge; the
subgraph's `n_active` varies between batches, which is the dominant
reason `CudaPatternCache` is **per-instance** not global.

Construction follows the same `open(path)` pattern as
`ConusAdjacencyStore`.

## Netcdf catchment attributes

**Status: TODO** per `CLAUDE.md`. Planned reader uses the `netcdf`
crate (already a hard dependency — see the `DataError::NetCdf`
variant). Target file:
`~/projects/ddr/data/merit_global_attributes_v2.nc`. Output will be
an `ndarray::Array2<f32>` indexed by `Comid` position via the
`ConusAdjacencyStore`'s `IdIndex`.

The variable names (`SoilGrids1km_clay`, `aridity`, `meanelevation`,
`meanP`, `NDVI`, `meanslope`, `log10_uparea`, `SoilGrids1km_sand`,
`ETPOT_Hargr`, `Porosity`) come from
`config/merit_training.yaml::mlp.input_var_names` and feed the MLP
head per [Architecture](../architecture.md).

## Icechunk forcing + USGS

**Status: TODO** per `CLAUDE.md`. Planned readers use the `icechunk`
crate behind the dataset's tokio runtime. Targets:

- Streamflow forcing:
  `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic`
- USGS observations:
  `/mnt/ssd1/data/icechunk/usgs_daily_observations`

Both are async-first stores. The dataset's `block_on(...)` keeps the
rest of ddrs synchronous — the icechunk async surface stays inside
`src/data/dataset.rs`. `DataError::IceChunk { path, source }` is
already wired for these.

## Newtype IDs

DDR's Python uses raw `int` for COMIDs and raw `str` for STAIDs, which
has been a recurring bug surface (forgot-to-zfill mistakes,
COMID-vs-divide_id mixups). Newtypes in `src/data/ids.rs` let the
compiler catch those:

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

`Staid::new("1563500")` zero-pads to `"01563500"` to match DDR's
canonical form (`base_geodataset.py:35`, `readers.py:131`). The unit
test in `src/data/ids.rs` locks the contract:

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

Every store builds one of these at `open` time; every read consumes
one to map domain IDs to integer array positions:

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

`positions_of` is the workhorse — it returns both the resolved
positions and the indices of the requested IDs that were missing.
Callers decide whether to warn, error, or fill with sentinels. The
roundtrip test:

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
`Dates.calculate_time_period` (`dataclasses.py:160-167`). The returned
`RhoWindow` carries the start day index, the rho count, and the
calendar date of the first day — enough state to slice the streamflow
and observation arrays along both daily and hourly axes.

**Daily ↔ hourly invariant:** when `rho` daily steps are selected,
the corresponding hourly range has `(rho - 1) * 24` entries — DDR's
`StreamflowReader.forward` relies on this when it does
`np.repeat(daily, 24)[:, :n_hourly]`. The Rust mirror is on
`RhoWindow::n_hourly`:

```rust
impl RhoWindow {
    pub fn daily_range(&self) -> Range<usize> {
        self.start_day_idx..self.start_day_idx + self.rho_days
    }

    pub fn n_hourly(&self) -> usize {
        (self.rho_days.saturating_sub(1)) * 24
    }
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

Don't break that semantic — if you change `rho_days` accounting, both
the daily-resolution observation reader and the hourly-resolution
forcing reader silently misalign.

## DataError convention

Every variant of `DataError` (`src/data/error.rs`) carries a `PathBuf`
so error context survives wrapping:

```rust
#[derive(thiserror::Error, Debug)]
pub enum DataError {
    #[error("zarr read failed at {path}: {source}")]
    Zarr {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("netcdf read failed at {path}: {source}")]
    NetCdf {
        path: PathBuf,
        #[source]
        source: netcdf::Error,
    },

    #[error("icechunk read failed at {path}: {source}")]
    IceChunk {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("missing {missing}/{total} {kind} in store at {path}")]
    MissingIds {
        path: PathBuf,
        kind: &'static str,
        missing: usize,
        total: usize,
    },

    #[error("malformed store at {path}: {message}")]
    Malformed { path: PathBuf, message: String },

    #[error("yaml parse error at {path}: {source}")]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },

    #[error("csv parse error at {path}: {source}")]
    Csv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
}
```

DDR's stack traces (`KeyError: 'gage_id'` from a wrapped pandas read)
are notoriously hard to debug — paying the extra `PathBuf` field once
here means callers don't have to wrap every read with their own
context.

## Gotchas

- **Zarr stores opened lazily, slice on demand.**
  `ConusAdjacencyStore` is eager (load once, ~30 MB), but per-batch
  slices into attributes/forcing/observations are read on demand.
  Don't pre-materialize the full attribute matrix — it doesn't fit
  cleanly into the training loop.
- **No `Box<dyn Store>` / no `trait Store`.** Premature unification
  was explicitly rejected (`src/data/mod.rs`): the five sources have
  different I/O models (sync zarr/netcdf vs async icechunk) and the
  call sites diverge too much. Each store is a focused module
  returning typed `ndarray::Array` buffers.
- **Gauge subgraphs differ per batch.** `n_active` varies with the
  gauge pick; downstream code can't cache shapes across batches. The
  static CONUS state lives in `ConusAdjacencyStore`; per-batch state
  lives in whatever `GagesAdjacencyStore` returns for the chosen
  gauge.
- **MERIT CONUS scale is small enough for consumer GPUs.** 346,321
  reaches × 338,814 edges. Not millions — the port targets consumer
  GPUs. Don't assume a "production HPC" footprint when planning
  memory budgets.
- **Adjacency is topologically ordered, lower-triangular.** `rows[k]
  >= cols[k]` holds for every COO edge. The forward-substitution
  sparse solver in `src/sparse/mod.rs` assumes this. The regression
  test `data_zarr_store::conus_adjacency_loads_real_merit_zarr`
  asserts it against the on-disk zarr.

## Verification

| Path | Covered by |
|---|---|
| Zarr adjacency loads + topo-order invariant | `cargo test --test data_zarr_store conus_adjacency_loads_real_merit_zarr` |
| `Staid` zero-pad | `cargo test --lib data::ids::tests::staid_zfill_8` |
| `IdIndex` roundtrip | `cargo test --lib data::ids::tests::id_index_roundtrip` |
| `TimeAxis` + rho sampler | `cargo test --lib data::dates::` |

The CONUS-zarr test is the cross-cutting one: it both verifies the
reader and locks the lower-triangular invariant that the routing
core depends on.

## See also

- [Setup](../setup.md) — the on-disk paths referenced above.
- [Formatting inputs](inputs-formatting.md) — the YAML
  `data_sources:` block that wires these paths into the dataloader.
- [Graph objects](graph-objects.md) — how `ConusAdjacencyStore`'s
  COO triplets become `SparseAdjacency` → `CsrPattern` →
  `MuskingumCunge`.
- [Architecture](../architecture.md) — module map showing where the
  data layer sits relative to the routing core.
