# Reading inputs

ddrs reads DDR's training data **in place** — there is no export or
conversion step. Eleven focused modules under `src/data/store/` back the
dataloader, one per on-disk format. Reads return `ndarray::Array`
buffers keyed by `Comid` / `Staid` newtypes; there is deliberately no
`trait Store` unifying them. This chapter walks through each reader, the
newtype IDs that index them, the `TimeAxis` / `RhoWindow` sampler, and
the `DataError` convention that gives every failure a source path.

## What it is

The data layer (`src/data/`) is the boundary between DDR's heterogeneous
on-disk formats and ddrs's sync routing core. Each source has a different
I/O model — sync zarr v2 and v3, sync netCDF, async-first icechunk, CSV —
so each gets its own small module rather than a shared abstraction.
`src/data/store/mod.rs` declares **eleven** store modules
(`camels_hourly`, `gage_csv`, `icechunk`, `netcdf`, `obs_writer`,
`param_dump`, `state_cache`, `zarr`, `zarr_aorc`, `zarr_obs`,
`zarr_qprime`) and re-exports fourteen names from them, plus two
format-dispatching enums (`ObservationsStore`, `StreamflowSource`)
defined in `mod.rs` itself.

### Readers reached from `data_sources:`

These are the config keys (see [Formatting inputs](inputs-formatting.md))
and the reader each one opens. Paths are the values shipped in
`config/merit_training.yaml` where that file sets the key.

| `data_sources` key | Example path | Reader |
|---|---|---|
| `attributes` (required) | `~/projects/ddr/data/merit_global_attributes_v2.nc` | `netcdf` — `AttributesStore::open` (`store/netcdf.rs`); a YAML **list** routes to `AttributesStore::open_multi` |
| `streamflow` (required) | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` | `StreamflowSource` (`store/mod.rs`) — sniffs icechunk (`StreamflowStore`) vs multi-zone zarr v2 (`GlobalStreamflowStore`) |
| `observations` (required) | `/mnt/ssd1/data/icechunk/usgs_daily_observations` | `ObservationsStore` (`store/mod.rs`) — sniffs icechunk (`UsgsObservationsStore`) vs zarr v2 group (`GlobalObservationsStore`) |
| `gages` (required) | `~/projects/ddr/references/gage_info/gages_3000.csv` | `csv` — `GageMetadata` (`store/gage_csv.rs`); also accepts a directory of per-zone CSVs |
| `geospatial_fabric` (managed adjacency) | `.../riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp` | `dbase` / `rusqlite` via `adjacency::cache::resolve_or_build`, which writes the two zarr stores below into `.ddrs/adjacency/<key>/` |
| `conus_adjacency` (explicit adjacency) | a pre-built zarr store | `zarrs` — `ConusAdjacencyStore` (`store/zarr.rs`) |
| `gages_adjacency` (explicit adjacency) | a pre-built zarr store | `zarrs` — `GagesAdjacencyStore` (`store/zarr.rs`) |
| `aorc_precip` (optional) | `/mnt/ssd1/data/aorc/merit_unit_catchments.zarr` | `zarrs` — `AorcPrecipStore` (`store/zarr_aorc.rs`) |

**Attribution caveat:** `config/merit_training.yaml` sets only
`attributes`, `geospatial_fabric`, `streamflow`, `observations`, and
`gages`. It contains **neither** adjacency key — the
`loads_merit_training_yaml` test in `src/config.rs` asserts
`conus_adjacency.is_none()` and `gages_adjacency.is_none()` — and no
`aorc_precip`. The adjacency zarr paths in the table above are what a
Strategy-B config supplies, or what the managed build materializes.

### Readers with no `data_sources` key

| Reader | Module | Reached from |
|---|---|---|
| `StateCache` | `store/state_cache.rs` | `experiment.state_cache:` — a day-boundary discharge netCDF (`(day, COMID)` `q_state` f32 + `day0` global attr), read one ~256 KB row at a time via `row_for_day` rather than materializing the ~1.3 GB array |
| `CamelsHourlyStore` | `store/camels_hourly.rs` | Real hourly USGS discharge + NLDAS precip netCDF (`(basin, date)`), used to pretrain the disaggregation head. **Not** on the production training/eval path |
| `load_comid_field` | `store/param_dump.rs` | COMID-keyed routing-parameter netCDF dumps (`--donor-params-nc`); returns a `HashMap<i64, f32>` because donor and target dumps are not guaranteed to share row order |
| `write_obs_zarr_v2` | `store/obs_writer.rs` | A **writer**, not a reader — emits a minimal zarr-v2 observations store for fixtures and synthetic experiments |

CONUS MERIT is **346,321 reaches × 338,814 edges** — not millions;
consumer GPUs (24 GB VRAM is comfortable) handle it. Backend types
(`zarrs::Array`, `netcdf::Variable`, `icechunk::Store`) never escape
their modules — callers see only `ndarray` and `data::ids` types.

## Adjacency: managed build vs pre-built zarr

Adjacency is an either/or, enforced at config load by
`validate_data_sources` (`src/config.rs`):

- **Managed build** — set `geospatial_fabric` and omit *both* adjacency
  keys. The first `ddrs plan` reads the fabric's attribute table (`.shp`
  → sibling `.dbf`, or a `.gpkg` via SQL; geometry is never opened),
  topologically sorts it, builds the CONUS graph plus every per-gauge
  subgraph, and writes both zarr stores into
  `.ddrs/adjacency/<key>/{conus,gages}`. Subsequent plans are cache hits.
- **Pre-built zarr** — set *both* `conus_adjacency` and `gages_adjacency`
  and omit `geospatial_fabric`.
- Exactly one adjacency key, or neither key and no fabric, is a load-time
  error.

The cache key is `blake3(fabric_fingerprint ∥ gages_fingerprint ∥
[layer] ∥ BUILDER_VERSION)` truncated to 16 hex chars, where the
fingerprints are content hashes of the file bytes — moving or renaming a
fabric does **not** invalidate the cache, and two byte-identical fabrics
share one entry. The optional gpkg `layer` participates only when set
(a multi-layer gpkg has identical bytes regardless of which layer you
pick). Builds are crash-safe: everything lands in
`<root>/adjacency/.tmp-<key>` and is atomically renamed into place. Entry
point: `adjacency::cache::resolve_or_build(workspace_root, fabric,
fabric_layer, gages_csv)` (`src/adjacency/cache.rs`).

Either way, what the dataset ultimately opens is the pair of zarr stores
described next — the managed builder produces output that matches an
engine-built store element-for-element (`tests/adjacency_parity.rs`).

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

impl GageSubgraph {
    /// True when the gauge's catchment is a single MERIT divide — the
    /// subgraph has zero edges. Training and the baseline both drop these.
    pub fn is_headwater(&self) -> bool { self.indices_0.is_empty() }

    /// Unique upstream COMIDs, sorted by CONUS position (stable across runs).
    /// EMPTY for a headwater subgraph — filter with `is_headwater` first.
    pub fn upstream_comids(&self, conus: &ConusAdjacencyStore) -> Vec<Comid>;
}
```

A subgraph's COO indices reference **CONUS** positions, not compressed
positions — the dataset compresses at batch time when it unions multiple
gauges' subgraphs. Each batch picks a gauge, so the active node count
varies between batches — see [Gotchas](#gotchas).

`is_headwater()` is load-bearing in two places, and both must agree:
`MeritGagesDataset::open` uses it (inlined as
`g.indices_0.is_empty()`) as the third stage of its gauge filter, and
`summed_q_prime` calls it to skip the same gauges. A zero-edge subgraph
has an empty `upstream_comids`, and summing an empty set silently yields
an all-zero prediction — which is why the baseline must skip rather than
impute. `single_divide_subgraph_is_headwater_and_has_empty_upstream`
(`src/data/store/zarr.rs`) locks the contract.

### The three-stage gauge filter

`MeritGagesDataset::open` (`src/data/dataset.rs`) narrows the gauge list
in three passes, each logged to stderr so a run's `run.log` records the
survivor counts:

1. **`DA_VALID` drop** — rows whose `DA_VALID` column is explicitly
   `false` are dropped. Rows *without* the column pass (the v3.1 global
   per-zone CSVs carry no `DA_VALID`), so absence is not failure. Logs
   `DA_VALID filter: kept N/M gauges`.
2. **Adjacency presence** — the survivors are passed to
   `GagesAdjacencyStore::open`, and any STAID with no subgroup in the
   store is dropped (mirrors DDR's `valid_gauges_mask`).
3. **Headwater drop** — any surviving subgraph with `indices_0.is_empty()`
   is dropped. Stages 2 and 3 log together:
   `gages_adjacency filter: kept N gauges (dropped X missing, Y headwater)`.

Downstream consumers must reproduce this population exactly; a baseline
computed over the unfiltered gauge list is not comparable to a trained
model's metrics.

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

### `open_multi` — several attribute files, one matrix

`data_sources.attributes` is a `Vec<PathBuf>`; a bare scalar path parses
as a one-element list. With two or more paths the dataset calls
`AttributesStore::open_multi(&paths, &attr_names, &comids)` instead,
which **feature-concatenates** rather than row-concatenates:

```rust
pub fn open_multi(
    paths: &[PathBuf],
    attr_names: &[String],
    comids: &[Comid],
) -> Result<Self>;
```

Pass 1 probes every file to decide which store *owns* each requested
variable. A variable found in two stores is a hard `DataError::Malformed`
("each variable must belong to exactly one store"); a variable found in
none is likewise an error. Pass 3 opens each store COMID-aligned to the
full requested list and copies its rows into a merged `(F, N)` matrix
pre-filled with `NaN` — so a COMID present in store A but absent from
store B keeps A's values and carries `NaN` for B's variables. The
resulting `index` covers **all** requested COMIDs, and column order
matches the requested order exactly
(`open_multi_two_store_alignment`, `src/data/store/netcdf.rs`). An empty
`attributes` list is rejected at config-deserialize time, before any file
is opened.

## Icechunk forcing + USGS observations

`StreamflowStore` and `UsgsObservationsStore` (`src/data/store/icechunk.rs`)
read the two CONUS time-series sources from local icechunk repositories. Because
the `icechunk` crate has no `zarrs` dependency, the module wraps an
`icechunk::Store` behind an `IcZarrStorage` adapter implementing zarrs's
`ReadableStorageTraits`; each store opens a read-only session on the `main`
branch and owns a `tokio::runtime::Runtime`, calling `block_on(...)` at the
icechunk boundary so the rest of ddrs stays sync:

```rust
pub struct StreamflowStore {
    pub path: PathBuf,
    pub index: IdIndex<Comid>,     // COMID -> column
    /// First calendar day covered (hourly stores: the day holding hour 0).
    pub time_start: NaiveDate,
    /// Length of the NATIVE axis: days (daily store) or hours (hourly store).
    pub n_time: usize,
    /// Native axis resolution, sniffed from the CF `units` attribute.
    pub resolution: Frequency,
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

### `resolution` — the daily/hourly sniff

`StreamflowStore::open` parses the `/time` coordinate's CF `units`
attribute via `parse_cf_units`, which accepts **either** `"days since …"`
or `"hours since …"` and returns the epoch plus a `Frequency`
(`Daily | Hourly`, `src/data/dates.rs`). That `resolution` field — not
the config, not the caller — is what decides how every subsequent read
slices:

| | `Frequency::Daily` | `Frequency::Hourly` |
|---|---|---|
| `read_window` | reads daily, then upsamples with repeat-24 + trailing-day trim (`daily_to_hourly_trim`) | slices the native hourly axis directly; no upsampling |
| `read_window_daily` | direct slab read | averages each day's 24 hours (Q′ is a rate, so the daily value is the day's mean flow), in 32-day chunks to bound the transient allocation |
| `open` extras | — | `validate_hourly_axis` enforces hour-0 alignment |

`UsgsObservationsStore` has no `resolution` field: observations are daily
by construction and its `open` parses only `"days since …"`. Dataset open
logs the sniffed value (`streamflow resolution: Daily|Hourly`) — check it
when validating a run.

### Windowed reads

```rust
let qr  = StreamflowStore::open(streamflow_path)?;
let obs = UsgsObservationsStore::open(observations_path)?;

// (n_hourly, N) — TIME-MAJOR: hours down the rows, reaches across
let forcing: Array2<f32> = qr.read_window(&window, &comids)?;

// (n_days, N) — takes a bare (start, n_days) pair, NOT a &RhoWindow
let daily: Array2<f32> =
    qr.read_window_daily(window.window_start, window.rho_days, &comids)?;

// (rho_days, G) — observations are already daily, no hourly transform
let targets: Array2<f32> = obs.read_window(&window, &staids)?;
```

Three things to get right:

- **Shapes are time-major.** `read_window` returns `(n_hourly, N)` —
  hours first — not `(n_reach, T)`.
- **`read_window_daily` does not take a `RhoWindow`.** Its signature is
  `(window_start: NaiveDate, n_days: usize, ids: &[…])`, because the
  summed-Q′ baseline calls it with a 15-year span that is not a
  rho-window. `read_window` is the `RhoWindow`-shaped wrapper.
- **Only `StreamflowStore` has `read_test_window`.**
  `read_test_window(&TestWindow, &comids)` reads `n_days * 24` hours with
  no trailing-day trim so eval chunks tile cleanly.
  `UsgsObservationsStore` exposes only `open`, `read_window_daily`, and
  `read_window`.

Missing IDs are handled differently by design: `StreamflowStore` fills
COMIDs absent from the store with `0.001` m³/s (mirrors DDR's
`readers.py:464-468`), while the observation readers raise
`DataError::MissingIds` — a missing observation series is a config bug,
a missing Q′ prediction is expected. An icechunk-level failure yields
`DataError::IceChunk { path, source }`. `qr_units()` exposes the `/Qr`
`units` attribute, used by `ddrs import` to check the m³/s contract.

## Format-dispatching enums

`data_sources.streamflow` and `data_sources.observations` are not opened
through the structs above directly. `MeritGagesDataset` goes through two
enums declared in `src/data/store/mod.rs` that sniff the on-disk format
and static-dispatch — a closed set, per the no-`Box<dyn Store>` rule:

| Enum | Variants | Sniff |
|---|---|---|
| `StreamflowSource` | `Icechunk(StreamflowStore)`, `GlobalZarr(GlobalStreamflowStore)` | zone groups (`.zgroup` + `streamflow/.zarray`, at the root or one level down) ⇒ global zarr v2; anything else ⇒ icechunk |
| `ObservationsStore` | `Usgs(UsgsObservationsStore)`, `Global(GlobalObservationsStore)` | `.zgroup` at the root ⇒ zarr v2 group; anything else ⇒ icechunk |

`StreamflowSource` forwards `read_window`, `read_window_daily`,
`read_test_window`, and `resolution()` (which reports
`Frequency::Daily` for the global zarr layout — daily by construction).
`ObservationsStore` forwards `read_window` / `read_window_daily` and adds
`contains(&staid)`, used to pre-filter gauge sets so reads never hit
`MissingIds` (gage CSVs can list gauges the observation product lacks —
63 of 5,975 in the v3.1 global set).

## Global zarr v2 stores

The non-CONUS arms of those enums live in their own modules. Neither
store carries units metadata on disk; the facts below were established
empirically and are recorded in the module docs.

### `GlobalStreamflowStore` (`store/zarr_qprime.rs`)

One zarr v2 group per pfaf-2 zone (`11` … `86`, 60 zones globally), each
holding `streamflow` `(time, COMID)` f64 (**time-major**, the transpose
of the AORC store), a `time` coord with CF `"days since 1980-01-01
00:00:00"`, and an int64 `COMID` coord whose first two digits are the
zone. Units are m³/s. COMIDs absent from the store are filled with
`0.001`, matching `StreamflowStore`. `n_comids()` / `n_zones()` report
the store's extent.

### `GlobalObservationsStore` (`store/zarr_obs.rs`)

A zarr v2 directory group holding one 1-D f64 array per gage, named
`<Provider>__<StationId>` (`USGS__01030500`, `BOMAustralia__003204A`);
6,051 gages from 25+ providers. Every array is `shape [14976]`,
blosc-lz4, single chunk. Units are m³/s, missing data is `NaN`, and
there is **no** time coordinate anywhere — the axis is implicit: daily,
1980-01-01 through 2020-12-31. `open_with_epoch` overrides that implicit
start when a store uses a different one. Read contract matches
`UsgsObservationsStore`: missing STAIDs are a hard `DataError::MissingIds`.

## Hourly AORC precipitation

`AorcPrecipStore` (`store/zarr_aorc.rs`) reads the zarr **v3**
`merit_unit_catchments.zarr` store behind the optional `aorc_precip`
data source. Layout: `total_precipitation` `(catchment, time)` f32,
**catchment-major**, chunks `(n_catchments, 48)`, `bytes`+`zstd`, fill
`0.0`, units **mm/hr**; `gauge_id` fixed-length UTF-32 strings that are
MERIT COMIDs; `date` hourly from 1980-01-01T00:00 to 2020-12-31T23:00
(14,976 days = 359,424 hours), so hour rows `[t·24 … (t+1)·24)` are day
`t` — byte-aligned with the streamflow Q′ axis.

It is **CONUS-only**: the AORC fabric covers 290,878 of the 346,321
CONUS MERIT reaches, and COMIDs absent from the store are filled with
`0.0` (dry-equivalent, so the disaggregation head's softmax sees a flat
window and falls back to the daily-Q / attribute shape). `coverage(&comids)`
reports how many of a requested set are actually present. Precip is a
*shape* signal, not a flow — it is never flow-scaled, and is normalized
per-reach in `src/data/dataset.rs` before the head sees it. Reads:
`read_window` / `read_test_window` / `read_window_hourly`.

## Gage metadata CSV

`GageMetadata` (`store/gage_csv.rs`) reads the **required** `gages`
source, mirroring DDR's `read_gage_info`. Required columns: `STAID`,
`STANAME`, `DRAIN_SQKM`, `LAT_GAGE`, `LNG_GAGE`. Optional: `COMID`,
`COMID_DRAIN_SQKM`, `COMID_UNITAREA_SQKM`, `ABS_DIFF`, `DA_VALID`,
`FLOW_SCALE`. `STAID` values are zero-padded to 8 characters at
construction via `Staid::new`. `open` accepts a single `.csv` **or** a
directory, in which case every `*.csv` in filename order is concatenated
(this is how the 57 per-zone global gage CSVs are consumed). A CSV-level
failure yields `DataError::Csv { path, source }`. The `DA_VALID` column
drives the first stage of the gauge filter above.

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
    #[error("eval chunk {chunk}/{total} looks corrupted ({message}); context path {path:?} — ...")]
    CorruptedEvalChunk { path: PathBuf, chunk: usize, total: usize, message: String },
}

pub type Result<T> = std::result::Result<T, DataError>;
```

DDR's stack traces (`KeyError: 'gage_id'` from a wrapped pandas read) are
notoriously hard to debug — paying the extra `PathBuf` field once here
means callers don't have to wrap every read with their own context.

`CorruptedEvalChunk` is the odd one out: it is not an I/O failure but a
*plausibility* failure, raised when an eval chunk's values look wrong.
Its `#[error]` string spells out the usual cause — a silent GPU
worker-thread failure (e.g. a cubecl OOM in a background thread, which
logs-and-drops the task instead of propagating) — and the remedy: retry
with `--backend cpu` or a smaller `batch_size_days`. If you see it, the
data on disk is probably fine.

## Gotchas

- **Zarr adjacency is eager; per-batch time-series reads are windowed.**
  `ConusAdjacencyStore` loads once (~30 MB), but `StreamflowStore` /
  `UsgsObservationsStore` slice a `RhoWindow` on demand. Don't
  pre-materialize the full attribute or forcing matrix — it doesn't fit
  cleanly into the training loop.
- **No `Box<dyn Store>` / no `trait Store`.** Premature unification was
  explicitly rejected (`src/data/mod.rs`): the sources have different
  I/O models (sync zarr v2/v3, sync netcdf, async icechunk, CSV) and the
  call sites diverge too much. Each store is a focused module returning
  typed `ndarray::Array` buffers — composition over abstraction. Where
  two formats *do* share a read contract (`streamflow`, `observations`),
  the unification is a closed-set `enum` with static dispatch
  (`StreamflowSource`, `ObservationsStore`), never a trait object.
- **Time-series reads are time-major.** `read_window` returns
  `(n_hourly, N)` and `read_window_daily` returns `(n_days, N)`. Reaches
  are the *columns*.
- **`StreamflowStore::resolution` is sniffed, not configured.** An
  hourly-native Q′ store is sliced directly and must not be combined with
  a `kan_head.disaggregation` block; a daily store is repeat-24
  upsampled. Read the `streamflow resolution: …` line in the run log
  before trusting a forcing comparison.
- **Headwater gauges must be skipped, not imputed.**
  `GageSubgraph::is_headwater()` is true for zero-edge subgraphs, whose
  `upstream_comids` is empty — summing an empty set yields a silent
  all-zero prediction. Training drops them; any baseline or diagnostic
  must drop the same ones or the populations aren't comparable.
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

| Contract | Covered by |
|---|---|
| Zarr adjacency loads + topo-order invariant | `cargo test --test data_zarr_store conus_adjacency_loads_real_merit_zarr` |
| `GageSubgraph::upstream_comids` dedup/order + `is_headwater` | `cargo test --lib data::store::zarr::tests` |
| Icechunk axes + `read_window` shape | `cargo test --lib data::store::icechunk::tests` |
| `AttributesStore::open_multi` alignment / ambiguity / missing-var | `cargo test --lib data::store::netcdf::tests` |
| AORC precip store opens + reads aligned | `cargo test --lib data::store::zarr_aorc::tests` |
| Managed adjacency builder (topo sort, cache key, gpkg) | `cargo test --test adjacency_build` |
| Managed build matches an engine-built store element-for-element | `cargo test --test adjacency_parity -- --ignored` (`#[ignore]`d: reads the real ~108 MB pfaf_7 `.dbf`, ~10 s) |
| `Staid` zero-pad | `cargo test --lib data::ids::tests::staid_zfill_8` |
| `IdIndex` roundtrip | `cargo test --lib data::ids::tests::id_index_roundtrip` |
| `TimeAxis` + rho sampler | `cargo test --lib data::dates::` |

Note the first row is an **integration** test (`tests/data_zarr_store.rs`),
so it needs `--test data_zarr_store`, not `--lib`. Running
`cargo test --lib data::store::zarr::tests::conus_adjacency_loads_real_merit_zarr`
matches nothing and reports a green "0 passed".

The CONUS-zarr test is the cross-cutting one: it both verifies the reader
and locks the lower-triangular invariant that the routing core depends on.

Source modules, all under `src/data/`:

- `store/mod.rs` — module declarations, public re-exports, the
  `ObservationsStore` / `StreamflowSource` dispatch enums, and the
  anti-`trait Store` design notes
- `store/zarr.rs` — `ConusAdjacencyStore`, `GagesAdjacencyStore`, `GageSubgraph`
- `store/netcdf.rs` — `AttributesStore` (`open`, `open_aligned`, `open_multi`)
- `store/icechunk.rs` — `StreamflowStore`, `UsgsObservationsStore`
- `store/zarr_qprime.rs` — `GlobalStreamflowStore`
- `store/zarr_obs.rs` — `GlobalObservationsStore`
- `store/zarr_aorc.rs` — `AorcPrecipStore`
- `store/gage_csv.rs` — `GageMetadata`, `GageRow`
- `store/state_cache.rs` — `StateCache`
- `store/camels_hourly.rs` — `CamelsHourlyStore`
- `store/param_dump.rs` — `load_comid_field`
- `store/obs_writer.rs` — `write_obs_zarr_v2` (writer)
- `dataset.rs` — `MeritGagesDataset`, the three-stage gauge filter
- `ids.rs` — `Comid`, `Staid`, `IdIndex<T>`
- `dates.rs` — `TimeAxis`, `RhoWindow`, `Frequency`
- `error.rs` — `DataError`, `Result<T>`

Adjacency construction lives one level up, in `src/adjacency/`
(`cache::resolve_or_build`, `build::topological_sort`).

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
