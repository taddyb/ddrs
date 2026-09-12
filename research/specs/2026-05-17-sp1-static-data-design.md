# SP-1 design: static-data layer (NetCDF attributes + gage CSV + stats)

**Status:** Draft, pending user review
**Parent:** [`2026-05-17-train_and_test-replication-design.md`](./2026-05-17-train_and_test-replication-design.md)
**Mirrors:** `ddr/io/readers.py::AttributesReader`,
`ddr/io/readers.py::read_gage_info`, `ddr/io/statistics.py::set_statistics`,
`ddr/io/readers.py::fill_nans` / `naninfmean`.

## Why this sub-project

SP-1 unblocks every downstream sub-project. Concretely, SP-3's
`MeritGagesDataset::collate` needs:

- per-reach attribute matrix `(F, N_active)` for the MLP input
- per-attribute (mean, std) for normalization
- per-gauge metadata for headwater/DA_VALID filtering and `flow_scale`

These three artifacts have no dependency on icechunk, BURN, or the dataset
layer — they read flat files into `ndarray::Array` buffers. Doing SP-1 first
is the lowest-risk way to start.

## Scope

In scope:

1. NetCDF read of `merit_global_attributes_v2.nc` for a configurable subset of
   COMIDs and attribute names.
2. Gage CSV read of `references/gage_info/*.csv`.
3. Statistics: load the pre-computed JSON
   (`data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json`).
4. NaN-handling helpers (`fill_nans`, `naninfmean`) used by SP-3.

Explicitly **out of scope**:

- Re-computing statistics from scratch. DDR's `set_statistics` does this on
  cold paths; the JSON is checked into the repo and we just read it. If a
  user later needs recompute we add it — YAGNI for now.
- Statistics for non-MERIT geodatasets (Lynker hydrofabric).
- Streamflow / observations / adjacency — SP-2 and existing code.
- The `means`/`stds` BURN-tensor materialization. That happens in SP-3 at the
  device boundary; SP-1 returns plain `ndarray::Array1<f32>`.

## Components

### 1. `src/data/store/netcdf.rs` — `AttributesStore`

```rust
pub struct AttributesStore {
    pub path: PathBuf,
    /// Materialized attribute matrix, shape (F, N).
    pub attrs: Array2<f32>,
    /// Attribute names in row order. `attr_names[f]` is row `f` of `attrs`.
    pub attr_names: Vec<String>,
    /// COMID → column index in `attrs`. Maps each requested COMID to its
    /// position. COMIDs that were missing from the NetCDF are absent from
    /// this map (the caller can detect via `lookup_or_nan`).
    pub index: IdIndex<Comid>,
    /// Per-attribute fill means, length F. Filled by `naninfmean` over the
    /// raw NetCDF column. Used by `fill_nans` to fill per-row NaNs at
    /// batch time.
    pub row_means: Array1<f32>,
}

impl AttributesStore {
    /// Open the NetCDF file and materialize `(attr_names, comids)`.
    ///
    /// The materialized `attrs` matrix is f32 (cast from NetCDF's f64).
    /// COMIDs not found in the file are dropped from `index` — they will
    /// appear as NaN-filled rows at batch-attribute time via `fill_nans`.
    ///
    /// `attr_names` order is preserved in the row order of `attrs`.
    pub fn open(
        path: impl Into<PathBuf>,
        attr_names: &[String],
        comids: &[Comid],
    ) -> Result<Self>;
}
```

**Implementation sketch:**

1. Open NetCDF via the `netcdf` crate. The file has one dim `COMID` (length
   2,939,404 in the global file) and one coord variable `COMID` (int64).
2. Read the full `COMID` coord into a `Vec<i64>`. Build a `HashMap<i64, usize>`
   from COMID value → position. (One-time cost, ~25 MB at 2.94M entries.)
3. For each requested COMID, look up the position. Skip missing ones (build
   `(requested_pos, file_pos)` pairs).
4. For each `attr_name`, open the variable and read the values **at the
   requested file positions**. The `netcdf` crate supports `Variable::get_values`
   with `Extents`; for 1D vars indexed by COMID, we use a *strided gather*
   only if positions are contiguous-ish. Otherwise we read the full column
   `(2.94M f64)` and select — that's 23 MB per attribute, 10 attrs = 230 MB
   peak during open. Acceptable for a one-time startup cost.
5. Cast f64 → f32 at assembly time.
6. Compute `row_means[f] = naninfmean(full_column)` (mean of finite values).

**Concerns:**

- The `netcdf` v0.12 crate's `Variable::values_arr` API may not support fancy
  indexing. If not, we read the full column and select. Trade memory for
  simplicity.
- DDR's `_get_attributes` per-batch lookup is **slow** because xarray's
  `isel(COMID=valid_indices).compute()` re-reads from disk every batch. Our
  in-memory `Array2` is strictly better.
- NaN propagation: NetCDF `_FillValue` becomes Rust `f32::NAN`. Match DDR's
  semantics.

### 2. `src/data/store/gage_csv.rs` — `GageMetadata`

```rust
pub struct GageRow {
    pub staid: Staid,
    pub staname: String,
    pub drain_sqkm: f64,
    pub lat_gage: f64,
    pub lng_gage: f64,
    pub comid: Option<i64>,
    pub comid_drain_sqkm: Option<f64>,
    pub comid_unitarea_sqkm: Option<f64>,
    pub abs_diff: Option<f64>,
    pub da_valid: Option<bool>,
    pub flow_scale: Option<f32>,
}

pub struct GageMetadata {
    pub path: PathBuf,
    /// Insertion-ordered rows.
    pub rows: Vec<GageRow>,
    /// Lookup by STAID.
    pub by_staid: HashMap<Staid, usize>,
}

impl GageMetadata {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self>;

    /// Convenience: STAIDs in file order, zero-padded.
    pub fn staids(&self) -> Vec<Staid>;
}
```

**Implementation sketch:**

1. `csv::Reader::from_path`; iterate rows.
2. STAID column → `Staid::new` (handles the zero-padding; already implemented).
3. `DA_VALID` parsing accepts `"True"`/`"False"`/`"1"`/`"0"`.
4. Optional columns get `None` if missing from the header — match DDR's
   "optional columns" semantics in `read_gage_info`.
5. Build the `HashMap` after collecting all rows.

**Verification:** integration test against `references/gage_info/gages_3000.csv`.
Assert: 3000 rows; all `STAID` zero-padded to length 8; `DA_VALID` field
parsed correctly.

### 3. `src/data/statistics.rs` — `AttrStats` + helpers

```rust
/// Loaded from `data/statistics/merit_attribute_statistics_*.json`.
pub struct AttrStats {
    /// Attribute name → (min, max, mean, std, p10, p90). All f64.
    pub by_name: HashMap<String, AttrStatRow>,
}

#[derive(Clone, Debug)]
pub struct AttrStatRow {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub p10: f64,
    pub p90: f64,
}

impl AttrStats {
    /// Read DDR's pre-computed JSON.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self>;

    /// Convenience: means in attribute order, as f32.
    pub fn means_f32(&self, attr_names: &[String]) -> Array1<f32>;
    pub fn stds_f32(&self,  attr_names: &[String]) -> Array1<f32>;
}

/// Replace NaNs with the per-row mean. Mirrors `fill_nans` in
/// `ddr/io/readers.py:332-368`.
///
/// - For a 1D tensor, `row_means.len() == 1` (or matches `attr.len()`).
/// - For a 2D `(F, N)` tensor, `row_means.len() == F` and broadcasts across N.
pub fn fill_nans(attr: ArrayViewMut2<f32>, row_means: &Array1<f32>);
pub fn fill_nans_1d(attr: ArrayViewMut1<f32>, row_mean: f32);

/// Mean over the finite values of an array. Returns NaN if no finite values.
/// Mirrors `naninfmean` in `ddr/io/readers.py:315-330`.
pub fn naninfmean(arr: &[f32]) -> f32;
```

**JSON layout (already in repo):**

```json
{
  "SoilGrids1km_clay": {
    "min": 2.59, "max": 52.78, "mean": 23.49, "std": 8.22,
    "p10": 12.77, "p90": 34.60
  },
  ...
}
```

**Implementation sketch:**

1. `serde_json` (add to `Cargo.toml`); deserialize directly into
   `HashMap<String, AttrStatRow>`.
2. `means_f32` / `stds_f32`: iterate `attr_names`, look up, cast.

### 4. Module surface (`src/data/mod.rs` re-exports)

```rust
pub use store::{AttributesStore, GageMetadata, GageRow};
pub use statistics::{AttrStats, AttrStatRow, fill_nans, fill_nans_1d, naninfmean};
```

## Verification protocol

Per the project-wide guidance: trust the cargo crates to read the same bytes
DDR's xarray reads; do not introduce a Python-side fixture export to
bit-validate reader output. Verification at the SP-1 layer is just sanity:
shape, count, and a couple of sampled known values. Real verification lives
at the SP-3 alignment layer (batch construction) and the SP-4 loss layer.

### Unit tests

- `statistics::naninfmean`: edge cases (all-NaN, all-Inf, mixed).
- `statistics::fill_nans`: 1D and 2D with hand-crafted inputs.
- `AttrStats::open`: read the checked-in JSON, assert a couple of known
  (attr_name → mean, std) pairs from the file.
- `GageMetadata::open`: parse `references/gage_info/gages_3000.csv`, assert
  3000 rows, a sampled STAID's fields, and that optional columns
  (`COMID_DRAIN_SQKM`, `DA_VALID`, `FLOW_SCALE`) are populated when present.

### Integration test against the live data sources

`tests/data_static.rs`:

1. Open `merit_global_attributes_v2.nc` with the 10 MERIT attributes from
   `config/merit_training.yaml` and the COMIDs from
   `ConusAdjacencyStore::open(...).order` (limited to the first ~1000 for
   test speed).
2. Assert: `attrs.shape == (10, ~1000)`, all `attr_names` present in the
   NetCDF, materialized matrix has the right dtype (f32), `row_means` are
   finite, and a small spot-check (e.g. `meanelevation` for a single known
   COMID against an inspection done at design time).
3. No fixture file, no Python export. Just open the production NetCDF and
   sanity-check.

## Concerns

- **`netcdf` v0.12 crate API ergonomics.** I have not personally read its
  fancy-indexing support. If `Variable::values_arr(Some(&Extents::...))`
  refuses non-contiguous selections, we fall back to "read full column, then
  select" — a 23 MB transient per attribute, 230 MB peak for 10 attributes.
  Acceptable. Documented as a planned simplification in the implementation
  plan.
- **HDF5 dependency.** The `netcdf` crate links to libnetcdf which links to
  libhdf5. The build assumes these are available on the dev machine (they
  are — DDR uses them too). If we ever want a pure-Rust read, we'd swap to
  `hdf5-metno` or write a `.nc → .zarr` pre-pass. Out of scope here.
- **Statistics JSON path is hard-coded in config.** Same as DDR
  (`data_sources.statistics`). We mirror it.
- **The optional-columns logic for the gage CSV requires careful header
  inspection.** DDR's `read_gage_info` checks `set(expected) ⊆ set(columns)`
  and treats unknown columns as ignorable. We use `serde`'s `#[serde(default)]`
  + `Option<T>` for the optional ones, but the CSV crate is row-by-row so
  we must read the header explicitly to know which optional fields exist.

## Assumptions

1. The pre-computed statistics JSON at
   `data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json`
   is authoritative. If a user changes attribute lists, they regenerate it
   via DDR.
2. f64 → f32 cast at the NetCDF boundary is acceptable. DDR also casts (its
   tensors are f32 throughout the routing core; the cast happens in
   `_get_attributes`'s `device=cfg.device, dtype=torch.float32`). The
   precision floor we target is f32; no fidelity loss.
3. `DA_VALID` column is present in the production CSVs (verified above for
   all four reference files). The implementation treats absence as a hard
   error in training mode — matching DDR's behavior when neither `DA_VALID`
   nor `max_area_diff_sqkm` is available.
4. `STANAME` may be missing (DDR's reader tolerates this by aliasing
   `STANAME ← STAID`). We mirror that aliasing.

## Module layout summary

```
src/data/
├── mod.rs                            (+) re-export new types
├── store/
│   ├── mod.rs                        (+) re-export AttributesStore, GageMetadata
│   ├── netcdf.rs                     (new) AttributesStore
│   └── gage_csv.rs                   (new) GageMetadata, GageRow
├── statistics.rs                     (new) AttrStats, fill_nans, naninfmean
└── ...

tests/
└── data_static.rs                    (new) integration tests against live files

Cargo.toml                            (+) netcdf (already), serde_json
```

No fixture export, no `scripts/` additions — readers exercise the production
files directly.

## Dependencies to add

```toml
netcdf = "0.12"   # already present, listed for completeness
serde_json = "1"  # new — for stats JSON
```

(`csv` and `serde` are already present.)

## What I'm NOT going to do (and why)

- **No `trait DataSource`.** Per the data-layer convention: concrete types,
  no premature unification.
- **No async in SP-1.** NetCDF and CSV reads are sync. Tokio enters at SP-2
  (icechunk).
- **No statistics recompute.** Cached JSON is authoritative. Adding a
  recompute path now is YAGNI.
- **No `Box<dyn>` for stat rows.** Just a plain struct.
- **No I/O retries.** Local-FS reads; bail with a `DataError` on first
  failure.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `netcdf` crate doesn't support fancy indexing | Medium | Fall back to full-column read + select. Already factored into the design. |
| HDF5 link errors on a different dev machine | Low | Out of scope — same dependency as DDR. |
| JSON layout changes | Low | DDR has not changed it; pinned by the integration test. |

## Open questions for review

None — this is small enough to spec in full. The implementation plan that
follows will turn this into ~8 ordered subtasks.

## Next step after approval

Invoke writing-plans to generate the implementation plan for SP-1, then
execute via subagent-driven development.
