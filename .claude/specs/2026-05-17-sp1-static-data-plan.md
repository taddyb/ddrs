# SP-1 Static-Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three "static" data-layer pieces needed to feed
`MeritGagesDataset` later: NetCDF attribute reader, gage CSV reader, and
pre-computed statistics loader. Plus the small NaN-handling helpers used by
later batch-construction code.

**Architecture:** Three focused modules under `src/data/` — each takes a path
and returns owned `ndarray::Array` buffers plus domain-typed metadata. No
async, no `trait`, no `Box<dyn>`. Verification is sanity-only (shape, count,
sampled values) against the production files in `~/projects/ddr/`; no
fixture-export step.

**Tech Stack:** `netcdf` v0.12 (HDF5-backed), `csv` v1 + `serde`,
`serde_json` (new), `ndarray` v0.16, existing `DataError` + `IdIndex<T>`.

**Spec:** `.claude/specs/2026-05-17-sp1-static-data-design.md`

**Parent spec:** `.claude/specs/2026-05-17-train_and_test-replication-design.md`

**DDR reference files (read-only, cite line numbers in comments):**
- `~/projects/ddr/src/ddr/io/readers.py` — `read_gage_info` (~lines 100-160),
  `fill_nans` (~lines 332-368), `naninfmean` (~lines 315-330),
  `AttributesReader` (~lines 470+).
- `~/projects/ddr/src/ddr/io/statistics.py` — `set_statistics` (the JSON
  layout we consume).

**Repository paths used by tests:**
- NetCDF: `/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc`
- Statistics JSON:
  `/home/tbindas/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json`
- Gage CSV:
  `/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv`

These paths match `config/merit_training.yaml`. Integration tests gate on
their presence (skip with a log if absent) so CI on machines without the
data still passes.

---

## File Structure

**Created:**
- `src/data/statistics.rs` — `AttrStats`, `AttrStatRow`, `naninfmean`,
  `fill_nans_1d`, `fill_nans`
- `src/data/store/gage_csv.rs` — `GageRow`, `GageMetadata`
- `src/data/store/netcdf.rs` — `AttributesStore`
- `tests/data_static.rs` — integration tests against production files

**Modified:**
- `Cargo.toml` — add `serde_json = "1"`
- `src/data/mod.rs` — `pub mod statistics;` + re-exports
- `src/data/store/mod.rs` — `pub mod gage_csv; pub mod netcdf;` + re-exports

---

## Conventions for this plan

- Every file we author opens with a short module doc-comment citing the
  DDR source line numbers it mirrors (see existing
  `src/data/store/zarr.rs` for the style).
- Errors are constructed via the existing `DataError` variants
  (`NetCdf`, `Csv`, `Io`, `Malformed`). Do **not** add new variants.
- Tests sit under `#[cfg(test)] mod tests` inside the source file when
  they don't touch external files. Tests against the production NetCDF /
  CSV / JSON go in `tests/data_static.rs`.
- Run `cargo test` (not `cargo test --release`) for the TDD cycle —
  faster. Use release only for the final regression sweep.
- After every passing test cycle, commit. Commit message style follows
  recent history (`git log --oneline` shows `Add ...`, `Port ...`).

---

### Task 1: Dependencies + statistics module skeleton + `naninfmean`

**Files:**
- Modify: `Cargo.toml`
- Create: `src/data/statistics.rs`
- Modify: `src/data/mod.rs`

- [ ] **Step 1: Add `serde_json` to `Cargo.toml`**

In the `[dependencies]` table, add after `serde_yaml = "0.9"`:

```toml
serde_json = "1"
```

- [ ] **Step 2: Create `src/data/statistics.rs` with module skeleton**

```rust
//! Pre-computed attribute statistics + NaN-handling helpers.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/statistics.py::set_statistics` (read
//! path only) and the `naninfmean` / `fill_nans` helpers in
//! `~/projects/ddr/src/ddr/io/readers.py:315-368`.
//!
//! We do **not** recompute statistics. DDR caches them as JSON next to the
//! attributes file; we load that JSON. If a user changes the attribute list
//! they regenerate the cache under DDR's `uv` venv and re-point
//! `config.data_sources.statistics` here.

/// Mean over the finite values of an array. Returns `f32::NAN` if no finite
/// values exist. Mirrors `naninfmean` (readers.py:315-330).
pub fn naninfmean(arr: &[f32]) -> f32 {
    let mut sum = 0.0_f64;
    let mut n = 0_usize;
    for &x in arr {
        if x.is_finite() {
            sum += x as f64;
            n += 1;
        }
    }
    if n == 0 {
        f32::NAN
    } else {
        (sum / n as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naninfmean_mixed() {
        let v: Vec<f32> = vec![1.0, 2.0, f32::NAN, 3.0, f32::INFINITY, -f32::INFINITY];
        assert_eq!(naninfmean(&v), 2.0); // (1+2+3)/3
    }

    #[test]
    fn naninfmean_all_nonfinite_returns_nan() {
        let v: Vec<f32> = vec![f32::NAN, f32::INFINITY, -f32::INFINITY];
        assert!(naninfmean(&v).is_nan());
    }

    #[test]
    fn naninfmean_empty_returns_nan() {
        assert!(naninfmean(&[]).is_nan());
    }
}
```

- [ ] **Step 3: Wire the module into `src/data/mod.rs`**

In `src/data/mod.rs`, after the existing `pub mod store;` line add:

```rust
pub mod statistics;
```

And at the bottom of the file, after the existing `pub use store::...` line,
add:

```rust
pub use statistics::naninfmean;
```

- [ ] **Step 4: Run the unit tests**

```
cargo test --lib data::statistics
```

Expected: 3 tests pass. If `naninfmean_mixed` fails check the comparison —
we expect exact equality because the sum is on whole numbers.

- [ ] **Step 5: Commit**

```
git add Cargo.toml src/data/statistics.rs src/data/mod.rs
git commit -m "Add statistics module skeleton and naninfmean helper"
```

---

### Task 2: `fill_nans_1d` + `fill_nans` (2D)

**Files:**
- Modify: `src/data/statistics.rs`

- [ ] **Step 1: Write failing tests for `fill_nans_1d` and `fill_nans`**

Append to the `#[cfg(test)] mod tests` block in `src/data/statistics.rs`:

```rust
    #[test]
    fn fill_nans_1d_replaces_nan_with_row_mean() {
        use ndarray::Array1;
        let mut a: Array1<f32> = Array1::from(vec![1.0, f32::NAN, 3.0, f32::NAN]);
        fill_nans_1d(a.view_mut(), 2.5);
        assert_eq!(a.as_slice().unwrap(), &[1.0, 2.5, 3.0, 2.5]);
    }

    #[test]
    fn fill_nans_2d_broadcasts_row_means_across_columns() {
        use ndarray::{Array1, Array2};
        // (F=2, N=3); F=0 row has mean 10, F=1 row has mean 20.
        let mut a: Array2<f32> = Array2::from_shape_vec(
            (2, 3),
            vec![1.0, f32::NAN, 3.0,
                 f32::NAN, 5.0, f32::NAN],
        )
        .unwrap();
        let row_means: Array1<f32> = Array1::from(vec![10.0, 20.0]);
        fill_nans(a.view_mut(), &row_means);
        assert_eq!(
            a.as_slice().unwrap(),
            &[1.0, 10.0, 3.0,  20.0, 5.0, 20.0]
        );
    }

    #[test]
    fn fill_nans_2d_wrong_row_means_length_panics() {
        use ndarray::{Array1, Array2};
        let mut a: Array2<f32> = Array2::zeros((2, 3));
        let row_means: Array1<f32> = Array1::from(vec![1.0]); // wrong: needs len 2
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fill_nans(a.view_mut(), &row_means);
        }));
        assert!(r.is_err());
    }
```

- [ ] **Step 2: Verify the tests fail**

```
cargo test --lib data::statistics::tests::fill_nans
```

Expected: compile error — `fill_nans_1d` and `fill_nans` don't exist yet.

- [ ] **Step 3: Implement both helpers**

In `src/data/statistics.rs`, **above** the `#[cfg(test)]` block, add:

```rust
use ndarray::{Array1, ArrayViewMut1, ArrayViewMut2};

/// Replace `NaN` entries in a 1D array with `row_mean`. Mirrors `fill_nans`
/// in `readers.py:332-368` for the 1D case.
pub fn fill_nans_1d(mut attr: ArrayViewMut1<f32>, row_mean: f32) {
    for v in attr.iter_mut() {
        if v.is_nan() {
            *v = row_mean;
        }
    }
}

/// Replace `NaN` entries in a `(F, N)` array with the per-row mean. `row_means`
/// has length `F`. Mirrors `fill_nans` (readers.py:332-368) for the 2D case
/// — specifically the branch that broadcasts a length-F vector across N
/// columns.
pub fn fill_nans(mut attr: ArrayViewMut2<f32>, row_means: &Array1<f32>) {
    let (f, _n) = attr.dim();
    assert_eq!(
        f,
        row_means.len(),
        "fill_nans: row_means length {} does not match F={}",
        row_means.len(),
        f
    );
    for (i, mut row) in attr.outer_iter_mut().enumerate() {
        let m = row_means[i];
        for v in row.iter_mut() {
            if v.is_nan() {
                *v = m;
            }
        }
    }
}
```

- [ ] **Step 4: Re-export from `src/data/mod.rs`**

Replace the existing `pub use statistics::naninfmean;` line with:

```rust
pub use statistics::{fill_nans, fill_nans_1d, naninfmean};
```

- [ ] **Step 5: Run the tests**

```
cargo test --lib data::statistics
```

Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```
git add src/data/statistics.rs src/data/mod.rs
git commit -m "Add fill_nans / fill_nans_1d helpers"
```

---

### Task 3: `AttrStats` (JSON loader + helpers)

**Files:**
- Modify: `src/data/statistics.rs`
- Reads (test): `/home/tbindas/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json`

DDR's JSON layout (sampled from disk at design time):

```json
{
  "SoilGrids1km_clay": {
    "min": 2.59, "max": 52.78, "mean": 23.49,
    "std": 8.22, "p10": 12.77, "p90": 34.60
  },
  "...": { "..." }
}
```

- [ ] **Step 1: Add a failing test that loads the production JSON**

Append to the `#[cfg(test)] mod tests` block in `src/data/statistics.rs`:

```rust
    #[test]
    fn attr_stats_open_reads_known_values() {
        let path = "/home/tbindas/projects/ddr/data/statistics/\
                    merit_attribute_statistics_merit_global_attributes_v2.nc.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: {path} not present");
            return;
        }
        let s = AttrStats::open(path).expect("load stats");
        // Sampled at design time from the same JSON.
        let clay = s
            .by_name
            .get("SoilGrids1km_clay")
            .expect("SoilGrids1km_clay present");
        assert!((clay.mean - 23.494225_f64).abs() < 1e-6);
        assert!((clay.std - 8.221468_f64).abs() < 1e-6);

        // Helpers preserve order and cast to f32.
        let names = vec![
            "SoilGrids1km_clay".to_string(),
            "meanslope".to_string(),
        ];
        let means = s.means_f32(&names);
        let stds = s.stds_f32(&names);
        assert_eq!(means.len(), 2);
        assert_eq!(stds.len(), 2);
        assert!((means[0] - 23.494225_f32).abs() < 1e-3);
    }
```

- [ ] **Step 2: Verify it fails**

```
cargo test --lib data::statistics::tests::attr_stats_open_reads_known_values
```

Expected: compile error — `AttrStats` doesn't exist.

- [ ] **Step 3: Add `AttrStats` + `AttrStatRow` + `open` + helpers**

Append to `src/data/statistics.rs` (above the `#[cfg(test)]` block):

```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::data::error::{DataError, Result};

/// One row of pre-computed statistics for a single attribute.
#[derive(Clone, Debug, Deserialize)]
pub struct AttrStatRow {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub p10: f64,
    pub p90: f64,
}

/// Pre-computed attribute statistics keyed by attribute name. Loaded from
/// DDR's JSON cache (see `~/projects/ddr/src/ddr/io/statistics.py`).
#[derive(Debug)]
pub struct AttrStats {
    pub path: PathBuf,
    pub by_name: HashMap<String, AttrStatRow>,
}

impl AttrStats {
    /// Read DDR's pre-computed JSON file. The file's top-level object is
    /// `{ attr_name: { min, max, mean, std, p10, p90 } }`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let bytes = std::fs::read(&path).map_err(|e| DataError::Io {
            path: path.clone(),
            source: e,
        })?;
        let by_name: HashMap<String, AttrStatRow> =
            serde_json::from_slice(&bytes).map_err(|e| DataError::Malformed {
                path: path.clone(),
                message: format!("stats JSON parse failed: {e}"),
            })?;
        Ok(Self { path, by_name })
    }

    /// Per-attribute means in the order given, cast to f32. Panics if any
    /// requested name is missing — callers know their attribute list up
    /// front and a typo there is a configuration bug.
    pub fn means_f32(&self, attr_names: &[String]) -> Array1<f32> {
        Array1::from(
            attr_names
                .iter()
                .map(|name| {
                    self.by_name
                        .get(name)
                        .unwrap_or_else(|| panic!("AttrStats: unknown attribute {name}"))
                        .mean as f32
                })
                .collect::<Vec<_>>(),
        )
    }

    /// Per-attribute stds in the order given, cast to f32. See `means_f32`.
    pub fn stds_f32(&self, attr_names: &[String]) -> Array1<f32> {
        Array1::from(
            attr_names
                .iter()
                .map(|name| {
                    self.by_name
                        .get(name)
                        .unwrap_or_else(|| panic!("AttrStats: unknown attribute {name}"))
                        .std as f32
                })
                .collect::<Vec<_>>(),
        )
    }
}
```

- [ ] **Step 4: Re-export `AttrStats` from `src/data/mod.rs`**

Replace the existing `pub use statistics::{fill_nans, fill_nans_1d, naninfmean};` line with:

```rust
pub use statistics::{fill_nans, fill_nans_1d, naninfmean, AttrStatRow, AttrStats};
```

- [ ] **Step 5: Run the tests**

```
cargo test --lib data::statistics
```

Expected: 7 tests pass (the new one skips if the JSON file is absent; on
the dev machine it should hit the assertion path).

- [ ] **Step 6: Commit**

```
git add src/data/statistics.rs src/data/mod.rs
git commit -m "Add AttrStats JSON loader with f32 means/stds helpers"
```

---

### Task 4: `GageRow` + `GageMetadata::open`

**Files:**
- Create: `src/data/store/gage_csv.rs`
- Modify: `src/data/store/mod.rs`
- Modify: `src/data/mod.rs`

CSV header observed in production files
(`references/gage_info/gages_3000.csv`):
`STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID,COMID_DRAIN_SQKM,COMID_UNITAREA_SQKM,ABS_DIFF,DA_VALID,FLOW_SCALE`

Required: first 5 columns. Optional: last 6. STAID is a string (must
zero-pad to 8). DA_VALID is `True`/`False` literal.

- [ ] **Step 1: Write a failing TDD test using an in-memory CSV string**

Create `src/data/store/gage_csv.rs` with the following content (header doc +
test only — implementation comes in step 3):

```rust
//! Gage metadata CSV reader.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/readers.py::read_gage_info` (lines
//! ~100-160). Required columns: `STAID, STANAME, DRAIN_SQKM, LAT_GAGE,
//! LNG_GAGE`. Optional columns: `COMID, COMID_DRAIN_SQKM,
//! COMID_UNITAREA_SQKM, ABS_DIFF, DA_VALID, FLOW_SCALE`.
//!
//! STAID values are zero-padded to 8 characters at construction (matches
//! DDR's canonical form).

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &str = "\
STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID,COMID_DRAIN_SQKM,COMID_UNITAREA_SQKM,ABS_DIFF,DA_VALID,FLOW_SCALE
14190500,\"LUCKIAMUTE RIVER NEAR SUVER, OR\",603.4942,44.783175,-123.234543,78022248,659.0194570316622,57.94836462236595,55.52525703166225,True,0.04181494346724791
1563500,FAKE,1.0,2.0,3.0,99,2.0,2.0,1.0,False,0.5
";

    #[test]
    fn parses_required_fields_and_zfills_staid() {
        let m = GageMetadata::from_reader(
            SAMPLE_CSV.as_bytes(),
            std::path::PathBuf::from("<inline>"),
        )
        .expect("parse");
        assert_eq!(m.rows.len(), 2);
        // Already 8-digit — preserved.
        assert_eq!(m.rows[0].staid.as_str(), "14190500");
        // Was 7-digit — must zero-pad.
        assert_eq!(m.rows[1].staid.as_str(), "01563500");
        assert!((m.rows[0].drain_sqkm - 603.4942).abs() < 1e-6);
    }

    #[test]
    fn parses_optional_fields() {
        let m = GageMetadata::from_reader(
            SAMPLE_CSV.as_bytes(),
            std::path::PathBuf::from("<inline>"),
        )
        .expect("parse");
        let r0 = &m.rows[0];
        assert_eq!(r0.comid, Some(78022248));
        assert_eq!(r0.da_valid, Some(true));
        assert!(r0.flow_scale.is_some());
        let r1 = &m.rows[1];
        assert_eq!(r1.da_valid, Some(false));
    }

    #[test]
    fn lookup_by_staid_uses_padded_form() {
        let m = GageMetadata::from_reader(
            SAMPLE_CSV.as_bytes(),
            std::path::PathBuf::from("<inline>"),
        )
        .expect("parse");
        use crate::data::ids::Staid;
        assert!(m.by_staid.contains_key(&Staid::new("1563500")));
        assert!(m.by_staid.contains_key(&Staid::new("01563500")));
    }
}
```

- [ ] **Step 2: Write the parser + struct above the test block**

Insert (in `src/data/store/gage_csv.rs`, above the `#[cfg(test)]` block):

```rust
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use serde::Deserialize;

use crate::data::error::{DataError, Result};
use crate::data::ids::Staid;

#[derive(Clone, Debug)]
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

#[derive(Debug)]
pub struct GageMetadata {
    pub path: PathBuf,
    pub rows: Vec<GageRow>,
    pub by_staid: HashMap<Staid, usize>,
}

#[derive(Debug, Deserialize)]
struct RawRow {
    #[serde(rename = "STAID")]
    staid: String,
    #[serde(rename = "STANAME")]
    staname: Option<String>,
    #[serde(rename = "DRAIN_SQKM")]
    drain_sqkm: f64,
    #[serde(rename = "LAT_GAGE")]
    lat_gage: f64,
    #[serde(rename = "LNG_GAGE")]
    lng_gage: f64,
    #[serde(rename = "COMID", default)]
    comid: Option<i64>,
    #[serde(rename = "COMID_DRAIN_SQKM", default)]
    comid_drain_sqkm: Option<f64>,
    #[serde(rename = "COMID_UNITAREA_SQKM", default)]
    comid_unitarea_sqkm: Option<f64>,
    #[serde(rename = "ABS_DIFF", default)]
    abs_diff: Option<f64>,
    #[serde(rename = "DA_VALID", default)]
    da_valid: Option<DaValid>,
    #[serde(rename = "FLOW_SCALE", default)]
    flow_scale: Option<f32>,
}

/// Newtype wrapper so we accept `True`/`False` (Python style) in addition to
/// csv's default `true`/`false`.
#[derive(Debug, Copy, Clone)]
struct DaValid(bool);

impl<'de> serde::Deserialize<'de> for DaValid {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s: String = String::deserialize(d)?;
        match s.as_str() {
            "True" | "true" | "1" => Ok(DaValid(true)),
            "False" | "false" | "0" => Ok(DaValid(false)),
            other => Err(serde::de::Error::custom(format!(
                "DA_VALID expected True/False/1/0, got {other}"
            ))),
        }
    }
}

impl GageMetadata {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file = std::fs::File::open(&path).map_err(|e| DataError::Io {
            path: path.clone(),
            source: e,
        })?;
        Self::from_reader(file, path)
    }

    /// Internal helper — takes any `Read` for ease of testing.
    fn from_reader<R: Read>(rdr: R, path: PathBuf) -> Result<Self> {
        let mut csv_rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(rdr);
        let mut rows: Vec<GageRow> = Vec::new();
        for rec in csv_rdr.deserialize::<RawRow>() {
            let raw = rec.map_err(|e| DataError::Csv {
                path: path.clone(),
                source: e,
            })?;
            rows.push(GageRow {
                staid: Staid::new(&raw.staid),
                staname: raw.staname.unwrap_or_else(|| raw.staid.clone()),
                drain_sqkm: raw.drain_sqkm,
                lat_gage: raw.lat_gage,
                lng_gage: raw.lng_gage,
                comid: raw.comid,
                comid_drain_sqkm: raw.comid_drain_sqkm,
                comid_unitarea_sqkm: raw.comid_unitarea_sqkm,
                abs_diff: raw.abs_diff,
                da_valid: raw.da_valid.map(|v| v.0),
                flow_scale: raw.flow_scale,
            });
        }
        let by_staid: HashMap<Staid, usize> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.staid.clone(), i))
            .collect();
        Ok(Self {
            path,
            rows,
            by_staid,
        })
    }

    /// STAIDs in file order, zero-padded.
    pub fn staids(&self) -> Vec<Staid> {
        self.rows.iter().map(|r| r.staid.clone()).collect()
    }
}
```

- [ ] **Step 3: Wire the module into `src/data/store/mod.rs`**

Update `src/data/store/mod.rs` so it reads:

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
pub mod zarr;

pub use gage_csv::{GageMetadata, GageRow};
pub use zarr::{ConusAdjacencyStore, GageSubgraph, GagesAdjacencyStore};
```

- [ ] **Step 4: Re-export from `src/data/mod.rs`**

Update the `pub use store::{...}` line in `src/data/mod.rs` to:

```rust
pub use store::{ConusAdjacencyStore, GageMetadata, GageRow, GageSubgraph, GagesAdjacencyStore};
```

- [ ] **Step 5: Run the new tests**

```
cargo test --lib data::store::gage_csv
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```
git add src/data/store/gage_csv.rs src/data/store/mod.rs src/data/mod.rs
git commit -m "Add gage CSV reader with required + optional columns"
```

---

### Task 5: Gage CSV integration test against `gages_3000.csv`

**Files:**
- Create: `tests/data_static.rs`

This is the first cross-process integration test. It reads the production
CSV; if the file isn't present the test logs and returns early so CI on a
clean machine still passes.

- [ ] **Step 1: Create the integration test file**

```rust
//! Integration tests for SP-1 static-data readers, exercised against the
//! production files in `~/projects/ddr/`.
//!
//! Each test path-checks the production file and skips with an eprintln if
//! absent — so CI on a clean machine still passes. On the dev machine
//! (where the files exist) the assertions are load-bearing.

use std::path::Path;

use ddrs::data::GageMetadata;
use ddrs::data::ids::Staid;

const GAGES_CSV: &str =
    "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";

#[test]
fn gages_3000_loads_with_expected_shape() {
    if !Path::new(GAGES_CSV).exists() {
        eprintln!("skipping: {GAGES_CSV} not present");
        return;
    }
    let m = GageMetadata::open(GAGES_CSV).expect("open gages_3000.csv");
    assert_eq!(m.rows.len(), 3000);
    // Known first row (verified at design time).
    assert_eq!(m.rows[0].staid.as_str(), "14190500");
    assert!((m.rows[0].drain_sqkm - 603.4942).abs() < 1e-6);
    assert_eq!(m.rows[0].da_valid, Some(true));
    // Lookup uses padded form.
    assert!(m.by_staid.contains_key(&Staid::new("14190500")));
}
```

- [ ] **Step 2: Run the test**

```
cargo test --test data_static gages_3000
```

Expected: 1 test passes (assuming the CSV is on disk). If the file is
missing the test still exits "ok" with the skip log.

- [ ] **Step 3: Commit**

```
git add tests/data_static.rs
git commit -m "Add gage CSV integration test against gages_3000.csv"
```

---

### Task 6: `AttributesStore::open` (NetCDF read)

**Files:**
- Create: `src/data/store/netcdf.rs`
- Modify: `src/data/store/mod.rs`
- Modify: `src/data/mod.rs`

The NetCDF strategy: open the file, read the `COMID` coord (length ~2.94M)
into a `HashMap<i64, usize>` for lookup, then for each requested attribute
read the **full column** (~24 MB at 2.94M f64), select the rows for the
requested COMIDs into a row of the output `(F, N)` matrix, cast to f32, and
compute `row_means[f] = naninfmean(full_column_as_f32)`.

Reading the full column is the simple, robust path. The implementer should
not invest time in fancy-indexing optimizations — startup cost is paid
once.

**netcdf v0.12 API notes (verify on first compile):**

- `netcdf::open(path)` → `Result<File, netcdf::Error>`
- `file.variable("COMID")` → `Option<Variable<'_>>`
- `Variable::values_arr::<T, _>(extents)` reads into an `ArrayD<T>`. To read
  the full 1D array: `var.values_arr::<i64, _>(..)`. Adjust to the actual
  call shape if v0.12 differs; the test fails fast and tells you.

- [ ] **Step 1: Stub the module**

Create `src/data/store/netcdf.rs`:

```rust
//! NetCDF attribute reader.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/readers.py::AttributesReader` and
//! `~/projects/ddr/src/ddr/geodatazoo/merit.py::_get_attributes` for the
//! MERIT branch (single `merit_global_attributes_v2.nc` file, 1D vars on a
//! `COMID` dim).
//!
//! Strategy: at `open` we materialize a `(F, N)` f32 matrix where `N` is
//! the number of requested COMIDs that were present in the file. The full
//! NetCDF column is read once per attribute (`~24 MB` at 2.94M f64),
//! cast to f32, then sliced — fancy indexing is unnecessary and the
//! peak transient is bounded by `F * 24 MB`.

use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::{Array1, Array2};

use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, IdIndex};
use crate::data::statistics::naninfmean;

pub struct AttributesStore {
    pub path: PathBuf,
    /// Attribute names in row order of `attrs`.
    pub attr_names: Vec<String>,
    /// Materialized matrix, shape `(F, N)` where `N` is the number of
    /// requested COMIDs that the NetCDF actually contains. Float32.
    pub attrs: Array2<f32>,
    /// Maps each present COMID to its column in `attrs`.
    pub index: IdIndex<Comid>,
    /// Per-attribute fill mean (`naninfmean` over the full file column),
    /// length F. Used by `fill_nans` at batch-time.
    pub row_means: Array1<f32>,
}

impl AttributesStore {
    pub fn open(
        path: impl Into<PathBuf>,
        attr_names: &[String],
        comids: &[Comid],
    ) -> Result<Self> {
        let path = path.into();
        let file = netcdf::open(&path).map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;

        // ----- COMID coord → HashMap<i64, file_pos> -----
        let comid_var = file
            .variable("COMID")
            .ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: "missing 'COMID' coord variable".to_string(),
            })?;
        let comid_arr = comid_var.values_arr::<i64, _>(..).map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;
        let comid_flat: Vec<i64> = comid_arr.iter().copied().collect();
        let comid_to_pos: HashMap<i64, usize> = comid_flat
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        // Resolve requested COMIDs → file positions; track present subset.
        let mut requested_positions: Vec<usize> = Vec::with_capacity(comids.len());
        let mut present_comids: Vec<Comid> = Vec::with_capacity(comids.len());
        for c in comids {
            if let Some(&p) = comid_to_pos.get(&c.0) {
                requested_positions.push(p);
                present_comids.push(*c);
            }
        }
        let n_present = present_comids.len();

        let f = attr_names.len();
        let mut attrs = Array2::<f32>::zeros((f, n_present));
        let mut row_means = Array1::<f32>::zeros(f);

        for (fi, name) in attr_names.iter().enumerate() {
            let var = file.variable(name).ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: format!("missing attribute variable '{name}'"),
            })?;
            // Read full column as f64 (NetCDF native).
            let col_f64 = var.values_arr::<f64, _>(..).map_err(|e| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?;
            // Cast to f32. iter().copied() works for ArrayD<f64>.
            let col_f32: Vec<f32> = col_f64.iter().map(|&x| x as f32).collect();
            // row_means: mean over finite values of the full column.
            row_means[fi] = naninfmean(&col_f32);
            // Select requested positions into row `fi`.
            for (out_col, &src_pos) in requested_positions.iter().enumerate() {
                attrs[(fi, out_col)] = col_f32[src_pos];
            }
        }

        let index = IdIndex::new(present_comids);
        Ok(Self {
            path,
            attr_names: attr_names.to_vec(),
            attrs,
            index,
            row_means,
        })
    }
}
```

- [ ] **Step 2: Wire the module into `src/data/store/mod.rs`**

Update `src/data/store/mod.rs` so it reads:

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
pub mod netcdf;
pub mod zarr;

pub use gage_csv::{GageMetadata, GageRow};
pub use netcdf::AttributesStore;
pub use zarr::{ConusAdjacencyStore, GageSubgraph, GagesAdjacencyStore};
```

- [ ] **Step 3: Re-export from `src/data/mod.rs`**

Update the `pub use store::{...}` line in `src/data/mod.rs` to:

```rust
pub use store::{
    AttributesStore, ConusAdjacencyStore, GageMetadata, GageRow, GageSubgraph,
    GagesAdjacencyStore,
};
```

- [ ] **Step 4: Compile (no runnable test yet — covered in Task 7)**

```
cargo build
```

Expected: clean build. If `values_arr::<T, _>(..)` does not match the
v0.12 API, adjust the call: common alternatives are
`var.get_values::<T, _>(..)` or constructing an explicit `Extents` value.
The fix is local to `src/data/store/netcdf.rs`; everything else stays.

- [ ] **Step 5: Commit**

```
git add src/data/store/netcdf.rs src/data/store/mod.rs src/data/mod.rs
git commit -m "Add NetCDF AttributesStore (full-column read + select)"
```

---

### Task 7: AttributesStore integration test against `merit_global_attributes_v2.nc`

**Files:**
- Modify: `tests/data_static.rs`

- [ ] **Step 1: Append the integration test**

Append to `tests/data_static.rs`:

```rust
use ddrs::data::{AttributesStore, ConusAdjacencyStore};

const ATTRS_NC: &str =
    "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc";
const CONUS_ADJ: &str =
    "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr";

#[test]
fn attributes_store_opens_against_conus_subset() {
    if !Path::new(ATTRS_NC).exists() || !Path::new(CONUS_ADJ).exists() {
        eprintln!("skipping: production data files not present");
        return;
    }
    let conus = ConusAdjacencyStore::open(CONUS_ADJ).expect("conus adj");
    // Limit to the first 500 COMIDs so the test stays fast (a full-column
    // read at 2.94M f64 × 10 attrs is ~250 MB peak — still OK, but we
    // don't need that here).
    let comids: Vec<_> = conus.order.iter().take(500).copied().collect();
    let attr_names = vec![
        "SoilGrids1km_clay".to_string(),
        "aridity".to_string(),
        "meanelevation".to_string(),
        "meanP".to_string(),
        "NDVI".to_string(),
        "meanslope".to_string(),
        "log10_uparea".to_string(),
        "SoilGrids1km_sand".to_string(),
        "ETPOT_Hargr".to_string(),
        "Porosity".to_string(),
    ];

    let store =
        AttributesStore::open(ATTRS_NC, &attr_names, &comids).expect("open attrs");

    assert_eq!(store.attr_names.len(), 10);
    assert_eq!(store.attrs.shape()[0], 10);
    assert!(store.attrs.shape()[1] > 0);
    assert!(store.attrs.shape()[1] <= 500); // some COMIDs may be absent
    // row_means are finite (the full file column has plenty of finite data).
    for &m in store.row_means.iter() {
        assert!(m.is_finite(), "row_mean unexpectedly non-finite: {m}");
    }
    // The first present COMID is round-trippable via the index.
    let first = *store.index.ids().first().expect("at least one COMID present");
    assert_eq!(store.index.position(&first), Some(0));
}
```

- [ ] **Step 2: Run the test**

```
cargo test --test data_static attributes_store_opens_against_conus_subset
```

Expected: 1 test passes. This is a slow-ish test — full-column reads of 10
attributes from a 2.94M-row NetCDF take a few seconds.

- [ ] **Step 3: If the test reveals a netcdf API mismatch, fix it**

Common fixes if v0.12 differs from the assumed API:

- `values_arr` → `get_values` or `values::<T>`. The compiler error will be
  specific.
- Extents argument: try `..` (full slice), `(..,)`, or `&[..]`.
- Casting via `col_f64.mapv(|x| x as f32)` may be more idiomatic than
  `iter().map(...)`.

Re-run after each fix. The test is the spec for the API.

- [ ] **Step 4: Commit**

```
git add tests/data_static.rs
git commit -m "Add AttributesStore integration test against MERIT NetCDF"
```

---

### Task 8: AttrStats integration test + final clippy/test sweep

**Files:**
- Modify: `tests/data_static.rs`

- [ ] **Step 1: Append the stats integration test**

Append to `tests/data_static.rs`:

```rust
use ddrs::data::AttrStats;

const STATS_JSON: &str =
    "/home/tbindas/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json";

#[test]
fn attr_stats_open_against_production_json() {
    if !Path::new(STATS_JSON).exists() {
        eprintln!("skipping: {STATS_JSON} not present");
        return;
    }
    let s = AttrStats::open(STATS_JSON).expect("open stats json");
    // The 10 attributes from config/merit_training.yaml should all be present.
    for name in [
        "SoilGrids1km_clay",
        "aridity",
        "meanelevation",
        "meanP",
        "NDVI",
        "meanslope",
        "log10_uparea",
        "SoilGrids1km_sand",
        "ETPOT_Hargr",
        "Porosity",
    ] {
        assert!(s.by_name.contains_key(name), "missing {name}");
    }
    // Spot-checked at design time.
    let clay = &s.by_name["SoilGrids1km_clay"];
    assert!((clay.mean - 23.494225_f64).abs() < 1e-6);
    assert!((clay.std - 8.221468_f64).abs() < 1e-6);
}
```

- [ ] **Step 2: Run the full test suite**

```
cargo test
```

Expected: all SP-1 tests pass and all pre-existing tests pass. If a
pre-existing test regresses, investigate — SP-1 should have changed nothing
outside the new modules and the re-export lines.

- [ ] **Step 3: Run clippy**

```
cargo clippy --all-targets -- -D warnings
```

Expected: no warnings. Common cleanups if it complains:

- Unused `mut` on locals: drop the `mut`.
- `clone_on_copy` for `Comid`: replace `.clone()` with `*`.
- Lints around `iter().copied().collect()`: idiomatic, should not fire.

- [ ] **Step 4: Run the regression benchmark to verify nothing broke**

```
cargo run --release --example compare_ddr_sandbox
```

Expected: still reports "ABSOLUTE MATCH". SP-1 doesn't touch the routing
core, but this is the existing invariant per `CLAUDE.md`.

- [ ] **Step 5: Commit**

```
git add tests/data_static.rs
git commit -m "Add AttrStats integration test against production JSON"
```

---

## Self-Review

### Spec coverage check

| Spec section | Covered by |
|---|---|
| `AttributesStore` (NetCDF) | Task 6 (impl) + Task 7 (live test) |
| `GageMetadata` (CSV) | Task 4 (impl + unit tests) + Task 5 (live test) |
| `AttrStats` (JSON) + helpers | Task 3 (impl + unit tests) + Task 8 (live test) |
| `naninfmean` | Task 1 |
| `fill_nans_1d` / `fill_nans` | Task 2 |
| Module re-exports | Tasks 1, 2, 3, 4, 6 (incrementally) |
| `serde_json` dependency | Task 1 |
| No fixture export / no Python script | Verified — none of the tasks produce one |
| Verification = "live files, sanity asserts only" | Tasks 5, 7, 8 |

### Placeholder scan

No "TBD", "TODO", "implement later", or "add appropriate X" lines. Every
code step shows the actual code; every test step shows the actual asserts.

### Type/identifier consistency

- `Staid::new(...)`, `Staid::as_str(...)`, `Comid(i64)`, `IdIndex<T>` —
  all match `src/data/ids.rs`.
- `DataError::Io`, `::NetCdf`, `::Csv`, `::Malformed` — all match
  `src/data/error.rs`.
- `Result<T>` alias from the same module — consistent across tasks.
- `AttrStats::by_name`, `AttrStatRow::{min,max,mean,std,p10,p90}` — used
  identically in Tasks 3 and 8.
- `GageMetadata::{rows, by_staid, staids()}` — used identically in Tasks 4
  and 5.
- `AttributesStore::{path, attr_names, attrs, index, row_means}` — used
  identically in Tasks 6 and 7.

No drift detected.

---

## Execution choice

Plan complete and saved to `.claude/specs/2026-05-17-sp1-static-data-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per
   task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using
   executing-plans, batch with checkpoints.

Which approach?
