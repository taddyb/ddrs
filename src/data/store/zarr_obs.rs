//! Global observations reader over `dMC_global_v3.1`-style zarr v2 groups.
//!
//! Layout (e.g. `/gpfs/hjj5218/data/dmc_forcing/observation/dMC_global_v3.1`):
//! a zarr **v2** directory group (`.zgroup` at the root) holding one 1-D
//! float64 array per gage, named `<Provider>__<StationId>` (e.g.
//! `USGS__01030500`, `BOMAustralia__003204A`). 6,051 gages from 25+ providers
//! (USGS, GRDC, BOM, the CAMELS family, national agencies, ...).
//!
//! Conventions of the format, established empirically (2026-06-10):
//!   - Every array is `shape [14976]`, `dtype <f8`, blosc-lz4, single chunk.
//!   - Units are m³/s. Missing data is NaN (the `.zarray` `fill_value: 0.0`
//!     never materializes — every array has its one chunk written).
//!   - There is NO time coordinate and no `.zattrs` anywhere. The time axis
//!     is implicit: daily, starting 1980-01-01 (index 14975 = 2020-12-31).
//!     Cross-checked against USGS NWIS: `USGS__01030500[0..14]` equals the
//!     gage's published daily cfs for 1980-01-01.. converted at 0.0283168.
//!
//! Same read contract as `UsgsObservationsStore` (`store/icechunk.rs`):
//! missing STAIDs are a hard `DataError::MissingIds` (observation misses are
//! configuration bugs), reads return `(n_days, G)` f32 with NaN preserved.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::NaiveDate;
use ndarray::Array2;
use zarrs::array::Array as ZarrArray;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorage;

use crate::data::dates::RhoWindow;
use crate::data::error::{DataError, Result};
use crate::data::ids::{IdIndex, Staid};

/// Implicit epoch of the dMC_global_v3.1 time axis (see module docs for the
/// NWIS cross-check pinning this).
pub const DMC_GLOBAL_EPOCH: (i32, u32, u32) = (1980, 1, 1);

/// Zarr-v2-group observations reader. One 1-D f64 array per gage.
pub struct GlobalObservationsStore {
    pub path: PathBuf,
    pub index: IdIndex<Staid>,
    /// Calendar date of array index 0.
    pub time_start: NaiveDate,
    /// Length of every per-gage array (validated per read).
    pub n_time: usize,
    storage: ReadableStorage,
}

impl GlobalObservationsStore {
    /// True if `path` looks like this format: a directory group (`.zgroup`)
    /// rather than an icechunk repo. Used by `ObservationsStore::open` to
    /// dispatch.
    pub fn sniff(path: &Path) -> bool {
        path.join(".zgroup").is_file()
    }

    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let (y, m, d) = DMC_GLOBAL_EPOCH;
        Self::open_with_epoch(path, NaiveDate::from_ymd_opt(y, m, d).unwrap())
    }

    /// `open` with an explicit index-0 date, for stores that share the layout
    /// but not the 1980-01-01 epoch.
    pub fn open_with_epoch(path: impl Into<PathBuf>, time_start: NaiveDate) -> Result<Self> {
        let path = path.into();
        if !Self::sniff(&path) {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: "not a zarr v2 group (no .zgroup at root)".into(),
            });
        }
        let storage: ReadableStorage =
            Arc::new(FilesystemStore::new(&path).map_err(|e| zarr_err(&path, e))?);

        // Enumerate gage arrays: every child directory holding a `.zarray`.
        // Sorted so the index is deterministic across filesystems.
        let mut names: Vec<String> = Vec::new();
        let entries = std::fs::read_dir(&path).map_err(|e| DataError::Io {
            path: path.clone(),
            source: e,
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| DataError::Io {
                path: path.clone(),
                source: e,
            })?;
            if entry.path().join(".zarray").is_file() {
                names.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        names.sort();
        if names.is_empty() {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: "zarr group contains no arrays".into(),
            });
        }

        // n_time from the first array; per-read validation catches stragglers.
        let first = open_gage_array(&storage, &path, &names[0])?;
        let n_time = first.shape()[0] as usize;

        // NOTE: gage names here are full `Provider__Id` strings (≥ 8 chars),
        // so `Staid::new`'s zero-padding never fires.
        let index = IdIndex::new(names.iter().map(|n| Staid::new(n)).collect());

        Ok(Self {
            path,
            index,
            time_start,
            n_time,
            storage,
        })
    }

    /// Read observations daily for `[window_start, window_start + n_days)`
    /// and `staids`. Returns `(n_days, G)` f32, NaN = missing. Missing
    /// STAIDs trigger `DataError::MissingIds`.
    pub fn read_window_daily(
        &self,
        window_start: NaiveDate,
        n_days: usize,
        staids: &[Staid],
    ) -> Result<Array2<f32>> {
        let store_start_day_i64 = (window_start - self.time_start).num_days();
        if store_start_day_i64 < 0 {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window starts {} before store start {}",
                    window_start, self.time_start
                ),
            });
        }
        let store_start_day = store_start_day_i64 as usize;
        let end_day = store_start_day + n_days;
        if end_day > self.n_time {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window extends to store day {end_day} but n_time={}",
                    self.n_time
                ),
            });
        }

        // Hard-error on misses, same semantics as UsgsObservationsStore.
        let (_, missing_indices) = self.index.positions_of(staids);
        if !missing_indices.is_empty() {
            return Err(DataError::MissingIds {
                path: self.path.clone(),
                kind: "gage_id",
                missing: missing_indices.len(),
                total: staids.len(),
            });
        }

        let mut out = Array2::<f32>::zeros((n_days, staids.len()));
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            (store_start_day as u64)..(end_day as u64),
        ]);
        for (col, staid) in staids.iter().enumerate() {
            let arr = open_gage_array(&self.storage, &self.path, staid.as_str())?;
            if arr.shape() != [self.n_time as u64] {
                return Err(DataError::Malformed {
                    path: self.path.clone(),
                    message: format!(
                        "array {} has shape {:?}, expected [{}]",
                        staid,
                        arr.shape(),
                        self.n_time
                    ),
                });
            }
            let vals: Vec<f64> = arr
                .retrieve_array_subset(&subset)
                .map_err(|e| zarr_err(&self.path, e))?;
            debug_assert_eq!(vals.len(), n_days);
            for (d, v) in vals.into_iter().enumerate() {
                out[(d, col)] = v as f32;
            }
        }
        Ok(out)
    }

    /// `RhoWindow`-shaped wrapper; observations are already daily.
    pub fn read_window(&self, window: &RhoWindow, staids: &[Staid]) -> Result<Array2<f32>> {
        self.read_window_daily(window.window_start, window.rho_days, staids)
    }
}

fn open_gage_array(
    storage: &ReadableStorage,
    path: &Path,
    name: &str,
) -> Result<ZarrArray<dyn zarrs::storage::ReadableStorageTraits>> {
    ZarrArray::open(storage.clone(), &format!("/{name}")).map_err(|e| zarr_err(path, e))
}

fn zarr_err<E: std::error::Error + Send + Sync + 'static>(path: &Path, e: E) -> DataError {
    DataError::Zarr {
        path: path.to_path_buf(),
        source: Box::new(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-write a minimal uncompressed zarr v2 group with two gage arrays
    /// (10 days each) — no python dependency.
    fn synthetic_group(dir: &Path) {
        std::fs::write(dir.join(".zgroup"), r#"{"zarr_format": 2}"#).unwrap();
        let zarray = r#"{
            "chunks": [10], "compressor": null, "dtype": "<f8",
            "fill_value": 0.0, "filters": null, "order": "C",
            "shape": [10], "zarr_format": 2
        }"#;
        for (name, base) in [("TestProv__A1", 0.0f64), ("TestProv__B2", 100.0f64)] {
            let d = dir.join(name);
            std::fs::create_dir(&d).unwrap();
            std::fs::write(d.join(".zarray"), zarray).unwrap();
            let mut bytes = Vec::with_capacity(80);
            for i in 0..10 {
                let v = if name.ends_with("B2") && i == 3 {
                    f64::NAN
                } else {
                    base + i as f64
                };
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(d.join("0"), bytes).unwrap();
        }
    }

    #[test]
    fn synthetic_group_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        synthetic_group(tmp.path());

        let store = GlobalObservationsStore::open_with_epoch(
            tmp.path(),
            NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(store.n_time, 10);
        assert_eq!(store.index.len(), 2);

        let staids = [Staid::new("TestProv__B2"), Staid::new("TestProv__A1")];
        let out = store
            .read_window_daily(NaiveDate::from_ymd_opt(2000, 1, 3).unwrap(), 4, &staids)
            .unwrap();
        assert_eq!(out.shape(), &[4, 2]);
        // Window covers store days 2..6; B2 is NaN at store day 3.
        assert_eq!(out[(0, 1)], 2.0); // A1 day 2
        assert_eq!(out[(0, 0)], 102.0); // B2 day 2
        assert!(out[(1, 0)].is_nan()); // B2 day 3 = NaN
        assert_eq!(out[(1, 1)], 3.0); // A1 day 3
        assert_eq!(out[(3, 0)], 105.0); // B2 day 5

        // Missing staid is a hard error.
        let err = store
            .read_window_daily(
                NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
                1,
                &[Staid::new("TestProv__ZZ")],
            )
            .unwrap_err();
        assert!(matches!(err, DataError::MissingIds { .. }));

        // Out-of-range windows are hard errors.
        assert!(store
            .read_window_daily(NaiveDate::from_ymd_opt(1999, 12, 31).unwrap(), 1, &staids)
            .is_err());
        assert!(store
            .read_window_daily(NaiveDate::from_ymd_opt(2000, 1, 8).unwrap(), 5, &staids)
            .is_err());
    }

    // ------------------------------------------------------------------
    // Gated tests against the real dMC_global_v3.1 store. Skipped (pass
    // trivially) when the cluster path is absent — same pattern as the
    // icechunk tests against /mnt/ssd1.
    // ------------------------------------------------------------------

    const REAL: &str = "/gpfs/hjj5218/data/dmc_forcing/observation/dMC_global_v3.1";

    #[test]
    fn real_store_opens_and_matches_nwis() {
        if !Path::new(REAL).exists() {
            eprintln!("skipping: {REAL} not present");
            return;
        }
        let store = GlobalObservationsStore::open(REAL).unwrap();
        assert_eq!(store.n_time, 14976, "1980-01-01..2020-12-31 daily");
        assert_eq!(store.index.len(), 6051);

        // USGS NWIS daily cfs for 01030500, 1980-01-01.., × 0.0283168 m³/s.
        // (2600, 2260, 1940, 1650 cfs.) Stored values are f32-quantized f64.
        let out = store
            .read_window_daily(
                NaiveDate::from_ymd_opt(1980, 1, 1).unwrap(),
                4,
                &[Staid::new("USGS__01030500")],
            )
            .unwrap();
        let expect = [73.624, 63.996, 54.935, 46.723];
        for (d, e) in expect.iter().enumerate() {
            assert!(
                (out[(d, 0)] - e).abs() < 5e-3,
                "day {d}: got {}, want {e}",
                out[(d, 0)]
            );
        }

        // BOM gage starts as NaN (missing early record).
        let bom = store
            .read_window_daily(
                NaiveDate::from_ymd_opt(1980, 1, 1).unwrap(),
                5,
                &[Staid::new("BOMAustralia__003204A")],
            )
            .unwrap();
        assert!(bom.iter().all(|v| v.is_nan()));
    }
}
