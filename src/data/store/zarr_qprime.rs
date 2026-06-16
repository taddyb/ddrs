//! Global unit-catchment streamflow (Q') reader over
//! `merit_global_v2.x`-style multi-zone zarr v2 stores.
//!
//! Layout (e.g. `/gpfs/hjj5218/data/dmc_forcing/streamflow/zarr/8km/merit_global_v2.7`):
//! one directory per pfaf-2 zone (`11` … `86`, 60 zones globally), each a
//! zarr **v2** group with:
//!
//!   - `streamflow` — `(time, COMID)` float64, blosc-lz4, fill NaN.
//!     **Time-major** (the icechunk CONUS store is divide-major).
//!   - `time`       — int64, CF `"days since 1980-01-01 00:00:00"`,
//!     proleptic_gregorian, contiguous daily steps.
//!   - `COMID`      — int64 MERIT COMIDs; first two digits = the zone.
//!
//! Units were established empirically (2026-06-11, no units attr on disk):
//! per-COMID magnitudes match the CONUS `merit_dhbv2_UH_retrospective`
//! reference (declared `m^3/s`), and summed upstream Q' over USGS gage
//! basins reproduces observed discharge in m³/s (best gage: ratio 1.05,
//! corr 0.95 over year 2000). Values are **m³/s**, like the CONUS store.
//!
//! Same read contract as `StreamflowStore` (`store/icechunk.rs`): reads
//! return `(n_days, N)` f32; COMIDs absent from the store are filled with
//! `0.001` (discharge minimum, mirrors DDR's `readers.py:464-468`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::NaiveDate;
use ndarray::Array2;
use zarrs::array::Array as ZarrArray;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ReadableStorage, ReadableStorageTraits};

use crate::data::dates::RhoWindow;
use crate::data::error::{DataError, Result};
use crate::data::ids::Comid;
use crate::data::store::icechunk::{daily_to_hourly_trim, parse_cf_epoch};

/// One pfaf-2 zone of the store.
struct Zone {
    /// Zone directory name (`"74"`), kept for error context.
    name: String,
    streamflow: ZarrArray<dyn ReadableStorageTraits>,
}

/// Multi-zone zarr v2 Q' reader. COMIDs are resolved to `(zone, column)`
/// at open time; reads group requested COMIDs per zone and do one
/// contiguous column-range retrieve per zone.
pub struct GlobalStreamflowStore {
    pub path: PathBuf,
    pub time_start: NaiveDate,
    pub n_time: usize,
    /// COMID → (index into `zones`, column on the zone's COMID axis).
    by_comid: HashMap<Comid, (usize, usize)>,
    zones: Vec<Zone>,
}

impl GlobalStreamflowStore {
    /// True if `path` looks like this format: either a zone group itself
    /// (`.zgroup` + `streamflow/.zarray`) or a directory of such zone
    /// groups. Used by `StreamflowSource::open` to dispatch.
    pub fn sniff(path: &Path) -> bool {
        if is_zone_group(path) {
            return true;
        }
        match std::fs::read_dir(path) {
            Ok(entries) => entries
                .flatten()
                .any(|e| is_zone_group(&e.path())),
            Err(_) => false,
        }
    }

    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();

        // Resolve the zone group directories: `path` is either one zone or
        // a parent holding many. Sorted for a deterministic open order.
        let mut zone_dirs: Vec<PathBuf> = if is_zone_group(&path) {
            vec![path.clone()]
        } else {
            let entries = std::fs::read_dir(&path).map_err(|e| DataError::Io {
                path: path.clone(),
                source: e,
            })?;
            let mut dirs: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| is_zone_group(p))
                .collect();
            dirs.sort();
            dirs
        };
        if zone_dirs.is_empty() {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: "no zone groups found (dirs with .zgroup + streamflow/.zarray)".into(),
            });
        }

        let mut zones: Vec<Zone> = Vec::with_capacity(zone_dirs.len());
        let mut by_comid: HashMap<Comid, (usize, usize)> = HashMap::new();
        let mut time_start: Option<NaiveDate> = None;
        let mut n_time: usize = 0;

        for zone_dir in zone_dirs.drain(..) {
            let name = zone_dir
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default();
            let storage: ReadableStorage = Arc::new(
                FilesystemStore::new(&zone_dir).map_err(|e| zarr_err(&zone_dir, e))?,
            );

            // Time axis: CF epoch + contiguity check; every zone must agree.
            let time_arr = ZarrArray::open(storage.clone(), "/time")
                .map_err(|e| zarr_err(&zone_dir, e))?;
            let epoch = parse_cf_epoch(time_arr.attributes(), &zone_dir)?;
            let time_i64: Vec<i64> = time_arr
                .retrieve_array_subset(&time_arr.subset_all())
                .map_err(|e| zarr_err(&zone_dir, e))?;
            if time_i64.is_empty() {
                return Err(DataError::Malformed {
                    path: zone_dir.clone(),
                    message: "time axis is empty".into(),
                });
            }
            let contiguous = time_i64
                .windows(2)
                .all(|w| w[1] == w[0] + 1);
            if !contiguous {
                return Err(DataError::Malformed {
                    path: zone_dir.clone(),
                    message: "time axis is not contiguous daily steps".into(),
                });
            }
            let zone_start = epoch + chrono::Duration::days(time_i64[0]);
            match time_start {
                None => {
                    time_start = Some(zone_start);
                    n_time = time_i64.len();
                }
                Some(expect) => {
                    if zone_start != expect || time_i64.len() != n_time {
                        return Err(DataError::Malformed {
                            path: zone_dir.clone(),
                            message: format!(
                                "zone {name} time axis ({zone_start}, n={}) disagrees with \
                                 first zone ({expect}, n={n_time})",
                                time_i64.len()
                            ),
                        });
                    }
                }
            }

            // COMID axis → (zone, column) map.
            let comid_arr = ZarrArray::open(storage.clone(), "/COMID")
                .map_err(|e| zarr_err(&zone_dir, e))?;
            let comids: Vec<i64> = comid_arr
                .retrieve_array_subset(&comid_arr.subset_all())
                .map_err(|e| zarr_err(&zone_dir, e))?;

            let streamflow = ZarrArray::open(storage, "/streamflow")
                .map_err(|e| zarr_err(&zone_dir, e))?;
            if streamflow.shape() != [n_time as u64, comids.len() as u64] {
                return Err(DataError::Malformed {
                    path: zone_dir.clone(),
                    message: format!(
                        "streamflow shape {:?} != (time={n_time}, comid={})",
                        streamflow.shape(),
                        comids.len()
                    ),
                });
            }

            let zone_idx = zones.len();
            for (col, c) in comids.into_iter().enumerate() {
                by_comid.insert(Comid(c), (zone_idx, col));
            }
            zones.push(Zone { name, streamflow });
        }

        Ok(Self {
            path,
            time_start: time_start.expect("at least one zone opened"),
            n_time,
            by_comid,
            zones,
        })
    }

    /// Number of COMIDs across all zones.
    pub fn n_comids(&self) -> usize {
        self.by_comid.len()
    }

    /// Number of zone groups opened.
    pub fn n_zones(&self) -> usize {
        self.zones.len()
    }

    /// Read Q' daily for `[window_start, window_start + n_days)` and
    /// `comids`. Returns `(n_days, N)` f32; missing COMIDs are filled with
    /// `0.001` (same semantics as the icechunk `StreamflowStore`).
    pub fn read_window_daily(
        &self,
        window_start: NaiveDate,
        n_days: usize,
        comids: &[Comid],
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

        // Pre-fill with the discharge minimum; missing COMIDs keep it.
        let mut daily = Array2::<f32>::from_elem((n_days, comids.len()), 0.001);

        // Group present COMIDs by zone: zone_idx → [(out_col, zone_col)].
        let mut per_zone: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (out_col, c) in comids.iter().enumerate() {
            if let Some(&(z, col)) = self.by_comid.get(c) {
                per_zone.entry(z).or_default().push((out_col, col));
            }
        }

        for (z, cols) in per_zone {
            let zone = &self.zones[z];
            let min_col = cols.iter().map(|&(_, c)| c).min().unwrap();
            let max_col = cols.iter().map(|&(_, c)| c).max().unwrap();
            let col_count = max_col - min_col + 1;

            // streamflow is (time, COMID) — time-major, unlike icechunk.
            let subset = zarrs::array::ArraySubset::new_with_ranges(&[
                (store_start_day as u64)..(end_day as u64),
                (min_col as u64)..((max_col + 1) as u64),
            ]);
            let raw_f64: Vec<f64> = zone
                .streamflow
                .retrieve_array_subset(&subset)
                .map_err(|e| {
                    zarr_err(&self.path.join(&zone.name), e)
                })?;
            debug_assert_eq!(raw_f64.len(), n_days * col_count);

            // raw is row-major (n_days, col_count): element (d, j) at
            // d * col_count + j.
            for &(out_col, zone_col) in &cols {
                let local = zone_col - min_col;
                for d in 0..n_days {
                    daily[(d, out_col)] = raw_f64[d * col_count + local] as f32;
                }
            }
        }
        Ok(daily)
    }

    /// `(n_hourly, N)` read for a training rho-window — daily values
    /// repeated 24× and trimmed, identical to the icechunk store.
    pub fn read_window(&self, window: &RhoWindow, comids: &[Comid]) -> Result<Array2<f32>> {
        let daily = self.read_window_daily(window.window_start, window.rho_days, comids)?;
        Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
    }

    /// `TestWindow` read returning `n_days * 24` hours (no trailing trim).
    pub fn read_test_window(
        &self,
        window: &crate::data::TestWindow,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        let daily = self.read_window_daily(window.window_start, window.n_days, comids)?;
        Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
    }
}

/// A zone group is a directory with zarr v2 group metadata and a
/// `streamflow` array inside.
fn is_zone_group(path: &Path) -> bool {
    path.join(".zgroup").is_file() && path.join("streamflow").join(".zarray").is_file()
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

    /// Hand-write a two-zone uncompressed zarr v2 store, 5 days × 2 COMIDs
    /// per zone. Zone "11" holds COMIDs 11000001-2, zone "74" 74000001-2.
    fn synthetic_store(dir: &Path) {
        for (zone, base) in [("11", 0.0f64), ("74", 100.0f64)] {
            let z = dir.join(zone);
            std::fs::create_dir(&z).unwrap();
            std::fs::write(z.join(".zgroup"), r#"{"zarr_format": 2}"#).unwrap();

            // time: 5 days starting at offset 0 from the epoch.
            std::fs::create_dir(z.join("time")).unwrap();
            std::fs::write(
                z.join("time/.zarray"),
                r#"{"chunks":[5],"compressor":null,"dtype":"<i8","fill_value":null,
                    "filters":null,"order":"C","shape":[5],"zarr_format":2}"#,
            )
            .unwrap();
            std::fs::write(
                z.join("time/.zattrs"),
                r#"{"units": "days since 2000-01-01 00:00:00"}"#,
            )
            .unwrap();
            let time_bytes: Vec<u8> = (0i64..5).flat_map(|t| t.to_le_bytes()).collect();
            std::fs::write(z.join("time/0"), time_bytes).unwrap();

            // COMID: two ids per zone.
            let zone_num: i64 = zone.parse().unwrap();
            std::fs::create_dir(z.join("COMID")).unwrap();
            std::fs::write(
                z.join("COMID/.zarray"),
                r#"{"chunks":[2],"compressor":null,"dtype":"<i8","fill_value":null,
                    "filters":null,"order":"C","shape":[2],"zarr_format":2}"#,
            )
            .unwrap();
            let comid_bytes: Vec<u8> = [zone_num * 1_000_000 + 1, zone_num * 1_000_000 + 2]
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .collect();
            std::fs::write(z.join("COMID/0"), comid_bytes).unwrap();

            // streamflow: (5, 2) C-order, value = base + day*10 + col.
            std::fs::create_dir(z.join("streamflow")).unwrap();
            std::fs::write(
                z.join("streamflow/.zarray"),
                r#"{"chunks":[5,2],"compressor":null,"dtype":"<f8","fill_value":"NaN",
                    "filters":null,"order":"C","shape":[5,2],"zarr_format":2}"#,
            )
            .unwrap();
            let mut sf = Vec::with_capacity(80);
            for d in 0..5 {
                for c in 0..2 {
                    sf.extend_from_slice(&(base + (d * 10 + c) as f64).to_le_bytes());
                }
            }
            std::fs::write(z.join("streamflow/0.0"), sf).unwrap();
        }
    }

    #[test]
    fn synthetic_multizone_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        synthetic_store(tmp.path());

        assert!(GlobalStreamflowStore::sniff(tmp.path()));
        let store = GlobalStreamflowStore::open(tmp.path()).unwrap();
        assert_eq!(store.n_zones(), 2);
        assert_eq!(store.n_comids(), 4);
        assert_eq!(store.n_time, 5);
        assert_eq!(
            store.time_start,
            NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()
        );

        // Cross-zone read with one missing COMID (→ 0.001 fill), days 1..4.
        let comids = [
            Comid(74000002), // zone 74, col 1
            Comid(99999999), // missing
            Comid(11000001), // zone 11, col 0
        ];
        let out = store
            .read_window_daily(NaiveDate::from_ymd_opt(2000, 1, 2).unwrap(), 3, &comids)
            .unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        // zone 74 col 1, day d: 100 + d*10 + 1.
        assert_eq!(out[(0, 0)], 111.0);
        assert_eq!(out[(2, 0)], 131.0);
        // missing → discharge minimum.
        assert_eq!(out[(0, 1)], 0.001);
        // zone 11 col 0, day d: d*10.
        assert_eq!(out[(0, 2)], 10.0);
        assert_eq!(out[(2, 2)], 30.0);

        // Out-of-range windows error.
        assert!(store
            .read_window_daily(NaiveDate::from_ymd_opt(1999, 12, 31).unwrap(), 1, &comids)
            .is_err());
        assert!(store
            .read_window_daily(NaiveDate::from_ymd_opt(2000, 1, 4).unwrap(), 3, &comids)
            .is_err());

        // Pointing at a single zone group directly also works.
        let single = GlobalStreamflowStore::open(tmp.path().join("74")).unwrap();
        assert_eq!(single.n_zones(), 1);
        assert_eq!(single.n_comids(), 2);
    }

    // ------------------------------------------------------------------
    // Gated tests against the real merit_global_v2.7 store; skipped when
    // the cluster path is absent.
    // ------------------------------------------------------------------

    const REAL: &str = "/gpfs/hjj5218/data/dmc_forcing/streamflow/zarr/8km/merit_global_v2.7";

    #[test]
    fn real_zone74_matches_python_reference() {
        let zone74 = Path::new(REAL).join("74");
        if !zone74.exists() {
            eprintln!("skipping: {REAL} not present");
            return;
        }
        // Open one zone, not all 60 — keeps the test fast.
        let store = GlobalStreamflowStore::open(&zone74).unwrap();
        assert_eq!(store.n_time, 14976, "1980-01-01..2020-12-31 daily");
        assert_eq!(store.n_comids(), 72659);
        assert_eq!(
            store.time_start,
            NaiveDate::from_ymd_opt(1980, 1, 1).unwrap()
        );

        // Reference values read via python zarr 2.18 (2026-06-11):
        // COMID 74071669 (USGS__02481000's reach), days 0..4 and day 7305.
        let comids = [Comid(74071669)];
        let out = store
            .read_window_daily(NaiveDate::from_ymd_opt(1980, 1, 1).unwrap(), 4, &comids)
            .unwrap();
        let expect = [0.00128416f32, 0.00218403, 0.00289403, 0.00355483];
        for (d, e) in expect.iter().enumerate() {
            assert!(
                (out[(d, 0)] - e).abs() < 1e-6,
                "day {d}: got {}, want {e}",
                out[(d, 0)]
            );
        }
        let y2000 = store
            .read_window_daily(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(), 1, &comids)
            .unwrap();
        assert!((y2000[(0, 0)] - 0.39761827).abs() < 1e-6);
    }

    /// Full 60-zone open (~2.9M COMIDs). Ignored by default — run with
    /// `cargo test --lib zarr_qprime -- --ignored` on the cluster.
    #[test]
    #[ignore]
    fn real_full_store_opens_all_zones() {
        if !Path::new(REAL).exists() {
            eprintln!("skipping: {REAL} not present");
            return;
        }
        let t0 = std::time::Instant::now();
        let store = GlobalStreamflowStore::open(REAL).unwrap();
        eprintln!(
            "opened {} zones / {} COMIDs in {:.1}s",
            store.n_zones(),
            store.n_comids(),
            t0.elapsed().as_secs_f32()
        );
        assert_eq!(store.n_zones(), 60);
        // 2,897,147 as of v2.7 — ~42k fewer than the fabric's 2,939,408;
        // reaches without predictions take the 0.001 fill at read time.
        assert!(store.n_comids() > 2_800_000, "got {}", store.n_comids());
        assert_eq!(store.n_time, 14976);

        // Cross-zone read straddling three zones in one call.
        let comids = [Comid(11000001), Comid(74071669), Comid(86000001)];
        let out = store
            .read_window_daily(NaiveDate::from_ymd_opt(1980, 1, 1).unwrap(), 2, &comids)
            .unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert!((out[(0, 1)] - 0.00128416).abs() < 1e-6);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
