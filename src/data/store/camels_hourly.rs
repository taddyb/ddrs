//! Real hourly USGS streamflow + NLDAS precip reader.
//!
//! This is the Gauch et al. "Rainfall-Runoff Prediction at Multiple
//! Timescales with a Single Long Short-Term Memory Network" (MTS-LSTM)
//! paper's INPUT dataset — real NLDAS forcings + real MEASURED USGS
//! streamflow, NOT that paper's (or any) model's output. It exists
//! specifically to pretrain the disaggregation head (`src/nn/disagg_head.rs`)
//! against real hourly ground truth with zero second-model bias — see
//! `docs/2026-07-1x-disagg-real-pretrain-*.md`. Not part of the production
//! training/eval data path.
//!
//! Layout: single NetCDF4 file, dims `(basin, date)`. `basin` is a
//! **vlen-string** coordinate (raw USGS STAID, e.g. `"01022500"` — NOT
//! guaranteed zero-padded in the file; re-padded here via [`Staid::new`]).
//! `date` is `int64`, CF units `"hours since <timestamp>"` where
//! `<timestamp>` is NOT midnight-aligned (verified: `1956-12-07 09:00:00`) —
//! all offset arithmetic is done via row-index counts against that exact
//! timestamp, never calendar-day semantics. Data vars used:
//! `qobs_mm_per_hour` f32 (real measured streamflow, basin-normalized to
//! mm/hr, `_FillValue: nan`) and `total_precipitation` f32 (real NLDAS
//! hourly precip, gap-free, already basin-aggregated). Unit conversion to
//! m³/s (via drainage area) is the CALLER's job, not this reader's — this
//! module reports exactly what's on disk.

use std::path::PathBuf;

use chrono::NaiveDateTime;
use ndarray::Array2;

use crate::data::error::{DataError, Result};
use crate::data::ids::{IdIndex, Staid};

/// Reader for `usgs-streamflow-nldas_hourly.nc`-layout files. Metadata
/// (basin index, time axis) is read once at [`open`]; [`read_window`]
/// reopens the file per call (same lazy-reopen convention as
/// [`crate::data::store::state_cache::StateCache`] — this file is ~4.5 GB,
/// keeping it open in the struct isn't worth the lifetime complexity for
/// the batch-per-call access pattern pretraining uses).
pub struct CamelsHourlyStore {
    pub path: PathBuf,
    pub index: IdIndex<Staid>,
    /// Calendar timestamp of hour-row 0, parsed from the `date` variable's
    /// `units` attribute. NOT midnight-aligned — see module docs.
    pub time_start: NaiveDateTime,
    /// Number of hourly rows in the store.
    pub n_hours: usize,
}

impl CamelsHourlyStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file = netcdf::open(&path).map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;

        // ----- basin (vlen string) -> Staid index -----
        let basin_var = file.variable("basin").ok_or_else(|| DataError::Malformed {
            path: path.clone(),
            message: "missing 'basin' variable".into(),
        })?;
        let n_basin = basin_var
            .dimensions()
            .first()
            .map(|d| d.len())
            .unwrap_or(0);
        let mut staids = Vec::with_capacity(n_basin);
        for i in 0..n_basin {
            let s = basin_var.get_string(i).map_err(|e| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?;
            staids.push(Staid::new(&s));
        }
        let index = IdIndex::new(staids);

        // ----- date (hours since <timestamp>) -----
        let date_var = file.variable("date").ok_or_else(|| DataError::Malformed {
            path: path.clone(),
            message: "missing 'date' variable".into(),
        })?;
        let n_hours = date_var.dimensions().first().map(|d| d.len()).unwrap_or(0);
        let units_attr = date_var.attribute("units").ok_or_else(|| DataError::Malformed {
            path: path.clone(),
            message: "'date' variable missing 'units' attribute".into(),
        })?;
        let units: String = units_attr
            .value()
            .map_err(|e| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?
            .try_into()
            .map_err(|e: netcdf::Error| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?;
        const PREFIX: &str = "hours since ";
        let ts_str = units.strip_prefix(PREFIX).ok_or_else(|| DataError::Malformed {
            path: path.clone(),
            message: format!("unexpected 'date' units {units:?}, expected {PREFIX:?} prefix"),
        })?;
        let time_start = NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S")
            .map_err(|e| DataError::Malformed {
                path: path.clone(),
                message: format!("failed to parse 'date' units timestamp {ts_str:?}: {e}"),
            })?;

        Ok(Self {
            path,
            index,
            time_start,
            n_hours,
        })
    }

    /// Read `n_hours` hourly `(qobs_mm_per_hour, total_precipitation)`
    /// starting at `start` for `staids`. Returns two `(n_hours, N)` f32
    /// arrays, NaN preserved (real data has gaps — unlike a model product,
    /// callers MUST mask incomplete days rather than assume gap-free
    /// coverage). Missing STAIDs are a hard [`DataError::MissingIds`]; an
    /// out-of-range window is a hard [`DataError::Malformed`].
    pub fn read_window(
        &self,
        start: NaiveDateTime,
        n_hours: usize,
        staids: &[Staid],
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let (positions, missing) = self.index.positions_of(staids);
        if !missing.is_empty() {
            return Err(DataError::MissingIds {
                path: self.path.clone(),
                kind: "basin",
                missing: missing.len(),
                total: staids.len(),
            });
        }

        let start_hour_i64 = (start - self.time_start).num_hours();
        if start_hour_i64 < 0 {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window starts {start} before store start {}",
                    self.time_start
                ),
            });
        }
        let start_hour = start_hour_i64 as usize;
        let end_hour = start_hour + n_hours;
        if end_hour > self.n_hours {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window extends to hour {end_hour} but n_hours={}",
                    self.n_hours
                ),
            });
        }

        let file = netcdf::open(&self.path).map_err(|e| DataError::NetCdf {
            path: self.path.clone(),
            source: e,
        })?;
        let qobs_var = file
            .variable("qobs_mm_per_hour")
            .ok_or_else(|| DataError::Malformed {
                path: self.path.clone(),
                message: "missing 'qobs_mm_per_hour' variable".into(),
            })?;
        let precip_var = file
            .variable("total_precipitation")
            .ok_or_else(|| DataError::Malformed {
                path: self.path.clone(),
                message: "missing 'total_precipitation' variable".into(),
            })?;

        let mut qobs = Array2::<f32>::zeros((n_hours, staids.len()));
        let mut precip = Array2::<f32>::zeros((n_hours, staids.len()));
        for (col, &pos) in positions.iter().enumerate() {
            let q_extents: netcdf::Extents = (&[pos, start_hour][..], &[1_usize, n_hours][..])
                .try_into()
                .map_err(|e| DataError::NetCdf {
                    path: self.path.clone(),
                    source: e,
                })?;
            let q_raw: Vec<f32> =
                qobs_var
                    .get_values::<f32, _>(q_extents)
                    .map_err(|e| DataError::NetCdf {
                        path: self.path.clone(),
                        source: e,
                    })?;
            debug_assert_eq!(q_raw.len(), n_hours);
            for (row, &v) in q_raw.iter().enumerate() {
                qobs[(row, col)] = v;
            }

            let p_extents: netcdf::Extents = (&[pos, start_hour][..], &[1_usize, n_hours][..])
                .try_into()
                .map_err(|e| DataError::NetCdf {
                    path: self.path.clone(),
                    source: e,
                })?;
            let p_raw: Vec<f32> =
                precip_var
                    .get_values::<f32, _>(p_extents)
                    .map_err(|e| DataError::NetCdf {
                        path: self.path.clone(),
                        source: e,
                    })?;
            debug_assert_eq!(p_raw.len(), n_hours);
            for (row, &v) in p_raw.iter().enumerate() {
                precip[(row, col)] = v;
            }
        }
        Ok((qobs, precip))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Hand-write a minimal real NetCDF4 file matching the production
    /// layout (vlen-string basin, hourly int64 date w/ non-midnight epoch,
    /// qobs + precip data vars) — no python dependency.
    fn synthetic_file(path: &Path) {
        let mut f = netcdf::create(path).unwrap();
        f.add_dimension("basin", 2).unwrap();
        f.add_dimension("date", 6).unwrap();

        let mut basin_var = f.add_string_variable("basin", &["basin"]).unwrap();
        basin_var.put_string("1022500", 0).unwrap(); // deliberately unpadded (7 chars)
        basin_var.put_string("01031500", 1).unwrap();

        let mut date_var = f.add_variable::<i64>("date", &["date"]).unwrap();
        date_var
            .put_values(&[0i64, 1, 2, 3, 4, 5], ..)
            .unwrap();
        date_var
            .put_attribute("units", "hours since 1956-12-07 09:00:00")
            .unwrap();

        let mut qobs_var = f
            .add_variable::<f32>("qobs_mm_per_hour", &["basin", "date"])
            .unwrap();
        // basin0: [0.1, 0.2, NaN, 0.4, 0.5, 0.6]; basin1: [1.0..1.5]
        qobs_var
            .put_values(
                &[0.1f32, 0.2, f32::NAN, 0.4, 0.5, 0.6, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                ..,
            )
            .unwrap();

        let mut precip_var = f
            .add_variable::<f32>("total_precipitation", &["basin", "date"])
            .unwrap();
        precip_var
            .put_values(
                &[0.0f32, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                ..,
            )
            .unwrap();
    }

    #[test]
    fn synthetic_file_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("synthetic.nc");
        synthetic_file(&path);

        let store = CamelsHourlyStore::open(&path).unwrap();
        assert_eq!(store.n_hours, 6);
        assert_eq!(store.index.len(), 2);
        // "1022500" -> zero-padded to "01022500" at read time.
        assert_eq!(
            store.time_start,
            NaiveDateTime::parse_from_str("1956-12-07 09:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        let staids = [Staid::new("01031500"), Staid::new("1022500")];
        let (qobs, precip) = store
            .read_window(store.time_start + chrono::Duration::hours(1), 4, &staids)
            .unwrap();
        assert_eq!(qobs.shape(), &[4, 2]);
        // Window covers store hours 1..5.
        assert_eq!(qobs[(0, 0)], 1.1); // basin1 hour1
        assert_eq!(qobs[(0, 1)], 0.2); // basin0 (zero-padded) hour1
        assert!(qobs[(1, 1)].is_nan()); // basin0 hour2 = NaN
        assert_eq!(precip[(1, 1)], 5.0); // basin0 hour2 precip spike

        // Missing staid is a hard error.
        let err = store
            .read_window(store.time_start, 1, &[Staid::new("99999999")])
            .unwrap_err();
        assert!(matches!(err, DataError::MissingIds { .. }));

        // Out-of-range windows are hard errors.
        assert!(store
            .read_window(store.time_start - chrono::Duration::hours(1), 1, &staids)
            .is_err());
        assert!(store.read_window(store.time_start, 10, &staids).is_err());
    }

    // ------------------------------------------------------------------
    // Gated test against the real file. Skipped (passes trivially) when
    // /mnt/ssd1 is absent — same pattern as the icechunk tests.
    // ------------------------------------------------------------------

    const REAL: &str = "/mnt/ssd1/data/camels_hourly/usgs-streamflow-nldas_hourly.nc";

    #[test]
    fn real_store_opens_with_expected_schema() {
        if !Path::new(REAL).exists() {
            eprintln!("skipping: {REAL} not present");
            return;
        }
        let store = CamelsHourlyStore::open(REAL).unwrap();
        assert_eq!(store.index.len(), 516);
        assert_eq!(store.n_hours, 556421);
        assert_eq!(
            store.time_start,
            NaiveDateTime::parse_from_str("1956-12-07 09:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );

        // A known real gauge must be present (zero-padded) and readable.
        // Row 0 (1956-12-07) predates this gauge's period of record (it
        // starts 1989-10-01) -- that's a real, expected data gap, not a
        // reader bug (mirrors GlobalObservationsStore's BOM-gauge test).
        // Read a window inside the known period of record instead.
        let staid = Staid::new("01022500");
        assert!(store.index.position(&staid).is_some());
        let record_start = NaiveDateTime::parse_from_str("1989-10-01 04:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let (qobs, precip) = store.read_window(record_start, 24, &[staid]).unwrap();
        assert_eq!(qobs.shape(), &[24, 1]);
        assert_eq!(precip.shape(), &[24, 1]);
        // Real data: not all-NaN, no negative flows.
        assert!(qobs.iter().any(|v| v.is_finite()));
        assert!(qobs.iter().all(|v| v.is_nan() || *v >= 0.0));
        assert!(precip.iter().all(|v| v.is_finite())); // precip verified gap-free
    }
}
