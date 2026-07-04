//! Day-boundary discharge state cache reader.
//!
//! Reads the netCDF produced by `--mode state-cache` in `probe_zeta_gradient`.
//! Layout: dims `(day, COMID)`, var `q_state` f32, var `COMID` i64, global
//! attr `day0` (ISO date string `"YYYY-MM-DD"`).
//!
//! **Memory trade-off:** the full array is ~1.3 GB for a CONUS-scale run;
//! lazy row reads (~256 KB per `row_for_day` call, one file-open per call)
//! are preferred. At `batch_size = 64` over ~10 K gauges (~156 batches/epoch)
//! the file-open overhead is negligible vs the routing kernel cost.

use std::path::PathBuf;

use chrono::NaiveDate;
use ndarray::Array1;

use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, IdIndex};

/// Lazy reader for the day-boundary discharge state cache.
///
/// Created once via [`StateCache::open`]; each [`StateCache::row_for_day`]
/// call reopens the file and reads a single row (~256 KB for 64 K reaches).
///
/// ### Row-0 approximation
///
/// Cache row 0 is approximate: it captures per-reach discharge after the
/// **first hourly routing step** of day 0 (hour 1 of the continuous run),
/// not at the true hour-0 boundary. The error is one hourly routing step and
/// affects only training windows whose `window_start` exactly equals `day0`.
/// All rows ≥ 1 are exact day-boundary states. Consumers read rows by date
/// and need not special-case this; the approximation is bounded and benign.
pub struct StateCache {
    /// Path to the netCDF file (carried for error messages + lazy re-opens).
    pub path: PathBuf,
    /// COMID → column position in `q_state`.
    index: IdIndex<Comid>,
    /// Calendar date of row 0 (= the continuous run's time-axis start).
    pub day0: NaiveDate,
    /// Number of day rows in the cache.
    pub n_days: usize,
    /// Number of COMID columns in the cache.
    pub n_comids: usize,
}

impl StateCache {
    /// Open the cache netCDF and read COMID index + `day0` metadata.
    ///
    /// `q_state` values are NOT read at open time (lazy; ~1.3 GB for CONUS).
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let file = netcdf::open(&path).map_err(|e| DataError::NetCdf {
            path: path.clone(),
            source: e,
        })?;

        // Global `day0` attribute — ISO date "YYYY-MM-DD".
        let day0_str: String = file
            .attribute("day0")
            .ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: "state cache missing 'day0' global attribute".into(),
            })?
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
        let day0 =
            NaiveDate::parse_from_str(&day0_str, "%Y-%m-%d").map_err(|e| DataError::Malformed {
                path: path.clone(),
                message: format!(
                    "state cache 'day0' attribute parse failed ({day0_str:?}): {e}"
                ),
            })?;

        // Dimension lengths.
        let n_days = file
            .dimension("day")
            .ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: "state cache missing 'day' dimension".into(),
            })?
            .len();
        let n_comids = file
            .dimension("COMID")
            .ok_or_else(|| DataError::Malformed {
                path: path.clone(),
                message: "state cache missing 'COMID' dimension".into(),
            })?
            .len();

        // COMID coordinate variable → IdIndex for fast lookup.
        let comid_var = file.variable("COMID").ok_or_else(|| DataError::Malformed {
            path: path.clone(),
            message: "state cache missing 'COMID' variable".into(),
        })?;
        let comid_raw: Vec<i64> = comid_var
            .get_values::<i64, _>(..)
            .map_err(|e| DataError::NetCdf {
                path: path.clone(),
                source: e,
            })?;
        let index = IdIndex::new(comid_raw.into_iter().map(Comid).collect());

        Ok(Self {
            path,
            index,
            day0,
            n_days,
            n_comids,
        })
    }

    /// Read the discharge state vector at `date` (day-0 boundary), reordered
    /// to match `comids` order.
    ///
    /// Hard errors:
    /// - `date` is before `day0` or ≥ `day0 + n_days days` → range error
    /// - any COMID in `comids` is absent from the cache index → missing-COMID
    ///   error (the cache must cover the full training network)
    ///
    /// ### Row-0 approximation
    /// When `date == day0`, row 0 is returned. See the struct-level doc for
    /// the one-hourly-step approximation that affects only day-0 windows.
    pub fn row_for_day(&self, date: NaiveDate, comids: &[Comid]) -> Result<Array1<f32>> {
        // Bounds check.
        let delta = (date - self.day0).num_days();
        if delta < 0 || delta as usize >= self.n_days {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "state cache: date {date} is out of range \
                     (cache covers {day0} through {day0} + {n} days)",
                    day0 = self.day0,
                    n = self.n_days,
                ),
            });
        }
        let day_idx = delta as usize;

        // Missing-COMID check (before the expensive file open).
        let missing: Vec<Comid> = comids
            .iter()
            .filter(|c| !self.index.contains(c))
            .copied()
            .collect();
        if !missing.is_empty() {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "state cache: {n} COMID(s) not in cache index \
                     (first few: {few:?}); cache must cover the full training network",
                    n = missing.len(),
                    few = &missing[..missing.len().min(3)],
                ),
            });
        }

        // Lazy row read: open the file and read exactly one row.
        let file = netcdf::open(&self.path).map_err(|e| DataError::NetCdf {
            path: self.path.clone(),
            source: e,
        })?;
        let var = file.variable("q_state").ok_or_else(|| DataError::Malformed {
            path: self.path.clone(),
            message: "state cache missing 'q_state' variable".into(),
        })?;

        // `(&[start], &[count]).try_into()` maps to start[0]=day_idx, count[0]=1
        // on the day dim, and start[1]=0, count[1]=n_comids on the COMID dim.
        let extents: netcdf::Extents =
            (&[day_idx, 0_usize][..], &[1_usize, self.n_comids][..])
                .try_into()
                .map_err(|e| DataError::NetCdf {
                    path: self.path.clone(),
                    source: e,
                })?;
        let raw: Vec<f32> = var
            .get_values::<f32, _>(extents)
            .map_err(|e| DataError::NetCdf {
                path: self.path.clone(),
                source: e,
            })?;

        // Reorder raw (cache column order) to requested `comids` order.
        let mut out = Array1::<f32>::zeros(comids.len());
        for (out_pos, comid) in comids.iter().enumerate() {
            // SAFETY: all comids were verified present above.
            out[out_pos] = raw[self.index.position(comid).unwrap()];
        }
        Ok(out)
    }
}
