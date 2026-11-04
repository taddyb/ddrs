//! Hourly water-level series for the stage block's prescribed downstream
//! boundary (`data_sources.stage_boundary`, `params.stage_routing.
//! downstream_boundary: prescribed_stage`).
//!
//! CSV header `time,stage_m`. Time is UTC and parsed with exactly
//! `%Y-%m-%dT%H:%M:%SZ` (e.g. `2000-01-01T00:00:00Z`). Rows are strictly
//! 1 h apart. Every value is finite (m, same vertical datum as the reach
//! `bed_z`).
//!
//! Length rule: routing step `k` interpolates between `series[k]` and
//! `series[k+1]`, so `H` steps need `H + 1` values. `MuskingumCunge::forward`
//! routes `T` hourly q' rows as `T - 1` steps, so the dataset slices `T`
//! values: value `k` is the instant of q' row `k`.

use std::path::{Path, PathBuf};

use chrono::{Duration, NaiveDateTime};

use crate::data::error::{DataError, Result};

/// The one accepted time format.
pub const TIME_FORMAT: &str = "%Y-%m-%dT%H:%M:%SZ";

#[derive(Debug, Clone)]
pub struct StageBoundaryStore {
    pub path: PathBuf,
    start: NaiveDateTime,
    values: Vec<f32>,
}

impl StageBoundaryStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let malformed = |message: String| DataError::Malformed {
            path: path.clone(),
            message,
        };
        let mut rdr = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_path(&path)
            .map_err(|source| DataError::Csv {
                path: path.clone(),
                source,
            })?;
        let headers = rdr
            .headers()
            .map_err(|source| DataError::Csv {
                path: path.clone(),
                source,
            })?
            .clone();
        let col = |name: &str| {
            headers
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| malformed(format!("missing required column `{name}`")))
        };
        let (ti, si) = (col("time")?, col("stage_m")?);

        let mut start: Option<NaiveDateTime> = None;
        let mut prev: Option<NaiveDateTime> = None;
        let mut values = Vec::new();
        for (i, rec) in rdr.records().enumerate() {
            let row = i + 1; // 1-based data row
            let rec = rec.map_err(|source| DataError::Csv {
                path: path.clone(),
                source,
            })?;
            let ts = rec.get(ti).unwrap_or("");
            let t = NaiveDateTime::parse_from_str(ts, TIME_FORMAT).map_err(|e| {
                malformed(format!(
                    "row {row}: time {ts:?} does not match {TIME_FORMAT} ({e})"
                ))
            })?;
            let vs = rec.get(si).unwrap_or("");
            let v: f32 = vs
                .parse()
                .map_err(|_| malformed(format!("row {row}: stage_m {vs:?} is not a number")))?;
            if !v.is_finite() {
                return Err(malformed(format!("row {row}: stage_m {vs:?} is not finite")));
            }
            if let Some(p) = prev {
                if t - p != Duration::hours(1) {
                    return Err(malformed(format!(
                        "row {row}: not contiguous, {} follows {} (rows must be exactly 1 h apart)",
                        t.format(TIME_FORMAT),
                        p.format(TIME_FORMAT)
                    )));
                }
            } else {
                start = Some(t);
            }
            prev = Some(t);
            values.push(v);
        }
        let start = start.ok_or_else(|| malformed("no data rows".into()))?;
        Ok(Self {
            path,
            start,
            values,
        })
    }

    /// `n_hours` values, the first at `start`. Errors when the range is not
    /// inside the record.
    pub fn slice(&self, start: NaiveDateTime, n_hours: usize) -> Result<Vec<f32>> {
        let offset = (start - self.start).num_hours();
        let end = self.start + Duration::hours(self.values.len() as i64);
        let inside = start >= self.start
            && (start - self.start) == Duration::hours(offset)
            && offset as usize + n_hours <= self.values.len();
        if !inside {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "requested {n_hours} hours from {} is outside the record [{}, {}) \
                     ({} values)",
                    start.format(TIME_FORMAT),
                    self.start.format(TIME_FORMAT),
                    end.format(TIME_FORMAT),
                    self.values.len()
                ),
            });
        }
        let o = offset as usize;
        Ok(self.values[o..o + n_hours].to_vec())
    }
}
