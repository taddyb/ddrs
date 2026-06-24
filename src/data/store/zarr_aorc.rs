//! Hourly AORC precipitation reader over the `merit_unit_catchments.zarr`
//! **zarr v3** store.
//!
//! Layout (`/mnt/ssd1/data/aorc/merit_unit_catchments.zarr`):
//!   - `total_precipitation` — `(catchment, time)` float32, **catchment-major**
//!     (the transpose of the global Q' store), chunks `(n_catchments, 48)`,
//!     `bytes`+`zstd`, fill `0.0`, units **mm/hr**.
//!   - `gauge_id` — `fixed_length_utf32` (`length_bytes=32`) strings that are
//!     MERIT COMIDs (`"71022453"`); parsed to `Comid(i64)`.
//!   - `date`     — hourly `datetime64[ns]`, `1980-01-01T00:00 …
//!     2020-12-31T23:00` (14,976 days = 359,424 hours). The time axis is
//!     therefore byte-aligned with the streamflow Q' axis: hour rows
//!     `[t·24 … (t+1)·24)` are day `t` (days since 1980-01-01). Established
//!     empirically 2026-06-22 (see `gauge_id`/`date` probe).
//!
//! This store provides the within-day *shape* signal for the precip-driven
//! disaggregation head (`src/nn/disagg_head.rs`). It is **CONUS-only**: the
//! AORC fabric covers 290,878 of the 346,321 CONUS MERIT reaches; COMIDs
//! absent from the store are filled with `0.0` (dry-equivalent → the head's
//! softmax sees a flat precip window for that reach and falls back to the
//! daily-Q / attribute shape).
//!
//! Unlike `q_prime`, precip is **not** flow-scaled — it is a shape signal,
//! normalized per-reach in the data-batching layer before the head sees it
//! (`src/data/dataset.rs`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::NaiveDate;
use ndarray::Array2;
use zarrs::array::{Array as ZarrArray, ArrayBytes, ArraySubset};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ReadableStorage, ReadableStorageTraits};

use crate::data::dates::RhoWindow;
use crate::data::error::{DataError, Result};
use crate::data::ids::Comid;
use crate::data::TestWindow;

/// Implicit epoch of the AORC time axis. Verified against `date[0]`
/// (`1980-01-01T00:00`) and hourly spacing (`date[1] = 01:00`).
pub const AORC_EPOCH: (i32, u32, u32) = (1980, 1, 1);

/// Hourly AORC precipitation reader.
pub struct AorcPrecipStore {
    pub path: PathBuf,
    /// Calendar date of hour index 0.
    pub time_start: NaiveDate,
    /// Total number of hourly steps (`total_precipitation` time extent).
    pub n_time: usize,
    /// Number of catchment rows.
    n_catchments: usize,
    /// Time-axis chunk extent (read window is tiled in these blocks so a read
    /// never materializes more than one time-chunk of the full catchment
    /// width at once).
    time_chunk: usize,
    /// COMID → catchment row index.
    by_comid: HashMap<Comid, usize>,
    precip: ZarrArray<dyn ReadableStorageTraits>,
    /// Hourly 2-m air temperature (K), same `(catchment, time)` layout. Used by
    /// the disaggregation head's optional temperature channel.
    temperature: ZarrArray<dyn ReadableStorageTraits>,
}

impl AorcPrecipStore {
    /// True if `path` looks like this store: a zarr v3 group (`zarr.json` with
    /// a `total_precipitation` array child). Used for opt-in dispatch.
    pub fn sniff(path: &Path) -> bool {
        path.join("zarr.json").is_file()
            && path.join("total_precipitation").join("zarr.json").is_file()
    }

    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let (y, m, d) = AORC_EPOCH;
        let time_start = NaiveDate::from_ymd_opt(y, m, d).unwrap();

        let storage: ReadableStorage =
            Arc::new(FilesystemStore::new(&path).map_err(|e| zarr_err(&path, e))?);

        let precip = ZarrArray::open(storage.clone(), "/total_precipitation")
            .map_err(|e| zarr_err(&path, e))?;
        let shape = precip.shape();
        if shape.len() != 2 {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: format!("total_precipitation must be 2-D, got shape {shape:?}"),
            });
        }
        let n_catchments = shape[0] as usize;
        let n_time = shape[1] as usize;
        let time_chunk = precip.chunk_grid_shape()[1] as usize;

        let temperature = ZarrArray::open(storage.clone(), "/temperature")
            .map_err(|e| zarr_err(&path, e))?;
        if temperature.shape() != shape {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: format!(
                    "temperature shape {:?} != total_precipitation {shape:?}",
                    temperature.shape()
                ),
            });
        }

        // gauge_id: fixed_length_utf32 → decode the raw bytes ourselves
        // (zarrs has no String element mapping for this extension dtype).
        let gid = ZarrArray::open(storage.clone(), "/gauge_id").map_err(|e| zarr_err(&path, e))?;
        let gid_len = gid.shape()[0] as usize;
        if gid_len != n_catchments {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: format!(
                    "gauge_id length {gid_len} != total_precipitation catchments {n_catchments}"
                ),
            });
        }
        let bytes: ArrayBytes = gid
            .retrieve_array_subset(&ArraySubset::new_with_ranges(&[0..gid_len as u64]))
            .map_err(|e| zarr_err(&path, e))?;
        let raw = bytes.into_fixed().map_err(|e| DataError::Malformed {
            path: path.clone(),
            message: format!("gauge_id not a fixed-length array: {e:?}"),
        })?;
        // fixed_length_utf32 → 4 bytes/char; the element size is the array's
        // total fixed-byte width / element count (32 for this store).
        let elem_bytes = raw.len() / n_catchments;

        let mut by_comid = HashMap::with_capacity(n_catchments);
        for row in 0..n_catchments {
            let off = row * elem_bytes;
            let s = decode_utf32le(&raw[off..off + elem_bytes]);
            let comid = s.parse::<i64>().map_err(|_| DataError::Malformed {
                path: path.clone(),
                message: format!("gauge_id[{row}] = {s:?} is not an integer COMID"),
            })?;
            by_comid.insert(Comid(comid), row);
        }

        Ok(Self {
            path,
            time_start,
            n_time,
            n_catchments,
            time_chunk,
            by_comid,
            precip,
            temperature,
        })
    }

    /// Number of catchments with precip series.
    pub fn n_catchments(&self) -> usize {
        self.n_catchments
    }

    /// How many of `comids` are present in the store (the rest take the 0.0
    /// fill). Used to log per-batch coverage.
    pub fn coverage(&self, comids: &[Comid]) -> usize {
        comids.iter().filter(|c| self.by_comid.contains_key(c)).count()
    }

    /// Read hourly precip for `n_hourly` hours starting at `window_start`,
    /// gathered to `comids`. Returns `(n_hourly, N)` f32; COMIDs absent from
    /// the store are filled with `0.0` (dry-equivalent).
    pub fn read_window_hourly(
        &self,
        window_start: NaiveDate,
        n_hourly: usize,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        self.read_var_hourly(&self.precip, window_start, n_hourly, comids)
    }

    /// Read hourly 2-m air temperature (K), same contract as
    /// [`read_window_hourly`]. Non-coverage / NaN catchments → `0.0` (a
    /// constant column → neutral after per-reach z-score in the data layer).
    pub fn read_temp_hourly(
        &self,
        window_start: NaiveDate,
        n_hourly: usize,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        self.read_var_hourly(&self.temperature, window_start, n_hourly, comids)
    }

    /// Gather `n_hourly` hours from `array` (a `(catchment, time)` AORC field)
    /// starting at `window_start`, to `comids`. Non-finite values and absent
    /// COMIDs → `0.0`.
    ///
    /// The read is tiled over the time axis in `time_chunk`-sized blocks so a
    /// single retrieve never assembles more than one time-chunk of the
    /// (catchment-major) array.
    fn read_var_hourly(
        &self,
        array: &ZarrArray<dyn ReadableStorageTraits>,
        window_start: NaiveDate,
        n_hourly: usize,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        let start_hour_i64 = (window_start - self.time_start).num_days() * 24;
        if start_hour_i64 < 0 {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window starts {window_start} before store start {}",
                    self.time_start
                ),
            });
        }
        let start_hour = start_hour_i64 as usize;
        let end_hour = start_hour + n_hourly;
        if end_hour > self.n_time {
            return Err(DataError::Malformed {
                path: self.path.clone(),
                message: format!(
                    "window extends to hour {end_hour} but n_time={}",
                    self.n_time
                ),
            });
        }

        let mut out = Array2::<f32>::zeros((n_hourly, comids.len()));

        // Present (out_col, catchment row), and the row span to retrieve.
        let mut present: Vec<(usize, usize)> = Vec::with_capacity(comids.len());
        for (out_col, c) in comids.iter().enumerate() {
            if let Some(&row) = self.by_comid.get(c) {
                present.push((out_col, row));
            }
        }
        if present.is_empty() {
            return Ok(out); // all-fill (e.g. global run) — keep zeros.
        }
        let row_min = present.iter().map(|&(_, r)| r).min().unwrap();
        let row_max = present.iter().map(|&(_, r)| r).max().unwrap();
        let row_span = row_max - row_min + 1;

        // Tile the time axis in chunk-sized blocks.
        let mut h = start_hour;
        while h < end_hour {
            let blk_end = (h - (h % self.time_chunk) + self.time_chunk).min(end_hour);
            let blk = blk_end - h;
            let subset = ArraySubset::new_with_ranges(&[
                (row_min as u64)..((row_max + 1) as u64),
                (h as u64)..(blk_end as u64),
            ]);
            // Row-major over (row_span, blk): element (r_local, t) at
            // r_local * blk + t.
            let raw: Vec<f32> = array
                .retrieve_array_subset(&subset)
                .map_err(|e| zarr_err(&self.path, e))?;
            debug_assert_eq!(raw.len(), row_span * blk);

            let out_row0 = h - start_hour;
            for &(out_col, row) in &present {
                let r_local = row - row_min;
                let base = r_local * blk;
                for t in 0..blk {
                    // The store carries real NaN (~14% of values: whole-catchment
                    // ocean / no AORC coverage) despite a 0.0 fill_value. Zero it
                    // so it can't poison the head (a constant column → neutral
                    // after per-reach normalization in the data layer).
                    let v = raw[base + t];
                    out[(out_row0 + t, out_col)] = if v.is_finite() { v } else { 0.0 };
                }
            }
            h = blk_end;
        }
        Ok(out)
    }

    /// Training rho-window read: `(rho_days-1)·24` hours from `window_start`.
    pub fn read_window(&self, window: &RhoWindow, comids: &[Comid]) -> Result<Array2<f32>> {
        self.read_window_hourly(window.window_start, window.n_hourly(), comids)
    }

    /// Test-window read: `n_days·24` hours from `window_start`.
    pub fn read_test_window(&self, window: &TestWindow, comids: &[Comid]) -> Result<Array2<f32>> {
        self.read_window_hourly(window.window_start, window.n_hourly(), comids)
    }

    /// Temperature rho-window read (training): `(rho_days-1)·24` hours.
    pub fn read_temp_window(&self, window: &RhoWindow, comids: &[Comid]) -> Result<Array2<f32>> {
        self.read_temp_hourly(window.window_start, window.n_hourly(), comids)
    }

    /// Temperature test-window read: `n_days·24` hours.
    pub fn read_temp_test_window(
        &self,
        window: &TestWindow,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        self.read_temp_hourly(window.window_start, window.n_hourly(), comids)
    }
}

/// Decode a fixed-length UTF-32-LE element to a `String`, stopping at the
/// first NUL codepoint (zarr's fixed-length string padding).
fn decode_utf32le(bytes: &[u8]) -> String {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .take_while(|&cp| cp != 0)
        .filter_map(char::from_u32)
        .collect()
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

    const REAL: &str = "/mnt/ssd1/data/aorc/merit_unit_catchments.zarr";

    #[test]
    fn decode_utf32le_parses_comid() {
        // "71022453" as UTF-32-LE, NUL-padded to 32 bytes.
        let mut buf = vec![0u8; 32];
        for (i, ch) in "71022453".chars().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&(ch as u32).to_le_bytes());
        }
        assert_eq!(decode_utf32le(&buf), "71022453");
    }

    // ------------------------------------------------------------------
    // Gated test against the real AORC store; skipped when absent.
    // ------------------------------------------------------------------
    #[test]
    fn real_store_opens_and_reads_aligned() {
        if !Path::new(REAL).exists() {
            eprintln!("skipping: {REAL} not present");
            return;
        }
        let store = AorcPrecipStore::open(REAL).unwrap();
        assert_eq!(store.n_catchments(), 290_878);
        assert_eq!(store.n_time, 359_424, "14976 days * 24");
        assert_eq!(store.time_start, NaiveDate::from_ymd_opt(1980, 1, 1).unwrap());

        // Read one day (24 h) for a known COMID + a missing one. The COMID
        // 71022453 is catchment row 0 (first gauge_id).
        let comids = [Comid(71022453), Comid(999_999_999)];
        let out = store
            .read_window_hourly(NaiveDate::from_ymd_opt(2000, 6, 1).unwrap(), 24, &comids)
            .unwrap();
        assert_eq!(out.shape(), &[24, 2]);
        // Missing COMID → all-zero fill.
        assert!(out.column(1).iter().all(|&v| v == 0.0));
        // Present COMID → finite, non-negative mm/hr.
        assert!(out.column(0).iter().all(|&v| v.is_finite() && v >= 0.0));
        // Coverage helper.
        assert_eq!(store.coverage(&comids), 1);

        // Block tiling across a chunk boundary (48-h chunk): read 100 h.
        let many = store
            .read_window_hourly(NaiveDate::from_ymd_opt(2000, 6, 1).unwrap(), 96, &comids)
            .unwrap();
        assert_eq!(many.shape(), &[96, 2]);
        assert!(many.column(0).iter().all(|&v| v.is_finite()));
    }
}
