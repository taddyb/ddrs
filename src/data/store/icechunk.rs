//! Icechunk-backed time-series readers.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/readers.py::read_ic` (lines ~376-403),
//! `StreamflowReader` (~lines 405-468), and `IcechunkUSGSReader`
//! (~lines 478-510). Both production stores live on a local filesystem
//! under `/mnt/ssd1/data/icechunk/`; S3 access is out of scope.
//!
//! Each store owns a `tokio::runtime::Runtime` and exposes a sync
//! `read_window(&RhoWindow, ids)` API — `block_on` happens at the icechunk
//! boundary. The dataset (SP-3) may later consolidate to a shared runtime
//! if profiling demands it.
//!
//! Adapter strategy: **B — icechunk-native `Store`**. The icechunk crate has
//! no zarrs dependency, so zarrs `ReadableStorage` is unavailable. Instead,
//! `IcSession` holds an `icechunk::Store` (a `Session`-backed Zarr key-value
//! handle) and chunk fetches go through `runtime.block_on(store.get(...))`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::NaiveDate;
use tokio::runtime::Runtime;
use tokio::sync::RwLock;

use icechunk::{Repository, Store, new_local_filesystem_storage};
use icechunk::repository::VersionInfo;
use zarrs::array::Array as ZarrArray;
use zarrs::storage::{
    MaybeBytesIterator, ReadableStorage, ReadableStorageTraits, StorageError, StoreKey,
};

use ndarray::Array2;

use crate::data::dates::RhoWindow;
use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, IdIndex, Staid};

/// Shared session handle. Internal — never leaks past this module.
#[allow(dead_code)] // Tasks 2-4 will use the fields.
pub(crate) struct IcSession {
    pub(crate) runtime: Arc<Runtime>,
    pub(crate) store: Arc<Store>,
}

/// Open an icechunk repo at `path` and start a read-only session on the
/// `main` branch.
#[allow(dead_code)] // Tasks 2-4 will call this.
pub(crate) fn open_session(path: &Path) -> Result<IcSession> {
    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| DataError::Io {
                path: path.to_path_buf(),
                source: e,
            })?,
    );

    let store = runtime.block_on(async {
        let storage = new_local_filesystem_storage(path)
            .await
            .map_err(|e| ic_err(path, e))?;

        let repo = Repository::open(None, storage, std::collections::HashMap::new())
            .await
            .map_err(|e| ic_err(path, e))?;

        let session = repo
            .readonly_session(&VersionInfo::BranchTipRef("main".into()))
            .await
            .map_err(|e| ic_err(path, e))?;

        let store = Store::from_session(Arc::new(RwLock::new(session))).await;
        Ok::<Arc<Store>, DataError>(Arc::new(store))
    })?;

    Ok(IcSession { runtime, store })
}

#[allow(dead_code)] // Used by the helpers in this module once filled in.
pub(crate) fn ic_err<E: std::error::Error + Send + Sync + 'static>(
    path: &Path,
    e: E,
) -> DataError {
    DataError::IceChunk {
        path: path.to_path_buf(),
        source: Box::new(e),
    }
}

// ---------------------------------------------------------------------------
// zarrs storage adapter
// ---------------------------------------------------------------------------

/// `zarrs::ReadableStorageTraits` adapter over an icechunk `Store`. Each
/// `get_partial_many` call `block_on`s `Store::get(key, &ByteRange::ALL)` and
/// then slices the result for each requested byte range. Caches no state — the
/// inner `Arc<Store>` is shared across the per-var Array handles owned by
/// `StreamflowStore`.
pub(crate) struct IcZarrStorage {
    pub(crate) store: Arc<Store>,
    pub(crate) runtime: Arc<Runtime>,
}

impl IcZarrStorage {
    pub(crate) fn shared(session: &IcSession) -> Arc<Self> {
        Arc::new(Self {
            store: session.store.clone(),
            runtime: session.runtime.clone(),
        })
    }
}

impl ReadableStorageTraits for IcZarrStorage {
    /// Retrieve partial bytes from a list of byte ranges for a store key.
    ///
    /// We don't support partial reads (see `supports_get_partial`), so zarrs
    /// will only call this with a single `FromStart(0, None)` range. We fetch
    /// the full chunk once and slice for every requested range.
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: zarrs::storage::byte_range::ByteRangeIterator<'a>,
    ) -> std::result::Result<MaybeBytesIterator<'a>, StorageError> {
        let key_str = key.as_str().to_string();

        // Fetch the whole value once.
        let maybe_bytes = self.runtime.block_on(async {
            match self.store.get(&key_str, &icechunk::format::ByteRange::ALL).await {
                Ok(bytes) => Ok(Some(bytes)),
                Err(e) if matches!(e.kind(), icechunk::store::StoreErrorKind::NotFound(_)) => {
                    Ok(None)
                }
                Err(e) => Err(StorageError::Other(e.to_string())),
            }
        })?;

        let Some(full_bytes) = maybe_bytes else {
            return Ok(None);
        };

        let size = full_bytes.len() as u64;
        let slices: Vec<std::result::Result<zarrs::storage::Bytes, StorageError>> = byte_ranges
            .map(|br| {
                let range = br.to_range_usize(size);
                Ok(zarrs::storage::Bytes::copy_from_slice(&full_bytes[range]))
            })
            .collect();

        Ok(Some(Box::new(slices.into_iter())))
    }

    /// Size is unknown without an extra round-trip; return `None`. zarrs falls
    /// back to reading the whole value when size is unknown, which is fine.
    fn size_key(
        &self,
        _key: &StoreKey,
    ) -> std::result::Result<Option<u64>, StorageError> {
        Ok(None)
    }

    /// We do not support efficient partial reads; zarrs will fall back to a
    /// full read for every `get_partial` call.
    fn supports_get_partial(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// StreamflowStore
// ---------------------------------------------------------------------------

/// `Qr` reader over `merit_dhbv2_UH_retrospective.ic`-style icechunk repos.
///
/// Opened once at dataset construction. Task 3 will add `read_window`.
pub struct StreamflowStore {
    pub path: PathBuf,
    pub index: IdIndex<Comid>,
    pub time_start: NaiveDate,
    pub n_time: usize,
    // SP-3 may consolidate to a shared runtime; keep the Arc alive so the
    // icechunk Store is not dropped while `qr` is in use.
    #[allow(dead_code)]
    storage: Arc<IcZarrStorage>,
    qr: ZarrArray<dyn ReadableStorageTraits>,
}

impl StreamflowStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let session = open_session(&path)?;
        let storage: Arc<IcZarrStorage> = IcZarrStorage::shared(&session);
        // zarrs Array::open takes Arc<dyn ReadableStorageTraits> — cast via type alias
        let readable: ReadableStorage = storage.clone();

        // 1. Read `time` coord: shape (n_time,), dtype int64.
        //    The encoding is CF-convention "days since YYYY-MM-DD" (units attr).
        let time_arr = ZarrArray::open(readable.clone(), "/time")
            .map_err(|e| ic_err(&path, e))?;
        // Parse the epoch from the `units` attribute ("days since 1980-01-01").
        let time_epoch = parse_cf_epoch(time_arr.attributes(), &path)?;
        let time_subset = time_arr.subset_all();
        let time_i64: Vec<i64> = time_arr
            .retrieve_array_subset(&time_subset)
            .map_err(|e| ic_err(&path, e))?;
        let n_time = time_i64.len();
        if n_time == 0 {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: "time axis is empty".into(),
            });
        }
        let time_start = time_epoch
            + chrono::Duration::days(time_i64[0]);

        // 2. Read `divide_id` coord; build IdIndex<Comid>.
        let div_arr = ZarrArray::open(readable.clone(), "/divide_id")
            .map_err(|e| ic_err(&path, e))?;
        let div_subset = div_arr.subset_all();
        let div_i64: Vec<i64> = div_arr
            .retrieve_array_subset(&div_subset)
            .map_err(|e| ic_err(&path, e))?;
        let index = IdIndex::new(div_i64.into_iter().map(Comid).collect());

        // 3. Open the `/Qr` data var (Task 3 will actually read from it).
        let qr = ZarrArray::open(readable.clone(), "/Qr")
            .map_err(|e| ic_err(&path, e))?;

        Ok(Self { path, index, time_start, n_time, storage, qr })
    }
}

/// Parse the CF `units` attribute of a time coordinate and return the epoch
/// plus the native axis resolution. Supported forms (see
/// docs/nh-qprime-store-contract.md):
///   "days since YYYY-MM-DD[ HH:MM:SS]"  → Daily
///   "hours since YYYY-MM-DD[ HH:MM:SS]" → Hourly
/// Anything else is a hard error naming the store and the units string — a
/// mis-scaled time axis must never be silently accepted.
pub(crate) fn parse_cf_units(
    attrs: &serde_json::Map<String, serde_json::Value>,
    path: &Path,
) -> Result<(NaiveDate, crate::data::dates::Frequency)> {
    use crate::data::dates::Frequency;

    let units = attrs
        .get("units")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DataError::Malformed {
            path: path.to_path_buf(),
            message: "time array missing 'units' attribute".into(),
        })?;
    let (date_str, resolution) = if let Some(rest) = units.strip_prefix("days since ") {
        (rest, Frequency::Daily)
    } else if let Some(rest) = units.strip_prefix("hours since ") {
        (rest, Frequency::Hourly)
    } else {
        return Err(DataError::Malformed {
            path: path.to_path_buf(),
            message: format!(
                "unsupported time units {units:?}: expected \"days since …\" \
                 or \"hours since …\""
            ),
        });
    };
    // The date portion may be followed by a time-of-day component, e.g.
    // "1981-01-01 00:00:00" — take only the first token.
    let date_part = date_str.split_whitespace().next().unwrap_or("");
    let epoch =
        NaiveDate::parse_from_str(date_part, "%Y-%m-%d").map_err(|e| DataError::Malformed {
            path: path.to_path_buf(),
            message: format!("cannot parse epoch from units {units:?}: {e}"),
        })?;
    Ok((epoch, resolution))
}

/// Daily-only wrapper for stores whose axis MUST be daily (USGS observations).
pub(crate) fn parse_cf_epoch(
    attrs: &serde_json::Map<String, serde_json::Value>,
    path: &Path,
) -> Result<NaiveDate> {
    match parse_cf_units(attrs, path)? {
        (epoch, crate::data::dates::Frequency::Daily) => Ok(epoch),
        (_, crate::data::dates::Frequency::Hourly) => Err(DataError::Malformed {
            path: path.to_path_buf(),
            message: "expected a daily time axis (\"days since …\"), got hourly".into(),
        }),
    }
}

/// Repeat a `(rho_days, N)` daily slab to `(n_hourly, N)` by replicating
/// each row 24 times along the time axis, then trim to `n_hourly` rows.
/// Mirrors `np.repeat(daily, 24, axis=1)[:, :n_hourly].T` in
/// `~/projects/ddr/src/ddr/io/readers.py:447-454` (DDR transposes after; we
/// yield time-major directly).
pub(crate) fn daily_to_hourly_trim(daily: &Array2<f32>, n_hourly: usize) -> Array2<f32> {
    let (rho_days, n_div) = daily.dim();
    debug_assert!(
        n_hourly <= rho_days * 24,
        "n_hourly={n_hourly} exceeds rho_days*24={}",
        rho_days * 24
    );
    let mut hourly = Array2::<f32>::zeros((n_hourly, n_div));
    for h in 0..n_hourly {
        let d = h / 24;
        for j in 0..n_div {
            hourly[(h, j)] = daily[(d, j)];
        }
    }
    hourly
}

impl StreamflowStore {
    /// Read `Qr` daily for `[window_start, window_start + n_days)` and
    /// `comids`. Returns `(n_days, N)` f32 matrix; missing COMIDs are
    /// filled with `0.001` (discharge minimum, mirrors DDR's
    /// `torch.full(..., fill_value=0.001)` in `readers.py:464-468`).
    ///
    /// Used directly by the summed Q' baseline (which needs daily output
    /// over a 15-yr window where the hourly form would be ~8.5 GB).
    /// `read_window` and `read_test_window` wrap this and add the
    /// daily → hourly repeat.
    pub fn read_window_daily(
        &self,
        window_start: NaiveDate,
        n_days: usize,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        // 1. Resolve time window to store-local day indices.
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

        // 2. Resolve COMIDs → divide-axis positions.
        // `positions_of` returns positions in the order of non-missing inputs,
        // plus a list of indices (into `comids`) that were missing.
        let (positions, missing_indices) = self.index.positions_of(comids);
        let missing_set: std::collections::HashSet<usize> =
            missing_indices.iter().copied().collect();
        let n_out = comids.len();

        // Pre-fill with the discharge minimum; missing COMIDs keep this value.
        let mut daily = Array2::<f32>::from_elem((n_days, n_out), 0.001);

        if positions.is_empty() {
            // All COMIDs missing — return filled daily result.
            return Ok(daily);
        }

        // 3. Contiguous divide-axis read covering [min_pos, max_pos].
        // Transient memory: (max_pos - min_pos + 1) * n_days * 4 bytes.
        // For 50 COMIDs spanning ~100K positions × 90 days = ~36 MB — acceptable
        // for SP-2. SP-3 may revisit with gather-style reads.
        let min_pos = *positions.iter().min().unwrap();
        let max_pos = *positions.iter().max().unwrap();
        let div_range_end = max_pos + 1;
        let div_count = div_range_end - min_pos;

        // Qr is stored as (divide_id, time). Subset: axis 0 = divide, axis 1 = time.
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            (min_pos as u64)..(div_range_end as u64),
            (store_start_day as u64)..(end_day as u64),
        ]);
        let raw_f32: Vec<f32> = self
            .qr
            .retrieve_array_subset(&subset)
            .map_err(|e| ic_err(&self.path, e))?;
        // raw_f32 is row-major: shape (div_count, n_days).
        // Element at (i, t) is at index i * n_days + t.
        debug_assert_eq!(raw_f32.len(), div_count * n_days);

        // 4. Scatter into the output. Walk `comids` in order; for each
        // non-missing entry consume the next element of `positions`.
        let mut next_present = 0usize;
        for (out_col, _) in comids.iter().enumerate() {
            if missing_set.contains(&out_col) {
                // Already pre-filled with 0.001.
                continue;
            }
            let div_pos = positions[next_present];
            next_present += 1;
            let local_div = div_pos - min_pos;
            for d in 0..n_days {
                let raw_idx = local_div * n_days + d;
                daily[(d, out_col)] = raw_f32[raw_idx];
            }
        }

        debug_assert_eq!(
            next_present,
            positions.len(),
            "scatter walked past `positions` — IdIndex::positions_of invariant broken"
        );

        Ok(daily)
    }

    /// Read `Qr` for `window` and `comids`. Returns `(n_hourly, N)` f32
    /// matrix; missing COMIDs (not in the store) are filled with `0.001`
    /// (discharge minimum, mirrors DDR's `torch.full(..., fill_value=0.001)`
    /// in `readers.py:464-468`).
    pub fn read_window(&self, window: &RhoWindow, comids: &[Comid]) -> Result<Array2<f32>> {
        let daily = self.read_window_daily(window.window_start, window.rho_days, comids)?;
        Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
    }

    /// Same as `read_window` but for `TestWindow` — returns `n_days * 24`
    /// hours (no trailing-day trim) so chunks tile cleanly. Used by SP-5
    /// `evaluate()`.
    pub fn read_test_window(
        &self,
        window: &crate::data::TestWindow,
        comids: &[Comid],
    ) -> Result<Array2<f32>> {
        let daily = self.read_window_daily(window.window_start, window.n_days, comids)?;
        Ok(daily_to_hourly_trim(&daily, window.n_hourly()))
    }
}

// ---------------------------------------------------------------------------
// UsgsObservationsStore
// ---------------------------------------------------------------------------

/// `streamflow` observations reader over `usgs_daily_observations`-style
/// icechunk repos.
///
/// Mirrors `~/projects/ddr/src/ddr/io/readers.py::IcechunkUSGSReader`
/// (lines ~478-510). Differences from `StreamflowStore`:
///   - Indexed by `Staid` (string), not `Comid` (int64).
///   - Missing STAIDs cause a hard `DataError::MissingIds` error — matches
///     DDR's `.sel(gage_id=...)` KeyError behavior. Streamflow misses are
///     a fact of life (not every COMID has DHBv2 coverage); observation
///     misses are a configuration bug.
///   - No daily→hourly transform: loss is computed at daily resolution.
///   - `streamflow` on disk is f64; we cast to f32 at read time.
pub struct UsgsObservationsStore {
    pub path: PathBuf,
    pub index: IdIndex<Staid>,
    pub time_start: NaiveDate,
    pub n_time: usize,
    #[allow(dead_code)]
    storage: Arc<IcZarrStorage>,
    streamflow: ZarrArray<dyn ReadableStorageTraits>,
}

impl UsgsObservationsStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let session = open_session(&path)?;
        let storage: Arc<IcZarrStorage> = IcZarrStorage::shared(&session);
        let readable: ReadableStorage = storage.clone();

        // 1. `time` coord — CF-convention "days since YYYY-MM-DD[ HH:MM:SS]".
        let time_arr = ZarrArray::open(readable.clone(), "/time")
            .map_err(|e| ic_err(&path, e))?;
        let time_epoch = parse_cf_epoch(time_arr.attributes(), &path)?;
        let time_subset = time_arr.subset_all();
        let time_i64: Vec<i64> = time_arr
            .retrieve_array_subset(&time_subset)
            .map_err(|e| ic_err(&path, e))?;
        let n_time = time_i64.len();
        if n_time == 0 {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: "time axis is empty".into(),
            });
        }
        let time_start = time_epoch + chrono::Duration::days(time_i64[0]);

        // 2. `gage_id` coord — zarr v3 `string` dtype with `vlen-utf8` codec.
        let staids = read_gage_id_coord(&readable, &path)?;
        let index = IdIndex::new(staids);

        // 3. Open `/streamflow` (f64 on disk; cast to f32 at read time).
        let streamflow = ZarrArray::open(readable.clone(), "/streamflow")
            .map_err(|e| ic_err(&path, e))?;

        Ok(Self {
            path,
            index,
            time_start,
            n_time,
            storage,
            streamflow,
        })
    }

    /// Read `streamflow` observations daily for
    /// `[window_start, window_start + n_days)` and `staids`. Returns
    /// `(n_days, G)` f32 matrix. Missing STAIDs trigger
    /// `DataError::MissingIds` — observation misses are configuration bugs.
    ///
    /// Used directly by the summed Q' baseline. `read_window` wraps this
    /// for `RhoWindow`-shaped callers; the body is identical because
    /// observations are already daily (no hourly transform).
    pub fn read_window_daily(
        &self,
        window_start: NaiveDate,
        n_days: usize,
        staids: &[Staid],
    ) -> Result<Array2<f32>> {
        // 1. Time window validation (same pattern as StreamflowStore).
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

        // 2. Resolve STAIDs. Hard-error on misses.
        let (positions, missing_indices) = self.index.positions_of(staids);
        if !missing_indices.is_empty() {
            return Err(DataError::MissingIds {
                path: self.path.clone(),
                kind: "gage_id",
                missing: missing_indices.len(),
                total: staids.len(),
            });
        }
        debug_assert_eq!(positions.len(), staids.len());

        if positions.is_empty() {
            return Ok(Array2::<f32>::zeros((n_days, 0)));
        }

        // 3. Read contiguous gage-axis range covering [min_pos, max_pos].
        let min_pos = *positions.iter().min().unwrap();
        let max_pos = *positions.iter().max().unwrap();
        let gage_range_end = max_pos + 1;
        let gage_count = gage_range_end - min_pos;

        // streamflow is stored as (gage_id, time) — axis 0 = gage, axis 1 = time.
        let subset = zarrs::array::ArraySubset::new_with_ranges(&[
            (min_pos as u64)..(gage_range_end as u64),
            (store_start_day as u64)..(end_day as u64),
        ]);
        // On disk the array is f64; cast to f32 at the scatter site.
        let raw_f64: Vec<f64> = self
            .streamflow
            .retrieve_array_subset(&subset)
            .map_err(|e| ic_err(&self.path, e))?;
        debug_assert_eq!(raw_f64.len(), gage_count * n_days);

        // 4. Scatter to output preserving input order. `positions[i]`
        // corresponds to `staids[i]` because all inputs are present (no
        // missing_indices path taken above).
        debug_assert_eq!(positions.len(), staids.len());
        let mut out = Array2::<f32>::zeros((n_days, staids.len()));
        for (out_col, _) in staids.iter().enumerate() {
            let pos = positions[out_col];
            let local_gage = pos - min_pos;
            for d in 0..n_days {
                let raw_idx = local_gage * n_days + d;
                out[(d, out_col)] = raw_f64[raw_idx] as f32;
            }
        }
        Ok(out)
    }

    /// `RhoWindow`-shaped wrapper for `read_window_daily`. Returns
    /// `(rho_days, G)` f32; observations are already daily so no
    /// hourly transform.
    pub fn read_window(
        &self,
        window: &RhoWindow,
        staids: &[Staid],
    ) -> Result<Array2<f32>> {
        self.read_window_daily(window.window_start, window.rho_days, staids)
    }
}

/// Read the `/gage_id` string coord from an icechunk-backed zarr store.
///
/// The on-disk dtype is zarr v3 `"string"` with `vlen-utf8` codec; zarrs
/// decodes it natively as `Vec<String>`. See zarr.json for the store:
///   `"data_type": "string"`, `"codecs": [{"name": "vlen-utf8"}, ...]`.
fn read_gage_id_coord(
    storage: &ReadableStorage,
    path: &Path,
) -> Result<Vec<Staid>> {
    let arr = ZarrArray::open(storage.clone(), "/gage_id")
        .map_err(|e| ic_err(path, e))?;
    let subset = arr.subset_all();

    // Approach A: zarrs native string decode (vlen-utf8 → Vec<String>).
    if let Ok(vs) = arr.retrieve_array_subset::<Vec<String>>(&subset) {
        return Ok(vs
            .into_iter()
            .map(|s| Staid::new(s.trim_end_matches('\0').trim()))
            .collect());
    }

    // Approach B: fixed-length UTF-32 ('<U8'): 8 u32 codepoints per gage.
    if let Ok(codepoints) = arr.retrieve_array_subset::<Vec<u32>>(&subset) {
        let chunk_size = 8usize;
        if codepoints.len() % chunk_size == 0 {
            let n = codepoints.len() / chunk_size;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let slice = &codepoints[i * chunk_size..(i + 1) * chunk_size];
                let s: String = slice
                    .iter()
                    .take_while(|&&c| c != 0)
                    .filter_map(|&c| char::from_u32(c))
                    .collect();
                out.push(Staid::new(&s));
            }
            return Ok(out);
        }
    }

    // Approach C: raw fixed-length ASCII/Latin-1 bytes (8 bytes per gage).
    if let Ok(bytes) = arr.retrieve_array_subset::<Vec<u8>>(&subset) {
        let chunk_size = 8usize;
        if bytes.len() % chunk_size == 0 {
            let n = bytes.len() / chunk_size;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let slice = &bytes[i * chunk_size..(i + 1) * chunk_size];
                let s = String::from_utf8_lossy(slice)
                    .trim_end_matches('\0')
                    .trim()
                    .to_string();
                out.push(Staid::new(&s));
            }
            return Ok(out);
        }
    }

    Err(DataError::Malformed {
        path: path.to_path_buf(),
        message: "could not decode /gage_id (tried Vec<String>, Vec<u32> UTF-32, Vec<u8>)".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn daily_to_hourly_trim_repeats_and_truncates() {
        use ndarray::Array2;
        // 3 daily values × 2 divides → expand to 72 hours, truncate to 47
        // (which is what (rho_days - 1) * 24 yields for rho_days=3, matching DDR's
        // pd.date_range(... inclusive="left") semantics).
        let daily: Array2<f32> = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
        )
        .unwrap();
        let hourly = daily_to_hourly_trim(&daily, 47);
        assert_eq!(hourly.shape(), &[47, 2]);
        for h in 0..24 {
            assert_eq!(hourly[(h, 0)], 1.0);
            assert_eq!(hourly[(h, 1)], 10.0);
        }
        // Hours 24..47 fall in day 1.
        for h in 24..47 {
            assert_eq!(hourly[(h, 0)], 2.0);
            assert_eq!(hourly[(h, 1)], 20.0);
        }
    }

    #[test]
    fn open_streamflow_store_if_present() {
        let p = Path::new("/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic");
        if !p.exists() {
            return;
        }
        assert!(open_session(p).is_ok());
    }

    #[test]
    fn streamflow_read_window_returns_expected_shape() {
        let p = Path::new("/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic");
        if !p.exists() {
            eprintln!("skipping: {p:?} not present");
            return;
        }
        let store = StreamflowStore::open(p).expect("open");

        // First 10 COMIDs from the store's own divide_id index — guaranteed
        // present, no fills needed.
        let comids: Vec<Comid> = store.index.ids().iter().take(10).copied().collect();

        // RhoWindow starting at 1981-10-01 (MERIT training start) + 90 days.
        let axis = crate::data::dates::TimeAxis::new(
            chrono::NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(1981, 12, 31).unwrap(),
        );
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let window = axis.sample_rho_window(&mut rng, 90);

        let q = store.read_window(&window, &comids).expect("read_window");
        assert_eq!(q.shape(), &[window.n_hourly(), 10]);
        // No fill column expected (we used real COMIDs from the store).
        for &v in q.iter() {
            assert!(v.is_finite(), "got non-finite: {v}");
        }
    }

    #[test]
    fn streamflow_store_open_sees_expected_axes() {
        let p = Path::new("/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic");
        if !p.exists() {
            eprintln!("skipping: {p:?} not present");
            return;
        }
        let s = StreamflowStore::open(p).expect("open streamflow");
        assert!(s.n_time > 14000, "expected many days; got {}", s.n_time);
        assert!(
            s.index.len() > 100_000,
            "expected many divides; got {}",
            s.index.len()
        );
        assert_eq!(
            s.time_start,
            chrono::NaiveDate::from_ymd_opt(1980, 1, 1).unwrap()
        );
    }

    #[test]
    fn observations_store_open_sees_expected_axes() {
        let p = Path::new("/mnt/ssd1/data/icechunk/usgs_daily_observations");
        if !p.exists() {
            eprintln!("skipping: {p:?} not present");
            return;
        }
        let store = UsgsObservationsStore::open(p).expect("open obs");
        assert!(store.n_time > 14000, "expected ~14610 days, got {}", store.n_time);
        assert!(store.index.len() > 8000, "expected ~9067 gages, got {}", store.index.len());
        assert_eq!(
            store.time_start,
            chrono::NaiveDate::from_ymd_opt(1980, 1, 1).unwrap()
        );
        // Spot-check that we got real STAIDs — known first one is "01011000".
        let first = &store.index.ids()[0];
        assert_eq!(
            first.as_str(),
            "01011000",
            "first STAID should be 01011000 (per design-time probe), got {}",
            first
        );
    }

    fn attrs_with_units(u: &str) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        m.insert("units".into(), serde_json::Value::String(u.into()));
        m
    }

    #[test]
    fn parse_cf_units_daily() {
        let (epoch, res) =
            parse_cf_units(&attrs_with_units("days since 1980-01-01"), Path::new("/t")).unwrap();
        assert_eq!(epoch, chrono::NaiveDate::from_ymd_opt(1980, 1, 1).unwrap());
        assert_eq!(res, crate::data::dates::Frequency::Daily);
    }

    #[test]
    fn parse_cf_units_daily_with_time_of_day() {
        // daily_lstm store encodes "days since 1981-01-01 00:00:00".
        let (epoch, res) =
            parse_cf_units(&attrs_with_units("days since 1981-01-01 00:00:00"), Path::new("/t"))
                .unwrap();
        assert_eq!(epoch, chrono::NaiveDate::from_ymd_opt(1981, 1, 1).unwrap());
        assert_eq!(res, crate::data::dates::Frequency::Daily);
    }

    #[test]
    fn parse_cf_units_hourly() {
        // hourly_lstm store encodes "hours since 1981-01-01 00:00:00".
        let (epoch, res) =
            parse_cf_units(&attrs_with_units("hours since 1981-01-01 00:00:00"), Path::new("/t"))
                .unwrap();
        assert_eq!(epoch, chrono::NaiveDate::from_ymd_opt(1981, 1, 1).unwrap());
        assert_eq!(res, crate::data::dates::Frequency::Hourly);
    }

    #[test]
    fn parse_cf_units_rejects_other_resolutions() {
        let err = parse_cf_units(&attrs_with_units("minutes since 1981-01-01"), Path::new("/t"))
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("minutes since"), "error must name the units: {msg}");
        assert!(msg.contains("days since"), "error must name what IS supported: {msg}");
    }

    #[test]
    fn parse_cf_epoch_rejects_hourly_axis() {
        // The daily-only wrapper (used by the USGS observations store) must
        // refuse an hourly axis rather than silently mis-scaling.
        let err = parse_cf_epoch(&attrs_with_units("hours since 1980-01-01"), Path::new("/t"))
            .unwrap_err();
        assert!(err.to_string().contains("daily"), "got: {err}");
    }
}
