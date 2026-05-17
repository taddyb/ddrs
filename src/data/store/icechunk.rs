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

use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, IdIndex};

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
    #[allow(dead_code)] // used by Task 3 (read_window)
    storage: Arc<IcZarrStorage>,
    #[allow(dead_code)] // used by Task 3 (read_window)
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

/// Parse `units` CF attribute of the form `"days since YYYY-MM-DD"` and
/// return the epoch as a `NaiveDate`.
fn parse_cf_epoch(
    attrs: &serde_json::Map<String, serde_json::Value>,
    path: &Path,
) -> Result<NaiveDate> {
    let units = attrs
        .get("units")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DataError::Malformed {
            path: path.to_path_buf(),
            message: "time array missing 'units' attribute".into(),
        })?;
    // Expected: "days since YYYY-MM-DD" (CF convention).
    let date_str = units
        .strip_prefix("days since ")
        .ok_or_else(|| DataError::Malformed {
            path: path.to_path_buf(),
            message: format!("unexpected time units format: {units:?}"),
        })?;
    NaiveDate::parse_from_str(date_str.trim(), "%Y-%m-%d").map_err(|e| DataError::Malformed {
        path: path.to_path_buf(),
        message: format!("cannot parse epoch from units {units:?}: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn open_streamflow_store_if_present() {
        let p = Path::new("/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic");
        if !p.exists() {
            return;
        }
        assert!(open_session(p).is_ok());
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
}
