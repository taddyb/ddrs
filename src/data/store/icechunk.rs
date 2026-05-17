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

use std::path::Path;
use std::sync::Arc;

use tokio::runtime::Runtime;
use tokio::sync::RwLock;

use icechunk::{Repository, Store, new_local_filesystem_storage};
use icechunk::repository::VersionInfo;

use crate::data::error::{DataError, Result};

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
}
