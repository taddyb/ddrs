//! Read MERIT's binsparse-COO zarr v3 stores.
//!
//! Two stores live here:
//!
//!   - `ConusAdjacencyStore`: the full CONUS COO graph + per-reach `length_m`
//!     and `slope`, plus the `order` array (COMIDs in topological order).
//!     Eager-loaded once — small (~30 MB at 346K reaches, zstd-compressed).
//!
//!   - `GagesAdjacencyStore`: per-STAID subgraph COOs keyed by gauge.
//!     Eager-loaded for the chosen-gauge set only (a few MB).
//!
//! Both targets are zarr v3 with int32/uint8 arrays and `bytes` + `zstd`
//! codecs — see `ddr/engine/src/ddr_engine/core/zarr_io.py` for the writer.
//! We never expose `zarrs::Array` to callers; reads return `Vec<T>` /
//! `ndarray::Array1` with the foreign types contained.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array1;
use zarrs::array::Array as ZarrArray;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::ReadableStorage;

use crate::data::error::{DataError, Result};
use crate::data::ids::{Comid, IdIndex, Staid};

/// Static CONUS-wide network state. Loaded once at dataset construction.
pub struct ConusAdjacencyStore {
    pub path: PathBuf,
    /// COMIDs in topological order — element `i` is the COMID at zarr position `i`.
    pub order: Vec<Comid>,
    /// `IdIndex` mapping COMID → topological position (for cross-store lookups).
    pub index: IdIndex<Comid>,
    /// Per-reach channel length in metres, aligned to `order`.
    pub length_m: Array1<f32>,
    /// Per-reach channel slope (dimensionless), aligned to `order`.
    pub slope: Array1<f32>,
    /// COO row indices (downstream segment index in CONUS position space).
    pub indices_0: Vec<i32>,
    /// COO column indices (upstream segment index in CONUS position space).
    pub indices_1: Vec<i32>,
    /// Number of reaches (== `order.len()`).
    pub n: usize,
    /// Number of non-zero edges in the COO.
    pub nnz: usize,
}

impl ConusAdjacencyStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let storage: ReadableStorage =
            Arc::new(FilesystemStore::new(&path).map_err(|e| zarr_err(&path, e))?);
        let _root = Group::open(storage.clone(), "/").map_err(|e| zarr_err(&path, e))?;

        let order_i32 = read_array_i32(&storage, &path, "/order")?;
        let order: Vec<Comid> = order_i32.into_iter().map(|c| Comid(c as i64)).collect();
        let n = order.len();
        let index = IdIndex::new(order.clone());

        let length_m = Array1::from(read_array_f32(&storage, &path, "/length_m")?);
        let slope = Array1::from(read_array_f32(&storage, &path, "/slope")?);
        if length_m.len() != n || slope.len() != n {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: format!(
                    "order/length_m/slope lengths disagree: {n} / {} / {}",
                    length_m.len(),
                    slope.len()
                ),
            });
        }

        let indices_0 = read_array_i32(&storage, &path, "/indices_0")?;
        let indices_1 = read_array_i32(&storage, &path, "/indices_1")?;
        if indices_0.len() != indices_1.len() {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: format!(
                    "indices_0 / indices_1 length mismatch: {} vs {}",
                    indices_0.len(),
                    indices_1.len()
                ),
            });
        }
        let nnz = indices_0.len();

        Ok(Self {
            path,
            order,
            index,
            length_m,
            slope,
            indices_0,
            indices_1,
            n,
            nnz,
        })
    }
}

/// Per-gauge upstream subgraph — indices reference *CONUS* positions, not
/// compressed positions. The dataset compresses at batch time when it unions
/// multiple gauges' subgraphs.
#[derive(Clone, Debug)]
pub struct GageSubgraph {
    pub staid: Staid,
    /// Position of the gauge outlet in the CONUS-wide array.
    pub gage_idx: usize,
    /// MERIT COMID of the gauge outlet (from `gage_catchment` attr).
    pub gage_catchment: String,
    /// COO row indices in CONUS position space.
    pub indices_0: Vec<i32>,
    /// COO column indices in CONUS position space.
    pub indices_1: Vec<i32>,
}

impl GageSubgraph {
    /// Returns the unique COMIDs in this gauge's upstream subgraph,
    /// sorted by CONUS position (stable across runs).
    ///
    /// Mirrors `gages_adjacency[gauge]["order"][:]` from
    /// `~/projects/ddr/scripts/summed_q_prime.py:198`. The COO indices
    /// (`indices_0` ∪ `indices_1`) cover exactly the same node set as the
    /// gauge's `order` array because every node either appears as an edge
    /// endpoint or would be unreferenced.
    pub fn upstream_comids(&self, conus: &ConusAdjacencyStore) -> Vec<Comid> {
        let mut positions: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
        positions.extend(self.indices_0.iter().copied());
        positions.extend(self.indices_1.iter().copied());
        positions
            .into_iter()
            .map(|pos| conus.order[pos as usize])
            .collect()
    }
}

/// Per-STAID subgraph store. Loaded eagerly for the chosen-gauge set at
/// dataset construction (architectural decision: option (b) from the prior
/// design discussion).
pub struct GagesAdjacencyStore {
    pub path: PathBuf,
    pub subgraphs: std::collections::HashMap<Staid, GageSubgraph>,
}

impl GagesAdjacencyStore {
    /// Eager-load only the requested STAIDs. Missing STAIDs are silently
    /// dropped (mirrors DDR's `valid_gauges_mask = np.isin(...)` in
    /// `_collate_gages`).
    pub fn open(path: impl Into<PathBuf>, staids: &[Staid]) -> Result<Self> {
        let path = path.into();
        let storage: ReadableStorage =
            Arc::new(FilesystemStore::new(&path).map_err(|e| zarr_err(&path, e))?);
        // Verify the root group exists.
        let _root = Group::open(storage.clone(), "/").map_err(|e| zarr_err(&path, e))?;

        let mut subgraphs = std::collections::HashMap::with_capacity(staids.len());
        for staid in staids {
            let group_path = format!("/{}", staid.as_str());
            // Open the gauge subgroup; if missing, skip rather than error.
            let group = match Group::open(storage.clone(), &group_path) {
                Ok(g) => g,
                Err(_) => continue,
            };
            let indices_0 = read_array_i32(&storage, &path, &format!("{group_path}/indices_0"))?;
            let indices_1 = read_array_i32(&storage, &path, &format!("{group_path}/indices_1"))?;

            // Required attrs: gage_idx, gage_catchment.
            let attrs = group.attributes();
            let gage_idx = attrs
                .get("gage_idx")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| DataError::Malformed {
                    path: path.clone(),
                    message: format!("missing or non-integer 'gage_idx' on {group_path}"),
                })? as usize;
            let gage_catchment = attrs
                .get("gage_catchment")
                .map(|v| match v.as_str() {
                    Some(s) => s.to_string(),
                    None => v.to_string(),
                })
                .unwrap_or_default();

            subgraphs.insert(
                staid.clone(),
                GageSubgraph {
                    staid: staid.clone(),
                    gage_idx,
                    gage_catchment,
                    indices_0,
                    indices_1,
                },
            );
        }
        Ok(Self { path, subgraphs })
    }

    pub fn get(&self, staid: &Staid) -> Option<&GageSubgraph> {
        self.subgraphs.get(staid)
    }

    pub fn len(&self) -> usize {
        self.subgraphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.subgraphs.is_empty()
    }
}

// ----------------------- private helpers -----------------------

fn read_array_i32(storage: &ReadableStorage, store_path: &Path, array_path: &str) -> Result<Vec<i32>> {
    let arr = ZarrArray::open(storage.clone(), array_path).map_err(|e| zarr_err(store_path, e))?;
    let subset = arr.subset_all();
    arr.retrieve_array_subset::<Vec<i32>>(&subset)
        .map_err(|e| zarr_err(store_path, e))
}

fn read_array_f32(storage: &ReadableStorage, store_path: &Path, array_path: &str) -> Result<Vec<f32>> {
    let arr = ZarrArray::open(storage.clone(), array_path).map_err(|e| zarr_err(store_path, e))?;
    let subset = arr.subset_all();
    arr.retrieve_array_subset::<Vec<f32>>(&subset)
        .map_err(|e| zarr_err(store_path, e))
}

fn zarr_err<E: std::error::Error + Send + Sync + 'static>(path: &Path, source: E) -> DataError {
    DataError::Zarr {
        path: path.to_path_buf(),
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_conus(comids: Vec<i64>) -> ConusAdjacencyStore {
        let order: Vec<Comid> = comids.into_iter().map(Comid).collect();
        let n = order.len();
        let index = IdIndex::new(order.clone());
        ConusAdjacencyStore {
            path: PathBuf::from("/dev/null"),
            order,
            index,
            length_m: Array1::zeros(n),
            slope: Array1::zeros(n),
            indices_0: vec![],
            indices_1: vec![],
            n,
            nnz: 0,
        }
    }

    #[test]
    fn upstream_comids_dedupes_and_orders_by_position() {
        let conus = fake_conus(vec![100, 200, 300, 400]);
        let sg = GageSubgraph {
            staid: Staid::from("00000001"),
            gage_idx: 3,
            gage_catchment: String::new(),
            // Mix of duplicates and out-of-order positions
            indices_0: vec![3, 2, 1, 3],
            indices_1: vec![2, 1, 0, 0],
        };
        let comids = sg.upstream_comids(&conus);
        // Position order 0,1,2,3 → COMIDs 100, 200, 300, 400.
        assert_eq!(comids, vec![Comid(100), Comid(200), Comid(300), Comid(400)]);
    }

    #[test]
    fn upstream_comids_empty_subgraph_returns_empty() {
        let conus = fake_conus(vec![100, 200]);
        let sg = GageSubgraph {
            staid: Staid::from("00000002"),
            gage_idx: 0,
            gage_catchment: String::new(),
            indices_0: vec![],
            indices_1: vec![],
        };
        assert!(sg.upstream_comids(&conus).is_empty());
    }
}
