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
///
/// ## Two index spaces (reach subdivision)
///
/// When the store was built with `params.subdivision.enabled`, one MERIT reach
/// occupies several consecutive rows. `order` then carries **duplicate** COMIDs
/// (one per sub-reach), so a COMID→row lookup on it would be ambiguous.
///
/// - **Parent space** (`parent_order`, length `n_parent`): one entry per MERIT
///   reach. `index` is built from THIS, so `index.position(comid)` always
///   returns a *parent* position.
/// - **Sub-reach space** (`order`, `length_m`, `slope`, `indices_*`, length `n`):
///   what the solver sees. Parent `p` owns rows
///   `parent_offset[p]..parent_offset[p + 1]`, ordered upstream→downstream, so
///   its outlet — the row a gauge must be read at — is `parent_offset[p+1] - 1`.
///
/// Stores written before subdivision existed carry neither array; `open`
/// synthesizes the identity (`parent_order == order`, `parent_offset == 0..=n`)
/// so they keep loading unchanged and every consumer sees one uniform contract.
pub struct ConusAdjacencyStore {
    pub path: PathBuf,
    /// COMIDs in topological order — element `i` is the COMID at zarr position `i`.
    /// Contains duplicates when the store is subdivided.
    pub order: Vec<Comid>,
    /// `IdIndex` mapping COMID → **parent** position. Built from `parent_order`,
    /// never from `order` (see the type-level note on the two index spaces).
    pub index: IdIndex<Comid>,
    /// One COMID per MERIT reach, in topological order. Identical to `order`
    /// when the store is not subdivided.
    pub parent_order: Vec<Comid>,
    /// Length `parent_order.len() + 1`. Rows `[parent_offset[p], parent_offset[p+1])`
    /// of the sub-reach arrays belong to parent `p`. `0..=n` when not subdivided.
    pub parent_offset: Vec<i32>,
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

        // Parent map. Absent in every store written before reach subdivision
        // (including the engine's own exports), so a missing array is NOT an
        // error — it means "one row per reach", and the identity below makes
        // such a store indistinguishable from a subdivided one with all m = 1.
        let parent_order: Vec<Comid> =
            match try_read_array_i32(&storage, "/parent_order") {
                Some(v) => v.into_iter().map(|c| Comid(c as i64)).collect(),
                None => order.clone(),
            };
        let parent_offset: Vec<i32> = match try_read_array_i32(&storage, "/parent_offset") {
            Some(v) => v,
            None => (0..=n as i32).collect(),
        };
        if parent_offset.len() != parent_order.len() + 1 {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: format!(
                    "parent_offset must have parent_order.len() + 1 entries: {} vs {}",
                    parent_offset.len(),
                    parent_order.len() + 1
                ),
            });
        }
        // The offsets must partition the sub-reach rows exactly; a truncated or
        // stale parent map would silently mis-address every gauge.
        if parent_offset.first() != Some(&0) || parent_offset.last() != Some(&(n as i32)) {
            return Err(DataError::Malformed {
                path: path.clone(),
                message: format!(
                    "parent_offset must run 0..{n}, got {:?}..{:?}",
                    parent_offset.first(),
                    parent_offset.last()
                ),
            });
        }
        // Built from `parent_order`: `order` has duplicates once subdivided.
        let index = IdIndex::new(parent_order.clone());

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
            parent_order,
            parent_offset,
            length_m,
            slope,
            indices_0,
            indices_1,
            n,
            nnz,
        })
    }

    /// Number of MERIT reaches (parent rows). Equals [`Self::n`] when the store
    /// is not subdivided.
    #[inline]
    pub fn n_parent(&self) -> usize {
        self.parent_order.len()
    }

    /// Last (most downstream) sub-reach row owned by `parent`. A gauge on that
    /// reach must be read here: any earlier piece omits the downstream fraction
    /// of the reach's own lateral inflow.
    #[inline]
    pub fn outlet_row(&self, parent: usize) -> usize {
        self.parent_offset[parent + 1] as usize - 1
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
    /// True when the gauge's catchment is a single MERIT divide: the
    /// subgraph has no edges (only a length-1 `order` array in the store).
    /// Training drops these as headwaters (`dataset.rs` "dropped N
    /// headwater"); every consumer must skip them the same way — for a
    /// zero-edge subgraph `upstream_comids` is empty, and summing an empty
    /// set silently yields an all-zero prediction.
    pub fn is_headwater(&self) -> bool {
        self.indices_0.is_empty()
    }

    /// Mirrors `gages_adjacency[gauge]["order"][:]` from
    /// `~/projects/ddr/scripts/summed_q_prime.py:198`. For subgraphs with at
    /// least one edge, the COO indices (`indices_0` ∪ `indices_1`) cover
    /// exactly the same node set as the gauge's `order` array because every
    /// node appears as an edge endpoint. Single-divide catchments have NO
    /// edges, so this returns empty — callers must filter with
    /// [`GageSubgraph::is_headwater`] first, as training does.
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

/// Read an optional int32 array: `None` when the array is absent OR unreadable.
///
/// Used for the subdivision parent map, which pre-subdivision stores (every
/// engine export, and every ddrs cache built before BUILDER_VERSION 2) simply
/// do not have. Collapsing "missing" and "corrupt" is acceptable here only
/// because the caller's fallback — the identity map — is itself a valid,
/// fully-consistent answer that `open` then range-checks against `n`.
fn try_read_array_i32(storage: &ReadableStorage, array_path: &str) -> Option<Vec<i32>> {
    let arr = ZarrArray::open(storage.clone(), array_path).ok()?;
    let subset = arr.subset_all();
    arr.retrieve_array_subset::<Vec<i32>>(&subset).ok()
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
            parent_order: order.clone(),
            parent_offset: (0..=n as i32).collect(),
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
    fn single_divide_subgraph_is_headwater_and_has_empty_upstream() {
        // Single-divide catchments have NO edges in the gages store — only a
        // length-1 `order` array. `upstream_comids` is empty for them, so
        // consumers (baseline included) must skip via `is_headwater`, exactly
        // as training's dataset filter does.
        let conus = fake_conus(vec![100, 200]);
        let sg = GageSubgraph {
            staid: Staid::from("00000002"),
            gage_idx: 1,
            gage_catchment: String::new(),
            indices_0: vec![],
            indices_1: vec![],
        };
        assert!(sg.is_headwater());
        assert!(sg.upstream_comids(&conus).is_empty());
    }

    #[test]
    fn edged_subgraph_is_not_headwater() {
        let sg = GageSubgraph {
            staid: Staid::from("00000003"),
            gage_idx: 1,
            gage_catchment: String::new(),
            indices_0: vec![1],
            indices_1: vec![0],
        };
        assert!(!sg.is_headwater());
    }
}
