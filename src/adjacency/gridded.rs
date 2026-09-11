//! Read a DDR gridded (ISIMIP DDM30) **sub-reach** adjacency zarr and express
//! it as the [`SubdividedAdjacency`] the managed builder already knows how to
//! write and the dataset already knows how to route.
//!
//! The store is the output of DDR's `scripts/build_subdivided_adjacency.py`
//! (DeepGroundwater/ddr #194). Per node (sub-reach) it carries:
//!
//! ```text
//!   order        int64   node id = cell_id * 1000 + sub_index (upstream-most first)
//!   parent_cell  int32   DDM30 cell id = row * 720 + col (row 0 = 55.75°S)
//!   indices_0/1  int32   COO (downstream row, upstream col), strictly lower-tri
//!   length_m     float32 piece length (the cell's flow length / pieces)
//!   slope        float32 inherited from the cell
//!   lat, lon     float32 cell centre (unused here)
//! ```
//!
//! and the pieces of a cell form a chain that is **contiguous in `order`**,
//! upstream first. That is exactly ddrs's subdivided layout (`zarr.rs:36-46`):
//! parent `p` owns rows `parent_offset[p]..parent_offset[p+1]` and its outlet —
//! the row a gauge is read at — is the last one. So the translation is a
//! relabelling, not a rebuild:
//!
//! ```text
//!   DDM30 store                         SubdividedAdjacency
//!   order       [c₀·1000+0, c₀·1000+1, c₁·1000+0, …]
//!   parent_cell [c₀, c₀, c₁, …]  ──►   order        = [c₀, c₀, c₁, …]
//!                                       parent_order = [c₀, c₁, …]
//!                                       parent_offset= [0, 2, 3, …]
//!   indices_0/1, length_m, slope ──►   rows/cols, length_m, slope  (copied)
//! ```
//!
//! Attributes and Q′ are keyed on the cell id (DDR writes both stores with
//! `COMID`/`divide_id` = cell id), so `AttributesStore::open(…, parent_order)`
//! and the streamflow reader need no change, and `pieces_per_row_divisor`
//! splits each cell's Q′ across its pieces the way DDR's trainers divide by
//! `k_per_node`.
//!
//! **Not supported:** DDR's unsplit cell adjacency (`ddm30_adjacency.zarr`,
//! no `parent_cell`). DDR's trainers use the sub-reach store; `open` refuses a
//! store without `parent_cell` with a message saying which script produces one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use zarrs::array::Array as ZarrArray;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::ReadableStorage;

use crate::adjacency::subdivide::SubdividedAdjacency;
use crate::data::error::{DataError, Result};

/// A DDM30 sub-reach network as read from disk, validated.
#[derive(Debug, Clone)]
pub struct GriddedNetwork {
    pub path: PathBuf,
    /// `order`: DDR node ids (`cell * 1000 + sub`). Kept for diagnostics only.
    pub node_ids: Vec<i64>,
    /// DDM30 cell id per sub-reach row.
    pub parent_cell: Vec<i32>,
    /// COO `indices_0` — downstream row.
    pub rows: Vec<i32>,
    /// COO `indices_1` — upstream row.
    pub cols: Vec<i32>,
    pub length_m: Vec<f32>,
    pub slope: Vec<f32>,
    /// Row where each cell's contiguous run starts, plus a final `n`.
    /// `parent_offset[p]..parent_offset[p+1]` are cell `p`'s rows.
    parent_offset: Vec<i32>,
}

impl GriddedNetwork {
    /// Open and validate a DDM30 sub-reach store.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let storage: ReadableStorage =
            Arc::new(FilesystemStore::new(&path).map_err(|e| zarr_err(&path, e))?);
        let _root = Group::open(storage.clone(), "/").map_err(|e| zarr_err(&path, e))?;

        let node_ids = read_i64_or_i32(&storage, &path, "/order")?;
        let n = node_ids.len();
        let parent_cell = match read_i32(&storage, &path, "/parent_cell") {
            Ok(v) => v,
            Err(_) => {
                return Err(DataError::Malformed {
                    path,
                    message: "gridded_network store has no `parent_cell` array — this is \
                              DDR's unsplit cell adjacency, not a sub-reach store. ddrs \
                              routes the sub-reach network only; build one with DDR's \
                              scripts/build_subdivided_adjacency.py"
                        .to_string(),
                });
            }
        };
        let rows = read_i32(&storage, &path, "/indices_0")?;
        let cols = read_i32(&storage, &path, "/indices_1")?;
        let length_m = read_f32(&storage, &path, "/length_m")?;
        let slope = read_f32(&storage, &path, "/slope")?;

        let malformed = |message: String| DataError::Malformed {
            path: path.clone(),
            message,
        };

        if parent_cell.len() != n || length_m.len() != n || slope.len() != n {
            return Err(malformed(format!(
                "order/parent_cell/length_m/slope lengths disagree: {n} / {} / {} / {}",
                parent_cell.len(),
                length_m.len(),
                slope.len()
            )));
        }
        if rows.len() != cols.len() {
            return Err(malformed(format!(
                "indices_0/indices_1 lengths disagree: {} / {}",
                rows.len(),
                cols.len()
            )));
        }
        // Routing invariant 3 (forward substitution): every edge must run from
        // a lower row (upstream) to a strictly higher row (downstream). A
        // self-edge would also break `subdivide`'s outlet→inlet argument.
        for (k, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
            if r < 0 || c < 0 || r as usize >= n || c as usize >= n {
                return Err(malformed(format!(
                    "edge {k} ({r} <- {c}) is outside 0..{n}"
                )));
            }
            if r <= c {
                return Err(malformed(format!(
                    "edge {k} ({r} <- {c}) is not strictly lower-triangular; the \
                     sub-reach store must be topologically ordered (downstream row > \
                     upstream row)"
                )));
            }
        }
        for (i, (&l, &s)) in length_m.iter().zip(slope.iter()).enumerate() {
            if !(l.is_finite() && l > 0.0) || !s.is_finite() {
                return Err(malformed(format!(
                    "row {i} (node {}) has length_m {l} / slope {s}; both must be finite \
                     and length positive",
                    node_ids[i]
                )));
            }
        }

        // Each cell's pieces must be one contiguous run: ddrs reads a gauge at
        // `parent_offset[p+1] - 1`, and `pieces_per_row_divisor` counts a
        // parent's rows from the offsets. A cell that appears in two runs
        // would be split into two "parents" with the same id, and the second
        // would shadow the first in `IdIndex`.
        let mut parent_offset = vec![0_i32];
        let mut seen = std::collections::HashSet::with_capacity(n);
        for i in 1..=n {
            if i == n || parent_cell[i] != parent_cell[i - 1] {
                let cell = parent_cell[i - 1];
                if !seen.insert(cell) {
                    return Err(malformed(format!(
                        "cell {cell} occupies more than one contiguous run of rows; the \
                         pieces of a cell must be adjacent in `order` (upstream first)"
                    )));
                }
                parent_offset.push(i as i32);
            }
        }

        Ok(Self {
            path,
            node_ids,
            parent_cell,
            rows,
            cols,
            length_m,
            slope,
            parent_offset,
        })
    }

    /// Number of sub-reach rows.
    pub fn n(&self) -> usize {
        self.node_ids.len()
    }

    /// Number of distinct DDM30 cells (parents).
    pub fn n_cells(&self) -> usize {
        self.parent_offset.len() - 1
    }

    /// Relabel into ddrs's subdivided layout (see the module diagram). Pure
    /// bookkeeping: no edge or length is recomputed.
    pub fn to_subdivided(&self) -> SubdividedAdjacency {
        let parent_order: Vec<i32> = self.parent_offset[..self.n_cells()]
            .iter()
            .map(|&start| self.parent_cell[start as usize])
            .collect();
        SubdividedAdjacency {
            order: self.parent_cell.clone(),
            parent_order,
            parent_offset: self.parent_offset.clone(),
            rows: self.rows.clone(),
            cols: self.cols.clone(),
            length_m: self.length_m.clone(),
            slope: self.slope.clone(),
        }
    }
}

// ---------- zarr helpers (mirror src/data/store/zarr.rs's private ones) ----------

fn read_i32(storage: &ReadableStorage, store_path: &Path, array_path: &str) -> Result<Vec<i32>> {
    let arr = ZarrArray::open(storage.clone(), array_path).map_err(|e| zarr_err(store_path, e))?;
    arr.retrieve_array_subset::<Vec<i32>>(&arr.subset_all())
        .map_err(|e| zarr_err(store_path, e))
}

fn read_f32(storage: &ReadableStorage, store_path: &Path, array_path: &str) -> Result<Vec<f32>> {
    let arr = ZarrArray::open(storage.clone(), array_path).map_err(|e| zarr_err(store_path, e))?;
    arr.retrieve_array_subset::<Vec<f32>>(&arr.subset_all())
        .map_err(|e| zarr_err(store_path, e))
}

/// DDR writes `order` as int64 (`build_subdivided_adjacency.py`); accept int32
/// too so a hand-rolled store still opens.
fn read_i64_or_i32(storage: &ReadableStorage, store_path: &Path, array_path: &str) -> Result<Vec<i64>> {
    let arr = ZarrArray::open(storage.clone(), array_path).map_err(|e| zarr_err(store_path, e))?;
    match arr.retrieve_array_subset::<Vec<i64>>(&arr.subset_all()) {
        Ok(v) => Ok(v),
        Err(_) => arr
            .retrieve_array_subset::<Vec<i32>>(&arr.subset_all())
            .map(|v| v.into_iter().map(i64::from).collect())
            .map_err(|e| zarr_err(store_path, e)),
    }
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
    use zarrs::array::{data_type, ArrayBuilder};
    use zarrs::group::GroupBuilder;

    /// Two cells, 100 (3 pieces) → 200 (2 pieces): a 5-row chain.
    ///
    /// ```text
    ///   100·0 → 100·1 → 100·2 → 200·0 → 200·1
    ///   row 0    1       2       3       4
    /// ```
    struct Synthetic {
        order: Vec<i64>,
        parent: Vec<i32>,
        rows: Vec<i32>,
        cols: Vec<i32>,
        length: Vec<f32>,
        slope: Vec<f32>,
    }

    fn chain() -> Synthetic {
        Synthetic {
            order: vec![100_000, 100_001, 100_002, 200_000, 200_001],
            parent: vec![100, 100, 100, 200, 200],
            rows: vec![1, 2, 3, 4],
            cols: vec![0, 1, 2, 3],
            length: vec![3000.0; 5],
            slope: vec![0.001, 0.001, 0.001, 0.002, 0.002],
        }
    }

    fn write_store(dir: &Path, s: &Synthetic, with_parent: bool) {
        let storage = Arc::new(FilesystemStore::new(dir).unwrap());
        GroupBuilder::new().build(storage.clone(), "/").unwrap().store_metadata().unwrap();
        macro_rules! put {
            ($name:expr, $dt:expr, $fill:expr, $data:expr) => {{
                let data = $data;
                let a = ArrayBuilder::new(vec![data.len() as u64], vec![data.len().max(1) as u64], $dt, $fill)
                    .build(storage.clone(), $name)
                    .unwrap();
                a.store_metadata().unwrap();
                if !data.is_empty() {
                    a.store_chunk(&[0], &data).unwrap();
                }
            }};
        }
        put!("/order", data_type::int64(), 0_i64, s.order.clone());
        if with_parent {
            put!("/parent_cell", data_type::int32(), 0_i32, s.parent.clone());
        }
        put!("/indices_0", data_type::int32(), 0_i32, s.rows.clone());
        put!("/indices_1", data_type::int32(), 0_i32, s.cols.clone());
        put!("/length_m", data_type::float32(), 0.0_f32, s.length.clone());
        put!("/slope", data_type::float32(), 0.0_f32, s.slope.clone());
    }

    fn open_synthetic(tag: &str, s: &Synthetic, with_parent: bool) -> Result<GriddedNetwork> {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join(format!("{tag}_subreach_adjacency.zarr"));
        write_store(&store, s, with_parent);
        GriddedNetwork::open(&store)
    }

    #[test]
    fn chain_relabels_into_two_parents_with_contiguous_pieces() {
        let g = open_synthetic("chain", &chain(), true).expect("open");
        assert_eq!(g.n(), 5);
        assert_eq!(g.n_cells(), 2);
        let sub = g.to_subdivided();
        assert_eq!(sub.order, vec![100, 100, 100, 200, 200]);
        assert_eq!(sub.parent_order, vec![100, 200]);
        assert_eq!(sub.parent_offset, vec![0, 3, 5]);
        assert_eq!(sub.pieces(0), 3);
        assert_eq!(sub.pieces(1), 2);
        // The gauge row for a cell is its LAST piece.
        assert_eq!(sub.outlet(0), 2);
        assert_eq!(sub.outlet(1), 4);
        assert_eq!(sub.rows, vec![1, 2, 3, 4]);
        assert_eq!(sub.cols, vec![0, 1, 2, 3]);
        assert_eq!(sub.length_m, vec![3000.0; 5]);
        assert_eq!(sub.slope[3], 0.002);
    }

    #[test]
    fn store_without_parent_cell_is_refused_with_the_fix() {
        let err = open_synthetic("noparent", &chain(), false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("parent_cell"), "{msg}");
        assert!(msg.contains("build_subdivided_adjacency.py"), "{msg}");
    }

    #[test]
    fn upper_triangular_edge_is_refused() {
        let mut s = chain();
        s.rows[0] = 0;
        s.cols[0] = 1; // 0 <- 1: downstream row below upstream row
        let msg = open_synthetic("upper", &s, true).unwrap_err().to_string();
        assert!(msg.contains("lower-triangular"), "{msg}");
    }

    #[test]
    fn non_contiguous_cell_is_refused() {
        let mut s = chain();
        s.parent = vec![100, 200, 100, 200, 200]; // cell 100 split in two runs
        let msg = open_synthetic("split", &s, true).unwrap_err().to_string();
        assert!(msg.contains("contiguous"), "{msg}");
    }

    #[test]
    fn non_positive_length_is_refused() {
        let mut s = chain();
        s.length[2] = 0.0;
        let msg = open_synthetic("zerolen", &s, true).unwrap_err().to_string();
        assert!(msg.contains("length_m 0"), "{msg}");
    }
}
