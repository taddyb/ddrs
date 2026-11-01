//! Device side of the tree LU: the structure arrays of a pattern on the GPU.

use super::cube_buf::{upload_i32, Cube};
use super::tree_lu::TreeLevels;

/// `TreeLevels` on the device. Same field names as the host struct.
pub struct TreeDev {
    pub n: usize,
    pub nnz: usize,
    pub n_levels: usize,
    pub crow: Cube,
    pub col: Cube,
    pub row: Cube,
    pub col_diag: Cube,
    pub down: Cube,
    pub down_slot: Cube,
    pub lvl_ptr: Cube,
    pub lvl_idx: Cube,
}

impl TreeDev {
    /// Copy every array of `t` to the device of `like`.
    pub fn upload(t: &TreeLevels, like: &Cube) -> Self {
        Self {
            n: t.n,
            nnz: t.nnz,
            n_levels: t.n_levels,
            crow: upload_i32(like, &t.crow),
            col: upload_i32(like, &t.col),
            row: upload_i32(like, &t.row),
            col_diag: upload_i32(like, &t.col_diag),
            down: upload_i32(like, &t.down),
            down_slot: upload_i32(like, &t.down_slot),
            lvl_ptr: upload_i32(like, &t.lvl_ptr),
            lvl_idx: upload_i32(like, &t.lvl_idx),
        }
    }
}
