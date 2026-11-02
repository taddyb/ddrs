//! Device side of the tree LU: the structure arrays of a pattern on the GPU,
//! the factor kernel, the grad-A kernel and the four triangular solves.
//!
//! The factor runs as one cube of 1024 threads. It walks the topological
//! levels in order and calls `sync_cube` between them, so a reach always sees
//! the finished diagonal of its upstream reaches. The solves reuse the
//! cuSPARSE calls of the MC path.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim};

use super::cube_buf::{empty_f32, from_cube, grid, into_cube, upload_i32, Cube, Gpu};
use super::cusparse::{cusparse_backward_solve, cusparse_forward};
use super::tree_lu::TreeLevels;
use super::CsrPattern;

/// `TreeLevels` on the device. Same field names as the host struct.
#[derive(Debug)]
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

/// Factors of a tree matrix on the device. Both buffers have length `nnz`
/// in the `CsrPattern` layout, as `TreeFactors` does on the host.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct TreeFactorsDev {
    pub l_values: Cube,
    pub u_values: Cube,
}

/// Zero-fill LU, one cube, level by level. Mirrors `tree_lu_factor`.
/// Threads of a level take reaches `UNIT_POS`, `UNIT_POS + CUBE_DIM`, ...
#[cube(launch)]
fn tree_factor_kernel(
    crow: &Tensor<i32>,
    col_diag: &Tensor<i32>,
    lvl_ptr: &Tensor<i32>,
    lvl_idx: &Tensor<i32>,
    a: &Tensor<f32>,
    l: &mut Tensor<f32>,
    u: &mut Tensor<f32>,
    nnz: u32,
    n_levels: u32,
) {
    let nz = usize::cast_from(nnz);
    let step = usize::cast_from(CUBE_DIM);
    for lv in 0..n_levels {
        let lo = usize::cast_from(lvl_ptr[lv as usize]);
        let hi = usize::cast_from(lvl_ptr[lv as usize + 1]);
        let mut t = lo + usize::cast_from(UNIT_POS);
        while t < hi {
            let i = usize::cast_from(lvl_idx[t]);
            let s0 = usize::cast_from(crow[i]);
            let ds = usize::cast_from(crow[i + 1]) - 1;
            let mut d = a[ds];
            for s in s0..ds {
                let lv_ = a[s] / u[usize::cast_from(col_diag[s])];
                let up = a[nz + s];
                l[s] = lv_;
                u[s] = up;
                // Rule F1: the same fused site as `tree_lu_factor` on the host.
                d = fma(-lv_, up, d);
            }
            l[ds] = 1.0f32;
            u[ds] = d;
            t += step;
        }
        sync_cube();
    }
}

/// `out[k] = -g[row k] * x[col k]`; `out[nnz + k] = -g[col k] * x[row k]`,
/// and exactly 0 on a diagonal slot. Mirrors `tree_lu_grad_a_host`.
#[cube(launch)]
fn tree_grad_a_kernel(
    row: &Tensor<i32>,
    col: &Tensor<i32>,
    g: &Tensor<f32>,
    x: &Tensor<f32>,
    out: &mut Tensor<f32>,
    nnz: u32,
) {
    let k = ABSOLUTE_POS;
    let nz = usize::cast_from(nnz);
    if k < nz {
        let r = usize::cast_from(row[k]);
        let c = usize::cast_from(col[k]);
        out[k] = -g[r] * x[c];
        let mut upper = 0.0f32;
        if r != c {
            upper = -g[c] * x[r];
        }
        out[nz + k] = upper;
    }
}

/// Factor `A` (`a_values` has length `2 * nnz`).
#[doc(hidden)]
pub fn tree_factor_dev(t: &TreeDev, a_values: &Cube) -> TreeFactorsDev {
    let l_values = empty_f32(a_values, t.nnz);
    let u_values = empty_f32(a_values, t.nnz);
    tree_factor_kernel::launch::<CudaRuntime>(
        &a_values.client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1024),
        t.crow.clone().into_tensor_arg(),
        t.col_diag.clone().into_tensor_arg(),
        t.lvl_ptr.clone().into_tensor_arg(),
        t.lvl_idx.clone().into_tensor_arg(),
        a_values.clone().into_tensor_arg(),
        l_values.clone().into_tensor_arg(),
        u_values.clone().into_tensor_arg(),
        t.nnz as u32,
        t.n_levels as u32,
    );
    TreeFactorsDev { l_values, u_values }
}

fn lower_solve(pattern: &CsrPattern, values: &Cube, rhs: &Cube) -> Cube {
    let device = values.device.clone();
    into_cube::<Gpu>(cusparse_forward::<Gpu>(
        pattern,
        &from_cube::<Gpu>(values.clone()),
        &from_cube::<Gpu>(rhs.clone()),
        &device,
    ))
}

fn upper_transposed_solve(pattern: &CsrPattern, values: &Cube, rhs: &Cube) -> Cube {
    let device = values.device.clone();
    into_cube::<Gpu>(cusparse_backward_solve::<Gpu>(
        pattern,
        &from_cube::<Gpu>(values.clone()),
        &from_cube::<Gpu>(rhs.clone()),
        &device,
    ))
}

/// Solve `A x = b`: `L y = b`, then `U x = y`.
#[doc(hidden)]
pub fn tree_solve_dev(pattern: &CsrPattern, f: &TreeFactorsDev, b: &Cube) -> Cube {
    let y = lower_solve(pattern, &f.l_values, b);
    upper_transposed_solve(pattern, &f.u_values, &y)
}

/// Solve `Aᵀ y = g`: `Uᵀ z = g`, then `Lᵀ y = z`.
#[doc(hidden)]
pub fn tree_solve_transposed_dev(pattern: &CsrPattern, f: &TreeFactorsDev, g: &Cube) -> Cube {
    let z = lower_solve(pattern, &f.u_values, g);
    upper_transposed_solve(pattern, &f.l_values, &z)
}

/// Gradient w.r.t. `A` (length `2 * nnz`) from `g = A⁻ᵀ·grad_x` and `x`.
#[doc(hidden)]
pub fn tree_grad_a_dev(t: &TreeDev, g: &Cube, x: &Cube) -> Cube {
    let out = empty_f32(g, 2 * t.nnz);
    let (count, dim) = grid(t.nnz);
    tree_grad_a_kernel::launch::<CudaRuntime>(
        &g.client,
        count,
        dim,
        t.row.clone().into_tensor_arg(),
        t.col.clone().into_tensor_arg(),
        g.clone().into_tensor_arg(),
        x.clone().into_tensor_arg(),
        out.clone().into_tensor_arg(),
        t.nnz as u32,
    );
    out
}
