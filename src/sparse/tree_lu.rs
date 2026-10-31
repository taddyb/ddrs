//! Zero-fill LU of a tree-structured matrix on the [`CsrPattern`] layout, with
//! its solves and its gradient rule (host side, f32).
//!
//! The matrix `A` has the pattern of a river tree: entry `(d, k)` for each
//! reach `k` with downstream reach `d`, plus the diagonal. Eliminating reaches
//! upstream first creates no fill, so `L` has the pattern of `N` and `U` has
//! the pattern of `Nᵀ`. Both fit the existing `CsrPattern` slots.
//!
//! # Value layout
//!
//! `A` is one slice of length `2 * nnz`, with `nnz = CsrPattern::nnz()`.
//!
//! | Half | Slot `(d, k)` holds | Diagonal slot `(i, i)` holds |
//! |---|---|---|
//! | first `nnz` | `A[d, k]` | `A[i, i]` |
//! | second `nnz` | `A[k, d]` | ignored (0) |
//!
//! The diagonal slot of row `i` is `crow[i + 1] - 1` (see [`diag_slot`]).
//!
//! Factors: `l_values` is `L` in row layout with 1.0 on the diagonal. `u_values`
//! has `U[k, d]` at slot `(d, k)` and `U[i, i]` at the diagonal slot.
//!
//! # Solves (reuse of the sweeps in `src/sparse/mod.rs`)
//!
//! | Step | Call |
//! |---|---|
//! | `L y = b` | `forward_sub_lower(pattern, l_values, b)` |
//! | `U x = y` | `back_sub_upper_transposed(pattern, u_values, y)` |
//! | `Uᵀ z = g` | `forward_sub_lower(pattern, u_values, g)` |
//! | `Lᵀ y = z` | `back_sub_upper_transposed(pattern, l_values, z)` |
//!
//! Gradient: with `g = A⁻ᵀ·grad_x`, `grad_b = g`, and
//! `grad_A[s] = -g[row(s)]·x[col(s)]` (first half),
//! `grad_A[nnz + s] = -g[col(s)]·x[row(s)]` (second half, 0 on the diagonal).
//! This is the `CsrSolveOp` rule applied to both halves.
//!
//! Stability: for a matrix with strictly column-dominant positive diagonal,
//! elimination without pivoting has growth factor at most 2.

use super::{back_sub_upper_transposed, forward_sub_lower, CsrPattern};

/// Factors of a tree matrix. Both vectors have length `nnz` (`CsrPattern` layout).
#[derive(Clone, Debug)]
pub struct TreeFactors {
    pub l_values: Vec<f32>,
    pub u_values: Vec<f32>,
}

/// Slot of the diagonal entry of `row`. `CsrPattern::from_sparse` emits it last.
pub fn diag_slot(pattern: &CsrPattern, row: usize) -> usize {
    pattern.crow[row + 1] as usize - 1
}

/// Check that the pattern is a forest: every off-diagonal has `row > col`, and
/// every reach appears at most once as an upstream column (one downstream).
pub fn assert_tree(pattern: &CsrPattern) -> Result<(), String> {
    let mut downstream_count = vec![0u32; pattern.n];
    for s in 0..pattern.nnz() {
        let r = pattern.row_for_nnz[s];
        let c = pattern.col[s];
        if r == c {
            continue;
        }
        if r <= c {
            return Err(format!(
                "entry (row {r}, col {c}) is not lower triangular (need row > col)"
            ));
        }
        downstream_count[c as usize] += 1;
        if downstream_count[c as usize] > 1 {
            return Err(format!("reach {c} has two downstream reaches"));
        }
    }
    Ok(())
}

/// Zero-fill LU. `a_values.len() == 2 * nnz` (layout in the module doc).
pub fn tree_lu_factor(pattern: &CsrPattern, a_values: &[f32]) -> TreeFactors {
    let nnz = pattern.nnz();
    assert_eq!(a_values.len(), 2 * nnz, "a_values must have length 2 * nnz");
    let mut l_values = vec![0.0f32; nnz];
    let mut u_values = vec![0.0f32; nnz];
    for i in 0..pattern.n {
        let start = pattern.crow[i] as usize;
        let ds = diag_slot(pattern, i);
        let mut diag = a_values[ds];
        for s in start..ds {
            let u = pattern.col[s] as usize;
            let l = a_values[s] / u_values[diag_slot(pattern, u)];
            let up = a_values[nnz + s];
            l_values[s] = l;
            u_values[s] = up;
            diag -= l * up;
        }
        l_values[ds] = 1.0;
        u_values[ds] = diag;
    }
    TreeFactors { l_values, u_values }
}

/// Solve `A x = b`.
pub fn tree_lu_solve_host(pattern: &CsrPattern, f: &TreeFactors, b: &[f32]) -> Vec<f32> {
    let y = forward_sub_lower(pattern, &f.l_values, b);
    back_sub_upper_transposed(pattern, &f.u_values, &y)
}

/// Solve `Aᵀ y = g`.
pub fn tree_lu_solve_transposed_host(
    pattern: &CsrPattern,
    f: &TreeFactors,
    g: &[f32],
) -> Vec<f32> {
    let z = forward_sub_lower(pattern, &f.u_values, g);
    back_sub_upper_transposed(pattern, &f.l_values, &z)
}

/// Gradient of the loss w.r.t. `A` (length `2 * nnz`), given `g = A⁻ᵀ·grad_x`
/// and the solution `x`.
pub fn tree_lu_grad_a_host(pattern: &CsrPattern, grad_b: &[f32], x: &[f32]) -> Vec<f32> {
    let nnz = pattern.nnz();
    let mut out = vec![0.0f32; 2 * nnz];
    for s in 0..nnz {
        let r = pattern.row_for_nnz[s] as usize;
        let c = pattern.col[s] as usize;
        out[s] = -grad_b[r] * x[c];
        if r != c {
            out[nnz + s] = -grad_b[c] * x[r];
        }
    }
    out
}
