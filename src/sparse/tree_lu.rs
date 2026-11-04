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

use std::sync::Arc;

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

use super::{back_sub_upper_transposed, forward_sub_lower, primitive_to_vec, CsrPattern};

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
/// Each row must also end with its diagonal entry, which `diag_slot` assumes.
pub fn assert_tree(pattern: &CsrPattern) -> Result<(), String> {
    // `diag_slot` assumes the last slot of each row is the diagonal.
    for i in 0..pattern.n {
        let (start, end) = (pattern.crow[i] as usize, pattern.crow[i + 1] as usize);
        if end == start || pattern.col[end - 1] as usize != i {
            return Err(format!("row {i} does not end with its diagonal entry"));
        }
    }
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

// =================================================================================
// Dispatch layer. One function per host step; the CUDA arms are filled in Phase G.
// Modelled on `src/sparse/dispatch.rs::backward_solve_primitive`.
// =================================================================================

pub(crate) const CUDA_TODO: &str = "stage_routing CUDA path: Phase G";

/// Factor `A` (length `2 * nnz`).
pub(crate) fn dispatch_factor(
    pattern: &CsrPattern,
    a_values: &[f32],
    use_cuda: bool,
) -> TreeFactors {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    tree_lu_factor(pattern, a_values)
}

/// Solve `L y = b`.
pub(crate) fn dispatch_l_solve(
    pattern: &CsrPattern,
    l_values: &[f32],
    b: &[f32],
    use_cuda: bool,
) -> Vec<f32> {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    forward_sub_lower(pattern, l_values, b)
}

/// Solve `U x = y`.
pub(crate) fn dispatch_u_solve(
    pattern: &CsrPattern,
    u_values: &[f32],
    y: &[f32],
    use_cuda: bool,
) -> Vec<f32> {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    back_sub_upper_transposed(pattern, u_values, y)
}

/// Solve `Uᵀ z = g`.
pub(crate) fn dispatch_ut_solve(
    pattern: &CsrPattern,
    u_values: &[f32],
    g: &[f32],
    use_cuda: bool,
) -> Vec<f32> {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    forward_sub_lower(pattern, u_values, g)
}

/// Solve `Lᵀ g = z`.
pub(crate) fn dispatch_lt_solve(
    pattern: &CsrPattern,
    l_values: &[f32],
    z: &[f32],
    use_cuda: bool,
) -> Vec<f32> {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    back_sub_upper_transposed(pattern, l_values, z)
}

/// Gradient w.r.t. `A` (length `2 * nnz`) from `g` and `x`.
pub(crate) fn dispatch_grad_a(
    pattern: &CsrPattern,
    g: &[f32],
    x: &[f32],
    use_cuda: bool,
) -> Vec<f32> {
    if use_cuda {
        unimplemented!("{}", CUDA_TODO)
    }
    tree_lu_grad_a_host(pattern, g, x)
}

// =================================================================================
// Autodiff op
// =================================================================================

/// Saved state for the backward pass. Everything is behind an `Arc`, so cloning
/// onto the tape is a refcount bump.
#[derive(Clone, Debug)]
struct TreeSolveState {
    factors: Arc<TreeFactors>,
    x: Arc<Vec<f32>>,
    pattern: Arc<CsrPattern>,
    use_cuda: bool,
}

#[derive(Debug)]
struct TreeSolveOp;

impl<B: Backend + 'static> Backward<B, 2> for TreeSolveOp
where
    B::FloatTensorPrimitive: 'static,
{
    type State = TreeSolveState;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let TreeSolveState { factors, x, pattern, use_cuda } = ops.state;
        let [parent_a, parent_b] = ops.parents;
        let grad_out = grads.consume::<B>(&ops.node);
        let device = B::float_device(&grad_out);
        let grad_x = primitive_to_vec::<B>(grad_out);

        // g = A^{-T} grad_x: Uᵀ z = grad_x, then Lᵀ g = z.
        let z = dispatch_ut_solve(&pattern, &factors.u_values, &grad_x, use_cuda);
        let g = dispatch_lt_solve(&pattern, &factors.l_values, &z, use_cuda);

        if let Some(p_a) = parent_a {
            let ga = dispatch_grad_a(&pattern, &g, &x, use_cuda);
            let len = ga.len();
            grads.register::<B>(
                p_a.id,
                B::float_from_data(TensorData::new(ga, [len]), &device),
            );
        }
        if let Some(p_b) = parent_b {
            let len = g.len();
            grads.register::<B>(
                p_b.id,
                B::float_from_data(TensorData::new(g, [len]), &device),
            );
        }
    }
}

/// Solve `A x = b` for a tree matrix with a hand-written adjoint.
///
/// `a_values` has length `2 * nnz` (layout in the module doc). `pattern` is
/// structural metadata and does not take part in autograd.
pub fn tree_lu_solve<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    a_values: Tensor<Autodiff<I>, 1>,
    b: Tensor<Autodiff<I>, 1>,
    use_cuda: bool,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let a_at = match a_values.into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
    };
    let b_at = match b.into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => panic!("expected float tensor"),
    };
    let device = I::float_device(&a_at.primitive);

    let a_host = primitive_to_vec::<I>(a_at.primitive.clone());
    let b_host = primitive_to_vec::<I>(b_at.primitive.clone());
    let factors = dispatch_factor(pattern, &a_host, use_cuda);
    let y = dispatch_l_solve(pattern, &factors.l_values, &b_host, use_cuda);
    let x = dispatch_u_solve(pattern, &factors.u_values, &y, use_cuda);
    let n = x.len();
    let out_prim = I::float_from_data(TensorData::new(x.clone(), [n]), &device);

    let result = match TreeSolveOp
        .prepare::<NoCheckpointing>([a_at.node.clone(), b_at.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let state = TreeSolveState {
                factors: Arc::new(factors),
                x: Arc::new(x),
                pattern: pattern.clone(),
                use_cuda,
            };
            prep.finish(state, out_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(out_prim),
    };
    Tensor::from_primitive(TensorPrimitive::Float(result))
}
