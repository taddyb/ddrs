//! Routing primitives: parameter denormalization and a forward-substitution
//! triangular solve.
//!
//! Ports `denormalize`, `triangular_sparse_solve`, and `compute_hotstart_discharge`
//! from `~/projects/ddr/src/ddr/routing/`. The Python version uses SciPy/CuPy
//! sparse solvers with a custom `torch.autograd.Function` for the backward pass;
//! here we instead express the forward substitution as a plain BURN op-loop so
//! autodiff falls out of the framework automatically.
//!
//! ### Why forward substitution?
//!
//! The routing matrix `A = I − c1·N` is lower triangular when the adjacency
//! `N` is built in topological order (DDR's engine guarantees this). For a
//! lower-triangular `A` with non-zero diagonal, `A·x = b` solves as:
//!
//! ```text
//! x[i] = (b[i] − Σ_{j<i} A[i,j] · x[j]) / A[i,i]
//! ```
//!
//! Each step is a pure BURN tensor op, so the autodiff tape captures gradients
//! through both `A` and `b` without bespoke `Backward` plumbing. For the small
//! networks the test suite exercises (≤100 reaches) this is fine; a sparse
//! perf pass can come later without changing the public API.

use burn::tensor::{backend::Backend, Tensor};

/// Denormalize a `[0, 1]` neural-net output to physical units.
///
/// Linear: `value · (max − min) + min`.
/// Log-space: `exp(value · (log(max) − log(min + ε)) + log(min + ε))`.
/// Matches `denormalize()` in `routing/utils.py`.
pub fn denormalize<B: Backend>(value: Tensor<B, 1>, bounds: [f32; 2], log_space: bool) -> Tensor<B, 1> {
    let [lo, hi] = bounds;
    if log_space {
        let log_min = (lo + 1e-6).ln();
        let log_max = hi.ln();
        (value * (log_max - log_min) + log_min).exp()
    } else {
        value * (hi - lo) + lo
    }
}

/// Solve `A · x = b` where `A` is square lower triangular (dense for now).
///
/// `a` has shape `[n, n]`; `b` has shape `[n]`. Performs forward substitution
/// with `n` BURN ops on the autograd tape. Used for both the per-timestep
/// Muskingum solve and the hot-start accumulation in `MuskingumCunge`.
pub fn triangular_solve_lower<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let dims = a.dims();
    debug_assert_eq!(dims[0], dims[1], "triangular_solve_lower expects square matrix");
    debug_assert_eq!(dims[0], b.dims()[0], "matrix/vector dim mismatch");
    let n = dims[0];
    let device = b.device();

    // Carry the partial solution as a Vec of rank-1 tensors so each step's
    // gradient routes back through the BURN tape.
    let mut x_rows: Vec<Tensor<B, 1>> = Vec::with_capacity(n);

    for i in 0..n {
        // a_row = A[i, :i+1]  (shape [i+1])
        let a_row: Tensor<B, 1> = a.clone().slice([i..i + 1, 0..i + 1]).reshape([i + 1]);
        let b_i: Tensor<B, 1> = b.clone().slice([i..i + 1]);

        let off_diag_sum = if i == 0 {
            Tensor::zeros([1], &device)
        } else {
            // Σ_{j<i} A[i,j] · x[j]
            let a_off: Tensor<B, 1> = a_row.clone().slice([0..i]);
            let x_prev: Tensor<B, 1> = Tensor::cat(x_rows.clone(), 0);
            (a_off * x_prev).sum().reshape([1])
        };

        let diag: Tensor<B, 1> = a_row.slice([i..i + 1]);
        let x_i = (b_i - off_diag_sum) / diag;
        x_rows.push(x_i);
    }

    Tensor::cat(x_rows, 0)
}

/// Cold-start discharge via topological accumulation of lateral inflows.
///
/// Solves `(I − N) · Q_0 = q'_0` so each node receives the sum of all upstream
/// lateral inflows. On a topologically ordered DAG, `I − N` is lower triangular
/// with unit diagonal, so the reduction is exact in one forward substitution.
/// Mirrors `compute_hotstart_discharge` in `mmc.py` — on a linear chain with
/// uniform inflow, the result is a cumulative sum.
pub fn compute_hotstart_discharge<B: Backend>(
    q_prime_t0: Tensor<B, 1>,
    network: Tensor<B, 2>,
    discharge_lb: f32,
) -> Tensor<B, 1> {
    let n = network.dims()[0];
    let device = network.device();
    let i_mat: Tensor<B, 2> = Tensor::eye(n, &device);
    let a = i_mat - network;
    triangular_solve_lower(a, q_prime_t0).clamp_min(discharge_lb)
}
