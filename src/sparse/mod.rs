//! CSR triangular solve for the routing equation `A · x = b` with
//! `A = I − c·N`, plus analytical backward.
//!
//! Port of DDR's `TriangularSparseSolver` (`~/projects/ddr/src/ddr/routing/utils.py:515`).
//! Where DDR delegates the forward solve to SciPy / CuPy via `torch.autograd.Function`,
//! we keep everything in-process: raw `Vec<f32>` forward substitution on the inner
//! backend, custom [`Backward`] implementation for the adjoint. The pattern of `A`
//! is fixed for a given network adjacency — we build [`CsrPattern`] once at
//! `setup_inputs` time and reuse it across timesteps.
//!
//! See `.claude/skills/burn_custom_backward.md` for the BURN-0.21 custom-backward
//! recipe this module uses.

pub mod cusparse;
pub(crate) mod dispatch;

use std::sync::Arc;

use crate::sparse::cusparse::UnsafeSendCache;

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData, TensorPrimitive};

/// Sparse adjacency in COO form, bundled with the per-reach channel attributes
/// that the Muskingum-Cunge solver requires.
///
/// Mirrors the on-disk layout that DDR's `ddr_engine` writes to zarr
/// (`engine/src/ddr_engine/core/zarr_io.py` — `indices_0`, `indices_1`,
/// `values`, plus the `length_m` / `slope` arrays aligned to the topological
/// `order`). The whole struct is plain CPU data; the engine uploads what it
/// needs to its backend at `setup_inputs` time.
///
/// Invariants (asserted at use sites, not in constructors — match DDR's
/// `merit/build.py` builder):
/// - `rows[k] >= cols[k]` for all k (lower triangular)
/// - reaches are in topological order (so the matrix is forward-substitutable)
/// - `length_m.len() == slope.len() == n`
#[derive(Clone, Debug)]
pub struct SparseAdjacency {
    pub n: usize,
    /// COO row indices (downstream segment idx), length `nnz`.
    pub rows: Vec<i32>,
    /// COO column indices (upstream segment idx), length `nnz`.
    pub cols: Vec<i32>,
    /// Edge weights at each (row, col), length `nnz`. Usually all `1.0` for
    /// pure connectivity; non-unit values are passed through unchanged.
    pub values: Vec<f32>,
    /// Channel length per reach in metres, length `n`, aligned to topological
    /// order.
    pub length_m: Vec<f32>,
    /// Channel slope per reach (dimensionless), length `n`, aligned to
    /// topological order. Engine clamps to `attribute_minimums.slope`.
    pub slope: Vec<f32>,
}

impl SparseAdjacency {
    /// Build from a dense row-major `[n, n]` adjacency slice. **Test/fixture
    /// use only** — scans the full `n × n` array, fine for small benchmarks
    /// (5×5 sandbox, 10×10 mock chains). Production loaders should construct
    /// `SparseAdjacency` directly from COO on disk.
    pub fn from_dense(
        n: usize,
        adj_dense: &[f32],
        length_m: Vec<f32>,
        slope: Vec<f32>,
    ) -> Self {
        assert_eq!(adj_dense.len(), n * n, "adj_dense must be n*n");
        assert_eq!(length_m.len(), n, "length_m must have length n");
        assert_eq!(slope.len(), n, "slope must have length n");
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in 0..i {
                // Lower-triangular scan: j < i only.
                let v = adj_dense[i * n + j];
                if v != 0.0 {
                    rows.push(i as i32);
                    cols.push(j as i32);
                    values.push(v);
                }
            }
        }
        Self { n, rows, cols, values, length_m, slope }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Cached non-zero structure of `A = I − c·N` in CSR layout.
///
/// `A` is square `[n, n]`, lower triangular under topological ordering of `N`.
/// The diagonal is always present (from `I`); off-diagonal entries come from `N`.
/// Within each row, off-diagonal entries are emitted first (in ascending column
/// order), followed by the diagonal — this matches both DDR's `PatternMapper`
/// output ordering and the natural forward-substitution traversal.
#[derive(Clone)]
pub struct CsrPattern {
    pub n: usize,
    /// CSR row pointers, length `n + 1`.
    pub crow: Vec<i32>,
    /// CSR column indices, length `nnz`.
    pub col: Vec<i32>,
    /// Row index of each non-zero entry, length `nnz`. Used by the analytical
    /// backward to scatter `gradA_values[k] = -gradb[row(k)] * x[col(k)]`.
    pub row_for_nnz: Vec<i32>,
    /// Adjacency weight `N[row, col]` at each non-zero position, length `nnz`.
    /// Stored as `0.0` at diagonal slots (their contribution comes from `I`,
    /// not from `N`).
    pub adj_values: Vec<f32>,
    /// `1.0` at diagonal slots, `0.0` elsewhere. Used to assemble `A_values`:
    /// `A[k] = diag_mask[k] + (1 − diag_mask[k]) · (−c[row(k)] · adj[k])`.
    pub diag_mask: Vec<f32>,

    // --- Transposed-CSR view for the backward solve `A^T · gradb = grad_out`. ---
    /// Row pointers of `A^T`, length `n + 1`.
    pub trans_crow: Vec<i32>,
    /// Column indices of `A^T`, length `nnz`.
    pub trans_col: Vec<i32>,
    /// `trans_to_orig[k]` = index in the original CSR arrays of the entry that
    /// the `k`-th entry of `A^T` corresponds to. Lets the backward read
    /// `A_values[trans_to_orig[k]]` without rebuilding any structure.
    pub trans_to_orig: Vec<i32>,

    /// Lazy GPU companion built on first cuSPARSE solve call. `None` on
    /// CPU-only runs. Not part of structural equality — the `Debug` impl
    /// skips this field.
    pub(crate) cuda_cache: UnsafeSendCache,
}

impl std::fmt::Debug for CsrPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsrPattern")
            .field("n", &self.n)
            .field("nnz", &self.col.len())
            .finish_non_exhaustive()
    }
}

impl Clone for UnsafeSendCache {
    fn clone(&self) -> Self {
        // The GPU cache is not cloned — each CsrPattern clone starts with an
        // empty cache and re-initializes on first GPU solve if needed.
        Self::new()
    }
}

impl CsrPattern {
    /// Build the pattern from a sparse adjacency in COO form. O(nnz log nnz)
    /// in the worst case (one sort by (row, col)); no `n × n` scan, no dense
    /// tensor materialization.
    ///
    /// `adj` is expected lower-triangular with topological-order indices. Each
    /// row's non-zeros are emitted in ascending column order, followed by the
    /// diagonal entry (always present from `I`).
    pub fn from_sparse(adj: &SparseAdjacency) -> Self {
        let nnz_off = adj.nnz();
        // Sort COO triplets by (row, col) ascending. Most builders already
        // emit in this order, but we don't trust it.
        let mut order: Vec<usize> = (0..nnz_off).collect();
        order.sort_unstable_by_key(|&k| (adj.rows[k], adj.cols[k]));

        let n = adj.n;
        // Final nnz: off-diagonals + 1 diagonal per row.
        let nnz_total = nnz_off + n;
        let mut crow = Vec::with_capacity(n + 1);
        let mut col = Vec::with_capacity(nnz_total);
        let mut row_for_nnz = Vec::with_capacity(nnz_total);
        let mut adj_values = Vec::with_capacity(nnz_total);
        let mut diag_mask = Vec::with_capacity(nnz_total);

        crow.push(0);
        let mut cursor = 0usize;
        for i in 0..n {
            // Emit off-diagonals belonging to row i (j < i), in sorted order.
            while cursor < nnz_off && adj.rows[order[cursor]] == i as i32 {
                let k = order[cursor];
                debug_assert!(adj.cols[k] < i as i32, "adjacency not lower-triangular at (row={}, col={})", adj.rows[k], adj.cols[k]);
                col.push(adj.cols[k]);
                row_for_nnz.push(i as i32);
                adj_values.push(adj.values[k]);
                diag_mask.push(0.0);
                cursor += 1;
            }
            // Diagonal entry from `I`.
            col.push(i as i32);
            row_for_nnz.push(i as i32);
            adj_values.push(0.0);
            diag_mask.push(1.0);

            crow.push(col.len() as i32);
        }
        debug_assert_eq!(cursor, nnz_off, "stray COO entries past row n");

        let (trans_crow, trans_col, trans_to_orig) =
            build_transposed_pattern(n, &col, &row_for_nnz);

        Self {
            n,
            crow,
            col,
            row_for_nnz,
            adj_values,
            diag_mask,
            trans_crow,
            trans_col,
            trans_to_orig,
            cuda_cache: UnsafeSendCache::new(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.col.len()
    }

    /// Build a pattern from explicit CSR structure (`crow`, `col`), without
    /// assuming the matrix came from an `I − c·N` decomposition.
    ///
    /// `adj_values` and `diag_mask` are left zero — callers using this
    /// constructor must supply `A_values` directly (e.g. the gradcheck test
    /// against DDR's solver). All structural metadata needed by the forward
    /// substitution and the analytical backward (row indices, transposed
    /// pattern) is computed here.
    pub fn from_csr_structure(n: usize, crow: Vec<i32>, col: Vec<i32>) -> Self {
        assert_eq!(crow.len(), n + 1, "crow length must be n+1");
        let nnz = col.len();
        assert_eq!(
            crow[n] as usize, nnz,
            "crow[n] must equal nnz (got {} vs {})",
            crow[n], nnz
        );
        let mut row_for_nnz = Vec::with_capacity(nnz);
        for i in 0..n {
            let start = crow[i] as usize;
            let end = crow[i + 1] as usize;
            for _ in start..end {
                row_for_nnz.push(i as i32);
            }
        }
        let adj_values = vec![0.0f32; nnz];
        let diag_mask = vec![0.0f32; nnz];
        let (trans_crow, trans_col, trans_to_orig) =
            build_transposed_pattern(n, &col, &row_for_nnz);
        Self {
            n,
            crow,
            col,
            row_for_nnz,
            adj_values,
            diag_mask,
            trans_crow,
            trans_col,
            trans_to_orig,
            cuda_cache: UnsafeSendCache::new(),
        }
    }
}

fn build_transposed_pattern(
    n: usize,
    col: &[i32],
    row_for_nnz: &[i32],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let nnz = col.len();
    // Count entries per transposed-row (= original column).
    let mut counts = vec![0i32; n + 1];
    for &c in col {
        counts[c as usize + 1] += 1;
    }
    // Prefix sum → trans_crow.
    let mut trans_crow = vec![0i32; n + 1];
    for i in 1..=n {
        trans_crow[i] = trans_crow[i - 1] + counts[i];
    }
    // Fill, using cursor copy of trans_crow.
    let mut cursor = trans_crow.clone();
    let mut trans_col = vec![0i32; nnz];
    let mut trans_to_orig = vec![0i32; nnz];
    for k in 0..nnz {
        let r = col[k] as usize;
        let pos = cursor[r] as usize;
        trans_col[pos] = row_for_nnz[k];
        trans_to_orig[pos] = k as i32;
        cursor[r] += 1;
    }
    (trans_crow, trans_col, trans_to_orig)
}

// =================================================================================
// Forward substitution on raw f32 data (lower triangular).
// =================================================================================

/// Forward substitution for lower-triangular CSR: solves `A · x = b`.
///
/// Pure-Rust port of `scipy.sparse.linalg.spsolve_triangular(A, b, lower=True)`.
/// `a_values` is laid out per [`CsrPattern`] (off-diag then diag within each row).
fn forward_sub_lower(pattern: &CsrPattern, a_values: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a_values.len(), pattern.nnz());
    debug_assert_eq!(b.len(), pattern.n);
    let n = pattern.n;
    let mut x = vec![0.0f32; n];
    for i in 0..n {
        let start = pattern.crow[i] as usize;
        let end = pattern.crow[i + 1] as usize;
        let mut sum = 0.0f32;
        let mut diag = 0.0f32;
        for k in start..end {
            let j = pattern.col[k] as usize;
            if j == i {
                diag = a_values[k];
            } else {
                sum += a_values[k] * x[j];
            }
        }
        x[i] = (b[i] - sum) / diag;
    }
    x
}

/// Back-substitution for upper-triangular CSR: solves `U · y = b` where `U = A^T`.
///
/// `pattern.trans_*` describe `A^T`'s CSR layout; we look up the underlying
/// `A_values` via `trans_to_orig`.
pub(crate) fn back_sub_upper_transposed(
    pattern: &CsrPattern,
    a_values: &[f32],
    b: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(a_values.len(), pattern.nnz());
    debug_assert_eq!(b.len(), pattern.n);
    let n = pattern.n;
    let mut y = vec![0.0f32; n];
    for i in (0..n).rev() {
        let start = pattern.trans_crow[i] as usize;
        let end = pattern.trans_crow[i + 1] as usize;
        let mut sum = 0.0f32;
        let mut diag = 0.0f32;
        for k in start..end {
            let j = pattern.trans_col[k] as usize;
            let val = a_values[pattern.trans_to_orig[k] as usize];
            if j == i {
                diag = val;
            } else {
                sum += val * y[j];
            }
        }
        y[i] = (b[i] - sum) / diag;
    }
    y
}

// =================================================================================
// Backend-level forward (no autograd): used both as the inner forward for the
// autodiff op and as a standalone API for forward-only callers.
// =================================================================================

pub(crate) fn cpu_forward_primitive<B: Backend>(
    pattern: &CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> (B::FloatTensorPrimitive, Vec<f32>) {
    // Pull data → CPU Vec<f32> via Tensor wrapper (sync to_data).
    let a_data: Vec<f32> = primitive_to_vec::<B>(a_values_prim.clone());
    let b_data: Vec<f32> = primitive_to_vec::<B>(b_prim.clone());
    let x = forward_sub_lower(pattern, &a_data, &b_data);
    let out = B::float_from_data(TensorData::from(x.as_slice()), device);
    (out, x)
}

/// Wrap a raw `FloatTensorPrimitive` in a `Tensor` wrapper just long enough to
/// call `.to_data()` synchronously, then yield a `Vec<f32>`.
pub(crate) fn primitive_to_vec<B: Backend>(prim: B::FloatTensorPrimitive) -> Vec<f32> {
    let t: Tensor<B, 1> = Tensor::from_primitive(TensorPrimitive::Float(prim));
    t.to_data().to_vec::<f32>().expect("expected f32 tensor")
}

// =================================================================================
// Autodiff op: forward + analytical backward.
// =================================================================================

/// Forward-solve output saved for the backward pass. The CPU path stores a
/// host-side `Vec<f32>` (cheap to share via `Arc`); the future GPU path
/// stores a `B::FloatTensorPrimitive` so `x` stays on device.
#[derive(Clone, Debug)]
pub(crate) enum SavedX<B: Backend> {
    Cpu(Arc<Vec<f32>>),
    #[allow(dead_code)]
    Cuda(B::FloatTensorPrimitive),
}

/// Saved state for the backward pass.
///
/// `x` is stored as a host `Vec<f32>` because the backward immediately needs it
/// as one (avoids a redundant H→D and D→H round-trip per timestep). `a_values`
/// stays as a primitive because the forward provides it that way already.
/// `pattern` is `Arc`-wrapped so per-timestep cloning is a refcount bump
/// instead of cloning O(nnz) index/value arrays onto the autograd tape.
#[derive(Clone, Debug)]
struct CsrSolveState<B: Backend> {
    a_values: B::FloatTensorPrimitive,
    x: SavedX<B>,
    pattern: Arc<CsrPattern>,
    use_cuda: bool,
}

#[derive(Debug)]
struct CsrSolveOp;

impl<B: Backend + 'static> Backward<B, 2> for CsrSolveOp
where
    B::FloatTensorPrimitive: 'static,
{
    type State = CsrSolveState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let CsrSolveState { a_values, x, pattern, use_cuda } = ops.state;
        let [parent_a, parent_b] = ops.parents;
        let grad_out = grads.consume::<B>(&ops.node);
        let device = B::float_device(&grad_out);

        // Dispatch the backward triangular solve (CPU back-sub or GPU SpSV TRANSPOSE).
        // Returns grad_b = (A^T)^{-1} · grad_out as a primitive.
        let gradb_prim = crate::sparse::dispatch::backward_solve_primitive::<B>(
            &pattern,
            &a_values,
            &grad_out,
            &device,
            use_cuda,
        );

        if let Some(p_b) = parent_b {
            grads.register::<B>(p_b.id, gradb_prim.clone());
        }

        if let Some(p_a) = parent_a {
            // For grada the per-nnz scatter still runs on host (Task 11 swaps
            // in a GPU kernel). Pull gradb and x to CPU.
            let gradb_data: Vec<f32> = primitive_to_vec::<B>(gradb_prim);
            let x_host: Vec<f32> = match x {
                SavedX::Cpu(arc) => (*arc).clone(),
                SavedX::Cuda(prim) => primitive_to_vec::<B>(prim),
            };
            let nnz = pattern.nnz();
            let mut grada = vec![0.0f32; nnz];
            for k in 0..nnz {
                let r = pattern.row_for_nnz[k] as usize;
                let c = pattern.col[k] as usize;
                grada[k] = -gradb_data[r] * x_host[c];
            }
            let grada_prim = B::float_from_data(TensorData::from(grada.as_slice()), &device);
            grads.register::<B>(p_a.id, grada_prim);
        }
    }
}

/// Solve the lower-triangular sparse system `A · x = b` with analytical adjoint.
///
/// `a_values` carries gradient → upstream `c1` (and thence routing params).
/// `b` carries gradient → upstream `Q_t`, `q_prime`, and routing params via the
/// per-timestep `c2, c3, c4` factors.
///
/// Note: `pattern` is **structural metadata only** — it does not participate in
/// autograd. Callers cache one [`CsrPattern`] per network and reuse it across
/// timesteps and epochs.
pub fn triangular_csr_solve<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    a_values: Tensor<Autodiff<I>, 1>,
    b: Tensor<Autodiff<I>, 1>,
    use_cuda: bool,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
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

    let (out_prim, saved_x) = crate::sparse::dispatch::forward_primitive::<I>(
        pattern,
        &a_at.primitive,
        &b_at.primitive,
        &device,
        use_cuda,
    );

    let result = match CsrSolveOp
        .prepare::<NoCheckpointing>([a_at.node.clone(), b_at.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let state = CsrSolveState::<I> {
                a_values: a_at.primitive.clone(),
                x: saved_x,
                pattern: pattern.clone(),
                use_cuda,
            };
            prep.finish(state, out_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(out_prim),
    };

    Tensor::from_primitive(TensorPrimitive::Float(result))
}

// =================================================================================
// Assemble `A_values = diag_mask + (1 − diag_mask) · (−c[row] · adj)` differentiably.
// =================================================================================

/// Pre-uploaded constants for assembling `A_values = I − c·N` every timestep
/// without paying H→D copies for the structural arrays.
///
/// Built once at `setup_inputs`, reused for every solve (hot-start + every
/// timestep). All three tensors are length `nnz` and have no autograd
/// dependence — they are constants of the network topology.
pub struct AValuesAssembler<I: Backend> {
    n: usize,
    /// Adjacency weight `N[row, col]` at each non-zero (0 at diagonal slots).
    adj: Tensor<Autodiff<I>, 1>,
    /// `1` at diagonal slots, `0` elsewhere.
    diag_mask: Tensor<Autodiff<I>, 1>,
    /// Row index per non-zero entry, for `c[row]` gather and SpMV scatter.
    row_idx: Tensor<Autodiff<I>, 1, Int>,
    /// Column index per non-zero entry, for the SpMV gather of `q_t`.
    col_idx: Tensor<Autodiff<I>, 1, Int>,
}

impl<I: Backend> AValuesAssembler<I> {
    pub fn new(pattern: &CsrPattern, device: &I::Device) -> Self {
        let adj = Tensor::<Autodiff<I>, 1>::from_floats(pattern.adj_values.as_slice(), device);
        let diag_mask =
            Tensor::<Autodiff<I>, 1>::from_floats(pattern.diag_mask.as_slice(), device);
        let row_idx = Tensor::<Autodiff<I>, 1, Int>::from_data(
            TensorData::from(pattern.row_for_nnz.as_slice()),
            device,
        );
        let col_idx = Tensor::<Autodiff<I>, 1, Int>::from_data(
            TensorData::from(pattern.col.as_slice()),
            device,
        );
        Self { n: pattern.n, adj, diag_mask, row_idx, col_idx }
    }

    /// Build the non-zero values of `A = I − c·N` for a given per-row
    /// coefficient vector `c` (length `n`). Result has shape `[nnz]`.
    ///
    /// Simplified form: `A_values = diag_mask + (−c[row] · adj)`. The original
    /// expression `diag_mask + (1 − diag_mask) · (−c[row] · adj)` is redundant
    /// because `adj[k] == 0` at diagonal slots — the masking with `(1 − diag_mask)`
    /// only zeros out terms that were already zero. Saves one multiply and one
    /// subtract per timestep, plus their tape nodes.
    pub fn assemble(&self, c: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1> {
        let c_at_rows = c.gather(0, self.row_idx.clone());
        self.diag_mask.clone() + c_at_rows.neg() * self.adj.clone()
    }

    /// Sparse `N · q` for the cached adjacency, without materializing a dense
    /// `[n, n]` matmul. Equivalent to `network.matmul(q.unsqueeze(1)).squeeze(1)`.
    ///
    /// Implementation: `q[col]` gather → multiply by `adj` → scatter-add by
    /// `row`. All three ops are BURN-native with built-in autograd, so the
    /// adjoint (SpMV by `N^T`) is registered automatically. Cost: `O(nnz)`.
    ///
    /// Note: `adj[k] = 0` at diagonal slots, so those entries contribute 0 —
    /// the result matches the dense matmul bit-for-bit when the network has
    /// zero diagonal (always true for topologically-ordered routing graphs).
    pub fn spmv(&self, q: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1> {
        let device = q.device();
        let q_at_cols = q.gather(0, self.col_idx.clone());
        let weighted = q_at_cols * self.adj.clone();
        let zeros = Tensor::<Autodiff<I>, 1>::zeros([self.n], &device);
        zeros.scatter(0, self.row_idx.clone(), weighted, IndexingUpdateOp::Add)
    }
}
