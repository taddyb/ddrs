//! Runtime dispatch between the CPU and cuSPARSE triangular solve paths.
//!
//! `effective_use_cuda` resolves the caller's request against the actual
//! backend type. If `use_cuda` is asked but the runtime backend is not
//! `Cuda<f32, i32>`, a one-shot WARN is emitted and the CPU path is taken.

use std::any::TypeId;
use std::sync::Arc;
use std::sync::Once;

use burn::tensor::backend::Backend;
use burn_cuda::Cuda;

use crate::sparse::{CsrPattern, SavedX};

/// True iff `B` is `burn::backend::Cuda<f32, i32>` (the only GPU backend
/// SP-6 specialises for).
///
/// Uses `TypeId` equality — zero overhead on a hot path (TypeId comparison
/// compiles to a single integer comparison).
pub(crate) fn backend_is_cuda<B: Backend + 'static>() -> bool {
    TypeId::of::<B>() == TypeId::of::<Cuda<f32, i32>>()
}

static FALLBACK_WARNED: Once = Once::new();

/// Resolve effective backend choice. If `use_cuda` is `true` but the backend
/// is not `Cuda<f32, i32>`, log a one-shot WARN and return `false`.
pub(crate) fn effective_use_cuda<B: Backend + 'static>(use_cuda: bool) -> bool {
    if use_cuda && !backend_is_cuda::<B>() {
        FALLBACK_WARNED.call_once(|| {
            eprintln!(
                "WARN [ddrs/dispatch]: sparse_solver=cuda requested but backend is not \
                 Cuda<f32, i32> — falling back to CPU path. \
                 (This message is logged once per process.)"
            );
        });
        return false;
    }
    use_cuda
}

/// Forward solve dispatch: routes to the cuSPARSE or CPU path and returns
/// both the output primitive and the `SavedX` variant for the autograd tape.
///
/// # CUDA path
///
/// Calls `cusparse_forward` which executes cusparseSpSV_solve on the dedicated
/// cuSPARSE stream, syncs, and round-trips `x` to host for the output primitive
/// (temporary fallback — see `cusparse_forward` docs). `SavedX::Cuda` stores the
/// output primitive so the backward can retrieve `x` without an extra host copy.
///
/// # CPU path
///
/// Calls the existing `cpu_forward_primitive` (renamed from `forward_primitive`
/// in mod.rs). Returns `SavedX::Cpu(Arc<Vec<f32>>)` for zero-copy backward access.
pub(crate) fn forward_primitive<B: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
    use_cuda: bool,
) -> (B::FloatTensorPrimitive, SavedX<B>)
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    if effective_use_cuda::<B>(use_cuda) {
        // CUDA path: GPU solve via cuSPARSE.
        let x_prim = crate::sparse::cusparse::cusparse_forward::<B>(
            pattern,
            a_values_prim,
            b_prim,
            device,
        );
        // Clone the primitive for SavedX. For the Cuda backend, primitives are
        // backed by refcounted handles (CubeTensor wraps a client + handle),
        // so this clone is cheap. For the host-roundtrip variant the primitive
        // is an NdArray slice — also cheap to clone.
        (x_prim.clone(), SavedX::Cuda(x_prim))
    } else {
        // CPU path: host-side forward substitution (unchanged from pre-Task-9).
        let (out_prim, x_vec) =
            crate::sparse::cpu_forward_primitive::<B>(pattern, a_values_prim, b_prim, device);
        (out_prim, SavedX::Cpu(Arc::new(x_vec)))
    }
}

/// Per-nnz grada gradient dispatch.
///
/// GPU path calls `cusparse_grada` which keeps everything on device via
/// pure BURN tensor ops (`Tensor::select` + multiply + negate). CPU path
/// uses the existing host-loop in `CsrSolveOp::backward` refactored here.
///
/// `gradb_prim` is consumed (not cloned) by the GPU path; the caller must
/// clone it first if it's also needed for registering `parent_b`.
pub(crate) fn grada_primitive<B: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    gradb_prim: B::FloatTensorPrimitive,
    x: crate::sparse::SavedX<B>,
    device: &B::Device,
    use_cuda: bool,
) -> B::FloatTensorPrimitive
where
    B::FloatTensorPrimitive: 'static,
{
    if effective_use_cuda::<B>(use_cuda) {
        let x_prim = match x {
            crate::sparse::SavedX::Cuda(p) => p,
            crate::sparse::SavedX::Cpu(_) => {
                panic!("GPU grada called with CPU-saved x — mismatched dispatch")
            }
        };
        crate::sparse::cusparse::cusparse_grada::<B>(pattern, gradb_prim, x_prim, device)
    } else {
        // CPU path: materialize gradb + x to host, compute, push back.
        let gradb_host: Vec<f32> = crate::sparse::primitive_to_vec::<B>(gradb_prim);
        let x_host: Vec<f32> = match x {
            crate::sparse::SavedX::Cpu(arc) => (*arc).clone(),
            crate::sparse::SavedX::Cuda(p) => crate::sparse::primitive_to_vec::<B>(p),
        };
        let grada: Vec<f32> = pattern
            .row_for_nnz
            .iter()
            .zip(pattern.col.iter())
            .map(|(&r, &c)| -gradb_host[r as usize] * x_host[c as usize])
            .collect();
        B::float_from_data(
            burn::tensor::TensorData::from(grada.as_slice()),
            device,
        )
    }
}

/// Site 1 dispatch: forward `y = N · q`.
pub(crate) fn spmv_forward_dispatch<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    q_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    if effective_use_cuda::<I>(use_cuda) {
        let cache = unsafe { crate::sparse::cusparse::ensure_cuda_cache(pattern) };
        crate::sparse::cusparse::cusparse_spmv_forward::<I>(cache, q_prim, device)
    } else {
        crate::sparse::cpu_spmv_forward::<I>(pattern, q_prim, device)
    }
}

/// Site 2 dispatch: backward `gq = N^T · gi`.
pub(crate) fn spmv_backward_dispatch<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    gi_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    if effective_use_cuda::<I>(use_cuda) {
        let cache = unsafe { crate::sparse::cusparse::ensure_cuda_cache(pattern) };
        crate::sparse::cusparse::cusparse_spmv_backward::<I>(cache, gi_prim, device)
    } else {
        crate::sparse::cpu_spmv_backward::<I>(pattern, gi_prim, device)
    }
}

/// Site 3 dispatch: `gc = -Σ_k(gA[k]·adj[k])` per row.
pub(crate) fn assemble_backward_dispatch<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    g_a_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    if effective_use_cuda::<I>(use_cuda) {
        let cache = unsafe { crate::sparse::cusparse::ensure_cuda_cache(pattern) };
        crate::sparse::cusparse::cusparse_assemble_backward::<I>(cache, g_a_prim, device)
    } else {
        crate::sparse::cpu_assemble_backward::<I>(pattern, g_a_prim, device)
    }
}

/// Backward-solve dispatch: CPU path uses `back_sub_upper_transposed`,
/// GPU path uses `cusparse_backward_solve`. Returns the gradient on `b`.
///
/// On the CPU path, `a_values_prim` and `grad_out_prim` are pulled to host,
/// the back-substitution is run, and the result is uploaded back via
/// `B::float_from_data`.
///
/// On the GPU path, the solve runs entirely on device via cuSPARSE's
/// TRANSPOSE op using the pre-analyzed `desc_backward` descriptor from the
/// pattern cache. The result is returned as a new `B::FloatTensorPrimitive`
/// (host-roundtrip fallback — same temporary strategy as `cusparse_forward`).
pub(crate) fn backward_solve_primitive<B: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    a_values_prim: &B::FloatTensorPrimitive,
    grad_out_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
    use_cuda: bool,
) -> B::FloatTensorPrimitive
where
    B::FloatTensorPrimitive: 'static,
{
    if effective_use_cuda::<B>(use_cuda) {
        crate::sparse::cusparse::cusparse_backward_solve::<B>(
            pattern,
            a_values_prim,
            grad_out_prim,
            device,
        )
    } else {
        // CPU path: existing back-substitution + round-trip.
        let a_data: Vec<f32> =
            crate::sparse::primitive_to_vec::<B>(a_values_prim.clone());
        let grad_out_data: Vec<f32> =
            crate::sparse::primitive_to_vec::<B>(grad_out_prim.clone());
        let gradb_data =
            crate::sparse::back_sub_upper_transposed(pattern, &a_data, &grad_out_data);
        B::float_from_data(
            burn::tensor::TensorData::from(gradb_data.as_slice()),
            device,
        )
    }
}
