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
