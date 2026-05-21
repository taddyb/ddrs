//! cuSPARSE-backed forward/backward triangular solves for the BURN Cuda
//! backend.
//!
//! Task 6 (SP-6) is the spike that proves we can reach a raw CUDA device
//! pointer from a `burn::backend::Cuda` tensor primitive. Later SP-6 tasks
//! build the cusparseSpSV pipeline on top.
//!
//! # Type chain
//!
//! ```text
//! burn::backend::Cuda<f32, i32>
//!   = burn_cubecl::CubeBackend<CudaRuntime, f32, i32, u8>   (non-fusion variant)
//!   FloatTensorPrimitive = burn_cubecl::tensor::CubeTensor<CudaRuntime>
//!     .client : cubecl_runtime::client::ComputeClient<CudaRuntime>
//!     .handle : cubecl_runtime::server::Handle
//!
//! ComputeClient::get_resource(handle) -> ManagedResource<GpuResource>
//!   GpuResource.ptr  : u64   <- raw CUdeviceptr (the value of the pointer)
//!   GpuResource.size : u64   <- bytes allocated
//! ```
//!
//! `GpuResource` lives in a private submodule of `cubecl-cuda` and cannot be
//! named here, but its public fields are accessible via type inference once we
//! obtain the `ManagedResource` through the public `get_resource` method.

use burn::tensor::backend::Backend;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cudarc::driver::sys::CUstream;

/// Type-erased view into a CUDA tensor as a raw device pointer.
///
/// `ptr` is the raw CUDA device pointer (value of the `CUdeviceptr`).
/// `len` is the element count (not bytes).
///
/// The pointer aliases the BURN tensor's backing allocation — the caller MUST
/// NOT drop the source primitive while this view is alive.
pub struct CudaView {
    pub ptr: *mut f32,
    pub len: usize,
}

// SAFETY: The pointer is a CUDA device pointer, which is valid to send across
// threads as long as no concurrent GPU access occurs without synchronization.
// All callers are responsible for ensuring proper stream serialization.
unsafe impl Send for CudaView {}

/// Extract a raw CUDA device pointer from a `CubeTensor<CudaRuntime>`.
///
/// # Safety
///
/// - The returned `CudaView.ptr` aliases the tensor's GPU allocation.
/// - The caller must not drop `cube_tensor` (or the originating `Tensor`) while
///   the `CudaView` is live.
/// - No BURN operations that reallocate or move the tensor's buffer may be
///   scheduled on `cube_tensor.client` while the `CudaView` is live without
///   explicit CUDA stream synchronization.
pub(crate) fn cuda_view_from_cube_tensor(
    cube_tensor: &CubeTensor<CudaRuntime>,
) -> CudaView {
    // SAFETY: get_resource submits a blocking command to the server and returns
    // a ManagedResource that keeps the ManagedMemoryBinding alive, preventing
    // the buffer from being reclaimed. We extract .ptr (a u64 CUdeviceptr)
    // from the returned GpuResource (pub fields, private-module type).
    let resource = cube_tensor
        .client
        .get_resource(cube_tensor.handle.clone())
        .expect("CubeTensor handle must be bound to a live GPU allocation");

    let gpu = resource.resource();
    // `gpu.ptr` is the raw CUdeviceptr value (u64). Cast to *mut f32.
    let ptr = gpu.ptr as *mut f32;
    // `handle.size_in_used()` is bytes; divide by 4 for f32 element count.
    let len = cube_tensor.handle.size_in_used() as usize / core::mem::size_of::<f32>();

    CudaView { ptr, len }
}

/// Borrow a `Cuda<f32, i32>` float primitive as a raw device pointer.
///
/// Returns `None` if `B` is not `burn::backend::Cuda<f32, i32>` (the
/// non-fusion variant, i.e. `CubeBackend<CudaRuntime, f32, i32, u8>`).
///
/// # Safety
///
/// The returned `CudaView.ptr` is owned by the BURN tensor; do not free it.
/// The lifetime of the pointer is tied to `prim`. See [`cuda_view_from_cube_tensor`]
/// for the full invariant list.
///
/// The `TypeId` equality check guarantees that `B::FloatTensorPrimitive` and
/// `CubeTensor<CudaRuntime>` are the same type, making the pointer cast sound.
pub fn primitive_as_cuda_view<B: Backend>(
    prim: &B::FloatTensorPrimitive,
) -> Option<CudaView>
where
    B::FloatTensorPrimitive: 'static,
{
    use std::any::TypeId;

    // The target concrete primitive type for the non-fusion Cuda backend.
    type Target = CubeTensor<CudaRuntime>;

    if TypeId::of::<B::FloatTensorPrimitive>() != TypeId::of::<Target>() {
        return None;
    }

    // SAFETY: TypeId equality is Rust's guarantee that two types are identical.
    // Both `B::FloatTensorPrimitive` and `Target` are the same type
    // (`CubeTensor<CudaRuntime>`), so they have the same memory layout.
    // We are merely reinterpreting a `&B::FloatTensorPrimitive` as a
    // `&CubeTensor<CudaRuntime>` — a zero-cost read-only alias with no
    // lifetime extension.
    let cube_tensor: &Target =
        unsafe { &*(prim as *const B::FloatTensorPrimitive as *const Target) };

    Some(cuda_view_from_cube_tensor(cube_tensor))
}

/// Test-only entry point that returns the device-slice element count extracted
/// from the primitive. Used by `tests/cusparse_ptr_spike.rs` to verify the
/// round-trip without exposing the raw pointer publicly.
#[doc(hidden)]
pub fn __spike_extract_len<B: Backend>(prim: &B::FloatTensorPrimitive) -> usize
where
    B::FloatTensorPrimitive: 'static,
{
    primitive_as_cuda_view::<B>(prim)
        .expect("expected Cuda<f32,i32> backend (non-fusion) with an extractable device pointer")
        .len
}

// ---------------------------------------------------------------------------
// Stream access — SP-6 Task 7
// ---------------------------------------------------------------------------

/// Newtype wrapper so the raw `CUstream` pointer can be stored in a
/// `OnceLock` (which requires `Send` + `Sync`).
///
/// SAFETY: CUDA stream handles are process-wide opaque integers.  Sending
/// the value across threads is safe as long as no thread destroys the stream
/// and all CUDA operations using it are properly serialized (cuSPARSE handles
/// internal locking; the caller owns the sync contract with BURN).
struct SendStream(CUstream);

// SAFETY: See doc on `SendStream`.
unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

/// Per-process dedicated cuSPARSE stream.  Created once, on first call.
static FALLBACK_STREAM: std::sync::OnceLock<SendStream> = std::sync::OnceLock::new();

/// Returns a CUDA stream handle suitable for passing to `cusparseSetStream`.
///
/// **Path B — dedicated stream (requires explicit sync on interop).**
///
/// ## Why Path A is blocked
///
/// cubecl-cuda 0.10 keeps all stream state private to the crate:
/// `CudaServer.streams: MultiStream<CudaStreamBackend>` (private field),
/// `Stream.sys: CUstream` (pub field, but the `stream` module is
/// `pub(crate)`), and `GpuStorage.stream` (private).  No public method on
/// `ComputeClient<CudaRuntime>` or the `ComputeServer` trait exposes the raw
/// `CUstream`.  Path A (sharing cubecl's own stream) would require forking
/// cubecl-cuda.
///
/// ## Path B implementation
///
/// `cuStreamCreate` requires the current thread to have an active CUDA
/// context.  That context is bound only on cubecl's server thread.  To
/// avoid `CUDA_ERROR_INVALID_CONTEXT` we dispatch the one-time creation via
/// `ComputeClient::exclusive`, which runs the closure on the server thread
/// where the CUDA context is already current.
///
/// ## Perf implication
///
/// Because this stream is independent of cubecl's scheduler, every
/// cuSPARSE call that hands off data to BURN (or vice-versa) requires one
/// explicit `cudaStreamSynchronize` on the boundary.  Expected cost: one
/// host-side sync per triangular solve.  Acceptable for Task 9; revisit if
/// profiling shows it to be a bottleneck.
///
/// The returned handle is valid for the lifetime of the process.  The
/// caller MUST NOT destroy it.
///
/// # Panics
///
/// Panics if `B` is not the CUDA backend (`B::Device` is not `CudaDevice`),
/// or if `cuStreamCreate` fails.
pub(crate) fn cubecl_cuda_stream<B: Backend>(device: &B::Device) -> CUstream
where
    B::Device: 'static,
{
    use std::any::TypeId;

    // Fast path — already created.
    if let Some(s) = FALLBACK_STREAM.get() {
        return s.0;
    }

    // Downcast B::Device to the concrete CudaDevice type.
    type CudaDev = cubecl::cuda::CudaDevice;
    assert_eq!(
        TypeId::of::<B::Device>(),
        TypeId::of::<CudaDev>(),
        "cubecl_cuda_stream requires the Cuda backend"
    );
    // SAFETY: TypeId equality guarantees same type; reinterpret reference.
    let cuda_device: &CudaDev =
        unsafe { &*(device as *const B::Device as *const CudaDev) };

    // Load a ComputeClient for this device (does NOT create a new server —
    // just borrows the existing DeviceHandle for the already-init'd device).
    // `cubecl::client` re-exports cubecl_runtime::client (via cubecl-core).
    let client = cubecl::client::ComputeClient::<CudaRuntime>::load(cuda_device);

    // Run stream creation on the server thread, where the CUDA context is
    // current.  `exclusive` blocks until the closure returns.
    // The closure returns `SendStream` (which is `Send`) to satisfy the
    // `Re: Send + 'static` bound on `exclusive`.
    let send_stream = client
        .exclusive(|| {
            // SAFETY: cuStreamCreate is called from the server thread where the
            // CUDA context is bound.  The handle is stored for the process
            // lifetime and never freed.
            let raw = cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .expect("cuStreamCreate failed on server thread");
            // SAFETY: wrapping the raw pointer in SendStream so it can cross
            // the channel boundary back to the calling thread.
            SendStream(raw)
        })
        .expect("exclusive task dispatched successfully");

    // Store (race-free: OnceLock ensures only one winner).
    FALLBACK_STREAM.get_or_init(|| send_stream).0
}

/// Test-only entry point for the stream spike.
/// Returns the stream handle so `tests/cusparse_ptr_spike.rs` can assert it
/// is non-null without depending on the private `SendStream` type.
#[doc(hidden)]
pub fn __spike_get_stream<B: Backend>(device: &B::Device) -> CUstream
where
    B::Device: 'static,
{
    cubecl_cuda_stream::<B>(device)
}
