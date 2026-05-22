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

// ---------------------------------------------------------------------------
// SP-6 Task 10: cusparse_backward_solve — GPU backward triangular solve
// ---------------------------------------------------------------------------

/// GPU backward triangular solve: `A^T · y = b` via cuSPARSE TRANSPOSE op.
/// Returns `y` as a new device-resident primitive (host-roundtrip fallback —
/// same temporary strategy as [`cusparse_forward`]).
///
/// This is the upper-triangular back-substitution used in the autograd
/// backward of `triangular_csr_solve` — given upstream gradient `grad_out`,
/// the input gradient on `b` is the solution to `A^T · grad_b = grad_out`.
///
/// Mirrors `cusparse_forward` with two changes:
/// - Uses `cache.desc_backward` (pre-analyzed for TRANSPOSE).
/// - Passes `CUSPARSE_OPERATION_TRANSPOSE` to `cusparseSpSV_solve`.
///
/// cuSPARSE transposes the matrix on the fly; the same `sp_mat` descriptor is
/// reused — no second sparse-matrix descriptor is needed.
///
/// SAFETY assumptions (caller responsibility, checked at dispatch):
/// - Active backend is `Cuda<f32, i32>` (non-fusion).
/// - `a_values_prim` and `b_prim` are on the same CUDA device.
pub(crate) fn cusparse_backward_solve<B: Backend>(
    pattern: &crate::sparse::CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive
where
    B::FloatTensorPrimitive: 'static,
{
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        cusparseSpSV_solve,
        cusparseSetStream,
    };

    // --- 1. Get dedicated stream (initialises the CUDA device via cubecl). ---
    // Must happen before ensure_cuda_cache to guarantee FALLBACK_STREAM is live.
    let stream = cubecl_cuda_stream::<B>(device);

    // --- 2. Lazy-build the pattern cache (one-time per CsrPattern). ---
    // SAFETY: SP-6 single-threaded contract — no concurrent access to cuda_cache.
    let cache = unsafe { ensure_cuda_cache(pattern) };

    // --- 3. Bind cuSPARSE handle to the dedicated stream. ---
    // SAFETY: stream is a non-null CUstream created in Task 7, valid for the
    // process lifetime. The driver and cuSPARSE CUstream_st types share the
    // same underlying CUDA ABI; the transmute is safe.
    unsafe {
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream failed in backward solve");
    }

    // --- 4. Flush cubecl's compute queue before extracting raw pointers. ---
    // a_values_prim and b_prim (= grad_out) are results of BURN tensor ops
    // queued on cubecl's stream. cuSPARSE runs on FALLBACK_STREAM. Without
    // a sync, cuSPARSE could read stale data. B::sync blocks the host until
    // all pending cubecl GPU work completes.
    B::sync(device).expect("cubecl device sync failed before cusparse_backward_solve");

    // --- 5. Extract raw device pointers from BURN tensors. ---
    // SAFETY: a_values_prim and b_prim must not be dropped while these views
    // are live. Both stay alive through the end of this function.
    // All cubecl GPU work is complete (see sync step above).
    let a_view = primitive_as_cuda_view::<B>(a_values_prim)
        .expect("cusparse_backward_solve: Cuda<f32, i32> backend required");
    let b_view = primitive_as_cuda_view::<B>(b_prim)
        .expect("cusparse_backward_solve: Cuda<f32, i32> backend required");

    let n = pattern.n;

    // --- 6. Allocate output y on device (zero-initialised) via FALLBACK_STREAM. ---
    // SAFETY: FALLBACK_STREAM is valid. cuMemAllocAsync + cuMemsetD8Async are
    // context-free on CUDA 12.2+.
    let y_ptr: u64 = unsafe { async_alloc::<f32>(n, stream) };
    unsafe { zero_device(y_ptr, n * 4, stream) };

    unsafe {
        // --- 7. Re-point sparse matrix descriptor at the current a_values. ---
        // SAFETY: a_view.ptr is the live device pointer of a_values_prim.
        // d_crow and d_col are stored as raw CUdeviceptr in the cache.
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            cache.sp_mat,
            cache.d_crow as *mut std::ffi::c_void,
            cache.d_col as *mut std::ffi::c_void,
            a_view.ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseCsrSetPointers failed in backward solve");

        // --- 7b. Notify cuSPARSE that matrix values changed since analysis. ---
        // Same rationale as in cusparse_forward: the SpSV descriptor caches
        // values from analysis time (dummy 1.0s); updateMatrix refreshes them.
        // SAFETY: desc_backward was analyzed; sp_mat has valid values pointer.
        cudarc::cusparse::sys::cusparseSpSV_updateMatrix(
            cache.handle,
            cache.desc_backward,
            a_view.ptr as *mut std::ffi::c_void,
            cudarc::cusparse::sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_GENERAL,
        )
        .result()
        .expect("cusparseSpSV_updateMatrix (backward) failed");

        // --- 8. Build transient dense vector descriptors for b (rhs) and y (out). ---
        // SAFETY: b_view.ptr and y_ptr are live device pointers of the
        // correct element count (pattern.n).
        let mut b_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut y_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(
            &mut b_dn,
            n as i64,
            b_view.ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec b failed in backward solve");

        cusparseCreateDnVec(
            &mut y_dn,
            n as i64,
            y_ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec y failed in backward solve");

        // --- 9. Execute the TRANSPOSE triangular solve: y = (A^T)^{-1} b. ---
        // SAFETY: cusparseSpSV_solve uses the pre-analyzed desc_backward, which
        // was built for CUSPARSE_OPERATION_TRANSPOSE during cache construction.
        // The TRANSPOSE op flag directs cuSPARSE to treat sp_mat (lower-tri L)
        // as its transpose (upper-tri U) and solve U · y = b. The workspace
        // registered during analysis is implicitly reused — no workspace arg here.
        let alpha: f32 = 1.0;
        cusparseSpSV_solve(
            cache.handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const _ as *const std::ffi::c_void,
            cache.sp_mat,
            b_dn,
            y_dn,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            cache.desc_backward,
        )
        .result()
        .expect("cusparseSpSV_solve (TRANSPOSE backward) failed");

        // --- Path B sync: synchronize before reading y back to host. ---
        // SAFETY: stream is the dedicated cuSPARSE stream. Synchronising ensures
        // the solve is complete before memcpy_dtoh_sync reads y.
        cudarc::driver::sys::cuStreamSynchronize(stream)
            .result()
            .expect("cuStreamSynchronize failed after backward cusparseSpSV_solve");

        // Clean up transient descriptors.
        cusparseDestroyDnVec(b_dn)
            .result()
            .expect("cusparseDestroyDnVec b failed in backward solve");
        cusparseDestroyDnVec(y_dn)
            .result()
            .expect("cusparseDestroyDnVec y failed in backward solve");
    }

    // --- 10. Host round-trip: copy y from device to host, then create BURN tensor. ---
    // TEMPORARY FALLBACK: same rationale as cusparse_forward — no public
    // CubeTensor constructor from raw CUdeviceptr in burn-cubecl 0.21.
    // We synchronised `stream` above, so y is fully written.
    let mut y_host = vec![0.0f32; n];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_sync(&mut y_host, y_ptr)
            .expect("cuMemcpyDtoH y failed in cusparse_backward_solve");
        // Free the temporary device y buffer.
        cudarc::driver::result::free_async(y_ptr, stream)
            .expect("cuMemFreeAsync y_ptr failed in backward solve");
    }

    B::float_from_data(
        burn::tensor::TensorData::from(y_host.as_slice()),
        device,
    )
}

// ---------------------------------------------------------------------------
// SP-6 Task 11: cusparse_grada — GPU per-nnz scatter via pure BURN tensor ops
// ---------------------------------------------------------------------------

/// GPU per-nnz grada scatter using pure BURN tensor ops:
///
///     grada[k] = -gradb[row_for_nnz[k]] * x[col[k]]
///
/// No custom CUDA kernel — `Tensor::select` (gather) compiles on any backend
/// (NdArray for CPU tests, Cuda for the GPU path). This path runs entirely on
/// device when invoked with a `Cuda<f32, i32>` backend.
///
/// Inputs are `FloatTensorPrimitive` to match the dispatch interface; they are
/// wrapped into `Tensor<B, 1>` for the select/multiply/negate ops and then
/// unwrapped back.
pub(crate) fn cusparse_grada<B: Backend>(
    pattern: &crate::sparse::CsrPattern,
    gradb_prim: B::FloatTensorPrimitive,
    x_prim: B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};

    // 1. Lift index arrays as BURN Int tensors on the device.
    let row_t: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::from(pattern.row_for_nnz.as_slice()),
        device,
    );
    let col_t: Tensor<B, 1, Int> = Tensor::from_data(
        TensorData::from(pattern.col.as_slice()),
        device,
    );

    // 2. Wrap the input primitives as Tensors.
    let gradb_t: Tensor<B, 1> =
        Tensor::from_primitive(TensorPrimitive::Float(gradb_prim));
    let x_t: Tensor<B, 1> =
        Tensor::from_primitive(TensorPrimitive::Float(x_prim));

    // 3. Gather + multiply + negate: grada[k] = -gradb[row[k]] * x[col[k]]
    let gradb_gathered: Tensor<B, 1> = gradb_t.select(0, row_t);
    let x_gathered: Tensor<B, 1> = x_t.select(0, col_t);
    let grada: Tensor<B, 1> = -(gradb_gathered * x_gathered);

    // 4. Unwrap back to FloatTensorPrimitive.
    match grada.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("grada is f32"),
    }
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
// SP-7 Task 4: compute_client, cubecl_stream_active, cube_tensor_to_primitive
// ---------------------------------------------------------------------------

/// Obtain the cubecl `ComputeClient` for the given BURN Cuda device.
///
/// Panics if `B` is not `Cuda<f32, i32>`. Callers must gate via
/// `dispatch::backend_is_cuda::<B>()` (or the TypeId check inline).
fn compute_client<B: Backend + 'static>(
    device: &B::Device,
) -> cubecl::client::ComputeClient<CudaRuntime> {
    use std::any::TypeId;
    // burn_cuda::Cuda<f32, i32> is the concrete non-fusion Cuda backend type.
    // burn::backend::Cuda is gated behind the "cuda" feature on the burn umbrella
    // crate; burn_cuda exposes it unconditionally.
    assert_eq!(
        TypeId::of::<B::Device>(),
        TypeId::of::<cubecl::cuda::CudaDevice>(),
        "compute_client requires Cuda<f32, i32> backend",
    );
    // SAFETY: TypeId match above guarantees layout compatibility between
    // B::Device and CudaDevice. Borrow only for client lookup.
    let cuda_device: &cubecl::cuda::CudaDevice =
        unsafe { &*(device as *const B::Device as *const cubecl::cuda::CudaDevice) };
    <CudaRuntime as cubecl::Runtime>::client(cuda_device)
}

/// Returns cubecl-cuda's active CUDA stream for the current logical stream.
///
/// Uses `ComputeClient::exclusive_with_server` (added in the SP-7 cubecl-runtime
/// fork) to run `CudaServer::stream(StreamId::current())` on the server-bound
/// thread with `&mut` access. Replaces SP-6's `cubecl_cuda_stream`
/// (FALLBACK_STREAM) in Tasks 5+6; Task 8 deletes the old function.
///
/// SAFETY: returned `CUstream` is owned by cubecl. Caller must not destroy
/// it or queue conflicting work without explicit stream serialization.
pub(crate) fn cubecl_stream_active<B: Backend + 'static>(
    device: &B::Device,
) -> cudarc::driver::sys::CUstream {
    use cubecl_common::stream_id::StreamId;
    let client = compute_client::<B>(device);
    // exclusive_with_server() is the SP-7 cubecl-runtime fork addition (Task 4b).
    // It runs the closure on the server-bound thread with mutable access.
    // stream() needs &mut self because cubecl lazy-inits the stream on first call.
    // CUstream is a raw pointer (*mut _) and not Send, so we cast to usize for
    // the cross-thread return and cast back. The value is a CUDA handle (opaque
    // integer); no dereference occurs during the transfer.
    let ptr: usize = client
        .exclusive_with_server(|server| server.stream(StreamId::current()) as usize)
        .expect("exclusive_with_server failed to read cubecl stream");
    ptr as cudarc::driver::sys::CUstream
}

/// Convert a `CubeTensor<CudaRuntime>` into the BURN backend's
/// `FloatTensorPrimitive`. Inverse of `primitive_as_cuda_view`.
///
/// SAFETY: caller verified via TypeId that `B == Cuda<f32, i32>`. The
/// `CubeTensor` and `B::FloatTensorPrimitive` share layout under that
/// equality. Ownership of the cubecl handle transfers into BURN's tape;
/// the cubecl-allocated buffer is freed when BURN drops the primitive.
pub(crate) fn cube_tensor_to_primitive<B: Backend + 'static>(
    cube: CubeTensor<CudaRuntime>,
) -> B::FloatTensorPrimitive {
    use std::any::TypeId;
    // burn_cuda::Cuda<f32, i32> is the concrete non-fusion Cuda backend. We
    // use burn_cuda directly because burn::backend::Cuda requires the "cuda"
    // feature on the burn umbrella crate (not enabled in ddrs's Cargo.toml).
    assert_eq!(
        TypeId::of::<B>(),
        TypeId::of::<burn_cuda::Cuda<f32, i32>>(),
        "cube_tensor_to_primitive requires Cuda<f32, i32> backend",
    );
    // SAFETY: TypeId equality guarantees identical layout between
    // `CubeTensor<CudaRuntime>` and `B::FloatTensorPrimitive` (they are the
    // same type). Use ManuallyDrop + ptr::read to move ownership without
    // running the source Drop on the reinterpreted bits.
    let cube = std::mem::ManuallyDrop::new(cube);
    unsafe {
        std::ptr::read(
            &*cube as *const CubeTensor<CudaRuntime> as *const B::FloatTensorPrimitive,
        )
    }
}

#[doc(hidden)]
pub fn __spike_active_stream<B: Backend + 'static>(
    device: &B::Device,
) -> cudarc::driver::sys::CUstream {
    cubecl_stream_active::<B>(device)
}

#[doc(hidden)]
pub fn __spike_cube_round_trip<B: Backend + 'static>(
    device: &B::Device,
    data: Vec<f32>,
) -> Vec<f32> {
    use burn::tensor::{Tensor, TensorPrimitive};
    use burn_cubecl::cubecl::server::Handle;
    use burn::tensor::{DType, Shape};

    let client = compute_client::<B>(device);

    // Allocate + upload: use client.create_from_slice which takes &[u8] and
    // returns a Handle without needing the private cubecl_common::Bytes type.
    // SAFETY: f32 slice reinterpreted as u8 bytes; alignment/size are valid.
    let data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    // `client.create_from_slice` copies bytes to device, returns a Handle.
    let handle: Handle = client.create_from_slice(data_bytes);

    // Wrap as CubeTensor via the SP-7 fork's from_handle constructor.
    // ARG ORDER (from Task 3 implementer): (client, device, shape, handle, dtype).
    let shape = Shape::from(vec![data.len()]);
    // SAFETY: TypeId of B::Device == CudaDevice asserted by compute_client.
    let cuda_device: cubecl::cuda::CudaDevice =
        unsafe { std::ptr::read(device as *const B::Device as *const cubecl::cuda::CudaDevice) };
    let cube = CubeTensor::<CudaRuntime>::from_handle(
        client,
        cuda_device,
        shape,
        handle,
        DType::F32,
    );

    let prim = cube_tensor_to_primitive::<B>(cube);
    let tensor: Tensor<B, 1> = Tensor::from_primitive(TensorPrimitive::Float(prim));
    tensor.into_data().to_vec::<f32>().unwrap()
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

// ---------------------------------------------------------------------------
// CudaPatternCache — SP-6 Task 8 + Task 9
// ---------------------------------------------------------------------------

use std::marker::PhantomData;

/// Per-pattern cuSPARSE state. Built lazily on first GPU solve call.
///
/// !Send because cuSPARSE descriptors and CUDA contexts are tied to the
/// thread that created them. Single-threaded training is the only supported
/// mode for SP-6.
///
/// Device allocations are stored as raw `CUdeviceptr` (u64) values rather
/// than cudarc `CudaSlice<T>` to avoid the `CudaContext`/`CudaStream` RAII
/// wrappers. This is necessary because:
///
/// 1. `CudaContext::new(0)` calls `cuCtxSetCurrent` on the calling thread,
///    which conflicts with cubecl's server thread holding the primary context.
/// 2. CUDA 12.2+ supports context-free stream-ordered allocation (`cuMemAllocAsync`),
///    and our target (CUDA 13.2) supports this. We use `FALLBACK_STREAM` for all
///    stream-ordered async allocation, bypassing the RAII wrappers.
///
/// Drop frees all device memory with `cuMemFreeAsync` on `FALLBACK_STREAM`.
pub(crate) struct CudaPatternCache {
    pub(crate) handle: cudarc::cusparse::sys::cusparseHandle_t,
    /// Device pointer for crow (i32 array, len = n+1). Freed in Drop.
    pub(crate) d_crow: u64,
    /// Device pointer for col (i32 array, len = nnz). Freed in Drop.
    pub(crate) d_col: u64,
    /// Device pointer for row_for_nnz (i32 array, len = nnz). Freed in Drop.
    pub(crate) d_row_for_nnz: u64,
    pub(crate) sp_mat: cudarc::cusparse::sys::cusparseSpMatDescr_t,
    pub(crate) desc_forward: cudarc::cusparse::sys::cusparseSpSVDescr_t,
    pub(crate) desc_backward: cudarc::cusparse::sys::cusparseSpSVDescr_t,
    /// Device pointer for forward workspace (u8 array). Freed in Drop.
    pub(crate) workspace_forward: u64,
    /// Device pointer for backward workspace (u8 array). Freed in Drop.
    pub(crate) workspace_backward: u64,
    /// `!Send` marker — cuSPARSE descriptors are thread-bound.
    _not_send: PhantomData<*mut ()>,
}

impl Drop for CudaPatternCache {
    fn drop(&mut self) {
        // SAFETY: All device pointers were allocated with cuMemAllocAsync on
        // FALLBACK_STREAM.  We free them on the same stream so that any pending
        // async work referencing them completes before the memory is reclaimed.
        // cuSPARSE descriptors are destroyed first (in dependency order), then
        // sp_mat and handle. Raw device memory is freed last.
        //
        // FALLBACK_STREAM may be None if the process is exiting without ever
        // using the GPU path — in that case we skip the CUDA cleanup and let
        // the OS reclaim GPU memory.
        unsafe {
            // Destroy SpSV descriptors.
            cudarc::cusparse::sys::cusparseSpSV_destroyDescr(self.desc_forward)
                .result()
                .expect("cusparseSpSV_destroyDescr (forward) failed");
            cudarc::cusparse::sys::cusparseSpSV_destroyDescr(self.desc_backward)
                .result()
                .expect("cusparseSpSV_destroyDescr (backward) failed");
            // Destroy sparse matrix descriptor.
            cudarc::cusparse::sys::cusparseDestroySpMat(self.sp_mat)
                .result()
                .expect("cusparseDestroySpMat failed");
            // Destroy cuSPARSE handle.
            cudarc::cusparse::sys::cusparseDestroy(self.handle)
                .result()
                .expect("cusparseDestroy failed");

            // Free device memory. Use FALLBACK_STREAM for async free if available.
            if let Some(stream) = FALLBACK_STREAM.get() {
                let s = stream.0;
                cudarc::driver::result::free_async(self.d_crow, s)
                    .expect("cuMemFreeAsync d_crow failed");
                cudarc::driver::result::free_async(self.d_col, s)
                    .expect("cuMemFreeAsync d_col failed");
                cudarc::driver::result::free_async(self.d_row_for_nnz, s)
                    .expect("cuMemFreeAsync d_row_for_nnz failed");
                cudarc::driver::result::free_async(self.workspace_forward, s)
                    .expect("cuMemFreeAsync workspace_forward failed");
                cudarc::driver::result::free_async(self.workspace_backward, s)
                    .expect("cuMemFreeAsync workspace_backward failed");
            }
        }
    }
}

/// A wrapper around `Option<CudaPatternCache>` stored in an `UnsafeCell` so
/// that `CsrPattern` (which is `Arc`-shared and stored in the autograd tape's
/// `Send` state) can hold it.
///
/// SAFETY: SP-6 single-threaded training guarantee — only the training thread
/// ever calls `ensure_cuda_cache` or reads/writes the inner value. The
/// `UnsafeCell` is never accessed from two threads concurrently. If this
/// invariant is violated the program will have a data race, which is UB; the
/// single-threaded contract must be maintained by all callers.
pub(crate) struct UnsafeSendCache(std::cell::UnsafeCell<Option<CudaPatternCache>>);

// SAFETY: see doc on `UnsafeSendCache` above.
unsafe impl Send for UnsafeSendCache {}
unsafe impl Sync for UnsafeSendCache {}

impl UnsafeSendCache {
    pub(crate) fn new() -> Self {
        Self(std::cell::UnsafeCell::new(None))
    }

    /// Initialize the cache if not yet set, then return a reference to it.
    ///
    /// SAFETY: caller must guarantee exclusive access (single-threaded context).
    pub(crate) unsafe fn get_or_init(
        &self,
        init: impl FnOnce() -> CudaPatternCache,
    ) -> &CudaPatternCache {
        let ptr = self.0.get();
        if (*ptr).is_none() {
            *ptr = Some(init());
        }
        (*ptr).as_ref().unwrap()
    }
}

/// Build or retrieve the GPU cache for this pattern. Allocates device
/// memory for crow/col/row_for_nnz on first call; subsequent calls return
/// the cached handle.
///
/// SAFETY: caller must guarantee the current thread has an active CUDA
/// context and that no other thread concurrently accesses the pattern's
/// `cuda_cache`. The returned reference is valid for the lifetime of `pattern`.
pub(crate) unsafe fn ensure_cuda_cache(
    pattern: &crate::sparse::CsrPattern,
) -> &CudaPatternCache {
    pattern
        .cuda_cache
        .get_or_init(|| build_cuda_pattern_cache(pattern))
}

// ---------------------------------------------------------------------------
// SP-6 Task 9: build_cuda_pattern_cache implementation
// ---------------------------------------------------------------------------

/// Allocate stream-ordered async device memory of `len` elements of type `T`.
///
/// Uses `cuMemAllocAsync` on `FALLBACK_STREAM`. Requires CUDA 12.2+ (CUDA 13.2
/// is installed). Returns the raw `CUdeviceptr` (u64).
///
/// SAFETY: caller must ensure FALLBACK_STREAM is initialised and valid.
unsafe fn async_alloc<T>(len: usize, stream: CUstream) -> u64 {
    let bytes = len * std::mem::size_of::<T>();
    cudarc::driver::result::malloc_async(stream, bytes)
        .expect("cuMemAllocAsync failed in build_cuda_pattern_cache")
}

/// Upload a `&[T]` to device memory at `d_ptr` using `cuMemcpyHtoDAsync`.
///
/// SAFETY: d_ptr must be a valid device allocation of at least `src.len()` elements,
/// and the allocation must be associated with `stream`.
unsafe fn upload_slice<T>(d_ptr: u64, src: &[T], stream: CUstream) {
    let bytes = std::mem::size_of_val(src);
    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
        d_ptr,
        src.as_ptr() as *const _,
        bytes,
        stream,
    )
    .result()
    .expect("cuMemcpyHtoDAsync failed");
}

/// Zero-fill `len` bytes at `d_ptr` using `cuMemsetD8Async`.
///
/// SAFETY: d_ptr must be a valid device allocation of at least `len` bytes.
unsafe fn zero_device(d_ptr: u64, len_bytes: usize, stream: CUstream) {
    cudarc::driver::result::memset_d8_async(d_ptr, 0, len_bytes, stream)
        .expect("cuMemsetD8Async failed");
}

/// Allocate device memory + create cuSPARSE descriptors for this pattern.
///
/// This runs once per `CsrPattern` lifetime and performs:
/// 1. Create cuSPARSE handle
/// 2. Upload crow/col/row_for_nnz to device via FALLBACK_STREAM
/// 3. Create sparse matrix descriptor (values=NULL, set per-solve)
/// 4. Set fill mode (lower) + diag type (non-unit)
/// 5. Create SpSV descriptors (forward + backward)
/// 6. Probe and allocate workspace buffers
/// 7. Run cusparseSpSV_analysis for both directions
///
/// All device allocations use `cuMemAllocAsync` on `FALLBACK_STREAM` (CUDA 12.2+
/// context-free async allocation). This avoids binding a CUDA context on the
/// calling thread, which would conflict with cubecl's server thread.
fn build_cuda_pattern_cache(pattern: &crate::sparse::CsrPattern) -> CudaPatternCache {
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateCsr,
        cusparseCreateDnVec,
        cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT,
        cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER,
        cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        cusparseIndexType_t::CUSPARSE_INDEX_32I,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        cusparseSpMatAttribute_t::{CUSPARSE_SPMAT_DIAG_TYPE, CUSPARSE_SPMAT_FILL_MODE},
        cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        cusparseSpSV_analysis,
        cusparseSpSV_bufferSize,
        cusparseSpSV_createDescr,
        cusparseSpMatSetAttribute,
        cusparseDestroyDnVec,
    };

    // All operations use FALLBACK_STREAM (created by cubecl_cuda_stream on
    // the server thread before this function is called).
    // SAFETY: FALLBACK_STREAM is guaranteed to be initialised because
    // build_cuda_pattern_cache is only called from ensure_cuda_cache, which
    // is only called from cusparse_forward, which calls cubecl_cuda_stream
    // (initialising FALLBACK_STREAM) before calling ensure_cuda_cache.
    let stream = FALLBACK_STREAM
        .get()
        .expect("FALLBACK_STREAM not initialised — ensure cubecl_cuda_stream was called first")
        .0;

    // --- Step 1: Create cuSPARSE handle. ---
    // SAFETY: cusparseCreate doesn't require a current CUDA context in CUDA 12+.
    let handle = cudarc::cusparse::result::create()
        .expect("cusparseCreate failed — CUDA setup broken");

    unsafe {
        // --- Step 2: Upload structural arrays to device via FALLBACK_STREAM. ---
        // SAFETY: FALLBACK_STREAM is a valid non-blocking stream created on the
        // server thread. cuMemAllocAsync and cuMemcpyHtoDAsync in CUDA 12.2+ do
        // not require a current context on the calling thread.

        let d_crow = async_alloc::<i32>(pattern.crow.len(), stream);
        upload_slice(d_crow, &pattern.crow, stream);

        let d_col = async_alloc::<i32>(pattern.col.len(), stream);
        upload_slice(d_col, &pattern.col, stream);

        let d_row_for_nnz = async_alloc::<i32>(pattern.row_for_nnz.len(), stream);
        upload_slice(d_row_for_nnz, &pattern.row_for_nnz, stream);

        // Synchronise so the uploads are complete before we pass pointers to cuSPARSE.
        cudarc::driver::sys::cuStreamSynchronize(stream)
            .result()
            .expect("stream sync after htod upload");

        let n = pattern.n as i64;
        let nnz = pattern.col.len() as i64;

        // --- Step 3: Create sparse matrix descriptor (values = NULL). ---
        // Values will be set per-call via cusparseCsrSetPointers in cusparse_forward.
        let mut sp_mat: cudarc::cusparse::sys::cusparseSpMatDescr_t = std::ptr::null_mut();
        cusparseCreateCsr(
            &mut sp_mat,
            n,
            n,
            nnz,
            d_crow as *mut std::ffi::c_void,
            d_col as *mut std::ffi::c_void,
            std::ptr::null_mut(), // values — set per-call
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateCsr failed");

        // --- Step 4: Set fill mode (lower triangular) and diag type (non-unit). ---
        // SAFETY: sp_mat is a valid descriptor. Attribute values are passed as
        // pointers to local variables per the cuSPARSE API contract.
        let fill_mode = CUSPARSE_FILL_MODE_LOWER;
        cusparseSpMatSetAttribute(
            sp_mat,
            CUSPARSE_SPMAT_FILL_MODE,
            &fill_mode as *const _ as *mut std::ffi::c_void,
            std::mem::size_of_val(&fill_mode),
        )
        .result()
        .expect("cusparseSpMatSetAttribute FILL_MODE failed");

        let diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(
            sp_mat,
            CUSPARSE_SPMAT_DIAG_TYPE,
            &diag_type as *const _ as *mut std::ffi::c_void,
            std::mem::size_of_val(&diag_type),
        )
        .result()
        .expect("cusparseSpMatSetAttribute DIAG_TYPE failed");

        // --- Step 5: Create SpSV descriptors for forward (L \ b) and backward (L^T \ b). ---
        let mut desc_forward: cudarc::cusparse::sys::cusparseSpSVDescr_t = std::ptr::null_mut();
        let mut desc_backward: cudarc::cusparse::sys::cusparseSpSVDescr_t = std::ptr::null_mut();
        cusparseSpSV_createDescr(&mut desc_forward)
            .result()
            .expect("cusparseSpSV_createDescr (forward) failed");
        cusparseSpSV_createDescr(&mut desc_backward)
            .result()
            .expect("cusparseSpSV_createDescr (backward) failed");

        // --- Step 6: Probe buffer sizes + run analysis. ---
        // cuSPARSE needs dense vector descriptors for buffer sizing and analysis.
        // Allocate small dummy device buffers (zero-filled) for b and x.
        let dummy_b = async_alloc::<f32>(pattern.n, stream);
        let dummy_x = async_alloc::<f32>(pattern.n, stream);
        zero_device(dummy_b, pattern.n * 4, stream);
        zero_device(dummy_x, pattern.n * 4, stream);
        cudarc::driver::sys::cuStreamSynchronize(stream)
            .result()
            .expect("sync after dummy alloc");

        let mut b_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut x_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut b_dn, n, dummy_b as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec b failed");
        cusparseCreateDnVec(&mut x_dn, n, dummy_x as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec x failed");

        let alpha: f32 = 1.0;

        // --- Forward: NON_TRANSPOSE (lower-tri solve L \ b). ---
        let mut fwd_buf_sz: usize = 0;
        cusparseSpSV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const _ as *const std::ffi::c_void,
            sp_mat,
            b_dn,
            x_dn,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            desc_forward,
            &mut fwd_buf_sz,
        )
        .result()
        .expect("cusparseSpSV_bufferSize (forward) failed");

        // --- Backward: TRANSPOSE (upper-tri solve L^T \ b for adjoint). ---
        let mut bwd_buf_sz: usize = 0;
        cusparseSpSV_bufferSize(
            handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const _ as *const std::ffi::c_void,
            sp_mat,
            b_dn,
            x_dn,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            desc_backward,
            &mut bwd_buf_sz,
        )
        .result()
        .expect("cusparseSpSV_bufferSize (backward) failed");

        // Allocate workspaces. Minimum 4 bytes to avoid zero-size alloc issues.
        let workspace_forward = async_alloc::<u8>(fwd_buf_sz.max(4), stream);
        let workspace_backward = async_alloc::<u8>(bwd_buf_sz.max(4), stream);

        // Allocate dummy values for the analysis step. cusparseSpSV_analysis
        // requires a valid (non-NULL) values pointer — NULL values is only
        // allowed at descriptor creation time. We use 1.0 values (valid lower-tri).
        let nnz_usize = pattern.col.len();
        let dummy_vals = async_alloc::<f32>(nnz_usize, stream);
        // Fill with 1.0 (all-ones matrix is valid for structural analysis).
        {
            let ones: Vec<f32> = vec![1.0f32; nnz_usize];
            upload_slice(dummy_vals, &ones, stream);
        }
        cudarc::driver::sys::cuStreamSynchronize(stream)
            .result()
            .expect("sync after workspace alloc");

        // Set the values pointer on sp_mat to the dummy values for analysis.
        // SAFETY: dummy_vals is a valid f32 device array of length nnz.
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            sp_mat,
            d_crow as *mut std::ffi::c_void,
            d_col as *mut std::ffi::c_void,
            dummy_vals as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseCsrSetPointers (analysis dummy) failed");

        // --- Analysis (pattern-once step, reused every timestep). ---
        // SAFETY: sp_mat now has valid values (dummy_vals). Analysis reads structure
        // and optionally values depending on the algorithm.
        cusparseSpSV_analysis(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const _ as *const std::ffi::c_void,
            sp_mat,
            b_dn,
            x_dn,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            desc_forward,
            workspace_forward as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseSpSV_analysis (forward) failed");

        cusparseSpSV_analysis(
            handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const _ as *const std::ffi::c_void,
            sp_mat,
            b_dn,
            x_dn,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            desc_backward,
            workspace_backward as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseSpSV_analysis (backward) failed");

        // Sync after analysis — analysis is async.
        cudarc::driver::sys::cuStreamSynchronize(stream)
            .result()
            .expect("sync after cusparseSpSV_analysis");

        // Destroy transient dense vector descriptors and free dummy buffers.
        cusparseDestroyDnVec(b_dn)
            .result()
            .expect("cusparseDestroyDnVec b failed");
        cusparseDestroyDnVec(x_dn)
            .result()
            .expect("cusparseDestroyDnVec x failed");
        cudarc::driver::result::free_async(dummy_b, stream)
            .expect("cuMemFreeAsync dummy_b failed");
        cudarc::driver::result::free_async(dummy_x, stream)
            .expect("cuMemFreeAsync dummy_x failed");
        cudarc::driver::result::free_async(dummy_vals, stream)
            .expect("cuMemFreeAsync dummy_vals failed");

        CudaPatternCache {
            handle,
            d_crow,
            d_col,
            d_row_for_nnz,
            sp_mat,
            desc_forward,
            desc_backward,
            workspace_forward,
            workspace_backward,
            _not_send: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// SP-7 Task 5: handle_device_ptr helper
// ---------------------------------------------------------------------------

/// Extract the raw CUDA device pointer (as u64) from a cubecl `Handle`.
///
/// Calls `client.get_resource(handle.clone())` to retrieve the `GpuResource`
/// associated with the handle, then reads `GpuResource.ptr` (a `CUdeviceptr`).
/// The `GpuResource` type is inferred — it does not need to be named explicitly.
///
/// # Safety
///
/// - Caller must not free the returned pointer manually; cubecl owns the buffer.
/// - The `handle` (or a clone of it) must stay alive while the pointer is in use,
///   so that the underlying allocation is not reclaimed.
/// - The returned pointer is valid for GPU access on cubecl's active stream.
///   The caller is responsible for ensuring stream-ordering between cubecl ops
///   and any external (cuSPARSE) ops that use the pointer.
unsafe fn handle_device_ptr(
    client: &cubecl::client::ComputeClient<CudaRuntime>,
    handle: &burn_cubecl::cubecl::server::Handle,
) -> u64 {
    // `get_resource` is a blocking server call that returns `ManagedResource<GpuResource>`.
    // `GpuResource` is a pub struct in cubecl-cuda with a `pub ptr: u64` field.
    // We access it via type inference — no need to name `GpuResource` explicitly.
    // SAFETY: We clone the handle to keep the allocation alive; we do not free the ptr.
    let resource = client
        .get_resource(handle.clone())
        .expect("handle_device_ptr: failed to get GpuResource for Handle");
    resource.resource().ptr
}

// ---------------------------------------------------------------------------
// SP-7 Task 5 (replaces SP-6 Task 9): cusparse_forward — shared-stream, zero-copy x
// ---------------------------------------------------------------------------

/// GPU forward solve `A · x = b` for lower-triangular `A` via cuSPARSE.
/// Returns a new `B::FloatTensorPrimitive` for `x` on the same device.
///
/// SP-7 Task 5 version — shared-stream, no host roundtrip:
/// - Binds cuSPARSE to cubecl's active CUDA stream (no FALLBACK_STREAM).
/// - Allocates x via `client.create_from_slice` (cubecl-owned buffer).
/// - Returns a `CubeTensor` wrapping the handle — no `cuMemcpyDtoH` or
///   `B::float_from_data`.
/// - No `B::sync`; a `client.flush()` submits cubecl's pending kernels onto
///   the shared stream so cuSPARSE reads them in order.
///
/// SAFETY assumptions (caller responsibility, checked at dispatch):
/// - Active backend is `Cuda<f32, i32>` (non-fusion).
/// - `a_values_prim` and `b_prim` are already on the same CUDA device.
pub(crate) fn cusparse_forward<B: Backend + 'static>(
    pattern: &crate::sparse::CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::server::Handle;
    use burn_cubecl::tensor::CubeTensor;
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        cusparseSpSV_solve,
        cusparseSetStream,
    };

    // 1. Ensure FALLBACK_STREAM is initialised before build_cuda_pattern_cache.
    //    build_cuda_pattern_cache still uses FALLBACK_STREAM for the one-time
    //    crow/col upload (Task 7 will migrate this to cubecl Handles; until
    //    then we must guarantee the stream exists before the first cache build).
    //    cubecl_cuda_stream is idempotent after the first call.
    cubecl_cuda_stream::<B>(device);

    // 2. Lazy-build the pattern cache (one-time per CsrPattern).
    // SAFETY: SP-6/SP-7 single-threaded training contract — no concurrent
    // access to cuda_cache from multiple threads.
    let cache = unsafe { ensure_cuda_cache(pattern) };

    // 2. Bind cuSPARSE to cubecl's active stream. cubecl queues kernels that
    //    write a_values + b onto this stream; cuSPARSE will run after them
    //    automatically because they share the same stream. No host sync needed.
    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        // SAFETY: `cudarc::driver::sys::CUstream_st` and
        // `cudarc::cusparse::sys::CUstream_st` are the same opaque CUDA ABI
        // type. cudarc generates separate FFI bindings per sub-crate; casting
        // the pointer integer is safe because no dereference occurs here.
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream forward failed");
    }

    // 3. Flush cubecl's kernel queue onto the active stream before cuSPARSE
    //    reads a_values / b. `client.flush()` is a non-blocking server-side
    //    operation: it submits pending work to the CUDA stream without blocking
    //    the host. cuSPARSE, running on the same stream, will execute after.
    //    This is cheaper than B::sync (which blocks the host via cuEventSynchronize).
    let client = compute_client::<B>(device);
    client.flush().expect("cubecl client flush failed before cusparse_forward");

    // 4. Extract device pointers for a_values + b.
    // SAFETY: a_values_prim and b_prim must stay alive for the duration of
    // this function (they are parameters). cubecl's pending kernels have been
    // submitted to the stream (step 3); cuSPARSE reads them in order.
    let a_view = primitive_as_cuda_view::<B>(a_values_prim)
        .expect("cusparse_forward: Cuda<f32, i32> backend required");
    let b_view = primitive_as_cuda_view::<B>(b_prim)
        .expect("cusparse_forward: Cuda<f32, i32> backend required");

    let n = pattern.n;

    // 5. Allocate x via the cubecl client. The returned Handle owns the GPU
    //    buffer; cubecl frees it when the resulting CubeTensor is dropped.
    //    Zero-initialise so cuSPARSE can safely accumulate into x.
    let n_bytes = n * std::mem::size_of::<f32>();
    let x_bytes = vec![0u8; n_bytes];
    // SAFETY: create_from_slice uploads bytes to device and returns a Handle.
    // The Handle keeps the allocation alive for as long as we (or the resulting
    // CubeTensor) hold it.
    let x_handle: Handle = client.create_from_slice(&x_bytes);

    // SAFETY: x_handle is a freshly allocated, still-owned handle. We clone it
    // to keep the allocation alive while extracting the raw ptr; the original
    // x_handle is consumed later by CubeTensor::from_handle.
    let x_ptr = unsafe { handle_device_ptr(&client, &x_handle) } as *mut f32;

    // 6. cuSPARSE solve — same descriptor + updateMatrix dance as SP-6.
    unsafe {
        // 6a. Re-point sparse matrix descriptor at the current a_values.
        // SAFETY: a_view.ptr is the live device pointer of a_values_prim.
        // cache.d_crow and cache.d_col are raw CUdeviceptr u64 values from SP-6;
        // they remain raw until Task 7 migrates them to cubecl Handles.
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            cache.sp_mat,
            cache.d_crow as *mut std::ffi::c_void,
            cache.d_col as *mut std::ffi::c_void,
            a_view.ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseCsrSetPointers forward failed");

        // 6b. Notify cuSPARSE that matrix values changed since analysis.
        // The SpSV descriptor caches values from analysis time (dummy 1.0s);
        // updateMatrix refreshes them so the solve uses the current a_values.
        // SAFETY: desc_forward was analyzed; sp_mat now has valid a_values ptr.
        cudarc::cusparse::sys::cusparseSpSV_updateMatrix(
            cache.handle,
            cache.desc_forward,
            a_view.ptr as *mut std::ffi::c_void,
            cudarc::cusparse::sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_GENERAL,
        )
        .result()
        .expect("cusparseSpSV_updateMatrix forward failed");

        // 6c. Build transient dense vector descriptors for b and x.
        // SAFETY: b_view.ptr and x_ptr are live device pointers of the
        // correct element count (n). They remain valid through the solve.
        let mut b_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut x_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(
            &mut b_dn,
            n as i64,
            b_view.ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec b failed in forward");
        cusparseCreateDnVec(
            &mut x_dn,
            n as i64,
            x_ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec x failed in forward");

        // 6d. Execute the triangular solve: x = A^{-1} b.
        // SAFETY: desc_forward was pre-analyzed for CUSPARSE_OPERATION_NON_TRANSPOSE.
        // The handle is bound to cubecl's active stream (step 2), so the solve
        // executes after all upstream cubecl kernels on that stream.
        // cusparseSpSV_solve does NOT take a workspace arg — the workspace is
        // registered implicitly during cusparseSpSV_analysis.
        let alpha: f32 = 1.0;
        cusparseSpSV_solve(
            cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const _ as *const std::ffi::c_void,
            cache.sp_mat,
            b_dn,
            x_dn,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            cache.desc_forward,
        )
        .result()
        .expect("cusparseSpSV_solve forward failed");
        // NO cuStreamSynchronize — cubecl's next op on the same stream will
        // execute after the cuSPARSE solve automatically.

        // Clean up transient dense vector descriptors.
        // SAFETY: b_dn and x_dn are valid descriptors created above.
        cusparseDestroyDnVec(b_dn)
            .result()
            .expect("cusparseDestroyDnVec b failed in forward");
        cusparseDestroyDnVec(x_dn)
            .result()
            .expect("cusparseDestroyDnVec x failed in forward");
    }

    // 7. Wrap x_handle as CubeTensor → B::FloatTensorPrimitive.
    //    No host roundtrip. cubecl owns the x buffer via x_handle; it will be
    //    freed when BURN drops the resulting tensor.
    let shape = Shape::from(vec![n]);
    // SAFETY: TypeId of B::Device == CudaDevice is asserted inside compute_client.
    // transmute_copy reads the device value by-copy without moving `device`.
    let cuda_device: <CudaRuntime as cubecl::Runtime>::Device =
        unsafe { std::mem::transmute_copy(device) };
    let cube = CubeTensor::<CudaRuntime>::from_handle(
        client,
        cuda_device,
        shape,
        x_handle,
        DType::F32,
    );
    cube_tensor_to_primitive::<B>(cube)
}
