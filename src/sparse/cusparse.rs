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
pub(crate) fn cusparse_backward_solve<B: Backend + 'static>(
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
        cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        cusparseSpSV_solve,
        cusparseSetStream,
    };

    // 1. Lazy-build the pattern cache (one-time per CsrPattern).
    //    Task 7: build_cuda_pattern_cache now uses cubecl Handles internally;
    // SAFETY: SP-7 single-threaded training contract — no concurrent access to
    // cuda_cache from multiple threads.
    let cache = unsafe { ensure_cuda_cache(pattern) };

    // 3. Bind cuSPARSE to cubecl's active stream. cubecl queues kernels that
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
            .expect("cusparseSetStream backward failed");
    }

    // 4. Flush cubecl's kernel queue onto the active stream before cuSPARSE
    //    reads a_values / b. `client.flush()` is a non-blocking server-side
    //    operation: it submits pending work to the CUDA stream without blocking
    //    the host. cuSPARSE, running on the same stream, will execute after.
    //    This is cheaper than B::sync (which blocks the host via cuEventSynchronize).
    let client = compute_client::<B>(device);
    // SP-10: use flush_no_sync so this call is safe inside a
    // cuStreamBeginCapture region (the regular client.flush() triggers
    // cuEventSynchronize via the drop_queue rotation, which would
    // invalidate the capture). flush_no_sync is also strictly cheaper on
    // the non-capture path: CUDA kernel/copy submission happens at issue
    // time, so there is no batch-submit step to drain.
    client.flush_no_sync().expect("cubecl client flush_no_sync failed before cusparse_backward_solve");

    // 5. Extract device pointers for a_values + b.
    // SAFETY: a_values_prim and b_prim must stay alive for the duration of
    // this function (they are parameters). cubecl's pending kernels have been
    // submitted to the stream (step 4); cuSPARSE reads them in order.
    let a_view = primitive_as_cuda_view::<B>(a_values_prim)
        .expect("cusparse_backward_solve: Cuda<f32, i32> backend required");
    let b_view = primitive_as_cuda_view::<B>(b_prim)
        .expect("cusparse_backward_solve: Cuda<f32, i32> backend required");

    let n = pattern.n;

    // 6. Allocate y via the cubecl client. The returned Handle owns the GPU
    //    buffer; cubecl frees it when the resulting CubeTensor is dropped.
    //    Zero-initialise so cuSPARSE can safely accumulate into y.
    let n_bytes = n * std::mem::size_of::<f32>();
    let y_bytes = vec![0u8; n_bytes];
    // SAFETY: create_from_slice uploads bytes to device and returns a Handle.
    // The Handle keeps the allocation alive for as long as we (or the resulting
    // CubeTensor) hold it.
    let y_handle: Handle = client.create_from_slice(&y_bytes);

    // SAFETY: y_handle is a freshly allocated, still-owned handle. We clone it
    // to keep the allocation alive while extracting the raw ptr; the original
    // y_handle is consumed later by CubeTensor::from_handle.
    let y_ptr = unsafe { handle_device_ptr(&client, &y_handle) } as *mut f32;

    // 7. cuSPARSE solve — same descriptor + updateMatrix dance as cusparse_forward,
    //    but using desc_backward and CUSPARSE_OPERATION_TRANSPOSE.
    unsafe {
        // 7a. Re-point sparse matrix descriptor at the current a_values.
        // SAFETY: a_view.ptr is the live device pointer of a_values_prim.
        // cache.d_crow and cache.d_col are cubecl Handles (Task 7); extract raw
        // device pointers via handle_device_ptr. The Handles stay alive through
        // `cache` (borrowed for the whole call).
        let crow_ptr = handle_device_ptr(&client, &cache.d_crow);
        let col_ptr  = handle_device_ptr(&client, &cache.d_col);
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            cache.sp_mat,
            crow_ptr as *mut std::ffi::c_void,
            col_ptr as *mut std::ffi::c_void,
            a_view.ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseCsrSetPointers backward failed");

        // 7b. Notify cuSPARSE that matrix values changed since analysis.
        // The SpSV descriptor caches values from analysis time (dummy 1.0s);
        // updateMatrix refreshes them so the solve uses the current a_values.
        // SAFETY: desc_backward was analyzed; sp_mat now has valid a_values ptr.
        cudarc::cusparse::sys::cusparseSpSV_updateMatrix(
            cache.handle,
            cache.desc_backward,
            a_view.ptr as *mut std::ffi::c_void,
            cudarc::cusparse::sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_GENERAL,
        )
        .result()
        .expect("cusparseSpSV_updateMatrix backward failed");

        // 7c. Build transient dense vector descriptors for b and y.
        // SAFETY: b_view.ptr and y_ptr are live device pointers of the
        // correct element count (n). They remain valid through the solve.
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

        // 7d. Execute the TRANSPOSE triangular solve: y = (A^T)^{-1} b.
        // SAFETY: desc_backward was pre-analyzed for CUSPARSE_OPERATION_TRANSPOSE.
        // The TRANSPOSE op flag directs cuSPARSE to treat sp_mat (lower-tri L)
        // as its transpose (upper-tri U) and solve U · y = b. The handle is
        // bound to cubecl's active stream (step 3), so the solve executes after
        // all upstream cubecl kernels on that stream.
        // cusparseSpSV_solve does NOT take a workspace arg — the workspace is
        // registered implicitly during cusparseSpSV_analysis.
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
        // NO cuStreamSynchronize — cubecl's next op on the same stream will
        // execute after the cuSPARSE solve automatically.

        // Clean up transient dense vector descriptors.
        // SAFETY: b_dn and y_dn are valid descriptors created above.
        cusparseDestroyDnVec(b_dn)
            .result()
            .expect("cusparseDestroyDnVec b failed in backward solve");
        cusparseDestroyDnVec(y_dn)
            .result()
            .expect("cusparseDestroyDnVec y failed in backward solve");
    }

    // 8. Wrap y_handle as CubeTensor → B::FloatTensorPrimitive.
    //    No host roundtrip. cubecl owns the y buffer via y_handle; it will be
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
        y_handle,
        DType::F32,
    );
    cube_tensor_to_primitive::<B>(cube)
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

// =================================================================================
// SP-9 Task 4: cusparse_spmv_forward — site 1, forward y = N · q via
// cusparseSpMV(NON_TRANSPOSE) on sp_mat_spmv. Stream-shared with cubecl,
// zero-copy.
// =================================================================================

/// Compute `y = N · q` via cuSPARSE SpMV (NON_TRANSPOSE). Returns the result
/// as a primitive tensor of shape `[n]`. No D↔H syncs — input and output
/// stay on device.
///
/// `cache` must come from `build_cuda_pattern_cache` for the matching
/// `CsrPattern`. `q_prim` is a `Tensor<Cuda<f32, i32>>::FloatTensorPrimitive`
/// of length `n`.
pub(crate) fn cusparse_spmv_forward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    q_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    use burn::tensor::{DType, Shape};
    use burn_cubecl::tensor::CubeTensor;
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseSpMV,
        cusparseSetStream,
    };

    // 1. Bind cuSPARSE to cubecl's active stream. cubecl queues kernels that
    //    write q onto this stream; cuSPARSE will run after them automatically
    //    because they share the same stream. No host sync needed.
    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        // SAFETY: `cudarc::driver::sys::CUstream_st` and
        // `cudarc::cusparse::sys::CUstream_st` are the same opaque CUDA ABI
        // type. cudarc generates separate FFI bindings; casting the pointer
        // integer is safe because no dereference occurs here.
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream forward SpMV failed");
    }

    // 2. Flush cubecl's kernel queue onto the active stream before cuSPARSE
    //    reads q. `client.flush()` is a non-blocking server-side operation: it
    //    submits pending work to the CUDA stream without blocking the host.
    //    cuSPARSE, running on the same stream, will execute after.
    let client = compute_client::<B>(device);
    // SP-10: flush_no_sync (see cusparse_backward_solve for rationale).
    client.flush_no_sync().expect("cubecl client flush_no_sync failed before cusparse_spmv_forward");

    // 3. Extract device pointer for q.
    // SAFETY: q_prim must stay alive for the duration of this function. cubecl's
    // pending kernels have been submitted to the stream (step 2); cuSPARSE reads
    // q in order.
    let q_view = primitive_as_cuda_view::<B>(q_prim)
        .expect("cusparse_spmv_forward: Cuda<f32, i32> backend required");

    let n = cache.n;

    // 4. Allocate y via the cubecl client. The returned Handle owns the GPU
    //    buffer; cubecl frees it when the resulting CubeTensor is dropped.
    //    Zero-initialise so cuSPARSE can safely accumulate into y.
    let n_bytes = n * std::mem::size_of::<f32>();
    let y_bytes = vec![0u8; n_bytes];
    // SAFETY: create_from_slice uploads bytes to device and returns a Handle.
    // The Handle keeps the allocation alive for as long as we (or the resulting
    // CubeTensor) hold it.
    let y_handle: burn_cubecl::cubecl::server::Handle = client.create_from_slice(&y_bytes);

    // SAFETY: y_handle is a freshly allocated, still-owned handle. The original
    // y_handle is consumed later by CubeTensor::from_handle.
    let y_ptr = unsafe { handle_device_ptr(&client, &y_handle) } as *mut f32;
    let workspace_ptr = unsafe { handle_device_ptr(&client, &cache.workspace_spmv_n) };

    // 5. Execute SpMV: y = 1.0 · N · q + 0.0 · y.
    unsafe {
        // 5a. Build transient dense vector descriptors for q and y.
        // SAFETY: q_view.ptr and y_ptr are live device pointers of the correct
        // element count (n). They remain valid through the SpMV.
        let mut q_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut y_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(
            &mut q_dn,
            n as i64,
            q_view.ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec q failed in forward SpMV");
        cusparseCreateDnVec(
            &mut y_dn,
            n as i64,
            y_ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec y failed in forward SpMV");

        // 5b. Execute: y = alpha * sp_mat_spmv * q + beta * y.
        // SAFETY: cache.sp_mat_spmv was built at cache-creation time with the
        // correct (n × n) shape and adj_values pointer. The handle is bound to
        // cubecl's active stream (step 1), so the SpMV executes after all
        // upstream cubecl kernels on that stream.
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        cusparseSpMV(
            cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const std::ffi::c_void,
            cache.sp_mat_spmv,
            q_dn,
            &beta as *const f32 as *const std::ffi::c_void,
            y_dn,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseSpMV forward failed");
        // NO cuStreamSynchronize — cubecl's next op on the same stream will
        // execute after the cuSPARSE SpMV automatically.

        // Clean up transient dense vector descriptors.
        // SAFETY: q_dn and y_dn are valid descriptors created above.
        cusparseDestroyDnVec(q_dn)
            .result()
            .expect("cusparseDestroyDnVec q failed in forward SpMV");
        cusparseDestroyDnVec(y_dn)
            .result()
            .expect("cusparseDestroyDnVec y failed in forward SpMV");
    }

    // 6. Wrap y_handle as CubeTensor → B::FloatTensorPrimitive.
    //    No host roundtrip. cubecl owns the y buffer via y_handle; it will be
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
        y_handle,
        DType::F32,
    );
    cube_tensor_to_primitive::<B>(cube)
}

// =================================================================================
// SP-9 Task 5: cusparse_spmv_backward — site 2, backward gq = N^T · gi via
// cusparseSpMV(TRANSPOSE) on sp_mat_spmv. Same matrix descriptor as forward;
// only the op flag and workspace differ.
// =================================================================================

/// Compute `gq = N^T · gi` via cuSPARSE SpMV with the TRANSPOSE op flag.
/// Returns the result as a primitive tensor of shape `[n]`. No D↔H syncs.
///
/// `cache` must come from `build_cuda_pattern_cache` for the matching
/// `CsrPattern`. `gi_prim` is a `Tensor<Cuda<f32, i32>>::FloatTensorPrimitive`
/// of length `n`.
pub(crate) fn cusparse_spmv_backward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    gi_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    use burn::tensor::{DType, Shape};
    use burn_cubecl::tensor::CubeTensor;
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseSpMV,
        cusparseSetStream,
    };

    // 1. Bind cuSPARSE to cubecl's active stream. cubecl queues kernels that
    //    write gi onto this stream; cuSPARSE will run after them automatically
    //    because they share the same stream. No host sync needed.
    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        // SAFETY: `cudarc::driver::sys::CUstream_st` and
        // `cudarc::cusparse::sys::CUstream_st` are the same opaque CUDA ABI
        // type. cudarc generates separate FFI bindings; casting the pointer
        // integer is safe because no dereference occurs here.
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream backward SpMV failed");
    }

    // 2. Flush cubecl's kernel queue onto the active stream before cuSPARSE
    //    reads gi. `client.flush()` is a non-blocking server-side operation: it
    //    submits pending work to the CUDA stream without blocking the host.
    //    cuSPARSE, running on the same stream, will execute after.
    let client = compute_client::<B>(device);
    // SP-10: flush_no_sync (see cusparse_backward_solve for rationale).
    client.flush_no_sync().expect("cubecl client flush_no_sync failed before cusparse_spmv_backward");

    // 3. Extract device pointer for gi.
    // SAFETY: gi_prim must stay alive for the duration of this function. cubecl's
    // pending kernels have been submitted to the stream (step 2); cuSPARSE reads
    // gi in order.
    let gi_view = primitive_as_cuda_view::<B>(gi_prim)
        .expect("cusparse_spmv_backward: Cuda<f32, i32> backend required");

    let n = cache.n;

    // 4. Allocate y via the cubecl client. The returned Handle owns the GPU
    //    buffer; cubecl frees it when the resulting CubeTensor is dropped.
    //    Zero-initialise so cuSPARSE can safely accumulate into y.
    let n_bytes = n * std::mem::size_of::<f32>();
    let y_bytes = vec![0u8; n_bytes];
    // SAFETY: create_from_slice uploads bytes to device and returns a Handle.
    // The Handle keeps the allocation alive for as long as we (or the resulting
    // CubeTensor) hold it.
    let y_handle: burn_cubecl::cubecl::server::Handle = client.create_from_slice(&y_bytes);

    // SAFETY: y_handle is a freshly allocated, still-owned handle. The original
    // y_handle is consumed later by CubeTensor::from_handle.
    let y_ptr = unsafe { handle_device_ptr(&client, &y_handle) } as *mut f32;
    let workspace_ptr = unsafe { handle_device_ptr(&client, &cache.workspace_spmv_nt) };

    // 5. Execute SpMV: y = 1.0 · N^T · gi + 0.0 · y.
    unsafe {
        // 5a. Build transient dense vector descriptors for gi and y.
        // SAFETY: gi_view.ptr and y_ptr are live device pointers of the correct
        // element count (n). They remain valid through the SpMV.
        let mut gi_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut y_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(
            &mut gi_dn,
            n as i64,
            gi_view.ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec gi failed in backward SpMV");
        cusparseCreateDnVec(
            &mut y_dn,
            n as i64,
            y_ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec y failed in backward SpMV");

        // 5b. Execute: y = alpha * sp_mat_spmv^T * gi + beta * y.
        // SAFETY: cache.sp_mat_spmv was built at cache-creation time with the
        // correct (n × n) shape and adj_values pointer. The handle is bound to
        // cubecl's active stream (step 1), so the SpMV executes after all
        // upstream cubecl kernels on that stream.
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        cusparseSpMV(
            cache.handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const f32 as *const std::ffi::c_void,
            cache.sp_mat_spmv,
            gi_dn,
            &beta as *const f32 as *const std::ffi::c_void,
            y_dn,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseSpMV backward failed");
        // NO cuStreamSynchronize — cubecl's next op on the same stream will
        // execute after the cuSPARSE SpMV automatically.

        // Clean up transient dense vector descriptors.
        // SAFETY: gi_dn and y_dn are valid descriptors created above.
        cusparseDestroyDnVec(gi_dn)
            .result()
            .expect("cusparseDestroyDnVec gi failed in backward SpMV");
        cusparseDestroyDnVec(y_dn)
            .result()
            .expect("cusparseDestroyDnVec y failed in backward SpMV");
    }

    // 6. Wrap y_handle as CubeTensor → B::FloatTensorPrimitive.
    //    No host roundtrip. cubecl owns the y buffer via y_handle; it will be
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
        y_handle,
        DType::F32,
    );
    cube_tensor_to_primitive::<B>(cube)
}

// =================================================================================
// SP-9 Task 6: cusparse_assemble_backward — site 3, per-row sum
// gc = -sp_mat_rowsum · gA via cusparseSpMV(NON_TRANSPOSE) on sp_mat_rowsum
// with α=-1 (embedded negation, no separate kernel launch).
// =================================================================================

/// Compute `gc[i] = -Σ_k adj[k] · gA[k]` for `k in row i` via cuSPARSE SpMV
/// with α=-1. Returns the result as a primitive tensor of shape `[n]`.
/// No D↔H syncs.
///
/// `gA_prim` has length `nnz` (one entry per off-diagonal CSR position).
/// Output `gc` has length `n` (one entry per reach).
pub(crate) fn cusparse_assemble_backward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    g_a_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive
where
    B::FloatTensorPrimitive: 'static,
    B::Device: 'static,
{
    use burn::tensor::{DType, Shape};
    use burn_cubecl::tensor::CubeTensor;
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseSpMV,
        cusparseSetStream,
    };

    // 1. Bind cuSPARSE to cubecl's active stream. cubecl queues kernels that
    //    write gA onto this stream; cuSPARSE will run after them automatically
    //    because they share the same stream. No host sync needed.
    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        // SAFETY: `cudarc::driver::sys::CUstream_st` and
        // `cudarc::cusparse::sys::CUstream_st` are the same opaque CUDA ABI
        // type. cudarc generates separate FFI bindings; casting the pointer
        // integer is safe because no dereference occurs here.
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream (assemble backward) failed");
    }

    // 2. Flush cubecl's kernel queue onto the active stream before cuSPARSE
    //    reads gA. `client.flush()` is a non-blocking server-side operation: it
    //    submits pending work to the CUDA stream without blocking the host.
    //    cuSPARSE, running on the same stream, will execute after.
    let client = compute_client::<B>(device);
    // SP-10: flush_no_sync (see cusparse_backward_solve for rationale).
    client.flush_no_sync().expect("cubecl client flush_no_sync failed before cusparse_assemble_backward");

    // 3. Extract device pointer for gA.
    // SAFETY: g_a_prim must stay alive for the duration of this function. cubecl's
    // pending kernels have been submitted to the stream (step 2); cuSPARSE reads
    // gA in order.
    let ga_view = primitive_as_cuda_view::<B>(g_a_prim)
        .expect("cusparse_assemble_backward: Cuda<f32, i32> backend required");

    let n = cache.n;
    let nnz = cache.nnz;

    // 4. Allocate gc via the cubecl client. The returned Handle owns the GPU
    //    buffer; cubecl frees it when the resulting CubeTensor is dropped.
    //    Zero-initialise so cuSPARSE can safely accumulate into gc.
    let n_bytes = n * std::mem::size_of::<f32>();
    let gc_bytes = vec![0u8; n_bytes];
    // SAFETY: create_from_slice uploads bytes to device and returns a Handle.
    // The Handle keeps the allocation alive for as long as we (or the resulting
    // CubeTensor) hold it.
    let gc_handle: burn_cubecl::cubecl::server::Handle = client.create_from_slice(&gc_bytes);

    // SAFETY: gc_handle is a freshly allocated, still-owned handle. The original
    // gc_handle is consumed later by CubeTensor::from_handle.
    let gc_ptr = unsafe { handle_device_ptr(&client, &gc_handle) } as *mut f32;
    let workspace_ptr = unsafe { handle_device_ptr(&client, &cache.workspace_rowsum) };

    // 5. Execute SpMV: gc = -1.0 · sp_mat_rowsum · gA + 0.0 · gc.
    unsafe {
        // 5a. Build transient dense vector descriptors for gA and gc.
        // SAFETY: ga_view.ptr is a live device pointer with nnz elements.
        // gc_ptr is a live device pointer with n elements.
        // Both remain valid through the SpMV.
        let mut ga_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut gc_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(
            &mut ga_dn,
            nnz as i64,
            ga_view.ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec gA failed in (assemble backward)");
        cusparseCreateDnVec(
            &mut gc_dn,
            n as i64,
            gc_ptr as *mut std::ffi::c_void,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateDnVec gc failed in (assemble backward)");

        // 5b. Execute: gc = alpha * sp_mat_rowsum * gA + beta * gc.
        // SAFETY: cache.sp_mat_rowsum was built at cache-creation time with the
        // correct (n × nnz) shape and adj_values pointer. The handle is bound to
        // cubecl's active stream (step 1), so the SpMV executes after all
        // upstream cubecl kernels on that stream.
        let alpha: f32 = -1.0;
        let beta: f32 = 0.0;
        cusparseSpMV(
            cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const std::ffi::c_void,
            cache.sp_mat_rowsum,
            ga_dn,
            &beta as *const f32 as *const std::ffi::c_void,
            gc_dn,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseSpMV (assemble backward) failed");
        // NO cuStreamSynchronize — cubecl's next op on the same stream will
        // execute after the cuSPARSE SpMV automatically.

        // Clean up transient dense vector descriptors.
        // SAFETY: ga_dn and gc_dn are valid descriptors created above.
        cusparseDestroyDnVec(ga_dn)
            .result()
            .expect("cusparseDestroyDnVec gA failed in (assemble backward)");
        cusparseDestroyDnVec(gc_dn)
            .result()
            .expect("cusparseDestroyDnVec gc failed in (assemble backward)");
    }

    // 6. Wrap gc_handle as CubeTensor → B::FloatTensorPrimitive.
    //    No host roundtrip. cubecl owns the gc buffer via gc_handle; it will be
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
        gc_handle,
        DType::F32,
    );
    cube_tensor_to_primitive::<B>(cube)
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
pub(crate) fn compute_client<B: Backend + 'static>(
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
/// thread with `&mut` access. Replaces SP-6's dedicated stream fallback.
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
pub fn __spike_compute_client<B: Backend + 'static>(
    device: &B::Device,
) -> cubecl::client::ComputeClient<CudaRuntime> {
    compute_client::<B>(device)
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

// ---------------------------------------------------------------------------
// CudaPatternCache — SP-6 Task 8 + Task 9
// ---------------------------------------------------------------------------

use std::marker::PhantomData;

/// SP-10 observability. Surfaces whether capture succeeded; if not, the
/// reason so training-log greps catch silent fallbacks.
#[derive(Debug, Clone)]
pub(crate) enum CaptureStatus {
    NotAttempted,
    Captured,
    FallbackReason(String),
}

/// Per-pattern cuSPARSE state. Built lazily on first GPU solve call.
///
/// !Send because cuSPARSE descriptors and CUDA contexts are tied to the
/// thread that created them. Single-threaded training is the only supported
/// mode for SP-6/SP-7.
///
/// Device allocations are cubecl `Handle`s. cubecl owns their lifetimes and
/// frees them when the Handles drop. Drop only needs to destroy cuSPARSE
/// descriptors; no `cuMemFreeAsync` is needed here.
pub(crate) struct CudaPatternCache {
    pub(crate) handle: cudarc::cusparse::sys::cusparseHandle_t,
    /// cubecl Handle for crow (i32 array, len = n+1).
    pub(crate) d_crow: burn_cubecl::cubecl::server::Handle,
    /// cubecl Handle for col (i32 array, len = nnz).
    pub(crate) d_col: burn_cubecl::cubecl::server::Handle,
    /// Held to keep the row-index buffer alive (cusparseSpMat references it
    /// via raw pointer). Never read directly — cubecl frees on Drop.
    #[allow(dead_code)]
    pub(crate) d_row_for_nnz: burn_cubecl::cubecl::server::Handle,
    /// Persistent device buffer of CsrPattern.adj_values (length nnz, f32).
    /// SP-9: shared by sp_mat_spmv and sp_mat_rowsum SpMV descriptors.
    pub(crate) d_adj_values: burn_cubecl::cubecl::server::Handle,
    pub(crate) sp_mat: cudarc::cusparse::sys::cusparseSpMatDescr_t,
    pub(crate) desc_forward: cudarc::cusparse::sys::cusparseSpSVDescr_t,
    pub(crate) desc_backward: cudarc::cusparse::sys::cusparseSpSVDescr_t,
    /// Held for cuSPARSE workspace lifetime; cubecl frees on Drop.
    #[allow(dead_code)]
    pub(crate) workspace_forward: burn_cubecl::cubecl::server::Handle,
    /// Held for cuSPARSE workspace lifetime; cubecl frees on Drop.
    #[allow(dead_code)]
    pub(crate) workspace_backward: burn_cubecl::cubecl::server::Handle,
    // ── SP-9 SpMV ─────────────────────────────────────────────────
    /// `(n × n)` cuSPARSE matrix descriptor for SpMV (values = adj). Used for
    /// site 1 (forward `y = N · q`) and site 2 (backward `gq = N^T · gi`).
    pub(crate) sp_mat_spmv: cudarc::cusparse::sys::cusparseSpMatDescr_t,
    /// `(n × nnz)` cuSPARSE matrix descriptor for site 3 row-sum
    /// (`gc = α · sp_mat_rowsum · gA`, with α=-1).
    pub(crate) sp_mat_rowsum: cudarc::cusparse::sys::cusparseSpMatDescr_t,
    /// `[0, 1, 2, ..., nnz-1]` i32 indices — col-index array for `sp_mat_rowsum`.
    pub(crate) d_col_identity: burn_cubecl::cubecl::server::Handle,
    /// Workspace for `cusparseSpMV(NON_TRANSPOSE, sp_mat_spmv, ...)`.
    pub(crate) workspace_spmv_n: burn_cubecl::cubecl::server::Handle,
    /// Workspace for `cusparseSpMV(TRANSPOSE, sp_mat_spmv, ...)`.
    pub(crate) workspace_spmv_nt: burn_cubecl::cubecl::server::Handle,
    /// Workspace for `cusparseSpMV(NON_TRANSPOSE, sp_mat_rowsum, ...)`.
    pub(crate) workspace_rowsum: burn_cubecl::cubecl::server::Handle,
    // ── SP-10 CUDA Graphs ─────────────────────────────────────────────
    // NB: declaration order matters — Rust drops struct fields in
    // declaration order. `graph_fwd` and `graph_bwd` must be declared
    // BEFORE `scratch` so the captured CUgraphExec objects (which hold
    // pointers into scratch buffers) are destroyed first. Drop::drop
    // also explicitly sets the graphs to None at its top, but the
    // declaration order is the backstop.
    /// Forward-direction captured CUDA graph. `None` if capture failed
    /// or `params.use_cuda_graphs == false`.
    pub(crate) graph_fwd: Option<crate::cuda_graph::CudaGraph>,
    /// Backward-direction captured CUDA graph. Same fallback semantics.
    pub(crate) graph_bwd: Option<crate::cuda_graph::CudaGraph>,
    /// Persistent per-instance scratch buffers. Allocated in
    /// `MuskingumCunge::setup_inputs`; consumed by both forward and
    /// backward graph capture/replay.
    pub(crate) scratch: Option<crate::cuda_graph::PersistentScratch>,
    /// SP-10: pinned primitives from the capture pass of
    /// `forward_chain_inner_pinned`. Keeps every intermediate cubecl Handle
    /// alive for the cache's lifetime so the persistent pool never recycles
    /// their device addresses — which would otherwise invalidate the captured
    /// CUDA graph and cause `CUDA_ERROR_ILLEGAL_ADDRESS` on the second
    /// replay.
    ///
    /// Type-erased via `Box<dyn Any + Send>` because `CudaPatternCache` is
    /// non-generic but the primitives are `I::FloatTensorPrimitive`. The
    /// capture-pass code stores the concrete type; the cache simply keeps
    /// them alive until Drop.
    ///
    /// Memory footprint: ~65 × n × 4 bytes per cache instance (~1.3 MB at
    /// n=5K, ~90 MB at full CONUS n=346,321). Drops when the cache drops.
    ///
    /// Declared AFTER `graph_fwd` / `graph_bwd` so Rust's field-drop order
    /// destroys the captured graphs BEFORE freeing the handles they
    /// reference. `Drop::drop` also explicitly nulls this field after the
    /// graphs as a belt-and-braces.
    pub(crate) pinned_intermediates: Option<Vec<Box<dyn std::any::Any + Send>>>,
    /// `(n_segments, sparse_solver_kind)` signature at capture time. If
    /// this changes between batches, drop both graphs and recapture.
    pub(crate) capture_sig: Option<(usize, crate::config::SparseSolver)>,
    /// Observability: surfaces silent fallbacks. Logged at end of
    /// setup_inputs.
    pub(crate) capture_status: CaptureStatus,
    /// Number of reaches (rows = cols of the square network matrix).
    pub(crate) n: usize,
    /// Number of non-zeros in the network adjacency.
    pub(crate) nnz: usize,
    /// `!Send` marker — cuSPARSE descriptors are thread-bound.
    _not_send: PhantomData<*mut ()>,
}

impl Drop for CudaPatternCache {
    fn drop(&mut self) {
        // SP-10: drop captured graphs first. CudaGraph::drop destroys the
        // CUgraphExec then the CUgraph template; both reference scratch
        // device pointers internally (baked into the exec by CUDA, not
        // tracked by Rust). Field-declaration order already places
        // graph_fwd/graph_bwd before scratch, but assigning None here is
        // an explicit belt-and-braces that runs before any other Drop
        // logic below.
        self.graph_fwd = None;
        self.graph_bwd = None;
        // SP-10 Task D: drop the pinned capture-pass intermediates AFTER the
        // graphs so the captured CUgraphExec is destroyed before the
        // referenced cubecl Handles are freed.
        self.pinned_intermediates = None;
        // `scratch` (Option<PersistentScratch>) is dropped automatically
        // in struct-field declaration order, after the cuSPARSE
        // descriptor teardown below.
        // SAFETY: cuSPARSE descriptors are destroyed before the Handle fields
        // drop (struct field declaration order: handle → d_crow → d_col →
        // d_row_for_nnz → d_adj_values → sp_mat → desc_forward → desc_backward →
        // workspace_forward → workspace_backward → sp_mat_spmv → sp_mat_rowsum →
        // d_col_identity → workspace_spmv_n → workspace_spmv_nt → workspace_rowsum).
        // cubecl-owned Handles release device buffers via their own Drop impl —
        // no cuMemFreeAsync needed.
        unsafe {
            // Destroy new SpMV descriptors (Task 2 — may be null placeholders).
            if !self.sp_mat_spmv.is_null() {
                cudarc::cusparse::sys::cusparseDestroySpMat(self.sp_mat_spmv)
                    .result()
                    .expect("cusparseDestroySpMat sp_mat_spmv failed");
            }
            if !self.sp_mat_rowsum.is_null() {
                cudarc::cusparse::sys::cusparseDestroySpMat(self.sp_mat_rowsum)
                    .result()
                    .expect("cusparseDestroySpMat sp_mat_rowsum failed");
            }
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
            // The nine Handle fields (d_crow, d_col, d_row_for_nnz, d_adj_values,
            // workspace_forward, workspace_backward, d_col_identity,
            // workspace_spmv_n, workspace_spmv_nt, workspace_rowsum) drop
            // automatically after this block and free their device allocations
            // via cubecl's Handle Drop impl. No explicit cuMemFreeAsync is needed.
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

/// Allocate device memory + create cuSPARSE descriptors for this pattern.
///
/// SP-7 Task 7 version — all device buffers are cubecl `Handle`s allocated
/// via `client.create_from_slice`. No `cuMemAllocAsync`, no `cuMemFreeAsync`.
///
/// This runs once per `CsrPattern` lifetime and performs:
/// 1. Materialise a cubecl client for `Cuda<f32, i32>` (default device).
/// 2. Upload crow / col / row_for_nnz via `client.create_from_slice`.
/// 3. `client.flush()` to submit writes onto the shared stream.
/// 4. Create cuSPARSE handle; bind it to cubecl's active stream.
/// 5. Create sparse matrix descriptor (values=NULL, set per-solve).
/// 6. Set fill mode (lower) + diag type (non-unit).
/// 7. Create SpSV descriptors (forward + backward).
/// 8. Probe workspace sizes (with dummy b/x Handles).
/// 9. Allocate workspaces + dummy values via `create_from_slice`.
/// 10. Run `cusparseSpSV_analysis` for both directions.
/// 11. Return `CudaPatternCache`; dummy Handles drop and free device memory.
fn build_cuda_pattern_cache(pattern: &crate::sparse::CsrPattern) -> CudaPatternCache {
    use burn_cubecl::cubecl::server::Handle;
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateCsr,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT,
        cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER,
        cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        cusparseIndexType_t::CUSPARSE_INDEX_32I,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        cusparseSpMatAttribute_t::{CUSPARSE_SPMAT_DIAG_TYPE, CUSPARSE_SPMAT_FILL_MODE},
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseSpMV_bufferSize,
        cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        cusparseSpSV_analysis,
        cusparseSpSV_bufferSize,
        cusparseSpSV_createDescr,
        cusparseSpMatSetAttribute,
    };

    // --- Step 1: Materialise the cubecl client. ---
    // We are guaranteed to be on Cuda<f32, i32> here (the dispatcher gates the
    // call). Default device is device index 0.
    // Note: burn::backend::Cuda is gated behind the "cuda" feature on the burn
    // umbrella crate (not enabled); use burn_cuda::Cuda directly instead.
    type B = burn_cuda::Cuda<f32, i32>;
    let device: cubecl::cuda::CudaDevice = Default::default();
    let client = compute_client::<B>(&device);

    // --- Step 2: Upload structural arrays via cubecl. ---
    // `create_from_slice` allocates device memory and copies bytes to it.
    // SAFETY: i32 slices reinterpreted as u8 bytes; alignment/size are valid.

    let crow_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pattern.crow.as_ptr() as *const u8,
            pattern.crow.len() * std::mem::size_of::<i32>(),
        )
    };
    let d_crow: Handle = client.create_from_slice(crow_bytes);

    let col_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pattern.col.as_ptr() as *const u8,
            pattern.col.len() * std::mem::size_of::<i32>(),
        )
    };
    let d_col: Handle = client.create_from_slice(col_bytes);

    let row_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pattern.row_for_nnz.as_ptr() as *const u8,
            pattern.row_for_nnz.len() * std::mem::size_of::<i32>(),
        )
    };
    let d_row_for_nnz: Handle = client.create_from_slice(row_bytes);

    // SP-9: Upload adj_values as a persistent device buffer. SpMV descriptors
    // (Tasks 3-7) will point at this buffer for the cache's lifetime.
    let adj_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pattern.adj_values.as_ptr() as *const u8,
            pattern.adj_values.len() * std::mem::size_of::<f32>(),
        )
    };
    let d_adj_values: Handle = client.create_from_slice(adj_bytes);

    // --- Step 3: Flush so writes hit the stream cuSPARSE will be using. ---
    client.flush().expect("cubecl flush after crow/col/row upload");

    // --- Step 4: Create cuSPARSE handle + bind to cubecl's active stream. ---
    // SAFETY: cusparseCreate doesn't require a current CUDA context in CUDA 12+.
    let handle = cudarc::cusparse::result::create()
        .expect("cusparseCreate failed — CUDA setup broken");

    let stream = cubecl_stream_active::<B>(&device);
    unsafe {
        // SAFETY: `cudarc::driver::sys::CUstream_st` and
        // `cudarc::cusparse::sys::CUstream_st` are the same opaque CUDA ABI
        // type. cudarc generates separate FFI bindings; casting the pointer
        // integer is safe because no dereference occurs here.
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cudarc::cusparse::sys::cusparseSetStream(handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream (build_cuda_pattern_cache) failed");
    }

    let n = pattern.n as i64;
    let nnz = pattern.col.len() as i64;

    // Extract raw device pointers from the three structural Handles.
    // SAFETY: Handles are alive for the rest of this function; pointers are valid.
    let crow_ptr = unsafe { handle_device_ptr(&client, &d_crow) };
    let col_ptr  = unsafe { handle_device_ptr(&client, &d_col) };

    // --- Step 5: Create sparse matrix descriptor (values = NULL). ---
    // Values will be set per-call via cusparseCsrSetPointers in cusparse_forward.
    let mut sp_mat: cudarc::cusparse::sys::cusparseSpMatDescr_t = std::ptr::null_mut();
    unsafe {
        // SAFETY: crow_ptr / col_ptr are valid device pointers for i32 arrays
        // of length n+1 and nnz respectively.
        cusparseCreateCsr(
            &mut sp_mat,
            n,
            n,
            nnz,
            crow_ptr as *mut std::ffi::c_void,
            col_ptr as *mut std::ffi::c_void,
            std::ptr::null_mut(), // values — set per-call
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateCsr failed");

        // --- Step 6: Set fill mode (lower triangular) and diag type (non-unit). ---
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
    }

    // --- Step 7: Create SpSV descriptors for forward and backward. ---
    let mut desc_forward: cudarc::cusparse::sys::cusparseSpSVDescr_t = std::ptr::null_mut();
    let mut desc_backward: cudarc::cusparse::sys::cusparseSpSVDescr_t = std::ptr::null_mut();
    unsafe {
        // SAFETY: cusparseSpSV_createDescr initialises an opaque descriptor;
        // no CUDA context is required.
        cusparseSpSV_createDescr(&mut desc_forward)
            .result()
            .expect("cusparseSpSV_createDescr (forward) failed");
        cusparseSpSV_createDescr(&mut desc_backward)
            .result()
            .expect("cusparseSpSV_createDescr (backward) failed");
    }

    // --- Step 8: Probe workspace sizes with dummy b/x Handles. ---
    // cusparseSpSV_bufferSize requires valid (non-NULL) dense vector pointers.
    // Allocate zero-filled dummy buffers via cubecl.
    let dummy_b: Handle = client.create_from_slice(&vec![0u8; pattern.n * std::mem::size_of::<f32>()]);
    let dummy_x: Handle = client.create_from_slice(&vec![0u8; pattern.n * std::mem::size_of::<f32>()]);

    // Upload dummy values (1.0f32) for the analysis step; cusparseSpSV_analysis
    // requires a non-NULL values pointer — NULL values is only allowed at
    // descriptor creation time.
    let nnz_usize = pattern.col.len();
    let ones: Vec<f32> = vec![1.0f32; nnz_usize];
    let dummy_vals_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(ones.as_ptr() as *const u8, nnz_usize * std::mem::size_of::<f32>())
    };
    let dummy_vals: Handle = client.create_from_slice(dummy_vals_bytes);

    // Flush so the dummy uploads land on the stream before cuSPARSE reads them.
    client.flush().expect("cubecl flush after dummy uploads");

    let dummy_b_ptr   = unsafe { handle_device_ptr(&client, &dummy_b) };
    let dummy_x_ptr   = unsafe { handle_device_ptr(&client, &dummy_x) };
    let dummy_a_ptr   = unsafe { handle_device_ptr(&client, &dummy_vals) };

    let alpha: f32 = 1.0;
    let mut fwd_buf_sz: usize = 0;
    let mut bwd_buf_sz: usize = 0;

    unsafe {
        // Set values pointer to dummy for buffer-size query.
        // SAFETY: dummy_a_ptr is a valid f32 device array of length nnz.
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            sp_mat,
            crow_ptr as *mut std::ffi::c_void,
            col_ptr as *mut std::ffi::c_void,
            dummy_a_ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseCsrSetPointers (analysis dummy) failed");

        let mut b_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut x_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        // SAFETY: dummy_b_ptr / dummy_x_ptr are valid f32 device arrays of length n.
        cusparseCreateDnVec(&mut b_dn, n, dummy_b_ptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec b (bufferSize) failed");
        cusparseCreateDnVec(&mut x_dn, n, dummy_x_ptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec x (bufferSize) failed");

        // --- Forward: NON_TRANSPOSE (lower-tri solve L \ b). ---
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

        cusparseDestroyDnVec(b_dn)
            .result()
            .expect("cusparseDestroyDnVec b (bufferSize) failed");
        cusparseDestroyDnVec(x_dn)
            .result()
            .expect("cusparseDestroyDnVec x (bufferSize) failed");
    }

    // --- Step 9: Allocate workspaces. Minimum 1 byte to avoid zero-size alloc. ---
    let workspace_forward:  Handle = client.create_from_slice(&vec![0u8; fwd_buf_sz.max(1)]);
    let workspace_backward: Handle = client.create_from_slice(&vec![0u8; bwd_buf_sz.max(1)]);
    let ws_fwd_ptr = unsafe { handle_device_ptr(&client, &workspace_forward) };
    let ws_bwd_ptr = unsafe { handle_device_ptr(&client, &workspace_backward) };

    // Flush workspace allocations onto the stream before analysis.
    client.flush().expect("cubecl flush before cusparseSpSV_analysis");

    // --- Step 10: Run cusparseSpSV_analysis for forward + backward. ---
    unsafe {
        // Re-create dense vector descriptors for analysis (the bufferSize ones were destroyed).
        let mut b_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut x_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        // SAFETY: dummy_b_ptr / dummy_x_ptr are still valid (Handles alive).
        cusparseCreateDnVec(&mut b_dn, n, dummy_b_ptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec b (analysis) failed");
        cusparseCreateDnVec(&mut x_dn, n, dummy_x_ptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec x (analysis) failed");

        // SAFETY: sp_mat has valid values (dummy_a_ptr set above). Analysis reads
        // structure and optionally values depending on the algorithm.
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
            ws_fwd_ptr as *mut std::ffi::c_void,
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
            ws_bwd_ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseSpSV_analysis (backward) failed");

        cusparseDestroyDnVec(b_dn)
            .result()
            .expect("cusparseDestroyDnVec b (analysis) failed");
        cusparseDestroyDnVec(x_dn)
            .result()
            .expect("cusparseDestroyDnVec x (analysis) failed");
        // dummy_vals, dummy_b, dummy_x drop here — cubecl frees the device buffers.
    }

    // --- Step 11: SP-9 Task 3 — create SpMV descriptors + workspaces. ---

    // Upload identity column indices [0, 1, ..., nnz-1] for sp_mat_rowsum.
    // sp_mat_rowsum is an (n × nnz) CSR matrix where each stored entry j
    // lives in column j — so col_ind[j] = j for all j in [0, nnz).
    let nnz_usize = pattern.col.len();
    let col_identity_vec: Vec<i32> = (0..nnz_usize as i32).collect();
    let col_identity_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            col_identity_vec.as_ptr() as *const u8,
            col_identity_vec.len() * std::mem::size_of::<i32>(),
        )
    };
    let d_col_identity: Handle = client.create_from_slice(col_identity_bytes);

    // Flush so d_col_identity lands on the stream before cuSPARSE reads it.
    client.flush().expect("cubecl flush after d_col_identity upload");

    let adj_ptr_spmv = unsafe { handle_device_ptr(&client, &d_adj_values) };
    let col_id_ptr   = unsafe { handle_device_ptr(&client, &d_col_identity) };

    // Create sp_mat_spmv: same (n × n) shape as sp_mat (used by SpSV) but a
    // separate descriptor so SpMV and SpSV calls don't share state.
    // Values point at d_adj_values (persistent cache field).
    let mut sp_mat_spmv: cudarc::cusparse::sys::cusparseSpMatDescr_t = std::ptr::null_mut();
    unsafe {
        cusparseCreateCsr(
            &mut sp_mat_spmv,
            n,                                        // rows
            n,                                        // cols
            nnz,                                      // nnz
            crow_ptr as *mut std::ffi::c_void,
            col_ptr as *mut std::ffi::c_void,
            adj_ptr_spmv as *mut std::ffi::c_void,   // values = adj_values
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateCsr sp_mat_spmv failed");
    }

    // Create sp_mat_rowsum: (n × nnz) CSR where row pointers are the SAME
    // crow as sp_mat_spmv and col_ind is d_col_identity. Each edge e in
    // [0, nnz) is stored in column e, so row-sum via SpMV sums adj_values
    // per upstream reach — used in the dq/dN gradient assembly (SP-9 Task 6).
    let mut sp_mat_rowsum: cudarc::cusparse::sys::cusparseSpMatDescr_t = std::ptr::null_mut();
    unsafe {
        cusparseCreateCsr(
            &mut sp_mat_rowsum,
            n,                                        // rows
            nnz,                                      // cols = nnz
            nnz,                                      // total nnz of this matrix
            crow_ptr as *mut std::ffi::c_void,        // same crow as sp_mat_spmv
            col_id_ptr as *mut std::ffi::c_void,      // identity col indices
            adj_ptr_spmv as *mut std::ffi::c_void,    // same adj_values
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateCsr sp_mat_rowsum failed");
    }

    // Query workspace sizes for the three SpMV configurations.
    // cusparseCreateDnVec requires valid device pointers — allocate real
    // device buffers sized to the largest of (n, nnz) elements as f32.
    // These are temporary and drop at the end of this block.
    let max_len = (nnz_usize).max(pattern.n);
    let dummy_vec_h: Handle = client.create_from_slice(
        &vec![0u8; max_len * std::mem::size_of::<f32>()]);
    client.flush().expect("cubecl flush after dummy_vec_h");
    let dummy_vec_ptr = unsafe { handle_device_ptr(&client, &dummy_vec_h) }
        as *mut std::ffi::c_void;

    // The closure captures `handle`, `dummy_vec_ptr`, and the cuSPARSE
    // type aliases via the use block above.
    let query_workspace_bytes = |
        sp_mat_q: cudarc::cusparse::sys::cusparseSpMatDescr_t,
        op: cudarc::cusparse::sys::cusparseOperation_t,
        input_len: i64,
        output_len: i64,
    | -> usize {
        let mut dnvec_in:  cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut dnvec_out: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let alpha: f32 = 1.0;
        let beta:  f32 = 0.0;
        let mut buf_bytes: usize = 0;
        unsafe {
            // Both descriptors point into the same dummy buffer — cusparseSpMV_bufferSize
            // only inspects shapes, not memory contents.
            cusparseCreateDnVec(&mut dnvec_in,  input_len,  dummy_vec_ptr, CUDA_R_32F)
                .result().expect("cusparseCreateDnVec in (SpMV workspace query)");
            cusparseCreateDnVec(&mut dnvec_out, output_len, dummy_vec_ptr, CUDA_R_32F)
                .result().expect("cusparseCreateDnVec out (SpMV workspace query)");
            cusparseSpMV_bufferSize(
                handle,
                op,
                &alpha as *const f32 as *const std::ffi::c_void,
                sp_mat_q,
                dnvec_in,
                &beta as *const f32 as *const std::ffi::c_void,
                dnvec_out,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &mut buf_bytes,
            )
            .result().expect("cusparseSpMV_bufferSize");
            cusparseDestroyDnVec(dnvec_in)
                .result().expect("cusparseDestroyDnVec in (SpMV workspace query)");
            cusparseDestroyDnVec(dnvec_out)
                .result().expect("cusparseDestroyDnVec out (SpMV workspace query)");
        }
        buf_bytes.max(1)  // cuSPARSE may return 0 — always allocate at least 1 byte
    };

    let bytes_spmv_n  = query_workspace_bytes(
        sp_mat_spmv, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n);
    let bytes_spmv_nt = query_workspace_bytes(
        sp_mat_spmv, CUSPARSE_OPERATION_TRANSPOSE,     n, n);
    let bytes_rowsum  = query_workspace_bytes(
        sp_mat_rowsum, CUSPARSE_OPERATION_NON_TRANSPOSE, nnz, n);
    // dummy_vec_h drops here, freeing the temporary device buffer.

    let workspace_spmv_n:  Handle = client.create_from_slice(&vec![0u8; bytes_spmv_n]);
    let workspace_spmv_nt: Handle = client.create_from_slice(&vec![0u8; bytes_spmv_nt]);
    let workspace_rowsum:  Handle = client.create_from_slice(&vec![0u8; bytes_rowsum]);

    CudaPatternCache {
        handle,
        d_crow,
        d_col,
        d_row_for_nnz,
        d_adj_values,
        sp_mat,
        desc_forward,
        desc_backward,
        workspace_forward,
        workspace_backward,
        sp_mat_spmv,
        sp_mat_rowsum,
        d_col_identity,
        workspace_spmv_n,
        workspace_spmv_nt,
        workspace_rowsum,
        // SP-10: graphs + scratch start unpopulated; setup_inputs fills
        // them after the cache is built when use_cuda_graphs is on.
        graph_fwd: None,
        graph_bwd: None,
        scratch: None,
        pinned_intermediates: None,
        capture_sig: None,
        capture_status: CaptureStatus::NotAttempted,
        n: pattern.n,
        nnz: pattern.col.len(),
        _not_send: PhantomData,
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
/// - Binds cuSPARSE to cubecl's active CUDA stream.
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

    // 1. Lazy-build the pattern cache (one-time per CsrPattern).
    //    Task 7: build_cuda_pattern_cache now uses cubecl Handles internally;
    // SAFETY: SP-7 single-threaded training contract — no concurrent access to
    // cuda_cache from multiple threads.
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
    // SP-10: flush_no_sync (see cusparse_backward_solve for rationale).
    // CRITICAL for graph capture: this call must NOT cuEventSynchronize,
    // because cusparse_forward runs inside the cuStreamBeginCapture
    // closure in try_capture_forward.
    client.flush_no_sync().expect("cubecl client flush_no_sync failed before cusparse_forward");

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
        // cache.d_crow and cache.d_col are cubecl Handles (Task 7); extract raw
        // device pointers via handle_device_ptr. The Handles stay alive for the
        // duration of this function through `cache` (borrowed for the whole call).
        let crow_ptr = handle_device_ptr(&client, &cache.d_crow);
        let col_ptr  = handle_device_ptr(&client, &cache.d_col);
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            cache.sp_mat,
            crow_ptr as *mut std::ffi::c_void,
            col_ptr as *mut std::ffi::c_void,
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

// ---------------------------------------------------------------------------
// SP-10 Phase 3: cuSPARSE variants that write into caller-owned dst Handles.
//
// The default `cusparse_spmv_forward` / `cusparse_forward` allocate a fresh
// output Handle via the cubecl client each call. That allocation is illegal
// inside `cuStreamBeginCapture` (the cubecl pool would record an alloc node
// in the graph). These variants take a pre-allocated dst Handle (typically
// pointing into `PersistentScratch`) and only enqueue stream-ordered work.
// ---------------------------------------------------------------------------

/// SP-10 Phase 3: `y = N · q` via cuSPARSE SpMV(NON_TRANSPOSE), writing into
/// a caller-owned dst Handle. No allocations on the cubecl pool.
///
/// `q_devptr` is the source device pointer (e.g. the devptr of
/// `scratch.in_q`); `dst_devptr` is the destination device pointer (e.g.
/// the devptr of `scratch.state_i_t`). The caller pre-resolves these so we
/// can run the entire body inside the capture region without touching the
/// cubecl Handle table.
///
/// # Safety
/// - `q_devptr` and `dst_devptr` must be valid CUDA device pointers of at
///   least `cache.n` f32 elements, allocated on cubecl's primary stream pool.
/// - The caller has bound cubecl's primary CUDA context on the current
///   thread (see `try_capture_forward` for the canonical pattern).
/// - This function may be called inside `cuStreamBeginCapture` — it does
///   no host-sync.
pub(crate) unsafe fn cusparse_spmv_into_devptrs(
    cache: &CudaPatternCache,
    client: &cubecl::client::ComputeClient<CudaRuntime>,
    q_devptr: u64,
    dst_devptr: u64,
    stream: cudarc::driver::sys::CUstream,
) {
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseSpMV,
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseSetStream,
    };

    // Bind cuSPARSE to the captured stream.
    unsafe {
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream forward SpMV (into_devptrs) failed");
    }

    // flush_no_sync is safe inside capture: it submits prior cubecl work
    // without cuEventSynchronize. Here our K1 already submitted directly to
    // the captured stream, so nothing to drain — but keep flush_no_sync for
    // consistency with the other cuSPARSE call sites.
    client
        .flush_no_sync()
        .expect("cubecl client flush_no_sync failed before cusparse_spmv_into_devptrs");

    let n = cache.n;
    let workspace_ptr = unsafe { handle_device_ptr(client, &cache.workspace_spmv_n) };

    unsafe {
        let mut q_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut y_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut q_dn, n as i64, q_devptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec q failed in forward SpMV (into_devptrs)");
        cusparseCreateDnVec(&mut y_dn, n as i64, dst_devptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec y failed in forward SpMV (into_devptrs)");
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        cusparseSpMV(
            cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const std::ffi::c_void,
            cache.sp_mat_spmv,
            q_dn,
            &beta as *const f32 as *const std::ffi::c_void,
            y_dn,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseSpMV forward (into_devptrs) failed");
        cusparseDestroyDnVec(q_dn).result().expect("cusparseDestroyDnVec q failed");
        cusparseDestroyDnVec(y_dn).result().expect("cusparseDestroyDnVec y failed");
    }
}

/// SP-10 Phase 3: triangular solve `A · x = b`, writing into a caller-owned
/// dst Handle's devptr. cuSPARSE-12.x SpSV. No cubecl allocations.
///
/// `a_values_devptr` is the device pointer to A's value array; `b_devptr` is
/// the rhs vector; `dst_devptr` is the destination x vector. The caller
/// pre-resolves all three so this function is safe inside a CUDA stream
/// capture region.
///
/// # Safety
/// Same contract as `cusparse_spmv_into_devptrs`.
pub(crate) unsafe fn cusparse_solve_into_devptrs(
    cache: &CudaPatternCache,
    client: &cubecl::client::ComputeClient<CudaRuntime>,
    a_values_devptr: u64,
    b_devptr: u64,
    dst_devptr: u64,
    stream: cudarc::driver::sys::CUstream,
) {
    use cudarc::cusparse::sys::{
        cudaDataType_t::CUDA_R_32F,
        cusparseCreateDnVec,
        cusparseDestroyDnVec,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        cusparseSpSV_solve,
        cusparseSetStream,
    };

    unsafe {
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream forward (into_devptrs) failed");
    }

    client
        .flush_no_sync()
        .expect("cubecl client flush_no_sync failed before cusparse_solve_into_devptrs");

    let n = cache.n;

    unsafe {
        // Re-point sparse-matrix descriptor at the current a_values.
        let crow_ptr = handle_device_ptr(client, &cache.d_crow);
        let col_ptr = handle_device_ptr(client, &cache.d_col);
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            cache.sp_mat,
            crow_ptr as *mut std::ffi::c_void,
            col_ptr as *mut std::ffi::c_void,
            a_values_devptr as *mut std::ffi::c_void,
        )
        .result()
        .expect("cusparseCsrSetPointers forward (into_devptrs) failed");

        cudarc::cusparse::sys::cusparseSpSV_updateMatrix(
            cache.handle,
            cache.desc_forward,
            a_values_devptr as *mut std::ffi::c_void,
            cudarc::cusparse::sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_GENERAL,
        )
        .result()
        .expect("cusparseSpSV_updateMatrix forward (into_devptrs) failed");

        let mut b_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut x_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut b_dn, n as i64, b_devptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec b failed in forward (into_devptrs)");
        cusparseCreateDnVec(&mut x_dn, n as i64, dst_devptr as *mut std::ffi::c_void, CUDA_R_32F)
            .result()
            .expect("cusparseCreateDnVec x failed in forward (into_devptrs)");

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
        .expect("cusparseSpSV_solve forward (into_devptrs) failed");
        cusparseDestroyDnVec(b_dn).result().expect("cusparseDestroyDnVec b failed");
        cusparseDestroyDnVec(x_dn).result().expect("cusparseDestroyDnVec x failed");
    }
}

// ---------------------------------------------------------------------------
// SP-10 Task 5: forward-graph capture helpers
// ---------------------------------------------------------------------------

/// Wrap an inner-backend `FloatTensorPrimitive` as a rank-1 `Tensor<B, 1>`.
/// No-op type juggling — convenience for the SP-10 capture path.
pub(crate) fn wrap_prim<B: Backend>(p: B::FloatTensorPrimitive) -> burn::tensor::Tensor<B, 1> {
    burn::tensor::Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(p))
}

/// Extract the raw `CUdeviceptr` (as `u64`) from a cubecl `Handle` using the
/// supplied client. Public-crate counterpart of the existing private
/// `handle_device_ptr` (kept for `cusparse_*` callers that already have a
/// client).
///
/// # Safety
/// - Caller must not free the returned pointer manually; cubecl owns the buffer.
/// - The `handle` (or a clone) must stay alive while the pointer is in use.
/// - The pointer is valid on cubecl's active stream; stream-ordering against
///   any external (cuSPARSE / raw CUDA) ops is the caller's responsibility.
pub(crate) unsafe fn handle_devptr(
    client: &cubecl::client::ComputeClient<CudaRuntime>,
    handle: &burn_cubecl::cubecl::server::Handle,
) -> u64 {
    // SAFETY: cloning the handle keeps the allocation alive past the
    // `get_resource` call; we only read the `ptr` field.
    let resource = client
        .get_resource(handle.clone())
        .expect("handle_devptr: failed to get GpuResource for Handle");
    resource.resource().ptr
}

/// Extract the raw `CUdeviceptr` (as `u64`) from an inner-backend
/// `FloatTensorPrimitive` (which must be a `CubeTensor<CudaRuntime>` — the
/// caller is responsible for ensuring `B == burn_cuda::Cuda<f32, i32>`).
///
/// # Safety
/// - Same constraints as [`primitive_as_cuda_view`].
/// - Returns `None` if the primitive is not a CUDA tensor — callers must
///   either `.expect(...)` or otherwise handle the `None` case explicitly.
///   Previously this returned `0` silently, which could let a captured graph
///   memcpy from the null page if a caller forgot to check the backend.
pub(crate) fn primitive_devptr<B: Backend>(
    prim: &B::FloatTensorPrimitive,
) -> Option<u64>
where
    B::FloatTensorPrimitive: 'static,
{
    primitive_as_cuda_view::<B>(prim).map(|v| v.ptr as u64)
}

/// SP-10: wrap a cubecl `Handle` (e.g. one of the `PersistentScratch.in_q*`
/// buffers) as an inner-backend `FloatTensorPrimitive` of shape `[n]`.
///
/// Clones the Handle so the underlying refcounted allocation stays alive
/// after the returned primitive is dropped — the caller's original Handle
/// (held inside `PersistentScratch`) remains valid for the cache lifetime.
///
/// Used by `try_capture_forward` so the captured graph's input reads come
/// from the persistent scratch buffers; on replay we just D2D-copy the
/// per-step `q_t` / `q_prime_t` into those scratch destinations before
/// `cuGraphLaunch`.
pub(crate) fn handle_to_primitive<B: Backend + 'static>(
    handle: &burn_cubecl::cubecl::server::Handle,
    n: usize,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    use burn::tensor::{DType, Shape};
    let client = compute_client::<B>(device);
    let shape = Shape::from(vec![n]);
    // SAFETY: TypeId of B::Device == CudaDevice is asserted inside compute_client.
    let cuda_device: cubecl::cuda::CudaDevice =
        unsafe { std::ptr::read(device as *const B::Device as *const cubecl::cuda::CudaDevice) };
    let cube = CubeTensor::<CudaRuntime>::from_handle(
        client,
        cuda_device,
        shape,
        handle.clone(),
        DType::F32,
    );
    cube_tensor_to_primitive::<B>(cube)
}

/// SP-10: allocate a fresh cubecl `Handle` of `[n × f32]`, D2D-copy from
/// `src_handle` into it, then wrap as an inner-backend `FloatTensorPrimitive`.
///
/// Used by `timestep_forward_via_graph` to publish the graph's scratch
/// outputs as autograd-tape-eligible primitives owned by the per-step result.
/// The fresh handle keeps the result alive past the next graph replay (which
/// would overwrite the scratch contents).
///
/// # Safety
/// - `src_handle` must contain valid f32 data of at least `n` elements.
/// - `stream` must be cubecl's primary stream, with context currently bound.
/// - Caller is responsible for stream-ordering with subsequent reads.
pub(crate) unsafe fn fresh_primitive_from_scratch<B: Backend + 'static>(
    src_handle: &burn_cubecl::cubecl::server::Handle,
    n: usize,
    stream: cudarc::driver::sys::CUstream,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    use burn::tensor::{DType, Shape};
    use burn_cubecl::cubecl::server::Handle;
    use cubecl::MemoryAllocationMode;
    let client = compute_client::<B>(device);
    let n_bytes = n * std::mem::size_of::<f32>();
    // SP-10 Phase 3 fix: allocate the fresh device buffer via the persistent
    // pool. The Auto pool may recycle slots whose Handle was just dropped
    // (e.g. the initial hot-start tensor that became `col0` via
    // `unsqueeze_dim`), corrupting Tensor::cat at end of `forward()`.
    // Persistent mode disables the recycling for this single allocation.
    //
    // Toggle the mode for the empty() call ONLY — restore Auto immediately
    // so subsequent cubecl ops keep their usual fast-recycle behavior.
    unsafe {
        client.allocation_mode(MemoryAllocationMode::Persistent);
    }
    let dst_handle: Handle = client.empty(n_bytes);
    unsafe {
        client.allocation_mode(MemoryAllocationMode::Auto);
    }

    // SAFETY: src is alive (caller-owned through `&Handle`); dst is the freshly
    // allocated handle (alive for this call and beyond, owned by `dst_handle`).
    // Both pointers are CUDA device pointers. The D2D copy is enqueued on
    // cubecl's active stream so it serializes with subsequent kernels.
    unsafe {
        let src_ptr = handle_devptr(&client, src_handle);
        let dst_ptr = handle_devptr(&client, &dst_handle);
        cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, n_bytes, stream)
            .expect("fresh_primitive_from_scratch: D2D copy failed");
    }

    let shape = Shape::from(vec![n]);
    // SAFETY: TypeId of B::Device == CudaDevice is asserted inside compute_client.
    let cuda_device: cubecl::cuda::CudaDevice =
        unsafe { std::ptr::read(device as *const B::Device as *const cubecl::cuda::CudaDevice) };
    let cube = CubeTensor::<CudaRuntime>::from_handle(
        client,
        cuda_device,
        shape,
        dst_handle,
        DType::F32,
    );
    cube_tensor_to_primitive::<B>(cube)
}

/// `&mut` variant of [`ensure_cuda_cache`]. Used by SP-10 to install
/// captured graphs onto the cache from `setup_inputs`.
///
/// # Safety
/// Same as [`ensure_cuda_cache`]: caller must guarantee single-threaded
/// access to the pattern's cuda cache. In practice this is invoked from
/// `MuskingumCunge::setup_inputs` on the training thread, before any
/// per-timestep call.
// SAFETY-INVERSION: caller must serialize access to the cache; same
// contract as `ensure_cuda_cache`. The `UnsafeSendCache` wrapper
// (cusparse.rs:1158) enforces single-threaded use via the marker.
#[allow(clippy::mut_from_ref)]
pub(crate) unsafe fn ensure_cuda_cache_mut(
    pattern: &crate::sparse::CsrPattern,
) -> &mut CudaPatternCache {
    // Initialize through the existing `get_or_init` if needed (so the
    // build_cuda_pattern_cache work isn't duplicated).
    // SAFETY: caller guarantees single-threaded access.
    let _ = unsafe { ensure_cuda_cache(pattern) };
    // Now grab `&mut` through the UnsafeCell. The OnceLock-style guard on
    // initialization is already satisfied by the line above; here we just
    // hand out a `&mut` to the inner `CudaPatternCache`.
    let ptr = pattern.cuda_cache.0.get();
    // SAFETY: single-threaded access guarantee from the caller means no
    // concurrent reads/writes of the inner Option. We initialized above so
    // unwrap() is safe.
    unsafe { (*ptr).as_mut().expect("cuda cache must be initialized") }
}

/// SP-10: capture the forward chain into `cache.graph_fwd`. Mutates the
/// cache to install the graph (or to record a fallback reason if capture
/// failed). Called from `MuskingumCunge::setup_inputs`.
///
/// Strategy: double-run. The chain is invoked once OUTSIDE capture to
/// observe the cubecl-allocator-assigned device pointers for each of the 24
/// outputs (Q_next + 23 saved intermediates). Then the chain is invoked
/// AGAIN inside `cuStreamBeginCapture` / `cuStreamEndCapture`, and we
/// schedule a `cuMemcpyDtoDAsync` for each output from its (assumed-stable)
/// src pointer into the persistent scratch destination. The resulting graph
/// is installed on `cache.graph_fwd`.
///
/// On failure (any error from capture, instantiate, or the closure itself),
/// the cache's `capture_status` is set to `FallbackReason(...)` and
/// `graph_fwd` is left `None` so subsequent `route_timestep` calls take the
/// SP-9 direct-launch path. Logs via `eprintln!` either way so silent
/// fallbacks are greppable in training logs.
///
/// Only called when `cfg.params.use_cuda_graphs == true` and
/// `sparse_solver == Cuda`. Bypassed entirely otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_capture_forward<I: Backend + 'static>(
    cache: &mut CudaPatternCache,
    cfg: &crate::config::Config,
    pattern: &std::sync::Arc<crate::sparse::CsrPattern>,
    n: I::FloatTensorPrimitive,
    q_spatial: I::FloatTensorPrimitive,
    p_spatial: I::FloatTensorPrimitive,
    length: I::FloatTensorPrimitive,
    slope: I::FloatTensorPrimitive,
    x_storage: I::FloatTensorPrimitive,
    device: &I::Device,
) where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    // SP-10 Phase 3: capture the fused-kernel forward chain
    //
    //     K1 (S1..S23) → cuSPARSE SpMV (S24) → K2 (S25 b_rhs)
    //     → assemble (S26) → cuSPARSE SpSV (S27) → K3 (S28 q_clamp)
    //
    // All intermediates live in either GPU registers (inside K1/K2/K3) or in
    // pre-allocated `PersistentScratch` slots. No cubecl Handle allocations
    // happen inside the capture region, so there is no allocator-vs-graph
    // recycling hazard. The captured graph reads from / writes to stable
    // scratch device pointers throughout.
    //
    // Per-step inputs (`q_t`, `q_prime_t`) live in `scratch.in_q` / `scratch.in_qp`;
    // `route_timestep` D2D-copies the current step's values into these
    // slots before `cuGraphLaunch`.
    //
    // Static inputs (`n`, `q_spatial`, `p_spatial`, `length`, `slope`,
    // `x_storage`, `pattern.diag_mask`) are seeded into scratch via a
    // one-time D2D before `cuStreamBeginCapture` — the captured K1 / assemble
    // kernels read those stable scratch addresses, not the caller's source
    // primitives' addresses.
    use crate::cuda_graph::{
        capture_on_stream, CaptureError, PersistentScratch,
        geometry_kernel::{assemble_kernel, b_rhs_kernel, forward_k1_kernel, q_clamp_kernel},
    };
    use burn_cubecl::cubecl::server::Handle;
    use burn_cubecl::cubecl::{CubeCount, CubeDim, prelude::TensorArg};

    let n_seg = pattern.n;
    let nnz = pattern.col.len();
    let bytes_n = (n_seg * std::mem::size_of::<f32>()) as usize;
    let bytes_nnz_i32 = (nnz * std::mem::size_of::<i32>()) as usize;
    let _ = bytes_nnz_i32;

    // 1. Allocate scratch on first capture. `allocate` also uploads
    //    `pattern.diag_mask` into `scratch.pattern_diag_mask`.
    if cache.scratch.is_none() {
        cache.scratch = Some(PersistentScratch::allocate::<I>(
            n_seg,
            nnz,
            pattern,
            device,
        ));
    }

    let client = compute_client::<I>(device);

    // Drain prior cubecl work before capture. Double-flush so the drop_queue
    // (double-buffered: staged -> pending) is empty before we enter capture.
    client.flush().expect("client flush #1 before forward capture");
    client.flush().expect("client flush #2 before forward capture");

    // Bind CUDA primary context on this thread BEFORE any cudarc capture /
    // launch call. Mirrors the Task 0 spike pattern.
    let cu_device: cudarc::driver::sys::CUdevice = match unsafe {
        cudarc::driver::result::device::get(0)
    } {
        Ok(d) => d,
        Err(e) => {
            cache.capture_status =
                CaptureStatus::FallbackReason(format!("forward: cuDeviceGet(0) failed: {e:?}"));
            eprintln!(
                "SP-10 forward graph capture FAILED, falling back: cuDeviceGet(0) failed: {e:?}"
            );
            return;
        }
    };
    let primary_ctx = match unsafe { cudarc::driver::result::primary_ctx::retain(cu_device) } {
        Ok(c) => c,
        Err(e) => {
            cache.capture_status = CaptureStatus::FallbackReason(format!(
                "forward: primary_ctx::retain failed: {e:?}"
            ));
            eprintln!(
                "SP-10 forward graph capture FAILED, falling back: primary_ctx::retain failed: {e:?}"
            );
            return;
        }
    };
    struct CtxGuard {
        cu_device: cudarc::driver::sys::CUdevice,
    }
    impl Drop for CtxGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = cudarc::driver::result::primary_ctx::release(self.cu_device);
            }
        }
    }
    let _ctx_guard = CtxGuard { cu_device };

    if let Err(e) = unsafe { cudarc::driver::result::ctx::set_current(primary_ctx) } {
        cache.capture_status =
            CaptureStatus::FallbackReason(format!("forward: ctx::set_current failed: {e:?}"));
        eprintln!(
            "SP-10 forward graph capture FAILED, falling back: ctx::set_current failed: {e:?}"
        );
        return;
    }

    let stream: cudarc::driver::sys::CUstream = cubecl_stream_active::<I>(device);

    // 2. Seed the static inputs into scratch via one-time D2D. After this
    //    runs, the K1/assemble kernels can read from scratch's stable
    //    devptrs.
    let scratch_ref = cache.scratch.as_ref().unwrap();
    let dst_in_n = unsafe { handle_devptr(&client, &scratch_ref.in_n) };
    let dst_in_qsp = unsafe { handle_devptr(&client, &scratch_ref.in_qsp) };
    let dst_in_psp = unsafe { handle_devptr(&client, &scratch_ref.in_psp) };
    let dst_in_length = unsafe { handle_devptr(&client, &scratch_ref.in_length) };
    let dst_in_slope = unsafe { handle_devptr(&client, &scratch_ref.in_slope) };
    let dst_in_xst = unsafe { handle_devptr(&client, &scratch_ref.in_xst) };

    let src_n = match primitive_devptr::<I>(&n) {
        Some(p) => p,
        None => {
            cache.capture_status = CaptureStatus::FallbackReason(
                "forward: n primitive is not CUDA".into(),
            );
            eprintln!("SP-10 forward graph capture FAILED, falling back: n primitive is not CUDA");
            return;
        }
    };
    let src_qsp = primitive_devptr::<I>(&q_spatial).expect("q_spatial must be CUDA");
    let src_psp = primitive_devptr::<I>(&p_spatial).expect("p_spatial must be CUDA");
    let src_length = primitive_devptr::<I>(&length).expect("length must be CUDA");
    let src_slope = primitive_devptr::<I>(&slope).expect("slope must be CUDA");
    let src_xst = primitive_devptr::<I>(&x_storage).expect("x_storage must be CUDA");

    // SAFETY: src and dst are valid CUDA device pointers of at least bytes_n;
    // stream is cubecl's primary stream; context is bound on this thread.
    // These D2Ds run BEFORE cuStreamBeginCapture, so they do not become
    // captured nodes — they are one-time seeds.
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(dst_in_n, src_n, bytes_n, stream)
            .expect("seed D2D n -> in_n failed");
        cudarc::driver::result::memcpy_dtod_async(dst_in_qsp, src_qsp, bytes_n, stream)
            .expect("seed D2D q_spatial -> in_qsp failed");
        cudarc::driver::result::memcpy_dtod_async(dst_in_psp, src_psp, bytes_n, stream)
            .expect("seed D2D p_spatial -> in_psp failed");
        cudarc::driver::result::memcpy_dtod_async(dst_in_length, src_length, bytes_n, stream)
            .expect("seed D2D length -> in_length failed");
        cudarc::driver::result::memcpy_dtod_async(dst_in_slope, src_slope, bytes_n, stream)
            .expect("seed D2D slope -> in_slope failed");
        cudarc::driver::result::memcpy_dtod_async(dst_in_xst, src_xst, bytes_n, stream)
            .expect("seed D2D x_storage -> in_xst failed");
    }
    // Submit the seed copies so they complete before capture begins. The
    // captured stream will see scratch.in_* populated when K1 launches.
    client.flush().expect("flush after seed D2D before capture");

    // 3. Resolve scratch devptrs for cuSPARSE callouts inside capture.
    //    Cube-launch kernels accept Handles directly (via TensorArg::from_raw_parts);
    //    cuSPARSE callouts need raw devptrs.
    let scratch_ref = cache.scratch.as_ref().unwrap();
    let dp_in_q = unsafe { handle_devptr(&client, &scratch_ref.in_q) };
    let dp_state_i_t = unsafe { handle_devptr(&client, &scratch_ref.state_i_t) };
    let dp_state_c1 = unsafe { handle_devptr(&client, &scratch_ref.state_c1) };
    let dp_state_a_values = unsafe { handle_devptr(&client, &scratch_ref.state_a_values) };
    let dp_state_b_rhs = unsafe { handle_devptr(&client, &scratch_ref.state_b_rhs) };
    let dp_state_x_sol = unsafe { handle_devptr(&client, &scratch_ref.state_x_sol) };
    let _ = dp_state_c1;

    // Scratch handles for cube kernels (clones bump Arc refcounts only).
    let h = |hd: &Handle| -> Handle { hd.clone() };
    let in_n_h = h(&scratch_ref.in_n);
    let in_qsp_h = h(&scratch_ref.in_qsp);
    let in_psp_h = h(&scratch_ref.in_psp);
    let in_q_h = h(&scratch_ref.in_q);
    let in_qp_h = h(&scratch_ref.in_qp);
    let in_length_h = h(&scratch_ref.in_length);
    let in_slope_h = h(&scratch_ref.in_slope);
    let in_xst_h = h(&scratch_ref.in_xst);
    let pattern_diag_mask_h = h(&scratch_ref.pattern_diag_mask);
    let pattern_row_for_nnz_h = h(&cache.d_row_for_nnz);
    let pattern_adj_values_h = h(&cache.d_adj_values);

    let s_depth = h(&scratch_ref.state_depth);
    let s_top_width = h(&scratch_ref.state_top_width);
    let s_side_slope = h(&scratch_ref.state_side_slope);
    let s_bottom_width = h(&scratch_ref.state_bottom_width);
    let s_hyd_radius = h(&scratch_ref.state_hydraulic_radius);
    let s_vel_un = h(&scratch_ref.state_velocity_unclamped);
    let s_vel_cl = h(&scratch_ref.state_velocity_clamped);
    let s_celerity = h(&scratch_ref.state_celerity);
    let s_k_musk = h(&scratch_ref.state_k_muskingum);
    let s_denom = h(&scratch_ref.state_denom);
    let s_c1 = h(&scratch_ref.state_c1);
    let s_c2 = h(&scratch_ref.state_c2);
    let s_c3 = h(&scratch_ref.state_c3);
    let s_c4 = h(&scratch_ref.state_c4);
    let s_ratio = h(&scratch_ref.state_ratio);
    let s_denominator = h(&scratch_ref.state_denominator);
    let s_q_eps = h(&scratch_ref.state_q_eps);
    let s_side_slope_raw = h(&scratch_ref.state_side_slope_raw);
    let s_bw_raw = h(&scratch_ref.state_bw_raw);
    let s_a_values = h(&scratch_ref.state_a_values);
    let s_b_rhs = h(&scratch_ref.state_b_rhs);
    let s_i_t = h(&scratch_ref.state_i_t);
    let s_x_sol = h(&scratch_ref.state_x_sol);
    let out_q_h = h(&scratch_ref.out_q);

    // Scalar bounds — match `forward_chain_inner`.
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;
    let dt = crate::routing::mmc::DT_SECONDS;

    // Common cube launch parameters. One thread per element; for assemble
    // the launch dimension is nnz instead of n_seg.
    //
    // For larger n we may need multi-cube tiling. To keep V1 sandbox (n=5)
    // simple we use one cube with CubeDim sized to the array; cubecl will
    // accept oversize dim because each kernel guards with `terminate!()`
    // when ABSOLUTE_POS >= len.
    //
    // n_seg/256 ceiling cubes of 256 threads each — handles arbitrary n.
    let block: u32 = 256;
    let grid_n = ((n_seg as u32 + block - 1) / block).max(1);
    let grid_nnz = ((nnz as u32 + block - 1) / block).max(1);

    let stride_n = vec![1_usize];
    let shape_n = vec![n_seg];
    let stride_nnz = vec![1_usize];
    let shape_nnz = vec![nnz];

    let cache_ptr: *const CudaPatternCache = &*cache;

    // 4. Capture the fused-kernel sequence.
    //
    // The closure runs INSIDE cuStreamBeginCapture / cuStreamEndCapture. It
    // issues 3 cube-launch kernels + 3 cuSPARSE calls, all writing to scratch
    // devptrs. No cubecl allocations happen inside; no host-sync calls.
    //
    // SAFETY: stream is cubecl's primary stream, primary context is bound on
    // this thread; the closure body issues only stream-ordered work; no
    // host-sync; all scratch devptrs are owned by `cache` (alive for the
    // returned CudaGraph's lifetime).
    //
    // SP-10 CONUS-scale fix: cubecl's kernel-launch path internally calls
    // `create_with_data` on every `cube_kernel::launch`, which (a) allocates
    // a small device buffer for kernel-argument-info and (b) does an H2D
    // copy into it. If those allocations happen INSIDE the capture region,
    // cubecl's CUDA storage hits `cuMemAllocAsync` on a stream that is
    // currently in capture mode — so the allocation is recorded as a graph
    // memalloc node instead of executing a real CUDA allocation. cubecl
    // returns the captured devptr to its pool's free-list when the
    // temporary info-buffer Handle drops. The pool later hands that slot
    // out for some unrelated `client.empty()`/`from_data` call (e.g. the
    // `Tensor::from_data` host-roundtrip in `forward()`), and the
    // subsequent `cuMemcpyHtoDAsync` to that "live in graph only" devptr
    // segfaults inside libcuda.
    //
    // At sandbox `n=5` the prior cubecl pool state happens to absorb the
    // capture-time growth into already-existing slots, so the crash doesn't
    // surface. At CONUS `n=3824` the pool grows during capture (more
    // / bigger bindings → bigger info buffer), and we hit it.
    //
    // Mitigation: run the entire kernel chain ONCE outside capture as a
    // warm-up so cubecl's pool grows under normal allocation (the
    // resulting kernel-info devptrs are real allocations). The temporary
    // info-buffer Handles drop at the end of the warm-up `launch` call and
    // return to the drop_queue; a `client.flush()` after the warm-up
    // recycles them onto the free-list. The subsequent in-capture launch
    // reuses those existing slots (no new `cuMemAllocAsync`), so capture
    // no longer absorbs growth nodes.
    let capture_result: Result<crate::cuda_graph::CudaGraph, String> = {
        // The fused kernel chain, factored so we can run it once as a warm-up
        // and again inside capture.
        let kernel_chain = || -> Result<(), String> {
            // ── K1: S1..S23 ───────────────────────────────────────────────
            let mk_in = |hd: &Handle| -> TensorArg<CudaRuntime> {
                unsafe {
                    TensorArg::from_raw_parts(
                        hd.clone(),
                        stride_n.clone().into(),
                        shape_n.clone().into(),
                    )
                }
            };
            let mk_out = mk_in;
            forward_k1_kernel::launch::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_1d(block),
                // 8 inputs
                mk_in(&in_n_h),
                mk_in(&in_qsp_h),
                mk_in(&in_psp_h),
                mk_in(&in_q_h),
                mk_in(&in_qp_h),
                mk_in(&in_length_h),
                mk_in(&in_slope_h),
                mk_in(&in_xst_h),
                // 19 outputs
                mk_out(&s_depth),
                mk_out(&s_top_width),
                mk_out(&s_side_slope),
                mk_out(&s_bottom_width),
                mk_out(&s_hyd_radius),
                mk_out(&s_vel_un),
                mk_out(&s_vel_cl),
                mk_out(&s_celerity),
                mk_out(&s_k_musk),
                mk_out(&s_denom),
                mk_out(&s_c1),
                mk_out(&s_c2),
                mk_out(&s_c3),
                mk_out(&s_c4),
                mk_out(&s_ratio),
                mk_out(&s_denominator),
                mk_out(&s_q_eps),
                mk_out(&s_side_slope_raw),
                mk_out(&s_bw_raw),
                // scalars
                bottom_width_lb,
                depth_lb,
                velocity_lb,
                dt,
            );
            // Drain cube's async submit queue onto the captured stream. cube
            // `launch` goes through `device.submit` (channel-async); without
            // this flush, the kernel may not have landed on `stream` by the
            // time cuStreamEndCapture runs.
            client.flush_no_sync().expect("flush_no_sync after K1");

            // ── SpMV: i_t = N · q_t ───────────────────────────────────────
            //
            // SAFETY: cache_ptr is a borrow of the live `cache` (we hold
            // `&mut cache` in the enclosing scope); we re-borrow here to
            // satisfy the call signature. dp_in_q / dp_state_i_t are stable
            // scratch devptrs allocated above. The stream is cubecl's.
            unsafe {
                cusparse_spmv_into_devptrs(
                    &*cache_ptr,
                    &client,
                    dp_in_q,
                    dp_state_i_t,
                    stream,
                );
            }

            // ── K2: b_rhs = c2*i_t + c3*q_t + c4*q_prime_t ───────────────
            b_rhs_kernel::launch::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_1d(block),
                mk_in(&s_c2),
                mk_in(&s_c3),
                mk_in(&s_c4),
                mk_in(&s_i_t),
                mk_in(&in_q_h),
                mk_in(&in_qp_h),
                mk_out(&s_b_rhs),
            );
            client.flush_no_sync().expect("flush_no_sync after K2");

            // ── assemble: a_values[k] = diag[k] - c1[row[k]] * adj[k] ───
            let mk_in_nnz = |hd: &Handle| -> TensorArg<CudaRuntime> {
                unsafe {
                    TensorArg::from_raw_parts(
                        hd.clone(),
                        stride_nnz.clone().into(),
                        shape_nnz.clone().into(),
                    )
                }
            };
            let mk_out_nnz = mk_in_nnz;
            assemble_kernel::launch::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(grid_nnz, 1, 1),
                CubeDim::new_1d(block),
                mk_in(&s_c1),                  // c (shape [n])
                mk_in_nnz(&pattern_row_for_nnz_h), // row_for_nnz (i32, [nnz])
                mk_in_nnz(&pattern_adj_values_h),  // adj (f32, [nnz])
                mk_in_nnz(&pattern_diag_mask_h),   // diag (f32, [nnz])
                mk_out_nnz(&s_a_values),
            );
            client.flush_no_sync().expect("flush_no_sync after assemble");

            // ── SpSV: A · x = b_rhs ───────────────────────────────────────
            unsafe {
                cusparse_solve_into_devptrs(
                    &*cache_ptr,
                    &client,
                    dp_state_a_values,
                    dp_state_b_rhs,
                    dp_state_x_sol,
                    stream,
                );
            }

            // ── K3: q_next = max(x_sol, discharge_lb) ─────────────────────
            q_clamp_kernel::launch::<f32, CudaRuntime>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_1d(block),
                mk_in(&s_x_sol),
                mk_out(&out_q_h),
                discharge_lb,
            );
            client.flush_no_sync().expect("flush_no_sync after K3");

            Ok(())
        };

        // WARM-UP: run the chain once OUTSIDE capture so cubecl's pool
        // grows under normal allocation. The kernel-info Handles dropped
        // at the end of each launch enter the drop_queue; the subsequent
        // double-flush below releases them back to the free-list, so the
        // in-capture run reuses existing slots and emits NO new
        // `cuMemAllocAsync` graph nodes.
        if let Err(e) = kernel_chain() {
            cache.capture_status =
                CaptureStatus::FallbackReason(format!("forward warm-up: {e}"));
            eprintln!("SP-10 forward graph WARM-UP FAILED, falling back: {e}");
            return;
        }
        // Two blocking flushes: the first submits the warm-up work to the
        // server and waits, ensuring all temporary info-buffer Handles
        // have been dropped on the server side; the second cycles the
        // drop_queue from staged→pending→free so reuse is enabled.
        client.flush().expect("client flush #1 after warm-up");
        client.flush().expect("client flush #2 after warm-up");

        // SAFETY: stream + ctx invariants validated above; closure is
        // host-sync-free; no cubecl allocations inside (post-warm-up).
        unsafe { capture_on_stream(stream, kernel_chain) }
            .map_err(|e: CaptureError| format!("{e}"))
    };

    match capture_result {
        Ok(graph) => {
            cache.graph_fwd = Some(graph);
            cache.capture_status = CaptureStatus::Captured;
            cache.capture_sig = Some((pattern.n, cfg.params.sparse_solver));
            eprintln!(
                "SP-10 forward graph captured (n={}, nnz={})",
                pattern.n,
                pattern.col.len()
            );
        }
        Err(e) => {
            cache.capture_status = CaptureStatus::FallbackReason(format!("forward: {e}"));
            eprintln!("SP-10 forward graph capture FAILED, falling back: {e}");
        }
    }
}
