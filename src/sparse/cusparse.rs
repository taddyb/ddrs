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
    client.flush().expect("cubecl client flush failed before cusparse_backward_solve");

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
    client.flush().expect("cubecl client flush failed before cusparse_spmv_forward");

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
    client.flush().expect("cubecl client flush failed before cusparse_spmv_backward");

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
    /// Number of reaches (rows = cols of the square network matrix).
    pub(crate) n: usize,
    /// Number of non-zeros in the network adjacency.
    pub(crate) nnz: usize,
    /// `!Send` marker — cuSPARSE descriptors are thread-bound.
    _not_send: PhantomData<*mut ()>,
}

impl Drop for CudaPatternCache {
    fn drop(&mut self) {
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
