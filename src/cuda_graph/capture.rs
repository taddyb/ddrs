//! Stream-capture helpers built on cudarc's raw driver API.
//!
//! We use the raw `cudarc::driver::result::{stream,graph}::*` functions
//! (not the safe `CudaGraph` in `cudarc::driver::safe::graph`) because cubecl
//! owns the stream and we work in terms of `sys::CUstream`, not a cudarc
//! `CudaStream` Arc.
//!
//! ## Context-binding contract (Task 0 finding)
//!
//! cubecl binds the CUDA primary context **only on its server-bound thread**.
//! Any host thread calling [`capture_on_stream`] or [`CudaGraph::launch`]
//! without that context bound gets `CUDA_ERROR_INVALID_CONTEXT` from
//! `cuGraphInstantiate` / `cuGraphLaunch`.
//!
//! Callers MUST invoke these helpers from inside
//! `cubecl::client::ComputeClient::exclusive_with_server(|server| ...)` (the
//! same pattern SP-7 added for `cubecl_stream_active`). That closure runs on
//! cubecl's server thread where the primary context is current.
//!
//! Internal context-binding was considered and rejected because (a) it would
//! couple this module to the burn/cubecl runtime types and (b) the hot path
//! (`route_timestep`) is going to wrap an entire forward+launch+backward block
//! in a single `exclusive_with_server` scope anyway, so paying the
//! closure-dispatch per individual `launch()` call would be wasteful.

use cudarc::driver::result::{graph as cu_graph_api, stream as cu_stream_api};
use cudarc::driver::sys::{
    CUgraph, CUgraphExec, CUgraphInstantiate_flags, CUstream, CUstreamCaptureMode_enum,
    CUstreamCaptureStatus_enum,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("begin_capture failed: {0}")]
    BeginFailed(cudarc::driver::DriverError),
    #[error("user closure failed during capture: {0}")]
    ClosureFailed(String),
    #[error("end_capture failed (likely host-sync inside region): {0}")]
    EndFailed(cudarc::driver::DriverError),
    #[error("graph::instantiate failed: {0}")]
    InstantiateFailed(cudarc::driver::DriverError),
    #[error("captured graph is null (region was empty or invalidated)")]
    EmptyCapture,
}

/// Owned `CUgraphExec` + `CUgraph`. Drops in correct order via `Drop`.
///
/// **Not Sync.** CUDA graphs are not internally synchronized.
pub struct CudaGraph {
    exec: CUgraphExec,
    template: CUgraph,
}

// SAFETY: `CUgraphExec` and `CUgraph` are opaque CUDA handles (raw pointers
// into the driver's tables). They are not thread-affine in the sense of
// requiring single-thread ownership — they only require that *uses* of them
// happen on a thread with the originating CUDA primary context bound. We
// enforce that contract at use sites (callers must be inside
// `exclusive_with_server`). Moving the handle across threads is sound.
unsafe impl Send for CudaGraph {}

impl CudaGraph {
    /// Launch the graph on `stream`.
    ///
    /// # Safety
    /// - `stream` must be a valid `CUstream`, typically cubecl's primary
    ///   stream from `cubecl_stream_active::<B>(device)`.
    /// - The current thread must have the originating CUDA primary context
    ///   bound. In practice this means the caller is inside
    ///   `ComputeClient::exclusive_with_server(|server| ...)`.
    /// - The stream must have compatible memory-pool configuration with the
    ///   stream the graph was captured on (cubecl's primary stream qualifies).
    pub unsafe fn launch(&self, stream: CUstream) -> Result<(), cudarc::driver::DriverError> {
        // SAFETY: `self.exec` is a valid `CUgraphExec` (constructed by
        // `capture_on_stream` and not yet dropped). `stream` validity and
        // context-binding are caller invariants per this method's safety
        // doc.
        unsafe { cu_graph_api::launch(self.exec, stream) }
    }

    /// Upload graph resources to device for first-launch overhead reduction.
    ///
    /// Useful between instantiation and the first replay so the driver can
    /// pre-stage kernels / parameters before we hit the steady-state loop.
    ///
    /// # Safety
    /// Same constraints as [`launch`](Self::launch).
    pub unsafe fn upload(&self, stream: CUstream) -> Result<(), cudarc::driver::DriverError> {
        // SAFETY: see `launch` — same invariants on `self.exec` and `stream`.
        unsafe { cu_graph_api::upload(self.exec, stream) }
    }

    /// Raw access to the underlying `CUgraphExec`. Intended for diagnostic
    /// / FFI callers; do not destroy it.
    pub fn exec_raw(&self) -> CUgraphExec {
        self.exec
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        // SAFETY: `self.exec` and `self.template` are valid handles owned by
        // `self` (constructed in `capture_on_stream`, not yet destroyed). The
        // CUDA API requires `cuGraphExecDestroy` before `cuGraphDestroy` when
        // both are alive — we honour that order. We swap each to null after
        // destroy so a re-entrant drop (impossible here, but defensive) is a
        // no-op. Errors are intentionally swallowed because Drop cannot
        // signal failure; in practice the only failure is "handle already
        // destroyed", which is benign at shutdown.
        unsafe {
            if !self.exec.is_null() {
                let _ = cu_graph_api::exec_destroy(self.exec);
                self.exec = std::ptr::null_mut();
            }
            if !self.template.is_null() {
                let _ = cu_graph_api::destroy(self.template);
                self.template = std::ptr::null_mut();
            }
        }
    }
}

/// Run `closure` inside a stream-capture region on `stream` and return the
/// resulting graph as an instantiated `CudaGraph`.
///
/// On instantiate failure, the captured graph template is destroyed before
/// the error is returned (no leak).
///
/// # Safety
/// - `stream` must be a valid `CUstream`.
/// - The current thread must have the originating CUDA primary context bound
///   (see module-level docs — call from inside `exclusive_with_server`).
/// - `closure` must not invoke any host-sync APIs (no `cuStreamSynchronize`,
///   no `cuEventSynchronize` on blocking events, no host-roundtrip tensor
///   reads such as `Tensor::into_data`). Any host-sync makes `end_capture`
///   fail.
pub unsafe fn capture_on_stream<F>(
    stream: CUstream,
    closure: F,
) -> Result<CudaGraph, CaptureError>
where
    F: FnOnce() -> Result<(), String>,
{
    // SAFETY: `stream` validity and context-binding are caller invariants.
    // THREAD_LOCAL is the strictest capture mode and gives the loudest error
    // if any cubecl thread leaks a CUDA call out of the captured scope.
    unsafe {
        cu_stream_api::begin_capture(
            stream,
            CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .map_err(CaptureError::BeginFailed)?;
    }

    let closure_result = closure();

    // SAFETY: same as `begin_capture` — caller invariants on stream + context.
    // Call `end_capture` unconditionally so the stream exits capture mode
    // even when the closure errored.
    let template = unsafe { cu_stream_api::end_capture(stream) }
        .map_err(CaptureError::EndFailed)?;

    // Surface closure failure AFTER end_capture so the stream is clean.
    closure_result.map_err(CaptureError::ClosureFailed)?;

    if template.is_null() {
        return Err(CaptureError::EmptyCapture);
    }

    // SAFETY: `template` is a valid CUgraph just produced by `end_capture`.
    // Context-binding is the caller's responsibility. We pass the auto-free-
    // on-launch flag so any cubecl async frees that ended up as graph nodes
    // are reclaimed by the driver after replay completes.
    let exec = match unsafe {
        cu_graph_api::instantiate(
            template,
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )
    } {
        Ok(exec) => exec,
        Err(e) => {
            // SAFETY: we own `template` (haven't handed it to a `CudaGraph`
            // yet). Destroying it here prevents the leak that would result
            // from early-returning without giving Drop a chance to run.
            unsafe {
                let _ = cu_graph_api::destroy(template);
            }
            return Err(CaptureError::InstantiateFailed(e));
        }
    };

    Ok(CudaGraph { exec, template })
}

/// Probe a stream's current capture status. Mostly diagnostic — useful in
/// tests to assert the stream entered/exited capture cleanly.
///
/// # Safety
/// - `stream` must be a valid `CUstream`.
/// - Current thread must have the originating CUDA primary context bound.
pub unsafe fn capture_status(
    stream: CUstream,
) -> Result<CUstreamCaptureStatus_enum, cudarc::driver::DriverError> {
    // SAFETY: `stream` validity and context-binding are caller invariants
    // (see this function's safety doc and the module-level docs).
    unsafe { cu_stream_api::is_capturing(stream) }
}
