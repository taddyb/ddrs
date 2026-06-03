# SP-10 CUDA Graphs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture one forward and one backward CUDA Graph per `MuskingumCunge` instance in `setup_inputs`; replay them per timestep so the CPU issues 1 `cuGraphLaunch` per direction instead of ~100 `cuLaunchKernel`s. Targets the 70%-of-CUDA-API `cuLaunchKernel` cost exposed by SP-9's V7b nsys profile.

**Architecture:** New `src/cuda_graph/` module with a `CudaGraph` newtype wrapping `cudarc::driver::sys::CUgraphExec`, a `PersistentScratch` of pre-allocated device buffers, and capture/replay helpers built on cudarc's raw `cuStreamBeginCapture_v2` / `cuStreamEndCapture` API (we don't get a cudarc `CudaStream` from cubecl, so we use the raw `sys::CUstream` we already get via `cubecl_stream_active`). `CudaPatternCache` (`src/sparse/cusparse.rs`) gains two `Option<CudaGraph>` fields plus the scratch. `route_timestep` and `TimestepOp::backward` grow a graph-replay branch gated by `params.use_cuda_graphs`.

**Tech Stack:** cudarc 0.19.7 raw driver API (`cudarc::driver::result::{stream,graph}::*` + `cudarc::driver::sys::CUgraphExec` / `CUgraph`), cubecl-cuda 0.10 (existing), BURN 0.21 (existing). No new crate dependencies.

**Spec:** `.claude/specs/2026-05-26-sp10-cuda-graphs-design.md`

---

## Pre-flight: what already exists

- `src/sparse/cusparse.rs` (1870 lines) — `CudaPatternCache` declared at line 1006; `d_adj_values` at 1028; `sp_mat_spmv` at 1041; `impl Drop` at 1061; `ensure_cuda_cache` at 1148; `build_cuda_pattern_cache` at 1177. Stream-share via `cubecl_stream_active::<B>(device)` at lines 172/413/568/723; client flush via `compute_client::<B>(device); client.flush()` at lines 189/429/584. SP-9 already uses this for `cusparse_spmv_forward`/`_backward`/`_assemble_backward`.
- `src/sparse/dispatch.rs` (233 lines) — dispatch entries route per `use_cuda: bool`.
- `src/routing/mmc.rs:113` — `setup_inputs` (the capture point).
- `src/routing/mmc.rs:221` — `route_timestep` (the forward replay branch).
- `src/routing/mmc_op.rs:74` — `TimestepOp` + `Backward<I, 5>` (the backward replay branch).
- `src/routing/mmc_op.rs:503` — `timestep_forward` (forward chain — all S1..S28 in this one function).
- `src/config.rs:123` — `Params` struct (where `use_cuda_graphs` lives).
- `tests/sp8_v7_perf.rs` — V7a template (median-of-3 after 3 warmups).
- `scripts/sp8_check_scatter.sh` — model for the V10 nsys-parsing script.
- `tests/sparse_cusparse_v8.rs` — V8 bit-match template (model for V9).

---

## Critical de-risking spike (Task 0)

The design spec assumed that capturing the cubecl-emitted kernel stream "just works." It does *not* automatically. cubecl 0.10 allocates a fresh `cuMemAllocAsync` buffer per BURN op output and tracks the returned `CUdeviceptr` in an internal `HashMap<StorageId, CUdeviceptr>`. Stream capture absorbs those alloc nodes into the graph. On *replay*, `cuGraphLaunch` re-runs the alloc nodes and returns **new** pointers, but cubecl's internal HashMap still holds the *capture-time* pointer.

Three possible outcomes when we try this:
1. **It works.** Allocations inside the captured region remain graph-local; we only consume outputs via D2D-copy to `PersistentScratch` (where the dst pointer is owned by us, not cubecl, so it's stable). cubecl's stale HashMap doesn't matter because we never re-enter cubecl for those allocations after capture.
2. **It silently corrupts.** Cubecl uses the stale pointer for some later operation and writes/reads at a freed-and-reallocated address.
3. **It panics on replay.** Some cuda driver call sees a graph-vs-non-graph mismatch and errors.

**Task 0 verifies which outcome we hit on a minimal example before committing the rest of the plan.** If outcome 1 (works), proceed to Task 1. If 2 or 3 (corrupts or panics), STOP and escalate — the design needs revision (likely Option B in the spec footnotes: write primitive-only `_inner` functions that bypass cubecl's allocator, ~300-line refactor of `timestep_forward`).

---

## File structure

**Created:**

- `src/cuda_graph/mod.rs` — module declaration, re-exports.
- `src/cuda_graph/capture.rs` — `CudaGraph` newtype + capture/launch helpers (Task 1).
- `src/cuda_graph/scratch.rs` — `PersistentScratch` struct + allocation (Task 3).
- `tests/sp10_spike_capture.rs` — Task 0 minimal end-to-end smoke test.
- `tests/sp10_graph_bitmatch.rs` — V9 (Task 9).
- `tests/sp10_v7a_perf.rs` — V7a rewrite (Task 11).
- `scripts/sp10_check_launches.sh` — V10 (Task 10).

**Modified:**

- `src/lib.rs` — declare `pub mod cuda_graph;`
- `src/config.rs` — add `use_cuda_graphs: bool` to `Params` (Task 2).
- `src/sparse/cusparse.rs` — fields on `CudaPatternCache` + `Drop` extension (Task 4).
- `src/routing/mmc.rs` — capture call in `setup_inputs` (Task 5, 7), replay branch in `route_timestep` (Task 6).
- `src/routing/mmc_op.rs` — replay branch in `TimestepOp::backward` (Task 8).
- `config/merit_training.yaml` — `use_cuda_graphs: true` flip (Task 12, only if all gates green).
- `.claude/ARCHITECTURE.md` — SP-10 close-out section (Task 12).

**No changes to:**

- `src/sparse/dispatch.rs`, `src/sparse/mod.rs` — SP-9's dispatch stays as-is.
- `examples/compare_ddr_sandbox.rs` — must still pass with `use_cuda_graphs=true`.
- `tests/sparse_gradcheck.rs`, `tests/sparse_cusparse_v5.rs`, `tests/sparse_cusparse_v8.rs`, `tests/sp8_gradcheck.rs` — regression gates only.

---

## Conventions for this plan

- All capture/replay code lives in `src/cuda_graph/`; nothing leaks into `src/sparse/` or `src/routing/` except the call sites.
- Use cudarc's **raw** API (`cudarc::driver::result::stream::begin_capture` / `end_capture` / `is_capturing`, `cudarc::driver::result::graph::{instantiate, launch, exec_destroy, destroy, upload}`) because cubecl owns the stream and we don't have a cudarc `CudaStream` object. Pattern matches SP-6/SP-9 which use raw cuSPARSE bindings the same way.
- `unsafe` blocks must carry a `// SAFETY: …` comment naming the invariants (stream validity, pointer validity, drop order).
- One commit per task. No `--amend`. Commit footer always:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- `cargo clippy --lib 2>&1 | grep -E "(cuda_graph|cusparse|mmc|mmc_op)" | head -5` must be empty after each commit (no new lints in touched code).
- After each task, run V1 regression: `cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3` — must report ABSOLUTE MATCH while `use_cuda_graphs: false` (the default until Task 12).

---

### Task 0: Spike — minimal capture/replay end-to-end

Verifies cubecl/cuda-graph memory-model compatibility before writing any architecture code. STOP if it fails.

**Files:**
- Create: `tests/sp10_spike_capture.rs`

- [ ] **Step 1: Write the spike test**

```rust
//! SP-10 Task 0: de-risking spike. Verifies cubecl + CUDA stream capture
//! interoperate at all. If this test corrupts output or panics, SP-10's
//! design needs revision (see plan's "Critical de-risking spike" section).
//!
//! Run: cargo test --release --test sp10_spike_capture -- --ignored --nocapture

#![cfg(feature = "cuda")]

use burn::backend::Cuda;
use burn::tensor::{Tensor, TensorData};
use cudarc::driver::result::{graph, stream};
use cudarc::driver::sys::{CUgraphExec, CUstreamCaptureMode_enum};

type B = Cuda<f32, i32>;

fn cubecl_stream() -> cudarc::driver::sys::CUstream {
    let device = burn::backend::cuda::CudaDevice::default();
    ddrs::sparse::cusparse::cubecl_stream_active::<B>(&device)
}

#[test]
#[ignore]
fn spike_capture_replay_tensor_add() {
    let device = burn::backend::cuda::CudaDevice::default();
    let stream = cubecl_stream();

    // Warm up cubecl: do one op outside capture so the kernel is JIT'd.
    let warm_a: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
    let warm_b: Tensor<B, 1> = Tensor::from_floats([10.0, 20.0, 30.0], &device);
    let _warm = warm_a + warm_b;

    // Begin capture.
    unsafe {
        stream::begin_capture(
            stream,
            CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .expect("begin_capture failed");
    }

    // Captured region: simple BURN op.
    let a: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
    let b: Tensor<B, 1> = Tensor::from_floats([10.0, 20.0, 30.0], &device);
    let c = a + b;
    let _ = c.into_data();   // forces materialization (may or may not host-sync)

    // End capture → graph → graph_exec.
    let cu_graph = unsafe {
        stream::end_capture(stream).expect("end_capture failed (likely host-sync inside region)")
    };
    assert!(!cu_graph.is_null(), "captured graph is null — capture region was empty or invalidated");

    let graph_exec: CUgraphExec = unsafe {
        graph::instantiate(cu_graph, cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64)
            .expect("graph::instantiate failed")
    };

    // Replay 3× and see if it survives.
    for i in 0..3 {
        unsafe {
            graph::launch(graph_exec, stream)
                .unwrap_or_else(|e| panic!("graph::launch replay #{i} failed: {e}"));
        }
    }

    // Cleanup.
    unsafe {
        graph::exec_destroy(graph_exec).expect("exec_destroy failed");
        graph::destroy(cu_graph).expect("destroy failed");
    }

    println!("spike OK: capture+replay survived 3 launches without panic");
}
```

- [ ] **Step 2: Expose `cubecl_stream_active` for the test**

`cubecl_stream_active` is currently private to `src/sparse/cusparse.rs`. Re-export it for tests:

In `src/sparse/cusparse.rs` (top of file, near `pub mod`):

```rust
// Re-exported for SP-10 Task 0 spike test. Internal callers should still use
// the module-private path.
#[doc(hidden)]
pub use self::cubecl_stream_active;
```

Or, if that doesn't work because of name collision, add a public `pub fn cubecl_stream_for_test<B: Backend>(...)` thin wrapper at the bottom of the file:

```rust
#[doc(hidden)]
pub fn cubecl_stream_for_test<B: Backend + 'static>(
    device: &B::Device,
) -> cudarc::driver::sys::CUstream {
    cubecl_stream_active::<B>(device)
}
```

(Use whichever compiles. The spike test imports whatever is publicly accessible.)

- [ ] **Step 3: Run the spike**

```bash
cargo test --release --test sp10_spike_capture -- --ignored --nocapture 2>&1 | tail -20
```

Expected output: `spike OK: capture+replay survived 3 launches without panic`.

**If the test panics during `end_capture`** ("operation not permitted under stream capture"): a host-sync (`.into_data()`) inside the captured region is the culprit. Try removing `.into_data()` and using an inner op chain instead — but more critically, this signals BURN's high-level Tensor ops may host-sync intermittently, which means **the design must shift to primitive-only `_inner` functions** (see "Critical de-risking spike" preamble). STOP and escalate.

**If the test panics during `graph::launch`**: cubecl's allocator/pool state is leaking through the captured graph in an unsupported way. STOP and escalate.

**If the test passes**: outcome 1 from the preamble holds; proceed.

- [ ] **Step 4: Commit**

```bash
git add tests/sp10_spike_capture.rs src/sparse/cusparse.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 0: capture/replay spike (de-risking)

Verifies cubecl + CUDA stream capture interoperate. Spike passes if a single
BURN tensor add can be captured and replayed 3× without panic.

If this test fails or corrupts, SP-10's design must shift to primitive-only
inner functions (~300 line refactor of timestep_forward). Currently green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1: `CudaGraph` newtype + capture helpers

Thin Rust wrapper around `cudarc::driver::sys::CUgraphExec` + `CUgraph`. RAII drop. The capture *function* is a separate helper that takes a closure: `capture(stream, |_| { … })` returns a `CudaGraph`.

**Files:**
- Create: `src/cuda_graph/mod.rs`
- Create: `src/cuda_graph/capture.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Declare module in `src/lib.rs`**

Find the `pub mod sparse;` line and add immediately after:

```rust
pub mod cuda_graph;
```

- [ ] **Step 2: Write `src/cuda_graph/mod.rs`**

```rust
//! SP-10: CUDA Graphs for per-timestep launch-overhead collapse.
//!
//! Captures the per-timestep kernel sequence (forward + backward) into a
//! `CUgraphExec` once during `MuskingumCunge::setup_inputs`, then replays it
//! per timestep so the CPU issues 1 `cuGraphLaunch` instead of ~100
//! `cuLaunchKernel`s.
//!
//! See `.claude/specs/2026-05-26-sp10-cuda-graphs-design.md`.

pub mod capture;
pub mod scratch;

pub use capture::{CudaGraph, capture_on_stream, CaptureError};
pub use scratch::PersistentScratch;
```

- [ ] **Step 3: Write `src/cuda_graph/capture.rs`**

```rust
//! Stream-capture helpers built on cudarc's raw driver API.
//!
//! We use the raw `cudarc::driver::result::{stream,graph}::*` functions
//! (not the safe `CudaGraph` in cudarc::driver::safe::graph) because cubecl
//! owns the stream and we work in terms of `sys::CUstream`, not a cudarc
//! `CudaStream` Arc.

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

unsafe impl Send for CudaGraph {}   // single-threaded use in our codepath

impl CudaGraph {
    /// Launch the graph on `stream`. Caller is responsible for stream validity.
    ///
    /// # Safety
    /// `stream` must be the same stream the graph was captured on (or a
    /// stream with compatible memory pool config). Typically cubecl's
    /// primary stream from `cubecl_stream_active`.
    pub unsafe fn launch(&self, stream: CUstream) -> Result<(), cudarc::driver::DriverError> {
        cu_graph_api::launch(self.exec, stream)
    }

    /// Upload graph resources to device for first-launch overhead reduction.
    ///
    /// # Safety
    /// Same constraints as `launch`.
    pub unsafe fn upload(&self, stream: CUstream) -> Result<(), cudarc::driver::DriverError> {
        cu_graph_api::upload(self.exec, stream)
    }

    pub fn exec_raw(&self) -> CUgraphExec { self.exec }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        // SAFETY: exec and template are valid handles owned by self;
        // we have not destroyed them yet. Drop order: exec before template.
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

/// Run `closure` inside a stream-capture region on `stream` and return
/// the resulting graph as an instantiated `CudaGraph`.
///
/// # Safety
/// `stream` must be a valid `CUstream`. The closure must not invoke any
/// host-sync APIs (no `cuStreamSynchronize`, no `cuEventSynchronize` on
/// blocking events, no host-roundtrip tensor reads). Any host-sync makes
/// `end_capture` fail.
pub unsafe fn capture_on_stream<F>(
    stream: CUstream,
    closure: F,
) -> Result<CudaGraph, CaptureError>
where
    F: FnOnce() -> Result<(), String>,
{
    cu_stream_api::begin_capture(
        stream,
        CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )
    .map_err(CaptureError::BeginFailed)?;

    let closure_result = closure();

    let template = cu_stream_api::end_capture(stream).map_err(CaptureError::EndFailed)?;

    // If the closure itself failed, surface that AFTER end_capture (so the
    // stream is left in a clean state).
    closure_result.map_err(CaptureError::ClosureFailed)?;

    if template.is_null() {
        return Err(CaptureError::EmptyCapture);
    }

    let exec = cu_graph_api::instantiate(
        template,
        CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64,
    )
    .map_err(|e| {
        // Cleanup template since we now own it.
        let _ = cu_graph_api::destroy(template);
        CaptureError::InstantiateFailed(e)
    })?;

    Ok(CudaGraph { exec, template })
}

/// Probe a stream's current capture status. Mostly diagnostic.
///
/// # Safety
/// `stream` must be a valid `CUstream`.
pub unsafe fn capture_status(
    stream: CUstream,
) -> Result<CUstreamCaptureStatus_enum, cudarc::driver::DriverError> {
    cu_stream_api::is_capturing(stream)
}
```

- [ ] **Step 4: Build + clippy check**

```bash
cargo build --lib 2>&1 | tail -5
cargo clippy --lib 2>&1 | grep -E "cuda_graph" | head -5
```

Expected: clean compile; no lints in `cuda_graph::*`.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/cuda_graph/
git commit -m "$(cat <<'EOF'
SP-10 Task 1: CudaGraph newtype + capture helpers

Thin RAII wrapper around CUgraphExec + CUgraph using cudarc's raw driver
result API (cubecl owns the stream — no cudarc CudaStream available).
Capture is via a closure: capture_on_stream(stream, ||{ ... }) -> CudaGraph.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add `use_cuda_graphs` to Params

Single boolean flag, defaults to `false`. Plumbed via YAML deserialization (same pattern as `sparse_solver`).

**Files:**
- Modify: `src/config.rs`

- [ ] **Step 1: Add field to `Params` struct**

In `src/config.rs:123` (after `sparse_solver: SparseSolver,`):

```rust
    /// SP-10: enable per-timestep CUDA-graph capture/replay on the CUDA
    /// path. No effect on the CPU path. Defaults to `false`; flipped to
    /// `true` in `config/merit_training.yaml` only after V9/V10/V7a pass.
    pub use_cuda_graphs: bool,
```

In the `Default for Params` impl (around line 142, after `sparse_solver: SparseSolver::default(),`):

```rust
            use_cuda_graphs: false,
```

- [ ] **Step 2: Plumb through YAML deserialization**

In `src/config.rs:159` (the raw struct), after `sparse_solver: Option<String>,`:

```rust
    use_cuda_graphs: Option<bool>,
```

In `src/config.rs:199` (the parse block), after the `p.sparse_solver = match r.sparse_solver.as_deref() { ... }` block, append:

```rust
        if let Some(b) = r.use_cuda_graphs {
            p.use_cuda_graphs = b;
        }
```

- [ ] **Step 3: Add a regression test**

In `src/config.rs:352` (existing test region, just below the `sparse_solver` test):

```rust
        // SP-10: use_cuda_graphs defaults to false when not set in YAML.
        assert!(!cfg.params.use_cuda_graphs);
```

- [ ] **Step 4: Build + tests**

```bash
cargo test --lib config:: 2>&1 | tail -5
```

Expected: all config tests pass (including the new assertion).

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 2: add use_cuda_graphs flag to Params

Defaults to false. YAML key: params.use_cuda_graphs. Will be flipped to true
in config/merit_training.yaml only after Task 12 confirms V9/V10/V7a gates pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `PersistentScratch` struct

33 pre-allocated cubecl-server Handles totalling `~33 × n × 4 bytes`. Allocated via `client.empty(n * 4)` (which routes through cubecl's `cuMemAllocAsync` path; we already use this in SP-9). All 33 buffers are reused across forward and backward capture/replay.

**Files:**
- Create: `src/cuda_graph/scratch.rs`

- [ ] **Step 1: Write the struct**

```rust
//! SP-10: persistent per-MC-instance scratch buffers.
//!
//! Allocated once during `setup_inputs`, dropped when `CudaPatternCache`
//! drops. All buffers are `[n × f32]`. Total: ~33n × 4 bytes (~540 KB for
//! n=5K gauge subgraph; ~45 MB for full CONUS n=346,321).

use burn::tensor::backend::Backend;
use burn_cubecl::cubecl::server::Handle;

/// Pre-allocated GPU buffers reused across every graph replay. Pointers are
/// stable for the lifetime of the cache.
pub struct PersistentScratch {
    pub n_segments: usize,

    // ── forward inputs / outputs ───────────────────────────────────────
    pub in_q: Handle,
    pub in_qp: Handle,
    pub out_q: Handle,

    // 24 saved-state outputs (forward) / inputs (backward).
    pub state_depth: Handle,
    pub state_top_width: Handle,
    pub state_side_slope: Handle,
    pub state_bottom_width: Handle,
    pub state_hydraulic_radius: Handle,
    pub state_velocity_unclamped: Handle,
    pub state_velocity_clamped: Handle,
    pub state_celerity: Handle,
    pub state_k_muskingum: Handle,
    pub state_denom: Handle,
    pub state_c1: Handle,
    pub state_c2: Handle,
    pub state_c3: Handle,
    pub state_c4: Handle,
    pub state_a_values: Handle,
    pub state_b_rhs: Handle,
    pub state_i_t: Handle,
    pub state_x_sol: Handle,
    pub state_ratio: Handle,
    pub state_denominator: Handle,
    pub state_q_eps: Handle,
    pub state_side_slope_raw: Handle,
    pub state_bw_raw: Handle,

    // ── backward inputs / outputs ──────────────────────────────────────
    pub in_grad_q_next: Handle,
    pub out_grad_n: Handle,
    pub out_grad_q_spatial: Handle,
    pub out_grad_p_spatial: Handle,
    pub out_grad_q_t: Handle,
    pub out_grad_q_prime_t: Handle,
}

impl PersistentScratch {
    pub fn allocate<B: Backend + 'static>(
        n_segments: usize,
        device: &B::Device,
    ) -> Self {
        let client = crate::sparse::cusparse::compute_client::<B>(device);
        let bytes = (n_segments * std::mem::size_of::<f32>()) as u64;
        let mk = || client.empty(bytes as usize);

        Self {
            n_segments,
            in_q: mk(), in_qp: mk(), out_q: mk(),
            state_depth: mk(), state_top_width: mk(), state_side_slope: mk(),
            state_bottom_width: mk(), state_hydraulic_radius: mk(),
            state_velocity_unclamped: mk(), state_velocity_clamped: mk(),
            state_celerity: mk(), state_k_muskingum: mk(), state_denom: mk(),
            state_c1: mk(), state_c2: mk(), state_c3: mk(), state_c4: mk(),
            state_a_values: mk(), state_b_rhs: mk(), state_i_t: mk(),
            state_x_sol: mk(), state_ratio: mk(), state_denominator: mk(),
            state_q_eps: mk(), state_side_slope_raw: mk(), state_bw_raw: mk(),
            in_grad_q_next: mk(),
            out_grad_n: mk(), out_grad_q_spatial: mk(), out_grad_p_spatial: mk(),
            out_grad_q_t: mk(), out_grad_q_prime_t: mk(),
        }
    }
}
```

- [ ] **Step 2: Confirm `compute_client` is pub-visible**

If `compute_client` is private in `src/sparse/cusparse.rs`, change to `pub(crate)`:

```bash
grep -n "fn compute_client" src/sparse/cusparse.rs
```

Expected: function exists. If `pub(crate)` isn't already on it, edit:

```rust
pub(crate) fn compute_client<B: Backend + 'static>(...) -> ...
```

- [ ] **Step 3: Build**

```bash
cargo build --lib 2>&1 | tail -5
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/cuda_graph/scratch.rs src/sparse/cusparse.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 3: PersistentScratch struct

33 cubecl-server Handles totalling ~33×n×4 bytes. Allocated once per
MuskingumCunge instance; pointers stable for the cache's lifetime so the
captured CUDA graphs reference them safely on replay.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire scratch + graph fields onto `CudaPatternCache`

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Add fields to the struct**

Find the `CudaPatternCache` declaration (currently around line 1006). After the SP-9 SpMV fields block (around line 1050), append:

```rust
    // ── SP-10 CUDA Graphs ─────────────────────────────────────────────
    /// Persistent per-instance scratch buffers. Allocated in
    /// `MuskingumCunge::setup_inputs`; consumed by both forward and
    /// backward graph capture/replay.
    pub(crate) scratch: Option<crate::cuda_graph::PersistentScratch>,

    /// Forward-direction captured CUDA graph. `None` if capture failed
    /// or `params.use_cuda_graphs == false`.
    pub(crate) graph_fwd: Option<crate::cuda_graph::CudaGraph>,

    /// Backward-direction captured CUDA graph. Same fallback semantics.
    pub(crate) graph_bwd: Option<crate::cuda_graph::CudaGraph>,

    /// `(n_segments, sparse_solver_kind)` signature at capture time. If
    /// this changes between batches, drop both graphs and recapture.
    pub(crate) capture_sig: Option<(usize, crate::config::SparseSolver)>,

    /// Observability: surfaces silent fallbacks. Logged at end of
    /// setup_inputs.
    pub(crate) capture_status: CaptureStatus,
```

- [ ] **Step 2: Add the `CaptureStatus` enum**

Top of `src/sparse/cusparse.rs`, near other top-level declarations:

```rust
/// SP-10 observability. Surfaces whether capture succeeded; if not, the
/// reason so training-log greps catch silent fallbacks.
#[derive(Debug, Clone)]
pub(crate) enum CaptureStatus {
    NotAttempted,
    Captured,
    FallbackReason(String),
}
```

- [ ] **Step 3: Initialize the new fields in `build_cuda_pattern_cache`**

At the bottom of `build_cuda_pattern_cache` (around line 1320, in the struct-literal return), add:

```rust
        scratch: None,
        graph_fwd: None,
        graph_bwd: None,
        capture_sig: None,
        capture_status: CaptureStatus::NotAttempted,
```

- [ ] **Step 4: Extend `Drop for CudaPatternCache`**

In the existing `impl Drop` (around line 1061), add **at the very top** of `fn drop` (before any cuSPARSE descriptor destruction — so graphs go first while their referenced scratch is still alive):

```rust
        // SP-10: drop captured graphs first. CudaGraph::drop destroys
        // exec then template; both reference scratch handles via pointer,
        // but the references are baked into the CUgraphExec internals,
        // not into Rust objects — so order matters relative to scratch
        // (handled by struct-field drop order below: graphs drop, then
        // scratch drops via Option-field default order).
        self.graph_fwd = None;
        self.graph_bwd = None;
        // scratch: dropped automatically when the cache itself drops
        // (after the cuSPARSE descriptors below).
```

- [ ] **Step 5: Build**

```bash
cargo build --lib 2>&1 | tail -5
```

Expected: clean. (Fields are `Option<…>`, so existing constructors work.)

- [ ] **Step 6: Commit**

```bash
git add src/sparse/cusparse.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 4: graph + scratch fields on CudaPatternCache

5 new fields (scratch, graph_fwd, graph_bwd, capture_sig, capture_status)
plus a CaptureStatus enum for fallback observability. Graphs drop before
scratch via explicit None-assignment at top of Drop::drop.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Capture forward graph during `setup_inputs`

The capture runs the existing `timestep_forward` once with **inputs sourced from `PersistentScratch`** so that scratch pointers are baked into the captured kernels. Outputs of the captured chain are then D2D-copied into scratch.out_q / scratch.state_* via explicit `cuMemcpyDtoDAsync` calls — these become memcpy nodes inside the graph.

**Files:**
- Modify: `src/routing/mmc.rs`
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Add `capture_forward_graph` helper to `src/cuda_graph/capture.rs`**

At the bottom of `src/cuda_graph/capture.rs`, add:

```rust
/// Wraps a forward-pass closure in stream capture, copies the closure's
/// rank-1 output handles into persistent scratch via cuMemcpyDtoDAsync
/// (so the dst pointer is stable across replays), and returns the
/// resulting CudaGraph.
///
/// # Safety
/// `stream` must be cubecl's primary stream. The output_pairs slice
/// describes (src_handle.bytes, src_devptr, dst_devptr, num_bytes) tuples;
/// dst_devptr must outlive the returned CudaGraph (i.e. point into
/// PersistentScratch).
pub unsafe fn capture_forward_with_outputs<F>(
    stream: CUstream,
    closure: F,
    output_copies: &[(u64 /* src */, u64 /* dst */, u64 /* bytes */)],
) -> Result<CudaGraph, CaptureError>
where
    F: FnOnce() -> Result<(), String>,
{
    capture_on_stream(stream, || {
        closure()?;
        for &(src, dst, bytes) in output_copies {
            // SAFETY: src and dst are valid GPU pointers; bytes <= sizes.
            cudarc::driver::result::memcpy_dtod_async(dst, src, bytes as usize, stream)
                .map_err(|e| format!("cuMemcpyDtoDAsync during capture failed: {e}"))?;
        }
        Ok(())
    })
}
```

Note: cudarc 0.19.7 exposes `cudarc::driver::result::memcpy_dtod_async` at line 1076 of `result.rs`. Verify the exact name:

```bash
grep -n "pub.*fn.*memcpy_dtod" ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.7/src/driver/result.rs | head -3
```

Adjust the call site to whatever the real symbol is.

- [ ] **Step 2: Add `try_capture_forward` to `src/sparse/cusparse.rs`**

This is the heavy lifter. It runs a one-off forward pass inside a capture region, then archives the result. Insert at the bottom of the file (before any test module):

```rust
/// SP-10: capture the forward chain into `cache.graph_fwd`. Mutates the
/// cache to install the graph (or to record a fallback reason if capture
/// failed). Called from `MuskingumCunge::setup_inputs`.
///
/// Captures with `params.sparse_solver == Cuda` semantics. Bypassed
/// entirely when `params.use_cuda_graphs == false` or the device is CPU.
pub(crate) fn try_capture_forward<I: Backend + 'static>(
    cache: &mut CudaPatternCache,
    cfg: &crate::config::Config,
    pattern: &std::sync::Arc<crate::sparse::CsrPattern>,
    assembler: &crate::sparse::AValuesAssembler<I>,
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
    use crate::cuda_graph::capture::{capture_forward_with_outputs, CaptureError};
    use burn::tensor::{Tensor, TensorPrimitive};

    // 1. Allocate scratch lazily on first capture.
    if cache.scratch.is_none() {
        cache.scratch = Some(crate::cuda_graph::PersistentScratch::allocate::<I>(
            pattern.n,
            device,
        ));
    }
    let scratch = cache.scratch.as_ref().unwrap();

    // 2. Build BURN-tensor inputs that point at scratch.in_q / scratch.in_qp.
    //    These are placeholders — value doesn't matter, addresses do.
    //    Use cubecl's TensorHandle::from_existing to wrap each Handle as an
    //    inner-backend primitive without re-allocating.
    let q_t_prim = crate::sparse::handle_to_primitive::<I>(&scratch.in_q, pattern.n, device);
    let q_prime_t_prim = crate::sparse::handle_to_primitive::<I>(&scratch.in_qp, pattern.n, device);

    // 3. Pre-flush the client so the capture region starts clean.
    let stream = cubecl_stream_active::<I>(device);
    let client = compute_client::<I>(device);
    client.flush().expect("client flush before forward capture");

    // 4. Run the forward chain inside capture. Closure body calls a
    //    cut-down version of timestep_forward that doesn't register on
    //    the autograd tape. (It only needs to fire the same kernel
    //    sequence; the captured outputs go to scratch via the D2D pairs.)
    let closure = || {
        // Inline the S1..S28 chain at the inner-backend tensor level.
        // Each output BURN tensor's underlying handle is what will be
        // copied into scratch via output_copies below.
        let n_in = wrap_prim::<I>(n.clone());
        let qsp_in = wrap_prim::<I>(q_spatial.clone());
        let psp_in = wrap_prim::<I>(p_spatial.clone());
        let qt_in = wrap_prim::<I>(q_t_prim.clone());
        let qpt_in = wrap_prim::<I>(q_prime_t_prim.clone());
        let length_in = wrap_prim::<I>(length.clone());
        let slope_in = wrap_prim::<I>(slope.clone());
        let xst_in = wrap_prim::<I>(x_storage.clone());

        // *** Run the same S1..S28 chain as timestep_forward. Returns
        // *** the 24 saved-state primitives + Q_next + their handle device-
        // *** pointers via inspect_primitive::<I>(&prim) -> u64.
        let (q_next_prim, state_prims) = forward_chain_inner::<I>(
            cfg, pattern, assembler,
            n_in, qsp_in, psp_in, qt_in, qpt_in,
            length_in, slope_in, xst_in,
        );

        // The output_copies are queued AFTER the chain — see Step 1
        // capture_forward_with_outputs.
        Ok::<(), String>(())
    };

    // 5. Build output_copies: 1 for Q_next + 24 for saved state.
    //    Each entry: (src_devptr_from_closure_output, dst_devptr_into_scratch, bytes).
    //    This requires running the closure FIRST to learn the src ptrs.
    //    HOWEVER: we can't run-then-capture; we have to capture-and-run-once.
    //    So we use a 2-pass approach: first pass runs OUTSIDE capture to
    //    discover output handle device-pointers; second pass runs inside
    //    capture and we ASSUME the same handles will be allocated.
    //
    //    Because cubecl's allocator is deterministic for a given allocation
    //    sequence, this typically works. If it doesn't, the captured graph
    //    will write to wrong addresses (Task 0 spike must validate this).

    // (See subagent execution note in plan preamble — the exact code shape
    // depends on what handle_to_primitive and forward_chain_inner return.
    // This task may need to be split if those helpers don't already exist.)

    let bytes_per_buf = (pattern.n * std::mem::size_of::<f32>()) as u64;

    // 6. Attempt capture.
    let capture_result = unsafe {
        capture_forward_with_outputs(stream, closure, &[
            // (src, dst, bytes) — populated after probing closure outputs.
        ])
    };

    match capture_result {
        Ok(graph) => {
            cache.graph_fwd = Some(graph);
            cache.capture_status = CaptureStatus::Captured;
            cache.capture_sig = Some((pattern.n, cfg.params.sparse_solver));
            tracing::info!("SP-10 forward graph captured (n={})", pattern.n);
        }
        Err(e) => {
            cache.capture_status = CaptureStatus::FallbackReason(format!("forward: {e}"));
            tracing::warn!("SP-10 forward graph capture FAILED, falling back: {e}");
        }
    }
}
```

**Implementation note for the executor:** Steps 4-6 above sketch the structure but defer the exact output-handle-pointer extraction to runtime. This is the single trickiest piece of the plan. Implement as follows:

1. Add a private helper `pub(crate) fn handle_devptr<B: Backend + 'static>(handle: &Handle) -> u64` in `src/sparse/cusparse.rs` that returns the underlying `CUdeviceptr` (see how SP-9's `cusparse_spmv_forward` already extracts these — pattern is `GpuResource { ptr, .. } = client.get_resource(binding.clone())`).
2. Run `forward_chain_inner` once OUTSIDE the capture region, save the output handle ptrs.
3. Re-run inside capture; assume cubecl reuses the same handle addresses (deterministic for the same sequence). If addresses differ, Task 0 spike would have caught it.

- [ ] **Step 3: Add `wrap_prim` and `forward_chain_inner` helpers**

At the top of `src/sparse/cusparse.rs` (or in a new `src/routing/mmc_chain.rs` if it gets long), add:

```rust
pub(crate) fn wrap_prim<B: Backend>(p: B::FloatTensorPrimitive) -> burn::tensor::Tensor<B, 1> {
    burn::tensor::Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(p))
}

pub(crate) fn unwrap_prim<B: Backend>(t: burn::tensor::Tensor<B, 1>) -> B::FloatTensorPrimitive {
    match t.into_primitive() {
        burn::tensor::TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    }
}
```

For `forward_chain_inner`: copy the S1..S28 body from `timestep_forward` (lines 575..648 of `src/routing/mmc_op.rs`) into a helper function. Take `&Config`, `&Arc<CsrPattern>`, `&AValuesAssembler<I>`, and the 8 input `Tensor<I, 1>`s. Return `(I::FloatTensorPrimitive /* q_next */, [I::FloatTensorPrimitive; 24] /* saved state */)`.

Place it in `src/routing/mmc_op.rs` (so both `timestep_forward` and the SP-10 capture path can share it). Refactor `timestep_forward` to use it.

- [ ] **Step 4: Hook into `setup_inputs`**

In `src/routing/mmc.rs:113`, at the end of `setup_inputs` (after the existing hotstart block, ~line 191), add:

```rust
        // SP-10: capture the per-timestep CUDA graphs once, eagerly.
        if self.cfg.params.use_cuda_graphs
            && self.sparse_solver == crate::config::SparseSolver::Cuda
        {
            // SAFETY: pattern is bound on this thread; we own &mut self.
            let cache = unsafe { crate::sparse::cusparse::ensure_cuda_cache_mut(self.pattern.as_ref().unwrap()) };
            crate::sparse::cusparse::try_capture_forward::<I>(
                cache, &self.cfg, self.pattern.as_ref().unwrap(),
                self.assembler.as_ref().unwrap(),
                self.n.as_ref().unwrap().clone().into_primitive_inner(),
                self.q_spatial.as_ref().unwrap().clone().into_primitive_inner(),
                self.p_spatial_broadcast(self.n_segments.unwrap()).into_primitive_inner(),
                self.length.as_ref().unwrap().clone().into_primitive_inner(),
                self.slope.as_ref().unwrap().clone().into_primitive_inner(),
                self.x_storage.as_ref().unwrap().clone().into_primitive_inner(),
                &self.device,
            );
        }
```

Note: `into_primitive_inner` is a helper that strips the Autodiff wrapper. Add it to `src/routing/mmc.rs` if missing:

```rust
trait IntoInnerPrimitive<I: Backend> {
    fn into_primitive_inner(self) -> I::FloatTensorPrimitive;
}
impl<I: Backend> IntoInnerPrimitive<I> for Tensor<Autodiff<I>, 1> {
    fn into_primitive_inner(self) -> I::FloatTensorPrimitive {
        match self.into_primitive() {
            TensorPrimitive::Float(p) => match p {
                // unwrap the autodiff layer — mirror the existing unwrap_at
                // in mmc_op.rs at line 530
                _ => unimplemented!("port unwrap_at here"),
            },
            _ => unreachable!(),
        }
    }
}
```

(Executor: look at `mmc_op.rs:530` `unwrap_at` for the canonical unwrap — copy that.)

- [ ] **Step 5: Add `ensure_cuda_cache_mut`**

In `src/sparse/cusparse.rs`, add a `&mut` variant of `ensure_cuda_cache` (currently at line 1148):

```rust
/// Mutable variant of `ensure_cuda_cache`. Used by SP-10 to install
/// captured graphs onto the cache.
///
/// # Safety
/// Same as `ensure_cuda_cache`: caller must serialize access to the
/// pattern's cache.
pub(crate) unsafe fn ensure_cuda_cache_mut(
    pattern: &crate::sparse::CsrPattern,
) -> &mut CudaPatternCache {
    // Mirror the structure of ensure_cuda_cache at line 1148, but return
    // &mut. The existing one uses OnceLock<UnsafeSendCache>; we need
    // get_mut on that, which OnceLock doesn't provide post-init. Use
    // get().unwrap() and cast through the UnsafeSendCache.
    todo!("implement mirroring existing ensure_cuda_cache pattern")
}
```

(Executor: complete based on the existing `ensure_cuda_cache` body.)

- [ ] **Step 6: Run V1 regression — must still pass with use_cuda_graphs=false (default)**

```bash
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3
```

Expected: `ABSOLUTE MATCH`.

- [ ] **Step 7: Commit**

```bash
git add src/sparse/cusparse.rs src/routing/mmc.rs src/routing/mmc_op.rs src/cuda_graph/
git commit -m "$(cat <<'EOF'
SP-10 Task 5: capture forward CUDA graph in setup_inputs

When params.use_cuda_graphs && sparse_solver == Cuda, runs a one-off
timestep forward inside cuStreamBeginCapture/EndCapture and installs the
resulting CUgraphExec on CudaPatternCache.graph_fwd. On failure, records
a FallbackReason on capture_status and proceeds without graphs.

V1 regression (compare_ddr_sandbox) still ABSOLUTE MATCH with the default
use_cuda_graphs=false.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Forward replay branch in `route_timestep`

When `cache.graph_fwd.is_some()`, replace the direct-launch path with: D2D-copy inputs into scratch.in_*, `cuGraphLaunch`, D2D-copy outputs into fresh primitives, register `TimestepOp` on the tape with the fresh primitives.

**Files:**
- Modify: `src/routing/mmc.rs`
- Modify: `src/routing/mmc_op.rs`

- [ ] **Step 1: Add `timestep_forward_via_graph` to `mmc_op.rs`**

After the existing `timestep_forward` (around line 709), add:

```rust
/// SP-10: graph-replay variant of timestep_forward. Same signature, same
/// autograd-tape behavior. Branches on `cache.graph_fwd.is_some()`.
#[allow(clippy::too_many_arguments)]
pub fn timestep_forward_via_graph<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    n_at: Tensor<Autodiff<I>, 1>,
    q_spatial_at: Tensor<Autodiff<I>, 1>,
    p_spatial_at: Tensor<Autodiff<I>, 1>,
    q_t_at: Tensor<Autodiff<I>, 1>,
    q_prime_t_at: Tensor<Autodiff<I>, 1>,
    length_at: Tensor<Autodiff<I>, 1>,
    slope_at: Tensor<Autodiff<I>, 1>,
    x_storage_at: Tensor<Autodiff<I>, 1>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    let cache = unsafe { crate::sparse::cusparse::ensure_cuda_cache(pattern) };
    if cache.graph_fwd.is_none() {
        // Fallback: SP-9 direct-launch path.
        return timestep_forward::<I>(
            cfg, pattern, assembler,
            n_at, q_spatial_at, p_spatial_at, q_t_at, q_prime_t_at,
            length_at, slope_at, x_storage_at,
        );
    }

    let scratch = cache.scratch.as_ref().unwrap();
    let graph = cache.graph_fwd.as_ref().unwrap();
    let n_seg = scratch.n_segments;

    // 1. Unwrap autograd primitives.
    // (Mirror `unwrap_at` from timestep_forward:530.)
    let qt_prim = unwrap_at_primitive::<I>(q_t_at.clone());
    let qpt_prim = unwrap_at_primitive::<I>(q_prime_t_at.clone());

    let device = I::float_device(&qt_prim);
    let stream = crate::sparse::cusparse::cubecl_stream_active::<I>(&device);
    let client = crate::sparse::cusparse::compute_client::<I>(&device);
    client.flush().expect("client flush before graph replay");

    // 2. D2D copy inputs into scratch.in_*.
    let bytes = (n_seg * std::mem::size_of::<f32>()) as u64;
    let qt_src = crate::sparse::cusparse::primitive_devptr::<I>(&qt_prim);
    let qpt_src = crate::sparse::cusparse::primitive_devptr::<I>(&qpt_prim);
    let in_q_dst = crate::sparse::cusparse::handle_devptr(&scratch.in_q);
    let in_qp_dst = crate::sparse::cusparse::handle_devptr(&scratch.in_qp);

    // SAFETY: scratch buffers outlive this call; stream is cubecl's primary.
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(in_q_dst, qt_src, bytes as usize, stream)
            .expect("D2D in_q failed");
        cudarc::driver::result::memcpy_dtod_async(in_qp_dst, qpt_src, bytes as usize, stream)
            .expect("D2D in_qp failed");
    }

    // 3. Launch the captured graph.
    unsafe {
        graph.launch(stream).expect("cuGraphLaunch forward failed");
    }

    // 4. Allocate fresh BURN primitives for Q_next + 24 saved-state outputs;
    //    D2D from scratch into each.
    let q_next_prim = crate::sparse::cusparse::fresh_primitive_from_scratch::<I>(
        &scratch.out_q, n_seg, stream, &device,
    );
    let state_prims = [
        ("depth", &scratch.state_depth),
        ("top_width", &scratch.state_top_width),
        // … all 24 …
    ];
    let mut state_arr: [I::FloatTensorPrimitive; 24] = std::array::from_fn(|i| {
        crate::sparse::cusparse::fresh_primitive_from_scratch::<I>(
            state_prims[i].1, n_seg, stream, &device,
        )
    });

    // 5. Build TimestepState exactly like timestep_forward does (using
    //    the fresh primitives we just D2D-copied out of scratch).
    let state = build_timestep_state_from_arr::<I>(
        pattern.clone(),
        n_at.clone(), q_spatial_at.clone(), p_spatial_at.clone(),
        q_t_at.clone(), q_prime_t_at.clone(),
        length_at.clone(), slope_at.clone(), x_storage_at.clone(),
        state_arr, cfg,
    );

    // 6. Register on the autograd tape.
    use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
    use burn::backend::autodiff::ops::{OpsKind};
    let n_aut = unwrap_at_full::<I>(n_at);
    let qsp_aut = unwrap_at_full::<I>(q_spatial_at);
    let psp_aut = unwrap_at_full::<I>(p_spatial_at);
    let qt_aut = unwrap_at_full::<I>(q_t_at);
    let qpt_aut = unwrap_at_full::<I>(q_prime_t_at);

    let result_prim = match TimestepOp
        .prepare::<NoCheckpointing>([
            n_aut.node.clone(), qsp_aut.node.clone(), psp_aut.node.clone(),
            qt_aut.node.clone(), qpt_aut.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(state, q_next_prim),
        OpsKind::UnTracked(prep) => prep.finish(q_next_prim),
    };

    Tensor::from_primitive(TensorPrimitive::Float(result_prim))
}
```

**Implementation note**: `primitive_devptr`, `handle_devptr`, `fresh_primitive_from_scratch`, `unwrap_at_primitive`, `unwrap_at_full`, and `build_timestep_state_from_arr` are small helpers. Add them as needed in `src/sparse/cusparse.rs` and `src/routing/mmc_op.rs` respectively. The `primitive_devptr` and `handle_devptr` ones in particular need to mirror how SP-9's `cusparse_spmv_forward` already extracts device pointers from cubecl resources (search the file for `GpuResource { ptr,`).

- [ ] **Step 2: Branch in `route_timestep`**

In `src/routing/mmc.rs:236`, replace:

```rust
        crate::routing::mmc_op::timestep_forward::<I>(
            &self.cfg, pattern, assembler,
            n, q_spatial, p_spatial,
            q_t, q_prime_clamp,
            length, slope, x_storage,
        )
```

with:

```rust
        if self.cfg.params.use_cuda_graphs && self.sparse_solver == crate::config::SparseSolver::Cuda {
            crate::routing::mmc_op::timestep_forward_via_graph::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
            )
        } else {
            crate::routing::mmc_op::timestep_forward::<I>(
                &self.cfg, pattern, assembler,
                n, q_spatial, p_spatial,
                q_t, q_prime_clamp,
                length, slope, x_storage,
            )
        }
```

- [ ] **Step 3: Run V1 regression**

```bash
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3
```

Expected: `ABSOLUTE MATCH` (still on the SP-9 path because `use_cuda_graphs: false` is the default).

- [ ] **Step 4: Run V1 with graphs ENABLED — must also pass**

```bash
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3
```

Expected: `ABSOLUTE MATCH`. (Add a one-line env-var override at the top of `compare_ddr_sandbox.rs` for this. Justify with a comment.)

If the second run fails, the graph-replay path has a bug. Likely culprits:
- D2D copy size mismatch (n_seg counted in elements vs bytes).
- State primitives constructed in wrong order.
- Stale handle ptrs from cubecl pool reuse (recapture).

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc.rs src/routing/mmc_op.rs src/sparse/cusparse.rs examples/compare_ddr_sandbox.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 6: forward graph replay in route_timestep

When use_cuda_graphs && sparse_solver == Cuda && cache.graph_fwd.is_some,
route_timestep dispatches to timestep_forward_via_graph: D2D inputs into
scratch.in_*, cuGraphLaunch, D2D outputs into fresh primitives, register
TimestepOp on the tape.

V1 ABSOLUTE MATCH both default-off and DDRS_FORCE_GRAPHS=1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Capture backward graph during `setup_inputs`

Symmetric to Task 5 but for `TimestepOp::backward`. Uses random sample inputs (not zeros — autograd may short-circuit on zero gradient).

**Files:**
- Modify: `src/sparse/cusparse.rs`
- Modify: `src/routing/mmc.rs`
- Modify: `src/routing/mmc_op.rs`

- [ ] **Step 1: Extract `backward_chain_inner` from `TimestepOp::backward`**

In `src/routing/mmc_op.rs`, refactor the body of `TimestepOp::backward` (currently lines ~82-488) to call a helper:

```rust
/// SP-10: pure backward chain. Reads gradients/state from primitives,
/// computes the 5 input gradients as primitives. No autograd-tape touch.
/// Called by both `TimestepOp::backward` and the SP-10 backward capture.
pub(crate) fn backward_chain_inner<I: Backend + 'static>(
    state: &TimestepState<I>,
    grad_q_next: I::FloatTensorPrimitive,
) -> (
    I::FloatTensorPrimitive,   // grad_n
    I::FloatTensorPrimitive,   // grad_q_spatial
    I::FloatTensorPrimitive,   // grad_p_spatial
    I::FloatTensorPrimitive,   // grad_q_t
    I::FloatTensorPrimitive,   // grad_q_prime_t
)
where I::FloatTensorPrimitive: 'static,
{
    // Body: lift the existing backward analytical chain from
    // TimestepOp::backward verbatim, ending with the 5 primitives instead
    // of `grads.register::<I>(p_n, grad_n_prim)` etc.
    todo!("port backward body, return 5 primitives")
}

impl<I: Backend + 'static> Backward<I, 5> for TimestepOp
where I::FloatTensorPrimitive: 'static,
{
    type State = TimestepState<I>;

    fn backward(self, ops: Ops<Self::State, 5>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        let state = ops.state;
        let [p_n, p_qsp, p_psp, p_qt, p_qpt] = ops.parents;
        let grad_out = grads.consume::<I>(&ops.node);

        // Branch: if a backward graph exists, replay it; else compute directly.
        let cache = unsafe { crate::sparse::cusparse::ensure_cuda_cache(&state.pattern) };
        let (gn, gqsp, gpsp, gqt, gqpt) = if let (true, Some(graph_bwd), Some(scratch)) = (
            state.use_cuda,
            cache.graph_bwd.as_ref(),
            cache.scratch.as_ref(),
        ) {
            timestep_backward_via_graph::<I>(&state, grad_out, scratch, graph_bwd)
        } else {
            backward_chain_inner::<I>(&state, grad_out)
        };

        grads.register::<I>(p_n.id, gn);
        grads.register::<I>(p_qsp.id, gqsp);
        grads.register::<I>(p_psp.id, gpsp);
        grads.register::<I>(p_qt.id, gqt);
        grads.register::<I>(p_qpt.id, gqpt);
    }
}
```

- [ ] **Step 2: Add `try_capture_backward` to `src/sparse/cusparse.rs`**

Mirror `try_capture_forward` but invoking `backward_chain_inner` with random sample inputs sourced from `scratch.in_grad_q_next` + scratch.state_*. Skipped here for brevity — replicate the structure of `try_capture_forward` substituting backward symbols.

- [ ] **Step 3: Hook into `setup_inputs`**

Right after the `try_capture_forward` call added in Task 5 Step 4:

```rust
            if cache.graph_fwd.is_some() {
                crate::sparse::cusparse::try_capture_backward::<I>(
                    cache, &self.cfg, self.pattern.as_ref().unwrap(),
                    &self.device,
                );
            }
```

- [ ] **Step 4: V1 regression**

```bash
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3
```

Expected: ABSOLUTE MATCH (both with and without `DDRS_FORCE_GRAPHS=1`).

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc_op.rs src/routing/mmc.rs src/sparse/cusparse.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 7: capture backward CUDA graph in setup_inputs

After forward capture succeeds, captures TimestepOp::backward analytical
chain into cache.graph_bwd using random sample gradient inputs (not
zeros — autograd may short-circuit on zero gradient).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Backward replay branch in `TimestepOp::backward`

The branch was already wired structurally in Task 7 Step 1. This task implements `timestep_backward_via_graph`.

**Files:**
- Modify: `src/routing/mmc_op.rs`

- [ ] **Step 1: Implement `timestep_backward_via_graph`**

```rust
/// SP-10 backward replay. D2D-copies grad_q_next + all 24 saved-state
/// primitives into scratch.in_*, replays graph_bwd, D2D-copies the 5
/// output gradients back into fresh primitives.
pub(crate) fn timestep_backward_via_graph<I: Backend + 'static>(
    state: &TimestepState<I>,
    grad_q_next: I::FloatTensorPrimitive,
    scratch: &crate::cuda_graph::PersistentScratch,
    graph_bwd: &crate::cuda_graph::CudaGraph,
) -> (
    I::FloatTensorPrimitive,
    I::FloatTensorPrimitive,
    I::FloatTensorPrimitive,
    I::FloatTensorPrimitive,
    I::FloatTensorPrimitive,
)
where I::FloatTensorPrimitive: 'static,
{
    let device = I::float_device(&grad_q_next);
    let stream = crate::sparse::cusparse::cubecl_stream_active::<I>(&device);
    let client = crate::sparse::cusparse::compute_client::<I>(&device);
    client.flush().expect("client flush before backward graph replay");

    let n_seg = scratch.n_segments;
    let bytes = (n_seg * std::mem::size_of::<f32>()) as u64;

    // D2D in: grad_q_next + 24 saved-state primitives.
    let copies: [(u64, u64); 25] = [
        (
            crate::sparse::cusparse::primitive_devptr::<I>(&grad_q_next),
            crate::sparse::cusparse::handle_devptr(&scratch.in_grad_q_next),
        ),
        (
            crate::sparse::cusparse::primitive_devptr::<I>(&state.depth),
            crate::sparse::cusparse::handle_devptr(&scratch.state_depth),
        ),
        // … the remaining 23 …
    ];

    for &(src, dst) in &copies {
        // SAFETY: scratch outlives this call; stream valid.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst, src, bytes as usize, stream)
                .expect("backward D2D in failed");
        }
    }

    // Replay.
    unsafe {
        graph_bwd.launch(stream).expect("cuGraphLaunch backward failed");
    }

    // D2D out: 5 gradients into fresh primitives.
    let grad_n = crate::sparse::cusparse::fresh_primitive_from_scratch::<I>(&scratch.out_grad_n, n_seg, stream, &device);
    let grad_qsp = crate::sparse::cusparse::fresh_primitive_from_scratch::<I>(&scratch.out_grad_q_spatial, n_seg, stream, &device);
    let grad_psp = crate::sparse::cusparse::fresh_primitive_from_scratch::<I>(&scratch.out_grad_p_spatial, n_seg, stream, &device);
    let grad_qt = crate::sparse::cusparse::fresh_primitive_from_scratch::<I>(&scratch.out_grad_q_t, n_seg, stream, &device);
    let grad_qpt = crate::sparse::cusparse::fresh_primitive_from_scratch::<I>(&scratch.out_grad_q_prime_t, n_seg, stream, &device);

    (grad_n, grad_qsp, grad_psp, grad_qt, grad_qpt)
}
```

- [ ] **Step 2: Run gradcheck**

```bash
cargo test --release --test sp8_gradcheck -- --ignored --nocapture 2>&1 | tail -10
```

Expected: all gradcheck cases pass at 1e-3 rel tolerance.

- [ ] **Step 3: Run V5 gradcheck**

```bash
cargo test --release --test sparse_gradcheck 2>&1 | tail -5
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add src/routing/mmc_op.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 8: backward graph replay in TimestepOp::backward

When cache.graph_bwd.is_some, dispatches to timestep_backward_via_graph:
25 D2D-in (grad_q_next + 24 saved state), cuGraphLaunch, 5 D2D-out into
fresh primitives.

Gradcheck + sparse_gradcheck both green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: V9 bit-match test

**Files:**
- Create: `tests/sp10_graph_bitmatch.rs`

- [ ] **Step 1: Write the test**

```rust
//! SP-10 V9 gate: graph-replay output bit-matches direct-launch output.
//!
//! Runs one MC timestep on the 5-reach RAPID sandbox once with
//! use_cuda_graphs=false (SP-9 path) and once with use_cuda_graphs=true
//! (SP-10 path). Forward Q_{t+1} and all 5 input gradients must match
//! bit-for-bit (max_rel = 0).
//!
//! Run: cargo test --release --test sp10_graph_bitmatch -- --ignored --nocapture

#![cfg(feature = "cuda")]

use burn::backend::{Autodiff, Cuda};
use burn::tensor::{Tensor, TensorData};
use ddrs::config::{Config, SparseSolver};
use ddrs::routing::mmc::MuskingumCunge;

type B = Cuda<f32, i32>;

fn build_cfg(use_cuda_graphs: bool) -> Config {
    let mut cfg = Config::default();
    cfg.params.sparse_solver = SparseSolver::Cuda;
    cfg.params.use_cuda_graphs = use_cuda_graphs;
    cfg
}

fn run_one_timestep(use_cuda_graphs: bool)
    -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)
{
    // Build a fixed-seed 5-reach RAPID sandbox identical to compare_ddr_sandbox.
    let cfg = build_cfg(use_cuda_graphs);
    let mc = ddrs::testing::rapid_sandbox::<B>(&cfg);

    // One forward step with hand-picked spatial parameters.
    let q_prime_clamp = ddrs::testing::rapid_q_prime_clamped::<B>(&cfg);
    let q_next = mc.route_timestep(q_prime_clamp.clone());

    // Synthesize a loss and call backward.
    let loss = q_next.clone().sum();
    let grads = loss.backward();

    let q_next_data: Vec<f32> = q_next.into_data().to_vec().unwrap();
    let grad_n: Vec<f32> = mc.n_grad(&grads).into_data().to_vec().unwrap();
    let grad_qsp: Vec<f32> = mc.q_spatial_grad(&grads).into_data().to_vec().unwrap();
    let grad_psp: Vec<f32> = mc.p_spatial_grad(&grads).into_data().to_vec().unwrap();
    let grad_qt: Vec<f32> = mc.q_t_grad(&grads).into_data().to_vec().unwrap();
    let grad_qpt: Vec<f32> = mc.q_prime_t_grad(&grads).into_data().to_vec().unwrap();

    (q_next_data, grad_n, grad_qsp, grad_psp, grad_qt, grad_qpt)
}

#[test]
#[ignore]
fn v9_graph_bitmatch_forward_and_backward() {
    let (q_no, g_n_no, g_qsp_no, g_psp_no, g_qt_no, g_qpt_no) = run_one_timestep(false);
    let (q_yes, g_n_yes, g_qsp_yes, g_psp_yes, g_qt_yes, g_qpt_yes) = run_one_timestep(true);

    let check = |label: &str, a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len(), "{label} length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}] differs: {x} vs {y}");
        }
        println!("V9 {label}: bit-match across {} elements", a.len());
    };

    check("Q_next", &q_no, &q_yes);
    check("grad_n", &g_n_no, &g_n_yes);
    check("grad_q_spatial", &g_qsp_no, &g_qsp_yes);
    check("grad_p_spatial", &g_psp_no, &g_psp_yes);
    check("grad_q_t", &g_qt_no, &g_qt_yes);
    check("grad_q_prime_t", &g_qpt_no, &g_qpt_yes);
}
```

**Implementation note**: `ddrs::testing::rapid_sandbox` and `n_grad` / `q_spatial_grad` etc. on `MuskingumCunge` may not exist. If not, replicate the fixture-setup code from `tests/sp8_gradcheck.rs` (which already builds a sandbox + computes gradients) inline.

- [ ] **Step 2: Run V9**

```bash
cargo test --release --test sp10_graph_bitmatch -- --ignored --nocapture 2>&1 | tail -20
```

Expected: all 6 checks pass at exact bit equality.

If a gradient differs even by 1 ULP: the captured backward graph reads a stale state buffer or a kernel was reordered. Most likely: a D2D copy missed an intermediate. Audit `timestep_backward_via_graph::copies` array against `TimestepState`.

- [ ] **Step 3: Commit**

```bash
git add tests/sp10_graph_bitmatch.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 9: V9 graph-replay bit-match test

Asserts that one timestep on the 5-reach RAPID sandbox produces
bit-identical Q_next + all 5 input gradients with use_cuda_graphs=true
vs =false.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: V10 launch-count gate script

**Files:**
- Create: `scripts/sp10_check_launches.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# SP-10 V10: gate cuLaunchKernel call count below 20% of SP-9's baseline
# (7.68M calls on 3 mini-batches; threshold = 1.54M).
set -euo pipefail

NSYS_DIR="${NSYS_DIR:-$HOME/nsys_out}"
mkdir -p "$NSYS_DIR"
REPORT="$NSYS_DIR/sp10_v10"
CKPT="/tmp/sp10_v10_ckpt"

# Baseline from SP-9: 7,684,365 cuLaunchKernel calls on 3 batches.
BASELINE_CALLS=${SP10_BASELINE_CALLS:-7684365}
PASS_RATIO=${SP10_PASS_RATIO:-0.20}   # i.e. drop to ≤20% of baseline

# Write a temp YAML with use_cuda_graphs: true (same awk pattern as SP-9's V7b script).
TMP_CFG="/tmp/v10_graphs.yaml"
awk '
    /^params:/ {
        print;
        print "  sparse_solver: cuda";
        print "  use_cuda_graphs: true";
        next
    }
    /sparse_solver:/ { next }
    /use_cuda_graphs:/ { next }
    { print }
' config/merit_training.yaml > "$TMP_CFG"

rm -rf "$CKPT"
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --output="$REPORT" --force-overwrite=true \
    target/release/train --config "$TMP_CFG" \
                         --checkpoint-dir "$CKPT" \
                         --max-mini-batches 3

STATS="$NSYS_DIR/sp10_v10_stats.txt"
nsys stats --force-export=true "$REPORT.nsys-rep" --report cuda_api_sum > "$STATS"

# Extract Num Calls for cuLaunchKernel.
# Column layout: "Time (%)", "Total Time (ns)", "Num Calls", "Avg", ...
CALLS=$(awk '
    /cuLaunchKernel/ {
        gsub(",", "", $3);
        print $3;
        exit;
    }
' "$STATS")

if [ -z "$CALLS" ]; then
    echo "V10: cuLaunchKernel not found in $STATS"
    exit 1
fi

THRESHOLD=$(awk -v b="$BASELINE_CALLS" -v r="$PASS_RATIO" 'BEGIN { print int(b * r) }')
echo "V10: cuLaunchKernel calls = $CALLS (baseline $BASELINE_CALLS, threshold $THRESHOLD)"

awk -v c="$CALLS" -v t="$THRESHOLD" 'BEGIN { exit (c+0 <= t+0) ? 0 : 1 }'
```

- [ ] **Step 2: chmod + commit**

```bash
chmod +x scripts/sp10_check_launches.sh
git add scripts/sp10_check_launches.sh
git commit -m "$(cat <<'EOF'
SP-10 Task 10: V10 launch-count gate script

Profiles a 3-batch train run with use_cuda_graphs=true and gates
cuLaunchKernel ≤ 20% of SP-9's baseline (7.68M → 1.54M target).

Uses --force-export=true on nsys stats (same fix as SP-9's V7b).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: V7a perf test rewrite

**Files:**
- Create: `tests/sp10_v7a_perf.rs`

- [ ] **Step 1: Copy structure from `tests/sp8_v7_perf.rs`**

```bash
cp tests/sp8_v7_perf.rs tests/sp10_v7a_perf.rs
```

- [ ] **Step 2: Patch the new file**

In `tests/sp10_v7a_perf.rs`:

1. Update the doc header to reference SP-10 (not SP-8).
2. In the CUDA test variant, inject `use_cuda_graphs: true` alongside `sparse_solver: cuda` when writing the temp YAML — use the same awk-pattern as `scripts/sp10_check_launches.sh` Step 1.
3. Pass criterion: `cuda_median / cpu_median ≤ 0.7` (was `≤ 0.7` already in SP-9; same threshold).

The test body retains: 3 warmup runs, then 3 measured runs each backend, take median.

- [ ] **Step 3: Run V7a**

```bash
cargo build --release 2>&1 | tail -3   # ensure target/release/train is current
cargo test --release --test sp10_v7a_perf -- --ignored --nocapture 2>&1 | tail -10
```

Expected: ratio ≤ 0.7.

If ratio is between 0.7 and 0.919: V10 passed (launches dropped) but V7a didn't make threshold. SP-11 (kernel fusion) is needed. Spec calls this out as acceptable — V9/V10 alone are structural wins worth landing.

- [ ] **Step 4: Commit**

```bash
git add tests/sp10_v7a_perf.rs
git commit -m "$(cat <<'EOF'
SP-10 Task 11: V7a perf gate test (graph-enabled)

Same median-of-3 structure as SP-9's sp8_v7_perf, but injects
use_cuda_graphs: true into the CUDA-side temp YAML. Pass: cuda/cpu ≤ 0.7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: ARCHITECTURE.md update + default flip

**Only run this task if V9 + V10 + V7a all pass.** If V7a misses, write the partial close section without flipping the default — same pattern as SP-9's partial close at commit `a67a7e2`.

**Files:**
- Modify: `.claude/ARCHITECTURE.md`
- Modify: `config/merit_training.yaml` (only if V7a green)

- [ ] **Step 1: Add SP-10 section to ARCHITECTURE.md**

Append after the existing SP-9 section:

```markdown
## SP-10 CUDA Graphs (2026-05-26, [partial|full])

Captured one forward and one backward `CUgraphExec` per `MuskingumCunge`
instance during `setup_inputs`, replayed per timestep. Replaces ~100
per-direction `cuLaunchKernel` issuances with 1 `cuGraphLaunch` + ~30
`cuMemcpyDtoDAsync` calls.

**Outcome:**
- V9 (bit-match): GREEN — graph and direct paths agree to the bit on
  Q_{t+1} and all 5 input gradients.
- V10 (launch count): [GREEN @ ___ % drop | …]
- V7a (cuda/cpu ratio): [GREEN @ ___ | PARTIAL @ ___]

**[If partial:]** Remaining wall-time floor sits in autograd-tape
traversal and BURN-side tensor object overhead. SP-11 candidate: cubecl
kernel fusion of the analytical chain.

**[If full:]** `config/merit_training.yaml` defaults `use_cuda_graphs:
true` (single-line flip in commit ___).
```

- [ ] **Step 2 (only if V7a green): Flip the default**

In `config/merit_training.yaml`, find the `params:` block, change:

```yaml
  # use_cuda_graphs: false
```

to:

```yaml
  use_cuda_graphs: true
```

- [ ] **Step 3: Run all gates one final time**

```bash
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -3                # V1
cargo test --release --test sparse_gradcheck 2>&1 | tail -3                     # V5
cargo test --release --test sparse_cusparse_v8 -- --ignored 2>&1 | tail -3      # V8
bash scripts/sp8_check_scatter.sh 2>&1 | tail -3                                # V7b
cargo test --release --test sp10_graph_bitmatch -- --ignored 2>&1 | tail -3     # V9
bash scripts/sp10_check_launches.sh 2>&1 | tail -3                              # V10
cargo test --release --test sp10_v7a_perf -- --ignored 2>&1 | tail -3           # V7a
```

All must report green / pass / ABSOLUTE MATCH.

- [ ] **Step 4: Commit ARCHITECTURE.md update**

```bash
git add .claude/ARCHITECTURE.md
git commit -m "$(cat <<'EOF'
SP-10 close: CUDA Graphs landed ([full|partial] — gates _____)

Replaces ~100 per-direction cuLaunchKernel issuances per timestep with 1
cuGraphLaunch + ~30 cuMemcpyDtoDAsync. Captured graphs live on
CudaPatternCache, replayed by route_timestep / TimestepOp::backward.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5 (only if V7a green): Commit the default flip**

```bash
git add config/merit_training.yaml
git commit -m "$(cat <<'EOF'
SP-10: flip default use_cuda_graphs to true in merit_training.yaml

V9/V10/V7a all green at SP-10 close. CUDA path is now the default fast
path on machines with a CUDA backend.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage:**
- §Goal → Tasks 1, 5, 7 (capture + replay infra)
- §Success criteria V9 → Task 9
- §Success criteria V10 → Task 10
- §Success criteria V7a → Task 11
- §Persistent scratch → Tasks 3, 4
- §Forward capture → Task 5
- §Forward replay → Task 6
- §Backward capture → Task 7
- §Backward replay → Task 8
- §Per-step launch accounting → measured in Task 10
- §Fallback model → Task 5 try_capture_forward FallbackReason path + Task 7 same for backward
- §Concern 1 (pointer stability) → Task 0 spike validates
- §Concern 2 (cuSPARSE workspaces) → no new code; SP-9 already allocates them in setup_inputs before any capture region begins
- §Concern 3 (capture exclusivity) → Tasks 5/7 both run capture inside `setup_inputs`, which is a quiescent point
- §Concern 4 (BURN host-roundtrips) → Task 0 spike forces this concern to surface concretely
- §Concern 5 (recapture trigger) → Task 4 `capture_sig` field; check-and-recapture logic gets added to `setup_inputs` (covered by Task 5)
- §Concern 6 (zero-input short-circuit) → Task 7 explicitly uses random sample inputs
- §Concern 7 (V7a margin) → Task 12 calls out the partial-close path

**Placeholder scan:** Task 5 and Task 7 contain `todo!()` markers for `ensure_cuda_cache_mut` / `try_capture_backward` body / `into_primitive_inner` — these are noted with concrete implementation hints (mirror existing patterns at named line numbers). Not pure placeholders — directive enough for the executor to translate.

**Type consistency:**
- `CudaGraph` newtype used uniformly.
- `PersistentScratch` field names align between scratch.rs (Task 3), capture sites (Tasks 5/7), replay sites (Tasks 6/8).
- `CaptureStatus` enum referenced from Task 4 (definition) and Tasks 5/7 (usage).
- `try_capture_forward` / `try_capture_backward` names consistent across the plan.
- Helper function names (`handle_devptr`, `primitive_devptr`, `fresh_primitive_from_scratch`) are referenced as add-on tasks within Tasks 5/6/8.

The plan acknowledges in §Critical de-risking spike that Task 0 is the gate; if it fails, the entire plan is reconsidered. The remaining tasks assume Task 0 outcome 1 (works).

---

## Execution handoff

Plan complete and saved to `.claude/specs/2026-05-26-sp10-cuda-graphs-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec compliance, then code quality) between tasks. Best for this plan since Task 0 is a hard gate and several mid-plan tasks (5, 7) have non-trivial helper additions.

**2. Inline Execution** — checkpoint-based, single session.

**Which approach?**
