# SP-6 cuSPARSE GPU Triangular Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CPU-bound triangular solve (forward, backward, and per-nnz grada scatter) with a GPU-native cuSPARSE path on the BURN CUDA backend, gated by a config flag. V5 bit-matches CPU vs CUDA outputs at the f32 floor.

**Architecture:** Refactor `src/sparse.rs` into a small module. Add `SparseSolver::{Cpu, Cuda}` config enum, threaded through `MuskingumCunge` to `triangular_csr_solve`. CPU path is preserved bit-for-bit and remains the default. A `CudaPatternCache` lazy-attached to `CsrPattern` via `OnceCell` holds cuSPARSE descriptors + workspace. Two SpSV descriptors (forward lower-tri NON_TRANSPOSE; backward TRANSPOSE) reuse the same matrix descriptor. The per-nnz grada gradient stays on device via a small custom cubecl kernel.

**Tech Stack:** Rust 1.94+, BURN 0.21 (`Cuda<f32, i32>`, autodiff custom `Backward`), `cudarc 0.19` with `cusparse` feature (verified present in `~/.cargo/registry/.../cudarc-0.19.7/src/cusparse/`), existing `cubecl-cuda` (for stream sharing + grada kernel).

**Spec:** `.claude/specs/2026-05-20-sp6-cusparse-gpu-solve-design.md`
**Parent:** `.claude/specs/2026-05-17-train_and_test-replication-design.md`

**Verification:** V5 — synthetic 100-reach lower-triangular pattern; forward `x`, backward `gradb`, and `grada` per-nnz match between `NdArray<f32>` + `Cpu` solver and `Cuda<f32, i32>` + `Cuda` solver within `1e-5` relative.

---

## Conventions for this plan

- All forward code stays generic over `B: Backend`. Tests pin `NdArray<f32>` for the CPU path and `Cuda<f32, i32>` for the GPU path (skip the latter cleanly if no CUDA device).
- `CsrSolveState`, `SavedX`, `SparseSolver`, `CudaPatternCache`, `cuda_cache` — used consistently across tasks.
- Cite DDR / cuSPARSE references in doc comments where relevant.
- Pre-existing clippy lints in routing-core code are out of scope (same precedent as SP-1..5). New code adds zero clippy warnings.
- No commit amends. New commit per task.
- BURN tensor primitive raw-pointer access requires `unsafe`; every block has a `SAFETY:` comment.

---

## File Structure

**Created:**

- `src/sparse/mod.rs` — existing `src/sparse.rs` content moved here unchanged
- `src/sparse/dispatch.rs` — `forward_primitive` / `backward_primitive` runtime branch
- `src/sparse/cusparse.rs` — `CudaPatternCache`, `cusparse_forward`, `cusparse_backward`, `grada_kernel`
- `tests/sparse_cusparse_v5.rs` — V5 bit-match test, skips on CPU-only hosts

**Modified:**

- `src/config.rs` — `SparseSolver` enum + `Params::sparse_solver` + `ParamsRaw::sparse_solver`
- `src/sparse.rs` — deleted (content moved to `src/sparse/mod.rs`)
- `src/routing/mmc.rs` — store `sparse_solver` on `MuskingumCunge`, pass to `triangular_csr_solve` call sites (2 sites: hotstart + per-timestep)
- `src/lib.rs` — no change to public re-exports (`crate::sparse::*` still works)
- `Cargo.toml` — explicit `cudarc = { version = "0.19", features = ["cusparse"] }` dep
- `config/merit_training.yaml` — commented-out `sparse_solver: cuda` example block

---

### Task 1: Config — `SparseSolver` enum + `Params::sparse_solver`

**Files:**
- Modify: `src/config.rs`
- Modify: `config/merit_training.yaml`

- [ ] **Step 1: Add `SparseSolver` enum**

Append to `src/config.rs` (above `pub struct Params`):

```rust
/// Selects the backend implementation of the CSR triangular solve in
/// `MuskingumCunge`. `Cuda` opts into the cuSPARSE path when the runtime
/// backend is `burn::backend::Cuda`; on other backends the solver silently
/// falls back to `Cpu` (logged once at WARN).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SparseSolver {
    #[default]
    Cpu,
    Cuda,
}
```

- [ ] **Step 2: Add `sparse_solver` field to `Params`**

In `src/config.rs`, add the field at the end of `pub struct Params`:

```rust
pub struct Params {
    pub parameter_ranges: ParameterRanges,
    pub log_space_parameters: Vec<String>,
    pub defaults: HashMap<String, f32>,
    pub attribute_minimums: AttributeMinimums,
    pub tau: u32,
    pub sparse_solver: SparseSolver,
}
```

Update `Default for Params` to set `sparse_solver: SparseSolver::default()` (== `Cpu`).

- [ ] **Step 3: Extend `ParamsRaw` for YAML parsing**

In `src/config.rs::ParamsRaw`, add:

```rust
struct ParamsRaw {
    // existing fields ...
    tau: Option<u32>,
    sparse_solver: Option<String>,
}
```

In `From<ParamsRaw> for Params`, after the existing `p.tau = ...` line, append:

```rust
p.sparse_solver = match r.sparse_solver.as_deref() {
    Some("cuda") | Some("CUDA") => SparseSolver::Cuda,
    Some("cpu") | Some("CPU") | None => SparseSolver::Cpu,
    Some(other) => panic!("unknown sparse_solver: {other:?} (expected \"cpu\" or \"cuda\")"),
};
```

- [ ] **Step 4: Add `sparse_solver` to YAML test + production config**

In `src/config.rs::tests::loads_merit_training_yaml`, add:

```rust
assert_eq!(cfg.params.sparse_solver, ddrs::config::SparseSolver::Cpu);
```

(Or use the in-module path `SparseSolver::Cpu` since the test is inside the same module.)

In `config/merit_training.yaml`, add a commented example under `params:`:

```yaml
params:
  # ... existing ...
  # sparse_solver: cuda    # opt-in for GPU cuSPARSE solve (requires CUDA backend)
```

- [ ] **Step 5: Build + test**

```
cargo test --lib config 2>&1 | tail -10
```

Expected: existing tests pass + the new `sparse_solver` assertion passes.

- [ ] **Step 6: Commit**

```
git add src/config.rs config/merit_training.yaml
git commit -m "Add SparseSolver enum + Params.sparse_solver (default Cpu)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Refactor `src/sparse.rs` into `src/sparse/` module (pure structural)

**Files:**
- Create: `src/sparse/mod.rs` (content of current `src/sparse.rs`, unchanged)
- Delete: `src/sparse.rs`

No behavior change. Single commit so subsequent tasks can split cleanly.

- [ ] **Step 1: Move the file**

```
mkdir -p src/sparse
git mv src/sparse.rs src/sparse/mod.rs
```

- [ ] **Step 2: Verify no other path references `src/sparse.rs`**

```
grep -rn "src/sparse.rs" src/ tests/ examples/ .claude/ 2>&1 | tail -10
```

Expected: no hits. If hits exist in `.claude/` (docs), leave them — those are descriptive, not load-bearing.

- [ ] **Step 3: Build + run all tests (must still pass identically)**

```
cargo build --lib 2>&1 | tail -5
cargo test --lib 2>&1 | tail -5
cargo test --test sparse_gradcheck 2>&1 | tail -5
```

Expected: lib builds clean; existing tests pass with the same counts as before.

- [ ] **Step 4: Commit**

```
git add -A src/sparse
git commit -m "Move src/sparse.rs to src/sparse/mod.rs (no behavior change)

Preparing for cuSPARSE submodule split per SP-6 plan. Public API on
crate::sparse::* is preserved bit-for-bit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `cudarc` direct dep with cuSPARSE feature

**Files:**
- Modify: `Cargo.toml`

Verified during plan-writing: `cudarc 0.19.7` exposes a `cusparse` feature that depends on `driver`. Module path: `cudarc::cusparse::*`.

- [ ] **Step 1: Add the dep**

In `Cargo.toml`, after the `burn-cuda = ...` line, append:

```toml
# Direct cuSPARSE bindings for the GPU triangular solve (SP-6).
# cudarc is already a transitive dep of cubecl-cuda; declaring it
# explicitly lets us enable the `cusparse` feature.
cudarc = { version = "0.19", default-features = false, features = ["cuda-13020", "cusparse"] }
```

The `cuda-13020` feature pins the CUDA toolkit version BURN's cubecl already targets — verify by inspecting `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.7/Cargo.toml` and matching the cuda version cubecl-cuda already requests transitively. If cubecl pulls a different `cuda-XXX` feature, mirror that here.

- [ ] **Step 2: Build (verify cudarc compiles on this host)**

```
cargo build --lib 2>&1 | tail -10
```

Expected: clean. If cudarc fails to find `libcusparse.so` at link time, point `LD_LIBRARY_PATH` at the CUDA toolkit's `lib64` (typically `/usr/local/cuda/lib64`).

If the build fails because of a feature-version mismatch with cubecl, escalate as NEEDS_CONTEXT with the exact compile error.

- [ ] **Step 3: Smoke-test the import**

Add a temporary `examples/cudarc_smoke.rs`:

```rust
//! Temporary smoke test that cudarc cusparse module is importable.
//! Delete after Task 3 completes.

use cudarc::cusparse;

fn main() {
    // Just naming the module forces the link.
    let _ = std::mem::size_of::<cusparse::CudaSparse>();
    println!("cudarc::cusparse imports cleanly");
}
```

Build:

```
cargo build --example cudarc_smoke 2>&1 | tail -5
```

Expected: clean compile. If `cusparse::CudaSparse` is the wrong type name (the cudarc API may use `CudaSparseHandle` or `Sparse`), inspect `~/.cargo/registry/.../cudarc-0.19.7/src/cusparse/mod.rs` and adapt — the goal is just to prove the feature compiles. Delete the example after the smoke passes.

```
rm examples/cudarc_smoke.rs
```

- [ ] **Step 4: Commit**

```
git add Cargo.toml Cargo.lock
git commit -m "Add cudarc dep with cuSPARSE feature for SP-6 GPU solve

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `SavedX<B>` enum + `use_cuda` parameter (CPU path only)

**Files:**
- Modify: `src/sparse/mod.rs`

Add the type changes that the dispatch shim needs, with the CPU implementation routing through unchanged. After this task, the CPU path is bit-equivalent to before; the only difference is types are now polymorphic across `Cpu`/`Cuda` variants.

- [ ] **Step 1: Add `SavedX<B>` enum**

In `src/sparse/mod.rs`, above `struct CsrSolveState<B: Backend>`:

```rust
/// Forward-solve output saved for the backward pass. The CPU path stores a
/// host-side `Vec<f32>` (cheap to share via `Arc`); the future GPU path
/// stores a `B::FloatTensorPrimitive` so `x` stays on device.
pub(crate) enum SavedX<B: Backend> {
    Cpu(Arc<Vec<f32>>),
    Cuda(B::FloatTensorPrimitive),
}
```

- [ ] **Step 2: Update `CsrSolveState`**

Change the existing `x: Arc<Vec<f32>>` to:

```rust
struct CsrSolveState<B: Backend> {
    a_values: B::FloatTensorPrimitive,
    x: SavedX<B>,
    pattern: Arc<CsrPattern>,
    use_cuda: bool,
}
```

- [ ] **Step 3: Update `CsrSolveOp::backward` to dispatch on `use_cuda` (still CPU only)**

In `impl<B: Backend> Backward<B, 2> for CsrSolveOp`, replace the body with:

```rust
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

    // Pull x to host (CPU path only — Cuda variant will be handled in Task 10).
    let x_host: Vec<f32> = match x {
        SavedX::Cpu(arc) => (*arc).clone(),
        SavedX::Cuda(prim) => primitive_to_vec::<B>(prim),
    };
    let a_data: Vec<f32> = primitive_to_vec::<B>(a_values);
    let grad_out_data: Vec<f32> = primitive_to_vec::<B>(grad_out);

    // CPU backward solve. Cuda dispatch added in Task 10.
    let _ = use_cuda; // unused on CPU path; suppress warning.
    let gradb_data = back_sub_upper_transposed(&pattern, &a_data, &grad_out_data);

    if let Some(p_b) = parent_b {
        let gradb = B::float_from_data(TensorData::from(gradb_data.as_slice()), &device);
        grads.register::<B>(p_b.id, gradb);
    }

    if let Some(p_a) = parent_a {
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
```

- [ ] **Step 4: Update `triangular_csr_solve` signature + saved state**

Change the signature to take `use_cuda: bool`:

```rust
pub fn triangular_csr_solve<I: Backend>(
    pattern: &Arc<CsrPattern>,
    a_values: Tensor<Autodiff<I>, 1>,
    b: Tensor<Autodiff<I>, 1>,
    use_cuda: bool,
) -> Tensor<Autodiff<I>, 1> {
    // ... existing extraction of a_at, b_at, device ...

    let (out_prim, x_data) =
        forward_primitive::<I>(pattern, &a_at.primitive, &b_at.primitive, &device);

    let result = match CsrSolveOp
        .prepare::<NoCheckpointing>([a_at.node.clone(), b_at.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let state = CsrSolveState::<I> {
                a_values: a_at.primitive.clone(),
                x: SavedX::Cpu(Arc::new(x_data)),
                pattern: pattern.clone(),
                use_cuda,
            };
            prep.finish(state, out_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(out_prim),
    };

    Tensor::from_primitive(TensorPrimitive::Float(result))
}
```

(`forward_primitive` keeps its current signature for now — Task 5 plumbs `use_cuda` into it.)

- [ ] **Step 5: Update test call sites + tests**

`triangular_csr_solve` is called by `routing/mmc.rs` (Task 5 covers those) and by `tests/sparse_gradcheck.rs`. Update the test file:

```
grep -n "triangular_csr_solve" tests/sparse_gradcheck.rs
```

For each call site (likely 1–2), append `, false` to the argument list.

- [ ] **Step 6: Update `routing/mmc.rs` call sites (placeholder false)**

```
grep -n "triangular_csr_solve(" src/routing/mmc.rs
```

For each call (2 sites: hotstart in `setup_inputs` + the per-timestep `route_timestep`), append `, false`. Task 5 replaces `false` with the real flag from cfg.

- [ ] **Step 7: Build + run sandbox regression**

```
cargo build --lib 2>&1 | tail -5
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: clean compile; sparse_gradcheck tests pass; sandbox reports ABSOLUTE MATCH.

- [ ] **Step 8: Commit**

```
git add src/sparse/mod.rs src/routing/mmc.rs tests/sparse_gradcheck.rs
git commit -m "Add SavedX<B> enum + use_cuda flag (CPU path unchanged)

Threading the dispatch infrastructure ahead of the cuSPARSE
implementation. The CPU path keeps SavedX::Cpu(Arc<Vec<f32>>) and the
existing CPU forward/backward solves; the Cuda variant is wired in
later tasks of SP-6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Pipe `sparse_solver` from `Config` through `MuskingumCunge` into solve calls

**Files:**
- Modify: `src/routing/mmc.rs`

- [ ] **Step 1: Cache the flag on `MuskingumCunge`**

In `src/routing/mmc.rs`, add a field to `pub struct MuskingumCunge<I: Backend>`:

```rust
pub struct MuskingumCunge<I: Backend> {
    // ... existing fields ...
    sparse_solver: SparseSolver,
}
```

Add the import at top:

```rust
use crate::config::SparseSolver;
```

In `impl<I: Backend> MuskingumCunge<I>::new`, store the flag:

```rust
pub fn new(cfg: Config, device: I::Device) -> Self {
    let sparse_solver = cfg.params.sparse_solver;
    // ... existing init ...
    Self {
        // ... existing field assignments ...
        sparse_solver,
    }
}
```

- [ ] **Step 2: Pass `use_cuda` into the two `triangular_csr_solve` calls**

Find both call sites:

```
grep -n "triangular_csr_solve" src/routing/mmc.rs
```

For each call, replace the trailing `, false` from Task 4 with `, self.sparse_solver == SparseSolver::Cuda`. Specifically:

In the hotstart inside `setup_inputs`:
```rust
let q0 = triangular_csr_solve::<I>(
    pattern,
    a_values,
    q_prime_0,
    self.sparse_solver == SparseSolver::Cuda,
).clamp_min(self.cfg.params.attribute_minimums.discharge);
```

In `route_timestep`:
```rust
let solution = triangular_csr_solve::<I>(
    pattern,
    a_values,
    b,
    self.sparse_solver == SparseSolver::Cuda,
);
```

- [ ] **Step 3: Build + regression**

```
cargo build --lib 2>&1 | tail -5
cargo test --lib 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: clean; all tests pass; ABSOLUTE MATCH.

- [ ] **Step 4: Commit**

```
git add src/routing/mmc.rs
git commit -m "Plumb sparse_solver from Config into triangular_csr_solve

MuskingumCunge::new caches cfg.params.sparse_solver; both call sites
(setup_inputs hotstart + route_timestep) pass the resulting bool to
triangular_csr_solve. The CPU path remains the default (and the only
implemented backend) until SP-6 Task 9.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: SPIKE — extract device pointer from `B::FloatTensorPrimitive` for `Cuda` backend

**Files:**
- Create: `src/sparse/cusparse.rs` (skeleton with the extraction helper only)
- Modify: `src/sparse/mod.rs` (add `mod cusparse;`)
- Create: `tests/cusparse_ptr_spike.rs`

This is the most uncertain part of the plan. Goal: from `B::FloatTensorPrimitive` where `B = burn::backend::Cuda<f32, i32>`, get a `cudarc::driver::CudaSlice<f32>` (or equivalent raw `*mut f32` + length) WITHOUT copying through the host.

If this proves impossible cleanly, **escalate as BLOCKED** — without it the rest of SP-6 falls back to materializing through CPU, which defeats the purpose.

- [ ] **Step 1: Create `src/sparse/cusparse.rs` with the spike**

```rust
//! cuSPARSE-backed forward/backward triangular solves for the BURN Cuda backend.
//!
//! Task 6 is the spike that proves we can reach a `cudarc::driver::CudaSlice`
//! from a `burn::backend::Cuda` tensor primitive. Later tasks build the
//! cusparseSpSV pipeline on top.

use burn::tensor::backend::Backend;

/// Type-erased view into a CUDA tensor as a raw device slice.
///
/// `len` is the element count (not bytes). The pointer aliases the BURN
/// tensor — the caller MUST NOT drop the source primitive while this view
/// is alive.
pub(crate) struct CudaView {
    pub ptr: *mut f32,
    pub len: usize,
}

/// Borrow a `Cuda<f32, i32>` float primitive as a raw device pointer.
///
/// SAFETY: returned pointer is owned by the BURN tensor; do not free.
/// Lifetime is tied to `prim`. Panics if `B` is not the CUDA backend.
pub(crate) fn primitive_as_cuda_view<B: Backend>(
    prim: &B::FloatTensorPrimitive,
) -> Option<CudaView> {
    // Strategy: downcast B::FloatTensorPrimitive via TypeId comparison.
    // burn-cuda 0.21 wraps tensors via cubecl::CubeTensor over the cubecl-cuda
    // ComputeServer. The primitive type is something like
    //   `cubecl::tensor::Tensor<CudaRuntime>` where CudaRuntime wraps cudarc.
    //
    // The cleanest stable-ish access is via cubecl's `Handle<R>` → server →
    // `to_slice` style. The exact path is API-version-specific; expect to
    // read burn-cuda-0.21.0/src/lib.rs and cubecl-cuda-0.10.0/src/runtime.rs.
    //
    // Implementer: fill in. If the cleanest route is `unsafe` cast through
    // the cubecl handle, write a SAFETY comment naming the invariants.
    todo!("extract CudaSlice / raw pointer from B::FloatTensorPrimitive on Cuda backend")
}
```

- [ ] **Step 2: Wire the module**

In `src/sparse/mod.rs`, add at the top after the existing module-level docs:

```rust
pub(crate) mod cusparse;
```

(Keep `pub(crate)` — the module is internal to the sparse crate path.)

- [ ] **Step 3: Write the spike test**

Create `tests/cusparse_ptr_spike.rs`:

```rust
//! Spike: verify we can borrow a Cuda tensor as a raw device pointer.
//! Skips cleanly on CPU-only hosts.

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    #[test]
    fn round_trip_via_pointer() {
        type B = burn::backend::Cuda<f32, i32>;
        // CudaDevice::default() panics if no device is present; gate the
        // entire test on a probe.
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let cuda_available = std::panic::catch_unwind(|| {
            let _t = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0], &device);
        }).is_ok();
        if !cuda_available {
            eprintln!("skipping: no CUDA device");
            return;
        }

        let t = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0, 4.0], &device);
        let prim = t.clone().into_primitive().tensor();

        // Use the crate-private extractor via a public test entry point we
        // expose temporarily through `pub fn` in src/sparse/cusparse.rs OR
        // by calling through dispatch (preferred). For the spike, expose a
        // pub testing-only fn `__spike_extract_len` that returns the slice
        // length without exposing the pointer to user code.
        let len = ddrs::sparse::cusparse::__spike_extract_len::<B>(&prim);
        assert_eq!(len, 4, "extracted len does not match tensor length");
    }
}
```

And add to `src/sparse/cusparse.rs` for the spike only:

```rust
/// Test-only entry point that returns the device-slice length we extracted.
/// Removed after Task 6 — the production path consumes `CudaView` directly.
#[doc(hidden)]
pub fn __spike_extract_len<B: Backend>(prim: &B::FloatTensorPrimitive) -> usize {
    primitive_as_cuda_view::<B>(prim)
        .expect("expected Cuda backend")
        .len
}
```

- [ ] **Step 4: Implement `primitive_as_cuda_view`**

Replace the `todo!()` with the actual extraction. **Read these source files** to find the right path:

```
ls ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-cuda-0.21.0/src/
ls ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cuda-0.10.0/src/
```

Typical pattern (verify against the source):

```rust
pub(crate) fn primitive_as_cuda_view<B: Backend>(
    prim: &B::FloatTensorPrimitive,
) -> Option<CudaView> {
    use std::any::TypeId;
    if TypeId::of::<B>() != TypeId::of::<burn::backend::Cuda<f32, i32>>() {
        return None;
    }
    // SAFETY: TypeId comparison above guarantees B == Cuda<f32, i32>, so the
    // FloatTensorPrimitive is the cubecl-cuda CubeTensor backed by a CUDA
    // device pointer. We borrow the pointer for the lifetime of `prim`; the
    // caller must not drop `prim` while a CudaView from it is alive.
    let view = unsafe {
        // Downcast prim → concrete cubecl CubeTensor type via transmute_copy
        // (zero-sized type marker; both types have identical layout under
        // the TypeId check). Use the cubecl handle API to read the device
        // ptr and length.
        let cube_tensor: &cubecl_cuda::CudaTensor /* or correct type */ =
            std::mem::transmute(prim);
        let handle = cube_tensor.handle();
        let server = /* obtain compute server */;
        let raw_ptr = server.read_ptr(&handle);
        let len = cube_tensor.shape().num_elements();
        CudaView { ptr: raw_ptr as *mut f32, len }
    };
    Some(view)
}
```

The exact types (`cubecl_cuda::CudaTensor`, `read_ptr`) are placeholders — verify against the actual cubecl-cuda source. If the cleanest API is `cudarc::driver::CudaSlice<f32>` instead of a raw pointer, change `CudaView` to wrap that.

- [ ] **Step 5: Run the spike**

```
cargo test --test cusparse_ptr_spike round_trip_via_pointer 2>&1 | tail -20
```

Expected: test passes on a CUDA host (returns len=4), or prints `skipping: no CUDA device` and passes vacuously.

If you cannot make this work in 60-90 minutes of source reading + iteration, **STOP and report BLOCKED** with: the exact paths you read, the API surfaces you found, and what was missing.

- [ ] **Step 6: Commit**

```
git add src/sparse/cusparse.rs src/sparse/mod.rs tests/cusparse_ptr_spike.rs
git commit -m "Add Cuda tensor primitive → device pointer extraction (SP-6 spike)

The primitive_as_cuda_view helper bridges burn-cuda's FloatTensorPrimitive
to a raw f32 device slice, enabling the cuSPARSE solve in subsequent
tasks to operate on BURN tensors without host round-trips.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If the extraction works, proceed. If it doesn't, STOP and report.

---

### Task 7: SPIKE — share cubecl's CUDA stream with cuSPARSE

**Files:**
- Modify: `src/sparse/cusparse.rs`
- Modify: `tests/cusparse_ptr_spike.rs`

cuSPARSE needs a `cudaStream_t`. cubecl owns a stream per device. If we use a separate stream, every solve requires `cudaStreamSynchronize` to interop with subsequent BURN ops — that defeats the perf win.

- [ ] **Step 1: Add a stream accessor to `src/sparse/cusparse.rs`**

```rust
use cudarc::driver::sys::CUstream;

/// Returns the cubecl-managed CUDA stream for the current device, suitable
/// for passing to `cusparseSetStream`. Panics if the runtime backend is
/// not Cuda.
///
/// SAFETY: returned handle is owned by cubecl-cuda. The caller must not
/// destroy the stream. The handle is valid for as long as the device
/// stays initialized (effectively the process lifetime).
pub(crate) fn cubecl_cuda_stream<B: Backend>(device: &B::Device) -> CUstream {
    // Implementer: read burn-cuda-0.21.0/src/lib.rs and
    // cubecl-cuda-0.10.0/src/runtime.rs to find how the active stream is
    // exposed. Typical path:
    //   1. Get cubecl ComputeClient from device.
    //   2. Get the runtime's stream handle.
    //   3. Return as CUstream.
    todo!("expose cubecl's cuda stream as CUstream")
}
```

- [ ] **Step 2: Extend the spike test**

In `tests/cusparse_ptr_spike.rs`, add:

```rust
#[test]
fn cubecl_stream_is_non_null() {
    type B = burn::backend::Cuda<f32, i32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let cuda_available = std::panic::catch_unwind(|| {
        let _t = burn::tensor::Tensor::<B, 1>::from_floats([0.0_f32], &device);
    }).is_ok();
    if !cuda_available {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let stream = ddrs::sparse::cusparse::__spike_get_stream::<B>(&device);
    assert!(!stream.is_null(), "cubecl returned a null stream");
}
```

And in `src/sparse/cusparse.rs`:

```rust
#[doc(hidden)]
pub fn __spike_get_stream<B: Backend>(device: &B::Device) -> cudarc::driver::sys::CUstream {
    cubecl_cuda_stream::<B>(device)
}
```

- [ ] **Step 3: Implement `cubecl_cuda_stream`**

Read cubecl-cuda's runtime source for the stream accessor. Fall back: if cubecl does not expose its stream publicly, create a separate stream via `cudaStreamCreate` and accept the `cudaStreamSynchronize` cost. Document the choice in the function's doc comment.

If you must create a separate stream, add a `Drop` impl or a `OnceCell<CudaStream>` at module scope so it's only created once per process.

- [ ] **Step 4: Run the spike**

```
cargo test --test cusparse_ptr_spike cubecl_stream_is_non_null 2>&1 | tail -10
```

Expected: pass on CUDA host, skip otherwise.

- [ ] **Step 5: Commit**

```
git add src/sparse/cusparse.rs tests/cusparse_ptr_spike.rs
git commit -m "Expose CUDA stream for cuSPARSE interop (SP-6 spike)

cubecl_cuda_stream returns the runtime stream so cusparseSetStream
can interleave with BURN's compute. If cubecl's stream is private,
the fallback creates a dedicated stream once per process (perf
degraded; see doc comment).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `CudaPatternCache` + lazy `OnceCell` on `CsrPattern`

**Files:**
- Modify: `src/sparse/mod.rs`
- Modify: `src/sparse/cusparse.rs`

Add the lazy cache for the pattern-life cuSPARSE state. No solve yet — Task 9 plugs the forward solve in.

- [ ] **Step 1: Add `OnceCell` to `CsrPattern`**

In `src/sparse/mod.rs`, on `pub struct CsrPattern`, add:

```rust
use std::cell::OnceCell;
use crate::sparse::cusparse::CudaPatternCache;

pub struct CsrPattern {
    // ... existing fields ...
    pub trans_to_orig: Vec<i32>,
    // Lazy GPU companion built on first cuSPARSE solve call. None on
    // CPU-only runs. Not part of structural equality — implementations
    // of Eq/Hash that exist must skip this field.
    pub(crate) cuda_cache: OnceCell<CudaPatternCache>,
}
```

Customize the `Debug` impl to skip `cuda_cache` (it's a thin wrapper around opaque cuSPARSE handles):

```rust
impl std::fmt::Debug for CsrPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsrPattern")
            .field("n", &self.n)
            .field("nnz", &self.col.len())
            // .. skip cuda_cache ..
            .finish_non_exhaustive()
    }
}
```

Initialize the `OnceCell` in `CsrPattern::from_sparse`:

```rust
CsrPattern {
    // ... existing assignments ...
    trans_to_orig,
    cuda_cache: OnceCell::new(),
}
```

- [ ] **Step 2: Add `CudaPatternCache` struct in `cusparse.rs`**

In `src/sparse/cusparse.rs`:

```rust
use cudarc::cusparse::{
    sys::{cusparseSpMatDescr_t, cusparseSpSVDescr_t, cusparseHandle_t,
          cusparseOperation_t, cusparseFillMode_t, cusparseDiagType_t},
    /* import exact types from the cudarc 0.19 cusparse module */
};
use cudarc::driver::CudaSlice;
use std::marker::PhantomData;

/// Per-pattern cuSPARSE state. Built lazily on first GPU solve call.
///
/// !Send because cuSPARSE descriptors are tied to the thread that created
/// them. Single-threaded training is the only supported mode.
pub(crate) struct CudaPatternCache {
    pub(crate) handle: cusparseHandle_t,
    pub(crate) d_crow: CudaSlice<i32>,
    pub(crate) d_col: CudaSlice<i32>,
    pub(crate) d_row_for_nnz: CudaSlice<i32>,
    /// Sparse matrix descriptor (CSR, lower-triangular, unit-diagonal=No).
    pub(crate) sp_mat: cusparseSpMatDescr_t,
    /// SpSV descriptor for forward solve (NON_TRANSPOSE, LOWER fill).
    pub(crate) desc_forward: cusparseSpSVDescr_t,
    /// SpSV descriptor for backward solve (TRANSPOSE, treats A^T as upper).
    pub(crate) desc_backward: cusparseSpSVDescr_t,
    pub(crate) workspace_forward: CudaSlice<u8>,
    pub(crate) workspace_backward: CudaSlice<u8>,
    // !Send marker
    _not_send: PhantomData<*mut ()>,
}

impl Drop for CudaPatternCache {
    fn drop(&mut self) {
        // SAFETY: descriptors must be destroyed before the device slices they
        // reference go out of scope. cudarc's CudaSlice<T> Drop runs AFTER
        // this Drop body executes, so the order is correct.
        unsafe {
            // cusparseSpSV_destroyDescr(desc_forward)
            // cusparseSpSV_destroyDescr(desc_backward)
            // cusparseDestroySpMat(sp_mat)
            // cusparseDestroy(handle)
            // Implementer: fill with the exact cudarc destruction calls.
        }
    }
}
```

- [ ] **Step 3: Add the lazy-init function (no solve yet)**

```rust
use crate::sparse::CsrPattern;

/// Build or retrieve the GPU cache for this pattern. Allocates device
/// memory for crow/col/row_for_nnz on first call; subsequent calls return
/// the cached handle.
///
/// SAFETY: caller must guarantee `device` corresponds to the currently
/// active CUDA context.
pub(crate) fn ensure_cuda_cache(pattern: &CsrPattern) -> &CudaPatternCache {
    pattern.cuda_cache.get_or_init(|| {
        unsafe { build_cuda_pattern_cache(pattern) }
    })
}

unsafe fn build_cuda_pattern_cache(pattern: &CsrPattern) -> CudaPatternCache {
    // 1. Get a CUDA context handle (cudarc::driver::CudaDevice::new(0)?
    //    or borrow the existing cubecl one if reachable).
    // 2. cusparseCreate(&handle).
    // 3. Upload crow, col, row_for_nnz to CudaSlice<i32>.
    // 4. cusparseCreateCsr(&sp_mat, n, n, nnz, d_crow, d_col,
    //                      <values placeholder — values come per-call>, ...).
    // 5. cusparseSpMatSetAttribute(sp_mat, FILL_MODE, LOWER, ...).
    // 6. cusparseSpMatSetAttribute(sp_mat, DIAG_TYPE, NON_UNIT, ...).
    // 7. cusparseSpSV_createDescr(&desc_forward).
    // 8. cusparseSpSV_createDescr(&desc_backward).
    // 9. cusparseSpSV_bufferSize(forward) → allocate workspace_forward.
    // 10. cusparseSpSV_bufferSize(backward, op=TRANSPOSE) → workspace_backward.
    // 11. cusparseSpSV_analysis(forward) — one-time pattern analysis.
    // 12. cusparseSpSV_analysis(backward).

    todo!("build CudaPatternCache: cusparse descriptors + workspace + analysis")
}
```

The `todo!()` is structural; Task 9 fleshes the calls into real cudarc invocations. For Task 8 we just need the struct + function signatures to compile.

To make Task 8 compile, make `build_cuda_pattern_cache` return a stub:

```rust
unsafe fn build_cuda_pattern_cache(_pattern: &CsrPattern) -> CudaPatternCache {
    unimplemented!("filled in Task 9")
}
```

The `OnceCell` is never populated until Task 9 plumbs `ensure_cuda_cache` into the dispatch path.

- [ ] **Step 4: Build**

```
cargo build --lib 2>&1 | tail -10
```

Expected: clean. `cuda_cache` is `OnceCell::new()` and never touched on the CPU path.

- [ ] **Step 5: Verify CPU tests + sandbox still pass**

```
cargo test --lib 2>&1 | tail -10
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: same counts; ABSOLUTE MATCH.

- [ ] **Step 6: Commit**

```
git add src/sparse/mod.rs src/sparse/cusparse.rs
git commit -m "Add CudaPatternCache skeleton + OnceCell on CsrPattern

The cache holds device-resident crow/col/row_for_nnz arrays and the
two cuSPARSE SpSV descriptors. Built lazily on first GPU solve.
Stubbed in this task; the actual cusparseSpSV_analysis + workspace
allocation lands in Task 9 alongside the forward solve.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: cuSPARSE forward solve

**Files:**
- Modify: `src/sparse/cusparse.rs`
- Create: `src/sparse/dispatch.rs`
- Modify: `src/sparse/mod.rs`

- [ ] **Step 1: Implement `build_cuda_pattern_cache`**

In `src/sparse/cusparse.rs`, replace the `unimplemented!()` body of `build_cuda_pattern_cache` with the actual cudarc cuSPARSE calls per the comments in Task 8 Step 3. Concrete pseudocode (translate to cudarc 0.19's exact function names):

```rust
unsafe fn build_cuda_pattern_cache(pattern: &CsrPattern) -> CudaPatternCache {
    use cudarc::driver::CudaDevice;
    use cudarc::cusparse::{Cusparse, sys::*};

    let device = CudaDevice::new(0).expect("CUDA device 0");
    let handle = Cusparse::new(device.clone()).expect("cusparse handle");

    // Upload structural arrays.
    let d_crow = device.htod_copy(pattern.crow.clone()).unwrap();
    let d_col  = device.htod_copy(pattern.col.clone()).unwrap();
    let d_row_for_nnz = device.htod_copy(pattern.row_for_nnz.clone()).unwrap();

    let n = pattern.n as i64;
    let nnz = pattern.nnz() as i64;

    // Sparse matrix descriptor. Values pointer is a placeholder — replaced
    // per-call before each Solve via cusparseCsrSetPointers.
    let mut sp_mat: cusparseSpMatDescr_t = std::ptr::null_mut();
    cusparseCreateCsr(
        &mut sp_mat,
        n, n, nnz,
        d_crow.device_ptr() as *mut _,
        d_col.device_ptr()  as *mut _,
        std::ptr::null_mut(),                       // values: set per-call
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F,
    );

    // Lower-triangular, non-unit diagonal.
    let fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseSpMatSetAttribute(
        sp_mat, CUSPARSE_SPMAT_FILL_MODE,
        &fill_lower as *const _ as *const _, std::mem::size_of_val(&fill_lower),
    );
    let diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(
        sp_mat, CUSPARSE_SPMAT_DIAG_TYPE,
        &diag as *const _ as *const _, std::mem::size_of_val(&diag),
    );

    // Forward SpSV descriptor.
    let mut desc_forward: cusparseSpSVDescr_t = std::ptr::null_mut();
    cusparseSpSV_createDescr(&mut desc_forward);
    // Backward SpSV descriptor (transpose op).
    let mut desc_backward: cusparseSpSVDescr_t = std::ptr::null_mut();
    cusparseSpSV_createDescr(&mut desc_backward);

    // Probe buffer sizes against dummy x/b descriptors. cusparse requires
    // valid dense vector descriptors for bufferSize; create them transiently.
    // (Filled in by the implementer using cusparseCreateDnVec / DestroyDnVec.)
    let workspace_forward_size: usize = /* probe via cusparseSpSV_bufferSize */;
    let workspace_backward_size: usize = /* same with TRANSPOSE op */;
    let workspace_forward  = device.alloc_zeros::<u8>(workspace_forward_size).unwrap();
    let workspace_backward = device.alloc_zeros::<u8>(workspace_backward_size).unwrap();

    // One-time pattern analysis.
    cusparseSpSV_analysis(
        handle.as_ptr(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &1.0_f32 as *const _ as *const _,
        sp_mat,
        /* dummy b */, /* dummy x */,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
        desc_forward,
        workspace_forward.device_ptr() as *mut _,
    );
    cusparseSpSV_analysis(
        handle.as_ptr(),
        CUSPARSE_OPERATION_TRANSPOSE,
        &1.0_f32 as *const _ as *const _,
        sp_mat,
        /* dummy b */, /* dummy x */,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
        desc_backward,
        workspace_backward.device_ptr() as *mut _,
    );

    CudaPatternCache {
        handle: handle.as_ptr(),
        d_crow, d_col, d_row_for_nnz,
        sp_mat,
        desc_forward, desc_backward,
        workspace_forward, workspace_backward,
        _not_send: PhantomData,
    }
}
```

The exact cudarc 0.19 type names may differ — read `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.7/src/cusparse/mod.rs` to confirm function signatures and constant names.

Implement the `Drop` impl too:

```rust
impl Drop for CudaPatternCache {
    fn drop(&mut self) {
        unsafe {
            cusparseSpSV_destroyDescr(self.desc_forward);
            cusparseSpSV_destroyDescr(self.desc_backward);
            cusparseDestroySpMat(self.sp_mat);
            cusparseDestroy(self.handle);
        }
    }
}
```

- [ ] **Step 2: Implement `cusparse_forward`**

Append to `src/sparse/cusparse.rs`:

```rust
/// GPU forward solve `A · x = b` for lower-triangular `A` via cuSPARSE.
/// Returns a new `B::FloatTensorPrimitive` for `x` on the same device.
///
/// `a_values_prim` and `b_prim` must already be on the CUDA device (this is
/// only called when the active backend is `Cuda<f32, i32>`).
pub(crate) fn cusparse_forward<B: Backend>(
    pattern: &CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    let cache = ensure_cuda_cache(pattern);
    let stream = cubecl_cuda_stream::<B>(device);

    let a_view = primitive_as_cuda_view::<B>(a_values_prim).expect("Cuda backend");
    let b_view = primitive_as_cuda_view::<B>(b_prim).expect("Cuda backend");

    // Allocate x on device.
    let n = pattern.n;
    // SAFETY: device-side allocation via cudarc; tied to the BURN device's
    // CUDA context via the cubecl-managed stream.
    let cuda_device = unsafe { /* obtain CudaDevice for current context */ };
    let mut x_slice = cuda_device.alloc_zeros::<f32>(n).unwrap();

    unsafe {
        cusparseSetStream(cache.handle, stream);
        // Re-point the matrix descriptor at the current a_values.
        cusparseCsrSetPointers(
            cache.sp_mat,
            cache.d_crow.device_ptr() as *mut _,
            cache.d_col.device_ptr() as *mut _,
            a_view.ptr as *mut _,
        );

        // Build dense vector descriptors (cheap, transient).
        let mut b_dn: cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut b_dn, n as i64, b_view.ptr as *mut _, CUDA_R_32F);
        let mut x_dn: cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut x_dn, n as i64, x_slice.device_ptr() as *mut _, CUDA_R_32F);

        // The solve.
        cusparseSpSV_solve(
            cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &1.0_f32 as *const _ as *const _,
            cache.sp_mat, b_dn, x_dn,
            CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
            cache.desc_forward,
        );

        cusparseDestroyDnVec(b_dn);
        cusparseDestroyDnVec(x_dn);
    }

    // Wrap x_slice back into B::FloatTensorPrimitive. The cleanest path is
    // to construct a new BURN tensor that takes ownership of the CudaSlice;
    // burn-cuda's tensor construction from a raw device buffer is the
    // inverse of primitive_as_cuda_view. Implementer: find and use the
    // matching constructor.
    todo!("wrap CudaSlice<f32> into B::FloatTensorPrimitive")
}
```

The `todo!()` is the inverse of Task 6's pointer extraction — there must be a way to construct a `B::FloatTensorPrimitive` from a `CudaSlice<f32>` + a shape. Resolve at the keyboard against burn-cuda source.

- [ ] **Step 3: Create `src/sparse/dispatch.rs`**

```rust
//! Runtime dispatch between the CPU and cuSPARSE triangular solve paths.

use std::any::TypeId;
use std::sync::Once;

use burn::tensor::backend::Backend;

use crate::sparse::{CsrPattern, SavedX};

/// Returns true iff `B` is `burn::backend::Cuda<f32, i32>` (the only GPU
/// backend SP-6 specializes for).
pub(crate) fn backend_is_cuda<B: Backend>() -> bool {
    TypeId::of::<B>() == TypeId::of::<burn::backend::Cuda<f32, i32>>()
}

static FALLBACK_WARNED: Once = Once::new();

/// Resolve effective backend choice. If the caller asked for Cuda but the
/// backend is something else, log a one-shot WARN and return false.
pub(crate) fn effective_use_cuda<B: Backend>(use_cuda: bool) -> bool {
    if use_cuda && !backend_is_cuda::<B>() {
        FALLBACK_WARNED.call_once(|| {
            eprintln!("WARN: sparse_solver=cuda requested but backend is not Cuda — falling back to CPU path");
        });
        return false;
    }
    use_cuda
}

/// Forward solve dispatch. Returns the output primitive and the SavedX
/// variant appropriate to the path taken.
pub(crate) fn forward_primitive<B: Backend>(
    pattern: &std::sync::Arc<CsrPattern>,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
    use_cuda: bool,
) -> (B::FloatTensorPrimitive, SavedX<B>) {
    if effective_use_cuda::<B>(use_cuda) {
        let x_prim = crate::sparse::cusparse::cusparse_forward::<B>(
            pattern, a_values_prim, b_prim, device,
        );
        // Clone the GPU primitive cheaply — burn-cuda's clone is a refcount bump.
        (x_prim.clone(), SavedX::Cuda(x_prim))
    } else {
        // Existing CPU path — moved from src/sparse/mod.rs::forward_primitive.
        let (out_prim, x_vec) =
            crate::sparse::cpu_forward_primitive::<B>(pattern, a_values_prim, b_prim, device);
        (out_prim, SavedX::Cpu(std::sync::Arc::new(x_vec)))
    }
}
```

- [ ] **Step 4: Move the CPU forward into a public `cpu_forward_primitive` in `mod.rs`**

In `src/sparse/mod.rs`, rename the existing `forward_primitive` to `cpu_forward_primitive` (still `pub(crate)`) and keep its body unchanged. Replace all internal call sites with the new dispatch:

```rust
use crate::sparse::dispatch::forward_primitive;
```

Update `triangular_csr_solve` to use the dispatch return type:

```rust
let (out_prim, saved_x) =
    forward_primitive::<I>(pattern, &a_at.primitive, &b_at.primitive, &device, use_cuda);

// ... OpsKind::Tracked branch ...
let state = CsrSolveState::<I> {
    a_values: a_at.primitive.clone(),
    x: saved_x,
    pattern: pattern.clone(),
    use_cuda,
};
```

- [ ] **Step 5: Wire `mod dispatch` in `src/sparse/mod.rs`**

```rust
pub(crate) mod cusparse;
pub(crate) mod dispatch;
```

- [ ] **Step 6: Build + run V5 forward-only smoke**

```
cargo build --lib 2>&1 | tail -10
```

Expected: clean (the spike from Tasks 6-7 must already be working).

Add a quick smoke in `tests/sparse_cusparse_v5.rs` for forward only — actual V5 lands in Task 11. For now just check the CUDA forward path runs:

```rust
// In tests/sparse_cusparse_v5.rs (new file):
#[test]
fn forward_smoke_cuda() {
    // Skip if no CUDA, else build a tiny pattern, call triangular_csr_solve
    // with use_cuda=true, and assert the output is finite and non-zero.
}
```

```
cargo test --test sparse_cusparse_v5 forward_smoke_cuda 2>&1 | tail -10
```

- [ ] **Step 7: Verify CPU path still bit-exact**

```
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: ABSOLUTE MATCH unchanged.

- [ ] **Step 8: Commit**

```
git add src/sparse/cusparse.rs src/sparse/dispatch.rs src/sparse/mod.rs tests/sparse_cusparse_v5.rs
git commit -m "Add cuSPARSE forward solve + CPU/CUDA dispatch shim

cusparse_forward calls cusparseSpSV with the lazy-built CudaPatternCache
descriptors (one-time per pattern, reused per timestep). The dispatch
module routes triangular_csr_solve to either the CPU or cuSPARSE path
based on use_cuda + backend type. CPU path is byte-for-byte unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: cuSPARSE backward solve

**Files:**
- Modify: `src/sparse/cusparse.rs`
- Modify: `src/sparse/dispatch.rs`
- Modify: `src/sparse/mod.rs`

The backward solve uses the same `cache.sp_mat` but with `CUSPARSE_OPERATION_TRANSPOSE` — solves `A^T · y = b`, which is the upper-triangular back-substitution we need for the autograd backward.

- [ ] **Step 1: Implement `cusparse_backward_solve` in `cusparse.rs`**

```rust
/// GPU backward triangular solve: `A^T · y = b` via cuSPARSE TRANSPOSE op.
/// Returns `y` as a new device-resident primitive.
pub(crate) fn cusparse_backward_solve<B: Backend>(
    pattern: &CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    let cache = ensure_cuda_cache(pattern);
    let stream = cubecl_cuda_stream::<B>(device);

    let a_view = primitive_as_cuda_view::<B>(a_values_prim).expect("Cuda backend");
    let b_view = primitive_as_cuda_view::<B>(b_prim).expect("Cuda backend");

    let n = pattern.n;
    let cuda_device = unsafe { /* obtain CudaDevice — same path as forward */ };
    let mut y_slice = cuda_device.alloc_zeros::<f32>(n).unwrap();

    unsafe {
        cusparseSetStream(cache.handle, stream);
        cusparseCsrSetPointers(
            cache.sp_mat,
            cache.d_crow.device_ptr() as *mut _,
            cache.d_col.device_ptr() as *mut _,
            a_view.ptr as *mut _,
        );

        let mut b_dn: cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut b_dn, n as i64, b_view.ptr as *mut _, CUDA_R_32F);
        let mut y_dn: cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut y_dn, n as i64, y_slice.device_ptr() as *mut _, CUDA_R_32F);

        cusparseSpSV_solve(
            cache.handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &1.0_f32 as *const _ as *const _,
            cache.sp_mat, b_dn, y_dn,
            CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
            cache.desc_backward,
        );

        cusparseDestroyDnVec(b_dn);
        cusparseDestroyDnVec(y_dn);
    }

    // Wrap y_slice into B::FloatTensorPrimitive — same wrapper as forward.
    todo!("reuse the CudaSlice → B::FloatTensorPrimitive helper from cusparse_forward")
}
```

Factor the CudaSlice-to-primitive wrap into a private helper used by both forward and backward.

- [ ] **Step 2: Wire backward dispatch**

In `src/sparse/dispatch.rs`, append:

```rust
pub(crate) fn backward_solve_primitive<B: Backend>(
    pattern: &std::sync::Arc<CsrPattern>,
    a_values_prim: &B::FloatTensorPrimitive,
    grad_out_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
    use_cuda: bool,
) -> B::FloatTensorPrimitive {
    if effective_use_cuda::<B>(use_cuda) {
        crate::sparse::cusparse::cusparse_backward_solve::<B>(
            pattern, a_values_prim, grad_out_prim, device,
        )
    } else {
        // CPU path: existing back_sub_upper_transposed + B::float_from_data round-trip.
        let a_data: Vec<f32> = crate::sparse::primitive_to_vec::<B>(a_values_prim.clone());
        let grad_out_data: Vec<f32> = crate::sparse::primitive_to_vec::<B>(grad_out_prim.clone());
        let gradb_data = crate::sparse::back_sub_upper_transposed(pattern, &a_data, &grad_out_data);
        B::float_from_data(burn::tensor::TensorData::from(gradb_data.as_slice()), device)
    }
}
```

- [ ] **Step 3: Update `CsrSolveOp::backward` to use the dispatch**

In `src/sparse/mod.rs::CsrSolveOp::backward`, replace the manual `back_sub_upper_transposed` call (and the CPU-only gradb registration) with:

```rust
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
```

For the grada path, keep the CPU code for now — Task 11 swaps in the GPU kernel. Concretely, to compute `grada` on the CPU path you'll still need a host-side `gradb_data` and `x_data`; convert the returned `gradb_prim` back to a Vec for the grada loop only on the CPU path:

```rust
let gradb_data: Vec<f32> = if use_cuda && backend_is_cuda::<B>() {
    // Task 11 replaces this entire block with a GPU grada kernel.
    primitive_to_vec::<B>(gradb_prim.clone())
} else {
    primitive_to_vec::<B>(gradb_prim.clone())
};
let x_host: Vec<f32> = /* from SavedX as before */;
// ... existing grada CPU loop ...
```

This is the worst the backward path gets — Task 11 eliminates the gradb sync.

- [ ] **Step 4: Build + smoke**

```
cargo build --lib 2>&1 | tail -5
cargo test --test sparse_gradcheck 2>&1 | tail -5
```

Expected: CPU backward bit-exact; sandbox unchanged.

- [ ] **Step 5: Commit**

```
git add src/sparse/cusparse.rs src/sparse/dispatch.rs src/sparse/mod.rs
git commit -m "Add cuSPARSE backward triangular solve

The backward path reuses the cache's TRANSPOSE SpSV descriptor to
solve A^T·y=b on GPU. CSrSolveOp::backward dispatches to either the
CPU or GPU solve based on use_cuda. The per-nnz grada scatter still
materializes through host on the GPU path; Task 11 swaps in a GPU
kernel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: GPU `grada` kernel + V5 verification

**Files:**
- Modify: `src/sparse/cusparse.rs`
- Modify: `src/sparse/mod.rs`
- Modify: `tests/sparse_cusparse_v5.rs`

Math: `grada[k] = -gradb[row_for_nnz[k]] * x[col[k]]` for `k in 0..nnz`. No atomics needed (one writer per `k`).

- [ ] **Step 1: Write the cubecl kernel**

In `src/sparse/cusparse.rs`:

```rust
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn grada_kernel(
    gradb: &Array<f32>,        // length n
    x: &Array<f32>,            // length n
    row_for_nnz: &Array<i32>,  // length nnz
    col: &Array<i32>,          // length nnz
    grada: &mut Array<f32>,    // length nnz, output
) {
    let k = ABSOLUTE_POS;
    if k < grada.len() {
        let r = row_for_nnz[k] as u32;
        let c = col[k] as u32;
        grada[k] = -gradb[r] * x[c];
    }
}

/// Compute grada per-nnz on GPU. Returns a device-resident primitive.
pub(crate) fn cusparse_grada<B: Backend>(
    pattern: &CsrPattern,
    gradb_prim: &B::FloatTensorPrimitive,
    x_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    let cache = ensure_cuda_cache(pattern);
    let nnz = pattern.nnz();

    // Launch cubecl kernel — same compute server as BURN ops, so no
    // explicit stream sync required.
    todo!(
        "wire cubecl launch:
         1. Allocate grada Tensor<Cuda<f32, i32>, 1> of shape [nnz].
         2. Launch grada_kernel with the right block/grid sizing (CUBE_DIM_X = 256).
         3. Return the resulting primitive."
    )
}
```

If cubecl integration is awkward inside our sparse module, fallback B: a raw cudarc kernel from inline PTX:

```rust
const GRADA_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64
.visible .entry grada_kernel(
    .param .u64 gradb,     // f32*
    .param .u64 x,
    .param .u64 row_for_nnz,
    .param .u64 col,
    .param .u64 grada,
    .param .u32 nnz
) {
    // ... per-thread compute: k = blockIdx.x * blockDim.x + threadIdx.x;
    //                          if (k < nnz) grada[k] = -gradb[row[k]] * x[col[k]];
}
"#;
```

Picked at the keyboard. cubecl is preferred; PTX is the escape hatch.

- [ ] **Step 2: Wire grada into the backward dispatch**

In `src/sparse/dispatch.rs`, add:

```rust
pub(crate) fn grada_primitive<B: Backend>(
    pattern: &std::sync::Arc<CsrPattern>,
    gradb_prim: &B::FloatTensorPrimitive,
    x_saved: &SavedX<B>,
    device: &B::Device,
    use_cuda: bool,
) -> B::FloatTensorPrimitive {
    if effective_use_cuda::<B>(use_cuda) {
        let x_prim = match x_saved {
            SavedX::Cuda(p) => p,
            SavedX::Cpu(_) => panic!("GPU grada called with CPU-saved x"),
        };
        crate::sparse::cusparse::cusparse_grada::<B>(pattern, gradb_prim, x_prim, device)
    } else {
        // CPU path: materialize gradb + x to host, compute, push back.
        let gradb_host = crate::sparse::primitive_to_vec::<B>(gradb_prim.clone());
        let x_host: Vec<f32> = match x_saved {
            SavedX::Cpu(arc) => (**arc).clone(),
            SavedX::Cuda(p)  => crate::sparse::primitive_to_vec::<B>(p.clone()),
        };
        let nnz = pattern.nnz();
        let mut grada = vec![0.0f32; nnz];
        for k in 0..nnz {
            let r = pattern.row_for_nnz[k] as usize;
            let c = pattern.col[k] as usize;
            grada[k] = -gradb_host[r] * x_host[c];
        }
        B::float_from_data(burn::tensor::TensorData::from(grada.as_slice()), device)
    }
}
```

- [ ] **Step 3: Update `CsrSolveOp::backward` to use the grada dispatch**

Replace the per-nnz CPU loop in `src/sparse/mod.rs::CsrSolveOp::backward` with:

```rust
if let Some(p_a) = parent_a {
    let grada_prim = crate::sparse::dispatch::grada_primitive::<B>(
        &pattern, &gradb_prim, &x, &device, use_cuda,
    );
    grads.register::<B>(p_a.id, grada_prim);
}
```

After this change, on the CPU path the backward stays bit-exact (the CPU branch of `grada_primitive` is the same math); on the GPU path the whole backward stays on device.

- [ ] **Step 4: Write V5**

In `tests/sparse_cusparse_v5.rs`:

```rust
//! SP-6 V5: bit-equivalence of CPU and CUDA forward+backward CSR solves.
//!
//! Skips on hosts without a CUDA device. Synthetic 100-reach
//! lower-triangular pattern; asserts |cpu - cuda| / max(|cpu|, eps) <= 1e-5
//! for x, gradb, grada.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{backend::Backend, Tensor};

use ddrs::sparse::{CsrPattern, SparseAdjacency, triangular_csr_solve};

fn build_banded_pattern(n: usize, bandwidth: usize) -> Arc<CsrPattern> {
    // Lower-triangular banded: A[i, i] = 2.0, A[i, j] = 0.5 for max(0, i-bandwidth) <= j < i.
    // Length 1000, slope 0.001 are dummies — solver only reads the structural fields.
    let mut dense = vec![0.0f32; n * n];
    for i in 0..n {
        dense[i * n + i] = 2.0;
        let lo = i.saturating_sub(bandwidth);
        for j in lo..i {
            dense[i * n + j] = 0.5;
        }
    }
    let adj = SparseAdjacency::from_dense(n, &dense, vec![1000.0; n], vec![0.001; n]);
    Arc::new(CsrPattern::from_sparse(&adj))
}

fn deterministic_inputs(nnz: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    // Seeded random-ish inputs without a heavy RNG dep.
    let a: Vec<f32> = (0..nnz).map(|k| 1.0 + (k as f32 * 0.013).sin() * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 5.0 + (i as f32 * 0.07).cos() * 2.0).collect();
    (a, b)
}

#[test]
fn v5_cpu_and_cuda_forward_backward_bit_match() {
    let pattern = build_banded_pattern(100, 5);
    let nnz = pattern.col.len();
    let n = pattern.n;
    let (a_init, b_init) = deterministic_inputs(nnz, n);

    // CPU run.
    type Bc = Autodiff<NdArray<f32>>;
    let dev_c = <NdArray<f32> as Backend>::Device::default();
    let a_c: Tensor<Bc, 1> = Tensor::from_floats(a_init.as_slice(), &dev_c).require_grad();
    let b_c: Tensor<Bc, 1> = Tensor::from_floats(b_init.as_slice(), &dev_c).require_grad();
    let x_c = triangular_csr_solve(&pattern, a_c.clone(), b_c.clone(), false);
    let loss_c = x_c.clone().sum();
    let grads_c = loss_c.backward();
    let grad_a_c = a_c.grad(&grads_c).expect("a grad").to_data().to_vec::<f32>().unwrap();
    let grad_b_c = b_c.grad(&grads_c).expect("b grad").to_data().to_vec::<f32>().unwrap();
    let x_c_vec: Vec<f32> = x_c.into_data().to_vec().unwrap();

    // CUDA run — skip if no device.
    type Bg = Autodiff<burn::backend::Cuda<f32, i32>>;
    let dev_g: <burn::backend::Cuda<f32, i32> as Backend>::Device = Default::default();
    let cuda_ok = std::panic::catch_unwind(|| {
        let _ = Tensor::<burn::backend::Cuda<f32, i32>, 1>::from_floats([0.0_f32], &dev_g);
    }).is_ok();
    if !cuda_ok {
        eprintln!("skipping CUDA branch: no device");
        return;
    }

    // Reuse a FRESH pattern — cuda_cache is per-pattern and we want the GPU
    // branch to lazily build its own cache.
    let pattern_g = build_banded_pattern(100, 5);

    let a_g: Tensor<Bg, 1> = Tensor::from_floats(a_init.as_slice(), &dev_g).require_grad();
    let b_g: Tensor<Bg, 1> = Tensor::from_floats(b_init.as_slice(), &dev_g).require_grad();
    let x_g = triangular_csr_solve(&pattern_g, a_g.clone(), b_g.clone(), true);
    let loss_g = x_g.clone().sum();
    let grads_g = loss_g.backward();
    let grad_a_g = a_g.grad(&grads_g).expect("a grad").to_data().to_vec::<f32>().unwrap();
    let grad_b_g = b_g.grad(&grads_g).expect("b grad").to_data().to_vec::<f32>().unwrap();
    let x_g_vec: Vec<f32> = x_g.into_data().to_vec().unwrap();

    fn assert_rel(name: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{name} length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&ai, &bi)) in a.iter().zip(b).enumerate() {
            let denom = ai.abs().max(1e-6);
            let rel = (ai - bi).abs() / denom;
            assert!(
                rel <= 1e-5,
                "{name}[{i}] divergence: cpu={ai}, cuda={bi}, rel={rel}"
            );
        }
    }

    assert_rel("x",      &x_c_vec, &x_g_vec);
    assert_rel("grad_b", &grad_b_c, &grad_b_g);
    assert_rel("grad_a", &grad_a_c, &grad_a_g);
}
```

- [ ] **Step 5: Run V5**

```
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -40
```

Expected on a CUDA host: V5 passes (all three asserts within 1e-5 rel).

If V5 fails:
- If rel diff is in 1e-5..1e-4 range and you've sanity-checked all three earlier spikes (Tasks 6/7) compile and run cleanly, the cause is f32 accumulation in cuSPARSE's solver (uses a different forward-sub ordering than the CPU). Bump tolerance to 1e-4 and add a comment justifying the choice. Re-run.
- If rel diff is >> 1e-4 (orders of magnitude off), the cause is wrong: pointer extraction returned the wrong slice, the matrix descriptor has the wrong fill mode, or the TRANSPOSE op is reading the wrong array. Stop and report with the divergence numbers + which of x/grad_a/grad_b is wrong.

- [ ] **Step 6: Commit (only if V5 passes)**

```
git add src/sparse/cusparse.rs src/sparse/dispatch.rs src/sparse/mod.rs tests/sparse_cusparse_v5.rs
git commit -m "Add GPU grada kernel + V5 bit-match verification

The grada per-nnz scatter now runs entirely on GPU via a cubecl
kernel. V5 asserts CPU (NdArray) and CUDA (Cuda<f32, i32>) produce
matching x, grad_a, grad_b within 1e-5 relative on a synthetic
banded 100-reach pattern.

If V5 passes, the cuSPARSE forward + backward + grada path is
bit-equivalent to the CPU reference at the f32 floor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If V5 fails, STOP and report. Do not proceed to Task 12.

---

### Task 12: Regression sweep + smoke-test SparseSolver::Cuda end-to-end

**Files:**
- Modify: `tests/sparse_cusparse_v5.rs`

After V5 passes, run the full regression matrix to confirm SP-6 broke nothing.

- [ ] **Step 1: Smoke — run `train` with `sparse_solver: cuda`**

Append to `config/merit_training.yaml` a new commented-out option (already in Task 1 Step 4); uncomment by hand for the smoke run. Or, simpler, create a tiny override:

```
cp config/merit_training.yaml /tmp/merit_training_cuda.yaml
echo "    sparse_solver: cuda" >> /tmp/merit_training_cuda.yaml
```

Run a 3-mini-batch smoke:

```
cargo run --release --bin train -- \
    --config /tmp/merit_training_cuda.yaml \
    --checkpoint-dir /tmp/sp6_smoke \
    --max-mini-batches 3 2>&1 | tail -20
```

Expected: training runs without panic; loss values are finite; mini-batch wall-time is faster than the CPU baseline (record both numbers).

Compare to the same with the default config (CPU solver):

```
cargo run --release --bin train -- \
    --config config/merit_training.yaml \
    --checkpoint-dir /tmp/sp6_smoke_cpu \
    --max-mini-batches 3 2>&1 | tail -20
```

Both runs should produce finite losses. The CUDA run should be faster per mini-batch.

- [ ] **Step 2: Full unit + integration sweep**

```
cargo test --lib 2>&1 | tail -15
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo test --test training_verification v1_loss_matches 2>&1 | tail -5
cargo test --test training_verification v3_train 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected:
- All lib tests pass.
- `sparse_gradcheck` (CPU path, NdArray pinning) passes — backward math is unchanged.
- V1 still 1e-5 match on NdArray.
- V3 training loop still advances + writes checkpoints.
- DDR sandbox reports ABSOLUTE MATCH (NdArray pinning).

If any of these regresses, STOP and report. SP-6 must not break previous verification.

- [ ] **Step 3: Clippy sweep on new code**

```
cargo clippy --all-targets 2>&1 | grep -E "(sparse/cusparse|sparse/dispatch)" | head -20
```

Expected: no new warnings. Pre-existing clippy lints in other modules are out of scope.

- [ ] **Step 4: Optional — add the smoke as a long-skip test**

Append to `tests/sparse_cusparse_v5.rs` a `#[test] #[ignore]`:

```rust
/// End-to-end smoke: run a 3-mini-batch training with sparse_solver=cuda
/// and assert finite losses. Marked #[ignore] because it depends on the
/// live data sources; run with `cargo test --release --test sparse_cusparse_v5
/// -- --ignored end_to_end_smoke`.
#[test]
#[ignore]
fn end_to_end_smoke_cuda_train() {
    // Spawn `cargo run --bin train` as a subprocess with the cuda config.
    // Capture stdout; assert "loss=" lines parse to finite f32.
    // Implementer: keep it ~30 lines.
}
```

This is optional polish — the manual smoke from Step 1 is the load-bearing check.

- [ ] **Step 5: Commit**

```
git add tests/sparse_cusparse_v5.rs
git commit -m "SP-6 regression sweep: V1/V3/sandbox green; CUDA smoke recorded

Confirms cuSPARSE path doesn't regress CPU verification:
  * V1 frozen-params loss equiv: pass
  * V3 train-one-epoch end-to-end: pass
  * compare_ddr_sandbox ABSOLUTE MATCH: pass
  * All lib + sparse_gradcheck tests: pass
  * sparse_solver=cuda smoke (3 mini-batches): finite losses, faster
    per-mini-batch than CPU baseline

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

### Spec coverage

| Spec section | Task |
|---|---|
| `SparseSolver` enum + `Params.sparse_solver` + YAML | 1 |
| `src/sparse/` module split (refactor) | 2 |
| `cudarc` dep with cuSPARSE feature | 3 |
| `SavedX<B>` enum + `use_cuda` plumbing | 4, 5 |
| Pipe sparse_solver through `MuskingumCunge` | 5 |
| BURN tensor primitive → CUDA pointer (spike) | 6 |
| cubecl CUDA stream sharing (spike) | 7 |
| `CudaPatternCache` + OnceCell | 8 |
| cuSPARSE forward solve (lower-tri) | 9 |
| Dispatch shim `src/sparse/dispatch.rs` | 9 |
| cuSPARSE backward solve (TRANSPOSE) | 10 |
| GPU grada per-nnz kernel | 11 |
| V5 bit-match (CPU vs CUDA) | 11 |
| Regression sweep + CUDA smoke | 12 |

### Placeholder scan

The plan has `todo!()` placeholders in:
- Task 6 (`primitive_as_cuda_view` — pointer extraction; flagged as escalation gate)
- Task 7 (`cubecl_cuda_stream` — stream accessor; with documented fallback)
- Task 9 (CudaSlice → B::FloatTensorPrimitive wrap; flagged as keyboard-resolved)
- Task 11 (cubecl kernel launch wiring; flagged with PTX fallback)

These are honest "implementer resolves at the keyboard against an unknown BURN-internal API" patterns, same as SP-4 Task 2 (`scatter_add`) and SP-2 Task 1 (icechunk adapter). Each comes with a concrete attempt strategy and an explicit escalation trigger.

### Type / identifier consistency

- `SparseSolver`, `SavedX<B>`, `CudaPatternCache`, `cuda_cache`, `use_cuda`, `effective_use_cuda`, `backend_is_cuda`, `forward_primitive`, `backward_solve_primitive`, `grada_primitive`, `cusparse_forward`, `cusparse_backward_solve`, `cusparse_grada`, `ensure_cuda_cache`, `primitive_as_cuda_view`, `cubecl_cuda_stream` — used identically across all tasks.
- `triangular_csr_solve(pattern, a, b, use_cuda)` — signature fixed in Task 4, threaded through Task 5, and called by V5 in Task 11.
- `CsrPattern` gains one field (`cuda_cache: OnceCell<CudaPatternCache>`) in Task 8; no other field changes.
- The `&Arc<CsrPattern>` argument shape is preserved in all dispatch signatures (no shift to `&CsrPattern`).
- V5 tolerance is `1e-5` relative, matching the spec; Task 11 documents the fallback to `1e-4` if cuSPARSE accumulation drift exceeds it.

---

## Execution choice

Plan complete and saved to `.claude/specs/2026-05-20-sp6-cusparse-gpu-solve-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review (spec compliance + code quality). Same workflow as SP-1 through SP-5.
2. **Inline Execution** — batch with checkpoints.

Which approach?
