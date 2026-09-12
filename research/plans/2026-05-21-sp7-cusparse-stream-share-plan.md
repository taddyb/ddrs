# SP-7 cuSPARSE Stream-Share + Zero-Copy x Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sparse_solver: cuda` materially faster than the CPU path by replacing SP-6's host-blocking sync model with shared-stream GPU ordering, and eliminating the per-solve host roundtrip of `x`. V6 asserts CUDA wall-time ≤ 0.7× CPU on a 3-mini-batch smoke.

**Architecture:** Two minimal forks of upstream Tracel-AI crates (`cubecl-cuda` + `burn-cubecl`) expose `pub` accessors hidden today: `CudaServer::stream()` and `CubeTensor::from_handle()`. `Cargo.toml [patch.crates-io]` redirects to git branches on our forks. With those in hand, `cusparse_forward` / `cusparse_backward_solve` run cuSPARSE on cubecl's own stream (no explicit syncs) and return a `CubeTensor` wrapping a cubecl-allocated buffer (no host roundtrip). FALLBACK_STREAM + every raw cuMem* call gets deleted.

**Tech Stack:** Rust 1.94+, BURN 0.21 (`Cuda<f32, i32>`), `cudarc 0.19` (cuSPARSE + driver), patched `cubecl-cuda 0.10` + `burn-cubecl 0.21` via git forks.

**Spec:** `.claude/specs/2026-05-21-sp7-cusparse-stream-share-design.md`
**Parent:** `.claude/specs/2026-05-20-sp6-cusparse-gpu-solve-design.md`

**Verification:**
- **V5 (correctness, preserved):** synthetic 100-reach lower-tri bit-match between `NdArray<f32>` Cpu and `Cuda<f32, i32>` Cuda paths — `x`, `grad_a`, `grad_b` within `1e-3` abs / `1e-4` rel. Must still pass after the rewrite.
- **V6 (load-bearing perf, NEW, `#[ignore]`'d):** `cargo run --release --bin train -- --max-mini-batches 3` with `sparse_solver: cuda` finishes in ≤ 0.7× the wall-time of the same command with `sparse_solver: cpu`.

---

## Conventions for this plan

- CPU path stays bit-exact. After every task, `cargo test --test sparse_gradcheck` passes and `cargo run --release --example compare_ddr_sandbox` reports ABSOLUTE MATCH.
- All `unsafe` blocks need `SAFETY:` comments.
- New code must add zero clippy warnings. Pre-existing routing-core lints are out of scope (SP-4..6 precedent).
- No commit amends. New commit per task.
- Fork URLs in the examples below use `https://github.com/ddrs-fork/<repo>` as placeholders. The Task 1 implementer substitutes the actual GitHub account they push to (typically the user's personal account).
- Every cusparse-side test that needs CUDA skips cleanly on CPU-only hosts via `std::panic::catch_unwind` around `Default::default()` on the device type.

---

## File Structure

**Created:**
- `vendor/README.md` — fork rationale, upstream commit hashes, recovery instructions, upstream PR checklist (SP-8 cleanup).
- `tests/sparse_cusparse_v6.rs` — V6 speedup assertion test (`#[ignore]`'d).

**Modified:**
- `Cargo.toml` — `[patch.crates-io]` block with two git forks.
- `src/sparse/cusparse.rs` — rewrite `cusparse_forward`, `cusparse_backward_solve`, `build_cuda_pattern_cache`; delete `FALLBACK_STREAM`, `cubecl_cuda_stream`'s body, `async_alloc`, `SendStream`, and every raw `cuMem*` call that becomes unreachable.

**External (forks, not in this repo):**
- `<gh-account>/cubecl@ddrs-sp7-stream-accessor` branch — patch adds `pub fn CudaServer::stream() -> CUstream`.
- `<gh-account>/burn@ddrs-sp7-primitive-ctor` branch — patch adds `pub fn CubeTensor::from_handle(...)`.

---

### Task 1: Fork prep — clone, branch, push, wire `[patch.crates-io]`

**Files:**
- Create: `vendor/README.md`
- Modify: `Cargo.toml`

**Anchors:**
- Upstream cubecl-cuda: `https://github.com/tracel-ai/cubecl/tree/main/crates/cubecl-cuda` — published version `0.10.0`.
- Upstream burn-cubecl: `https://github.com/tracel-ai/burn/tree/main/crates/burn-cubecl` — published version `0.21.0`.
- Both are monorepos; the patch lives at `crates/<name>/` inside each.

- [ ] **Step 1: Find the exact upstream tags**

```
git ls-remote --tags https://github.com/tracel-ai/cubecl | grep -E "v?0\.10\.0$" | head -3
git ls-remote --tags https://github.com/tracel-ai/burn    | grep -E "v?0\.21\.0$" | head -3
```

Record the commit hashes — they go into `vendor/README.md`. If the published `0.10.0` / `0.21.0` doesn't tag directly, fall back to the `main` SHA that was current when those crates.io versions were released (visible on crates.io's "repository" link → look at the cargo `Cargo.lock`-pinned date).

- [ ] **Step 2: Fork + branch + push**

Manual step the implementer takes through a GitHub web UI or `gh repo fork`:

```
gh repo fork tracel-ai/cubecl --clone=false
gh repo fork tracel-ai/burn   --clone=false
```

Clone each fork locally to `/tmp/sp7-forks/`:

```
git clone https://github.com/<gh-account>/cubecl /tmp/sp7-forks/cubecl
git clone https://github.com/<gh-account>/burn   /tmp/sp7-forks/burn
```

Check out the recorded upstream commit hashes and create branches:

```
cd /tmp/sp7-forks/cubecl
git checkout -b ddrs-sp7-stream-accessor <upstream-commit-hash>
git push -u origin ddrs-sp7-stream-accessor

cd /tmp/sp7-forks/burn
git checkout -b ddrs-sp7-primitive-ctor   <upstream-commit-hash>
git push -u origin ddrs-sp7-primitive-ctor
```

If you don't want to publish forks, use in-tree vendor instead:

```
mkdir -p vendor
cp -r /tmp/sp7-forks/cubecl/crates/cubecl-cuda vendor/cubecl-cuda
cp -r /tmp/sp7-forks/burn/crates/burn-cubecl   vendor/burn-cubecl
```

…and the `[patch.crates-io]` block below uses `{ path = "..." }` instead of `{ git = "..." }`. Pick the route that fits the implementer's setup; the rest of the plan reads either form.

- [ ] **Step 3: Write `vendor/README.md`**

Create with this content (filling in the bracketed bits with the actual hashes/URLs from Step 1):

```markdown
# Vendored cubecl + burn patches for SP-7

SP-7 (cuSPARSE stream-share + zero-copy x wrap) needs two tiny `pub`
accessors that are `pub(crate)` on crates.io today:

| Crate          | Version | Added accessor                                  |
|----------------|---------|-------------------------------------------------|
| `cubecl-cuda`  | 0.10.0  | `pub fn CudaServer::stream() -> CUstream`       |
| `burn-cubecl`  | 0.21.0  | `pub fn CubeTensor::from_handle(...) -> Self`   |

## Forks

- cubecl: `<gh-account>/cubecl` branch `ddrs-sp7-stream-accessor`
  - Forked from upstream commit `<UPSTREAM-CUBECL-SHA>` (= crates.io 0.10.0).
- burn:   `<gh-account>/burn`   branch `ddrs-sp7-primitive-ctor`
  - Forked from upstream commit `<UPSTREAM-BURN-SHA>` (= crates.io 0.21.0).

## Diffs

Both patches are ≤ 50 lines. See each branch's first commit for the
exact diff.

## Upgrading

When upgrading `burn` / `cubecl` major versions in ddrs:
1. On the fork, `git fetch upstream && git rebase upstream/<new-tag>`.
2. Re-apply the patch (small risk of merge conflict if the touched
   files moved).
3. `git push --force-with-lease origin <branch>`.
4. In ddrs `Cargo.toml`, bump the upstream version constraint if needed.

## Upstream PR (SP-8 cleanup)

The accessors are vanilla "expose what already exists as `pub`" PRs.
Plan to upstream both as SP-8 when SP-7 is stable. Once merged + released,
delete `[patch.crates-io]` entries and `vendor/README.md`.
```

- [ ] **Step 4: Add `[patch.crates-io]` to `Cargo.toml`**

Append at the bottom of `Cargo.toml`:

```toml
# Vendored forks of cubecl-cuda + burn-cubecl with `pub` accessors needed
# by SP-7's cuSPARSE GPU solve. See vendor/README.md for rationale +
# upstream PR plan.
[patch.crates-io]
cubecl-cuda = { git = "https://github.com/<gh-account>/cubecl", branch = "ddrs-sp7-stream-accessor" }
burn-cubecl = { git = "https://github.com/<gh-account>/burn",   branch = "ddrs-sp7-primitive-ctor"  }
```

(In-tree fallback: `cubecl-cuda = { path = "vendor/cubecl-cuda" }`, `burn-cubecl = { path = "vendor/burn-cubecl" }`.)

- [ ] **Step 5: Verify `cargo build` resolves with the patched deps**

```
cargo update -p cubecl-cuda -p burn-cubecl 2>&1 | tail -10
cargo build --lib 2>&1 | tail -10
```

Expected: `cargo update` shows the fork URLs in the diagnostic; `cargo build` succeeds with no behavior change (the fork branches at this point are byte-identical to upstream).

If `[patch.crates-io]` is rejected with "patched crate is not in the workspace's dependency graph" or "version mismatch", the most common cause is a transitive crate (e.g., `burn-cuda`) pinning an exact version. Inspect `Cargo.lock` and adjust. If `burn-cuda 0.21` requires `cubecl-cuda = "=0.10.5"` (an exact patch version different from 0.10.0), the fork branches must rebase to that tag instead. Report and revise the plan.

- [ ] **Step 6: Confirm full regression with patched (unmodified) deps**

```
cargo test --lib 2>&1 | grep "test result" | tail -5
cargo test --test sparse_gradcheck 2>&1 | grep "test result" | tail -3
cargo test --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | grep "test result" | tail -3
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: same counts as before SP-7 started. The forks haven't been patched yet — these runs only confirm the `[patch.crates-io]` resolution doesn't break the build.

- [ ] **Step 7: Commit**

```
git add Cargo.toml Cargo.lock vendor/README.md
git commit -m "Wire [patch.crates-io] forks of cubecl-cuda + burn-cubecl

Prep for SP-7 perf work. Both fork branches are byte-identical to
upstream at this point; subsequent SP-7 tasks add the pub accessors.
vendor/README.md records the upstream commit hashes and SP-8 cleanup
plan to upstream the patches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: cubecl-cuda fork — `pub fn CudaServer::stream() -> CUstream`

**Files (on the cubecl fork, NOT this repo):**
- Modify: `crates/cubecl-cuda/src/compute/server.rs`

This task lands a commit on `<gh-account>/cubecl@ddrs-sp7-stream-accessor`, not in ddrs itself.

- [ ] **Step 1: Read the existing private path**

```
grep -n "streams:\|MultiStream\|fn current\|pub.*Stream\|sys: CUstream\|Stream {" \
    /tmp/sp7-forks/cubecl/crates/cubecl-cuda/src/compute/server.rs \
    /tmp/sp7-forks/cubecl/crates/cubecl-cuda/src/compute/stream*.rs 2>/dev/null | head -30
```

Locate `CudaServer.streams: MultiStream<CudaStreamBackend>` and the `current()` (or equivalent) method on `MultiStream`. Find where `Stream.sys: CUstream` is the actual handle.

- [ ] **Step 2: Add the public method**

In `crates/cubecl-cuda/src/compute/server.rs`, inside `impl CudaServer`, append:

```rust
    /// Return the active CUDA stream for this server's primary execution
    /// path.
    ///
    /// Exposed for external CUDA libraries (cuSPARSE, cuBLAS, cuFFT) that
    /// need `cusparseSetStream`-style stream sharing to avoid host syncs
    /// at the boundary with cubecl-managed kernels.
    pub fn stream(&self) -> cudarc::driver::sys::CUstream {
        self.streams.current().sys
    }
```

Adjust `self.streams.current().sys` to the actual field path discovered in Step 1. If `MultiStream::current()` doesn't exist by that name, use whatever returns the primary stream (e.g., `self.streams.active()`, `self.streams.primary()`); inline a one-line doc comment naming the chosen field if non-obvious.

- [ ] **Step 3: Verify the fork still builds**

```
cd /tmp/sp7-forks/cubecl
cargo build -p cubecl-cuda 2>&1 | tail -10
```

Expected: clean compile.

- [ ] **Step 4: Commit + push the fork**

```
cd /tmp/sp7-forks/cubecl
git add crates/cubecl-cuda/src/compute/server.rs
git commit -m "Expose CudaServer::stream() for external CUDA library interop

External libraries like cuSPARSE need the active CUDA stream to call
cusparseSetStream(...) and avoid host syncs at the cubecl boundary.
This commit exposes the existing stream handle via a one-method pub
accessor. No behavioral change.

(SP-7 patch for downstream ddrs project; will upstream as a real PR.)"
git push origin ddrs-sp7-stream-accessor
```

- [ ] **Step 5: Refresh the patched dep in ddrs + verify**

In `/home/tbindas/projects/ddrs`:

```
cargo update -p cubecl-cuda 2>&1 | tail -5
cargo build --lib 2>&1 | tail -10
cargo test --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | grep "test result" | tail -3
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: V5 passes, ABSOLUTE MATCH. The new `stream()` method is reachable from ddrs but not yet called.

- [ ] **Step 6: No ddrs commit yet — Task 4 picks it up**

The fork-only change is recorded by the fork's commit history. ddrs's `Cargo.lock` may show a new rev hash on the fork URL; commit that:

```
git add Cargo.lock
git commit -m "Bump cubecl-cuda fork to ddrs-sp7-stream-accessor with stream() accessor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: burn-cubecl fork — `pub fn CubeTensor::from_handle(...)`

**Files (on the burn fork):**
- Modify: `crates/burn-cubecl/src/tensor/base.rs` (or wherever `CubeTensor` is defined)

- [ ] **Step 1: Read the existing struct + private constructors**

```
grep -n "pub struct CubeTensor\|impl.*CubeTensor\|fn new\|client:\|handle:\|meta:" \
    /tmp/sp7-forks/burn/crates/burn-cubecl/src/tensor/base.rs | head -30
```

Identify which fields exist (`client`, `handle`, `meta`, `device`, `dtype` per the SP-6 Task 6 finding) and whether any existing constructor takes a handle.

- [ ] **Step 2: Add the public constructor**

Inside `impl<R: CubeRuntime> CubeTensor<R>`, append:

```rust
    /// Construct a tensor from an existing cubecl `Handle` plus shape
    /// and dtype.
    ///
    /// Used by external libraries that allocate via the cubecl client
    /// (`client.create(size_bytes) -> Handle`) and need to register the
    /// result as a BURN tensor without copying.
    pub fn from_handle(
        client: ComputeClient<R::Server, R::Channel>,
        handle: Handle,
        shape: Shape,
        dtype: DType,
        device: R::Device,
    ) -> Self {
        Self {
            client,
            handle,
            meta: TensorMetadata::from(shape),
            device,
            dtype,
        }
    }
```

The exact `meta` construction depends on the existing `TensorMetadata`/`Shape` API. If `TensorMetadata::from(shape)` doesn't compile, look at the existing `pub(crate)` constructor (or `from_data`) for the right initialization. The contract is: pass a contiguous rank-1 shape and dtype, get a valid CubeTensor over the externally-allocated buffer.

If `Handle` is not nameable from outside the crate, add a `pub use crate::handle::Handle` at the crate root.

- [ ] **Step 3: Verify the fork still builds**

```
cd /tmp/sp7-forks/burn
cargo build -p burn-cubecl 2>&1 | tail -10
```

- [ ] **Step 4: Commit + push the fork**

```
cd /tmp/sp7-forks/burn
git add crates/burn-cubecl/src/tensor/base.rs
git commit -m "Expose CubeTensor::from_handle for foreign-buffer registration

External libraries (cuSPARSE, cuBLAS, cuRAND) allocate via
client.create(size) -> Handle and need to register the allocation as
a CubeTensor without copying. This pub constructor exposes the
private path. No behavioral change.

(SP-7 patch for downstream ddrs; will upstream as a real PR.)"
git push origin ddrs-sp7-primitive-ctor
```

- [ ] **Step 5: Refresh + verify in ddrs**

```
cd /home/tbindas/projects/ddrs
cargo update -p burn-cubecl 2>&1 | tail -5
cargo build --lib 2>&1 | tail -10
cargo test --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | grep "test result" | tail -3
```

- [ ] **Step 6: Bump Cargo.lock + commit**

```
git add Cargo.lock
git commit -m "Bump burn-cubecl fork to ddrs-sp7-primitive-ctor with from_handle ctor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add `cubecl_stream_active` + `cube_tensor_to_primitive` + `compute_client` helpers

**Files:**
- Modify: `src/sparse/cusparse.rs`

Add three new helper functions that use the freshly-exposed accessors. Don't touch `cusparse_forward` / `cusparse_backward_solve` yet — Tasks 5–7 swap them over.

- [ ] **Step 1: Add `compute_client::<B>(device)` helper**

In `src/sparse/cusparse.rs`, around line 425 (above the existing `cubecl_cuda_stream`):

```rust
use cubecl::client::ComputeClient;

/// Obtain the cubecl ComputeClient for the given BURN Cuda device.
///
/// Panics if `B` is not `Cuda<f32, i32>` — caller must gate via
/// `dispatch::backend_is_cuda::<B>()`.
fn compute_client<B: Backend + 'static>(
    device: &B::Device,
) -> ComputeClient<<CudaRuntime as cubecl::Runtime>::Server, <CudaRuntime as cubecl::Runtime>::Channel> {
    use std::any::TypeId;
    assert_eq!(
        TypeId::of::<B::Device>(),
        TypeId::of::<<burn::backend::Cuda<f32, i32> as Backend>::Device>(),
        "compute_client requires Cuda<f32, i32> backend",
    );
    // SAFETY: TypeId match above guarantees layout compatibility between
    // B::Device and CudaDevice. We borrow the address for cubecl::Runtime
    // initialization.
    let cuda_device: &cubecl::cuda::CudaDevice = unsafe { std::mem::transmute(device) };
    ComputeClient::<_, _>::load(cuda_device)
}
```

The exact `ComputeClient::load` path may differ — refer to how `cubecl_cuda_stream` already does this in the current code (the SP-6 Task 7 implementer wrote it). Use the same pattern.

- [ ] **Step 2: Add `cubecl_stream_active::<B>(device) -> CUstream`**

Below `compute_client`:

```rust
/// Returns cubecl-cuda's active stream via the SP-7 fork's `pub stream()`
/// accessor. Replaces SP-6's `cubecl_cuda_stream` (which returned a
/// dedicated FALLBACK_STREAM and required host syncs).
pub(crate) fn cubecl_stream_active<B: Backend + 'static>(
    device: &B::Device,
) -> CUstream {
    let client = compute_client::<B>(device);
    // exclusive() runs the closure on the server's CUDA-context-bound thread.
    client.exclusive(|server| server.stream())
}
```

If the cubecl `ComputeClient` exposes `server()` directly rather than `exclusive(|s| ...)`, use that; the SP-6 Task 7 implementer already explored this API surface and used `exclusive`. Match whatever compiles.

- [ ] **Step 3: Add `cube_tensor_to_primitive::<B>` helper**

Below `cubecl_stream_active`:

```rust
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;

/// Convert a `CubeTensor<CudaRuntime>` into the BURN backend's
/// `FloatTensorPrimitive`. Inverse of `primitive_as_cuda_view`.
///
/// SAFETY: the caller has verified via TypeId that `B == Cuda<f32, i32>`.
/// The `CubeTensor` and `B::FloatTensorPrimitive` share layout under that
/// equality. The transmute borrows ownership of the cubecl handle into
/// BURN's tape; the cubecl-allocated buffer is freed when BURN drops the
/// tensor.
pub(crate) fn cube_tensor_to_primitive<B: Backend + 'static>(
    cube: CubeTensor<CudaRuntime>,
) -> B::FloatTensorPrimitive {
    use std::any::TypeId;
    assert_eq!(
        TypeId::of::<B>(),
        TypeId::of::<burn::backend::Cuda<f32, i32>>(),
        "cube_tensor_to_primitive requires Cuda<f32, i32> backend",
    );
    // SAFETY: TypeId equality guarantees identical layout.
    unsafe { std::mem::transmute_copy::<CubeTensor<CudaRuntime>, B::FloatTensorPrimitive>(&cube) }
}
```

`transmute_copy` because `CubeTensor` is not `Copy`; we move it through the transmute. Drop semantics belong to whichever struct holds the handle on the other side — BURN's primitive type takes ownership.

- [ ] **Step 4: SPIKE test — both helpers round-trip**

Append to `tests/cusparse_ptr_spike.rs`:

```rust
#[test]
fn cubecl_active_stream_is_non_null_and_not_fallback() {
    type B = burn::backend::Cuda<f32, i32>;
    type Dev = <B as burn::tensor::backend::Backend>::Device;
    let cuda_available = std::panic::catch_unwind(|| {
        let _d: Dev = Default::default();
    }).is_ok();
    if !cuda_available {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let device: Dev = Default::default();
    let _t = burn::tensor::Tensor::<B, 1>::from_floats([0.0_f32], &device);
    let active = ddrs::sparse::cusparse::__spike_active_stream::<B>(&device);
    let fallback = ddrs::sparse::cusparse::__spike_get_stream::<B>(&device);
    assert!(!active.is_null(), "cubecl active stream is null");
    assert_ne!(active, fallback,
        "expected active stream and FALLBACK_STREAM to be different handles");
}

#[test]
fn cube_tensor_round_trip_to_primitive() {
    type B = burn::backend::Cuda<f32, i32>;
    type Dev = <B as burn::tensor::backend::Backend>::Device;
    let cuda_available = std::panic::catch_unwind(|| {
        let _d: Dev = Default::default();
    }).is_ok();
    if !cuda_available {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let device: Dev = Default::default();
    // Use the test-only spike helper added below in Step 5.
    let recovered: Vec<f32> = ddrs::sparse::cusparse::__spike_cube_round_trip::<B>(&device, vec![10.0, 20.0, 30.0]);
    assert_eq!(recovered, vec![10.0, 20.0, 30.0]);
}
```

- [ ] **Step 5: Add `#[doc(hidden)]` test entry points**

Below the three helpers in `src/sparse/cusparse.rs`:

```rust
#[doc(hidden)]
pub fn __spike_active_stream<B: Backend + 'static>(device: &B::Device) -> CUstream {
    cubecl_stream_active::<B>(device)
}

#[doc(hidden)]
pub fn __spike_cube_round_trip<B: Backend + 'static>(
    device: &B::Device,
    data: Vec<f32>,
) -> Vec<f32> {
    use burn::tensor::Tensor;
    use cubecl::Runtime;
    let client = compute_client::<B>(device);
    let bytes = data.len() * std::mem::size_of::<f32>();
    let handle = client.create(bytes);
    // Upload data into the cubecl-owned handle.
    client.write(&handle.binding(), bytemuck::cast_slice(&data));
    let cube = CubeTensor::<CudaRuntime>::from_handle(
        client.clone(),
        handle,
        cubecl::ir::Shape::new(vec![data.len()]),
        burn::tensor::DType::F32,
        // SAFETY: TypeId of B::Device == CudaDevice (asserted in compute_client).
        unsafe { std::mem::transmute_copy(device) },
    );
    let prim = cube_tensor_to_primitive::<B>(cube);
    let tensor: Tensor<B, 1> = Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(prim));
    tensor.into_data().to_vec::<f32>().unwrap()
}
```

The exact `client.write(...)` / `client.create(...)` signatures depend on cubecl's `ComputeClient` API. Look at how `burn-cubecl` itself constructs CubeTensors for data inputs — the same pattern applies. If `bytemuck` is not already in `Cargo.toml`, add it as a test dep or use a manual byte slice cast inside an `unsafe` block.

- [ ] **Step 6: Run the spike tests**

```
cargo test --test cusparse_ptr_spike cubecl_active_stream 2>&1 | tail -10
cargo test --test cusparse_ptr_spike cube_tensor_round_trip 2>&1 | tail -10
```

Expected: both pass on CUDA host.

If `cube_tensor_round_trip` fails with garbage data, the `from_handle` constructor in the burn-cubecl fork is missing a metadata field. Inspect with the existing `pub(crate)` constructors in burn-cubecl and add what's missing to the fork (Task 3 may need a revision; report and revise).

- [ ] **Step 7: Verify CPU regression unchanged**

```
cargo test --lib 2>&1 | grep "test result" | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: same counts; ABSOLUTE MATCH.

- [ ] **Step 8: Commit**

```
git add src/sparse/cusparse.rs tests/cusparse_ptr_spike.rs
git commit -m "Add cubecl_stream_active + cube_tensor_to_primitive + compute_client

The three helpers use the SP-7 fork accessors (CudaServer::stream and
CubeTensor::from_handle) to make subsequent SP-7 tasks drop SP-6's
FALLBACK_STREAM + cuMemAllocAsync + host-roundtrip pattern. Spike
tests confirm round-trip via cubecl-owned handle preserves data and
that cubecl's active stream differs from the dedicated fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Rewrite `cusparse_forward` for cubecl stream + zero-copy x

**Files:**
- Modify: `src/sparse/cusparse.rs`

Replace the FALLBACK_STREAM-and-host-roundtrip body with the new flow. Backward solve (Task 6) is the same pattern.

- [ ] **Step 1: Replace the body of `cusparse_forward`**

In `src/sparse/cusparse.rs`, the existing `pub(crate) fn cusparse_forward<B: Backend>(...)` (around line 968) gets rewritten. Replace ONLY the body — keep the signature.

```rust
pub(crate) fn cusparse_forward<B: Backend + 'static>(
    pattern: &CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    // 1. Lazy-build the pattern cache (one-time per CsrPattern).
    let cache = unsafe { ensure_cuda_cache(pattern) };

    // 2. Bind cuSPARSE to cubecl's active stream — no FALLBACK_STREAM,
    //    no host sync needed. cubecl's queued kernels writing a_values + b
    //    are already on this stream; cuSPARSE runs after them on the same
    //    stream automatically.
    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        let rc = cudarc::cusparse::sys::cusparseSetStream(cache.handle, stream);
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS,
                   "cusparseSetStream forward failed: {:?}", rc);
    }

    // 3. Extract device pointers for a_values and b. No B::sync — the
    //    pointers may reference buffers still being written by cubecl,
    //    but cuSPARSE on the same stream will wait for them naturally.
    let a_view = primitive_as_cuda_view::<B>(a_values_prim).expect("Cuda required");
    let b_view = primitive_as_cuda_view::<B>(b_prim).expect("Cuda required");
    let n = pattern.n;

    // 4. Allocate x via cubecl's client. The returned Handle owns the
    //    device memory; freeing happens when the CubeTensor below is
    //    dropped by BURN's autograd tape.
    let client = compute_client::<B>(device);
    let n_bytes = n * std::mem::size_of::<f32>();
    let x_handle = client.create(n_bytes);
    let x_ptr_u64 = x_handle.binding().resource_handle();  // exact accessor TBD
    let x_ptr = x_ptr_u64 as *mut f32;

    // 5. cuSPARSE solve. Same descriptor + updateMatrix dance as SP-6.
    unsafe {
        let rc = cudarc::cusparse::sys::cusparseCsrSetPointers(
            cache.sp_mat,
            cache.d_crow as *mut _,
            cache.d_col  as *mut _,
            a_view.ptr as *mut _,
        );
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);

        let rc = cudarc::cusparse::sys::cusparseSpSV_updateMatrix(
            cache.handle, cache.desc_forward,
            a_view.ptr as *mut _,
            cudarc::cusparse::sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_GENERAL,
        );
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);

        let mut b_dn = std::ptr::null_mut();
        let rc = cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut b_dn, n as i64, b_view.ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);

        let mut x_dn = std::ptr::null_mut();
        let rc = cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut x_dn, n as i64, x_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);

        let alpha: f32 = 1.0;
        let rc = cudarc::cusparse::sys::cusparseSpSV_solve(
            cache.handle,
            cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const _ as *const _,
            cache.sp_mat, b_dn, x_dn,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
            cudarc::cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
            cache.desc_forward,
        );
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS,
                   "cusparseSpSV_solve forward failed: {:?}", rc);

        cudarc::cusparse::sys::cusparseDestroyDnVec(b_dn);
        cudarc::cusparse::sys::cusparseDestroyDnVec(x_dn);
        // NO cuStreamSynchronize. The next BURN op on cubecl's stream will
        // see x ready because both ran on the same stream.
    }

    // 6. Wrap the cubecl Handle as a CubeTensor → B::FloatTensorPrimitive.
    //    The handle ownership transfers to the resulting primitive; BURN
    //    drops the cubecl allocation when the autograd tape releases it.
    let cube = CubeTensor::<CudaRuntime>::from_handle(
        client,
        x_handle,
        cubecl::ir::Shape::new(vec![n]),
        burn::tensor::DType::F32,
        // SAFETY: B::Device == CudaDevice asserted by compute_client.
        unsafe { std::mem::transmute_copy(device) },
    );
    cube_tensor_to_primitive::<B>(cube)
}
```

Several names above (`resource_handle()`, `Shape::new(vec![n])`, `CudaDevice` direct transmute) are based on the SP-6 Task 6/7 implementer's notes. Refine each to the exact API the burn-cubecl fork exposes — the test in Task 4 Step 4 already exercises round-trip through these calls, so any name mismatch breaks Task 4 before this one.

- [ ] **Step 2: Add `+ 'static` bound everywhere `cusparse_forward` is called**

`dispatch::forward_primitive` currently has the bound; verify it compiles cleanly. If not, propagate `+ 'static` up the chain.

- [ ] **Step 3: Run V5**

```
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -30
```

Expected: V5 PASSES with the rewritten forward. If max abs / rel jumps past `1e-3 / 1e-4`, the new shared-stream model has a subtle ordering bug or the `from_handle` constructor produces a malformed CubeTensor. STOP and report with the divergence numbers.

- [ ] **Step 4: Run sparse_cusparse_v5 forward_cuda_smoke + CPU regression**

```
cargo test --test sparse_cusparse_v5 forward_cuda_smoke 2>&1 | tail -10
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: forward smoke passes, gradcheck passes (CPU path unchanged), ABSOLUTE MATCH.

- [ ] **Step 5: Commit**

```
git add src/sparse/cusparse.rs
git commit -m "Rewrite cusparse_forward for shared stream + zero-copy x

Drops B::sync(device), FALLBACK_STREAM, cuMemcpyDtoH, and B::float_from_data.
The new path runs cuSPARSE on cubecl's active stream and returns a
CubeTensor wrapping a cubecl-allocated buffer. No host roundtrip.

V5 bit-match preserved; the math is identical, only the staging changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Rewrite `cusparse_backward_solve` for cubecl stream + zero-copy y

**Files:**
- Modify: `src/sparse/cusparse.rs`

Mirror Task 5 with `desc_backward` + `CUSPARSE_OPERATION_TRANSPOSE`.

- [ ] **Step 1: Replace the body of `cusparse_backward_solve`**

Use the same structure as the rewritten `cusparse_forward`. Two differences:
- `cache.desc_backward` instead of `cache.desc_forward`.
- `CUSPARSE_OPERATION_TRANSPOSE` instead of `NON_TRANSPOSE`.

```rust
pub(crate) fn cusparse_backward_solve<B: Backend + 'static>(
    pattern: &CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    let cache = unsafe { ensure_cuda_cache(pattern) };

    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        let rc = cudarc::cusparse::sys::cusparseSetStream(cache.handle, stream);
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS,
                   "cusparseSetStream backward failed: {:?}", rc);
    }

    let a_view = primitive_as_cuda_view::<B>(a_values_prim).expect("Cuda required");
    let b_view = primitive_as_cuda_view::<B>(b_prim).expect("Cuda required");
    let n = pattern.n;

    let client = compute_client::<B>(device);
    let n_bytes = n * std::mem::size_of::<f32>();
    let y_handle = client.create(n_bytes);
    let y_ptr = y_handle.binding().resource_handle() as *mut f32;

    unsafe {
        cudarc::cusparse::sys::cusparseCsrSetPointers(
            cache.sp_mat,
            cache.d_crow as *mut _,
            cache.d_col  as *mut _,
            a_view.ptr as *mut _,
        );
        cudarc::cusparse::sys::cusparseSpSV_updateMatrix(
            cache.handle, cache.desc_backward,
            a_view.ptr as *mut _,
            cudarc::cusparse::sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_GENERAL,
        );

        let mut b_dn = std::ptr::null_mut();
        cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut b_dn, n as i64, b_view.ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );
        let mut y_dn = std::ptr::null_mut();
        cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut y_dn, n as i64, y_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );

        let alpha: f32 = 1.0;
        let rc = cudarc::cusparse::sys::cusparseSpSV_solve(
            cache.handle,
            cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const _ as *const _,
            cache.sp_mat, b_dn, y_dn,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
            cudarc::cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
            cache.desc_backward,
        );
        assert_eq!(rc, cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS,
                   "cusparseSpSV_solve backward failed: {:?}", rc);

        cudarc::cusparse::sys::cusparseDestroyDnVec(b_dn);
        cudarc::cusparse::sys::cusparseDestroyDnVec(y_dn);
    }

    let cube = CubeTensor::<CudaRuntime>::from_handle(
        client,
        y_handle,
        cubecl::ir::Shape::new(vec![n]),
        burn::tensor::DType::F32,
        unsafe { std::mem::transmute_copy(device) },
    );
    cube_tensor_to_primitive::<B>(cube)
}
```

- [ ] **Step 2: Run V5 + backward smoke + regression**

```
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -30
cargo test --test sparse_cusparse_v5 backward_cuda_smoke 2>&1 | tail -10
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: V5 passes, backward smoke passes, gradcheck passes, ABSOLUTE MATCH.

- [ ] **Step 3: Commit**

```
git add src/sparse/cusparse.rs
git commit -m "Rewrite cusparse_backward_solve for shared stream + zero-copy y

Mirror of Task 5's forward rewrite: cusparse runs on cubecl's active
stream, output gradient wraps a cubecl-allocated buffer. Drops
B::sync, FALLBACK_STREAM, host roundtrip.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Rewrite `build_cuda_pattern_cache` to use cubecl allocations for structural arrays + workspaces

**Files:**
- Modify: `src/sparse/cusparse.rs`

The existing builder uses `cuMemAllocAsync` on FALLBACK_STREAM for `d_crow`, `d_col`, `d_row_for_nnz`, `workspace_forward`, `workspace_backward`. To delete FALLBACK_STREAM (Task 8), these allocations must move to cubecl-managed buffers as well.

- [ ] **Step 1: Rewrite the body**

Replace `cuMemAllocAsync` + `cuMemcpyHtoDAsync` with cubecl `client.create(bytes)` + `client.write(handle.binding(), bytes_slice)` for each structural array. Store the resulting `Handle`s on `CudaPatternCache` (renaming the fields from `*_u64: u64` to `*_handle: cubecl::server::Handle`):

```rust
pub(crate) struct CudaPatternCache {
    pub(crate) handle: cudarc::cusparse::sys::cusparseHandle_t,
    pub(crate) d_crow: cubecl::server::Handle,
    pub(crate) d_col: cubecl::server::Handle,
    pub(crate) d_row_for_nnz: cubecl::server::Handle,
    pub(crate) sp_mat: cudarc::cusparse::sys::cusparseSpMatDescr_t,
    pub(crate) desc_forward: cudarc::cusparse::sys::cusparseSpSVDescr_t,
    pub(crate) desc_backward: cudarc::cusparse::sys::cusparseSpSVDescr_t,
    pub(crate) workspace_forward: cubecl::server::Handle,
    pub(crate) workspace_backward: cubecl::server::Handle,
    _not_send: std::marker::PhantomData<*mut ()>,
}
```

The exact `cubecl::server::Handle` path matches what Task 4 already imports. Verify.

Update `build_cuda_pattern_cache`:

```rust
fn build_cuda_pattern_cache(pattern: &crate::sparse::CsrPattern) -> CudaPatternCache {
    // Borrow the CUDA device the BURN backend is using. This function is
    // only called inside ensure_cuda_cache from cusparse_{forward,backward};
    // those callers have already invoked compute_client which initialises
    // the cubecl client.
    type B = burn::backend::Cuda<f32, i32>;
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let client = compute_client::<B>(&device);

    let crow_bytes  = (pattern.n + 1) * std::mem::size_of::<i32>();
    let col_bytes   = pattern.nnz()   * std::mem::size_of::<i32>();
    let row_bytes   = pattern.nnz()   * std::mem::size_of::<i32>();

    let d_crow = client.create(crow_bytes);
    client.write(&d_crow.binding(), bytemuck::cast_slice(&pattern.crow));
    let d_col = client.create(col_bytes);
    client.write(&d_col.binding(),  bytemuck::cast_slice(&pattern.col));
    let d_row_for_nnz = client.create(row_bytes);
    client.write(&d_row_for_nnz.binding(), bytemuck::cast_slice(&pattern.row_for_nnz));

    // cuSPARSE handle, sp_mat descriptor — same as SP-6 but read pointers
    // from the cubecl handles via .resource_handle().
    let crow_ptr = d_crow.binding().resource_handle() as *mut i32;
    let col_ptr  = d_col.binding().resource_handle()  as *mut i32;

    let mut handle = std::ptr::null_mut();
    unsafe {
        cudarc::cusparse::sys::cusparseCreate(&mut handle);
    }

    let mut sp_mat = std::ptr::null_mut();
    let n = pattern.n as i64;
    let nnz = pattern.nnz() as i64;
    unsafe {
        cudarc::cusparse::sys::cusparseCreateCsr(
            &mut sp_mat,
            n, n, nnz,
            crow_ptr as *mut _,
            col_ptr as *mut _,
            std::ptr::null_mut(),  // values: set per-call
            cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cudarc::cusparse::sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );
        let fill = cudarc::cusparse::sys::cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER;
        cudarc::cusparse::sys::cusparseSpMatSetAttribute(
            sp_mat,
            cudarc::cusparse::sys::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_FILL_MODE,
            &fill as *const _ as *const _, std::mem::size_of_val(&fill),
        );
        let diag = cudarc::cusparse::sys::cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT;
        cudarc::cusparse::sys::cusparseSpMatSetAttribute(
            sp_mat,
            cudarc::cusparse::sys::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_DIAG_TYPE,
            &diag as *const _ as *const _, std::mem::size_of_val(&diag),
        );
    }

    // SpSV descriptors + workspaces.
    let mut desc_forward = std::ptr::null_mut();
    let mut desc_backward = std::ptr::null_mut();
    unsafe {
        cudarc::cusparse::sys::cusparseSpSV_createDescr(&mut desc_forward);
        cudarc::cusparse::sys::cusparseSpSV_createDescr(&mut desc_backward);
    }

    // Buffer-size probe via dummy dense vector descriptors (need a valid
    // values pointer for SetPointers + updateMatrix during analysis).
    // The SP-6 analysis uses an all-ones dummy a_values; mirror that.
    let dummy_a = client.create(nnz as usize * std::mem::size_of::<f32>());
    {
        let ones = vec![1.0_f32; nnz as usize];
        client.write(&dummy_a.binding(), bytemuck::cast_slice(&ones));
    }
    let dummy_a_ptr = dummy_a.binding().resource_handle() as *mut f32;
    let dummy_b = client.create(pattern.n * std::mem::size_of::<f32>());
    let dummy_b_ptr = dummy_b.binding().resource_handle() as *mut f32;
    let dummy_x = client.create(pattern.n * std::mem::size_of::<f32>());
    let dummy_x_ptr = dummy_x.binding().resource_handle() as *mut f32;

    let alpha: f32 = 1.0;

    let mut ws_fwd_size: usize = 0;
    let mut ws_bwd_size: usize = 0;

    unsafe {
        cudarc::cusparse::sys::cusparseCsrSetPointers(sp_mat, crow_ptr as *mut _, col_ptr as *mut _, dummy_a_ptr as *mut _);

        let mut b_dn = std::ptr::null_mut();
        cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut b_dn, n, dummy_b_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );
        let mut x_dn = std::ptr::null_mut();
        cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut x_dn, n, dummy_x_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );

        cudarc::cusparse::sys::cusparseSpSV_bufferSize(
            handle,
            cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const _ as *const _,
            sp_mat, b_dn, x_dn,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
            cudarc::cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
            desc_forward,
            &mut ws_fwd_size as *mut _,
        );

        cudarc::cusparse::sys::cusparseSpSV_bufferSize(
            handle,
            cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const _ as *const _,
            sp_mat, b_dn, x_dn,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
            cudarc::cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
            desc_backward,
            &mut ws_bwd_size as *mut _,
        );

        cudarc::cusparse::sys::cusparseDestroyDnVec(b_dn);
        cudarc::cusparse::sys::cusparseDestroyDnVec(x_dn);
    }

    // Allocate workspaces via cubecl.
    let workspace_forward  = client.create(ws_fwd_size.max(1));
    let workspace_backward = client.create(ws_bwd_size.max(1));

    // Re-do dense descriptors for analysis (they were destroyed above).
    unsafe {
        let mut b_dn = std::ptr::null_mut();
        cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut b_dn, n, dummy_b_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );
        let mut x_dn = std::ptr::null_mut();
        cudarc::cusparse::sys::cusparseCreateDnVec(
            &mut x_dn, n, dummy_x_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
        );

        cudarc::cusparse::sys::cusparseSpSV_analysis(
            handle,
            cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const _ as *const _,
            sp_mat, b_dn, x_dn,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
            cudarc::cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
            desc_forward,
            workspace_forward.binding().resource_handle() as *mut _,
        );

        cudarc::cusparse::sys::cusparseSpSV_analysis(
            handle,
            cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const _ as *const _,
            sp_mat, b_dn, x_dn,
            cudarc::cusparse::sys::cudaDataType_t::CUDA_R_32F,
            cudarc::cusparse::sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
            desc_backward,
            workspace_backward.binding().resource_handle() as *mut _,
        );

        cudarc::cusparse::sys::cusparseDestroyDnVec(b_dn);
        cudarc::cusparse::sys::cusparseDestroyDnVec(x_dn);
    }

    // dummy_a, dummy_b, dummy_x drop here — cubecl frees the buffers.

    CudaPatternCache {
        handle,
        d_crow, d_col, d_row_for_nnz,
        sp_mat,
        desc_forward, desc_backward,
        workspace_forward, workspace_backward,
        _not_send: std::marker::PhantomData,
    }
}
```

- [ ] **Step 2: Update the `Drop` impl**

The old `Drop` used `cuMemFreeAsync` on FALLBACK_STREAM. Now the cubecl `Handle`s drop automatically when the cache is dropped — they free themselves via cubecl's allocator. Reduce the `Drop` to descriptor + handle cleanup only:

```rust
impl Drop for CudaPatternCache {
    fn drop(&mut self) {
        // SAFETY: descriptors must be destroyed before any device buffer
        // they reference. The cubecl Handle fields (d_crow, d_col,
        // d_row_for_nnz, workspace_*) drop AFTER this Drop body returns,
        // so descriptor destruction happens first. cubecl manages buffer
        // lifetime via the Handle.
        unsafe {
            cudarc::cusparse::sys::cusparseSpSV_destroyDescr(self.desc_forward);
            cudarc::cusparse::sys::cusparseSpSV_destroyDescr(self.desc_backward);
            cudarc::cusparse::sys::cusparseDestroySpMat(self.sp_mat);
            cudarc::cusparse::sys::cusparseDestroy(self.handle);
        }
    }
}
```

- [ ] **Step 3: Update all callers reading `cache.d_crow` etc.**

The forward + backward rewrites in Tasks 5/6 already read `cache.d_crow as *mut _`. Change to:

```rust
cache.d_crow.binding().resource_handle() as *mut _
```

(And likewise for `d_col`, `d_row_for_nnz`, `workspace_forward`, `workspace_backward`.)

- [ ] **Step 4: Run V5 + smokes + regression**

```
cargo test --release --test sparse_cusparse_v5 2>&1 | tail -20
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: V5 + forward + backward smokes pass; gradcheck unchanged; ABSOLUTE MATCH.

- [ ] **Step 5: Commit**

```
git add src/sparse/cusparse.rs
git commit -m "Allocate CudaPatternCache buffers via cubecl client (zero raw cuMem*)

build_cuda_pattern_cache uses client.create + client.write for the
five structural arrays and two workspaces. Handles drop with the
cache; cubecl frees the underlying buffers. The Drop impl is now
descriptor-only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Delete `FALLBACK_STREAM` and all unreachable code

**Files:**
- Modify: `src/sparse/cusparse.rs`

With Tasks 5–7 routing every cuSPARSE call through cubecl's stream and cubecl-allocated buffers, the entire FALLBACK_STREAM apparatus is dead code. Remove it.

- [ ] **Step 1: Identify and delete**

Delete the following from `src/sparse/cusparse.rs`:
- `static FALLBACK_STREAM: std::sync::OnceLock<SendStream> = ...`
- `struct SendStream(CUstream)` + the `unsafe impl Send/Sync` wrappers
- `fn cubecl_cuda_stream<B>(device) -> CUstream` body referencing FALLBACK_STREAM (rename or delete; if any test still imports it, route those to `cubecl_stream_active` via the test-only `__spike_get_stream`)
- `fn async_alloc<T>(n: usize, stream: CUstream) -> u64`
- Any helper that calls `cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemsetD8Async`, `cuMemcpyDtoH`, `cuMemcpyHtoDAsync`
- Any remaining `B::sync(device)` calls — they were already removed in Tasks 5–6, but a final grep confirms

After deletion:
```
grep -n "FALLBACK_STREAM\|cuMemAllocAsync\|cuMemFreeAsync\|cuMemcpyDtoH\|cuMemcpyHtoDAsync\|B::sync\|async_alloc\|SendStream" src/sparse/cusparse.rs
```

Expected: zero hits.

- [ ] **Step 2: Update `__spike_get_stream` (if kept)**

If `tests/cusparse_ptr_spike.rs::cubecl_stream_is_non_null` still calls `__spike_get_stream`, repoint it to return `cubecl_stream_active`:

```rust
#[doc(hidden)]
pub fn __spike_get_stream<B: Backend + 'static>(device: &B::Device) -> CUstream {
    cubecl_stream_active::<B>(device)
}
```

(Or delete both `__spike_get_stream` and the test that uses it — Task 4 added `__spike_active_stream` which is the proper successor.)

- [ ] **Step 3: Full regression sweep**

```
cargo build --lib 2>&1 | tail -10
cargo test --lib 2>&1 | grep "test result" | tail -5
cargo test --test sparse_gradcheck 2>&1 | tail -5
cargo test --test cusparse_ptr_spike 2>&1 | tail -10
cargo test --release --test sparse_cusparse_v5 2>&1 | tail -20
cargo test --test training_verification v1_loss_matches 2>&1 | tail -5
cargo test --test training_verification v3_train 2>&1 | tail -5
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected: every test passes with same counts; ABSOLUTE MATCH.

- [ ] **Step 4: Clippy on the new code**

```
cargo clippy --all-targets 2>&1 | grep -E "(sparse/cusparse|sparse/dispatch)" | head -20
```

Expected: no new warnings. Fix anything inline (e.g., `unused_imports`, `needless_borrow`).

- [ ] **Step 5: Commit**

```
git add src/sparse/cusparse.rs tests/cusparse_ptr_spike.rs
git commit -m "Delete FALLBACK_STREAM + raw cuMem* + SP-6 host-roundtrip path

Every cuSPARSE call now runs on cubecl's active stream with cubecl-
allocated buffers, so the SP-6 fallback (FALLBACK_STREAM, SendStream,
async_alloc, cuMemcpyDtoH, cuMemcpyHtoDAsync, cuMemFreeAsync,
B::sync(device)) is unreachable. Remove it.

V5 / sparse_gradcheck / V1 / V3 / sandbox all green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: V6 speedup assertion test

**Files:**
- Create: `tests/sparse_cusparse_v6.rs`

- [ ] **Step 1: Write the test**

```rust
//! SP-7 V6: assert CUDA training is materially faster than CPU.
//!
//! Run manually: `cargo test --release -- --ignored v6_cuda_is_faster`.
//! Marked #[ignore] because it spawns `cargo run --bin train` (slow) and
//! depends on live data files at the paths in config/merit_training.yaml.

use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn data_files_present() -> bool {
    // Probe a couple of the required paths from merit_training.yaml.
    Path::new("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr").exists()
}

fn cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        type B = burn::backend::Cuda<f32, i32>;
        let _d: <B as burn::tensor::backend::Backend>::Device = Default::default();
    }).is_ok()
}

/// Build a temp YAML that's a copy of merit_training.yaml with the
/// sparse_solver param set to `value`.
fn write_override_yaml(value: &str) -> std::path::PathBuf {
    let base = std::fs::read_to_string("config/merit_training.yaml")
        .expect("read merit_training.yaml");
    // Remove any existing sparse_solver line (commented or not).
    let mut lines: Vec<String> = base.lines()
        .filter(|l| !l.trim_start().starts_with("sparse_solver:")
                 && !l.trim_start().starts_with("# sparse_solver:"))
        .map(String::from)
        .collect();
    // Insert under params:. Naïve search — production config has `params:`
    // on its own line; append `  sparse_solver: <value>` right after.
    let params_idx = lines.iter().position(|l| l.trim_start_matches(' ').starts_with("params:"))
        .expect("params: block not found in merit_training.yaml");
    lines.insert(params_idx + 1, format!("  sparse_solver: {value}"));
    let path = std::path::PathBuf::from(format!("/tmp/v6_{value}.yaml"));
    std::fs::write(&path, lines.join("\n")).expect("write override yaml");
    path
}

fn run_train(config_path: &Path) -> f32 {
    let ckpt_dir = format!("/tmp/v6_smoke_{}", config_path.file_stem().unwrap().to_string_lossy());
    let _ = std::fs::remove_dir_all(&ckpt_dir);
    let start = Instant::now();
    let output = Command::new("cargo")
        .args(&["run", "--release", "--bin", "train", "--",
                "--config", config_path.to_str().unwrap(),
                "--checkpoint-dir", &ckpt_dir,
                "--max-mini-batches", "3"])
        .output()
        .expect("spawn cargo run");
    let elapsed = start.elapsed();
    assert!(output.status.success(),
        "train binary failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr));
    elapsed.as_secs_f32() / 60.0
}

#[test]
#[ignore]
fn v6_cuda_is_faster_than_cpu_on_smoke_train() {
    if !data_files_present() {
        eprintln!("skipping: data files not present");
        return;
    }
    if !cuda_available() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    let cpu_yaml  = write_override_yaml("cpu");
    let cuda_yaml = write_override_yaml("cuda");

    let cpu_minutes  = run_train(&cpu_yaml);
    let cuda_minutes = run_train(&cuda_yaml);

    eprintln!("V6 timing: cpu={cpu_minutes:.2} min, cuda={cuda_minutes:.2} min, ratio={:.3}",
              cuda_minutes / cpu_minutes);
    assert!(cuda_minutes < cpu_minutes * 0.7,
        "SP-7 speedup goal missed: cuda={cuda_minutes:.2} min, \
         cpu={cpu_minutes:.2} min, ratio={:.3} (target ≤ 0.7)",
        cuda_minutes / cpu_minutes);
}
```

- [ ] **Step 2: Run V6**

```
cargo test --release --test sparse_cusparse_v6 -- --ignored v6_cuda_is_faster 2>&1 | tail -30
```

Expected: test passes. Note the recorded `cpu_minutes` / `cuda_minutes` / `ratio` line for the commit.

If V6 fails:
- Record the timings.
- Determine whether the failure is "no speedup at all" (ratio ~1.0) or "marginal speedup" (ratio in 0.7..0.9). The former points to a still-present sync; the latter is borderline and needs deeper profiling.
- STOP and report. SP-7 must not claim success without V6.

- [ ] **Step 3: Commit (only if V6 passes)**

```
git add tests/sparse_cusparse_v6.rs
git commit -m "Add V6 speedup assertion: CUDA wall-time ≤ 0.7× CPU on smoke train

V6 spawns the train binary twice (sparse_solver: cpu, then cuda),
times the 3-mini-batch smoke each way, and asserts CUDA is ≤ 0.7×
CPU. Marked #[ignore] — depends on live data + slow cargo subprocess
spawn; run manually with --ignored.

Recorded timings (this host): cpu=<X.XX> min, cuda=<Y.YY> min,
ratio=<R.RRR>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(Fill in the actual numbers from Step 2's output.)

---

### Task 10: Final regression sweep + close-out

**Files:**
- None modified — pure verification + commit.

- [ ] **Step 1: Full unit + integration test sweep**

```
cargo test --lib 2>&1 | grep "test result" | tail -10
cargo test --test sparse_gradcheck 2>&1 | grep "test result" | tail -3
cargo test --test sparse_cusparse_v5 2>&1 | grep "test result" | tail -3
cargo test --test cusparse_ptr_spike 2>&1 | grep "test result" | tail -3
cargo test --test training_verification v1_loss_matches 2>&1 | grep "test result" | tail -3
cargo test --test training_verification v3_train 2>&1 | grep "test result" | tail -3
cargo run --release --example compare_ddr_sandbox 2>&1 | tail -10
```

Expected:
- All lib tests pass.
- sparse_gradcheck: 1 PASS.
- sparse_cusparse_v5: V5 + forward + backward smokes pass.
- cusparse_ptr_spike: round-trip + active-stream + cube-tensor round-trip pass.
- V1 (NdArray, 8 gauges, frozen-params): 1e-5 match.
- V3 (NdArray, training loop): pass.
- `compare_ddr_sandbox`: ABSOLUTE MATCH.

- [ ] **Step 2: Clippy final pass**

```
cargo clippy --all-targets 2>&1 | grep -vE "(routing/|geometry|nn/mlp|data/)" | head -30
```

Expected: no warnings in SP-6/SP-7 files (`sparse/cusparse.rs`, `sparse/dispatch.rs`, `tests/sparse_*`).

- [ ] **Step 3: Optional CUDA train smoke (one more confirmation)**

```
cp config/merit_training.yaml /tmp/sp7_close_cuda.yaml
sed -i 's|^  # sparse_solver: cuda.*|  sparse_solver: cuda|' /tmp/sp7_close_cuda.yaml
cargo run --release --bin train -- \
    --config /tmp/sp7_close_cuda.yaml \
    --checkpoint-dir /tmp/sp7_close_cuda \
    --max-mini-batches 3 2>&1 | tail -10
```

Expected: finite losses, faster wall-time than the SP-6 baseline (6.10 min on this hardware).

- [ ] **Step 4: Close-out commit**

If clippy needed fixes, those land here. Otherwise, this is a doc-only commit summarising SP-7 numbers:

```
git commit --allow-empty -m "SP-7 close: V5 bit-match + V6 speedup + regression all green

Tasks 1-10 land the cuSPARSE stream-share + zero-copy x perf fix.
SP-6's host-blocking sync + n-element host roundtrip are replaced
with cubecl-stream-shared GPU ordering + cubecl-allocated output
buffers.

Verification:
  V5 bit-match (CPU vs CUDA):                     PASS
  V6 speedup (CUDA ≤ 0.7× CPU):                   PASS
  V1 / V3 / sparse_gradcheck / compare_ddr_sandbox: PASS

Vendor patches at:
  https://github.com/<gh-account>/cubecl @ ddrs-sp7-stream-accessor
  https://github.com/<gh-account>/burn   @ ddrs-sp7-primitive-ctor
SP-8 follow-up: upstream both as PRs against tracel-ai/{cubecl,burn}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(Replace `<gh-account>` with the actual fork owner.)

---

## Self-Review

### Spec coverage

| Spec section | Task |
|---|---|
| Fork prep + `[patch.crates-io]` wiring | 1 |
| cubecl-cuda `pub stream()` patch | 2 |
| burn-cubecl `pub from_handle()` patch | 3 |
| `cubecl_stream_active` / `cube_tensor_to_primitive` / `compute_client` helpers | 4 |
| `cusparse_forward` rewrite (shared stream + zero-copy x) | 5 |
| `cusparse_backward_solve` rewrite | 6 |
| `build_cuda_pattern_cache` cubecl allocations | 7 |
| Delete FALLBACK_STREAM + raw cuMem* + B::sync | 8 |
| V6 speedup assertion | 9 |
| V5 still passes (preserved correctness) | 5, 6, 7, 8, 10 |
| Regression: V1, V3, sparse_gradcheck, sandbox | every task that touches GPU code + Task 10 |
| `vendor/README.md` documentation + SP-8 upstream PR plan | 1 |

### Placeholder scan

The plan has explicit "implementer resolves at keyboard" gates in:
- Task 2 (`MultiStream::current()` field path — verify against actual cubecl source).
- Task 3 (`TensorMetadata::from(shape)` — verify against actual burn-cubecl source).
- Task 4 (`ComputeClient::exclusive` vs `.server()` access — match cubecl-runtime API).
- Task 7 (`Handle::binding().resource_handle()` exact accessor name — match cubecl-runtime API).

Each gate cites the SP-6 spike findings (Task 6 + Task 7 implementers' work in `src/sparse/cusparse.rs`) so the SP-7 implementer doesn't start from scratch. No vague "add error handling" or "implement later" — every code block shows the intended structure with concrete cudarc + cubecl call sequences.

### Type / identifier consistency

- `cubecl_stream_active`, `cube_tensor_to_primitive`, `compute_client`, `CudaPatternCache`, `cubecl::server::Handle`, `CubeTensor::from_handle`, `CudaServer::stream` — used identically across all tasks.
- `cusparse_forward(pattern, a_values_prim, b_prim, device)` — signature unchanged from SP-6.
- `cusparse_backward_solve` — signature unchanged from SP-6.
- `B: Backend + 'static` bound — added in Task 4, propagated in Tasks 5–7.
- `CudaPatternCache` field renames (`*_u64: u64 → *_handle: cubecl::server::Handle`) — consistent across Tasks 7 + 8.
- V5 / V6 / V1 / V3 references all match the names from SP-5 / SP-6 plans.

---

## Execution choice

Plan complete and saved to `.claude/specs/2026-05-21-sp7-cusparse-stream-share-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review. Same workflow as SP-4 / SP-5 / SP-6.
2. **Inline Execution** — batch with checkpoints.

Which approach?
