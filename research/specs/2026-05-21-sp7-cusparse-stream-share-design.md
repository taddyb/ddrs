# SP-7: cuSPARSE stream-share + zero-copy x wrap

**Status:** Draft, pending user review
**Date:** 2026-05-21
**Parent:** `.claude/specs/2026-05-20-sp6-cusparse-gpu-solve-design.md`
**Reference:**
- SP-6 close-out: CPU = 5.58 min, CUDA = 6.10 min on 3-mini-batch smoke. Losses bit-exact between paths; CUDA is 10% SLOWER.
- The bottleneck is two-fold: host syncs (`B::sync(device)` + `cuStreamSynchronize`) and the n-element host roundtrip of `x` at the end of every `cusparseSpSV_solve`.

## Goal

Make `sparse_solver: cuda` materially faster than the CPU path. Replace SP-6's host-blocking sync model with GPU-side stream ordering, and eliminate the host roundtrip of `x` in `cusparse_forward` / `cusparse_backward_solve`. V6 asserts CUDA training wall-time ≤ 0.7× CPU training wall-time on a 3-mini-batch smoke. CPU path stays default and bit-exact; V5 bit-match still passes.

## Two SP-6 wounds, one root cause

Both wounds trace to the same problem: `cubecl-cuda 0.10` and `burn-cubecl 0.21` keep the relevant APIs `pub(crate)`.

**Wound 1 — host syncs (Path B fallback from Task 7).** SP-6's `cusparse_forward` calls `B::sync(device)` to wait for cubecl's stream to finish writing `a_values` and `b`, then runs cuSPARSE on `FALLBACK_STREAM`, then `cuStreamSynchronize(FALLBACK_STREAM)` before reading the output. Two host blocks per solve.

**Wound 2 — host roundtrip of `x` (Task 9 "no clean CubeTensor constructor" fallback).** After cuSPARSE writes `x` to a raw `CUdeviceptr` allocated via `cuMemAllocAsync`, the function does `cuMemcpyDtoH` to copy `x` to a host `Vec<f32>`, then `B::float_from_data` re-uploads it as a new BURN tensor. For CONUS scale (~65k active reaches × 89 hourly steps × 5 epochs × hundreds of batches), this dominates.

A small vendor patch of two cubecl/burn crates exposes the missing APIs and lets both wounds heal in the same SP-7.

## Verification — V5 (preserved) + V6 (load-bearing speedup)

### V5 — must still pass

The synthetic 100-reach lower-triangular pattern bit-match from SP-6 Task 11 remains the correctness gate. Under SP-7's new sync model, `x`, `grad_a`, `grad_b` must still match CPU and CUDA within `1e-3` abs or `1e-4` rel.

If V5 fails, SP-7 has broken correctness. STOP.

### V6 — `#[ignore]`'d hard speedup assertion (NEW)

`tests/sparse_cusparse_v6.rs`:

```rust
#[test]
#[ignore] // run manually: cargo test --release -- --ignored v6_cuda_is_faster
fn v6_cuda_is_faster_than_cpu_on_smoke_train() {
    // 1. Skip if no CUDA device or data sources absent.
    // 2. Build two override configs: /tmp/v6_cuda.yaml (sparse_solver: cuda)
    //    and /tmp/v6_cpu.yaml (sparse_solver: cpu — explicit, in case
    //    default ever changes).
    // 3. Run `cargo run --release --bin train -- --config X --max-mini-batches 3`
    //    twice as subprocesses. Capture stdout.
    // 4. Parse the "Training complete in X.XX min" line out of each.
    // 5. Assert cuda_minutes < cpu_minutes * 0.7.
    //    Print both timings on assertion failure.
}
```

`#[ignore]` because the test depends on live data files and spawns `cargo run` (slow). The workstation runs it manually post-implementation; CI doesn't gate on it. Failing the assertion means SP-7 didn't deliver and we should figure out why before claiming the goal.

Threshold rationale: SP-6 had CUDA at 6.10 min vs CPU 5.58 min (CUDA 1.09× slower). A 0.7× threshold means CUDA must reach ≤ 3.90 min — roughly a 1.6× speedup over current CUDA, 1.4× over current CPU. Aggressive enough to be a real win; modest enough to absorb wall-clock noise on a non-isolated workstation.

## File changes

```
vendor/
  cubecl-cuda/                 NEW — fork of 0.10.0 with stream() accessor
  burn-cubecl/                 NEW — fork of 0.21.0 with primitive-from-handle constructor
  README.md                    NEW — fork rationale + upstream commit hashes

Cargo.toml                     [patch.crates-io] block pointing at vendor/

src/sparse/cusparse.rs         REWRITE forward + backward solve bodies:
                               - drop FALLBACK_STREAM + B::sync + cuStreamSynchronize
                               - use cubecl_server.stream() directly
                               - allocate x/y via cubecl client.create(size_bytes)
                               - wrap the cubecl Handle as CubeTensor → B::FloatTensorPrimitive

tests/sparse_cusparse_v5.rs    UNCHANGED assertions — runs under new internals
tests/sparse_cusparse_v6.rs    NEW — speedup assertion (#[ignore]'d)
```

`src/sparse/dispatch.rs` and `src/sparse/mod.rs` have NO surface changes. Existing callers see the same `triangular_csr_solve(pattern, a, b, use_cuda)` signature; only the GPU implementation under it gets faster.

## Strategy detail

### 1. Vendor forks (git deps to personal repos)

Two forks, one tiny `pub` surface change each. Use git deps in `Cargo.toml` so the diffs are visible in commit history of the fork repos and can be upstreamed cleanly later.

**Cargo.toml:**

```toml
[patch.crates-io]
cubecl-cuda = { git = "https://github.com/<user>/cubecl",   branch = "ddrs-sp7-stream-accessor" }
burn-cubecl = { git = "https://github.com/<user>/burn",     branch = "ddrs-sp7-primitive-ctor" }
```

(Both upstream projects are monorepos with the relevant crate at `crates/cubecl-cuda` and `crates/burn-cubecl` — the git dep specifies the path implicitly via `package = "cubecl-cuda"` or just the crate-name lookup in the monorepo.)

`vendor/README.md` documents:
- Upstream commit hash the fork branched from.
- The exact diff (5–50 lines per fork).
- A checklist for upstreaming as SP-8 follow-up cleanup.
- Recovery instructions if cubecl/burn release a new version: rebase the branch on the new tag.

**Fallback if git forks are inconvenient:** in-tree `vendor/cubecl-cuda/` + `vendor/burn-cubecl/` as full source copies of the published crate plus the patch. `Cargo.toml [patch.crates-io]` points at `{ path = "vendor/cubecl-cuda" }`. More reproducible (no network on build); higher repo size (~1MB per fork). Decision deferred to the SP-7 plan Task 1 implementer.

### 2. Patch surface — cubecl-cuda

Add a public method on `CudaServer`:

```rust
// In cubecl-cuda/src/compute/server.rs
impl CudaServer {
    /// Return the active CUDA stream for this server.
    ///
    /// Used by external CUDA libraries (cuSPARSE, cuBLAS, etc.) that need
    /// to interleave their calls with cubecl's compute via
    /// `cusparseSetStream` / `cublasSetStream`.
    pub fn stream(&self) -> cudarc::driver::sys::CUstream {
        // Pick the current active stream from MultiStream<CudaStreamBackend>.
        self.streams.current().sys
    }
}
```

(Field paths verified by Task 7 implementer in SP-6: `CudaServer.streams: MultiStream<CudaStreamBackend>`; `Stream.sys: CUstream` is already `pub` — only the wrapping module's visibility blocks us today.)

If `MultiStream` has multiple streams and selects one per kernel, expose the *primary*/default stream. The SP-7 plan Task 1 implementer verifies the multi-stream semantics; for SP-7's single-threaded training use case, one shared stream is correct.

Caller side in ddrs (`src/sparse/cusparse.rs`):

```rust
fn cubecl_stream<B: Backend>(device: &B::Device) -> CUstream {
    let client = ComputeClient::<CudaRuntime>::load(/* cuda_device from B::Device */);
    client.server().stream()  // OR client.exclusive(|s| s.stream()) — pick what compiles
}
```

The exact accessor (`server()` vs `with_server(...)` vs `exclusive(...)`) depends on how cubecl-runtime exposes the underlying server. Task 7's Path B fallback used `ComputeClient::load` + `exclusive(|| {})`. Same pattern, new accessor.

### 3. Patch surface — burn-cubecl

Add a `pub` constructor that builds a `CubeTensor<R>` from an externally-allocated cubecl `Handle`:

```rust
// In burn-cubecl/src/tensor/base.rs (or wherever CubeTensor is defined).
impl<R: CubeRuntime> CubeTensor<R> {
    /// Construct a tensor from an existing cubecl Handle plus shape and dtype.
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
}
```

(Fields verified to exist by Task 6 implementer: `CubeTensor { client, handle, meta, device, dtype }`. The struct itself may or may not have `pub(crate)` constructors today; if a private `new` exists, this `from_handle` wraps it.)

If the `Handle` type also needs `pub` exposure for the caller to construct one, add a one-line re-export.

### 4. New `cusparse_forward` body

Pseudo-Rust showing the structural change. Existing pre-amble (cache lazy-init, primitive_as_cuda_view) stays.

```rust
pub(crate) fn cusparse_forward<B: Backend>(
    pattern: &CsrPattern,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    let cache = unsafe { ensure_cuda_cache(pattern) };

    // 1. Get cubecl's stream — drop B::sync; cubecl's queued kernels write
    //    a_values + b on this same stream, so cuSPARSE on the same stream
    //    is naturally ordered after them. No host block.
    let stream = cubecl_stream::<B>(device);
    unsafe { cudarc::cusparse::sys::cusparseSetStream(cache.handle, stream); }

    // 2. Extract device pointers — same as SP-6.
    let a_view = primitive_as_cuda_view::<B>(a_values_prim).expect("Cuda required");
    let b_view = primitive_as_cuda_view::<B>(b_prim).expect("Cuda required");
    let n = pattern.n;

    // 3. Allocate x via cubecl's client (NOT raw cuMemAllocAsync). The
    //    returned Handle owns the buffer; freeing happens when the
    //    CubeTensor we build below is dropped — by BURN's tape.
    let client = compute_client::<B>(device);
    let x_handle = client.create(n * std::mem::size_of::<f32>());
    let x_ptr = x_handle.binding().device_ptr();  // raw CUdeviceptr

    // 4. cuSPARSE solve. cusparseCsrSetPointers + updateMatrix from SP-6.
    //    NO post-solve cuStreamSynchronize — cubecl's next op on the same
    //    stream waits naturally.
    unsafe {
        cusparseCsrSetPointers(cache.sp_mat, cache.d_crow, cache.d_col,
                               a_view.ptr as *mut _);
        cusparseSpSV_updateMatrix(cache.handle, cache.desc_forward,
                                  a_view.ptr as *mut _,
                                  CUSPARSE_SPSV_UPDATE_GENERAL);
        // ... createDnVec for b and x ... SpSV_solve ... destroyDnVec ...
    }

    // 5. Wrap the handle as a CubeTensor → B::FloatTensorPrimitive.
    //    No memcpy_dtoh, no float_from_data. The buffer x_handle points at
    //    is on the GPU; the next BURN op consumes it on the same stream.
    let cube_tensor = CubeTensor::<CudaRuntime>::from_handle(
        client.clone(),
        x_handle,
        Shape::new([n]),
        DType::F32,
        device.clone(),
    );
    cube_tensor_to_primitive::<B>(cube_tensor)
}
```

`cube_tensor_to_primitive::<B>` is the type-id-gated transmute back to `B::FloatTensorPrimitive` — the inverse of Task 6's `primitive_as_cuda_view`. Same SAFETY contract.

`cusparse_backward_solve` follows the identical structure with `desc_backward` + `CUSPARSE_OPERATION_TRANSPOSE`.

### 5. `cusparse_grada` already on device

SP-6's `cusparse_grada` uses pure BURN tensor ops (`Tensor::select` + multiply + negate). No change needed in SP-7 — it already runs entirely on whatever backend the tensors live on, and the new zero-copy `x` from `cusparse_forward` flows through it cleanly without host roundtrip.

## Decisions made

These resolve the three sanity-check questions from the brainstorming session:

1. **Fork workflow:** Git deps to personal forks (one branch per fork), with `vendor/README.md` recording upstream commit hashes and the diff. In-tree `vendor/<crate>/` is the fallback if the implementer hits friction publishing forks.
2. **V6 threshold:** `0.7×` (CUDA wall-time ≤ 70% of CPU wall-time). Aggressive but absorbs ~10% workstation noise.
3. **Possible third fork (burn-cuda):** If `burn-cuda` 0.21 pins `cubecl-cuda` transitively in a way that blocks Cargo `[patch]` resolution, SP-7 Task 1 escalates and the spec gets revised to add a third fork or pivot to in-tree vendor.

## Concerns

1. **Vendor patches drift across cubecl/burn version bumps.** Each upgrade requires re-applying ≤ 50-line diffs. Mitigation: small diffs, documented commit hash, upstream PR planned as SP-8 cleanup.
2. **`[patch.crates-io]` compatibility.** The fork must declare the same `name` and `version` as the original crate for Cargo's patch table to resolve. Task 1 verifies the patched dep graph builds.
3. **Transitive cubecl-cuda version pinning.** `burn-cuda 0.21` may pin `cubecl-cuda = "=0.10.0"` — fine for our fork. If it pins a different version, Task 1 reports and we revise. Same risk applies to `burn-cubecl`.
4. **CubeTensor's `from_handle` may need more than visibility flip.** burn-cubecl's CubeTensor today might have no constructor that takes external buffers — only `client.create(...) → Handle → CubeTensor` via private builders. If so, Task 1 patches a real new method, not just a `pub` toggle.
5. **`MultiStream` semantics.** cubecl-cuda 0.10 uses `MultiStream<CudaStreamBackend>` — may select different streams for different kernel categories. Our `stream()` accessor exposes a single stream. If cubecl migrates the kernel for `a_values` computation to a different stream than the one we return, cuSPARSE on our exposed stream won't be ordered after it. Mitigation: read `MultiStream::current()` and document the assumption; if it falls apart, fall back to `B::sync(device)` for correctness and report the perf regression.
6. **Stream-sharing eliminates explicit syncs but cuSPARSE may still need workspace fences.** Per-stream serialization should be sufficient — SpSV writes `x` to a buffer cubecl-allocated, subsequent BURN ops on the same stream read the buffer. V5 catches any subtle ordering bug.
7. **V6 wall-clock noise.** A 3-mini-batch smoke is short (~5 min). System load can swing wall-time by ±10%. The `0.7×` threshold absorbs most realistic noise; if it flakes, the fix is to extend the smoke to 10 batches or pin CPU affinity.
8. **`B::FloatTensorPrimitive` reverse cast.** Task 6 went `B::FloatTensorPrimitive → CudaView` via TypeId-gated `transmute_copy`. SP-7 needs the inverse: `CubeTensor<CudaRuntime> → B::FloatTensorPrimitive`. Same trick, opposite direction; same SAFETY conditions.

## Open assumptions

1. cubecl-cuda 0.10's `CudaServer` holds streams in a way where exposing one `pub stream()` accessor is sufficient for the SP-7 use case. Verified by Task 7 source-reading; multi-stream edge cases may surface and are explicit in the SP-7 plan's Task 1 escalation criteria.
2. burn-cubecl 0.21's `CubeTensor` has all the fields needed (`client`, `handle`, `meta`, `device`, `dtype`) for the `from_handle` constructor. Task 6 confirmed the struct layout.
3. Host syncs + host roundtrip together explain the SP-6 perf regression. If V6 fails even after both are fixed, profiling moves to BURN autograd tape bookkeeping, icechunk reads, or the per-batch subgraph rebuild (cubecl autotune re-compile on new shapes). Each of those is a separate SP-8+ effort.
4. Single-GPU, single-threaded training.

## Out of scope

- Upstream PRs to cubecl / burn (SP-8 follow-up).
- Replacing BURN's internal scatter kernel hotspot (separate problem; SP-6 nsys flagged it).
- f64 cuSPARSE path.
- Multi-GPU.

## Next steps

1. You review this spec and request changes or approve.
2. After approval: invoke `superpowers:writing-plans` to produce the SP-7 implementation plan (`.claude/specs/2026-05-21-sp7-cusparse-stream-share-plan.md`).
3. After plan approval: subagent-driven execution per the SP-4/5/6 pattern.
