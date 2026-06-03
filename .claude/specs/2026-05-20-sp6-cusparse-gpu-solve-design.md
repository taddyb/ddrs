# SP-6: GPU cuSPARSE triangular solve

**Status:** Draft, pending user review
**Date:** 2026-05-20
**Parent:** `.claude/specs/2026-05-17-train_and_test-replication-design.md`
**Reference:**
- The smoking gun: `src/sparse.rs:335` (`forward_primitive`) — CPU forward solve forces D→H + H→D every timestep.
- The other CPU bottleneck: `src/sparse.rs:380` (`CsrSolveOp::backward`) — pulls `a_values + grad_out` to host, runs `back_sub_upper_transposed`, builds `grada` per-nnz on CPU.
- CuPy's `spsolve_triangular` is the reference algorithm; it dispatches to cuSPARSE `cusparseSpSV_*`.

## Goal

Make the CSR triangular solve GPU-native end-to-end on the BURN CUDA backend.
Eliminate the per-timestep D↔H syncs in the forward solve, the backward
upper-triangular solve, and the per-nnz `grada` scatter. Add a config flag
`params.sparse_solver: "cpu" | "cuda"` that gates the GPU path; CPU stays the
default and remains a fallback.

The win is measured by removing the `cuEventSynchronize` storm (12.2 sec /
134K calls in the nsys profile of the training loop) and the implicit syncs
behind every `primitive_to_vec` call inside the per-timestep loop.

## Architectural invariants preserved

- `f32` throughout the routing core (cuSPARSE f32 path; no mixed precision).
- Adjacency stays topologically ordered, lower-triangular. `pattern.col[k] ≤ pattern.row_for_nnz[k]` for every k.
- The hand-written `CsrSolveOp impl Backward` stays. Only its inner CPU
  primitives are swapped for cuSPARSE.
- The `compare_ddr_sandbox` regression must continue to report ABSOLUTE MATCH.
  It pins `NdArray<f32>`, so the CPU path is exercised end-to-end.
- The pattern-once-fill-many model (`CsrPattern` + `AValuesAssembler`)
  carries over directly — cuSPARSE has the same shape (`cusparseSpSV_analysis`
  once per pattern, `cusparseSpSV_solve` per timestep).

## Verification — V5 (load-bearing)

Single integration test: `tests/sparse_cusparse_v5.rs`. Skips cleanly if no
CUDA device is available.

1. Build a synthetic lower-triangular `CsrPattern` for `n = 100` reaches with
   nonzero off-diagonal entries (e.g., a banded matrix with bandwidth 5, plus
   the diagonal). All elements have nontrivial gradients.
2. Build random but seeded `a_values` (length `nnz`) and `b` (length `n`).
3. Run CPU path (`sparse_solver = Cpu`, `NdArray<f32>` backend):
   - Forward → `x_cpu`.
   - Backward (autograd `loss = x_cpu.sum()`) → `grad_a_cpu`, `grad_b_cpu`.
4. Run CUDA path (`sparse_solver = Cuda`, `Cuda<f32, i32>` backend):
   - Forward → `x_gpu`.
   - Backward → `grad_a_gpu`, `grad_b_gpu`.
5. Assert per-element f32 match for `x`, `grad_a`, `grad_b` within
   `1e-5` relative.

If V5 passes, the GPU implementation is bit-equivalent to the CPU one at the
f32 floor.

## Config surface

Add to `src/config.rs::Params`:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SparseSolver {
    #[default]
    Cpu,
    Cuda,
}

pub struct Params {
    // ... existing fields ...
    pub sparse_solver: SparseSolver,
}
```

YAML field `params.sparse_solver: "cpu" | "cuda"` (default `cpu`). The serde
intermediate `ParamsRaw` reads `sparse_solver: Option<String>` and the `From`
impl maps `"cpu" → Cpu`, `"cuda" → Cuda`, missing → `Cpu`.

The flag is plumbed through:
- `MuskingumCunge::new` reads `cfg.params.sparse_solver` and stores it on `self`.
- `MuskingumCunge::route_timestep` (and the hotstart call in `setup_inputs`)
  passes it to `triangular_csr_solve` calls.
- `triangular_csr_solve` adds a `use_cuda: bool` parameter to its public signature.
- Saved into `CsrSolveState` so the autograd backward uses the same path as
  the forward.

Fallback behavior: if `sparse_solver == Cuda` but the backend is not CUDA
(e.g., `NdArray<f32>` in a test), silently fall back to the CPU path. Log
once at WARN level the first time the fallback triggers in a process. This
prevents accidental test breakage and keeps the public API uniform.

## File layout

```
src/sparse/                         NEW directory
  mod.rs                            Existing src/sparse.rs content moves here
                                    (re-exports preserved for callers via crate::sparse::*)
  cusparse.rs                       NEW — cuSPARSE forward + backward + grada GPU path
  dispatch.rs                       NEW — runtime branch CPU vs cuSPARSE
src/config.rs                       MODIFIED — SparseSolver enum + Params.sparse_solver
src/routing/mmc.rs                  MODIFIED — pipe sparse_solver into triangular_csr_solve calls
tests/sparse_cusparse_v5.rs         NEW — V5 bit-match test (skips when no CUDA)
config/merit_training.yaml          MODIFIED — add commented-out `sparse_solver: cuda` example
Cargo.toml                          MODIFIED — explicit `cudarc` dep (verify cuSPARSE feature)
```

`src/sparse.rs` is at 536 lines and growing; splitting into `src/sparse/{mod,cusparse,dispatch}.rs`
is a focused split tied to this task, not a drive-by refactor. The public
API on `crate::sparse::*` is preserved bit-for-bit.

## Key components

### 1. cuSPARSE bindings

`cudarc 0.19.7` is a transitive dep already in the registry. Task 1 of the
implementation plan must verify whether it exposes `cusparseSpSV_*` directly.

- If yes: add explicit dep `cudarc = { version = "0.19", features = ["cusparse"] }`.
- If no: hand-roll FFI for the three functions we need:
  - `cusparseSpSV_bufferSize`
  - `cusparseSpSV_analysis`
  - `cusparseSpSV_solve`
  via `bindgen` against `cusparse.h` from the CUDA toolkit. Add a tiny
  `build.rs`.

Either way the binding surface is small (~3 functions + 2 enum types + 3
descriptor types).

### 2. Pattern-life descriptor cache

cuSPARSE requires a one-time `cusparseSpSV_analysis` call per CSR pattern.
`CsrPattern` is `Arc`'d and rebuilt per batch. Two descriptor handles are
needed:
- forward: lower-triangular, no-transpose (`CUSPARSE_OPERATION_NON_TRANSPOSE`,
  `CUSPARSE_FILL_MODE_LOWER`).
- backward: upper-triangular via transpose op (`CUSPARSE_OPERATION_TRANSPOSE`,
  same matrix, fill mode interpreted post-transpose).

Both are lazy-built on first GPU solve call and stored alongside the pattern:

```rust
pub struct CsrPattern {
    // ... existing CPU-side index arrays ...
    cuda_cache: OnceCell<CudaPatternCache>,
}

struct CudaPatternCache {
    d_crow: CudaSlice<i32>,                 // device copy of pattern.crow
    d_col:  CudaSlice<i32>,                 // device copy of pattern.col
    sp_mat_desc: cusparseSpMatDescr_t,      // built once, refers to d_crow/d_col
    desc_forward: cusparseSpSVDescr_t,
    desc_backward: cusparseSpSVDescr_t,
    workspace_forward: CudaSlice<u8>,
    workspace_backward: CudaSlice<u8>,
    stream: cudaStream_t,                   // shared with cubecl — see Concerns
}
```

`OnceCell` is sufficient because `CsrPattern` is built single-threaded per
batch. `CudaPatternCache` implements `Drop` that releases descriptors before
the device slices go out of scope (descriptor-before-data is the correct
cuSPARSE teardown order).

### 3. Saved state for backward

`CsrSolveState::x` is currently `Arc<Vec<f32>>` (CPU). For GPU residency,
the saved `x` and `a_values` must stay on whichever device produced them:

```rust
enum SavedX<B: Backend> {
    Cpu(Arc<Vec<f32>>),                   // existing CPU path
    Cuda(B::FloatTensorPrimitive),        // GPU path; never touches host
}

struct CsrSolveState<B: Backend> {
    a_values: B::FloatTensorPrimitive,    // already a GPU primitive on GPU path
    x: SavedX<B>,
    pattern: Arc<CsrPattern>,
    use_cuda: bool,                       // dispatched in backward
}
```

The backward path reads `use_cuda` and dispatches to CPU or cuSPARSE accordingly.

### 4. The `grada` kernel

Math: `grada[k] = -gradb[row_for_nnz[k]] * x[col[k]]` for `k in 0..nnz`.

On GPU: a small custom kernel with one thread per `k`. Inputs:
- `d_row_for_nnz: &[i32]` (length `nnz`)
- `d_col: &[i32]`         (length `nnz`)
- `d_gradb: &[f32]`       (length `n`)
- `d_x: &[f32]`           (length `n`)

Output: `d_grada: &mut [f32]` (length `nnz`).

Implementation choice: write via `cubecl::cube` macro (same JIT path BURN
uses) so it integrates with the existing CUDA context and stream. If cubecl
specialization is awkward inside our sparse module, fall back to raw cudarc
+ inline PTX or a small `.cu` kernel compiled at build.rs time. Pick the
cleanest at task 6 of the plan.

`d_row_for_nnz` and `d_col` already live on device as part of
`CudaPatternCache` (added as new fields if not already covered).

### 5. Dispatch shim

`src/sparse/dispatch.rs` exposes:

```rust
pub(crate) fn forward_primitive<B: Backend>(
    pattern: &Arc<CsrPattern>,
    a_values_prim: &B::FloatTensorPrimitive,
    b_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
    use_cuda: bool,
) -> ForwardResult<B>;

pub(crate) struct ForwardResult<B: Backend> {
    pub out_prim: B::FloatTensorPrimitive,
    pub saved_x: SavedX<B>,
}
```

Internal branching:
```rust
if use_cuda && backend_is_cuda::<B>() {
    cusparse::forward(pattern, a_values_prim, b_prim, device)
} else {
    cpu::forward(pattern, a_values_prim, b_prim, device)
}
```

`backend_is_cuda::<B>()` uses `std::any::TypeId` to compare `B` against
`burn::backend::cuda::Cuda<f32, i32>`. If the comparison is false but
`use_cuda` was requested, the first hit logs WARN once via `std::sync::Once`.

Backward dispatch follows the same pattern in
`src/sparse/dispatch.rs::backward_primitive`.

## Data flow on the GPU path

```
forward (per timestep):
  a_values_prim, b_prim                  (BURN CUDA tensors, on device)
    ↓ extract device ptrs from B::FloatTensorPrimitive
  cusparseSpSV_solve(desc_forward, a_values, b) -> x          (on device)
    ↓
  return x_prim as B::FloatTensorPrimitive                    (no host round-trip)
  save SavedX::Cuda(x_prim.clone()) for backward

backward (per backward pass):
  saved a_values_prim (GPU), saved x_prim (GPU), grad_out_prim (GPU)
    ↓
  cusparseSpSV_solve(desc_backward, a_values, grad_out) -> gradb   (transpose op)
    ↓
  custom kernel: grada[k] = -gradb[row_for_nnz[k]] * x[col[k]]      (on device)
    ↓
  register gradb_prim + grada_prim as gradients                     (no host round-trip)
```

For the CPU path, the existing `forward_sub_lower` + `back_sub_upper_transposed`
flow is unchanged. Only the `CsrSolveState::x` typing changes (now an enum;
the `Cpu` variant is the existing `Arc<Vec<f32>>`).

## Concerns

1. **`cudarc` cuSPARSE coverage.** Task 1 of the plan verifies whether
   `cudarc 0.19` exposes `cusparseSpSV_*`. If missing, the fallback is
   bindgen + a small build.rs against `cusparse.h`. Adds a half-day of
   integration work but no architectural change. Document the chosen path
   in the SP-6 plan's Task 1.
2. **Extracting raw device pointers from BURN tensors.** `burn-cuda` 0.21
   wraps tensors via `cubecl`'s `ComputeServer`. The path to a `*mut f32`
   (or `CudaSlice<f32>`) from a `B::FloatTensorPrimitive` is not part of
   BURN's stable API and may require `unsafe` access through cubecl's
   server handle. Task 2 of the plan is a spike to nail this down. If the
   abstraction is unworkable, we fall back to materializing the values
   via `primitive_to_vec` (defeats the purpose) — STOP and escalate.
3. **Stream / context sharing with cubecl.** cuSPARSE needs a
   `cudaStream_t`. cubecl owns a stream per device. If we use a separate
   stream, we must `cudaStreamSynchronize` to interop — silently kills the
   perf win. Best path: get cubecl's stream and pass it to
   `cusparseSetStream`. May require digging into `burn-cuda`/`cubecl-cuda`
   internals. Task 3 of the plan covers this.
4. **Descriptor lifetime + Send/Sync.** cuSPARSE descriptors are not
   thread-safe. `Arc<CsrPattern>` may be cloned across threads in the
   future. Single-threaded training is fine today; the `OnceCell` is
   `Send + Sync` but the descriptor inside must not actually cross
   threads. Document `CudaPatternCache: !Send` (via `PhantomData<*mut ()>`)
   and accept the constraint.
5. **f32 invariant preserved.** cuSPARSE supports f32 directly; no
   precision drift vs CPU path expected beyond bit-level f32 atomic
   ordering in the per-nnz `grada` kernel. V5's `1e-5` rel should hold.
   If it doesn't, tighten to `1e-4` and document. Note: the `grada`
   kernel has NO atomics (write-only by `k`, no contention) so order is
   deterministic.
6. **`CsrPattern` struct extension blast radius.** `CsrPattern` is `pub`
   in `crate::sparse`. Adding `cuda_cache: OnceCell<...>` is non-breaking
   (no constructor changes) but changes struct size + alignment. All
   `CsrPattern::from_sparse` call sites should rebuild cleanly. The
   `cuda_cache` field is `#[derive(Debug)]`-skipped via a custom impl —
   `OnceCell<T>` is `Debug` only when `T: Debug`, and the GPU buffers
   inside are not.
7. **No new clippy warnings.** Unsafe blocks need `SAFETY:` comments.
   cudarc usage needs care around drop order (descriptors before device →
   use-after-free if reversed).
8. **CUDA-only at compile time.** The project already pulls `burn-cuda`
   unconditionally. Adding direct `cudarc` dep compiles fine without a
   CUDA device — the `OnceCell` stays `None` and we never call into it.
   The V5 test gates on `cuda_available()` (cudarc query) and skips on CPU-only hosts.
9. **CPU-path performance.** The CPU path stays untouched semantically;
   the only change is the `SavedX::Cpu` wrapper around the existing
   `Arc<Vec<f32>>`. V1, V2, V4, sandbox regression — all must still pass.

## Decisions made

These resolve the three sanity-check questions from the brainstorming
session:

1. **Config naming and default.** `params.sparse_solver: "cpu" | "cuda"`,
   default `"cpu"`. CPU as default keeps existing behavior and all current
   tests green without YAML changes.
2. **Saved state representation.** Use the `SavedX<B>` enum with `Cpu` and
   `Cuda` variants. Both paths carry their natural representation; the
   backward dispatches on `use_cuda`. Avoids needless cross-path conversions.
3. **V5 scope.** Synthetic-pattern bit-match only. The end-to-end
   regression (running V1 on the Cuda backend) is deferred to Task 12
   of the SP-6 plan as a smoke pass (`cargo test --release ...` against
   both `sparse_solver` settings) — not part of V5's load-bearing assertions.

## Open assumptions

1. `cudarc 0.19` includes cuSPARSE bindings sufficient for
   `SpSV_bufferSize`, `SpSV_analysis`, `SpSV_solve` — VERIFY at Task 1.
   If missing, plan accommodates a bindgen spike (Task 1b).
2. The DDR-MATCH regression (`examples/compare_ddr_sandbox`) keeps passing —
   it pins `NdArray<f32>` and the CPU path is unchanged.
3. V1 (`NdArray`, 8 gauges, frozen-params) keeps passing with default
   `sparse_solver: cpu`. V5 adds the CUDA equivalence test.
4. Single-GPU only. Multi-GPU is out of scope.
5. `cubecl-cuda` exposes its stream handle to user code (we can call
   `cusparseSetStream` with it). If it doesn't, fallback is a separate
   stream + explicit `cudaStreamSynchronize` — perf-degraded but correct.
   Task 3 verifies and reports.

## Out of scope

- Replacing BURN's internal scatter kernel (the `scatter_kernel_t_f32_i_i32`
  hotspot from the nsys profile is in BURN's autograd machinery, not our
  per-nnz grada scatter — separate problem).
- Multi-GPU sparse solve.
- Migrating other ops (geometry, MLP, downsample) to custom GPU kernels.
- f64 path for cuSPARSE.

## Next steps

1. You review this spec and request changes or approve.
2. After approval: invoke `superpowers:writing-plans` to produce the
   SP-6 implementation plan (`.claude/specs/2026-05-20-sp6-cusparse-gpu-solve-plan.md`).
3. After plan approval: subagent-driven execution per the SP-4/SP-5 pattern.
