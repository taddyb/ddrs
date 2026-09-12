# SP-9: cuSPARSE SpMV (kill the scatter hotspot for real)

**Status:** Draft, pending user review
**Date:** 2026-05-22
**Parent:** SP-8 (MC timestep fusion — landed but V7a/V7b both missed)
**Predecessor close-out:** SP-8's fusion dropped wall time 27% (autograd-graph
collapse) but didn't move `scatter_kernel_t_f32_i_i32` (77.5% of GPU compute,
96,090 invocations). The scatter sources are NOT autograd-induced; they're the
three explicit `Tensor::scatter(0, ..., IndexingUpdateOp::Add)` calls in
`src/sparse/mod.rs:663, 687, 712` inside `spmv_primitive`,
`assemble_backward_primitive`, and `spmv_backward_primitive`. Those calls
lower through `K::scatter` → `B::float_scatter_add` → `kernel::scatter` →
`scatter_kernel_t_f32_i_i32` — exactly the bottleneck kernel.

## Goal

Replace the three `.scatter(..., Add)` call sites with `cusparseSpMV` calls on
the CUDA backend. Reuse the existing SP-6/SP-7 cuSPARSE infrastructure
(`CudaPatternCache`, stream-share via `cubecl_stream_active`, zero-copy via
`cube_tensor_to_primitive`). CPU (NdArray) path stays on the existing
`.scatter` implementation — unchanged.

## Verification: V7a + V7b + V8 (all three must pass)

### V7a — perf gate (re-run of `tests/sp8_v7_perf.rs`)

Median CUDA wall ÷ median CPU wall ≤ **0.7** on the smoke train
(`bin/train --max-mini-batches 3`). SP-8 ran this at ratio 1.000.
SP-9 expects cuSPARSE SpMV to eliminate most of the 29.7 sec spent on
`scatter_kernel`, which should push CUDA below 3 min and the ratio below 0.7.

### V7b — profile gate (re-run of `tests/sp8_v7_profile.rs`)

`scatter_kernel_t_f32_i_i32` percentage of GPU compute time falls below
**30%**. SP-8 ran this at 77.5%. After SP-9 we expect near 0% (no `.scatter`
calls left in the GPU path).

### V8 — new: SpMV CPU/CUDA bit-match (`tests/sparse_cusparse_v8.rs`)

For a small synthetic CSR pattern, verify all three new cuSPARSE SpMV functions
match their CPU `.scatter`-based counterparts within 1e-5 relative tolerance:

- `cusparse_spmv_forward` vs CPU `spmv_primitive` (NdArray with sparse_solver: cpu)
- `cusparse_spmv_backward` vs CPU `spmv_backward_primitive`
- `cusparse_assemble_backward` vs CPU `assemble_backward_primitive`

Distinct test file from V5 — V5 is "CSR solve bit-match", V8 is "SpMV bit-match".

### Regression gates (must remain green)

V1 (8 gauges, frozen-params), V2 (all gauges), V4 (full test period),
V5 (CSR solve CPU/CUDA bit-match), gradcheck (5 parents). All on CPU
backend by default, so SP-9's CUDA-only changes shouldn't touch them.

## Architecture

### Three operations, two new sparse-matrix descriptors

| Site | Math | cuSPARSE call | Matrix |
|---|---|---|---|
| `spmv_primitive` (forward `y = N · q`) | `out[i] = Σ_k adj[k]·q[col[k]]` | `cusparseSpMV(NON_TRANSPOSE, sp_mat_spmv, q, y, α=1, β=0)` | `(n × n)`, values = adj |
| `spmv_backward_primitive` (`gq = N^T · gi`) | `out[j] = Σ_k adj[k]·gi[row[k]]` for `col[k]=j` | `cusparseSpMV(TRANSPOSE, sp_mat_spmv, gi, gq, α=1, β=0)` | same `sp_mat_spmv` |
| `assemble_backward_primitive` (`gc = -sum_k(gA[k]·adj[k])` per row) | `gc[i] = -Σ_k adj[k]·gA[k]` for `k in row i` | `cusparseSpMV(NON_TRANSPOSE, sp_mat_rowsum, gA, gc, α=-1, β=0)` | `(n × nnz)`, values = adj |

The first two share one matrix descriptor (`sp_mat_spmv`, n × n, values = adj).
The third needs a separate `(n × nnz)` descriptor because `gA` has length
`nnz`, not `n`.

### `CudaPatternCache` extension

```rust
pub(crate) struct CudaPatternCache {
    // ── Existing (SP-6/7) ─────────────────────────────────────────
    pub(crate) handle: cusparseHandle_t,
    pub(crate) d_crow: Handle,
    pub(crate) d_col: Handle,
    pub(crate) d_row_for_nnz: Handle,
    pub(crate) sp_mat: cusparseSpMatDescr_t,       // (n × n) for SpSV
    pub(crate) desc_forward: cusparseSpSVDescr_t,
    pub(crate) desc_backward: cusparseSpSVDescr_t,
    pub(crate) workspace_forward: Handle,
    pub(crate) workspace_backward: Handle,

    // ── NEW (SP-9) ────────────────────────────────────────────────
    pub(crate) sp_mat_spmv: cusparseSpMatDescr_t,    // (n × n) for SpMV sites 1+2
    pub(crate) sp_mat_rowsum: cusparseSpMatDescr_t,  // (n × nnz) for SpMV site 3
    pub(crate) d_col_identity: Handle,               // [0..nnz) i32 indices
    pub(crate) d_crow_eye: Handle,                   // [0, 1, 2, ..., nnz] crow for rowsum
    pub(crate) workspace_spmv_n: Handle,             // SpMV NON_TRANSPOSE on sp_mat_spmv
    pub(crate) workspace_spmv_nt: Handle,            // SpMV TRANSPOSE on sp_mat_spmv
    pub(crate) workspace_rowsum: Handle,             // SpMV NON_TRANSPOSE on sp_mat_rowsum
}
```

Six new fields. Three Handle buffers (`d_col_identity`, `d_crow_eye`,
plus three workspaces). Two cuSPARSE descriptors.

**Why `d_crow_eye` is needed**: the `(n × nnz)` `sp_mat_rowsum` reuses the
*shape* of N's CSR structure for the row partition. Its `crow` is the same as
N's `crow` (length n+1, summing to nnz). Its `col` is `[0, 1, 2, ..., nnz-1]`
(identity, length nnz). Its `values` are `adj_values` (length nnz, same as
N's adj). The structure is: row i contains exactly the nnz-positions k such
that `row[k] = i`, with col indices `k` and values `adj[k]`.

Actually, on review: `sp_mat_rowsum`'s `crow` IS exactly N's `crow` — same
partition of nnz into rows. So we **don't need a separate `d_crow_eye`** —
we reuse the existing `cache.d_crow`. The only new array is `d_col_identity`.

Updated field list (one fewer Handle):

```rust
    // NEW (SP-9):
    pub(crate) sp_mat_spmv: cusparseSpMatDescr_t,
    pub(crate) sp_mat_rowsum: cusparseSpMatDescr_t,
    pub(crate) d_col_identity: Handle,
    pub(crate) workspace_spmv_n: Handle,
    pub(crate) workspace_spmv_nt: Handle,
    pub(crate) workspace_rowsum: Handle,
```

Five new fields total.

### Setup in `build_cuda_pattern_cache`

Append to the existing setup function in `src/sparse/cusparse.rs`:

1. Build `d_col_identity` as a `Vec<i32>` of `[0, 1, 2, ..., nnz-1]`, upload via
   `client.create_from_slice`.
2. Create `sp_mat_spmv` via `cusparseCreateCsr` with `(rows=n, cols=n, nnz=nnz,
   csrRowOffsets=d_crow, csrColInd=d_col, csrValues=d_adj_values, ...)`.
   Reuse `d_crow`/`d_col` and a new device buffer for `adj_values` (or reuse
   the existing one if SP-6 already uploaded it — verify).
3. Create `sp_mat_rowsum` via `cusparseCreateCsr` with `(rows=n, cols=nnz,
   nnz=nnz, csrRowOffsets=d_crow, csrColInd=d_col_identity,
   csrValues=d_adj_values, ...)`. Note `cols=nnz`.
4. For each of the three SpMV configurations, call `cusparseSpMV_bufferSize`
   with the matching op flag and a dummy alpha/beta, allocate the returned
   workspace size via `client.create(size)`, and store as
   `workspace_spmv_n` / `workspace_spmv_nt` / `workspace_rowsum`.
5. Optional but recommended: `cusparseSpMV_preprocess` for each configuration
   to reduce per-call overhead.

### Three new public-in-crate functions in `src/sparse/cusparse.rs`

All three follow the existing SP-7 pattern (`cusparse_forward` /
`cusparse_backward_solve` / `cusparse_grada`): extract device pointers via
`handle_device_ptr`, set the cubecl-active stream on the cuSPARSE handle via
`cusparseSetStream`, call `cusparseSpMV` with the right descriptor + op flag
+ alpha + beta, return the result as `B::FloatTensorPrimitive` via
`cube_tensor_to_primitive`.

```rust
/// y = N · q. Site 1 — forward SpMV.
pub(crate) fn cusparse_spmv_forward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    q_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive;

/// gq = N^T · gi. Site 2 — transpose SpMV. Reuses sp_mat_spmv.
pub(crate) fn cusparse_spmv_backward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    gi_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive;

/// gc = -sp_mat_rowsum · gA. Site 3 — per-row sum with embedded negation.
pub(crate) fn cusparse_assemble_backward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    g_a_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive;
```

The `α=-1.0` in `cusparseSpMV` for site 3 handles the negation directly —
no separate `.neg()` step needed.

### Dispatch entries in `src/sparse/dispatch.rs`

Three new dispatch entries that mirror the existing
`forward_primitive` / `backward_solve_primitive` / `grada_primitive`:

```rust
pub fn spmv_forward_primitive<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    q_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive;

pub fn spmv_backward_primitive_dispatch<I: Backend + 'static>(  // renamed to disambiguate
    pattern: &Arc<CsrPattern>,
    gi_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive;

pub fn assemble_backward_primitive_dispatch<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    g_a_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive;
```

`use_cuda=true` AND `TypeId::of::<I>() == TypeId::of::<Cuda<f32, i32>>()` →
cuSPARSE path. Otherwise → existing CPU `.scatter` path.

Name collision note: `spmv_backward_primitive` already exists as the top-level
sparse helper in `src/sparse/mod.rs:698`. The dispatch entries below mod live
in `src/sparse/dispatch.rs` — the existing name in `mod.rs` becomes a
thin wrapper calling `dispatch::*`. Renaming the dispatch entries to
`*_dispatch` keeps the mental model clear; or we drop the `_primitive`
suffix on the mod helpers since they're no longer pure primitives. Pick one:
the implementation plan locks the naming.

### Call-site rewrite in `src/sparse/mod.rs`

The three helpers become thin dispatches. Example for `spmv_primitive`:

```rust
pub fn spmv_primitive<I: Backend + 'static>(
    pattern: &Arc<CsrPattern>,
    q_prim: I::FloatTensorPrimitive,
    device: &I::Device,
) -> I::FloatTensorPrimitive {
    // SP-9: dispatch CPU `.scatter` path vs cuSPARSE SpMV per cfg.params.sparse_solver.
    // The `use_cuda` flag is plumbed from the caller via the engine's stored solver flag.
    // ...
    crate::sparse::dispatch::spmv_forward_primitive::<I>(
        pattern, &q_prim, device,
        /* use_cuda */ ??? // see below
    )
}
```

Plumbing the `use_cuda` flag: today the engine knows its `sparse_solver`
via `cfg.params.sparse_solver`. The three primitive helpers are called from
`mmc_op::timestep_forward` (forward direction) and `TimestepOp::backward`
(backward direction). Both have access to the engine's `Config`. Add a
`use_cuda: bool` parameter to each primitive helper's signature; pass it
through from the call sites in `mmc_op.rs`. Two new parameters total
(`mmc_op.rs` already plumbs the flag for the existing `triangular_csr_solve`
call — we extend the same pattern).

### Drop impl

The existing `Drop for CudaPatternCache` releases the SpSV descriptors and
flushes via cubecl. Extend it to release the two new SpMV matrix descriptors:

```rust
cusparseDestroySpMat(self.sp_mat_spmv);
cusparseDestroySpMat(self.sp_mat_rowsum);
```

Place in cache-field declaration order so Drop runs in reverse-declaration
order (Rust default). Workspaces and `d_col_identity` are cubecl Handles —
they drop automatically.

## File layout

```
src/sparse/cusparse.rs        MODIFIED
  + add cusparse_spmv_forward, cusparse_spmv_backward, cusparse_assemble_backward
  + extend CudaPatternCache with 5 fields
  + extend build_cuda_pattern_cache setup
  + extend Drop impl

src/sparse/dispatch.rs        MODIFIED
  + add 3 new dispatch entries

src/sparse/mod.rs             MODIFIED
  + rewrite 3 primitive helpers as thin dispatches
  + add use_cuda parameter

src/routing/mmc_op.rs         MODIFIED
  + plumb use_cuda through to the 3 primitive helper calls
  + (no math changes — same call sites as today)

tests/sparse_cusparse_v8.rs   NEW
  + 3 bit-match tests (CPU vs CUDA) for the new SpMV functions
```

No changes outside the sparse module + the call sites in mmc_op.

## Concerns

1. **cuSPARSE matrix-descriptor lifetime.** The two new `cusparseSpMatDescr_t`
   handles live in `CudaPatternCache` alongside the existing `sp_mat`. They
   share device pointers (d_crow, d_col, d_adj_values) with the existing
   SpSV descriptor. cuSPARSE doesn't copy data into descriptors — they hold
   raw pointers. We must keep the underlying `d_crow`/`d_col`/d_adj_values
   Handles alive for the cache's lifetime. They already are.

2. **Workspace sizing is per-op-flag.** `cusparseSpMV_bufferSize` returns
   different sizes for NON_TRANSPOSE vs TRANSPOSE on the same matrix. Sites
   1 + 2 share the matrix but need two workspaces. Site 3 needs its own.
   Three workspaces total. Small (typically < 1 MB each).

3. **`d_adj_values` not yet a cache field.** SP-6/7 likely uploaded
   `adj_values` ad-hoc per call (via `cube_tensor_to_primitive` of the values
   tensor). For SP-9 we need a *persistent* `d_adj_values` Handle in the
   cache since the cuSPARSE matrix descriptors hold the pointer across
   calls. If `CsrPattern::adj_values` is already on host, upload it once in
   `build_cuda_pattern_cache` via `client.create_from_slice`. If it's
   already a cubecl Handle somewhere, reuse that. Plan Task 1 confirms.

4. **`alpha = -1.0` for site 3.** cuSPARSE accepts `alpha` and `beta` as
   `*const c_void` pointing at the matching numeric type. For f32 we pass
   `&-1.0f32 as *const f32 as *const c_void`. Existing SpSV plumbing uses
   `alpha = 1.0` only — site 3 is the first SP that exercises non-unit alpha.
   Verify cuSPARSE's expected calling convention.

5. **V7a's 0.7× target without margin analysis.** If cuSPARSE SpMV on our
   matrix size (~65K × 65K, ~3 nnz/row, very sparse) only achieves 2-3×
   speedup vs scatter-add, the wall-time win may not reach the 0.7× ratio.
   The SP-8 close-out's "fusion sped both backends symmetrically" warning
   still applies — CPU may also speed up if we ever revisit it. Tentative
   plan: if V7a misses but V7b passes by a large margin (e.g., scatter at
   5%), document the partial win and propose a follow-up (CUDA Graphs or
   kernel fusion for the remaining 1 μs binops).

6. **CPU path on NdArray is unchanged.** The dispatch's `use_cuda=false`
   branch returns the existing `.scatter`-based implementation verbatim.
   V1/V2/V4 are NdArray-pinned by default and stay green.

7. **The name collision between `spmv_backward_primitive` (mod) and
   `spmv_backward_primitive_dispatch` (dispatch).** Cosmetic only. Plan
   resolves at file edit time.

8. **cudarc 0.19 cuSPARSE SpMV surface.** Verify availability in
   `~/.cargo/registry/src/.../cudarc-0.19.7/src/cusparse/`. Functions needed:
   `cusparseSpMV`, `cusparseSpMV_bufferSize`, `cusparseSpMV_preprocess`,
   `cusparseCreateCsr`, `cusparseDestroySpMat`, `cusparseCreateDnVec`,
   `cusparseDestroyDnVec`. All standard. Plan Task 1 verifies.

9. **CLAUDE.md invariant #2 preserved.** f32 throughout. cuSPARSE SpMV
   supports f32 directly (`CUDA_R_32F` data types in the descriptor).

## Open assumptions

1. cuSPARSE SpMV at our matrix scale (65K × 65K, very sparse) is at least
   5× faster than the `scatter_kernel` it replaces. If it's only 2-3×,
   V7a misses; V7b still passes; we close as partial and propose follow-up.
2. The cuSPARSE bufferSize for our matrices is small enough that 3
   workspace Handles per pattern don't bloat memory. At nnz ≈ 65K × 3 = 200K,
   workspace is typically a few MB total — negligible vs GPU memory budget.
3. `cusparseSpMV_preprocess` reduces per-call overhead enough to justify
   the one-time setup cost (~1-10 ms per pattern). If not, drop the
   preprocess call — still functionally correct.
4. The fused TimestepOp's analytical backward (SP-8) routes through these
   three primitive helpers unchanged. SP-9 swaps the implementation under
   the same API. No backward-math changes needed.

## What's deferred

- **MLP / Adam scatter sources.** If V7b passes by a large margin (say,
  scatter at < 5%) but V7a still misses, the remaining wall-time floor is
  the small-binop / launch-overhead surface (see SP-8 close-out: 8M
  `cuLaunchKernel` calls at 2.3 μs each = 18 sec host). CUDA Graphs or
  cubecl kernel fusion is the next lever — a separate SP.
- **Upstreaming the SP-7 vendor patches** to tracel-ai. Still pending.
- **Multi-GPU, Wgpu backend.** SP-9 targets the Cuda backend only.

## Next steps

1. You review this spec.
2. After approval: write `.claude/specs/2026-05-22-sp9-cusparse-spmv-plan.md`
   with the full task-by-task plan (cuSPARSE setup, three new functions,
   dispatch wiring, V8 bit-match test).
3. Subagent-driven execution per SP-6/7/8 precedent.
