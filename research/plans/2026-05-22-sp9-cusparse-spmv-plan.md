# SP-9 cuSPARSE SpMV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three `Tensor::scatter(0, ..., IndexingUpdateOp::Add)` call sites in `src/sparse/mod.rs` (forward SpMV, backward SpMV-transpose, assemble-backward row-sum) with `cusparseSpMV` calls on the CUDA backend. Crush the 77.5% `scatter_kernel_t_f32_i_i32` GPU bottleneck.

**Architecture:** Extend the existing SP-6/SP-7 cuSPARSE infrastructure (`CudaPatternCache` in `src/sparse/cusparse.rs`) with two new sparse-matrix descriptors (`sp_mat_spmv` for sites 1+2; `sp_mat_rowsum` for site 3) and three new SpMV functions following the existing stream-share + zero-copy plumbing. CPU (NdArray) path stays on `.scatter` unchanged. Three new dispatch entries route per `sparse_solver` config.

**Tech Stack:** cudarc 0.19.7 cuSPARSE bindings (already a transitive dep), cubecl Handles for GPU memory, BURN 0.21 backend primitives. No new crate dependencies.

**Spec:** `.claude/specs/2026-05-22-sp9-cusparse-spmv-design.md`

---

## Pre-flight: what already exists

- `src/sparse/cusparse.rs` (1215 lines) — `CudaPatternCache` + `cusparse_forward` + `cusparse_backward_solve` + `cusparse_grada` + `build_cuda_pattern_cache` + `Drop for CudaPatternCache`. SP-6/7 plumbing for stream-share via `cubecl_stream_active` and zero-copy via `cube_tensor_to_primitive` is here.
- `src/sparse/dispatch.rs` (176 lines) — existing dispatch entries `forward_primitive`, `backward_solve_primitive`, `grada_primitive` switch CPU vs CUDA per `use_cuda: bool`.
- `src/sparse/mod.rs:629-715` — the three primitive helpers to rewrite: `spmv_primitive`, `assemble_primitive` (no scatter, untouched), `assemble_backward_primitive`, `spmv_backward_primitive`.
- `src/routing/mmc_op.rs` — call sites for the primitive helpers (TimestepOp forward uses `spmv_primitive` + `assemble_primitive`; TimestepOp backward uses `assemble_backward_primitive` + `spmv_backward_primitive`).
- `tests/sparse_cusparse_v5.rs` — V5 CSR-solve CPU/CUDA bit-match. Mirror its structure for V8.
- `tests/sp8_v7_perf.rs` and `tests/sp8_v7_profile.rs` — V7a/V7b gates (currently red). SP-9 makes them green.

---

## File Structure

**Created:**

- `tests/sparse_cusparse_v8.rs` — Task 9: SpMV CPU/CUDA bit-match for all 3 sites.

**Modified:**

- `src/sparse/cusparse.rs` — Tasks 1-4 (cache extension + setup) + Tasks 5-7 (three new SpMV functions).
- `src/sparse/dispatch.rs` — Task 8: three new dispatch entries.
- `src/sparse/mod.rs` — Task 8: rewrite three primitive helpers as thin dispatches with `use_cuda` parameter.
- `src/routing/mmc_op.rs` — Task 8: plumb `use_cuda` through to the three primitive-helper call sites.

**No changes:**

- `src/routing/mmc.rs` (already plumbs `sparse_solver` for the existing CSR solve — same flag is what we use).
- `tests/training_verification.rs`, `tests/sparse_gradcheck.rs`, `tests/sp8_gradcheck.rs`, `tests/sparse_cusparse_v5.rs` — regression gates only.

---

## Conventions for this plan

- All new `cusparse_*` functions mirror the existing pattern in `src/sparse/cusparse.rs:140-340` (`cusparse_backward_solve` / `cusparse_grada`): backend-generic over `B: Backend + 'static`, runtime `TypeId` check for `Cuda<f32, i32>`, `client.flush()` before cuSPARSE call, no D↔H syncs.
- Cite line numbers in `src/sparse/mod.rs` when replacing scatter sites so a future reader can map dispatch back to the original.
- One new commit per task. No `--amend`.
- New code stays clippy-clean: `cargo clippy --lib 2>&1 | grep -E "(cusparse|dispatch|sparse/mod)" | head -5` must be empty after each commit.
- Pre-existing clippy lints in routing-core code are out of scope.

---

## Naming decision (locked here)

The dispatch entries are named with the `_dispatch` suffix to avoid the
collision with the existing thin helpers in `src/sparse/mod.rs`:

- Dispatch entries (in `src/sparse/dispatch.rs`):
  - `spmv_forward_dispatch`
  - `spmv_backward_dispatch`
  - `assemble_backward_dispatch`
- Thin helpers (in `src/sparse/mod.rs`, unchanged names):
  - `spmv_primitive`
  - `spmv_backward_primitive`
  - `assemble_backward_primitive`

Each thin helper becomes: parse autograd-irrelevant params, call into the
matching dispatch entry, return the result.

---

### Task 1: Verify cudarc cuSPARSE SpMV API + add `d_adj_values` cache field

The two new sparse-matrix descriptors point at the same `adj_values` buffer.
Today SP-6 may upload it per call. SP-9 needs it persistent in the cache.

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Inspect cudarc cuSPARSE SpMV surface**

Run:
```bash
grep -rn "cusparseSpMV\|cusparseCreateCsr\|cusparseSpMV_bufferSize\|cusparseSpMV_preprocess\|cusparseDestroySpMat" ~/.cargo/registry/src/index.crates.io-*/cudarc-0.19.7/src/cusparse/ 2>/dev/null | head -20
```

Verify these symbols are exported:
- `cusparseSpMV` (per-call)
- `cusparseSpMV_bufferSize` (one-time per matrix + op)
- `cusparseSpMV_preprocess` (one-time per matrix + op; cuSPARSE 11.7+)
- `cusparseCreateCsr` (creates `cusparseSpMatDescr_t`)
- `cusparseDestroySpMat` (releases)
- `cusparseCreateDnVec` / `cusparseDestroyDnVec` (already used by SP-6/7)

If any symbol is missing or named differently in 0.19, note the actual name —
the rest of the plan uses these names. If the entire `cusparseSpMV` surface
is missing, STOP and report; we may need a cudarc bump.

- [ ] **Step 2: Locate where `adj_values` is currently uploaded**

Run:
```bash
grep -n "adj_values\|d_adj_values\|cusparseCreateCsr" src/sparse/cusparse.rs | head -20
```

Three possibilities:
1. `adj_values` is uploaded per-call inside `cusparse_forward` / `cusparse_backward_solve` (likely, given the SP-6 plumbing).
2. It's already a persistent `d_adj_values` Handle in `CudaPatternCache` — search for the field name.
3. It's referenced by the existing `sp_mat` descriptor (in which case there's already a persistent buffer somewhere).

Inspect `build_cuda_pattern_cache` (~line 618) to confirm. SP-9 requires
`adj_values` as a persistent cache field; if it isn't already, add it.

- [ ] **Step 3: Add (or confirm) `d_adj_values` cache field**

In `src/sparse/cusparse.rs`, append to `CudaPatternCache` declaration (line ~543):

```rust
    /// Persistent device buffer of CsrPattern.adj_values (length nnz, f32).
    /// SP-9: shared by sp_mat_spmv and sp_mat_rowsum SpMV descriptors.
    pub(crate) d_adj_values: burn_cubecl::cubecl::server::Handle,
```

(Skip this step if the field already exists. Note in the commit message
whether it pre-existed.)

In `build_cuda_pattern_cache` (after the existing `d_crow` / `d_col` setup,
roughly line 730), upload `adj_values`:

```rust
    // SP-9: persistent device buffer for adj_values, shared by the SpMV
    // sparse-matrix descriptors in Task 3.
    let adj_bytes: &[u8] = bytemuck::cast_slice(&pattern.adj_values);
    let d_adj_values: burn_cubecl::cubecl::server::Handle =
        client.create_from_slice(adj_bytes);
```

Add `bytemuck` if not already a dependency (it's already in the project for
the existing `d_crow`/`d_col` casts).

In the struct construction at the end of `build_cuda_pattern_cache`, include:
```rust
        d_adj_values,
```

- [ ] **Step 4: Build + V5 regression check**

Run:
```
cargo build --lib 2>&1 | tail -5
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -3
```
Expected: clean compile; V5 passes (this task only adds a field, doesn't change existing behavior).

- [ ] **Step 5: Commit**

```bash
git add src/sparse/cusparse.rs
git commit -m "SP-9 Task 1: persist d_adj_values in CudaPatternCache

Preparation for SP-9 SpMV descriptors (Tasks 3-7), which need a stable
device pointer to adj_values that lives for the cache's lifetime. No
behavior change to existing SpSV / grada paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `d_col_identity` + 5 new cache field declarations

Declarations only — population happens in Task 3.

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Extend `CudaPatternCache` struct**

In `src/sparse/cusparse.rs` (around line 543), append five new fields:

```rust
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
```

The struct's `Drop` (around line 571) must release the new descriptors. Update it:

```rust
impl Drop for CudaPatternCache {
    fn drop(&mut self) {
        // ... existing drop body ...
        // SP-9 additions: destroy the SpMV descriptors before the underlying
        // Handles drop. Order matters — Rust drops fields in declaration
        // order, so descriptors are released here in the Drop body BEFORE
        // d_crow / d_col / d_adj_values / d_col_identity go away.
        unsafe {
            cudarc::cusparse::sys::cusparseDestroySpMat(self.sp_mat_spmv)
                .result()
                .expect("cusparseDestroySpMat sp_mat_spmv failed");
            cudarc::cusparse::sys::cusparseDestroySpMat(self.sp_mat_rowsum)
                .result()
                .expect("cusparseDestroySpMat sp_mat_rowsum failed");
        }
        // existing destroy calls follow ...
    }
}
```

Locate the existing destroy calls in `Drop` and place the new ones at the
top of the unsafe block — they need to run BEFORE the existing `sp_mat`
descriptor is destroyed (their cleanup is independent, but ordering is
defensive).

- [ ] **Step 2: Build (the new fields are unused — expect warnings, no errors)**

Run:
```
cargo build --lib 2>&1 | tail -10
```

Expected: clean compile. The new fields will trigger `dead_code` warnings
because Task 3 hasn't populated them yet — that's fine; Tasks 3+ resolve
them. Don't add `#[allow(dead_code)]` — the warnings are correct, just
temporary.

The struct won't construct in `build_cuda_pattern_cache` until Task 3
because the new fields aren't initialized. The compiler error from missing
field initializers is the gate Task 3 must clear. **If Step 2 reports a
"missing field" error, that's expected** — Task 3 fills them.

Actually, the compile WILL fail with "missing field" errors at the
`CudaPatternCache { ... }` construction site. To keep this task's commit
buildable, add the field declarations now but leave construction for Task 3.
This means Task 2 doesn't ship a clean compile — only the declarations.

**Revised Step 2:** Add temporary `todo!()` initializers in
`build_cuda_pattern_cache`'s struct constructor to make the build compile.
Task 3 replaces them.

In `build_cuda_pattern_cache` (find the final `Self { ... }` near the bottom):

```rust
    Self {
        // ... existing fields ...
        sp_mat_spmv: unsafe { std::mem::zeroed() }, // Task 3
        sp_mat_rowsum: unsafe { std::mem::zeroed() }, // Task 3
        d_col_identity: client.create(0),          // Task 3 — dummy zero-byte handle
        workspace_spmv_n: client.create(0),        // Task 3
        workspace_spmv_nt: client.create(0),       // Task 3
        workspace_rowsum: client.create(0),        // Task 3
    }
```

The `unsafe { std::mem::zeroed() }` for the descriptor pointers is a
deliberate temporary hack — they're raw `*mut` pointers, and a null pointer
is acceptable for the build to pass. Task 3 immediately replaces both.

The Drop impl Step 1 added will panic on null descriptors if the cache is
dropped between Tasks 2 and 3 (uncommon since the cache is built per-batch
in production code, and we won't run the production path between tasks).
Defensive: wrap the `cusparseDestroySpMat` calls in `if !ptr.is_null()`.

Updated Drop fix (also in Step 1, refining):

```rust
        unsafe {
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
        }
```

Run:
```
cargo build --lib 2>&1 | tail -5
```
Expected: clean compile.

- [ ] **Step 3: V5 regression check**

Run:
```
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -3
```
Expected: PASS — Task 2 added declarations + null-placeholder construction,
no behavior change to existing code paths.

- [ ] **Step 4: Commit**

```bash
git add src/sparse/cusparse.rs
git commit -m "SP-9 Task 2: declare 5 new SpMV fields in CudaPatternCache

Adds sp_mat_spmv, sp_mat_rowsum, d_col_identity, and three workspace
Handles. Initialized to null/zero-byte placeholders — Task 3 populates
them. Drop impl is null-safe for the placeholder state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Populate SpMV descriptors + workspaces in `build_cuda_pattern_cache`

The setup work. After this task the cache holds two ready-to-use SpMV
descriptors with their workspaces.

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Add `d_col_identity` construction**

In `build_cuda_pattern_cache`, after the existing `d_crow` / `d_col` /
`d_row_for_nnz` / `d_adj_values` uploads, add:

```rust
    // SP-9: identity column indices [0, 1, 2, ..., nnz-1] for sp_mat_rowsum.
    let nnz = pattern.nnz();
    let col_identity_vec: Vec<i32> = (0..nnz as i32).collect();
    let col_identity_bytes: &[u8] = bytemuck::cast_slice(&col_identity_vec);
    let d_col_identity: burn_cubecl::cubecl::server::Handle =
        client.create_from_slice(col_identity_bytes);
```

- [ ] **Step 2: Create `sp_mat_spmv` (n × n) descriptor**

After the existing `sp_mat` creation (search for `cusparseCreateCsr`), add:

```rust
    // SP-9: create sp_mat_spmv for SpMV sites 1+2 (forward y = N·q and
    // backward gq = N^T·gi). Same (n × n) shape as the existing sp_mat (used
    // by SpSV), but a separate cuSPARSE descriptor so SpMV and SpSV calls
    // don't share state.
    let crow_ptr = unsafe { handle_device_ptr(&client, &d_crow) };
    let col_ptr = unsafe { handle_device_ptr(&client, &d_col) };
    let adj_ptr = unsafe { handle_device_ptr(&client, &d_adj_values) };

    let mut sp_mat_spmv: cudarc::cusparse::sys::cusparseSpMatDescr_t =
        std::ptr::null_mut();
    unsafe {
        cudarc::cusparse::sys::cusparseCreateCsr(
            &mut sp_mat_spmv,
            pattern.n as i64,     // rows
            pattern.n as i64,     // cols
            nnz as i64,
            crow_ptr as *mut _,
            col_ptr as *mut _,
            adj_ptr as *mut _,
            cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I, // crow type
            cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I, // col type
            cudarc::cusparse::sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateCsr sp_mat_spmv failed");
    }
```

- [ ] **Step 3: Create `sp_mat_rowsum` (n × nnz) descriptor**

Immediately after Step 2:

```rust
    let col_id_ptr = unsafe { handle_device_ptr(&client, &d_col_identity) };
    let mut sp_mat_rowsum: cudarc::cusparse::sys::cusparseSpMatDescr_t =
        std::ptr::null_mut();
    unsafe {
        cudarc::cusparse::sys::cusparseCreateCsr(
            &mut sp_mat_rowsum,
            pattern.n as i64,     // rows
            nnz as i64,           // cols = nnz
            nnz as i64,
            crow_ptr as *mut _,                  // SAME crow as sp_mat_spmv
            col_id_ptr as *mut _,                // identity col indices
            adj_ptr as *mut _,                   // SAME adj_values
            cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cudarc::cusparse::sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cudarc::cusparse::sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F,
        )
        .result()
        .expect("cusparseCreateCsr sp_mat_rowsum failed");
    }
```

- [ ] **Step 4: Compute workspace sizes + allocate**

After Step 3:

```rust
    // SP-9: allocate three SpMV workspaces. cusparseSpMV_bufferSize takes the
    // sparse-mat descriptor, dense-vec descriptor placeholders, op flag,
    // alpha, beta, algorithm; we construct dummies for the dnvecs since
    // bufferSize only inspects shapes.

    use cudarc::cusparse::sys::{
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseOperation_t::{CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE},
        cusparseSpMV_bufferSize,
        cusparseCreateDnVec, cusparseDestroyDnVec,
    };

    // Helper: query bufferSize for an (op, sp_mat, input_len, output_len) combo.
    let query_workspace_bytes = |
        sp_mat: cudarc::cusparse::sys::cusparseSpMatDescr_t,
        op: cudarc::cusparse::sys::cusparseOperation_t,
        input_len: usize,
        output_len: usize,
    | -> usize {
        let mut dummy_in_ptr: u64 = 1; // non-null dummy
        let mut dummy_out_ptr: u64 = 1;
        let mut dnvec_in: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut dnvec_out: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let mut buf_bytes: usize = 0;
        unsafe {
            cusparseCreateDnVec(&mut dnvec_in, input_len as i64,
                &mut dummy_in_ptr as *mut _ as *mut _,
                cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
                .result().expect("dnvec_in create");
            cusparseCreateDnVec(&mut dnvec_out, output_len as i64,
                &mut dummy_out_ptr as *mut _ as *mut _,
                cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
                .result().expect("dnvec_out create");
            cusparseSpMV_bufferSize(
                handle, op,
                &alpha as *const f32 as *const _,
                sp_mat,
                dnvec_in,
                &beta as *const f32 as *const _,
                dnvec_out,
                cudarc::cusparse::sys::cudaDataType::CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &mut buf_bytes,
            )
            .result().expect("cusparseSpMV_bufferSize");
            cusparseDestroyDnVec(dnvec_in).result().expect("destroy dnvec_in");
            cusparseDestroyDnVec(dnvec_out).result().expect("destroy dnvec_out");
        }
        buf_bytes.max(1)  // cuSPARSE may return 0 — allocate at least 1 byte
    };

    let bytes_spmv_n  = query_workspace_bytes(sp_mat_spmv, CUSPARSE_OPERATION_NON_TRANSPOSE, pattern.n, pattern.n);
    let bytes_spmv_nt = query_workspace_bytes(sp_mat_spmv, CUSPARSE_OPERATION_TRANSPOSE,     pattern.n, pattern.n);
    let bytes_rowsum  = query_workspace_bytes(sp_mat_rowsum, CUSPARSE_OPERATION_NON_TRANSPOSE, nnz,       pattern.n);

    let workspace_spmv_n: burn_cubecl::cubecl::server::Handle  = client.create(bytes_spmv_n);
    let workspace_spmv_nt: burn_cubecl::cubecl::server::Handle = client.create(bytes_spmv_nt);
    let workspace_rowsum: burn_cubecl::cubecl::server::Handle  = client.create(bytes_rowsum);
```

The `handle` variable in `query_workspace_bytes` references the cuSPARSE
handle constructed earlier in `build_cuda_pattern_cache`. Capture it in the
closure if the closure can't see it — easier to just inline the calls if
borrow checker complains.

- [ ] **Step 5: Replace the Task 2 placeholders with real values in the struct constructor**

In the `Self { ... }` block at the bottom of `build_cuda_pattern_cache`, replace:

```rust
        sp_mat_spmv: unsafe { std::mem::zeroed() },
        sp_mat_rowsum: unsafe { std::mem::zeroed() },
        d_col_identity: client.create(0),
        workspace_spmv_n: client.create(0),
        workspace_spmv_nt: client.create(0),
        workspace_rowsum: client.create(0),
```

with:

```rust
        sp_mat_spmv,
        sp_mat_rowsum,
        d_col_identity,
        workspace_spmv_n,
        workspace_spmv_nt,
        workspace_rowsum,
```

- [ ] **Step 6: Build + V5 regression**

Run:
```
cargo build --lib 2>&1 | tail -10
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -3
```
Expected: clean compile, V5 PASS. The added descriptors are unused at call
time — but they exist and get cleaned up properly via Drop.

- [ ] **Step 7: Commit**

```bash
git add src/sparse/cusparse.rs
git commit -m "SP-9 Task 3: populate SpMV descriptors + workspaces in pattern cache

build_cuda_pattern_cache now creates sp_mat_spmv (n × n) and
sp_mat_rowsum (n × nnz), uploads d_col_identity, queries cuSPARSE
bufferSize for each SpMV configuration, and allocates the three
workspaces via cubecl. Descriptors hold raw pointers to d_crow,
d_col, d_col_identity, and d_adj_values — all persistent cache fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add `cusparse_spmv_forward` (site 1)

Forward SpMV: `y = N · q`. Replaces `Tensor::scatter` at `src/sparse/mod.rs:663`.

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Write the function**

After the existing `cusparse_grada` (around line 340) in `src/sparse/cusparse.rs`:

```rust
// SP-9: cusparse_spmv_forward — site 1, forward y = N · q via cusparseSpMV
// (NON_TRANSPOSE) on sp_mat_spmv. Stream-shared with cubecl, zero-copy.

/// Compute `y = N · q` via cuSPARSE SpMV. Returns the result as a primitive
/// tensor of shape `[n]`. No D↔H syncs — input and output stay on device.
///
/// `cache` must come from `build_cuda_pattern_cache` for the matching
/// `CsrPattern`. `q_prim` is a `Tensor<Cuda<f32, i32>>` of length `n`.
pub(crate) fn cusparse_spmv_forward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    q_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    use cudarc::cusparse::sys::{
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseCreateDnVec, cusparseDestroyDnVec,
        cusparseSetStream, cusparseSpMV,
    };

    let client = compute_client::<B>(device);
    let n = (cache_pattern_size::<B>(cache)) as i64; // helper — returns pattern.n

    // Allocate the output y on device, shape [n], zero-initialized.
    let y_bytes = (n as usize) * std::mem::size_of::<f32>();
    let y_handle: burn_cubecl::cubecl::server::Handle = client.create(y_bytes);

    // Get raw device pointers for q (input) and y (output).
    let q_view = cuda_view_from_cube_tensor::<B>(q_prim);
    let q_ptr = q_view.ptr;
    let y_ptr = unsafe { handle_device_ptr(&client, &y_handle) };
    let workspace_ptr = unsafe { handle_device_ptr(&client, &cache.workspace_spmv_n) };

    // Set the stream cuSPARSE should use to cubecl's active stream.
    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream forward SpMV failed");
    }

    // Flush cubecl's queued kernels before launching the cuSPARSE op.
    client.flush().expect("cubecl flush before cusparse_spmv_forward failed");

    // Build dense-vector descriptors and run SpMV.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        let mut q_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut y_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut q_dn, n, q_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("cusparseCreateDnVec q (forward SpMV) failed");
        cusparseCreateDnVec(&mut y_dn, n, y_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("cusparseCreateDnVec y (forward SpMV) failed");

        cusparseSpMV(
            cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            cache.sp_mat_spmv,
            q_dn,
            &beta as *const f32 as *const _,
            y_dn,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr as *mut _,
        )
        .result().expect("cusparseSpMV forward failed");

        cusparseDestroyDnVec(q_dn).result().expect("cusparseDestroyDnVec q (fwd)");
        cusparseDestroyDnVec(y_dn).result().expect("cusparseDestroyDnVec y (fwd)");
    }

    cube_tensor_to_primitive::<B>(client, device, y_handle, [n as usize])
}

/// Returns `cache.pattern_n`. The cache stores it explicitly for fast access.
#[inline]
fn cache_pattern_size<B: Backend>(cache: &CudaPatternCache) -> usize {
    // If `CudaPatternCache` already has an `n: usize` field from SP-6/7, use it.
    // Otherwise the implementer adds `pub(crate) n: usize` to the struct
    // (initialized in build_cuda_pattern_cache from pattern.n) and uses it
    // here. Either way: O(1) read.
    cache.n
}
```

The `cache.n` field may not exist yet. If it doesn't, add it to the struct
in this task (one extra line of declaration, initialized as `n: pattern.n`
in the constructor).

The `compute_client::<B>` and `cube_tensor_to_primitive::<B>` helpers
already exist in `src/sparse/cusparse.rs` (from SP-7). The `cuda_view_from_cube_tensor::<B>`
already exists too (around line 57). Reuse all of them.

- [ ] **Step 2: Build**

Run:
```
cargo build --lib 2>&1 | tail -10
```
Expected: clean compile. The new function isn't called yet (Task 8 wires
the dispatch).

- [ ] **Step 3: Commit**

```bash
git add src/sparse/cusparse.rs
git commit -m "SP-9 Task 4: add cusparse_spmv_forward (site 1)

Forward SpMV y = N · q via cusparseSpMV(NON_TRANSPOSE) on sp_mat_spmv.
Reuses SP-7 stream-share + zero-copy plumbing. Not wired yet — Task 8
adds the dispatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add `cusparse_spmv_backward` (site 2)

Backward SpMV: `gq = N^T · gi`. Replaces `Tensor::scatter` at `src/sparse/mod.rs:712`.

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Write the function**

After `cusparse_spmv_forward`:

```rust
// SP-9: cusparse_spmv_backward — site 2, backward gq = N^T · gi via
// cusparseSpMV(TRANSPOSE) on sp_mat_spmv. Same matrix descriptor as forward.

/// Compute `gq = N^T · gi` via cuSPARSE SpMV with TRANSPOSE op. Returns the
/// result as a primitive tensor of shape `[n]`.
pub(crate) fn cusparse_spmv_backward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    gi_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    use cudarc::cusparse::sys::{
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        cusparseCreateDnVec, cusparseDestroyDnVec,
        cusparseSetStream, cusparseSpMV,
    };

    let client = compute_client::<B>(device);
    let n = cache.n as i64;

    let y_bytes = (n as usize) * std::mem::size_of::<f32>();
    let y_handle: burn_cubecl::cubecl::server::Handle = client.create(y_bytes);

    let gi_view = cuda_view_from_cube_tensor::<B>(gi_prim);
    let gi_ptr = gi_view.ptr;
    let y_ptr = unsafe { handle_device_ptr(&client, &y_handle) };
    let workspace_ptr = unsafe { handle_device_ptr(&client, &cache.workspace_spmv_nt) };

    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream backward SpMV failed");
    }
    client.flush().expect("cubecl flush before cusparse_spmv_backward failed");

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        let mut gi_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut y_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut gi_dn, n, gi_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("cusparseCreateDnVec gi (bwd SpMV) failed");
        cusparseCreateDnVec(&mut y_dn, n, y_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("cusparseCreateDnVec y (bwd SpMV) failed");

        cusparseSpMV(
            cache.handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            &alpha as *const f32 as *const _,
            cache.sp_mat_spmv,
            gi_dn,
            &beta as *const f32 as *const _,
            y_dn,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr as *mut _,
        )
        .result().expect("cusparseSpMV backward (transpose) failed");

        cusparseDestroyDnVec(gi_dn).result().expect("cusparseDestroyDnVec gi (bwd)");
        cusparseDestroyDnVec(y_dn).result().expect("cusparseDestroyDnVec y (bwd)");
    }

    cube_tensor_to_primitive::<B>(client, device, y_handle, [n as usize])
}
```

- [ ] **Step 2: Build + commit**

```
cargo build --lib 2>&1 | tail -5
```

```bash
git add src/sparse/cusparse.rs
git commit -m "SP-9 Task 5: add cusparse_spmv_backward (site 2)

Backward SpMV gq = N^T · gi via cusparseSpMV(TRANSPOSE). Reuses
sp_mat_spmv descriptor + workspace_spmv_nt. Not wired yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Add `cusparse_assemble_backward` (site 3)

Site 3: `gc = -sp_mat_rowsum · gA`. The `α=-1` embeds the negation.

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Write the function**

After `cusparse_spmv_backward`:

```rust
// SP-9: cusparse_assemble_backward — site 3, gc = -sp_mat_rowsum · gA via
// cusparseSpMV(NON_TRANSPOSE) on sp_mat_rowsum with α=-1.

/// Compute `gc[i] = -Σ_k adj[k] · gA[k]` for `k in row i` via cuSPARSE SpMV
/// with α=-1 (negation embedded in the SpMV — no separate .neg() step).
/// Returns the result as a primitive tensor of shape `[n]`.
///
/// `gA_prim` has length `nnz`.
pub(crate) fn cusparse_assemble_backward<B: Backend + 'static>(
    cache: &CudaPatternCache,
    g_a_prim: &B::FloatTensorPrimitive,
    device: &B::Device,
) -> B::FloatTensorPrimitive {
    use cudarc::cusparse::sys::{
        cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparseCreateDnVec, cusparseDestroyDnVec,
        cusparseSetStream, cusparseSpMV,
    };

    let client = compute_client::<B>(device);
    let n = cache.n as i64;
    let nnz = cache.nnz as i64; // add `nnz: usize` field to CudaPatternCache if missing

    let y_bytes = (n as usize) * std::mem::size_of::<f32>();
    let y_handle: burn_cubecl::cubecl::server::Handle = client.create(y_bytes);

    let ga_view = cuda_view_from_cube_tensor::<B>(g_a_prim);
    let ga_ptr = ga_view.ptr;
    let y_ptr = unsafe { handle_device_ptr(&client, &y_handle) };
    let workspace_ptr = unsafe { handle_device_ptr(&client, &cache.workspace_rowsum) };

    let stream = cubecl_stream_active::<B>(device);
    unsafe {
        let stream_for_cusparse = stream as *mut cudarc::cusparse::sys::CUstream_st;
        cusparseSetStream(cache.handle, stream_for_cusparse)
            .result()
            .expect("cusparseSetStream assemble_backward SpMV failed");
    }
    client.flush().expect("cubecl flush before cusparse_assemble_backward failed");

    let alpha: f32 = -1.0;
    let beta: f32 = 0.0;
    unsafe {
        let mut ga_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut y_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut ga_dn, nnz, ga_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("cusparseCreateDnVec gA (assemble_bwd) failed");
        cusparseCreateDnVec(&mut y_dn, n, y_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("cusparseCreateDnVec y (assemble_bwd) failed");

        cusparseSpMV(
            cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha as *const f32 as *const _,
            cache.sp_mat_rowsum,
            ga_dn,
            &beta as *const f32 as *const _,
            y_dn,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace_ptr as *mut _,
        )
        .result().expect("cusparseSpMV assemble_backward failed");

        cusparseDestroyDnVec(ga_dn).result().expect("cusparseDestroyDnVec gA");
        cusparseDestroyDnVec(y_dn).result().expect("cusparseDestroyDnVec y (assemble_bwd)");
    }

    cube_tensor_to_primitive::<B>(client, device, y_handle, [n as usize])
}
```

If `cache.nnz` doesn't exist on `CudaPatternCache`, add it as a `pub(crate)
nnz: usize` field in the struct declaration and initialize from `pattern.nnz()`.

- [ ] **Step 2: Build + commit**

```
cargo build --lib 2>&1 | tail -5
```

```bash
git add src/sparse/cusparse.rs
git commit -m "SP-9 Task 6: add cusparse_assemble_backward (site 3)

Per-row sum gc = -sp_mat_rowsum · gA via cusparseSpMV(NON_TRANSPOSE)
on sp_mat_rowsum with α=-1. The negation embeds in the SpMV — no
separate kernel launch. Not wired yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Optional `cusparseSpMV_preprocess` for cold-call reduction

cuSPARSE 11.7+ exposes `cusparseSpMV_preprocess`, called once per matrix +
op flag, that warms internal caches for subsequent calls. Skipping it is
correct (cuSPARSE will just preprocess on the first call) but adds a few
ms per pattern. Recommended.

**Files:**
- Modify: `src/sparse/cusparse.rs`

- [ ] **Step 1: Add preprocess calls in `build_cuda_pattern_cache`**

After Task 3's workspace allocation, before the final `Self { ... }`:

```rust
    // SP-9: preprocess each SpMV configuration to warm cuSPARSE's internal
    // caches. Skips the first-call overhead at training time. Optional but
    // cheap. cuSPARSE 11.7+ — if cusparseSpMV_preprocess is missing in your
    // cudarc 0.19, comment this block out.
    use cudarc::cusparse::sys::cusparseSpMV_preprocess;
    let warm_input_handle_n: burn_cubecl::cubecl::server::Handle =
        client.create((pattern.n as usize) * std::mem::size_of::<f32>());
    let warm_input_handle_nnz: burn_cubecl::cubecl::server::Handle =
        client.create((nnz as usize) * std::mem::size_of::<f32>());
    let warm_output_handle: burn_cubecl::cubecl::server::Handle =
        client.create((pattern.n as usize) * std::mem::size_of::<f32>());

    let warm_in_n_ptr   = unsafe { handle_device_ptr(&client, &warm_input_handle_n) };
    let warm_in_nnz_ptr = unsafe { handle_device_ptr(&client, &warm_input_handle_nnz) };
    let warm_out_ptr    = unsafe { handle_device_ptr(&client, &warm_output_handle) };

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let preprocess_spmv = |sp_mat, op, in_ptr, in_len, out_ptr, ws_ptr| unsafe {
        let mut in_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        let mut out_dn: cudarc::cusparse::sys::cusparseDnVecDescr_t = std::ptr::null_mut();
        cusparseCreateDnVec(&mut in_dn, in_len as i64, in_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("preprocess in dnvec");
        cusparseCreateDnVec(&mut out_dn, pattern.n as i64, out_ptr as *mut _,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F)
            .result().expect("preprocess out dnvec");
        cusparseSpMV_preprocess(
            handle, op,
            &alpha as *const f32 as *const _,
            sp_mat,
            in_dn,
            &beta as *const f32 as *const _,
            out_dn,
            cudarc::cusparse::sys::cudaDataType::CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            ws_ptr as *mut _,
        )
        .result().expect("cusparseSpMV_preprocess");
        cusparseDestroyDnVec(in_dn).result().expect("destroy preprocess in dnvec");
        cusparseDestroyDnVec(out_dn).result().expect("destroy preprocess out dnvec");
    };

    let ws_n_ptr  = unsafe { handle_device_ptr(&client, &workspace_spmv_n) };
    let ws_nt_ptr = unsafe { handle_device_ptr(&client, &workspace_spmv_nt) };
    let ws_rs_ptr = unsafe { handle_device_ptr(&client, &workspace_rowsum) };

    preprocess_spmv(sp_mat_spmv,
        cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        warm_in_n_ptr, pattern.n, warm_out_ptr, ws_n_ptr);
    preprocess_spmv(sp_mat_spmv,
        cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
        warm_in_n_ptr, pattern.n, warm_out_ptr, ws_nt_ptr);
    preprocess_spmv(sp_mat_rowsum,
        cudarc::cusparse::sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        warm_in_nnz_ptr, nnz as usize, warm_out_ptr, ws_rs_ptr);

    // The warm_* Handles drop at end of scope — cuSPARSE has already
    // recorded what it needs in the descriptors.
```

If cudarc 0.19 doesn't expose `cusparseSpMV_preprocess`, comment out this
entire block. Functionality still correct, just slightly slower first call.

- [ ] **Step 2: Build + V5 regression**

```
cargo build --lib 2>&1 | tail -5
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -3
```
Expected: clean compile, V5 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/sparse/cusparse.rs
git commit -m "SP-9 Task 7: cuSPARSE SpMV preprocess (optional perf hint)

Warm cuSPARSE caches for the three SpMV configurations in
build_cuda_pattern_cache. Skips the first-call setup overhead at
training time. Skippable if cudarc lacks cusparseSpMV_preprocess.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Wire the three new functions through dispatch + thin helpers

The connective tissue: dispatch entries → existing thin helpers in `mod.rs`
→ already-correct call sites in `mmc_op.rs`.

**Files:**
- Modify: `src/sparse/dispatch.rs`
- Modify: `src/sparse/mod.rs`
- Modify: `src/routing/mmc_op.rs`

- [ ] **Step 1: Add three dispatch entries**

Read the current `src/sparse/dispatch.rs` shape:
```bash
grep -n "pub fn\|fn cpu_\|fn cuda_" src/sparse/dispatch.rs | head -10
```

Following the pattern of `forward_primitive` / `backward_solve_primitive` /
`grada_primitive`, append three new entries:

```rust
/// Site 1 dispatch: forward `y = N · q`.
pub fn spmv_forward_dispatch<I: Backend + 'static>(
    pattern: &std::sync::Arc<crate::sparse::CsrPattern>,
    q_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive {
    if use_cuda && is_cuda_backend::<I>() {
        let cache = pattern
            .cuda_cache
            .get()
            .expect("CudaPatternCache not initialized — call ensure_cuda_cache() first");
        crate::sparse::cusparse::cusparse_spmv_forward::<I>(cache, q_prim, device)
    } else {
        crate::sparse::cpu_spmv_forward::<I>(pattern, q_prim, device)
    }
}

/// Site 2 dispatch: backward `gq = N^T · gi`.
pub fn spmv_backward_dispatch<I: Backend + 'static>(
    pattern: &std::sync::Arc<crate::sparse::CsrPattern>,
    gi_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive {
    if use_cuda && is_cuda_backend::<I>() {
        let cache = pattern.cuda_cache.get()
            .expect("CudaPatternCache not initialized");
        crate::sparse::cusparse::cusparse_spmv_backward::<I>(cache, gi_prim, device)
    } else {
        crate::sparse::cpu_spmv_backward::<I>(pattern, gi_prim, device)
    }
}

/// Site 3 dispatch: `gc = -sum_k(gA[k] · adj[k])` per row.
pub fn assemble_backward_dispatch<I: Backend + 'static>(
    pattern: &std::sync::Arc<crate::sparse::CsrPattern>,
    g_a_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive {
    if use_cuda && is_cuda_backend::<I>() {
        let cache = pattern.cuda_cache.get()
            .expect("CudaPatternCache not initialized");
        crate::sparse::cusparse::cusparse_assemble_backward::<I>(cache, g_a_prim, device)
    } else {
        crate::sparse::cpu_assemble_backward::<I>(pattern, g_a_prim, device)
    }
}
```

The `is_cuda_backend::<I>()` helper already exists in `src/sparse/dispatch.rs`
from SP-6 (search for it; if not, copy the pattern from
`forward_primitive`). The `pattern.cuda_cache.get()` access matches the
existing dispatch pattern.

The `crate::sparse::cpu_spmv_forward` etc. are the new CPU-path helpers —
extracted from the current `spmv_primitive` / `spmv_backward_primitive` /
`assemble_backward_primitive` bodies. Step 2 defines them.

- [ ] **Step 2: Refactor the existing primitive helpers in `src/sparse/mod.rs`**

Currently the three helpers do the `.scatter` work directly. SP-9 splits
them into:
- A new `cpu_*` function with the existing `.scatter` body (for dispatch).
- The existing `*_primitive` function rewritten as a thin dispatch caller
  with a new `use_cuda: bool` parameter.

In `src/sparse/mod.rs`, after the existing helper bodies, add:

```rust
/// SP-9 CPU fallback: forward SpMV via Tensor::scatter (the previous
/// implementation, now reused only when use_cuda=false or backend is not Cuda).
pub(crate) fn cpu_spmv_forward<I: Backend>(
    pattern: &CsrPattern,
    q_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
) -> I::FloatTensorPrimitive {
    // verbatim copy of the existing spmv_primitive body (lines 650-665),
    // taking q_prim by reference instead of by value.
    let row_idx = tensor_from_pattern_i32::<I>(&pattern.row_for_nnz, device);
    let col_idx = tensor_from_pattern_i32::<I>(&pattern.col, device);
    let adj = tensor_from_pattern_f32::<I>(&pattern.adj_values, device);
    let q: Tensor<I, 1> = Tensor::from_primitive(TensorPrimitive::Float(q_prim.clone()));

    let q_at_cols = q.gather(0, col_idx);
    let weighted = q_at_cols * adj;
    let zeros: Tensor<I, 1> = Tensor::zeros([pattern.n], device);
    let out = zeros.scatter(0, row_idx, weighted, IndexingUpdateOp::Add);
    match out.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    }
}

/// SP-9 CPU fallback: backward SpMV via Tensor::scatter.
pub(crate) fn cpu_spmv_backward<I: Backend>(
    pattern: &CsrPattern,
    gi_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
) -> I::FloatTensorPrimitive {
    // verbatim copy of spmv_backward_primitive body (lines 698-714).
    let row_idx = tensor_from_pattern_i32::<I>(&pattern.row_for_nnz, device);
    let col_idx = tensor_from_pattern_i32::<I>(&pattern.col, device);
    let adj = tensor_from_pattern_f32::<I>(&pattern.adj_values, device);
    let g_i: Tensor<I, 1> =
        Tensor::from_primitive(TensorPrimitive::Float(gi_prim.clone()));

    let g_i_at_rows = g_i.gather(0, row_idx);
    let weighted = g_i_at_rows * adj;
    let zeros: Tensor<I, 1> = Tensor::zeros([pattern.n], device);
    let out = zeros.scatter(0, col_idx, weighted, IndexingUpdateOp::Add);
    match out.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    }
}

/// SP-9 CPU fallback: assemble backward via Tensor::scatter.
pub(crate) fn cpu_assemble_backward<I: Backend>(
    pattern: &CsrPattern,
    g_a_prim: &I::FloatTensorPrimitive,
    device: &I::Device,
) -> I::FloatTensorPrimitive {
    // verbatim copy of assemble_backward_primitive body (lines 675-693).
    let row_idx = tensor_from_pattern_i32::<I>(&pattern.row_for_nnz, device);
    let adj = tensor_from_pattern_f32::<I>(&pattern.adj_values, device);
    let g_a: Tensor<I, 1> =
        Tensor::from_primitive(TensorPrimitive::Float(g_a_prim.clone()));

    let weighted = g_a * adj;
    let zeros: Tensor<I, 1> = Tensor::zeros([pattern.n], device);
    let summed = zeros.scatter(0, row_idx, weighted, IndexingUpdateOp::Add);
    let out = summed.neg();
    match out.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    }
}
```

Now rewrite the existing three `*_primitive` helpers to dispatch:

```rust
pub fn spmv_primitive<I: Backend + 'static>(
    pattern: &std::sync::Arc<CsrPattern>,
    q_prim: I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive {
    crate::sparse::dispatch::spmv_forward_dispatch::<I>(pattern, &q_prim, device, use_cuda)
}

pub fn spmv_backward_primitive<I: Backend + 'static>(
    pattern: &std::sync::Arc<CsrPattern>,
    gi_prim: I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive {
    crate::sparse::dispatch::spmv_backward_dispatch::<I>(pattern, &gi_prim, device, use_cuda)
}

pub fn assemble_backward_primitive<I: Backend + 'static>(
    pattern: &std::sync::Arc<CsrPattern>,
    g_a_prim: I::FloatTensorPrimitive,
    device: &I::Device,
    use_cuda: bool,
) -> I::FloatTensorPrimitive {
    crate::sparse::dispatch::assemble_backward_dispatch::<I>(pattern, &g_a_prim, device, use_cuda)
}
```

Note the **signature changes**: each helper now takes `&Arc<CsrPattern>`
(not `&CsrPattern`) and a new `use_cuda: bool` parameter. Callers must
update.

- [ ] **Step 3: Update callers in `src/routing/mmc_op.rs`**

Read the call sites:
```bash
grep -n "spmv_primitive\|spmv_backward_primitive\|assemble_backward_primitive\|assemble_primitive" src/routing/mmc_op.rs
```

Update each call to pass `&pattern` (Arc reference, not deref) and the
`use_cuda` flag. The engine config flag was added in SP-6 — search for the
existing `sparse_solver == SparseSolver::Cuda` check (it's in `mmc.rs:186`
for the SpSV call) and use the same pattern.

In `mmc_op.rs`, the existing calls look approximately:

```rust
let i_t_prim = sparse::spmv_primitive::<I>(pattern, qt_p.clone(), &device);
let a_values_prim = sparse::assemble_primitive::<I>(pattern, c1_prim.clone(), &device);
// ... and in the backward:
sparse::spmv_backward_primitive::<I>(&state.pattern, gi_t_prim, &device);
sparse::assemble_backward_primitive::<I>(&state.pattern, g_a_values_prim, &device);
```

Update to:

```rust
let i_t_prim = sparse::spmv_primitive::<I>(pattern, qt_p.clone(), &device, use_cuda);
let a_values_prim = sparse::assemble_primitive::<I>(pattern, c1_prim.clone(), &device);
// ↑ assemble_primitive (forward) has no scatter — unchanged
// ... and in the backward:
sparse::spmv_backward_primitive::<I>(&state.pattern, gi_t_prim, &device, use_cuda);
sparse::assemble_backward_primitive::<I>(&state.pattern, g_a_values_prim, &device, use_cuda);
```

`use_cuda` is plumbed in from `timestep_forward`'s caller, which has access
to the engine config. Add a `use_cuda: bool` parameter to `timestep_forward`
and to the `TimestepOp::backward` save state (or pass through `TimestepState`).
Mirror the SP-6 plumbing for the existing CSR-solve flag.

For the backward path: the `use_cuda` flag is captured in the `TimestepState`
struct at forward time and read back in `TimestepOp::backward`. Add a
`pub use_cuda: bool` field to `TimestepState<B>` (declaration in `mmc_op.rs`).

- [ ] **Step 4: Build + V1 + V5 regression**

```
cargo build --lib 2>&1 | tail -10
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -3
cargo test --release --test training_verification v1_loss_matches 2>&1 | tail -3
cargo test --release --test sp8_gradcheck 2>&1 | tail -3
```

Expected: all pass. V1 runs on NdArray (sparse_solver: cpu → use_cuda=false
→ dispatches to cpu_* helpers → math identical to before). V5 still passes.
gradcheck on NdArray still passes.

- [ ] **Step 5: Commit**

```bash
git add src/sparse/cusparse.rs src/sparse/dispatch.rs src/sparse/mod.rs src/routing/mmc_op.rs
git commit -m "SP-9 Task 8: wire cuSPARSE SpMV through dispatch

Adds three dispatch entries (spmv_forward_dispatch, spmv_backward_dispatch,
assemble_backward_dispatch) that route between cuSPARSE on CUDA and the
existing .scatter path on NdArray. The three *_primitive helpers in
src/sparse/mod.rs gain a use_cuda parameter; mmc_op.rs plumbs the flag
from the engine config (mirroring SP-6's SpSV flag).

V1, V5, gradcheck all still green on NdArray.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: V8 — SpMV CPU/CUDA bit-match test

The math-correctness gate. Three tests, one per new cuSPARSE function.

**Files:**
- Create: `tests/sparse_cusparse_v8.rs`

- [ ] **Step 1: Write the test**

```rust
//! SP-9 V8: per-site CPU/CUDA bit-match for cuSPARSE SpMV.
//!
//! Compares the three new cusparse_spmv_* functions against their CPU
//! .scatter-based counterparts on a small synthetic CsrPattern. Tolerance
//! 1e-5 relative (matches the existing V5 CSR-solve bit-match).
//!
//! Run manually:
//!   cargo test --release --test sparse_cusparse_v8 -- --ignored --nocapture

use std::sync::Arc;

use burn::backend::{NdArray, Wgpu};
use burn::tensor::{backend::Backend, Tensor, TensorPrimitive};
use burn_cuda::Cuda;

use ddrs::sparse::{
    CsrPattern, SparseAdjacency,
    spmv_primitive, spmv_backward_primitive, assemble_backward_primitive,
};

const N: usize = 8;
const REL_TOL: f32 = 1e-5;

fn linear_chain_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![1000.0; N], vec![0.001; N])
}

fn cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        type CudaInner = burn_cuda::Cuda<f32, i32>;
        type Dev = <CudaInner as burn::tensor::backend::BackendTypes>::Device;
        let _d: Dev = Default::default();
    })
    .is_ok()
}

fn build_pattern() -> Arc<CsrPattern> {
    let adj = linear_chain_sparse();
    Arc::new(CsrPattern::from_sparse(&adj))
}

fn vec_max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let denom = av.abs().max(bv.abs()).max(1e-12);
            (av - bv).abs() / denom
        })
        .fold(0.0_f32, f32::max)
}

#[test]
#[ignore]
fn v8_spmv_forward_cpu_vs_cuda_bit_match() {
    type CudaB = Cuda<f32, i32>;
    type NdB = NdArray<f32>;
    if !cuda_available() {
        eprintln!("V8 spmv_forward: skip — no CUDA");
        return;
    }
    let pattern = build_pattern();

    // CPU input.
    let nd_dev = <NdB as Backend>::Device::default();
    let q_cpu_data: Vec<f32> = (0..N).map(|i| 1.0 + i as f32 * 0.1).collect();
    let q_cpu: Tensor<NdB, 1> = Tensor::from_floats(q_cpu_data.as_slice(), &nd_dev);
    let q_cpu_prim = match q_cpu.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let y_cpu_prim = spmv_primitive::<NdB>(&pattern, q_cpu_prim, &nd_dev, /*use_cuda*/ false);
    let y_cpu: Vec<f32> = Tensor::<NdB, 1>::from_primitive(TensorPrimitive::Float(y_cpu_prim))
        .to_data().to_vec().unwrap();

    // CUDA input (same numerical values).
    let cuda_dev = <CudaB as Backend>::Device::default();
    let q_cuda: Tensor<CudaB, 1> = Tensor::from_floats(q_cpu_data.as_slice(), &cuda_dev);
    let q_cuda_prim = match q_cuda.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!(),
    };
    let y_cuda_prim = spmv_primitive::<CudaB>(&pattern, q_cuda_prim, &cuda_dev, /*use_cuda*/ true);
    let y_cuda: Vec<f32> = Tensor::<CudaB, 1>::from_primitive(TensorPrimitive::Float(y_cuda_prim))
        .to_data().to_vec().unwrap();

    let rel = vec_max_rel_diff(&y_cpu, &y_cuda);
    eprintln!("V8 spmv_forward: y_cpu={y_cpu:?}, y_cuda={y_cuda:?}, max_rel={rel}");
    assert!(rel < REL_TOL, "V8 spmv_forward: max_rel={rel} >= {REL_TOL}");
}

#[test]
#[ignore]
fn v8_spmv_backward_cpu_vs_cuda_bit_match() {
    type CudaB = Cuda<f32, i32>;
    type NdB = NdArray<f32>;
    if !cuda_available() {
        eprintln!("V8 spmv_backward: skip — no CUDA");
        return;
    }
    let pattern = build_pattern();
    let nd_dev = <NdB as Backend>::Device::default();
    let cuda_dev = <CudaB as Backend>::Device::default();

    let gi_data: Vec<f32> = (0..N).map(|i| 0.5 + i as f32 * 0.2).collect();
    let gi_cpu: Tensor<NdB, 1> = Tensor::from_floats(gi_data.as_slice(), &nd_dev);
    let gi_cpu_prim = match gi_cpu.into_primitive() {
        TensorPrimitive::Float(p) => p, _ => unreachable!(),
    };
    let y_cpu_prim = spmv_backward_primitive::<NdB>(&pattern, gi_cpu_prim, &nd_dev, false);
    let y_cpu: Vec<f32> = Tensor::<NdB, 1>::from_primitive(TensorPrimitive::Float(y_cpu_prim))
        .to_data().to_vec().unwrap();

    let gi_cuda: Tensor<CudaB, 1> = Tensor::from_floats(gi_data.as_slice(), &cuda_dev);
    let gi_cuda_prim = match gi_cuda.into_primitive() {
        TensorPrimitive::Float(p) => p, _ => unreachable!(),
    };
    let y_cuda_prim = spmv_backward_primitive::<CudaB>(&pattern, gi_cuda_prim, &cuda_dev, true);
    let y_cuda: Vec<f32> = Tensor::<CudaB, 1>::from_primitive(TensorPrimitive::Float(y_cuda_prim))
        .to_data().to_vec().unwrap();

    let rel = vec_max_rel_diff(&y_cpu, &y_cuda);
    eprintln!("V8 spmv_backward: max_rel={rel}");
    assert!(rel < REL_TOL, "V8 spmv_backward: max_rel={rel} >= {REL_TOL}");
}

#[test]
#[ignore]
fn v8_assemble_backward_cpu_vs_cuda_bit_match() {
    type CudaB = Cuda<f32, i32>;
    type NdB = NdArray<f32>;
    if !cuda_available() {
        eprintln!("V8 assemble_backward: skip — no CUDA");
        return;
    }
    let pattern = build_pattern();
    let nd_dev = <NdB as Backend>::Device::default();
    let cuda_dev = <CudaB as Backend>::Device::default();

    let nnz = pattern.nnz();
    let ga_data: Vec<f32> = (0..nnz).map(|i| 0.3 + i as f32 * 0.05).collect();

    let ga_cpu: Tensor<NdB, 1> = Tensor::from_floats(ga_data.as_slice(), &nd_dev);
    let ga_cpu_prim = match ga_cpu.into_primitive() {
        TensorPrimitive::Float(p) => p, _ => unreachable!(),
    };
    let y_cpu_prim = assemble_backward_primitive::<NdB>(&pattern, ga_cpu_prim, &nd_dev, false);
    let y_cpu: Vec<f32> = Tensor::<NdB, 1>::from_primitive(TensorPrimitive::Float(y_cpu_prim))
        .to_data().to_vec().unwrap();

    let ga_cuda: Tensor<CudaB, 1> = Tensor::from_floats(ga_data.as_slice(), &cuda_dev);
    let ga_cuda_prim = match ga_cuda.into_primitive() {
        TensorPrimitive::Float(p) => p, _ => unreachable!(),
    };
    let y_cuda_prim = assemble_backward_primitive::<CudaB>(&pattern, ga_cuda_prim, &cuda_dev, true);
    let y_cuda: Vec<f32> = Tensor::<CudaB, 1>::from_primitive(TensorPrimitive::Float(y_cuda_prim))
        .to_data().to_vec().unwrap();

    let rel = vec_max_rel_diff(&y_cpu, &y_cuda);
    eprintln!("V8 assemble_backward: max_rel={rel}");
    assert!(rel < REL_TOL, "V8 assemble_backward: max_rel={rel} >= {REL_TOL}");
}
```

The `Wgpu` import in the use list is incidental — drop it if it triggers a
warning. The tests pin Cuda<f32, i32> and NdArray<f32> backends explicitly.

- [ ] **Step 2: Run V8**

```
cargo test --release --test sparse_cusparse_v8 -- --ignored --nocapture 2>&1 | tail -20
```

Expected: 3 PASS, each with `max_rel < 1e-5`. The `eprintln!` lines show the
actual rel diff for diagnosis.

**If V8 FAILS** with a divergence between CPU and CUDA (rel > 1e-5):
1. Pinpoint which site (spmv_forward / spmv_backward / assemble_backward)
   failed.
2. Print intermediate vectors at the failing site (q_at_cols, weighted)
   for both backends.
3. Most likely cause: descriptor setup mismatch (wrong dim, wrong index
   type, wrong base). Re-verify Task 3's setup.
4. STOP and report — don't proceed to V7 until V8 is green.

- [ ] **Step 3: Commit (if V8 passes)**

```bash
git add tests/sparse_cusparse_v8.rs
git commit -m "SP-9 Task 9: V8 SpMV CPU/CUDA bit-match

Three tests verify the new cuSPARSE SpMV path matches the CPU .scatter
path within 1e-5 rel on a small linear-chain CsrPattern: forward SpMV,
backward SpMV-transpose, and assemble-backward row-sum (with α=-1).

If V8 passes, the SP-9 SpMV swap is mathematically correct on the
CUDA backend.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Re-run V7b (profile gate)

Now check the bottleneck actually moved.

**Files:**
- None modified — uses the existing `scripts/sp8_check_scatter.sh` +
  `tests/sp8_v7_profile.rs`.

- [ ] **Step 1: Run V7b**

```
cargo test --release --test sp8_v7_profile -- --ignored --nocapture 2>&1 | tail -20
```

Expected runtime: ~5 minutes (nsys profile + parse).

**Outcome interpretation:**
- `scatter_kernel percentage = N%` where N < 30 → V7b PASS, proceed to Task 11.
- `scatter_kernel percentage` not found (kernel doesn't appear in stats) →
  V7b PASS trivially, proceed.
- N >= 30 → V7b FAIL. Diagnose: re-read `$HOME/nsys_out/sp8_v7b_stats.txt`,
  see what the new top kernel is. If `scatter_kernel` shrunk but
  `scatter_nd_kernel` grew, cuSPARSE may be using internal scatter — that's
  a cuSPARSE algorithm issue. Try `CUSPARSE_SPMV_ALG2` or `CUSPARSE_SPMV_ALG3`.
  STOP and report.

- [ ] **Step 2: Capture the result**

If V7b passes: write the scatter percentage and the new top-kernel ranking
to a short status note. If it fails: same but document the failure mode.

No commit yet — V7b doesn't add files. The status note is for the Task 11
close-out commit.

---

### Task 11: Re-run V7a (perf gate) + close

The final perf gate. After this either SP-9 succeeds end-to-end or we close
as partial.

**Files:**
- Modify: `.claude/ARCHITECTURE.md`

- [ ] **Step 1: Run V7a**

```
cargo test --release --test sp8_v7_perf -- --ignored --nocapture 2>&1 | tail -20
```

Expected runtime: ~25-30 minutes (4 runs × 2 backends × ~3 min each).

**Outcome interpretation:**
- `ratio = X.XXX where X < 0.7` → V7a PASS. Both gates green; SP-9 fully closes.
- `0.7 < ratio <= 0.85` → V7a FAIL but close to threshold. Document the
  partial win.
- `ratio > 0.85` → V7a FAIL with significant gap. Document and recommend
  CUDA Graphs / kernel fusion follow-up.

- [ ] **Step 2: Update ARCHITECTURE.md**

Read `.claude/ARCHITECTURE.md`. Locate the SP-8 section (added by SP-8 Task 7).
Append an SP-9 close-out section:

```markdown
## SP-9 cuSPARSE SpMV (2026-05-22, <status>)

Three `Tensor::scatter(0, ..., IndexingUpdateOp::Add)` call sites in
`src/sparse/mod.rs` were the source of the 78%-of-GPU `scatter_kernel`
hotspot. SP-9 replaced them with `cusparseSpMV` calls via two new
`cusparseSpMatDescr_t` descriptors on `CudaPatternCache`:

- `sp_mat_spmv` (n × n, values = adj) — sites 1 + 2 (forward y=N·q,
  backward gq=N^T·gi via TRANSPOSE op).
- `sp_mat_rowsum` (n × nnz, values = adj) — site 3 (`gc = -sp_mat_rowsum · gA`
  with α=-1, embeddedly negating).

**Outcome:**
- V7a (cuda/cpu ratio ≤ 0.7): <result>
- V7b (scatter_kernel < 30% of GPU time): <result>
- V8 (SpMV CPU/CUDA bit-match): green at 1e-5 rel for all 3 sites.

CPU (NdArray) path keeps using `Tensor::scatter` unchanged. The dispatch
in `src/sparse/dispatch.rs` routes between the two paths per the engine's
`cfg.params.sparse_solver` flag (SP-6).
```

Replace `<status>` with `fully landed` (both gates green) or `partial` (V7b
green, V7a not green) or `inconclusive` (V7b not green). Fill in the V7a
ratio and V7b scatter % from Steps 1 + Task 10 Step 2.

- [ ] **Step 3: Commit ARCHITECTURE update**

```bash
git add -f .claude/ARCHITECTURE.md
git commit -m "SP-9 close: cuSPARSE SpMV landed (<one-line status>)

V7a: <ratio>
V7b: <scatter %>
V8: green (1e-5 rel CPU/CUDA bit-match for all 3 sites).

V1, V5, gradcheck all still green on NdArray.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Replace placeholders with real numbers from Tasks 10 + 11.

If both V7 gates passed:
```bash
git tag sp9-cusparse-spmv-landed
```

- [ ] **Step 4: Final sanity sweep**

Run all regression gates one more time:

```
cargo test --release --test training_verification v1_loss_matches 2>&1 | tail -3
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -3
cargo test --release --test sp8_gradcheck 2>&1 | tail -3
cargo test --release --test sparse_cusparse_v8 -- --ignored --nocapture 2>&1 | tail -10
```

Each must report PASS. No new clippy warnings on the SP-9 surface:

```
cargo clippy --lib 2>&1 | grep -E "(cusparse|dispatch|sparse/mod)" | head -5
```

Should be empty.

---

## Self-Review

### Spec coverage

| Spec section | Task |
|---|---|
| Persist d_adj_values in CudaPatternCache | 1 |
| Declare 5 new SpMV fields | 2 |
| Build sp_mat_spmv + sp_mat_rowsum + workspaces | 3 |
| cusparse_spmv_forward (site 1) | 4 |
| cusparse_spmv_backward (site 2) | 5 |
| cusparse_assemble_backward (site 3) | 6 |
| cusparseSpMV_preprocess (optional perf) | 7 |
| Dispatch entries + thin helpers + mmc_op plumbing | 8 |
| V8 SpMV CPU/CUDA bit-match | 9 |
| V7b profile gate | 10 |
| V7a perf gate + ARCHITECTURE.md + close | 11 |

### Placeholder scan

- The `<status>`, `<result>`, `<ratio>`, `<scatter %>` placeholders in Task 11
  Step 2 + Step 3 are intentional — the implementer fills them in from the
  Task 10 + 11 Step 1 outputs.
- The `// existing destroy calls follow ...` comment in Task 2 Step 1 is a
  reference to read-then-extend the existing Drop body. Implementer reads
  the current code first.
- The `// verbatim copy of the existing ... body` comments in Task 8 Step 2
  show the full helper bodies inline — they're not "implement later"
  placeholders, just `cpu_*` rename of the existing logic.

### Type consistency

- `cusparseSpMatDescr_t`, `cusparseSpSVDescr_t`, `cusparseDnVecDescr_t` are
  raw pointer types from cudarc 0.19 — Task 1 Step 1 verifies their names.
- `burn_cubecl::cubecl::server::Handle` is the persistent device-buffer type
  (same as SP-7).
- `use_cuda: bool` parameter added to three primitive helpers + three dispatch
  entries — consistent across the SP-9 surface.
- `pattern: &Arc<CsrPattern>` (Arc reference, not plain reference) — matches
  the existing SP-6 `triangular_csr_solve` signature in `mmc.rs:182`.
- `cache.n` and `cache.nnz` fields on `CudaPatternCache` — Task 4/6
  conditionally add them if missing. Implementer verifies on first read.

### Risk recheck

- Task 1: cudarc surface check is non-skippable. If symbols missing → STOP.
- Task 3: cuSPARSE matrix-descriptor pointers reference cache-owned Handles.
  Lifetime correctness depends on Drop order being correct (Step 1 sets
  it up).
- Task 8: signature change on three public-in-crate functions. All callers
  in `mmc_op.rs` must be updated; Step 3 enumerates them.
- Task 9: V8 must pass before V7. Don't run V7 if V8 fails — the answer
  would be uninterpretable.
- Task 11: V7a may miss even with V7b green. Plan handles that path
  (partial close).

---

## Execution choice

Plan complete and saved to `.claude/specs/2026-05-22-sp9-cusparse-spmv-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage
   spec-then-quality review. Same workflow as SP-1..SP-8.
2. **Inline Execution** — batch with checkpoints.

Which approach?
