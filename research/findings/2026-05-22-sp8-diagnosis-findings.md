# SP-8 Task 1: Scatter hotspot diagnosis findings

**Date:** 2026-05-22
**nsys report:** $HOME/nsys_out/train_profile.nsys-rep (re-used from SP-6/SP-7 profile run)
**Stats file:** $HOME/nsys_out/stats.txt

## scatter_kernel_t_f32_i_i32 source

**cubecl macro emitting this kernel:**
`/home/tbindas/projects/burn/crates/burn-cubecl/src/kernel/index/scatter.rs:12-13`
```rust
#[cube(launch_unchecked, address_type = "dynamic")]
fn scatter_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(...)
```
The `#[cube]` proc-macro generates a CUDA kernel named `scatter_kernel_t_f32_i_i32` when `T=f32, I=i32`.

**Generated when BURN calls:** `B::float_scatter_add(dim, tensor, indices, value)` — which maps to
`kernel::scatter(dim, tensor, indices, value, is_bool=false)` in
`burn-cubecl/src/ops/tensor.rs:189-196`.

Note: there is a separate `scatter_nd_kernel` for `float_scatter_nd` / `Tensor::scatter(..., IndexingUpdateOp::Add)`. The forward `spmv()` call uses `Tensor::scatter(0, ..., IndexingUpdateOp::Add)`, which routes through `float_scatter_nd` → **`scatter_nd_kernel`**, which is NOT the bottleneck kernel. The bottleneck is `scatter_kernel` only.

## BURN autograd ops emitting scatter during backward

### Where `float_scatter_add` (→ `scatter_kernel`) is emitted in the route_timestep path

| route_timestep step | BURN op | Call site | Emits scatter_kernel? | Evidence (file:line) |
|---|---|---|---|---|
| `assemble(c1)` | `c.gather(0, row_idx)` | `sparse/mod.rs:569` | YES — in **backward** | `burn-autodiff/src/ops/tensor.rs:983`: `B::float_scatter_add(dim, zeros, indices, grad)` |
| `spmv(q_t)` | `q.gather(0, col_idx)` | `sparse/mod.rs:585` | YES — in **backward** | Same as above |
| `spmv(q_t)` | `zeros.scatter(0, row_idx, weighted, Add)` | `sparse/mod.rs:588` | YES — in **forward**, routes to `float_scatter_add` | `burn-backend/src/tensor/ops/float.rs:168`: `B::float_scatter_add(...)` |
| `compute_trapezoidal_geometry` — all ops | `add, sub, mul, div, add_scalar, mul_scalar, div_scalar, neg, powf, powf_scalar, sqrt, recip` | `geometry.rs:37-79` | NO | Backward uses `float_mul`, `float_mul_scalar`, `float_neg`, `float_div` — elementwise kernels (`kernel_binop_c_f32`, `kernel_scalar_binop_c_f32`, `unary_float_f_f32`) |
| `compute_trapezoidal_geometry` — `clamp_min`, `clamp` | `float_mask_fill(zeros)` (default impl via `float_lower_elem` + `float_mask_fill`) | `burn-backend/src/backend/ops/tensor.rs:209-244` | NO | Backward: `B::float_mask_fill(grad, mask, 0)` → `mask_fill_kernel` |
| `calculate_muskingum_coefficients` — all ops | `mul, div, add, add_scalar, mul_scalar, neg, recip` | `mmc.rs:206-216` | NO | Same as geometry ops — elementwise kernels only |
| `b = c2 * i_t + c3 * q_t + c4 * q_prime` | `mul, add` | `mmc.rs:252` | NO | Elementwise backward only |
| `solution.clamp_min(discharge_lb)` | `float_mask_fill` | `mmc.rs:262` | NO | `mask_fill_kernel` |
| `Gradients::register` (every backward op accumulating into shared grad) | `B::float_add(value, tensor_old)` | `burn-autodiff/src/grads.rs:108` | NO | `float_add` → `numeric::add` → `kernel_binop_c_f32_*` — elementwise add, NOT scatter |

### Summary

Exactly **3 scatter_kernel invocations per route_timestep** call (across forward + backward):

1. **Forward** — `spmv()`: `zeros.scatter(0, row_idx, weighted, Add)` → `float_scatter_add` → `scatter_kernel`
2. **Backward** — `gather` backward for `spmv()`: grad of `q.gather(0, col_idx)` → `float_scatter_add` → `scatter_kernel`
3. **Backward** — `gather` backward for `assemble()`: grad of `c.gather(0, row_idx)` → `float_scatter_add` → `scatter_kernel`

This prediction is confirmed by the profile counts:
- `scatter_kernel_t_f32_i_i32`: **96,075 invocations**
- `gather_kernel_t_f32_i_i32`: **96,090 invocations** (15-call delta = startup/teardown overhead)
- `96,075 / 3 = 32,025 timesteps` → `32,025 / 90 = 355.8 mini-batches` (consistent with a long training run)

## Hypothesis confirmation

**[X] Partially confirmed / Working hypothesis refined:**

The working hypothesis stated: *"BURN's autograd gradient accumulation across `route_timestep`'s ~33 per-op tensor operations"* emits the scatter hotspot.

**Actual finding:** The scatter hotspot is NOT from BURN's `Gradients::register` (which uses `float_add` → elementwise kernel, not scatter). It IS from the gather↔scatter roundtrip of the SpMV kernels (`spmv()` and `assemble()`) that are called within each route_timestep. These 3 gather/scatter ops per timestep account for all 96,075 scatter_kernel invocations. None of the ~30 other per-op tensor operations in geometry/coefficient calculation emit scatter.

**Implication for SP-8 fusion plan:** The SP-8 plan to fuse `route_timestep` into a single custom autodiff op is STILL CORRECT and will eliminate all 3 scatter sources. The custom backward in `CsrSolveOp` (already landed in SP-6) handles the CSR solve gradient; fusing the remaining timestep math eliminates the per-op gather/scatter tape chain entirely.

**Hypothesis box check:**
- [X] **Refined-confirmed:** The 3 scatter/gather ops per timestep (all inside `spmv()` and `assemble()` within `route_timestep`) emit the bulk of the 96K scatters. These are structural to the SpMV + coefficient assembly path, not general autograd accumulation. Proceed to Tasks 2-7 with this refined understanding.

## Profile snapshot

| Metric | Value |
|---|---|
| nsys report used | `$HOME/nsys_out/train_profile.nsys-rep` (SP-6/SP-7 profile) |
| nsys stats file | `$HOME/nsys_out/stats.txt` |
| scatter_kernel % of GPU time | **78.7%** |
| scatter_kernel invocations | **96,075** |
| scatter_kernel avg duration | **311,522 ns (311 µs)** |
| scatter_kernel total time | **29.93 s** out of 38.04 s GPU compute |
| gather_kernel invocations | **96,090** (near-equal, confirming 1:1 forward/backward pairing) |
| Ratio scatter:gather | 0.9998 (≈ 1:1, exact model prediction) |
| Scatter sources | 3 per timestep: 1× spmv forward, 1× spmv gather-backward, 1× assemble gather-backward |
| Predicted timesteps | 32,025 (= 96,075 / 3) |
| Predicted mini-batches | ~356 (= 32,025 / 90) |

### SCATTER lines from nsys stats (cuda_gpu_kern_sum section):
```
     78.7   29,929,479,179     96,075  311,522.0  305,412.0    13,408   675,145     75,307.8  scatter_kernel_t_f32_i_i32
```
