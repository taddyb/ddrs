# ddrs — BURN port architecture

Reference: `~/projects/ddr/src/ddr/routing/` (Python/PyTorch).
Algorithm reference: `~/projects/ddr/CLAUDE.md`.

## Per-timestep dataflow

```
            spatial parameters [0,1] from NN
                       │
            denormalize(value, bounds, log_space?)
                       │
              n, q_spatial, p_spatial  (physical units)
                       │
                       ▼
        compute_trapezoidal_geometry(n, p, q, Q_t, slope)
            │
            ├── depth = ((Q·n·(q+1)) / (p·√s))^(3/(5+3q))
            ├── top_width = p · depth^q
            ├── side_slope = clamp(TW·q / (2·d), 0.5, 50)
            ├── bottom_width = clamp(TW − 2·ss·d, btm_lb)
            ├── R = ((TW+BW)·d/2) / (BW + 2·d·√(1+ss²))
            └── v = (1/n) · R^(2/3) · √s
                       │
                       ▼
           celerity c = clamp(v, v_lb, 15) · 5/3
                       │
                       ▼
            k = length / c     (per reach)
            denom = 2k(1−x) + dt           (dt = 3600 s)
            c1 = (dt − 2kx)/denom
            c2 = (dt + 2kx)/denom
            c3 = (2k(1−x) − dt)/denom
            c4 = 2·dt / denom
                       │
                       ▼
          A = I − c1·N      (lower triangular)
          b = c2·(N·Q_t) + c3·Q_t + c4·q'
                       │
                       ▼
        triangular_solve_lower(A, b)     ← forward substitution
                       │
                       ▼
          Q_{t+1} = clamp(x, discharge_lb)
```

## Cold start (hot-start at t=0)

```
(I − N) · Q_0 = q'_0
        │
        ▼   linear-chain network → Q_0[i] = Σ_{j ≤ i} q'_0[j]   (cumulative sum)
```

## Why dense forward substitution, not sparse CSR + custom autograd

DDR's PyTorch path wraps SciPy/CuPy `spsolve_triangular` in a custom
`torch.autograd.Function` that hand-rolls `∇A = -gradb[rows]·x[cols]`. We chose
not to replicate that in BURN because:

* BURN 0.21's `Backward`/`Ops` plumbing is in flux and version-pinned wiring is
  fragile.
* Forward substitution over a topologically sorted adjacency is `O(n²)` worst
  case but the test suite never exceeds 100 reaches.
* Every step is a plain BURN tensor op, so autograd is automatic — no `unsafe`,
  no custom-`Backward` boilerplate.

Sparse + custom backward is a perf pass for later; the public API
(`MuskingumCunge::forward`) stays unchanged.

## Module map

| File | Mirrors (in ~/projects/ddr) | Purpose |
|---|---|---|
| `src/config.rs` | `validation/configs.py` (Params subset) | Parameter ranges, attribute minimums |
| `src/geometry.rs` | `geometry/trapezoidal.py` | Trapezoidal channel geometry |
| `src/routing/utils.rs` | `routing/utils.py` | `denormalize`, `triangular_solve_lower`, `compute_hotstart_discharge` |
| `src/routing/mmc.rs` | `routing/mmc.py` | `MuskingumCunge` engine |
| `tests/geometry.rs` | — (Python tests via mmc only) | Geometry sanity + gradients |
| `tests/routing_utils.rs` | `tests/routing/test_routing_utils.py` | Denormalize + triangular solve |
| `tests/mmc.rs` | `tests/routing/test_mmc.py` | Hotstart, coefficients, forward, autodiff |

## SP-8 fused MC timestep (2026-05-22, partial)

`MuskingumCunge::route_timestep` is a thin wrapper around
`mmc_op::timestep_forward` (`src/routing/mmc_op.rs`). The forward chain
runs at the backend-primitive level — no autograd nodes — and the
saved-state struct holds all 24 intermediates the analytical backward
needs. One autograd node per timestep instead of ~33.

**Outcome:** wall-time dropped from 5.58 → 4.06 min on the smoke train
(27% improvement, both backends). V1/V5/gradcheck all green.

**Did NOT meet either V7 gate:**
- V7a (cuda/cpu ratio ≤ 0.7): **ratio = 1.000**. Fusion sped up BOTH
  backends symmetrically because the win is autograd-graph collapse
  (a Rust-side cost shared by CPU + GPU), not GPU-specific.
- V7b (scatter_kernel < 30% of GPU time): **77.5%**. The primitive
  helpers in `src/sparse/mod.rs` (`spmv_primitive`, `assemble_*_primitive`)
  still use `Tensor::scatter(0, ..., IndexingUpdateOp::Add)`, which
  lowers to `scatter_kernel_t_f32_i_i32` — the exact kernel the
  diagnosis named. The fusion moved scatters from
  autograd-gather-backward to explicit-scatter-in-primitive, net zero.

**SP-9 (next):** replace `.scatter(..., Add)` in the CSR primitive helpers
with either `cusparseSpMV` or a no-atomic warp-reduction kernel. That's
the remaining unlock on the V7 gates.

## SP-9 cuSPARSE SpMV (2026-05-22, partial)

Three `Tensor::scatter(0, ..., IndexingUpdateOp::Add)` call sites in
`src/sparse/mod.rs` (`spmv_primitive`, `spmv_backward_primitive`,
`assemble_backward_primitive`) were the source of the 77.5%-of-GPU
`scatter_kernel` hotspot identified in SP-8. SP-9 replaced them with
`cusparseSpMV` calls via two new `cusparseSpMatDescr_t` descriptors on
`CudaPatternCache`:

- `sp_mat_spmv` (n × n, values = adj) — sites 1 + 2 (forward y=N·q,
  backward gq=N^T·gi via TRANSPOSE op).
- `sp_mat_rowsum` (n × nnz, values = adj) — site 3
  (`gc = -sp_mat_rowsum · gA` with α=-1, negation embedded in SpMV).

CPU (NdArray) path keeps `Tensor::scatter` unchanged. Dispatch in
`src/sparse/dispatch.rs` routes between paths per `cfg.params.sparse_solver`.

**Outcome:**
- V8 (SpMV CPU/CUDA bit-match): GREEN — `max_rel = 0` exact match for all
  3 sites on the linear-chain test pattern.
- V7b (scatter_kernel < 30% of GPU time): **GREEN — 0.0%** (down from
  77.5%). The `scatter_kernel_t_f32_i_i32` is GONE from the kernel
  profile. New top kernels are `cusparse::spsm_v2_kernel` (31%, SP-6's
  triangular solve, unchanged) and the various `kernel_binop_*` /
  `kernel_scalar_binop_*` families.
- V7a (cuda/cpu ratio ≤ 0.7): **PARTIAL — ratio = 0.919**. CUDA is now
  21 sec / 8% faster than CPU on the smoke train (4.34 → 3.99 min), but
  the 30% target was missed. The cuSPARSE SpMV win was capped at the
  fraction of wall time scatter_kernel had been consuming
  (~29.7 sec on a ~5 min run = ~10%, matching the observed 8% drop).

**The remaining wall-time floor** is launch overhead: 8M+
`cuLaunchKernel` calls at ~2.3 μs each (per SP-8's nsys profile), spread
across millions of small (~1 μs) `kernel_binop_*` ops. CPU and CUDA pay
this roughly equally, hence the small relative ratio improvement
despite the big absolute win on scatter.

**SP-10 candidate**: CUDA Graphs or cubecl kernel fusion to attack the
launch-overhead surface. Either would help GPU disproportionately
(small per-op kernels are exactly where launch overhead dominates).

## Deferred from the Python original

These exist in DDR but are not load-bearing for the MC solver itself and were
left out of the harness:

* `flow_scale` multiplier on `q_prime` (test_flow_scaling.py)
* Observed top_width / side_slope override (`_apply_data_override`)
* Gauge-subset scatter output (`output_indices` / `_flat_indices`)
* `tau` boundary trimming
* KAN parameterization (separate module — out of scope here)
* CUDA backend (drop in `Wgpu`/`CudaJit` later by swapping the backend generic)
