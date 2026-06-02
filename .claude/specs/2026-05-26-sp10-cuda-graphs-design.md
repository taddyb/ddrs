# SP-10: CUDA Graphs for per-timestep launch-overhead collapse

**Date:** 2026-05-26
**Status:** Design — awaiting plan
**Predecessors:** SP-7 (cubecl-cuda backend), SP-8 (MC timestep fusion), SP-9 (cuSPARSE SpMV)

## Motivation

SP-9 eliminated the `scatter_kernel_t_f32_i_i32` hotspot (77.5% → 0.0% of GPU
time) and brought the CUDA path 8% faster than CPU (`cuda_median = 3.99 min` vs
`cpu_median = 4.34 min` on the 9-batch smoke train). But the V7a target
(ratio ≤ 0.7) missed at 0.919. The remaining bottleneck is on the CPU side, not
the GPU.

The SP-9 V7b nsys profile (3-batch smoke) shows:

| API call            | Time   | Count    | Avg     |
|---------------------|--------|----------|---------|
| `cuLaunchKernel`    | 17.5 s | 7,684,365 | 2.28 μs |
| `cuEventSynchronize`| 2.24 s |   225,449 | —       |
| `cuCtxSetCurrent`   | 1.39 s | 12,454,497|   111 ns|
| `cuMemcpyHtoDAsync_v2` | 1.27 s | 412,887 | —      |

Total GPU compute is only ~13 s. CPU spends **more time issuing kernels
(17.5 s) than the GPU spends executing them (13 s)** — the GPU is starved
waiting for the host to enqueue work. The fix textbook-matches the diagnosis:
CUDA Graphs.

## Goal

Capture one forward and one backward CUDA Graph per `MuskingumCunge` instance.
Replay them per timestep. CPU issues 1 `cuGraphLaunch` per direction instead
of ~100 `cuLaunchKernel`s.

## Success criteria (gates)

| Gate | Condition | Verification |
|------|-----------|--------------|
| V1   | DDR sandbox absolute match (existing) | `examples/compare_ddr_sandbox` |
| V5   | Gradient correctness (existing)       | `tests/sparse_gradcheck.rs` |
| V8   | SpMV CPU/CUDA bit-match (SP-9)        | `tests/sparse_cusparse_v8.rs` |
| V7b  | scatter_kernel < 30% of GPU time      | `scripts/sp8_check_scatter.sh` (trivially) |
| **V9** *(new)* | Graph-replay output bit-matches direct-launch output. Forward Q_{t+1} and all 5 input gradients: `max_rel = 0`. | `tests/sp10_graph_bitmatch.rs` |
| V7a  | cuda/cpu wall-time ratio ≤ 0.7        | `tests/sp10_v7a_perf.rs` (rewrite of SP-9 V7a) |
| **V10** *(new)* | `cuLaunchKernel` count drops ≥ 80% in 3-batch nsys profile (≤ 1.54M calls vs SP-9's 7.68M). | `scripts/sp10_check_launches.sh` |

V9 and V10 are structural gates: they verify the mechanism fired even if V7a
is muddied by data-loading or autograd overhead outside the GPU path.

## Non-goals

- No changes to CPU path (NdArray backend untouched).
- No changes to autograd math or analytical chain in `mmc_op.rs`.
- No changes to SP-6 SpSV or SP-9 SpMV implementations.
- No precision shifts (f32 throughout).
- No multi-stream / async-batch.

## Architecture

### Module layout

```
src/cuda_graph/                NEW
├── mod.rs                     CudaGraph wrapper (newtype around CUgraphExec)
├── capture.rs                 cuStreamBeginCapture/EndCapture + instantiate
└── scratch.rs                 PersistentScratch buffer set per MC instance

src/routing/mmc.rs             route_timestep grows a graph-replay branch
src/routing/mmc_op.rs          TimestepOp::backward grows a graph-replay branch
src/sparse/cusparse.rs         CudaPatternCache gains graph_fwd, graph_bwd, scratch
src/config.rs                  params.use_cuda_graphs: bool (default false on land)
```

### Persistent scratch (`PersistentScratch`)

Allocated once per `MuskingumCunge` instance inside `setup_inputs`, lives on
`CudaPatternCache`, freed when the cache drops. All via `cuMemAllocAsync`.

| Buffer | Shape | Direction |
|--------|-------|-----------|
| `in_q` | `[n]` | forward in (Q_t) |
| `in_qp` | `[n]` | forward in (q'_t) |
| `out_q` | `[n]` | forward out (Q_{t+1}) |
| `state_*` × 24 | `[n]` each | forward out / backward in (24 saved intermediates) |
| `in_grad_q_next` | `[n]` | backward in |
| `out_grad_n` | `[n]` | backward out (gradient w.r.t. spatial n) |
| `out_grad_q_spatial` | `[n]` | backward out |
| `out_grad_p_spatial` | `[n]` | backward out |
| `out_grad_q_t` | `[n]` | backward out |
| `out_grad_q_prime_t` | `[n]` | backward out |

Total ~33 × n × 4 bytes ≈ 660 KB per gauge subgraph (n=5K), ~45 MB for full
CONUS (n=346,321). Trivial.

### Forward capture

Eager, in `setup_inputs`, after the existing SP-6/SP-9 cuSPARSE descriptor
build. Uses cubecl's primary CUDA stream (the one SP-9 already reaches via
`cusparseSetStream`).

```rust
fn capture_fwd_graph(cache, scratch):
    let stream = cubecl_stream(device);
    cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    timestep_forward_inner(
        scratch.in_q, scratch.in_qp,
        cache.params,                        // already on device
        cache.adj_values,                    // SP-9 persistent
        scratch.out_q,
        scratch.state_*,                     // 24 outputs
    );
    let graph_template = cuStreamEndCapture(stream);
    let graph_exec = cuGraphInstantiate(graph_template);
    cuGraphDestroy(graph_template);          // keep only the exec
    cache.graph_fwd = Some(CudaGraph(graph_exec));
```

cubecl kernel launches and cuSPARSE SpMV/SpSV calls on the captured stream are
absorbed as graph nodes automatically.

### Forward replay (per timestep)

```rust
fn route_timestep(Q_t, q_prime_t):
    if let Some(graph_fwd) = &cache.graph_fwd {
        cuMemcpyDtoDAsync(scratch.in_q,  Q_t.handle,       n*4, stream);
        cuMemcpyDtoDAsync(scratch.in_qp, q_prime_t.handle, n*4, stream);

        cuGraphLaunch(graph_fwd, stream);              // ONE launch

        let Q_next = fresh_burn_primitive(n);
        cuMemcpyDtoDAsync(Q_next.handle, scratch.out_q, n*4, stream);

        let state = TimestepState {
            depth: fresh; cuMemcpyDtoDAsync(_.handle, scratch.state_depth, n*4, stream),
            // ... × 24 ...
        };

        register_timestep_op(Q_next, state);
        return Q_next;
    } else {
        // SP-9 direct-launch path
    }
```

### Backward capture

Eager, in `setup_inputs`, right after forward capture. Synthesize **random**
sample inputs (not zeros — autograd may short-circuit on zeros) to drive one
backward and capture its kernel sequence. Discard outputs.

```rust
fn capture_bwd_graph(cache, scratch):
    populate(scratch.in_grad_q_next, scratch.state_*, with random f32);
    cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    timestep_backward_inner(
        scratch.in_grad_q_next,
        scratch.state_*,                                // 24 saved-state inputs
        cache.params,
        cache.adj_values,
        scratch.out_grad_n,
        scratch.out_grad_q_spatial,
        scratch.out_grad_p_spatial,
        scratch.out_grad_q_t,
        scratch.out_grad_q_prime_t,
    );
    let graph_template = cuStreamEndCapture(stream);
    let graph_exec = cuGraphInstantiate(graph_template);
    cuGraphDestroy(graph_template);
    cache.graph_bwd = Some(CudaGraph(graph_exec));
```

### Backward replay

```rust
impl Backward<B, 5> for TimestepOp {
    fn backward(&self, state: TimestepState, grad_q_next: B::FloatTensorPrimitive):
        if let Some(graph_bwd) = &cache.graph_bwd {
            cuMemcpyDtoDAsync(scratch.in_grad_q_next, grad_q_next, n*4, stream);
            cuMemcpyDtoDAsync(scratch.state_depth, state.depth, n*4, stream);
            // ... × 24 ...

            cuGraphLaunch(graph_bwd, stream);          // ONE launch

            let grad_n         = fresh; cuMemcpyDtoDAsync(_, scratch.out_grad_n,         n*4, stream);
            let grad_q_spatial = fresh; cuMemcpyDtoDAsync(_, scratch.out_grad_q_spatial, n*4, stream);
            let grad_p_spatial = fresh; cuMemcpyDtoDAsync(_, scratch.out_grad_p_spatial, n*4, stream);
            let grad_q_t       = fresh; cuMemcpyDtoDAsync(_, scratch.out_grad_q_t,       n*4, stream);
            let grad_q_prime_t = fresh; cuMemcpyDtoDAsync(_, scratch.out_grad_q_prime_t, n*4, stream);

            return (grad_n, grad_q_spatial, grad_p_spatial, grad_q_t, grad_q_prime_t);
        } else {
            // SP-9 direct-launch backward
        }
}
```

### Per-step launch accounting

| Path           | `cuLaunchKernel` | `cuMemcpyDtoDAsync` |
|----------------|------------------|---------------------|
| SP-9 (current) | ~100 fwd + ~100 bwd = ~200 | 0 |
| SP-10 (graph)  | 1 fwd + 1 bwd = 2 | 27 fwd + 30 bwd = 57 |

`cuLaunchKernel` drop per step: **~99%**. Total API issuance time per step:
`200 × 2.28 μs = 456 μs` → `2 × ~5 μs + 57 × ~2.5 μs ≈ 152 μs`. ~67% CPU-side
savings per step, multiplied by tens-of-thousands of steps per train.

### Data flow

```
┌─────────────────────────────────────────────────────────────────┐
│ MuskingumCunge::setup_inputs   (once per batch)                 │
│                                                                 │
│   alloc PersistentScratch                                       │
│   build SP-6 SpSV descriptors                                   │
│   build SP-9 SpMV descriptors                                   │
│   capture forward  -> graph_fwd : CUgraphExec                   │
│   capture backward -> graph_bwd : CUgraphExec                   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ route_timestep   (T_hours × batches × … times)                  │
│                                                                 │
│   Q_t, q'_t ──D2D──▶ scratch.in_*                               │
│                          │                                      │
│                          ▼                                      │
│                   cuGraphLaunch(graph_fwd)                      │
│                          │                                      │
│                          ▼                                      │
│   scratch.out_q, scratch.state_*  ──D2D──▶ fresh BURN primitives│
│   register TimestepOp(state)                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ TimestepOp::backward   (T_hours times via autograd tape)        │
│                                                                 │
│   grad_q_next, state.* ──D2D──▶ scratch.in_grad_*, scratch.state_*│
│                          │                                      │
│                          ▼                                      │
│                   cuGraphLaunch(graph_bwd)                      │
│                          │                                      │
│                          ▼                                      │
│   scratch.out_grad_*  ──D2D──▶ 5 fresh BURN primitives          │
└─────────────────────────────────────────────────────────────────┘
```

## Fallback model

Stream capture may fail because of:
- Uncaptured driver call leaking into stream (host-roundtrip).
- cuSPARSE workspace allocated non-async during the captured region.
- BURN's autograd tape causing a host-side allocation we didn't predict.

Detection: `cuStreamEndCapture` returns `CUDA_ERROR_STREAM_CAPTURE_*`, or
`cuGraphInstantiate` returns a graph-instantiation error.

On any capture error:
1. `tracing::warn!` with the captured error and reason.
2. Set `cache.graph_fwd = None` (and `graph_bwd = None`).
3. Set `cache.capture_status = FallbackReason(error_string)` for observability.
4. All future `route_timestep` / `TimestepOp::backward` calls take the SP-9
   direct-launch path. No crash, no behavioral change.

V7a regresses to SP-9's 0.919, but V1/V5/V8 still hold.

## Concerns

1. **Pointer stability across replays.** Captured graphs operate on the GPU
   pointers visible at capture time. `PersistentScratch` holds those pointers
   steady; data movement is via D2D-async into/out of those buffers. Standard
   PyTorch-style idiom.

2. **cuSPARSE workspace allocations.** SP-9 allocates these once in
   `setup_inputs` *before* capture begins. The captured calls only invoke
   `cusparseSpMV` / `cusparseSpSV_solve` with already-allocated workspaces.
   Captured stream sees no allocations. ✓

3. **Capture-mode exclusivity.** Nothing else can use the captured stream
   during capture. `setup_inputs` is already a serialized "build all CUDA
   resources" point — the stream is quiescent. ✓

4. **BURN host-roundtrips.** Any debug print, shape-check via `into_data`, or
   synchronous read mid-timestep would trigger D2H sync and fail capture. The
   current `route_timestep` and `TimestepOp::backward` use pure primitive
   ops — no host-side reads — so capture should succeed. Worth one targeted
   grep before capture goes in.

5. **Recapture trigger.** If batch size or `n_active` changes between batches,
   the captured graphs are invalid. Detection: hash `(n_active, T_hours)` and
   compare against the cache's captured signature. Mismatch ⇒ destroy graphs
   (`cuGraphExecDestroy`) and recapture lazily on next `setup_inputs`.

6. **Zero-input autograd short-circuiting (backward capture).** If backward
   uses zero gradient inputs at capture time, BURN's autograd may elide a
   kernel that would otherwise fire on non-zero inputs. Mitigation: random
   f32 sample data during backward capture; the captured kernels are the
   same regardless of input *values*, only counts/shapes/addresses matter.

7. **V7a margin.** With ~22% of full-run wall time being `cuLaunchKernel` API
   time (extrapolated from V7b's 3-batch numbers), eliminating ~99% of those
   recovers ~20% wall time on CUDA — pushing the ratio from 0.919 toward
   ~0.74. **We may still miss the 0.7 gate.** SP-10 lands V9 + V10
   unconditionally as structural wins; V7a may need a follow-up SP (kernel
   fusion in cubecl) if it misses.

## Assumptions

- **`params.use_cuda_graphs` defaults to `false` on initial land.** Flipped to
  `true` only after V9, V10, V7a all pass. Mirrors SP-9's `sparse_solver:
  cuda` flip ritual (commit `dbcf6e6`).

- **One `MuskingumCunge` per (network, batch).** The training loop reuses a
  single instance for the rollout. Consistent with SP-7/SP-9 plumbing; worth
  a one-line confirmation in `bin/train.rs` during implementation.

- **f32 throughout.** No precision shifts. ✓

- **cubecl persistent handles survive cache lifetime.** SP-9 already relies
  on this for `d_adj_values`; `PersistentScratch` follows the same pattern.

- **BURN's autograd is value-independent in kernel dispatch.** TimestepOp's
  analytical chain emits the same kernels regardless of input values. Backward
  capture with random data captures the same sequence as backward with real
  data.

## Tests & rollout

### New tests

| File | Gate | Purpose |
|------|------|---------|
| `tests/sp10_graph_bitmatch.rs` | V9 | 5-reach RAPID sandbox: run one timestep with `use_cuda_graphs=false`, then `=true`. Both forward Q_{t+1} and all 5 input gradients must match bit-for-bit (`max_rel = 0`). |
| `tests/sp10_v7a_perf.rs` | V7a | Rewrite of SP-9's `sp8_v7_perf` with `use_cuda_graphs=true`. Median-of-3 measurements after 3 warmups. Pass: `cuda_median / cpu_median ≤ 0.7`. |
| `scripts/sp10_check_launches.sh` | V10 | Run nsys profile (3 mini-batches), parse `cuda_api_sum` for `cuLaunchKernel` Num Calls. Pass: `≤ 1.54M` (≥80% drop from SP-9's 7.68M). |

### Rollout sequence

1. Land all code with `use_cuda_graphs: false` default. All existing gates
   (V1, V5, V8, V7b) green.
2. Run V9 (graph-replay bit-match) — must show `max_rel = 0` for forward and
   all 5 backward gradients.
3. Run V10 (launch-count gate) — must show ≥ 80% drop.
4. Run V7a (wall-time ratio) — must show ≤ 0.7.
5. **Only if V9, V10, V7a pass:** flip default to `true` in
   `config/merit_training.yaml`. Single-line commit.
6. Update `.claude/ARCHITECTURE.md` SP-10 section with measured numbers.

### Observability

Add `cache.capture_status: CaptureStatus` enum with `Captured`,
`FallbackReason(String)` variants. Surface via `tracing::info!` at the end of
`setup_inputs` so we can grep training logs for silent fallbacks.

## Plan task outline (for `writing-plans` skill)

1. cudarc graph bindings: thin newtype wrapper around `CUgraphExec`,
   capture helpers, error mapping. New `src/cuda_graph/` module.
2. `PersistentScratch` struct + allocation in `setup_inputs`.
3. Wire scratch + graph fields onto `CudaPatternCache`; recapture-trigger hash.
4. Forward capture: `capture_fwd_graph()`, hook into `setup_inputs`.
5. Forward replay: branch in `route_timestep`, D2D plumbing.
6. Backward capture: `capture_bwd_graph()` with random sample inputs, hook
   into `setup_inputs`.
7. Backward replay: branch in `TimestepOp::backward`.
8. V9 test (`tests/sp10_graph_bitmatch.rs`).
9. V10 script (`scripts/sp10_check_launches.sh`).
10. V7a test rewrite (`tests/sp10_v7a_perf.rs`).
11. ARCHITECTURE.md update + default-flip commit (only if all gates green).
