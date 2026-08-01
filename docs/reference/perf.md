# Performance & CUDA Graphs

The GPU-vs-CPU performance story for ddrs is the arc from SP-7
(a working `cubecl-cuda` backend with no measurable speedup) to SP-10
(forward-only CUDA Graphs giving CUDA a 2.6× wall-time advantage on
the smoke train). The canonical number is the **V7a wall-time ratio**:
`cuda_wall_time / cpu_wall_time` measured on 3 mini-batches of the
standard training loop.

SP-8 sat at 1.000 (symmetric fusion win, no GPU advantage), SP-9
dropped it to 0.919 (cuSPARSE killed the scatter hotspot), and
**SP-10 landed at 0.385** — CUDA finishes in 1.96 min vs CPU's 5.09
min. Forward CUDA Graphs only; the backward path still runs on the
SP-9 direct-launch route, which is why V10 (cuLaunchKernel drop)
only reached 29.2% rather than the 40% target.

Two YAML toggles control the whole stack: `sparse_solver: cuda|cpu`
and `use_cuda_graphs: true|false`.

## The journey

| Spike | Date | Change | Key result |
|---|---|---|---|
| SP-7 | early | Wire `burn_cubecl::cubecl-cuda` as a backend behind the same `Backend` generic. Patched cubecl fork adds `stream_accessor` so ddrs can submit work on a thread-bound stream. | CUDA backend runs end-to-end; no V7a measurement yet. |
| SP-8 | 2026-05-22 | Collapse the per-timestep dataflow into a single fused autograd op (`mmc_op::timestep_forward`, `src/routing/mmc_op.rs`). One autograd node per step instead of ~33. Saved-state struct holds the 24 intermediates the analytical backward needs. | Wall-time 5.58 → 4.06 min on both backends (27% — symmetric). V7a = **1.000** — fusion benefit was Rust-side graph collapse, shared by CPU + GPU. V7b = **77.5%** — `scatter_kernel_t_f32_i_i32` still dominated GPU time. |
| SP-9 | 2026-05-22 → 2026-05-26 | Replace three `Tensor::scatter(0, ..., IndexingUpdateOp::Add)` call sites in `src/sparse/mod.rs` with `cusparseSpMV` (two descriptors: `sp_mat_spmv` for forward/backward y=N·q, `sp_mat_rowsum` for the backward `gc = -N·gA`). Dispatch in `src/sparse/dispatch.rs` selects per `cfg.params.sparse_solver`. | V8 bit-match: GREEN (max_rel = 0). V7b: GREEN — scatter hotspot is **0.0%** of GPU time. V7a: PARTIAL — **0.919** (CUDA 8% faster). New floor: ~8M `cuLaunchKernel` calls at ~2.3 μs each, dominated by `kernel_binop_*` ops. |
| SP-10 | 2026-05-29 | Capture the forward per-timestep kernel sequence into a `CUgraphExec` once during `setup_inputs`; replay via one `cuGraphLaunch` per timestep. Required three fused `#[cube]` kernels (K1, K2, K3) plus a `PersistentScratch` buffer pool, plus a `flush_no_sync` patch to the cubecl fork. | V9 bit-match: recorded GREEN (ABSOLUTE MATCH under `DDRS_FORCE_GRAPHS=1`) — but see the V9 caveat under [Gates](#gates): that command exercises the CUDA *backend*, not graph capture. V10: PARTIAL — **29.2%** launch drop (5,442,735 vs SP-9's 7,684,365); below 40% target because backward still direct-launches. V7a: GREEN — **0.385**. 2.4× improvement over SP-9. |

Full prose in `.claude/ARCHITECTURE.md` sections "SP-8 fused MC
timestep", "SP-9 cuSPARSE SpMV", and "SP-10 CUDA Graphs" —
particularly the SP-10 "7-layer architectural journey" subsection,
which catalogues every wall the naive capture approach hit
(host-sync inside `flush`, re-entrant `exclusive_with_server`,
transient cubecl-pool allocations baked into the graph,
pinning-was-the-bug, cuSPARSE-internal-allocations as false positive,
fused `#[cube]` kernels as the real fix, per-batch pool fragmentation).

## Toggles

Two booleans under `params:` in the training YAML control the entire
perf stack — but **the Rust defaults and the shipped-YAML values are
opposites**, so it matters a great deal whether a config sets the keys at
all:

| Key | Rust default (key omitted) | `config/merit_training.yaml` |
|---|---|---|
| `sparse_solver` | **`cpu`** — `SparseSolver::#[default] Cpu`, `src/config.rs:407-410` | `cuda` (line 138) |
| `use_cuda_graphs` | **`false`** — `src/config.rs:454` | `true` (line 141) |

The fast path is therefore **opt-in**. A config that omits both keys runs
the CPU sparse solver with no graph capture, silently and without
warning — which is exactly what the DDR sandbox does (see
[Comparing to DDR](ddr-comparison.md)). The values that pass the gates
today, as shipped in `config/merit_training.yaml`:

```yaml
params:
  sparse_solver: cuda    # YAML value since commit dbcf6e6 (SP-9 close)
  use_cuda_graphs: true  # YAML value since the SP-10 close commit (e35af29)
```

To force the CPU-only baseline (e.g. for debugging or a fairness
comparison) — also what you get by leaving both keys out:

```yaml
params:
  sparse_solver: cpu
  use_cuda_graphs: false
```

`sparse_solver: cuda` selects the `cusparseSpMV`-backed path; `cpu`
keeps the `Tensor::scatter(0, ..., Add)` path. `use_cuda_graphs: true`
activates the SP-10 capture path on top of the cuda sparse solver —
it has no effect unless `sparse_solver: cuda` is also set, because
the captured kernel sequence is the SP-9 kernels with K1/K2/K3 fused
around the cuSPARSE calls.

## Capture architecture

The SP-10 captured forward region is six kernel launches per
timestep, captured once during `setup_inputs` and replayed via
`cuGraphLaunch`:

```
K1  →  cuSPARSE SpMV (y = N·q_t)  →  K2  →  assemble_kernel  →
cuSPARSE SpSV (solve L·q_new = b)  →  K3
```

Each of K1, K2, K3 is a fused `#[cube]` kernel in
`src/cuda_graph/geometry_kernel.rs`, mirroring the corresponding
stages of `forward_chain_inner` in `src/routing/mmc_op.rs`
line-by-line:

- **K1 — `forward_k1_kernel` (S1..S23).** Geometry + Muskingum
  coefficients. Eight tensor inputs — `n`, `q_spatial`, `p_spatial`,
  `q_t`, `q_prime_t` (accepted but unused: `c4` multiplies it in S25,
  not here), `length`, `slope`, `x_storage` — plus four `F` scalars
  (`bottom_width_lb`, `depth_lb`, `velocity_lb`, `dt`). Outputs: 19
  buffers of shape `[n_segments]`, in `forward_saved_idx` order **minus
  `{14 A_VALUES, 15 B_RHS, 16 I_T, 17 X_SOL}`, which are produced later**
  (`src/cuda_graph/geometry_kernel.rs:197-218`): `depth`, `top_width`,
  `side_slope`, `bottom_width`, `hydraulic_radius`,
  `velocity_unclamped`, `velocity_clamped`, `celerity`, `k_muskingum`,
  `denom`, `c1`, `c2`, `c3`, `c4`, `ratio`, `denominator`, `q_eps`,
  `side_slope_raw`, `bw_raw`. In particular `a_values` is **not** a K1
  output — `assemble_kernel` (S26) produces it, and it is the one
  `[nnz]`-sized saved-state buffer. One thread per segment (rank-1, n
  threads). Every intermediate (alpha_1, alpha_2, hydraulic radius, etc.)
  stays in GPU registers — zero cubecl-pool slot allocations inside
  the captured region.
- **K2 — `b_rhs_kernel` (S25).** Builds the RHS for the triangular
  solve: `b = c2·(N·q_t) + c3·q_t + c4·q'`. Single launch,
  register-resident.
- **K3 — `q_clamp_kernel` (S28).** Post-solve clamp `max(q_new,
  q_eps)`, writing into the per-timestep `out_q` handle.

The two cuSPARSE calls (`SpMV` for S24, `SpSV` for S27) sit between
them and capture cleanly because cuSPARSE 12.x takes
externally-managed workspace buffers at `bufferSize`/`analysis`
time — the per-call routine itself is allocation-free.
`assemble_kernel` (S26, `src/cuda_graph/geometry_kernel.rs:382-408`) is
the fourth `#[cube]` kernel in the captured region. It builds the sparse
matrix values themselves — `a_values[k] = diag_mask[k] − c[row[k]] ·
adj[k]`, i.e. `A = I − c·N` — one thread per nnz, writing straight into
the caller-owned `state_a_values` handle. (It mirrors `assemble_primitive`
in `src/sparse/mod.rs:642`; it does **not** touch the RHS `b`, which is
K2's job.)

The reason the fused-kernel design was load-bearing: **the cubecl
memory allocator recycles pool slots between successive tensor ops.**
A captured `cuMalloc` address becomes a fixed pointer in the replay;
if any subsequent allocation reuses that slot, the replay reads or
writes garbage. Pinning handles to disable recycling was the obvious
"fix" — and was the actual bug in spike #4
(`CUDA_ERROR_ILLEGAL_ADDRESS` on replay even with no other
allocations running, because the allocator reshuffles pinned handles
internally). Moving all intermediates into K1/K2/K3 registers
sidesteps the pool entirely for the captured region.

## PersistentScratch

`src/cuda_graph/scratch.rs` defines `PersistentScratch`: **39** cubecl
`Handle`s (`src/cuda_graph/scratch.rs:14-71`) pre-allocated once per
`MuskingumCunge` instance during `setup_inputs` and held for the lifetime
of the `CudaPatternCache`. Layout:

- **3 forward I/O** — `in_q`, `in_qp`, `out_q`.
- **6 static-input mirrors** (SP-10 Phase 3) — `in_n`, `in_qsp`,
  `in_psp`, `in_length`, `in_slope`, `in_xst`. Constant per training
  batch, populated once via D2D copy from caller primitives *before*
  `cuStreamBeginCapture` so the captured K1 reads from these stable
  handles. Required because the caller's source primitives' device
  pointers can move between batches.
- **1 pattern buffer** — `pattern_diag_mask` (nnz f32), persistent
  device upload of `pattern.diag_mask`, read by `assemble_kernel`.
- **23 saved-state buffers** — `state_depth`, `state_top_width`, …
  `state_bw_raw` (`:40-62`). Outputs of forward K1/K2/K3 *and* inputs to
  the analytical backward. The forward writes these and the backward
  reads them across the timestep loop. All are `[n_segments]` f32 except
  `state_a_values`, which is `[nnz]`.
- **6 backward I/O** (`:64-70`) — one input, `in_grad_q_next`, and five
  outputs, `out_grad_n`, `out_grad_q_spatial`, `out_grad_p_spatial`,
  `out_grad_q_t`, `out_grad_q_prime_t`. These are allocated even though
  the backward is not itself captured (SP-11 open work, below).

That is 3 + 6 + 1 + 23 + 6 = **39** handles: 37 sized `[n_segments]` f32
and 2 sized `[nnz]` f32 (`state_a_values` and `pattern_diag_mask`).

Counting only the n-sized buffers, total memory is **~37n × 4 bytes** —
**~740 KB** for an n=5K gauge subgraph and **~51 MB** for full CONUS at
n=346,321, plus `2 × nnz × 4` bytes for the two pattern-sized buffers.
Pointers are stable for the lifetime of the cache, which is exactly the
property the captured graph requires.

> The module docstring at `src/cuda_graph/scratch.rs:4-7` still says
> "32 Handles total … ~32n × 4 bytes (~525 KB … ~44 MB)". That count
> predates the six Phase-3 static-input mirrors and `pattern_diag_mask`;
> the field list immediately below it is authoritative. (The old
> "~525 KB" was never self-consistent either — 32 × 5,000 × 4 is 640 KB.)

## Gates

Three gates protect the perf path:

| Gate | Tests | Threshold | Most recent |
|---|---|---|---|
| **V1** | `examples/compare_ddr_sandbox` (also under `DDRS_FORCE_GRAPHS=1`) | max abs diff < 1e-3 m³/s | ABSOLUTE MATCH (1.5e-5 m³/s) |
| **V7a** | `tests/sp10_v7a_perf.rs` (CUDA-with-graphs vs CPU; median of 3 timed runs after warmup) | `cuda_wall / cpu_wall ≤ 0.7` | **0.385** |
| **V10** | `scripts/sp10_check_launches.sh` (nsys `cuda_api_sum` row for `cuLaunchKernel`) | `(1 - calls/SP9_BASELINE) ≥ 40%` | **29.2%** — partial, capped by backward direct-launch |

V1 is the absolute correctness invariant from `CLAUDE.md`. It was reused
unchanged as **V9**, recorded at the SP-10 close as "graph vs no-graph
bit-match: GREEN" on the strength of
`DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox`.

> **That V9 framing does not hold up, and the recorded result is weaker
> than the label.** `DDRS_FORCE_GRAPHS` only selects the `Cuda<f32, i32>`
> inner backend (`examples/compare_ddr_sandbox.rs:113-117`); capture
> additionally requires `use_cuda_graphs && sparse_solver ==
> SparseSolver::Cuda` (`src/routing/mmc.rs:289-294`), and the sandbox
> `Config` is `Config::default()` (`src/sandbox.rs:88-89`) — `false` and
> `Cpu` — with `fixtures/sandbox/config.csv` setting neither. So the
> ABSOLUTE MATCH recorded for V9 certifies the **CUDA backend against the
> CPU backend with the CPU sparse solver**, not the graph-capture path.
> The capture path has no standing bit-match gate. Treat "graphs are
> bit-identical to no-graphs" as **unverified by this command**; a real
> V9 needs a sandbox config (or a harness) that sets both toggles. See
> [Comparing to DDR](ddr-comparison.md) for the full chain of evidence.

V7a is what answers "is GPU actually worth using?" and the V7a test
pre-warms the JIT with one discarded run before taking the median of
three.
V10 measures the mechanical objective of CUDA Graphs — collapsing
host-side launch count — and exposes the gap left by the un-captured
backward.

## Open work

- **Backward graph capture (SP-11 candidate).** The bit-equivalent
  backward path would need its own fused kernels for the gradient
  versions of K1, K2, K3. The autograd tape currently dispatches each
  backward stage as individual `kernel_binop_*` ops. Closing this
  would push V10 above 50% and is the only plausible route to V7a
  below 0.3.
- **cubecl fork patches.** All on `taddyb/cubecl` branch
  `ddrs-release`:
  - `stream_accessor` (SP-7) — exposes the thread-bound cubecl
    stream so ddrs can submit graph work on the same stream the JIT
    compiles into.
  - `exclusive_with_server` (SP-9) — re-entrant safe acquisition of
    the cubecl server context, needed to bind the thread once before
    capture.
  - `flush_no_sync` (SP-10, commit `d562ab99`) — flush variant that
    submits queued work without the `cuEventSynchronize` that would
    invalidate a stream mid-capture.

## Gotchas

1. **Capture is invoked once per batch in `setup_inputs`, not once
   per instance.** Gauge subgraphs make `n_active` vary between
   batches, so the `PersistentScratch` allocation and graph capture
   both rebuild when the active subnetwork changes. Within a batch,
   every timestep replays the same captured graph.
2. **`PersistentScratch` sizing depends on `n_active` and `nnz`.**
   Sized at allocation time, fixed for the cache lifetime. A batch
   that needs a larger subgraph drops the cache and re-allocates;
   never resize in place.
3. **`client.memory_cleanup()` MUST be called after each
   `optimizer.step()`** at end-of-batch. The first multi-batch run
   grew the cubecl persistent pool monotonically until OOM at batch
   4, because gradient buffers leaked into the pool while the graph
   held handle slots that the pool refused to reclaim. The cleanup
   forces a hard pool reset.
4. **`flush_no_sync` is load-bearing inside the capture region.**
   Using stock `flush()` instead triggers `cuEventSynchronize`,
   which invalidates stream capture and produces an opaque CUDA
   error on the next `cuStreamEndCapture`. If the cubecl fork is
   updated, re-verify that the patched flush variant is still
   exported.
5. **JIT warm-up + double-flush is required for CONUS-scale runs.**
   Without it, the first replay segfaults on a buffer that the
   warm-up path recycled. See the SP-10 "Per-batch pool fragmentation"
   subsection of `.claude/ARCHITECTURE.md`.
6. **`use_cuda_graphs: true` with `sparse_solver: cpu` is a silent
   no-op.** The captured kernel sequence assumes the cuSPARSE path;
   the CPU sparse solver has nothing to capture. No error is raised
   — the run just degrades to SP-8 fusion only.
7. **cuSPARSE dispatches into the thread-current CUDA context — on
   multi-GPU hosts that context can belong to the wrong device.**
   cubecl only sets the calling thread's context when it first
   *creates* a client for a device; afterwards command execution
   re-binds inside cubecl's server, not on the caller. Once a second
   device has been touched in the process, a cuSPARSE call can run
   under device A's context with device B's buffers/stream and die
   with an integer divide-by-zero (SIGFPE) inside
   `cusparseSpSV_solve`. `ensure_cuda_cache` therefore calls
   `bind_primary_context` (src/sparse/cusparse.rs) on every call —
   all raw-cuSPARSE entry points pass through it. Graph
   capture/replay bind the context themselves. Never add a cuSPARSE
   call path that bypasses `ensure_cuda_cache` without binding the
   target device's primary context first. Regression test:
   `tests/device_selection.rs` (needs ≥2 GPUs; skips otherwise).

## Verification

```bash
# V1 — correctness floor; both the CPU backend and the CUDA backend pass.
# (Neither run enables graph capture — see the V9 caveat under Gates.)
cargo run --release --example compare_ddr_sandbox
DDRS_FORCE_GRAPHS=1 cargo run --release --example compare_ddr_sandbox

# V7a — wall-time ratio gate (CUDA with graphs vs CPU)
cargo test --release --test sp10_v7a_perf -- --ignored --nocapture

# V10 — cuLaunchKernel drop gate (requires nsys in PATH)
bash scripts/sp10_check_launches.sh
```

After any change under `src/cuda_graph/`, `src/sparse/`, or
`src/routing/`, run V1 first (correctness gate) and V7a second
(performance gate). V10 is slower and only needs running when
investigating launch-count regressions or the SP-11 backward-capture
work.

## See also

- [Architecture](../architecture.md) — module map and per-timestep
  dataflow that K1/K2/K3 mirror.
- [Algorithm](../algorithm.md) — the math K1 implements (S1..S23).
- [BURN autograd recipe](burn-autograd.md) — the custom Backward for
  the sparse triangular solve that sits next to the captured region.
- [Comparing to DDR](ddr-comparison.md) — V1 details, reused as V9.
- [Formatting inputs](../usage/inputs-formatting.md) — the
  `sparse_solver` and `use_cuda_graphs` keys this chapter references.
