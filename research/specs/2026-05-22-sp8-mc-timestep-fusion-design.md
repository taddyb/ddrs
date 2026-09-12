# SP-8: MC Timestep Fusion (kill the scatter hotspot)

**Status:** Draft, pending user review
**Date:** 2026-05-22
**Parent:** SP-7 (cusparse stream-share + zero-copy x wrap)
**Predecessor close-out:** V6 came in at 0.998× (parity, not the 0.7× target). nsys
profile of the SP-6/SP-7 training loop showed `scatter_kernel_t_f32_i_i32` at
**78.7% of GPU compute time** — 96,075 invocations averaging 311 μs each.
That hotspot is NOT in our cusparse path (our `cusparse_grada` is already
gather+multiply, not scatter). It's BURN's autograd gradient accumulation
across the MC engine's many per-timestep tensor ops.

## Goal

Fuse the MC engine's per-timestep forward (`route_timestep` in `src/routing/mmc.rs`)
into a single custom autodiff op with an analytical backward. The cusparse
triangular solve (SP-6) already has its own custom Backward; we extend that
pattern up one level. Collapsing the BURN-tensor-op chain eliminates
per-op gradient scatter, which should reduce `scatter_kernel` invocations
by roughly the count of intermediate ops × timesteps × backward passes.

## Verification: V7 (hard gate, BOTH must pass)

### V7a — Perf gate
- Re-run V6-style smoke benchmark (5 epochs × 3 mini-batches × `bin/train`).
- Median CUDA wall-time across 3 runs ≤ **0.7×** median CPU wall-time across 3 runs.
- Lives in `tests/sp8_v7_perf.rs`.

### V7b — Profile gate
- Re-run `nsys profile target/release/train --max-mini-batches 3` and parse the
  `cuda_gpu_kern_sum` report.
- `scatter_kernel_t_f32_i_i32` < **30%** of total GPU compute time.
- Lives in `scripts/sp8_check_scatter.sh` (returns nonzero on failure).
- Test wrapper: `tests/sp8_v7_profile.rs` that invokes the script and asserts on exit code.

Both gates must pass. If only V7a passes (speedup for an unrelated reason), the
fix is suspect and we re-investigate. If only V7b passes (scatter gone but no
speedup), we have a different bottleneck and re-scope.

## Architecture

### Before (today)

```text
route_timestep(...) -> Tensor<Autodiff<I>, 1>
  ├── geometry(...)               BURN ops × ~8       autograd nodes × 8
  ├── velocity_clamped(...)       BURN ops × 1        autograd node × 1
  ├── celerity = v_clamped * 5/3  BURN ops × 1        autograd node × 1
  ├── calculate_muskingum_coeffs  BURN ops × ~10      autograd nodes × 10
  ├── A_values_assembler.spmv     BURN ops × ~3       autograd nodes × 3
  ├── A_values_assembler.assemble BURN ops × ~3       autograd nodes × 3
  ├── b construction              BURN ops × ~6       autograd nodes × 6
  ├── triangular_csr_solve        custom Backward     ONE custom node
  └── clamp_min                   BURN ops × 1        autograd node × 1
                                  ───────             ─────────────────
                                  ~33 autograd nodes per timestep,
                                  each registering grads via scatter
```

With 89 timesteps × ~185 mini-batches × 5 epochs ≈ 82K backward calls; each
backward traverses 33 nodes; if each emits one scatter, that's ~2.7M scatters
expected. The profile shows 96K — so it's emitter consolidation by BURN
(scatters per output tensor, not per op), but the dominance is consistent.

### After (this spec)

```text
route_timestep(...) -> Tensor<Autodiff<I>, 1>
  └── TimestepOp::apply(...)      ONE custom Backward
        forward:
          all ~33 ops as BACKEND-LEVEL primitives (no autograd nodes)
        save:
          inputs (n, q_spatial, p_spatial, length, slope, x_storage, Q_t, q'_t)
          intermediates (depth, top_width, side_slope, bottom_width, R, v, c, k, c1..c4, A_values, b, x)
          pattern (Arc<CsrPattern>)
        backward:
          analytical ∂L/∂{Q_t, q'_t, n, q_spatial, p_spatial}
          via chain rule through the saved chain
                                  ───────             ─────────────────
                                  1 autograd node per timestep
                                  → ~1 scatter per timestep × backwards
```

The expected scatter count drops from ~33×N to ~1×N. That's the lever for the
78.7% → <30% gate.

## File layout

```
src/routing/
  mmc.rs                MODIFIED — route_timestep delegates to TimestepOp::apply
  mmc_op.rs             NEW — TimestepOp impl Backward<B, 1>, saved state, analytical backward
  utils.rs              UNCHANGED

src/sparse.rs           UNCHANGED — CsrSolveOp stays; we call it as a sub-op
src/sparse/cusparse.rs  UNCHANGED — cusparse_grada already optimal (gather+multiply)
src/geometry.rs         UNCHANGED — exposed as backend-level primitives for the fused op

tests/sp8_diagnosis.rs           NEW — instrumented run that emits scatter attribution table
tests/sp8_v7_perf.rs             NEW — V7a perf gate
tests/sp8_v7_profile.rs          NEW — V7b profile gate (wraps the bash script)
scripts/sp8_check_scatter.sh     NEW — nsys + parse + threshold check
```

## Conventions for this plan

- Custom op pattern mirrors `CsrSolveOp` in `src/sparse.rs:374-422`.
- All forward code generic over `B: Backend`. `TimestepOp` is generic.
- `gradcheck` against the original autograd path (NdArray, small synthetic
  network) is the math-correctness gate. Tolerance 1e-3 relative for f32.
- Cite line numbers in `src/routing/mmc.rs::route_timestep` as we replace each
  block so a future reader can map fused-op math back to the original
  decomposed source.
- No commit amends — new commit per task per SP-6/SP-7 precedent.
- Pre-existing clippy lints are out of scope; new code stays clean.

## Tasks (7 tasks)

### Task 1 — Diagnosis: confirm scatter root cause

Non-skippable. Authority to STOP the plan and force re-scope if findings
diverge from the working hypothesis.

**Deliverable:** `tests/sp8_diagnosis.rs` instrumented run + a short report in
`.claude/specs/2026-05-22-sp8-diagnosis-findings.md` containing:
1. The full disassembly comment from cubecl's `scatter_kernel_t_f32_i_i32`
   (read the cubecl source — it's a JIT'd kernel; we want the macro source it
   compiles from).
2. The list of BURN backward methods that emit this scatter, found by
   grepping `~/projects/burn/crates/burn-autodiff/src/ops/`.
3. Per-MC-timestep scatter attribution table: for each of the ~33 BURN ops in
   `route_timestep`, which one(s) emit scatter in their backward, with
   instance counts from a short run.

**Stop condition:** if the dominant scatter source is NOT per-op autograd
gradient accumulation in `route_timestep`, STOP and report. The rest of the
plan assumes a per-timestep fusion is the right surgery.

**Verification:** the report file exists, is committed, and links to specific
cubecl/burn source files with line numbers.

### Task 2 — TimestepOp boilerplate

Create `src/routing/mmc_op.rs` mirroring `CsrSolveOp`:

```rust
pub struct TimestepOp;

pub struct TimestepState<B: Backend> {
    pub pattern: Arc<CsrPattern>,
    // inputs (read by backward)
    pub n: B::FloatTensorPrimitive,
    pub q_spatial: B::FloatTensorPrimitive,
    pub p_spatial: B::FloatTensorPrimitive,
    pub length: B::FloatTensorPrimitive,
    pub slope: B::FloatTensorPrimitive,
    pub x_storage: B::FloatTensorPrimitive,
    pub q_t: B::FloatTensorPrimitive,
    pub q_prime_t: B::FloatTensorPrimitive,
    // intermediates (read by backward)
    pub depth: B::FloatTensorPrimitive,
    pub top_width: B::FloatTensorPrimitive,
    pub side_slope: B::FloatTensorPrimitive,
    pub bottom_width: B::FloatTensorPrimitive,
    pub hydraulic_radius: B::FloatTensorPrimitive,
    pub velocity: B::FloatTensorPrimitive,
    pub celerity: B::FloatTensorPrimitive,
    pub k_muskingum: B::FloatTensorPrimitive,
    pub c1: B::FloatTensorPrimitive,
    pub c2: B::FloatTensorPrimitive,
    pub c3: B::FloatTensorPrimitive,
    pub c4: B::FloatTensorPrimitive,
    pub a_values: B::FloatTensorPrimitive,
    pub b: B::FloatTensorPrimitive,
    pub x: B::FloatTensorPrimitive,
}

impl<B: Backend> Backward<B, 1> for TimestepOp {
    type State = TimestepState<B>;
    fn backward(self, ops: Ops<Self::State, ...>, grads: &mut Gradients, _cp: &mut Checkpointer) {
        todo!("Task 3")
    }
}
```

Saved-state primitive types (not autograd tensors) so the tape can't grow
through them. Five autograd-tracked parents: `n, q_spatial, p_spatial, q_t, q_prime_t`.
Five backward output gradients to register.

**Verification:** `cargo build --lib` clean. No tests yet — Task 3 fills the backward.

### Task 3 — Analytical backward implementation

The plan derives ∂L/∂{Q_t, q'_t, n, q_spatial, p_spatial} via chain rule
through the saved intermediates. The full derivation is in the plan doc, not
this spec — too dense for design review. The pattern mirrors `CsrSolveOp::backward`:
work backwards from `∂L/∂x` (the output gradient), unwinding each chain link.

**Math-correctness gate:** `gradcheck` against the original autograd path on a
small synthetic NdArray network (`tests/sp8_gradcheck.rs`). Numerical vs
analytical relative diff < 1e-3 for each of the five backward outputs.

**Verification:** gradcheck test passes. V1 / V2 / V4 / V5 all still pass
after Task 4 (which actually wires this in).

### Task 4 — Wire TimestepOp into route_timestep

Replace the BURN-op-chained body of `route_timestep` in `src/routing/mmc.rs`
with a single `TimestepOp::apply(...)` call that:

1. Unwraps autograd tensors to backend primitives at the boundary.
2. Computes the entire forward as backend primitives (no autograd nodes).
3. Saves all inputs + intermediates to `TimestepState`.
4. Returns a single autograd-tracked output via `prep.finish(state, x_prim)`.

V1 / V2 / V4 / V5 must still pass.

**Verification:** all four regression tests still green.

### Task 5 — V7a perf benchmark

Add `tests/sp8_v7_perf.rs` mirroring `tests/sparse_cusparse_v6.rs` but with the
hard gate restored: median CUDA wall ÷ median CPU wall ≤ 0.7. Three runs each
for stability.

**Verification:** test passes.

### Task 6 — V7b profile gate

Add `scripts/sp8_check_scatter.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
NSYS_DIR="${NSYS_DIR:-$HOME/nsys_out}"
mkdir -p "$NSYS_DIR"
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --output="$NSYS_DIR/sp8_profile" --force-overwrite=true \
    target/release/train --config config/merit_training.yaml \
                          --checkpoint-dir /tmp/sp8_ckpt \
                          --max-mini-batches 3
nsys stats "$NSYS_DIR/sp8_profile.nsys-rep" --report cuda_gpu_kern_sum \
    > "$NSYS_DIR/sp8_stats.txt"
PCT=$(awk '/scatter_kernel_t_f32_i_i32/ {gsub(",","",$1); print $1; exit}' "$NSYS_DIR/sp8_stats.txt")
echo "scatter_kernel percentage: $PCT"
awk -v p="$PCT" 'BEGIN { exit (p+0 < 30.0) ? 0 : 1 }'
```

Wrap in `tests/sp8_v7_profile.rs`:
```rust
#[test]
fn v7b_scatter_below_30_percent() {
    let status = std::process::Command::new("bash")
        .arg("scripts/sp8_check_scatter.sh")
        .status()
        .expect("nsys script");
    assert!(status.success(), "scatter % >= 30");
}
```

**Verification:** test passes.

### Task 7 — Close + commit

Tag the cleaned-up tip with `sp8-fusion-landed`. Update `.claude/ARCHITECTURE.md`
to note that `route_timestep` now wraps a fused custom op. Push to origin.

**Verification:** git tag exists; ARCHITECTURE.md updated; both V7 gates pass on
master tip.

## Concerns

1. **Analytical backward derivation is the hard part.** ~50 lines of chain-rule
   math in `TimestepOp::backward`. Mitigated by `gradcheck` as the math gate
   (Task 3). If it can't be derived cleanly (e.g., the `clamp` in
   `velocity_clamped` breaks differentiability somewhere), escalate as
   BLOCKED — we keep the autograd chain for that branch and fuse only the
   rest.
2. **Saved tape size.** 23 saved tensors × N reaches × f32 ≈ 6 MB per timestep
   × 89 timesteps × ~185 mini-batches (one batch at a time alive) = ~530 MB
   peak. Within consumer-GPU budget but watch nsys for spikes.
3. **Regression risk for V1/V2/V4/V5.** These are NdArray-pinned and exercise
   the engine math. A subtle derivation error will surface here before V7.
   We run them after Task 4 and refuse to proceed to Tasks 5-6 unless they're
   green.
4. **The diagnosis (Task 1) might find a different scatter source.** Stop
   condition is explicit; do NOT silently re-scope without user input.
5. **V7a 0.7× target lift is set without margin analysis.** If the real win
   only gets us to 0.85× we'll have correctness landed but no green test.
   Plan provision: if V7a misses but V7b passes, document and propose SP-9
   for the rest. Don't ship a broken hard gate.
6. **cubecl autotune cache freshness.** First V7a run will pay JIT cost. The
   median-of-three protocol absorbs that; first run is discarded.
7. **`gradcheck` infrastructure.** ddrs may not have a shared gradcheck
   helper. Task 3 may need a small new helper in `tests/common.rs`.
8. **CLAUDE.md invariant #2 (f32 throughout):** preserved. The backward
   derivation stays in f32. If precision becomes a problem, escalate before
   silently switching to f64.

## Open assumptions

1. The 78.7% scatter is dominated by per-op autograd registration in
   `route_timestep`. Task 1 verifies; if false, STOP.
2. The custom Backward pattern from `CsrSolveOp` works the same way one level
   up — same BURN `Backward<B, 1>` trait, same `Ops` mechanism. Confirmed by
   reading `src/sparse.rs:374-422`.
3. Fusing the timestep doesn't break SP-6/SP-7's cusparse path; the CSR solve
   is still called as a backend-level primitive inside `TimestepOp::forward`.
4. The V6 CPU baseline (5.59 min) is the comparable for V7a. Run on the same
   hardware, same config, median of 3.
5. The user's GPU (consumer-grade) can hold the 530 MB peak tape comfortably
   alongside MLP + data + intermediates.

## What's deferred

- **Upstreaming the SP-7 vendor patches** (cubecl + burn forks) — tracked
  separately, not blocking SP-8.
- **MLP / Adam scatter elimination.** If V7b passes, MLP+Adam scatters land
  below 30% naturally. If not, follow-up work.
- **Multi-GPU / Wgpu backends.** SP-8 targets the Cuda backend only. NdArray
  + Wgpu paths use the existing autograd chain.

## Next steps

1. You review this spec.
2. After approval: write `.claude/specs/2026-05-22-sp8-mc-timestep-fusion-plan.md`
   with the full task-by-task plan (including the analytical backward
   derivation that this spec deferred).
3. Subagent-driven execution per SP-6/SP-7 precedent.
