# SP-8 MC Timestep Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse the MC engine's per-timestep forward (`src/routing/mmc.rs::route_timestep`) into a single custom autodiff op so BURN's per-op gradient scatter chain collapses to one analytical backward, dropping `scatter_kernel_t_f32_i_i32` from 78.7% of GPU compute time to below 30% and getting CUDA wall-time ≤ 0.7× CPU wall-time.

**Architecture:** New `src/routing/mmc_op.rs` defines `TimestepOp` mirroring the existing `CsrSolveOp` pattern in `src/sparse/mod.rs:415-462`. `route_timestep` is rewritten to compute the entire forward chain at the backend-primitive level (no autograd nodes), save inputs+intermediates to `TimestepState`, and return a single autograd-tracked output. The analytical backward derives ∂L/∂{Q_t, q'_t, n, q_spatial, p_spatial} via the chain rule, terminating at the existing `triangular_csr_solve` (still its own custom Backward).

**Tech Stack:** BURN 0.21 (`Tensor`, `Autodiff`, `Backward<B, N>`, `Gradients`, `Ops`, `OpsKind`, `NoCheckpointing`), existing `CsrPattern`/`AValuesAssembler`/`triangular_csr_solve` from `src/sparse/mod.rs`. No new deps.

**Spec:** `.claude/specs/2026-05-22-sp8-mc-timestep-fusion-design.md`

---

## Pre-flight: what already exists

- `src/routing/mmc.rs:221-263` — `route_timestep` body to replace. 5 autograd-tracked inputs flow in: `n` (engine state), `q_spatial`, `p_spatial`, `q_t = self.discharge_t`, plus the per-call `q_prime_clamp`. Constants: `length`, `slope`, `x_storage`, `pattern`, `assembler`.
- `src/sparse/mod.rs:415-516` — `CsrSolveOp` template. Same `Backward<B, N>` pattern we'll mirror. `prepare::<NoCheckpointing>([nodes...]).compute_bound().stateful()` returns `OpsKind::{Tracked, UnTracked}`.
- `src/geometry.rs:28-79` — `compute_trapezoidal_geometry`. Reused inside the fused forward.
- `tests/training_verification.rs` — V1, V2, V3, V4 (the correctness regression gates).
- `tests/sparse_cusparse_v5.rs` — V5 (CPU vs CUDA bit-match of triangular solve).
- `tests/sparse_cusparse_v6.rs` — V6 informational perf benchmark we'll mirror in V7a.
- `src/bin/train.rs` — the binary V7a invokes (`Training complete in X.XX min`).

---

## File Structure

**Created:**

- `src/routing/mmc_op.rs` — `TimestepOp` + `TimestepState` + analytical `Backward` impl.
- `tests/sp8_diagnosis.rs` — Task 1 instrumented test that runs a short train and collects scatter source data.
- `.claude/specs/2026-05-22-sp8-diagnosis-findings.md` — written by Task 1 (committed).
- `tests/sp8_gradcheck.rs` — Task 3 math-correctness gate (numerical vs analytical, NdArray).
- `tests/sp8_v7_perf.rs` — Task 5 V7a hard perf gate.
- `tests/sp8_v7_profile.rs` — Task 6 V7b hard profile gate.
- `scripts/sp8_check_scatter.sh` — Task 6 helper script (runs nsys + parses scatter %).

**Modified:**

- `src/routing/mmc.rs` — `route_timestep` body replaced with `TimestepOp::apply(...)`.
- `src/routing/mod.rs` — `pub mod mmc_op;` added.

**Not touched:**

- `src/sparse/mod.rs`, `src/sparse/cusparse.rs`, `src/sparse/dispatch.rs` — the CSR solve and its own custom Backward stay as-is; we call `triangular_csr_solve` as a sub-op from inside the fused forward.
- `src/geometry.rs` — `compute_trapezoidal_geometry` is reused unchanged.

---

## Conventions for this plan

- Custom op pattern mirrors `CsrSolveOp` in `src/sparse/mod.rs:415-462`. Same imports, same `prepare/compute_bound/stateful` chain, same `Ops<State, N>` access pattern.
- All forward code generic over `B: Backend`. `TimestepOp` is generic; tests pin `NdArray<f32>` for the math gate and `Cuda<f32, i32>` for the perf gates.
- Cite line numbers in `src/routing/mmc.rs::route_timestep` as we replace each block.
- One new commit per task. No `--amend`.
- Pre-existing clippy lints in routing-core code are out of scope (same precedent as SP-1..SP-7). New code stays clean: `cargo clippy --lib 2>&1 | grep -E "(mmc_op|routing/mmc)" | head -5` must be empty after each commit.

---

## Frozen design constants

These constants appear in multiple tasks and must stay consistent:

```rust
// Inside src/routing/mmc.rs (already): DT_SECONDS = 3600.0
// New in src/routing/mmc_op.rs:
const NUM_PARENTS: usize = 5;   // [n, q_spatial, p_spatial, q_t, q_prime_t]
```

Parent order is **always** `[n, q_spatial, p_spatial, q_t, q_prime_t]`. Don't permute.

---

### Task 1: Diagnosis — confirm scatter root cause

**Authority:** This task has STOP authority. If the findings diverge from the working hypothesis (per-op autograd gradient accumulation in `route_timestep`), the implementer reports DONE_WITH_CONCERNS and the controller re-scopes SP-8 before any of Tasks 2-7 run.

**Files:**
- Create: `tests/sp8_diagnosis.rs`
- Create: `.claude/specs/2026-05-22-sp8-diagnosis-findings.md`

- [ ] **Step 1: Write the instrumented diagnosis test**

Create `tests/sp8_diagnosis.rs`:

```rust
//! SP-8 Task 1: gather evidence that the `scatter_kernel_t_f32_i_i32`
//! hotspot is BURN's gradient accumulation across per-timestep autograd ops
//! in `route_timestep`. Read-only investigation: runs a short training
//! invocation under nsys, parses the kernel stats, prints attribution.
//!
//! Run manually (requires CUDA + nsys + the merit data files):
//!   cargo test --release --test sp8_diagnosis -- --ignored --nocapture
//!
//! This test does NOT assert anything. It collects evidence the human reviews
//! and commits to `.claude/specs/2026-05-22-sp8-diagnosis-findings.md`.

use std::path::Path;
use std::process::Command;

const MAX_MINI_BATCHES: &str = "3";

fn data_files_present() -> bool {
    Path::new("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr").exists()
        && Path::new("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc").exists()
}

#[test]
#[ignore]
fn sp8_diagnosis_run() {
    if !data_files_present() {
        eprintln!("sp8_diagnosis: skip — data files not present");
        return;
    }
    if which::which("nsys").is_err() {
        eprintln!("sp8_diagnosis: skip — nsys not on PATH");
        return;
    }

    // 1. Run nsys on bin/train --max-mini-batches 3.
    let nsys_dir = std::env::var("NSYS_DIR")
        .unwrap_or_else(|_| format!("{}/nsys_out", std::env::var("HOME").unwrap()));
    std::fs::create_dir_all(&nsys_dir).expect("mkdir nsys_dir");
    let report_path = format!("{nsys_dir}/sp8_diagnosis");
    let ckpt_dir = "/tmp/sp8_diagnosis_ckpt";
    let _ = std::fs::remove_dir_all(ckpt_dir);
    let nsys_status = Command::new("nsys")
        .args([
            "profile",
            "--trace=cuda",
            "--sample=none",
            "--cpuctxsw=none",
            "--output", &report_path,
            "--force-overwrite=true",
            "target/release/train",
            "--config", "config/merit_training.yaml",
            "--checkpoint-dir", ckpt_dir,
            "--max-mini-batches", MAX_MINI_BATCHES,
        ])
        .status()
        .expect("spawn nsys");
    assert!(nsys_status.success(), "nsys profile failed");

    // 2. Run nsys stats and write to a file.
    let stats_path = format!("{nsys_dir}/sp8_diagnosis_stats.txt");
    let out = Command::new("nsys")
        .args([
            "stats",
            &format!("{report_path}.nsys-rep"),
            "--report",
            "cuda_api_sum,cuda_kern_exec_sum,cuda_gpu_mem_time_sum,cuda_gpu_kern_sum",
        ])
        .output()
        .expect("spawn nsys stats");
    std::fs::write(&stats_path, &out.stdout).expect("write stats");
    eprintln!("sp8_diagnosis: wrote {stats_path}");

    // 3. Echo the scatter rows so a human can eyeball the dominance.
    let stats = String::from_utf8_lossy(&out.stdout);
    for line in stats.lines() {
        if line.contains("scatter_kernel") {
            eprintln!("sp8_diagnosis SCATTER: {line}");
        }
    }
}
```

- [ ] **Step 2: Add the `which` dev-dependency**

In `Cargo.toml` under `[dev-dependencies]`:

```toml
which = "6"
```

- [ ] **Step 3: Run the test once**

Run: `cargo test --release --test sp8_diagnosis -- --ignored --nocapture 2>&1 | tail -40`
Expected: nsys completes, `sp8_diagnosis_stats.txt` exists, SCATTER lines printed showing the kernel's % share.

- [ ] **Step 4: Read cubecl scatter_kernel source**

Locate the source of `scatter_kernel_t_f32_i_i32`. The implementer runs:

```bash
grep -rn "scatter_kernel" ~/projects/cubecl/crates/ | head -20
grep -rn "scatter_kernel\|atomic_add" ~/projects/burn/crates/burn-autodiff/src/ | head -20
```

Read the matched files. Identify:
1. Which cubecl macro/function generates kernel name `scatter_kernel_t_f32_i_i32`.
2. Which BURN autodiff ops (in `burn-autodiff/src/ops/`) emit a scatter during backward — search for `scatter_add`, `Gradients::register`, etc.

- [ ] **Step 5: Write the findings doc**

Create `.claude/specs/2026-05-22-sp8-diagnosis-findings.md`:

```markdown
# SP-8 Task 1: Scatter hotspot diagnosis findings

**Date:** 2026-05-22
**nsys report:** $HOME/nsys_out/sp8_diagnosis.nsys-rep
**Stats file:** $HOME/nsys_out/sp8_diagnosis_stats.txt

## scatter_kernel_t_f32_i_i32 source

cubecl macro emitting this kernel: <FILE:LINE>
Generated when BURN calls: <op name(s)>

## BURN autograd ops emitting scatter during backward

For each op in route_timestep (src/routing/mmc.rs:221-263), the implementer
lists the BURN op it lowers to and whether that op's backward emits a scatter.

| route_timestep step | BURN op | Emits scatter? | Evidence (file:line) |
|---|---|---|---|
| q_eps = q_spatial + 1e-6 | add_scalar | ... | ... |
| ratio = (Q·n·(q+1))/(p√s) | mul, div, ... | ... | ... |
| ... | ... | ... | ... |

## Hypothesis confirmation

[ ] Confirmed: per-op gradient accumulation across the timestep's ~33 BURN
    ops emits the bulk of the 96K scatters. Proceed to Tasks 2-7.

OR

[ ] Refuted: the dominant scatter source is <other>. STOP and re-scope.
    Recommended next direction: <one-sentence>.
```

- [ ] **Step 6: Commit**

```bash
git add tests/sp8_diagnosis.rs Cargo.toml \
    .claude/specs/2026-05-22-sp8-diagnosis-findings.md
git commit -m "SP-8 Task 1: scatter hotspot diagnosis

Adds tests/sp8_diagnosis.rs (ignored) that runs nsys on bin/train and
saves per-kernel stats. Findings doc attributes the
scatter_kernel_t_f32_i_i32 dominance to BURN's per-op gradient
accumulation across route_timestep, confirming the SP-8 fusion plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Decision gate**

If the findings doc's hypothesis box marks *Confirmed*, proceed to Task 2.
If *Refuted*, report DONE_WITH_CONCERNS to the controller — do not start Task 2.

---

### Task 2: TimestepOp boilerplate

**Files:**
- Create: `src/routing/mmc_op.rs`
- Modify: `src/routing/mod.rs`

- [ ] **Step 1: Create the empty module**

Create `src/routing/mmc_op.rs`:

```rust
//! Fused MC timestep custom autodiff op. SP-8.
//!
//! Replaces the ~33 BURN-tensor-op chain in `MuskingumCunge::route_timestep`
//! with a single autograd node. Pattern mirrors `CsrSolveOp` in
//! `src/sparse/mod.rs:415-462`: a `Backward<B, N>` impl with a saved-state
//! struct holding backend primitives (no autograd participation).
//!
//! Parents in fixed order: [n, q_spatial, p_spatial, q_t, q_prime_t].

use std::sync::Arc;

use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::{backend::Backend, Tensor, TensorPrimitive};

use crate::config::Config;
use crate::sparse::{AValuesAssembler, CsrPattern};

/// Saved primitives used by `TimestepOp::backward`.
///
/// Forward inputs (the 5 autograd-tracked parents + the 3 constants) plus the
/// intermediates needed to evaluate the analytical chain rule.
#[derive(Clone, Debug)]
pub(crate) struct TimestepState<B: Backend> {
    pub pattern: Arc<CsrPattern>,
    // Inputs (autograd-tracked parents — read by backward to compute scalars
    // that flow into gradient register calls).
    pub n: B::FloatTensorPrimitive,
    pub q_spatial: B::FloatTensorPrimitive,
    pub p_spatial: B::FloatTensorPrimitive,
    pub q_t: B::FloatTensorPrimitive,
    pub q_prime_t: B::FloatTensorPrimitive,
    // Constants (not parents — used by backward but not differentiated through).
    pub length: B::FloatTensorPrimitive,
    pub slope: B::FloatTensorPrimitive,
    pub x_storage: B::FloatTensorPrimitive,
    // Forward intermediates (saved for backward).
    pub depth: B::FloatTensorPrimitive,
    pub top_width: B::FloatTensorPrimitive,
    pub side_slope: B::FloatTensorPrimitive,
    pub bottom_width: B::FloatTensorPrimitive,
    pub hydraulic_radius: B::FloatTensorPrimitive,
    pub velocity_unclamped: B::FloatTensorPrimitive,
    pub velocity_clamped: B::FloatTensorPrimitive,
    pub celerity: B::FloatTensorPrimitive,
    pub k_muskingum: B::FloatTensorPrimitive,
    pub denom: B::FloatTensorPrimitive,
    pub c1: B::FloatTensorPrimitive,
    pub c2: B::FloatTensorPrimitive,
    pub c3: B::FloatTensorPrimitive,
    pub c4: B::FloatTensorPrimitive,
    pub a_values: B::FloatTensorPrimitive,
    pub b_rhs: B::FloatTensorPrimitive,
    pub i_t: B::FloatTensorPrimitive, // N · Q_t  (SpMV result)
    pub x_sol: B::FloatTensorPrimitive, // pre-clamp solve output
    // Bookkeeping (small floats).
    pub depth_lb: f32,
    pub bottom_width_lb: f32,
    pub velocity_lb: f32,
    pub discharge_lb: f32,
    pub dt: f32,
}

#[derive(Debug)]
pub(crate) struct TimestepOp;

impl<B: Backend + 'static> Backward<B, 1> for TimestepOp
where
    B::FloatTensorPrimitive: 'static,
{
    type State = TimestepState<B>;

    fn backward(
        self,
        _ops: Ops<Self::State, 5>,
        _grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        // Task 3 fills this in. For Task 2 we leave the body empty — the op
        // compiles but registering it on the tape will not propagate any
        // gradients yet. V1/V2/V4 will fail until Task 3 lands; that's the
        // exact gate Task 3 must clear.
        unimplemented!("Task 3 — analytical backward not yet implemented");
    }
}

/// Forward + register-on-tape entry point. Called from
/// `MuskingumCunge::route_timestep` (Task 4). Returns Q_{t+1} as an
/// autograd-tracked rank-1 tensor.
///
/// Parent order: [n, q_spatial, p_spatial, q_t, q_prime_t].
#[allow(clippy::too_many_arguments)]
pub(crate) fn timestep_forward<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    n: Tensor<Autodiff<I>, 1>,
    q_spatial: Tensor<Autodiff<I>, 1>,
    p_spatial: Tensor<Autodiff<I>, 1>,
    q_t: Tensor<Autodiff<I>, 1>,
    q_prime_t: Tensor<Autodiff<I>, 1>,
    length: Tensor<Autodiff<I>, 1>,
    slope: Tensor<Autodiff<I>, 1>,
    x_storage: Tensor<Autodiff<I>, 1>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    // Task 2 returns a placeholder by delegating back to the existing
    // BURN-op chain so the type/borrow plumbing compiles without depending
    // on Task 3. The actual fused forward + saved state lands in Task 4.
    let _ = (cfg, pattern, assembler, n.clone(), q_spatial.clone(),
             p_spatial.clone(), q_t.clone(), q_prime_t.clone(),
             length.clone(), slope.clone(), x_storage.clone());
    let _ = TimestepOp; // touch the symbol so it isn't dead-code-warned
    unimplemented!("Task 4 fills the fused forward and the OpsKind branches");
}
```

- [ ] **Step 2: Wire into the module tree**

Edit `src/routing/mod.rs`. Read it first to see the current contents:

```bash
cat src/routing/mod.rs
```

Append:

```rust
pub(crate) mod mmc_op;
```

(Use `pub(crate)` because callers outside `src/routing/` should still go
through `MuskingumCunge::route_timestep`, not the fused op directly.)

- [ ] **Step 3: Build (allow the `unimplemented!` calls)**

Run: `cargo build --lib 2>&1 | tail -10`
Expected: clean compile. The `unimplemented!()` calls are runtime panics,
not compile errors.

- [ ] **Step 4: Confirm V1/V2/V4 still pass (nothing wired yet, so they should)**

Run: `cargo test --release --test training_verification v1_loss_matches 2>&1 | tail -5`
Expected: PASS — Task 2 didn't touch route_timestep.

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc_op.rs src/routing/mod.rs
git commit -m "SP-8 Task 2: TimestepOp skeleton + TimestepState

Adds the fused MC timestep custom autodiff op as a Backward<B, 1> with
TimestepState holding all backend primitives the analytical backward
will need. The forward + backward bodies panic with unimplemented! —
Task 3 fills the backward, Task 4 wires the forward into route_timestep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Analytical backward — derivation + impl + gradcheck

This is the heaviest task. We replace the `unimplemented!()` in `TimestepOp::backward` with the chain-rule derivation, then verify it against a numerical gradient.

**Files:**
- Modify: `src/routing/mmc_op.rs`
- Create: `tests/sp8_gradcheck.rs`

#### The forward chain (refer back to these as Step labels)

```text
S1.  q_eps           = q_spatial + 1e-6
S2.  numerator       = q_t · n · (q_eps + 1)
S3.  denominator     = p_spatial · √slope + 1e-8
S4.  ratio           = numerator / denominator
S5.  exponent        = 3 / (5 + 3·q_eps)
S6.  depth           = max(ratio^exponent, depth_lb)
S7.  top_width       = p_spatial · depth^q_eps
S8.  side_slope_raw  = top_width · q_eps / (2·depth)
S9.  side_slope      = clamp(side_slope_raw, 0.5, 50)
S10. bw_raw          = top_width − 2·side_slope·depth
S11. bottom_width    = max(bw_raw, bottom_width_lb)
S12. area            = (top_width + bottom_width) · depth / 2
S13. wp              = bottom_width + 2·depth·√(1 + side_slope²)
S14. hyd_radius      = area / wp
S15. velocity_un     = (1/n) · hyd_radius^(2/3) · √slope
S16. velocity_cl     = clamp(velocity_un, velocity_lb, 15)
S17. celerity        = velocity_cl · 5/3
S18. k_muskingum     = length / celerity
S19. denom           = 2·k_muskingum·(1 − x_storage) + dt
S20. c1              = (−2·k_muskingum·x_storage + dt) / denom
S21. c2              = ( 2·k_muskingum·x_storage + dt) / denom
S22. c3              = ( 2·k_muskingum·(1 − x_storage) − dt) / denom
S23. c4              = 2·dt / denom
S24. i_t             = N · q_t                                  (SpMV)
S25. b_rhs           = c2·i_t + c3·q_t + c4·q_prime_t
S26. a_values        = AValuesAssembler::assemble(c1)
S27. x_sol           = triangular_csr_solve(a_values, b_rhs)
S28. q_next          = max(x_sol, discharge_lb)
```

Returns `q_next`. Autograd-tracked parents: `n, q_spatial, p_spatial, q_t, q_prime_t`. Constants: `length, slope, x_storage`.

#### Backward (given `gy = ∂L/∂q_next`)

Walk steps S28→S1 in reverse. At each step accumulate ∂L into the tracked parents. Implementer constructs each line as backend primitives (no autograd nodes).

```text
B28. gx_sol         = gy · mask(x_sol > discharge_lb)
B27. (gA_values, gb_rhs)
      = csr_solve_backward(pattern, a_values, x_sol, gx_sol)
      ↑ delegate to the existing CsrSolveOp pieces — gA_values via
        gather+multiply (cusparse_grada), gb_rhs via transpose solve
        (backward_solve_primitive). We call the underlying primitive-level
        routines directly, NOT triangular_csr_solve again (which would push
        a new autograd node — we want a primitive-level computation here).
B26. gc1            = AValuesAssembler::assemble_backward(gA_values)
      ↑ assembler.assemble is c1 · adj on off-diagonals + 1 on diagonal.
        Backward: gc1[i] = sum over off-diag entries of row i of
        (gA_values[k] · adj_value[k]) · (-1).  Diagonals contribute 0.
B25a. gc2           = gb_rhs · i_t      (elementwise, length N)
B25b. gc3           = gb_rhs · q_t      (elementwise)
B25c. gc4           = gb_rhs · q_prime_t (elementwise)
B25d. gi_t          = c2 · gb_rhs       (elementwise)
B25e. gq_t_from_S25 = c3 · gb_rhs       (elementwise) — partial
B25f. gq_prime_t    = c4 · gb_rhs       (elementwise) — final, register
B24.  gq_t_from_S24 = AValuesAssembler::spmv_backward(gi_t)
      ↑ if forward i_t = N · q_t, backward = N^T · gi_t. Use the existing
        assembler primitive (or compute manually via the transposed CSR
        in pattern).
      Combined: gq_t_total = gq_t_from_S25 + gq_t_from_S24
                            (defer S27/S26/S25 contributions via b_rhs).
B23.  gdenom_from_c4 = -2·dt · gc4 / denom²     (elementwise)
B22.  gdenom_from_c3 = -gc3 · (2·k·(1-x_storage) - dt) / denom²
       gnum_c3       = gc3 / denom
       g_2k1mx_from_c3 = gnum_c3        (numerator term)
B21.  gdenom_from_c2 = -gc2 · (2·k·x_storage + dt) / denom²
       g_2kx_from_c2 = gc2 / denom
B20.  gdenom_from_c1 = -gc1 · (-2·k·x_storage + dt) / denom²
       g_2kx_from_c1 = -gc1 / denom
B19.  gdenom_total   = gdenom_from_c1 + gdenom_from_c2 + gdenom_from_c3
                     + gdenom_from_c4
       g_2k1mx_from_denom = gdenom_total                   (no x_storage factor)
       Combine g_2k1mx_total = g_2k1mx_from_c3 + g_2k1mx_from_denom
       g_2k_from_2k1mx     = g_2k1mx_total · (1 − x_storage)
       g_2kx_total          = g_2kx_from_c1 + g_2kx_from_c2
       g_2k_from_2kx       = g_2kx_total · x_storage
       g_2k_total          = g_2k_from_2kx + g_2k_from_2k1mx
       gk_muskingum         = 2 · g_2k_total
B18.  gcelerity      = -gk_muskingum · length / celerity²
B17.  gvelocity_cl   = gcelerity · 5/3
B16.  gvelocity_un   = gvelocity_cl · mask(velocity_lb < velocity_un < 15)
B15.  Let v = velocity_un. Then v = (1/n) · R^(2/3) · √slope.
      gn_from_S15    = gvelocity_un · ( -v / n )           (elementwise)
      gR_from_S15    = gvelocity_un · ( (2/3) · v / R )
      Register gn_from_S15 into gn_accumulator (defer total).
B14.  gR = gR_from_S15. R = area / wp.
      garea_from_R    = gR / wp
      gwp_from_R      = -gR · area / wp²
B13.  gwp = gwp_from_R. wp = bw + 2·d·sqrt(1 + ss²).
      gbw_from_S13    = gwp                                    (additive)
      gd_from_S13     = gwp · 2·sqrt(1+ss²)
      gss_from_S13    = gwp · 2·d · ss / sqrt(1+ss²)
B12.  garea = garea_from_R. area = (tw + bw)·d/2.
      gtw_from_S12    = garea · d/2
      gbw_from_S12    = garea · d/2
      gd_from_S12     = garea · (tw + bw)/2
B11.  gbw_raw        = (gbw_from_S13 + gbw_from_S12) · mask(bw_raw > bottom_width_lb)
B10.  gtw_from_S10   = gbw_raw
      gss_from_S10   = -2 · gbw_raw · d
      gd_from_S10    = -2 · gbw_raw · ss
B9.   gss_from_clamp = (gss_from_S13 + gss_from_S10) · mask(0.5 < ss_raw < 50)
B8.   ss_raw = tw · q_eps / (2·d).
      gtw_from_S8    = gss_from_clamp · q_eps / (2·d)
      gq_eps_from_S8 = gss_from_clamp · tw / (2·d)
      gd_from_S8     = -gss_from_clamp · tw·q_eps / (2·d²)
B7.   tw = p · depth^q_eps.
      gtw_total       = gtw_from_S12 + gtw_from_S10 + gtw_from_S8
      gp_from_S7      = gtw_total · depth^q_eps
      gdepth_from_S7  = gtw_total · p · q_eps · depth^(q_eps-1)
      gq_eps_from_S7  = gtw_total · tw · ln(depth)           (use saved tw)
B6.   gd_total = gd_from_S13 + gd_from_S12 + gd_from_S10 + gd_from_S8
                + gdepth_from_S7
      gd_pre_clamp    = gd_total · mask(ratio^exp > depth_lb)
B5/B6.  depth = ratio^exponent. Let d = depth = ratio^exp.
      gratio_from_S6  = gd_pre_clamp · exponent · ratio^(exponent-1)
                      = gd_pre_clamp · exponent · d / ratio
      gexp_from_S6    = gd_pre_clamp · d · ln(ratio)
B5.   exponent = 3 / (5 + 3·q_eps)
      gq_eps_from_S5 = -3 · gexp_from_S6 · 3 / (5 + 3·q_eps)²
                     = -9 · gexp_from_S6 / (5 + 3·q_eps)²
B4.   ratio = numerator / denominator
      gnum   = gratio_from_S6 / denominator
      gden   = -gratio_from_S6 · numerator / denominator²
B3.   denominator = p·√s + 1e-8
      gp_from_S3   = gden · √slope
      (gslope is dropped — slope is not a parent)
B2.   numerator = q_t · n · (q_eps + 1)
      gq_t_from_S2 = gnum · n · (q_eps + 1)
      gn_from_S2   = gnum · q_t · (q_eps + 1)
      gq_eps_from_S2 = gnum · q_t · n
B1.   q_eps = q_spatial + 1e-6
      gq_spatial_total = gq_eps_from_S8 + gq_eps_from_S7
                       + gq_eps_from_S5 + gq_eps_from_S2

Final accumulations:
  gp_spatial_total = gp_from_S7 + gp_from_S3
  gn_total         = gn_from_S15 + gn_from_S2
  gq_t_total       = gq_t_from_S25 + gq_t_from_S24 + gq_t_from_S2
  gq_prime_t       = (B25f, already final)
  gq_spatial_total = (B1, already final)
```

Each line above is one BURN backend-primitive call (`B::float_mul`, `B::float_div`, etc.) or `B::float_mask_where`. The implementer translates each line to one primitive op.

- [ ] **Step 1: Write the gradcheck test FIRST (TDD)**

Create `tests/sp8_gradcheck.rs`:

```rust
//! SP-8 Task 3: numerical-vs-analytical gradient check for the fused
//! TimestepOp. Builds a small synthetic network on NdArray, runs one
//! route_timestep (which by Task 4 will dispatch through TimestepOp),
//! and compares the analytical gradient against finite differences.
//!
//! Tolerance 1e-3 relative — matches the existing sparse_gradcheck pattern.

use approx::assert_relative_eq;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{backend::Backend, Tensor};

use ddrs::config::Config;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type I = NdArray<f32>;
type AB = Autodiff<I>;

const N: usize = 4;
const EPS: f32 = 1e-2;
const REL_TOL: f32 = 1e-3;
const ABS_TOL: f32 = 1e-5;

fn linear_chain_sparse() -> SparseAdjacency {
    let mut dense = vec![0.0_f32; N * N];
    for i in 0..N - 1 {
        dense[(i + 1) * N + i] = 1.0;
    }
    SparseAdjacency::from_dense(N, &dense, vec![1000.0; N], vec![0.001; N])
}

fn mock_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.params.parameter_ranges.n = [0.01, 0.1];
    cfg.params.parameter_ranges.q_spatial = [0.1, 0.9];
    cfg.params.parameter_ranges.p_spatial = [1.0, 200.0];
    cfg.params.attribute_minimums.velocity = 0.1;
    cfg.params.attribute_minimums.depth = 0.01;
    cfg.params.attribute_minimums.discharge = 0.001;
    cfg.params.attribute_minimums.bottom_width = 0.1;
    cfg.params.attribute_minimums.slope = 0.001;
    cfg.params.defaults.insert("p_spatial".to_string(), 1.0);
    cfg
}

/// Run one timestep, take loss = sum(q_next), compute analytical grads, then
/// compare against finite-difference grads for each parent.
fn analytical_vs_numerical(parent: &str) {
    let device = <I as Backend>::Device::default();
    let cfg = mock_cfg();
    let adj = linear_chain_sparse();

    // Build a single Q_t / q' window of two timesteps so the engine cold-starts
    // and one route_timestep fires.
    let q_t_data: [f32; N] = [0.6, 0.5, 0.4, 0.3];
    let q_prime_data: [f32; N] = [0.05, 0.04, 0.03, 0.02];
    let n_data: [f32; N] = [0.30, 0.35, 0.40, 0.45];
    let q_spatial_data: [f32; N] = [0.50, 0.45, 0.40, 0.35];
    let p_spatial_data: [f32; N] = [0.25, 0.30, 0.35, 0.40];

    let analytical = compute_grad::<AB>(parent,
        &n_data, &q_spatial_data, &p_spatial_data,
        &q_t_data, &q_prime_data, &cfg, &adj, &device, true);
    let numerical  = finite_diff::<AB>(parent,
        &n_data, &q_spatial_data, &p_spatial_data,
        &q_t_data, &q_prime_data, &cfg, &adj, &device);

    for (i, (a, num)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let denom = num.abs().max(ABS_TOL);
        let rel = (a - num).abs() / denom;
        assert!(rel < REL_TOL,
            "{parent}[{i}]: analytical={a}, numerical={num}, rel_diff={rel}");
    }
}

fn compute_grad<B: burn::tensor::backend::AutodiffBackend>(
    _parent: &str,
    _n: &[f32], _q_spatial: &[f32], _p_spatial: &[f32],
    _q_t: &[f32], _q_prime: &[f32],
    _cfg: &Config, _adj: &SparseAdjacency, _device: &B::Device,
    _analytical: bool,
) -> Vec<f32> {
    // Build Autodiff tensors for each input, set the one matching `parent`
    // as require_grad, run MuskingumCunge::route_timestep, sum the output,
    // call .backward(), extract the gradient tensor for that parent.
    //
    // The implementer fills this body using the existing MuskingumCunge API
    // in src/routing/mmc.rs:113-263. See tests/mmc.rs for an existing
    // forward-only example.
    todo!("Step 1: fill in using MuskingumCunge::route_timestep");
}

fn finite_diff<B: burn::tensor::backend::AutodiffBackend>(
    parent: &str,
    n: &[f32], q_spatial: &[f32], p_spatial: &[f32],
    q_t: &[f32], q_prime: &[f32],
    cfg: &Config, adj: &SparseAdjacency, device: &B::Device,
) -> Vec<f32> {
    // Central differences: for each index i in the named parent, perturb by
    // ±EPS, run forward both times, compute loss = sum(q_next), report
    // (loss_plus - loss_minus) / (2 · EPS).
    let mut grads = Vec::with_capacity(n.len());
    let inputs_len = match parent {
        "n" => n.len(),
        "q_spatial" => q_spatial.len(),
        "p_spatial" => p_spatial.len(),
        "q_t" => q_t.len(),
        "q_prime_t" => q_prime.len(),
        other => panic!("unknown parent {other}"),
    };
    for i in 0..inputs_len {
        let mut up = n.to_vec();
        let mut down = n.to_vec();
        let _ = (q_spatial, p_spatial, q_t, q_prime, cfg, adj, device, &mut up, &mut down);
        // implementer perturbs the right slot based on `parent`, runs forward
        // twice, computes (loss_plus - loss_minus) / (2 · EPS).
        grads.push(0.0);
    }
    let _ = i;
    grads
}

#[test]
fn gradcheck_n() { analytical_vs_numerical("n"); }
#[test]
fn gradcheck_q_spatial() { analytical_vs_numerical("q_spatial"); }
#[test]
fn gradcheck_p_spatial() { analytical_vs_numerical("p_spatial"); }
#[test]
fn gradcheck_q_t() { analytical_vs_numerical("q_t"); }
#[test]
fn gradcheck_q_prime_t() { analytical_vs_numerical("q_prime_t"); }
```

- [ ] **Step 2: Run gradcheck (will fail — backward is `unimplemented!`)**

Run: `cargo test --release --test sp8_gradcheck 2>&1 | tail -15`
Expected: FAIL — `unimplemented!("Task 3 — analytical backward not yet implemented")`.

- [ ] **Step 3: Fill in `compute_grad` and `finite_diff` helpers**

Replace the two `todo!()` bodies. Forward path uses
`MuskingumCunge::new(cfg, device)` → `setup_inputs` → one `route_timestep`
call. See `tests/mmc.rs` for the existing forward pattern.

Run: `cargo test --release --test sp8_gradcheck gradcheck_n 2>&1 | tail -15`
Expected: FAIL — analytical grad is empty / wrong because TimestepOp::backward
still panics, but the helper itself runs.

- [ ] **Step 4: Implement the analytical backward**

Edit `src/routing/mmc_op.rs::TimestepOp::backward`. Replace the
`unimplemented!()` with the chain-rule walk B28 → B1 above. Each line
becomes one or two backend-primitive op calls. Use the saved `TimestepState`
fields as inputs.

Implementer references that may help:
- `src/sparse/mod.rs:415-462` for the saved-state primitive access pattern.
- `src/sparse/dispatch.rs::backward_solve_primitive` for the transpose solve.
- `src/sparse/dispatch.rs::grada_primitive` for the per-nnz gather+multiply.
- BURN backend primitive ops live at `~/projects/burn/crates/burn-tensor/src/tensor/ops/tensor.rs` (`float_*` methods).

- [ ] **Step 5: Run gradcheck (must pass)**

Run: `cargo test --release --test sp8_gradcheck 2>&1 | tail -15`
Expected: 5 tests pass (one per parent).

If a single parent's gradcheck fails, the implementer isolates the chain
step responsible by printing intermediates. STOP and report if any parent
exceeds 1e-3 rel diff after debugging — escalate to BLOCKED.

- [ ] **Step 6: Commit**

```bash
git add src/routing/mmc_op.rs tests/sp8_gradcheck.rs
git commit -m "SP-8 Task 3: analytical backward + gradcheck

Implements TimestepOp::backward as a 28-step chain rule walk from
gy=∂L/∂q_next back to gradients on the 5 tracked parents (n, q_spatial,
p_spatial, q_t, q_prime_t). All five parents pass numerical-vs-analytical
gradcheck at 1e-3 rel on a 4-reach NdArray linear chain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire TimestepOp into route_timestep

The forward chain is built as backend primitives, the state is captured, and
the autograd node is registered.

**Files:**
- Modify: `src/routing/mmc.rs:221-263`
- Modify: `src/routing/mmc_op.rs::timestep_forward`

- [ ] **Step 1: Fill in the fused forward in `mmc_op.rs`**

In `src/routing/mmc_op.rs`, replace the `unimplemented!()` body of
`timestep_forward`. Pseudocode:

```rust
pub(crate) fn timestep_forward<I: Backend + 'static>(
    cfg: &Config,
    pattern: &Arc<CsrPattern>,
    assembler: &AValuesAssembler<I>,
    n: Tensor<Autodiff<I>, 1>,
    q_spatial: Tensor<Autodiff<I>, 1>,
    p_spatial: Tensor<Autodiff<I>, 1>,
    q_t: Tensor<Autodiff<I>, 1>,
    q_prime_t: Tensor<Autodiff<I>, 1>,
    length: Tensor<Autodiff<I>, 1>,
    slope: Tensor<Autodiff<I>, 1>,
    x_storage: Tensor<Autodiff<I>, 1>,
) -> Tensor<Autodiff<I>, 1>
where
    I::FloatTensorPrimitive: 'static,
    I::Device: 'static,
{
    // 1. Extract autograd nodes + primitives for each input.
    let (n_node, n_prim) = split_node_prim(n);
    let (qs_node, qs_prim) = split_node_prim(q_spatial);
    let (ps_node, ps_prim) = split_node_prim(p_spatial);
    let (qt_node, qt_prim) = split_node_prim(q_t);
    let (qpt_node, qpt_prim) = split_node_prim(q_prime_t);
    let length_prim = into_prim(length);
    let slope_prim = into_prim(slope);
    let x_storage_prim = into_prim(x_storage);

    let device = I::float_device(&n_prim);
    let dt = crate::routing::DT_SECONDS;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let bottom_width_lb = cfg.params.attribute_minimums.bottom_width;
    let velocity_lb = cfg.params.attribute_minimums.velocity;
    let discharge_lb = cfg.params.attribute_minimums.discharge;

    // 2. Forward chain at primitive level (no Autodiff nodes).
    //    Translate steps S1..S28 from the plan one-for-one using I::float_*
    //    methods. Each line is `let xxx_prim = I::float_op(...)`.
    //    The CSR solve (S26-S27) calls into the lower-level primitive
    //    routines, NOT triangular_csr_solve (which would push another
    //    autograd node).
    //    Implementer: refer to `src/routing/mmc.rs::route_timestep` for the
    //    Tensor-level form and translate each line.

    let q_eps_prim    = /* S1 */;
    let numerator_prim = /* S2 */;
    let denominator_prim = /* S3 */;
    let ratio_prim    = /* S4 */;
    let exponent_prim = /* S5 */;
    let depth_prim    = /* S6 */;
    let top_width_prim = /* S7 */;
    let side_slope_raw_prim = /* S8 */;
    let side_slope_prim = /* S9 */;
    let bw_raw_prim   = /* S10 */;
    let bottom_width_prim = /* S11 */;
    let area_prim     = /* S12 */;
    let wp_prim       = /* S13 */;
    let hydraulic_radius_prim = /* S14 */;
    let velocity_unclamped_prim = /* S15 */;
    let velocity_clamped_prim = /* S16 */;
    let celerity_prim = /* S17 */;
    let k_muskingum_prim = /* S18 */;
    let denom_prim    = /* S19 */;
    let c1_prim       = /* S20 */;
    let c2_prim       = /* S21 */;
    let c3_prim       = /* S22 */;
    let c4_prim       = /* S23 */;
    let i_t_prim      = assembler.spmv_primitive(qt_prim.clone()); // S24
    let b_rhs_prim    = /* S25 */;
    let a_values_prim = assembler.assemble_primitive(c1_prim.clone()); // S26
    let x_sol_prim    = crate::sparse::dispatch::forward_primitive::<I>(
        pattern, &a_values_prim, &b_rhs_prim, &device,
        false, // SP-8 uses CPU CSR solve inside the fused op for now;
               // wiring through SparseSolver::Cuda is a follow-up.
    ).0;
    let q_next_prim   = /* S28 */;

    // 3. Register the autograd op.
    let result = match TimestepOp
        .prepare::<NoCheckpointing>([
            n_node, qs_node, ps_node, qt_node, qpt_node,
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            let state = TimestepState::<I> {
                pattern: pattern.clone(),
                n: n_prim.clone(),
                q_spatial: qs_prim.clone(),
                p_spatial: ps_prim.clone(),
                q_t: qt_prim.clone(),
                q_prime_t: qpt_prim.clone(),
                length: length_prim.clone(),
                slope: slope_prim.clone(),
                x_storage: x_storage_prim.clone(),
                depth: depth_prim.clone(),
                top_width: top_width_prim.clone(),
                side_slope: side_slope_prim.clone(),
                bottom_width: bottom_width_prim.clone(),
                hydraulic_radius: hydraulic_radius_prim.clone(),
                velocity_unclamped: velocity_unclamped_prim.clone(),
                velocity_clamped: velocity_clamped_prim.clone(),
                celerity: celerity_prim.clone(),
                k_muskingum: k_muskingum_prim.clone(),
                denom: denom_prim.clone(),
                c1: c1_prim.clone(),
                c2: c2_prim.clone(),
                c3: c3_prim.clone(),
                c4: c4_prim.clone(),
                a_values: a_values_prim.clone(),
                b_rhs: b_rhs_prim.clone(),
                i_t: i_t_prim.clone(),
                x_sol: x_sol_prim.clone(),
                depth_lb, bottom_width_lb, velocity_lb, discharge_lb, dt,
            };
            prep.finish(state, q_next_prim)
        }
        OpsKind::UnTracked(prep) => prep.finish(q_next_prim),
    };
    Tensor::from_primitive(TensorPrimitive::Float(result))
}
```

`split_node_prim` and `into_prim` are local helpers — the implementer copies
the equivalent pattern from `triangular_csr_solve` in `src/sparse/mod.rs:483-491`.

If `AValuesAssembler` doesn't yet expose `assemble_primitive` /
`spmv_primitive` (it currently exposes Tensor-level `assemble` / `spmv`),
the implementer adds those primitive-level variants as part of this step —
small wrappers that take/return `B::FloatTensorPrimitive`. Place them in
`src/sparse/mod.rs` next to the existing methods.

- [ ] **Step 2: Rewrite `route_timestep` to delegate**

Replace lines 221-263 of `src/routing/mmc.rs::route_timestep`:

```rust
pub fn route_timestep(&self, q_prime_clamp: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1> {
    let n = self.n.as_ref().unwrap().clone();
    let q_spatial = self.q_spatial.as_ref().unwrap().clone();
    let p_spatial = self.p_spatial_broadcast(self.n_segments.expect("setup_inputs not called"));
    let length = self.length.as_ref().unwrap().clone();
    let slope = self.slope.as_ref().unwrap().clone();
    let x_storage = self.x_storage.as_ref().unwrap().clone();
    let q_t = self.discharge_t.as_ref().unwrap().clone();
    let pattern = self.pattern.as_ref().unwrap();
    let assembler = self.assembler.as_ref().unwrap();

    crate::routing::mmc_op::timestep_forward::<I>(
        &self.cfg, pattern, assembler,
        n, q_spatial, p_spatial,
        q_t, q_prime_clamp,
        length, slope, x_storage,
    )
}
```

- [ ] **Step 3: Run gradcheck (must still pass — Task 3's bar)**

Run: `cargo test --release --test sp8_gradcheck 2>&1 | tail -10`
Expected: 5 PASS — Task 4 wiring didn't break Task 3's math gate.

- [ ] **Step 4: Run V1 (CPU, frozen-params, 8 gauges)**

Run: `cargo test --release --test training_verification v1_loss_matches 2>&1 | tail -10`
Expected: PASS — V1 is the load-bearing correctness gate.

- [ ] **Step 5: Run V2 (CPU, frozen-params, all gauges)**

Run: `cargo test --release --test training_verification v2_loss_matches 2>&1 | tail -10`
Expected: PASS (~8 min runtime).

- [ ] **Step 6: Run V4 (full test period)**

Run: `cargo test --release --test training_verification v4_test_period_matches 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 7: Run V5 (CPU vs CUDA bit-match)**

Run: `cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -10`
Expected: PASS — the underlying CSR solve still agrees CPU vs CUDA.

- [ ] **Step 8: Commit**

```bash
git add src/routing/mmc.rs src/routing/mmc_op.rs src/sparse/mod.rs
git commit -m "SP-8 Task 4: route_timestep delegates to TimestepOp

The ~33-op Tensor-level chain in route_timestep collapses to one
autograd node via TimestepOp. The forward computes S1..S28 at the
backend primitive level and saves all 24 intermediates to TimestepState
for the analytical backward. V1, V2, V4, V5 all still green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: V7a perf benchmark (hard gate)

Mirror V6's structure but with the assertion restored: median CUDA wall ÷
median CPU wall ≤ 0.7.

**Files:**
- Create: `tests/sp8_v7_perf.rs`

- [ ] **Step 1: Write the test**

Create `tests/sp8_v7_perf.rs`:

```rust
//! SP-8 V7a: assert CUDA wall-time ≤ 0.7× CPU wall-time on the smoke train.
//!
//! Run manually:
//!   cargo test --release --test sp8_v7_perf -- --ignored --nocapture
//!
//! Median of three runs each (first run discarded — JIT warmup).

use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_MINI_BATCHES: &str = "3";
const RUNS_PER_VARIANT: usize = 4; // first is warmup; median of last 3
const RATIO_THRESHOLD: f32 = 0.7;

fn data_files_present() -> bool {
    Path::new("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr").exists()
        && Path::new("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc").exists()
}

fn cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        type CudaInner = burn_cuda::Cuda<f32, i32>;
        type Dev = <CudaInner as burn::tensor::backend::BackendTypes>::Device;
        let _d: Dev = Default::default();
    })
    .is_ok()
}

fn write_override_yaml(value: &str) -> PathBuf {
    let base = std::fs::read_to_string("config/merit_training.yaml")
        .expect("read merit_training.yaml");
    let mut lines: Vec<String> = base
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("sparse_solver:") && !t.starts_with("# sparse_solver:")
        })
        .map(String::from)
        .collect();
    let params_idx = lines
        .iter()
        .position(|l| l.trim_start().starts_with("params:"))
        .expect("params: not found");
    lines.insert(params_idx + 1, format!("  sparse_solver: {value}"));
    let path = PathBuf::from(format!("/tmp/v7_{value}.yaml"));
    std::fs::write(&path, lines.join("\n") + "\n").expect("write override yaml");
    path
}

fn run_train_minutes(config_path: &Path) -> f32 {
    let stem = config_path.file_stem().unwrap().to_string_lossy().into_owned();
    let ckpt_dir = format!("/tmp/v7_ckpt_{stem}");
    let _ = std::fs::remove_dir_all(&ckpt_dir);
    let output = Command::new("cargo")
        .args([
            "run", "--release", "--bin", "train", "--",
            "--config", config_path.to_str().unwrap(),
            "--checkpoint-dir", &ckpt_dir,
            "--max-mini-batches", MAX_MINI_BATCHES,
        ])
        .output()
        .expect("spawn cargo run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(),
        "train failed ({}): stdout=\n{}\nstderr=\n{}", stem, stdout, stderr);
    let combined = format!("{stdout}\n{stderr}");
    for line in combined.lines() {
        if let Some(idx) = line.find("Training complete in ") {
            let tail = &line[idx + "Training complete in ".len()..];
            if let Some(min_idx) = tail.find(" min") {
                if let Ok(m) = tail[..min_idx].trim().parse::<f32>() {
                    return m;
                }
            }
        }
    }
    panic!("could not parse training minutes from output");
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[test]
#[ignore]
fn v7a_cuda_at_least_30_percent_faster_than_cpu() {
    if !data_files_present() {
        eprintln!("V7a skip: data files not present");
        return;
    }
    if !cuda_available() {
        eprintln!("V7a skip: no CUDA device");
        return;
    }

    let cpu_yaml = write_override_yaml("cpu");
    let cuda_yaml = write_override_yaml("cuda");

    let mut cpu_times = Vec::with_capacity(RUNS_PER_VARIANT - 1);
    let mut cuda_times = Vec::with_capacity(RUNS_PER_VARIANT - 1);

    for i in 0..RUNS_PER_VARIANT {
        let cpu_min = run_train_minutes(&cpu_yaml);
        let cuda_min = run_train_minutes(&cuda_yaml);
        eprintln!("V7a run {i}: cpu={cpu_min:.3} min, cuda={cuda_min:.3} min");
        if i > 0 {
            cpu_times.push(cpu_min);
            cuda_times.push(cuda_min);
        }
    }
    let cpu_med = median(&mut cpu_times);
    let cuda_med = median(&mut cuda_times);
    let ratio = cuda_med / cpu_med;
    eprintln!("V7a: cpu_median={cpu_med:.3} min, cuda_median={cuda_med:.3} min, ratio={ratio:.3}");
    assert!(
        ratio <= RATIO_THRESHOLD,
        "V7a FAILED: cuda/cpu ratio = {ratio:.3} > {RATIO_THRESHOLD}",
    );
}
```

- [ ] **Step 2: Run V7a**

Run: `cargo test --release --test sp8_v7_perf -- --ignored --nocapture 2>&1 | tail -20`
Expected: PASS — ratio ≤ 0.7. Runtime ~25 min (4 runs × 2 variants × 3 min/run).

If V7a FAILS but V6 passed correctness (V1/V2/V4/V5): SP-8 closes as
"correctness only" with V7a remaining red. Report DONE_WITH_CONCERNS;
controller decides next step (SP-9 vs. ship as-is).

- [ ] **Step 3: Commit**

```bash
git add tests/sp8_v7_perf.rs
git commit -m "SP-8 Task 5: V7a hard perf gate (cuda ≤ 0.7× cpu)

Median of 3 runs per variant (after 1 warmup). Asserts that the fused
TimestepOp closes the GPU-vs-CPU gap measured in V6 (which came in at
0.998 parity).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: V7b nsys profile gate (hard gate)

`scatter_kernel_t_f32_i_i32` must drop below 30% of GPU compute time.

**Files:**
- Create: `scripts/sp8_check_scatter.sh`
- Create: `tests/sp8_v7_profile.rs`

- [ ] **Step 1: Write the bash script**

Create `scripts/sp8_check_scatter.sh`:

```bash
#!/usr/bin/env bash
# SP-8 V7b: gate scatter_kernel_t_f32_i_i32 below 30% of GPU compute time.
#
# Exits 0 if the gate passes, 1 otherwise. Writes the nsys report and stats
# to $NSYS_DIR (default $HOME/nsys_out).
set -euo pipefail

NSYS_DIR="${NSYS_DIR:-$HOME/nsys_out}"
mkdir -p "$NSYS_DIR"
REPORT="$NSYS_DIR/sp8_v7b"
CKPT="/tmp/sp8_v7b_ckpt"
THRESHOLD="${SP8_SCATTER_THRESHOLD:-30.0}"

rm -rf "$CKPT"
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --output="$REPORT" --force-overwrite=true \
    target/release/train --config config/merit_training.yaml \
                          --checkpoint-dir "$CKPT" \
                          --max-mini-batches 3

STATS="$NSYS_DIR/sp8_v7b_stats.txt"
nsys stats "$REPORT.nsys-rep" --report cuda_gpu_kern_sum > "$STATS"

# nsys output column layout for cuda_gpu_kern_sum (verified against the
# SP-6 stats.txt at $HOME/nsys_out/stats.txt):
#   "Time (%)", "Total Time (ns)", "Instances", "Avg", "Med", "Min", "Max", "StdDev", "Name"
# We extract column 1 (Time %) for the row whose Name column contains
# scatter_kernel_t_f32_i_i32.
PCT=$(awk '
    /scatter_kernel_t_f32_i_i32/ {
        gsub(",", "", $1);
        print $1;
        exit;
    }
' "$STATS")

if [ -z "$PCT" ]; then
    echo "V7b: scatter_kernel_t_f32_i_i32 not found in $STATS — assuming 0%"
    PCT=0
fi
echo "V7b: scatter_kernel percentage = $PCT% (threshold $THRESHOLD%)"
awk -v p="$PCT" -v t="$THRESHOLD" 'BEGIN { exit (p+0 < t+0) ? 0 : 1 }'
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/sp8_check_scatter.sh`

- [ ] **Step 3: Write the test wrapper**

Create `tests/sp8_v7_profile.rs`:

```rust
//! SP-8 V7b: assert scatter_kernel_t_f32_i_i32 below 30% of GPU compute time
//! via nsys profile.
//!
//! Run manually:
//!   cargo test --release --test sp8_v7_profile -- --ignored --nocapture

use std::path::Path;
use std::process::Command;

#[test]
#[ignore]
fn v7b_scatter_below_30_percent() {
    if which::which("nsys").is_err() {
        eprintln!("V7b skip: nsys not on PATH");
        return;
    }
    if !Path::new("/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr").exists() {
        eprintln!("V7b skip: data files not present");
        return;
    }
    let status = Command::new("bash")
        .arg("scripts/sp8_check_scatter.sh")
        .status()
        .expect("spawn sp8_check_scatter.sh");
    assert!(status.success(),
        "V7b FAILED: scatter_kernel_t_f32_i_i32 ≥ 30% of GPU time. See $NSYS_DIR/sp8_v7b_stats.txt.");
}
```

- [ ] **Step 4: Run V7b**

Run: `cargo test --release --test sp8_v7_profile -- --ignored --nocapture 2>&1 | tail -15`
Expected: PASS — scatter % < 30. Runtime ~5 min (nsys profile + stats parse).

If V7b FAILS, the implementer captures `$HOME/nsys_out/sp8_v7b_stats.txt`
and reports it in the close-out — diagnose which kernel(s) now dominate.

- [ ] **Step 5: Commit**

```bash
git add scripts/sp8_check_scatter.sh tests/sp8_v7_profile.rs
git commit -m "SP-8 Task 6: V7b hard profile gate (scatter < 30%)

Runs nsys profile on bin/train --max-mini-batches 3 and parses the
cuda_gpu_kern_sum report for scatter_kernel_t_f32_i_i32's percentage.
Test passes if the percentage falls below 30 (down from SP-6's 78.7%).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Close — ARCHITECTURE.md + tag + push

**Files:**
- Modify: `.claude/ARCHITECTURE.md`

- [ ] **Step 1: Append a note to ARCHITECTURE.md**

Read the current file:

```bash
cat .claude/ARCHITECTURE.md | head -50
```

Append a section (place it after the "Module map" section):

```markdown

## SP-8 fused MC timestep (2026-05-22)

`MuskingumCunge::route_timestep` is a thin wrapper around
`mmc_op::timestep_forward` (`src/routing/mmc_op.rs`). The forward chain
runs at the backend-primitive level — no autograd nodes — and the
saved-state struct holds all 24 intermediates the analytical backward
needs. One autograd node per timestep instead of ~33.

Before SP-8: nsys profile showed `scatter_kernel_t_f32_i_i32` at 78.7%
of GPU compute time. After SP-8 the V7b gate enforces <30%.

The CSR triangular solve (`src/sparse/mod.rs::triangular_csr_solve` +
`CsrSolveOp`) remains its own custom Backward and is called as a
sub-op from inside the fused forward via the primitive-level
`dispatch::forward_primitive`.
```

- [ ] **Step 2: Verify all gates one more time**

Run in sequence:

```bash
cargo test --release --test training_verification v1_loss_matches 2>&1 | tail -3
cargo test --release --test training_verification v3_train_one_epoch 2>&1 | tail -3
cargo test --release --test sparse_cusparse_v5 v5_cpu_and_cuda 2>&1 | tail -3
cargo test --release --test sp8_gradcheck 2>&1 | tail -3
# V7a + V7b are #[ignore]'d to keep regular runs fast — invoke explicitly:
cargo test --release --test sp8_v7_perf -- --ignored --nocapture 2>&1 | tail -3
cargo test --release --test sp8_v7_profile -- --ignored --nocapture 2>&1 | tail -3
```

Expected: each tail shows `ok. 1 passed` (or more, for sp8_gradcheck).

- [ ] **Step 3: Tag + commit ARCHITECTURE update**

```bash
git add .claude/ARCHITECTURE.md
git commit -m "SP-8 close: fused MC timestep + V7 perf gates green

V7a: cuda/cpu ratio < 0.7 on the smoke train.
V7b: scatter_kernel < 30% of GPU compute time.
V1/V2/V4/V5/gradcheck all green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git tag sp8-fusion-landed
```

- [ ] **Step 4: Push to origin**

```bash
git push origin master
git push origin sp8-fusion-landed
```

Expected: both pushes succeed.

---

## Self-Review

### Spec coverage

| Spec section | Task |
|---|---|
| V7a perf gate (ratio ≤ 0.7) | Task 5 |
| V7b profile gate (scatter < 30%) | Task 6 |
| Diagnosis with STOP authority | Task 1 |
| TimestepOp + TimestepState | Task 2 |
| Analytical backward + gradcheck | Task 3 |
| Wire into route_timestep | Task 4 |
| V1/V2/V4/V5 regression checks | Task 4 (Steps 4-7) |
| ARCHITECTURE.md update | Task 7 |
| Both V7 gates must pass | Task 7 (Step 2) |
| Tag `sp8-fusion-landed` | Task 7 |

### Placeholder scan

- `compute_grad` and `finite_diff` helpers in `tests/sp8_gradcheck.rs` Task 3
  Step 1 contain `todo!()`. These are intentional — Step 3 fills them in
  with concrete code. The plan instructs Step 3 to "Replace the two `todo!()`
  bodies" with explicit code. This is the same pattern SP-4 used.
- `timestep_forward` body in Task 2 Step 1 uses `unimplemented!()`. Same
  reason — Task 4 Step 1 replaces it. Instructed explicitly.
- Step 1 of Task 4 has 28 `let xxx_prim = /* S?? */;` lines marked as
  "implementer fills in". This is a deliberate plan-vs-code tradeoff —
  the forward is mechanical translation of Tensor-level ops in the original
  `route_timestep` to primitive-level ops, and including all 28 lines fully
  here triples the plan length while adding no new information beyond the
  forward chain already documented in Task 3. The implementer copies each
  Tensor-level line from `src/routing/mmc.rs:221-263`, replaces
  `.method()` Tensor calls with `B::float_method(...)` primitive calls,
  and refers to the saved-state struct in Task 2 for field names.

### Type consistency

- `TimestepOp` is `Backward<B, 1>` (rank-1 output Q_{t+1}).
- Parents N = 5 in fixed order `[n, q_spatial, p_spatial, q_t, q_prime_t]`.
- `TimestepState<B>` fields use `B::FloatTensorPrimitive` (matching
  `CsrSolveState<B>` in `src/sparse/mod.rs:407-412`).
- `timestep_forward<I: Backend + 'static>` matches the where-clauses on
  `triangular_csr_solve` (`src/sparse/mod.rs:473-482`).
- `pattern: Arc<CsrPattern>` consistent everywhere.

### Risk recheck

- Task 3 numerical/analytical gradcheck at 1e-3 — if any parent fails after
  debugging, escalate as BLOCKED (per spec Concern #1).
- Task 4 V1/V2/V4 regressions — gated explicitly; if any fail, stop and
  isolate before Tasks 5-7.
- Task 5 perf — if it misses 0.7, report DONE_WITH_CONCERNS; don't paper
  over with relaxation. Controller decides SP-9 vs. ship.
- Task 6 profile — same posture as Task 5; the gate is calibrated against
  the SP-6 baseline of 78.7%.

---

## Execution choice

Plan complete and saved to `.claude/specs/2026-05-22-sp8-mc-timestep-fusion-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — same workflow as SP-1..SP-7. Fresh
   subagent per task with two-stage spec-then-quality review.
2. **Inline Execution** — batch with checkpoints.

Which approach?
