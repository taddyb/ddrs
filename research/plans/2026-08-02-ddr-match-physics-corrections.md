# `ddr_match` Physics Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `params.ddr_match: bool` flag (default `true`) that, when `false`, replaces four physically-incorrect behaviours in the Muskingum-Cunge core — the `5/3` celerity factor, the constant `X = 0.3`, unmonitored negative-discharge clamping, and the absent Courant guard — each with a gradient-exact corrected implementation.

**Architecture:** All four corrections live inside `forward_chain_inner` / `timestep_backward_core` in `src/routing/mmc_op.rs`, selected by a single boolean threaded through `SavedState`. `ddr_match: true` reproduces DDR bit-for-bit so `examples/compare_ddr_sandbox` stays an ABSOLUTE MATCH (invariant 1). Every new forward term gets a hand-derived backward branch (invariant 4 — no autograd-tape unrolling) validated by finite-difference gradcheck.

**Tech Stack:** Rust, BURN 0.21 autograd (`Backward<I, N>` custom ops), `cargo test`, NdArray backend for deterministic tests.

---

## Concerns for the user

**What could go wrong, and why:**

1. **Silent parity loss.** If `ddr_match` is not threaded into *every* branch point, a config with `ddr_match: true` could still take a corrected path and break invariant 1 without an obvious symptom. Mitigated by Task 1's parity test, which must run before any physics change.
2. **Gradient-exactness regression (invariant 4).** Both β and Cunge `X` introduce new dependencies on tensors that *already* carry gradient paths (`area`, `top_width`, `wetted_perimeter`, `side_slope`, `q_t`, `celerity`). A missed chain-rule term produces a plausible-but-wrong gradient that training will silently absorb — exactly the failure mode that cost the AdaDelta run. Mitigated by a dedicated gradcheck per correction, run *before* the correction is used in any training run.
3. **Cunge X makes Courant worse before it makes it better.** `X ≈ 0.49` narrows the non-negative-coefficient window from `[0.6, 1.4]` to roughly `[0.98, 1.02]`. Enabling Task 4 without Task 5 will increase negative-discharge frequency. **Tasks 4 and 5 must be evaluated together**, and Task 2's counter is the instrument that proves it.
4. **Sub-stepping multiplies tape depth.** `n_sub` sub-steps per hourly timestep multiply autograd tape entries by `n_sub`. At `n_sub = 4` over 2160 hourly steps this is a real memory increase on GPU. Mitigated by making `n_sub` adaptive (per-reach, capped) rather than global.
5. **The corrections may not improve skill.** Each is individually absorbable by `n` (proven for the slope clamp; likely for a near-constant β). The scientific payoff is an interpretable parameter field, not necessarily a better NSE. Do not promote `ddr_match: false` on the basis of physics alone — gate it on a measured comparison.

**Assumptions made:**

- **DDR parity is worth preserving** as the only end-to-end reference check the port has, so `true` is the default and DDR itself is not modified. If the team decides to fix DDR too, Tasks 3–5 become the reference and the fixture is regenerated.
- **`X` becomes a derived quantity, not a learnable one.** `x_storage` stays out of `kan_head.learnable_parameters`; under `ddr_match: false` it is computed from Cunge's formula. Making it *both* learnable and Cunge-derived is contradictory and is explicitly out of scope.
- **Instrumentation is safe in both modes.** Counting negative solves changes no numerics, so Task 2 is unconditional and lands first.
- **Hourly `Δt = 3600 s` stays hardcoded** (`mmc.rs:33`). Sub-stepping divides it internally rather than changing the forcing cadence.

**Blast radius:**

| File | Change | Risk |
|---|---|---|
| `src/config.rs` | +1 field, +1 default fn | low — additive, defaulted |
| `src/routing/mmc_op.rs` | forward branches + backward branches + `SavedState` field | **high** — invariants 1 and 4 both live here |
| `src/routing/mmc.rs` | thread flag; read + log the counter | low |
| `tests/` | 4 new test files | none |
| `examples/compare_ddr_sandbox.rs` | none (default preserves it) | none |

Diagram: `.claude/PHYSICS-CORRECTIONS.md`.

**Out of scope (separate plan):** every disaggregation-head finding. That is an independent subsystem; see "Follow-up" at the end.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/config.rs` | Declare `ParamsSection.ddr_match: bool`, default `true` |
| `src/routing/mmc_op.rs` | Both forward paths, both backward paths, `SavedState.ddr_match`, negative-solve counter |
| `src/routing/mmc.rs` | Pass `cfg.params.ddr_match` down; report the counter per forward |
| `tests/ddr_match_flag.rs` | Flag defaults, and `false` actually changes output |
| `tests/negative_discharge_counter.rs` | Counter fires on a known-unstable network |
| `tests/celerity_beta.rs` | β analytic vs finite-difference `dQ/dA`; limits; gradcheck |
| `tests/cunge_x.rs` | `X` formula, clamping, diffusion match; gradcheck |
| `tests/courant_substep.rs` | Sub-stepping keeps `Cr` in range and conserves mass |

---

## Task 1: `ddr_match` config flag (no behaviour change)

**Files:**
- Modify: `src/config.rs` (`ParamsSection`, near `use_cuda_graphs`)
- Modify: `src/routing/mmc_op.rs` (`SavedState`, `forward_chain_inner` signature, `timestep_forward`)
- Test: `tests/ddr_match_flag.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tests/ddr_match_flag.rs
//! `ddr_match` defaults to true so every existing config and the DDR sandbox
//! parity example keep their current behaviour (invariant 1).
use ddrs::config::Config;

#[test]
fn ddr_match_defaults_to_true() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;
    let cfg: Config = serde_yaml::from_str(yaml).expect("parse");
    assert!(cfg.params.ddr_match, "ddr_match must default to true");
}

#[test]
fn ddr_match_can_be_disabled() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 42
np_seed: 42
params:
  ddr_match: false
  parameter_ranges:
    n: [0.015, 0.25]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 200.0]
"#;
    let cfg: Config = serde_yaml::from_str(yaml).expect("parse");
    assert!(!cfg.params.ddr_match);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test ddr_match_flag`
Expected: FAIL — `no field 'ddr_match' on type 'ParamsSection'`

- [ ] **Step 3: Add the config field**

In `src/config.rs`, inside `ParamsSection` (next to `use_cuda_graphs`):

```rust
    /// When `true` (default) the routing core reproduces DDR's formulation
    /// bit-for-bit, including two known physical approximations:
    ///   * celerity `c = v · 5/3` (the wide-rectangular Kleitz-Seddon limit,
    ///     ~22-27% high for the trapezoid this code actually builds), and
    ///   * Muskingum `X ≡ 0.3` (constant, NOT Cunge-derived, giving a median
    ///     10-30x excess numerical diffusion).
    ///
    /// Set `false` to enable the corrected physics. This CHANGES FORWARD
    /// OUTPUT and will break `examples/compare_ddr_sandbox`'s ABSOLUTE MATCH
    /// (invariant 1) — which is why the default preserves DDR behaviour.
    /// See `.claude/PHYSICS-CORRECTIONS.md`.
    #[serde(default = "default_ddr_match")]
    pub ddr_match: bool,
```

Add the default fn near the other `default_*` helpers:

```rust
fn default_ddr_match() -> bool {
    true
}
```

Add `ddr_match: default_ddr_match(),` to `ParamsSection`'s `Default` impl.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test ddr_match_flag`
Expected: PASS, 2 tests

- [ ] **Step 5: Thread the flag into the op (still unused)**

In `src/routing/mmc_op.rs`, add to `SavedState` (after `depth_lb: f32`):

```rust
    pub ddr_match: bool,
```

Add a parameter to `forward_chain_inner` after `discharge_lb: f32`:

```rust
    ddr_match: bool,
```

Populate it in the `SavedState` construction (`ddr_match,`) and pass it at the call site in `timestep_forward`:

```rust
    let ddr_match = cfg.params.ddr_match;
```

- [ ] **Step 6: Verify nothing changed**

Run: `cargo build --release && cargo run --release --example compare_ddr_sandbox`
Expected: `ABSOLUTE MATCH` (max abs diff < 1e-3 m³/s)

Run: `cargo test --test sp8_gradcheck --test sparse_gradcheck --test mmc`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/config.rs src/routing/mmc_op.rs tests/ddr_match_flag.rs
git commit -m "feat(config): ddr_match flag, defaulting to DDR-matching behaviour"
```

---

## Task 2: Negative-discharge instrumentation (both modes, no physics change)

The `clamp_min(1e-4)` at `mmc_op.rs:919` converts every negative solve to `+1e-4`, creating mass and hiding Courant instability. Frequency has never been measured. This task measures it and changes nothing else.

**Files:**
- Modify: `src/routing/mmc_op.rs` (counter + increment at S28)
- Modify: `src/routing/mmc.rs` (report per forward)
- Test: `tests/negative_discharge_counter.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tests/negative_discharge_counter.rs
//! The S28 clamp silently turns negative solves into +1e-4. This counter is
//! the only way to see how often Muskingum's non-negative-coefficient
//! condition (2X <= Cr <= 2(1-X)) is violated in a real run.
use ddrs::routing::mmc_op::{negative_solve_stats, reset_negative_solve_stats};

#[test]
fn counter_starts_at_zero_and_resets() {
    reset_negative_solve_stats();
    let (neg, total) = negative_solve_stats();
    assert_eq!(neg, 0);
    assert_eq!(total, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test negative_discharge_counter`
Expected: FAIL — `unresolved import ddrs::routing::mmc_op::negative_solve_stats`

- [ ] **Step 3: Implement the counter**

At the top of `src/routing/mmc_op.rs`:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

/// Count of solve outputs that came out NEGATIVE before the S28
/// `clamp_min(discharge_lb)` rewrote them to `+1e-4`.
///
/// Why this exists: Muskingum coefficients are only non-negative for
/// `2X <= Cr <= 2(1-X)` with `Cr = dt/K`. Measured on CONUS at mean flow with
/// `X = 0.3`, 69.8% of reaches sit outside that window (28.4% give `c1 < 0`,
/// 41.4% give `c3 < 0`), so negative discharge is expected — and the clamp
/// both CREATES MASS and removes the only symptom. Nothing in the codebase
/// measured this before 2026-08-02.
static NEG_SOLVES: AtomicU64 = AtomicU64::new(0);
static TOTAL_SOLVES: AtomicU64 = AtomicU64::new(0);

/// `(negative_count, total_count)` since the last reset.
pub fn negative_solve_stats() -> (u64, u64) {
    (NEG_SOLVES.load(Ordering::Relaxed), TOTAL_SOLVES.load(Ordering::Relaxed))
}

pub fn reset_negative_solve_stats() {
    NEG_SOLVES.store(0, Ordering::Relaxed);
    TOTAL_SOLVES.store(0, Ordering::Relaxed);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --test negative_discharge_counter`
Expected: PASS

- [ ] **Step 5: Increment at S28**

In `forward_chain_inner`, immediately before the existing clamp (`mmc_op.rs:919`):

```rust
    // Count negatives BEFORE the clamp masks them. Diagnostic only: reads the
    // solve to host, changes no numerics, and is identical in both ddr_match
    // modes.
    {
        let v: Vec<f32> = wrap(x_sol_prim.clone()).into_data().to_vec::<f32>().unwrap();
        let neg = v.iter().filter(|x| **x < 0.0).count() as u64;
        NEG_SOLVES.fetch_add(neg, Ordering::Relaxed);
        TOTAL_SOLVES.fetch_add(v.len() as u64, Ordering::Relaxed);
    }
    // S28: q_next = max(x_sol, discharge_lb)
    let q_next = x_sol.clone().clamp_min(discharge_lb);
```

(Substitute the actual identifier for the pre-clamp solve primitive in scope; it is `x_sol` wrapped from the solver output.)

- [ ] **Step 6: Report it once per forward**

In `src/routing/mmc.rs`, at the end of the timestep loop in `forward`:

```rust
        let (neg, total) = crate::routing::mmc_op::negative_solve_stats();
        if total > 0 && neg > 0 {
            eprintln!(
                "  negative solves before clamp: {neg}/{total} ({:.3}%) — \
                 Muskingum coefficient sign violation (see .claude/PHYSICS-CORRECTIONS.md)",
                100.0 * neg as f64 / total as f64
            );
        }
        crate::routing::mmc_op::reset_negative_solve_stats();
```

- [ ] **Step 7: Verify parity still holds and observe the number**

Run: `cargo run --release --example compare_ddr_sandbox`
Expected: `ABSOLUTE MATCH`, plus a `negative solves before clamp:` line

- [ ] **Step 8: Commit**

```bash
git add src/routing/mmc_op.rs src/routing/mmc.rs tests/negative_discharge_counter.rs
git commit -m "feat(diag): count negative Muskingum solves before the S28 clamp"
```

---

## Task 3: Trapezoidal celerity β (`ddr_match: false`)

**Physics.** For a trapezoid, `c = dQ/dA = (dQ/dy)/T` with `Q = (1/n)A^(5/3)P^(-2/3)√S` gives

```
c = v · β,    β = 5/3 − (4/3)·A·√(1+z²)/(T·P)
```

`β → 5/3` as `b/y → ∞` (wide rectangular) and `β → 4/3` as `b → 0` (triangular). The code's channels have `κ = b/y ∈ [0.7, 1.8]`, giving `β ≈ 1.31–1.36` — so the hardcoded `5/3` is **22–27% high**.

**Files:**
- Modify: `src/routing/mmc_op.rs` (S17 forward; B17 backward)
- Test: `tests/celerity_beta.rs`

- [ ] **Step 1: Write the physics test (no code yet)**

```rust
// tests/celerity_beta.rs
//! beta = 5/3 - (4/3)·A·sqrt(1+z^2)/(T·P) is the exact kinematic-wave
//! celerity ratio c/v for a trapezoid. Verified three ways: against the
//! wide-rectangular limit (5/3), the triangular limit (4/3), and a
//! finite-difference dQ/dA on the same section.

fn beta(b: f64, z: f64, y: f64) -> f64 {
    let a = (b + z * y) * y;
    let t = b + 2.0 * z * y;
    let p = b + 2.0 * y * (1.0 + z * z).sqrt();
    5.0 / 3.0 - (4.0 / 3.0) * a * (1.0 + z * z).sqrt() / (t * p)
}

fn q_manning(b: f64, z: f64, y: f64, n: f64, s: f64) -> (f64, f64) {
    let a = (b + z * y) * y;
    let p = b + 2.0 * y * (1.0 + z * z).sqrt();
    ((1.0 / n) * a.powf(5.0 / 3.0) * p.powf(-2.0 / 3.0) * s.sqrt(), a)
}

#[test]
fn beta_recovers_wide_rectangular_limit() {
    assert!((beta(1e6, 0.0, 2.0) - 5.0 / 3.0).abs() < 1e-4);
}

#[test]
fn beta_recovers_triangular_limit() {
    assert!((beta(0.0, 2.0, 2.0) - 4.0 / 3.0).abs() < 1e-12);
}

#[test]
fn beta_matches_finite_difference_dq_da() {
    let (n, s) = (0.08, 2e-3);
    for &(b, z, y) in &[(6.3, 0.50, 9.31), (13.8, 0.57, 8.09), (8.8, 2.76, 6.38)] {
        let h = y * 1e-6;
        let (q1, a1) = q_manning(b, z, y - h, n, s);
        let (q2, a2) = q_manning(b, z, y + h, n, s);
        let (q0, a0) = q_manning(b, z, y, n, s);
        let fd = ((q2 - q1) / (a2 - a1)) / (q0 / a0);
        assert!((fd - beta(b, z, y)).abs() < 1e-6, "b={b} z={z} y={y}: fd={fd}");
    }
}

#[test]
fn ddr_five_thirds_is_biased_high_for_these_channels() {
    // Regression guard on the MAGNITUDE of the defect ddr_match=false fixes.
    let bt = beta(13.8, 0.57, 8.09);
    let err = (5.0 / 3.0) / bt - 1.0;
    assert!(err > 0.20 && err < 0.30, "expected +20..30% bias, got {err:.3}");
}
```

- [ ] **Step 2: Run to verify the physics tests pass immediately**

Run: `cargo test --test celerity_beta`
Expected: PASS, 4 tests (these validate the formula, not yet the implementation)

- [ ] **Step 3: Implement the forward branch**

In `forward_chain_inner`, replace S17 (`mmc_op.rs:849`):

```rust
    // S17: celerity.
    //   ddr_match=true  -> c = v·5/3, the wide-rectangular Kleitz-Seddon
    //                      limit. Matches ddr/mmc.py:167. WRONG for the
    //                      trapezoid built above (kappa = b/y ~ 0.7-1.8 here,
    //                      so the true ratio is ~1.31-1.36, not 1.667).
    //   ddr_match=false -> exact trapezoidal c = dQ/dA = (dQ/dy)/T:
    //                      beta = 5/3 - (4/3)·A·sqrt(1+z^2)/(T·P)
    let celerity = if ddr_match {
        velocity_cl.clone() * (5.0_f32 / 3.0_f32)
    } else {
        let root = (side_slope.clone().powf_scalar(2.0) + 1.0).sqrt();
        let beta = -(_area.clone() * root) / (top_width.clone() * wp.clone()) * (4.0 / 3.0)
            + (5.0 / 3.0);
        velocity_cl.clone() * beta
    };
```

- [ ] **Step 4: Implement the backward branch**

Derivation. With `G ≡ 5/3 − β` (so `β = 5/3 − G` and `G = (4/3)·A·u/(T·P)`, `u = √(1+z²)`):

```
∂β/∂A = −G/A      ∂β/∂T = +G/T      ∂β/∂P = +G/P      ∂β/∂z = −G·z/(1+z²)
```

In `timestep_backward_core`, replace B17 (`mmc_op.rs:352`):

```rust
        // B17. celerity = velocity_cl · beta
        //   ddr_match: beta is the constant 5/3, so only gvelocity_cl exists.
        //   otherwise: beta depends on area/top_width/wp/side_slope, all of
        //   which already have gradient paths — these are ADDITIONAL
        //   contributions, not replacements.
        let (gvelocity_cl, gbeta_terms) = if state.ddr_match {
            (gcelerity.clone() * (5.0 / 3.0), None)
        } else {
            let ss = wrap(state.side_slope.clone());
            let tw = wrap(state.top_width.clone());
            let area = (tw.clone() + wrap(state.bottom_width.clone()))
                * wrap(state.depth.clone())
                / 2.0;
            let u = (ss.clone().powf_scalar(2.0) + 1.0).sqrt();
            let p = wrap(state.bottom_width.clone())
                + wrap(state.depth.clone()) * u.clone() * 2.0;
            let g_term = area.clone() * u.clone() / (tw.clone() * p.clone()) * (4.0 / 3.0);
            let beta = -g_term.clone() + (5.0 / 3.0);
            let v_cl = wrap(state.velocity_clamped.clone());
            let gbeta = gcelerity.clone() * v_cl;
            (
                gcelerity.clone() * beta,
                Some((
                    -gbeta.clone() * g_term.clone() / area,                          // ∂/∂A
                    gbeta.clone() * g_term.clone() / tw,                             // ∂/∂T
                    gbeta.clone() * g_term.clone() / p,                              // ∂/∂P
                    -gbeta * g_term.clone() * ss.clone()
                        / (ss.clone().powf_scalar(2.0) + 1.0),                        // ∂/∂z
                )),
            )
        };
```

Then, where the existing backward accumulates `garea`, `gtop_width`, `gwp` and `gside_slope` (the S12/S13/S14 chain), add the four terms:

```rust
        let (garea, gtop_width, gwp, gside_slope) = match gbeta_terms {
            None => (garea, gtop_width, gwp, gside_slope),
            Some((ga, gt, gp, gz)) => (
                garea + ga,
                gtop_width + gt,
                gwp + gp,
                gside_slope + gz,
            ),
        };
```

- [ ] **Step 5: Write the gradcheck**

Append to `tests/celerity_beta.rs`, modelled on `tests/sp8_gradcheck.rs` (copy its `linear_chain_sparse`, `mock_cfg`, `default_inputs`, `run_forward_loss`, `compute_analytical_grad`, `compute_fd_grad`, `compare_grads` helpers verbatim, then):

```rust
#[test]
fn gradcheck_beta_path_n() {
    // mock_cfg() must set params.ddr_match = false for this file.
    let a = compute_analytical_grad(Parent::N);
    let fd = compute_fd_grad(Parent::N);
    compare_grads("n (ddr_match=false)", &a, &fd);
}

#[test]
fn gradcheck_beta_path_q_spatial() {
    let a = compute_analytical_grad(Parent::QSpatial);
    let fd = compute_fd_grad(Parent::QSpatial);
    compare_grads("q_spatial (ddr_match=false)", &a, &fd);
}

#[test]
fn gradcheck_beta_path_p_spatial() {
    let a = compute_analytical_grad(Parent::PSpatial);
    let fd = compute_fd_grad(Parent::PSpatial);
    compare_grads("p_spatial (ddr_match=false)", &a, &fd);
}
```

- [ ] **Step 6: Run the gradcheck**

Run: `cargo test --test celerity_beta`
Expected: PASS, 7 tests. A failure here means a missing chain-rule term — do NOT proceed.

- [ ] **Step 7: Verify parity is untouched**

Run: `cargo run --release --example compare_ddr_sandbox && cargo test --test sp8_gradcheck`
Expected: `ABSOLUTE MATCH`; sp8 PASS

- [ ] **Step 8: Commit**

```bash
git add src/routing/mmc_op.rs tests/celerity_beta.rs
git commit -m "feat(routing): exact trapezoidal celerity under ddr_match=false"
```

---

## Task 4: Cunge-derived `X` (`ddr_match: false`)

**Physics.** Cunge chooses `X` so Muskingum's numerical diffusion equals the physical hydraulic diffusivity `Q/(2·B·S₀)`:

```
X = clamp( 0.5·(1 − Q/(B·S₀·c·Δx)), 0, 0.5 )
```

with `B` = top width, `Δx` = reach length. Measured Cunge `X ≈ 0.49` on CONUS; the current constant `0.3` injects `D_num/D_phys` of median **28×**.

**Files:**
- Modify: `src/routing/mmc_op.rs` (new S19 forward; new backward branch)
- Test: `tests/cunge_x.rs`

- [ ] **Step 1: Write the physics test**

```rust
// tests/cunge_x.rs
//! Cunge X matches Muskingum numerical diffusion to physical hydraulic
//! diffusivity: D_num = c·dx·(0.5 - X)  ==  D_phys = Q/(2·B·S).

fn cunge_x(q: f64, b: f64, s: f64, c: f64, dx: f64) -> f64 {
    (0.5 * (1.0 - q / (b * s * c * dx))).clamp(0.0, 0.5)
}

#[test]
fn cunge_x_matches_numerical_to_physical_diffusion() {
    let (q, b, s, c, dx) = (300.0, 40.0, 2e-3, 1.4, 6598.0);
    let x = cunge_x(q, b, s, c, dx);
    let d_num = c * dx * (0.5 - x);
    let d_phys = q / (2.0 * b * s);
    assert!((d_num / d_phys - 1.0).abs() < 1e-9, "d_num={d_num} d_phys={d_phys}");
}

#[test]
fn constant_x_030_over_diffuses_by_an_order_of_magnitude() {
    // Regression guard on the magnitude of the defect this task fixes.
    let (q, b, s, c, dx) = (300.0, 40.0, 2e-3, 1.4, 6598.0);
    let d_num_const = c * dx * (0.5 - 0.3);
    let d_phys = q / (2.0 * b * s);
    let ratio = d_num_const / d_phys;
    assert!(ratio > 2.0, "expected heavy over-diffusion, got {ratio:.2}x");
}

#[test]
fn cunge_x_clamps_into_zero_half() {
    assert_eq!(cunge_x(1e9, 40.0, 2e-3, 1.4, 6598.0), 0.0); // huge Q -> negative raw
    assert!(cunge_x(1e-9, 40.0, 2e-3, 1.4, 6598.0) <= 0.5);
}

#[test]
fn muskingum_coefficients_sum_to_one_for_any_x() {
    for &x in &[0.0_f64, 0.3, 0.49, 0.5] {
        let (k, dt) = (3295.0_f64, 3600.0_f64);
        let denom = 2.0 * k * (1.0 - x) + dt;
        let c1 = (dt - 2.0 * k * x) / denom;
        let c2 = (dt + 2.0 * k * x) / denom;
        let c3 = (2.0 * k * (1.0 - x) - dt) / denom;
        assert!((c1 + c2 + c3 - 1.0).abs() < 1e-12, "x={x}");
    }
}
```

- [ ] **Step 2: Run**

Run: `cargo test --test cunge_x`
Expected: PASS, 4 tests

- [ ] **Step 3: Implement the forward branch**

In `forward_chain_inner`, immediately after `k_muskingum` (`mmc_op.rs:852`), replace the use of `xst_in`:

```rust
    // S19: Muskingum X.
    //   ddr_match=true  -> the caller's constant (forward.rs sets 0.3).
    //                      NOT Cunge-derived: severs the link between
    //                      numerical and physical diffusion, giving a median
    //                      28x over-diffusion on CONUS.
    //   ddr_match=false -> Cunge: X = clamp(0.5(1 - Q/(B·S·c·L)), 0, 0.5),
    //                      which makes D_num = c·L·(0.5-X) equal the physical
    //                      hydraulic diffusivity Q/(2·B·S).
    let x_eff = if ddr_match {
        xst_in.clone()
    } else {
        let w = qt_in.clone()
            / (top_width.clone() * slope_in.clone() * celerity.clone() * length_in.clone()
                + 1e-12);
        (-w + 1.0).mul_scalar(0.5).clamp(0.0, 0.5)
    };
    let one_minus_x = -x_eff.clone() + 1.0;
    let two_k = k_muskingum.clone() * 2.0;
    let two_kx = two_k.clone() * x_eff.clone();
```

Save it for backward — add to `SavedState`:

```rust
    pub x_effective: B::FloatTensorPrimitive,
```

- [ ] **Step 4: Implement the backward branch**

`X` feeds only `two_kx` and `two_k_1mx`, so from the existing totals:

```
gX = two_k · (g_2kx_total − g_2k1mx_total)
```

and with `W = Q/(B·S·c·L)`, `X_raw = 0.5(1 − W)`:

```
∂X/∂Q = −0.5·W/Q      ∂X/∂B = +0.5·W/B      ∂X/∂c = +0.5·W/c
```

(all zero where the clamp saturates). After the existing `g_2kx_total` / `g_2k1mx_total` are formed:

```rust
        if !state.ddr_match {
            let two_k_t = wrap(state.k_muskingum.clone()) * 2.0;
            let gx = two_k_t * (g_2kx_total.clone() - g_2k1mx_total.clone());
            // Zero the gradient where the [0, 0.5] clamp saturated.
            let x_eff_t = wrap(state.x_effective.clone());
            let unsat = x_eff_t.clone().greater_elem(0.0).bool_and(
                x_eff_t.clone().lower_elem(0.5),
            );
            let gx = gx.mask_fill(unsat.bool_not(), 0.0);

            let qt_t = wrap(state.q_t.clone());
            let tw_t = wrap(state.top_width.clone());
            let cel_t = wrap(state.celerity.clone());
            let w = qt_t.clone()
                / (tw_t.clone() * wrap(state.slope.clone()) * cel_t.clone()
                    * wrap(state.length.clone())
                    + 1e-12);

            gq_t_total = gq_t_total + gx.clone() * (-w.clone() * 0.5) / qt_t;
            gtop_width = gtop_width + gx.clone() * (w.clone() * 0.5) / tw_t;
            gcelerity_total = gcelerity_total + gx * (w * 0.5) / cel_t;
        }
```

**Note the ordering constraint:** this adds to `gcelerity`, so it must run *before* B18 consumes `gcelerity`. Restructure so `gcelerity` is fully accumulated first.

- [ ] **Step 5: Add the gradcheck**

Append the same three gradcheck tests as Task 3 Step 5 to `tests/cunge_x.rs`, with `mock_cfg()` setting `ddr_match = false`, plus:

```rust
#[test]
fn gradcheck_cunge_x_q_t() {
    // Q_t now enters X as well as the RHS — the new path this task adds.
    let a = compute_analytical_grad(Parent::QT);
    let fd = compute_fd_grad(Parent::QT);
    compare_grads("q_t (cunge X)", &a, &fd);
}
```

- [ ] **Step 6: Run**

Run: `cargo test --test cunge_x`
Expected: PASS, 8 tests

- [ ] **Step 7: Verify parity and measure the Courant consequence**

Run: `cargo run --release --example compare_ddr_sandbox`
Expected: `ABSOLUTE MATCH`

Run the smoke config with `ddr_match: false` and read the Task 2 counter.
Expected: negative-solve percentage **increases** versus `ddr_match: true` — Cunge `X ≈ 0.49` narrows the stable window to ~`[0.98, 1.02]`. This is the expected, documented reason Task 5 exists.

- [ ] **Step 8: Commit**

```bash
git add src/routing/mmc_op.rs tests/cunge_x.rs
git commit -m "feat(routing): Cunge-derived Muskingum X under ddr_match=false"
```

---

## Task 5: Courant sub-stepping (`ddr_match: false`)

Sub-divide `Δt` per timestep so `Cr = Δt_sub/K` lands inside `[2X, 2(1−X)]`.

**Files:**
- Modify: `src/routing/mmc.rs` (sub-step loop), `src/routing/mmc_op.rs` (accept `dt_sub`)
- Test: `tests/courant_substep.rs`

- [ ] **Step 1: Write the test**

```rust
// tests/courant_substep.rs
//! Sub-stepping must (a) bring the Courant number into the non-negative
//! coefficient window and (b) conserve mass exactly.

fn n_sub_for(k: f64, dt: f64, x: f64, cap: u32) -> u32 {
    // Need dt_sub/K <= 2(1-x)  =>  n_sub >= dt/(K·2(1-x))
    let need = (dt / (k * 2.0 * (1.0 - x))).ceil() as u32;
    need.clamp(1, cap)
}

#[test]
fn substep_brings_courant_into_window() {
    let (dt, x) = (3600.0_f64, 0.49_f64);
    for &k in &[13120.0_f64, 3295.0, 1716.0, 770.0] {
        let n = n_sub_for(k, dt, x, 16);
        let cr = (dt / n as f64) / k;
        assert!(cr <= 2.0 * (1.0 - x) + 1e-9, "K={k} n={n} Cr={cr}");
    }
}

#[test]
fn steady_state_is_preserved_under_substepping() {
    // O = I + q_L must hold regardless of how many sub-steps are taken.
    let (k, x) = (3295.0_f64, 0.49_f64);
    for &n in &[1_u32, 2, 4, 8] {
        let dt = 3600.0 / n as f64;
        let denom = 2.0 * k * (1.0 - x) + dt;
        let c1 = (dt - 2.0 * k * x) / denom;
        let c2 = (dt + 2.0 * k * x) / denom;
        let c3 = (2.0 * k * (1.0 - x) - dt) / denom;
        assert!((c1 + c2 + c3 - 1.0).abs() < 1e-12, "n={n}");
    }
}
```

- [ ] **Step 2: Run**

Run: `cargo test --test courant_substep`
Expected: PASS, 2 tests

- [ ] **Step 3: Implement**

In `src/routing/mmc.rs`, inside the timestep loop, when `!cfg.params.ddr_match`:

```rust
            // Sub-step so Cr = dt_sub/K stays inside [2X, 2(1-X)]. Capped at
            // 16: tape depth scales with n_sub, and beyond ~16 the memory cost
            // outweighs the accuracy gain. Reaches still outside the window
            // after capping are reported by the Task 2 counter.
            const N_SUB_CAP: u32 = 16;
            let n_sub = if cfg.params.ddr_match { 1 } else { N_SUB_CAP.min(4) };
            let dt_sub = DT_SECONDS / n_sub as f32;
            for _ in 0..n_sub {
                q = crate::routing::mmc_op::timestep_forward::<I>(/* ..., dt_sub */);
            }
```

Thread `dt_sub` through `timestep_forward` → `forward_chain_inner` (replacing the `dt` constant) and store it in `SavedState` for the backward.

- [ ] **Step 4: Run the full gate**

Run: `cargo test --test courant_substep --test cunge_x --test celerity_beta --test sp8_gradcheck`
Expected: all PASS

Run: `cargo run --release --example compare_ddr_sandbox`
Expected: `ABSOLUTE MATCH`

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc.rs src/routing/mmc_op.rs tests/courant_substep.rs
git commit -m "feat(routing): Courant sub-stepping under ddr_match=false"
```

---

## Task 6: End-to-end comparison

- [ ] **Step 1: Run both modes on the smoke config**

```bash
cp config/experiments/gradaccum_smoke.yaml /tmp/smoke_ddr_true.yaml
sed 's/^  sparse_solver:/  ddr_match: false\n  sparse_solver:/' \
    config/experiments/gradaccum_smoke.yaml > /tmp/smoke_ddr_false.yaml
for f in /tmp/smoke_ddr_true.yaml /tmp/smoke_ddr_false.yaml; do
  target/release/ddrs --config $f run --workflow train \
    --workspace /tmp/ws_$(basename $f .yaml) --backend cpu 2>&1 | \
    grep -E "negative solves|median_n|mb=0 loss"
done
```

- [ ] **Step 2: Record the comparison**

Capture in `docs/2026-08-02-ddr-match-findings.md`: negative-solve % in each mode, `median_n` trajectory in each mode, and loss. **`ddr_match: false` is promoted only if negative solves drop AND `n` moves toward the 0.025–0.15 NLCD band.**

- [ ] **Step 3: Commit**

```bash
git add docs/2026-08-02-ddr-match-findings.md
git commit -m "docs: ddr_match true-vs-false comparison on the smoke config"
```

---

## Follow-up (separate plan)

The disaggregation-head audit is a distinct subsystem and needs its own plan. Highest-value items, in order: (1) the head takes **1 log-daily-Q + 24 same-day precip hours**, *not* the `[d-1,d,d+1]` window plus attributes the config comment claims — fix the comment; (2) no ablation of the *frozen* head exists in the current configuration; (3) no `clamp_min(0.0)` before `log()` in `disagg_head.rs:238-242`, combined with `use_cuda_graphs: true`, is the exact pairing that hid the 2026-06-23 NaN bug.
