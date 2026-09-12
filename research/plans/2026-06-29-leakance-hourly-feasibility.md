# Leakance (water-loss term) × hourly-disaggregation feasibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a flag-gated, gradient-exact leakance (GW–SW water-loss) term to ddrs's fused MC timestep, then run the pre-registered 2×2 (forcing × leakance) experiment to decide whether the hourly disaggregation signal makes leakance identifiable and helpful.

**Architecture:** A parallel custom autodiff op `TimestepLeakanceOp` (`Backward<I, 8>`) extends the existing fused `TimestepOp` (`Backward<I, 5>`) with three new parents (`K_D, d_gw, leakance_factor`) and the DDR `zeta` term subtracted from `b_rhs`. The shared S1..S28 chain is reused via an `Option`-gated `forward_chain_inner`; the existing 5-parent op and the `triangular_csr_solve` backward are untouched. Leakance is off by default and forces `use_cuda_graphs: false`. Parameters come from the KAN head automatically by listing them in `kan_head.learnable_parameters`.

**Tech Stack:** Rust, BURN 0.21 (Autodiff custom ops), rskan, ndarray, serde_yaml. f32 throughout the routing core.

**Spec:** `docs/superpowers/specs/2026-06-29-leakance-hourly-feasibility-design.md`

**DDR reference (read first):** `~/projects/ddr` commit `c2bd0f9`, `src/ddr/routing/mmc.py` `_compute_zeta` (lines 146–197) and `route_timestep` (the `b = ... - zeta` branch). View with:
`git -C ~/projects/ddr show c2bd0f9:src/ddr/routing/mmc.py | sed -n '146,197p'`

---

## The math (single source of truth for every task)

DDR `_compute_zeta`, reusing ddrs's **saved** `depth` (verified identical power law):

```
q_eps   = q_spatial + 1e-6                         # ddrs S1 (shared)
depth   = ( q_t·n·(q_eps+1) / (p·√s0 + 1e-8) )^(3/(5+3·q_eps))   # ddrs S6 (shared, clamped ≥ depth_lb)
width_z = (p · depth)^q_eps                        # NEW (plan-view; NOT the trapezoidal top_width)
area_z  = width_z · length                         # NEW
zeta    = leakance_factor · area_z · K_D · (depth − d_gw)        # NEW
b_rhs   = c2·i_t + c3·q_t + c4·q_prime_t − zeta     # ddrs S25 with −zeta
```

Forward solve and clamp are unchanged: `x_sol = solve(A, b_rhs)`, `q_next = max(x_sol, discharge_lb)`.

**Analytical gradients** (let `g_b = gb_rhs`, the upstream grad of `b_rhs`; `m = depth − d_gw`):

```
gzeta            = -g_b                                   # b_rhs = ... − zeta
g_leakance_factor = gzeta · area_z · K_D · m
g_K_D             = gzeta · leakance_factor · area_z · m
g_d_gw            = gzeta · leakance_factor · area_z · K_D · (−1)

# zeta depends on p_spatial and q_eps only through area_z = (p·depth)^q_eps · length:
#   ∂area_z/∂p     = area_z · q_eps / p
#   ∂area_z/∂q_eps = area_z · ln(p·depth)
common            = gzeta · leakance_factor · K_D · m     # = ∂zeta/∂area_z
g_p_from_zeta     = common · area_z · q_eps / p
g_qeps_from_zeta  = common · area_z · (p·depth).ln()

# zeta depends on depth two ways (direct m, and through area_z):
#   ∂zeta/∂depth = factor·K_D·area_z  +  factor·K_D·m · (area_z · q_eps / depth)
g_depth_from_zeta = gzeta · leakance_factor · K_D · ( area_z + m · area_z · q_eps / depth )
```

`g_depth_from_zeta` is **added into `gd_total`** (mmc_op.rs:410) so it flows through the existing depth→ratio→{q_t, n, p} chain. `g_p_from_zeta` is added to `gp_total` (mmc_op.rs:467). `g_qeps_from_zeta` is added to `gq_spatial` (mmc_op.rs:461). The three new parent grads are registered on the 3 new parent nodes.

---

## File structure

| Path | Create/Modify | Responsibility |
|---|---|---|
| `src/config.rs` | Modify | `Params.use_leakance` flag; `ParameterRanges` gains `k_d`/`d_gw`/`leakance_factor`; `ParamsRaw` mapping; reject `use_cuda_graphs && use_leakance` |
| `src/routing/leakance.rs` | Create | Pure `zeta`/`width_z`/`area_z` forward helper + its analytical grad helper, both on inner-backend `Tensor<I,1>`; unit-tested in isolation |
| `src/routing/mmc_op.rs` | Modify | `Option`-gated `forward_chain_inner`; `TimestepLeakanceState`; `TimestepLeakanceOp` (`Backward<I,8>`); `timestep_forward_leakance` |
| `src/routing/mmc.rs` | Modify | `SpatialParameters` gains optional `k_d/d_gw/leakance_factor`; `setup_inputs` denormalizes them; `route_timestep` dispatches to the leakance op when present |
| `src/routing/mod.rs` | Modify | `pub mod leakance;` |
| `src/training/forward.rs` | Modify | Thread `K_D/d_gw/leakance_factor` from head HashMap into `SpatialParameters` (all 3 build sites) |
| `tests/leakance_gradcheck.rs` | Create | Finite-difference gradcheck of all new grads (mirrors `tests/sp8_gradcheck.rs`) |
| `tests/leakance_off_parity.rs` | Create | `use_leakance=false` ⇒ identical routed output vs current code |
| `config/sources/conus-hourly.yaml` + run configs | Reference | Experiment configs (Task 12) |
| `scripts/leakance_subset_analysis.py` | Create | Losing-stream subset slice + go/no-go metrics (Task 13) |

---

## Phase 1 — Config plumbing

### Task 1: `use_leakance` flag + leakance parameter ranges

**Files:**
- Modify: `src/config.rs` (`ParameterRanges`, `Params`, `ParamsRaw`, `From<ParamsRaw>`)
- Test: `src/config.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/config.rs`:

```rust
#[test]
fn leakance_flag_and_ranges_parse() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
params:
  use_leakance: true
  parameter_ranges:
    K_D: [1.0e-8, 1.0e-6]
    d_gw: [-2.0, 2.0]
    leakance_factor: [0.0, 1.0]
  log_space_parameters: [p_spatial, K_D]
"#;
    let path = std::env::temp_dir().join("ddrs_leakance_cfg.yaml");
    std::fs::write(&path, yaml).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("load yaml");
    assert!(cfg.params.use_leakance);
    assert!((cfg.params.parameter_ranges.k_d[0] - 1e-8).abs() < 1e-12);
    assert!((cfg.params.parameter_ranges.k_d[1] - 1e-6).abs() < 1e-12);
    assert_eq!(cfg.params.parameter_ranges.d_gw, [-2.0, 2.0]);
    assert_eq!(cfg.params.parameter_ranges.leakance_factor, [0.0, 1.0]);
    assert!(cfg.params.log_space_parameters.iter().any(|s| s == "K_D"));
}

#[test]
fn use_leakance_defaults_false() {
    assert!(!Params::default().use_leakance);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::tests::leakance_flag_and_ranges_parse`
Expected: FAIL — `no field use_leakance` / `no field k_d`.

- [ ] **Step 3: Implement**

In `src/config.rs`, extend `ParameterRanges` (the struct near line 276) with three fields and defaults:

```rust
pub struct ParameterRanges {
    pub n: [f32; 2],
    pub q_spatial: [f32; 2],
    pub p_spatial: [f32; 2],
    pub x_storage: [f32; 2],
    /// Leakance (GW–SW) ranges — verbatim from DDR `configs.py` (commit c2bd0f9).
    /// Only consumed when `Params.use_leakance` and the params are listed in
    /// `kan_head.learnable_parameters`.
    pub k_d: [f32; 2],
    pub d_gw: [f32; 2],
    pub leakance_factor: [f32; 2],
}
```

In `impl Default for ParameterRanges`:

```rust
            x_storage: [0.0, 0.5],
            k_d: [1e-8, 1e-6],
            d_gw: [-2.0, 2.0],
            leakance_factor: [0.0, 1.0],
```

Add `use_leakance` to `Params` (struct near line 309) after `use_cuda_graphs`:

```rust
    /// Enable the leakance (GW–SW water-loss) term in routing. Off by default;
    /// when on, `K_D`/`d_gw`/`leakance_factor` must be in
    /// `kan_head.learnable_parameters`, and `use_cuda_graphs` must be false.
    pub use_leakance: bool,
```

In `impl Default for Params`, add `use_leakance: false,`.

In `ParamsRaw` (near line 344) add `use_leakance: Option<bool>,`.

In `From<ParamsRaw> for Params`, map the new ranges and flag (after the existing `x_storage` mapping and the `use_cuda_graphs` block):

```rust
        if let Some(v) = r.parameter_ranges.get("K_D") {
            p.parameter_ranges.k_d = *v;
        }
        if let Some(v) = r.parameter_ranges.get("d_gw") {
            p.parameter_ranges.d_gw = *v;
        }
        if let Some(v) = r.parameter_ranges.get("leakance_factor") {
            p.parameter_ranges.leakance_factor = *v;
        }
        if let Some(b) = r.use_leakance {
            p.use_leakance = b;
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib config::tests::leakance_flag_and_ranges_parse config::tests::use_leakance_defaults_false`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "feat(config): use_leakance flag + leakance parameter ranges (off by default)"
```

### Task 2: Reject `use_cuda_graphs && use_leakance` at load

**Files:**
- Modify: `src/config.rs` (`from_yaml_file_with_mode` validation + a new `validate_leakance` fn)
- Test: `src/config.rs` inline

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn leakance_with_cuda_graphs_rejected() {
    let yaml = r#"
mode: training
geodataset: merit
seed: 1
np_seed: 1
params:
  use_leakance: true
  use_cuda_graphs: true
"#;
    let path = std::env::temp_dir().join("ddrs_leakance_graphs.yaml");
    std::fs::write(&path, yaml).unwrap();
    let err = Config::from_yaml_file(&path).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("use_leakance") && msg.contains("use_cuda_graphs"),
        "expected leakance/graphs conflict, got: {msg}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib config::tests::leakance_with_cuda_graphs_rejected`
Expected: FAIL — config loads instead of erroring.

- [ ] **Step 3: Implement**

Add a validator and call it in `from_yaml_file_with_mode` right after `validate_data_sources(&cfg)`:

```rust
fn validate_leakance(cfg: &Config) -> std::result::Result<(), String> {
    if cfg.params.use_leakance && cfg.params.use_cuda_graphs {
        return Err(
            "params: `use_leakance: true` requires `use_cuda_graphs: false` — the \
             CUDA-graph capture path bakes the non-leakance b_rhs into the graph."
                .to_string(),
        );
    }
    Ok(())
}
```

Call site (mirror the existing `.map_err(... DataError::Yaml ...)` pattern):

```rust
        validate_leakance(&cfg).map_err(|msg| DataError::Yaml {
            path: path.to_path_buf(),
            source: serde_yaml::Error::custom(msg),
        })?;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib config::tests::leakance_with_cuda_graphs_rejected`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "feat(config): reject use_leakance with use_cuda_graphs"
```

---

## Phase 2 — The zeta forward + its gradient, in isolation

Building `zeta` and its analytical grad as a standalone, unit-tested module **before** wiring it into the fused op de-risks the hardest part (the derivation) away from the autograd machinery.

### Task 3: `leakance::zeta_forward` (pure inner-backend op)

**Files:**
- Create: `src/routing/leakance.rs`
- Modify: `src/routing/mod.rs` (add `pub mod leakance;`)
- Test: `src/routing/leakance.rs` inline

- [ ] **Step 1: Write the failing test**

Create `src/routing/leakance.rs`:

```rust
//! Leakance (GW–SW water-loss) term `zeta`, ported from DDR `_compute_zeta`
//! (`~/projects/ddr/src/ddr/routing/mmc.py:146-197`, commit c2bd0f9).
//!
//! `zeta = leakance_factor · area_z · K_D · (depth − d_gw)`, where
//! `width_z = (p·depth)^q_eps`, `area_z = width_z · length`, and `depth` is the
//! SHARED power-law depth already computed by `forward_chain_inner` (S6).
//! Subtracted from `b_rhs`. Positive ⇒ losing stream. All ops are plain inner-
//! backend `Tensor<I,1>` (no autograd tape).

use burn::tensor::{backend::Backend, Tensor};

/// `(width_z, area_z, zeta)` from the shared `depth` and the three leakance
/// params. `q_eps = q_spatial + 1e-6` (consistency with the shared depth).
pub fn zeta_forward<I: Backend>(
    depth: Tensor<I, 1>,
    p_spatial: Tensor<I, 1>,
    q_eps: Tensor<I, 1>,
    length: Tensor<I, 1>,
    k_d: Tensor<I, 1>,
    d_gw: Tensor<I, 1>,
    leakance_factor: Tensor<I, 1>,
) -> (Tensor<I, 1>, Tensor<I, 1>, Tensor<I, 1>) {
    let p_depth = p_spatial * depth.clone();
    let width_z = p_depth.powf(q_eps);
    let area_z = width_z.clone() * length;
    let m = depth - d_gw;
    let zeta = leakance_factor * area_z.clone() * k_d * m;
    (width_z, area_z, zeta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    fn t(v: &[f32]) -> Tensor<B, 1> {
        Tensor::from_floats(v, &Default::default())
    }

    #[test]
    fn zeta_matches_hand_computed_value() {
        // depth=2, p=10, q_eps=0.5, length=1000, K_D=1e-6, d_gw=1, factor=0.5
        // width_z = (10·2)^0.5 = sqrt(20) = 4.472136
        // area_z  = 4.472136·1000 = 4472.136
        // m       = 2−1 = 1
        // zeta    = 0.5·4472.136·1e-6·1 = 0.002236068
        let (w, a, z) = zeta_forward::<B>(
            t(&[2.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[1.0]), t(&[0.5]),
        );
        assert!((w.into_scalar() - 4.472_136).abs() < 1e-4);
        assert!((a.into_scalar() - 4472.136).abs() < 1e-1);
        assert!((z.into_scalar() - 0.002_236_068).abs() < 1e-7);
    }

    #[test]
    fn gaining_stream_is_negative() {
        // depth < d_gw ⇒ m < 0 ⇒ zeta < 0 (gaining stream).
        let (_, _, z) = zeta_forward::<B>(
            t(&[1.0]), t(&[10.0]), t(&[0.5]), t(&[1000.0]),
            t(&[1e-6]), t(&[3.0]), t(&[1.0]),
        );
        assert!(z.into_scalar() < 0.0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

First add `pub mod leakance;` to `src/routing/mod.rs`. Then:
Run: `cargo test --lib routing::leakance::tests`
Expected: FAIL until the module compiles + asserts pass — run it to confirm it builds and the hand-computed values match (this step is also the numeric check).

- [ ] **Step 3: Implement** — already written in Step 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib routing::leakance::tests`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/routing/leakance.rs src/routing/mod.rs
git commit -m "feat(routing): leakance zeta_forward (DDR _compute_zeta port), unit-tested"
```

### Task 4: `leakance::zeta_backward` (analytical grads) + finite-diff check

**Files:**
- Modify: `src/routing/leakance.rs`
- Test: `src/routing/leakance.rs` inline (finite-difference)

- [ ] **Step 1: Write the failing test**

Add to `src/routing/leakance.rs`:

```rust
/// Per-parent gradient contributions of `zeta`. `g_b` is ∂L/∂b_rhs; since
/// `b_rhs = … − zeta`, `gzeta = −g_b`. Returns grads for the three leakance
/// params plus zeta's contributions into `depth`, `p_spatial`, `q_eps`.
pub struct ZetaGrads<I: Backend> {
    pub g_k_d: Tensor<I, 1>,
    pub g_d_gw: Tensor<I, 1>,
    pub g_leakance_factor: Tensor<I, 1>,
    pub g_depth: Tensor<I, 1>,
    pub g_p_spatial: Tensor<I, 1>,
    pub g_q_eps: Tensor<I, 1>,
}

#[allow(clippy::too_many_arguments)]
pub fn zeta_backward<I: Backend>(
    g_b: Tensor<I, 1>,
    depth: Tensor<I, 1>,
    p_spatial: Tensor<I, 1>,
    q_eps: Tensor<I, 1>,
    area_z: Tensor<I, 1>,
    k_d: Tensor<I, 1>,
    d_gw: Tensor<I, 1>,
    leakance_factor: Tensor<I, 1>,
) -> ZetaGrads<I> {
    let gzeta = -g_b;
    let m = depth.clone() - d_gw;
    let g_leakance_factor = gzeta.clone() * area_z.clone() * k_d.clone() * m.clone();
    let g_k_d = gzeta.clone() * leakance_factor.clone() * area_z.clone() * m.clone();
    let g_d_gw = -(gzeta.clone() * leakance_factor.clone() * area_z.clone() * k_d.clone());
    let common = gzeta * leakance_factor * k_d; // = ∂zeta/∂area_z (× m below)
    let common_m = common.clone() * m.clone();
    let g_p_spatial = common_m.clone() * area_z.clone() * q_eps.clone() / p_spatial.clone();
    let g_q_eps = common_m.clone() * area_z.clone() * (p_spatial * depth.clone()).log();
    // ∂zeta/∂depth = factor·K_D·area_z (direct m) + factor·K_D·m·(area_z·q_eps/depth)
    let g_depth = common.clone() * area_z.clone()
        + common_m * area_z * q_eps / depth;
    ZetaGrads { g_k_d, g_d_gw, g_leakance_factor, g_depth, g_p_spatial, g_q_eps }
}

#[cfg(test)]
mod grad_tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    fn s(v: f32) -> Tensor<B, 1> { Tensor::from_floats(&[v][..], &Default::default()) }
    fn val(t: Tensor<B, 1>) -> f32 { t.into_scalar() }

    // Scalar zeta as a plain f64 closure for central differences.
    #[allow(clippy::too_many_arguments)]
    fn zeta_scalar(depth: f64, p: f64, q_eps: f64, length: f64, k_d: f64, d_gw: f64, factor: f64) -> f64 {
        let width_z = (p * depth).powf(q_eps);
        let area_z = width_z * length;
        factor * area_z * k_d * (depth - d_gw)
    }

    #[test]
    fn zeta_grads_match_central_differences() {
        // Base point.
        let (depth, p, q_eps, length, k_d, d_gw, factor) =
            (2.0_f64, 10.0, 0.5, 1000.0, 1e-6, 1.0, 0.5);
        let area_z = (p * depth).powf(q_eps) * length;
        // g_b = 1 ⇒ gzeta = −1, so analytical grads below are −∂zeta/∂param.
        let g = zeta_backward::<B>(
            s(1.0), s(depth as f32), s(p as f32), s(q_eps as f32),
            s(area_z as f32), s(k_d as f32), s(d_gw as f32), s(factor as f32),
        );
        let h = 1e-4;
        let cd = |f: &dyn Fn(f64) -> f64, x: f64| (f(x + h) - f(x - h)) / (2.0 * h);

        // Each analytical grad equals −∂zeta/∂param (because gzeta = −1).
        let d_kd = cd(&|x| zeta_scalar(depth, p, q_eps, length, x, d_gw, factor), k_d);
        assert!(((val(g.g_k_d) as f64) - (-d_kd)).abs() / d_kd.abs().max(1.0) < 1e-3);

        let d_dgw = cd(&|x| zeta_scalar(depth, p, q_eps, length, k_d, x, factor), d_gw);
        assert!(((val(g.g_d_gw) as f64) - (-d_dgw)).abs() < 1e-7);

        let d_fac = cd(&|x| zeta_scalar(depth, p, q_eps, length, k_d, d_gw, x), factor);
        assert!(((val(g.g_leakance_factor) as f64) - (-d_fac)).abs() / d_fac.abs() < 1e-3);

        let d_p = cd(&|x| zeta_scalar(depth, x, q_eps, length, k_d, d_gw, factor), p);
        assert!(((val(g.g_p_spatial) as f64) - (-d_p)).abs() / d_p.abs() < 1e-2);

        let d_q = cd(&|x| zeta_scalar(depth, p, x, length, k_d, d_gw, factor), q_eps);
        assert!(((val(g.g_q_eps) as f64) - (-d_q)).abs() / d_q.abs() < 1e-2);

        let d_depth = cd(&|x| zeta_scalar(x, p, q_eps, length, k_d, d_gw, factor), depth);
        assert!(((val(g.g_depth) as f64) - (-d_depth)).abs() / d_depth.abs() < 1e-2);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib routing::leakance::grad_tests`
Expected: FAIL — `zeta_backward` not defined.

- [ ] **Step 3: Implement** — already written in Step 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib routing::leakance::grad_tests`
Expected: PASS — every analytical grad within tolerance of central differences.

- [ ] **Step 5: Commit**

```bash
git add src/routing/leakance.rs
git commit -m "feat(routing): leakance zeta_backward analytical grads, finite-diff verified"
```

---

## Phase 3 — Wire zeta into the fused timestep op

### Task 5: `Option`-gate `forward_chain_inner` with leakance (None = byte-identical)

**Files:**
- Modify: `src/routing/mmc_op.rs`
- Test: `tests/leakance_off_parity.rs` (Create)

- [ ] **Step 1: Write the failing test**

Create `tests/leakance_off_parity.rs`:

**This test is a verbatim port of an existing passing test.** Open `tests/mmc.rs`, find the linear-chain routing test (e.g. `mc_routes_linear_chain`), and copy its entire body — network setup, `SpatialParameters`, `forward()` call, and the concrete expected-output assertion — into `leakance_none_matches_baseline_chain` in the new file. Do **not** change any value: leakance is absent (`k_d: None, …`), so the routed hydrograph must equal the existing test's committed expected values bit-for-bit. The file header:

```rust
//! `forward_chain_inner` with `leakance = None` must produce byte-identical
//! `q_next` to the pre-leakance code path. This is a verbatim copy of the
//! linear-chain assertion in `tests/mmc.rs`, run after the Phase-3 changes to
//! prove the leakance-off path is untouched.
```

The single concrete assertion (copied from `tests/mmc.rs`) is of the form
`assert!((routed[i] - EXPECTED[i]).abs() < 1e-6)` over the committed expected
hydrograph. There is no new expected-value computation — you are re-running an
existing fixture under the modified code.

- [ ] **Step 2: Run test to verify it fails (or is a faithful copy that passes pre-change)**

Run: `cargo test --test leakance_off_parity`
Expected: PASS on the unmodified code (this is the regression guard you must not break in Steps 3+).

- [ ] **Step 3: Implement the gate**

In `src/routing/mmc_op.rs`, add a small input struct near the top:

```rust
/// Inner-backend leakance inputs threaded into `forward_chain_inner`.
#[derive(Clone)]
pub(crate) struct LeakanceTensors<I: Backend> {
    pub k_d: Tensor<I, 1>,
    pub d_gw: Tensor<I, 1>,
    pub leakance_factor: Tensor<I, 1>,
}

/// Extra saved-state (inner-backend primitives) the leakance backward needs,
/// beyond what `TimestepState` already saves (`depth`, `p_spatial`, `q_eps` are
/// reused from there). This is the ONE leakance saved-state type — reused as the
/// `leak` field of `TimestepLeakanceState` below (do not introduce a second).
#[derive(Clone, Debug)]
pub(crate) struct LeakanceSaved<I: Backend> {
    pub area_z: I::FloatTensorPrimitive,
    pub k_d: I::FloatTensorPrimitive,
    pub d_gw: I::FloatTensorPrimitive,
    pub leakance_factor: I::FloatTensorPrimitive,
}
```

Change `forward_chain_inner`'s signature to take `leakance: Option<LeakanceTensors<I>>` and `leak_out: &mut Option<LeakanceSaved<I>>` as trailing args. Inside, **after S7 computes `top_width` and you have `depth`, `q_eps`, `psp_in`, `length_in`** (note `length_in` is already a param), and **before S25 builds `b_rhs`**, insert:

```rust
    // Leakance: subtract zeta from b_rhs when active. None ⇒ byte-identical.
    let zeta_opt = leakance.as_ref().map(|lk| {
        let (_w, area_z, zeta) = crate::routing::leakance::zeta_forward::<I>(
            depth.clone(), psp_in.clone(), q_eps.clone(), length_in.clone(),
            lk.k_d.clone(), lk.d_gw.clone(), lk.leakance_factor.clone(),
        );
        *leak_out = Some(LeakanceSaved {
            area_z: unwrap(area_z.clone()),
            k_d: unwrap(lk.k_d.clone()),
            d_gw: unwrap(lk.d_gw.clone()),
            leakance_factor: unwrap(lk.leakance_factor.clone()),
        });
        zeta
    });
```

Then change the S25 line to subtract zeta when present:

```rust
    let b_rhs_base =
        c2.clone() * i_t.clone() + c3.clone() * qt_in.clone() + c4.clone() * qpt_in.clone();
    let b_rhs = match zeta_opt {
        Some(zeta) => b_rhs_base - zeta,
        None => b_rhs_base,
    };
```

Update both existing call sites of `forward_chain_inner` (in `timestep_forward` ~line 1099 and the capture path) to pass `None, &mut None`. The `forward_chain_inner_pinned` variant (graph path) keeps `None` permanently — leakance never uses graphs.

- [ ] **Step 4: Run the parity test**

Run: `cargo test --test leakance_off_parity && cargo run --release --example compare_ddr_sandbox`
Expected: parity test PASS; sandbox reports **ABSOLUTE MATCH** (leakance-off path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc_op.rs tests/leakance_off_parity.rs
git commit -m "feat(routing): Option-gate forward_chain_inner with leakance (None byte-identical)"
```

### Task 6: `TimestepLeakanceState` + `TimestepLeakanceOp` forward

**Files:**
- Modify: `src/routing/mmc_op.rs`

- [ ] **Step 1: Add the test signature (body finished in Task 8)**

This op cannot be *invoked* until Task 8 wires `route_timestep` dispatch, so here only add the forward op + state so the crate compiles. The behavioural test `leakance_removes_water_on_losing_config` is **authored in Task 8 Step 1** (it needs the `SpatialParameters` leakance fields). Do not stub it with `assert!(true)` here — simply do not add it yet. This task's verification is compilation (Step 2) plus the Task-7 gradcheck.

- [ ] **Step 2: Run to verify current state**

Run: `cargo build`
Expected: compiles (the new op is added but not yet dispatched).

- [ ] **Step 3: Implement the state + op**

In `src/routing/mmc_op.rs` add:

```rust
#[derive(Clone, Debug)]
pub(crate) struct TimestepLeakanceState<I: Backend> {
    pub base: TimestepState<I>,
    pub leak: LeakanceSaved<I>, // the same type defined in Task 5 — do not duplicate
}

#[derive(Debug)]
pub(crate) struct TimestepLeakanceOp;
```

Add `timestep_forward_leakance` — a copy of `timestep_forward` (lines 1045–1184) with: (a) three extra `Tensor<Autodiff<I>,1>` params `k_d_at, d_gw_at, leakance_factor_at`; (b) it unwraps their autodiff nodes; (c) it calls `forward_chain_inner` with `Some(LeakanceTensors{…})` and a `&mut Option<LeakanceSaved>`; (d) it builds `TimestepLeakanceState { base, leak }`; (e) it prepares `TimestepLeakanceOp.prepare::<NoCheckpointing>([n,qsp,psp,qt,qpt,k_d,d_gw,leakance_factor])` (8 nodes). Backward is added in Task 7. Until then, stub `TimestepLeakanceOp`'s backward with `todo!()` is NOT allowed (it must compile and the parity test must not call it) — so implement Task 7's backward in the same change if compilation requires the trait impl. (Practically: do Tasks 6 and 7 as one commit because `prepare` requires the `Backward` impl to exist.)

- [ ] **Step 4: Build**

Run: `cargo build`
Expected: compiles once Task 7's `Backward<I,8>` impl is present (combine commits).

- [ ] **Step 5: Commit** — combined with Task 7.

### Task 7: `TimestepLeakanceOp::backward` (extend the analytical chain)

**Files:**
- Modify: `src/routing/mmc_op.rs`
- Test: `tests/leakance_gradcheck.rs` (Create)

- [ ] **Step 1: Write the failing test**

Create `tests/leakance_gradcheck.rs`, mirroring `tests/sp8_gradcheck.rs` (open it and copy its harness: build a tiny network on `Autodiff<NdArray>`, set `require_grad` on each parameter, run one `route_timestep`, sum the output, `.backward()`, then compare each parameter's autograd grad to a central-difference estimate). Add the three leakance parameters to the swept set:

```rust
// For each of {n, q_spatial, p_spatial, x_storage, K_D, d_gw, leakance_factor}:
//   analytical = param.grad(&grads)
//   numerical  = (loss(param+h) − loss(param−h)) / 2h   (forward-only reruns)
//   assert max rel err < 2e-2  (f32 finite-diff tolerance, matches sp8_gradcheck)
```

Use leakance params in the **interior** of their ranges (e.g. `K_D=5e-7`, `d_gw=0.0`, `leakance_factor=0.5`) and a losing config (`depth > d_gw`) so no clamp saturates.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test leakance_gradcheck`
Expected: FAIL — backward not implemented (grads are zero/None for the 3 new parents).

- [ ] **Step 3: Implement the backward**

Add `impl<I: Backend + 'static> Backward<I, 8> for TimestepLeakanceOp`. The body is the existing `TimestepOp::backward` (lines 82–488) with these changes:

1. Destructure 8 parents: `let [p_n, p_qsp, p_psp, p_qt, p_qpt, p_kd, p_dgw, p_fac] = ops.parents;`
2. Use `state.base.*` for all existing saved fields; wrap `area_z, k_d, d_gw, leakance_factor` from `state.leak`.
3. Compute `gzeta`/parent grads from `crate::routing::leakance::zeta_backward` using `gb_rhs` (already computed at line 169 as `gb_rhs`). The function returns `ZetaGrads` — but here operate on already-`wrap`ped inner tensors; either call `zeta_backward` with wrapped tensors (it is generic over `Backend = I`) or inline the six expressions. Calling it is cleaner:

```rust
        let zg = crate::routing::leakance::zeta_backward::<I>(
            gb_rhs.clone(),                 // g_b = ∂L/∂b_rhs
            depth.clone(), p_spatial.clone(), q_eps.clone(),
            wrap(state.leak.area_z.clone()),
            wrap(state.leak.k_d.clone()),
            wrap(state.leak.d_gw.clone()),
            wrap(state.leak.leakance_factor.clone()),
        );
```

   (`q_eps` is `state.base.q_eps`; bind it via `wrap(state.base.q_eps.clone())` near the other wraps.)

4. **Fold zeta's geometry contributions into the existing accumulators**, at the exact points noted in "The math":
   - At line 410, change `gd_total` to also add `zg.g_depth`.
   - At line 461, change `gq_spatial` to also add `zg.g_q_eps`.
   - At line 467, change `gp_total` to also add `zg.g_p_spatial`.
5. **Register the three new parents** after the existing five:

```rust
        if let Some(node) = p_kd { grads.register::<I>(node.id, unwrap(zg.g_k_d)); }
        if let Some(node) = p_dgw { grads.register::<I>(node.id, unwrap(zg.g_d_gw)); }
        if let Some(node) = p_fac { grads.register::<I>(node.id, unwrap(zg.g_leakance_factor)); }
```

To avoid duplicating ~400 lines, factor the shared body (lines 88–468, ending at the five accumulators `gn_total, gq_spatial, gp_total, gq_t_total, gq_prime_t`) into a private helper `fn timestep_backward_core<I>(state: &TimestepState<I>, grad_out, …) -> FiveGrads<I>` that BOTH ops call; the existing `TimestepOp::backward` becomes "call core, register 5". The leakance op calls core, then folds zeta into three of the five (the `depth`/`p`/`q_eps` corrections must happen **inside** core's depth chain, so pass an `Option<&ZetaGeom>` into core that injects `g_depth/g_p/g_q_eps` at lines 410/461/467). This extraction is guarded by the existing `compare_ddr_sandbox` + `sp8_gradcheck` tests.

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
cargo test --test leakance_gradcheck
cargo test --test sp8_gradcheck
cargo run --release --example compare_ddr_sandbox
```
Expected: leakance gradcheck PASS (all 7 params within tol); sp8_gradcheck PASS (core extraction didn't regress); sandbox ABSOLUTE MATCH.

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc_op.rs tests/leakance_gradcheck.rs tests/leakance_off_parity.rs
git commit -m "feat(routing): TimestepLeakanceOp Backward<I,8> with analytical zeta grads"
```

---

## Phase 4 — Thread leakance params from head to routing

### Task 8: `SpatialParameters` gains optional leakance params; `route_timestep` dispatches

**Files:**
- Modify: `src/routing/mmc.rs`
- Test: `tests/leakance_off_parity.rs` (finish the `leakance_removes_water_on_losing_config` test)

- [ ] **Step 1: Finish the failing test**

Author `leakance_removes_water_on_losing_config` in `tests/leakance_off_parity.rs` (deferred from Task 6, now that `SpatialParameters` carries the leakance fields): build the tiny network twice — once with `SpatialParameters { …, k_d: None, d_gw: None, leakance_factor: None }`, once with `Some` losing-config tensors (`depth > d_gw`, `factor>0`, `K_D>0`) — call `forward()`, and assert `sum(with_leakance) < sum(without)` and both finite.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test leakance_off_parity::leakance_removes_water_on_losing_config`
Expected: FAIL — `SpatialParameters` has no `k_d` field.

- [ ] **Step 3: Implement**

In `src/routing/mmc.rs`, extend `SpatialParameters` (line 48):

```rust
pub struct SpatialParameters<I: Backend> {
    pub n: Tensor<Autodiff<I>, 1>,
    pub q_spatial: Tensor<Autodiff<I>, 1>,
    pub p_spatial: Option<Tensor<Autodiff<I>, 1>>,
    /// Leakance params (all-or-nothing). Present ⇒ route via `TimestepLeakanceOp`.
    pub k_d: Option<Tensor<Autodiff<I>, 1>>,
    pub d_gw: Option<Tensor<Autodiff<I>, 1>>,
    pub leakance_factor: Option<Tensor<Autodiff<I>, 1>>,
}
```

Add three fields to `MuskingumCunge<I>` (`k_d/d_gw/leakance_factor: Option<Tensor<Autodiff<I>,1>>`), init `None` in `new`. In `setup_inputs`, after the existing denormalize block, denormalize when present:

```rust
        let ranges = &self.cfg.params.parameter_ranges;
        let log_space = &self.cfg.params.log_space_parameters;
        self.k_d = params.k_d.map(|t| denormalize(t, ranges.k_d, log_space.iter().any(|s| s == "K_D")));
        self.d_gw = params.d_gw.map(|t| denormalize(t, ranges.d_gw, log_space.iter().any(|s| s == "d_gw")));
        self.leakance_factor = params.leakance_factor
            .map(|t| denormalize(t, ranges.leakance_factor, log_space.iter().any(|s| s == "leakance_factor")));
```

In `route_timestep`, dispatch: if all three are `Some`, call `crate::routing::mmc_op::timestep_forward_leakance::<I>(…, k_d, d_gw, leakance_factor)`; else the existing path. (Leakance forces `use_cuda_graphs=false`, so only the non-graph branch needs the leakance variant.)

Update the three `SpatialParameters { … }` literals in `src/training/forward.rs` (lines 133, 207, 279) to add `k_d: None, d_gw: None, leakance_factor: None` for now (Task 9 fills them).

- [ ] **Step 4: Run tests**

Run: `cargo test --test leakance_off_parity && cargo test --test mmc`
Expected: PASS — water removed on losing config; existing mmc tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/routing/mmc.rs src/training/forward.rs tests/leakance_off_parity.rs
git commit -m "feat(routing): optional leakance params on SpatialParameters + route_timestep dispatch"
```

### Task 9: Populate leakance params from the KAN head HashMap

**Files:**
- Modify: `src/training/forward.rs` (the three build sites)
- Test: `tests/leakance_off_parity.rs` (head-driven smoke)

- [ ] **Step 1: Write the failing test**

Add a test that builds a KAN head with `learnable_parameters = [n, q_spatial, p_spatial, K_D, d_gw, leakance_factor]`, runs the training forward over the tiny network with `use_leakance=true`, and asserts the routed output is finite and differs from the `use_leakance=false` run. (Mirror an existing `src/training/forward.rs` test or `tests/` driver test for setup.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test leakance_off_parity head_driven`
Expected: FAIL — params not threaded (leakance has no effect).

- [ ] **Step 3: Implement**

In each of the three `forward.rs` build sites, after extracting `n_param`/`q_param`/`p_param`, pull the leakance keys when present (mirror the `x_storage` optional pattern at lines 180/244):

```rust
    let (k_d, d_gw, leakance_factor) = if cfg.params.use_leakance {
        (
            params_map.get("K_D").cloned(),
            params_map.get("d_gw").cloned(),
            params_map.get("leakance_factor").cloned(),
        )
    } else {
        (None, None, None)
    };
```

Then set those fields in the `SpatialParameters { … }` literal. For the non-autodiff site (the `Tensor<I,1>` eval path, ~line 207), match its tensor type as the existing code does for `x_storage`.

Add a guard: if `cfg.params.use_leakance` and any of the three keys is missing from `params_map`, `panic!` with a clear message naming the missing key (so a misconfigured `learnable_parameters` fails fast, not silently).

- [ ] **Step 4: Run tests**

Run: `cargo test --test leakance_off_parity && cargo test --lib training::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/forward.rs tests/leakance_off_parity.rs
git commit -m "feat(training): thread K_D/d_gw/leakance_factor from KAN head into routing"
```

### Task 10: Full-suite regression gate

**Files:** none (verification only)

- [ ] **Step 1: Run the parity-critical suite**

Run:
```
cargo test
cargo run --release --example compare_ddr_sandbox
cargo test --features fixtures --test kan_head_init_repro --test kan_head_init_parity --test kan_head_fixture_forward --test kan_head_fixture_backward
```
Expected: all green; sandbox **ABSOLUTE MATCH**; KAN parity intact (invariants #1, #5, #7).

- [ ] **Step 2: Commit (if any doc/CHANGELOG touch-ups)** — otherwise skip.

---

## Phase 5 — The experiment

### Task 11: Document the math in code + CLAUDE.md pointer

**Files:**
- Modify: `CLAUDE.md` (add a "Leakance (experimental, off by default)" subsection under Training objective / routing), `.claude/ARCHITECTURE.md` (one paragraph + the zeta equation)

- [ ] **Step 1:** Add a short subsection to `CLAUDE.md` documenting: the flag, that it forces `use_cuda_graphs:false`, the param ranges, that it's gated behind `learnable_parameters`, and the gradient-exactness guard (`tests/leakance_gradcheck.rs`). Reference the spec + this plan.
- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md .claude/ARCHITECTURE.md
git commit -m "docs: leakance term (experimental, off by default)"
```

### Task 12: Experiment run configs (the 2×2)

**Files:**
- Create: `config/experiments/leakance_hourly_on.yaml` (hourly precip-disagg + `use_leakance: true`, same seed as the 2026-06-23 precip+L1 run)
- Create: `config/experiments/leakance_daily_on.yaml` (flat repeat-24, no disagg block, + `use_leakance: true`)

Each config: `loss: l1`, `use_cuda_graphs: false`, and `kan_head.learnable_parameters` extended with `K_D, d_gw, leakance_factor`; `params.use_leakance: true`; `params.log_space_parameters` includes `K_D`.

- [ ] **Step 1:** Copy the precip+L1 config used by run `2026-06-23T02-49-12Z-conus-hourly` (find it under `.ddrs/runs/<id>/config.yaml`) into `leakance_hourly_on.yaml`; add the leakance keys above; keep the seed identical for paired batches.
- [ ] **Step 2:** Copy it to `leakance_daily_on.yaml`; remove the `kan_head.disaggregation` block (⇒ flat repeat-24 daily forcing); keep leakance keys + seed.
- [ ] **Step 3:** Validate both parse:

```bash
cargo run --release -- --config config/experiments/leakance_hourly_on.yaml plan --workflow train-and-test
cargo run --release -- --config config/experiments/leakance_daily_on.yaml plan --workflow train-and-test
```
Expected: both `plan` succeed (and reject if `use_cuda_graphs` accidentally true).

- [ ] **Step 4: Commit**

```bash
git add config/experiments/leakance_hourly_on.yaml config/experiments/leakance_daily_on.yaml
git commit -m "config: leakance 2x2 experiment run configs (hourly-on, daily-on)"
```

### Task 13: Run the experiment + losing-stream subset analysis

**Files:**
- Create: `scripts/leakance_subset_analysis.py`

- [ ] **Step 1: Run both ON cells**

```bash
cargo run --release -- --config config/experiments/leakance_hourly_on.yaml run --workflow train-and-test
cargo run --release -- --config config/experiments/leakance_daily_on.yaml  run --workflow train-and-test
```
Record both run IDs. The hourly-OFF and daily-OFF cells are the existing runs (2026-06-23 precip+L1; the 2026-06-19 trained daily routing run) — if no same-seed daily-OFF exists, run one paired daily-OFF alongside.

- [ ] **Step 2: Dump learned leakance params**

Use the existing `dump_parameters` path (writes `kan_parameters.nc`) to export per-COMID denormalized `K_D, d_gw, leakance_factor`. Then compute per-reach net `zeta` over the eval window using `leakance::zeta_forward`'s formula on the routed depth (a small offline reuse, or add a `--dump-zeta` diagnostic if simpler).

- [ ] **Step 3: Write the subset analysis script**

`scripts/leakance_subset_analysis.py` (run under DDR's uv venv) must:
1. Load each run's `predictions.f32` + `observations.f32` + gage IDs from the run manifests.
2. Define the **losing-stream subset**: gauges where the summed-Q′ baseline ratio `mean(pred_baseline)/mean(obs) > 1`.
3. Compute paired (ON−OFF) per-gauge **NSE, KGE, and KGE-β** on the subset, for both hourly and daily arms.
4. Compute the learned **net `|zeta|`** distribution (m³/s) per reach on the subset; report the fraction of reaches with `|zeta| > 0.01`.
5. Report the **go/no-go** verdict per the spec:
   - **GO** if (NSE or KGE improves on the subset, hourly ON−OFF) **and** `|zeta|>0.01 m³/s` on a meaningful set of reaches **and** the effect is absent/weaker in the daily arm.
   - **NO-GO** otherwise (sub-0.01 collapse under hourly, or no skill gain, or water-deletion-everywhere with no spatial coherence).
6. Print the `K_D` distribution vs its `[1e-8,1e-6]` bounds (floor pile-up diagnostic) and a spatial-coherence summary (losing reaches by HUC region).

- [ ] **Step 4: Run the analysis**

```bash
cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/leakance_subset_analysis.py \
    --hourly-on <run_id> --daily-on <run_id> \
    --hourly-off <run_id> --daily-off <run_id>
```
Expected: a printed GO/NO-GO verdict + the metric tables.

- [ ] **Step 5: Write up findings + commit**

Create `docs/2026-XX-XX-leakance-hourly-findings.md` (mirror the style of `docs/2026-06-23-precip-disaggregation-findings.md`): the 2×2 table, the subset deltas, the `|zeta|` distribution, the verdict, and the recommendation (full port vs drop).

```bash
git add scripts/leakance_subset_analysis.py docs/2026-XX-XX-leakance-hourly-findings.md
git commit -m "experiment: leakance x hourly 2x2 results + go/no-go verdict"
```

---

## Self-review notes

- **Spec coverage:** Part A (testbed) → Tasks 1–10; Part B (experiment) → Tasks 12–13; eval cohort + go/no-go → Task 13; risks (depth/area, K_D floor, fudge factor) → Tasks 4/7 (grad correctness), 13 (floor + coherence diagnostics).
- **Invariant guards:** #1 sandbox → Tasks 5/7/10; #4 sparse backward reused unchanged → Tasks 5/7 (only `mmc_op` analytical backward extended); #5/#7 KAN parity → Task 10 (head untouched).
- **The risk is concentrated in Task 7's backward.** It is fully de-risked by Task 4 (the same grads finite-diff-checked in isolation) and Task 7's autograd gradcheck. Do not skip either.
- **Tasks 6+7 ship as one commit** (BURN's `prepare` needs the `Backward` impl to exist).
