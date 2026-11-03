//! Gradient checks of the reservoir dam rows on `geometry_form: power_law`.
//! WET copies of `tests/reservoir_override.rs` (replace row: gradcheck and
//! exactly zero dam-row gradient) and `tests/reservoir_additive.rs` (one
//! additive row gradcheck). Tolerances and step sizes are the source files'.
//!
//! The dam row replaces (or adds to) K and X after the section, so the dam
//! row's celerity gradient is cut before `power_law.rs::section_backward`.
//! On the replace row the dam's `n`, `q_spatial` and `p_spatial` get exactly 0.
//!
//! Base points are the source files'. Each check asserts the analytical
//! gradient is nonzero (except the exact-zero dam row), so none is vacuous.
//!
//! FD step: the rule of `tests/power_law_gradcheck.rs::step_of`.
//! `h = max(h0, h_rt)`, with `h0 = 1e-3 * max(1, |x|)` and
//! `h_rt = 4 * eps_f32 * S / (REL_TOL * |g|)`, capped at
//! `STEP_CAP * max(1, |x|)` with `STEP_CAP = 0.1`. Here the loss is a weighted
//! sum over all steps and reaches, not one `q_next`, so the round-off scale
//! `S` is the sum over steps and reaches of `|w * q|` from the base-point
//! forward. A slot whose cap binds is printed as `(capped)` and fails the
//! check. A slot with an exactly zero analytical gradient (the replace-row dam)
//! uses `h0`, since there is no gradient to resolve.
//!
//! Kink-free check: before each FD, the clamp selections (depth floor,
//! velocity band, Cunge X band) of every reach on every step of the window must
//! be equal at `x - h`, `x` and `x + h`. Each step is read by running the
//! section on that step's `Q_t` from the stencil point's own routed series.
//! A stencil that crosses a switch is refused, not passed.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, DamRow, GeometryForm};
use ddrs::routing::mmc::DamRelease;
use ddrs::routing::mmc_op::__spike_forward_chain_k1_outputs;
use ddrs::routing::utils::denormalize;
use ddrs::sparse::CsrPattern;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 1e-4;
const EPS: f32 = 1e-3;
/// Largest FD step, as a fraction of `max(1, |x|)`.
const STEP_CAP: f32 = 0.1;

fn power_law_cfg() -> Config {
    let mut cfg = Config::default();
    cfg.params.geometry_form = GeometryForm::PowerLaw;
    cfg.params.ddr_match = false;
    cfg
}

#[derive(Clone)]
struct Params {
    n: Vec<f32>,
    q_spatial: Vec<f32>,
    p_spatial: Vec<f32>,
}

impl Params {
    fn uniform(n_reach: usize) -> Self {
        Self {
            n: vec![0.5; n_reach],
            q_spatial: vec![0.5; n_reach],
            p_spatial: vec![0.575; n_reach],
        }
    }
}

fn network(n_reach: usize, edges: &[(usize, usize)], long: Option<(usize, f32)>) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n_reach * n_reach];
    for &(up, down) in edges {
        assert!(down > up);
        dense[down * n_reach + up] = 1.0;
    }
    let mut length = vec![5000.0_f32; n_reach];
    if let Some((r, l)) = long {
        length[r] = l;
    }
    SparseAdjacency::from_dense(n_reach, &dense, length, vec![0.001; n_reach])
}

/// The 5-reach RAPID sandbox topology. Reach 3 is the only interior reach.
fn sandbox5() -> SparseAdjacency {
    network(5, &[(1, 3), (2, 3), (0, 4), (3, 4)], None)
}

struct Routed {
    q: Tensor<AB, 2>,
    n: Tensor<AB, 1>,
    q_spatial: Tensor<AB, 1>,
    p_spatial: Tensor<AB, 1>,
}

const DAM: usize = 3;
const UPSTREAM: usize = 1;
const DOWNSTREAM: usize = 4;
const GRAD_STEPS: usize = 24;
const DAM_T_DAYS: f32 = 1.2;

fn sandbox_q_prime(steps: usize, t_scale: f32) -> Vec<f32> {
    let base = [22.0_f32, 11.0, 11.0, 11.0, 22.0];
    let mut q = Vec::with_capacity((steps + 1) * base.len());
    for t in 0..=steps {
        let s = (t as f32 / t_scale * std::f32::consts::PI).sin();
        let pulse = 1.0 + 1.5 * s * s;
        q.extend(base.iter().map(|b| b * pulse));
    }
    q
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    /// `set_reservoir_rows_as`: constant `T` (days), dam row form.
    OptionC(f32, DamRow),
    /// `set_dam_release`: `T0` (days), constant in time, additive row.
    Release(f32),
}

fn route(
    cfg: &Config,
    adjacency: SparseAdjacency,
    q_prime: &[f32],
    params: &Params,
    arm: Arm,
    require_grad: bool,
) -> Routed {
    let device = Device::default();
    let n_reach = adjacency.n;
    let n_rows = q_prime.len() / n_reach;
    let leaf = |v: &[f32]| {
        let t = Tensor::<AB, 1>::from_floats(v, &device);
        if require_grad { t.require_grad() } else { t }
    };
    let n = leaf(&params.n);
    let q_spatial = leaf(&params.q_spatial);
    let p_spatial = leaf(&params.p_spatial);
    let inputs = RoutingInputs::<I> {
        adjacency,
        x_storage: Tensor::ones([n_reach], &device) * 0.3,
    };
    let spatial = SpatialParameters::<I> {
        n: n.clone(),
        q_spatial: q_spatial.clone(),
        p_spatial: Some(p_spatial.clone()),
        k_d: None,
        d_gw: None,
        leakance_factor: None,
        impervious_mask: None,
        gamma: None,
    };
    let q_prime = Tensor::<AB, 1>::from_floats(q_prime, &device).reshape([n_rows, n_reach]);
    let mut mc = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    mc.setup_inputs(inputs, q_prime, spatial, false, None);
    match arm {
        Arm::OptionC(t, row) => mc.set_reservoir_rows_as(&[DAM], &[t], row).expect("rows"),
        Arm::Release(t0) => {
            let t0 = Tensor::<AB, 1>::from_floats([t0], &device);
            mc.set_dam_release(DamRelease {
                rows: vec![DAM],
                t0_days: t0,
                seasonal: None,
                phase: vec![[0.0, 1.0]; n_rows],
                dam_row: DamRow::Additive,
                rule_curve: None,
            })
            .expect("release");
        }
    }
    Routed { q: mc.forward(), n, q_spatial, p_spatial }
}

fn host1(t: Tensor<I, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// Weighted loss, so a symmetric error cannot cancel. Column 0 is left out.
fn weights(n_reach: usize, steps: usize) -> Vec<f32> {
    (0..n_reach * (steps + 1))
        .map(|i| {
            let (r, t) = (i / (steps + 1), i % (steps + 1));
            if t == 0 { 0.0 } else { 1.0 + 0.1 * r as f32 + 0.01 * (t % 7) as f32 }
        })
        .collect()
}

fn loss_f64(q: &Tensor<AB, 2>, n_reach: usize, steps: usize) -> f64 {
    let v: Vec<f32> = q.clone().into_data().to_vec().unwrap();
    v.iter().zip(weights(n_reach, steps)).map(|(&a, w)| a as f64 * w as f64).sum()
}

fn analytical_grads(cfg: &Config, arm: Arm, steps: usize, qp: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = Device::default();
    let r = route(cfg, sandbox5(), qp, &Params::uniform(5), arm, true);
    let w = Tensor::<AB, 1>::from_floats(weights(5, steps).as_slice(), &device).reshape([5, steps + 1]);
    let grads = (r.q * w).sum().backward();
    (
        host1(r.n.grad(&grads).expect("grad on n")),
        host1(r.q_spatial.grad(&grads).expect("grad on q_spatial")),
        host1(r.p_spatial.grad(&grads).expect("grad on p_spatial")),
    )
}

#[derive(Copy, Clone, Debug)]
enum Parent {
    N,
    QSpatial,
}

fn params_with(parent: Parent, reach: usize, x: f32) -> Params {
    let mut p = Params::uniform(5);
    match parent {
        Parent::N => p.n[reach] = x,
        Parent::QSpatial => p.q_spatial[reach] = x,
    }
    p
}

fn base_value(parent: Parent, reach: usize) -> f32 {
    let b = Params::uniform(5);
    match parent {
        Parent::N => b.n[reach],
        Parent::QSpatial => b.q_spatial[reach],
    }
}

/// FD step for one slot (see the module doc). Returns `(h, capped)`.
fn step_of(x: f32, g: f32, s: f32) -> (f32, bool) {
    let scale = x.abs().max(1.0);
    let h0 = EPS * scale;
    if g == 0.0 {
        return (h0, false);
    }
    let h_rt = 4.0 * f32::EPSILON * s / (REL_TOL as f32 * g.abs());
    let cap = STEP_CAP * scale;
    let h = h0.max(h_rt);
    if h > cap { (cap, true) } else { (h, false) }
}

fn routed_host(cfg: &Config, arm: Arm, qp: &[f32], p: &Params) -> Vec<f32> {
    let r = route(cfg, sandbox5(), qp, p, arm, false);
    r.q.into_data().to_vec().unwrap()
}

/// Clamp selections `[depth floor, velocity band, X band]` for every step and
/// reach, from the stencil point's own routed series (`q` is `[5, steps + 1]`).
fn selections(cfg: &Config, p: &Params, q: &[f32], qp: &[f32], steps: usize) -> Vec<[u8; 3]> {
    let adj = sandbox5();
    let dev = Device::default();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let t = |d: &[f32]| -> Tensor<I, 1> { Tensor::from_floats(d, &dev) };
    let ranges = &cfg.params.parameter_ranges;
    let logs = &cfg.params.log_space_parameters;
    let log_of = |k: &str| logs.iter().any(|s| s == k);
    let n = denormalize(t(&p.n), ranges.n, log_of("n"));
    let qsp = denormalize(t(&p.q_spatial), ranges.q_spatial, log_of("q_spatial"));
    let psp = denormalize(t(&p.p_spatial), ranges.p_spatial, log_of("p_spatial"));
    let cols = steps + 1;
    let depth_lb = cfg.params.attribute_minimums.depth;
    let mut out = Vec::new();
    for step in 0..steps {
        let qt: Vec<f32> = (0..5).map(|r| q[r * cols + step]).collect();
        let qpt = &qp[step * 5..step * 5 + 5];
        let k = __spike_forward_chain_k1_outputs::<I>(
            cfg,
            &pattern,
            n.clone(),
            qsp.clone(),
            psp.clone(),
            t(&qt),
            t(qpt),
            t(&adj.length_m),
            t(&adj.slope),
            t(&[0.3; 5]),
        );
        for r in 0..5 {
            let (depth, top, v_un, v_cl, c) = (k[0][r], k[1][r], k[5][r], k[6][r], k[7][r]);
            let depth_floor = (depth <= depth_lb) as u8;
            let v_band = if v_cl > v_un { 1 } else if v_cl < v_un { 2 } else { 0 };
            let w = qt[r] / (top * adj.slope[r] * c * adj.length_m[r] + 1e-12);
            let x_raw = (-w + 1.0) * 0.5;
            let x_band = if x_raw < 0.0 { 1 } else if x_raw > 0.5 { 2 } else { 0 };
            out.push([depth_floor, v_band, x_band]);
        }
    }
    out
}

/// Panics when any clamp selection differs between `x - h`, `x` and `x + h`.
fn assert_kink_free(cfg: &Config, arm: Arm, steps: usize, qp: &[f32], parent: Parent, reach: usize, h: f32) {
    let x0 = base_value(parent, reach);
    let sel = |x: f32| {
        let p = params_with(parent, reach, x);
        let q = routed_host(cfg, arm, qp, &p);
        selections(cfg, &p, &q, qp, steps)
    };
    let (sm, s0, sp) = (sel(x0 - h), sel(x0), sel(x0 + h));
    for (i, ((a, b), c)) in sm.iter().zip(&s0).zip(&sp).enumerate() {
        assert!(
            a == b && b == c,
            "{parent:?}@{reach}: a clamp selection changes inside the FD stencil at step {} reach {} (minus {a:?}, base {b:?}, plus {c:?})",
            i / 5,
            i % 5
        );
    }
}

fn fd_grad(cfg: &Config, arm: Arm, steps: usize, qp: &[f32], parent: Parent, reach: usize, eps: f32) -> f64 {
    let x0 = base_value(parent, reach);
    let eval = |x: f32| {
        let r = route(cfg, sandbox5(), qp, &params_with(parent, reach, x), arm, false);
        loss_f64(&r.q, 5, steps)
    };
    // Divide by the step actually taken in f32, not by `2·eps`.
    let (hi, lo) = (x0 + eps, x0 - eps);
    (eval(hi) - eval(lo)) / (hi as f64 - lo as f64)
}

/// Gradcheck of n and q_spatial at the upstream, downstream and dam reaches.
/// `dam_is_zero`: the dam row's analytical gradient must be exactly 0 and its
/// FD must be within `ABS_TOL` of 0 (replace row). Returns (worst rel, worst abs).
fn gradcheck(label: &str, arm: Arm, steps: usize, t_scale: f32, dam_is_zero: bool) -> (f64, f64) {
    let cfg = power_law_cfg();
    let qp = sandbox_q_prime(steps, t_scale);
    let (gn, gq, _) = analytical_grads(&cfg, arm, steps, &qp);
    // Round-off scale of the loss: sum of |w * q| at the base point.
    let q0 = routed_host(&cfg, arm, &qp, &Params::uniform(5));
    let scale: f32 = q0.iter().zip(weights(5, steps)).map(|(&a, w)| (a * w).abs()).sum();
    println!("{label}: round-off scale S = sum|w q| = {scale:.6e}");
    let (mut worst_rel, mut worst_abs) = (0.0f64, 0.0f64);
    let mut failures = Vec::new();
    for (parent, analytical) in [(Parent::N, &gn), (Parent::QSpatial, &gq)] {
        for (name, reach) in [("upstream", UPSTREAM), ("downstream", DOWNSTREAM), ("dam", DAM)] {
            let a = analytical[reach] as f64;
            let (h, capped) = step_of(base_value(parent, reach), a as f32, scale);
            assert_kink_free(&cfg, arm, steps, &qp, parent, reach, h);
            let f = fd_grad(&cfg, arm, steps, &qp, parent, reach, h);
            let abs = (a - f).abs();
            let rel = abs / a.abs().max(f.abs()).max(1e-12);
            println!(
                "{label} {parent:?} @ reach {reach} ({name}): h={h:.3e}{} analytical={a:.6e} fd={f:.6e} abs={abs:.3e} rel={rel:.3e}",
                if capped { " (capped)" } else { "" }
            );
            if capped {
                failures.push(format!("{label} {parent:?} @ reach {reach}: the step cap binds"));
            }
            if reach == DAM && dam_is_zero {
                assert_eq!(a, 0.0, "{label} {parent:?}: dam row gradient must be exactly 0");
                if f.abs() >= ABS_TOL {
                    failures.push(format!("{label} {parent:?} dam FD {f:.3e} is not near 0"));
                }
                continue;
            }
            assert!(a != 0.0 && a.is_finite(), "{label} {parent:?} reach {reach}: vacuous gradient {a}");
            worst_rel = worst_rel.max(rel);
            worst_abs = worst_abs.max(abs);
            if !(rel < REL_TOL || abs < ABS_TOL) {
                failures.push(format!("{label} {parent:?} @ reach {reach} ({name}): rel {rel:.3e} abs {abs:.3e}"));
            }
        }
    }
    println!("{label}: worst rel {worst_rel:.3e} worst abs {worst_abs:.3e}");
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
    (worst_rel, worst_abs)
}

#[test]
fn replace_row_gradcheck() {
    gradcheck(
        "replace",
        Arm::OptionC(DAM_T_DAYS, DamRow::Replace),
        GRAD_STEPS,
        GRAD_STEPS as f32,
        true,
    );
}

#[test]
fn replace_row_parameters_have_zero_gradient() {
    for enforce_positivity in [false, true] {
        let mut cfg = power_law_cfg();
        cfg.params.enforce_positivity = enforce_positivity;
        let qp = sandbox_q_prime(GRAD_STEPS, GRAD_STEPS as f32);
        let (gn, gq, gp) =
            analytical_grads(&cfg, Arm::OptionC(DAM_T_DAYS, DamRow::Replace), GRAD_STEPS, &qp);
        for (name, g) in [("n", &gn), ("q_spatial", &gq), ("p_spatial", &gp)] {
            assert_eq!(
                g[DAM], 0.0,
                "enforce_positivity={enforce_positivity}: d loss / d {name} at the dam row must be exactly 0"
            );
            for reach in [UPSTREAM, DOWNSTREAM] {
                assert!(
                    g[reach] != 0.0 && g[reach].is_finite(),
                    "enforce_positivity={enforce_positivity}: {name} gradient at reach {reach} is {}",
                    g[reach]
                );
            }
        }
    }
}

/// One additive-row gradcheck, as `additive_gradcheck_outlet_dam_constant_t`
/// does it for the dam's neighbours, on the interior dam (reach 3) with a
/// constant `T0` of 1.5 d. The dam reach's own `n` reaches the row through
/// K_r and X_r, so the dam's gradient is nonzero here.
#[test]
fn additive_row_gradcheck() {
    gradcheck("additive", Arm::Release(1.5), 72, 24.0, false);
}
