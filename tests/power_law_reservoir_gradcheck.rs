//! Gradient checks of the reservoir dam rows on `geometry_form: power_law`.
//! WET copies of `tests/reservoir_override.rs` (replace row: gradcheck and
//! exactly zero dam-row gradient) and `tests/reservoir_additive.rs` (one
//! additive row gradcheck). Tolerances and step sizes are the source files'.
//!
//! The dam row replaces (or adds to) K and X after the section, so the dam
//! row's celerity gradient is cut before `power_law.rs::section_backward`.
//! On the replace row the dam's `n`, `q_spatial` and `p_spatial` get exactly 0.
//!
//! Base points are the source files'. Each gradcheck first asserts that the
//! power-law routed discharge series is finite and that the dam reach has a
//! nonzero analytical gradient, so none of the checks is vacuous.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, DamRow, GeometryForm};
use ddrs::routing::mmc::DamRelease;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 1e-4;
/// FD step on the normalized `[0, 1]` parents (see `tests/reservoir_override.rs::EPS`).
const EPS: f32 = 1e-2;

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

fn fd_grad(cfg: &Config, arm: Arm, steps: usize, qp: &[f32], parent: Parent, reach: usize) -> f64 {
    let base = Params::uniform(5);
    let x0 = match parent {
        Parent::N => base.n[reach],
        Parent::QSpatial => base.q_spatial[reach],
    };
    let eps = (EPS * x0.abs()).max(EPS);
    let eval = |x: f32| {
        let mut p = base.clone();
        match parent {
            Parent::N => p.n[reach] = x,
            Parent::QSpatial => p.q_spatial[reach] = x,
        }
        let r = route(cfg, sandbox5(), qp, &p, arm, false);
        loss_f64(&r.q, 5, steps)
    };
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
    let (mut worst_rel, mut worst_abs) = (0.0f64, 0.0f64);
    let mut failures = Vec::new();
    for (parent, analytical) in [(Parent::N, &gn), (Parent::QSpatial, &gq)] {
        for (name, reach) in [("upstream", UPSTREAM), ("downstream", DOWNSTREAM), ("dam", DAM)] {
            let a = analytical[reach] as f64;
            let f = fd_grad(&cfg, arm, steps, &qp, parent, reach);
            let abs = (a - f).abs();
            let rel = abs / a.abs().max(f.abs()).max(1e-12);
            println!(
                "{label} {parent:?} @ reach {reach} ({name}): analytical={a:.6e} fd={f:.6e} abs={abs:.3e} rel={rel:.3e}"
            );
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
