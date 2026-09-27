//! Finite-difference gradcheck of the learned dam release: the per-dam `T`
//! (seconds) is an autodiff parent of the timestep op (`TimestepReleaseOp`,
//! `TimestepReleaseGammaOp` in `src/routing/mmc_op.rs`), and `T_d(t)` is built
//! from `(T0, a, b)` in ordinary Burn autodiff by `MuskingumCunge::forward`.
//!
//! Sandbox topology (`1 → 3`, `2 → 3`, `0 → 4`, `3 → 4`), dam on the interior
//! reach 3 and, separately, on the outlet 4. The phase table runs one full
//! seasonal cycle every 48 steps so `a`, `b` and `T0` have independent
//! gradients. Checked at a small (`T0 = 0.1` d) and a large (`T0 = 20` d)
//! residence time with `a`, `b` nonzero, with the learned-gamma sibling op,
//! and at a clamped base where the gradient must be exactly zero.
//!
//! The storage-conserving dam row reads `T` at both ends of a step, so the op
//! has two `T` parents: `T_{t+1}` (every coefficient's denominator) and `T_t`
//! (c3's numerator). `step_gradcheck_t_prev_and_t_next_separately_and_together`
//! checks each on its own at one step (via `timestep_forward_release`), and one
//! leaf in both slots (constant `T`), whose gradient must be their sum. At
//! constant `T` the two nearly cancel, so that case is judged against the
//! parts' scale.
//!
//! ε: every parent here is nonlinear in the output (`T0`, `a`, `b` enter
//! through `exp` and the Muskingum coefficients), so the step is relative,
//! `max(1e-2·|x|, 1e-2)`, except `a`, `b` at the large `T0`. There the dam's
//! `c3 = (2T − dt)/(2T + dt)` sits within 2e-3 of 1, a 1e-2 step in `a` moves
//! it by a few hundred f32 ulps, and the finite difference scatters
//! non-monotonically around the analytical value. Measured 2026-09-26 at
//! `T0 = 20` d (rel. error of `b`): ε 3e-3 → 8.6e-3, 1e-2 → 8.0e-3,
//! 3e-2 → 3.8e-3, 1e-1 → 3.9e-4, 2e-1 → 1.3e-3 (truncation). The analytical
//! value is the fixed point, so that case uses ε = 0.1. The loss is
//! accumulated in f64 on the host for the finite differences; see
//! `.claude/skills/ddrs-dev/references/testing.md` §Authoring patterns.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc::DamRelease;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 1e-4;
const STEPS: usize = 72;
const N_REACH: usize = 5;

fn sandbox5() -> SparseAdjacency {
    let n = N_REACH;
    let mut dense = vec![0.0_f32; n * n];
    for (up, down) in [(1, 3), (2, 3), (0, 4), (3, 4)] {
        dense[down * n + up] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, vec![5000.0; n], vec![0.001; n])
}

fn q_prime() -> Vec<f32> {
    let base = [22.0_f32, 11.0, 11.0, 11.0, 22.0];
    let mut q = Vec::with_capacity((STEPS + 1) * N_REACH);
    for t in 0..=STEPS {
        let s = (t as f32 / 24.0 * std::f32::consts::PI).sin();
        let pulse = 1.0 + 1.5 * s * s;
        q.extend(base.iter().map(|b| b * pulse));
    }
    q
}

fn phase() -> Vec<[f32; 2]> {
    (0..=STEPS)
        .map(|t| {
            let w = 2.0 * std::f32::consts::PI * t as f32 / 48.0;
            [w.sin(), w.cos()]
        })
        .collect()
}

/// Per-(reach, step) loss weights, so the loss is not a plain sum that a
/// symmetric error could cancel in.
fn weights() -> Vec<f32> {
    (0..N_REACH * (STEPS + 1))
        .map(|i| {
            let (r, t) = (i / (STEPS + 1), i % (STEPS + 1));
            if t == 0 { 0.0 } else { 1.0 + 0.1 * r as f32 + 0.01 * (t % 7) as f32 }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Base {
    dam: usize,
    t0: f32,
    a: f32,
    b: f32,
    /// Normalised learned gamma, when the gamma sibling op is exercised.
    gamma: Option<f32>,
    /// Finite-difference step for `a` and `b` (see the module docs).
    eps_ab: f32,
}

struct Leaves {
    t0: Tensor<AB, 1>,
    a: Tensor<AB, 1>,
    b: Tensor<AB, 1>,
    n: Tensor<AB, 1>,
    gamma: Option<Tensor<AB, 1>>,
}

fn route(base: Base, require_grad: bool) -> (Tensor<AB, 2>, Leaves) {
    let device = Device::default();
    let leaf = |v: Vec<f32>| {
        let t = Tensor::<AB, 1>::from_floats(v.as_slice(), &device);
        if require_grad { t.require_grad() } else { t }
    };
    let n = leaf(vec![0.5; N_REACH]);
    let gamma = base.gamma.map(|g| leaf(vec![g; N_REACH]));
    let spatial = SpatialParameters::<I> {
        n: n.clone(),
        q_spatial: Tensor::from_floats(vec![0.5_f32; N_REACH].as_slice(), &device),
        p_spatial: Some(Tensor::from_floats(vec![0.575_f32; N_REACH].as_slice(), &device)),
        k_d: None,
        d_gw: None,
        leakance_factor: None,
        impervious_mask: None,
        gamma: gamma.clone(),
    };
    let inputs = RoutingInputs::<I> {
        adjacency: sandbox5(),
        x_storage: Tensor::ones([N_REACH], &device) * 0.3,
    };
    let q = Tensor::<AB, 1>::from_floats(q_prime().as_slice(), &device).reshape([STEPS + 1, N_REACH]);
    let mut mc = MuskingumCunge::<I>::new(Config::default(), device);
    mc.setup_inputs(inputs, q, spatial, false, None);
    let (t0, a, b) = (leaf(vec![base.t0]), leaf(vec![base.a]), leaf(vec![base.b]));
    mc.set_dam_release(DamRelease {
        rows: vec![base.dam],
        t0_days: t0.clone(),
        seasonal: Some((a.clone(), b.clone())),
        phase: phase(),
    })
    .expect("release");
    (mc.forward(), Leaves { t0, a, b, n, gamma })
}

fn loss_f64(base: Base) -> f64 {
    let q = route(base, false).0.into_data().to_vec::<f32>().unwrap();
    q.iter().zip(weights()).map(|(&v, w)| v as f64 * w as f64).sum()
}

#[derive(Clone, Copy, Debug)]
enum Parent {
    T0,
    A,
    B,
    Gamma,
}

fn perturbed(base: Base, p: Parent, x: f32) -> Base {
    let mut b = base;
    match p {
        Parent::T0 => b.t0 = x,
        Parent::A => b.a = x,
        Parent::B => b.b = x,
        Parent::Gamma => b.gamma = Some(x),
    }
    b
}

fn value(base: Base, p: Parent) -> f32 {
    match p {
        Parent::T0 => base.t0,
        Parent::A => base.a,
        Parent::B => base.b,
        Parent::Gamma => base.gamma.expect("gamma base"),
    }
}

fn fd(base: Base, p: Parent) -> f64 {
    let x0 = value(base, p);
    let eps = match p {
        Parent::A | Parent::B => base.eps_ab,
        _ => (1e-2 * x0.abs()).max(1e-2),
    };
    let (hi, lo) = (x0 + eps, x0 - eps);
    (loss_f64(perturbed(base, p, hi)) - loss_f64(perturbed(base, p, lo))) / (hi as f64 - lo as f64)
}

fn analytical(base: Base) -> (f64, f64, f64, Option<f64>, Vec<f32>) {
    let device = Device::default();
    let (q, leaves) = route(base, true);
    let w = Tensor::<AB, 1>::from_floats(weights().as_slice(), &device).reshape([N_REACH, STEPS + 1]);
    let grads = (q * w).sum().backward();
    let g1 = |t: &Tensor<AB, 1>| -> Vec<f32> {
        t.grad(&grads).expect("tracked parent has a gradient").into_data().to_vec::<f32>().unwrap()
    };
    // The FD perturbs gamma on EVERY reach (one leaf value broadcast), so
    // compare against the sum of the per-reach gradients.
    let gamma = leaves.gamma.as_ref().map(|g| g1(g).iter().map(|&v| v as f64).sum());
    (
        g1(&leaves.t0)[0] as f64,
        g1(&leaves.a)[0] as f64,
        g1(&leaves.b)[0] as f64,
        gamma,
        g1(&leaves.n),
    )
}

fn check(label: &str, base: Base) {
    let (gt0, ga, gb, ggamma, gn) = analytical(base);
    let mut cases = vec![(Parent::T0, gt0), (Parent::A, ga), (Parent::B, gb)];
    if let Some(g) = ggamma {
        cases.push((Parent::Gamma, g));
    }
    let mut failures = Vec::new();
    for (p, a) in cases {
        let f = fd(base, p);
        let abs = (a - f).abs();
        let rel = abs / a.abs().max(f.abs()).max(1e-12);
        println!("{label} {p:?}: analytical={a:.6e} fd={f:.6e} abs={abs:.3e} rel={rel:.3e}");
        assert!(a != 0.0 && a.is_finite(), "{label} {p:?}: analytical gradient {a} is vacuous");
        if !(rel < REL_TOL || abs < ABS_TOL) {
            failures.push(format!("{label} {p:?}: rel {rel:.3e}"));
        }
    }
    // K and X on the dam row are T and 0, so its hydraulic parameters get
    // exactly zero gradient, while the reaches above it still learn.
    assert_eq!(gn[base.dam], 0.0, "{label}: dam row n gradient must be exactly 0");
    assert!(gn[1] != 0.0, "{label}: upstream n gradient vanished");
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn release_gradcheck_small_t_interior_dam() {
    check("small T, dam 3", Base { dam: 3, t0: 0.1, a: 0.7, b: -0.4, gamma: None, eps_ab: 1e-2 });
}

#[test]
fn release_gradcheck_large_t_interior_dam() {
    check("large T, dam 3", Base { dam: 3, t0: 20.0, a: 0.7, b: -0.4, gamma: None, eps_ab: 1e-1 });
}

#[test]
fn release_gradcheck_outlet_dam() {
    check("T 1.5, dam 4", Base { dam: 4, t0: 1.5, a: -0.8, b: 0.5, gamma: None, eps_ab: 1e-2 });
}

#[test]
fn release_gradcheck_with_learned_gamma() {
    check("gamma op, dam 3", Base { dam: 3, t0: 0.8, a: 0.7, b: -0.4, gamma: Some(0.4), eps_ab: 1e-2 });
}

/// End to end: one weight of the release head's read-out, through
/// `release_params` (sigmoid, log-space `T0`, linear `a`, `b`), the per-step
/// seasonal `T`, the timestep op's `T` parent and the routing solve. The
/// FD perturbs the weight value directly in a rebuilt head.
#[test]
fn release_head_weight_gradcheck_end_to_end() {
    use burn::module::Param;
    use ddrs::config::{ParameterRanges, ReleaseHeadSection};
    use ddrs::nn::release_head::{init_release_head, release_params};

    let device = Device::default();
    let ranges = ParameterRanges::default();
    let section = ReleaseHeadSection {
        hidden_size: 6,
        num_hidden_layers: 1,
        grid: 5,
        k: 3,
        input_var_names: vec!["f1".into(), "f2".into()],
        seasonal: true,
    };
    let features = || Tensor::<AB, 2>::from_floats([[0.8_f32, -1.2]], &device);
    // Move the read-out off its init so T0 is a few days and a, b are nonzero.
    let head0 = {
        let mut h = init_release_head::<AB>(&section, &ranges, 42, &device);
        h.output.bias = Some(Param::from_tensor(Tensor::from_floats([0.3_f32, 0.6, -0.5], &device)));
        h
    };
    let (row, col) = (2, 0); // read-out weight [H, P]: hidden unit 2 -> T0
    let with_weight = |delta: f32| {
        let mut h = head0.clone();
        let w = h.output.weight.val();
        let mut v: Vec<f32> = w.clone().into_data().to_vec().unwrap();
        let [_, p] = w.dims();
        v[row * p + col] += delta;
        h.output.weight = Param::from_tensor(
            Tensor::<AB, 1>::from_floats(v.as_slice(), &device).reshape(w.dims()),
        );
        h
    };

    let route_with = |head: &ddrs::nn::KanHead<AB>| -> Tensor<AB, 2> {
        let rp = release_params(head, features(), &ranges);
        let spatial = SpatialParameters::<I> {
            n: Tensor::from_floats(vec![0.5_f32; N_REACH].as_slice(), &device),
            q_spatial: Tensor::from_floats(vec![0.5_f32; N_REACH].as_slice(), &device),
            p_spatial: Some(Tensor::from_floats(vec![0.575_f32; N_REACH].as_slice(), &device)),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        };
        let inputs = RoutingInputs::<I> {
            adjacency: sandbox5(),
            x_storage: Tensor::ones([N_REACH], &device) * 0.3,
        };
        let q = Tensor::<AB, 1>::from_floats(q_prime().as_slice(), &device).reshape([STEPS + 1, N_REACH]);
        let mut mc = MuskingumCunge::<I>::new(Config::default(), device);
        mc.setup_inputs(inputs, q, spatial, false, None);
        mc.set_dam_release(DamRelease {
            rows: vec![3],
            t0_days: rp.t0_days,
            seasonal: rp.seasonal,
            phase: phase(),
        })
        .unwrap();
        mc.forward()
    };
    let w = Tensor::<AB, 1>::from_floats(weights().as_slice(), &device).reshape([N_REACH, STEPS + 1]);
    let grads = (route_with(&head0) * w).sum().backward();
    let gw: Vec<f32> = head0.output.weight.val().grad(&grads).expect("weight grad").into_data().to_vec().unwrap();
    let p = head0.output.weight.val().dims()[1];
    let analytical = gw[row * p + col] as f64;

    let loss = |h: &ddrs::nn::KanHead<AB>| -> f64 {
        let q = route_with(h).into_data().to_vec::<f32>().unwrap();
        q.iter().zip(weights()).map(|(&v, w)| v as f64 * w as f64).sum()
    };
    let eps = 2e-2_f32;
    let fd = (loss(&with_weight(eps)) - loss(&with_weight(-eps))) / (2.0 * eps as f64);
    let abs = (analytical - fd).abs();
    let rel = abs / analytical.abs().max(fd.abs());
    println!("release head weight [{row},{col}]: analytical={analytical:.6e} fd={fd:.6e} rel={rel:.3e}");
    assert!(analytical != 0.0 && analytical.is_finite(), "vacuous weight gradient");
    assert!(rel < REL_TOL || abs < ABS_TOL, "release head weight gradcheck rel {rel:.3e}");
}

// ---------------------------------------------------------------------------
// One timestep, the two T parents on their own: `T_t` (the step's start,
// only in c3's numerator) and `T_{t+1}` (its end, in every coefficient's
// denominator). `timestep_forward_release` takes them as separate leaves.
// ---------------------------------------------------------------------------

/// One routed step on the sandbox with dams on reaches 3 and 4. Returns the
/// weighted loss and the two T leaves (seconds, `[2]`). `same_leaf` routes
/// ONE leaf into both parents, the constant-T case.
fn one_step(t_prev: [f32; 2], t_next: [f32; 2], same_leaf: bool, grad: bool)
    -> (Tensor<AB, 1>, Tensor<AB, 1>, Tensor<AB, 1>, Tensor<AB, 1>)
{
    use std::sync::Arc;
    use ddrs::sparse::{AValuesAssembler, CsrPattern};
    let device = Device::default();
    let adj = sandbox5();
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);
    let c = |v: Vec<f32>| Tensor::<AB, 1>::from_floats(v.as_slice(), &device);
    let leaf = |v: [f32; 2]| {
        let t = Tensor::<AB, 1>::from_floats(v, &device);
        if grad { t.require_grad() } else { t }
    };
    let tn = leaf(t_next);
    let tp = if same_leaf { tn.clone() } else { leaf(t_prev) };
    // A transient state: the dams hold more than their inflow, the rest less.
    let q_t = c(vec![20.0, 9.0, 13.0, 40.0, 70.0]);
    let q_prime = c(vec![22.0, 11.0, 11.0, 11.0, 22.0]);
    let cfg = Config::default();
    // Denormalized geometry, as `MuskingumCunge` hands the op (n, q, p).
    let q = ddrs::routing::mmc_op::timestep_forward_release::<I>(
        &cfg,
        &pattern,
        &assembler,
        c(vec![0.05; N_REACH]),
        c(vec![0.5; N_REACH]),
        c(vec![21.0; N_REACH]),
        q_t,
        q_prime,
        c(adj.length_m.clone()),
        c(adj.slope.clone()),
        c(vec![0.3; N_REACH]),
        vec![3, 4],
        tn.clone(),
        tp.clone(),
    );
    let w = c(STEP_WEIGHTS.to_vec());
    ((q.clone() * w).sum(), tp, tn, q)
}

const STEP_WEIGHTS: [f32; 5] = [1.0, 1.1, 1.2, 1.3, 1.4];

/// The weighted loss accumulated in f64 on the host, for the finite differences.
fn one_step_loss(t_prev: [f32; 2], t_next: [f32; 2]) -> f64 {
    let q: Vec<f32> = one_step(t_prev, t_next, false, false).3.into_data().to_vec().unwrap();
    q.iter().zip(STEP_WEIGHTS).map(|(&v, w)| v as f64 * w as f64).sum()
}

#[test]
fn step_gradcheck_t_prev_and_t_next_separately_and_together() {
    let mut failures = Vec::new();
    // Small (2 h, 3 h) and large (5 d, 30 d) residence times, with T_t != T_{t+1}.
    for (label, tp0, tn0) in [
        ("small", [7_200.0_f32, 10_800.0], [9_000.0_f32, 8_000.0]),
        ("large", [432_000.0_f32, 2_592_000.0], [480_000.0_f32, 2_400_000.0]),
    ] {
        let (loss, tp, tn, _) = one_step(tp0, tn0, false, true);
        let grads = loss.backward();
        let g = |t: &Tensor<AB, 1>| -> Vec<f32> { t.grad(&grads).expect("grad").into_data().to_vec().unwrap() };
        let (gp, gn) = (g(&tp), g(&tn));
        for dam in 0..2 {
            for (which, analytical) in [("T_t", gp[dam] as f64), ("T_t+1", gn[dam] as f64)] {
                let base = if which == "T_t" { tp0[dam] } else { tn0[dam] };
                let eps = 1e-2 * base;
                let bump = |d: f32| {
                    let (mut p, mut n) = (tp0, tn0);
                    if which == "T_t" { p[dam] += d } else { n[dam] += d }
                    one_step_loss(p, n)
                };
                let fd = (bump(eps) - bump(-eps)) / (2.0 * eps as f64);
                let abs = (analytical - fd).abs();
                let rel = abs / analytical.abs().max(fd.abs()).max(1e-30);
                println!("{label} dam {dam} {which}: analytical={analytical:.6e} fd={fd:.6e} rel={rel:.3e}");
                assert!(analytical != 0.0, "{label} {which}: vacuous gradient");
                if !(rel < REL_TOL || abs < 1e-9) {
                    failures.push(format!("{label} dam {dam} {which}: rel {rel:.3e}"));
                }
            }
        }
        // Together: one leaf in both slots gets the sum of the two parents'
        // gradients (constant T), and it matches a FD that moves both.
        let t_same = tn0;
        let (loss, _, t, _) = one_step(t_same, t_same, true, true);
        let grads = loss.backward();
        let g_same: Vec<f32> = t.grad(&grads).expect("grad").into_data().to_vec().unwrap();
        let (loss2, tp2, tn2, _) = one_step(t_same, t_same, false, true);
        let grads2 = loss2.backward();
        let gp2: Vec<f32> = tp2.grad(&grads2).unwrap().into_data().to_vec().unwrap();
        let gn2: Vec<f32> = tn2.grad(&grads2).unwrap().into_data().to_vec().unwrap();
        for dam in 0..2 {
            let eps = 1e-2 * t_same[dam];
            let bump = |d: f32| {
                let mut v = t_same;
                v[dam] += d;
                one_step_loss(v, v)
            };
            let fd = (bump(eps) - bump(-eps)) / (2.0 * eps as f64);
            let sum = gp2[dam] as f64 + gn2[dam] as f64;
            // At constant T the two parents' gradients nearly cancel (a step
            // with dT = 0 is storage-neutral in T), so the together-gradient
            // is judged on the scale of its parts, not of itself.
            let scale = (gp2[dam] as f64).abs().max((gn2[dam] as f64).abs());
            let rel = (g_same[dam] as f64 - fd).abs() / scale;
            let rel_sum = (g_same[dam] as f64 - sum).abs() / scale;
            println!("{label} dam {dam} together: one leaf {:.6e} = T_t + T_t+1 {sum:.6e} (rel {rel_sum:.1e}), fd {fd:.6e} (err / part {rel:.3e})", g_same[dam]);
            if !(rel < REL_TOL) || !(rel_sum < 1e-6) {
                failures.push(format!("{label} dam {dam} together: rel {rel:.3e} / sum {rel_sum:.1e}"));
            }
        }
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn clamped_release_has_exactly_zero_gradient() {
    // T0 = 1.5 h, a = -2, b = -2: T0·exp(-2 sin - 2 cos) < 1 h wherever
    // sin + cos > 0.2, so the clamp binds on part of the cycle and those
    // steps contribute nothing. Put the dam in a phase where it binds on
    // EVERY step: a constant phase (sin, cos) = (1, 0).
    let device = Device::default();
    let mut mc = {
        let spatial = SpatialParameters::<I> {
            n: Tensor::from_floats(vec![0.5_f32; N_REACH].as_slice(), &device),
            q_spatial: Tensor::from_floats(vec![0.5_f32; N_REACH].as_slice(), &device),
            p_spatial: Some(Tensor::from_floats(vec![0.575_f32; N_REACH].as_slice(), &device)),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        };
        let inputs = RoutingInputs::<I> {
            adjacency: sandbox5(),
            x_storage: Tensor::ones([N_REACH], &device) * 0.3,
        };
        let q = Tensor::<AB, 1>::from_floats(q_prime().as_slice(), &device).reshape([STEPS + 1, N_REACH]);
        let mut mc = MuskingumCunge::<I>::new(Config::default(), device);
        mc.setup_inputs(inputs, q, spatial, false, None);
        mc
    };
    let t0 = Tensor::<AB, 1>::from_floats([1.5_f32 / 24.0], &device).require_grad();
    let a = Tensor::<AB, 1>::from_floats([-2.0_f32], &device).require_grad();
    let b = Tensor::<AB, 1>::from_floats([0.5_f32], &device).require_grad();
    mc.set_dam_release(DamRelease {
        rows: vec![3],
        t0_days: t0.clone(),
        seasonal: Some((a.clone(), b.clone())),
        phase: vec![[1.0, 0.0]; STEPS + 1],
    })
    .unwrap();
    let grads = mc.forward().sum().backward();
    for (name, t) in [("T0", &t0), ("a", &a), ("b", &b)] {
        let g = t.grad(&grads).expect("gradient").into_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(g, 0.0, "{name}: clamped T must pass no gradient, got {g}");
    }
}
