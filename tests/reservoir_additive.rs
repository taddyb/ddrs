//! The additive dam row (`release_head.dam_row: additive`, S19'''' / B19''''
//! in `src/routing/mmc_op.rs`): the reservoir's storage `T·Q` is ADDED to the
//! reach's own Muskingum channel storage,
//!
//! ```text
//! S = K_r·[X_r·I + (1 − X_r)·Q] + T·Q
//! D  = K_r·(1 − X_r) + T_{t+1} + dt/2
//! c1 = (dt/2 − K_r·X_r)/D   c2 = (dt/2 + K_r·X_r)/D
//! c3 = (K_r·(1 − X_r) + T_t − dt/2)/D   c4 = dt/D
//! ```
//!
//! 1. `T = 0` is the channel row: routing with an additive dam at `T = 0`
//!    (release path, seasonal or constant, and the option C path) is bitwise
//!    the routing with no dam, and so is the gradient into `n`.
//! 2. `K_r = 0` (a zero-length dam reach) is today's replace row (S19'''),
//!    bitwise in the forward and in the `T0`, `a`, `b` gradients.
//! 3. The row conserves storage for a time-varying `T`, one step at a time,
//!    at a headwater and at an interior dam, with `K_r`, `X_r` recovered from
//!    the channel row's own coefficients (the solve is affine in `q'`).
//! 4. Gradcheck of `T0`, `a`, `b`, the dam reach's `n` (which reaches the dam
//!    row through `K_r`, `X_r`) and an upstream `n`, at `T0` in
//!    `{0, 0.1, 1.5, 20}` d, with `K_r·X_r` below and above `dt/2` (checked
//!    by the sign of `c1`); with a learned gamma; with `enforce_positivity`;
//!    the two `T` parents on their own at one step; one release-head read-out
//!    weight end to end.
//! 5. Opt-out (the review synthesis' check 1b): on the Juniata bundle, the
//!    additive row with the release head saturated at its `T0` floor
//!    reproduces the no-dam gauge series to NSE > 0.9999, while the replace
//!    row at its 1 h floor does not come as close and the additive row at the
//!    init `T0` (4.5 h) visibly acts.
//!
//! ε follows `tests/reservoir_release_gradcheck.rs`: relative 1e-2 on `T0`
//! (absolute 1e-3 d at `T0 = 0`, where the additive row is smooth through
//! zero and a 1e-2 d step is not small against `D`), 1e-2 on `a`, `b` except
//! 0.1 at `T0 = 20` d (c3 within 2e-3 of 1; see that file's step sweep), and
//! 1e-2 on normalized `n` (`tests/reservoir_override.rs::EPS`), except the dam
//! reach's `n` at `T0 = 20` d, which sits at the loss's f32 noise floor (step
//! sweep recorded at that case).

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::{Config, DamRow};
use ddrs::routing::mmc::{DamRelease, DT_SECONDS};
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 1e-4;
const STEPS: usize = 72;
const N_REACH: usize = 5;
const DT: f64 = DT_SECONDS as f64;

/// The RAPID sandbox topology `1 → 3`, `2 → 3`, `0 → 4`, `3 → 4`; reach
/// `long` gets `long_m` metres, every other reach 5 km.
fn sandbox(long: usize, long_m: f32) -> SparseAdjacency {
    let n = N_REACH;
    let mut dense = vec![0.0_f32; n * n];
    for (up, down) in [(1, 3), (2, 3), (0, 4), (3, 4)] {
        dense[down * n + up] = 1.0;
    }
    let mut length = vec![5000.0_f32; n];
    length[long] = long_m;
    SparseAdjacency::from_dense(n, &dense, length, vec![0.001; n])
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

/// Per-(reach, step) loss weights, so a symmetric error cannot cancel.
fn weights() -> Vec<f32> {
    (0..N_REACH * (STEPS + 1))
        .map(|i| {
            let (r, t) = (i / (STEPS + 1), i % (STEPS + 1));
            if t == 0 { 0.0 } else { 1.0 + 0.1 * r as f32 + 0.01 * (t % 7) as f32 }
        })
        .collect()
}

fn assert_bitwise(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: idx {i}: {x} vs {y}");
    }
}

fn host<B: burn::tensor::backend::Backend>(t: Tensor<B, 1>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

/// How the dam is armed.
#[derive(Clone, Copy, Debug)]
enum Arm {
    None,
    /// `set_reservoir_rows_as`, constant `T` (days).
    OptionC(f32, DamRow),
    /// `set_dam_release`: `T0` (days), `Some((a, b))` for seasonal.
    Release(f32, Option<(f32, f32)>, DamRow),
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    dam: usize,
    /// Length of the dam reach, metres (sets `K_r`).
    dam_len: f32,
    /// Normalized `n` on every reach, and on the dam reach.
    n: f32,
    n_dam: f32,
    gamma: Option<f32>,
    enforce_positivity: bool,
}

impl Setup {
    fn new(dam: usize, dam_len: f32) -> Self {
        Self { dam, dam_len, n: 0.5, n_dam: 0.5, gamma: None, enforce_positivity: false }
    }
}

struct Routed {
    q: Tensor<AB, 2>,
    n: Tensor<AB, 1>,
    t0: Option<Tensor<AB, 1>>,
    ab: Option<(Tensor<AB, 1>, Tensor<AB, 1>)>,
    gamma: Option<Tensor<AB, 1>>,
}

fn route(s: Setup, arm: Arm, grad: bool) -> Routed {
    let device = Device::default();
    let leaf = |v: Vec<f32>| {
        let t = Tensor::<AB, 1>::from_floats(v.as_slice(), &device);
        if grad { t.require_grad() } else { t }
    };
    let mut nv = vec![s.n; N_REACH];
    nv[s.dam] = s.n_dam;
    let n = leaf(nv);
    let gamma = s.gamma.map(|g| leaf(vec![g; N_REACH]));
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
        adjacency: sandbox(s.dam, s.dam_len),
        x_storage: Tensor::ones([N_REACH], &device) * 0.3,
    };
    let q = Tensor::<AB, 1>::from_floats(q_prime().as_slice(), &device).reshape([STEPS + 1, N_REACH]);
    let mut cfg = Config::default();
    cfg.params.enforce_positivity = s.enforce_positivity;
    let mut mc = MuskingumCunge::<I>::new(cfg, device);
    mc.setup_inputs(inputs, q, spatial, false, None);
    let (mut t0_leaf, mut ab_leaf) = (None, None);
    match arm {
        Arm::None => {}
        Arm::OptionC(t, row) => mc.set_reservoir_rows_as(&[s.dam], &[t], row).expect("rows"),
        Arm::Release(t0, ab, row) => {
            let t0 = leaf(vec![t0]);
            let ab = ab.map(|(a, b)| (leaf(vec![a]), leaf(vec![b])));
            mc.set_dam_release(DamRelease {
                rows: vec![s.dam],
                t0_days: t0.clone(),
                seasonal: ab.clone(),
                phase: phase(),
                dam_row: row,
            })
            .expect("release");
            t0_leaf = Some(t0);
            ab_leaf = ab;
        }
    }
    Routed { q: mc.forward(), n, t0: t0_leaf, ab: ab_leaf, gamma }
}

fn forward(s: Setup, arm: Arm) -> Vec<f32> {
    route(s, arm, false).q.into_data().to_vec().unwrap()
}

fn loss_f64(s: Setup, arm: Arm) -> f64 {
    forward(s, arm).iter().zip(weights()).map(|(&v, w)| v as f64 * w as f64).sum()
}

/// Analytical gradients of the weighted loss: (n per reach, T0, (a, b), gamma summed).
fn analytical(s: Setup, arm: Arm) -> (Vec<f32>, Option<f64>, Option<(f64, f64)>, Option<f64>) {
    let device = Device::default();
    let r = route(s, arm, true);
    let w = Tensor::<AB, 1>::from_floats(weights().as_slice(), &device).reshape([N_REACH, STEPS + 1]);
    let grads = (r.q * w).sum().backward();
    let g = |t: &Tensor<AB, 1>| host(t.grad(&grads).expect("tracked"));
    (
        g(&r.n),
        r.t0.as_ref().map(|t| g(t)[0] as f64),
        r.ab.as_ref().map(|(a, b)| (g(a)[0] as f64, g(b)[0] as f64)),
        r.gamma.as_ref().map(|t| g(t).iter().map(|&v| v as f64).sum()),
    )
}

// ---------------------------------------------------------------------------
// 1. T = 0 is the channel row.
// ---------------------------------------------------------------------------

#[test]
fn additive_row_at_zero_t_routes_bitwise_like_no_dam() {
    for (dam, len) in [(3, 5000.0), (1, 5000.0), (4, 20_000.0)] {
        let s = Setup::new(dam, len);
        let none = forward(s, Arm::None);
        let label = |what: &str| format!("dam {dam}: {what}");
        assert_bitwise(&forward(s, Arm::Release(0.0, None, DamRow::Additive)), &none, &label("constant T = 0"));
        assert_bitwise(
            &forward(s, Arm::Release(0.0, Some((0.7, -0.4)), DamRow::Additive)),
            &none,
            &label("seasonal T0 = 0"),
        );
        assert_bitwise(&forward(s, Arm::OptionC(0.0, DamRow::Additive)), &none, &label("option C T = 0"));
        // Not vacuous: a nonzero T acts.
        assert!(forward(s, Arm::Release(0.5, None, DamRow::Additive)) != none, "{}", label("T = 0.5 d"));
    }
}

#[test]
fn additive_row_at_zero_t_passes_the_channel_gradient_into_n() {
    let s = Setup::new(3, 5000.0);
    let (g_none, ..) = analytical(s, Arm::None);
    let (g_add, ..) = analytical(s, Arm::Release(0.0, None, DamRow::Additive));
    println!("no dam {g_none:?}\nadditive T=0 {g_add:?}");
    assert_bitwise(&g_add, &g_none, "d loss / d n, additive T = 0 vs no dam");
    assert!(g_none[3] != 0.0);
}

// ---------------------------------------------------------------------------
// 2. K_r = 0 is the replace row.
// ---------------------------------------------------------------------------

#[test]
fn additive_row_on_a_zero_length_reach_is_the_replace_row() {
    let s = Setup::new(3, 0.0);
    for arm_t in [(0.3_f32, Some((0.7_f32, -0.4_f32))), (1.5, None), (20.0, Some((-0.8, 0.5)))] {
        let (t0, ab) = arm_t;
        let add = Arm::Release(t0, ab, DamRow::Additive);
        let rep = Arm::Release(t0, ab, DamRow::Replace);
        assert_bitwise(&forward(s, add), &forward(s, rep), &format!("T0 {t0}: additive at K_r = 0 vs replace"));
        let (_, ta, aba, _) = analytical(s, add);
        let (_, tr, abr, _) = analytical(s, rep);
        println!("T0 {t0}: dT0 additive {ta:?} replace {tr:?}; d(a,b) {aba:?} vs {abr:?}");
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1e-30);
        assert!(close(ta.unwrap(), tr.unwrap()), "T0 {t0}: dT0 differs");
        if let (Some((a1, b1)), Some((a2, b2))) = (aba, abr) {
            assert!(close(a1, a2) && close(b1, b2), "T0 {t0}: d(a, b) differ");
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Storage conservation, one step at a time.
// ---------------------------------------------------------------------------

/// One routed step on the sandbox (dam reach `dam`, length `dam_len`), dams
/// armed through `timestep_forward_release_as`. Returns every reach's
/// `Q_{t+1}` and, when `grad`, the loss and the two T leaves.
fn one_step(
    dam: usize,
    dam_len: f32,
    q_t: [f32; 5],
    q_lat: [f32; 5],
    t_prev: f32,
    t_next: f32,
    row: DamRow,
    grad: bool,
) -> (Vec<f32>, Option<(Tensor<AB, 1>, Tensor<AB, 1>, Tensor<AB, 1>)>) {
    use ddrs::sparse::{AValuesAssembler, CsrPattern};
    let device = Device::default();
    let adj = sandbox(dam, dam_len);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);
    let c = |v: Vec<f32>| Tensor::<AB, 1>::from_floats(v.as_slice(), &device);
    let leaf = |v: f32| {
        let t = Tensor::<AB, 1>::from_floats([v], &device);
        if grad { t.require_grad() } else { t }
    };
    let (tp, tn) = (leaf(t_prev), leaf(t_next));
    let q = ddrs::routing::mmc_op::timestep_forward_release_as::<I>(
        &Config::default(),
        &pattern,
        &assembler,
        c(vec![0.05; N_REACH]),
        c(vec![0.5; N_REACH]),
        c(vec![21.0; N_REACH]),
        c(q_t.to_vec()),
        c(q_lat.to_vec()),
        c(adj.length_m.clone()),
        c(adj.slope.clone()),
        c(vec![0.3; N_REACH]),
        vec![dam],
        tn.clone(),
        tp.clone(),
        row,
    );
    let out = host(q.clone());
    let extra = grad.then(|| {
        let w = c(STEP_WEIGHTS.to_vec());
        ((q * w).sum(), tp, tn)
    });
    (out, extra)
}

const STEP_WEIGHTS: [f32; 5] = [1.0, 1.1, 1.2, 1.3, 1.4];
const Q_T: [f32; 5] = [20.0, 9.0, 13.0, 40.0, 70.0];
const Q_LAT: [f32; 5] = [22.0, 11.0, 11.0, 11.0, 22.0];

/// The dam reach's channel `A = K_r·(1 − X_r)` and `B = K_r·X_r` at this
/// step, from the channel row (T = 0) alone: the solve is affine in the
/// lateral inflows, and `K_r`, `X_r` depend only on `q_t`, so
/// `c4 = ∂Q_dam/∂q'_dam = 2dt/(2A + dt)` and, for an interior dam fed by
/// reach `up`, `c1 = ∂Q_dam/∂Q_up = (dt − 2B)/(2A + dt)`.
fn channel_a_b(dam: usize, dam_len: f32, up: Option<usize>) -> (f64, f64) {
    let solve = |lat: [f32; 5]| -> Vec<f64> {
        one_step(dam, dam_len, Q_T, lat, 0.0, 0.0, DamRow::Additive, false).0.iter().map(|&v| v as f64).collect()
    };
    let base = solve(Q_LAT);
    let mut bump = Q_LAT;
    bump[dam] += 40.0;
    let c4 = (solve(bump)[dam] - base[dam]) / 40.0;
    let a = DT / c4 - DT / 2.0;
    let b = match up {
        None => f64::NAN,
        Some(u) => {
            let mut bump = Q_LAT;
            bump[u] += 40.0;
            let moved = solve(bump);
            let c1 = (moved[dam] - base[dam]) / (moved[u] - base[u]);
            (DT - c1 * (2.0 * a + DT)) / 2.0
        }
    };
    (a, b)
}

#[test]
fn additive_row_conserves_storage_for_a_time_varying_t() {
    // Headwater dam (reach 1) and interior dam (reach 3, fed by 1 and 2),
    // at short and long reaches, T rising and falling across the step.
    for (dam, len, up) in [(1, 5000.0, None), (3, 5000.0, Some(1)), (3, 20_000.0, Some(1))] {
        let (a, b) = channel_a_b(dam, len, up);
        println!("dam {dam} L {len}: K_r(1 - X_r) = {a:.1} s, K_r X_r = {b:.1} s");
        assert!(a > 0.0 && a.is_finite());
        for (tp, tn) in [(3_600.0_f32, 90_000.0_f32), (500_000.0, 40_000.0), (0.0, 7_200.0)] {
            let (q, _) = one_step(dam, len, Q_T, Q_LAT, tp, tn, DamRow::Additive, false);
            let (qn, qt, lat) = (q[dam] as f64, Q_T[dam] as f64, Q_LAT[dam] as f64);
            // Inflow from upstream at the step's start and end (reaches 1, 2 feed 3).
            let (i_t, i_n) = match up {
                None => (0.0, 0.0),
                Some(_) => ((Q_T[1] + Q_T[2]) as f64, (q[1] + q[2]) as f64),
            };
            let b = if up.is_some() { b } else { 0.0 };
            // S_{t+1} − S_t with S = B·I + A·Q + T·Q, against dt·[(I_t + I_{t+1})/2 + q' − (Q_t + Q_{t+1})/2].
            let ds = b * (i_n - i_t) + (a + tn as f64) * qn - (a + tp as f64) * qt;
            let flux = DT * (0.5 * (i_t + i_n) + lat - 0.5 * (qt + qn));
            let scale = (a + tn as f64) * qn + (a + tp as f64) * qt + DT * lat;
            let err = (ds - flux).abs() / scale;
            println!("  T_t {tp} T_t+1 {tn}: dS {ds:.6e} flux {flux:.6e} err/scale {err:.2e}");
            assert!(err < 2e-6, "dam {dam} L {len}: storage imbalance {err:.2e}");
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Gradchecks.
// ---------------------------------------------------------------------------

/// Sign of the dam row's `c1`: the dam's `Q_{t+1}` response to more inflow
/// from upstream at the same step. `c1 < 0` exactly when `K_r·X_r > dt/2`.
fn c1_sign(dam_len: f32) -> f64 {
    let (_, b) = channel_a_b(3, dam_len, Some(1));
    (DT / 2.0 - b).signum()
}

#[derive(Clone, Copy, Debug)]
enum Parent {
    T0,
    A,
    B,
    NDam,
    NUp,
    Gamma,
}

fn fd(s: Setup, t0: f32, ab: Option<(f32, f32)>, p: Parent, eps_ab: f32, eps_n: f32) -> f64 {
    let arm = |t0: f32, ab: Option<(f32, f32)>| Arm::Release(t0, ab, DamRow::Additive);
    let eval = |s: Setup, t0: f32, ab: Option<(f32, f32)>| loss_f64(s, arm(t0, ab));
    let (x0, eps) = match p {
        // At T0 = 0 the step is absolute: 1e-3 d (86 s). A 1e-2 d step (864 s) is
        // not small against D = K_r(1 − X_r) + T + dt/2 on a 3 km reach, and the
        // rational dependence on T then costs ~1 % of truncation error.
        Parent::T0 => (t0, if t0 == 0.0 { 1e-3 } else { 1e-2 * t0.abs() }),
        Parent::A => (ab.unwrap().0, eps_ab),
        Parent::B => (ab.unwrap().1, eps_ab),
        Parent::NDam => (s.n_dam, eps_n),
        Parent::NUp => (s.n, (1e-2 * s.n.abs()).max(1e-2)),
        Parent::Gamma => (s.gamma.unwrap(), (1e-2 * s.gamma.unwrap().abs()).max(1e-2)),
    };
    let at = |x: f32| match p {
        Parent::T0 => eval(s, x, ab),
        Parent::A => eval(s, t0, ab.map(|(_, b)| (x, b))),
        Parent::B => eval(s, t0, ab.map(|(a, _)| (a, x))),
        Parent::NDam => eval(Setup { n_dam: x, ..s }, t0, ab),
        // Every reach but the dam's: an upstream parameter field.
        Parent::NUp => eval(Setup { n: x, ..s }, t0, ab),
        Parent::Gamma => eval(Setup { gamma: Some(x), ..s }, t0, ab),
    };
    let (hi, lo) = (x0 + eps, x0 - eps);
    (at(hi) - at(lo)) / (hi as f64 - lo as f64)
}

/// `abs_tol_n`: the absolute tolerance on the dam reach's `n` (its FD noise floor; see the
/// T0 = 20 d case). Every other parent is judged on `REL_TOL` or `ABS_TOL`.
fn check(
    label: &str,
    s: Setup,
    t0: f32,
    ab: Option<(f32, f32)>,
    eps_ab: f32,
    eps_n: f32,
    abs_tol_n: f64,
) -> Vec<String> {
    let arm = Arm::Release(t0, ab, DamRow::Additive);
    let (gn, gt0, gab, ggamma) = analytical(s, arm);
    let up_sum: f64 = (0..N_REACH).filter(|&r| r != s.dam).map(|r| gn[r] as f64).sum();
    let mut cases = vec![(Parent::T0, gt0.unwrap()), (Parent::NDam, gn[s.dam] as f64), (Parent::NUp, up_sum)];
    if t0 != 0.0 {
        // At T0 = 0 the seasonal factor multiplies zero: a, b get exactly 0.
        if let Some((ga, gb)) = gab {
            cases.push((Parent::A, ga));
            cases.push((Parent::B, gb));
        }
    }
    if let Some(g) = ggamma {
        cases.push((Parent::Gamma, g));
    }
    let mut failures = Vec::new();
    for (p, a) in cases {
        let f = fd(s, t0, ab, p, eps_ab, eps_n);
        let abs = (a - f).abs();
        let rel = abs / a.abs().max(f.abs()).max(1e-12);
        println!("{label} {p:?}: analytical={a:.6e} fd={f:.6e} abs={abs:.3e} rel={rel:.3e}");
        assert!(a != 0.0 && a.is_finite(), "{label} {p:?}: analytical gradient {a} is vacuous");
        let abs_tol = if matches!(p, Parent::NDam) { abs_tol_n } else { ABS_TOL };
        if !(rel < REL_TOL || abs < abs_tol) {
            failures.push(format!("{label} {p:?}: rel {rel:.3e} abs {abs:.3e}"));
        }
    }
    if t0 == 0.0 {
        if let Some((ga, gb)) = gab {
            assert_eq!((ga, gb), (0.0, 0.0), "{label}: a, b must get exactly 0 at T0 = 0");
        }
    }
    failures
}

#[test]
fn additive_gradcheck_over_t_and_both_c1_regimes() {
    // Short dam reach: K_r·X_r < dt/2 (c1 > 0); long: K_r·X_r > dt/2 (c1 < 0).
    // 3 km, not shorter: at 1 to 2 km the Cunge X reaches its [0, 0.5] floor (W = Q/(B·S·c·L) >= 1)
    // on part of the window, and a finite difference on the dam reach's n straddles that kink.
    assert!(c1_sign(3000.0) > 0.0, "3 km dam reach should have K_r X_r < dt/2");
    assert!(c1_sign(20_000.0) < 0.0, "20 km dam reach should have K_r X_r > dt/2");
    let mut failures = Vec::new();
    for (regime, len) in [("KX<dt/2", 3000.0_f32), ("KX>dt/2", 20_000.0)] {
        for (t0, eps_ab) in [(0.0_f32, 1e-2_f32), (0.1, 1e-2), (1.5, 1e-2), (20.0, 1e-1)] {
            let label = format!("{regime} T0 {t0}");
            // At T0 = 20 d the dam reach's own K_r is ~1e-3 of its storage, and on the
            // 3 km reach d loss / d n_dam is 0.50 against 20 to 6,000 for every other
            // parent: inside the f32 noise of this 72-step loss. Measured on that case,
            // the FD at steps 5e-3 / 1e-2 / 2e-2 / 5e-2 / 1e-1 / 2e-1 is 0.659 / 0.378 /
            // 0.482 / 0.516 / 0.514 / 0.576 (analytical 0.502): a loss noise of ~1.5e-3,
            // i.e. ±0.0075 at step 0.1, with truncation showing only at 0.2. So that
            // case takes step 0.1 and an absolute tolerance of 2e-2; the 20 km reach
            // (gradient 31.7) passes at rel < 3e-3 regardless. Every other case: 1e-2.
            let (eps_n, abs_tol_n) = if t0 >= 20.0 { (1e-1, 2e-2) } else { (1e-2, ABS_TOL) };
            failures.extend(check(
                &label,
                Setup::new(3, len),
                t0,
                Some((0.7, -0.4)),
                eps_ab,
                eps_n,
                abs_tol_n,
            ));
        }
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn additive_gradcheck_outlet_dam_constant_t() {
    let failures = check("outlet, constant T", Setup::new(4, 5000.0), 1.5, None, 1e-2, 1e-2, ABS_TOL);
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn additive_gradcheck_with_learned_gamma() {
    let s = Setup { gamma: Some(0.4), ..Setup::new(3, 5000.0) };
    let failures = check("gamma op", s, 0.8, Some((0.7, -0.4)), 1e-2, 1e-2, ABS_TOL);
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn additive_gradcheck_with_enforce_positivity() {
    // The S18'/S19' clamps act on the dam reach's own K_r, X_r, and B19' opens
    // the X-cap path into K_r, so the dam row's n gradient runs through it too.
    let s = Setup { enforce_positivity: true, ..Setup::new(3, 5000.0) };
    let failures = check("enforce_positivity", s, 0.8, Some((0.7, -0.4)), 1e-2, 1e-2, ABS_TOL);
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn additive_step_gradcheck_t_prev_and_t_next() {
    let mut failures = Vec::new();
    for (label, len, tp0, tn0) in [
        ("zero", 5000.0_f32, 0.0_f32, 0.0_f32),
        ("small", 5000.0, 7_200.0, 9_000.0),
        ("large, KX>dt/2", 20_000.0, 432_000.0, 480_000.0),
    ] {
        let (_, extra) = one_step(3, len, Q_T, Q_LAT, tp0, tn0, DamRow::Additive, true);
        let (loss, tp, tn) = extra.unwrap();
        let grads = loss.backward();
        let g = |t: &Tensor<AB, 1>| host(t.grad(&grads).expect("grad"))[0] as f64;
        for (which, analytical) in [("T_t", g(&tp)), ("T_t+1", g(&tn))] {
            let base = if which == "T_t" { tp0 } else { tn0 };
            let eps = (1e-2 * base).max(100.0);
            let bump = |d: f32| {
                let (p, n) = if which == "T_t" { (tp0 + d, tn0) } else { (tp0, tn0 + d) };
                let q = one_step(3, len, Q_T, Q_LAT, p, n, DamRow::Additive, false).0;
                q.iter().zip(STEP_WEIGHTS).map(|(&v, w)| v as f64 * w as f64).sum::<f64>()
            };
            let fd = (bump(eps) - bump(-eps)) / (2.0 * eps as f64);
            let abs = (analytical - fd).abs();
            let rel = abs / analytical.abs().max(fd.abs()).max(1e-30);
            println!("{label} {which}: analytical={analytical:.6e} fd={fd:.6e} rel={rel:.3e}");
            assert!(analytical != 0.0, "{label} {which}: vacuous gradient");
            if !(rel < REL_TOL || abs < 1e-9) {
                failures.push(format!("{label} {which}: rel {rel:.3e}"));
            }
        }
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

/// End to end: one read-out weight of the release head, through
/// `release_params`, the per-step seasonal `T`, the additive row and the solve.
#[test]
fn additive_release_head_weight_gradcheck_end_to_end() {
    use burn::module::Param;
    use ddrs::config::{ParameterRanges, ReleaseHeadSection};
    use ddrs::nn::release_head::{init_release_head, release_params};

    let device = Device::default();
    let mut ranges = ParameterRanges::default();
    ranges.reservoir_t0 = [1e-4, 365.0];
    let section = ReleaseHeadSection {
        hidden_size: 6,
        num_hidden_layers: 1,
        grid: 5,
        k: 3,
        input_var_names: vec!["f1".into(), "f2".into()],
        seasonal: true,
        dam_row: DamRow::Additive,
        routing_checkpoint: None,
        freeze_routing: false,
    };
    let features = || Tensor::<AB, 2>::from_floats([[0.8_f32, -1.2]], &device);
    let head0 = {
        let mut h = init_release_head::<AB>(&section, &ranges, 42, &device);
        h.output.bias = Some(Param::from_tensor(Tensor::from_floats([0.3_f32, 0.6, -0.5], &device)));
        h
    };
    let (row, col) = (2, 0);
    let with_weight = |delta: f32| {
        let mut h = head0.clone();
        let w = h.output.weight.val();
        let mut v: Vec<f32> = w.clone().into_data().to_vec().unwrap();
        let [_, p] = w.dims();
        v[row * p + col] += delta;
        h.output.weight = Param::from_tensor(Tensor::<AB, 1>::from_floats(v.as_slice(), &device).reshape(w.dims()));
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
        let inputs = RoutingInputs::<I> { adjacency: sandbox(3, 5000.0), x_storage: Tensor::ones([N_REACH], &device) * 0.3 };
        let q = Tensor::<AB, 1>::from_floats(q_prime().as_slice(), &device).reshape([STEPS + 1, N_REACH]);
        let mut mc = MuskingumCunge::<I>::new(Config::default(), device);
        mc.setup_inputs(inputs, q, spatial, false, None);
        mc.set_dam_release(DamRelease {
            rows: vec![3],
            t0_days: rp.t0_days,
            seasonal: rp.seasonal,
            phase: phase(),
            dam_row: DamRow::Additive,
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
    println!("additive release head weight [{row},{col}]: analytical={analytical:.6e} fd={fd:.6e} rel={rel:.3e}");
    assert!(analytical != 0.0 && analytical.is_finite(), "vacuous weight gradient");
    assert!(rel < REL_TOL || abs < ABS_TOL, "release head weight gradcheck rel {rel:.3e}");
}

// ---------------------------------------------------------------------------
// Validation.
// ---------------------------------------------------------------------------

#[test]
fn additive_option_c_accepts_zero_t_and_replace_keeps_its_hour() {
    let s = Setup::new(3, 5000.0);
    let device = Device::default();
    let engine = || {
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
        let inputs = RoutingInputs::<I> { adjacency: sandbox(s.dam, s.dam_len), x_storage: Tensor::ones([N_REACH], &device) * 0.3 };
        let q = Tensor::<AB, 1>::from_floats(q_prime().as_slice(), &device).reshape([STEPS + 1, N_REACH]);
        let mut mc = MuskingumCunge::<I>::new(Config::default(), device);
        mc.setup_inputs(inputs, q, spatial, false, None);
        mc
    };
    assert!(engine().set_reservoir_rows_as(&[3], &[0.0], DamRow::Additive).is_ok());
    let err = engine().set_reservoir_rows_as(&[3], &[-0.1], DamRow::Additive).unwrap_err();
    assert!(err.contains(">= 0"), "{err}");
    let err = engine().set_reservoir_rows_as(&[3], &[0.01], DamRow::Replace).unwrap_err();
    assert!(err.contains("1/24"), "{err}");
    let err = engine().set_reservoir_rows(&[3], &[0.01]).unwrap_err();
    assert!(err.contains("1/24"), "{err}");
}

// ---------------------------------------------------------------------------
// 5. Opt-out on the Juniata bundle (the review synthesis' check 1b).
// ---------------------------------------------------------------------------

const JUNIATA_CONFIG: &str = "examples/juniata/ddrs.yaml";
const JUNIATA_DAMS: &str = "examples/juniata/data/juniata_dam_features.csv";
const JUNIATA_DAM_FEATURES: [&str; 19] = [
    "log10_storage",
    "log10_storage_max",
    "log10_surface",
    "log10_drainage",
    "log10_storage_per_area",
    "log10_max_discharge",
    "height",
    "year",
    "log10_storage_max_missing",
    "log10_surface_missing",
    "log10_max_discharge_missing",
    "year_missing",
    "purpose_flood",
    "purpose_hydro",
    "purpose_supply",
    "purpose_irrigation",
    "purpose_recreation",
    "purpose_navigation",
    "purpose_other",
];

/// The Juniata bundle's config, with the learned release on Raystown Dam in
/// `row` form when `row` is `Some`, and `T0` in `[t0_lo, 365]` d.
fn juniata_cfg(dir: &std::path::Path, row: Option<DamRow>, t0_lo: f32) -> Config {
    let mut cfg: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(JUNIATA_CONFIG).unwrap()).unwrap();
    if let Some(row) = row {
        cfg["params"]["use_reservoirs"] = true.into();
        cfg["params"]["reservoir_release"] = "learned".into();
        cfg["params"]["parameter_ranges"]["reservoir_T0"] =
            serde_yaml::Value::Sequence(vec![t0_lo.into(), 365.0_f32.into()]);
        cfg["data_sources"]["reservoirs"] = JUNIATA_DAMS.into();
        let mut rh = serde_yaml::Mapping::new();
        rh.insert(
            "input_var_names".into(),
            serde_yaml::Value::Sequence(JUNIATA_DAM_FEATURES.iter().map(|s| (*s).into()).collect()),
        );
        rh.insert("seasonal".into(), false.into());
        let name = match row {
            DamRow::Replace => "replace",
            DamRow::Additive => "additive",
        };
        rh.insert("dam_row".into(), name.into());
        cfg["release_head"] = serde_yaml::Value::Mapping(rh);
    }
    let path = dir.join(format!("{row:?}.yaml"));
    std::fs::write(&path, serde_yaml::to_string(&cfg).unwrap()).unwrap();
    Config::from_yaml_file(&path).expect("config loads")
}

/// Hourly gauge series of one 90-day window, routed with the seed-42
/// routing head and, when the config has the learned release, a release head
/// whose `T0` read-out bias is `t0_bias` (`-40` saturates it at the floor).
fn juniata_series(cfg: &Config, t0_bias: Option<f32>) -> Vec<f64> {
    use burn::module::Param;
    use rand::SeedableRng;
    let device = Device::default();
    let ds = ddrs::data::MeritGagesDataset::open(cfg).unwrap();
    let rho = cfg.experiment.as_ref().unwrap().rho.unwrap();
    let window = ds
        .time_axis()
        .sample_rho_window(&mut rand_chacha::ChaCha12Rng::seed_from_u64(11), rho);
    let staids = ds.staids().to_vec();
    let tensors = ds.collate(&staids, &window).unwrap().to_tensors::<AB>(&device);
    let head = ddrs::config::kan_config(cfg.kan_head.as_ref().unwrap(), cfg.seed).init::<AB>(&device);
    let release = t0_bias.map(|bias| {
        let mut h = ddrs::nn::release_head::init_release_head::<AB>(
            cfg.release_head.as_ref().unwrap(),
            &cfg.params.parameter_ranges,
            cfg.seed,
            &device,
        );
        h.output.bias = Some(Param::from_tensor(Tensor::from_floats([bias], &device)));
        h
    });
    let q = ddrs::training::forward::forward_with_release::<I>(
        cfg,
        &tensors,
        &head,
        release.as_ref(),
        &device,
        false,
        None,
    );
    q.into_data().to_vec::<f32>().unwrap().iter().map(|&v| v as f64).collect()
}

fn nse(sim: &[f64], obs: &[f64]) -> f64 {
    let mean = obs.iter().sum::<f64>() / obs.len() as f64;
    let num: f64 = sim.iter().zip(obs).map(|(s, o)| (s - o).powi(2)).sum();
    let den: f64 = obs.iter().map(|o| (o - mean).powi(2)).sum();
    1.0 - num / den
}

#[test]
fn additive_row_at_its_t0_floor_opts_out_on_juniata() {
    let tmp = tempfile::tempdir().unwrap();
    let no_dam = juniata_series(&juniata_cfg(tmp.path(), None, 0.0), None);
    let lo = 1e-4_f32; // 8.64 s
    let additive = juniata_cfg(tmp.path(), Some(DamRow::Additive), lo);
    let floor = juniata_series(&additive, Some(-40.0));
    // The init bias: T0 = 4.5 h, the release head's starting point.
    let init_bias = ddrs::nn::release_head::release_init_bias(false, &additive.params.parameter_ranges)[0];
    let init = juniata_series(&additive, Some(init_bias));
    // A 10-day bucket, the scale the learned heads reach at large dams.
    let [t_lo, t_hi] = additive.params.parameter_ranges.reservoir_t0;
    let u = ((10.0_f32).ln() - t_lo.ln()) / (t_hi.ln() - t_lo.ln());
    let ten_days = juniata_series(&additive, Some((u / (1.0 - u)).ln()));
    let replace_floor = juniata_series(&juniata_cfg(tmp.path(), Some(DamRow::Replace), 1.0 / 24.0), Some(-40.0));
    let [n_floor, n_init, n_ten, n_replace] =
        [&floor, &init, &ten_days, &replace_floor].map(|s| nse(s, &no_dam));
    println!(
        "Juniata gauge vs no dam (1 − NSE): additive at the T0 floor ({lo} d) {:.3e}; additive at \
         the 4.5 h init {:.3e}; additive at 10 d {:.3e}; replace at its 1 h floor {:.3e}",
        1.0 - n_floor,
        1.0 - n_init,
        1.0 - n_ten,
        1.0 - n_replace
    );
    // Check 1b: the additive row at its floor IS the no-dam routing.
    assert!(n_floor > 0.9999, "additive row at its floor does not opt out: NSE {n_floor}");
    // Not vacuous: Raystown is one tributary of the Newport gauge, so a
    // 4.5 h bucket moves it little, but it moves it far more than the floor,
    // and a 10-day bucket visibly.
    assert!(1.0 - n_init > 10.0 * (1.0 - n_floor), "the 4.5 h additive row should act more than its floor");
    assert!(n_ten < 0.999, "a 10-day additive bucket should visibly act: NSE {n_ten}");
    // The replace row cannot opt out: at its 1 h floor it is further from the
    // no-dam routing than the additive row at its floor (it drops the reach's K_r).
    assert!(n_replace < n_floor, "replace at 1 h should be further from no-dam than additive at the floor");
}
