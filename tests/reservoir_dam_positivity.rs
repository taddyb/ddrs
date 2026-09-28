//! Dam-row positivity (`release_head.dam_row_positivity` /
//! `params.reservoir_dam_row_positivity`, S19p / B19p in
//! `src/routing/mmc_op.rs`). The additive dam row keeps its reach's channel
//! wedge `K_r·X_r·I` in its storage, so its inflow coefficient
//! `c1 = (dt − 2·K_r·X_r)/D` is negative wherever `K_r·X_r > dt/2`: at low
//! outflow the Cunge `K_r` is days and `X_r` is 0.5. With positivity on, the
//! dam rows (only) route with `X_eff = min(X_r, 0.5·(1 − δ)·dt/K_r)`, so
//! `c1 >= δ·dt/D > 0`, and every coefficient and `D` read `X_eff`.
//!
//! 1. Off by default; where the cap does not bind (`K_r·X_r < dt/2` on every
//!    step) positivity on is bitwise the additive row, outputs and gradients,
//!    on the release path and option C rows; under `enforce_positivity` S19p
//!    is skipped (S19' already caps every row at the same bound).
//! 2. `c1 >= 0` on every dam-row step of a forced low-flow case, where the
//!    additive row without the cap has `c1 < 0` on most steps; the reach above
//!    the dam is untouched.
//! 3. Gradcheck of `T0`, `a`, `b`, the dam reach's `n` (through `K_r`, which
//!    alone carries the gradient where the cap binds) and an upstream `n`, at
//!    `T0` in `{0, 0.1, 1.5, 20}` d, with `K_r·X_r` below and above `dt/2`
//!    (cap inactive / active on every step); under `ddr_match` (a constant
//!    `X`, where the cap is `X`'s only path); and the two `T` parents on
//!    their own at one step, with the one-step `c1` measured.
//! 4. The v4 synthetic debt pump (diurnal upstream swing, a debt larger than
//!    the dam's storage, the carried floor): without the cap the dam creates
//!    new debt of the order of its opening debt and still owes at the end;
//!    with it the dam creates next to nothing, clears the debt within the
//!    days its inflow allows, and the carried account closes.
//! 5. With the cap binding on every step and `K`, `X` held constant, the
//!    carried dam row's volume balance read off the routed series closes
//!    over one rule-curve year, with `S = K·X_eff·I + (K·(1 − X_eff) + T)·Q`.

use std::sync::Arc;

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use chrono::{Duration, NaiveDate};

use ddrs::config::{Config, DamFloor, DamRow};
use ddrs::routing::mmc::{DamClampAccount, DamRelease, RuleCurve, DT_SECONDS};
use ddrs::routing::mmc_op::POSITIVITY_DELTA;
use ddrs::routing::release::{rule_curve_h, rule_curve_phase_start, seasonal_phase, OMEGA_RAD_PER_S};
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

fn network(n: usize, edges: &[(usize, usize)], length: Vec<f32>) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n * n];
    for &(up, down) in edges {
        dense[down * n + up] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, length, vec![0.001; n])
}

/// The RAPID sandbox `1 → 3`, `2 → 3`, `0 → 4`, `3 → 4`; reach `long` gets
/// `long_m` metres, every other reach 5 km.
fn sandbox(long: usize, long_m: f32) -> SparseAdjacency {
    let mut length = vec![5000.0_f32; N_REACH];
    length[long] = long_m;
    network(N_REACH, &[(1, 3), (2, 3), (0, 4), (3, 4)], length)
}

/// `0 → 1 → 2`, every reach `length` metres.
fn chain3(length: f32) -> SparseAdjacency {
    network(3, &[(0, 1), (1, 2)], vec![length; 3])
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

fn spatial(n: Tensor<AB, 1>, reaches: usize, device: &Device) -> SpatialParameters<I> {
    SpatialParameters::<I> {
        n,
        q_spatial: Tensor::from_floats(vec![0.5_f32; reaches].as_slice(), device),
        p_spatial: Some(Tensor::from_floats(vec![0.575_f32; reaches].as_slice(), device)),
        k_d: None,
        d_gw: None,
        leakance_factor: None,
        impervious_mask: None,
        gamma: None,
    }
}

// ---------------------------------------------------------------------------
// The sandbox harness (a dam on reach 3, fed by 1 and 2).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Arm {
    /// `set_reservoir_rows_as(.., Additive)`, constant `T` (days).
    OptionC(f32),
    /// `set_dam_release` on the additive row: `T0` (days), `Some((a, b))` seasonal.
    Release(f32, Option<(f32, f32)>),
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    dam: usize,
    dam_len: f32,
    n: f32,
    n_dam: f32,
    positivity: bool,
    ddr_match: bool,
    enforce_positivity: bool,
}

impl Setup {
    fn new(dam_len: f32, positivity: bool) -> Self {
        Self { dam: 3, dam_len, n: 0.5, n_dam: 0.5, positivity, ddr_match: false, enforce_positivity: false }
    }

    fn cfg(&self) -> Config {
        let mut cfg = Config::default();
        cfg.params.reservoir_dam_row_positivity = Some(self.positivity);
        cfg.params.ddr_match = self.ddr_match;
        cfg.params.enforce_positivity = self.enforce_positivity;
        cfg
    }
}

struct Routed {
    q: Tensor<AB, 2>,
    n: Tensor<AB, 1>,
    t0: Option<Tensor<AB, 1>>,
    ab: Option<(Tensor<AB, 1>, Tensor<AB, 1>)>,
    account: DamClampAccount,
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
    let inputs = RoutingInputs::<I> {
        adjacency: sandbox(s.dam, s.dam_len),
        x_storage: Tensor::ones([N_REACH], &device) * 0.3,
    };
    let q = Tensor::<AB, 1>::from_floats(q_prime().as_slice(), &device).reshape([STEPS + 1, N_REACH]);
    let mut mc = MuskingumCunge::<I>::new(s.cfg(), device);
    assert_eq!(mc.dam_row_positivity(), s.positivity, "the engine takes the config's setting");
    mc.setup_inputs(inputs, q, spatial(n.clone(), N_REACH, &device), false, None);
    let (mut t0_leaf, mut ab_leaf) = (None, None);
    match arm {
        Arm::OptionC(t) => mc.set_reservoir_rows_as(&[s.dam], &[t], DamRow::Additive).expect("rows"),
        Arm::Release(t0, ab) => {
            let t0 = leaf(vec![t0]);
            let ab = ab.map(|(a, b)| (leaf(vec![a]), leaf(vec![b])));
            mc.set_dam_release(DamRelease {
                rows: vec![s.dam],
                t0_days: t0.clone(),
                seasonal: ab.clone(),
                phase: phase(),
                dam_row: DamRow::Additive,
                rule_curve: None,
            })
            .expect("release");
            t0_leaf = Some(t0);
            ab_leaf = ab;
        }
    }
    let q = mc.forward();
    Routed { q, n, t0: t0_leaf, ab: ab_leaf, account: mc.dam_account().expect("dams armed") }
}

fn forward(s: Setup, arm: Arm) -> Vec<f32> {
    route(s, arm, false).q.into_data().to_vec().unwrap()
}

fn loss_f64(s: Setup, arm: Arm) -> f64 {
    forward(s, arm).iter().zip(weights()).map(|(&v, w)| v as f64 * w as f64).sum()
}

/// Gradients of the weighted loss: `n` per reach, `T0`, `(a, b)`.
struct Grads {
    n: Vec<f32>,
    t0: Option<Vec<f32>>,
    ab: Option<(Vec<f32>, Vec<f32>)>,
}

fn analytical(s: Setup, arm: Arm) -> (Vec<f32>, Grads, DamClampAccount) {
    let device = Device::default();
    let r = route(s, arm, true);
    let out = host(r.q.clone().reshape([N_REACH * (STEPS + 1)]));
    let w = Tensor::<AB, 1>::from_floats(weights().as_slice(), &device).reshape([N_REACH, STEPS + 1]);
    let grads = (r.q * w).sum().backward();
    let g = |t: &Tensor<AB, 1>| host(t.grad(&grads).expect("tracked"));
    let grads = Grads {
        n: g(&r.n),
        t0: r.t0.as_ref().map(|t| g(t)),
        ab: r.ab.as_ref().map(|(a, b)| (g(a), g(b))),
    };
    (out, grads, r.account)
}

// ---------------------------------------------------------------------------
// 1. Off by default; bitwise where the cap does not bind.
// ---------------------------------------------------------------------------

#[test]
fn positivity_is_off_by_default_and_follows_the_config() {
    let device = Device::default();
    assert!(!Config::default().dam_row_positivity());
    let mut mc = MuskingumCunge::<I>::new(Config::default(), device);
    assert!(!mc.dam_row_positivity());
    mc.set_dam_row_positivity(true);
    assert!(mc.dam_row_positivity());
    assert!(MuskingumCunge::<I>::new(Setup::new(3000.0, true).cfg(), device).dam_row_positivity());
}

#[test]
fn a_cap_that_never_binds_is_bitwise_the_additive_row() {
    // 3 km dam reach: K_r·X_r < dt/2 on every step (c1 > 0 without the cap).
    let arms = [
        Arm::Release(0.0, Some((0.7, -0.4))),
        Arm::Release(0.1, Some((0.7, -0.4))),
        Arm::Release(1.5, None),
        Arm::Release(20.0, Some((-0.8, 0.5))),
        Arm::OptionC(0.8),
    ];
    for arm in arms {
        let (off_q, off_g, off_acc) = analytical(Setup::new(3000.0, false), arm);
        let (on_q, on_g, on_acc) = analytical(Setup::new(3000.0, true), arm);
        println!("{arm:?}: c1 min {:.4e} (off) / {:.4e} (on)", off_acc.c1_min[0], on_acc.c1_min[0]);
        assert_eq!(off_acc.neg_c1_steps, vec![0], "{arm:?}: precondition, c1 > 0 without the cap");
        assert_bitwise(&on_q, &off_q, &format!("{arm:?}: routed"));
        assert_bitwise(&on_g.n, &off_g.n, &format!("{arm:?}: dL/dn"));
        if let (Some(a), Some(b)) = (&on_g.t0, &off_g.t0) {
            assert_bitwise(a, b, &format!("{arm:?}: dL/dT0"));
        }
        if let (Some((a1, b1)), Some((a2, b2))) = (&on_g.ab, &off_g.ab) {
            assert_bitwise(a1, a2, &format!("{arm:?}: dL/da"));
            assert_bitwise(b1, b2, &format!("{arm:?}: dL/db"));
        }
        assert_eq!(on_acc, off_acc, "{arm:?}: account");
    }
    // Not vacuous: on a 20 km dam reach (K_r·X_r > dt/2) the cap acts ...
    let arm = Arm::Release(1.5, None);
    assert!(forward(Setup::new(20_000.0, true), arm) != forward(Setup::new(20_000.0, false), arm));
    // ... except under enforce_positivity, whose S19' caps every row at the
    // same bound: S19p is skipped and the key changes nothing.
    let enforced = |positivity: bool| Setup { enforce_positivity: true, ..Setup::new(20_000.0, positivity) };
    let (a, ga, _) = analytical(enforced(true), arm);
    let (b, gb, _) = analytical(enforced(false), arm);
    assert_bitwise(&a, &b, "enforce_positivity: routed");
    assert_bitwise(&ga.n, &gb.n, "enforce_positivity: dL/dn");
}

// ---------------------------------------------------------------------------
// 2. c1 >= 0 on every dam-row step at low flow.
// ---------------------------------------------------------------------------

/// `0 → 1 → 2`, 5 km reaches, the dam on row 1, the Cunge K and X
/// (`ddr_match: false`), `upstream` the lateral inflow of row 0 per step.
/// Returns the routed `[3, steps + 1]` output and the dam's account.
#[allow(clippy::too_many_arguments)]
fn chain_run(
    cfg: &Config,
    upstream: &[f32],
    own: f32,
    t_days: f32,
    option_c: bool,
    start: NaiveDate,
    initial: Option<&[f32]>,
    owed: Option<f64>,
) -> (Vec<f32>, DamClampAccount) {
    let device = Device::default();
    let rows = upstream.len();
    let q: Vec<f32> = upstream.iter().flat_map(|&u| [u, own, 0.0]).collect();
    let mut mc = MuskingumCunge::<I>::new(cfg.clone(), device);
    mc.setup_inputs(
        RoutingInputs { adjacency: chain3(5000.0), x_storage: Tensor::ones([3], &device) * 0.3 },
        Tensor::<AB, 1>::from_floats(q.as_slice(), &device).reshape([rows, 3]),
        spatial(Tensor::from_floats([0.5_f32; 3], &device), 3, &device),
        false,
        initial.map(|v| Tensor::<AB, 1>::from_floats(v, &device)),
    );
    if option_c {
        mc.set_reservoir_rows_as(&[1], &[t_days], DamRow::Additive).unwrap();
    } else {
        mc.set_dam_release(DamRelease {
            rows: vec![1],
            t0_days: Tensor::from_floats([t_days], &device),
            seasonal: None,
            phase: seasonal_phase(start, rows),
            dam_row: DamRow::Additive,
            rule_curve: None,
        })
        .unwrap();
    }
    if let Some(o) = owed {
        mc.set_dam_owed(&[o]).unwrap();
    }
    let out: Vec<f32> = mc.forward().into_data().to_vec().unwrap();
    (out, mc.dam_account().unwrap())
}

/// `mean·(1 + amp·sin(2π·s/24))` for rows `0..=steps`.
fn diurnal(steps: usize, mean: f32, amp: f32) -> Vec<f32> {
    (0..=steps)
        .map(|s| mean * (1.0 + amp * (2.0 * std::f32::consts::PI * s as f32 / 24.0).sin()))
        .collect()
}

fn with(floor: DamFloor, positivity: bool) -> Config {
    let mut cfg = Config::default();
    cfg.params.reservoir_dam_floor = Some(floor);
    cfg.params.reservoir_dam_row_positivity = Some(positivity);
    cfg
}

fn start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2001, 1, 1).unwrap()
}

#[test]
fn c1_stays_nonnegative_on_every_dam_row_step_at_low_flow() {
    // 0.5 m3/s ± 30 % from upstream, 0.05 m3/s of its own: the dam's outflow
    // stays low, where the Cunge K_r is long and X_r near 0.5.
    let steps = 10 * 24;
    let upstream = diurnal(steps, 0.5, 0.3);
    for (t_days, option_c) in [(0.05_f32, false), (1.67, false), (0.5, true)] {
        let run = |pos: bool| chain_run(&with(DamFloor::Forgive, pos), &upstream, 0.05, t_days, option_c, start(), None, None);
        let (off, off_acc) = run(false);
        let (on, on_acc) = run(true);
        println!(
            "T {t_days} d (option C {option_c}): off: c1 min {:.4e}, {} of {} steps c1 < 0, {} clamp steps | \
             on: c1 min {:.4e}, {} steps c1 < 0, {} clamp steps",
            off_acc.c1_min[0], off_acc.neg_c1_steps[0], off_acc.steps, off_acc.clamp_steps[0],
            on_acc.c1_min[0], on_acc.neg_c1_steps[0], on_acc.clamp_steps[0]
        );
        assert!(
            off_acc.c1_min[0] < 0.0 && off_acc.neg_c1_steps[0] > off_acc.steps / 2,
            "T {t_days}: the case must force c1 < 0 on most steps without the cap"
        );
        assert_eq!(on_acc.neg_c1_steps[0], 0, "T {t_days}: c1 < 0 on a dam-row step with the cap");
        assert!(on_acc.c1_min[0] > 0.0, "T {t_days}: c1 min {}", on_acc.c1_min[0]);
        // The reach above the dam is a channel row: untouched.
        let cols = steps + 1;
        assert_bitwise(&on[..cols], &off[..cols], &format!("T {t_days}: the reach above the dam"));
        assert!(on[cols..2 * cols] != off[cols..2 * cols], "T {t_days}: the dam row changes");
    }
}

// ---------------------------------------------------------------------------
// 3. Gradchecks.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Parent {
    T0,
    A,
    B,
    NDam,
    NUp,
}

fn fd(s: Setup, t0: f32, ab: Option<(f32, f32)>, p: Parent, eps_ab: f32, eps_n: f32) -> f64 {
    let eval = |s: Setup, t0: f32, ab: Option<(f32, f32)>| loss_f64(s, Arm::Release(t0, ab));
    let (x0, eps) = match p {
        Parent::T0 => (t0, if t0 == 0.0 { 1e-3 } else { 1e-2 * t0.abs() }),
        Parent::A => (ab.unwrap().0, eps_ab),
        Parent::B => (ab.unwrap().1, eps_ab),
        Parent::NDam => (s.n_dam, eps_n),
        Parent::NUp => (s.n, (1e-2 * s.n.abs()).max(1e-2)),
    };
    let at = |x: f32| match p {
        Parent::T0 => eval(s, x, ab),
        Parent::A => eval(s, t0, ab.map(|(_, b)| (x, b))),
        Parent::B => eval(s, t0, ab.map(|(a, _)| (a, x))),
        Parent::NDam => eval(Setup { n_dam: x, ..s }, t0, ab),
        Parent::NUp => eval(Setup { n: x, ..s }, t0, ab),
    };
    let (hi, lo) = (x0 + eps, x0 - eps);
    (at(hi) - at(lo)) / (hi as f64 - lo as f64)
}

fn check(label: &str, s: Setup, t0: f32, ab: Option<(f32, f32)>, eps_ab: f32, eps_n: f32, abs_tol_n: f64) -> Vec<String> {
    let (_, g, _) = analytical(s, Arm::Release(t0, ab));
    let up_sum: f64 = (0..N_REACH).filter(|&r| r != s.dam).map(|r| g.n[r] as f64).sum();
    let mut cases = vec![(Parent::T0, g.t0.as_ref().unwrap()[0] as f64), (Parent::NDam, g.n[s.dam] as f64), (Parent::NUp, up_sum)];
    if t0 != 0.0 {
        if let Some((ga, gb)) = &g.ab {
            cases.push((Parent::A, ga[0] as f64));
            cases.push((Parent::B, gb[0] as f64));
        }
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
    failures
}

#[test]
fn gradcheck_with_positivity_over_t_and_both_regimes() {
    // Two regimes, each held on every step of the window at every T0:
    // - cap inactive: a 2.5 km dam reach at n = 0.3, where K_r·X_r stays
    //   below (1 − δ)·dt/2, so positivity on routes bitwise like off (checked
    //   here per T0; at 3 km and n = 0.5 the seasonal T0 = 1.5 and 20 d cases
    //   dip below on 12 and 37 steps, which would put a finite difference on
    //   the min's kink);
    // - cap active: a 20 km dam reach, c1 < 0 on every step without the cap.
    let ab = Some((0.7_f32, -0.4_f32));
    let mut failures = Vec::new();
    for (regime, len, n) in [("cap inactive", 2500.0_f32, 0.3_f32), ("cap active", 20_000.0, 0.5)] {
        let setup = |pos: bool| Setup { n, n_dam: n, ..Setup::new(len, pos) };
        for (t0, eps_ab) in [(0.0_f32, 1e-2_f32), (0.1, 1e-2), (1.5, 1e-2), (20.0, 1e-1)] {
            let label = format!("{regime} T0 {t0}");
            let on = route(setup(true), Arm::Release(t0, ab), false);
            let off = route(setup(false), Arm::Release(t0, ab), false);
            let (on_c1, on_neg) = (on.account.c1_min[0], on.account.neg_c1_steps[0]);
            assert!(on_c1 > 0.0 && on_neg == 0, "{label}: c1 min {on_c1}");
            let flat = |q: Tensor<AB, 2>| host(q.reshape([N_REACH * (STEPS + 1)]));
            let (on_q, off_q) = (flat(on.q), flat(off.q));
            if len < 5000.0 {
                assert!(off.account.c1_min[0] > 0.0, "{label}: c1 min {} without the cap", off.account.c1_min[0]);
                assert_bitwise(&on_q, &off_q, &format!("{label}: the cap never binds"));
            } else {
                assert_eq!(off.account.neg_c1_steps[0], STEPS as u64, "{label}: c1 < 0 on every step without the cap");
                assert!(on_q != off_q, "{label}: the cap acts");
            }
            // The dam reach's n at T0 = 20 d sits at this loss's f32 noise floor
            // (`tests/reservoir_additive.rs` records a step sweep): a larger step
            // and an absolute tolerance of 2e-2 there, as in that file. Cap
            // inactive (gradient -0.425), measured FD at steps 5e-3 / 1e-2 /
            // 2e-2 / 5e-2 / 1e-1 / 2e-1: -0.203 / -0.254 / -0.399 / -0.433 /
            // -0.460 / -1.038. Noise below 5e-2; at 1e-1 the n = 0.4 end already
            // binds the cap on some steps (without positivity: -0.450), so 5e-2.
            // Cap active (gradient 2.19): 1e-1, as in that file.
            let (eps_n, abs_tol_n) = match (t0 >= 20.0, len < 5000.0) {
                (true, true) => (5e-2, 2e-2),
                (true, false) => (1e-1, 2e-2),
                _ => (1e-2, ABS_TOL),
            };
            failures.extend(check(&label, setup(true), t0, ab, eps_ab, eps_n, abs_tol_n));
        }
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn gradcheck_with_positivity_under_ddr_match() {
    // A constant X = 0.3 (`x_storage`): the Cunge chain is absent, so where
    // the cap binds its K term is X's only path. 20 km: K_r·0.3 > dt/2.
    let s = Setup { ddr_match: true, ..Setup::new(20_000.0, true) };
    let off = route(Setup { positivity: false, ..s }, Arm::Release(0.8, None), false).account;
    assert_eq!(off.neg_c1_steps, vec![STEPS as u64], "precondition: the cap binds on every step");
    let mut failures = Vec::new();
    for t0 in [0.1_f32, 1.5] {
        failures.extend(check(&format!("ddr_match T0 {t0}"), s, t0, Some((0.7, -0.4)), 1e-2, 1e-2, ABS_TOL));
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

const STEP_WEIGHTS: [f32; 5] = [1.0, 1.1, 1.2, 1.3, 1.4];
const Q_T: [f32; 5] = [20.0, 9.0, 13.0, 40.0, 70.0];
const Q_LAT: [f32; 5] = [22.0, 11.0, 11.0, 11.0, 22.0];

/// One routed step on the sandbox, dam on reach 3 of length `dam_len`,
/// through `timestep_forward_release_as` (which reads the positivity from
/// the config). Returns every reach's `Q_{t+1}` and, when `grad`, the loss
/// and the two T leaves.
#[allow(clippy::type_complexity)]
fn one_step(
    dam_len: f32,
    q_lat: [f32; 5],
    t_prev: f32,
    t_next: f32,
    positivity: bool,
    grad: bool,
) -> (Vec<f32>, Option<(Tensor<AB, 1>, Tensor<AB, 1>, Tensor<AB, 1>)>) {
    use ddrs::sparse::{AValuesAssembler, CsrPattern};
    let device = Device::default();
    let adj = sandbox(3, dam_len);
    let pattern = Arc::new(CsrPattern::from_sparse(&adj));
    let assembler = AValuesAssembler::<I>::new(&pattern, &device);
    let c = |v: Vec<f32>| Tensor::<AB, 1>::from_floats(v.as_slice(), &device);
    let leaf = |v: f32| {
        let t = Tensor::<AB, 1>::from_floats([v], &device);
        if grad { t.require_grad() } else { t }
    };
    let (tp, tn) = (leaf(t_prev), leaf(t_next));
    let q = ddrs::routing::mmc_op::timestep_forward_release_as::<I>(
        &Setup::new(dam_len, positivity).cfg(),
        &pattern,
        &assembler,
        c(vec![0.05; N_REACH]),
        c(vec![0.5; N_REACH]),
        c(vec![21.0; N_REACH]),
        c(Q_T.to_vec()),
        c(q_lat.to_vec()),
        c(adj.length_m.clone()),
        c(adj.slope.clone()),
        c(vec![0.3; N_REACH]),
        vec![3],
        tn.clone(),
        tp.clone(),
        DamRow::Additive,
    );
    let out = host(q.clone());
    let extra = grad.then(|| ((q * c(STEP_WEIGHTS.to_vec())).sum(), tp, tn));
    (out, extra)
}

/// The dam row's `c1` at this step, `∂Q_dam/∂Q_up` within the step: more
/// lateral inflow on reach 1 moves reach 1's solve, and the dam row (3)
/// reads it only through `c1` (its own RHS uses the step-start inflow).
fn one_step_c1(dam_len: f32, t: f32, positivity: bool) -> f64 {
    let base = one_step(dam_len, Q_LAT, t, t, positivity, false).0;
    let mut bump = Q_LAT;
    bump[1] += 40.0;
    let moved = one_step(dam_len, bump, t, t, positivity, false).0;
    (moved[3] - base[3]) as f64 / (moved[1] - base[1]) as f64
}

#[test]
fn step_gradcheck_t_prev_and_t_next_with_positivity() {
    let mut failures = Vec::new();
    for (label, tp0, tn0) in [("zero", 0.0_f32, 0.0_f32), ("small", 7_200.0, 9_000.0), ("large", 432_000.0, 480_000.0)] {
        let (c1_off, c1_on) = (one_step_c1(20_000.0, tn0, false), one_step_c1(20_000.0, tn0, true));
        println!("{label}: c1 without the cap {c1_off:.4e}, with it {c1_on:.4e}");
        assert!(c1_off < 0.0, "{label}: precondition, K_r X_r > dt/2");
        assert!(c1_on >= 0.0, "{label}: c1 {c1_on:.4e} with the cap");
        let (_, extra) = one_step(20_000.0, Q_LAT, tp0, tn0, true, true);
        let (loss, tp, tn) = extra.unwrap();
        let grads = loss.backward();
        let g = |t: &Tensor<AB, 1>| host(t.grad(&grads).expect("grad"))[0] as f64;
        for (which, analytical) in [("T_t", g(&tp)), ("T_t+1", g(&tn))] {
            let base = if which == "T_t" { tp0 } else { tn0 };
            let eps = (1e-2 * base).max(100.0);
            let bump = |d: f32| {
                let (p, n) = if which == "T_t" { (tp0 + d, tn0) } else { (tp0, tn0 + d) };
                let q = one_step(20_000.0, Q_LAT, p, n, true, false).0;
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

// ---------------------------------------------------------------------------
// 4. The debt pump (review v3 finding 3 / the v4 re-evaluations).
// ---------------------------------------------------------------------------

/// The v4 scratch case: `0 → 1 → 2`, 5 km reaches, Cunge K and X, an
/// additive dam on row 1 with `T = t_days`, 10 m³/s ± 30 % diurnal from
/// upstream and 1 m³/s of its own, 30 days, carried floor, opening debt
/// `owed0` (more than the dam holds). Routed as daily chunks, threaded as the
/// test phase threads them (final discharge column and owed volume carried).
/// Returns the owed volume at the end of each day and the summed account
/// `(created, repaid, clamp steps, negative-c1 steps)`.
fn pump(positivity: bool, t_days: f32, owed0: f64) -> (Vec<f64>, (f64, f64, u64, u64)) {
    let days = 30;
    let upstream = diurnal(days * 24, 10.0, 0.3);
    let cfg = with(DamFloor::Carry, positivity);
    let (mut owed, mut state): (f64, Option<Vec<f32>>) = (owed0, None);
    let (mut created, mut repaid, mut clamps, mut neg) = (0.0, 0.0, 0, 0);
    let mut by_day = Vec::with_capacity(days);
    for d in 0..days {
        let rows = &upstream[d * 24..=(d + 1) * 24];
        let day = start() + Duration::days(d as i64);
        let (out, acc) = chain_run(&cfg, rows, 1.0, t_days, false, day, state.as_deref(), Some(owed));
        // The carried account closes step by step: owed_0 + created − repaid − owed = 0.
        let resid = owed + acc.created_m3[0] - acc.repaid_m3[0] - acc.owed_m3[0];
        assert!(resid.abs() <= 1e-5 * owed0, "day {d}: carried account residual {resid:.3e}");
        created += acc.created_m3[0];
        repaid += acc.repaid_m3[0];
        clamps += acc.clamp_steps[0];
        neg += acc.neg_c1_steps[0];
        owed = acc.owed_m3[0];
        by_day.push(owed);
        state = Some((0..3).map(|r| out[r * 25 + 24]).collect());
    }
    (by_day, (created, repaid, clamps, neg))
}

#[test]
fn positivity_stops_the_debt_pump() {
    let owed0 = 1e7;
    for t_days in [0.05_f32, 1.67] {
        let (off_days, (off_created, off_repaid, off_clamps, off_neg)) = pump(false, t_days, owed0);
        let (on_days, (on_created, on_repaid, on_clamps, on_neg)) = pump(true, t_days, owed0);
        let clear = |days: &[f64]| days.iter().position(|&o| o == 0.0);
        println!(
            "T {t_days} d, opening debt {owed0:.2e} m3 | without the cap: new debt {off_created:.4e}, repaid \
             {off_repaid:.4e}, owed at day 30 {:.4e}, cleared on day {:?}, {off_clamps} clamp steps, {off_neg} \
             steps c1 < 0 | with it: new debt {on_created:.4e}, repaid {on_repaid:.4e}, owed at day 30 {:.4e}, \
             cleared on day {:?}, {on_clamps} clamp steps, {on_neg} steps c1 < 0",
            off_days[29],
            clear(&off_days),
            on_days[29],
            clear(&on_days)
        );
        // Without the cap, at T = 0.05 d, the pump runs: the dam's negative c1
        // makes new debt of the order of the opening one, and it still owes
        // after 30 days of an inflow (2.85e7 m3) nearly three times the debt.
        // At T = 1.67 d the same c1 makes far less (4.6e5 m3): the dam's own
        // outflow takes days to fall to the floor, where the pump runs.
        assert!(off_neg > 0, "T {t_days}: c1 < 0 without the cap");
        if t_days < 0.1 {
            assert!(off_created > 0.3 * owed0, "T {t_days}: the case must pump without the cap");
            assert!(off_days[29] > 0.0, "T {t_days}: without the cap the debt outlives the window");
        }
        // With it: no negative c1, next to no new debt, the debt paid down
        // every day and cleared within the days the inflow needs.
        assert_eq!(on_neg, 0, "T {t_days}: c1 < 0 on a dam-row step with the cap");
        assert!(on_created <= 1e-3 * owed0, "T {t_days}: new debt {on_created:.3e} with the cap");
        let mut prev = owed0;
        for (d, &o) in on_days.iter().enumerate() {
            assert!(o <= prev, "T {t_days}: the debt grew on day {d}: {prev:.6e} -> {o:.6e}");
            prev = o;
        }
        let cleared = clear(&on_days).expect("the debt is cleared within the window");
        assert!(cleared <= 12, "T {t_days}: cleared only on day {cleared}");
        assert!(
            (on_repaid - owed0 - on_created).abs() <= 1e-5 * owed0,
            "T {t_days}: repaid {on_repaid:.6e} vs opening debt + new debt {:.6e}",
            owed0 + on_created
        );
    }
}

// ---------------------------------------------------------------------------
// 5. The carried volume balance closes with the cap binding.
// ---------------------------------------------------------------------------

/// One rule-curve period, 365.25 d, in hourly steps.
const YEAR_STEPS: usize = 8766;

/// `row` of a routed output `[n, cols]`, f64.
fn series(out: &[f32], cols: usize, row: usize) -> Vec<f64> {
    out[row * cols..(row + 1) * cols].iter().map(|&v| v as f64).collect()
}

/// `Σ dt·(v_t + v_{t+1})/2`, m³.
fn trapezoid(v: &[f64]) -> f64 {
    v.windows(2).map(|w| 0.5 * (w[0] + w[1]) * DT).sum()
}

#[test]
fn carry_closes_the_dam_row_volume_balance_with_the_cap_binding() {
    // K and X constant: `ddr_match` takes X from `x_storage` (0.3), and a
    // 2 m/s velocity floor, above every velocity the case reaches, fixes the
    // celerity at 2·5/3 m/s. On a 40 km reach K = 12,000 s, so
    // K·X = 3,600 s > dt/2 and the cap binds on every step:
    // X_eff = 0.5·(1 − δ)·dt/K = 0.1485.
    let steps = YEAR_STEPS;
    let length = 40_000.0_f32;
    let (t_days, ibar) = (0.05_f32, 8.0_f32);
    let q_dam = diurnal(steps, 2.0, 0.5);
    let q: Vec<f32> = q_dam.iter().flat_map(|&v| [4.0, v, 0.0]).collect();
    let k = (length / (2.0_f32 * (5.0_f32 / 3.0_f32))) as f64;
    let x_eff = 0.5 * (1.0 - POSITIVITY_DELTA as f64) * DT / k;
    assert!(0.3 > x_eff, "the case must bind the cap");
    let route = |floor: DamFloor, positivity: bool| {
        let device = Device::default();
        let mut cfg = with(floor, positivity);
        cfg.params.ddr_match = true;
        cfg.params.attribute_minimums.velocity = 2.0;
        let mut mc = MuskingumCunge::<I>::new(cfg, device);
        mc.setup_inputs(
            RoutingInputs { adjacency: chain3(length), x_storage: Tensor::ones([3], &device) * 0.3 },
            Tensor::<AB, 1>::from_floats(q.as_slice(), &device).reshape([steps + 1, 3]),
            spatial(Tensor::from_floats([0.5_f32; 3], &device), 3, &device),
            false,
            None,
        );
        mc.set_dam_release(DamRelease {
            rows: vec![1],
            t0_days: Tensor::from_floats([t_days], &device),
            seasonal: None,
            phase: seasonal_phase(start(), steps + 1),
            dam_row: DamRow::Additive,
            rule_curve: Some(RuleCurve {
                coeffs: Tensor::<AB, 1>::from_floats([1.0_f32, 0.0, 0.0, 0.0], &device).reshape([1, 4]),
                inflow_mean: Tensor::from_floats([ibar], &device),
                phase0: rule_curve_phase_start(start()),
            }),
        })
        .unwrap();
        let out: Vec<f32> = mc.forward().into_data().to_vec().unwrap();
        (out, mc.dam_account().unwrap())
    };
    let (out, acc) = route(DamFloor::Carry, true);
    let (_, off) = route(DamFloor::Carry, false);
    let cols = steps + 1;
    let (dam, up) = (series(&out, cols, 1), series(&out, cols, 0));
    let inflow = trapezoid(&up) + q_dam[..steps].iter().map(|&v| v as f64 * DT).sum::<f64>();
    let outflow = trapezoid(&dam);
    let t_s = t_days as f64 * 86_400.0;
    let storage = |i: usize| k * x_eff * up[i] + (k * (1.0 - x_eff) + t_s) * dam[i];
    let ds = storage(steps) - storage(0);
    let w0 = rule_curve_phase_start(start());
    let flux = ibar as f64 * (rule_curve_h(w0 + OMEGA_RAD_PER_S * DT * steps as f64)[0] - rule_curve_h(w0)[0]);
    let (created, repaid, owed) = (acc.created_m3[0], acc.repaid_m3[0], acc.owed_m3[0]);
    let resid = inflow - flux - outflow - ds;
    println!(
        "with the cap: inflow {inflow:.6e}, outflow {outflow:.6e}, dS {ds:.3e}, flux {flux:.3e}, created {created:.6e}, \
         repaid {repaid:.6e}, owed {owed:.3e}, balance {resid:.6e}, {} clamp steps, c1 min {:.4e}, {} steps c1 < 0 | \
         without it: {} steps c1 < 0, c1 min {:.4e}, created {:.6e}",
        acc.clamp_steps[0], acc.c1_min[0], acc.neg_c1_steps[0], off.neg_c1_steps[0], off.c1_min[0], off.created_m3[0]
    );
    assert_eq!(off.neg_c1_steps[0], steps as u64, "precondition: c1 < 0 on every step without the cap");
    assert_eq!(acc.neg_c1_steps[0], 0);
    assert!(acc.c1_min[0] > 0.0);
    assert!(acc.clamp_steps[0] > 0 && created > 0.01 * inflow, "the rule curve must force the clamp");
    assert!((created - repaid - owed).abs() <= 1e-4 * created, "carried account");
    assert!((resid + owed).abs() <= 1e-4 * created, "balance {resid:.6e} vs −owed {owed:.6e}");
    assert!((outflow - inflow).abs() <= 1e-3 * inflow, "outflow {outflow:.6e} vs inflow {inflow:.6e}");
}
