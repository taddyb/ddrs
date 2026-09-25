//! Linear-reservoir override of the Muskingum K and X on dam rows
//! (`MuskingumCunge::set_reservoir_rows`, `.claude/RESERVOIRS.md` option C).
//!
//! Muskingum's storage `S = K·[X·I + (1−X)·Q]` at `X = 0`, `K = T` is the
//! linear reservoir `S = T·Q`. On a dam row the timestep op replaces the K and
//! X it computed with the prescribed `T` (seconds) and 0. Every other row must
//! be bitwise unchanged, and the hand-written backward must give the dam row's
//! hydraulic parameters exactly zero gradient, since K and X are their only
//! consumers on a non-leakance row.
//!
//! All tests run on the deterministic `NdArray<f32>` backend with
//! `Config::default()` (`ddr_match: false`, `enforce_positivity: false`), the
//! supported combination; test 5 additionally covers `enforce_positivity`.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use ddrs::config::Config;
use ddrs::routing::mmc::DT_SECONDS;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const SECONDS_PER_DAY: f64 = 86_400.0;

/// Gradcheck tolerances, as in `tests/leakance_gradcheck.rs`.
const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 1e-4;

/// FD step, `max(EPS·|base|, EPS)` on the normalized `[0, 1]` parents.
///
/// `tests/leakance_gradcheck.rs` uses `EPS = 1e-3` on ONE timestep. On this
/// 24-step, 5-reach window a 1e-3 step puts the change in the loss at the f32
/// round-off floor of the routed discharge. Measured on `q_spatial` at reach
/// 1: FD steps of 3e-4, 1e-3 and 3e-3 scatter non-monotonically by up to 0.8%
/// around the analytical value, with or without the reservoir, while 1e-2 and
/// 3e-2 agree with it to < 1e-4. Every clamp is interior at this base point
/// (velocity 0.28 to 0.61 m/s, raw side slope 2.2 to 4.0, Cunge X 0.16 to
/// 0.40, Q >= 11 m³/s), so a 1e-2 step cannot straddle a kink. This is the same
/// trade `leakance_gradcheck` makes for its large K_D / d_gw / factor steps.
const EPS: f32 = 1e-2;

/// Normalized `[0, 1]` head outputs, one entry per reach.
#[derive(Clone)]
struct Params {
    n: Vec<f32>,
    q_spatial: Vec<f32>,
    p_spatial: Vec<f32>,
}

impl Params {
    /// Mid-box roughness and shape. `p_spatial = 0.575` denormalizes (log
    /// space, `[1, 200]`) to ~21, the config default.
    fn uniform(n_reach: usize) -> Self {
        Self {
            n: vec![0.5; n_reach],
            q_spatial: vec![0.5; n_reach],
            p_spatial: vec![0.575; n_reach],
        }
    }
}

/// Lower-triangular adjacency from `(upstream, downstream)` edges. Reaches are
/// 5 km long at slope 1e-3, the sandbox's attributes (`src/sandbox.rs::smoke`).
fn network(n_reach: usize, edges: &[(usize, usize)]) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n_reach * n_reach];
    for &(up, down) in edges {
        assert!(down > up, "topological order: downstream row after upstream");
        dense[down * n_reach + up] = 1.0;
    }
    SparseAdjacency::from_dense(n_reach, &dense, vec![5000.0; n_reach], vec![0.001; n_reach])
}

/// `0 → 1 → 2`.
fn chain3() -> SparseAdjacency {
    network(3, &[(0, 1), (1, 2)])
}

/// The 5-reach RAPID sandbox topology (`fixtures/sandbox/adjacency_topo.csv`):
/// `1 → 3`, `2 → 3`, `0 → 4`, `3 → 4`. Reach 3 is the only interior reach.
fn sandbox5() -> SparseAdjacency {
    network(5, &[(1, 3), (2, 3), (0, 4), (3, 4)])
}

/// A routed window plus the parameter leaves it was built from.
struct Routed {
    /// `[n_reach, steps + 1]`, column 0 is the initial state.
    q: Tensor<AB, 2>,
    n: Tensor<AB, 1>,
    q_spatial: Tensor<AB, 1>,
    p_spatial: Tensor<AB, 1>,
}

/// One routing window through `MuskingumCunge::forward`.
///
/// `q_prime` is row-major `[steps + 1, n_reach]`. `reservoir` is passed to
/// `set_reservoir_rows` AFTER `setup_inputs`, as the API requires.
fn route(
    cfg: &Config,
    adjacency: SparseAdjacency,
    q_prime: &[f32],
    params: &Params,
    reservoir: Option<(&[usize], &[f32])>,
    initial_state: Option<&[f32]>,
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
    let initial = initial_state.map(|v| Tensor::<AB, 1>::from_floats(v, &device));

    let mut mc = MuskingumCunge::<I>::new(cfg.clone(), device);
    mc.setup_inputs(inputs, q_prime, spatial, false, initial);
    if let Some((rows, t_days)) = reservoir {
        mc.set_reservoir_rows(rows, t_days).expect("valid reservoir rows");
    }
    Routed { q: mc.forward(), n, q_spatial, p_spatial }
}

fn host(t: Tensor<AB, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn host1(t: Tensor<I, 1>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

// ---------------------------------------------------------------------------
// 1. Forward: a dam row is exactly a linear reservoir; rows above it are
//    bitwise untouched.
// ---------------------------------------------------------------------------

#[test]
fn override_rows_route_as_linear_reservoir() {
    let cfg = Config::default();
    let n_reach = 3;
    let steps = 48;
    // Hot start from row 0, then a step in reach 0's lateral inflow from row
    // 1 on. `forward` routes step t with row t−1, so the step enters the
    // network at step 2 and the reservoir sees a rising inflow from then on.
    let mut q_prime = Vec::with_capacity((steps + 1) * n_reach);
    for t in 0..=steps {
        q_prime.extend_from_slice(&[if t == 0 { 10.0 } else { 40.0 }, 2.0, 3.0]);
    }
    let params = Params::uniform(n_reach);
    let t_days = 1.5_f32;

    let plain = host(route(&cfg, chain3(), &q_prime, &params, None, None, false).q);
    let res = host(
        route(&cfg, chain3(), &q_prime, &params, Some((&[1], &[t_days])), None, false).q,
    );
    let cols = steps + 1;

    // Reach 0 is upstream of the dam: the forward substitution never reads a
    // downstream row, so its series must be bit-for-bit the no-reservoir one.
    for t in 0..cols {
        assert_eq!(
            res[t].to_bits(),
            plain[t].to_bits(),
            "reach 0 step {t}: {} vs no-reservoir {}",
            res[t],
            plain[t]
        );
    }

    // Reach 1 (the dam): K = T, X = 0 turn the Muskingum coefficients into
    //   c1 = c2 = dt/(2T+dt),  c3 = (2T−dt)/(2T+dt),  c4 = 2·dt/(2T+dt)
    // and the solve row into
    //   q1[t] = c1·q0[t] + c2·q0[t−1] + c3·q1[t−1] + c4·q'1[t−1].
    // The lateral term is `c4·q'` with q' the clamped (never binding here) and
    // un-divided (no subdivision ⇒ `pieces_per_row` is None) row t−1.
    let dt = DT_SECONDS as f64;
    let t_sec = t_days as f64 * SECONDS_PER_DAY;
    let denom = 2.0 * t_sec + dt;
    let (c1, c2, c3, c4) = (dt / denom, dt / denom, (2.0 * t_sec - dt) / denom, 2.0 * dt / denom);
    let q0 = |t: usize| res[t] as f64;
    let q1 = |t: usize| res[cols + t] as f64;
    let mut worst = 0.0_f64;
    for t in 1..cols {
        let lateral = q_prime[(t - 1) * n_reach + 1] as f64;
        let expected = c1 * q0(t) + c2 * q0(t - 1) + c3 * q1(t - 1) + c4 * lateral;
        let rel = (q1(t) - expected).abs() / expected.abs();
        worst = worst.max(rel);
        assert!(
            rel < 1e-5,
            "reach 1 step {t}: routed {} vs linear reservoir {expected} (rel {rel:.3e})",
            q1(t)
        );
    }
    println!("linear-reservoir recurrence: worst rel {worst:.3e}");

    // The override must actually bite: the dam's own Muskingum K is far from
    // 1.5 days on a 5 km reach, so the two runs differ at reach 1.
    assert!(
        (1..cols).any(|t| res[cols + t] != plain[cols + t]),
        "reach 1 is identical with and without the reservoir"
    );
}

// ---------------------------------------------------------------------------
// 2. Disabled ⇒ bit-identical.
// ---------------------------------------------------------------------------

#[test]
fn disabled_is_bitwise_identical() {
    let cfg = Config::default();
    let n_reach = 3;
    let mut q_prime = Vec::new();
    for t in 0..=24 {
        let pulse = 1.0 + (t as f32 / 24.0 * std::f32::consts::PI).sin();
        q_prime.extend_from_slice(&[10.0 * pulse, 2.0, 3.0 * pulse]);
    }
    let params = Params::uniform(n_reach);

    let never = host(route(&cfg, chain3(), &q_prime, &params, None, None, false).q);
    let empty = host(route(&cfg, chain3(), &q_prime, &params, Some((&[], &[])), None, false).q);

    assert_eq!(never.len(), empty.len());
    for (i, (a, b)) in never.iter().zip(&empty).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "idx {i}: {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// 3. Mass balance: a reservoir row relaxes to outflow = inflow.
// ---------------------------------------------------------------------------

#[test]
fn reservoir_row_mass_balance() {
    let cfg = Config::default();
    let n_reach = 3;
    let steps = 2000;
    let lateral = [10.0_f32, 2.0, 3.0];
    let q_prime: Vec<f32> = (0..=steps).flat_map(|_| lateral).collect();
    let params = Params::uniform(n_reach);

    // Start well below the steady state [10, 12, 15] so the reservoir has to
    // fill: 2,000 hours is ~83 residence times at T = 1 day.
    let r = route(
        &cfg,
        chain3(),
        &q_prime,
        &params,
        Some((&[1], &[1.0])),
        Some(&[1.0, 1.0, 1.0]),
        false,
    );
    let q = host(r.q);
    let cols = steps + 1;
    let last = cols - 1;

    let outflow = q[cols + last] as f64;
    let inflow = q[last] as f64 + lateral[1] as f64;
    let rel = (outflow - inflow).abs() / inflow;
    println!("reservoir outflow {outflow} vs steady inflow {inflow} (rel {rel:.3e})");
    assert!(rel < 1e-5, "reservoir outflow {outflow} vs steady inflow {inflow} (rel {rel:.3e})");
}

// ---------------------------------------------------------------------------
// 4 & 5. Gradients on the 5-reach sandbox with reach 3 as a reservoir.
// ---------------------------------------------------------------------------

const DAM: usize = 3;
const UPSTREAM: usize = 1;
const DOWNSTREAM: usize = 4;
const GRAD_STEPS: usize = 24;
const DAM_T_DAYS: f32 = 1.2;

/// Sandbox lateral inflows (`fixtures/sandbox/qprime_topo.csv` row 0) scaled
/// by a smooth one-day pulse, so the window is a transient: at a steady state
/// the routed discharge does not depend on K or X and every gradient is 0.
fn sandbox_q_prime() -> Vec<f32> {
    let base = [22.0_f32, 11.0, 11.0, 11.0, 22.0];
    let mut q = Vec::with_capacity((GRAD_STEPS + 1) * base.len());
    for t in 0..=GRAD_STEPS {
        let s = (t as f32 / GRAD_STEPS as f32 * std::f32::consts::PI).sin();
        let pulse = 1.0 + 1.5 * s * s;
        q.extend(base.iter().map(|b| b * pulse));
    }
    q
}

fn sandbox_params() -> Params {
    Params::uniform(5)
}

#[derive(Copy, Clone, Debug)]
enum Parent {
    N,
    QSpatial,
}

/// `Σ` of routed discharge over the 24 routed steps (column 0, the hot start,
/// does not depend on the parameters and is left out).
fn loss_f64(q: &[f32]) -> f64 {
    let cols = GRAD_STEPS + 1;
    q.chunks(cols).flat_map(|row| &row[1..]).map(|&v| v as f64).sum()
}

fn analytical_grads(cfg: &Config) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let r = route(
        cfg,
        sandbox5(),
        &sandbox_q_prime(),
        &sandbox_params(),
        Some((&[DAM], &[DAM_T_DAYS])),
        None,
        true,
    );
    let loss = r.q.slice([0..5, 1..GRAD_STEPS + 1]).sum();
    let grads = loss.backward();
    (
        host1(r.n.grad(&grads).expect("grad on n")),
        host1(r.q_spatial.grad(&grads).expect("grad on q_spatial")),
        host1(r.p_spatial.grad(&grads).expect("grad on p_spatial")),
    )
}

fn fd_grad(cfg: &Config, parent: Parent, reach: usize) -> f64 {
    let base = sandbox_params();
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
        let r = route(
            cfg,
            sandbox5(),
            &sandbox_q_prime(),
            &p,
            Some((&[DAM], &[DAM_T_DAYS])),
            None,
            false,
        );
        loss_f64(&host(r.q))
    };
    // Divide by the step actually taken in f32, not by `2·eps`.
    let (hi, lo) = (x0 + eps, x0 - eps);
    (eval(hi) - eval(lo)) / (hi as f64 - lo as f64)
}

#[test]
fn reservoir_gradcheck() {
    let cfg = Config::default();
    let (gn, gq, _) = analytical_grads(&cfg);
    let mut failures = Vec::new();
    for (parent, analytical) in [(Parent::N, &gn), (Parent::QSpatial, &gq)] {
        for (label, reach) in [("upstream", UPSTREAM), ("downstream", DOWNSTREAM), ("dam", DAM)] {
            let a = analytical[reach] as f64;
            let f = fd_grad(&cfg, parent, reach);
            let abs = (a - f).abs();
            let rel = abs / a.abs().max(f.abs()).max(1e-12);
            println!(
                "{parent:?} @ reach {reach} ({label}): analytical={a:.6e} fd={f:.6e} abs={abs:.3e} rel={rel:.3e}"
            );
            if !(rel < REL_TOL || abs < ABS_TOL) {
                failures.push(format!("{parent:?} @ reach {reach} ({label}): rel {rel:.3e}"));
            }
        }
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

#[test]
fn dam_row_parameters_have_zero_gradient() {
    // `enforce_positivity` adds the S19' X-cap path `x_eff → cr → K`, the one
    // route into K that does not pass through c1..c4. Both configs must zero
    // the dam row.
    for enforce_positivity in [false, true] {
        let mut cfg = Config::default();
        cfg.params.enforce_positivity = enforce_positivity;
        let (gn, gq, gp) = analytical_grads(&cfg);
        println!("enforce_positivity={enforce_positivity}: n {gn:?}\n  q_spatial {gq:?}\n  p_spatial {gp:?}");
        for (name, g) in [("n", &gn), ("q_spatial", &gq), ("p_spatial", &gp)] {
            assert_eq!(
                g[DAM], 0.0,
                "enforce_positivity={enforce_positivity}: d loss / d {name} at the dam row must be exactly 0"
            );
            // Not vacuous: the reaches around the dam still learn.
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

// ---------------------------------------------------------------------------
// 6. Input validation.
// ---------------------------------------------------------------------------

#[test]
fn set_reservoir_rows_validates() {
    let device = Device::default();
    let n_reach = 3;
    let mut mc = MuskingumCunge::<I>::new(Config::default(), device.clone());
    let q_prime = Tensor::<AB, 1>::from_floats([10.0_f32, 2.0, 3.0, 10.0, 2.0, 3.0], &device)
        .reshape([2, n_reach]);
    let p = Params::uniform(n_reach);
    mc.setup_inputs(
        RoutingInputs::<I> {
            adjacency: chain3(),
            x_storage: Tensor::ones([n_reach], &device) * 0.3,
        },
        q_prime,
        SpatialParameters::<I> {
            n: Tensor::from_floats(p.n.as_slice(), &device),
            q_spatial: Tensor::from_floats(p.q_spatial.as_slice(), &device),
            p_spatial: None,
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        None,
    );

    let cases: [(&str, &[usize], &[f32]); 6] = [
        ("out-of-range row", &[3], &[1.0]),
        ("duplicate row", &[1, 1], &[1.0, 2.0]),
        ("length mismatch", &[1], &[1.0, 2.0]),
        ("T below one hour", &[1], &[0.5 / 24.0]),
        ("NaN T", &[1], &[f32::NAN]),
        ("infinite T", &[1], &[f32::INFINITY]),
    ];
    for (what, rows, t_days) in cases {
        let r = mc.set_reservoir_rows(rows, t_days);
        println!("{what}: {r:?}");
        assert!(r.is_err(), "{what}: expected Err, got Ok");
    }
    // The floor itself is admissible.
    assert!(mc.set_reservoir_rows(&[1], &[1.0 / 24.0]).is_ok());
}

// ---------------------------------------------------------------------------
// Unsupported combination: leakance.
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "leakance")]
fn reservoir_rows_with_leakance_panic() {
    let device = Device::default();
    let n_reach = 3;
    let mut mc = MuskingumCunge::<I>::new(Config::default(), device.clone());
    let q_prime = Tensor::<AB, 1>::from_floats([10.0_f32, 2.0, 3.0, 10.0, 2.0, 3.0], &device)
        .reshape([2, n_reach]);
    let half = || Tensor::<AB, 1>::ones([n_reach], &device) * 0.5;
    mc.setup_inputs(
        RoutingInputs::<I> {
            adjacency: chain3(),
            x_storage: Tensor::ones([n_reach], &device) * 0.3,
        },
        q_prime,
        SpatialParameters::<I> {
            n: half(),
            q_spatial: half(),
            p_spatial: None,
            k_d: Some(half()),
            d_gw: Some(half()),
            leakance_factor: Some(half()),
            impervious_mask: None,
            gamma: None,
        },
        false,
        None,
    );
    mc.set_reservoir_rows(&[1], &[1.0]).unwrap();
    let _ = mc.forward();
}

// ---------------------------------------------------------------------------
// A new `setup_inputs` clears the reservoir rows of the previous network.
// ---------------------------------------------------------------------------

/// `setup_inputs` binds a (possibly different) network, so reservoir rows set
/// for the previous one must not survive it: routing after the second
/// `setup_inputs` must be bitwise the no-reservoir result.
#[test]
fn setup_inputs_clears_reservoir_rows() {
    let cfg = Config::default();
    let device = Device::default();
    let n_reach = 3;
    let mut q_prime = Vec::new();
    for t in 0..=24 {
        q_prime.extend_from_slice(&[if t == 0 { 10.0 } else { 40.0 }, 2.0, 3.0]);
    }
    let params = Params::uniform(n_reach);
    let plain = host(route(&cfg, chain3(), &q_prime, &params, None, None, false).q);

    let setup = |mc: &mut MuskingumCunge<I>| {
        let v = |x: &[f32]| Tensor::<AB, 1>::from_floats(x, &device);
        mc.setup_inputs(
            RoutingInputs::<I> {
                adjacency: chain3(),
                x_storage: Tensor::ones([n_reach], &device) * 0.3,
            },
            v(&q_prime).reshape([q_prime.len() / n_reach, n_reach]),
            SpatialParameters::<I> {
                n: v(&params.n),
                q_spatial: v(&params.q_spatial),
                p_spatial: Some(v(&params.p_spatial)),
                k_d: None,
                d_gw: None,
                leakance_factor: None,
                impervious_mask: None,
                gamma: None,
            },
            false,
            None,
        );
    };
    let mut mc = MuskingumCunge::<I>::new(cfg.clone(), device.clone());
    setup(&mut mc);
    mc.set_reservoir_rows(&[1], &[1.5]).expect("valid reservoir rows");
    setup(&mut mc);
    let rerouted = host(mc.forward());

    assert_eq!(plain.len(), rerouted.len());
    for (i, (a, b)) in plain.iter().zip(&rerouted).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "idx {i}: no-reservoir {a} vs after re-setup {b}");
    }
}
