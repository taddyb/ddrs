//! The harmonic rule curve (`release_head.rule_curve`, `src/routing/release.rs`):
//! `S = T·Q + S0_d(t)`, the flux `r_d = Ibar_d·Σ_k(c_{k,s} sin kω + c_{k,c} cos kω)`
//! taken off the dam row's lateral inflow as `(S0_{t+1} − S0_t)/dt`, with
//! per-dam free coefficients `c = rule_curve_max·tanh(θ)` and a per-dam `T0`
//! multiplier `exp(δ)` (`src/nn/dam_params.rs`).
//!
//! 1. `θ = 0` (zero coefficients) routes bitwise like no rule curve, on both
//!    dam rows, at the engine and through `apply_reservoir_rows_with`.
//! 2. Over one calendar year of constant inflow the flux moves no volume: the
//!    dam's outflow volume equals its inflow volume less the bucket's storage
//!    change, to f32 tolerance, while the flux visibly reshapes the outflow.
//! 3. Finite-difference gradcheck of `θ` (all four coefficients) and `δ` on
//!    the sandbox network, replace and additive rows: the gradient reaches
//!    them through the timestep op's `q'` parent and ordinary autodiff.
//! 4. The training path (per-dam parameters on the batch's rows) and the
//!    resolved test-phase table (coefficients, effective `T0`, `Ibar` per dam)
//!    route bitwise identically, and `release_params.csv` carries them.
//! 5. `rule_curve: true` on a table without `inflow_mean_m3s` fails at
//!    dataset open.
//! 6. The engine's per-dam clamp account (`MuskingumCunge::dam_account`):
//!    a flux that stores more than the dam's inflow forces the S28 clamp, and
//!    the created volume `Σ max(lb − x, 0)·dt` matches a hand-rolled f64
//!    recurrence of the dam row; with no rule curve it is exactly zero; the
//!    inflow is the routed upstream inflow plus the reach's own `q'`.
//! 7. `params.reservoir_dam_row: additive` puts a `fixed` table on the
//!    additive row: a fixed CSV carrying the learned path's resolved table
//!    (`T0`, `c = 0`, `Ibar`) routes bitwise like that resolved table under
//!    the learned config, and a plain fixed table routes like
//!    `set_reservoir_rows_as(.., Additive)`.

use burn::backend::{Autodiff, NdArray};
use burn::module::Param;
use burn::tensor::Tensor;
use chrono::NaiveDate;

use ddrs::config::{Config, DamRow, ReleaseHeadSection, ReservoirRelease};
use ddrs::data::ids::Comid;
use ddrs::data::store::{map_reservoir_rows, read_fixed_release_table, DamFeatures, ReservoirTable};
use ddrs::nn::dam_params::DamParams;
use ddrs::nn::release_head::init_release_head;
use ddrs::routing::mmc::{DamRelease, RuleCurve, DT_SECONDS};
use ddrs::routing::release::{
    rule_curve_h, rule_curve_increments, rule_curve_phase_start, seasonal_phase, OMEGA_RAD_PER_S,
};
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::forward::apply_reservoir_rows_with;
use ddrs::training::release_eval::{resolve_release_table_with, write_release_params_csv};

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const REL_TOL: f64 = 5e-3;
const ABS_TOL: f64 = 1e-4;

fn network(n: usize, edges: &[(usize, usize)]) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n * n];
    for &(up, down) in edges {
        dense[down * n + up] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, vec![5000.0; n], vec![0.001; n])
}

/// `1 → 3`, `2 → 3`, `0 → 4`, `3 → 4`.
fn sandbox5() -> SparseAdjacency {
    network(5, &[(1, 3), (2, 3), (0, 4), (3, 4)])
}

/// `0 → 1 → 2`.
fn chain3() -> SparseAdjacency {
    network(3, &[(0, 1), (1, 2)])
}

fn engine(cfg: &Config, adjacency: SparseAdjacency, q_prime: &[f32]) -> MuskingumCunge<I> {
    let device = Device::default();
    let n = adjacency.n;
    let rows = q_prime.len() / n;
    let leaf = |v: f32| Tensor::<AB, 1>::from_floats(vec![v; n].as_slice(), &device);
    let q = Tensor::<AB, 1>::from_floats(q_prime, &device).reshape([rows, n]);
    let mut mc = MuskingumCunge::<I>::new(cfg.clone(), device);
    mc.setup_inputs(
        RoutingInputs { adjacency, x_storage: Tensor::ones([n], &device) * 0.3 },
        q,
        SpatialParameters {
            n: leaf(0.5),
            q_spatial: leaf(0.5),
            p_spatial: Some(leaf(0.575)),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        None,
    );
    mc
}

fn sandbox_q_prime(steps: usize) -> Vec<f32> {
    let base = [22.0_f32, 11.0, 11.0, 11.0, 22.0];
    (0..=steps)
        .flat_map(|t| {
            let s = (t as f32 / 24.0 * std::f32::consts::PI).sin();
            base.map(|b| b * (1.0 + 1.5 * s * s))
        })
        .collect()
}

fn assert_bitwise(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: idx {i}: {x} vs {y}");
    }
}

/// Route the sandbox with one dam on reach 3: `T0` days (seasonal `a`, `b`
/// when `Some`), and a rule curve `coeffs` with `Ibar` when `Some`.
fn route_sandbox(
    steps: usize,
    start: NaiveDate,
    row: DamRow,
    t0: Tensor<AB, 1>,
    ab: Option<(f32, f32)>,
    rule: Option<(Tensor<AB, 2>, f32)>,
) -> Tensor<AB, 2> {
    let device = Device::default();
    let mut mc = engine(&Config::default(), sandbox5(), &sandbox_q_prime(steps));
    let t = |v: f32| Tensor::<AB, 1>::from_floats([v], &device);
    mc.set_dam_release(DamRelease {
        rows: vec![3],
        t0_days: t0,
        seasonal: ab.map(|(a, b)| (t(a), t(b))),
        phase: seasonal_phase(start, steps + 1),
        dam_row: row,
        rule_curve: rule.map(|(coeffs, ibar)| RuleCurve {
            coeffs,
            inflow_mean: t(ibar),
            phase0: rule_curve_phase_start(start),
        }),
    })
    .expect("release");
    mc.forward()
}

fn host2(q: Tensor<AB, 2>) -> Vec<f32> {
    q.into_data().to_vec().unwrap()
}

// ---------------------------------------------------------------------------
// 1. θ = 0 is no rule curve, bit for bit.
// ---------------------------------------------------------------------------

#[test]
fn zero_coefficients_route_bitwise_like_no_rule_curve() {
    let device = Device::default();
    let start = NaiveDate::from_ymd_opt(2001, 5, 10).unwrap();
    for row in [DamRow::Replace, DamRow::Additive] {
        for ab in [None, Some((0.6_f32, -0.3_f32))] {
            let t0 = || Tensor::<AB, 1>::from_floats([0.7_f32], &device);
            let none = host2(route_sandbox(72, start, row, t0(), ab, None));
            let zero = host2(route_sandbox(
                72,
                start,
                row,
                t0(),
                ab,
                Some((Tensor::zeros([1, 4], &device), 25.0)),
            ));
            assert_bitwise(&zero, &none, &format!("{row:?} {ab:?}: c = 0 vs no rule curve"));
            let some = host2(route_sandbox(
                72,
                start,
                row,
                t0(),
                ab,
                Some((Tensor::from_floats([[0.3_f32, -0.2, 0.1, 0.15]], &device), 25.0)),
            ));
            assert!(some != none, "{row:?}: a nonzero rule curve must act");
        }
    }
}

// ---------------------------------------------------------------------------
// 2. One year: the rule curve moves no volume.
// ---------------------------------------------------------------------------

#[test]
fn rule_curve_moves_no_volume_over_a_year_of_constant_inflow() {
    let device = Device::default();
    // One period of the continuous phase: 365.25 d = 8,766 hours, so the last
    // phase row is the first one again (mod 2π) and S0 returns to its start.
    // The window crosses the 2001 -> 2002 year boundary.
    let steps = 8766;
    let start = NaiveDate::from_ymd_opt(2001, 1, 1).unwrap();
    let q_in = 50.0_f32;
    let q: Vec<f32> = (0..=steps).flat_map(|_| [q_in, 0.0, 0.0]).collect();
    let t0_days = 1.0_f32;
    let c = [0.3_f32, -0.2, 0.1, 0.15];
    let route = |rule: bool| -> Vec<f32> {
        let mut mc = engine(&Config::default(), chain3(), &q);
        let t = |v: &[f32]| Tensor::<AB, 1>::from_floats(v, &device);
        mc.set_dam_release(DamRelease {
            rows: vec![0],
            t0_days: t(&[t0_days]),
            seasonal: None,
            phase: seasonal_phase(start, steps + 1),
            dam_row: DamRow::Replace,
            rule_curve: rule.then(|| RuleCurve {
                coeffs: Tensor::<AB, 1>::from_floats(c, &device).reshape([1, 4]),
                inflow_mean: t(&[q_in]),
                phase0: rule_curve_phase_start(start),
            }),
        })
        .unwrap();
        // Reach 0 (the dam): row 0 of [n, steps + 1].
        host2(mc.forward())[..=steps].to_vec()
    };
    let with = route(true);
    let without = route(false);
    let dt = DT_SECONDS as f64;
    let t_sec = t0_days as f64 * 86_400.0;
    let balance = |out: &[f32]| {
        let inflow = steps as f64 * dt * q_in as f64;
        let outflow: f64 = (1..=steps).map(|s| dt * 0.5 * (out[s - 1] as f64 + out[s] as f64)).sum();
        let ds = t_sec * (out[steps] as f64 - out[0] as f64);
        (inflow, outflow, (inflow - outflow - ds) / inflow)
    };
    let (inflow, out_rc, imb_rc) = balance(&with);
    let (_, out_plain, imb_plain) = balance(&without);
    let min_q = with.iter().copied().fold(f32::INFINITY, f32::min);
    let max_dev = with.iter().zip(&without).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    println!(
        "one year: inflow {inflow:.6e} m3, outflow with rule curve {out_rc:.6e} (imbalance {imb_rc:.2e}), \
         without {out_plain:.6e} (imbalance {imb_plain:.2e}); min Q {min_q:.2} m3/s, max |dQ| {max_dev:.2} m3/s"
    );
    assert!(min_q > 1.0, "the dam never reaches the discharge floor in this test");
    assert!(max_dev > 5.0, "the rule curve must visibly reshape the outflow (max |dQ| {max_dev})");
    assert!(imb_rc.abs() < 1e-5, "the rule curve moved volume over a year: {imb_rc:.3e}");
}

// ---------------------------------------------------------------------------
// 2b. The phase is continuous across year boundaries (leap and non-leap).
// ---------------------------------------------------------------------------

/// The flux of every step across Dec 31 → Jan 1 is one hour of phase: the
/// increments `ΔH_s` (the flux is `Ibar/dt · c·ΔH_s`) match the exact
/// one-hour difference of `H` at the continuous phase, change smoothly from
/// step to step (no +7 h / −17 h jump), and telescope to `H(end) − H(start)`.
/// Windows starting on different dates line up with one phase.
#[test]
fn rule_curve_flux_is_continuous_across_year_boundaries() {
    let dw = OMEGA_RAD_PER_S * DT_SECONDS as f64;
    // 2003 -> 2004 enters a leap year (the old day-of-year phase jumped +7 h
    // there); 2004 -> 2005 leaves one (−17 h).
    for (start, label) in [
        (NaiveDate::from_ymd_opt(2003, 12, 30).unwrap(), "2003->2004 (non-leap Dec 31)"),
        (NaiveDate::from_ymd_opt(2004, 12, 30).unwrap(), "2004->2005 (leap Dec 31)"),
    ] {
        let n_rows = 24 * 4 + 1; // Dec 30 00:00 .. Jan 3 00:00
        let w0 = rule_curve_phase_start(start);
        let dh = rule_curve_increments(w0, n_rows);
        let rows: Vec<[f32; 4]> = dh.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        // The boundary step: Dec 31 23:00 -> Jan 1 00:00 is step 47.
        for (s, r) in rows.iter().enumerate() {
            let (a, b) = (rule_curve_h(w0 + dw * s as f64), rule_curve_h(w0 + dw * (s + 1) as f64));
            for j in 0..4 {
                let exact = b[j] - a[j];
                // One hour of phase: |ΔH_j| <= dt (H' has unit amplitude in seconds).
                assert!(
                    (r[j] as f64 - exact).abs() <= 1e-6 * (DT_SECONDS as f64) + 1e-7 * exact.abs(),
                    "{label}: step {s} coeff {j}: {} vs exact {exact}",
                    r[j]
                );
                assert!(exact.abs() <= DT_SECONDS as f64 * 1.0001, "{label}: step {s} spans more than an hour");
            }
            if s > 0 {
                // Smooth: consecutive steps differ by O(Ω·dt) of their size.
                let prev = rows[s - 1];
                for j in 0..4 {
                    let jump = (r[j] - prev[j]).abs() as f64;
                    assert!(
                        jump <= 4.0 * dw * DT_SECONDS as f64,
                        "{label}: step {s} coeff {j} jumps by {jump:.3e} s (old day-of-year phase: \
                         a 7 h or 17 h step)"
                    );
                }
            }
        }
        // Telescoping over the window.
        let (h0, h1) = (rule_curve_h(w0), rule_curve_h(w0 + dw * (n_rows - 1) as f64));
        for j in 0..4 {
            let sum: f64 = rows.iter().map(|r| r[j] as f64).sum();
            let rel = (sum - (h1[j] - h0[j])).abs() / (h1[j] - h0[j]).abs().max(DT_SECONDS as f64);
            assert!(rel < 1e-4, "{label}: coeff {j}: increments sum {sum} vs H(end) - H(start) {}", h1[j] - h0[j]);
        }
        // A window starting a day later lines up with this one's row 24.
        let w1 = rule_curve_phase_start(start + chrono::Duration::days(1));
        let d = (w1 - (w0 + 24.0 * dw)).rem_euclid(2.0 * std::f64::consts::PI);
        assert!(d.min(2.0 * std::f64::consts::PI - d) < 1e-9, "{label}: day-to-day phase offset {d:.3e}");
    }
    // Over one whole period (8,766 h) the increments telescope to zero.
    let w0 = rule_curve_phase_start(NaiveDate::from_ymd_opt(2004, 3, 1).unwrap());
    let dh = rule_curve_increments(w0, 8767);
    for j in 0..4 {
        let sum: f64 = dh.iter().skip(j).step_by(4).map(|&v| v as f64).sum();
        let scale = 1.0 / OMEGA_RAD_PER_S;
        assert!(sum.abs() / scale < 1e-6, "coeff {j}: one period sums to {sum:.3e} s (scale {scale:.3e})");
    }
}

// ---------------------------------------------------------------------------
// 3. Gradcheck of θ and δ.
// ---------------------------------------------------------------------------

const MAX_C: f32 = 0.8;
const T0_HEAD: f32 = 0.6;
const IBAR: f32 = 10.0;

fn weights(steps: usize) -> Vec<f32> {
    (0..5 * (steps + 1))
        .map(|i| {
            let (r, t) = (i / (steps + 1), i % (steps + 1));
            if t == 0 { 0.0 } else { 1.0 + 0.1 * r as f32 + 0.01 * (t % 7) as f32 }
        })
        .collect()
}

/// The sandbox routed with the dam's `T0 = T0_HEAD·exp(δ)` and coefficients
/// `MAX_C·tanh(θ)`, from leaves `theta` `[1, 4]` and `delta` `[1]`.
fn route_leaves(row: DamRow, theta: Tensor<AB, 2>, delta: Tensor<AB, 1>) -> Tensor<AB, 2> {
    let device = Device::default();
    let t0 = Tensor::<AB, 1>::from_floats([T0_HEAD], &device) * delta.exp();
    let c = theta.tanh() * MAX_C;
    let start = NaiveDate::from_ymd_opt(2001, 5, 10).unwrap();
    route_sandbox(72, start, row, t0, Some((0.5, -0.4)), Some((c, IBAR)))
}

fn loss_at(row: DamRow, theta: [f32; 4], delta: f32) -> f64 {
    let device = Device::default();
    let q = host2(route_leaves(
        row,
        Tensor::<AB, 1>::from_floats(theta, &device).reshape([1, 4]),
        Tensor::from_floats([delta], &device),
    ));
    q.iter().zip(weights(72)).map(|(&v, w)| v as f64 * w as f64).sum()
}

#[test]
fn theta_and_delta_gradcheck() {
    let device = Device::default();
    let theta0 = [0.4_f32, -0.3, 0.2, 0.5];
    let delta0 = 0.3_f32;
    let mut failures = Vec::new();
    for row in [DamRow::Replace, DamRow::Additive] {
        let theta = Tensor::<AB, 1>::from_floats(theta0, &device).reshape([1, 4]).require_grad();
        let delta = Tensor::<AB, 1>::from_floats([delta0], &device).require_grad();
        let w = Tensor::<AB, 1>::from_floats(weights(72).as_slice(), &device).reshape([5, 73]);
        let grads = (route_leaves(row, theta.clone(), delta.clone()) * w).sum().backward();
        let gt: Vec<f32> = theta.grad(&grads).expect("theta grad").into_data().to_vec().unwrap();
        let gd: f32 = delta.grad(&grads).expect("delta grad").into_scalar();
        let eps = 1e-2_f32;
        let mut cases: Vec<(String, f64, f64)> = (0..4)
            .map(|k| {
                let bump = |d: f32| {
                    let mut th = theta0;
                    th[k] += d;
                    loss_at(row, th, delta0)
                };
                let fd = (bump(eps) - bump(-eps)) / (2.0 * eps as f64);
                (format!("{row:?} theta[{k}]"), gt[k] as f64, fd)
            })
            .collect();
        let fd_delta = (loss_at(row, theta0, delta0 + eps) - loss_at(row, theta0, delta0 - eps)) / (2.0 * eps as f64);
        cases.push((format!("{row:?} delta"), gd as f64, fd_delta));
        for (label, a, f) in cases {
            let abs = (a - f).abs();
            let rel = abs / a.abs().max(f.abs()).max(1e-12);
            println!("{label}: analytical={a:.6e} fd={f:.6e} rel={rel:.3e}");
            assert!(a != 0.0 && a.is_finite(), "{label}: vacuous gradient");
            if !(rel < REL_TOL || abs < ABS_TOL) {
                failures.push(format!("{label}: rel {rel:.3e}"));
            }
        }
    }
    assert!(failures.is_empty(), "gradcheck failed: {failures:?}");
}

// ---------------------------------------------------------------------------
// 4. Training path = resolved test-phase table.
// ---------------------------------------------------------------------------

fn section(dam_row: DamRow) -> ReleaseHeadSection {
    ReleaseHeadSection {
        hidden_size: 6,
        num_hidden_layers: 1,
        grid: 5,
        k: 3,
        input_var_names: vec!["f1".into(), "f2".into()],
        seasonal: true,
        dam_row,
        routing_checkpoint: None,
        freeze_routing: false,
        rule_curve: true,
        rule_curve_max: MAX_C,
        per_dam_t0: true,
        per_dam_lr: 0.05,
        per_dam_l2: 0.0,
        rule_curve_penalty: 0.0,
        rule_curve_alpha: 0.9,
    }
}

fn learned_cfg(dam_row: DamRow) -> Config {
    let mut cfg = Config::default();
    cfg.params.use_reservoirs = true;
    cfg.params.reservoir_release = ReservoirRelease::Learned;
    cfg.release_head = Some(section(dam_row));
    cfg
}

/// Dams on reaches 3 and 4 (COMIDs 103, 104) plus one outside the network.
fn features() -> DamFeatures {
    DamFeatures {
        comids: vec![Comid(104), Comid(999), Comid(103)],
        names: vec!["f1".into(), "f2".into()],
        values: ndarray::array![[0.8_f32, -1.2], [0.0, 0.0], [-0.5, 1.7]],
        years: vec![None, None, None],
        inflow_mean: Some(vec![40.0, 5.0, 12.0]),
    }
}

/// Per-dam parameters with distinct rows (table order: 104, 999, 103).
fn dam_params() -> DamParams<AB> {
    let device = Device::default();
    let mut d = DamParams::<AB>::zeros(3, true, true, &device);
    d.theta = Some(Param::from_tensor(Tensor::from_floats(
        [[0.4, -0.3, 0.2, 0.5], [0.0; 4], [-0.6, 0.1, 0.3, -0.2]],
        &device,
    )));
    d.delta = Some(Param::from_tensor(Tensor::from_floats([0.5, 0.0, -0.4], &device)));
    d
}

fn sandbox_engine(cfg: &Config, steps: usize) -> MuskingumCunge<I> {
    engine(cfg, sandbox5(), &sandbox_q_prime(steps))
}

#[test]
fn training_and_resolved_rule_curve_route_identically() {
    use burn::module::AutodiffModule;
    let steps = 96;
    let start = NaiveDate::from_ymd_opt(1990, 4, 20).unwrap();
    let network: Vec<Comid> = (0..5).map(|i| Comid(100 + i)).collect();
    for row in [DamRow::Replace, DamRow::Additive] {
        let cfg = learned_cfg(row);
        let device = Device::default();
        let head = init_release_head::<AB>(&section(row), &cfg.params.parameter_ranges, 42, &device);
        let dams = dam_params();

        let rows = map_reservoir_rows(&ReservoirTable::Learned(features()), &network);
        assert_eq!(rows.table_index, vec![2, 0], "reach 3 is table row 2, reach 4 row 0");
        let mut a = sandbox_engine(&cfg, steps);
        apply_reservoir_rows_with(&cfg, &mut a, Some(&rows), start, steps + 1, Some(&head), Some(&dams));
        let train = host2(a.forward());

        let table = resolve_release_table_with::<I>(&head.valid(), Some(&dams.valid()), &features(), &cfg);
        let rc = table.rule_curve.as_ref().expect("the resolved table carries the rule curve");
        assert_eq!(table.inflow_mean, Some(vec![40.0, 5.0, 12.0]));
        assert!((rc[0][0] - MAX_C * 0.4_f32.tanh()).abs() < 1e-6, "c of table row 0");
        assert_eq!(rc[1], [0.0; 4]);
        // Effective T0 = head T0 · exp(δ): the head's T0 is the same for every
        // dam at init, so the ratio of rows 0 and 1 is exp(0.5).
        let ratio = table.dams[0].t_days / table.dams[1].t_days;
        assert!((ratio - 0.5_f32.exp()).abs() < 1e-5, "effective T0 ratio {ratio}");

        let rows_b = map_reservoir_rows(&ReservoirTable::Fixed(table.clone()), &network);
        let mut b = sandbox_engine(&cfg, steps);
        apply_reservoir_rows_with(&cfg, &mut b, Some(&rows_b), start, steps + 1, None, None);
        let resolved = host2(b.forward());
        assert_bitwise(&train, &resolved, &format!("{row:?}: training vs resolved rule curve"));

        let dir = tempfile::tempdir().unwrap();
        let csv = dir.path().join("release_params.csv");
        write_release_params_csv(&csv, &table, row).unwrap();
        let text = std::fs::read_to_string(&csv).unwrap();
        assert!(
            text.starts_with("COMID,T0_days,a,b,T_min_days,T_max_days,c1s,c1c,c2s,c2c,inflow_mean_m3s\n"),
            "{text}"
        );
    }
}

#[test]
#[should_panic(expected = "per-dam parameters")]
fn learned_rule_curve_without_dam_params_is_refused() {
    let cfg = learned_cfg(DamRow::Additive);
    let device = Device::default();
    let head = init_release_head::<AB>(&section(DamRow::Additive), &cfg.params.parameter_ranges, 42, &device);
    let network: Vec<Comid> = (0..5).map(|i| Comid(100 + i)).collect();
    let rows = map_reservoir_rows(&ReservoirTable::Learned(features()), &network);
    let mut a = sandbox_engine(&cfg, 24);
    let start = NaiveDate::from_ymd_opt(1990, 4, 20).unwrap();
    apply_reservoir_rows_with(&cfg, &mut a, Some(&rows), start, 25, Some(&head), None);
}

// ---------------------------------------------------------------------------
// 5. The inflow column is required.
// ---------------------------------------------------------------------------

#[test]
fn rule_curve_without_inflow_column_fails_at_dataset_open() {
    let mut cfg: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string("examples/juniata/ddrs.yaml").unwrap()).unwrap();
    cfg["params"]["use_reservoirs"] = true.into();
    cfg["params"]["reservoir_release"] = "learned".into();
    cfg["data_sources"]["reservoirs"] = "examples/juniata/data/juniata_dam_features.csv".into();
    let mut rh = serde_yaml::Mapping::new();
    rh.insert("input_var_names".into(), serde_yaml::Value::Sequence(vec!["log10_storage".into()]));
    rh.insert("rule_curve".into(), true.into());
    cfg["release_head"] = serde_yaml::Value::Mapping(rh);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ddrs.yaml");
    std::fs::write(&path, serde_yaml::to_string(&cfg).unwrap()).unwrap();
    let cfg = Config::from_yaml_file(&path).expect("the config itself is valid");
    let err = match ddrs::data::MeritGagesDataset::open(&cfg) {
        Ok(_) => panic!("a rule curve without inflow_mean_m3s must not open"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("inflow_mean_m3s") && err.contains("rule_curve"), "{err}");
}

// ---------------------------------------------------------------------------
// 6. Clamp accounting.
// ---------------------------------------------------------------------------

/// A headwater dam (row 0 of `0 → 1 → 2`) on the replace row at constant `T`,
/// constant `q'`, and a rule curve `c = (1, 0, 0, 0)` with `Ibar = 3·q'` in
/// spring (`sin ω ≈ 1`): the flux stores ~3× the inflow, `q'_eff < 0`, and
/// the dam row's solve goes below the floor. Returns the engine (after
/// `forward`) and the routed output `[3, steps + 1]`.
fn headwater_clamp_case(rule: bool, steps: usize) -> (MuskingumCunge<I>, Vec<f32>) {
    let device = Device::default();
    let q_dam = 5.0_f32;
    let q: Vec<f32> = (0..=steps).flat_map(|_| [q_dam, 0.0, 0.0]).collect();
    let mut mc = engine(&Config::default(), chain3(), &q);
    let t = |v: &[f32]| Tensor::<AB, 1>::from_floats(v, &device);
    let start = NaiveDate::from_ymd_opt(2001, 3, 20).unwrap();
    mc.set_dam_release(DamRelease {
        rows: vec![0],
        t0_days: t(&[0.1]),
        seasonal: None,
        phase: seasonal_phase(start, steps + 1),
        dam_row: DamRow::Replace,
        rule_curve: rule.then(|| RuleCurve {
            coeffs: Tensor::<AB, 1>::from_floats([1.0_f32, 0.0, 0.0, 0.0], &device).reshape([1, 4]),
            inflow_mean: t(&[3.0 * q_dam]),
            phase0: rule_curve_phase_start(start),
        }),
    })
    .unwrap();
    let out = host2(mc.forward());
    (mc, out)
}

#[test]
fn clamp_account_matches_a_hand_computation_when_the_flux_forces_the_clamp() {
    let steps = 72;
    let (mc, out) = headwater_clamp_case(true, steps);
    let acc = mc.dam_account().expect("dam rows are armed");
    assert_eq!(acc.rows, vec![0]);
    assert_eq!(acc.steps, steps as u64);

    // Hand computation of the dam row (headwater: I = 0), replace row at
    // constant T = 0.1 d: x = c3·Q_t + c4·(q' − r_s), Q_{t+1} = max(x, lb),
    // c3 = (2T − dt)/(2T + dt), c4 = 2dt/(2T + dt); r_s = Ibar/dt·(c·ΔH_s)
    // with ΔH_s the exact one-hour difference of H at the continuous phase.
    let (dt, lb) = (DT_SECONDS as f64, 1e-4_f64);
    let t_sec = 0.1 * 86_400.0;
    let (c3, c4) = ((2.0 * t_sec - dt) / (2.0 * t_sec + dt), 2.0 * dt / (2.0 * t_sec + dt));
    let (q_dam, ibar) = (5.0_f64, 15.0_f64);
    let w0 = rule_curve_phase_start(NaiveDate::from_ymd_opt(2001, 3, 20).unwrap());
    let dw = OMEGA_RAD_PER_S * dt;
    let mut q_t = out[0] as f64; // the engine's cold start, Q_0 = q'_0
    let (mut created, mut clamp_steps) = (0.0_f64, 0u64);
    for s in 0..steps {
        let dh = rule_curve_h(w0 + dw * (s + 1) as f64)[0] - rule_curve_h(w0 + dw * s as f64)[0];
        let r = ibar / dt * dh;
        let x = c3 * q_t + c4 * (q_dam - r);
        if x < lb {
            clamp_steps += 1;
            created += (lb - x) * dt;
        }
        q_t = x.max(lb);
        // The engine's routed dam outflow follows the same recurrence.
        let engine_q = out[s + 1] as f64;
        assert!((engine_q - q_t).abs() <= 1e-3 * q_t.abs().max(1.0), "step {s}: engine {engine_q} vs hand {q_t}");
    }
    let inflow = steps as f64 * q_dam * dt;
    println!(
        "clamp account: created {:.6e} m3 (hand {created:.6e}), inflow {:.6e} m3 (hand {inflow:.6e}), \
         clamp steps {} (hand {clamp_steps}) of {steps}",
        acc.created_m3[0], acc.inflow_m3[0], acc.clamp_steps[0]
    );
    assert!(clamp_steps > steps as u64 / 2, "the case must actually force the clamp ({clamp_steps} steps)");
    assert_eq!(acc.clamp_steps[0], clamp_steps, "clamp steps");
    assert!((acc.created_m3[0] - created).abs() <= 1e-4 * created, "created {} vs hand {created}", acc.created_m3[0]);
    assert!((acc.inflow_m3[0] - inflow).abs() <= 1e-5 * inflow, "inflow {} vs hand {inflow}", acc.inflow_m3[0]);
    assert_eq!(acc.pooled(), (acc.created_m3[0], acc.inflow_m3[0]));
    assert_eq!(acc.clamp_share(), (clamp_steps, steps as u64));
    // The penalty's inflow record: Qin = I_t + q' = q' on a headwater dam.
    let qin: Vec<f32> = mc.dam_inflow_record().expect("rule curve armed").into_data().to_vec().unwrap();
    assert_eq!(qin.len(), steps);
    assert!(qin.iter().all(|&v| v == q_dam as f32), "Qin on a headwater dam is its own q'");
}

#[test]
fn clamp_account_is_zero_without_a_rule_curve() {
    let steps = 72;
    let (mc, _) = headwater_clamp_case(false, steps);
    let acc = mc.dam_account().expect("dam rows are armed");
    assert_eq!(acc.created_m3, vec![0.0], "no rule curve: the clamp creates nothing");
    assert_eq!(acc.clamp_steps, vec![0]);
    let inflow = steps as f64 * 5.0 * DT_SECONDS as f64;
    assert!((acc.inflow_m3[0] - inflow).abs() <= 1e-5 * inflow);
    assert!(mc.dam_inflow_record().is_none(), "no rule curve: no penalty record");
}

/// An interior dam (row 1 of `0 → 1 → 2`): the account's inflow is the routed
/// upstream discharge at the step start plus the dam reach's own `q'`, and
/// option C rows (`set_reservoir_rows_as`) are accounted too.
#[test]
fn clamp_account_inflow_is_routed_upstream_plus_own_lateral() {
    let device = Device::default();
    let steps = 48;
    let q: Vec<f32> = (0..=steps)
        .flat_map(|t| [6.0 + 2.0 * (t as f32 / 7.0).sin(), 2.0, 1.0])
        .collect();
    for option_c in [false, true] {
        let mut mc = engine(&Config::default(), chain3(), &q);
        if option_c {
            mc.set_reservoir_rows_as(&[1], &[0.5], DamRow::Additive).unwrap();
        } else {
            let t = |v: &[f32]| Tensor::<AB, 1>::from_floats(v, &device);
            mc.set_dam_release(DamRelease {
                rows: vec![1],
                t0_days: t(&[0.5]),
                seasonal: None,
                phase: seasonal_phase(NaiveDate::from_ymd_opt(2001, 7, 1).unwrap(), steps + 1),
                dam_row: DamRow::Additive,
                rule_curve: None,
            })
            .unwrap();
        }
        let out = host2(mc.forward());
        let acc = mc.dam_account().expect("dam rows are armed");
        // Row 0 of [3, steps + 1] is the upstream reach's routed outflow.
        let expected: f64 = (0..steps)
            .map(|s| (out[s] as f64 + 2.0) * DT_SECONDS as f64)
            .sum();
        let rel = (acc.inflow_m3[0] - expected).abs() / expected;
        assert!(rel < 1e-5, "option C {option_c}: inflow {} vs {expected} (rel {rel:.2e})", acc.inflow_m3[0]);
        assert_eq!(acc.created_m3, vec![0.0]);
    }
}

// ---------------------------------------------------------------------------
// 7. Fixed tables on the additive row (`params.reservoir_dam_row`).
// ---------------------------------------------------------------------------

/// A `reservoir_release: fixed` config with `params.reservoir_dam_row`.
fn fixed_cfg(row: Option<DamRow>) -> Config {
    let mut cfg = Config::default();
    cfg.params.use_reservoirs = true;
    cfg.params.reservoir_release = ReservoirRelease::Fixed;
    cfg.params.reservoir_dam_row = row;
    cfg
}

#[test]
fn fixed_additive_table_with_zero_rule_curve_routes_like_the_resolved_learned_table() {
    use burn::module::AutodiffModule;
    let steps = 96;
    let start = NaiveDate::from_ymd_opt(1990, 4, 20).unwrap();
    let network: Vec<Comid> = (0..5).map(|i| Comid(100 + i)).collect();
    let device = Device::default();

    // The learned path's resolved test-phase table at theta = 0 (c = 0) and
    // delta = 0: the head's init T0 (4.5 h), a = b = 0, Ibar per dam.
    let learned = learned_cfg(DamRow::Additive);
    let head = init_release_head::<AB>(&section(DamRow::Additive), &learned.params.parameter_ranges, 42, &device);
    let zeros = DamParams::<AB>::zeros(3, true, true, &device);
    let table = resolve_release_table_with::<I>(&head.valid(), Some(&zeros.valid()), &features(), &learned);
    assert!(table.rule_curve.as_ref().unwrap().iter().all(|c| *c == [0.0; 4]), "theta = 0 is c = 0");
    let rows_a = map_reservoir_rows(&ReservoirTable::Fixed(table.clone()), &network);
    let mut a = sandbox_engine(&learned, steps);
    apply_reservoir_rows_with(&learned, &mut a, Some(&rows_a), start, steps + 1, None, None);
    let resolved = host2(a.forward());

    // The same table as a fixed CSV (the reader's own columns), routed under a
    // fixed config with params.reservoir_dam_row: additive.
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("fixed.csv");
    let (rc, ibar) = (table.rule_curve.as_ref().unwrap(), table.inflow_mean.as_ref().unwrap());
    let mut text = String::from("COMID,T_days,a,b,c1s,c1c,c2s,c2c,inflow_mean_m3s\n");
    for (i, d) in table.dams.iter().enumerate() {
        let [c1s, c1c, c2s, c2c] = rc[i];
        text.push_str(&format!("{},{},{},{},{c1s},{c1c},{c2s},{c2c},{}\n", d.comid.0, d.t_days, d.a, d.b, ibar[i]));
    }
    std::fs::write(&csv, text).unwrap();
    let fixed = read_fixed_release_table(&csv).unwrap();
    assert_eq!(fixed, table, "the CSV round-trips the resolved table exactly");
    let rows_b = map_reservoir_rows(&ReservoirTable::Fixed(fixed.clone()), &network);
    let cfg_b = fixed_cfg(Some(DamRow::Additive));
    assert_eq!(cfg_b.dam_row(), DamRow::Additive);
    let mut b = sandbox_engine(&cfg_b, steps);
    apply_reservoir_rows_with(&cfg_b, &mut b, Some(&rows_b), start, steps + 1, None, None);
    assert_bitwise(&host2(b.forward()), &resolved, "fixed additive vs resolved learned (c = 0)");

    // The key is live: the default (replace) routes differently.
    let cfg_c = fixed_cfg(None);
    let mut c = sandbox_engine(&cfg_c, steps);
    apply_reservoir_rows_with(&cfg_c, &mut c, Some(&rows_b), start, steps + 1, None, None);
    assert!(host2(c.forward()) != resolved, "replace must differ from additive");
}

#[test]
fn plain_fixed_table_on_the_additive_row_is_the_engines_additive_option_c() {
    let steps = 48;
    let start = NaiveDate::from_ymd_opt(1990, 7, 1).unwrap();
    let network: Vec<Comid> = (0..5).map(|i| Comid(100 + i)).collect();
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("plain.csv");
    std::fs::write(&csv, "COMID,T_days,year_completed\n103,0.8,1970\n104,2.5,\n").unwrap();
    let table = read_fixed_release_table(&csv).unwrap();
    let rows = map_reservoir_rows(&ReservoirTable::Fixed(table), &network);
    let cfg = fixed_cfg(Some(DamRow::Additive));
    let mut a = sandbox_engine(&cfg, steps);
    apply_reservoir_rows_with(&cfg, &mut a, Some(&rows), start, steps + 1, None, None);
    let via_config = host2(a.forward());
    let mut b = sandbox_engine(&cfg, steps);
    b.set_reservoir_rows_as(&[3, 4], &[0.8, 2.5], DamRow::Additive).unwrap();
    assert_bitwise(&via_config, &host2(b.forward()), "fixed additive table vs engine option C additive");
}
