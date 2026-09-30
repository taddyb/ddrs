//! The carried dam floor (`release_head.dam_floor` / `params.reservoir_dam_floor:
//! carry`, `ddrs::config::DamFloor`). The S28 clamp holds a dam row at the
//! floor `lb`; with `carry` the water it creates there,
//! `Σ max(lb − x, 0)·dt/c4`, is owed by the dam and paid back out of its later
//! inflow (`ddrs::routing::mmc`'s dam account), so the dam row conserves mass.
//!
//! 1. With no clamp on any dam row, `carry` routes bitwise like `forgive`,
//!    outputs and gradients, on option C rows and on the release path with a
//!    rule curve.
//! 2. Over one year (one rule-curve period) with forced clamps the carried
//!    account closes: `created − repaid − owed = 0`, the dam row's balance
//!    read off the routed series is `inflow − flux-stored − outflow − ΔS =
//!    −owed`, and the dam passes on its inflow (outflow = inflow to 1e-3),
//!    where `forgive` creates percents of it. Replace row (headwater) and
//!    additive row (interior dam, with the channel wedge).
//! 3. The owed state runs across test-phase chunks: a year routed as 15-day
//!    chunks, threaded as `training::eval` threads them (final discharge
//!    column, `training::forward::carry_dam_owed`, `DamClampSums::merge`),
//!    matches the year routed by one engine; the same chunks with the owed
//!    volume dropped at every boundary do not.
//! 4. `MuskingumCunge::set_dam_owed` refuses bad input.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use chrono::{Duration, NaiveDate};

use ddrs::config::{Config, DamFloor, DamRow};
use ddrs::routing::mmc::{DamRelease, RuleCurve, DT_SECONDS};
use ddrs::routing::release::{rule_curve_h, rule_curve_phase_start, seasonal_phase, OMEGA_RAD_PER_S};
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;
use ddrs::training::forward::carry_dam_owed;
use ddrs::training::release_eval::DamClampSums;

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

const DT: f64 = DT_SECONDS as f64;
/// One rule-curve period, 365.25 d, in hourly steps.
const YEAR_STEPS: usize = 8766;

fn network(n: usize, edges: &[(usize, usize)], length: f32) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n * n];
    for &(up, down) in edges {
        dense[down * n + up] = 1.0;
    }
    SparseAdjacency::from_dense(n, &dense, vec![length; n], vec![0.001; n])
}

/// `1 → 3`, `2 → 3`, `0 → 4`, `3 → 4`.
fn sandbox5() -> SparseAdjacency {
    network(5, &[(1, 3), (2, 3), (0, 4), (3, 4)], 5000.0)
}

/// `0 → 1 → 2`.
fn chain3(length: f32) -> SparseAdjacency {
    network(3, &[(0, 1), (1, 2)], length)
}

/// An engine on `adjacency` with `q_prime` (`[rows, n]` row-major), starting
/// from `initial` (the previous chunk's final discharge) or the cold start.
fn engine(cfg: &Config, adjacency: SparseAdjacency, q_prime: &[f32], initial: Option<&[f32]>) -> MuskingumCunge<I> {
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
        initial.map(|v| Tensor::<AB, 1>::from_floats(v, &device)),
    );
    mc
}

fn with_floor(mut cfg: Config, floor: DamFloor) -> Config {
    cfg.params.reservoir_dam_floor = Some(floor);
    cfg
}

fn host2(q: Tensor<AB, 2>) -> Vec<f32> {
    q.into_data().to_vec().unwrap()
}

fn assert_bitwise(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: idx {i}: {x} vs {y}");
    }
}

/// `row` of a routed output `[n, cols]`, f64.
fn series(out: &[f32], cols: usize, row: usize) -> Vec<f64> {
    out[row * cols..(row + 1) * cols].iter().map(|&v| v as f64).collect()
}

/// `Σ dt·(v_t + v_{t+1})/2`, m³.
fn trapezoid(v: &[f64]) -> f64 {
    v.windows(2).map(|w| 0.5 * (w[0] + w[1]) * DT).sum()
}

/// A diurnal lateral inflow `mean·(1 + amp·sin(2π·s/24))`, rows `0..=steps`.
fn diurnal(steps: usize, mean: f32, amp: f32) -> Vec<f32> {
    (0..=steps)
        .map(|s| mean * (1.0 + amp * (2.0 * std::f32::consts::PI * s as f32 / 24.0).sin()))
        .collect()
}

/// A rule curve `c = (1, 0, 0, 0)` with `Ibar` from `start`: the dam stores
/// `Ibar·sin ω` (spring and early summer) and releases it in autumn.
fn rule_curve(ibar: f32, start: NaiveDate) -> RuleCurve<I> {
    let device = Device::default();
    RuleCurve {
        coeffs: Tensor::<AB, 1>::from_floats([1.0_f32, 0.0, 0.0, 0.0], &device).reshape([1, 4]),
        inflow_mean: Tensor::from_floats([ibar], &device),
        phase0: rule_curve_phase_start(start),
    }
}

/// `Σ r·dt` for [`rule_curve`] over `steps` steps from `start`, f64.
fn stored_by_flux(start: NaiveDate, steps: usize, ibar: f64) -> f64 {
    let w0 = rule_curve_phase_start(start);
    ibar * (rule_curve_h(w0 + OMEGA_RAD_PER_S * DT * steps as f64)[0] - rule_curve_h(w0)[0])
}

// ---------------------------------------------------------------------------
// 1. No clamp: carry is forgive, bit for bit.
// ---------------------------------------------------------------------------

fn sandbox_q_prime(steps: usize) -> Vec<f32> {
    let base = [22.0_f32, 11.0, 11.0, 11.0, 22.0];
    (0..=steps)
        .flat_map(|t| {
            let s = (t as f32 / 24.0 * std::f32::consts::PI).sin();
            base.map(|b| b * (1.0 + 1.5 * s * s))
        })
        .collect()
}

#[test]
fn carry_is_forgive_bitwise_when_no_clamp_occurs() {
    let device = Device::default();
    let steps = 96;
    let start = NaiveDate::from_ymd_opt(1995, 4, 10).unwrap();
    let q = sandbox_q_prime(steps);

    // Option C rows, replace and additive.
    for row in [DamRow::Replace, DamRow::Additive] {
        let route = |floor: DamFloor| {
            let mut mc = engine(&with_floor(Config::default(), floor), sandbox5(), &q, None);
            mc.set_reservoir_rows_as(&[3, 4], &[0.8, 2.5], row).unwrap();
            let out = host2(mc.forward());
            (out, mc.dam_account().unwrap())
        };
        let (a, acc_a) = route(DamFloor::Forgive);
        let (b, acc_b) = route(DamFloor::Carry);
        assert_eq!(acc_b.created_m3, vec![0.0, 0.0], "{row:?}: precondition, no clamp");
        assert_eq!((acc_b.repaid_m3.clone(), acc_b.owed_m3.clone()), (vec![0.0; 2], vec![0.0; 2]));
        assert_bitwise(&a, &b, &format!("option C {row:?}: carry vs forgive"));
        assert_eq!(acc_a, acc_b);
    }

    // The release path with a rule curve: outputs and the gradients of T0 and
    // the coefficients.
    let route = |floor: DamFloor| {
        let t0 = Tensor::<AB, 1>::from_floats([0.4_f32], &device).require_grad();
        let c = Tensor::<AB, 1>::from_floats([0.05_f32, 0.03, -0.02, 0.01], &device)
            .reshape([1, 4])
            .require_grad();
        let mut mc = engine(&with_floor(Config::default(), floor), sandbox5(), &q, None);
        mc.set_dam_release(DamRelease {
            rows: vec![3],
            t0_days: t0.clone(),
            seasonal: None,
            phase: seasonal_phase(start, steps + 1),
            dam_row: DamRow::Additive,
            rule_curve: Some(RuleCurve {
                coeffs: c.clone(),
                inflow_mean: Tensor::from_floats([10.0_f32], &device),
                phase0: rule_curve_phase_start(start),
            }),
        })
        .unwrap();
        let out = mc.forward();
        let w = Tensor::<AB, 1>::from_floats(
            (0..5 * (steps + 1)).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect::<Vec<_>>().as_slice(),
            &device,
        )
        .reshape([5, steps + 1]);
        let grads = (out.clone() * w).sum().backward();
        let g = |t: &Tensor<AB, 1>| -> Vec<f32> { t.grad(&grads).unwrap().into_data().to_vec().unwrap() };
        let gc: Vec<f32> = c.grad(&grads).unwrap().into_data().to_vec().unwrap();
        (host2(out), g(&t0), gc, mc.dam_account().unwrap())
    };
    let (a, ga_t0, ga_c, acc_a) = route(DamFloor::Forgive);
    let (b, gb_t0, gb_c, acc_b) = route(DamFloor::Carry);
    assert_eq!(acc_b.created_m3, vec![0.0], "precondition, no clamp");
    assert_bitwise(&a, &b, "release + rule curve: carry vs forgive");
    assert_bitwise(&ga_t0, &gb_t0, "dL/dT0");
    assert_bitwise(&ga_c, &gb_c, "dL/dc");
    assert!(ga_c.iter().any(|&v| v != 0.0), "the coefficients get a gradient");
    assert_eq!(acc_a, acc_b);
}

// ---------------------------------------------------------------------------
// 2. One year with forced clamps: the carried dam row conserves mass.
// ---------------------------------------------------------------------------

/// A dam on `0 → 1 → 2` routed for one year with a rule curve that stores
/// more than the dam receives through spring and early summer.
struct YearCase {
    name: &'static str,
    cfg: Config,
    length: f32,
    /// The dam's row; row 0 is the upstream reach when the dam is row 1.
    dam: usize,
    /// Lateral inflow per row, `[steps + 1, 3]` row-major.
    q: Vec<f32>,
    /// The dam's own lateral inflow.
    q_dam: Vec<f32>,
    t_days: f32,
    ibar: f32,
    row: DamRow,
    /// `K·X` and `K·(1 − X)` of the dam reach's channel part (0 on the
    /// replace row), seconds, held constant by the case.
    kx: f64,
    k1x: f64,
}

const YEAR_START: (i32, u32, u32) = (2001, 1, 1);

fn year_start() -> NaiveDate {
    let (y, m, d) = YEAR_START;
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

/// Replace row, headwater dam (row 0), `T = 0.1 d`, `q'` 1..9 m³/s, `Ibar = 7`.
fn replace_headwater(steps: usize) -> YearCase {
    let q_dam = diurnal(steps, 5.0, 0.8);
    YearCase {
        name: "replace, headwater",
        cfg: Config::default(),
        length: 5000.0,
        dam: 0,
        q: q_dam.iter().flat_map(|&v| [v, 0.0, 0.0]).collect(),
        q_dam,
        t_days: 0.1,
        ibar: 7.0,
        row: DamRow::Replace,
        kx: 0.0,
        k1x: 0.0,
    }
}

/// Additive row, interior dam (row 1) below a reach carrying a steady
/// 4 m³/s, own `q'` 1..3 m³/s, `T = 0.05 d`, `Ibar = 8`. `K`, `X` constant:
/// `ddr_match` takes `X` from `x_storage` (0.3) and a 2 m/s velocity floor,
/// above every velocity the case reaches, fixes the celerity at `2·5/3` m/s.
fn additive_interior(steps: usize) -> YearCase {
    let mut cfg = Config::default();
    cfg.params.ddr_match = true;
    cfg.params.attribute_minimums.velocity = 2.0;
    let length = 10_000.0_f32;
    let k = (length / (2.0_f32 * (5.0_f32 / 3.0_f32))) as f64;
    let x = 0.3_f32 as f64;
    let q_dam = diurnal(steps, 2.0, 0.5);
    YearCase {
        name: "additive, interior",
        cfg,
        length,
        dam: 1,
        q: q_dam.iter().flat_map(|&v| [4.0, v, 0.0]).collect(),
        q_dam,
        t_days: 0.05,
        ibar: 8.0,
        row: DamRow::Additive,
        kx: k * x,
        k1x: k * (1.0 - x),
    }
}

/// Route `case` for `steps` with `floor`; the routed output and the account.
fn route_year(case: &YearCase, steps: usize, floor: DamFloor) -> (Vec<f32>, ddrs::routing::mmc::DamClampAccount) {
    let device = Device::default();
    let mut mc = engine(&with_floor(case.cfg.clone(), floor), chain3(case.length), &case.q, None);
    mc.set_dam_release(DamRelease {
        rows: vec![case.dam],
        t0_days: Tensor::from_floats([case.t_days], &device),
        seasonal: None,
        phase: seasonal_phase(year_start(), steps + 1),
        dam_row: case.row,
        rule_curve: Some(rule_curve(case.ibar, year_start())),
    })
    .unwrap();
    let out = host2(mc.forward());
    (out, mc.dam_account().unwrap())
}

/// `(inflow, outflow, ΔS, flux-stored)` of the dam row, m³, from the routed
/// series: inflow = routed upstream (trapezoid) + own lateral, `S` without `S0`.
fn dam_volumes(case: &YearCase, out: &[f32], steps: usize) -> (f64, f64, f64, f64) {
    let cols = steps + 1;
    let dam = series(out, cols, case.dam);
    let up = (case.dam == 1).then(|| series(out, cols, 0));
    let upstream = up.as_ref().map_or(0.0, |u| trapezoid(u));
    let lateral: f64 = case.q_dam[..steps].iter().map(|&v| v as f64 * DT).sum();
    let t_s = case.t_days as f64 * 86_400.0;
    let storage = |i: usize| case.kx * up.as_ref().map_or(0.0, |u| u[i]) + (case.k1x + t_s) * dam[i];
    let flux = stored_by_flux(year_start(), steps, case.ibar as f64);
    (upstream + lateral, trapezoid(&dam), storage(steps) - storage(0), flux)
}

#[test]
fn carry_closes_the_dam_row_volume_balance_over_a_year() {
    let steps = YEAR_STEPS;
    for case in [replace_headwater(steps), additive_interior(steps)] {
        let (fo, fa) = route_year(&case, steps, DamFloor::Forgive);
        let (co, ca) = route_year(&case, steps, DamFloor::Carry);
        let (f_in, f_out, f_ds, flux) = dam_volumes(&case, &fo, steps);
        let (c_in, c_out, c_ds, _) = dam_volumes(&case, &co, steps);
        let (created, repaid, owed) = (ca.created_m3[0], ca.repaid_m3[0], ca.owed_m3[0]);
        println!(
            "{}: forgive: inflow {f_in:.6e}, outflow {f_out:.6e}, dS {f_ds:.3e}, created {:.6e} ({} clamp steps) | \
             carry: inflow {c_in:.6e}, outflow {c_out:.6e}, dS {c_ds:.3e}, created {created:.6e}, repaid {repaid:.6e}, \
             owed {owed:.3e} ({} clamp steps) | flux-stored over the period {flux:.3e}",
            case.name, fa.created_m3[0], fa.clamp_steps[0], ca.clamp_steps[0]
        );
        // One period: the rule curve stores nothing net.
        assert!(flux.abs() < 1e-6 * f_in, "{}: one period moves no volume ({flux:.3e})", case.name);

        // Forgive: the balance lacks exactly the created volume, and it is large.
        assert!(fa.created_m3[0] > 0.01 * f_in, "{}: the case must force the clamp", case.name);
        let f_resid = f_in - flux - f_out - f_ds;
        assert!(
            (f_resid + fa.created_m3[0]).abs() <= 1e-4 * fa.created_m3[0],
            "{}: forgive balance {f_resid:.6e} vs −created {:.6e}",
            case.name,
            fa.created_m3[0]
        );

        // Carry: the clamp still binds, the account closes, and the dam row's
        // balance lacks only what is still owed.
        assert!(ca.clamp_steps[0] > 0 && created > 0.0, "{}: carry still clamps", case.name);
        assert!(repaid > 0.0 && owed >= 0.0, "{}: repaid {repaid}, owed {owed}", case.name);
        assert!(
            (created - repaid - owed).abs() <= 1e-4 * created,
            "{}: created {created:.6e} − repaid {repaid:.6e} − owed {owed:.6e} ≠ 0",
            case.name
        );
        let c_resid = c_in - flux - c_out - c_ds;
        assert!(
            (c_resid + owed).abs() <= 1e-4 * created,
            "{}: carry balance {c_resid:.6e} vs −owed {owed:.6e}",
            case.name
        );
        // The dam passes on its inflow: downstream volume = inflow volume.
        let (c_gap, f_gap) = ((c_out - c_in).abs(), (f_out - f_in).abs());
        assert!(c_gap <= 1e-3 * c_in, "{}: carry outflow {c_out:.6e} vs inflow {c_in:.6e}", case.name);
        assert!(f_gap > 10.0 * c_gap, "{}: forgive gap {f_gap:.3e} vs carry gap {c_gap:.3e}", case.name);
    }
}

// ---------------------------------------------------------------------------
// 3. The owed state runs across test-phase chunks.
// ---------------------------------------------------------------------------

/// Route `case` over one year as `chunk_days` chunks, threaded like
/// `training::eval::evaluate` threads the test phase: each chunk starts from
/// the previous one's final discharge column, the dams from what they still
/// owed (`carry_dam_owed`, when `carry_owed`), and `DamClampSums::merge`
/// keeps the running account. Consecutive chunks share their boundary
/// lateral-inflow row so that chunked and unchunked route the same steps.
/// Returns the dam's routed series (hours `0..=YEAR_STEPS`) and the account.
fn route_chunked(case: &YearCase, chunk_days: usize, carry_owed: bool) -> (Vec<f64>, DamClampSums) {
    let device = Device::default();
    let cfg = with_floor(case.cfg.clone(), DamFloor::Carry);
    let mut sums = DamClampSums::default();
    let mut dam = Vec::with_capacity(YEAR_STEPS + 1);
    let mut state: Option<Vec<f32>> = None;
    let mut h0 = 0;
    while h0 < YEAR_STEPS {
        let h1 = (h0 + chunk_days * 24).min(YEAR_STEPS);
        // Chunks start at midnight, as the test phase's 15-day chunks do.
        let chunk_start = year_start() + Duration::days((h0 / 24) as i64);
        let q = &case.q[h0 * 3..(h1 + 1) * 3];
        let mut mc = engine(&cfg, chain3(case.length), q, state.as_deref());
        mc.set_dam_release(DamRelease {
            rows: vec![case.dam],
            t0_days: Tensor::from_floats([case.t_days], &device),
            seasonal: None,
            phase: seasonal_phase(chunk_start, h1 - h0 + 1),
            dam_row: case.row,
            rule_curve: Some(RuleCurve {
                phase0: rule_curve_phase_start(chunk_start),
                ..rule_curve(case.ibar, year_start())
            }),
        })
        .unwrap();
        if carry_owed {
            carry_dam_owed(&mut mc, &sums);
        }
        let out = host2(mc.forward());
        sums.merge(&mc.dam_account().unwrap());
        let cols = h1 - h0 + 1;
        let row = series(&out, cols, case.dam);
        if h0 == 0 {
            dam.push(row[0]);
        }
        dam.extend_from_slice(&row[1..]);
        state = Some((0..3).map(|r| out[r * cols + cols - 1]).collect());
        h0 = h1;
    }
    (dam, sums)
}

#[test]
fn carry_runs_across_test_phase_chunks() {
    let case = replace_headwater(YEAR_STEPS);
    let (single_out, single) = route_year(&case, YEAR_STEPS, DamFloor::Carry);
    let single_dam = series(&single_out, YEAR_STEPS + 1, case.dam);
    let (chunked, sums) = route_chunked(&case, 15, true);
    let (dropped, dropped_sums) = route_chunked(&case, 15, false);
    assert_eq!(chunked.len(), single_dam.len());

    let max_diff = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max);
    let scale = single_dam.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let (d_carry, d_drop) = (max_diff(&chunked, &single_dam), max_diff(&dropped, &single_dam));
    let rec = sums.by_row[&case.dam];
    let dropped_rec = dropped_sums.by_row[&case.dam];
    println!(
        "one engine vs 15-day chunks: max |dQ| {d_carry:.3e} m3/s with the owed volume carried, {d_drop:.3e} dropped \
         (scale {scale:.3e}); created {:.6e} / {:.6e} / {:.6e}, repaid {:.6e} / {:.6e}, owed {:.3e} / {:.3e}",
        single.created_m3[0], rec.created_m3, dropped_rec.created_m3, single.repaid_m3[0], rec.repaid_m3,
        single.owed_m3[0], rec.owed_m3
    );
    assert!(d_carry <= 1e-4 * scale, "chunked with the owed volume carried differs by {d_carry:.3e}");
    assert!(d_drop > 1e-2 * scale, "dropping the owed volume must change the routing ({d_drop:.3e})");
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0);
    assert!(close(rec.created_m3, single.created_m3[0]), "created");
    assert!(close(rec.repaid_m3, single.repaid_m3[0]), "repaid");
    assert!(close(rec.owed_m3, single.owed_m3[0]), "owed");
    assert_eq!(rec.steps, YEAR_STEPS as u64);
    assert!(dropped_rec.repaid_m3 < 0.99 * dropped_rec.created_m3, "dropped debt is never repaid");
}

// ---------------------------------------------------------------------------
// 4. `set_dam_owed` input checks.
// ---------------------------------------------------------------------------

#[test]
fn set_dam_owed_refuses_bad_input() {
    let q = sandbox_q_prime(4);
    let cfg = with_floor(Config::default(), DamFloor::Carry);
    let mut bare = engine(&cfg, sandbox5(), &q, None);
    assert!(bare.set_dam_owed(&[]).unwrap_err().contains("no dam rows"));
    let mut mc = engine(&cfg, sandbox5(), &q, None);
    mc.set_reservoir_rows_as(&[3, 4], &[0.8, 2.5], DamRow::Additive).unwrap();
    assert_eq!(mc.dam_rows(), Some(&[3_usize, 4][..]));
    assert_eq!(mc.dam_floor(), DamFloor::Carry);
    assert!(mc.set_dam_owed(&[1.0]).unwrap_err().contains("2 armed dams"));
    assert!(mc.set_dam_owed(&[1.0, -1.0]).unwrap_err().contains(">= 0"));
    assert!(mc.set_dam_owed(&[f64::NAN, 0.0]).is_err());
    mc.set_dam_owed(&[5.0, 0.0]).unwrap();
    mc.forward();
    let acc = mc.dam_account().unwrap();
    // No clamp here, so the carried-in 5 m3 is repaid out of the first step's
    // ample inflow and nothing is owed.
    assert_eq!((acc.repaid_m3[0], acc.owed_m3[0]), (5.0, 0.0));
    assert_eq!(acc.created_m3, vec![0.0, 0.0]);
}
