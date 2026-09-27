//! Seasonal dam release in the engine (`MuskingumCunge::set_dam_release`):
//! `T_d(t) = T0_d·exp(a_d·sin ω_t + b_d·cos ω_t)`, clamped at one hour, routed
//! as a linear reservoir (Muskingum `K := T_d(t)`, `X := 0`) on dam rows.
//!
//! Forward-only checks here; the gradients are in
//! `tests/reservoir_release_gradcheck.rs`.
//!
//! 1. `a = b = 0` (and the non-seasonal form) is bitwise option C
//!    (`set_reservoir_rows`) with the same `T`.
//! 2. A headwater dam follows the storage-conserving seasonal recurrence
//!    (trapezoid on `S = T·Q`): `c3 = (2·T_t − dt)/(2·T_{t+1} + dt)`, with
//!    `T_t` at phase row `t − 1` and `T_{t+1}` at row `t`, and conserves
//!    storage exactly: `T_{t+1}·Q_{t+1} − T_t·Q_t = dt·(q' − (Q_t + Q_{t+1})/2)`.
//! 3. `seasonal_phase` is the fractional day of year, pandas-style (1-based).
//! 4. Rows upstream of the dam are bitwise untouched.
//! 5. Input validation.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use chrono::NaiveDate;

use ddrs::config::Config;
use ddrs::routing::mmc::{DamRelease, DT_SECONDS};
use ddrs::routing::release::seasonal_phase;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::sparse::SparseAdjacency;

type I = NdArray<f32>;
type AB = Autodiff<I>;
type Device = <I as burn::tensor::backend::BackendTypes>::Device;

fn network(n_reach: usize, edges: &[(usize, usize)]) -> SparseAdjacency {
    let mut dense = vec![0.0_f32; n_reach * n_reach];
    for &(up, down) in edges {
        assert!(down > up);
        dense[down * n_reach + up] = 1.0;
    }
    SparseAdjacency::from_dense(n_reach, &dense, vec![5000.0; n_reach], vec![0.001; n_reach])
}

/// `1 → 3`, `2 → 3`, `0 → 4`, `3 → 4` (the RAPID sandbox topology).
fn sandbox5() -> SparseAdjacency {
    network(5, &[(1, 3), (2, 3), (0, 4), (3, 4)])
}

/// `0 → 1 → 2`.
fn chain3() -> SparseAdjacency {
    network(3, &[(0, 1), (1, 2)])
}

enum Dam<'a> {
    None,
    OptionC(&'a [usize], &'a [f32]),
    /// rows, T0 days, Some((a, b)), phase
    Release(&'a [usize], &'a [f32], Option<(&'a [f32], &'a [f32])>, Vec<[f32; 2]>),
}

fn engine(adjacency: SparseAdjacency, q_prime: &[f32]) -> MuskingumCunge<I> {
    let device = Device::default();
    let n_reach = adjacency.n;
    let n_rows = q_prime.len() / n_reach;
    let leaf = |v: f32| Tensor::<AB, 1>::from_floats(vec![v; n_reach].as_slice(), &device);
    let inputs = RoutingInputs::<I> {
        adjacency,
        x_storage: Tensor::ones([n_reach], &device) * 0.3,
    };
    let spatial = SpatialParameters::<I> {
        n: leaf(0.5),
        q_spatial: leaf(0.5),
        p_spatial: Some(leaf(0.575)),
        k_d: None,
        d_gw: None,
        leakance_factor: None,
        impervious_mask: None,
        gamma: None,
    };
    let q_prime = Tensor::<AB, 1>::from_floats(q_prime, &device).reshape([n_rows, n_reach]);
    let mut mc = MuskingumCunge::<I>::new(Config::default(), device);
    mc.setup_inputs(inputs, q_prime, spatial, false, None);
    mc
}

fn route(adjacency: SparseAdjacency, q_prime: &[f32], dam: Dam) -> Vec<f32> {
    let device = Device::default();
    let mut mc = engine(adjacency, q_prime);
    match dam {
        Dam::None => {}
        Dam::OptionC(rows, t) => mc.set_reservoir_rows(rows, t).expect("rows"),
        Dam::Release(rows, t0, seasonal, phase) => {
            let t = |v: &[f32]| Tensor::<AB, 1>::from_floats(v, &device);
            mc.set_dam_release(DamRelease {
                rows: rows.to_vec(),
                t0_days: t(t0),
                seasonal: seasonal.map(|(a, b)| (t(a), t(b))),
                phase,
                dam_row: ddrs::config::DamRow::Replace,
            })
            .expect("release")
        }
    }
    mc.forward().into_data().to_vec::<f32>().unwrap()
}

fn pulse_q_prime(n_reach: usize, steps: usize) -> Vec<f32> {
    let mut q = Vec::with_capacity((steps + 1) * n_reach);
    for t in 0..=steps {
        let s = (t as f32 / steps as f32 * std::f32::consts::PI).sin();
        let pulse = 1.0 + 1.5 * s * s;
        q.extend((0..n_reach).map(|r| (11.0 + 5.0 * r as f32) * pulse));
    }
    q
}

fn assert_bitwise(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: idx {i}: {x} vs {y}");
    }
}

// ---------------------------------------------------------------------------
// 1. a = b = 0 is option C, bit for bit.
// ---------------------------------------------------------------------------

#[test]
fn zero_seasonal_coefficients_are_bitwise_option_c() {
    let steps = 72;
    let q = pulse_q_prime(5, steps);
    let phase = seasonal_phase(NaiveDate::from_ymd_opt(2001, 3, 15).unwrap(), steps + 1);
    let option_c = route(sandbox5(), &q, Dam::OptionC(&[3], &[0.4]));
    let zero = route(
        sandbox5(),
        &q,
        Dam::Release(&[3], &[0.4], Some((&[0.0], &[0.0])), phase.clone()),
    );
    let constant = route(sandbox5(), &q, Dam::Release(&[3], &[0.4], None, phase));
    assert_bitwise(&option_c, &zero, "a = b = 0 vs option C");
    assert_bitwise(&option_c, &constant, "non-seasonal release vs option C");
    let plain = route(sandbox5(), &q, Dam::None);
    assert!(option_c != plain, "the dam must change the routing");
}

// ---------------------------------------------------------------------------
// 2. Seasonal recurrence at a headwater dam.
// ---------------------------------------------------------------------------

#[test]
fn headwater_dam_follows_the_storage_conserving_seasonal_recurrence() {
    let steps = 96;
    let n_reach = 3;
    let q = pulse_q_prime(n_reach, steps);
    // A full seasonal cycle every 48 steps, so T swings by exp(±|(a, b)|).
    let phase: Vec<[f32; 2]> = (0..=steps)
        .map(|t| {
            let w = 2.0 * std::f32::consts::PI * t as f32 / 48.0;
            [w.sin(), w.cos()]
        })
        .collect();
    let (t0, a, b) = (0.3_f32, 0.9_f32, -0.6_f32);
    let res = route(chain3(), &q, Dam::Release(&[0], &[t0], Some((&[a], &[b])), phase.clone()));
    let cols = steps + 1;
    let dt = DT_SECONDS as f64;
    let mut worst = 0.0_f64;
    let t_sec = |row: usize| {
        let [s, c] = phase[row];
        (t0 as f64 * (a as f64 * s as f64 + b as f64 * c as f64).exp()).max(1.0 / 24.0) * 86_400.0
    };
    for t in 1..cols {
        // Headwater: no upstream term, q0[t] = c3·q0[t−1] + c4·q'0[t−1] with
        // c3 = (2·T_t − dt)/(2·T_{t+1} + dt), c4 = 2·dt/(2·T_{t+1} + dt):
        // T_t at phase row t − 1 (the step's start), T_{t+1} at row t (its end).
        let (t_prev, t_next) = (t_sec(t - 1), t_sec(t));
        let denom = 2.0 * t_next + dt;
        let expected = (2.0 * t_prev - dt) / denom * res[t - 1] as f64
            + 2.0 * dt / denom * q[(t - 1) * n_reach] as f64;
        let rel = (res[t] as f64 - expected).abs() / expected.abs();
        worst = worst.max(rel);
        assert!(rel < 2e-5, "step {t}: routed {} vs recurrence {expected} (rel {rel:.3e})", res[t]);
    }
    println!("seasonal recurrence: worst rel {worst:.3e}");
}

/// The dam row conserves `S = T·Q` whatever `T` does: over a window with a
/// violent seasonal swing (a full cycle every 48 h, `T` from 0.7 to 13 d),
/// inflow minus outflow equals the change in storage. The pre-fix row
/// (`K := T_{t+1}` alone) carries `Q` across a change in `T` and fails this
/// by a wide margin (it adds `Q·dT/dt`).
#[test]
fn seasonal_release_conserves_storage() {
    // 10.25 cycles, so the window ends at a different T than it starts (a
    // whole number of cycles with a steady inflow would hide the old row's
    // error: Q stays at the inflow and T returns to where it began).
    let steps = 492;
    let n_reach = 3;
    // Pulsed lateral inflow into the headwater dam: 10 m³/s with a
    // 40 m³/s, 12-hour storm every 100 hours.
    let lateral_at = |t: usize| if t % 100 < 12 { 40.0_f32 } else { 10.0 };
    let q: Vec<f32> = (0..=steps).flat_map(|t| [lateral_at(t), 0.0, 0.0]).collect();
    let phase: Vec<[f32; 2]> = (0..=steps)
        .map(|t| {
            let w = 2.0 * std::f32::consts::PI * t as f32 / 48.0;
            [w.sin(), w.cos()]
        })
        .collect();
    let (t0, a, b) = (3.0_f32, 1.5_f32, 0.0_f32);
    let res = route(chain3(), &q, Dam::Release(&[0], &[t0], Some((&[a], &[b])), phase.clone()));
    let dt = DT_SECONDS as f64;
    let t_sec = |row: usize| {
        let [s, c] = phase[row];
        (t0 as f64 * (a as f64 * s as f64 + b as f64 * c as f64).exp()).max(1.0 / 24.0) * 86_400.0
    };
    // Reach 0 (the dam) only: q0 is res[0..=steps].
    let mut inflow = 0.0_f64;
    let mut outflow = 0.0_f64;
    for t in 1..=steps {
        // Step t routes lateral row t − 1, held over the step.
        inflow += dt * lateral_at(t - 1) as f64;
        outflow += dt * 0.5 * (res[t - 1] as f64 + res[t] as f64);
    }
    let ds = t_sec(steps) * res[steps] as f64 - t_sec(0) * res[0] as f64;
    let imbalance = (inflow - outflow - ds).abs() / inflow;
    println!(
        "storage balance: inflow {inflow:.4e} outflow {outflow:.4e} dS {ds:.4e} imbalance {imbalance:.3e}; \
         mean out / mean in {:.6}",
        outflow / inflow
    );
    assert!(imbalance < 1e-4, "dam row does not conserve storage: imbalance {imbalance:.3e}");
}

#[test]
fn clamp_holds_t_at_one_hour() {
    // T0 at 1.5 h with a = -2 at sin ω = 1: T0·e^-2 ≈ 0.2 h, clamped to 1 h.
    let steps = 24;
    let n_reach = 3;
    let q = pulse_q_prime(n_reach, steps);
    let phase = vec![[1.0_f32, 0.0_f32]; steps + 1];
    let clamped = route(
        chain3(),
        &q,
        Dam::Release(&[0], &[1.5 / 24.0], Some((&[-2.0], &[0.0])), phase.clone()),
    );
    let one_hour = route(chain3(), &q, Dam::OptionC(&[0], &[1.0 / 24.0]));
    assert_bitwise(&one_hour, &clamped, "clamped T vs T = 1 h");
}

// ---------------------------------------------------------------------------
// 3. Phase table.
// ---------------------------------------------------------------------------

#[test]
fn seasonal_phase_is_fractional_day_of_year() {
    let w = |doy: f64| {
        let x = 2.0 * std::f64::consts::PI * doy / 365.25;
        [x.sin() as f32, x.cos() as f32]
    };
    let p = seasonal_phase(NaiveDate::from_ymd_opt(2001, 1, 1).unwrap(), 49);
    assert_eq!(p.len(), 49);
    assert_eq!(p[0], w(1.0), "Jan 1 00:00 is doy 1, as pandas dayofyear");
    assert_eq!(p[12], w(1.5));
    assert_eq!(p[24], w(2.0));
    // Year rollover: Dec 31 of a non-leap year is doy 365, then Jan 1 is 1.
    let q = seasonal_phase(NaiveDate::from_ymd_opt(2001, 12, 31).unwrap(), 30);
    assert_eq!(q[0], w(365.0));
    assert_eq!(q[24], w(1.0));
    assert_eq!(q[25], w(1.0 + 1.0 / 24.0));
}

// ---------------------------------------------------------------------------
// 4. Rows upstream of the dam are untouched.
// ---------------------------------------------------------------------------

#[test]
fn rows_not_below_the_dam_are_bitwise_untouched() {
    let steps = 48;
    let q = pulse_q_prime(5, steps);
    let phase = seasonal_phase(NaiveDate::from_ymd_opt(2001, 7, 1).unwrap(), steps + 1);
    let plain = route(sandbox5(), &q, Dam::None);
    let dam = route(sandbox5(), &q, Dam::Release(&[3], &[2.0], Some((&[0.5], &[0.5])), phase));
    let cols = steps + 1;
    for reach in [0, 1, 2] {
        let r = reach * cols..(reach + 1) * cols;
        assert_bitwise(&plain[r.clone()], &dam[r], &format!("reach {reach}"));
    }
    assert!(plain[3 * cols..] != dam[3 * cols..], "dam and outlet must change");
}

// ---------------------------------------------------------------------------
// 5. Validation.
// ---------------------------------------------------------------------------

#[test]
fn set_dam_release_rejects_bad_input() {
    let device = Device::default();
    let steps = 10;
    let q = pulse_q_prime(3, steps);
    let t = |v: &[f32]| Tensor::<AB, 1>::from_floats(v, &device);
    let phase = vec![[0.0_f32, 1.0_f32]; steps + 1];
    let err = |rel: DamRelease<I>| {
        let mut mc = engine(chain3(), &q);
        mc.set_dam_release(rel).expect_err("must be rejected")
    };
    let ok = || DamRelease::<I> {
        rows: vec![1],
        t0_days: t(&[1.0]),
        seasonal: None,
        phase: phase.clone(),
        dam_row: ddrs::config::DamRow::Replace,
    };
    assert!(err(DamRelease { rows: vec![3], ..ok() }).contains("outside"));
    assert!(err(DamRelease { rows: vec![1, 1], t0_days: t(&[1.0, 2.0]), ..ok() }).contains("twice"));
    assert!(err(DamRelease { t0_days: t(&[1.0, 2.0]), ..ok() }).contains("T0"));
    assert!(err(DamRelease { seasonal: Some((t(&[0.0, 0.0]), t(&[0.0]))), ..ok() }).contains("a"));
    assert!(err(DamRelease { phase: vec![[0.0, 1.0]; steps], ..ok() }).contains("phase"));

    // Option C and a release on the same engine are exclusive.
    let mut mc = engine(chain3(), &q);
    mc.set_reservoir_rows(&[1], &[1.0]).unwrap();
    assert!(mc.set_dam_release(ok()).unwrap_err().contains("set_reservoir_rows"));

    // Empty rows leave the engine exactly as with no release.
    let mut mc = engine(chain3(), &q);
    mc.set_dam_release(DamRelease { rows: vec![], t0_days: t(&[]), ..ok() }).unwrap();
    let empty = mc.forward().into_data().to_vec::<f32>().unwrap();
    assert_bitwise(&route(chain3(), &q, Dam::None), &empty, "empty release");
}
