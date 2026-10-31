//! Stage block on the whole Juniata basin (213 reaches), in-process.
//!
//! (b) free outfall: the stage block (`examples/juniata/ddrs_stage.yaml`) and
//!     plain Muskingum-Cunge route the same 90-day window with the same fixed
//!     uniform parameters; the gauge volume must agree.
//! (c) synthetic tide: a prescribed 12.42 h stage at the outlet reach must
//!     reach upstream, damped, and not change the volume.
//!
//! Neither case skips. Debug wall times are in the report
//! (`.superpowers/sdd/2026-10-31-tree-solve-juniata/task-12-report.md`).
//! SKIP RULE: any case over 60 s in a debug build would use the
//! `cfg!(debug_assertions)` skip of `tests/juniata_acceptance.rs`; none does.
//!
//! The tide formula is also written in `scripts/make_synthetic_stage.py`. The
//! duplication is deliberate (no Python at test time); keep the two in step.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use ddrs::config::Config;
use ddrs::data::{MeritGagesDataset, RhoWindow, Staid};
use ddrs::geometry::compute_trapezoidal_geometry;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::training::forward::apply_stage_block;

type I = NdArray<f32>;
type Ad = Autodiff<I>;

const MC_CONFIG: &str = "examples/juniata/ddrs.yaml";
const STAGE_CONFIG: &str = "examples/juniata/ddrs_stage.yaml";
const GAUGE: &str = "01567000";
const OUTLET_COMID: i64 = 73005278;
const OUTLET: usize = 212;
const WARMUP_H: usize = 5 * 24;
/// Normalized `[n, q_spatial, p_spatial]` (C6): n 0.035, q 0.5, p 21.04.
const NORM: [f32; 3] = [0.0851, 0.5, 0.575];
const TIDE_PERIOD_H: f64 = 12.42;
const TIDE_AMP_M: f64 = 0.5;

/// Same formula as `scripts/make_synthetic_stage.py` (C33).
fn tide(t_hours: usize, h0: f64) -> f64 {
    h0 + TIDE_AMP_M * (2.0 * std::f64::consts::PI * t_hours as f64 / TIDE_PERIOD_H).sin()
}

struct Routed {
    /// `q[reach][hour]`.
    q: Vec<Vec<f32>>,
    mass_rel: Option<f32>,
    /// Per hour `(dt_phi, alpha)` when recording was on.
    coeffs: Vec<(Vec<f32>, Vec<f32>)>,
    boundary: Option<Vec<f32>>,
    q_rows: usize,
    slope_outlet: f32,
}

fn route(cfg: &Config, record: bool) -> Routed {
    let device = Default::default();
    let ds = MeritGagesDataset::open(cfg).expect("open Juniata dataset");
    let axis = ds.time_axis();
    let start = chrono::NaiveDate::from_ymd_opt(1990, 1, 1).unwrap();
    let window = RhoWindow {
        start_day_idx: (start - axis.start).num_days() as usize,
        rho_days: 90,
        window_start: start,
    };
    let batch = ds.collate(&[Staid::from(GAUGE)], &window).expect("collate");
    let tensors = batch.to_tensors::<I>(&device);
    let n = tensors.adjacency.n;
    assert_eq!(n, 213);
    let slope_outlet = tensors.adjacency.slope[OUTLET];
    let q_rows = tensors.q_prime.dims()[0];
    let col = |v: f32| Tensor::<Ad, 1>::full([n], v, &device);
    let mut engine = MuskingumCunge::<I>::new(cfg.clone(), device);
    engine.setup_inputs(
        RoutingInputs { adjacency: tensors.adjacency.clone(), x_storage: col(0.3) },
        Tensor::from_inner(tensors.q_prime.clone()),
        SpatialParameters {
            n: col(NORM[0]),
            q_spatial: col(NORM[1]),
            p_spatial: Some(col(NORM[2])),
            k_d: None,
            d_gw: None,
            leakance_factor: None,
            impervious_mask: None,
            gamma: None,
        },
        false,
        None,
    );
    apply_stage_block(
        cfg,
        &mut engine,
        tensors.stage.as_ref(),
        tensors.reservoir_rows.as_ref(),
        tensors.window.window_start,
    );
    if record {
        engine.stage_block_mut().expect("stage block armed").record_coefficients(true);
    }
    let out: Vec<f32> = engine.forward().inner().into_data().to_vec().unwrap();
    let t = out.len() / n;
    let q = (0..n).map(|r| out[r * t..(r + 1) * t].to_vec()).collect();
    let coeffs = engine
        .stage_block_mut()
        .map(|b| b.coefficient_record().to_vec())
        .unwrap_or_default();
    Routed {
        q,
        mass_rel: engine.stage_mass_balance().map(|m| m.rel()),
        coeffs,
        boundary: tensors.stage.as_ref().and_then(|s| s.boundary.clone()),
        q_rows,
        slope_outlet,
    }
}

fn nse(sim: &[f32], obs: &[f32]) -> f64 {
    let m = obs.iter().map(|&v| v as f64).sum::<f64>() / obs.len() as f64;
    let num: f64 = sim.iter().zip(obs).map(|(&s, &o)| (s as f64 - o as f64).powi(2)).sum();
    let den: f64 = obs.iter().map(|&o| (o as f64 - m).powi(2)).sum();
    1.0 - num / den
}

fn sum_after_warmup(q: &[f32]) -> f64 {
    q[WARMUP_H..].iter().map(|&v| v as f64).sum()
}

fn all_finite(r: &Routed) -> bool {
    r.q.iter().all(|row| row.iter().all(|v| v.is_finite()))
}

fn quantile(v: &mut Vec<f32>, p: f64) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Amplitude of the 12.42 h component of `q[WARMUP_H..]` (least squares fit of
/// a sine and a cosine at that period).
fn amp_1242(q: &[f32]) -> f64 {
    let (mut cs, mut ss, mut n) = (0.0, 0.0, 0.0);
    let mean = q[WARMUP_H..].iter().map(|&v| v as f64).sum::<f64>() / (q.len() - WARMUP_H) as f64;
    for (k, &v) in q.iter().enumerate().skip(WARMUP_H) {
        let w = 2.0 * std::f64::consts::PI * k as f64 / TIDE_PERIOD_H;
        cs += (v as f64 - mean) * w.cos();
        ss += (v as f64 - mean) * w.sin();
        n += 1.0;
    }
    2.0 / n * (cs * cs + ss * ss).sqrt()
}

/// Longest run of consecutive hourly `dQ` of alternating sign. Changes below
/// 1e-4 of the mean flow count as zero and break a run.
fn longest_alternating_run(q: &[f32]) -> usize {
    let mean = q[WARMUP_H..].iter().sum::<f32>() / (q.len() - WARMUP_H) as f32;
    let tol = 1e-4 * mean;
    let d: Vec<f32> = q[WARMUP_H..].windows(2).map(|w| w[1] - w[0]).collect();
    let (mut best, mut run) = (0usize, 0usize);
    for k in 0..d.len() {
        if d[k].abs() <= tol {
            run = 0;
            continue;
        }
        if k > 0 && d[k - 1].abs() > tol && d[k - 1] * d[k] < 0.0 {
            run += 1;
        } else {
            run = 1;
        }
        best = best.max(run);
    }
    best
}

#[test]
fn juniata_stage_vs_mc_whole_basin_free_outfall() {
    let a = route(&Config::from_yaml_file(MC_CONFIG).unwrap(), false);
    let b = route(&Config::from_yaml_file(STAGE_CONFIG).unwrap(), true);
    assert!(all_finite(&a) && all_finite(&b), "non-finite Q");

    let (sa, sb) = (sum_after_warmup(&a.q[OUTLET]), sum_after_warmup(&b.q[OUTLET]));
    let ratio = sb / sa;
    println!("volume ratio stage/MC at the gauge (after 5 d warmup): {ratio:.5}");
    assert!((sb - sa).abs() / sa < 0.02, "gauge volume differs by {:.3} %", 100.0 * (ratio - 1.0));

    let rel = b.mass_rel.expect("stage block armed");
    println!("stage mass balance rel (gross-flux normalized): {rel:.3e}");
    assert!(rel <= 1e-4, "mass balance rel {rel}");

    for (name, r) in [("gauge 212", 212usize), ("reach 60", 60), ("reach 199", 199)] {
        let v = nse(&b.q[r][WARMUP_H..], &a.q[r][WARMUP_H..]);
        println!("NSE(stage vs MC) at {name}: {v:.4}");
    }

    let mut dtphi = Vec::new();
    let mut alpha = Vec::new();
    for (p, al) in &b.coeffs {
        for (x, y) in p.iter().zip(al) {
            if *x > 0.0 {
                dtphi.push(*x);
                alpha.push(*y);
            }
        }
    }
    assert!(!dtphi.is_empty(), "no recorded friction coefficients");
    println!(
        "dt*phi: median {:.3}, p10 {:.3}; median alpha {:.4} ({} samples)",
        quantile(&mut dtphi.clone(), 0.5),
        quantile(&mut dtphi, 0.1),
        quantile(&mut alpha, 0.5),
        alpha.len()
    );

    let run = longest_alternating_run(&b.q[183]);
    println!("longest alternating dQ run at reach 183: {run} h (MC: {})", longest_alternating_run(&a.q[183]));
}

/// Normal depth (m) at the outlet for mean flow `q`, from the engine's own
/// trapezoid inversion with the test parameters (n 0.035, q 0.5, p 21.04).
fn outlet_normal_depth(q: f32, slope: f32) -> f64 {
    let device = Default::default();
    let t = |v: f32| Tensor::<I, 1>::from_floats([v], &device);
    let g = compute_trapezoidal_geometry::<I>(
        t(0.015 + NORM[0] * (0.25 - 0.015)),
        t((NORM[2] * 200f32.ln()).exp()),
        t(NORM[1]),
        t(q),
        t(slope.max(1e-3)),
        0.01,
        0.01,
    );
    g.depth.into_data().to_vec::<f32>().unwrap()[0] as f64
}

#[test]
fn juniata_synthetic_tide_at_outlet_propagates_upstream() {
    let free = route(&Config::from_yaml_file(STAGE_CONFIG).unwrap(), false);
    let mean_q = (sum_after_warmup(&free.q[OUTLET]) / (free.q[OUTLET].len() - WARMUP_H) as f64) as f32;
    // Anchor 0, so z_out[212] = 0 and h0 = normal depth at the mean flow.
    let h0 = outlet_normal_depth(mean_q, free.slope_outlet);
    println!("outlet mean Q {mean_q:.2} m3/s, h0 = {h0:.3} m");

    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("tide.csv");
    let mut body = String::from("time,stage_m\n");
    let t0 = chrono::NaiveDate::from_ymd_opt(1990, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let series: Vec<f64> = (0..2400).map(|t| tide(t, h0)).collect();
    for (t, h) in series.iter().enumerate() {
        let stamp = (t0 + chrono::Duration::hours(t as i64)).format("%Y-%m-%dT%H:%M:%SZ");
        body.push_str(&format!("{stamp},{h:.6}\n"));
    }
    std::fs::write(&csv, body).unwrap();

    let text = std::fs::read_to_string(STAGE_CONFIG).unwrap();
    let from = "downstream_boundary: normal_depth";
    assert!(text.contains(from));
    let text = text.replace(
        from,
        &format!("downstream_boundary: prescribed_stage\n    boundary_comid: {OUTLET_COMID}\n    anchor_elevation_m: 0.0"),
    );
    assert!(text.contains("data_sources:\n"));
    let text = text.replace("data_sources:\n", &format!("data_sources:\n  stage_boundary: {}\n", csv.display()));
    let yaml = dir.path().join("ddrs_stage_tide.yaml");
    std::fs::write(&yaml, text).unwrap();
    let cfg = Config::from_yaml_file(&yaml).expect("prescribed-stage variant loads and validates");

    let tidal = route(&cfg, false);
    assert!(all_finite(&tidal), "non-finite Q");

    // Carry-over (Task 10): the boundary slice lines up with the q' rows.
    let bc = tidal.boundary.as_ref().expect("prescribed boundary reaches the batch");
    assert_eq!(bc.len(), tidal.q_rows, "boundary length equals the q' row count");
    assert!((bc[0] as f64 - series[0]).abs() < 1e-5, "boundary[0] is the 00:00 UTC value");
    assert!((bc[3] as f64 - series[3]).abs() < 1e-5, "boundary[3] is the 03:00 UTC value");

    // The free run has a small 12.42 h component of its own (hourly
    // interpolation of daily Q'), so the tide response is the tide-minus-free
    // difference. Raw amplitudes are printed too.
    let diff = |r: usize| -> Vec<f32> {
        tidal.q[r].iter().zip(&free.q[r]).map(|(a, b)| a - b).collect()
    };
    let (a212, a211, a199) = (amp_1242(&diff(212)), amp_1242(&diff(211)), amp_1242(&diff(199)));
    println!(
        "12.42 h amplitude of Q, raw (m3/s): 212 {:.4}, 211 {:.4}, 199 {:.4}; free outfall 212 {:.4}, 211 {:.4}, 199 {:.4}",
        amp_1242(&tidal.q[212]), amp_1242(&tidal.q[211]), amp_1242(&tidal.q[199]),
        amp_1242(&free.q[212]), amp_1242(&free.q[211]), amp_1242(&free.q[199])
    );
    println!("12.42 h amplitude of the tide response (tide minus free): 212 {a212:.4}, 211 {a211:.4}, 199 {a199:.4}");
    println!("ratios: 211/212 {:.4}, 199/212 {:.4}", a211 / a212, a199 / a212);
    assert!(a211 > 0.0 && a211 < a212, "211 must carry a damped tide");
    assert!(a199 < 0.1 * a212, "reach 199 must be under 10 % of the outlet amplitude");

    let (vt, vf) = (sum_after_warmup(&tidal.q[OUTLET]), sum_after_warmup(&free.q[OUTLET]));
    println!("volume ratio tide/free at the gauge: {:.5}", vt / vf);
    assert!((vt - vf).abs() / vf < 0.02, "tide changes the gauge volume by {:.3} %", 100.0 * (vt / vf - 1.0));
}
