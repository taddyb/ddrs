//! Stage block on the tidal Hudson (756 reaches, 41 on the stage block), in
//! process, against an MC-only control.
//!
//! Release only: each routed 90-day window takes tens of seconds. Debug builds
//! skip with the `cfg!(debug_assertions)` pattern of `tests/juniata_acceptance.rs`.
//!
//! The run uses uniform normalized head parameters `[n, q_spatial, p_spatial]`
//! = `[0.0851, 0.5, 0.575]` (n 0.035, q 0.5, p 21.04) on the MC reaches. The
//! 41 stage reaches take their width, bed and `n` from the geometry table, so
//! the head never touches them. A trained head therefore sees the same
//! estuary physics.
//!
//! Asserted: finiteness, stage mass balance, the volume bar against MC, depth
//! above the floor, the network shape. Everything else is printed with
//! `println!` and copied into the task report (tide shape, mean ranges, ringing,
//! chunk carry).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use ddrs::config::{kan_config, Config, ConfigMode};
use ddrs::data::test_window::TestWindow;
use ddrs::data::{MeritGagesDataset, RhoWindow, Staid};
use ddrs::routing::stage::StageMassBalance;
use ddrs::routing::{MuskingumCunge, RoutingInputs, SpatialParameters};
use ddrs::training::eval::EvalParams;
use ddrs::training::forward::{apply_stage_block, forward_eval_reaches_with_dams, scatter_add_by_group, StageEvalState};
use ddrs::training::loss::tau_trim_and_downsample;
use ddrs::training::release_eval::DamClampSums;
use ddrs::training::tidal_filter::godin_filter;

type I = NdArray<f32>;
type Ad = Autodiff<I>;

const STAGE_CONFIG: &str = "examples/hudson/ddrs.yaml";
const MC_CONFIG: &str = "examples/hudson/ddrs_mc_only.yaml";
const GEOMETRY_CSV: &str = "examples/hudson/data/hudson_channel_geometry.csv";

// Reach COMIDs nearest each tide station (L10). Found with the one-off
// geopandas script `hudson_nearest_reaches.py` (scratchpad, not committed):
// it reads the MERIT fabric
// `/mnt/ssd1/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`,
// keeps the 41 stage COMIDs of the geometry table, takes each reach's
// mid-point latitude and picks the reach nearest the station latitude:
// Haverstraw Bay 41.218 N, Turkey Point 42.014 N, Albany 42.65 N. The Battery
// reach is the outlet (40.7006 N is south of every stage reach).
const BATTERY_COMID: i64 = 73005067;
const HAVERSTRAW_COMID: i64 = 73002951;
const TURKEY_POINT_COMID: i64 = 73002968;
const ALBANY_COMID: i64 = 73001652;
const POUGHKEEPSIE_COMID: i64 = 73002961;
const POUGHKEEPSIE_GAUGE: &str = "01372058";
/// Mean tidal range, m (NOAA CO-OPS datum tables): Battery, Haverstraw, Turkey Point, Albany.
const NOAA_RANGE_M: [f64; 4] = [1.38, 0.98, 1.15, 1.52];
const GAUGES: [&str; 4] = ["01375000", "01358000", "01372058", "73005067"];

/// Normalized `[n, q_spatial, p_spatial]`: n 0.035, q 0.5, p 21.04.
const NORM: [f32; 3] = [0.0851, 0.5, 0.575];
const WINDOW_DAYS: usize = 90;
const WARMUP_DAYS: usize = 5;
const M2_PERIOD_H: f64 = 12.4206;

macro_rules! release_only {
    () => {
        if cfg!(debug_assertions) {
            eprintln!(
                "skipping: the Hudson stage window needs opt-level 3 - \
                 run `cargo test --release --test hudson_stage -- --nocapture`"
            );
            return;
        }
    };
}

#[derive(Clone, Copy)]
struct Spec {
    mc_only: bool,
    theta: Option<f32>,
    substeps: Option<u32>,
    /// Add 2.5 m to the Battery series for 12 h from hour 720.
    surge: bool,
}

const STAGE: Spec = Spec { mc_only: false, theta: None, substeps: None, surge: false };

struct Run {
    /// `q[reach][hour]`.
    q: Vec<Vec<f32>>,
    /// `h[hour][reach]`, empty for MC only.
    h: Vec<Vec<f32>>,
    mass: Option<StageMassBalance>,
    wall: Duration,
    row_of: HashMap<i64, usize>,
    /// Stage rows in geometry-table order (seed to outlet).
    stage_rows: Vec<usize>,
    bed: Vec<f32>,
    length_m: Vec<f32>,
    n_reaches: usize,
    n_outlets: usize,
    boundary: Option<Vec<f32>>,
    /// Filtered (stage) or raw (MC) daily Q at Poughkeepsie after the driver's tau trim.
    pough_daily_godin: Vec<f32>,
    pough_obs: Vec<f32>,
    depth_lb: f32,
}

fn stage_comids() -> (Vec<i64>, usize) {
    let text = std::fs::read_to_string(GEOMETRY_CSV).expect("geometry csv");
    let mut comids = Vec::new();
    let mut with_source = 0;
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()).skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        comids.push(f[0].trim().parse::<i64>().unwrap());
        if f.get(4).is_some_and(|s| !s.trim().is_empty()) {
            with_source += 1;
        }
    }
    (comids, with_source)
}

fn route(spec: Spec) -> Run {
    let path = if spec.mc_only { MC_CONFIG } else { STAGE_CONFIG };
    let mut cfg = Config::from_yaml_file(path).expect("hudson config");
    if let Some(sr) = cfg.params.stage_routing.as_mut() {
        if let Some(t) = spec.theta {
            sr.theta = t;
        }
        if let Some(s) = spec.substeps {
            sr.substeps = s;
        }
    }
    let device = Default::default();
    let ds = MeritGagesDataset::open(&cfg).expect("open Hudson dataset");
    let axis = ds.time_axis();
    let start = chrono::NaiveDate::from_ymd_opt(2010, 3, 1).unwrap();
    let window = RhoWindow {
        start_day_idx: (start - axis.start).num_days() as usize,
        rho_days: WINDOW_DAYS,
        window_start: start,
    };
    let staids: Vec<Staid> = GAUGES.iter().map(|s| Staid::from(*s)).collect();
    let mut batch = ds.collate(&staids, &window).expect("collate");
    if spec.surge {
        let b = batch.stage.as_mut().and_then(|s| s.boundary.as_mut()).expect("prescribed boundary");
        for v in &mut b[720..732] {
            *v += 2.5;
        }
    }
    let row_of: HashMap<i64, usize> =
        batch.divide_comids.iter().enumerate().map(|(i, c)| (c.0, i)).collect();
    let pough_col = batch.gauge_staids.iter().position(|s| s.as_str() == POUGHKEEPSIE_GAUGE).expect("gauge");
    let tensors = batch.to_tensors::<I>(&device);
    let n = tensors.adjacency.n;
    let n_outlets = {
        let mut has_down = vec![false; n];
        for &c in &tensors.adjacency.cols {
            has_down[c as usize] = true;
        }
        has_down.iter().filter(|d| !**d).count()
    };
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
        &cfg,
        &mut engine,
        tensors.stage.as_ref(),
        tensors.reservoir_rows.as_ref(),
        tensors.window.window_start,
    );
    if !spec.mc_only {
        engine.stage_block_mut().expect("stage block armed").record_coefficients(true);
    }
    let t0 = Instant::now();
    let out: Vec<f32> = engine.forward().inner().into_data().to_vec().unwrap();
    let wall = t0.elapsed();
    let t = out.len() / n;
    let q: Vec<Vec<f32>> = (0..n).map(|r| out[r * t..(r + 1) * t].to_vec()).collect();
    let h = engine.stage_block_mut().map(|b| b.stage_record().to_vec()).unwrap_or_default();

    let (comids, _) = stage_comids();
    let stage_rows: Vec<usize> = comids.iter().map(|c| row_of[c]).collect();
    let bed = tensors
        .stage
        .as_ref()
        .and_then(|s| s.geometry.as_ref())
        .map(|g| stage_rows.iter().map(|r| g.bed_z[g.rows.iter().position(|x| x == r).unwrap()]).collect())
        .unwrap_or_default();

    // Godin-filtered (stage) or raw (MC) daily Q at Poughkeepsie, the way the driver does it.
    let pr = row_of[&POUGHKEEPSIE_COMID];
    let series = Tensor::<I, 1>::from_floats(q[pr].as_slice(), &device).reshape([1, t]);
    let series = if spec.mc_only { series } else { godin_filter(series, &[0]) };
    let tau = cfg.params.tau as u32;
    let daily: Vec<f32> = tau_trim_and_downsample(series, tau).into_data().to_vec().unwrap();
    let pough_obs: Vec<f32> = tensors.observations.column(pough_col).to_vec();

    Run {
        q,
        h,
        mass: engine.stage_mass_balance(),
        wall,
        row_of,
        stage_rows,
        bed,
        length_m: tensors.adjacency.length_m.clone(),
        n_reaches: n,
        n_outlets,
        boundary: tensors.stage.as_ref().and_then(|s| s.boundary.clone()),
        pough_daily_godin: daily,
        pough_obs,
        depth_lb: cfg.params.attribute_minimums.depth,
    }
}

fn all_finite(r: &Run) -> bool {
    r.q.iter().all(|row| row.iter().all(|v| v.is_finite())) && r.h.iter().all(|row| row.iter().all(|v| v.is_finite()))
}

fn nse(sim: &[f32], obs: &[f32]) -> f64 {
    let pairs: Vec<(f64, f64)> =
        sim.iter().zip(obs).filter(|(s, o)| s.is_finite() && o.is_finite()).map(|(&s, &o)| (s as f64, o as f64)).collect();
    let m = pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len() as f64;
    let num: f64 = pairs.iter().map(|p| (p.0 - p.1).powi(2)).sum();
    let den: f64 = pairs.iter().map(|p| (p.1 - m).powi(2)).sum();
    1.0 - num / den
}

fn volume(daily: &[f32]) -> f64 {
    daily[WARMUP_DAYS..].iter().map(|&v| v as f64).sum()
}

/// Mean over whole days of the daily max minus min of `x`, last `days` days.
fn mean_range(x: &[f32], days: usize) -> f64 {
    let tail = &x[x.len() - 24 * days..];
    tail.chunks(24)
        .map(|d| {
            let hi = d.iter().cloned().fold(f32::MIN, f32::max);
            let lo = d.iter().cloned().fold(f32::MAX, f32::min);
            (hi - lo) as f64
        })
        .sum::<f64>()
        / days as f64
}

/// Least-squares M2 fit over the last 15 days: `(amplitude, phase_rad)` of `A cos(w t - phi)`.
fn m2_fit(x: &[f32]) -> (f64, f64) {
    let tail = &x[x.len() - 15 * 24..];
    let w = 2.0 * std::f64::consts::PI / M2_PERIOD_H;
    let mean = tail.iter().map(|&v| v as f64).sum::<f64>() / tail.len() as f64;
    let (mut cs, mut ss) = (0.0, 0.0);
    for (k, &v) in tail.iter().enumerate() {
        cs += (v as f64 - mean) * (w * k as f64).cos();
        ss += (v as f64 - mean) * (w * k as f64).sin();
    }
    let n = tail.len() as f64;
    let (a, b) = (2.0 * cs / n, 2.0 * ss / n);
    ((a * a + b * b).sqrt(), b.atan2(a))
}

/// Longest run of consecutive hourly `dQ` of alternating sign. Changes below
/// 1e-4 of the mean flow count as zero and break a run. Same rule as
/// `tests/juniata_stage.rs`.
fn longest_alternating_run(q: &[f32], warmup_h: usize) -> usize {
    let mean = q[warmup_h..].iter().sum::<f32>() / (q.len() - warmup_h) as f32;
    let tol = 1e-4 * mean.abs();
    let d: Vec<f32> = q[warmup_h..].windows(2).map(|w| w[1] - w[0]).collect();
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

/// Share of the (mean-removed) variance at periods of 2 to 4 h, by direct DFT.
fn variance_share_2_to_4h(q: &[f32], warmup_h: usize) -> f64 {
    let x = &q[warmup_h..];
    let n = x.len();
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let x: Vec<f64> = x.iter().map(|&v| v as f64 - mean).collect();
    let (mut total, mut band) = (0.0, 0.0);
    for k in 1..=n / 2 {
        let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let (mut re, mut im) = (0.0, 0.0);
        for (j, v) in x.iter().enumerate() {
            re += v * (w * j as f64).cos();
            im += v * (w * j as f64).sin();
        }
        let p = re * re + im * im;
        total += p;
        let period = n as f64 / k as f64;
        if (2.0..=4.0).contains(&period) {
            band += p;
        }
    }
    band / total
}

#[test]
fn hudson_stage_window_is_finite_and_conserves() {
    release_only!();
    let a = route(STAGE);
    assert_eq!(a.n_reaches, 756, "the batch routes the whole harbor domain");
    assert_eq!(a.n_outlets, 1, "one outlet");
    assert_eq!(a.stage_rows.len(), 41);
    // The stage set drains to the outlet: the last table row is the outlet.
    assert_eq!(*a.stage_rows.last().unwrap(), a.row_of[&BATTERY_COMID]);
    assert_eq!(a.n_outlets, 1);
    assert!(all_finite(&a), "non-finite Q or stage");

    let mb = a.mass.expect("stage block armed");
    let rel = mb.rel();
    println!(
        "stage mass balance rel (gross-flux normalized): {rel:.3e} over {} steps (gross {:.3e} m3, floor_created {:.3e} m3)",
        mb.steps, mb.gross_flux_m3, mb.floor_created_m3
    );
    assert!(rel <= 1e-4, "mass balance rel {rel}");

    // Depth above the floor at > 99 % of (reach, hour) pairs.
    let (mut ok, mut tot) = (0usize, 0usize);
    for hour in &a.h {
        for (j, &r) in a.stage_rows.iter().enumerate() {
            tot += 1;
            if hour[r] - a.bed[j] > a.depth_lb {
                ok += 1;
            }
        }
    }
    let frac = ok as f64 / tot as f64;
    println!("depth above the floor ({} m): {:.4} % of {tot} (reach, hour) pairs", a.depth_lb, 100.0 * frac);
    assert!(frac > 0.99, "depth above floor at only {:.3} %", 100.0 * frac);

    // Start transient: max |dh| per hour over the first 24 h on the 41 reaches.
    let mut per_hour = Vec::new();
    for k in 1..24.min(a.h.len()) {
        let m = a.stage_rows.iter().map(|&r| (a.h[k][r] - a.h[k - 1][r]).abs()).fold(0.0f32, f32::max);
        per_hour.push(m);
    }
    println!(
        "start transient: max |dh| per hour over the first 24 h (m): first {:.4}, hour 12 {:.4}, last {:.4}, max {:.4}",
        per_hour[0],
        per_hour[11],
        per_hour[per_hour.len() - 1],
        per_hour.iter().cloned().fold(0.0, f32::max)
    );
    let bc = a.boundary.as_ref().expect("boundary");
    let outlet = a.row_of[&BATTERY_COMID];
    println!("hour 0: outlet h {:.4} m, Battery boundary {:.4} m", a.h[0][outlet], bc[0]);

    // Mean tidal range at the four stations (last 15 days).
    let stations = [
        ("Battery (outlet)", BATTERY_COMID),
        ("Haverstraw Bay", HAVERSTRAW_COMID),
        ("Turkey Point", TURKEY_POINT_COMID),
        ("Albany", ALBANY_COMID),
    ];
    let series = |c: i64| -> Vec<f32> { a.h.iter().map(|row| row[a.row_of[&c]]).collect() };
    let ranges: Vec<f64> = stations.iter().map(|(_, c)| mean_range(&series(*c), 15)).collect();
    for (i, (name, c)) in stations.iter().enumerate() {
        println!(
            "mean tidal range, {name} (COMID {c}): modelled {:.3} m (ratio to modelled Battery {:.3}); NOAA {:.2} m (ratio {:.3})",
            ranges[i],
            ranges[i] / ranges[0],
            NOAA_RANGE_M[i],
            NOAA_RANGE_M[i] / NOAA_RANGE_M[0]
        );
    }
    let boundary_tail = &bc[bc.len() - 15 * 24..];
    println!("mean range of the prescribed Battery series over the last 15 days: {:.3} m", {
        boundary_tail
            .chunks(24)
            .map(|d| (d.iter().cloned().fold(f32::MIN, f32::max) - d.iter().cloned().fold(f32::MAX, f32::min)) as f64)
            .sum::<f64>()
            / 15.0
    });

    // Tide shape (P1): asserted only on a real, non-fallback geometry.
    let (_, k) = stage_comids();
    if k > 0 {
        println!(
            "tide shape: not asserted, {k} of 41 geometry rows are fallback rows (Albany range {:.3} m, Haverstraw range {:.3} m)",
            ranges[3], ranges[1]
        );
    } else {
        println!(
            "tide shape: asserted (Albany range {:.3} m above Haverstraw range {:.3} m)",
            ranges[3], ranges[1]
        );
        assert!(ranges[3] > ranges[1], "Albany range {:.3} m is not above Haverstraw {:.3} m", ranges[3], ranges[1]);
    }

    // Poughkeepsie against the Battery: M2 amplitude ratio and phase lag.
    let pough = series(POUGHKEEPSIE_COMID);
    let battery = series(BATTERY_COMID);
    let (ap, pp) = m2_fit(&pough);
    let (ab, pb) = m2_fit(&battery);
    let mut lag = pp - pb;
    while lag < -std::f64::consts::PI {
        lag += 2.0 * std::f64::consts::PI;
    }
    while lag > std::f64::consts::PI {
        lag -= 2.0 * std::f64::consts::PI;
    }
    println!(
        "Poughkeepsie M2 stage amplitude {ap:.3} m (Battery {ab:.3} m, ratio {:.3}); phase lag {:.1} deg = {:.2} h; {} 0.2 m",
        ap / ab,
        lag.to_degrees(),
        lag / (2.0 * std::f64::consts::PI) * M2_PERIOD_H,
        if ap > 0.2 { "above" } else { "BELOW" }
    );

    // Fast modes (P7): theta 0.6 (a) and 0.55 (b).
    let warm = WARMUP_DAYS * 24;
    let ringing = |run: &Run, tag: &str| {
        let mut idx: Vec<usize> = run.stage_rows.clone();
        idx.sort_by(|&x, &y| run.length_m[x].partial_cmp(&run.length_m[y]).unwrap());
        let pr = run.row_of[&POUGHKEEPSIE_COMID];
        let runs: Vec<String> = idx[..3]
            .iter()
            .map(|&r| format!("{:.0} m: {} h", run.length_m[r], longest_alternating_run(&run.q[r], warm)))
            .collect();
        println!(
            "[{tag}] longest hour-to-hour dQ alternation at the 3 shortest stage reaches: {}; 2 to 4 h share of Poughkeepsie hourly-Q variance {:.3e}",
            runs.join(", "),
            variance_share_2_to_4h(&run.q[pr], warm)
        );
    };
    ringing(&a, "theta 0.6");
    let b = route(Spec { theta: Some(0.55), ..STAGE });
    assert!(all_finite(&b), "non-finite at theta 0.55");
    ringing(&b, "theta 0.55");
    let mc = route(Spec { mc_only: true, ..STAGE });
    ringing(&mc, "MC only");
    println!(
        "theta 0.55: mass rel {:.3e}; tide ranges (Battery, Haverstraw, Turkey Point, Albany): {}",
        b.mass.unwrap().rel(),
        stations
            .iter()
            .map(|(_, c)| format!("{:.3}", mean_range(&b.h.iter().map(|row| row[b.row_of[c]]).collect::<Vec<_>>(), 15)))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Wall time: substeps 4 against substeps 1, and the MC-only control.
    let s1 = route(Spec { substeps: Some(1), ..STAGE });
    println!(
        "wall time of forward(): MC only {:.1} s, stage substeps 1 {:.1} s, substeps 4 (theta 0.6) {:.1} s, substeps 4 (theta 0.55) {:.1} s; ratio substeps 4 / MC {:.2}, substeps 4 / substeps 1 {:.2}",
        mc.wall.as_secs_f64(),
        s1.wall.as_secs_f64(),
        a.wall.as_secs_f64(),
        b.wall.as_secs_f64(),
        a.wall.as_secs_f64() / mc.wall.as_secs_f64(),
        a.wall.as_secs_f64() / s1.wall.as_secs_f64()
    );
    println!("substeps 1: finite {}, mass rel {:.3e}", all_finite(&s1), s1.mass.map(|m| m.rel()).unwrap_or(f32::NAN));
}

#[test]
fn hudson_filtered_daily_discharge_at_poughkeepsie() {
    release_only!();
    let mc = route(Spec { mc_only: true, ..STAGE });
    let st = route(STAGE);
    assert!(mc.pough_daily_godin.iter().chain(&st.pough_daily_godin).all(|v| v.is_finite()), "non-finite daily Q");
    let (va, vb) = (volume(&mc.pough_daily_godin), volume(&st.pough_daily_godin));
    println!(
        "Poughkeepsie volume after {WARMUP_DAYS} d warmup (sum of daily Q, m3/s-days): MC {va:.1}, stage (Godin) {vb:.1}, ratio {:.5}",
        vb / va
    );
    assert!((vb - va).abs() / va < 0.03, "volume differs by {:.3} %", 100.0 * (vb / va - 1.0));
    let days = mc.pough_daily_godin.len().min(mc.pough_obs.len());
    let obs = &mc.pough_obs[WARMUP_DAYS..days];
    let finite_obs = obs.iter().filter(|v| v.is_finite()).count();
    println!(
        "NSE against 72137 over days {WARMUP_DAYS}..{days} ({finite_obs} finite obs days): MC only {:.4}, stage + Godin {:.4}",
        nse(&mc.pough_daily_godin[WARMUP_DAYS..days], obs),
        nse(&st.pough_daily_godin[WARMUP_DAYS..days], obs)
    );
}

#[test]
fn surge_day_is_finite() {
    release_only!();
    let s = route(Spec { surge: true, ..STAGE });
    assert!(all_finite(&s), "non-finite under the surge");
    let mb = s.mass.expect("stage block armed");
    println!(
        "surge (+2.5 m for 12 h from hour 720): mass rel {:.3e}, floor created {:.3e} m3 (gross {:.3e} m3)",
        mb.rel(),
        mb.floor_created_m3,
        mb.gross_flux_m3
    );
    assert!(mb.rel() <= 1e-4, "surge mass balance rel {}", mb.rel());
    let outlet = s.row_of[&BATTERY_COMID];
    let peak = s.h.iter().map(|row| row[outlet]).fold(f32::MIN, f32::max);
    println!("surge: outlet peak stage {peak:.3} m");
}

/// Hourly gauge series (per gauge, chunks concatenated) and the carry after
/// the last chunk. WET copy of the chunk loop of
/// `src/training/eval.rs::eval_chunks_carry_the_stage_state` (a private test
/// module there), extended to several gauges.
fn eval_hourly(
    cfg: &Config,
    dataset: &MeritGagesDataset,
    head: &ddrs::nn::kan_head::KanHead<I>,
    chunk_days: usize,
    carry: bool,
    stage: bool,
) -> (Vec<Vec<f32>>, Vec<String>, Option<Vec<f32>>) {
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let mut cfg = cfg.clone();
    if !stage {
        cfg.params.stage_routing = None;
    }
    let axis = dataset.time_axis().clone();
    let mut state = StageEvalState::default();
    let mut dams = DamClampSums::default();
    let mut prev: Option<Vec<f32>> = None;
    let mut per_gauge: Vec<Vec<f32>> = Vec::new();
    let mut staids: Vec<String> = Vec::new();
    let mut day = 0;
    while day < axis.num_days {
        let n = (axis.num_days - day).min(chunk_days);
        let win = TestWindow::new(&axis, day, n);
        let batch = dataset.collate_window(&win).expect("collate");
        let mut tensors = batch.to_tensors::<I>(&device);
        staids = tensors.gauge_staids.iter().map(|s| s.as_str().to_string()).collect();
        if let Some(q0) = prev.as_ref() {
            tensors.initial_state = Some(Tensor::<I, 1>::from_floats(q0.as_slice(), &device));
        }
        if !carry {
            state.carry = None;
        }
        let reaches = forward_eval_reaches_with_dams::<I>(
            &cfg,
            &tensors,
            head,
            &device,
            false,
            None,
            &mut dams,
            stage.then_some(&mut state),
        );
        let [nr, th] = reaches.dims();
        prev = Some(reaches.clone().slice([0..nr, th - 1..th]).reshape([nr]).into_data().into_vec().unwrap());
        let g = tensors.num_gauges;
        let pred: Vec<f32> = scatter_add_by_group(reaches, tensors.flat_indices.clone(), tensors.group_ids.clone(), g)
            .into_data()
            .into_vec()
            .unwrap();
        let t = pred.len() / g;
        if per_gauge.is_empty() {
            per_gauge = vec![Vec::new(); g];
        }
        for (gi, row) in per_gauge.iter_mut().enumerate() {
            row.extend_from_slice(&pred[gi * t..(gi + 1) * t]);
        }
        day += n;
    }
    (per_gauge, staids, state.carry.map(|c| c.h))
}

/// Daily means of the Godin-filtered hourly series.
fn godin_daily(hourly: &[f32]) -> Vec<f32> {
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let t = hourly.len();
    let x = Tensor::<I, 1>::from_floats(hourly, &device).reshape([1, t]);
    let f: Vec<f32> = godin_filter(x, &[0]).into_data().into_vec().unwrap();
    f.chunks_exact(24).map(|d| d.iter().sum::<f32>() / 24.0).collect()
}

#[test]
fn hudson_eval_chunk_carry_is_recorded() {
    release_only!();
    let mut cfg = Config::from_yaml_file_with_mode(STAGE_CONFIG, ConfigMode::Testing).expect("hudson testing config");
    {
        let exp = cfg.experiment.as_mut().unwrap();
        exp.start_time = "2012/10/01".into();
        exp.end_time = "2012/10/30".into();
    }
    let dataset = MeritGagesDataset::open(&cfg).expect("open hudson");
    let device = <I as burn::tensor::backend::BackendTypes>::Device::default();
    let head = kan_config(cfg.kan_head.as_ref().unwrap(), cfg.seed).init::<I>(&device);
    let _ = EvalParams::KanHead(&head); // the real `evaluate` takes this; see the eval.rs test
    assert_eq!(dataset.time_axis().num_days, 30);

    let (one, staids, carry_one) = eval_hourly(&cfg, &dataset, &head, 30, true, true);
    let (two, _, carry_two) = eval_hourly(&cfg, &dataset, &head, 15, true, true);
    let (mc_one, _, _) = eval_hourly(&cfg, &dataset, &head, 30, true, false);
    let (mc_two, _, _) = eval_hourly(&cfg, &dataset, &head, 15, true, false);
    let (two_no_carry, _, carry_nc) = eval_hourly(&cfg, &dataset, &head, 15, false, true);
    let pg = staids.iter().position(|s| s == POUGHKEEPSIE_GAUGE).expect("Poughkeepsie gauge");
    for v in one.iter().chain(&two).chain(&mc_one).chain(&mc_two).flatten() {
        assert!(v.is_finite(), "non-finite hourly gauge value");
    }

    // Stage rows: the 41 COMIDs of the table, through the dataset's row order.
    let (comids, _) = stage_comids();
    let batch = dataset.collate_window(&TestWindow::new(dataset.time_axis(), 0, 15)).unwrap();
    let rows: Vec<usize> = comids
        .iter()
        .map(|c| batch.divide_comids.iter().position(|x| x.0 == *c).expect("stage comid in network"))
        .collect();
    let (h1, h2, hn) = (carry_one.unwrap(), carry_two.unwrap(), carry_nc.unwrap());
    assert!(h1.iter().chain(&h2).all(|v| v.is_finite()));
    let dh = |a: &[f32], b: &[f32]| rows.iter().map(|&r| (a[r] - b[r]).abs()).fold(0.0f32, f32::max);
    println!(
        "chunk carry: max abs dh on the 41 stage rows after day 30, 1 chunk vs 2 chunks with carry {:.4e} m (2 chunks without carry {:.4e} m)",
        dh(&h1, &h2),
        dh(&h1, &hn)
    );

    let diff = |a: &[f32], b: &[f32]| -> (f32, f32) {
        let (da, db) = (godin_daily(a), godin_daily(b));
        assert_eq!(da.len(), db.len());
        let abs = da.iter().zip(&db).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let rel = da.iter().zip(&db).map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-3)).fold(0.0f32, f32::max);
        (abs, rel)
    };
    let (sa, sr) = diff(&one[pg], &two[pg]);
    let (na, nr) = diff(&one[pg], &two_no_carry[pg]);
    let (ma, mr) = diff(&mc_one[pg], &mc_two[pg]);
    println!(
        "Godin-filtered daily Q at Poughkeepsie, 1 vs 2 chunks: stage with carry max abs {sa:.4e} m3/s, max rel {sr:.4e}; \
         stage without carry {na:.4e} m3/s, {nr:.4e}; MC only (the floor) {ma:.4e} m3/s, {mr:.4e}"
    );
    // Where the difference sits: the day of the largest gap, and the hourly shift that best aligns chunk 2.
    {
        let (d1, d2) = (godin_daily(&one[pg]), godin_daily(&two[pg]));
        let (day, _) = d1.iter().zip(&d2).enumerate().map(|(i, (x, y))| (i, (x - y).abs())).fold((0, 0.0f32), |b, c| if c.1 > b.1 { c } else { b });
        println!("largest daily gap at day {day} (of {}): 1 chunk {:.2}, 2 chunks {:.2} m3/s", d1.len(), d1[day], d2[day]);
        let half = one[pg].len() / 2;
        println!("hourly Q lengths: 1 chunk {}, 2 chunks {}", one[pg].len(), two[pg].len());
        for t in half - 4..half + 4 {
            println!("  hour {t}: 1 chunk {:.1}, 2 chunks {:.1}", one[pg][t], two[pg][t]);
        }
        for shift in [-1i64, 0, 1] {
            let worst = (half + 48..one[pg].len() - 48)
                .map(|t| (one[pg][t] - two[pg][(t as i64 + shift) as usize]).abs())
                .fold(0.0f32, f32::max);
            println!("hourly Q at Poughkeepsie, chunk-2 half, 1 chunk[t] vs 2 chunks[t{shift:+}]: max abs {worst:.2} m3/s");
        }
    }
    for v in [sa, sr, na, nr, ma, mr] {
        assert!(v.is_finite());
    }
}
