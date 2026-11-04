//! Godin tidal filter (`ddrs::training::tidal_filter`) on hourly series.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData};
use ddrs::config::{Config, ConfigMode, TidalFilter};
use ddrs::data::{MeritGagesDataset, Staid};
use ddrs::nn::KanHead;
use ddrs::training::eval::{evaluate, EvalParams};
use ddrs::training::loss::tau_trim_and_downsample;
use ddrs::training::tidal_filter::{godin_filter, godin_log_line};
use std::f64::consts::PI;

type I = NdArray<f32>;
type Ad = Autodiff<I>;

const M2_H: f64 = 12.42;

fn series(rows: usize, hours: usize, f: impl Fn(usize, usize) -> f64) -> Vec<f32> {
    let mut v = Vec::with_capacity(rows * hours);
    for r in 0..rows {
        for t in 0..hours {
            v.push(f(r, t) as f32);
        }
    }
    v
}

fn tensor<B: burn::tensor::backend::Backend>(v: Vec<f32>, g: usize, t: usize) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(v, [g, t]), &Default::default())
}

fn to_vec<B: burn::tensor::backend::Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

/// Largest |value| of `row` outside the first and last 36 h.
fn interior_max(row: &[f32]) -> f32 {
    row[36..row.len() - 36].iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

#[test]
fn godin_removes_an_m2_sinusoid() {
    let t = 24 * 40;
    let x = tensor::<I>(series(1, t, |_, h| (2.0 * PI * h as f64 / M2_H).sin()), 1, t);
    let y = to_vec(godin_filter(x, &[0]));
    let amp = interior_max(&y);
    println!("M2 amplitude after the Godin filter: {amp:.6} (input 1.0)");
    assert!(amp < 0.01, "M2 left {amp}");
}

#[test]
fn godin_keeps_a_slow_signal() {
    let t = 24 * 120;
    let period_h = 30.0 * 24.0;
    let x = tensor::<I>(series(1, t, |_, h| (2.0 * PI * h as f64 / period_h).sin()), 1, t);
    let y = to_vec(godin_filter(x, &[0]));
    let amp = interior_max(&y);
    println!("30-day amplitude after the Godin filter: {amp:.6}");
    assert!(amp > 0.99 && amp < 1.001, "30-day amplitude ratio {amp}");
}

#[test]
fn godin_touches_only_the_listed_rows() {
    let (g, t) = (5, 24 * 10);
    let data = series(g, t, |r, h| (r as f64 + 1.0) * (h as f64 * 0.37).sin() + r as f64);
    let y = to_vec(godin_filter(tensor::<I>(data.clone(), g, t), &[1, 3]));
    for r in 0..g {
        let same = (0..t).all(|h| y[r * t + h].to_bits() == data[r * t + h].to_bits());
        assert_eq!(same, r != 1 && r != 3, "row {r}");
    }
}

#[test]
fn godin_filter_is_on_the_tape() {
    let (g, t) = (3, 24 * 8);
    let data = series(g, t, |r, h| (h as f64 * 0.5).sin() + r as f64);
    let x = tensor::<Ad>(data, g, t).require_grad();
    let loss = godin_filter(x.clone(), &[1]).sum();
    let grads = loss.backward();
    let gx: Vec<f32> = x.grad(&grads).unwrap().into_data().to_vec().unwrap();
    assert!(gx.iter().all(|v| v.is_finite()));
    let row = &gx[t..2 * t];
    assert!(row.iter().any(|v| *v != 0.0), "no gradient on the listed row");
    // Each hour of a filtered row feeds the filter weights, which sum to 1 in
    // the interior. The unlisted rows are copies, gradient 1.
    assert!((row[t / 2] - 1.0).abs() < 1e-4, "interior gradient {}", row[t / 2]);
    assert!(gx[..t].iter().all(|v| *v == 1.0));
}

#[test]
fn godin_log_line_matches_the_pinned_format() {
    assert_eq!(
        godin_log_line("eval", &[Staid::new("01372058")]),
        "eval: tidal filter: godin at 1 gauge (01372058)"
    );
    assert_eq!(
        godin_log_line("train", &[Staid::new("01372058"), Staid::new("01567000")]),
        "train: tidal filter: godin at 2 gauges (01372058, 01567000)"
    );
}

/// Amplitude of the best-fit sinusoid at `freq` cycles per sample.
fn fitted_amplitude(y: &[f32], freq: f64) -> f64 {
    let (mut c, mut s, mut cc, mut ss, mut cs) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, v) in y.iter().enumerate() {
        let (sn, cn) = (2.0 * PI * freq * i as f64).sin_cos();
        let v = *v as f64;
        c += v * cn;
        s += v * sn;
        cc += cn * cn;
        ss += sn * sn;
        cs += cn * sn;
    }
    let det = cc * ss - cs * cs;
    let a = (c * ss - s * cs) / det;
    let b = (s * cc - c * cs) / det;
    (a * a + b * b).sqrt()
}

#[test]
fn daily_mean_of_m2_aliases_at_14_8_days() {
    let t = 24 * 200;
    let data = series(1, t, |_, h| (2.0 * PI * h as f64 / M2_H).sin());
    // Daily samples of a 24 / 12.42 = 1.9324 cycles/day tide alias to 0.0676
    // cycles/day, a period of 14.8 days.
    let alias_freq = 2.0 - 24.0 / M2_H;
    println!("alias period: {:.2} days", 1.0 / alias_freq);
    assert!((1.0 / alias_freq - 14.8).abs() < 0.1);

    let raw = tau_trim_and_downsample(tensor::<I>(data.clone(), 1, t), 0);
    let raw = to_vec(raw);
    let raw_amp = fitted_amplitude(&raw, alias_freq);

    let filtered = tau_trim_and_downsample(godin_filter(tensor::<I>(data, 1, t), &[0]), 0);
    let filtered = to_vec(filtered);
    let n = filtered.len();
    // Skip the 2 edge days the 36 h edge-hold padding biases.
    let filt_amp = filtered[2..n - 2].iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    println!("14.8-day residual: unfiltered {raw_amp:.5}, filtered {filt_amp:.6} (hourly amplitude 1)");
    assert!(raw_amp > 0.02, "unfiltered alias {raw_amp}");
    assert!(filt_amp < 0.001, "filtered alias {filt_amp}");
}

const STAGE_YAML: &str = "examples/juniata/ddrs_stage.yaml";

/// Eval of a fresh seed-42 head over a short test window, as a daily matrix.
fn eval_with(yaml: &str, tag: &str) -> Vec<f32> {
    let dir = std::env::temp_dir().join(format!("ddrs_tidal_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cfg.yaml");
    std::fs::write(&path, yaml).unwrap();
    let mut cfg = Config::from_yaml_file_with_mode(&path, ConfigMode::Testing).unwrap();
    let sr = cfg.params.stage_routing.as_ref().unwrap();
    assert_eq!(sr.tidal_filter, TidalFilter::None, "{tag} must parse to none");
    {
        let e = cfg.experiment.as_mut().unwrap();
        e.start_time = "1995/10/01".into();
        e.end_time = "1995/12/31".into();
    }
    let device = Default::default();
    let head_cfg = ddrs::config::kan_config(cfg.kan_head.as_ref().unwrap(), cfg.seed);
    let head: KanHead<I> = head_cfg.init::<I>(&device);
    let ds = MeritGagesDataset::open(&cfg).unwrap();
    let out = evaluate::<I>(&cfg, &ds, EvalParams::KanHead(&head), &device, 15, &dir).unwrap();
    out.predictions_daily.iter().copied().collect()
}

#[test]
fn tidal_filter_none_parses_as_absent_in_eval() {
    let base = std::fs::read_to_string(STAGE_YAML).unwrap();
    assert!(!base.contains("tidal_filter"));
    let with_none = base.replace(
        "    downstream_boundary: normal_depth\n",
        "    downstream_boundary: normal_depth\n    tidal_filter: none\n",
    );
    assert_ne!(base, with_none);
    let a = eval_with(&base, "a");
    let b = eval_with(&with_none, "b");
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
}
