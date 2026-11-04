//! End-to-end acceptance of the Hudson tidal example: full `train-and-test` on
//! CPU, MC-only control (`examples/hudson/ddrs_mc_only.yaml`) and the stage
//! block with the Godin filter at Poughkeepsie (`examples/hudson/ddrs.yaml`).
//!
//! Per-gauge NSE and KGE come from `eval/predictions.zarr` (the manifest holds
//! only aggregates), sliced past `experiment.warmup` as
//! `src/training/eval.rs::evaluate` does, and joined to the baseline by gauge id.
//!
//! The eval runs as ONE chunk (`testing.batch_size` 2600 days). A 15-day chunk
//! seam leaks into the 71 h Godin filter (see `examples/hudson/README.md`).
//! There is no tide-shape assert here: `ddrs run` writes daily Q only, and the
//! geometry is the fallback table. Tide checks live in `tests/hudson_stage.rs`.
//!
//! Release-only. Local only, not in CI. The test greps `run.log` (stderr), so it
//! needs `--nocapture`:
//!
//!     cargo test --release --test hudson_acceptance -- --nocapture
//!
//! Must run from the repo root. Workspaces go to tempdirs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ddrs::cli::run::{run, RunInput};
use ddrs::cli::workspace::Workspace;
use ddrs::training::Metrics;
use ndarray::{s, Array2};

const STAGE_CONFIG: &str = "examples/hudson/ddrs.yaml";
const MC_CONFIG: &str = "examples/hudson/ddrs_mc_only.yaml";
const POUGHKEEPSIE: &str = "01372058";
const GREEN_ISLAND: &str = "01358000";
const CROTON: &str = "01375000";
const HARBOR: &str = "73005067";
/// `experiment.warmup` in both configs.
const WARMUP: usize = 5;

/// Floors (measured minus 0.04, as `juniata_acceptance`). Measured 2026-11-01 (seed 42, CPU): stage NSE 0.3083 / KGE 0.6253 at Poughkeepsie.
const STAGE_POUGHKEEPSIE_NSE_FLOOR: f64 = 0.26;
const STAGE_POUGHKEEPSIE_KGE_FLOOR: f64 = 0.58;

/// WET copy of `run_juniata` in `tests/juniata_acceptance.rs` (kept parallel
/// copies). CPU `train-and-test` of `config` into a fresh workspace.
fn run_hudson(config: &Path, tmp: &Path) -> PathBuf {
    run(RunInput {
        workspace: Workspace::with_root(tmp.join(".ddrs")),
        config_path: config.to_path_buf(),
        workflow: None,
        plot: false,
        strict: false,
        max_mini_batches: None,
        batch_order_from: None,
        backend: "cpu".into(),
    })
    .expect("hudson train-and-test run failed")
}

struct Eval {
    ids: Vec<String>,
    pred: Array2<f32>,
    obs: Array2<f32>,
}

/// `/predictions`, `/observations`, `/gage_ids` of `eval/predictions.zarr`.
fn read_eval(run_dir: &Path) -> Eval {
    use zarrs::array::Array;
    use zarrs::filesystem::FilesystemStore;
    let store = std::sync::Arc::new(
        FilesystemStore::new(run_dir.join("eval/predictions.zarr")).expect("open predictions.zarr"),
    );
    let read2 = |name: &str| -> Array2<f32> {
        let arr = Array::open(store.clone(), name).expect("open array");
        let shape = (arr.shape()[0] as usize, arr.shape()[1] as usize);
        let v: Vec<f64> = arr.retrieve_array_subset(&arr.subset_all()).expect("read array");
        Array2::from_shape_vec(shape, v.into_iter().map(|x| x as f32).collect()).unwrap()
    };
    let ids_arr = Array::open(store.clone(), "/gage_ids").expect("open /gage_ids");
    let width = ids_arr.shape()[1] as usize;
    let raw: Vec<u8> = ids_arr.retrieve_array_subset(&ids_arr.subset_all()).expect("read gage_ids");
    let ids = raw
        .chunks(width)
        .map(|c| String::from_utf8_lossy(c).trim_end_matches('\0').to_string())
        .collect();
    Eval { ids, pred: read2("/predictions"), obs: read2("/observations") }
}

/// Per-gauge metrics after the warmup slice, keyed by gauge id.
fn per_gauge(e: &Eval) -> HashMap<String, (f64, f64)> {
    let m = Metrics::compute(
        &e.pred.slice(s![.., WARMUP..]).to_owned(),
        &e.obs.slice(s![.., WARMUP..]).to_owned(),
    );
    e.ids.iter().enumerate().map(|(i, id)| (id.clone(), (m.nse[i] as f64, m.kge[i] as f64))).collect()
}

/// Amplitude of the best-fit sinusoid of period `period` days on the days where
/// `mask` holds (least squares with a constant term).
fn sine_amplitude(series: &[f64], mask: &[bool], period: f64) -> f64 {
    let (mut n, mut sc, mut ss, mut scc, mut sss, mut scs) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sy, mut syc, mut sys) = (0.0, 0.0, 0.0);
    for (t, (&y, &ok)) in series.iter().zip(mask).enumerate() {
        if !ok {
            continue;
        }
        let w = 2.0 * std::f64::consts::PI * t as f64 / period;
        let (c, s_) = (w.cos(), w.sin());
        n += 1.0;
        sc += c;
        ss += s_;
        scc += c * c;
        sss += s_ * s_;
        scs += c * s_;
        sy += y;
        syc += y * c;
        sys += y * s_;
    }
    // Solve the 3x3 normal equations [1, c, s] by Cramer's rule.
    let a = [[n, sc, ss], [sc, scc, scs], [ss, scs, sss]];
    let b = [sy, syc, sys];
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    let col = |k: usize| {
        let mut m = a;
        for r in 0..3 {
            m[r][k] = b[r];
        }
        det(m) / d
    };
    (col(1).powi(2) + col(2).powi(2)).sqrt()
}

/// Spectral peak ratio (model / observed) near the 14.77 day spring-neap cycle:
/// the exact period, and the maximum over periods 13.5 to 16 days.
fn spring_neap_ratio(e: &Eval, gauge: &str) -> (f64, f64) {
    let i = e.ids.iter().position(|g| g == gauge).expect("gauge in eval");
    let p: Vec<f64> = e.pred.row(i).iter().skip(WARMUP).map(|&v| v as f64).collect();
    let o: Vec<f64> = e.obs.row(i).iter().skip(WARMUP).map(|&v| v as f64).collect();
    let mask: Vec<bool> = p.iter().zip(&o).map(|(a, b)| a.is_finite() && b.is_finite()).collect();
    let at = |per: f64| sine_amplitude(&p, &mask, per) / sine_amplitude(&o, &mask, per);
    let peak = |x: &[f64]| {
        (0..=50)
            .map(|k| sine_amplitude(x, &mask, 13.5 + 0.05 * k as f64))
            .fold(0.0_f64, f64::max)
    };
    (at(14.77), peak(&p) / peak(&o))
}

fn json_u64(v: &serde_json::Value, key: &str) -> u64 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or_else(|| panic!("metrics missing `{key}`: {v}"))
}

fn read_json(p: PathBuf) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}")))
        .expect("parse json")
}

#[test]
fn hudson_mc_only_and_stage_runs_meet_floors() {
    if cfg!(debug_assertions) {
        eprintln!(
            "skipping: hudson acceptance needs opt-level 3, run \
             `cargo test --release --test hudson_acceptance -- --nocapture`"
        );
        return;
    }
    let tmp_mc = tempfile::tempdir().unwrap();
    let t0 = Instant::now();
    let mc_dir = run_hudson(Path::new(MC_CONFIG), tmp_mc.path());
    let mc_wall = t0.elapsed().as_secs_f64();
    let tmp_st = tempfile::tempdir().unwrap();
    let t1 = Instant::now();
    let st_dir = run_hudson(Path::new(STAGE_CONFIG), tmp_st.path());
    let st_wall = t1.elapsed().as_secs_f64();

    let (mc, st) = (read_eval(&mc_dir), read_eval(&st_dir));
    let (mc_m, st_m) = (per_gauge(&mc), per_gauge(&st));

    // Baseline per gauge, joined by id (the pseudo-gauge's NSE is null).
    let base = read_json(st_dir.join("baseline/manifest.json"));
    let base_ids: Vec<String> =
        base["gage_ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let base_at = |id: &str, key: &str| -> f64 {
        let i = base_ids.iter().position(|g| g == id).unwrap_or_else(|| panic!("{id} not in baseline"));
        base["metrics"][key][i].as_f64().unwrap_or(f64::NAN)
    };

    for id in [POUGHKEEPSIE, GREEN_ISLAND, CROTON, HARBOR] {
        eprintln!(
            "hudson per-gauge {id}: MC NSE {:.4} KGE {:.4} | stage NSE {:.4} KGE {:.4} | baseline NSE {:.4} KGE {:.4}",
            mc_m[id].0, mc_m[id].1, st_m[id].0, st_m[id].1, base_at(id, "nse"), base_at(id, "kge")
        );
    }
    let (r_exact, r_peak) = spring_neap_ratio(&st, POUGHKEEPSIE);
    let (m_exact, m_peak) = spring_neap_ratio(&mc, POUGHKEEPSIE);
    eprintln!(
        "hudson 14.77 d amplitude ratio model/observed at Poughkeepsie: stage exact {r_exact:.3} \
         band-max {r_peak:.3} | MC exact {m_exact:.3} band-max {m_peak:.3}"
    );
    eprintln!(
        "hudson wall: MC {mc_wall:.1} s, stage {st_wall:.1} s, ratio {:.2}",
        st_wall / mc_wall
    );

    // Gauge counts: 4 gauges, the harbor pseudo-gauge has no observations.
    for (name, dir, m) in [("mc", &mc_dir, &mc_m), ("stage", &st_dir, &st_m)] {
        let manifest = read_json(dir.join("manifest.json"));
        let metrics = &manifest["metrics"];
        assert_eq!(json_u64(metrics, "n_gauges_total"), 4, "{name}: {metrics}");
        assert_eq!(json_u64(metrics, "n_gauges_finite_nse"), 3, "{name}: {metrics}");
        assert!(m[HARBOR].0.is_nan(), "{name}: harbor pseudo-gauge NSE should be NaN");
        for id in [POUGHKEEPSIE, GREEN_ISLAND, CROTON] {
            assert!(m[id].0.is_finite() && m[id].1.is_finite(), "{name}: {id} metrics not finite");
        }
    }

    // Stage run: beats the baseline at Poughkeepsie, keeps Green Island.
    let base_pk = base_at(POUGHKEEPSIE, "nse");
    assert!(
        st_m[POUGHKEEPSIE].0 > base_pk,
        "stage Poughkeepsie NSE {:.4} does not beat baseline {base_pk:.4}",
        st_m[POUGHKEEPSIE].0
    );
    assert!(
        st_m[POUGHKEEPSIE].0 >= STAGE_POUGHKEEPSIE_NSE_FLOOR,
        "stage Poughkeepsie NSE {:.4} < floor {STAGE_POUGHKEEPSIE_NSE_FLOOR}",
        st_m[POUGHKEEPSIE].0
    );
    assert!(
        st_m[POUGHKEEPSIE].1 >= STAGE_POUGHKEEPSIE_KGE_FLOOR,
        "stage Poughkeepsie KGE {:.4} < floor {STAGE_POUGHKEEPSIE_KGE_FLOOR}",
        st_m[POUGHKEEPSIE].1
    );
    assert!(
        st_m[GREEN_ISLAND].0 >= mc_m[GREEN_ISLAND].0 - 0.02,
        "stage Green Island NSE {:.4} < MC {:.4} - 0.02",
        st_m[GREEN_ISLAND].0,
        mc_m[GREEN_ISLAND].0
    );

    // Stage run.log lines.
    let log = std::fs::read_to_string(st_dir.join("run.log")).expect("read run.log");
    for want in [
        "stage routing: 41 of 756 reaches in the stage block (40 added by downstream closure)",
        "stage routing: substeps: 4, theta: 0.6, theta_control: 0.5",
    ] {
        assert!(log.contains(want), "run.log lacks `{want}`:\n{log}");
    }
    let count = |needle: &str| log.lines().filter(|l| l.contains(needle)).count();
    assert_eq!(count("eval: tidal filter: godin at 1 gauge (01372058)"), 1, "eval filter line");
    // One-chunk eval: a chunk seam leaks into the 71 h Godin filter.
    assert!(
        log.lines().any(|l| l.contains("chunk 1/1:")),
        "stage eval did not run as one chunk (expected `chunk 1/1:` in run.log)"
    );
    assert!(!log.contains("chunk 1/2:"), "stage eval ran in more than one chunk");
    assert!(!log.contains("chunk seams leak"), "run.log carries the chunk-seam warning");
    assert!(count("train: tidal filter: godin at 1 gauge (01372058)") >= 1, "train filter line");
    assert!(
        !log.lines().any(|l| l.contains("anchor_elevation_m") && l.contains("overridden")),
        "anchor override line in run.log"
    );
    assert_eq!(count("stage mass balance: no steps"), 0, "stage block took no steps");
    let mb: Vec<&str> = log.lines().filter(|l| l.contains("stage mass balance: rel ")).collect();
    assert_eq!(mb.len(), 1, "expected one mass balance line, got {mb:?}");
    let rel: f32 = mb[0]
        .split("stage mass balance: rel ")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .expect("mass balance number parses as f32");
    assert!(rel <= 1e-4, "stage mass balance rel {rel} > 1e-4");
    let tail = mb[0].split(" over ").nth(1).expect("step count");
    let steps: usize = tail.split_whitespace().next().unwrap().parse().expect("step count parses");
    assert!(steps > 0, "stage block took no steps: {}", mb[0]);
    if let Some(g) = tail.split("gross ").nth(1) {
        let gross: f32 = g.split_whitespace().next().unwrap().parse().expect("gross parses");
        assert!(gross > 0.0, "stage gross flux is zero: {}", mb[0]);
    }

    // MC-only run.log carries no stage lines.
    let mc_log = std::fs::read_to_string(mc_dir.join("run.log")).expect("read MC run.log");
    assert!(
        !mc_log.lines().any(|l| l.contains("stage routing:") || l.contains("stage mass balance:")),
        "MC-only run.log has stage lines"
    );
}
