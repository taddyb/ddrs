//! Manual measurement of `params.geometry_form` on the committed Juniata and
//! Hudson bundles. It records numbers; it gates nothing beyond the log lines.
//!
//! Ignored, release-only, local. Run from the repo root:
//!
//!     cargo test --release --test geometry_form_measure -- --ignored --nocapture
//!
//! All runs share ONE workspace, `output/geometry-form/.ddrs` (gitignored), so
//! later probes can read the checkpoints. The run ids go to
//! `output/geometry-form/RUN_IDS.txt`. That file is appended to, so remove it
//! before a rerun.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ddrs::cli::run::{run, RunInput};
use ddrs::cli::workspace::Workspace;
use ddrs::training::Metrics;
use ndarray::{s, Array2};

const OUT: &str = "output/geometry-form";
const JUNIATA: &str = "examples/juniata/ddrs.yaml";
const HUDSON_MC: &str = "examples/hudson/ddrs_mc_only.yaml";
const HUDSON_STAGE: &str = "examples/hudson/ddrs.yaml";
const GAUGES: [&str; 3] = ["01372058", "01358000", "01375000"];
/// `experiment.warmup` in the Hudson configs.
const WARMUP: usize = 5;

const INFO_LINE: &str = "params.geometry_form: power_law (one power-law section";
const WARN_LINE: &str = "params.geometry_form: power_law applies to Muskingum-Cunge rows only";

/// WET copy of `RUN_LOCK` in `tests/juniata_acceptance.rs`. `cli::tee`
/// redirects the process-wide fds, so runs must not overlap.
static RUN_LOCK: Mutex<()> = Mutex::new(());

/// Copy `src` to `dst` with `  geometry_form: <form>` right after the single
/// `params:` line.
fn with_form(src: &Path, form: &str, dst: &Path) {
    let text = std::fs::read_to_string(src).unwrap();
    assert_eq!(text.lines().filter(|l| *l == "params:").count(), 1, "{src:?}: one params: line");
    let mut out = String::new();
    for l in text.lines() {
        out.push_str(l);
        out.push('\n');
        if l == "params:" {
            out.push_str(&format!("  geometry_form: {form}\n"));
        }
    }
    std::fs::write(dst, out).unwrap();
}

fn run_one(config: &Path, plot: bool) -> PathBuf {
    let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run(RunInput {
        workspace: Workspace::with_root(PathBuf::from(OUT).join(".ddrs")),
        config_path: config.to_path_buf(),
        workflow: None,
        plot,
        strict: false,
        max_mini_batches: None,
        batch_order_from: None,
        backend: "cpu".into(),
    })
    .expect("train-and-test run failed")
}

/// One run per (bundle, form). Returns the run dir.
fn run_form(src: &str, tag: &str, form: &str, plot: bool) -> PathBuf {
    let dir = PathBuf::from(OUT).join("configs");
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = if form == "trapezoid" {
        PathBuf::from(src)
    } else {
        let p = dir.join(format!("{tag}_{form}.yaml"));
        with_form(Path::new(src), form, &p);
        p
    };
    let d = run_one(&cfg, plot);
    eprintln!("RUN {tag} {form}: {}", d.display());
    let id = d.file_name().unwrap().to_string_lossy().to_string();
    let mut ids = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(PathBuf::from(OUT).join("RUN_IDS.txt"))
        .unwrap();
    use std::io::Write;
    writeln!(ids, "{tag} {form} {id}").unwrap();
    d
}

fn read_json(p: PathBuf) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}")))
        .expect("parse json")
}

fn log_checks(dir: &Path, form: &str, want_warn: bool) {
    let log = std::fs::read_to_string(dir.join("run.log")).expect("read run.log");
    let has_info = log.contains(INFO_LINE);
    if form == "power_law" {
        assert!(has_info, "{dir:?}: power_law run.log lacks the INFO line");
        assert_eq!(log.contains(WARN_LINE), want_warn, "{dir:?}: WARN line presence");
    } else {
        assert!(!has_info, "{dir:?}: trapezoid run.log has the INFO line");
        assert!(!log.contains(WARN_LINE), "{dir:?}: trapezoid run.log has the WARN line");
    }
}

#[test]
#[ignore]
fn measure_juniata() {
    if cfg!(debug_assertions) {
        eprintln!("skipping: needs opt-level 3, run with --release");
        return;
    }
    for form in ["trapezoid", "power_law"] {
        let d = run_form(JUNIATA, "juniata", form, true);
        log_checks(&d, form, false);
        let m = read_json(d.join("manifest.json"));
        let (nse, kge) = (
            m["metrics"]["median_nse_finite"].as_f64().unwrap(),
            m["metrics"]["median_kge_finite"].as_f64().unwrap(),
        );
        let b = read_json(d.join("baseline/manifest.json"));
        eprintln!(
            "MEASURE juniata {form}: NSE {nse:.4} KGE {kge:.4} | baseline NSE {:.4} KGE {:.4}",
            b["metrics"]["nse"][0].as_f64().unwrap_or(f64::NAN),
            b["metrics"]["kge"][0].as_f64().unwrap_or(f64::NAN)
        );
    }
}

struct Eval {
    ids: Vec<String>,
    pred: Array2<f32>,
    obs: Array2<f32>,
}

/// WET copy of `read_eval` in `tests/hudson_acceptance.rs`.
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

/// WET copy of `per_gauge` in `tests/hudson_acceptance.rs`.
fn per_gauge(e: &Eval) -> HashMap<String, (f64, f64)> {
    let m = Metrics::compute(
        &e.pred.slice(s![.., WARMUP..]).to_owned(),
        &e.obs.slice(s![.., WARMUP..]).to_owned(),
    );
    e.ids.iter().enumerate().map(|(i, id)| (id.clone(), (m.nse[i] as f64, m.kge[i] as f64))).collect()
}

#[test]
#[ignore]
fn measure_hudson() {
    if cfg!(debug_assertions) {
        eprintln!("skipping: needs opt-level 3, run with --release");
        return;
    }
    for (src, tag, stage) in [(HUDSON_MC, "hudson_mc", false), (HUDSON_STAGE, "hudson_stage", true)] {
        for form in ["trapezoid", "power_law"] {
            let d = run_form(src, tag, form, false);
            log_checks(&d, form, stage && form == "power_law");
            let m = per_gauge(&read_eval(&d));
            let b = read_json(d.join("baseline/manifest.json"));
            let ids: Vec<String> = b["gage_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            for g in GAUGES {
                let i = ids.iter().position(|x| x == g).expect("gauge in baseline");
                eprintln!(
                    "MEASURE {tag} {form} {g}: NSE {:.4} KGE {:.4} | baseline NSE {:.4} KGE {:.4}",
                    m[g].0,
                    m[g].1,
                    b["metrics"]["nse"][i].as_f64().unwrap_or(f64::NAN),
                    b["metrics"]["kge"][i].as_f64().unwrap_or(f64::NAN)
                );
            }
        }
    }
}
