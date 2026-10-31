//! End-to-end acceptance of stage routing on the committed Juniata bundle:
//! full `train-and-test` on CPU with the `stage:` block on
//! (`examples/juniata/ddrs_stage.yaml`), then metric floors on the manifest.
//!
//! Reference (2026-10-31, seed 42): NSE 0.7928 / KGE 0.8808 vs baseline NSE 0.6947,
//! wall 853 s on CPU (MC: about 21 s).
//!
//! Release-only (a debug train takes minutes). The test greps `run.log` for the
//! `stage routing:` lines, which go to stderr, so it needs `--nocapture`: plain
//! `cargo test` captures `eprintln!` in-process and the fd-level tee
//! (`cli::tee`) never sees it.
//!
//!     cargo test --release --test juniata_stage_acceptance -- --nocapture
//!
//! Must run from the repo root. The workspace goes to a tempdir.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ddrs::cli::run::{run, RunInput};
use ddrs::cli::workspace::Workspace;

const STAGE_CONFIG: &str = "examples/juniata/ddrs_stage.yaml";

/// Floors: measured NSE/KGE minus 0.04 (the margin `juniata_acceptance` uses).
const NSE_FLOOR: f64 = 0.75;
const KGE_FLOOR: f64 = 0.84;
/// Same baseline band as `juniata_acceptance`.
const BASELINE_NSE_RANGE: (f64, f64) = (0.67, 0.72);

/// WET copy of `run_juniata` in `tests/juniata_acceptance.rs` (ruling C33: keep
/// parallel copies). CPU `train-and-test` of `config` into a fresh workspace.
fn run_juniata(config: &Path, tmp: &Path) -> PathBuf {
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
    .expect("juniata stage train-and-test run failed")
}

fn json_f64(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("manifest metrics missing numeric `{key}`: {v}"))
}

#[test]
fn juniata_stage_train_and_test_beats_the_baseline() {
    if cfg!(debug_assertions) {
        eprintln!(
            "skipping: juniata stage acceptance needs opt-level 3, run \
             `cargo test --release --test juniata_stage_acceptance -- --nocapture`"
        );
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let t0 = Instant::now();
    let run_dir = run_juniata(Path::new(STAGE_CONFIG), tmp.path());
    let wall = t0.elapsed().as_secs_f64();

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    let metrics = &manifest["metrics"];
    assert_eq!(metrics["n_gauges_total"].as_u64(), Some(1), "{metrics}");
    let nse = json_f64(metrics, "median_nse_finite");
    let kge = json_f64(metrics, "median_kge_finite");
    assert!(nse.is_finite() && kge.is_finite(), "non-finite metrics: {metrics}");

    let baseline: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("baseline/manifest.json"))
            .expect("read baseline/manifest.json"),
    )
    .expect("parse baseline/manifest.json");
    let baseline_nse = baseline["metrics"]["nse"][0].as_f64().expect("baseline nse[0]");

    eprintln!(
        "juniata stage acceptance: routed NSE {nse:.4} / KGE {kge:.4}, baseline NSE {baseline_nse:.4}, wall {wall:.1} s"
    );

    assert!(
        (BASELINE_NSE_RANGE.0..=BASELINE_NSE_RANGE.1).contains(&baseline_nse),
        "baseline NSE {baseline_nse:.4} outside {BASELINE_NSE_RANGE:?}"
    );
    assert!(nse > baseline_nse, "stage NSE {nse:.4} does not beat baseline {baseline_nse:.4}");
    assert!(nse >= NSE_FLOOR, "stage NSE {nse:.4} < floor {NSE_FLOOR}");
    assert!(kge >= KGE_FLOOR, "stage KGE {kge:.4} < floor {KGE_FLOOR}");

    let log = std::fs::read_to_string(run_dir.join("run.log")).expect("read run.log");
    assert!(
        log.contains("stage routing: 213 of 213 reaches in the stage block (0 added by downstream closure)"),
        "run.log lacks the stage block line:\n{log}"
    );
    assert!(
        log.contains("stage routing: substeps: 1, theta: 1, theta_control: 0.5"),
        "run.log lacks the stage settings line:\n{log}"
    );
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
}
