//! End-to-end routing acceptance on the committed Juniata bundle: full
//! `train-and-test` on CPU, then metric floors on the manifest.
//!
//! This is the only test that verifies the whole data → train → route →
//! eval → metric chain. Reference result (2026-08-19, seed 42, 30 epochs,
//! corrected physics): routed NSE 0.790 / KGE 0.881 vs summed-Q' baseline
//! NSE 0.695. Floors below are deliberately looser than the deterministic
//! seed-42 result so legitimate op-reordering noise doesn't trip them;
//! anything under them is a real regression.
//!
//! Runs only at opt-level 3 (~25 s; a debug build would take minutes):
//!
//!     cargo test --release --test juniata_acceptance -- --nocapture
//!
//! Must run from the repo root (bundle data paths in
//! `examples/juniata/ddrs.yaml` are repo-root-relative — cargo test's
//! default CWD). The workspace goes to a tempdir, so the sanctioned
//! `examples/juniata/.ddrs/` from manual runs is untouched.
//!
//! A second test runs the bundle again with `params.use_reservoirs: true` and
//! the committed Raystown Lake table (`examples/juniata/data/
//! juniata_reservoirs.csv`, option C of `.claude/RESERVOIRS.md`) and checks
//! that the reservoir changes the routed gauge series.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ddrs::cli::run::{run, RunInput};
use ddrs::cli::workspace::Workspace;

const JUNIATA_CONFIG: &str = "examples/juniata/ddrs.yaml";

/// `run` tees the process-global fds 1/2 into each run's `run.log`
/// (`cli::tee`), and libtest runs this file's tests on parallel threads, so
/// runs must not overlap.
static RUN_LOCK: Mutex<()> = Mutex::new(());

/// The harness: CPU `train-and-test` of `config` (its `workflow:`) into a
/// fresh workspace under `tmp`, returning the run directory.
fn run_juniata(config: &Path, tmp: &Path) -> PathBuf {
    let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run(RunInput {
        workspace: Workspace::with_root(tmp.join(".ddrs")),
        config_path: config.to_path_buf(),
        workflow: None, // config says train-and-test
        plot: false,
        strict: false,
        max_mini_batches: None,
        batch_order_from: None,
        backend: "cpu".into(),
    })
    .expect("juniata train-and-test run failed")
}

/// Routed-metric floors: below the seed-42 result (NSE 0.790 / KGE 0.881)
/// by a margin, above the baseline (NSE 0.695).
const NSE_FLOOR: f64 = 0.75;
const KGE_FLOOR: f64 = 0.80;
/// Baseline window: the summed-Q' NSE is 0.695 and deterministic (no RNG).
/// It doubles as a cross-implementation reader check (matches DDR's Python
/// readers to rounding), so it gets a tight band.
const BASELINE_NSE_RANGE: (f64, f64) = (0.67, 0.72);

fn json_f64(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("manifest metrics missing numeric `{key}`: {v}"))
}

#[test]
fn juniata_train_and_test_meets_metric_floors_and_beats_baseline() {
    if cfg!(debug_assertions) {
        eprintln!(
            "skipping: juniata acceptance needs opt-level 3 — \
             run `cargo test --release --test juniata_acceptance -- --nocapture`"
        );
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let run_dir = run_juniata(Path::new(JUNIATA_CONFIG), tmp.path());

    // Routed metrics from the run manifest.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    let metrics = &manifest["metrics"];
    assert_eq!(
        metrics["n_gauges_total"].as_u64(),
        Some(1),
        "expected the single Juniata gauge, got: {metrics}"
    );
    let nse = json_f64(metrics, "median_nse_finite");
    let kge = json_f64(metrics, "median_kge_finite");

    // Baseline NSE from the copy `run` places in <run_dir>/baseline/.
    let baseline: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("baseline/manifest.json"))
            .expect("read baseline/manifest.json — baseline copy is informational in `run` but mandatory here"),
    )
    .expect("parse baseline/manifest.json");
    let baseline_nse = baseline["metrics"]["nse"][0]
        .as_f64()
        .expect("baseline metrics.nse[0] missing or null");

    eprintln!(
        "juniata acceptance: routed NSE {nse:.4} / KGE {kge:.4}, baseline NSE {baseline_nse:.4}"
    );

    assert!(
        (BASELINE_NSE_RANGE.0..=BASELINE_NSE_RANGE.1).contains(&baseline_nse),
        "summed-Q' baseline NSE {baseline_nse:.4} outside {BASELINE_NSE_RANGE:?} — \
         the baseline has no RNG, so this is a data-reader or baseline regression, not noise"
    );
    assert!(
        nse >= NSE_FLOOR,
        "routed NSE {nse:.4} < floor {NSE_FLOOR} (seed-42 reference 0.790)"
    );
    assert!(
        kge >= KGE_FLOOR,
        "routed KGE {kge:.4} < floor {KGE_FLOOR} (seed-42 reference 0.881)"
    );
    assert!(
        nse > baseline_nse,
        "routed NSE {nse:.4} does not beat the summed-Q' baseline {baseline_nse:.4} — \
         the routing isn't earning its keep"
    );
}

/// Daily routed predictions at the gauge, `eval/predictions.zarr` `/predictions`.
fn gauge_predictions(run_dir: &Path) -> Vec<f64> {
    use zarrs::array::Array;
    use zarrs::filesystem::FilesystemStore;
    let store = std::sync::Arc::new(
        FilesystemStore::new(run_dir.join("eval/predictions.zarr")).expect("open predictions.zarr"),
    );
    let arr = Array::open(store, "/predictions").expect("open /predictions");
    assert_eq!(arr.shape()[0], 1, "expected the single Juniata gauge");
    arr.retrieve_array_subset::<Vec<f64>>(&arr.subset_all()).expect("read /predictions")
}

/// Raystown Lake (COMID 73005301) sits upstream of the Newport gauge. Routed
/// as a linear reservoir it must be found in the network, logged once, and
/// change the gauge series; the same bundle without the flag is the control.
#[test]
fn juniata_reservoir_is_matched_logged_and_changes_the_gauge_series() {
    if cfg!(debug_assertions) {
        eprintln!(
            "skipping: juniata reservoir run needs opt-level 3 — \
             run `cargo test --release --test juniata_acceptance -- --nocapture`"
        );
        return;
    }

    // Control: the bundle as committed.
    let tmp_plain = tempfile::tempdir().unwrap();
    let plain_dir = run_juniata(Path::new(JUNIATA_CONFIG), tmp_plain.path());

    // Same config plus the two reservoir keys.
    let mut cfg: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(JUNIATA_CONFIG).unwrap()).unwrap();
    cfg["params"]["use_reservoirs"] = true.into();
    cfg["data_sources"]["reservoirs"] = "examples/juniata/data/juniata_reservoirs.csv".into();
    let tmp_res = tempfile::tempdir().unwrap();
    let res_config = tmp_res.path().join("ddrs.yaml");
    std::fs::write(&res_config, serde_yaml::to_string(&cfg).unwrap()).unwrap();
    let res_dir = run_juniata(&res_config, tmp_res.path());

    const MATCH_LINE: &str = "reservoirs: 1 of 1 table COMIDs are in the network";
    const MATCH_TAIL: &str = "table COMIDs are in the network";
    let res_log = std::fs::read_to_string(res_dir.join("run.log")).expect("read reservoir run.log");
    assert!(res_log.contains(MATCH_LINE), "run.log lacks {MATCH_LINE:?}:\n{res_log}");
    let n_lines = res_log.lines().filter(|l| l.contains(MATCH_TAIL)).count();
    assert_eq!(n_lines, 1, "the match must be logged once, not per batch");
    let plain_log = std::fs::read_to_string(plain_dir.join("run.log")).expect("read control run.log");
    assert!(!plain_log.contains(MATCH_TAIL), "control run logged a reservoir match");

    // The table is fingerprinted with the other data sources.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(res_dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    assert!(
        manifest["sources"]["reservoirs"].is_object(),
        "manifest sources lack `reservoirs`: {}",
        manifest["sources"]
    );

    let plain = gauge_predictions(&plain_dir);
    let res = gauge_predictions(&res_dir);
    assert_eq!(plain.len(), res.len());
    assert!(res.iter().all(|v| v.is_finite()), "reservoir run has non-finite predictions");
    let n_diff = plain.iter().zip(&res).filter(|(a, b)| a != b).count();
    let max_abs = plain.iter().zip(&res).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    eprintln!(
        "juniata reservoir: {n_diff} of {} gauge days differ, max |diff| {max_abs:.3} m3/s",
        plain.len()
    );
    assert!(n_diff > 0, "the reservoir did not change the routed gauge series");
}
