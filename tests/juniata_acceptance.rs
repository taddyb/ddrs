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

use std::path::PathBuf;

use ddrs::cli::run::{run, RunInput};
use ddrs::cli::workspace::Workspace;

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
    let run_dir = run(RunInput {
        workspace: Workspace::with_root(tmp.path().join(".ddrs")),
        config_path: PathBuf::from("examples/juniata/ddrs.yaml"),
        workflow: None, // config says train-and-test
        plot: false,
        strict: false,
        max_mini_batches: None,
        batch_order_from: None,
        backend: "cpu".into(),
    })
    .expect("juniata train-and-test run failed");

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
