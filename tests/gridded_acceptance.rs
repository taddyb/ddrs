//! End-to-end routing acceptance on the committed gridded (ISIMIP DDM30)
//! Juniata bundle: full `train-and-test` on CPU through the gridded managed
//! build, then metric floors on the manifest.
//!
//! Reference result (2026-09-09, seed 42, 30 epochs): routed NSE 0.751 /
//! KGE 0.730 vs summed-Q' baseline NSE 0.594 / KGE 0.672. Seeds {42, 7, 123}
//! spanned NSE 0.744–0.758 / KGE 0.728–0.732. DDR's Python trainer reports
//! 0.795 / 0.737 on the same bundle: the gap is the different RNG window
//! stream plus ddrs's parent-space KAN head (one parameter set per cell;
//! DDR varies `log10_uparea` per sub-reach) — see
//! `docs/superpowers/specs/2026-09-08-ddrs-gridded-routing-design.md` §7.
//! The baseline has no RNG and matches DDR's 0.594 / 0.671 to rounding, so it
//! gets a tight band and doubles as the cross-implementation reader check —
//! it is what caught the time-major `Qr(time, divide_id)` layout of DDR's
//! gridded Q' store.
//!
//! Runs only at opt-level 3 (~30 s; a debug build would take minutes):
//!
//!     cargo test --release --test gridded_acceptance -- --nocapture
//!
//! Must run from the repo root (bundle data paths in
//! `examples/juniata_gridded/ddrs.yaml` are repo-root-relative). The
//! workspace goes to a tempdir, so `examples/juniata_gridded/.ddrs/` from
//! manual runs is untouched.

use std::path::PathBuf;

use ddrs::cli::run::{run, RunInput};
use ddrs::cli::workspace::Workspace;

/// Routed-metric floors: below the three-seed range (NSE 0.744–0.758 /
/// KGE 0.728–0.732) by a margin, above the baseline (NSE 0.594 / KGE 0.672).
const NSE_FLOOR: f64 = 0.70;
const KGE_FLOOR: f64 = 0.69;
/// Deterministic summed-Q' NSE (0.594; DDR 0.594).
const BASELINE_NSE_RANGE: (f64, f64) = (0.57, 0.62);

fn json_f64(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("manifest metrics missing numeric `{key}`: {v}"))
}

#[test]
fn gridded_juniata_train_and_test_meets_metric_floors_and_beats_baseline() {
    if cfg!(debug_assertions) {
        eprintln!(
            "skipping: gridded acceptance needs opt-level 3 — \
             run `cargo test --release --test gridded_acceptance -- --nocapture`"
        );
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let run_dir = run(RunInput {
        workspace: Workspace::with_root(tmp.path().join(".ddrs")),
        config_path: PathBuf::from("examples/juniata_gridded/ddrs.yaml"),
        workflow: None, // config says train-and-test
        plot: false,
        strict: false,
        max_mini_batches: None,
        batch_order_from: None,
        backend: "cpu".into(),
    })
    .expect("gridded juniata train-and-test run failed");

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

    let baseline: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("baseline/manifest.json"))
            .expect("read baseline/manifest.json"),
    )
    .expect("parse baseline/manifest.json");
    let baseline_nse = baseline["metrics"]["nse"][0]
        .as_f64()
        .expect("baseline metrics.nse[0] missing or null");

    eprintln!(
        "gridded juniata acceptance: routed NSE {nse:.4} / KGE {kge:.4}, baseline NSE {baseline_nse:.4}"
    );

    assert!(
        (BASELINE_NSE_RANGE.0..=BASELINE_NSE_RANGE.1).contains(&baseline_nse),
        "summed-Q' baseline NSE {baseline_nse:.4} outside {BASELINE_NSE_RANGE:?} — \
         no RNG here, so this is a reader/baseline regression (check the Qr axis \
         order: DDR's gridded store is (time, divide_id)), not noise"
    );
    assert!(nse >= NSE_FLOOR, "routed NSE {nse:.4} < floor {NSE_FLOOR} (seed-42 reference 0.751)");
    assert!(kge >= KGE_FLOOR, "routed KGE {kge:.4} < floor {KGE_FLOOR} (seed-42 reference 0.730)");
    assert!(
        nse > baseline_nse,
        "routed NSE {nse:.4} does not beat the summed-Q' baseline {baseline_nse:.4}"
    );
}
