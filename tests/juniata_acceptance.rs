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
//! Runs only at opt-level 3 (~53 s for the file's three Juniata trainings; a
//! debug build would take minutes):
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
//! that the reservoir changes the routed gauge series. A third trains the
//! bundle with the learned dam release (`reservoir_release: learned`,
//! Raystown Dam as the one release-head dam) and checks its `T0` trains.

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

/// The 19 dam-feature columns of `examples/juniata/data/juniata_dam_features.csv`
/// (`experiments/reservoir/release_head/build_dam_features.py`).
const DAM_FEATURES: [&str; 19] = [
    "log10_storage",
    "log10_storage_max",
    "log10_surface",
    "log10_drainage",
    "log10_storage_per_area",
    "log10_max_discharge",
    "height",
    "year",
    "log10_storage_max_missing",
    "log10_surface_missing",
    "log10_max_discharge_missing",
    "year_missing",
    "purpose_flood",
    "purpose_hydro",
    "purpose_supply",
    "purpose_irrigation",
    "purpose_recreation",
    "purpose_navigation",
    "purpose_other",
];

/// The learned dam release end to end: the bundle with Raystown Dam as a
/// release-head dam (`reservoir_release: learned`), trained jointly with the
/// routing head. The release head must train (its `T0` leaves the 4.5 h
/// init), the test phase must resolve it and write `release_params.csv`, and
/// the gauge series must stay finite.
#[test]
fn juniata_learned_release_trains_t0_and_writes_release_params() {
    if cfg!(debug_assertions) {
        eprintln!(
            "skipping: juniata learned-release run needs opt-level 3 — \
             run `cargo test --release --test juniata_acceptance -- --nocapture`"
        );
        return;
    }
    let mut cfg: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(JUNIATA_CONFIG).unwrap()).unwrap();
    cfg["params"]["use_reservoirs"] = true.into();
    cfg["params"]["reservoir_release"] = "learned".into();
    cfg["data_sources"]["reservoirs"] = "examples/juniata/data/juniata_dam_features.csv".into();
    let mut rh = serde_yaml::Mapping::new();
    rh.insert(
        "input_var_names".into(),
        serde_yaml::Value::Sequence(DAM_FEATURES.iter().map(|s| (*s).into()).collect()),
    );
    cfg["release_head"] = serde_yaml::Value::Mapping(rh);
    let tmp = tempfile::tempdir().unwrap();
    let config = tmp.path().join("ddrs.yaml");
    std::fs::write(&config, serde_yaml::to_string(&cfg).unwrap()).unwrap();
    let run_dir = run_juniata(&config, tmp.path());

    let log = std::fs::read_to_string(run_dir.join("run.log")).expect("read run.log");
    assert!(log.contains("release_T0_median="), "training log lacks the release T0 line");
    assert!(log.contains("reservoirs: 1 of 1 table COMIDs are in the network"), "{log}");

    let csv = std::fs::read_to_string(run_dir.join("release_params.csv"))
        .expect("the test phase writes release_params.csv");
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "COMID,T0_days,a,b,T_min_days,T_max_days");
    assert_eq!(lines.len(), 2, "one dam: {csv}");
    let f: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(f[0], "73005301");
    let t0: f64 = f[1].parse().unwrap();
    let (a, b): (f64, f64) = (f[2].parse().unwrap(), f[3].parse().unwrap());
    let init = 4.5 / 24.0;
    eprintln!(
        "juniata learned release: Raystown T0 {t0:.4} d ({:.2} h, init 4.5 h), a {a:.4}, b {b:.4}",
        t0 * 24.0
    );
    assert!(
        (t0 - init).abs() / init > 0.01,
        "T0 {t0} d did not move off its 4.5 h init: the release head did not train"
    );

    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    let nse = json_f64(&manifest["metrics"], "median_nse_finite");
    let kge = json_f64(&manifest["metrics"], "median_kge_finite");
    eprintln!("juniata learned release: routed NSE {nse:.4} / KGE {kge:.4}");
    assert!(gauge_predictions(&run_dir).iter().all(|v| v.is_finite()));

    // The test phase's per-dam clamp account, and its pooled manifest record.
    let clamp = std::fs::read_to_string(run_dir.join("release_clamp.csv"))
        .expect("the test phase writes release_clamp.csv");
    let lines: Vec<&str> = clamp.lines().collect();
    assert_eq!(
        lines[0],
        "COMID,created_m3,storage_m3,repaid_m3,owed_m3,inflow_m3,created_share,clamp_steps,steps"
    );
    assert_eq!(lines.len(), 2, "one dam: {clamp}");
    let f: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(f[0], "73005301");
    let (inflow, steps): (f64, u64) = (f[5].parse().unwrap(), f[8].parse().unwrap());
    // `forgive` (the default): nothing is repaid or owed.
    assert_eq!((f[3], f[4]), ("0.000000e0", "0.000000e0"), "{clamp}");
    assert!(inflow > 0.0 && steps > 0, "{clamp}");
    let rc = &manifest["metrics"]["release_clamp"];
    assert_eq!(rc["n_dams"].as_u64(), Some(1), "manifest release_clamp: {rc}");
    eprintln!("juniata learned release: release_clamp.csv {}", lines[1]);
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
    // The table itself is logged at dataset open, so train-only runs carry it too.
    assert!(res_log.contains("reservoirs: table "), "run.log lacks the table line:\n{res_log}");
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

    // Dam rows write the test phase's clamp account; the control has none.
    assert!(res_dir.join("release_clamp.csv").is_file(), "reservoir run lacks release_clamp.csv");
    assert!(!plain_dir.join("release_clamp.csv").exists(), "control run wrote release_clamp.csv");

    // The in-engine replay recipe: resume at the run's LAST checkpoint with the
    // same epochs. No optimizer step is left, so Phase 1 writes no checkpoint and
    // the test phase evaluates the resumed one: the gauge series must be the
    // reservoir run's bit for bit.
    let mut ckpts: Vec<(usize, usize, PathBuf)> = std::fs::read_dir(res_dir.join("checkpoints"))
        .expect("read checkpoints")
        .filter_map(|e| {
            let p = e.ok()?.path();
            let name = p.file_name()?.to_str()?.to_string();
            let rest = name.strip_prefix("epoch_")?;
            let (ep, mb) = rest.split_once("_mb_")?;
            Some((ep.parse().ok()?, mb.parse().ok()?, p))
        })
        .collect();
    ckpts.sort();
    let (_, _, last) = ckpts.last().expect("the reservoir run wrote checkpoints").clone();
    let mut replay = cfg.clone();
    replay["experiment"]["checkpoint"] = last.display().to_string().into();
    let tmp_rep = tempfile::tempdir().unwrap();
    let rep_config = tmp_rep.path().join("ddrs.yaml");
    std::fs::write(&rep_config, serde_yaml::to_string(&replay).unwrap()).unwrap();
    let rep_dir = run_juniata(&rep_config, tmp_rep.path());
    let rep_log = std::fs::read_to_string(rep_dir.join("run.log")).expect("read replay run.log");
    assert!(
        rep_log.contains("Phase 1 took no optimizer step and wrote no checkpoint; testing the resumed checkpoint"),
        "replay run.log lacks the zero-step line:\n{rep_log}"
    );
    assert!(!rep_log.contains("  mb="), "the replay took a training step:\n{rep_log}");
    let rep = gauge_predictions(&rep_dir);
    assert!(
        rep.iter().zip(&gauge_predictions(&res_dir)).all(|(a, b)| a.to_bits() == b.to_bits()),
        "the zero-step replay does not reproduce the reservoir run's test phase"
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
