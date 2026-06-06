//! Smokes the documented first-run flow:
//!   1. write ddrs.yaml (substitute for $EDITOR step)
//!   2. ddrs::cli::init::run_init (skip_smoke=true to keep test fast)
//!   3. ddrs::cli::plan::plan (no --workflow flag — must resolve from yaml)
//!
//! `ddrs run` is NOT exercised end-to-end here — it needs real CONUS data.
//! The pre-flight test in cli_run_preflight covers the workflow=train branch.

use std::fs;
use std::sync::Mutex;

// Serialize chdir-based tests (process global state) — must match
// the lock other chdir tests use if they're in the same binary.
static CHDIR_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn first_run_flow_init_then_plan() {
    let _g = CHDIR_LOCK.lock().unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let proj = tmp.path();
    let cfg_path = proj.join("ddrs.yaml");
    fs::write(&cfg_path, r#"
mode: training
workflow: train-and-test
geodataset: merit
seed: 1
np_seed: 1
data_sources:
  attributes: /dev/null
  conus_adjacency: /dev/null
  gages_adjacency: /dev/null
  streamflow: /dev/null
  observations: /dev/null
  gages: /dev/null
experiment:
  batch_size: 1
  start_time: "2000-01-01"
  end_time: "2000-01-02"
  epochs: 1
  warmup: 1
"#).unwrap();
    let ws_root = proj.join(".ddrs");

    // Patch yaml to point at real (empty) files instead of /dev/null
    // so init can fingerprint them.
    for name in ["attributes", "streamflow", "observations", "gages"] {
        let p = proj.join(format!("{name}.bin"));
        fs::write(&p, b"x").unwrap();
        let s = fs::read_to_string(&cfg_path).unwrap();
        let s = s.replace(&format!("{name}: /dev/null"),
                          &format!("{name}: {}", p.display()));
        fs::write(&cfg_path, s).unwrap();
    }

    // Adjacency keys must point at real zarr store skeletons so plan's
    // up-front layout validation passes (Task 7 — explicit-path branch).
    let conus = proj.join("conus.zarr");
    fs::create_dir_all(&conus).unwrap();
    fs::write(conus.join("zarr.json"), "{}").unwrap();
    for array in ["order", "length_m", "slope", "indices_0", "indices_1"] {
        fs::create_dir_all(conus.join(array)).unwrap();
        fs::write(conus.join(array).join("zarr.json"), "{}").unwrap();
    }
    let gages = proj.join("gages_adj.zarr");
    fs::create_dir_all(&gages).unwrap();
    fs::write(gages.join("zarr.json"), "{}").unwrap();
    for (name, p) in [("conus_adjacency", &conus), ("gages_adjacency", &gages)] {
        let s = fs::read_to_string(&cfg_path).unwrap();
        let s = s.replace(&format!("{name}: /dev/null"),
                          &format!("{name}: {}", p.display()));
        fs::write(&cfg_path, s).unwrap();
    }

    let init_out = ddrs::cli::init::run_init(ddrs::cli::init::InitInput {
        workspace: ws_root.clone(),
        config_path: Some(cfg_path.clone()),
        min_free_gpu_gb: 0.0,
        force: false,
        skip_smoke: true,
    }).expect("init succeeds");
    assert!(init_out.smoke_passed);
    assert!(ws_root.join("sources.lock").is_file());

    let ws = ddrs::cli::workspace::Workspace::with_root(&ws_root);
    let pr = ddrs::cli::plan::plan(&cfg_path, None, &ws)
        .expect("plan resolves workflow from yaml");
    assert_eq!(pr.workflow, ddrs::cli::Workflow::TrainAndTest);
    assert!(pr.drift.is_empty(), "no drift expected on fresh init");
}
