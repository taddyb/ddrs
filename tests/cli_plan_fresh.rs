//! Merged `plan` initializes a fresh workspace itself — no separate `init`.
//! Covers spec §9 tests 1 (fresh dir) and the smoke/lock artifacts.

use ddrs::cli::plan::{plan, PlanInput};
use ddrs::cli::workspace::Workspace;
use std::fs;
use std::path::{Path, PathBuf};

/// Minimal valid config in `dir`: real (1-byte) data files + zarr store
/// skeletons that pass plan's up-front layout validation. Mirrors the
/// fixture in cli_first_run_e2e.rs.
pub fn write_fixture_config(dir: &Path) -> PathBuf {
    let cfg_path = dir.join("ddrs.yaml");
    let mut yaml = String::from(
        "mode: training\nworkflow: train-and-test\ngeodataset: merit\nseed: 1\nnp_seed: 1\ndata_sources:\n",
    );
    for name in ["attributes", "streamflow", "observations", "gages"] {
        let p = dir.join(format!("{name}.bin"));
        fs::write(&p, b"x").unwrap();
        yaml.push_str(&format!("  {name}: {}\n", p.display()));
    }
    let conus = dir.join("conus.zarr");
    fs::create_dir_all(&conus).unwrap();
    fs::write(conus.join("zarr.json"), "{}").unwrap();
    for array in ["order", "length_m", "slope", "indices_0", "indices_1"] {
        fs::create_dir_all(conus.join(array)).unwrap();
        fs::write(conus.join(array).join("zarr.json"), "{}").unwrap();
    }
    let gages = dir.join("gages_adj.zarr");
    fs::create_dir_all(&gages).unwrap();
    fs::write(gages.join("zarr.json"), "{}").unwrap();
    yaml.push_str(&format!("  conus_adjacency: {}\n", conus.display()));
    yaml.push_str(&format!("  gages_adjacency: {}\n", gages.display()));
    yaml.push_str(
        "experiment:\n  batch_size: 1\n  start_time: \"2000-01-01\"\n  end_time: \"2000-01-02\"\n  epochs: 1\n  warmup: 1\n",
    );
    fs::write(&cfg_path, yaml).unwrap();
    cfg_path
}

#[test]
fn plan_initializes_fresh_workspace() {
    let d = tempfile::tempdir().unwrap();
    let cfg = write_fixture_config(d.path());
    let ws = Workspace::with_root(d.path().join(".ddrs"));
    let pr = plan(
        PlanInput { config_path: Some(cfg), skip_smoke: true, ..Default::default() },
        &ws,
    )
    .expect("plan must initialize a fresh workspace and succeed");
    assert!(ws.root().join("version").is_file());
    assert!(ws.system_json().is_file());
    assert!(ws.lockfile().is_file(), "first plan writes the lock");
    assert!(pr.drift.is_empty(), "no prior lock → no drift");
}
