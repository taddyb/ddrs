//! Smokes the documented lifecycle: `plan → plan (idempotent) → drift →
//! relock → strict`. Covers spec §9 tests 2 (idempotency), 3 (drift +
//! relock), and 4 (strict preserves the lock).
//!
//! `ddrs run` is NOT exercised end-to-end here — it needs real CONUS data.

use ddrs::cli::plan::{plan, PlanInput};
use ddrs::cli::workspace::Workspace;
use std::fs;
use std::path::{Path, PathBuf};

// Same fixture as tests/cli_plan_fresh.rs (integration tests are separate
// crates; the ~30 lines are duplicated rather than reshaping tests/common.rs,
// which is routing-focused).
fn write_fixture_config(dir: &Path) -> PathBuf {
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

fn plan_input(cfg: &Path) -> PlanInput {
    PlanInput {
        config_path: Some(cfg.to_path_buf()),
        skip_smoke: true,
        ..Default::default()
    }
}

#[test]
fn plan_lifecycle_idempotent_drift_relock_strict() {
    let d = tempfile::tempdir().unwrap();
    let cfg = write_fixture_config(d.path());
    let ws = Workspace::with_root(d.path().join(".ddrs"));

    // 1. Fresh plan: initializes + locks.
    let pr = plan(plan_input(&cfg), &ws).expect("fresh plan succeeds");
    assert_eq!(pr.workflow, ddrs::cli::Workflow::TrainAndTest);
    assert!(pr.drift.is_empty());
    let lock_1 = fs::read_to_string(ws.lockfile()).unwrap();

    // 2. Idempotency: second plan → no drift, lock byte-identical.
    let pr2 = plan(plan_input(&cfg), &ws).expect("second plan succeeds");
    assert!(pr2.drift.is_empty());
    let lock_2 = fs::read_to_string(ws.lockfile()).unwrap();
    assert_eq!(lock_1, lock_2, "unchanged sources must not rewrite the lock");

    // 3. Drift + auto-relock: mutate a source, plan reports + relocks.
    fs::write(d.path().join("gages.bin"), b"yy").unwrap();
    let pr3 = plan(plan_input(&cfg), &ws).expect("drifted plan still succeeds");
    assert_eq!(pr3.drift, vec!["gages".to_string()]);
    let lock_3 = fs::read_to_string(ws.lockfile()).unwrap();
    assert_ne!(lock_2, lock_3, "drift must refresh the lock");

    // 4. Post-relock: drift is gone.
    let pr4 = plan(plan_input(&cfg), &ws).expect("post-relock plan succeeds");
    assert!(pr4.drift.is_empty());

    // 5. Strict aborts BEFORE relocking (evidence preserved).
    fs::write(d.path().join("gages.bin"), b"zzz").unwrap();
    let err = plan(
        PlanInput { strict: true, ..plan_input(&cfg) },
        &ws,
    )
    .unwrap_err();
    assert!(
        matches!(err, ddrs::cli::CliError::LockDrift { .. }),
        "expected LockDrift, got: {err:?}"
    );
    let lock_5 = fs::read_to_string(ws.lockfile()).unwrap();
    assert_eq!(lock_3, lock_5, "strict abort must leave the lock untouched");
}
