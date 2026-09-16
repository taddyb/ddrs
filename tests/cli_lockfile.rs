use ddrs::cli::lockfile::{Lockfile, diff_against_live};
use ddrs::cli::fingerprint::Fingerprint;
use std::collections::BTreeMap;
use std::fs;

fn fp(path: &str, fp: &str) -> Fingerprint {
    Fingerprint {
        path: path.into(), mtime: "2026-05-30T00:00:00Z".into(),
        size: 1, fp: fp.into(), snapshot: None,
    }
}

#[test]
fn lockfile_round_trips() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("sources.lock");
    let mut sources = BTreeMap::new();
    sources.insert("attributes".into(), fp("/x", "blake3:aaa"));
    let lock = Lockfile { ddrs_version: "0.1.0".into(),
        created_at: "2026-05-30T00:00:00Z".into(), sources };
    lock.write_atomic(&p).unwrap();
    let loaded = Lockfile::read(&p).unwrap();
    assert_eq!(loaded, lock);
}

#[test]
fn diff_lists_drifted_keys() {
    let mut sources = BTreeMap::new();
    sources.insert("attributes".into(), fp("/x", "blake3:aaa"));
    sources.insert("conus_adjacency".into(), fp("/y", "blake3:bbb"));
    let lock = Lockfile { ddrs_version: "x".into(),
        created_at: "x".into(), sources };
    let mut live = BTreeMap::new();
    live.insert("attributes".into(), fp("/x", "blake3:aaa"));         // unchanged
    live.insert("conus_adjacency".into(), fp("/y", "blake3:CHANGED"));// drifted
    let drift = diff_against_live(&lock, &live);
    assert_eq!(drift, vec!["conus_adjacency".to_string()]);
}

// ── icechunk snapshots in the lock (pins plan, Task 2) ──────────────────────

fn repo_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Juniata-bundle config with absolute paths, so `plan` needs no chdir.
fn write_juniata_config(dir: &std::path::Path, pins: &str) -> std::path::PathBuf {
    let b = |n: &str| repo_path(&format!("examples/juniata/data/{n}"));
    let cfg = dir.join("ddrs.yaml");
    fs::write(
        &cfg,
        format!(
            "mode: training\nworkflow: train\ngeodataset: merit\nseed: 1\nnp_seed: 1\n\
             data_sources:\n\
             \x20 attributes: {}\n\
             \x20 conus_adjacency: {}\n\
             \x20 gages_adjacency: {}\n\
             \x20 streamflow: {}\n\
             \x20 observations: {}\n\
             \x20 gages: {}\n{pins}\
             experiment:\n\
             \x20 batch_size: 1\n  start_time: 1981/10/01\n  end_time: 1981/12/31\n\
             \x20 epochs: 1\n  rho: 10\n  warmup: 1\n",
            b("juniata_attributes.nc").display(),
            b("juniata_conus_adjacency.zarr").display(),
            b("juniata_gages_adjacency.zarr").display(),
            b("juniata_qprime.ic").display(),
            b("juniata_obs.ic").display(),
            b("juniata_gage.csv").display(),
        ),
    )
    .unwrap();
    cfg
}

fn plan_into_tmp(pins: &str) -> (tempfile::TempDir, Lockfile) {
    let d = tempfile::tempdir().unwrap();
    let cfg = write_juniata_config(d.path(), pins);
    let ws = ddrs::cli::workspace::Workspace::with_root(d.path().join(".ddrs"));
    ddrs::cli::plan::plan(
        ddrs::cli::plan::PlanInput {
            config_path: Some(cfg),
            skip_smoke: true,
            ..Default::default()
        },
        &ws,
    )
    .expect("plan on the Juniata bundle must succeed");
    let lock = Lockfile::read(&ws.lockfile()).unwrap();
    (d, lock)
}

#[test]
fn plan_locks_icechunk_sources_with_their_snapshot_ids() {
    let (_d, lock) = plan_into_tmp("");

    for (key, store) in [
        ("streamflow", "juniata_qprime.ic"),
        ("observations", "juniata_obs.ic"),
    ] {
        let tip = ddrs::data::store::icechunk::main_branch_snapshot(&repo_path(&format!(
            "examples/juniata/data/{store}"
        )))
        .unwrap();
        let e = lock.sources.get(key).expect("locked");
        assert_eq!(e.snapshot.as_deref(), Some(tip.as_str()), "{key} snapshot");
        assert_eq!(e.fp, format!("icechunk:{tip}"), "{key} fp");
    }

    // Non-icechunk sources stay snapshot-less.
    assert!(lock.sources["attributes"].snapshot.is_none());
    assert!(lock.sources["conus_adjacency"].snapshot.is_none());
}

#[test]
fn a_pin_is_what_gets_locked() {
    let qp = repo_path("examples/juniata/data/juniata_qprime.ic");
    let tip = ddrs::data::store::icechunk::main_branch_snapshot(&qp).unwrap();
    let (_d, lock) = plan_into_tmp(&format!("  pins:\n    streamflow: {tip}\n"));
    let e = &lock.sources["streamflow"];
    assert_eq!(e.snapshot.as_deref(), Some(tip.as_str()));
    assert_eq!(e.fp, format!("icechunk:{tip}"));
}
