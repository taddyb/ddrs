//! `ddrs import` behavior against the checked-in fixture stores.

use std::fs;
use std::path::{Path, PathBuf};

use ddrs::cli::import::{run_import, ImportInput};
use ddrs::cli::workspace::Workspace;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Minimal parseable ddrs.yaml (mirrors src/cli/sources.rs test CFG).
const CFG: &str = "\
mode: training
geodataset: merit
seed: 1
np_seed: 1
data_sources:
  attributes: /dev/null/attrs.nc
  conus_adjacency: /dev/null/conus.zarr
  gages_adjacency: /dev/null/gages.zarr
  streamflow: /dev/null/sf.ic
  observations: /dev/null/obs.ic
  gages: /dev/null/gages.csv
";

fn setup() -> (tempfile::TempDir, PathBuf, Workspace) {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tmp.path().join("ddrs.yaml");
    fs::write(&cfg, CFG).unwrap();
    let ws = Workspace::with_root(tmp.path().join(".ddrs"));
    (tmp, cfg, ws)
}

#[test]
fn dry_run_validates_without_writing_a_group() {
    let (_tmp, cfg, ws) = setup();
    run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_hourly.ic"),
            name: None,
            dry_run: true,
            force: false,
        },
    )
    .expect("dry-run import of hourly fixture");
    assert!(
        !cfg.parent().unwrap().join("config/sources").exists(),
        "dry-run must not create a group"
    );
}

#[test]
fn import_registers_group_with_swapped_streamflow() {
    let (_tmp, cfg, ws) = setup();
    run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_daily.ic"),
            name: Some("test-daily".into()),
            dry_run: false,
            force: false,
        },
    )
    .expect("import daily fixture");

    let group = cfg.parent().unwrap().join("config/sources/test-daily.yaml");
    let text = fs::read_to_string(&group).expect("group file written");
    assert!(text.contains("qr_daily.ic"), "streamflow swapped: {text}");
    assert!(
        text.contains("observations: /dev/null/obs.ic"),
        "other keys carried over from ddrs.yaml: {text}"
    );
    // Registering again without --force refuses; with force succeeds.
    let again = ImportInput {
        store_path: fixture("qr_daily.ic"),
        name: Some("test-daily".into()),
        dry_run: false,
        force: false,
    };
    assert!(run_import(Some(&cfg), &ws, again).is_err());
    let force = ImportInput {
        store_path: fixture("qr_daily.ic"),
        name: Some("test-daily".into()),
        dry_run: false,
        force: true,
    };
    run_import(Some(&cfg), &ws, force).expect("--force overwrites the existing group");
}

#[test]
fn import_rejects_nonconforming_store() {
    let (_tmp, cfg, ws) = setup();
    let err = run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_minutes.ic"),
            name: None,
            dry_run: true,
            force: false,
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("unsupported time units"),
        "got: {err}"
    );
}

#[test]
fn register_without_name_or_dry_run_is_an_error() {
    let (_tmp, cfg, ws) = setup();
    let err = run_import(
        Some(&cfg),
        &ws,
        ImportInput {
            store_path: fixture("qr_daily.ic"),
            name: None,
            dry_run: false,
            force: false,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("--name"), "got: {err}");
}
