//! Pinned-snapshot opens of the two icechunk-backed stores
//! (`data_sources.pins`, Task 2 of the pins plan).
//!
//! Uses the committed Juniata bundle (`examples/juniata/data/`), so this
//! test never skips — the same reason `tests/juniata_bundle.rs` uses it.

use ddrs::data::store::icechunk::main_branch_snapshot;
use ddrs::data::store::{ObservationsStore, StreamflowSource, StreamflowStore, UsgsObservationsStore};

/// A syntactically valid `SnapshotId` (12 bytes → 20 Crockford base32 chars)
/// that no repository will ever hold: icechunk's all-zero `FAKE` id.
const ABSENT_SNAPSHOT: &str = "00000000000000000000";

fn bundle(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/juniata/data")
        .join(name)
}

#[test]
fn streamflow_pinned_at_main_tip_reads_the_same_values_as_unpinned() {
    let p = bundle("juniata_qprime.ic");
    let tip = main_branch_snapshot(&p).expect("resolve main tip");

    let unpinned = StreamflowStore::open(&p).expect("open");
    let pinned = StreamflowStore::open_at(&p, Some(&tip)).expect("open_at main tip");

    assert_eq!(pinned.snapshot, tip, "store records the snapshot it opened");
    assert_eq!(unpinned.snapshot, tip, "unpinned open resolves main to the same tip");

    let comids: Vec<_> = unpinned.index.ids()[..4].to_vec();
    let a = unpinned
        .read_window_daily(unpinned.time_start, 5, &comids)
        .expect("read unpinned");
    let b = pinned
        .read_window_daily(pinned.time_start, 5, &comids)
        .expect("read pinned");
    assert_eq!(a, b, "a pin at the main tip must read byte-identical values");
}

#[test]
fn streamflow_pinned_at_an_absent_snapshot_errors_naming_path_and_id() {
    let p = bundle("juniata_qprime.ic");
    let err = match StreamflowStore::open_at(&p, Some(ABSENT_SNAPSHOT)) {
        Err(e) => e,
        Ok(_) => panic!("an absent snapshot id must not open"),
    };
    let msg = format!("{err}");
    assert!(msg.contains("juniata_qprime.ic"), "error must name the source path: {msg}");
    assert!(msg.contains(ABSENT_SNAPSHOT), "error must name the snapshot id: {msg}");
}

#[test]
fn streamflow_pinned_at_a_malformed_snapshot_errors_naming_path_and_id() {
    let p = bundle("juniata_qprime.ic");
    let err = match StreamflowStore::open_at(&p, Some("not-a-snapshot-id")) {
        Err(e) => e,
        Ok(_) => panic!("a malformed snapshot id must not open"),
    };
    let msg = format!("{err}");
    assert!(msg.contains("juniata_qprime.ic"), "error must name the source path: {msg}");
    assert!(msg.contains("not-a-snapshot-id"), "error must name the snapshot id: {msg}");
}

#[test]
fn observations_pinned_at_main_tip_reads_the_same_values_as_unpinned() {
    let p = bundle("juniata_obs.ic");
    let tip = main_branch_snapshot(&p).expect("resolve main tip");

    let unpinned = UsgsObservationsStore::open(&p).expect("open");
    let pinned = UsgsObservationsStore::open_at(&p, Some(&tip)).expect("open_at main tip");
    assert_eq!(pinned.snapshot, tip);

    let staids: Vec<_> = unpinned.index.ids().to_vec();
    let a = unpinned
        .read_window_daily(unpinned.time_start, 5, &staids)
        .expect("read unpinned");
    let b = pinned
        .read_window_daily(pinned.time_start, 5, &staids)
        .expect("read pinned");
    assert_eq!(a, b);
}

#[test]
fn observations_pinned_at_an_absent_snapshot_errors_naming_path_and_id() {
    let p = bundle("juniata_obs.ic");
    let err = match UsgsObservationsStore::open_at(&p, Some(ABSENT_SNAPSHOT)) {
        Err(e) => e,
        Ok(_) => panic!("an absent snapshot id must not open"),
    };
    let msg = format!("{err}");
    assert!(msg.contains("juniata_obs.ic"), "error must name the source path: {msg}");
    assert!(msg.contains(ABSENT_SNAPSHOT), "error must name the snapshot id: {msg}");
}

#[test]
fn format_dispatching_wrappers_thread_the_pin_and_report_the_snapshot() {
    let qp = bundle("juniata_qprime.ic");
    let ob = bundle("juniata_obs.ic");
    let qp_tip = main_branch_snapshot(&qp).unwrap();
    let ob_tip = main_branch_snapshot(&ob).unwrap();

    let sf = StreamflowSource::open_at(&qp, Some(&qp_tip)).expect("open_at streamflow");
    assert_eq!(sf.snapshot(), Some(qp_tip.as_str()));

    let obs = ObservationsStore::open_at(&ob, Some(&ob_tip)).expect("open_at observations");
    assert_eq!(obs.snapshot(), Some(ob_tip.as_str()));

    // Unpinned goes through the same accessor and still reports the tip.
    assert_eq!(
        StreamflowSource::open(&qp).unwrap().snapshot(),
        Some(qp_tip.as_str())
    );

    let err = match StreamflowSource::open_at(&qp, Some(ABSENT_SNAPSHOT)) {
        Err(e) => e,
        Ok(_) => panic!("an absent snapshot id must not open"),
    };
    assert!(format!("{err}").contains(ABSENT_SNAPSHOT));
}

/// End-to-end: `data_sources.pins` reaches the two stores through
/// `Config` → `MeritGagesDataset::open`. Run with `--nocapture` to see the
/// `streamflow snapshot: <id> (pinned)` line this exercises.
#[test]
fn a_pinned_config_opens_the_juniata_dataset() {
    use ddrs::config::Config;
    use ddrs::data::MeritGagesDataset;

    let tip_q = main_branch_snapshot(&bundle("juniata_qprime.ic")).unwrap();
    let tip_o = main_branch_snapshot(&bundle("juniata_obs.ic")).unwrap();

    // Integration tests run with the package root as cwd, so the example
    // config's repo-root-relative data paths resolve unchanged.
    let anchor = "  gages: examples/juniata/data/juniata_gage.csv";
    let src = std::fs::read_to_string("examples/juniata/ddrs.yaml").unwrap();
    assert!(src.contains(anchor), "example config layout changed");
    let pinned = src.replace(
        anchor,
        &format!("{anchor}\n  pins:\n    streamflow: {tip_q}\n    observations: {tip_o}"),
    );

    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("pinned.yaml");
    std::fs::write(&p, pinned).unwrap();

    let cfg = Config::from_yaml_file(&p).expect("pinned config loads");
    let ds = MeritGagesDataset::open(&cfg).expect("open the dataset at pinned snapshots");
    assert_eq!(ds.len(), 1);
}
