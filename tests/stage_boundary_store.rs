//! Hourly stage boundary reader (`data_sources.stage_boundary`).

use std::path::PathBuf;

use chrono::{Duration, NaiveDate, NaiveDateTime};
use ddrs::data::error::DataError;
use ddrs::data::store::StageBoundaryStore;

const FMT: &str = "%Y-%m-%dT%H:%M:%SZ";

fn t0(s: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(s, FMT).unwrap()
}

fn write_csv(dir: &tempfile::TempDir, body: &str) -> PathBuf {
    let p = dir.path().join("stage.csv");
    std::fs::write(&p, body).unwrap();
    p
}

/// `n` hourly rows starting at `start`; value = hour index.
fn hourly_csv(start: &str, n: usize) -> String {
    let s = t0(start);
    let mut out = String::from("time,stage_m\n");
    for i in 0..n {
        out.push_str(&format!(
            "{},{}\n",
            (s + Duration::hours(i as i64)).format(FMT),
            i as f32
        ));
    }
    out
}

#[test]
fn stage_boundary_reads_hourly_csv() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_csv(
        &dir,
        "time,stage_m\n2000-01-01T00:00:00Z,0.12\n2000-01-01T01:00:00Z,0.25\n2000-01-01T02:00:00Z,-0.5\n",
    );
    let store = StageBoundaryStore::open(&p).unwrap();
    assert_eq!(store.path, p);
    let v = store.slice(t0("2000-01-01T00:00:00Z"), 3).unwrap();
    assert_eq!(v, vec![0.12, 0.25, -0.5]);
    let v = store.slice(t0("2000-01-01T01:00:00Z"), 2).unwrap();
    assert_eq!(v, vec![0.25, -0.5]);
}

#[test]
fn stage_boundary_rejects_other_time_formats() {
    let dir = tempfile::tempdir().unwrap();
    for bad in [
        "2000-01-01 00:00",
        "2000-01-01T00:00:00",
        "2000-01-01T00:00:00+00:00",
    ] {
        let p = write_csv(
            &dir,
            &format!("time,stage_m\n2000-01-01T00:00:00Z,0.1\n{bad},0.2\n"),
        );
        let err = StageBoundaryStore::open(&p).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(p.to_str().unwrap()), "{msg}");
        assert!(msg.contains("row 2"), "{msg}");
        assert!(msg.contains("%Y-%m-%dT%H:%M:%SZ"), "{msg}");
    }
    // The first row is checked too.
    let p = write_csv(&dir, "time,stage_m\n2000-01-01 00:00,0.1\n");
    let msg = StageBoundaryStore::open(&p).unwrap_err().to_string();
    assert!(msg.contains("row 1") && msg.contains("%Y-%m-%dT%H:%M:%SZ"), "{msg}");
}

#[test]
fn stage_boundary_rejects_nan_and_short_series() {
    let dir = tempfile::tempdir().unwrap();
    for bad in ["NaN", "inf", "abc", ""] {
        let p = write_csv(
            &dir,
            &format!("time,stage_m\n2000-01-01T00:00:00Z,0.1\n2000-01-01T01:00:00Z,{bad}\n"),
        );
        let err = StageBoundaryStore::open(&p).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(p.to_str().unwrap()), "{msg}");
        assert!(msg.contains("row 2"), "{msg}");
    }
    // Empty file body.
    let p = write_csv(&dir, "time,stage_m\n");
    assert!(StageBoundaryStore::open(&p).is_err());

    let p = write_csv(&dir, &hourly_csv("2000-01-01T00:00:00Z", 10));
    let store = StageBoundaryStore::open(&p).unwrap();
    let err = store.slice(t0("2000-01-01T05:00:00Z"), 6).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, DataError::Malformed { .. }));
    assert!(msg.contains(p.to_str().unwrap()), "{msg}");
    assert!(msg.contains("2000-01-01T05:00:00Z"), "{msg}");
    assert!(msg.contains("6 hours"), "{msg}");
    // Before the record.
    assert!(store.slice(t0("1999-12-31T23:00:00Z"), 2).is_err());
}

#[test]
fn stage_boundary_rejects_gaps() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_csv(
        &dir,
        "time,stage_m\n2000-01-01T00:00:00Z,0.1\n2000-01-01T01:00:00Z,0.2\n2000-01-01T03:00:00Z,0.3\n",
    );
    let msg = StageBoundaryStore::open(&p).unwrap_err().to_string();
    assert!(msg.contains("not contiguous"), "{msg}");
    assert!(msg.contains(p.to_str().unwrap()), "{msg}");
    assert!(msg.contains("row 3"), "{msg}");
    // Duplicate and backwards rows are also not +1 h.
    let p = write_csv(
        &dir,
        "time,stage_m\n2000-01-01T00:00:00Z,0.1\n2000-01-01T00:00:00Z,0.2\n",
    );
    assert!(StageBoundaryStore::open(&p)
        .unwrap_err()
        .to_string()
        .contains("not contiguous"));
}

#[test]
fn stage_boundary_slice_starts_at_window_midnight_utc() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_csv(&dir, &hourly_csv("1999-12-31T22:00:00Z", 40));
    let store = StageBoundaryStore::open(&p).unwrap();
    let midnight = NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    assert_eq!(store.slice(midnight, 3).unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn stage_boundary_slice_has_one_value_per_q_prime_row() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_csv(&dir, &hourly_csv("2000-01-01T00:00:00Z", 400));
    let store = StageBoundaryStore::open(&p).unwrap();
    let start = t0("2000-01-01T00:00:00Z");
    for t in [2usize, 25, 361] {
        assert_eq!(store.slice(start, t).unwrap().len(), t);
    }
    // The store accepts one value; the stage block rejects it (H steps need H + 1).
    assert_eq!(store.slice(start, 1).unwrap().len(), 1);
}
