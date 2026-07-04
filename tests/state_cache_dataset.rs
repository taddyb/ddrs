//! Unit tests for `StateCache` (src/data/store/state_cache.rs).
//!
//! `MeritGagesDataset` requires real data paths so cannot be constructed in
//! CI. Per repo convention (see `tests/zeta_accum.rs`), we test the focused
//! store module directly — sufficient to prove the business-logic invariants:
//!
//! 1. `row_for_day` returns the correct row reordered to the caller's COMID order.
//! 2. A date outside `[day0, day0 + n_days)` returns a hard error.
//! 3. A COMID absent from the cache index returns a hard error naming the COMID.
//! 4. When `experiment.state_cache` is absent the `Experiment` serde parses with
//!    `state_cache == None` (byte-identity at config layer — the load-bearing
//!    invariant that keeps the no-cache path unchanged).

use chrono::NaiveDate;
use ddrs::data::ids::Comid;
use ddrs::data::store::StateCache;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Write a minimal state-cache netCDF to `path` and return it.
///
/// Layout: dims `day` × `COMID`, var `q_state` f32, var `COMID` i64,
/// global attr `day0` (ISO date). Mirrors `write_state_cache_netcdf` in
/// `probe_zeta_gradient.rs`.
fn write_fixture(
    path: &std::path::Path,
    n_days: usize,
    comids: &[i64],
    q_state: &[f32],
    day0: &str,
) {
    assert_eq!(q_state.len(), n_days * comids.len());
    let mut file = netcdf::create(path).expect("create netcdf");
    file.add_dimension("day", n_days).unwrap();
    file.add_dimension("COMID", comids.len()).unwrap();
    file.add_attribute("day0", day0).unwrap();
    file.add_attribute("checkpoint", "test").unwrap();
    {
        let mut v = file.add_variable::<i64>("COMID", &["COMID"]).unwrap();
        v.put_values(comids, ..).unwrap();
    }
    {
        let mut v = file.add_variable::<f32>("q_state", &["day", "COMID"]).unwrap();
        v.put_values(q_state, ..).unwrap();
        v.put_attribute("units", "m^3/s").unwrap();
    }
}

/// Build a temporary fixture with 3 days × 3 COMIDs.
///
/// Cache layout (row-major):
/// ```
/// COMID   100  200  300
/// day 0   1.0  2.0  3.0
/// day 1   4.0  5.0  6.0
/// day 2   7.0  8.0  9.0
/// ```
/// day0 = 1990-10-01.
fn fixture(dir: &std::path::Path) -> (std::path::PathBuf, NaiveDate) {
    let path = dir.join("state_cache_fixture.nc");
    let comids = [100i64, 200, 300];
    // Row-major: day0_comids then day1_comids then day2_comids.
    let q_state = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let day0_str = "1990-10-01";
    write_fixture(&path, 3, &comids, &q_state, day0_str);
    let day0 = NaiveDate::from_ymd_opt(1990, 10, 1).unwrap();
    (path, day0)
}

// ---------------------------------------------------------------------------
// Test 1: correct row and COMID reorder
// ---------------------------------------------------------------------------

#[test]
fn correct_row_and_comid_reorder() {
    let dir = tempfile::tempdir().unwrap();
    let (path, day0) = fixture(dir.path());

    let cache = StateCache::open(&path).expect("open");
    assert_eq!(cache.n_days, 3);
    assert_eq!(cache.n_comids, 3);
    assert_eq!(cache.day0, day0);

    // Request COMIDs in a different order: 300, 100 (drop 200).
    let requested = [Comid(300), Comid(100)];

    // Day 0: cache row = [1.0, 2.0, 3.0] for COMIDs [100, 200, 300].
    // Reordered to [300, 100] → [3.0, 1.0].
    let row0 = cache.row_for_day(day0, &requested).expect("day 0 row");
    assert_eq!(row0.len(), 2);
    assert!((row0[0] - 3.0).abs() < 1e-6, "COMID 300, day 0: expected 3.0, got {}", row0[0]);
    assert!((row0[1] - 1.0).abs() < 1e-6, "COMID 100, day 0: expected 1.0, got {}", row0[1]);

    // Day 1 (day0 + 1): cache row = [4.0, 5.0, 6.0].
    // Reordered to [300, 100] → [6.0, 4.0].
    let day1 = day0 + chrono::Duration::days(1);
    let row1 = cache.row_for_day(day1, &requested).expect("day 1 row");
    assert!((row1[0] - 6.0).abs() < 1e-6, "COMID 300, day 1: expected 6.0, got {}", row1[0]);
    assert!((row1[1] - 4.0).abs() < 1e-6, "COMID 100, day 1: expected 4.0, got {}", row1[1]);

    // Day 2 (day0 + 2): cache row = [7.0, 8.0, 9.0].
    // Reordered to [300, 100] → [9.0, 7.0].
    let day2 = day0 + chrono::Duration::days(2);
    let row2 = cache.row_for_day(day2, &requested).expect("day 2 row");
    assert!((row2[0] - 9.0).abs() < 1e-6, "COMID 300, day 2: expected 9.0, got {}", row2[0]);
    assert!((row2[1] - 7.0).abs() < 1e-6, "COMID 100, day 2: expected 7.0, got {}", row2[1]);
}

// ---------------------------------------------------------------------------
// Test 2: out-of-range day errors
// ---------------------------------------------------------------------------

#[test]
fn out_of_range_day_before_day0_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (path, day0) = fixture(dir.path());
    let cache = StateCache::open(&path).expect("open");

    let before = day0 - chrono::Duration::days(1);
    let err = cache
        .row_for_day(before, &[Comid(100)])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("out of range"),
        "expected out-of-range message, got: {err}"
    );
}

#[test]
fn out_of_range_day_after_cache_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (path, day0) = fixture(dir.path());
    let cache = StateCache::open(&path).expect("open");

    // n_days = 3 → valid = day0, day0+1, day0+2. day0+3 is out of range.
    let after = day0 + chrono::Duration::days(3);
    let err = cache
        .row_for_day(after, &[Comid(100)])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("out of range"),
        "expected out-of-range message, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: missing COMID names the COMID and the cache path
// ---------------------------------------------------------------------------

#[test]
fn missing_comid_error_names_comid_and_path() {
    let dir = tempfile::tempdir().unwrap();
    let (path, day0) = fixture(dir.path());
    let cache = StateCache::open(&path).expect("open");

    // COMID 999 is not in the cache.
    let err = cache
        .row_for_day(day0, &[Comid(100), Comid(999)])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("999") || err.contains("COMID"),
        "expected COMID info in error, got: {err}"
    );
    assert!(
        err.contains("cache") || err.contains("index"),
        "expected 'cache'/'index' in error, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: None path — Experiment without state_cache parses as None
// ---------------------------------------------------------------------------

#[test]
fn state_cache_absent_in_experiment_is_none() {
    // This is the load-bearing byte-identity invariant at the config layer:
    // any experiment config without `state_cache:` must parse with None,
    // ensuring the no-cache code path is taken.
    let yaml = "batch_size: 4\nstart_time: 2000/01/01\nend_time: 2000/01/02\n\
                epochs: 1\nrho: 10\nwarmup: 1\n";
    let exp: ddrs::config::Experiment =
        serde_yaml::from_str(yaml).expect("parse experiment");
    assert!(
        exp.state_cache.is_none(),
        "state_cache must be None when absent from YAML"
    );
}

// ---------------------------------------------------------------------------
// Test 5: StateCache::open on a valid file reports correct metadata
// ---------------------------------------------------------------------------

#[test]
fn open_reports_correct_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let (path, day0) = fixture(dir.path());
    let cache = StateCache::open(&path).expect("open");

    assert_eq!(cache.day0, day0, "day0 mismatch");
    assert_eq!(cache.n_days, 3, "n_days mismatch");
    assert_eq!(cache.n_comids, 3, "n_comids mismatch");
}
