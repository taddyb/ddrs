//! Integration tests for SP-1 static-data readers, exercised against the
//! production files in `~/projects/ddr/`.
//!
//! Each test path-checks the production file and skips with an eprintln if
//! absent — so CI on a clean machine still passes. On the dev machine
//! (where the files exist) the assertions are load-bearing.

use std::path::Path;

use ddrs::data::GageMetadata;
use ddrs::data::ids::Staid;

const GAGES_CSV: &str =
    "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";

#[test]
fn gages_3000_loads_with_expected_shape() {
    if !Path::new(GAGES_CSV).exists() {
        eprintln!("skipping: {GAGES_CSV} not present");
        return;
    }
    let m = GageMetadata::open(GAGES_CSV).expect("open gages_3000.csv");
    assert_eq!(m.rows.len(), 3211);
    assert_eq!(m.rows[0].staid.as_str(), "14190500");
    assert!((m.rows[0].drain_sqkm - 603.4942).abs() < 1e-6);
    assert_eq!(m.rows[0].da_valid, Some(true));
    assert!(m.by_staid.contains_key(&Staid::new("14190500")));
}
