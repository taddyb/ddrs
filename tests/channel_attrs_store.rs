//! Phase-A deliverable gate: merit_channel_attributes_v1.nc must open through
//! the SAME AttributesStore reader as merit_global_attributes_v2.nc (spec
//! §3: identical schema, zero reader changes).
//!
//! Path-gated: skips gracefully if the file is absent (before Phase A runs).
//! Exercises the exact code path MeritGagesDataset::open uses — AttributesStore::open
//! with attr_names + comids — and fails if any of the seven Phase-A variables
//! is missing from the file.

use std::path::Path;

use ddrs::data::{AttributesStore, Comid};

const CHANNEL_NC: &str = "/home/tbindas/projects/ddr/data/merit_channel_attributes_v1.nc";

#[test]
fn channel_attributes_open_through_attributes_store() {
    if !Path::new(CHANNEL_NC).exists() {
        eprintln!("skipping: {CHANNEL_NC} not present (Phase A not yet run)");
        return;
    }

    let attr_names: Vec<String> = [
        "channel_wtd_bed_rel",
        "losing_fraction",
        "corridor_impervious",
        "alluvium_fraction",
        "bfi",
        "drainage_density",
        "bankfull_depth",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    // A handful of real CONUS MERIT COMIDs (from bankfull.parquet, 156k reaches).
    // At least one must resolve to a finite value in the channel attrs file.
    let comids: Vec<Comid> = [71039327i64, 71039450, 71039451, 71039463, 71039472]
        .iter()
        .map(|&c| Comid(c))
        .collect();

    let store = AttributesStore::open(CHANNEL_NC, &attr_names, &comids)
        .expect("opens like global.nc — all seven Phase-A variables must be present");

    assert_eq!(store.attr_names.len(), 7, "all 7 Phase-A variable names resolved");
    assert!(
        store.attrs.shape()[1] > 0,
        "expected ≥1 CONUS COMID to resolve in the channel attrs file"
    );
    // Every resolved value should be finite (these COMIDs are in bankfull.parquet coverage).
    let all_finite = store.attrs.iter().all(|v| v.is_finite());
    assert!(all_finite, "resolved attr values should be finite for covered COMIDs");
}
