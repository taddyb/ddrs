//! Integration tests for SP-1 static-data readers, exercised against the
//! production files in `~/projects/ddr/`.
//!
//! Each test path-checks the production file and skips with an eprintln if
//! absent — so CI on a clean machine still passes. On the dev machine
//! (where the files exist) the assertions are load-bearing.

use std::path::Path;

use ddrs::data::ids::Staid;
use ddrs::data::{AttrStats, AttributesStore, ConusAdjacencyStore, GageMetadata};

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

const ATTRS_NC: &str =
    "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc";
const CONUS_ADJ: &str =
    "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr";

#[test]
fn attributes_store_opens_against_conus_subset() {
    if !Path::new(ATTRS_NC).exists() || !Path::new(CONUS_ADJ).exists() {
        eprintln!("skipping: production data files not present");
        return;
    }
    let conus = ConusAdjacencyStore::open(CONUS_ADJ).expect("conus adj");
    let comids: Vec<_> = conus.order.iter().take(500).copied().collect();
    let attr_names = vec![
        "SoilGrids1km_clay".to_string(),
        "aridity".to_string(),
        "meanelevation".to_string(),
        "meanP".to_string(),
        "NDVI".to_string(),
        "meanslope".to_string(),
        "log10_uparea".to_string(),
        "SoilGrids1km_sand".to_string(),
        "ETPOT_Hargr".to_string(),
        "Porosity".to_string(),
    ];

    let store =
        AttributesStore::open(ATTRS_NC, &attr_names, &comids).expect("open attrs");

    assert_eq!(store.attr_names.len(), 10);
    assert_eq!(store.attrs.shape()[0], 10);
    assert!(store.attrs.shape()[1] > 0);
    assert!(store.attrs.shape()[1] <= 500);
    for &m in store.row_means.iter() {
        assert!(m.is_finite(), "row_mean unexpectedly non-finite: {m}");
    }
    let first = *store.index.ids().first().expect("at least one COMID present");
    assert_eq!(store.index.position(&first), Some(0));
}

const STATS_JSON: &str =
    "/home/tbindas/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json";

#[test]
fn attr_stats_open_against_production_json() {
    if !Path::new(STATS_JSON).exists() {
        eprintln!("skipping: {STATS_JSON} not present");
        return;
    }
    let s = AttrStats::open(STATS_JSON).expect("open stats json");
    for name in [
        "SoilGrids1km_clay",
        "aridity",
        "meanelevation",
        "meanP",
        "NDVI",
        "meanslope",
        "log10_uparea",
        "SoilGrids1km_sand",
        "ETPOT_Hargr",
        "Porosity",
    ] {
        assert!(s.by_name.contains_key(name), "missing {name}");
    }
    let clay = &s.by_name["SoilGrids1km_clay"];
    assert!((clay.mean - 23.494225_f64).abs() < 1e-6);
    assert!((clay.std - 8.221468_f64).abs() < 1e-6);
}
