//! Contract test for the committed Juniata sample bundle
//! (`examples/juniata/data/`, mirrored from DeepGroundwater/ddr PR #193).
//!
//! Asserts the bundle through the REAL readers — the same code paths
//! `ddrs plan` / `ddrs run` use — so a bundle regeneration or a reader
//! change that breaks the sample fails `cargo test`, not a fresh user's
//! first run. Unlike the live-store tests, this never skips: the bundle
//! is committed in-repo.

use ddrs::config::Config;
use ddrs::data::{AttrStats, ConusAdjacencyStore, GagesAdjacencyStore, MeritGagesDataset, Staid};

const BUNDLE: &str = "examples/juniata/data";
const GAGE: &str = "01567000";

#[test]
fn conus_adjacency_is_213_reaches_lower_triangular() {
    let store = ConusAdjacencyStore::open(format!("{BUNDLE}/juniata_conus_adjacency.zarr"))
        .expect("open bundle conus adjacency");

    // Re-indexed compact 0..212 space: 213 reaches, 212 edges (a tree).
    assert_eq!(store.n, 213);
    assert_eq!(store.nnz, 212);
    assert_eq!(store.order.len(), store.n);
    assert_eq!(store.length_m.len(), store.n);
    assert_eq!(store.slope.len(), store.n);

    // Topological, lower-triangular: downstream row >= upstream col.
    let violations = store
        .indices_0
        .iter()
        .zip(&store.indices_1)
        .filter(|(r, c)| r < c)
        .count();
    assert_eq!(violations, 0, "{violations} edges violate row >= col");
}

#[test]
fn gage_subgraph_covers_the_whole_bundle() {
    let conus = ConusAdjacencyStore::open(format!("{BUNDLE}/juniata_conus_adjacency.zarr"))
        .expect("open bundle conus adjacency");
    let gages = GagesAdjacencyStore::open(
        format!("{BUNDLE}/juniata_gages_adjacency.zarr"),
        &[Staid::from(GAGE)],
    )
    .expect("open bundle gages adjacency");

    let sub = gages.subgraphs.get(&Staid::from(GAGE)).expect("01567000 subgraph present");
    assert!(!sub.is_headwater(), "Juniata gauge must not be filtered as a headwater");
    // Single-gauge bundle: the upstream set IS the whole 213-reach network.
    assert_eq!(sub.upstream_comids(&conus).len(), 213);
}

#[test]
fn statistics_cover_all_ten_kan_inputs() {
    let stats = AttrStats::open(format!(
        "{BUNDLE}/statistics/merit_attribute_statistics_juniata_attributes.nc.json"
    ))
    .expect("open bundle attribute statistics");

    let vars: Vec<String> = [
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
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let means = stats.means_f32(&vars);
    let stds = stats.stds_f32(&vars);
    for (i, v) in vars.iter().enumerate() {
        assert!(means[i].is_finite(), "mean for {v} not finite");
        assert!(stds[i].is_finite() && stds[i] > 0.0, "std for {v} not usable");
    }
}

/// End-to-end: the shipped example config opens through the full dataset
/// pipeline (attributes + statistics + both icechunk stores + gage CSV +
/// the training filter), exactly as `ddrs run` would.
#[test]
fn example_config_opens_the_dataset() {
    let cfg = Config::from_yaml_file("examples/juniata/ddrs.yaml").expect("load example config");
    let ds = MeritGagesDataset::open(&cfg).expect("open Juniata dataset");
    assert_eq!(ds.len(), 1);
    assert_eq!(ds.staids(), &[Staid::from(GAGE)]);
}
