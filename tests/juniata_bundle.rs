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

/// `geometry: table` is checked once at dataset open: a table that lacks one
/// stage reach fails, naming the COMID; a full table opens.
#[test]
fn table_geometry_coverage_is_checked_at_open() {
    use ddrs::config::{StageGeometry, StageRouting};

    let conus = ConusAdjacencyStore::open(format!("{BUNDLE}/juniata_conus_adjacency.zarr"))
        .expect("open bundle conus adjacency");
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, skip: Option<usize>| {
        let mut body = String::from("COMID,top_width_m,bed_z_m,n\n");
        for (i, c) in conus.order.iter().enumerate() {
            if Some(i) != skip {
                body.push_str(&format!("{},50.0,0.0,0.035\n", c.0));
            }
        }
        let p = dir.path().join(name);
        std::fs::write(&p, body).unwrap();
        p
    };
    let open = |path: std::path::PathBuf| {
        let mut cfg = Config::from_yaml_file("examples/juniata/ddrs.yaml").expect("load config");
        cfg.params.stage_routing = Some(StageRouting {
            enabled: true,
            geometry: StageGeometry::Table,
            ..StageRouting::default()
        });
        cfg.data_sources.as_mut().unwrap().channel_geometry = Some(path);
        MeritGagesDataset::open(&cfg)
    };

    let missing = conus.order[100];
    let err = open(write("gap.csv", Some(100))).err().expect("a gap must fail at open");
    let msg = err.to_string();
    assert!(msg.contains(&missing.0.to_string()), "error should name COMID {}: {msg}", missing.0);
    assert!(msg.contains("gap.csv"), "{msg}");

    let ds = open(write("full.csv", None)).expect("a full table opens");
    assert_eq!(ds.len(), 1);
}

/// `reaches: [COMID list]` seeds the open-time coverage check from the listed
/// COMIDs only. The table needs a row for the seed and everything downstream of
/// it, and nothing else.
#[test]
fn table_geometry_coverage_follows_a_comid_list() {
    use ddrs::config::{StageGeometry, StageReaches, StageRouting};

    let conus = ConusAdjacencyStore::open(format!("{BUNDLE}/juniata_conus_adjacency.zarr"))
        .expect("open bundle conus adjacency");
    // Topological order: the last reach has nothing downstream, so its stage set is itself.
    let seed = *conus.order.last().expect("non-empty network");
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, only: Option<i64>| {
        let mut body = String::from("COMID,top_width_m,bed_z_m,n\n");
        for c in &conus.order {
            if only.is_none_or(|o| o != c.0) {
                body.push_str(&format!("{},50.0,0.0,0.035\n", c.0));
            }
        }
        let p = dir.path().join(name);
        std::fs::write(&p, body).unwrap();
        p
    };
    let open = |path: std::path::PathBuf| {
        let mut cfg = Config::from_yaml_file("examples/juniata/ddrs.yaml").expect("load config");
        cfg.params.stage_routing = Some(StageRouting {
            enabled: true,
            geometry: StageGeometry::Table,
            reaches: StageReaches::Comids(vec![seed.0]),
            ..StageRouting::default()
        });
        cfg.data_sources.as_mut().unwrap().channel_geometry = Some(path);
        MeritGagesDataset::open(&cfg)
    };

    // A table that lacks the seed fails and names it.
    let err = open(write("no_seed.csv", Some(seed.0))).err().expect("a missing seed row must fail");
    assert!(err.to_string().contains(&seed.0.to_string()), "{err}");

    // A table holding only the seed row opens: no other reach is in the set.
    let mut body = String::from("COMID,top_width_m,bed_z_m,n\n");
    body.push_str(&format!("{},50.0,0.0,0.035\n", seed.0));
    let p = dir.path().join("seed_only.csv");
    std::fs::write(&p, body).unwrap();
    let ds = open(p).expect("a table covering the seed set opens");
    assert_eq!(ds.len(), 1);
}
