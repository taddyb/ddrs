//! Contract test for the committed gridded (ISIMIP DDM30) Juniata bundle
//! (`examples/juniata_gridded/data/`, byte-identical to DeepGroundwater/ddr
//! PR #194's `examples/juniata_gridded/data/`).
//!
//! Exercises the whole gridded ingestion path through the REAL code:
//! `GriddedNetwork` reads DDR's sub-reach zarr, the managed build relabels
//! it into ddrs's subdivided layout and cuts the gauge subgraph, and
//! `MeritGagesDataset::open` accepts the result. Never skips — the bundle
//! is committed in-repo.

use ddrs::adjacency::cache::resolve_or_build_gridded;
use ddrs::adjacency::gridded::GriddedNetwork;
use ddrs::config::Config;
use ddrs::data::{
    AttrStats, ConusAdjacencyStore, GageMetadata, GagesAdjacencyStore, MeritGagesDataset, Staid,
};

const BUNDLE: &str = "examples/juniata_gridded/data";
const CONFIG: &str = "examples/juniata_gridded/ddrs.yaml";
const GAGE: &str = "01567000";
/// The four DDM30 cells of the Juniata network, upstream first, and the
/// number of sub-reaches DDR cut each into (`length_m` ≈ 7.1–7.8 km).
const CELLS: [(i32, usize); 4] = [(139163, 9), (138443, 6), (138444, 6), (138445, 6)];
/// The gauge's snapped cell (`juniata_gage.csv`'s `cell` column).
const GAGE_CELL: i64 = 138445;

#[test]
fn subreach_store_reads_as_27_rows_over_4_cells() {
    let g = GriddedNetwork::open(format!("{BUNDLE}/juniata_subreach_adjacency.zarr"))
        .expect("open bundle sub-reach store");
    assert_eq!(g.n(), 27);
    assert_eq!(g.n_cells(), 4);
    assert_eq!(g.rows.len(), 26, "a 27-node tree has 26 edges");

    let sub = g.to_subdivided();
    let cells: Vec<i32> = CELLS.iter().map(|c| c.0).collect();
    assert_eq!(sub.parent_order, cells);
    for (p, &(cell, pieces)) in CELLS.iter().enumerate() {
        assert_eq!(sub.pieces(p), pieces, "cell {cell} piece count");
        // Every piece of a cell carries the cell id in `order`.
        for r in sub.inlet(p)..=sub.outlet(p) {
            assert_eq!(sub.order[r], cell);
        }
    }
    // DDR's `length_m` is the cell flow length / pieces: all pieces of a cell
    // are equal, and the cell is ~7 km per piece.
    for p in 0..4 {
        let l0 = sub.length_m[sub.inlet(p)];
        assert!((7000.0..8000.0).contains(&l0), "piece length {l0} m");
        for r in sub.inlet(p)..=sub.outlet(p) {
            assert_eq!(sub.length_m[r], l0);
        }
    }
}

#[test]
fn gauge_csv_cell_column_names_the_outlet_cell() {
    let meta = GageMetadata::open(format!("{BUNDLE}/juniata_gage.csv")).expect("open gauge csv");
    assert_eq!(meta.rows.len(), 1);
    assert_eq!(meta.rows[0].staid, Staid::from(GAGE));
    assert_eq!(meta.rows[0].comid, Some(GAGE_CELL));
    // No FLOW_SCALE / COMID_DRAIN_SQKM: the gauge reach's inflow scale stays 1,
    // matching DDR's train_gridded.py (flow_scale=None).
    assert!(meta.rows[0].flow_scale.is_none());
}

/// The managed build into a scratch workspace: the CONUS store round-trips
/// through `ConusAdjacencyStore` as a subdivided store, and the gauge is read
/// at cell 138445's most downstream sub-reach (the last row).
#[test]
fn managed_build_cuts_the_gauge_at_the_cells_outlet_piece() {
    let tmp = tempfile::tempdir().unwrap();
    let out = resolve_or_build_gridded(
        tmp.path(),
        std::path::Path::new(&format!("{BUNDLE}/juniata_subreach_adjacency.zarr")),
        std::path::Path::new(&format!("{BUNDLE}/juniata_gage.csv")),
    )
    .expect("gridded managed build");
    assert!(!out.cache_hit);
    assert!(out.paths.conus.ends_with("juniata_subreach_adjacency.zarr"));
    assert!(out.paths.gages.ends_with("juniata_subreach_gages_adjacency.zarr"));

    let conus = ConusAdjacencyStore::open(&out.paths.conus).expect("open built conus store");
    assert_eq!(conus.n, 27);
    assert_eq!(conus.nnz, 26);
    assert_eq!(conus.n_parent(), 4);
    assert_eq!(conus.parent_offset, vec![0, 9, 15, 21, 27]);
    let violations = conus
        .indices_0
        .iter()
        .zip(&conus.indices_1)
        .filter(|(r, c)| r <= c)
        .count();
    assert_eq!(violations, 0, "{violations} edges are not strictly lower-triangular");

    let gages = GagesAdjacencyStore::open(&out.paths.gages, &[Staid::from(GAGE)])
        .expect("open built gages store");
    let sg = gages.subgraphs.get(&Staid::from(GAGE)).expect("01567000 subgraph present");
    assert!(!sg.is_headwater());
    assert_eq!(sg.gage_catchment, GAGE_CELL.to_string());
    // Outlet piece of the last cell = row 26 (DDR's `outlet_of_cell`).
    assert_eq!(sg.gage_idx, 26);
    // Single-gauge bundle: the upstream set is every cell, each once (the
    // baseline sums a cell's Q' once, not once per piece).
    let upstream = sg.upstream_comids(&conus);
    assert_eq!(upstream.len(), 4);

    // Second resolve is a cache hit with the same key.
    let again = resolve_or_build_gridded(
        tmp.path(),
        std::path::Path::new(&format!("{BUNDLE}/juniata_subreach_adjacency.zarr")),
        std::path::Path::new(&format!("{BUNDLE}/juniata_gage.csv")),
    )
    .expect("second resolve");
    assert!(again.cache_hit);
    assert_eq!(again.key, out.key);
}

#[test]
fn statistics_cover_all_ten_kan_inputs() {
    let stats = AttrStats::open(format!(
        "{BUNDLE}/statistics/merit_attribute_statistics_ddm30_conus_attributes.nc.json"
    ))
    .expect("open bundle attribute statistics");
    let cfg = Config::from_yaml_file(CONFIG).expect("load example config");
    let vars = cfg.kan_head.as_ref().unwrap().input_var_names.clone();
    assert_eq!(vars.len(), 10);
    let means = stats.means_f32(&vars);
    let stds = stats.stds_f32(&vars);
    for (i, v) in vars.iter().enumerate() {
        assert!(means[i].is_finite(), "mean for {v} not finite");
        assert!(stds[i].is_finite() && stds[i] > 0.0, "std for {v} not usable");
    }
}

/// End-to-end: the shipped config resolves its adjacency through the gridded
/// managed build and opens through the full dataset pipeline, exactly as
/// `ddrs run` would.
#[test]
fn example_config_opens_the_dataset() {
    let mut cfg = Config::from_yaml_file(CONFIG).expect("load example config");
    let ds = cfg.data_sources.as_mut().unwrap();
    assert!(ds.gridded_network.is_some());
    let tmp = tempfile::tempdir().unwrap();
    let out = resolve_or_build_gridded(
        tmp.path(),
        ds.gridded_network.as_ref().unwrap(),
        &ds.gages,
    )
    .expect("gridded managed build");
    ds.conus_adjacency = Some(out.paths.conus);
    ds.gages_adjacency = Some(out.paths.gages);

    let dataset = MeritGagesDataset::open(&cfg).expect("open gridded Juniata dataset");
    assert_eq!(dataset.len(), 1);
    assert_eq!(dataset.staids(), &[Staid::from(GAGE)]);
}
