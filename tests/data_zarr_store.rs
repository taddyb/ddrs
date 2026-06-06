//! Integration test for the zarr-store readers, against DDR's live data.
//!
//! Touches `~/projects/ddr/data/merit_conus_adjacency.zarr` and
//! `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` directly. Skips
//! gracefully if either is absent so CI on a clean checkout doesn't break.
//!
//! These intentionally target the **explicit engine-built** zarr stores: their
//! purpose is to validate the *reader* against a real on-disk store, which stays
//! valid regardless of how the store was produced. Parity of the *managed*
//! builder's output (`src/adjacency/`) against an engine store is covered
//! separately by `tests/adjacency_parity.rs`.
use std::path::PathBuf;

use ddrs::data::{Comid, ConusAdjacencyStore, GagesAdjacencyStore, Staid};

const CONUS_ADJ: &str = "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr";
const GAGES_ADJ: &str = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr";

fn skip_if_missing(path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    p.exists().then_some(p)
}

#[test]
fn conus_adjacency_loads_real_merit_zarr() {
    let Some(path) = skip_if_missing(CONUS_ADJ) else {
        eprintln!("skip: {CONUS_ADJ} not present");
        return;
    };

    let store = ConusAdjacencyStore::open(&path).expect("open conus adjacency");

    // Numbers cross-checked against `cat .../merit_conus_adjacency.zarr/zarr.json`
    // (shape [346321, 346321]) and `.../indices_0/zarr.json` (shape [338814]).
    assert_eq!(store.n, 346_321);
    assert_eq!(store.nnz, 338_814);
    assert_eq!(store.order.len(), store.n);
    assert_eq!(store.length_m.len(), store.n);
    assert_eq!(store.slope.len(), store.n);
    assert_eq!(store.indices_0.len(), store.nnz);
    assert_eq!(store.indices_1.len(), store.nnz);

    // Adjacency invariant: lower-triangular in topological order
    // → indices_0[i] (downstream) >= indices_1[i] (upstream) for all i.
    let violations = store
        .indices_0
        .iter()
        .zip(&store.indices_1)
        .filter(|(r, c)| r < c)
        .count();
    assert_eq!(violations, 0, "{violations} edges violate row >= col");

    // length_m / slope should be non-negative where finite. Some NaNs are
    // expected (DDR fills them with row means downstream).
    let bad_length = store
        .length_m
        .iter()
        .filter(|v| v.is_finite() && **v < 0.0)
        .count();
    let bad_slope = store
        .slope
        .iter()
        .filter(|v| v.is_finite() && **v < 0.0)
        .count();
    assert_eq!(bad_length, 0);
    assert_eq!(bad_slope, 0);

    // IdIndex roundtrip on a real COMID.
    let sample_comid = store.order[1000];
    assert_eq!(store.index.position(&sample_comid), Some(1000));
    assert_eq!(store.index.id_at(1000), Some(&sample_comid));
}

#[test]
fn gages_adjacency_loads_known_staid() {
    let Some(path) = skip_if_missing(GAGES_ADJ) else {
        eprintln!("skip: {GAGES_ADJ} not present");
        return;
    };

    // 01011000 is the first row of dhbv2_gages.csv; should be in the store.
    let staid = Staid::new("01011000");
    let store = GagesAdjacencyStore::open(&path, &[staid.clone()])
        .expect("open gages adjacency");

    // If 01011000 wasn't in the store, eager-load just skips it silently
    // (mirrors DDR's valid_gauges_mask). Don't hard-fail — log and exit
    // since gauge presence depends on the upstream catalogue version.
    let Some(subgraph) = store.get(&staid) else {
        eprintln!("skip: gauge 01011000 not present in store (catalogue drift)");
        return;
    };
    assert_eq!(subgraph.staid, staid);
    assert!(
        !subgraph.indices_0.is_empty(),
        "01011000 should have at least one upstream edge"
    );
    assert_eq!(
        subgraph.indices_0.len(),
        subgraph.indices_1.len(),
        "subgraph COO indices must be paired"
    );
    // gage_idx must be a valid position in the CONUS 346K-reach space.
    assert!(subgraph.gage_idx < 346_321);
    // Row >= col on the subgraph too (still topological).
    let violations = subgraph
        .indices_0
        .iter()
        .zip(&subgraph.indices_1)
        .filter(|(r, c)| r < c)
        .count();
    assert_eq!(violations, 0);
}

#[test]
fn gages_adjacency_skips_unknown_staids_silently() {
    let Some(path) = skip_if_missing(GAGES_ADJ) else {
        eprintln!("skip: {GAGES_ADJ} not present");
        return;
    };
    let fake = Staid::new("99999999");
    let store = GagesAdjacencyStore::open(&path, &[fake.clone()])
        .expect("open with bogus staid should not error");
    assert!(store.get(&fake).is_none());
    assert_eq!(store.len(), 0);
}

#[test]
fn comid_newtype_blocks_mixups() {
    // Compile-time check — uncomment to see the error:
    //     let _: Comid = 12345_i64;     // does not compile
    //     let _: Staid = "01011000";    // does not compile
    let a = Comid(12345);
    let b = Comid(12345);
    assert_eq!(a, b);
}
