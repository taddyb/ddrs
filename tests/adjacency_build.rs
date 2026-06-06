//! Synthetic-network tests for the CONUS adjacency builder
//! (`src/adjacency/build.rs`).
//!
//! Expected `order` / `rows` / `cols` values are NOT hand-derived: they were
//! produced by running the real `ddr_engine` pipeline (rustworkx 0.17.1
//! `build_upstream_dict` → `build_graph` → `topological_sort`) on the same
//! synthetic networks, so these tests pin element-for-element parity with the
//! engine.
//!
//! Element-for-element `order` parity holds at CONUS scale too: the engine's
//! `topological_sort` (petgraph DFS finish-time order) is deterministic and the
//! managed build reproduces it exactly. See `tests/adjacency_parity.rs`.

use ddrs::adjacency::build::{build_conus_adjacency, BuildError};
use ddrs::adjacency::dbf::FlowpathRecord;

/// Build a record. `up` lists upstream COMIDs (zero-padded to 4); `lengthkm`
/// and `slope` default to finite placeholders unless overridden.
fn rec(comid: i64, up: &[i64], lengthkm: f64, slope: f64) -> FlowpathRecord {
    let mut up_arr = [0i64; 4];
    for (i, &u) in up.iter().enumerate() {
        up_arr[i] = u;
    }
    let next_down_id = 0; // unused by the builder (edges come from up1..up4)
    FlowpathRecord {
        comid,
        lengthkm,
        slope,
        next_down_id,
        up: up_arr,
    }
}

/// Lower-triangular invariant 3 must hold on every produced COO.
fn assert_lower_triangular(adj: &ddrs::adjacency::build::ConusAdjacency) {
    for (&r, &c) in adj.rows.iter().zip(adj.cols.iter()) {
        assert!(r >= c, "not lower triangular: row {r} < col {c}");
    }
}

#[test]
fn chain_a_b_c() {
    // 10 -> 20 -> 30 (20.up=[10], 30.up=[20]). Engine order: [10,20,30].
    let records = vec![
        rec(10, &[], 1.0, 0.01),
        rec(20, &[10], 2.0, 0.02),
        rec(30, &[20], 3.0, 0.03),
    ];
    let adj = build_conus_adjacency(&records).unwrap();
    assert_eq!(adj.order, vec![10, 20, 30]);
    assert_eq!(adj.rows, vec![1, 2]);
    assert_eq!(adj.cols, vec![0, 1]);
    assert!(adj.dropped_comids.is_empty());
    assert_lower_triangular(&adj);
}

#[test]
fn confluence_two_upstreams() {
    // 10 -> 30, 20 -> 30 (30.up=[10,20]). Engine order: [20,10,30].
    let records = vec![
        rec(10, &[], 1.0, 0.01),
        rec(20, &[], 2.0, 0.02),
        rec(30, &[10, 20], 3.0, 0.03),
    ];
    let adj = build_conus_adjacency(&records).unwrap();
    assert_eq!(adj.order, vec![20, 10, 30]);
    assert_eq!(adj.rows, vec![2, 2]);
    assert_eq!(adj.cols, vec![0, 1]);
    assert_lower_triangular(&adj);
}

#[test]
fn isolated_reaches_appended_sorted() {
    // 10 -> 20 connected; 99 and 5 isolated. Engine order: [10,20,5,99].
    let records = vec![
        rec(10, &[], 1.0, 0.01),
        rec(20, &[10], 2.0, 0.02),
        rec(99, &[], 9.0, 0.09),
        rec(5, &[], 0.5, 0.005),
    ];
    let adj = build_conus_adjacency(&records).unwrap();
    assert_eq!(adj.order, vec![10, 20, 5, 99]);
    assert_eq!(adj.rows, vec![1]);
    assert_eq!(adj.cols, vec![0]);
    assert_lower_triangular(&adj);
}

#[test]
fn sandbox_like_network() {
    // 10,20 -> 30 -> 50 ; 40 -> 50. Engine order: [40,20,10,30,50].
    let records = vec![
        rec(10, &[], 1.0, 0.01),
        rec(20, &[], 2.0, 0.02),
        rec(30, &[10, 20], 3.0, 0.03),
        rec(40, &[], 4.0, 0.04),
        rec(50, &[30, 40], 5.0, 0.05),
    ];
    let adj = build_conus_adjacency(&records).unwrap();
    assert_eq!(adj.order, vec![40, 20, 10, 30, 50]);
    assert_eq!(adj.rows, vec![4, 3, 3, 4]);
    assert_eq!(adj.cols, vec![0, 1, 2, 3]);
    assert_lower_triangular(&adj);
}

#[test]
fn cycle_removal_drops_cycle_comids() {
    // 2-cycle 10<->20 (10.up=[20], 20.up=[10]) plus normal 30 -> 40.
    // Engine drops {10,20}, recurses → order [30,40].
    let records = vec![
        rec(10, &[20], 1.0, 0.01),
        rec(20, &[10], 2.0, 0.02),
        rec(30, &[], 3.0, 0.03),
        rec(40, &[30], 4.0, 0.04),
    ];
    let adj = build_conus_adjacency(&records).unwrap();
    assert_eq!(adj.order, vec![30, 40]);
    assert_eq!(adj.rows, vec![1]);
    assert_eq!(adj.cols, vec![0]);
    assert_eq!(adj.dropped_comids, vec![10, 20]);
    assert_lower_triangular(&adj);
}

#[test]
fn nan_and_inf_filled_with_finite_mean() {
    // Chain 10->20->30. lengthkm: [2,NaN,inf] -> finite mean over {2}=2.0
    //   (in metres: {2000} -> mean 2000). slope: [0.01, 0.03, NaN] ->
    //   finite mean over {0.01,0.03}=0.02.
    let records = vec![
        rec(10, &[], 2.0, 0.01),
        rec(20, &[10], f64::NAN, 0.03),
        rec(30, &[20], f64::INFINITY, f64::NAN),
    ];
    let adj = build_conus_adjacency(&records).unwrap();
    assert_eq!(adj.order, vec![10, 20, 30]);

    // length_m mean over finite metres values = 2000 (only COMID 10 is finite).
    // COMID 10 -> 2000, COMID 20 (NaN) -> 2000, COMID 30 (inf) -> 2000.
    for (i, &v) in adj.length_m.iter().enumerate() {
        assert!(v.is_finite(), "length_m[{i}] not finite: {v}");
        assert!((v - 2000.0).abs() < 1e-3, "length_m[{i}] = {v}, want 2000");
    }

    // slope finite mean over {0.01,0.03} = 0.02. COMID 30 (NaN) -> 0.02.
    // order is [10,20,30] -> slopes [0.01, 0.03, fill=0.02].
    assert!((adj.slope[0] - 0.01).abs() < 1e-6);
    assert!((adj.slope[1] - 0.03).abs() < 1e-6);
    assert!((adj.slope[2] - 0.02).abs() < 1e-6, "filled slope = {}", adj.slope[2]);
    for &v in &adj.slope {
        assert!(v.is_finite());
    }
}

#[test]
fn length_m_is_lengthkm_times_1000() {
    // Single headwater + one downstream so there is at least one edge.
    let records = vec![rec(10, &[], 1.5, 0.01), rec(20, &[10], 2.25, 0.02)];
    let adj = build_conus_adjacency(&records).unwrap();
    // order [10,20]; lengths 1500, 2250 metres.
    assert!((adj.length_m[0] - 1500.0).abs() < 1e-3);
    assert!((adj.length_m[1] - 2250.0).abs() < 1e-3);
}

#[test]
fn not_dendritic_is_an_error() {
    // Force a node with two successors: 10 is upstream of BOTH 20 and 30.
    // up-lists: 20.up=[10], 30.up=[10]. Node 10 then has out-degree 2.
    let records = vec![
        rec(10, &[], 1.0, 0.01),
        rec(20, &[10], 2.0, 0.02),
        rec(30, &[10], 3.0, 0.03),
    ];
    let err = build_conus_adjacency(&records).unwrap_err();
    match err {
        BuildError::NotDendritic { comid, n_successors } => {
            assert_eq!(comid, 10);
            assert_eq!(n_successors, 2);
        }
        other => panic!("expected NotDendritic, got {other:?}"),
    }
}

/// Fuzz parity: replay 1000 random dendritic MERIT-like networks through the
/// public builder and compare `order` element-for-element against rustworkx
/// 0.17.1 (the engine's `topological_sort`). The fixture
/// `tests/fixtures/toposort_fuzz.jsonl` is generated under DDR's venv by
/// `scripts/dump_toposort_fixtures.py`; each line carries raw FlowpathRecord
/// rows plus the engine's topological order. This is the regression guard for
/// the DFS-finish-time toposort port in `src/adjacency/build.rs` — a LIFO-Kahn
/// queue (the prior, buggy port) fails ~85% of these graphs.
#[test]
fn fuzz_toposort_matches_rustworkx() {
    use std::io::BufRead;

    #[derive(serde::Deserialize)]
    struct Case {
        records: Vec<Vec<i64>>, // [comid, up1, up2, up3, up4]
        order: Vec<i64>,
    }

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/toposort_fuzz.jsonl");
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e} (regenerate with scripts/dump_toposort_fixtures.py)"));

    let mut total = 0usize;
    for (li, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(&line).unwrap();
        let records: Vec<FlowpathRecord> = case
            .records
            .iter()
            .map(|row| {
                let comid = row[0];
                let ups: Vec<i64> = row[1..].iter().copied().filter(|&u| u > 0).collect();
                rec(comid, &ups, 1.0, 0.01)
            })
            .collect();
        let adj = build_conus_adjacency(&records)
            .unwrap_or_else(|e| panic!("line {li}: build failed: {e:?}"));
        let ours: Vec<i64> = adj.order.iter().map(|&c| c as i64).collect();
        assert_eq!(
            ours, case.order,
            "line {li}: order mismatch vs rustworkx\n  ours = {ours:?}\n  rx   = {:?}",
            case.order
        );
        total += 1;
    }
    assert!(total >= 1000, "expected >= 1000 fuzz cases, ran {total}");
    eprintln!("fuzz_toposort_matches_rustworkx: {total} graphs match rustworkx element-for-element");
}

#[test]
fn position_lookup_maps_comid_to_index() {
    let records = vec![
        rec(10, &[], 1.0, 0.01),
        rec(20, &[10], 2.0, 0.02),
        rec(30, &[20], 3.0, 0.03),
    ];
    let adj = build_conus_adjacency(&records).unwrap();
    let lut = adj.position_lookup();
    assert_eq!(lut[&10], 0);
    assert_eq!(lut[&20], 1);
    assert_eq!(lut[&30], 2);
}
