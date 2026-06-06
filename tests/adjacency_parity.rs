//! Parity test: managed adjacency build vs the engine-built zarr stores.
//!
//! `#[ignore]` by default (like `conus_adjacency_loads_real_merit_zarr`) because
//! it reads the real ~108 MB pfaf_7 `.dbf` and runs the full topo sort + 3000
//! gauge BFS (~10 s). Run explicitly:
//!
//! ```bash
//! cargo test --test adjacency_parity -- --ignored --nocapture
//! ```
//!
//! ## IMPORTANT: the engine store's `order` is NOT reproducible
//!
//! The original Task-8 contract was "`order`/`indices_0`/`indices_1` match the
//! engine store element-for-element." Running this on the real CONUS fabric
//! disproves that contract: the node SETS, edge SETS (in COMID space), dropped
//! set, lower-triangularity, and isolated-tail length all match exactly, but the
//! topological *order* is a different (equally valid) permutation.
//!
//! Root cause: the engine's `build_upstream_dict` (`graph.py:9-52`) returns a
//! dict whose **key/edge iteration order is polars `group_by` order** — hash-
//! based and non-deterministic across machines/versions. `build_graph`
//! (`graph.py:82-85`) then inserts edges in that `items()` order, and rustworkx's
//! `topological_sort` uses per-node successor-insertion order as its tie-break.
//! So the engine's stored `order` array is an artifact of one irreproducible
//! polars run; two engine builds on different machines would not agree either.
//! (Task 3's element-for-element synthetic tests still hold — on tiny graphs the
//! edge order is incidentally stable; at 346 K nodes it is not.)
//!
//! The reader contract does not depend on the specific permutation: gauge
//! subgraphs reference CONUS *positions*, always resolved through the SAME
//! `order` array they were built with. So the load-bearing invariants are
//! structural, and that is what this test asserts:
//!   1. CONUS node set, edge set (COMID space), dropped set, lower-triangular,
//!      and length all match the engine store.
//!   2. length_m/slope spot-checks (NOT an element compare — our builder fills
//!      NaN/inf with the finite-column mean, the documented bug fix).
//!   3. Gauges: per sampled STAID, node SETS and edge SETS **in COMID space**
//!      plus `gage_catchment` match. `gage_idx` is position-space and therefore
//!      order-dependent; we assert instead that it resolves (in each store's own
//!      `order`) to the same outlet COMID.
//!
//! ## Data location
//!
//! Mirrors `data_zarr_store.rs`: hard-coded real paths, `skip_if_missing` returns
//! early if absent so a clean checkout doesn't break. The flowlines `.dbf` path
//! can be overridden with `DDRS_MERIT_DBF` (the config cites a wukong path that
//! does not exist on every machine); it falls back to the local `/mnt/ssd1` copy.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use ddrs::adjacency::build::build_conus_adjacency;
use ddrs::adjacency::dbf::{read_flowpath_records, FlowpathRecord};
use ddrs::adjacency::gauges::build_gauge_subgraphs;
use ddrs::data::{Comid, ConusAdjacencyStore, GagesAdjacencyStore, Staid};

const CONUS_ADJ: &str = "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr";
const GAGES_ADJ: &str = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr";
const GAGES_CSV: &str = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv";
/// Default flowlines `.dbf` (override with `DDRS_MERIT_DBF`). The config cites
/// `/projects/mhpi/data/MERIT/...` which does not exist on every machine.
const MERIT_DBF: &str =
    "/mnt/ssd1/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.dbf";

fn skip_if_missing(path: &str) -> Option<PathBuf> {
    let p = PathBuf::from(path);
    p.exists().then_some(p)
}

/// Resolve the flowlines `.dbf`: `DDRS_MERIT_DBF` env var first, then the
/// local default. Returns `None` if neither exists.
fn resolve_dbf() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("DDRS_MERIT_DBF") {
        let p = PathBuf::from(env);
        return p.exists().then_some(p);
    }
    skip_if_missing(MERIT_DBF)
}

#[test]
#[ignore = "reads the real ~108 MB pfaf_7 dbf + full topo/BFS build (~10 s)"]
fn managed_build_matches_engine_store() {
    // ---- Locate inputs; skip gracefully if any are absent -------------------
    let Some(dbf_path) = resolve_dbf() else {
        eprintln!("skip: no MERIT dbf (set DDRS_MERIT_DBF or place {MERIT_DBF})");
        return;
    };
    let Some(conus_path) = skip_if_missing(CONUS_ADJ) else {
        eprintln!("skip: {CONUS_ADJ} not present");
        return;
    };
    let Some(gages_path) = skip_if_missing(GAGES_ADJ) else {
        eprintln!("skip: {GAGES_ADJ} not present");
        return;
    };
    let Some(gages_csv) = skip_if_missing(GAGES_CSV) else {
        eprintln!("skip: {GAGES_CSV} not present");
        return;
    };

    // ---- Build ONCE; run every assertion against this single build ----------
    eprintln!("reading dbf {} ...", dbf_path.display());
    let records: Vec<FlowpathRecord> =
        read_flowpath_records(&dbf_path).expect("read flowpath records");
    eprintln!("  dbf records: {}", records.len());

    eprintln!("building CONUS adjacency ...");
    let conus = build_conus_adjacency(&records).expect("build conus adjacency");
    eprintln!(
        "  order: {}  nnz: {}  dropped: {}",
        conus.order.len(),
        conus.rows.len(),
        conus.dropped_comids.len()
    );

    eprintln!("opening engine CONUS store {} ...", conus_path.display());
    let engine = ConusAdjacencyStore::open(&conus_path).expect("open engine conus store");
    eprintln!("  engine order: {}  nnz: {}", engine.n, engine.nnz);

    // ========================================================================
    // Section 1: CONUS structural parity (node set, edge set, lower-triangular)
    //   NOT element-for-element order — see the module header for why the engine
    //   store's `order` permutation is irreproducible.
    // ========================================================================
    assert_eq!(conus.order.len(), engine.n, "order length");
    assert_eq!(conus.rows.len(), engine.nnz, "nnz");

    // Node SETS (COMID space) equal.
    let our_nodes: BTreeSet<i64> = conus.order.iter().map(|&c| c as i64).collect();
    let eng_nodes: BTreeSet<i64> = engine.order.iter().map(|c| c.0).collect();
    assert_eq!(
        our_nodes, eng_nodes,
        "CONUS node sets differ (ours {} vs engine {})",
        our_nodes.len(),
        eng_nodes.len()
    );

    // Edge SETS in COMID space equal (translate each store's COO positions
    // through its OWN `order` array, then compare the COMID-pair sets).
    let our_edges: BTreeSet<(i64, i64)> = conus
        .rows
        .iter()
        .zip(conus.cols.iter())
        .map(|(&r, &c)| (conus.order[r as usize] as i64, conus.order[c as usize] as i64))
        .collect();
    let eng_edges: BTreeSet<(i64, i64)> = engine
        .indices_0
        .iter()
        .zip(engine.indices_1.iter())
        .map(|(&r, &c)| (engine.order[r as usize].0, engine.order[c as usize].0))
        .collect();
    assert_eq!(
        our_edges, eng_edges,
        "CONUS edge sets (COMID space) differ (ours {} vs engine {})",
        our_edges.len(),
        eng_edges.len()
    );

    // Both COOs lower-triangular (routing invariant 3).
    let our_lt_violations = conus
        .rows
        .iter()
        .zip(conus.cols.iter())
        .filter(|(r, c)| r < c)
        .count();
    assert_eq!(our_lt_violations, 0, "our COO not lower-triangular");

    eprintln!("section 1 OK: node set + edge set (COMID space) + lower-triangular match");

    // ========================================================================
    // Section 2: cycle-removal delta == engine's own delta (dbf − engine_order)
    // ========================================================================
    let dbf_comids: BTreeSet<i64> = records.iter().map(|r| r.comid).collect();
    let engine_delta: BTreeSet<i64> = dbf_comids.difference(&eng_nodes).copied().collect();
    let our_dropped: BTreeSet<i64> = conus.dropped_comids.iter().copied().collect();
    eprintln!(
        "section 2: dbf={} engine_order={} engine_delta={} our_dropped={}",
        dbf_comids.len(),
        eng_nodes.len(),
        engine_delta.len(),
        our_dropped.len()
    );
    assert_eq!(
        our_dropped, engine_delta,
        "dropped-COMID set must equal the engine's (dbf − engine_order) delta"
    );
    eprintln!(
        "section 2 OK: dropped {} COMIDs == engine delta: {:?}",
        our_dropped.len(),
        our_dropped
    );

    // ========================================================================
    // Section 3: length_m / slope spot-checks (NOT element compare)
    // ========================================================================
    let length_mean = naninfmean(records.iter().map(|r| r.lengthkm * 1000.0));
    let slope_mean = naninfmean(records.iter().map(|r| r.slope));
    eprintln!("section 3: length_mean={length_mean:.6}  slope_mean={slope_mean:.9}");

    let pos_of: HashMap<i32, usize> = conus
        .order
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // A few finite-valued COMIDs: assert length_m == lengthkm*1000 and slope
    // passthrough (hand-computed straight from the dbf records).
    let mut finite_checked = 0;
    for rec in &records {
        if finite_checked >= 5 {
            break;
        }
        if rec.lengthkm.is_finite() && rec.slope.is_finite() {
            let Some(&pos) = pos_of.get(&(rec.comid as i32)) else {
                continue; // dropped on a cycle (rare)
            };
            let expected_len = (rec.lengthkm * 1000.0) as f32;
            let expected_slope = rec.slope as f32;
            assert!(
                (conus.length_m[pos] - expected_len).abs() <= 1e-2 * expected_len.abs().max(1.0),
                "COMID {} length_m {} != lengthkm*1000 {}",
                rec.comid,
                conus.length_m[pos],
                expected_len
            );
            assert!(
                (conus.slope[pos] - expected_slope).abs() <= 1e-6 + 1e-3 * expected_slope.abs(),
                "COMID {} slope {} != dbf slope {}",
                rec.comid,
                conus.slope[pos],
                expected_slope
            );
            finite_checked += 1;
        }
    }
    assert!(finite_checked > 0, "no finite-valued COMIDs found to spot-check");
    eprintln!("section 3a OK: {finite_checked} finite COMIDs pass length/slope passthrough");

    // A NaN-in-dbf COMID (if any survive to `order`): assert it was filled to
    // the finite-column mean.
    match records
        .iter()
        .find(|r| r.lengthkm.is_nan() && pos_of.contains_key(&(r.comid as i32)))
    {
        Some(rec) => {
            let pos = pos_of[&(rec.comid as i32)];
            assert!(
                (conus.length_m[pos] as f64 - length_mean).abs()
                    <= 1e-2 * length_mean.abs().max(1.0),
                "NaN-length COMID {} not filled to mean: {} vs {}",
                rec.comid,
                conus.length_m[pos],
                length_mean
            );
            eprintln!(
                "section 3b OK: NaN-length COMID {} filled to mean {}",
                rec.comid, conus.length_m[pos]
            );
        }
        None => eprintln!("section 3b: no NaN-length COMID in dbf (fill path untested here)"),
    }
    match records
        .iter()
        .find(|r| r.slope.is_nan() && pos_of.contains_key(&(r.comid as i32)))
    {
        Some(rec) => {
            let pos = pos_of[&(rec.comid as i32)];
            assert!(
                (conus.slope[pos] as f64 - slope_mean).abs() <= 1e-6 + 1e-3 * slope_mean.abs(),
                "NaN-slope COMID {} not filled to mean: {} vs {}",
                rec.comid,
                conus.slope[pos],
                slope_mean
            );
            eprintln!(
                "section 3b OK: NaN-slope COMID {} filled to mean {}",
                rec.comid, conus.slope[pos]
            );
        }
        None => eprintln!("section 3b: no NaN-slope COMID in dbf"),
    }

    // ========================================================================
    // Section 4: gauges — node SETS + edge SETS (COMID space) + outlet COMID
    //   gage_idx is position-space (order-dependent), so we compare what it
    //   resolves to: engine.order[gage_idx] (engine) and conus.order[gage_idx]
    //   (ours) must each equal the outlet COMID.
    // ========================================================================
    eprintln!("building gauge subgraphs ...");
    let our_gauges = build_gauge_subgraphs(&conus, &gages_csv).expect("build gauge subgraphs");
    eprintln!("  built {} gauge subgraphs", our_gauges.len());

    let mut our_staids: Vec<Staid> = our_gauges.iter().map(|g| g.staid.clone()).collect();
    our_staids.sort_by(|a, b| a.as_str().cmp(b.as_str()));

    let engine_gauges =
        GagesAdjacencyStore::open(&gages_path, &our_staids).expect("open engine gauges store");

    let our_by_staid: HashMap<&str, &ddrs::adjacency::gauges::GaugeSubgraph> =
        our_gauges.iter().map(|g| (g.staid.as_str(), g)).collect();

    // First 10 sorted STAIDs present in BOTH our build and the engine store.
    let sampled: Vec<Staid> = our_staids
        .iter()
        .filter(|s| engine_gauges.get(s).is_some())
        .take(10)
        .cloned()
        .collect();
    assert!(
        !sampled.is_empty(),
        "no STAIDs present in both our build and the engine gauges store"
    );
    eprintln!("section 4: sampling {} STAIDs", sampled.len());

    for staid in &sampled {
        let ours = our_by_staid[staid.as_str()];
        let eng = engine_gauges.get(staid).expect("engine subgraph");

        // Outlet COMID matches, and each store's gage_idx resolves (through its
        // own `order`) to that outlet COMID.
        let outlet = ours.gage_catchment;
        assert_eq!(
            outlet.to_string(),
            eng.gage_catchment,
            "STAID {}: gage_catchment ours {} vs engine {}",
            staid.as_str(),
            outlet,
            eng.gage_catchment
        );
        assert_eq!(
            conus.order[ours.gage_idx] as i64,
            outlet,
            "STAID {}: our gage_idx {} resolves to {} not outlet {}",
            staid.as_str(),
            ours.gage_idx,
            conus.order[ours.gage_idx],
            outlet
        );
        assert_eq!(
            engine.order[eng.gage_idx],
            Comid(outlet),
            "STAID {}: engine gage_idx {} resolves to {:?} not outlet {}",
            staid.as_str(),
            eng.gage_idx,
            engine.order[eng.gage_idx],
            outlet
        );

        // Node SETS in COMID space (translate each side through its own order).
        let our_node_set: BTreeSet<i64> = ours
            .rows
            .iter()
            .chain(ours.cols.iter())
            .map(|&p| conus.order[p as usize] as i64)
            .collect();
        let eng_node_set: BTreeSet<i64> = eng
            .indices_0
            .iter()
            .chain(eng.indices_1.iter())
            .map(|&p| engine.order[p as usize].0)
            .collect();
        assert_eq!(
            our_node_set, eng_node_set,
            "STAID {}: node sets (COMID space) differ (ours {} vs engine {})",
            staid.as_str(),
            our_node_set.len(),
            eng_node_set.len()
        );

        // Edge SETS in COMID space (sort pairs implicitly via BTreeSet).
        let our_edge_set: BTreeSet<(i64, i64)> = ours
            .rows
            .iter()
            .zip(ours.cols.iter())
            .map(|(&r, &c)| (conus.order[r as usize] as i64, conus.order[c as usize] as i64))
            .collect();
        let eng_edge_set: BTreeSet<(i64, i64)> = eng
            .indices_0
            .iter()
            .zip(eng.indices_1.iter())
            .map(|(&r, &c)| (engine.order[r as usize].0, engine.order[c as usize].0))
            .collect();
        assert_eq!(
            our_edge_set, eng_edge_set,
            "STAID {}: edge sets (COMID space) differ (ours {} vs engine {})",
            staid.as_str(),
            our_edge_set.len(),
            eng_edge_set.len()
        );

        eprintln!(
            "  STAID {} OK: outlet={} nodes={} edges={}",
            staid.as_str(),
            outlet,
            our_node_set.len(),
            our_edge_set.len()
        );
    }

    eprintln!("ALL SECTIONS PASS");
}

/// Mean over finite values only (excludes NaN and ±inf). Local copy of the
/// builder's private `naninfmean` so the test hand-computes expected fills.
fn naninfmean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        f64::NAN
    }
}
