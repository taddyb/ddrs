//! Per-batch subgraph union + compression helpers.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/builders.py::construct_network_matrix`
//! (lines ~55-110) and the COO-build portion of
//! `~/projects/ddr/src/ddr/geodatazoo/merit.py::_collate_gages`
//! (starts at line 197; COO-build at ~202-237).
//!
//! `build_flow_scale` mirrors `~/projects/ddr/src/ddr/io/readers.py::build_flow_scale_tensor`
//! (line 299) plus `compute_flow_scale_factor` (line 259).

use std::collections::BTreeSet;

use crate::data::ids::Staid;
use crate::data::store::{GageMetadata, GageSubgraph, GagesAdjacencyStore};

/// Output of the per-batch subgraph union. Edges are deduplicated and
/// returned in CONUS-position coordinates, sorted lex by `(row, col)`.
#[derive(Debug)]
pub struct UnionedCoo {
    pub edges: Vec<(usize, usize)>,
    /// One entry per gauge that was present in `gages_adj`:
    /// `(staid, gage_idx, gage_catchment)`. Carrying the STAID here lets
    /// `collate` derive `RoutingBatch.gauge_staids` directly.
    pub gauges: Vec<(Staid, usize, String)>,
}

/// Build the union of per-gauge subgraph COOs.
///
/// Mirrors `construct_network_matrix`. Missing gauges (not in `gages_adj`)
/// are silently skipped — matches DDR's `try / except KeyError` behavior.
pub fn union_subgraphs(
    staids: &[Staid],
    gages_adj: &GagesAdjacencyStore,
) -> UnionedCoo {
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut gauges: Vec<(Staid, usize, String)> = Vec::with_capacity(staids.len());
    for s in staids {
        let Some(g): Option<&GageSubgraph> = gages_adj.get(s) else { continue };
        gauges.push((s.clone(), g.gage_idx, g.gage_catchment.clone()));
        for (r, c) in g.indices_0.iter().zip(g.indices_1.iter()) {
            edges.insert((*r as usize, *c as usize));
        }
    }
    UnionedCoo {
        edges: edges.into_iter().collect(),
        gauges,
    }
}

use std::collections::HashMap;
use std::path::PathBuf;

use crate::data::error::{DataError, Result};
use crate::data::ids::Comid;

/// Compressed adjacency built from a unioned COO.
#[derive(Debug)]
pub struct CompressedAdj {
    /// Compressed COMIDs in topological order, length `N_active`.
    pub divide_comids: Vec<Comid>,
    /// Compressed-position rows (i32 for `SparseAdjacency`).
    pub rows: Vec<i32>,
    /// Compressed-position cols (i32 for `SparseAdjacency`).
    pub cols: Vec<i32>,
    /// Per-gauge compressed position of the gauge outlet, length `G_present`.
    pub gauge_compressed: Vec<usize>,
    /// For each gauge, the compressed cols whose row index equals the
    /// gauge's outlet. Mirrors DDR's `outflow_idx`.
    pub outflow_idx: Vec<Vec<usize>>,
}

/// Compress a unioned COO into dense compressed-position space, preserving
/// topological order via `BTreeSet` sort. The CONUS adjacency's `order`
/// array is itself topological — so a sorted subset stays topological.
///
/// Hard-asserts the lower-triangular invariant (`rows >= cols`); fails
/// with `DataError::Malformed` if violated.
pub fn compress(
    unioned: &UnionedCoo,
    conus_order: &[Comid],
) -> Result<CompressedAdj> {
    use std::collections::BTreeSet;

    // 1. Active set = union of edge endpoints + gauge outlets, sorted.
    let mut active: BTreeSet<usize> = BTreeSet::new();
    for &(r, c) in &unioned.edges {
        active.insert(r);
        active.insert(c);
    }
    for (_, g, _) in &unioned.gauges {
        active.insert(*g);
    }
    if active.is_empty() {
        return Err(DataError::Malformed {
            path: PathBuf::from("<collate>"),
            message: "compress: empty active set (no gauges + no edges)".into(),
        });
    }

    // 2. Map CONUS-position → compressed-position.
    let active_vec: Vec<usize> = active.into_iter().collect();
    let mut mapping: HashMap<usize, usize> = HashMap::with_capacity(active_vec.len());
    for (compressed_pos, &conus_pos) in active_vec.iter().enumerate() {
        mapping.insert(conus_pos, compressed_pos);
    }

    let divide_comids: Vec<Comid> = active_vec.iter().map(|&p| conus_order[p]).collect();

    // 3. Compress edges; assert lower-triangular.
    let nnz = unioned.edges.len();
    let mut rows: Vec<i32> = Vec::with_capacity(nnz);
    let mut cols: Vec<i32> = Vec::with_capacity(nnz);
    for &(r, c) in &unioned.edges {
        let rc = mapping[&r] as i32;
        let cc = mapping[&c] as i32;
        if rc < cc {
            return Err(DataError::Malformed {
                path: PathBuf::from("<collate>"),
                message: format!(
                    "lower-triangular violated: compressed edge ({rc},{cc}) — \
                     CONUS edge ({r},{c}) is upstream of itself"
                ),
            });
        }
        rows.push(rc);
        cols.push(cc);
    }

    // 4. Gauge compressed positions.
    let gauge_compressed: Vec<usize> =
        unioned.gauges.iter().map(|(_, g, _)| mapping[g]).collect();

    // 5. outflow_idx[g] = list of cols where rows[k] == gauge_compressed[g].
    // Fallback (matches DDR `_collate_gages` lines ~226-235): when a gauge
    // has no incoming edges in this batch's union, use the gauge's own
    // compressed index as the sole outflow. Headwater gauges are filtered
    // upstream of compress(), so this fallback only fires for gauges at
    // merge nodes whose upstream edges weren't in the batch.
    let mut outflow_idx: Vec<Vec<usize>> = Vec::with_capacity(gauge_compressed.len());
    for &g_comp in &gauge_compressed {
        let g_row = g_comp as i32;
        let cols_for_g: Vec<usize> = rows
            .iter()
            .zip(cols.iter())
            .filter(|(r, _)| **r == g_row)
            .map(|(_, c)| *c as usize)
            .collect();
        if cols_for_g.is_empty() {
            outflow_idx.push(vec![g_comp]);
        } else {
            outflow_idx.push(cols_for_g);
        }
    }

    Ok(CompressedAdj {
        divide_comids,
        rows,
        cols,
        gauge_compressed,
        outflow_idx,
    })
}

/// Per-segment flow scale factors of length `n_segments`. Default `1.0`;
/// the compressed-position of each gauge's outlet gets the gauge's scale.
///
/// Mirrors `build_flow_scale_tensor` in `~/projects/ddr/src/ddr/io/readers.py:270-330`:
/// fast path uses the `FLOW_SCALE` CSV column; fallback computes the factor
/// from `(DRAIN_SQKM, COMID_DRAIN_SQKM, COMID_UNITAREA_SQKM)`.
pub(crate) fn build_flow_scale(
    batch_staids: &[Staid],
    gauge_compressed: &[usize],
    gages: &GageMetadata,
    n_segments: usize,
) -> Vec<f32> {
    debug_assert_eq!(batch_staids.len(), gauge_compressed.len());
    let mut scale = vec![1.0_f32; n_segments];
    for (s, &seg) in batch_staids.iter().zip(gauge_compressed.iter()) {
        let Some(&i) = gages.by_staid.get(s) else { continue };
        let row = &gages.rows[i];
        if let Some(fs) = row.flow_scale {
            if fs.is_finite() {
                scale[seg] = fs;
                continue;
            }
        }
        if let (Some(comid_drain), Some(comid_unit)) =
            (row.comid_drain_sqkm, row.comid_unitarea_sqkm)
        {
            scale[seg] = compute_flow_scale_factor(
                row.drain_sqkm,
                comid_drain,
                comid_unit,
            );
        }
        // else: stays 1.0.
    }
    scale
}

/// Per-gauge scaling factor in `[0, 1]`. Mirrors
/// `compute_flow_scale_factor` in `readers.py:240-270`.
fn compute_flow_scale_factor(
    drain_sqkm: f64,
    comid_drain_sqkm: f64,
    comid_unitarea_sqkm: f64,
) -> f32 {
    if drain_sqkm.is_nan() || comid_drain_sqkm.is_nan() || comid_unitarea_sqkm.is_nan() {
        return 1.0;
    }
    if comid_unitarea_sqkm <= 0.0 {
        return 1.0;
    }
    let diff = drain_sqkm - comid_drain_sqkm;
    if diff >= 0.0 {
        return 1.0;
    }
    if diff.abs() >= comid_unitarea_sqkm {
        return 1.0;
    }
    ((comid_unitarea_sqkm - diff.abs()) / comid_unitarea_sqkm) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build a tiny in-memory `GagesAdjacencyStore` for unit tests.
    fn synthetic_store(
        gauges: &[(&str, usize, Vec<(i32, i32)>)],
    ) -> GagesAdjacencyStore {
        let mut subgraphs = HashMap::new();
        for (id, gage_idx, edges) in gauges {
            let staid = Staid::new(id);
            let indices_0: Vec<i32> = edges.iter().map(|(r, _)| *r).collect();
            let indices_1: Vec<i32> = edges.iter().map(|(_, c)| *c).collect();
            subgraphs.insert(
                staid.clone(),
                GageSubgraph {
                    staid,
                    gage_idx: *gage_idx,
                    gage_catchment: format!("comid{gage_idx}"),
                    indices_0,
                    indices_1,
                },
            );
        }
        GagesAdjacencyStore {
            path: std::path::PathBuf::from("<inline>"),
            subgraphs,
        }
    }

    #[test]
    fn union_deduplicates_shared_edges() {
        // Two gauges with overlapping ancestry: gauge A has edges {(3,1),
        // (3,2), (2,1)}; gauge B has {(4,2), (2,1)}. Shared edge (2,1)
        // appears once in the union.
        let store = synthetic_store(&[
            ("0000000A", 3, vec![(3, 1), (3, 2), (2, 1)]),
            ("0000000B", 4, vec![(4, 2), (2, 1)]),
        ]);
        let staids = vec![Staid::new("0000000A"), Staid::new("0000000B")];
        let u = union_subgraphs(&staids, &store);
        assert_eq!(u.edges.len(), 4);
        assert_eq!(u.edges, vec![(2, 1), (3, 1), (3, 2), (4, 2)]);
        assert_eq!(u.gauges.len(), 2);
        // Verify STAIDs carry through.
        assert_eq!(u.gauges[0].0, Staid::new("0000000A"));
        assert_eq!(u.gauges[0].1, 3);
        assert_eq!(u.gauges[0].2, "comid3");
        assert_eq!(u.gauges[1].0, Staid::new("0000000B"));
    }

    #[test]
    fn union_skips_missing_gauges() {
        let store = synthetic_store(&[("0000000A", 3, vec![(3, 1)])]);
        let staids = vec![Staid::new("0000000A"), Staid::new("00000099")];
        let u = union_subgraphs(&staids, &store);
        assert_eq!(u.gauges.len(), 1);
        assert_eq!(u.gauges[0].0, Staid::new("0000000A"));
        assert_eq!(u.edges.len(), 1);
    }

    #[test]
    fn union_empty_batch_returns_empty() {
        let store = synthetic_store(&[("0000000A", 3, vec![(3, 1)])]);
        let staids: Vec<Staid> = vec![];
        let u = union_subgraphs(&staids, &store);
        assert!(u.gauges.is_empty());
        assert!(u.edges.is_empty());
    }

    use crate::data::ids::Comid;

    #[test]
    fn compress_preserves_topological_order() {
        // CONUS positions [0, 1, 2, 3, 4], COMIDs in topological order.
        let conus_order = vec![Comid(100), Comid(200), Comid(300), Comid(400), Comid(500)];
        // Edges in CONUS positions, lower-triangular (rows >= cols).
        let unioned = UnionedCoo {
            edges: vec![(2, 0), (3, 1), (4, 2), (4, 3)],
            gauges: vec![
                (Staid::new("0000000A"), 4, "comid500".to_string()),
                (Staid::new("0000000B"), 3, "comid400".to_string()),
            ],
        };
        let c = compress(&unioned, &conus_order).expect("compress");
        // Active = {0, 1, 2, 3, 4} → all 5. Compressed positions match.
        assert_eq!(c.divide_comids, conus_order);
        assert_eq!(c.rows, vec![2, 3, 4, 4]);
        assert_eq!(c.cols, vec![0, 1, 2, 3]);
        assert_eq!(c.gauge_compressed, vec![4, 3]);
        // outflow_idx: gauge A at row 4 receives from cols 2, 3.
        // gauge B at row 3 receives from col 1.
        assert_eq!(c.outflow_idx[0], vec![2, 3]);
        assert_eq!(c.outflow_idx[1], vec![1]);
    }

    #[test]
    fn compress_remaps_sparse_active_to_dense_compressed() {
        // Sparse active set: CONUS positions {2, 5, 7, 9} → compressed {0,1,2,3}.
        let conus_order: Vec<Comid> = (0..10).map(|i| Comid(i as i64 * 100)).collect();
        let unioned = UnionedCoo {
            edges: vec![(9, 7), (9, 5), (7, 2)],
            gauges: vec![(Staid::new("0000000A"), 9, "comid900".to_string())],
        };
        let c = compress(&unioned, &conus_order).expect("compress");
        assert_eq!(c.divide_comids, vec![Comid(200), Comid(500), Comid(700), Comid(900)]);
        // Edges in compressed space: (3,2), (3,1), (2,0). Same order as input edges,
        // but mapped through the compressed index space.
        assert_eq!(c.rows.len(), 3);
        for k in 0..c.rows.len() {
            assert!(c.rows[k] >= c.cols[k], "lower-triangular violated at k={k}");
        }
        assert_eq!(c.gauge_compressed, vec![3]);
    }

    #[test]
    fn compress_errors_on_non_topological_edges() {
        let conus_order = vec![Comid(0), Comid(1), Comid(2)];
        // Bogus edge: row 0, col 1 — violates lower-triangular (upstream
        // referenced as downstream of itself).
        let unioned = UnionedCoo {
            edges: vec![(0, 1)],
            gauges: vec![(Staid::new("0000000A"), 0, "x".to_string())],
        };
        let err = compress(&unioned, &conus_order).unwrap_err();
        match err {
            crate::data::error::DataError::Malformed { .. } => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn compress_empty_unioned_errors() {
        let conus_order = vec![Comid(0)];
        let unioned = UnionedCoo {
            edges: vec![],
            gauges: vec![],
        };
        let err = compress(&unioned, &conus_order).unwrap_err();
        match err {
            crate::data::error::DataError::Malformed { .. } => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn outflow_idx_falls_back_to_self_when_no_incoming_edges() {
        // Gauge at CONUS-position 2 with no upstream edges in this batch
        // (active = {2} as a single-node graph). DDR's fallback yields the
        // gauge's own compressed index as the sole outflow column.
        let conus_order = vec![Comid(0), Comid(1), Comid(2)];
        let unioned = UnionedCoo {
            edges: vec![],
            gauges: vec![(Staid::new("0000000A"), 2, "comid2".to_string())],
        };
        let c = compress(&unioned, &conus_order).expect("compress");
        assert_eq!(c.gauge_compressed, vec![0]);
        assert_eq!(c.outflow_idx[0], vec![0], "self-edge fallback");
    }

    use crate::data::store::{GageMetadata, GageRow};

    fn synthetic_gage_meta(rows: Vec<GageRow>) -> GageMetadata {
        let by_staid = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.staid.clone(), i))
            .collect();
        GageMetadata {
            path: std::path::PathBuf::from("<inline>"),
            rows,
            by_staid,
        }
    }

    fn make_row(staid: &str, flow_scale: Option<f32>) -> GageRow {
        GageRow {
            staid: Staid::new(staid),
            staname: staid.into(),
            drain_sqkm: 100.0,
            lat_gage: 0.0,
            lng_gage: 0.0,
            comid: None,
            comid_drain_sqkm: None,
            comid_unitarea_sqkm: None,
            abs_diff: None,
            da_valid: Some(true),
            flow_scale,
        }
    }

    #[test]
    fn flow_scale_fast_path_uses_csv_column() {
        let meta = synthetic_gage_meta(vec![
            make_row("00000001", Some(0.5)),
            make_row("00000002", Some(0.8)),
        ]);
        let staids = vec![Staid::new("00000001"), Staid::new("00000002")];
        let gauge_compressed = vec![3, 7];
        let scale = build_flow_scale(&staids, &gauge_compressed, &meta, 10);
        assert_eq!(scale.len(), 10);
        assert!((scale[3] - 0.5).abs() < 1e-9);
        assert!((scale[7] - 0.8).abs() < 1e-9);
        for &i in &[0, 1, 2, 4, 5, 6, 8, 9] {
            assert!((scale[i] - 1.0).abs() < 1e-9, "expected 1.0 at {i}, got {}", scale[i]);
        }
    }

    #[test]
    fn flow_scale_fallback_to_factor_when_csv_missing() {
        let mut row = make_row("00000001", None);
        row.drain_sqkm = 50.0;
        row.comid_drain_sqkm = Some(100.0);
        row.comid_unitarea_sqkm = Some(60.0);
        let meta = synthetic_gage_meta(vec![row]);
        let staids = vec![Staid::new("00000001")];
        let scale = build_flow_scale(&staids, &vec![2], &meta, 5);
        // diff = 50 - 100 = -50; abs(diff) = 50 < 60 = unitarea
        // factor = (60 - 50) / 60 = 1/6
        let expected = (60.0_f64 - 50.0_f64) / 60.0_f64;
        assert!(
            (scale[2] as f64 - expected).abs() < 1e-6,
            "scale[2]={} expected={expected}",
            scale[2]
        );
    }

    #[test]
    fn flow_scale_unknown_staid_keeps_default_one() {
        let meta = synthetic_gage_meta(vec![make_row("00000001", Some(0.3))]);
        // Caller asks for a STAID that isn't in the metadata — should leave
        // the corresponding segment at 1.0.
        let staids = vec![Staid::new("99999999")];
        let scale = build_flow_scale(&staids, &vec![0], &meta, 3);
        for i in 0..3 {
            assert!((scale[i] - 1.0).abs() < 1e-9);
        }
    }
}
