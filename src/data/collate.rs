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
    /// For each gauge, the compressed reach indices whose routed discharge is
    /// summed to form that gauge's prediction. Which reaches those are depends
    /// on `params.ddr_match` — see `compress` step 5:
    ///
    /// * `ddr_match: false` (physically correct) — a single element: the
    ///   gauge's OWN reach, at its outlet piece. The MC solve there already
    ///   integrates the whole upstream network plus that reach's own lateral
    ///   inflow. Without subdivision that is `gauge_compressed[g]` itself.
    /// * `ddr_match: true` (DDR-faithful default) — the gauge's UPSTREAM
    ///   neighbours, which omit the gauge reach's own local drainage.
    pub outflow_idx: Vec<Vec<usize>>,
    /// Reach-subdivision parent map in **compressed** space, length
    /// `n_parent_active + 1`: compressed rows
    /// `[parent_offset[p], parent_offset[p + 1])` are the sub-reach pieces of
    /// the `p`-th parent present in this batch. Feeds
    /// `SparseAdjacency::parent_offset`, which the engine turns into the
    /// per-row lateral-inflow divisor.
    ///
    /// `None` when the caller passed no CONUS parent map. Identity
    /// (`0..=n_active`) when the store is not subdivided — the engine treats
    /// both alike and skips the split.
    pub parent_offset: Option<Vec<i32>>,
}

/// Parent index owning sub-reach row `row`. `parent_offset` is strictly
/// increasing, so the owner is the last offset that is `<= row`.
#[inline]
fn parent_of_row(parent_offset: &[i32], row: usize) -> usize {
    parent_offset.partition_point(|&o| o <= row as i32) - 1
}

/// Re-express a CONUS-space `parent_offset` in compressed space.
///
/// `active` is the sorted list of CONUS sub-reach positions this batch kept.
/// A parent's pieces occupy a contiguous ascending run of CONUS rows, and a
/// gauge subgraph always enters a parent at its outlet and then walks the
/// internal chain upstream — so every parent present in `active` must be
/// present *in full*, as an unbroken run. That is asserted here rather than
/// assumed: a partially-present parent would hand the engine a piece count
/// `m` smaller than the one the piece lengths were derived from (silently
/// creating mass) and would move the parent's outlet row.
fn compressed_parent_offset(
    active: &[usize],
    conus_parent_offset: &[i32],
) -> Result<Vec<i32>> {
    let n_rows = *conus_parent_offset.last().unwrap_or(&0);
    let mut offsets: Vec<i32> = vec![0];
    let mut i = 0usize;
    while i < active.len() {
        let row = active[i];
        if row as i32 >= n_rows {
            return Err(DataError::Malformed {
                path: PathBuf::from("<collate>"),
                message: format!(
                    "compress: CONUS row {row} is outside the parent map \
                     (which covers {n_rows} rows)"
                ),
            });
        }
        let p = parent_of_row(conus_parent_offset, row);
        let lo = conus_parent_offset[p] as usize;
        let hi = conus_parent_offset[p + 1] as usize;
        let m = hi - lo;
        if active.len() - i < m || (0..m).any(|k| active[i + k] != lo + k) {
            return Err(DataError::Malformed {
                path: PathBuf::from("<collate>"),
                message: format!(
                    "compress: parent {p} owns CONUS rows {lo}..{hi}, but the active \
                     set does not hold them contiguously from compressed row {i} — a \
                     partial sub-reach chain would mis-scale lateral inflow and move \
                     the gauge outlet"
                ),
            });
        }
        i += m;
        offsets.push(i as i32);
    }
    Ok(offsets)
}

/// Compress a unioned COO into dense compressed-position space, preserving
/// topological order via `BTreeSet` sort. The CONUS adjacency's `order`
/// array is itself topological — so a sorted subset stays topological.
///
/// Hard-asserts the lower-triangular invariant (`rows >= cols`); fails
/// with `DataError::Malformed` if violated.
///
/// `ddr_match` selects the `outflow_idx` convention (see step 5): `true`
/// reproduces DDR's `merit.py:226-234` bit-for-bit, `false` uses the
/// physically correct gauge-reach index. Comes from `params.ddr_match`.
///
/// `conus_parent_offset` is `ConusAdjacencyStore::parent_offset` — the
/// reach-subdivision map in CONUS sub-reach space. Pass it whenever it is
/// available (it is the identity `0..=n` on un-subdivided stores, which costs
/// nothing); `None` disables both the compressed parent map and the
/// outlet-piece resolution below, which is what the pure-topology unit tests
/// want.
pub fn compress(
    unioned: &UnionedCoo,
    conus_order: &[Comid],
    ddr_match: bool,
    conus_parent_offset: Option<&[i32]>,
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

    // 4b. Reach-subdivision parent map, re-expressed in compressed space.
    let parent_offset: Option<Vec<i32>> = match conus_parent_offset {
        Some(off) => Some(compressed_parent_offset(&active_vec, off)?),
        None => None,
    };

    // 5. outflow_idx — which reaches are summed to form a gauge's prediction.
    //
    // `ddr_match: false` (CORRECT) — the gauge's OWN reach, read at the
    // OUTLET piece of that reach.
    // A USGS gauge measures every drop of drainage above it, including the
    // lateral inflow of the reach the gauge sits on, and we do not know where
    // along that reach the gauge physically sits. The Muskingum-Cunge solve at
    // the gauge reach already accumulates all upstream contributions plus its
    // own `q_prime` by mass conservation, so the gauge's prediction is that
    // ONE reach's routed discharge.
    //
    // Under reach subdivision that reach spans several rows. The solve is
    // mass-conserving down the internal chain, so the whole reach's runoff —
    // upstream network plus its own lateral inflow, which was split `q'/m`
    // across the pieces — only arrives at the LAST piece,
    // `parent_offset[p + 1] - 1`. Reading an earlier piece would drop the
    // downstream fraction of the reach's own lateral inflow: the same class of
    // bug as the upstream-cols defect below, in a new form.
    //
    // `gauge_compressed` holds COMPRESSED SUB-REACH positions, not parent
    // indices, so the parent must be recovered from the *compressed*
    // `parent_offset` before its outlet can be taken. (Indexing the CONUS
    // parent map with a compressed row is a category error — the compression
    // renumbers rows.) In practice `outlet == gauge_compressed[g]` already,
    // because the gauge subgraph builder resolves a gauge COMID to its last
    // matching row (`cache.rs::resolve_or_build`); deriving it here keeps the
    // guarantee local instead of resting on that builder detail.
    //
    // `ddr_match: true` (DEFAULT, DDR-FAITHFUL) — the gauge's UPSTREAM
    // neighbours. Reproduces DDR's `_collate_gages`
    // (`~/projects/ddr/src/ddr/geodatazoo/merit.py:226-234`), which collects
    // the COO *cols* whose row equals the gauge outlet and only falls back to
    // the gauge's own index when that list is empty. In this adjacency
    // `indices_0` = rows = DOWNSTREAM segment and `indices_1` = cols =
    // UPSTREAM segment (`src/data/store/zarr.rs:39-42`), so this silently
    // drops the gauge reach's own local runoff from every prediction.
    //
    // Why `false` is the physical answer: gauge 01457000 (366.8 km² drainage,
    // of which the gauge reach alone is 250.1 km² = 68%) read 1.58 m³/s
    // against an observed 7.60 and a summed-Q' baseline of 7.38 — a constant
    // 0.215× suppression across all 15 eval years, on peaks as well as means.
    // 26 of 1841 gauges fell below 0.5× baseline, all of them small basins
    // where the gauge reach is a large share of the area. Because the omitted
    // mass is always positive, this biases EVERY ddrs-vs-baseline comparison
    // against ddrs.
    //
    // DDR's Lynker path validates `outflow_idx` against the flowpath `toid`
    // column (`~/projects/ddr/src/ddr/geodatazoo/lynker_hydrofabric.py:239-250`);
    // the MERIT path has no such check — that is where this would have been
    // caught upstream.
    let outflow_idx: Vec<Vec<usize>> = if ddr_match {
        gauge_compressed
            .iter()
            .map(|&g_comp| {
                let g_row = g_comp as i32;
                let cols_for_g: Vec<usize> = rows
                    .iter()
                    .zip(cols.iter())
                    .filter(|(r, _)| **r == g_row)
                    .map(|(_, c)| *c as usize)
                    .collect();
                if cols_for_g.is_empty() {
                    vec![g_comp]
                } else {
                    cols_for_g
                }
            })
            .collect()
    } else {
        match &parent_offset {
            Some(off) => gauge_compressed
                .iter()
                .map(|&g_comp| {
                    let p = parent_of_row(off, g_comp);
                    vec![off[p + 1] as usize - 1]
                })
                .collect(),
            None => gauge_compressed.iter().map(|&g_comp| vec![g_comp]).collect(),
        }
    };

    Ok(CompressedAdj {
        divide_comids,
        rows,
        cols,
        gauge_compressed,
        outflow_idx,
        parent_offset,
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
        let c = compress(&unioned, &conus_order, true, None).expect("compress");
        // Active = {0, 1, 2, 3, 4} → all 5. Compressed positions match.
        assert_eq!(c.divide_comids, conus_order);
        assert_eq!(c.rows, vec![2, 3, 4, 4]);
        assert_eq!(c.cols, vec![0, 1, 2, 3]);
        assert_eq!(c.gauge_compressed, vec![4, 3]);
        // Pins the `ddr_match: true` (DDR-faithful) convention: outflow_idx is
        // the gauge's UPSTREAM cols — gauge A at row 4 receives from cols 2, 3;
        // gauge B at row 3 from col 1. Both omit the gauge reach itself; see
        // `outflow_idx_includes_the_gauge_reach_when_not_ddr_match` for the
        // corrected convention.
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
        let c = compress(&unioned, &conus_order, true, None).expect("compress");
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
        let err = compress(&unioned, &conus_order, true, None).unwrap_err();
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
        let err = compress(&unioned, &conus_order, true, None).unwrap_err();
        match err {
            crate::data::error::DataError::Malformed { .. } => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn outflow_idx_includes_the_gauge_reach_when_not_ddr_match() {
        // We do NOT know where along its reach a gauge physically sits, so the
        // gauge reach's own lateral inflow MUST be counted. The MC solve at that
        // reach already accumulates everything upstream by mass conservation, so
        // the gauge's prediction is that ONE reach.
        //
        // Regression: gauge 01457000 (366.8 km2; its own reach is 250.1 km2 = 68%
        // of the basin) read 1.58 m3/s against an observed 7.60 and a summed-Q'
        // baseline of 7.38 -- a constant 0.215x suppression for 15 straight years,
        // affecting 26/1841 gauges and biasing every ddrs-vs-baseline comparison.
        //
        // The affected case is a gauge that HAS incoming edges: two headwaters
        // (CONUS positions 0, 1) draining into the gauge reach (position 2) --
        // exactly the 01457000 topology.
        let conus_order = vec![Comid(73006562), Comid(73006585), Comid(73005764)];
        let unioned = UnionedCoo {
            edges: vec![(2, 0), (2, 1)],
            gauges: vec![(Staid::new("01457000"), 2, "73005764".to_string())],
        };

        let corrected = compress(&unioned, &conus_order, false, None).expect("compress");
        assert_eq!(corrected.gauge_compressed, vec![2]);
        assert_eq!(
            corrected.outflow_idx[0],
            vec![2],
            "ddr_match=false: outflow_idx must be the gauge's OWN reach, not its \
             upstream cols [0, 1]"
        );

        // And the DDR-faithful path is preserved byte-for-byte under the flag.
        let ddr = compress(&unioned, &conus_order, true, None).expect("compress");
        assert_eq!(ddr.gauge_compressed, vec![2]);
        assert_eq!(
            ddr.outflow_idx[0],
            vec![0, 1],
            "ddr_match=true must reproduce DDR merit.py:226-234 (upstream cols)"
        );
    }

    #[test]
    fn outflow_idx_falls_back_to_self_when_no_incoming_edges() {
        // Gauge at CONUS-position 2 with no upstream edges in this batch
        // (active = {2} as a single-node graph). Pins the `ddr_match: true`
        // convention: DDR's empty-cols fallback yields the gauge's own
        // compressed index. Under `ddr_match: false` this is not a fallback at
        // all -- it is the general rule -- so both flag values agree here.
        let conus_order = vec![Comid(0), Comid(1), Comid(2)];
        let unioned = UnionedCoo {
            edges: vec![],
            gauges: vec![(Staid::new("0000000A"), 2, "comid2".to_string())],
        };
        let ddr = compress(&unioned, &conus_order, true, None).expect("compress");
        assert_eq!(ddr.gauge_compressed, vec![0]);
        assert_eq!(ddr.outflow_idx[0], vec![0], "self-edge fallback");

        let corrected = compress(&unioned, &conus_order, false, None).expect("compress");
        assert_eq!(corrected.outflow_idx[0], vec![0]);
    }

    /// Subdivided CONUS space: parents 0 and 1 are single-piece headwaters,
    /// parent 2 (the gauge reach) is split 4 ways into rows 2..6.
    ///
    ///     0 ─┐
    ///        ├─> 2 → 3 → 4 → 5   (gauge reach, outlet = 5)
    ///     1 ─┘
    fn subdivided_gauge_reach() -> (Vec<Comid>, Vec<i32>, UnionedCoo) {
        let conus_order = vec![
            Comid(73006562),
            Comid(73006585),
            Comid(73005764),
            Comid(73005764),
            Comid(73005764),
            Comid(73005764),
        ];
        let conus_parent_offset = vec![0, 1, 2, 6];
        let unioned = UnionedCoo {
            // Two external edges onto the gauge reach's INLET piece, plus the
            // internal chain 2→3→4→5.
            edges: vec![(2, 0), (2, 1), (3, 2), (4, 3), (5, 4)],
            // The subgraph builder resolves a gauge COMID to its parent's last
            // row, so `gage_idx` is already the outlet piece.
            gauges: vec![(Staid::new("01457000"), 5, "73005764".to_string())],
        };
        (conus_order, conus_parent_offset, unioned)
    }

    #[test]
    fn compressed_parent_offset_is_the_conus_map_renumbered() {
        let (order, off, unioned) = subdivided_gauge_reach();
        let c = compress(&unioned, &order, false, Some(&off)).expect("compress");
        // Active set is all 6 rows here, so compression is the identity and the
        // compressed map equals the CONUS map. What matters is the CONTRACT:
        // the map is expressed in the same space as `rows`/`cols`.
        assert_eq!(c.parent_offset, Some(vec![0, 1, 2, 6]));
        assert_eq!(c.divide_comids.len(), 6);
    }

    #[test]
    fn outflow_idx_reads_the_outlet_piece_of_a_subdivided_gauge_reach() {
        let (order, off, unioned) = subdivided_gauge_reach();
        let c = compress(&unioned, &order, false, Some(&off)).expect("compress");
        assert_eq!(c.gauge_compressed, vec![5]);
        assert_eq!(
            c.outflow_idx[0],
            vec![5],
            "the gauge's whole reach discharges at its LAST piece (compressed row \
             `parent_offset[p + 1] - 1` = 5); pieces 2, 3, 4 each omit part of the \
             reach's own lateral inflow"
        );
    }

    #[test]
    fn compress_rejects_a_partial_sub_reach_chain() {
        // Every edge touching row 2 is gone, so the gauge reach's inlet piece
        // is missing from the active set. The parent's chain is then only 3 of
        // its 4 rows — the engine would divide q' by 3 while the lengths were
        // cut for 4, creating mass. That must be an error, not a silent pass.
        let (order, off, _) = subdivided_gauge_reach();
        let unioned = UnionedCoo {
            edges: vec![(4, 3), (5, 4)],
            gauges: vec![(Staid::new("01457000"), 5, "73005764".to_string())],
        };
        let err = compress(&unioned, &order, false, Some(&off)).unwrap_err();
        match err {
            crate::data::error::DataError::Malformed { message, .. } => {
                assert!(
                    message.contains("contiguously"),
                    "expected a partial-chain diagnostic, got: {message}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn identity_parent_offset_leaves_outflow_idx_at_the_gauge_reach() {
        // Un-subdivided store: `parent_offset` is `0..=n`, so every parent owns
        // exactly one row and the outlet resolution must be a no-op.
        let conus_order = vec![Comid(73006562), Comid(73006585), Comid(73005764)];
        let unioned = UnionedCoo {
            edges: vec![(2, 0), (2, 1)],
            gauges: vec![(Staid::new("01457000"), 2, "73005764".to_string())],
        };
        let identity: Vec<i32> = vec![0, 1, 2, 3];
        let c = compress(&unioned, &conus_order, false, Some(&identity)).expect("compress");
        assert_eq!(c.parent_offset, Some(vec![0, 1, 2, 3]));
        assert_eq!(c.outflow_idx[0], vec![2]);
        // Byte-identical to passing no map at all.
        let bare = compress(&unioned, &conus_order, false, None).expect("compress");
        assert_eq!(c.outflow_idx, bare.outflow_idx);
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
