//! Per-batch subgraph union + compression helpers.
//!
//! Mirrors `~/projects/ddr/src/ddr/io/builders.py::construct_network_matrix`
//! (lines ~55-110) and the COO-build portion of
//! `~/projects/ddr/src/ddr/geodatazoo/merit.py::_collate_gages`
//! (lines ~245-285).

use std::collections::BTreeSet;

use crate::data::ids::Staid;
use crate::data::store::{GageSubgraph, GagesAdjacencyStore};

/// Output of the per-batch subgraph union. Edges are deduplicated and
/// returned in CONUS-position coordinates, sorted lex by `(row, col)`.
#[derive(Debug)]
pub(crate) struct UnionedCoo {
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
pub(crate) fn union_subgraphs(
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
}
