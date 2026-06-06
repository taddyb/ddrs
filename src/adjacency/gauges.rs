//! Build per-gauge upstream subgraphs from the CONUS adjacency.
//!
//! Port of `build_gauge_adjacencies` (`engine/src/ddr_engine/merit/build.py:206-290`),
//! `subset_upstream` (`engine/src/ddr_engine/merit/graph.py:89-118`) and
//! `create_subset_coo` (`engine/src/ddr_engine/merit/io.py:96-147`).
//!
//! ## What the engine emits per gauge (the compatibility contract)
//!
//! `coo_to_zarr_group_generic` (`core/zarr_io.py:337-393`) writes, per STAID
//! subgroup:
//!   - `indices_0` = `coo.row` = CONUS position of the **downstream** reach,
//!   - `indices_1` = `coo.col` = CONUS position of the **upstream** reach,
//!   - `order`     = the subset COMIDs (`subset_upstream` output),
//!   - `values`    = uint8 ones (written by Task 5, not modelled here),
//!   - attr `gage_catchment` = origin COMID,
//!   - attr `gage_idx`       = `merit_mapping[origin]` (origin's CONUS position).
//!
//! The ddrs reader (`src/data/store/zarr.rs:102-116`, `GageSubgraph`) consumes
//! `indices_0`/`indices_1` (CONUS position space) + the two attrs; it recovers
//! the node set from the union of the COO indices, sorted by CONUS position
//! (`GageSubgraph::upstream_comids`, zarr.rs:127-135).
//!
//! ## Determinism note (parity vs the engine)
//!
//! `subset_upstream` builds its COMID list as `rx.ancestors(...)` (a Python
//! `set`, iterated in **non-deterministic** order) `+ [origin]` (graph.py:115-118).
//! `create_subset_coo` then iterates that list to emit edges, so the engine's
//! `order` array and the *row order* of `indices_0`/`indices_1` are not stable
//! across runs. The reader is set-based (sorts by CONUS position), so the
//! **node set** and `gage_idx` are the invariants that matter — and those ARE
//! deterministic. We therefore emit `order` / COO rows sorted by ascending
//! CONUS position: identical node set + attrs to the engine, in a canonical
//! order. Task 8's parity test compares node sets and `gage_idx`, not row order.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::adjacency::build::ConusAdjacency;
use crate::data::error::Result;
use crate::data::ids::Staid;
use crate::data::store::gage_csv::GageMetadata;

/// One gauge's upstream subgraph, in CONUS position space.
///
/// `gage_idx`/`gage_catchment` and `indices_0`/`indices_1` mirror the
/// `GageSubgraph` reader contract (`src/data/store/zarr.rs:102-116`); `order`
/// is the subset's COMIDs (for Task 5's `order` array), sorted by CONUS
/// position (see module-level determinism note).
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeSubgraph {
    /// USGS station ID (zero-padded to 8 chars via `Staid`).
    pub staid: Staid,
    /// Outlet COMID (`gage_catchment` attr; from the gauges CSV `COMID` column).
    pub gage_catchment: i64,
    /// Outlet's position in the CONUS `order` array (`gage_idx` attr).
    pub gage_idx: usize,
    /// Subset COMIDs in ascending CONUS-position order (the `order` array).
    pub order: Vec<i32>,
    /// COO `indices_0` — downstream CONUS position.
    pub rows: Vec<i32>,
    /// COO `indices_1` — upstream CONUS position.
    pub cols: Vec<i32>,
}

/// Build per-gauge upstream subgraphs from a CONUS adjacency and the gauges CSV.
///
/// Mirrors `build_gauge_adjacencies` (`build.py:206-290`): read+validate the
/// gauges (STAID zfilled to 8, COMID required — `MERITGauge`), then per gauge
/// in CSV order, BFS upstream over the CONUS graph and emit the COO.
///
/// Gauges whose COMID is missing from the CONUS network are skipped with a
/// warning (mirrors `build.py:267-270`'s `root.__delitem__` + the reader's
/// silent drop in `GagesAdjacencyStore::open`, zarr.rs:160-164). Gauge rows
/// whose CSV `COMID` column is empty are likewise skipped.
pub fn build_gauge_subgraphs(
    conus: &ConusAdjacency,
    gages_csv: &Path,
) -> Result<Vec<GaugeSubgraph>> {
    // Read+validate the gauges CSV. `GageMetadata` already zfills STAID to 8
    // (matching `MERITGauge`'s `zfill_staid`, dataclasses.py:26-30) and parses
    // the COMID column. Iteration follows CSV row order, the same as
    // `gauge_set.gauges` (build.py:257). NOTE: `GageMetadata` requires the full
    // gage-info column set (STANAME/LAT_GAGE/LNG_GAGE), which is *stricter* than
    // `MERITGauge`'s `extra="ignore"` (STAID/DRAIN_SQKM/COMID only). The
    // production gauges CSV (`gages_3000.csv`) carries every column, so reusing
    // the existing reader over a duplicate STAID/COMID parser is the surgical
    // choice; a barer CSV would error here rather than parse partially.
    let meta = GageMetadata::open(gages_csv)?;

    // merit_mapping: COMID -> CONUS position (build.py:252).
    let position: HashMap<i64, usize> = conus.position_lookup();

    // Upstream adjacency over the CONUS graph in position space: for each COO
    // edge, `cols[k]` (upstream pos) feeds `rows[k]` (downstream pos). We invert
    // it to "downstream -> [upstream]" so BFS from the outlet walks upstream.
    // This is the same edge set `rx.ancestors` walks (the CONUS COO and the
    // gauge graph are both `build_graph(build_upstream_dict(fp))`).
    let upstream_of = build_upstream_adjacency(conus);

    let mut subgraphs = Vec::new();
    for row in &meta.rows {
        let staid = row.staid.clone();
        // MERITGauge requires COMID; a row without it can't be a MERIT gauge.
        let origin_comid = match row.comid {
            Some(c) => c,
            None => {
                eprintln!(
                    "warning: gauge {staid} has no COMID in {}; skipping",
                    gages_csv.display()
                );
                continue;
            }
        };

        // COMID not in the CONUS network -> skip (build.py:267-270).
        let origin_pos = match position.get(&origin_comid) {
            Some(&p) => p,
            None => {
                eprintln!(
                    "warning: COMID {origin_comid} for gauge {staid} not found in \
                     CONUS adjacency; skipping"
                );
                continue;
            }
        };

        // subset_upstream: ancestors(origin) ∪ {origin}, in CONUS position space
        // (graph.py:89-118). BFS upstream over `upstream_of`.
        let subset_positions = subset_upstream(origin_pos, &upstream_of);

        // create_subset_coo (io.py:96-147): for each subset node, emit an edge
        // to each of its in-subset downstream neighbours. Walking our
        // downstream->upstream `upstream_of` map: an edge (ds, us) exists when
        // both endpoints are in the subset. Equivalent to the engine's
        // successor scan; emitted in ascending downstream-position order for
        // determinism.
        let (rows, cols) = subset_coo(&subset_positions, &upstream_of);

        // `order` array = subset COMIDs (engine stores subset_comids). We sort
        // by CONUS position for a canonical, deterministic order (see module
        // determinism note); node set is identical to the engine's.
        let order: Vec<i32> = subset_positions
            .iter()
            .map(|&pos| conus.order[pos])
            .collect();

        subgraphs.push(GaugeSubgraph {
            staid,
            gage_catchment: origin_comid,
            gage_idx: origin_pos,
            order,
            rows,
            cols,
        });
    }

    Ok(subgraphs)
}

/// Downstream CONUS position -> sorted upstream CONUS positions.
///
/// Inverts the CONUS COO (`rows` = downstream, `cols` = upstream) so BFS from
/// the outlet walks upstream. Sorted+deduped for deterministic traversal.
fn build_upstream_adjacency(conus: &ConusAdjacency) -> HashMap<usize, Vec<usize>> {
    let mut up: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&r, &c) in conus.rows.iter().zip(conus.cols.iter()) {
        up.entry(r as usize).or_default().push(c as usize);
    }
    for ups in up.values_mut() {
        ups.sort_unstable();
        ups.dedup();
    }
    up
}

/// Positions of every reach upstream of `origin_pos`, including the origin.
///
/// Mirrors `subset_upstream` (graph.py:89-118): all ancestors of the outlet
/// plus the outlet itself. Returned ascending by CONUS position (the engine
/// returns a non-deterministically-ordered list; we canonicalise — see the
/// module determinism note). A node with no upstream returns `[origin_pos]`,
/// matching `subset_upstream`'s headwater branch.
fn subset_upstream(origin_pos: usize, upstream_of: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
    let mut visited: BTreeSet<usize> = BTreeSet::new();
    visited.insert(origin_pos);
    let mut frontier = vec![origin_pos];
    while let Some(node) = frontier.pop() {
        if let Some(ups) = upstream_of.get(&node) {
            for &u in ups {
                if visited.insert(u) {
                    frontier.push(u);
                }
            }
        }
    }
    visited.into_iter().collect()
}

/// COO edges (downstream, upstream) for the subset, in CONUS position space.
///
/// Mirrors `create_subset_coo` (io.py:96-147): an edge is kept iff both
/// endpoints are in the subset. Emitted in ascending downstream-position order
/// (then upstream-position) for determinism. Lower-triangular by construction
/// (downstream position >= upstream position in a topo-ordered CONUS array).
fn subset_coo(
    subset_positions: &[usize],
    upstream_of: &HashMap<usize, Vec<usize>>,
) -> (Vec<i32>, Vec<i32>) {
    let subset: BTreeSet<usize> = subset_positions.iter().copied().collect();
    let mut rows: Vec<i32> = Vec::new();
    let mut cols: Vec<i32> = Vec::new();
    // Iterate downstream positions ascending for a canonical row order.
    for &ds in subset_positions {
        if let Some(ups) = upstream_of.get(&ds) {
            for &us in ups {
                if subset.contains(&us) {
                    rows.push(ds as i32);
                    cols.push(us as i32);
                }
            }
        }
    }
    (rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Synthetic 6-reach dendritic network, already topo-ordered (headwaters
    /// first, outlet last) so it is lower-triangular. COMIDs == positions*10 to
    /// keep the COMID/position distinction visible in assertions.
    ///
    /// Topology (arrows point downstream):
    ///
    /// ```text
    ///   COMID 10 (pos0) ─┐
    ///                     ├─► COMID 30 (pos2) ─┐
    ///   COMID 20 (pos1) ─┘                     ├─► COMID 50 (pos4)
    ///   COMID 40 (pos3) ──────────────────────┘
    ///   COMID 60 (pos5)  (isolated)
    /// ```
    ///
    /// Edges (downstream pos, upstream pos): (2,0) (2,1) (4,2) (4,3).
    fn synthetic_conus() -> ConusAdjacency {
        ConusAdjacency {
            order: vec![10, 20, 30, 40, 50, 60],
            rows: vec![2, 2, 4, 4],
            cols: vec![0, 1, 2, 3],
            length_m: vec![100.0; 6],
            slope: vec![0.001; 6],
            dropped_comids: vec![],
        }
    }

    fn write_csv(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        f.write_all(contents.as_bytes()).expect("write");
        f.flush().expect("flush");
        f
    }

    #[test]
    fn two_gauges_node_sets_and_gage_idx() {
        // Gauge at COMID 50 (outlet, pos4) -> whole connected network.
        // Gauge at COMID 30 (pos2) -> {10,20,30}.
        let csv = "\
STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID
00000050,OUTLET,10.0,1.0,2.0,50
00000030,MID,5.0,1.0,2.0,30
";
        let f = write_csv(csv);
        let conus = synthetic_conus();
        let subs = build_gauge_subgraphs(&conus, f.path()).expect("build");

        assert_eq!(subs.len(), 2);

        // Gauge 50: outlet pos4, node set = positions {0,1,2,3,4} = COMIDs all.
        let g50 = &subs[0];
        assert_eq!(g50.staid, Staid::new("50"));
        assert_eq!(g50.gage_catchment, 50);
        assert_eq!(g50.gage_idx, 4);
        let node_set: BTreeSet<i32> = g50.order.iter().copied().collect();
        assert_eq!(node_set, BTreeSet::from([10, 20, 30, 40, 50]));
        // Edges cover the four CONUS edges between in-subset nodes.
        let edges: BTreeSet<(i32, i32)> =
            g50.rows.iter().copied().zip(g50.cols.iter().copied()).collect();
        assert_eq!(edges, BTreeSet::from([(2, 0), (2, 1), (4, 2), (4, 3)]));

        // Gauge 30: outlet pos2, ancestors {pos0,pos1} + self -> COMIDs {10,20,30}.
        let g30 = &subs[1];
        assert_eq!(g30.gage_catchment, 30);
        assert_eq!(g30.gage_idx, 2);
        let node_set: BTreeSet<i32> = g30.order.iter().copied().collect();
        assert_eq!(node_set, BTreeSet::from([10, 20, 30]));
        let edges: BTreeSet<(i32, i32)> =
            g30.rows.iter().copied().zip(g30.cols.iter().copied()).collect();
        assert_eq!(edges, BTreeSet::from([(2, 0), (2, 1)]));
    }

    #[test]
    fn headwater_gauge_has_self_only_no_edges() {
        // COMID 10 is a headwater (pos0, no upstream).
        let csv = "STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID\n00000010,HW,1.0,1.0,2.0,10\n";
        let f = write_csv(csv);
        let conus = synthetic_conus();
        let subs = build_gauge_subgraphs(&conus, f.path()).expect("build");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].order, vec![10]);
        assert!(subs[0].rows.is_empty());
        assert!(subs[0].cols.is_empty());
        assert_eq!(subs[0].gage_idx, 0);
    }

    #[test]
    fn missing_comid_in_network_is_skipped_others_survive() {
        // COMID 99999 is not in the network; COMID 30 is. The missing one is
        // dropped with a warning; the present one survives.
        let csv = "\
STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID
00099999,A,1.0,1.0,2.0,99999
00000030,B,5.0,1.0,2.0,30
";
        let f = write_csv(csv);
        let conus = synthetic_conus();
        let subs = build_gauge_subgraphs(&conus, f.path()).expect("build");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].gage_catchment, 30);
        assert_eq!(subs[0].gage_idx, 2);
    }

    #[test]
    fn row_without_comid_column_value_is_skipped() {
        // Empty COMID cell -> Option<i64> None -> skipped (no MERIT linkage).
        let csv = "STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID\n\
00000030,A,5.0,1.0,2.0,\n\
00000050,B,1.0,1.0,2.0,50\n";
        let f = write_csv(csv);
        let conus = synthetic_conus();
        let subs = build_gauge_subgraphs(&conus, f.path()).expect("build");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].gage_catchment, 50);
    }
}
