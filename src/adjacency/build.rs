//! Build the CONUS adjacency from raw MERIT flowpath records.
//!
//! Port of `ddr_engine/merit/build.py::create_adjacency_matrix`
//! (`engine/src/ddr_engine/merit/build.py:20-107`) and its graph helpers in
//! `engine/src/ddr_engine/merit/graph.py` (`build_upstream_dict:9-52`,
//! `build_graph:55-86`). The output `order` / `indices_0` / `indices_1` arrays
//! describe the SAME graph the engine writes (identical node set, edge set, and
//! cycle drops), so the two zarr stores are interchangeable for routing.
//!
//! ## Topological order is NOT byte-identical to the engine store (Task 8)
//!
//! The synthetic tests (`tests/adjacency_build.rs`) pin element-for-element
//! `order` parity on small graphs, but the real CONUS engine store's `order` is
//! a *different valid permutation* of ours — see `tests/adjacency_parity.rs`.
//! Root cause: the engine's `build_upstream_dict` returns a dict whose edge
//! iteration order is polars `group_by` order (hash-based, non-deterministic),
//! and rustworkx's `topological_sort` tie-breaks on edge-insertion order. The
//! engine's stored order is therefore an irreproducible artifact of one polars
//! run. Our LIFO-Kahn order (below) is deterministic and equally valid; the
//! load-bearing invariants are structural (node/edge sets, lower-triangular,
//! cycle drops), which `adjacency_parity.rs` asserts.
//!
//! ## Topological-sort parity
//!
//! The engine calls `rx.topological_sort` (rustworkx 0.17.1). That is Kahn's
//! algorithm with a **LIFO stack**: the stack is seeded with every
//! zero-indegree node pushed in node-index order `0..N`, then nodes are popped;
//! when a popped node's successor reaches indegree 0 it is pushed. We replicate
//! this exactly (verified empirically against rustworkx 0.17.1 on random DAGs —
//! a `BTreeMap`/COMID-keyed queue would NOT match). Node *index* order is the
//! tie-break key, so we must build nodes in the same insertion order the engine
//! does:
//!   1. iterate downstream COMIDs in **ascending** order (`sorted(keys)`),
//!   2. add the downstream node if unseen, then
//!   3. add each upstream node (upstream lists are sorted ascending) if unseen.
//!
//! Edges carry no tie-break weight here because MERIT is dendritic (one
//! successor per node), so per-node successor visit order is irrelevant.
//!
//! ## length_m / slope fill — deliberate divergence from the engine
//!
//! The engine's `write_merit_flowpath_attributes` (`build.py:110-161`) writes
//! `length_m = lengthkm * 1000` and raw `slope` to zarr **with NaN preserved**;
//! the NaN/inf fill happens later, at load time, in DDR's consumer
//! (`merit.py:73-80` builds `phys_means` via `naninfmean`, `merit.py:304-311`
//! calls `fill_nans`). Per the Managed-Adjacency plan ("the actual bug fix")
//! we move that fill into the builder so the stored arrays are already clean:
//! NaN **and** ±inf are replaced with the column mean over finite values
//! (`naninfmean`, `readers.py:365-379`). This is a documented divergence:
//!   - DDR's `fill_nans` (`readers.py:382-410`) only replaces `isnan`, leaving
//!     inf untouched; we additionally replace inf (the plan says "NaN/inf").
//!   - The engine's stored zarr therefore differs from ours on NaN/inf reaches.
//!     Parity for Task 8 is on the graph (`order`/`indices_0`/`indices_1`); the
//!     filled float columns are the intended fix, not a regression.

use std::collections::HashMap;

/// CONUS adjacency built from raw MERIT flowpaths.
///
/// `order` is the topological order of COMIDs (connected nodes first, then
/// isolated COMIDs appended sorted). `rows`/`cols` are the lower-triangular COO
/// (`row = downstream position`, `col = upstream position`) into `order`.
/// `length_m`/`slope` are aligned to `order` (f32, NaN/inf already filled).
#[derive(Debug, Clone)]
pub struct ConusAdjacency {
    /// COMIDs in topological order (engine stores this as int32).
    pub order: Vec<i32>,
    /// COO `indices_0` — downstream position in `order`.
    pub rows: Vec<i32>,
    /// COO `indices_1` — upstream position in `order`.
    pub cols: Vec<i32>,
    /// Per-reach channel length in metres, aligned to `order`.
    pub length_m: Vec<f32>,
    /// Per-reach channel slope (dimensionless), aligned to `order`.
    pub slope: Vec<f32>,
    /// COMIDs dropped because they sat on a simple cycle (for the manifest).
    pub dropped_comids: Vec<i64>,
}

impl ConusAdjacency {
    /// COMID → position in `order`. Built on demand for Task 4's subsetter.
    pub fn position_lookup(&self) -> HashMap<i64, usize> {
        self.order
            .iter()
            .enumerate()
            .map(|(idx, &comid)| (comid as i64, idx))
            .collect()
    }
}

/// Errors raised while building the adjacency. Cycle removal is *not* an error
/// (it is handled by recursion, mirroring the engine); a non-dendritic node is.
#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    /// A node has more than one successor — MERIT must be dendritic.
    /// Mirrors `build.py:94`'s assert.
    #[error("node COMID {comid} has {n_successors} successors, not dendritic")]
    NotDendritic { comid: i32, n_successors: usize },

    /// The lower-triangular invariant (routing invariant 3) was violated.
    /// Mirrors `build.py:105`'s assert.
    #[error("COO is not lower triangular: row {row} < col {col}")]
    NotLowerTriangular { row: i32, col: i32 },
}

/// Build the CONUS adjacency from raw flowpath records.
///
/// Mirrors `create_adjacency_matrix` (`build.py:20-107`) end to end, including
/// the `rx.DAGHasCycle` recursion that drops cycle COMIDs and retries.
pub fn build_conus_adjacency(
    records: &[crate::adjacency::dbf::FlowpathRecord],
) -> Result<ConusAdjacency, BuildError> {
    // Length/slope fill is over the FULL record set, not the cycle-dropped
    // subset: DDR computes `naninfmean` over the whole stored column
    // (merit.py:73-80) and the engine writes attributes for every COMID in
    // `order`. We align to `order` below; the means here use every finite
    // record value (cycle drops are rare — 6 of 346k — and excluding them
    // would shift the mean by < 1e-4, but matching the full-column semantics
    // is the safe parity choice).
    let comid_to_record: HashMap<i64, &crate::adjacency::dbf::FlowpathRecord> =
        records.iter().map(|r| (r.comid, r)).collect();

    let length_mean = naninfmean(records.iter().map(|r| r.lengthkm * 1000.0));
    let slope_mean = naninfmean(records.iter().map(|r| r.slope));

    // All COMIDs present in the fabric (engine: `fp["COMID"].values`,
    // build.py:78). Used for isolated-node detection.
    let all_comids: Vec<i64> = records.iter().map(|r| r.comid).collect();

    let (order_i64, rows, cols, dropped_comids) = build_topo_and_coo(records, &all_comids)?;

    // Engine stores COMIDs as int32 (build.py:128 comment, zarr.rs `/order`).
    let order: Vec<i32> = order_i64.iter().map(|&c| c as i32).collect();

    // length_m / slope aligned to `order`, with NaN/inf filled to the mean.
    // Mirrors write_merit_flowpath_attributes (build.py:143-151) for the
    // `* 1000.0` + alignment, plus the deliberate fill (see module docs).
    let mut length_m = Vec::with_capacity(order.len());
    let mut slope = Vec::with_capacity(order.len());
    for &comid in &order_i64 {
        let rec = comid_to_record.get(&comid);
        let (len_val, slope_val) = match rec {
            Some(r) => (r.lengthkm * 1000.0, r.slope),
            // A COMID can appear in `order` (as an upstream `up1..up4`) without
            // its own record row. The engine's comid_to_idx.get returns None
            // and the slot stays NaN (build.py:144-145) → then filled.
            None => (f64::NAN, f64::NAN),
        };
        length_m.push(fill_one(len_val, length_mean) as f32);
        slope.push(fill_one(slope_val, slope_mean) as f32);
    }

    Ok(ConusAdjacency {
        order,
        rows,
        cols,
        length_m,
        slope,
        dropped_comids,
    })
}

/// `(order_i64, rows, cols, dropped_comids)` — the structural output of the
/// topo/COO build, before length/slope alignment.
type TopoCoo = (Vec<i64>, Vec<i32>, Vec<i32>, Vec<i64>);

/// Build the topological order and lower-triangular COO, recursing on cycles.
///
/// Returns `(order_i64, rows, cols, dropped_comids)`. `order_i64` is connected
/// topo order followed by sorted isolated COMIDs. Mirrors `build.py:37-107`.
fn build_topo_and_coo(
    records: &[crate::adjacency::dbf::FlowpathRecord],
    all_comids: &[i64],
) -> Result<TopoCoo, BuildError> {
    // --- build_upstream_dict (graph.py:9-52) ---------------------------------
    // Map downstream COMID -> sorted-ascending unique upstream COMIDs, taken
    // ONLY from up1..up4 where up > 0. NextDownID is not used for edges here.
    let upstream_dict = build_upstream_dict(records);

    // --- build_graph (graph.py:55-86) ----------------------------------------
    // Node insertion order = sorted(downstream keys), each followed by its
    // (sorted) upstream COMIDs, skipping already-seen nodes. This insertion
    // order is the tie-break key for the topological sort below.
    let graph = build_graph(&upstream_dict);

    // --- rx.topological_sort with cycle handling (build.py:50-73) ------------
    match graph.topological_sort() {
        Ok(ts_order_idx) => {
            // Connected COMIDs in topo order (build.py:75).
            let id_order: Vec<i64> = ts_order_idx.iter().map(|&i| graph.node_data[i]).collect();

            // Isolated COMIDs: in the fabric but not in the connected graph,
            // appended sorted (build.py:77-83).
            let connected: std::collections::HashSet<i64> = id_order.iter().copied().collect();
            let mut isolated: Vec<i64> = all_comids
                .iter()
                .copied()
                .filter(|c| !connected.contains(c))
                .collect::<std::collections::HashSet<i64>>()
                .into_iter()
                .collect();
            isolated.sort_unstable();

            let mut full_order = id_order;
            full_order.extend(isolated);

            // idx_map: COMID -> position in full_order (build.py:85).
            let idx_map: HashMap<i64, i32> = full_order
                .iter()
                .enumerate()
                .map(|(idx, &c)| (c, idx as i32))
                .collect();

            // COO indices: for each node in topo order with out-degree > 0,
            // assert single successor, emit (row=ds_idx, col=us_idx)
            // (build.py:90-97).
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            for &node in &ts_order_idx {
                let succs = &graph.successors[node];
                if succs.is_empty() {
                    continue;
                }
                let comid = graph.node_data[node];
                if succs.len() != 1 {
                    return Err(BuildError::NotDendritic {
                        comid: comid as i32,
                        n_successors: succs.len(),
                    });
                }
                let ds_comid = graph.node_data[succs[0]];
                cols.push(idx_map[&comid]);
                rows.push(idx_map[&ds_comid]);
            }

            // Final lower-triangular assert (build.py:105, routing invariant 3).
            for (&r, &c) in rows.iter().zip(cols.iter()) {
                if r < c {
                    return Err(BuildError::NotLowerTriangular { row: r, col: c });
                }
            }

            Ok((full_order, rows, cols, Vec::new()))
        }
        Err(cycle_comids) => {
            // Drop every COMID on a simple cycle, then recurse on the filtered
            // records (build.py:52-73). Dropped COMIDs accumulate across the
            // recursion so the manifest sees all of them.
            let mut drop_set: std::collections::HashSet<i64> = cycle_comids.iter().copied().collect();
            let filtered: Vec<crate::adjacency::dbf::FlowpathRecord> = records
                .iter()
                .filter(|r| !drop_set.contains(&r.comid))
                .cloned()
                .collect();
            let filtered_all: Vec<i64> = filtered.iter().map(|r| r.comid).collect();

            let (order, rows, cols, mut more_dropped) =
                build_topo_and_coo(&filtered, &filtered_all)?;
            // Merge this level's drops with any from deeper recursion.
            for c in more_dropped.drain(..) {
                drop_set.insert(c);
            }
            let mut dropped: Vec<i64> = drop_set.into_iter().collect();
            dropped.sort_unstable();
            Ok((order, rows, cols, dropped))
        }
    }
}

/// Mirror of `build_upstream_dict` (graph.py:9-52).
///
/// Downstream COMID -> sorted-ascending list of upstream COMIDs from up1..up4
/// where `up > 0`. Duplicates within a downstream's upstreams are removed
/// (polars `group_by` + per-row uniqueness; in practice up1..up4 are distinct,
/// but a defensive dedup keeps node insertion order well-defined).
fn build_upstream_dict(
    records: &[crate::adjacency::dbf::FlowpathRecord],
) -> HashMap<i64, Vec<i64>> {
    let mut dict: HashMap<i64, Vec<i64>> = HashMap::new();
    for rec in records {
        for &up in &rec.up {
            if up > 0 {
                dict.entry(rec.comid).or_default().push(up);
            }
        }
    }
    for ups in dict.values_mut() {
        ups.sort_unstable();
        ups.dedup();
    }
    dict
}

/// A minimal directed graph mirroring the parts of `rx.PyDiGraph` the engine
/// uses: node data (COMID per index) and per-node successors, with a Kahn-stack
/// topological sort matching rustworkx 0.17.1.
struct DiGraph {
    /// COMID stored at each node index (insertion order = index order).
    node_data: Vec<i64>,
    /// Per-node successor node indices, in edge-insertion order.
    successors: Vec<Vec<usize>>,
    /// Per-node in-degree.
    indegree: Vec<usize>,
}

impl DiGraph {
    /// Build the graph from the upstream dict exactly as `build_graph`
    /// (graph.py:55-86) does: nodes in `sorted(keys)` order then each key's
    /// sorted upstreams; edges from upstream -> downstream.
    fn build(upstream_dict: &HashMap<i64, Vec<i64>>) -> Self {
        let mut node_index: HashMap<i64, usize> = HashMap::new();
        let mut node_data: Vec<i64> = Vec::new();

        let add_node = |comid: i64, node_index: &mut HashMap<i64, usize>, node_data: &mut Vec<i64>| {
            node_index.entry(comid).or_insert_with(|| {
                node_data.push(comid);
                node_data.len() - 1
            });
        };

        // Node insertion: sorted downstream keys, each followed by its upstreams.
        let mut keys: Vec<i64> = upstream_dict.keys().copied().collect();
        keys.sort_unstable();
        for &to_comid in &keys {
            add_node(to_comid, &mut node_index, &mut node_data);
            for &from_comid in &upstream_dict[&to_comid] {
                add_node(from_comid, &mut node_index, &mut node_data);
            }
        }

        // Edge insertion: iterate keys (deterministic via sorted keys), adding
        // from_comid -> to_comid. The engine iterates dict.items() (unordered),
        // but edge order does not affect the result for a dendritic graph;
        // using sorted keys keeps this fully deterministic.
        let n = node_data.len();
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indegree: Vec<usize> = vec![0; n];
        for &to_comid in &keys {
            let to_idx = node_index[&to_comid];
            for &from_comid in &upstream_dict[&to_comid] {
                let from_idx = node_index[&from_comid];
                successors[from_idx].push(to_idx);
                indegree[to_idx] += 1;
            }
        }

        DiGraph {
            node_data,
            successors,
            indegree,
        }
    }

    /// Kahn's algorithm with a LIFO stack, matching rustworkx 0.17.1
    /// `topological_sort` (verified empirically). Returns node indices in topo
    /// order, or `Err(cycle_comids)` listing every COMID on a simple cycle.
    fn topological_sort(&self) -> Result<Vec<usize>, Vec<i64>> {
        let n = self.node_data.len();
        let mut indegree = self.indegree.clone();
        // Seed: zero-indegree nodes pushed in ascending node-index order, so
        // popping yields descending index order (rustworkx behaviour).
        let mut stack: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(node) = stack.pop() {
            order.push(node);
            for &succ in &self.successors[node] {
                indegree[succ] -= 1;
                if indegree[succ] == 0 {
                    stack.push(succ);
                }
            }
        }
        if order.len() == n {
            Ok(order)
        } else {
            // Cycle: collect COMIDs of every node on a simple cycle. The engine
            // uses `rx.simple_cycles`; the union of all simple-cycle node sets
            // is exactly the set of nodes still having indegree > 0 plus those
            // reachable only through them — i.e. every node NOT emitted that is
            // part of a strongly-connected component of size > 1 or feeds one.
            // The engine drops every COMID returned by simple_cycles; that set
            // equals the nodes participating in cycles. We compute it directly
            // via Tarjan's SCC and take all nodes in any SCC of size > 1, plus
            // self-loops.
            Err(self.cycle_comids())
        }
    }

    /// COMIDs on simple cycles, matching the union of `rx.simple_cycles` node
    /// sets (build.py:55-63). A COMID is on a simple cycle iff it belongs to a
    /// strongly-connected component of size > 1, or has a self-loop.
    fn cycle_comids(&self) -> Vec<i64> {
        let sccs = self.strongly_connected_components();
        let mut on_cycle: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for scc in &sccs {
            if scc.len() > 1 {
                on_cycle.extend(scc.iter().copied());
            }
        }
        // Self-loops form a size-1 SCC that is still a cycle.
        for (idx, succs) in self.successors.iter().enumerate() {
            if succs.contains(&idx) {
                on_cycle.insert(idx);
            }
        }
        let mut comids: Vec<i64> = on_cycle.iter().map(|&i| self.node_data[i]).collect();
        comids.sort_unstable();
        comids
    }

    /// Tarjan's strongly-connected-components (iterative).
    fn strongly_connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_data.len();
        let mut index_counter = 0usize;
        let mut indices = vec![usize::MAX; n];
        let mut lowlink = vec![0usize; n];
        let mut on_stack = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut sccs: Vec<Vec<usize>> = Vec::new();

        // Iterative DFS with an explicit work stack of (node, successor cursor).
        for start in 0..n {
            if indices[start] != usize::MAX {
                continue;
            }
            let mut work: Vec<(usize, usize)> = vec![(start, 0)];
            while let Some(&(v, ci)) = work.last() {
                if ci == 0 {
                    indices[v] = index_counter;
                    lowlink[v] = index_counter;
                    index_counter += 1;
                    stack.push(v);
                    on_stack[v] = true;
                }
                if ci < self.successors[v].len() {
                    let w = self.successors[v][ci];
                    work.last_mut().unwrap().1 += 1;
                    if indices[w] == usize::MAX {
                        work.push((w, 0));
                    } else if on_stack[w] {
                        lowlink[v] = lowlink[v].min(indices[w]);
                    }
                } else {
                    // Done with v: propagate lowlink to parent, maybe close SCC.
                    if lowlink[v] == indices[v] {
                        let mut scc = Vec::new();
                        loop {
                            let w = stack.pop().unwrap();
                            on_stack[w] = false;
                            scc.push(w);
                            if w == v {
                                break;
                            }
                        }
                        sccs.push(scc);
                    }
                    work.pop();
                    if let Some(&(parent, _)) = work.last() {
                        lowlink[parent] = lowlink[parent].min(lowlink[v]);
                    }
                }
            }
        }
        sccs
    }
}

/// Thin constructor so callers read `build_graph(&dict)` like the Python.
fn build_graph(upstream_dict: &HashMap<i64, Vec<i64>>) -> DiGraph {
    DiGraph::build(upstream_dict)
}

/// Mean over finite values only (excludes NaN and ±inf).
/// Mirrors `naninfmean` (readers.py:365-379). Returns NaN if no finite values.
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

/// Replace a non-finite value (NaN or ±inf) with `mean`. See module docs for
/// the deliberate inf handling vs DDR's NaN-only `fill_nans`.
fn fill_one(value: f64, mean: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        mean
    }
}
