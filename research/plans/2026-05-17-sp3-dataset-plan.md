# SP-3 Dataset + Collate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `MeritGagesDataset` + per-batch `collate` that fuses SP-1
(attributes, gage CSV, statistics), SP-2 (icechunk streamflow + obs), and
the existing adjacency stores into a `RoutingBatch` consumable by SP-4's
training loop.

**Architecture:** Three new modules under `src/data/`:
`dataset.rs` (orchestrator), `collate.rs` (pure-math helpers), and
`sampler.rs` (RandomSampler + SequentialSampler). One config extension
adding `DataSources` + `Experiment` sections. Per-batch output is plain
`ndarray::Array2<f32>` + `SparseAdjacency`; BURN tensor materialization
happens at SP-4's device boundary.

**Tech Stack:** Existing crates only — `ndarray`, `serde`, `serde_yaml`,
`rand`, `chrono`, `std::collections::{BTreeSet, HashMap}`.

**Spec:** `.claude/specs/2026-05-17-sp3-dataset-design.md`
**Parent:** `.claude/specs/2026-05-17-train_and_test-replication-design.md`

**DDR reference (read-only, cite line numbers):**
- `~/projects/ddr/src/ddr/geodatazoo/merit.py::Merit::_collate_gages` (~245-330)
- `~/projects/ddr/src/ddr/geodatazoo/merit.py::_build_common_tensors` (~175-220)
- `~/projects/ddr/src/ddr/io/builders.py::construct_network_matrix` (~55-110)
- `~/projects/ddr/src/ddr/io/readers.py::build_flow_scale_tensor` (~270-330)
- `~/projects/ddr/src/ddr/io/readers.py::filter_gages_by_da_valid` (~185-220)
- `~/projects/ddr/src/ddr/io/readers.py::filter_headwater_gages` (~235-265)

---

## File Structure

**Created:**
- `src/data/dataset.rs` — `MeritGagesDataset`, `RoutingBatch`
- `src/data/collate.rs` — `union_subgraphs`, `compress`, `build_flow_scale`
- `src/data/sampler.rs` — `RandomSampler<R>`, `SequentialSampler`
- `tests/data_dataset.rs` — V2 integration test against live stores

**Modified:**
- `src/config.rs` — extend with `DataSources`, `Experiment` sections + YAML parsing helper
- `src/data/mod.rs` — wire new modules + re-exports

**No new dependencies.**

---

### Task 1: Extend `Config` with `DataSources` + `Experiment` + YAML loader

**Files:**
- Modify: `src/config.rs`

The existing `Config` struct only models the routing-engine knobs. SP-3
needs the full surface DDR's `config/merit_training_config.yaml` carries.

- [ ] **Step 1: Write a failing test that loads the production YAML**

Append to `src/config.rs` (inside a new `#[cfg(test)] mod tests` block at
file end, OR create one if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_merit_training_yaml() {
        let path = "config/merit_training.yaml";
        let cfg = Config::from_yaml_file(path).expect("load yaml");
        assert_eq!(cfg.experiment.batch_size, 64);
        assert_eq!(cfg.experiment.rho, Some(90));
        assert_eq!(cfg.experiment.warmup, 5);
        assert!(cfg.data_sources.conus_adjacency.ends_with("merit_conus_adjacency.zarr"));
        assert!(cfg.data_sources.streamflow.ends_with(".ic"));
    }

    #[test]
    fn existing_params_still_round_trip() {
        // Sanity: pre-existing default Config still constructs.
        let cfg = Config::default();
        assert!(cfg.params.parameter_ranges.n[0] > 0.0);
    }
}
```

- [ ] **Step 2: Verify it fails**

```
cargo test --lib config::tests::loads_merit_training_yaml
```

Expected: compile error — `from_yaml_file`, `data_sources`, `experiment`
don't exist.

- [ ] **Step 3: Extend `Config`**

In `src/config.rs`, add new types and update `Config`. Place above the
`#[cfg(test)]` block:

```rust
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::data::error::{DataError, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct DataSources {
    pub attributes: PathBuf,
    pub conus_adjacency: PathBuf,
    pub gages_adjacency: PathBuf,
    pub streamflow: PathBuf,
    pub observations: PathBuf,
    pub gages: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Experiment {
    pub batch_size: usize,
    /// Daily date strings in "YYYY/MM/DD" format (DDR convention).
    pub start_time: String,
    pub end_time: String,
    pub epochs: usize,
    /// Number of consecutive days in each training window; `None` for
    /// full-period testing.
    pub rho: Option<usize>,
    #[serde(default)]
    pub shuffle: bool,
    pub warmup: usize,
    /// Mapping `epoch → lr`. Serde reads this as `{1: 0.001, 3: 0.0005}`.
    #[serde(default)]
    pub learning_rate: std::collections::BTreeMap<usize, f32>,
    #[serde(default)]
    pub grad_clip_max_norm: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MlpConfigSection {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub input_var_names: Vec<String>,
    pub learnable_parameters: Vec<String>,
}
```

Then change the root `Config` struct to:

```rust
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub params: Params,
    #[serde(default)]
    pub data_sources: Option<DataSources>,
    #[serde(default)]
    pub experiment: Option<Experiment>,
    #[serde(default)]
    pub mlp: Option<MlpConfigSection>,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_geodataset")]
    pub geodataset: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_seed")]
    pub np_seed: u64,
}

fn default_mode() -> String { "training".to_string() }
fn default_geodataset() -> String { "merit".to_string() }
fn default_seed() -> u64 { 42 }

impl Config {
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| DataError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        serde_yaml::from_slice(&bytes).map_err(|e| DataError::Yaml {
            path: path.to_path_buf(),
            source: e,
        })
    }
}
```

You also need `Params` to deserialize from YAML. Add `#[derive(Deserialize)]`
+ `#[serde(default)]` to `Params`, `ParameterRanges`, `AttributeMinimums`,
and adjust field rename/default attributes as needed.

Inspect `config/merit_training.yaml` to match the structure:

```
mode: training
geodataset: merit
seed: 42
np_seed: 42
data_sources: { ... }
experiment: { ... }
mlp: { ... }
params:
  parameter_ranges: { n: [0.015, 0.25], ... }
  attribute_minimums: { discharge: 1.0e-4, ... }
  defaults: { p_spatial: 21.0 }
  log_space_parameters: [ n ]
```

The two tricky pieces:
- `parameter_ranges` in YAML is a dict mapping name → [min, max]. The
  existing `ParameterRanges` struct has named fields. Use a manual
  Deserialize impl or rename via `#[serde(rename = "...")]`. Simplest:
  deserialize into a `HashMap<String, [f32; 2]>` intermediate, then
  build the `ParameterRanges`.
- `learning_rate` in YAML is `{ 1: 0.001, 3: 0.0005 }` — int keys.
  `BTreeMap<usize, f32>` deserializes that natively.

If `Params` resists clean deserialize because of the field-vs-dict
mismatch, add a `ParamsRaw` intermediate that maps cleanly to YAML and
implement `From<ParamsRaw>` for `Params`.

- [ ] **Step 4: Run the tests**

```
cargo test --lib config 2>&1 | tail -10
```

Expected: both new tests pass + any existing config tests still pass.

- [ ] **Step 5: Commit**

```
git add src/config.rs
git commit -m "Extend Config with DataSources, Experiment, mlp sections

Adds YAML deserialization via serde_yaml. Existing routing-core fields
(parameter_ranges, attribute_minimums, log_space_parameters, defaults)
remain accessible at cfg.params and back-compatible with Default::default().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `RoutingBatch` struct + module wire-up

**Files:**
- Create: `src/data/dataset.rs`
- Modify: `src/data/mod.rs`

This task adds the public output type. The dataset struct body comes in
Task 7; this task just locks in the surface.

- [ ] **Step 1: Create `src/data/dataset.rs`**

```rust
//! `MeritGagesDataset` + `RoutingBatch`.
//!
//! Mirrors `~/projects/ddr/src/ddr/geodatazoo/merit.py::Merit` for the
//! training-mode (`_init_training` + `_collate_gages`) path. Other modes
//! (target_catchments, all_catchments) are out of scope for SP-3.

use ndarray::Array2;

use crate::data::dates::RhoWindow;
use crate::data::ids::{Comid, Staid};
use crate::sparse::SparseAdjacency;

/// One batch of inputs for the MC routing engine + MLP head.
///
/// All tensors are plain `ndarray::Array` here — SP-4's training loop
/// materializes them onto a BURN backend at the device boundary.
#[derive(Debug)]
pub struct RoutingBatch {
    pub adjacency: SparseAdjacency,
    /// Normalized attributes, shape `(N, F)`. Caller-major to match the
    /// MLP head input contract (`src/nn/mlp.rs::Mlp::forward`).
    pub spatial_attributes_normalized: Array2<f32>,
    /// q' streamflow forcing, shape `(T_hours, N)`. Already multiplied by
    /// `flow_scale` per column.
    pub q_prime: Array2<f32>,
    /// USGS observations, shape `(T_days, G)`. NaN-tolerant.
    pub observations: Array2<f32>,
    /// For each gauge in `gauge_staids`, list of compressed-cols whose row
    /// equals the gauge's outlet position. SP-4 reads gauge predictions
    /// out of the engine's `(N, T)` output via these indices.
    pub outflow_idx: Vec<Vec<usize>>,
    pub gauge_staids: Vec<Staid>,
    /// Compressed COMIDs in topological position order, length `N`.
    pub divide_comids: Vec<Comid>,
    /// Per-segment flow scaling factors, length `N`. Already applied to
    /// `q_prime` — kept here for diagnostics / loss reconstruction.
    pub flow_scale: Vec<f32>,
    pub window: RhoWindow,
}
```

- [ ] **Step 2: Wire the module into `src/data/mod.rs`**

Add `pub mod dataset;` after the existing `pub mod store;` line, and
update the re-exports:

```rust
pub mod collate;   // added in Task 3 (declared here to lock wire-up)
pub mod dataset;
pub mod sampler;   // added in Task 6
// ...
pub use dataset::RoutingBatch;
```

**Wait** — if `collate` and `sampler` modules don't exist yet, declaring
them here will fail to compile. So **only declare `dataset` in this
task**:

```rust
pub mod dataset;
// ...
pub use dataset::RoutingBatch;
```

Tasks 3 and 6 will add their own `pub mod` lines.

- [ ] **Step 3: Build**

```
cargo build 2>&1 | tail -5
```

Expected: clean.

- [ ] **Step 4: Commit**

```
git add src/data/dataset.rs src/data/mod.rs
git commit -m "Add RoutingBatch + dataset module skeleton

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `collate.rs` — `union_subgraphs` + `UnionedCoo`

**Files:**
- Create: `src/data/collate.rs`
- Modify: `src/data/mod.rs`

The collate logic is the gnarly math of SP-3. Factoring it into its own
module makes the pieces unit-testable.

- [ ] **Step 1: Failing TDD test for `union_subgraphs`**

Create `src/data/collate.rs`:

```rust
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
/// returned in CONUS-position coordinates. Sorted lex by `(row, col)`
/// so callers don't reshuffle.
#[derive(Debug)]
pub(crate) struct UnionedCoo {
    pub edges: Vec<(usize, usize)>,
    /// One entry per gauge that was present in `gages_adj`: `(gage_idx,
    /// gage_catchment)` from the subgraph attrs.
    pub gauges: Vec<(usize, String)>,
}

/// Build the union of per-gauge subgraph COOs. Mirrors
/// `construct_network_matrix`. Missing gauges (not in `gages_adj`) are
/// silently skipped — matches DDR's `try / except KeyError` behavior.
pub(crate) fn union_subgraphs(
    staids: &[Staid],
    gages_adj: &GagesAdjacencyStore,
) -> UnionedCoo {
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut gauges: Vec<(usize, String)> = Vec::with_capacity(staids.len());
    for s in staids {
        let Some(g): Option<&GageSubgraph> = gages_adj.get(s) else { continue };
        gauges.push((g.gage_idx, g.gage_catchment.clone()));
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
    use crate::data::ids::Staid;

    /// Build a tiny in-memory `GagesAdjacencyStore` for unit tests.
    fn synthetic_store(
        gauges: &[(&str, usize, Vec<(i32, i32)>)],
    ) -> GagesAdjacencyStore {
        use std::collections::HashMap;
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
        // (3,2), (2,1)}; gauge B has {(4,2), (2,1)}.  Shared edge (2,1)
        // appears once in the union.
        let store = synthetic_store(&[
            ("0000000A", 3, vec![(3, 1), (3, 2), (2, 1)]),
            ("0000000B", 4, vec![(4, 2), (2, 1)]),
        ]);
        let staids = vec![Staid::new("0000000A"), Staid::new("0000000B")];
        let u = union_subgraphs(&staids, &store);
        assert_eq!(u.edges.len(), 4); // (2,1), (3,1), (3,2), (4,2)
        assert_eq!(u.edges, vec![(2, 1), (3, 1), (3, 2), (4, 2)]);
        assert_eq!(u.gauges.len(), 2);
    }

    #[test]
    fn union_skips_missing_gauges() {
        let store = synthetic_store(&[("0000000A", 3, vec![(3, 1)])]);
        let staids = vec![Staid::new("0000000A"), Staid::new("00000099")];
        let u = union_subgraphs(&staids, &store);
        assert_eq!(u.gauges.len(), 1);
        assert_eq!(u.edges.len(), 1);
    }
}
```

Note the test uses `GagesAdjacencyStore { path, subgraphs }` struct
constructor directly. Verify the existing `GagesAdjacencyStore` struct
has `subgraphs: HashMap<Staid, GageSubgraph>` as a `pub` field. If not,
add a `#[cfg(test)] pub(crate) fn from_subgraphs(...)` test helper.

- [ ] **Step 2: Wire the module into `src/data/mod.rs`**

```rust
pub mod collate;
```

(Don't re-export anything — it's `pub(crate)`-internal.)

- [ ] **Step 3: Run the tests**

```
cargo test --lib data::collate
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```
git add src/data/collate.rs src/data/mod.rs
git commit -m "Add collate module with subgraph union

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `collate.rs` — `compress` + `CompressedAdj` (lower-triangular invariant)

**Files:**
- Modify: `src/data/collate.rs`

- [ ] **Step 1: Failing TDD tests**

Append inside the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn compress_preserves_topological_order() {
        use crate::data::ids::Comid;
        // CONUS positions [0, 1, 2, 3, 4], COMIDs in topological order.
        let conus_order = vec![Comid(100), Comid(200), Comid(300), Comid(400), Comid(500)];
        // Edges in CONUS positions, lower-triangular (rows >= cols).
        let unioned = UnionedCoo {
            edges: vec![(2, 0), (3, 1), (4, 2), (4, 3)],
            gauges: vec![(4, "comid500".to_string()), (3, "comid400".to_string())],
        };
        let c = compress(&unioned, &conus_order).expect("compress");
        // Active = {0, 1, 2, 3, 4} → all 5. Compressed positions match.
        assert_eq!(c.divide_comids, conus_order);
        assert_eq!(c.rows, vec![2, 3, 4, 4]);
        assert_eq!(c.cols, vec![0, 1, 2, 3]);
        assert_eq!(c.gauge_compressed, vec![4, 3]);
        // outflow_idx: gauge at row 4 receives from cols 2, 3.
        // gauge at row 3 receives from col 1.
        assert_eq!(c.outflow_idx[0], vec![2, 3]);
        assert_eq!(c.outflow_idx[1], vec![1]);
    }

    #[test]
    fn compress_remaps_to_dense_compressed_space() {
        use crate::data::ids::Comid;
        // Sparse active set: CONUS positions {2, 5, 7, 9} → compressed {0,1,2,3}.
        let conus_order: Vec<Comid> = (0..10).map(|i| Comid(i as i64 * 100)).collect();
        let unioned = UnionedCoo {
            edges: vec![(9, 7), (9, 5), (7, 2)],
            gauges: vec![(9, "comid900".to_string())],
        };
        let c = compress(&unioned, &conus_order).expect("compress");
        assert_eq!(c.divide_comids, vec![Comid(200), Comid(500), Comid(700), Comid(900)]);
        // Edges in compressed space: (3,2), (3,1), (2,0).  Sorted by edge order.
        assert_eq!(c.rows.len(), 3);
        for k in 0..c.rows.len() {
            assert!(c.rows[k] >= c.cols[k], "lower-triangular violated");
        }
        assert_eq!(c.gauge_compressed, vec![3]);
    }

    #[test]
    fn compress_errors_on_non_topological_edges() {
        use crate::data::ids::Comid;
        let conus_order = vec![Comid(0), Comid(1), Comid(2)];
        // Bogus edge: row 0, col 1 — violates lower-triangular.
        let unioned = UnionedCoo {
            edges: vec![(0, 1)],
            gauges: vec![(0, "x".to_string())],
        };
        let err = compress(&unioned, &conus_order).unwrap_err();
        match err {
            crate::data::error::DataError::Malformed { .. } => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }
```

- [ ] **Step 2: Verify fail**

```
cargo test --lib data::collate::tests::compress
```

Expected: compile error — `compress` and `CompressedAdj` don't exist.

- [ ] **Step 3: Implement `compress` + `CompressedAdj`**

Append to `src/data/collate.rs` (above `#[cfg(test)]`):

```rust
use std::collections::HashMap;
use std::path::PathBuf;

use crate::data::error::{DataError, Result};
use crate::data::ids::Comid;

/// Compressed adjacency built from a unioned COO.
#[derive(Debug)]
pub(crate) struct CompressedAdj {
    pub divide_comids: Vec<Comid>,
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    pub gauge_compressed: Vec<usize>,
    pub outflow_idx: Vec<Vec<usize>>,
}

/// Compress a unioned COO into dense compressed-position space, preserving
/// topological order via `BTreeSet` sort. The CONUS adjacency's `order`
/// array is itself topological — so a sorted subset stays topological.
///
/// Hard-asserts the lower-triangular invariant (`rows >= cols`) — fails
/// with `DataError::Malformed` if violated.
pub(crate) fn compress(
    unioned: &UnionedCoo,
    conus_order: &[Comid],
) -> Result<CompressedAdj> {
    // 1. Active set = union of edge endpoints + gauge outlets, sorted.
    let mut active: BTreeSet<usize> = BTreeSet::new();
    for &(r, c) in &unioned.edges {
        active.insert(r);
        active.insert(c);
    }
    for &(g, _) in &unioned.gauges {
        active.insert(g);
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
        unioned.gauges.iter().map(|&(g, _)| mapping[&g]).collect();

    // 5. outflow_idx[g] = list of cols where rows[k] == gauge_compressed[g].
    //    Mirrors DDR's per-gauge "which segments feed this outlet" lookup.
    let mut outflow_idx: Vec<Vec<usize>> = Vec::with_capacity(gauge_compressed.len());
    for &g_comp in &gauge_compressed {
        let g_row = g_comp as i32;
        let cols_for_g: Vec<usize> = rows
            .iter()
            .zip(cols.iter())
            .filter(|(r, _)| **r == g_row)
            .map(|(_, c)| *c as usize)
            .collect();
        outflow_idx.push(cols_for_g);
    }

    Ok(CompressedAdj {
        divide_comids,
        rows,
        cols,
        gauge_compressed,
        outflow_idx,
    })
}
```

- [ ] **Step 4: Run the tests**

```
cargo test --lib data::collate
```

Expected: 5 tests pass (2 from Task 3 + 3 new).

- [ ] **Step 5: Commit**

```
git add src/data/collate.rs
git commit -m "Add compress() preserving topological lower-triangular order

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `collate.rs` — `build_flow_scale`

**Files:**
- Modify: `src/data/collate.rs`

Mirrors `~/projects/ddr/src/ddr/io/readers.py::build_flow_scale_tensor`
(~lines 270-330) and `compute_flow_scale_factor` (~lines 240-270). Fast
path: read pre-computed `FLOW_SCALE` column. Fallback: compute from
`(DRAIN_SQKM, COMID_DRAIN_SQKM, COMID_UNITAREA_SQKM)`.

- [ ] **Step 1: Failing TDD tests**

Append to the test block in `src/data/collate.rs`:

```rust
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
            assert!((scale[i] - 1.0).abs() < 1e-9);
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
        let expected = (60.0 - 50.0) / 60.0;
        assert!((scale[2] - expected).abs() < 1e-6);
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
```

- [ ] **Step 2: Implement `build_flow_scale`**

Append above the test block:

```rust
use crate::data::store::GageMetadata;

/// Per-segment flow scale factors of length `n_segments`. Default `1.0`;
/// the compressed-position of each gauge's outlet gets the gauge's scale.
///
/// Mirrors `build_flow_scale_tensor` in readers.py:270-330 with the same
/// fast path (`FLOW_SCALE` column) and fallback (`compute_flow_scale_factor`).
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
        // Fallback: compute factor from drainage areas.
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

/// Computes the per-gauge scaling factor in `[0, 1]`.
///
/// Mirrors `compute_flow_scale_factor` in readers.py:240-270. Returns
/// `1.0` for degenerate / NaN inputs (the `unwrap_or` paths there).
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
```

- [ ] **Step 3: Run the tests**

```
cargo test --lib data::collate
```

Expected: 8 tests pass (5 from prior + 3 new).

- [ ] **Step 4: Commit**

```
git add src/data/collate.rs
git commit -m "Add build_flow_scale with FLOW_SCALE fast path + area fallback

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `sampler.rs` — RandomSampler + SequentialSampler

**Files:**
- Create: `src/data/sampler.rs`
- Modify: `src/data/mod.rs`

- [ ] **Step 1: Failing TDD tests**

Create `src/data/sampler.rs`:

```rust
//! Batch index samplers.
//!
//! Mirrors `torch.utils.data.{RandomSampler, SequentialSampler}` for our
//! batching needs. Not PyTorch-bit-identical — the project's verification
//! bar doesn't require it. Determinism via `rand::SeedableRng`.

use rand::seq::SliceRandom;
use rand::Rng;

pub struct RandomSampler {
    indices: Vec<usize>,
    batch_size: usize,
    cursor: usize,
    drop_last: bool,
}

impl RandomSampler {
    pub fn new(n: usize, batch_size: usize, drop_last: bool) -> Self {
        Self {
            indices: (0..n).collect(),
            batch_size,
            cursor: 0,
            drop_last,
        }
    }

    /// Permute the index list for a fresh epoch.
    pub fn reshuffle<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.indices.shuffle(rng);
        self.cursor = 0;
    }

    /// Return the next batch's indices, or `None` if the epoch is done.
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        let remaining = self.indices.len().saturating_sub(self.cursor);
        if remaining == 0 {
            return None;
        }
        let take = if remaining >= self.batch_size {
            self.batch_size
        } else if self.drop_last {
            return None;
        } else {
            remaining
        };
        let out = self.indices[self.cursor..self.cursor + take].to_vec();
        self.cursor += take;
        Some(out)
    }
}

pub struct SequentialSampler {
    n: usize,
    batch_size: usize,
    cursor: usize,
}

impl SequentialSampler {
    pub fn new(n: usize, batch_size: usize) -> Self {
        Self {
            n,
            batch_size,
            cursor: 0,
        }
    }

    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.cursor >= self.n {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.n);
        let out: Vec<usize> = (self.cursor..end).collect();
        self.cursor = end;
        Some(out)
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn random_sampler_covers_all_indices_with_drop_last_false() {
        let mut s = RandomSampler::new(10, 3, false);
        let mut rng = StdRng::seed_from_u64(42);
        s.reshuffle(&mut rng);
        let mut seen: Vec<usize> = Vec::new();
        while let Some(b) = s.next_batch() {
            seen.extend(b);
        }
        let mut seen_sorted = seen.clone();
        seen_sorted.sort();
        assert_eq!(seen_sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn random_sampler_drop_last_skips_partial_batch() {
        let mut s = RandomSampler::new(10, 3, true);
        let mut rng = StdRng::seed_from_u64(42);
        s.reshuffle(&mut rng);
        let mut total = 0;
        while let Some(b) = s.next_batch() {
            assert_eq!(b.len(), 3);
            total += b.len();
        }
        assert_eq!(total, 9); // 10 / 3 → 3 full batches of 3, drop 1
    }

    #[test]
    fn random_sampler_seeded_reproducible() {
        let mut s1 = RandomSampler::new(8, 2, false);
        let mut r1 = StdRng::seed_from_u64(7);
        s1.reshuffle(&mut r1);
        let b1: Vec<Vec<usize>> = std::iter::from_fn(|| s1.next_batch()).collect();

        let mut s2 = RandomSampler::new(8, 2, false);
        let mut r2 = StdRng::seed_from_u64(7);
        s2.reshuffle(&mut r2);
        let b2: Vec<Vec<usize>> = std::iter::from_fn(|| s2.next_batch()).collect();

        assert_eq!(b1, b2);
    }

    #[test]
    fn sequential_sampler_yields_in_order_with_partial_tail() {
        let mut s = SequentialSampler::new(7, 3);
        assert_eq!(s.next_batch(), Some(vec![0, 1, 2]));
        assert_eq!(s.next_batch(), Some(vec![3, 4, 5]));
        assert_eq!(s.next_batch(), Some(vec![6]));
        assert_eq!(s.next_batch(), None);
    }

    #[test]
    fn sequential_sampler_reset_restarts() {
        let mut s = SequentialSampler::new(4, 2);
        let _ = s.next_batch();
        let _ = s.next_batch();
        assert_eq!(s.next_batch(), None);
        s.reset();
        assert_eq!(s.next_batch(), Some(vec![0, 1]));
    }
}
```

- [ ] **Step 2: Wire into `src/data/mod.rs`**

Add `pub mod sampler;` and update re-exports:

```rust
pub mod sampler;
// ...
pub use sampler::{RandomSampler, SequentialSampler};
```

- [ ] **Step 3: Run the tests**

```
cargo test --lib data::sampler
```

Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```
git add src/data/sampler.rs src/data/mod.rs
git commit -m "Add RandomSampler + SequentialSampler over usize indices

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `dataset.rs` — `MeritGagesDataset::open` (filter pipeline)

**Files:**
- Modify: `src/data/dataset.rs`

This task adds the dataset struct + filter pipeline (DA_VALID → adjacency
presence → headwater drop). No `collate` yet — that's Task 8.

- [ ] **Step 1: Add struct + open method**

Append to `src/data/dataset.rs`:

```rust
use std::sync::Arc;

use ndarray::Array1;

use crate::config::Config;
use crate::data::dates::TimeAxis;
use crate::data::error::{DataError, Result};
use crate::data::statistics::AttrStats;
use crate::data::store::{
    AttributesStore, ConusAdjacencyStore, GageMetadata, GagesAdjacencyStore,
    StreamflowStore, UsgsObservationsStore,
};

pub struct MeritGagesDataset {
    pub(crate) conus: Arc<ConusAdjacencyStore>,
    pub(crate) gages_adj: Arc<GagesAdjacencyStore>,
    pub(crate) attrs: Arc<AttributesStore>,
    #[allow(dead_code)] // used by Task 8 (collate)
    pub(crate) stats: Arc<AttrStats>,
    pub(crate) gages: Arc<GageMetadata>,
    pub(crate) streamflow: Arc<StreamflowStore>,
    pub(crate) observations: Arc<UsgsObservationsStore>,
    pub(crate) time_axis: TimeAxis,
    pub(crate) attr_names: Vec<String>,
    pub(crate) means: Array1<f32>,
    pub(crate) stds: Array1<f32>,
    /// Filtered training gauges (DA_VALID + adjacency + non-headwater).
    pub(crate) gauges: Vec<Staid>,
}

impl MeritGagesDataset {
    /// Open all five stores + apply the training-mode filter pipeline.
    /// Mirrors `Merit.__init__` + `_init_training` in `geodatazoo/merit.py`.
    pub fn open(cfg: &Config) -> Result<Self> {
        let ds = cfg.data_sources.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "data_sources section missing".into(),
        })?;
        let exp = cfg.experiment.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "experiment section missing".into(),
        })?;
        let mlp = cfg.mlp.as_ref().ok_or_else(|| DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: "mlp section missing".into(),
        })?;

        // ---------- 1. Adjacency + gage CSV (needed for filtering) ----------
        let conus = Arc::new(ConusAdjacencyStore::open(&ds.conus_adjacency)?);

        // Read the gage CSV first; we'll need the STAID list for both
        // filtering and the gages-adjacency open.
        let gage_meta = GageMetadata::open(&ds.gages)?;

        // Filter 1: DA_VALID drop.
        let pre_filter = gage_meta.rows.len();
        let da_valid: Vec<Staid> = gage_meta
            .rows
            .iter()
            .filter(|r| r.da_valid == Some(true))
            .map(|r| r.staid.clone())
            .collect();
        log::info!(
            "DA_VALID filter: kept {}/{} gauges",
            da_valid.len(),
            pre_filter
        );

        // Open the gages adjacency store with just the DA_VALID set.
        let gages_adj = Arc::new(GagesAdjacencyStore::open(&ds.gages_adjacency, &da_valid)?);

        // Filter 2 + 3: adjacency presence + headwater drop.
        let mut gauges: Vec<Staid> = Vec::new();
        let mut n_missing = 0;
        let mut n_headwater = 0;
        for s in &da_valid {
            let Some(g) = gages_adj.get(s) else {
                n_missing += 1;
                continue;
            };
            if g.indices_0.is_empty() {
                n_headwater += 1;
                continue;
            }
            gauges.push(s.clone());
        }
        log::info!(
            "gages_adjacency filter: kept {} gauges (dropped {} missing, {} headwater)",
            gauges.len(),
            n_missing,
            n_headwater
        );

        // ---------- 2. Attributes + statistics ----------
        // Materialize only the CONUS COMIDs (about 346K) into the attribute
        // matrix. We use `conus.order` as the COMID slice.
        let attr_names: Vec<String> = mlp.input_var_names.clone();
        let attrs = Arc::new(AttributesStore::open(
            &ds.attributes,
            &attr_names,
            &conus.order,
        )?);

        // Statistics JSON path: DDR convention puts it under
        // `data_sources.statistics` but our YAML omits that field. Use a
        // sibling-of-attributes default.
        let stats_path = stats_path_from_attrs(&ds.attributes);
        let stats = Arc::new(AttrStats::open(&stats_path)?);
        let means = stats.means_f32(&attr_names);
        let stds = stats.stds_f32(&attr_names);

        // ---------- 3. Icechunk stores ----------
        let streamflow = Arc::new(StreamflowStore::open(&ds.streamflow)?);
        let observations = Arc::new(UsgsObservationsStore::open(&ds.observations)?);

        // ---------- 4. Time axis from experiment dates ----------
        let time_axis = parse_experiment_axis(&exp.start_time, &exp.end_time)?;

        Ok(Self {
            conus,
            gages_adj,
            attrs,
            stats,
            gages: Arc::new(gage_meta),
            streamflow,
            observations,
            time_axis,
            attr_names,
            means,
            stds,
            gauges,
        })
    }

    pub fn len(&self) -> usize {
        self.gauges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gauges.is_empty()
    }

    pub fn staids(&self) -> &[Staid] {
        &self.gauges
    }

    pub fn time_axis(&self) -> &TimeAxis {
        &self.time_axis
    }
}

/// Default statistics JSON location: same directory as the NetCDF, named
/// `merit_attribute_statistics_{filename}.json`. Mirrors DDR's
/// `set_statistics` cache-file convention.
fn stats_path_from_attrs(attrs_path: &std::path::Path) -> std::path::PathBuf {
    let dir = attrs_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let fname = attrs_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();
    dir.join("statistics")
        .join(format!("merit_attribute_statistics_{fname}.json"))
}

/// Parse two `"YYYY/MM/DD"` strings (DDR convention) into a `TimeAxis`.
fn parse_experiment_axis(start: &str, end: &str) -> Result<TimeAxis> {
    let start_date = chrono::NaiveDate::parse_from_str(start, "%Y/%m/%d").map_err(|e| {
        DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: format!("invalid experiment.start_time {start:?}: {e}"),
        }
    })?;
    let end_date = chrono::NaiveDate::parse_from_str(end, "%Y/%m/%d").map_err(|e| {
        DataError::Malformed {
            path: std::path::PathBuf::from("<config>"),
            message: format!("invalid experiment.end_time {end:?}: {e}"),
        }
    })?;
    Ok(TimeAxis::new(start_date, end_date))
}
```

- [ ] **Step 2: Re-export from `src/data/mod.rs`**

Update the `pub use dataset::...` line:

```rust
pub use dataset::{MeritGagesDataset, RoutingBatch};
```

- [ ] **Step 3: Build**

```
cargo build 2>&1 | tail -10
```

Fix any compile issues. Common pitfalls:
- `Staid` and `Comid` imports — `use crate::data::ids::{Comid, Staid};` at
  the top of `dataset.rs`.
- `log` crate isn't in `Cargo.toml`. Drop `log::info!` calls in favor of
  `eprintln!` for now; SP-4 can introduce structured logging later.

- [ ] **Step 4: Commit**

```
git add src/data/dataset.rs src/data/mod.rs
git commit -m "Add MeritGagesDataset::open with DA_VALID + headwater filtering

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `dataset.rs` — `MeritGagesDataset::collate`

**Files:**
- Modify: `src/data/dataset.rs`

The body of `collate`. Calls the Task 3-5 helpers, fetches q' and
observations, applies flow_scale + normalization, assembles the
`RoutingBatch`.

- [ ] **Step 1: Implement `collate`**

Append to `src/data/dataset.rs`:

```rust
use ndarray::Array2;

use crate::data::collate::{build_flow_scale, compress, union_subgraphs};
use crate::data::dates::RhoWindow;
use crate::data::statistics::fill_nans;
use crate::sparse::SparseAdjacency;

impl MeritGagesDataset {
    /// Build one `RoutingBatch` from a STAID subset + a time window.
    ///
    /// Mirrors `Merit._collate_gages` in `geodatazoo/merit.py:245-330`.
    pub fn collate(&self, batch_staids: &[Staid], window: &RhoWindow)
        -> Result<RoutingBatch>
    {
        // ----- 1. Subgraph union + compression -----
        let unioned = union_subgraphs(batch_staids, &self.gages_adj);
        if unioned.gauges.is_empty() {
            return Err(DataError::Malformed {
                path: std::path::PathBuf::from("<collate>"),
                message: format!(
                    "no batch gauges present in gages_adjacency (asked for {})",
                    batch_staids.len()
                ),
            });
        }
        let compressed = compress(&unioned, &self.conus.order)?;
        let n = compressed.divide_comids.len();

        // ----- 2. SparseAdjacency: rows/cols + length/slope sliced -----
        let mut length_m: Vec<f32> = Vec::with_capacity(n);
        let mut slope: Vec<f32> = Vec::with_capacity(n);
        for c in &compressed.divide_comids {
            let pos = self.conus.index.position(c).ok_or_else(|| DataError::Malformed {
                path: self.conus.path.clone(),
                message: format!("compressed COMID {c:?} not found in CONUS order"),
            })?;
            length_m.push(self.conus.length_m[pos]);
            slope.push(self.conus.slope[pos]);
        }
        let values: Vec<f32> = vec![1.0; compressed.rows.len()];
        let adjacency = SparseAdjacency {
            n,
            rows: compressed.rows.clone(),
            cols: compressed.cols.clone(),
            values,
            length_m,
            slope,
        };

        // ----- 3. flow_scale + q_prime fusion -----
        let flow_scale = build_flow_scale(
            batch_staids,
            &compressed.gauge_compressed,
            &self.gages,
            n,
        );
        let mut q_prime = self.streamflow.read_window(window, &compressed.divide_comids)?;
        // shape: (T_hours, N). Multiply each column by flow_scale[col].
        for col in 0..n {
            let s = flow_scale[col];
            if (s - 1.0).abs() < 1e-9 {
                continue;
            }
            for t in 0..q_prime.shape()[0] {
                q_prime[(t, col)] *= s;
            }
        }

        // ----- 4. Attributes: slice + fill_nans + normalize + transpose -----
        let f = self.attr_names.len();
        let mut attrs_present: Array2<f32> = Array2::zeros((f, n));
        for (out_col, comid) in compressed.divide_comids.iter().enumerate() {
            if let Some(src_col) = self.attrs.index.position(comid) {
                for fi in 0..f {
                    attrs_present[(fi, out_col)] = self.attrs.attrs[(fi, src_col)];
                }
            } else {
                // Missing — fill with NaN so fill_nans handles it via row_means.
                for fi in 0..f {
                    attrs_present[(fi, out_col)] = f32::NAN;
                }
            }
        }
        fill_nans(attrs_present.view_mut(), &self.attrs.row_means);

        // Normalize: (attrs - means) / stds, broadcast along axis 1.
        for fi in 0..f {
            let mean = self.means[fi];
            let std = self.stds[fi];
            for col in 0..n {
                attrs_present[(fi, col)] = (attrs_present[(fi, col)] - mean) / std;
            }
        }
        // Transpose to (N, F) for the MLP head's input contract.
        let spatial_attributes_normalized: Array2<f32> = attrs_present.reversed_axes();

        // ----- 5. Observations -----
        let observations = self.observations.read_window(window, batch_staids)?;

        // ----- 6. Assemble. -----
        let gauge_staids: Vec<Staid> = unioned
            .gauges
            .iter()
            .zip(batch_staids.iter())
            .map(|(_, s)| s.clone())
            .collect();
        Ok(RoutingBatch {
            adjacency,
            spatial_attributes_normalized,
            q_prime,
            observations,
            outflow_idx: compressed.outflow_idx,
            gauge_staids,
            divide_comids: compressed.divide_comids,
            flow_scale,
            window: *window,
        })
    }
}
```

Notes:
- The `gauge_staids` line: `unioned.gauges` only contains entries for
  STAIDs that were found in `gages_adj`. So we want the STAIDs in the
  *same order* as `unioned.gauges`. Using `zip` against `batch_staids` is
  wrong when there's a missing-STAID skip; better to track this in
  `union_subgraphs` (it already does — `gauges: Vec<(usize, String)>`).
  Refactor to return `staids` alongside `gauges`. Or simpler: just
  re-derive — match `unioned.gauges[i].0` (gage_idx) back to
  `batch_staids` via a lookup. **The clean fix:** make `UnionedCoo.gauges`
  hold `(staid, gage_idx, gage_catchment)` triples. Apply this fix in
  Task 3's commit (amend) OR do it here. Either way, end result:
  `gauge_staids` has length equal to the present-gauge subset.

  **Decision:** change `UnionedCoo` to carry STAIDs explicitly. Modify
  `union_subgraphs` to push `(staid.clone(), g.gage_idx, g.gage_catchment.clone())`.
  Adjust `compress` to use the new shape. Update tests in Task 3
  accordingly. This is a one-line change per function. Do it inline in
  this task and amend the Task 3 helper.

- [ ] **Step 2: Build + run all data tests**

```
cargo test --lib data 2>&1 | tail -20
```

Expected: all data-module tests still pass; `collate` is integration-tested
in Task 9.

- [ ] **Step 3: Commit**

```
git add src/data/collate.rs src/data/dataset.rs
git commit -m "Add MeritGagesDataset::collate building RoutingBatch end-to-end

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(If the `collate.rs` change to `UnionedCoo` is non-trivial, split into a
separate commit landing first.)

---

### Task 9: V2 integration test against live stores

**Files:**
- Create: `tests/data_dataset.rs`

- [ ] **Step 1: Create the integration test**

```rust
//! V2 integration test for SP-3: open all five stores, build the dataset,
//! sample one batch, assemble one `RoutingBatch`, sanity-check shapes +
//! invariants. Skip cleanly if production data is absent.

use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;

use ddrs::config::Config;
use ddrs::data::{MeritGagesDataset, RandomSampler};

#[test]
fn collate_one_batch_against_live_stores() {
    let cfg_path = "config/merit_training.yaml";
    if !Path::new(cfg_path).exists() {
        eprintln!("skipping: {cfg_path} not present");
        return;
    }
    let cfg = Config::from_yaml_file(cfg_path).expect("load yaml");
    // Spot-check that all referenced paths exist; otherwise skip.
    let ds_paths = cfg.data_sources.as_ref().expect("data_sources");
    for p in &[
        &ds_paths.attributes,
        &ds_paths.conus_adjacency,
        &ds_paths.gages_adjacency,
        &ds_paths.streamflow,
        &ds_paths.observations,
        &ds_paths.gages,
    ] {
        if !p.exists() {
            eprintln!("skipping: {} not present", p.display());
            return;
        }
    }

    let dataset = MeritGagesDataset::open(&cfg).expect("open dataset");
    assert!(dataset.len() > 100, "expected many filtered gauges, got {}", dataset.len());

    let mut rng = StdRng::seed_from_u64(42);
    let mut sampler = RandomSampler::new(dataset.len(), 8, true);
    sampler.reshuffle(&mut rng);
    let batch_idx = sampler.next_batch().expect("batch");
    let staids: Vec<_> = batch_idx.iter().map(|&i| dataset.staids()[i].clone()).collect();
    let window = dataset.time_axis().sample_rho_window(&mut rng, 90);

    let batch = dataset.collate(&staids, &window).expect("collate");

    // Adjacency: at least as many segments as gauges (ancestry expanded),
    // but compressed below CONUS scale.
    assert!(batch.adjacency.n >= staids.len());
    assert!(batch.adjacency.n < 200_000);

    // q' shape: (T_hours, N), where T_hours = window.n_hourly().
    assert_eq!(batch.q_prime.shape(), &[window.n_hourly(), batch.adjacency.n]);

    // Normalized attrs shape: (N, F).
    assert_eq!(
        batch.spatial_attributes_normalized.shape()[0],
        batch.adjacency.n
    );

    // Observations shape: (rho_days, G).
    assert_eq!(
        batch.observations.shape(),
        &[window.rho_days, batch.gauge_staids.len()]
    );

    // Lower-triangular invariant.
    for k in 0..batch.adjacency.nnz() {
        assert!(
            batch.adjacency.rows[k] >= batch.adjacency.cols[k],
            "lower-triangular violated at nnz={k}"
        );
    }

    // outflow_idx has one entry per gauge.
    assert_eq!(batch.outflow_idx.len(), batch.gauge_staids.len());

    // flow_scale length matches adjacency.
    assert_eq!(batch.flow_scale.len(), batch.adjacency.n);
}
```

- [ ] **Step 2: Run the test**

```
cargo test --test data_dataset 2>&1 | tail -10
```

Expected: 1 test passes (or skips cleanly). Time: a few seconds — opens
five stores + reads a 90-day window for ~thousands of active segments.

- [ ] **Step 3: Commit**

```
git add tests/data_dataset.rs
git commit -m "Add SP-3 V2 integration test against live data sources

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Final clippy + regression sweep

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

```
cargo test 2>&1 | grep "test result" | tail -15
```

Expected: every file shows `ok`. Total ~75+ tests pass.

- [ ] **Step 2: Clippy on SP-3 code**

```
cargo clippy --all-targets -- -D warnings 2>&1 | grep -E "data/dataset.rs|data/collate.rs|data/sampler.rs|data_dataset"
```

Expected: nothing matches. Pre-existing lints in `dates.rs` and
`routing/utils.rs` remain (out of SP-3 scope per the established
precedent).

- [ ] **Step 3: Regression benchmark**

```
cargo run --release --example compare_ddr_sandbox 2>&1 | grep "verdict"
```

Expected: `verdict: ABSOLUTE MATCH`.

- [ ] **Step 4: If anything needs fixing**

If clippy fires on SP-3 code, fix inline. If `compare_ddr_sandbox`
regresses (it shouldn't — SP-3 doesn't touch the routing core), STOP and
report. If the V2 test reveals a runtime bug in `collate`, debug there —
that's the load-bearing test.

No commit unless fixes were applied.

---

## Self-Review

### Spec coverage

| Spec section | Task |
|---|---|
| `Config` extension (DataSources, Experiment, mlp) | 1 |
| `RoutingBatch` struct | 2 |
| `union_subgraphs` (subgraph union) | 3 |
| `compress` (compressed adjacency + lower-triangular invariant) | 4 |
| `build_flow_scale` (fast path + fallback) | 5 |
| `RandomSampler`, `SequentialSampler` | 6 |
| Filter pipeline (DA_VALID + adjacency + headwater) | 7 |
| `MeritGagesDataset::collate` (orchestrator) | 8 |
| V2 integration test (shape + invariants) | 9 |
| Final clippy + regression sweep | 10 |

### Placeholder scan

No "TBD", "TODO" without concrete fix-it-here instructions. Every code
step shows actual code; every test step shows actual asserts.

The one piece of structural advice in Task 8 (changing `UnionedCoo` to
carry STAIDs) is the cleanest seam between Task 3 and Task 8. Mark it as
"amend Task 3's struct" in the commit when it happens — both options
(amend or inline) are fine, the implementer picks.

### Type/identifier consistency

- `MeritGagesDataset`, `RoutingBatch`, `RandomSampler`, `SequentialSampler`
  — all public types used identically across tasks.
- `UnionedCoo`, `CompressedAdj` — `pub(crate)`, internal.
- `Staid`, `Comid`, `IdIndex<T>`, `SparseAdjacency`, `RhoWindow`,
  `TimeAxis`, `Config`, `AttrStats`, `AttributesStore`, `GageMetadata`,
  `GagesAdjacencyStore`, `ConusAdjacencyStore`, `StreamflowStore`,
  `UsgsObservationsStore` — all match existing module surfaces.

No drift detected.

---

## Execution choice

Plan complete and saved. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-
   stage review. Same workflow as SP-1 + SP-2.
2. **Inline Execution** — `executing-plans`, batch with checkpoints.

Which approach?
