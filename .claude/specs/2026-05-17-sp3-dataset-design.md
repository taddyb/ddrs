# SP-3 design: `MeritGagesDataset` + batch collate

**Status:** Draft, pending user review
**Parent:** [`2026-05-17-train_and_test-replication-design.md`](./2026-05-17-train_and_test-replication-design.md)
**Mirrors:** `ddr/geodatazoo/merit.py::Merit` and especially `_collate_gages`
(~lines 245-330), `_build_common_tensors` (~lines 175-220), and
`ddr/io/builders.py::construct_network_matrix` (~lines 55-110).

## Why this sub-project

SP-3 is the load-bearing piece of the data pipeline. It glues the static
readers (SP-1: attributes, gage CSV, statistics) and the async readers
(SP-2: streamflow + observations) plus the existing adjacency stores into
**one `RoutingBatch` per training step**. Without it, the MC engine has no
inputs and SP-4's training loop is just an empty optimizer.

This is also where the project's verification bar materially binds — per
the master spec, "given the same network inputs and same fixed MLP outputs,
ddrs's per-batch L1 loss equals DDR's." For that to hold, ddrs and DDR must
agree on the *contents* of each batch: the compressed adjacency, the
attribute ordering, the q' window, and the gauge subset. SP-3 produces
those contents; SP-4 will spot-check loss equivalence using a small Python
side-by-side run.

## Scope

In scope:

1. **`RoutingBatch`** — plain-ndarray output struct (no BURN tensors here;
   the tensor materialization is at the SP-4 boundary).
2. **`MeritGagesDataset`** — owns `Arc`s to all five upstream stores +
   filtered gauge list. `collate(staids, window)` builds a `RoutingBatch`.
3. **`flow_scale` computation** — port of
   `~/projects/ddr/src/ddr/io/readers.py::build_flow_scale_tensor` (~lines
   270-330) plus `compute_flow_scale_factor`. Applied to q' at collate
   time (DDR multiplies inside the engine; we multiply at the data
   boundary — same result).
4. **Gauge filtering** — DA_VALID (mirrors `filter_gages_by_da_valid`,
   readers.py:185-220) and headwater drop (`filter_headwater_gages`,
   readers.py:235-265). Done once at dataset construction.
5. **Subgraph union + compression** — port of `construct_network_matrix`
   plus the `_collate_gages` body. The active-COMID list is sorted (via
   `BTreeSet`) so the topological order is preserved and lower-
   triangularity holds in the compressed matrix.
6. **Sampler types** — `RandomSampler<R: Rng>` (training, seeded) and
   `SequentialSampler` (testing). Both yield batches of size `batch_size`
   over the filtered gauge list. **Not PyTorch-bit-identical** — the
   verification bar doesn't require it.

Out of scope:

- BURN tensor materialization. SP-4 turns `Array2<f32>` →
  `Tensor<Autodiff<I>, 2>` at the device boundary.
- `target_catchments` mode and `all_catchments` mode from DDR's `Merit`.
  These are inference-time paths; training only needs `_collate_gages`.
- Training loop, optimizer, loss. That's SP-4.
- Hourly streamflow stores (`is_hourly: true` branch). MERIT config is
  daily.
- `tau` boundary trimming. Stays in SP-4 (it's part of the daily-runoff
  downsample, after the engine, not a batch input).
- Caching. The dataset has no per-batch cache; every `collate` call hits
  icechunk fresh. Profile-driven if needed later.

## Architecture

```
                    Sampler::next() →  Vec<Staid> (size = batch_size)
                                            │
                                            ▼
              ┌──────────────────────────────────────────────────────┐
              │  MeritGagesDataset::collate(staids, window)          │
              │                                                      │
              │  1. construct_network_matrix(staids, gages_adj)      │
              │       → set<(row, col)> in CONUS positions           │
              │       → list of (gage_idx, gage_catchment)           │
              │                                                      │
              │  2. active_indices = BTreeSet<usize>(edge endpts +   │
              │                                      gauge outlets)  │
              │     → preserves topological order via sort           │
              │                                                      │
              │  3. compress: HashMap<conus_idx, compressed_idx>     │
              │     compressed_rows, compressed_cols (i32)           │
              │                                                      │
              │  4. SparseAdjacency { rows, cols, values=1.0,        │
              │                       length_m, slope }              │
              │     ← length/slope sliced from ConusAdjacencyStore   │
              │     assert lower-triangular (rows >= cols)           │
              │                                                      │
              │  5. outflow_idx[g] = compressed cols where row==g    │
              │                                                      │
              │  6. flow_scale[N] from GageMetadata (FLOW_SCALE col  │
              │       fast path, factor fallback)                    │
              │                                                      │
              │  7. attrs[F, N] = AttributesStore.attrs[:, active]   │
              │     fill_nans by row_means                           │
              │     normalized = ((attrs - means) / stds).T   → (N,F)│
              │                                                      │
              │  8. q_prime = StreamflowStore.read_window(window,    │
              │                                           active_co) │
              │     multiply by flow_scale per column                │
              │                                                      │
              │  9. observations = UsgsObservationsStore.read_window │
              │                              (window, staids)        │
              └──────────────────────────────────────────────────────┘
                                            │
                                            ▼
                                     RoutingBatch
```

## Components

### 1. `src/data/dataset.rs` — `MeritGagesDataset` + `RoutingBatch`

```rust
pub struct RoutingBatch {
    /// Compressed sparse adjacency + per-reach length/slope.
    pub adjacency: SparseAdjacency,
    /// Normalized attributes, shape `(N, F)` (caller-major to match the MLP
    /// input contract from `src/nn/mlp.rs`).
    pub spatial_attributes_normalized: Array2<f32>,
    /// q' streamflow forcing, shape `(T_hours, N)`, pre-scaled by flow_scale.
    pub q_prime: Array2<f32>,
    /// USGS observations, shape `(T_days, G)`. NaN values preserved.
    pub observations: Array2<f32>,
    /// For each gauge in `gauge_staids`, the list of compressed-cols whose
    /// row index is the gauge's outlet position. The MC engine writes
    /// `Q_{t+1}[outflow_idx[g]]` to the gauge's predicted discharge.
    pub outflow_idx: Vec<Vec<usize>>,
    /// Gauges represented in this batch (filtered to those present in the
    /// gages_adjacency store).
    pub gauge_staids: Vec<Staid>,
    /// Compressed COMIDs in topological position order, length `N`.
    pub divide_comids: Vec<Comid>,
    /// Per-segment flow_scale factors in `[0, 1]`, length `N`. Already
    /// multiplied into `q_prime` — kept here for diagnostics + later loss
    /// reconstruction.
    pub flow_scale: Vec<f32>,
    pub window: RhoWindow,
}

pub struct MeritGagesDataset {
    conus: Arc<ConusAdjacencyStore>,
    gages_adj: Arc<GagesAdjacencyStore>,
    attrs: Arc<AttributesStore>,
    stats: Arc<AttrStats>,
    gages: Arc<GageMetadata>,
    streamflow: Arc<StreamflowStore>,
    observations: Arc<UsgsObservationsStore>,
    time_axis: TimeAxis,
    /// `attr_names` in row order — used to align stats lookup with attrs rows.
    attr_names: Vec<String>,
    /// `means[F]` and `stds[F]` in attribute-name order, materialized once.
    means: Array1<f32>,
    stds: Array1<f32>,
    /// Filtered training gauges — passed DA_VALID + headwater drop +
    /// gages-adjacency presence check.
    gauges: Vec<Staid>,
    /// `attribute_minimums.discharge` — used to fill missing q' columns.
    discharge_min: f32,
}

impl MeritGagesDataset {
    /// Open all five stores and apply the training-mode gauge filters.
    ///
    /// Mirrors `Merit.__init__` + `_init_training` in geodatazoo/merit.py.
    pub fn open(cfg: &Config) -> Result<Self>;

    /// Number of filtered gauges available for batching.
    pub fn len(&self) -> usize { self.gauges.len() }

    /// All filtered gauge STAIDs in stable order.
    pub fn staids(&self) -> &[Staid] { &self.gauges }

    /// Build one batch from a STAID subset + a time window.
    ///
    /// Mirrors `Merit._collate_gages` in geodatazoo/merit.py.
    pub fn collate(&self, batch_staids: &[Staid], window: &RhoWindow)
        -> Result<RoutingBatch>;
}
```

`MeritGagesDataset::open` takes a `&Config` (the existing
`src/config.rs` struct, possibly extended). The five paths come from
the YAML at `config/merit_training.yaml`. SP-3 adds the
`data_sources` and `experiment` sections to `Config` if they aren't
already there.

### 2. `src/data/sampler.rs` — RandomSampler + SequentialSampler

```rust
pub struct RandomSampler<R: Rng + ?Sized> {
    indices: Vec<usize>,
    batch_size: usize,
    cursor: usize,
    drop_last: bool,
    _rng: std::marker::PhantomData<R>,
}

impl<R: Rng + SeedableRng> RandomSampler<R> {
    /// Build a sampler over `n` items. The order is permuted once per
    /// epoch; consumers call `next_batch(&mut rng)` to drive it.
    pub fn new(n: usize, batch_size: usize, drop_last: bool) -> Self;

    /// Yield the next `Vec<usize>` of size `batch_size` (or shorter if
    /// `drop_last=false` and we're at the tail). Returns `None` when the
    /// epoch is exhausted; call `reshuffle(rng)` to start the next epoch.
    pub fn next_batch(&mut self) -> Option<Vec<usize>>;
    pub fn reshuffle(&mut self, rng: &mut R);
}

pub struct SequentialSampler {
    n: usize,
    batch_size: usize,
    cursor: usize,
}

impl SequentialSampler {
    pub fn new(n: usize, batch_size: usize) -> Self;
    pub fn next_batch(&mut self) -> Option<Vec<usize>>;
    pub fn reset(&mut self);
}
```

Both samplers yield index lists. SP-4's training loop converts indices
→ STAIDs via `&dataset.staids()[idx]` and passes the slice to
`dataset.collate(...)`.

### 3. `src/data/collate.rs` — helpers (separate file for testability)

The collate logic is the gnarly part of SP-3. Factoring it into a
sibling module of `dataset.rs` lets us unit-test the pure-math pieces on
synthetic data without spinning up icechunk.

```rust
/// Output of the first stage: union of subgraph COOs across a batch of
/// gauges, in CONUS-position coordinates (not yet compressed).
pub(crate) struct UnionedCoo {
    /// `(downstream_pos, upstream_pos)` pairs, deduplicated. Sorted lex.
    pub edges: Vec<(usize, usize)>,
    /// Per-gauge `gage_idx` (CONUS-position of the gauge outlet) +
    /// `gage_catchment` (the gauge's outlet COMID as a string from the
    /// subgraph attrs).
    pub gauges: Vec<(usize, String)>,
}

/// Mirrors `~/projects/ddr/src/ddr/io/builders.py::construct_network_matrix`.
pub(crate) fn union_subgraphs(
    staids: &[Staid],
    gages_adj: &GagesAdjacencyStore,
) -> UnionedCoo;

/// Compressed adjacency built by mapping the union to a dense compressed
/// index space (preserving topological order via sort).
pub(crate) struct CompressedAdj {
    /// Compressed COMIDs in topological order, length `N`.
    pub divide_comids: Vec<Comid>,
    /// Compressed-position rows / cols (i32 for `SparseAdjacency`).
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    /// Per-gauge compressed position of the gauge outlet, length `G`.
    pub gauge_compressed: Vec<usize>,
    /// For each gauge, the compressed cols whose row index equals the
    /// gauge's outlet — used by SP-4 to read predictions out of the
    /// engine's `(N, T)` output. Mirrors DDR's `outflow_idx`.
    pub outflow_idx: Vec<Vec<usize>>,
}

pub(crate) fn compress(
    unioned: &UnionedCoo,
    conus_order: &[Comid],
) -> Result<CompressedAdj>;

/// Per-segment flow scale for a batch. Mirrors `build_flow_scale_tensor`.
/// `FLOW_SCALE` column from `GageMetadata` is the fast path; the
/// drainage-area fallback uses `compute_flow_scale_factor`.
pub(crate) fn build_flow_scale(
    batch_staids: &[Staid],
    gauge_compressed: &[usize],
    gages: &GageMetadata,
    n_segments: usize,
) -> Vec<f32>;
```

`CompressedAdj` is the canonical intermediate. `MeritGagesDataset::collate`
calls these helpers in sequence, then assembles `RoutingBatch`.

## Filtering pipeline (one-time, at `MeritGagesDataset::open`)

1. **Source**: `gage_csv::GageMetadata` from `config.data_sources.gages`.
2. **DA_VALID drop**: keep rows with `da_valid == Some(true)`. Mirrors
   `filter_gages_by_da_valid` in readers.py:185-220. Error if the column
   is absent (matches DDR's behavior when DA_VALID is missing and
   `max_area_diff_sqkm` isn't set).
3. **Adjacency presence**: drop STAIDs not in
   `GagesAdjacencyStore::subgraphs`.
4. **Headwater drop**: drop STAIDs whose subgraph has `indices_0.len() == 0`
   (no upstream connectivity). Mirrors `filter_headwater_gages`,
   readers.py:235-265.

Log each filter's `(kept, dropped)` count via `log::info`.

Final `gauges: Vec<Staid>` is the universe SP-3's samplers iterate over.

## Subgraph union + compression details

`construct_network_matrix`-equivalent in Rust:

```rust
pub(crate) fn union_subgraphs(
    staids: &[Staid],
    gages_adj: &GagesAdjacencyStore,
) -> UnionedCoo {
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut gauges = Vec::with_capacity(staids.len());
    for s in staids {
        let Some(g) = gages_adj.get(s) else { continue };
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
```

`compress` then:

1. `active: BTreeSet<usize> = edges.iter().flat_map(|&(r,c)| [r,c]).chain(gauges.iter().map(|(g,_)| *g))`.
2. `divide_comids: Vec<Comid> = active.iter().map(|&i| conus_order[i]).collect();`.
   The sort comes from `BTreeSet`'s natural order — preserves topological
   position because `conus_order` is itself topological.
3. `index_mapping: HashMap<usize, usize>` from CONUS-position →
   compressed-position.
4. `rows[k] = mapping[edges[k].0] as i32; cols[k] = mapping[edges[k].1] as i32`.
5. **Invariant assertion**: `for k in 0..nnz { assert!(rows[k] >= cols[k]); }`.
   Fail with a clear `DataError::Malformed` if violated — that means the
   upstream zarr broke its own contract and SP-3 won't paper over it.
6. `outflow_idx[g]` = list of `cols[k]` for `k` where `rows[k] ==
   gauge_compressed[g]`.

`length_m` and `slope` for the compressed `SparseAdjacency` come from
`ConusAdjacencyStore::{length_m, slope}` sliced by the `active` positions.

## Attribute path

```rust
// `AttributesStore::attrs` is shape (F, N_conus), aligned to its own
// (full-CONUS or pre-slice) IdIndex. We slice to the active COMIDs.
let mut attrs_present: Array2<f32> = Array2::zeros((f, n_active));
for (out_col, comid) in divide_comids.iter().enumerate() {
    if let Some(src_col) = self.attrs.index.position(comid) {
        attrs_present.column_mut(out_col).assign(&self.attrs.attrs.column(src_col));
    } // else: column stays zero — filled by fill_nans next.
}
// Replace any NaNs with the per-row mean (from AttributesStore.row_means).
fill_nans(attrs_present.view_mut(), &self.attrs.row_means);

// Normalize: ((attrs - means) / stds), broadcasting over N. Then transpose
// to (N, F) for the MLP input contract (Mlp::forward expects shape [N, F]).
let normalized = ((attrs_present - &means_col_broadcast) / &stds_col_broadcast).reversed_axes();
```

`means` and `stds` are derived once at `MeritGagesDataset::open` via
`AttrStats::means_f32` / `stds_f32`. They have length `F` and are
broadcast across N at collate time.

## flow_scale + q' fusion

Per DDR's pattern, `flow_scale` multiplies into q' before the routing
engine consumes it. The convention is per-segment (length `N_active`):
default `1.0` for non-gauge segments; the gauge's outlet segment gets the
fractional scale.

SP-3's `collate` does:

```rust
let flow_scale = build_flow_scale(staids, &compressed.gauge_compressed,
                                   &self.gages, n_active);  // Vec<f32>, len N
let mut q_prime = self.streamflow.read_window(window, &divide_comids)?;
// q_prime: shape (T_hours, N). Multiply each column by flow_scale[col].
for col in 0..n_active {
    let s = flow_scale[col];
    if (s - 1.0).abs() < 1e-9 { continue; }  // skip the 99% common case
    for t in 0..q_prime.shape()[0] {
        q_prime[(t, col)] *= s;
    }
}
```

DDR multiplies inside the engine (in `mmc.py`'s setup_inputs); we
multiply at the data boundary. Same outcome, fewer load-bearing engine
changes.

## Observations path

```rust
let observations = self.observations.read_window(window, batch_staids)?;
// shape (rho_days, G). NaN-tolerant — SP-4 filters all-NaN gauges
// at loss time via a mask. No collate-time filtering.
```

## Verification protocol

Per the project verification philosophy — alignment over bytes — SP-3's
verification has three layers, easiest first:

### V1 — unit tests on small synthetic networks (load-bearing for collate logic)

A handful of `#[test]` functions inside `collate.rs` and `dataset.rs`:

1. **Subgraph union dedupe**: two gauges with overlapping subgraphs →
   union has the right edge count.
2. **Compression preserves topological order**: hand-crafted 3-gauge
   diamond network in CONUS positions `[0, 1, 2, 3, 4]`, assert
   `divide_comids` is in ascending CONUS-position order and `rows >=
   cols` everywhere.
3. **outflow_idx semantics**: synthetic gauge whose outlet receives
   inflow from two upstream segments → `outflow_idx[gauge]` has length 2.
4. **flow_scale fast path**: pass a `GageMetadata` with explicit
   `FLOW_SCALE` values → assert the right segments get those values, the
   rest stay 1.0.
5. **flow_scale fallback**: pass a `GageMetadata` without `FLOW_SCALE`
   but with the `COMID_DRAIN_SQKM` triple → assert computed factor matches
   DDR's `compute_flow_scale_factor` formula.

These tests don't need any live data sources. They run in milliseconds.

### V2 — integration test against live data

One end-to-end test that opens all five stores, builds the dataset, runs
the filter pipeline, samples one window, and assembles one batch. Asserts
on shape + ranges only — no DDR cross-check.

```rust
#[test]
fn collate_one_batch_against_live_stores() {
    if !all_paths_exist(&[...]) { return; }
    let cfg = Config::from_yaml("config/merit_training.yaml")?;
    let ds = MeritGagesDataset::open(&cfg)?;
    let mut rng = StdRng::seed_from_u64(42);
    let mut sampler = RandomSampler::new(ds.len(), 8, true);
    sampler.reshuffle(&mut rng);
    let batch_idx = sampler.next_batch().unwrap();
    let staids: Vec<Staid> = batch_idx.iter().map(|&i| ds.staids()[i].clone()).collect();
    let window = ds.time_axis().sample_rho_window(&mut rng, 90);
    let batch = ds.collate(&staids, &window)?;

    assert!(batch.adjacency.n > staids.len());           // expanded by ancestry
    assert!(batch.adjacency.n < 200_000);                // compressed, not full CONUS
    assert_eq!(batch.q_prime.shape()[0], window.n_hourly());
    assert_eq!(batch.q_prime.shape()[1], batch.adjacency.n);
    assert_eq!(batch.spatial_attributes_normalized.shape()[0], batch.adjacency.n);
    assert_eq!(batch.observations.shape(), &[window.rho_days, staids.len()]);
    for k in 0..batch.adjacency.nnz() {
        assert!(batch.adjacency.rows[k] >= batch.adjacency.cols[k]);
    }
}
```

### V3 — DDR cross-check (deferred to SP-4)

The full bit-for-bit match against DDR's collate output is deferred to
SP-4, where it pays off as part of the loss-equivalence test. SP-3 just
needs to be deterministic given `(staids, window)`; SP-4 will verify the
batch contents match DDR's at the loss boundary.

## Concerns

1. **`Config` schema gap.** The current `src/config.rs` only models the
   routing-engine knobs (`params`). SP-3 needs `data_sources` (the five
   paths) and `experiment.{batch_size, start_time, end_time, rho, warmup}`.
   We extend `Config` here. **YAML round-trip is via `serde_yaml`** (already
   pinned).

2. **Subgraph union may produce a non-contiguous active set.** When two
   gauges' subgraphs share no edges, the union is two disconnected
   components. The MC engine forward-substitution doesn't care (each
   component is solved independently in the same triangular pass), but
   we should verify the lower-triangular invariant still holds across
   the disjoint pieces. Sort-by-position takes care of this — if the
   underlying CONUS order is topological, the disjoint union remains so.

3. **Memory budget at CONUS scale.** Worst case for the training config:
   64 gauges, each with ~5000-10000 upstream segments. Union maxes out
   around 100K segments. Attributes are `10 × 100K × f32 = 4 MB`. q' is
   `~2160 hours × 100K × f32 = 864 MB` — substantial. **This needs
   profiling.** If we OOM on a developer machine, the first knob is
   batch_size; the second is to refactor q' to time-chunked reads.
   Documented but not pre-optimized.

4. **`build_flow_scale` requires CSV columns `FLOW_SCALE` or
   `(COMID_DRAIN_SQKM, COMID_UNITAREA_SQKM)`.** `gages_3000.csv` has
   `FLOW_SCALE` (verified at SP-1 time), so the fast path applies. The
   fallback exists for other gauge files (e.g., `dhbv2_gages.csv`).

5. **The `Sampler` decision is non-load-bearing.** SP-4 will use these
   samplers but the verification bar doesn't depend on PyTorch
   reproducibility. Picking `rand::rngs::StdRng` is fine.

6. **`Comid` index alignment between stores.** `AttributesStore.index` and
   `ConusAdjacencyStore.order` are independent — the NetCDF has 2.94M
   global COMIDs; CONUS adjacency has 346K. The intersection is what
   `MeritGagesDataset` actually uses. We rely on every CONUS COMID
   appearing in the NetCDF (true for MERIT v2). Misses are
   filled by `fill_nans` with the per-attribute row mean.

7. **The `time_axis` used for window sampling is the *experiment*
   range** (e.g., 1981-10-01 to 1995-09-30), not the icechunk store's
   full range. SP-4 will respect this.

## Assumptions

1. The five stores' paths in `config/merit_training.yaml` all exist on
   disk. Tested at SP-1 + SP-2 time.
2. `gages_3000.csv`'s STAID set has substantial overlap with the
   `gages_adjacency.zarr` subgraph set (verified empirically at SP-1).
3. Per-batch memory fits in RAM. If not, the dev machine notices first
   and we add batch-size knobs.
4. f32 throughout. No f64 sneaks past the icechunk-store boundary
   (where the obs store is f64-native; SP-2 already casts).
5. The DDR `gage_catchment` attribute on each subgraph is a string
   (`"comidNNN"`); SP-3 doesn't parse it — it's used only for
   debugging output.

## Module layout summary

```
src/data/
├── mod.rs                  (+) re-export MeritGagesDataset, RoutingBatch,
│                                samplers
├── dataset.rs              (new) MeritGagesDataset, RoutingBatch
├── collate.rs              (new) union_subgraphs, compress,
│                                 build_flow_scale (pub(crate) helpers)
├── sampler.rs              (new) RandomSampler, SequentialSampler
├── statistics.rs           (unchanged)
├── dates.rs                (unchanged)
├── error.rs                (unchanged)
├── ids.rs                  (unchanged)
└── store/                  (unchanged)

src/config.rs               (+) extend with DataSources + Experiment

tests/
└── data_dataset.rs         (new) integration test (V2)
```

Approximate code size: collate.rs ~250 LOC, dataset.rs ~200 LOC,
sampler.rs ~80 LOC, config extension ~80 LOC, tests ~150 LOC. Total
~750 LOC.

## No new dependencies

Everything SP-3 needs is already in `Cargo.toml`: `ndarray`, `serde`,
`serde_yaml`, `rand`, `chrono`, the data-layer crates. The existing
`std::collections::{BTreeSet, HashMap}` cover the dedupe and mapping
needs.

## Risks summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| Lower-triangular invariant violated post-compression | Low | Hard-assert at compress time; fail loudly. |
| `q'` memory blowup at CONUS scale | Medium | Profile; knob batch_size; document threshold. |
| `Config` extension breaks existing tests | Low | Make new fields `Option<>` with `#[serde(default)]`; existing tests still pass. |
| Subgraph-union order depends on `BTreeSet` iteration order | Low | `BTreeSet<usize>` iterates in natural-int order — deterministic. |
| `outflow_idx` semantics differ from DDR | Medium | Unit-test on a synthetic 3-gauge network with known expected output. |
| Attribute alignment between NetCDF and CONUS adj | Low | `AttributesStore.index.position()` returns `Option`; misses fall through to `fill_nans`. |

## What I'm NOT going to do

- No `trait Dataset` / no `Box<dyn Dataset>`. Concrete type per the
  existing data-layer convention.
- No async API. SP-3's `collate` is sync. The icechunk reads inside it
  block_on the per-store runtimes (set up in SP-2).
- No caching, no parallelism. Profile-driven if SP-4 measures slow.
- No PyTorch-bit-identical sampler. Rust-side `rand` is enough.
- No fixture export. V1 + V2 above are the verification floor; V3 is
  SP-4's job.

## Open questions for review

None — small enough to spec in full given the existing infrastructure.
The implementation plan that follows will turn this into ~9 ordered
tasks (`Config` extension, `RoutingBatch`, collate helpers,
samplers, dataset, integration test).

## Next step after approval

Invoke writing-plans → SP-3 implementation plan → subagent-driven
execution (same workflow as SP-1 / SP-2).
