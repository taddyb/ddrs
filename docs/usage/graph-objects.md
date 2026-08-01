# Graph objects

A ddrs run takes raw on-disk adjacency — zarr COO triplets aligned to a
topological `order` array, plus per-reach `length_m` and `slope` — and turns
it into a `MuskingumCunge<I>` that is ready to step once per timestep. This
chapter walks through every graph object that construction touches: the
newtype IDs, the cached `CsrPattern`, the constant `AValuesAssembler`, and
`MuskingumCunge::setup_inputs` as the single binding point. Each one carries
an invariant; getting them right once at `setup_inputs` is what lets every
subsequent timestep be a cheap reuse.

## Where the adjacency comes from

`SparseAdjacency` — the head of the chain below — is built from the COO
triplets in a `ConusAdjacencyStore` / `GagesAdjacencyStore` zarr pair. Those
stores are usually **managed**: with `data_sources.geospatial_fabric` set and
both adjacency keys absent, `ddrs plan` calls

```rust
adjacency::cache::resolve_or_build(workspace_root, fabric, fabric_layer, gages_csv)
```

(`src/adjacency/cache.rs`), which reads the fabric's attribute table
(`.shp` → sibling `.dbf`, or a `.gpkg` via SQL — geometry is never opened),
runs the topological sort, BFSes every gauge subgraph, and writes both zarr
stores into `.ddrs/adjacency/<key>/`. The key is

```
blake3(fabric_fingerprint ∥ gages_fingerprint ∥ [layer] ∥ BUILDER_VERSION)[..16]
```

where the fingerprints are content hashes of the file *bytes*, so moving or
renaming an input does not invalidate the cache and two byte-identical fabrics
share one entry. The `layer` term participates only when set. Builds are
crash-safe (temp dir + atomic rename), and the output matches an engine-built
store element-for-element (`tests/adjacency_parity.rs`). Providing both
`conus_adjacency` and `gages_adjacency` explicitly skips the build entirely.

One filter applies before any of this reaches the solver:
`GageSubgraph::is_headwater()` — true when a gauge's catchment is a single
MERIT divide, so the subgraph has zero edges. `MeritGagesDataset::open` drops
those gauges, and every downstream consumer must drop the same ones: a
zero-edge subgraph has an empty `upstream_comids`, and summing an empty set
silently yields an all-zero prediction rather than an error.

## The construction chain

The path from on-disk adjacency to a stepping solver is four objects deep,
and each is built exactly once:

```
SparseAdjacency   (COO + length_m + slope, plain CPU Vec<f32>)
   │
   ▼ CsrPattern::from_sparse(&adj)
CsrPattern        (Arc-shared; structural-only, no learnable values)
   │
   ▼ AValuesAssembler::<I>::new(&pattern, &device)
AValuesAssembler  (constant uploads of adj_values, diag_mask, row_idx, col_idx)
   │
   ▼ MuskingumCunge::setup_inputs(adj, streamflow, params, carry_state, initial_state)
MuskingumCunge<I> (discharge_t seeded, ready to .forward(...))
```

`setup_inputs` is the **only** function that builds graph objects. After it
returns, every per-timestep call (`route_timestep`) reuses the same
`Arc<CsrPattern>` and `AValuesAssembler`. Rebuilding either inside the loop
defeats the entire SP-6/SP-9 design and tanks GPU throughput — never call
`CsrPattern::from_sparse` inside a training loop.

## Newtype IDs

`src/data/ids.rs` defines two domain ID types:

```rust
pub struct Comid(pub i64);    // MERIT catchment ID
pub struct Staid(String);     // USGS gauge ID, zero-padded to 8 chars
```

Why newtypes — DDR's Python uses raw `int`/`str`, which has been a recurring
bug surface (forgot-to-zfill mistakes, COMID-vs-divide_id mixups). The Rust
newtypes let the compiler catch those mismatches. The convention everywhere in
`ddrs`: use these types, never raw `i64`/`String`. `Staid::new("1563500")`
zero-pads to `"01563500"` to match DDR's canonical form
(`base_geodataset.py:35`, `readers.py:131`):

```rust
pub fn new(s: &str) -> Self {
    let mut padded = s.to_string();
    while padded.len() < 8 {
        padded.insert(0, '0');
    }
    Self(padded)
}
```

The padding is a one-directional pad to width 8 — a string already 8 or more
characters is left untouched, with no upper-bound check.

`IdIndex<T>` is the cross-store boilerplate: every store (`ConusAdjacencyStore`,
`GagesAdjacencyStore`, attribute/streamflow stores) builds one at open time.
Reads consume it via

```rust
pub fn positions_of(&self, ids: &[Id]) -> (Vec<usize>, Vec<usize>)
```

which returns both the resolved positions **and** the indices of the requested
IDs that were missing, so callers can choose to warn, error, or fill with
sentinels (`positions.len() + missing.len() == ids.len()`). See
[Reading inputs](inputs-reading.md) for the full ID-layer story.

## CsrPattern

`CsrPattern` (`src/sparse/mod.rs`) is the cached non-zero structure of the
routing matrix `A = I − c·N`. It is square `[n, n]`, lower-triangular under
topological ordering of `N`, with the diagonal always present — its
contribution comes from `I`, not from `N`.

```rust
pub struct CsrPattern {
    pub n: usize,
    pub crow: Vec<i32>,         // CSR row pointers, length n+1
    pub col: Vec<i32>,          // CSR column indices, length nnz
    pub row_for_nnz: Vec<i32>,  // row index per non-zero, length nnz
    pub adj_values: Vec<f32>,   // N[row,col] at non-zeros (0 at diagonal slots)
    pub diag_mask: Vec<f32>,    // 1 at diagonal slots, 0 elsewhere

    // Transposed-CSR view for the backward solve A^T · gradb = grad_out:
    pub trans_crow: Vec<i32>,
    pub trans_col: Vec<i32>,
    pub trans_to_orig: Vec<i32>,

    pub(crate) cuda_cache: UnsafeSendCache,  // lazy GPU companion
}
```

Within each row, off-diagonals come first in ascending column order, then the
diagonal — this ordering matches both DDR's `PatternMapper` output and the
natural forward-substitution traversal.

`CsrPattern::from_sparse(&adj)` is the primary constructor: `O(nnz log nnz)`,
one sort by `(row, col)`, no `n × n` scan, no dense tensor materialization. It
emits `nnz_off + n` entries — the off-diagonals from the COO plus one diagonal
per row — and carries a `debug_assert!` that each off-diagonal satisfies
`col < row` (lower triangular). The struct is `Clone`, but the `cuda_cache`
field is not cloned — each clone starts with an empty GPU companion that
re-initializes on first GPU solve. A second constructor,
`CsrPattern::from_csr_structure(n, crow, col)`, builds the pattern from
explicit CSR arrays without assuming the `I − c·N` decomposition; it leaves
`adj_values` / `diag_mask` zero and is used by the gradcheck test against
DDR's solver.

The transposed view (`trans_crow`, `trans_col`, `trans_to_orig`) is pre-built
once so the backward solve `A^T · gradb = grad_out` does not have to re-sort at
every timestep. `trans_to_orig[k]` maps the `k`-th non-zero of `A^T` back to
the corresponding slot in `A`'s value array, so the backward can read
`A_values[trans_to_orig[k]]` without rebuilding any structure.

## AValuesAssembler

`AValuesAssembler<I>` (`src/sparse/mod.rs`) holds the four constant tensors
needed to assemble `A_values` differentiably every timestep:

```rust
pub struct AValuesAssembler<I: Backend> {
    n: usize,
    adj: Tensor<Autodiff<I>, 1>,        // adj_values, length nnz
    diag_mask: Tensor<Autodiff<I>, 1>,  // length nnz
    row_idx: Tensor<Autodiff<I>, 1, Int>,
    col_idx: Tensor<Autodiff<I>, 1, Int>,
}
```

All four are pre-uploaded to the device at `setup_inputs` time with no autograd
dependence — they are constants of the network topology, never
gradient-tracked.

`assemble(c)` produces the non-zero values of `A = I − c·N` for a per-row
coefficient vector `c` (length `n`):

```rust
pub fn assemble(&self, c: Tensor<Autodiff<I>, 1>) -> Tensor<Autodiff<I>, 1> {
    let c_at_rows = c.gather(0, self.row_idx.clone());
    self.diag_mask.clone() + c_at_rows.neg() * self.adj.clone()
}
```

Simplified form: `A_values = diag_mask + (−c[row] · adj)`. The naïve
`diag_mask + (1 − diag_mask) · (−c[row] · adj)` is redundant because
`adj[k] == 0` at diagonal slots — the masking with `(1 − diag_mask)` only zeros
out terms that were already zero. Dropping it saves one multiply and one
subtract per timestep, plus their autograd tape nodes, which matters once you
multiply through `O(timesteps) × O(batches)`.

`spmv(q)` does sparse `N · q` for the cached adjacency without a dense matmul:
gather `q[col]`, multiply by `adj`, scatter-add by `row`. All three ops are
BURN-native with built-in autograd, so the adjoint (SpMV by `N^T`) is
registered automatically. Cost: `O(nnz)`. It computes the upstream-inflow term
`c2·(N·Q_t)` in the per-timestep update (`mmc.rs:11`).

## MuskingumCunge::setup_inputs

`setup_inputs` (`src/routing/mmc.rs`) is the binding boundary — the single call
where the raw inputs from the dataloader and the learned head become a
ready-to-step solver. Signature:

```rust
pub fn setup_inputs(
    &mut self,
    inputs: RoutingInputs<I>,                       // adjacency + x_storage
    streamflow: Tensor<Autodiff<I>, 2>,             // [T, n] lateral inflow q'
    params: SpatialParameters<I>,                   // NN outputs in [0,1]
    carry_state: bool,
    initial_state: Option<Tensor<Autodiff<I>, 1>>,  // window-start Q_0, m³/s
)
```

Five parameters, not four. `initial_state` is the state-cache injection
point: when `Some`, that per-reach `Q_0` vector (same order as the network) is
used as-is; when `None`, the `carry_state` / hotstart logic runs exactly as it
did before the field existed, so every call site that passes `None` is
byte-identical to the pre-state-cache code path.

What it does, in order:

1. **Upload `length_m` and `slope`** from the bundled `SparseAdjacency` to
   `Autodiff<I>` tensors. They live as plain `Vec<f32>` on disk and only
   become tensors at the solver boundary. Clamp `slope` to
   `attribute_minimums.slope`.
2. **Build the CSR pattern** — `CsrPattern::from_sparse(&inputs.adjacency)`,
   wrapped in `Arc`. Build the `AValuesAssembler` against it. Store both on
   `self` for the lifetime of this engine instance.
3. **Stash the per-batch state** — `n_segments`, `length`, `slope`,
   `x_storage`, `q_prime`.
4. **Denormalize the NN parameters** — `params.n` and `params.q_spatial`
   always; `params.p_spatial` when provided; and the leakance trio
   `k_d` / `d_gw` / `leakance_factor` when provided (cleared to `None`
   otherwise). Each runs through `denormalize` (`src/routing/utils.rs`) with
   the configured range and log-space flag from `cfg.params` — note the
   log-space lookup for `k_d` matches the string `"K_D"`, DDR's uppercase
   spelling. `params.impervious_mask` is stored **as-is**: it is a constant on
   the inner backend, not autograd-tracked, and is never denormalized.
5. **Seed `discharge_t`** — see [Initial state](#initial-state-carry_state-vs-initial_state)
   below.
6. **SP-10 optional** — eagerly capture the per-timestep CUDA graph if
   `use_cuda_graphs && sparse_solver == Cuda && backend_is_cuda::<I>()`.

After this returns, the engine can be stepped indefinitely without rebuilding
any graph object.

`RoutingInputs<I>` is intentionally minimal — `adjacency`, `length_m`, and
`slope` are bundled inside `SparseAdjacency` (same topological order, loaded
together). `x_storage` (the Muskingum storage weight) is kept separate so it
can be supplied as a learnable or per-batch tensor.

### `SpatialParameters<I>` has eight fields

```rust
pub struct SpatialParameters<I: Backend> {
    pub n: Tensor<Autodiff<I>, 1>,
    pub q_spatial: Tensor<Autodiff<I>, 1>,
    pub p_spatial: Option<Tensor<Autodiff<I>, 1>>,
    /// Leakance params — all-or-nothing. Any `None` ⇒ the non-leakance path.
    pub k_d: Option<Tensor<Autodiff<I>, 1>>,
    pub d_gw: Option<Tensor<Autodiff<I>, 1>>,
    pub leakance_factor: Option<Tensor<Autodiff<I>, 1>>,
    /// Per-reach hard-zero mask (0.0 = impervious, 1.0 = normal).
    /// Inner backend `I` — constant, no gradient. `None` ⇒ all-ones no-op.
    pub impervious_mask: Option<Tensor<I, 1>>,
}
```

Only `n` and `q_spatial` are unconditional. The leakance trio is
**all-or-nothing**: `route_timestep` dispatches to
`timestep_forward_leakance` only when all three are `Some`; any `None` routes
the ordinary path. `impervious_mask` is precomputed by the caller from the
`corridor_impervious` attribute using `cfg.params.leakance_impervious_threshold`
(`MeritGagesDataset::build_impervious_mask`), which is why it arrives as a
plain constant tensor rather than something the head emits.

### Initial state: `carry_state` vs `initial_state`

The two are not peers — `initial_state` wins outright:

```
initial_state = Some(q0)  ──► discharge_t = q0.clamp_min(discharge_lb)
                              (hotstart solve SKIPPED, carry_state IGNORED)

initial_state = None      ──► if !carry_state || discharge_t.is_none():
                                  solve (I − N)·Q_0 = q'_0 via
                                  triangular_csr_solve with c = 1,
                                  clamp to attribute_minimums.discharge
                              else:
                                  keep the existing discharge_t
```

So the cold-start does **not** always run on a first call: with
`initial_state: Some(...)` it never runs at all, even when `discharge_t` is
`None`. Both branches apply the same `attribute_minimums.discharge` floor.

## The `Arc<CsrPattern>` single-instance rule

The sparse path uses **one** `Arc<CsrPattern>` per `MuskingumCunge` instance.
It is built once at `setup_inputs` and reused for every timestep. Never rebuild
it per step.

```rust
self.pattern = Some(Arc::new(CsrPattern::from_sparse(&inputs.adjacency)));
// ...later, per timestep:
let pattern = self.pattern.as_ref().unwrap();          // Arc bump only
let a_values = self.assembler.as_ref().unwrap().assemble(c1);
triangular_csr_solve::<I>(pattern, a_values, rhs, /* cuda */ ...);
```

Why `Arc` — the per-timestep autograd state needs a handle to the pattern
without copying the structural arrays (`crow`, `col`, `row_for_nnz`, plus the
transposed view and the `f32` value arrays). The `Arc` clone is a refcount
bump.

`cusparse.rs` (`CudaPatternCache`, `UnsafeSendCache`) holds a lazy GPU
companion *inside* the `CsrPattern`. The cuSPARSE descriptor handles and the
upload of `crow` / `col` / `adj` to GPU memory happen on the first cuSPARSE
solve call and persist for the lifetime of the pattern. This cache is
per-instance, not global — sharing it across batches with different `n` or
adjacency would be undefined.

## Gotchas

- **Adjacency MUST be topologically sorted and lower-triangular**
  (`rows[k] >= cols[k]`). The forward-sub solver assumes it, and `from_sparse`
  has a `debug_assert!` that fires on the first off-diagonal that breaks it.
  The invariant is tested against the real MERIT CONUS zarr by
  `conus_adjacency_loads_real_merit_zarr`, an **integration** test at
  `tests/data_zarr_store.rs` — run it with
  `cargo test --test data_zarr_store conus_adjacency_loads_real_merit_zarr`.
  If you load adjacency from a new source, run that test first.
- **`setup_inputs` is the ONLY place `CsrPattern` is built.** No public API
  rebuilds it. If you find yourself wanting to call `from_sparse` inside a
  training loop, you are doing something wrong — re-instantiate the
  `MuskingumCunge` instead.
- **`carry_state` semantics — only when `initial_state` is `None`.**
  Within the `None` branch: `carry_state == true` preserves `discharge_t` from
  the previous setup (skips the cold-start solve), `carry_state == false`
  reruns the cold-start, and if `discharge_t.is_none()` (first call) the
  cold-start runs regardless of the flag. But when `initial_state` is `Some`,
  the hotstart solve is skipped unconditionally and `carry_state` is not
  consulted at all — don't reason about `carry_state` without first checking
  which branch you're in.
- **`n` varies between batches.** Gauge subgraphs from `GagesAdjacencyStore`
  are different sizes per batch. The `CudaPatternCache` is **per-instance**
  (inside the `CsrPattern`), not global — different `MuskingumCunge` instances
  with different `n` have independent caches. Don't try to share a
  `CsrPattern` across batches with different topology.
- **`SparseAdjacency::from_dense` is fixtures-only.** It scans the full
  `n × n` array (fine for the 5×5 sandbox and small mock chains). Production
  loaders construct `SparseAdjacency` directly from COO on disk
  (`data::store::zarr`).
- **`Staid::new` zero-pads silently.** Passing `"1563500"` yields `"01563500"`.
  Passing already-padded `"01563500"` is a no-op. Passing a 9-character string
  is left untouched — there is no upper-bound check.

## Reference

| Object | Where | Built at | Role |
|---|---|---|---|
| `Comid` / `Staid` | `src/data/ids.rs` | reader open time | typed domain IDs (newtypes over `i64` / padded `String`) |
| `IdIndex<T>` | `src/data/ids.rs` | store open time | ID → array-position map (`positions_of` reports missing) |
| `GageSubgraph` | `src/data/store/zarr.rs` | `GagesAdjacencyStore::open` | per-gauge COO in CONUS position space; `is_headwater()` gates the dataset's third filter stage |
| managed adjacency cache | `src/adjacency/cache.rs` | first `ddrs plan` | `resolve_or_build` → `.ddrs/adjacency/<key>/`, content-addressed |
| `SparseAdjacency` | `src/sparse/mod.rs` | dataloader | COO triplets + `length_m` + `slope`, plain CPU `Vec<f32>` |
| `CsrPattern` | `src/sparse/mod.rs` | `setup_inputs` | structural-only CSR of `A = I − c·N`, `Arc`-shared, lower-triangular |
| `AValuesAssembler<I>` | `src/sparse/mod.rs` | `setup_inputs` | constant device tensors; `assemble(c)` and `spmv(q)` per timestep |
| `CudaPatternCache` / `UnsafeSendCache` | `src/sparse/cusparse.rs` | first cuSPARSE solve | lazy GPU companion inside `CsrPattern` |
| `MuskingumCunge<I>` | `src/routing/mmc.rs` | `setup_inputs` | the stepping solver; owns the `Arc<CsrPattern>` and assembler |

### Verification

```bash
cargo test --test mmc          # 13 tests
```

The ones that exercise the construction chain end to end —
`SparseAdjacency::from_dense` → `CsrPattern::from_sparse` →
`MuskingumCunge::setup_inputs` → `forward()`:

| Test | Locks |
|---|---|
| `forward_different_network_sizes` | the chain builds and steps at `n ∈ {1, 5, 50, 100}`; output shape `[n, t]`, all values finite and ≥ `attribute_minimums.discharge` |
| `forward_reproducible` | same inputs → same outputs across two independent engines |
| `setup_inputs_uses_hotstart` / `carry_state_skips_hotstart` | the `carry_state` branch of the `initial_state = None` path |
| `carry_state_preserves_discharge_across_setup_inputs_calls` | `discharge_t` survives a second `setup_inputs` with a new forcing window — what chunked eval depends on |
| `setup_inputs_slope_clamping` | `attribute_minimums.slope` is applied at upload |
| `forward_gradients_flow_to_spatial_params` | autograd reaches `n` and `q_spatial` through the whole chain |

Run one with e.g. `cargo test --test mmc forward_different_network_sizes`.

For the full CONUS adjacency invariant (lower-triangular, topological) — this
is an integration test, so `--test`, not `--lib`:

```bash
cargo test --test data_zarr_store conus_adjacency_loads_real_merit_zarr
```

For the managed adjacency builder:

```bash
cargo test --test adjacency_build                 # 10 tests
cargo test --test adjacency_parity -- --ignored   # element-for-element vs the engine store (~10 s, reads the real dbf)
```

## See also

- [Reading inputs](inputs-reading.md) — `ConusAdjacencyStore`, `Comid` /
  `Staid`, and where the COO triplets that feed `from_sparse` come from.
- [Architecture](../architecture.md) — module map and how `MuskingumCunge`
  sits relative to the rest of the routing core.
- [Algorithm](../algorithm.md) — the per-timestep math that runs over the
  assembled `CsrPattern` and `AValuesAssembler`.
- [Performance & CUDA Graphs](../reference/perf.md) — the `CudaPatternCache`
  and SP-10 capture path that live on top of these objects.
