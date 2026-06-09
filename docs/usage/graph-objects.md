# Graph objects

A ddrs run takes raw on-disk adjacency — zarr COO triplets aligned to a
topological `order` array, plus per-reach `length_m` and `slope` — and turns
it into a `MuskingumCunge<I>` that is ready to step once per timestep. This
chapter walks through every graph object that construction touches: the
newtype IDs, the cached `CsrPattern`, the constant `AValuesAssembler`, and
`MuskingumCunge::setup_inputs` as the single binding point. Each one carries
an invariant; getting them right once at `setup_inputs` is what lets every
subsequent timestep be a cheap reuse.

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
   ▼ MuskingumCunge::setup_inputs(adj, streamflow, params, carry_state)
MuskingumCunge<I> (cold-start solved, discharge_t seeded, ready to .forward(...))
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
    inputs: RoutingInputs<I>,           // adjacency + x_storage
    streamflow: Tensor<Autodiff<I>, 2>, // [T, n] lateral inflow q'
    params: SpatialParameters<I>,       // n, q_spatial, p_spatial in [0,1]
    carry_state: bool,
)
```

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
4. **Denormalize the NN parameters** — `params.n`, `params.q_spatial`, and
   `params.p_spatial` (if provided), each run through `denormalize`
   (`src/routing/utils.rs`) with the configured range and log-space flag from
   `cfg.params`.
5. **Cold-start** — if `!carry_state || discharge_t.is_none()`, solve
   `(I − N) · Q_0 = q'_0` by calling `triangular_csr_solve` with `c = 1` (an
   all-ones vector assembled through the same `AValuesAssembler`). Clamp the
   result to `attribute_minimums.discharge` and store in `self.discharge_t`.
6. **SP-10 optional** — eagerly capture the per-timestep CUDA graph if
   `use_cuda_graphs && sparse_solver == Cuda && backend_is_cuda::<I>()`.

After this returns, the engine can be stepped indefinitely without rebuilding
any graph object.

`RoutingInputs<I>` is intentionally minimal — `adjacency`, `length_m`, and
`slope` are bundled inside `SparseAdjacency` (same topological order, loaded
together). `x_storage` (the Muskingum storage weight) is kept separate so it
can be supplied as a learnable or per-batch tensor.

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
  The invariant is tested against the real MERIT CONUS zarr via
  `data::store::zarr::tests::conus_adjacency_loads_real_merit_zarr`. If you
  load adjacency from a new source, run that test first.
- **`setup_inputs` is the ONLY place `CsrPattern` is built.** No public API
  rebuilds it. If you find yourself wanting to call `from_sparse` inside a
  training loop, you are doing something wrong — re-instantiate the
  `MuskingumCunge` instead.
- **`carry_state` semantics.** `carry_state == true` preserves `discharge_t`
  from the previous setup (skips the cold-start solve). `carry_state == false`
  reruns the cold-start. If `discharge_t.is_none()` (first call), the
  cold-start runs regardless of the flag.
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
| `SparseAdjacency` | `src/sparse/mod.rs` | dataloader | COO triplets + `length_m` + `slope`, plain CPU `Vec<f32>` |
| `CsrPattern` | `src/sparse/mod.rs` | `setup_inputs` | structural-only CSR of `A = I − c·N`, `Arc`-shared, lower-triangular |
| `AValuesAssembler<I>` | `src/sparse/mod.rs` | `setup_inputs` | constant device tensors; `assemble(c)` and `spmv(q)` per timestep |
| `CudaPatternCache` / `UnsafeSendCache` | `src/sparse/cusparse.rs` | first cuSPARSE solve | lazy GPU companion inside `CsrPattern` |
| `MuskingumCunge<I>` | `src/routing/mmc.rs` | `setup_inputs` | the stepping solver; owns the `Arc<CsrPattern>` and assembler |

### Verification

```bash
cargo test --test mmc mc_routes_linear_chain
```

The `mc_routes_linear_chain` test (5-reach linear chain) exercises the full
chain — `SparseAdjacency::from_dense` → `CsrPattern::from_sparse` →
`MuskingumCunge::setup_inputs` → repeated `route_timestep` — and compares the
output to a hand-rolled cumsum baseline. Passing it confirms the graph objects
are built and reused correctly.

For the full CONUS adjacency invariant (lower-triangular, topological):

```bash
cargo test --lib data::store::zarr::tests::conus_adjacency_loads_real_merit_zarr
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
