# Gridded (ISIMIP DDM30) routing in ddrs — design

**Date:** 2026-09-08
**Source:** DeepGroundwater/ddr PR #194 ("Gridded (ISIMIP DDM30) routing:
network, attributes, forcing, and training", merged to DDR master).
**Status:** designed autonomously from the user's directive; assumptions are
listed in §7 for review.

## 1. What DDR #194 actually is

DDR #194 is a **data-preparation** feature, not a solver change. Its PR body
says so: "The routing engine, `mmc.py`, `merit.py` and the MERIT/Lynker paths
are untouched. Everything here is additive." It adds:

- `ddr_engine.gridded` — builds a river network from the ISIMIP DDM30 0.5°
  D8 grid, aggregates 25 attributes from native rasters onto cells, regrids
  per-catchment Q′ onto cells (mass-conserving area weights), splits cells
  into Courant≈1 sub-reaches, and validates the products.
- Data products keyed on the **cell id** (`row·720 + col`, row 0 at 55.75°S):
  an attributes NetCDF in the `merit_global_attributes_v2` layout (`COMID` =
  cell id), a Q′ icechunk store (`Qr(divide_id, time)`, `divide_id` = cell id),
  and a **sub-reach adjacency zarr** carrying `order` (node id =
  `cell·1000 + sub`), `parent_cell`, `indices_0/1`, `values`, `length_m`,
  `slope`, `lat`, `lon`.
- Two trainers (`examples/juniata_gridded/train_gridded.py`, `train_conus.py`)
  that feed those products to the unchanged `dmc` router, plus a 1.6 MB
  self-contained Juniata bundle.

ddrs has no `engine/` module and the user asked to keep it that way: **ddrs
consumes DDR's gridded products as another data source; it does not port the
raster/regridding engine.**

## 2. The key observation: ddrs already has the abstraction

ddrs's reach subdivision (`params.subdivision`, `.claude/REACH-SUBDIVISION.md`)
already models exactly the structure the gridded sub-reach network has:

| Gridded sub-reach zarr (DDR) | ddrs subdivided store |
|---|---|
| `parent_cell` per row (contiguous, upstream-first) | `order` (parent COMID repeated per piece), `parent_order`, `parent_offset` |
| attributes looked up by parent cell | `AttributesStore::open(…, &conus.parent_order)` |
| Q′ per cell ÷ `k_per_node` | `pieces_per_row_divisor` in `MuskingumCunge::setup_inputs` |
| gauge outlet = cell's last sub-reach | `position_lookup` = COMID → **last** row; `outlet(p) = parent_offset[p+1]−1` |
| `length_m`, `slope` in the adjacency zarr | `length_m`, `slope` in the adjacency zarr |
| summed-Q′ baseline sums each cell once | `upstream_comids` dedups via `parent_order` |

So the whole routing/training/eval/baseline path is untouched. What is new is
**one builder input**: a reader that turns the DDM30 sub-reach zarr into a
`SubdividedAdjacency`, and the plumbing that lets `ddrs plan` build the managed
stores from it.

```
  data_sources.gridded_network            data_sources.gages (CSV, `cell` col)
  (DDM30 sub-reach zarr from DDR)                     │
            │                                         │
            ▼                                         │
  adjacency::gridded::GriddedNetwork::open            │
    order/parent_cell/indices/length_m/slope          │
    validate: lower-tri, contiguous parents           │
            │                                         │
            ▼ to_subdivided()                         │
  SubdividedAdjacency { order = parent_cell,          │
     parent_order, parent_offset, rows, cols,         │
     length_m, slope }                                │
            │                                         │
            ├──► write_conus_store_subdivided ──► <stem>_adjacency.zarr
            │                                         │
            └──► build_gauge_subgraphs(expanded_view, csv) ◄──┘
                       │  (BFS upstream from the cell's OUTLET piece)
                       ▼
                 write_gauges_store ──► <stem>_gages_adjacency.zarr

  .ddrs/adjacency/<key>/ … then MeritGagesDataset, baseline, train, eval
  run exactly as they do for a subdivided MERIT network.
```

## 3. Changes (blast radius)

| Area | Change | Tier |
|---|---|---|
| `src/config.rs` | `data_sources.gridded_network: Option<PathBuf>`; guards (§4) | C |
| `src/data/store/gage_csv.rs` | `COMID` column accepts `cell` as a serde alias | C |
| `src/adjacency/gridded.rs` (new) | `GriddedNetwork::open`, validation, `to_subdivided` | C |
| `src/adjacency/cache.rs` | `resolve_or_build_gridded`; zarr-dir fingerprint; manifest `network_kind` | C |
| `src/adjacency/zarr_write.rs` | writers take a `geodataset` attr value (`"ddm30"`) | C |
| `src/cli/plan.rs`, `src/cli/sources.rs` | third `resolve_adjacency` branch; lock `gridded_network` | C |
| `examples/juniata_gridded/` | DDR's bundle byte-identical + `ddrs.yaml` + statistics JSON + README | — |
| `scripts/build_attribute_statistics.py` | writes the statistics sidecar for any attributes NetCDF (DDR venv) | — |
| `scripts/snap_gridded_gauges.py` | CONUS gauge→cell snapping (DDR venv) → gauge CSV with `COMID` = cell | — |
| `config/sources/conus-gridded.yaml` | source group for the on-disk CONUS gridded stores | D |
| `tests/gridded_bundle.rs`, `tests/gridded_acceptance.rs` | contract + end-to-end floors | — |

Not touched: `src/routing/`, `src/sparse/`, `src/nn/`, `src/training/`,
`src/data/dataset.rs`, `src/baseline/`. Invariants 1–7 are unaffected; the
Tier A gates still run because `cargo test` includes `ddr_sandbox_match`.

## 4. Config contract

```yaml
data_sources:
  gridded_network: examples/juniata_gridded/data/juniata_subreach_adjacency.zarr
  attributes:      examples/juniata_gridded/data/ddm30_conus_attributes.nc
  streamflow:      examples/juniata_gridded/data/juniata_qprime.ic
  observations:    examples/juniata_gridded/data/juniata_obs.ic
  gages:           examples/juniata_gridded/data/juniata_gage.csv
```

Load-time guards (all in `validate_data_sources` / `validate_subdivision`):

- `gridded_network` is a **third mutually-exclusive adjacency source**:
  with `geospatial_fabric` → error; with `conus_adjacency`/`gages_adjacency`
  → error; with neither of the other two → managed build.
- `geospatial_fabric_layer` with `gridded_network` → error (existing rule:
  the layer key needs a `.gpkg` fabric).
- `params.subdivision.enabled: true` with `gridded_network` → error. The
  gridded graph is already subdivided upstream (DDR sizes sub-reaches from
  celerity at mean flow); ddrs's MERIT subdivision does not apply to it and
  the flag would be silently inert.

The gauge CSV column that names the outlet is `COMID` **or** `cell` (DDR's
gridded CSVs write `cell`). Its value is the DDM30 cell id; the gauge is read
at that cell's most downstream sub-reach.

## 5. Cache key and manifest

`key = blake3(gridded_fp ∥ gages_fp ∥ "gridded" ∥ BUILDER_VERSION)[..16]`
where `gridded_fp` hashes every file under the zarr store directory in sorted
relative-path order (name ∥ NUL ∥ bytes). `params.subdivision` does not enter
the key because it is rejected with `gridded_network`. The manifest reuses
`CacheManifest` with `fabric_path`/`fabric_resolved_path` = the store dir and a
new `network_kind: "ddm30"` field (`#[serde(default)]`, so MERIT manifests
stay readable). Store names use the zarr stem
(`juniata_subreach_adjacency_adjacency.zarr` is ugly; the stem's trailing
`_adjacency` is stripped first → `juniata_subreach_adjacency.zarr` /
`juniata_subreach_gages_adjacency.zarr`).

## 6. Example bundle and tests

`examples/juniata_gridded/data/` is DDR's bundle **byte-identical** (same
readers on both sides is the point, as with `examples/juniata`), plus
`statistics/merit_attribute_statistics_ddm30_conus_attributes.nc.json`
generated by `scripts/build_attribute_statistics.py` (same six statistics DDR's
`set_statistics` writes; `merit_` prefix kept because ddrs derives the sidecar
name from a fixed template). `ddrs.yaml` mirrors `train_gridded.py`:
`n ∈ [0.02, 0.2]`, `q ∈ [0, 1]`, `p ∈ [1, 200]` log-space, `p` default 21,
lr 1e-3 constant, 30 epochs, rho 90, warmup 5, seed 42, CPU solver.

Tests:

- `tests/gridded_bundle.rs` (never skips): bundle opens through
  `GriddedNetwork`; 27 rows / 26 edges / 4 parents with pieces 9,6,6,6; the
  managed build into a tempdir yields a gauges store whose `01567000` subgraph
  covers all 27 rows with `gage_idx` = the last row of cell 138445; the
  statistics cover the ten KAN inputs; `MeritGagesDataset::open` succeeds.
- `tests/gridded_acceptance.rs` (release-only, like `juniata_acceptance`):
  full `train-and-test`, summed-Q′ baseline in a tight band around DDR's
  0.594 NSE (deterministic, a cross-implementation reader check), routed NSE/KGE
  above floors set from a measured ddrs run, routed beats baseline.
- Unit tests in `gridded.rs` (synthetic store), `config.rs` (guards),
  `gage_csv.rs` (`cell` alias), `cache.rs` (key sensitivity).

## 7. Assumptions and known deviations from DDR

1. **KAN head runs in parent (cell) space.** ddrs's subdivision design feeds
   the head `(n_parent, F)` attributes and gathers its outputs onto the pieces.
   DDR's gridded trainer runs the head per sub-reach with `log10_uparea`
   recomputed per piece by accumulating `cell_area/k`. In ddrs every piece of
   a cell shares the cell's parameters and the cell's `log10_uparea`.
   *Why accepted:* mirroring it would change `training/forward.rs` and the
   attribute path for every network (Tier C on the training core) to add
   within-cell variation that DDR's own README calls minor ("with only four
   cells the KAN has four distinct attribute rows"). Revisit only if a CONUS
   comparison shows a material gap.
2. **No `da_ratio` output correction.** `train_conus.py` divides predictions
   by the cell/gauge drainage-area ratio; `train_gridded.py` (the Juniata
   reference) does not. ddrs's `FLOW_SCALE` scales the gauge reach's own
   lateral inflow, a different quantity, so it is *not* used to emulate it.
   The CONUS gauge CSV carries `da_ratio` as an informational column only.
3. **Normalisation statistics are CONUS-wide** (all 5,198 cells), matching
   DDR's `conus.describe()`; the script uses `nanstd` (ddof 0) like DDR's
   MERIT sidecar rather than pandas' ddof 1 — a 1e-4 relative difference.
4. **Cells with no Q′** (21 % of CONUS sub-reach parents) get the icechunk
   reader's 0.001 m³/s fill, DDR's gridded trainer uses 0. Irrelevant for the
   Juniata bundle (all four cells present).
5. **The sub-reach zarr is trusted as topologically ordered and
   lower-triangular**; `GriddedNetwork::open` verifies `rows > cols` and
   parent contiguity and refuses otherwise, rather than re-sorting.
6. **The `gridded_network` key is the dispatch**, not `geodataset:`. The
   example config sets `geodataset: ddm30` for documentation; the field is
   still unread (as it is today for `merit`).

## 8. Concerns

- **Routed skill will not reproduce DDR's 0.795/0.737 exactly**: different
  RNG windows (as with `examples/juniata`) plus deviation 1. The acceptance
  floors are set from ddrs's own measured run, and the baseline band is the
  DDR cross-check.
- **Only the sub-reach form is supported.** DDR also ships an unsplit cell
  adjacency (`ddm30_adjacency.zarr`, no `parent_cell`). It is out of scope:
  DDR's trainers use the sub-reach store, and its README calls the cell
  network "available for cheap iteration" only. A store without
  `parent_cell` is rejected with a message saying so.
- **`ddrs sources use conus-gridded` needs a gauge CSV that does not exist in
  DDR's references yet**; `scripts/snap_gridded_gauges.py` produces it under
  DDR's venv, following the `build_gages_2000_area_balanced.py` precedent.

## 9. Outcome (2026-09-09)

Implemented on branch `worktree-gridded-routing` as designed, with two
findings the design did not anticipate:

1. **`GageSubgraph::upstream_comids` was per-piece on subdivided stores.** It
   deduplicated on row position, so a cell with `k` sub-reaches appeared `k`
   times and the summed-Q′ baseline counted its Q′ `k` times. Fixed to
   deduplicate on the id (a no-op for un-subdivided stores). The same bug
   applied to MERIT `params.subdivision` baselines.
2. **DDR's gridded Q′ stores are `Qr(time, divide_id)`**, the transpose of the
   contract, and zarrs returns fill values for an out-of-range subset rather
   than an error — the fixed-axis reader produced an all-NaN baseline with no
   message. `StreamflowStore::open` now detects the axis order from
   `dimension_names` (shape fallback, refuses neither) and `read_slab` addresses
   the raw buffer accordingly; `tests/fixtures/qr_daily_time_major.ic` pins it.

Results on the Juniata bundle (CPU, 30 epochs):

| | NSE | KGE |
|---|---|---|
| summed-Q′ baseline, ddrs / DDR | 0.594 / 0.594 | 0.672 / 0.671 |
| routed, seed 42 | 0.751 | 0.730 |
| routed, seeds {42, 7, 123} | 0.744–0.758 | 0.728–0.732 |
| routed, DDR Python | 0.795 | 0.737 |

CONUS (`config/experiments/gridded_conus.yaml`, 620 gauges, test window
1995-10-01..1997-09-30): `plan` builds 5,198 cells → 91,867 sub-reaches in
0.6 s; baseline median NSE 0.354 / KGE 0.505. DDR reports 0.390 for its summed
baseline, which includes the `da_ratio` output correction ddrs omits
(deviation 2 in §7). Training beyond a mechanics smoke was not run.

Environment note: the worktree build needed `CUDARC_CUDA_VERSION=13020`
(host CUDA 13.3.1 vs cudarc 0.19.7's table); recorded as trap T12.
