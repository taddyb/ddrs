---
name: ddrs-setup
description: How to set up a fresh ddrs checkout — Rust toolchain, the taddyb/cubecl fork @ ddrs-release, data file paths (zarr/netcdf/icechunk), DDR reference clone for fixture regeneration, and the uv environment.
output: setup.md
sources:
  - Cargo.toml
  - CLAUDE.md
  - README.md
---

# ddrs-setup

> Canonical agent-readable skill. Published chapter at `docs/setup.md`
> is regenerated from this file by `/regenerate-docs`.

## What to know

ddrs has four external dependencies the build resolves against:
1. **Rust stable** (≥ 1.80; tested on 1.94.0).
2. **The taddyb/cubecl and taddyb/burn forks**, both pinned as **git dependencies** via ddrs's `[patch.crates-io]` block in `Cargo.toml` — `cubecl-*` crates → `{ git = "https://github.com/taddyb/cubecl.git", branch = "ddrs-release" }`, `burn-*` crates → `{ git = "https://github.com/taddyb/burn.git", branch = "ddrs-sp7-primitive-ctor" }`. No local clones required — `git clone && cargo build` fetches the forks for you. The cubecl fork carries three ddrs-specific patches over upstream cubecl 0.10: stream accessor, exclusive_with_server, flush_no_sync; the burn fork carries the primitive-constructor accessors for the cuSPARSE GPU solve.
3. **CUDA Toolkit 12+ with a CUDA-12-capable driver** for the GPU path (`sparse_solver: cuda` + `use_cuda_graphs: true` defaults in `config/merit_training.yaml`); validated on driver 575.57.08 (8× A100, sm_80) and a desktop RTX 4080. The CPU path needs none of this and uses `burn-ndarray`.
4. **DDR reference repository** at `~/projects/ddr` for V1 fixture regeneration (`scripts/export_ddr_sandbox.py` runs under DDR's `uv` venv). NB: a valid V1 fixture currently requires the desktop's DDR working tree — see `.claude/references/ddrs-comparing-to-ddr.md` §Regenerating fixtures.

One more pinned dependency, an ordinary git dependency (not a `[patch.crates-io]` override): **`rskan`**, the KAN routing head, pinned to **tag `v0.1.3`** (`rskan = { git = "https://github.com/taddyb/rskan.git", tag = "v0.1.3" }`). When updating it, bump the tag and re-run the KAN parity sweep (CLAUDE.md invariant 7).

## Data file paths

From `CLAUDE.md` "Data sources". The adjacency stores are now **managed**:
the config points at the raw `geospatial_fabric` (MERIT flowlines as
`.shp`/`.dbf`, or a `.gpkg`) and `ddrs plan` builds the CONUS + per-gauge
zarr adjacency stores into `.ddrs/adjacency/<key>/` on first run
(content-addressed, reused afterwards). Only the fabric's attribute table is
read — geometry is never opened.

| Source | Path |
|---|---|
| Geospatial fabric | `riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp` (+ sibling `.dbf`), or a `.gpkg` |
| MERIT adjacency | managed → `.ddrs/adjacency/<key>/` |
| Per-gauge subgraphs | managed → `.ddrs/adjacency/<key>/` |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` |
| Gauges list | `~/projects/ddr/references/gage_info/gages_3000.csv` |

These paths are referenced from `config/merit_training.yaml`'s `data_sources:`
block. If they live elsewhere on a new machine, edit that YAML. To skip the
managed build, drop `geospatial_fabric` and set both `conus_adjacency` and
`gages_adjacency` to pre-built zarr stores instead.

## Setup steps

```bash
# 1. Clone ddrs. The cubecl/burn forks resolve as git dependencies,
#    so no sibling clones are needed.
git clone git@github.com:taddyb/ddrs ~/projects/ddrs

# 2. Clone the DDR reference (for V1 fixtures).
git clone git@github.com:mhpi/ddr ~/projects/ddr
cd ~/projects/ddr && uv sync --all-packages

# 3. Build.
cd ~/projects/ddrs
cargo build --release

# 4. Sanity check: V1 regression must report ABSOLUTE MATCH.
cargo run --release --example compare_ddr_sandbox
```

## Gotchas

- **The cubecl fork branch was renamed** from `ddrs-sp7-stream-accessor` to
  `ddrs-release`. If you have a stale `Cargo.lock` pinning the old branch,
  `cargo update` against the current `Cargo.toml` picks up `ddrs-release`.
- **fixtures/sandbox/** is gitignored. If missing, regenerate via
  `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py`.
- **CUDA defaults are on.** If on a CPU-only machine, override via temp
  YAML: `sparse_solver: cpu`, `use_cuda_graphs: false`.

## Verification

`cargo run --release --example compare_ddr_sandbox` must report
`ABSOLUTE MATCH` with `max abs < 1e-3 m³/s`. This is the V1 invariant from
`CLAUDE.md`; if it fails, the setup is wrong.
