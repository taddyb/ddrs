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
2. **The taddyb/cubecl fork**, pinned as a git dependency (`github.com/taddyb/cubecl`, branch `ddrs-release`) via ddrs's `[patch.crates-io]` block in `Cargo.toml` — no local clone required. Carries three ddrs-specific patches over upstream cubecl 0.10: stream accessor, exclusive_with_server, flush_no_sync.
3. **CUDA Toolkit 12+ with a CUDA-12-capable driver** for the GPU path (`sparse_solver: cuda` + `use_cuda_graphs: true` defaults in `config/merit_training.yaml`); validated on driver 575.57.08 (8× A100, sm_80) and a desktop RTX 4080. The CPU path needs none of this and uses `burn-ndarray`.
4. **DDR reference repository** at `~/projects/ddr` for V1 fixture regeneration (`scripts/export_ddr_sandbox.py` runs under DDR's `uv` venv). NB: a valid V1 fixture currently requires the desktop's DDR working tree — see `.claude/skills/ddrs-comparing-to-ddr.md` §Regenerating fixtures.

## Data file paths

From `CLAUDE.md` "Data sources":

| Source | Path |
|---|---|
| MERIT adjacency | `~/projects/ddr/data/merit_conus_adjacency.zarr` |
| Per-gauge subgraphs | `~/projects/ddr/data/merit_gages_conus_adjacency.zarr` |
| Catchment attributes | `~/projects/ddr/data/merit_global_attributes_v2.nc` |
| Streamflow forcing | `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic` |
| USGS observations | `/mnt/ssd1/data/icechunk/usgs_daily_observations` |

These paths are referenced from `config/merit_training.yaml`. If they live
elsewhere on a new machine, edit that YAML.

## Setup steps

```bash
# 1. Clone ddrs + cubecl fork.
git clone git@github.com:taddyb/ddrs ~/projects/ddrs
git clone -b ddrs-release git@github.com:taddyb/cubecl ~/projects/cubecl

# 2. Verify the cubecl path in Cargo.toml matches your clone location.
grep "/home/.*cubecl/crates/" ~/projects/ddrs/Cargo.toml

# 3. Clone the DDR reference (for V1 fixtures).
git clone git@github.com:mhpi/ddr ~/projects/ddr
cd ~/projects/ddr && uv sync --all-packages

# 4. Build.
cd ~/projects/ddrs
cargo build --release

# 5. Sanity check: V1 regression must report ABSOLUTE MATCH.
cargo run --release --example compare_ddr_sandbox
```

## Gotchas

- **The fork's branch was renamed** from `ddrs-sp7-stream-accessor` to
  `ddrs-release`. If you cloned earlier, `git fetch origin && git checkout
  ddrs-release` on the fork.
- **fixtures/sandbox/** is gitignored. If missing, regenerate via
  `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/export_ddr_sandbox.py`.
- **CUDA defaults are on.** If on a CPU-only machine, override via temp
  YAML: `sparse_solver: cpu`, `use_cuda_graphs: false`.

## Verification

`cargo run --release --example compare_ddr_sandbox` must report
`ABSOLUTE MATCH` with `max abs < 1e-3 m³/s`. This is the V1 invariant from
`CLAUDE.md`; if it fails, the setup is wrong.
