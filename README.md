# ddrs

Differentiable distributed routing. A BURN-based Rust port of the
Muskingum-Cunge routing solver from DDR (Python/PyTorch),
gradient-exact against the reference at single precision.

## Getting started

### Install

```bash
cargo install --path .
```

This puts the `ddrs` binary in `~/.cargo/bin/`. If that directory isn't on
your `PATH`:

```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### First-time setup

From your project root:

```bash
ddrs plan      # probes GPU + smoke test (first run), opens $EDITOR on
               # ddrs.yaml if missing, locks data sources, validates,
               # builds adjacency/baseline caches, prints the plan
ddrs run       # executes the workflow, writes manifest + outputs
```

The first `ddrs plan` runs a 5-reach RAPID sandbox parity check on CUDA when
available and falls back to CPU otherwise — so the install path works on
laptops and CI. The verdict is cached; later plans are fast. When no
`ddrs.yaml` exists, `plan` asks whether to start from your last successful
run's config or the clean bundled template (`config/merit_training.yaml`).
To start fresh at any time: `rm ddrs.yaml && ddrs plan`.

The adjacency stores are **managed**, so `data_sources` only needs the raw
inputs:

```yaml
data_sources:
  geospatial_fabric: /path/to/riv_pfaf_7_..._bugfix1.shp  # MERIT flowlines (.shp/.dbf/.gpkg)
  attributes:        /path/to/merit_global_attributes_v2.nc
  streamflow:        /path/to/merit_dhbv2_UH_retrospective.ic
  observations:      /path/to/usgs_daily_observations
  gages:             /path/to/gages_3000.csv
```

On the first `ddrs plan`, the CONUS and per-gauge adjacency zarr stores are
built from the fabric's attribute table into `.ddrs/adjacency/<key>/` (~10 s
for the CONUS dbf, content-addressed and reused afterwards). The fabric may
also be a GeoPackage — e.g. a merged global MERIT flowlines `.gpkg` — in
which case attributes are read via SQL and geometry is never touched; if the
gpkg holds more than one feature layer, pick one with
`geospatial_fabric_layer: <name>`. If you already have pre-built zarr
stores, drop `geospatial_fabric` and set both `conus_adjacency` and
`gages_adjacency` to their paths instead.

### What lives where

| Path | Written by | Purpose |
|---|---|---|
| `ddrs.yaml` | `ddrs plan` (via `$EDITOR`) | Workflow + experiment config |
| `.ddrs/system.json` | `ddrs plan` | GPU/driver/smoke-test record |
| `.ddrs/sources.lock` | `ddrs plan` | Fingerprints of `data_sources` paths |
| `.ddrs/adjacency/<key>/` | `ddrs plan` (managed build) | Cached CONUS + gauges adjacency zarr stores |
| `.ddrs/runs/<id>/manifest.json` | `ddrs run` | Per-run manifest (config + sources + git SHA + outputs) |
| `output/predictions_latest.zarr` | `ddrs run --workflow eval` / `train-and-test` Phase 2 | Predictions for plotting |
| `output/saved_models_*/epoch_*_mb_*.mpk` | `ddrs run --workflow train` / `train-and-test` Phase 1 | KAN checkpoints |

### Override workflow on the command line

The `workflow:` key in `ddrs.yaml` is what `plan`/`run` use by default. To
override for a single invocation:

```bash
ddrs plan --workflow eval
ddrs run --workflow train
```

`mode:` and `workflow:` must agree (`mode: training` ↔ `workflow ∈ {train, train-and-test}`; `mode: testing` ↔ `workflow: eval`). `ddrs plan` will reject contradictions at load time.

The top-level `device:` key in `ddrs.yaml` selects the CUDA device ordinal
(default `0`, mirrors DDR's `device:` key) — on multi-GPU hosts set e.g.
`device: 1` to keep training off the display/shared GPU.

### Advanced

- `ddrs show <run_id>` — inspect a past run's manifest
- `ddrs status` — list runs
- `ddrs gc` — clean up old run directories
- `ddrs <cmd> --help` for full flag list
