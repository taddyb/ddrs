# Gridded (ISIMIP DDM30) Juniata sample

The gridded counterpart of [`examples/juniata`](../juniata/README.md), and
the ddrs mirror of DDR's `examples/juniata_gridded` (DeepGroundwater/ddr
PR #194). Same basin and gauge (USGS 01567000, 8,657 km²), but the river
network is the ISIMIP DDM30 0.5° drainage-direction grid instead of MERIT
flowlines: four grid cells, split by DDR into 27 sub-reaches of about 7 km.

Everything needed is in `data/` (1.6 MB, byte-identical to DDR's bundle plus
the statistics sidecar). No HPC, S3, CUDA, or external stores.

## Quickstart

From the **repo root** (the config's data paths are repo-root-relative):

    cargo build --release --bin ddrs
    target/release/ddrs --config examples/juniata_gridded/ddrs.yaml plan
    target/release/ddrs --config examples/juniata_gridded/ddrs.yaml run --workflow train-and-test --backend cpu

`plan` relabels DDR's sub-reach store into ddrs's subdivided adjacency
layout, cuts the gauge's upstream subgraph, caches both under
`examples/juniata_gridded/.ddrs/adjacency/<key>/` (gitignored), and prints
the summed-Q' baseline. `run` trains the KAN head for 30 optimizer steps
(one random 90-day window per epoch on a single gauge), then evaluates over
water years 1996–2010. The whole workflow takes well under a minute on CPU.

Reference results (CPU, 30 epochs, test 1995-10-01 – 2010-09-30):

| | NSE | KGE |
|---|---|---|
| ddrs routed (seed 42) | 0.751 | 0.730 |
| ddrs routed, seeds {42, 7, 123} | 0.744–0.758 | 0.728–0.732 |
| DDR (Python) routed | 0.795 | 0.737 |
| summed-Q' baseline (ddrs) | 0.594 | 0.672 |
| summed-Q' baseline (DDR) | 0.594 | 0.671 |

The baseline is deterministic and matches DDR to rounding: the two
implementations read the same bundle through independent readers, so it is
a live cross-implementation check (it is what exposed that DDR's gridded Q'
store is written `Qr(time, divide_id)`, the transpose of the MERIT layout;
ddrs now detects both). The routed gap to DDR has two known sources: the 30
random training windows come from different RNG streams (as in
`examples/juniata`), and ddrs's KAN head runs **per cell** — every
sub-reach of a cell shares the cell's parameters and the cell's
`log10_uparea` — where DDR's gridded trainer runs it per sub-reach with an
accumulated `log10_uparea`. With four cells the field is nearly uniform
either way; DDR's own README calls the sample a demonstration that the
gridded network trains and routes, not a converged model.

## How the gridded inputs map onto ddrs

ddrs has no `engine/` module and ports none of DDR's raster aggregation or
Q' regridding. It consumes DDR's products as a data source: the new
`data_sources.gridded_network` key points at DDR's sub-reach adjacency
zarr, and everything downstream is the existing subdivided-network path:

| DDR gridded product | ddrs reads it as |
|---|---|
| `juniata_subreach_adjacency.zarr` — `order` (node id = cell·1000 + sub), `parent_cell`, `indices_0/1`, `length_m`, `slope` | a subdivided CONUS store: parent = cell, pieces = the cell's contiguous sub-reaches, outlet = last piece |
| `ddm30_conus_attributes.nc` — 25 attributes, `COMID` = cell id (all 5,198 CONUS cells, so normalisation matches a CONUS run) | the attributes table, looked up per cell; sidecar `data/statistics/merit_attribute_statistics_ddm30_conus_attributes.nc.json` from `scripts/build_attribute_statistics.py` |
| `juniata_qprime.ic` — `Qr(time, divide_id)` daily m³/s per cell, 1980–2010 | daily Q' per cell, split evenly across the cell's sub-reaches at routing time (`pieces_per_row_divisor`) |
| `juniata_obs.ic` — `streamflow(gage_id, time)`, USGS 01567000 | observations, unchanged |
| `juniata_gage.csv` — one row with the snapped `cell` and `da_ratio` | the gauge table; `cell` is read as the outlet id, routed at that cell's last sub-reach. `da_ratio` is informational (no output correction, as in DDR's `train_gridded.py`) |

Design and the full list of deviations:
`docs/superpowers/specs/2026-09-08-ddrs-gridded-routing-design.md`.

## Tests

    cargo test --test gridded_bundle                 # bundle contract, never skips
    cargo test --release --test gridded_acceptance   # end-to-end metric floors

## CONUS

`config/sources/conus-gridded.yaml` names the on-disk CONUS products
(91,867 sub-reaches, 620 gauges from `scripts/snap_gridded_gauges.py`), and
`config/experiments/gridded_conus.yaml` mirrors DDR's `train_conus.py`
defaults. Both need the workstation stores under `/mnt/ssd1` and
`~/projects/ddr/data/ddm30`.

## Regenerating the bundle

Maintainer only. Regenerate in the ddr repo
(`examples/juniata_gridded/extract_bundle.py`), copy `data/` here, then

    ~/projects/ddr/.venv/bin/python scripts/build_attribute_statistics.py \
        examples/juniata_gridded/data/ddm30_conus_attributes.nc
