---
name: ddrs-eval-plots
description: Generate and run plotting notebooks for the output of a ddrs training run, and interpret the results. Use whenever the user wants to visualize or evaluate a run — hydrographs of predicted vs observed streamflow, spatial maps of learned Manning's n / Leopold-Maddock p / q over MERIT basins, parameter-convergence drift across epochs, metric distributions (NSE, KGE, bias, RMSE, FHV, FLV) compared against the summed-Q' baseline, or DDR-vs-ddrs parameter parity. Trigger on "plot my trained model", "eval plots", "make a hydrograph", "plot Manning's n", "NSE distribution", "CDF", "box plot", "did it beat the baseline", "have the parameters converged", or a vague gesture at "the latest run". For building, configuring, testing, or debugging ddrs itself, use ddrs-dev instead.
---

# ddrs eval plots

Visualize and interpret what a trained ddrs KAN routing model produced. Mirrors the
plot families from DDR's reference notebooks (`~/projects/ddr/examples/eval/`) and
`ddr.validation.plots`, adapted to ddrs's output schemas.

## Contents

**This file** — Where everything lives (run layout; the two NetCDF files) · Python
environment · Workflow (pick family → read reference → generate/report/run) ·
Missing inputs · Interpreting what you plotted (baseline bar, convergence, zeta) ·
Conventions · When NOT to use · Files

**`references/hydrograph.md`** (98 lines)
Inputs (predictions zarr schema; user selection) · Notebook template · Notes
(warmup, `obs<=0` sentinel, per-slice metrics)

**`references/metrics.md`** (211 lines)
Inputs (predictions zarr; **raw-f32 baseline**; optional gauges CSV) · Metric helper
(`ddr.validation.Metrics` field list) · Notebook template: box plot of 6 metrics,
NSE CDF, drainage-area boxplots, gauge map · **§Is this a win?** — the baseline bar
and the two population traps · Notes

**`references/parameter_map.md`** (455 lines)
Inputs (`plot/kan_parameters.nc` variables; MERIT fabric) · Notebook template ·
Global-fabric runs (incl. producing the NetCDF for a managed-adjacency run) ·
Companion cells: distribution histogram, parameter vs log10(drainage area) hexbin ·
**§Convergence: has training actually moved the parameters?** — per-epoch dumps,
the four diagnostics, template · Notes

**`references/channel_geometry.md`**
Baseflow width & depth for all 346,321 CONUS reaches from the learned `n`, `p`,
`q` and post-clamp slope · CONUS maps of width / depth / `w:d` · **downstream
hydraulic-geometry exponents vs Leopold & Maddock** — the only internal test of
whether `p_spatial`/`q_spatial` are physically sensible, since the attributes
carry no width or depth to validate against · plausibility bands

**`references/parity.md`** (235 lines)
Init-time parity (when to use, inputs, load → histograms → pass/fail, KS criterion) ·
Trained parity (inputs, load → per-distribution stats → histograms → per-reach
scatter → verdict, KS + Spearman criteria)

**`scripts/load_ddrs_predictions.py`** — `load_predictions_zarr` and
`load_baseline_f32`. Always use these; they handle the zarr-v3 `dimension_names`
gap and the `(G, W) uint8` gage_ids encoding.

## Where everything lives

**One run directory holds every input you need.** There is no `output/saved_models*/`
layout anymore — those directories are flat pre-2026-06 `.mpk` files and are dead.

```
.ddrs/runs/<id>/                        # <id> = <UTC ts>-[<group>-]<workflow>
├── config.yaml                         # the config that produced this run
├── manifest.json                       # metrics, git sha, outputs, sources
├── run.log
├── checkpoints/epoch_E_mb_M/           # head.mpk, optim.mpk, state.json
├── eval/predictions.zarr               # predictions + observations (G,T)
├── baseline/                           # predictions.f32, observations.f32, manifest.json
├── kan_parameters.nc                   # eval-network zeta diagnostic (leakance runs only)
├── plot/kan_parameters.nc              # full-CONUS learned parameters (`ddrs run --plot`)
└── plots/                              # ← WRITE NOTEBOOKS AND PNGs HERE
```

**Always save into `<RUN_DIR>/plots/`** so artifacts travel with the run. Never
`<CKPT_DIR>/plots/` — that would bury them inside `checkpoints/epoch_30_mb_0/`.

Find the right run: `ls -t .ddrs/runs/ | head`, or `ddrs status`, or — most
authoritative — read the predictions zarr's root `model` attribute, which records the
checkpoint that produced it.

### Two different NetCDF files, often confused

| File | Dimension | Variables | Writer |
|---|---|---|---|
| `plot/kan_parameters.nc` | `COMID` (346,321 full CONUS) | `n`, `q_spatial`, `p_spatial`, `x_storage`, `slope` (+ `K_D`, `d_gw`, `leakance_factor` when `use_leakance`) | `dump_parameters` / `ddrs run --plot` |
| `kan_parameters.nc` | `COMID_eval` (64,892 gauge-subgraph union) | `zeta` (mean \|ζ\|, m³/s), `zeta_net` (signed; + = losing), `depth_mean`, `area_z_mean`, `q_mean` | `eval --zeta-output`, or `train-and-test` Phase 2 |

`dump_parameters` can never write `zeta` — it needs routed per-timestep depth, which
only exists during eval.

## Python environment

Everything runs from `./ddrs-py`. First-time setup (idempotent):

```bash
cd ./ddrs-py && uv sync --extra plots
```

Execute a notebook in place:

```bash
cd ./ddrs-py && uv run jupyter nbconvert --to notebook --execute \
    <absolute_notebook_path> --output <basename> --output-dir <RUN_DIR>/plots
```

Verified working on this host (2026-07-30): `ddr` 0.5.3.dev3, `torch` 2.12.0+cu130,
`xarray` 2026.4.0, `zarr` 3.2.1, `geopandas` 1.1.3, `contextily`, `matplotlib`,
`pyogrio`, `netCDF4`, `scipy`. `cartopy` is **absent and not needed** —
`plot_gauge_map` uses geopandas + contextily.

`from ddr.validation import Metrics, plot_box_fig, plot_cdf, plot_drainage_area_boxplots, plot_gauge_map`
all resolve; do not reimplement them.

**If `uv` can't build `ddrs-py`** (hosts without the Rust `burn-*` path deps, e.g.
wukong): skip `uv` and call the interpreter directly —
`ddrs-py/.venv/bin/python`, `ddrs-py/.venv/bin/jupyter nbconvert …`. This bypasses
the maturin rebuild and runs against what is already installed. Confirm importability
first before assuming `ddr` is available.

Do **not** shell into `~/projects/ddr` and run from there — that pollutes the project
boundary.

## Workflow

### 1. Pick the family

| User asks about | Family | Reference |
|---|---|---|
| hydrograph, time series, predicted vs observed, gauge X in year Y | **hydrograph** | `references/hydrograph.md` |
| NSE, KGE, bias, RMSE, FHV, FLV, CDF, box plot, "did it beat the baseline" | **metrics** | `references/metrics.md` |
| Manning's n, p_spatial, q_spatial, slope, map, basin, spatial pattern | **parameter_map** | `references/parameter_map.md` |
| "have the parameters converged", epoch drift, movement across epochs | **parameter convergence** | `references/parameter_map.md` §Convergence |
| width, depth, channel geometry, w:d ratio, "are the geometry parameters right", hydraulic geometry, Leopold & Maddock | **channel_geometry** | `references/channel_geometry.md` |
| DDR-vs-ddrs parameter distributions, at init or trained | **parity** | `references/parity.md` |

Vague request ("plot my trained model")? Offer the default bundle:
hydrograph + metrics + parameter_map.

### 2. Read the reference before writing the notebook

Each reference carries the exact schema, a runnable template, and the DDR-inherited
conventions. **Do not invent column names** — the schemas are documented there and
the writers are cited.

### 3. Generate, report, offer to run

Write `<RUN_DIR>/plots/<name>.ipynb` with a top markdown cell recording: run id,
checkpoint, inputs, region/gauge/year selected, date generated.

Then report the paths — **non-optional**:

```
notebook → <absolute path to .ipynb>
plots will save to → <RUN_DIR>/plots/
```

Offer to execute; only run if the user agrees (zarr reads and the MERIT shapefile
join are slow). **After execution, list every PNG with its absolute path.** Do not
summarize as "plots are in `<dir>`".

## Missing inputs — generate them

**`plot/kan_parameters.nc` absent (parameter maps).** One deterministic command with
an output path you control — offer to run it, and on consent run it. It is a
`cargo build` + GPU job (minutes, ~70 MB), so confirm first unless already
authorized.

```bash
cargo run --release --bin dump_parameters -- \
  --config <run_dir>/config.yaml \
  --checkpoint <run_dir>/checkpoints/<epoch_E_mb_M>/head \
  --output <run_dir>/plot/kan_parameters.nc
```

Note `--checkpoint` takes the **head base** (no `.mpk`) here, while `eval` takes the
**directory**. Optional `--batch-size` (default 50000) and `--backend` (default
`cuda`).

If it errors with `conus_adjacency not resolved — invoke via ddrs run --plot`, use
the throwaway-config recipe in `references/parameter_map.md`.

**`eval/predictions.zarr` absent.** That is a full routing run, not one forward pass.
Quote the command (`ddrs run --workflow train-and-test`, or the `eval` binary against
an existing checkpoint) and let the user run it.

## Interpreting what you plotted

### Did it beat the baseline?

The bar is the **summed-Q′ baseline on the same gauge population**. For CONUS with
the dHBV2-UH store that is **median NSE 0.6781 / KGE 0.7172 on 2,365 gauges**.

> Never use `0.689 / 0.723` as a CONUS bar — that is a *global* MERIT number on a
> different 5,224-gauge network. And never compare a 2,365-gauge trained median to a
> 3,211-gauge baseline median: the raw gage list contains 513 single-divide gauges
> that scored phantom zeros before the 2026-07-28 fix. Both series must come from the
> same population and the same metric code.

Best documented trained result: **0.7152 NSE / 0.7106 KGE** (precip-driven disagg +
L1, run `2026-06-23T02-49-12Z-conus-hourly-train-and-test`) — that is +0.037 NSE and
−0.007 KGE against the baseline. Full table and the 2026-07-30 KGE qualification:
`ddrs-dev` → `references/research-status.md`.

`ddrs show <id> | grep -E "nse|kge|loss"` is the cheap pre-plot triage.

### Have the parameters converged?

Read `references/parameter_map.md` §Convergence. The signals that matter: late-half
movement as a fraction of total movement, whether the IQR is contracting or
expanding, realized span as a percentage of the declared range, and the fraction of
reaches at a bound. Expanding IQR plus a realized span of a few percent of the
declared range means **near initialization, not saturated** — under-training, not
convergence.

For comparing two runs' parameter fields, the equifinality read-out is
`median(|Δn|) / IQR(n)` on a same-seed ON/OFF pair: **< 0.10** none · **0.10–0.49**
weak or ambiguous · **≥ 0.50** confirmed. Caveats: single seed per arm, and CUDA
scatter-add nondeterminism adds ~2–5% to parameter distributions — treat as
suggestive, and if ambiguous run each arm twice to compare within-arm spread against
the cross-arm shift.

### Reading a zeta variable

`zeta` is dimensioned by the **eval network** (64,892 reaches for the 2,365-gauge
set), not the 346,321-reach CONUS network. The historical activity bar was
`|zeta| > 0.01 m³/s` on ≥10% of eval reaches. That bar was met (10.4%) and the term
was still ruled NO-GO — passing it is necessary, not sufficient.

## Conventions (match DDR's `plots.py` and `evaluate.ipynb`)

- **Warmup**: drop the first 3 timesteps from hydrographs.
- **Metric clipping**: NSE/KGE clipped to `[-1, 1]` before plotting.
- **Basemap**: CartoDB.Positron, alpha 0.6, attribution off.
- **CONUS bounds**: `xlim=(-125, -66)`, `ylim=(24, 53)`.
- **Colormaps**: `plasma_r` for Manning's n (high n = rough = red), `viridis` for
  p/q, `Blues` for depth/width, `bamako` or `plasma` for NSE.
- **Save kwargs**: `dpi=300, bbox_inches="tight", facecolor="white"`.
- Report **medians**, not means — the per-gauge NSE/KGE distribution is strongly
  left-skewed and a few catastrophic gauges dominate the mean.

## When NOT to use this skill

- Building, configuring, testing, or debugging ddrs → `ddrs-dev`.
- Plotting DDR-Python output directly → DDR's own notebooks.
- `examples/benchmark_hydrograph.rs` (the 10-reach synthetic chain) → a Rust example,
  not a trained model.
- Debugging gradient parity against DDR → `examples/compare_ddr_sandbox.rs`.
- **H5/H6 selective-equifinality plots** — campaign CLOSED, both **INCONCLUSIVE**.
  The authoritative analysis is `scripts/h5_h6_audit_analysis.py` +
  `docs/2026-07-09-h5-h6-equifinality-v2-findings.md`. Do **not** regenerate the v1
  figures: the `f_n ≥ 2/3` bars used an unpaired variance estimate (the correct
  paired test has 15–40× smaller variance), and the `min × 1.05` sublevel contour
  saturated — it swallowed 100–105 of 121 grid points. Both instruments are refuted.

## Files

- `references/hydrograph.md` — single-gauge predictions vs observations
- `references/metrics.md` — NSE/KGE/bias distributions, CDFs, box plots vs baseline
- `references/parameter_map.md` — learned parameters over MERIT polygons, plus
  epoch-to-epoch convergence drift
- `references/channel_geometry.md` — baseflow width/depth over MERIT, plus the
  Leopold & Maddock exponent check on `p_spatial`/`q_spatial`
- `references/parity.md` — DDR-vs-ddrs parameter distributions at init and trained
- `scripts/load_ddrs_predictions.py` — **always use this** to open the predictions
  zarr and the f32 baseline. It handles two pitfalls every notebook otherwise hits:
  ddrs writes zarr v3 with `_ARRAY_DIMENSIONS` but no `dimension_names` (so
  `xr.open_zarr` raises `KeyError`), and `gage_ids` is `(G, W) uint8` fixed-width
  ASCII (W = max(id length, 8)), not 1D bytes.
- `evals/evals.json` — test prompts
