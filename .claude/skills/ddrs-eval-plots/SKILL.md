---
name: ddrs-eval-plots
description: Generate plotting notebooks for the output of a ddrs-trained KAN routing model. Use this whenever the user wants to visualize a ddrs training run — hydrographs of predicted vs. observed streamflow, spatial maps of learned Manning's n / Leopold-Maddock p / q parameters over MERIT-Hydro basins, or metrics distributions (NSE, KGE, bias, RMSE, FHV, FLV) across evaluation gauges. Trigger on phrases like "plot my trained model", "visualize the trained KAN", "make a hydrograph for gauge X", "plot Manning's n over basin Y", "show NSE distribution", "evaluation plots", or any request to inspect a checkpoint under `output/saved_models*/`. Also trigger when the user gestures vaguely at "the latest model" or "my training run" — the skill knows how to find inputs.
---

# ddrs trained-model plotting

This skill generates Jupyter notebooks that visualize the output of a trained ddrs KAN routing model. It mirrors the plot families from DDR's reference notebooks (`~/projects/ddr/examples/eval/evaluate.ipynb`, `~/projects/ddr/examples/merit/plot_parameter_map.ipynb`) and DDR's `ddr.validation.plots` library, but adapted to ddrs's output schemas.

## Why this exists

ddrs writes two artifacts after training:

| Artifact | Producer | Schema |
|---|---|---|
| Predictions zarr | `cargo run --bin eval` | groups `gage_ids`, `time`, `predictions (G,T)`, `observations (G,T)` |
| KAN parameter NetCDF | `cargo run --bin dump_parameters` | dim `COMID`, vars `n`, `q_spatial`, `p_spatial`, `slope` |

Visualizing them well requires three families of plots, each with its own data dependencies and conventions. Instead of asking the user to assemble matplotlib calls from scratch every time, this skill picks the right recipe, generates a notebook, and saves output PNGs next to the checkpoint so the artifacts travel with the model run.

## How it works

The skill produces a Jupyter notebook (and runs it if asked) that:
1. Loads the appropriate ddrs output artifact(s)
2. Uses `ddr.validation` helpers (`Metrics`, `plot_*`) — DDR is installed as a
   dependency in `ddrs-py`'s `plots` extra; the skill does not reimplement these
3. Saves PNGs into `<RUN_DIR>/plots/` (or `<CKPT_DIR>/plots/` for legacy `output/saved_models*/` layouts) so plots travel with the run

**All execution happens from `./ddrs-py`** — the Python venv lives there, not under `~/projects/ddr`. First-time setup (idempotent):

```bash
cd ./ddrs-py && uv sync --extra plots
```

Run the resulting notebook in place:

```bash
cd ./ddrs-py && uv run jupyter nbconvert --to notebook --execute \
    <absolute_notebook_path> --output <basename> --output-dir <PLOT_DIR>
```

Or open interactively: `cd ./ddrs-py && uv run jupyter notebook <absolute_notebook_path>`.

If `uv sync --extra plots` fails because `ddr` isn't at `file:///home/tbindas/projects/ddr` on this machine, update the `ddr @ file://...` line in `ddrs-py/pyproject.toml` to point at the local DDR checkout. Do NOT shell into `~/projects/ddr` and run from there — that's the old workflow and pollutes the ddrs project boundary.

**Host where `uv` can't build `ddrs-py` (e.g. wukong).** On nodes where the
Rust `burn-*` path deps are absent, `uv sync`/`uv run` abort trying to rebuild
the maturin package, so `ddr` + `torch` never install. If the venv was synced
once before the breakage it still has the plotting stack (xarray, zarr,
geopandas, contextily, matplotlib, pyogrio, netCDF4 — but NOT `cartopy`,
`torch`, or `ddr`). Two adjustments:
1. **Skip `uv` and call the interpreter directly**: `.venv/bin/python`,
   `.venv/bin/jupyter nbconvert --to notebook --execute ...`. This bypasses the
   rebuild and runs against the already-installed packages.
2. **The metrics family can't `from ddr.validation import Metrics, plot_*`** —
   reimplement them inline (NSE/KGE/bias/RMSE in numpy; FHV top-2% / FLV
   bottom-30% log-space per Yilmaz 2008; boxplot/CDF/gauge-map in
   matplotlib+geopandas). Compute BOTH the run and the baseline from raw arrays
   with the same numpy code so the two series stay apples-to-apples. Confirm
   what's importable first: `for m in ("torch","ddr","cartopy","geopandas","pyogrio"): ...`.

## Workflow

### Step 1 — Identify the plot family

Pick one (or more — emit one notebook per family):

| If user asks about… | Use family |
|---|---|
| hydrograph, time series, predicted vs observed, gauge X over year Y | **hydrograph** → `references/hydrograph.md` |
| Manning's n, p_spatial, q_spatial, slope, map, basin, MERIT, area | **parameter_map** → `references/parameter_map.md` |
| NSE, KGE, bias, RMSE, FHV, FLV, distribution, CDF, boxplot, metrics | **metrics** → `references/metrics.md` |

If the user is vague ("plot my trained model"), ask which family — but offer to emit all three as a default plot bundle if they don't have a preference.

### Step 2 — Locate inputs

Detect the checkpoint and infer where its outputs should live. ddrs convention:

```
output/
├── saved_models_1/                       ← checkpoint directory ($CKPT_DIR)
│   ├── epoch_5_mb_35.mpk
│   └── plots/                            ← write notebooks + PNGs here
├── predictions_latest.zarr               ← from `cargo run --bin eval`
└── kan_parameters_latest.nc              ← from `cargo run --bin dump_parameters`
```

**Finding the right checkpoint dir** — multiple `saved_models_*` directories may coexist on disk. Use this priority order:

1. **Predictions zarr's `model` attribute** (highest authority). `eval` records the source checkpoint path in the zarr's metadata — read it with `xr.open_zarr(path).attrs["model"]` and parse out the parent dir. This guarantees plots land next to the checkpoint that actually produced the predictions.
2. **User-specified checkpoint** in the prompt.
3. **Newest `saved_models_*` by mtime** as a last resort.

Tie-breaker hint when comparing two checkpoint dirs: KAN-head checkpoints (`rskan`) are typically ~20 KB; older MLP-placeholder checkpoints are ~3 KB. If the user's working with the current architecture, prefer the larger.

**Missing input artifacts — generate them, don't dead-end.** The plot families
depend on two producible artifacts:

- **Parameter NetCDF (`kan_parameters.nc`) for the parameter-map family.** This
  is a single deterministic command whose output path you control, so when it's
  absent **offer to run it and, on consent, run it for the user** — don't just
  quote the command and stop. It IS a `cargo build` + CUDA GPU job (minutes, GPU
  memory, writes ~70 MB), so confirm first unless the user already told you to
  proceed; then launch it in the background and monitor. Command (resolve
  `--config`/`--checkpoint` from the run dir; checkpoint base is the predictions
  zarr's `model` attr without the `.mpk`):

  ```bash
  cargo run --release --bin dump_parameters -- \
    --config <run_dir>/config.yaml \
    --checkpoint <run_dir>/checkpoints/<epoch_E_mb_M>/head \
    --output <run_dir>/kan_parameters.nc
  ```
  After it writes, execute the parameter-map notebook as usual. (If `cargo`
  itself won't build on the host, fall back to quoting the command.)

- **Predictions zarr for hydrograph/metrics.** Produced by the eval phase of
  `ddrs run`. This is a much longer routing run (not a single forward pass), so
  here it's correct to quote the command (`ddrs run --workflow eval`, or
  `src/bin/eval.rs`) and let the user run it rather than launching it yourself.

**Always save plots into `<CKPT_DIR>/plots/`** so artifacts travel with the run. Create the directory if it doesn't exist.

### Step 3 — Read the relevant reference

Each reference file contains:
- The exact ddrs output schema it expects
- A complete, runnable notebook template (imports, data loading, plot calls, save lines)
- Conventions inherited from DDR (warmup periods, NaN handling, metric clipping)

Read the reference before writing the notebook. Don't invent column names — the schemas are documented there.

### Step 4 — Generate the notebook

Write `<CKPT_DIR>/plots/<plot_name>.ipynb`. Suggested names:
- `hydrograph_<gage_id>_<year>.ipynb`
- `parameter_map_<variable>_<region>.ipynb`
- `metrics_distribution.ipynb`

Each notebook ends by saving PNGs to the same `<CKPT_DIR>/plots/` directory. Include a markdown cell at the top documenting: which checkpoint, which inputs, what region/gauge/year was selected, the date generated.

**After writing the notebook, report the absolute path back to the user.** Format:

```
notebook → <absolute path to .ipynb>
plots will save to → <absolute path to CKPT_DIR/plots/>
```

This is non-optional. The user needs to know where artifacts land before deciding whether to execute.

### Step 5 — Offer to run it

After writing, offer to execute the notebook from `ddrs-py`'s venv:

```bash
cd ./ddrs-py && uv run jupyter nbconvert --to notebook --execute \
    <full_notebook_path> --output <basename> --output-dir <PLOT_DIR>
```

Only run if the user agrees — execution can be slow (zarr reads, MERIT shapefile join) and the user may want to tweak the notebook first. If the venv hasn't been set up yet, run `cd ./ddrs-py && uv sync --extra plots` first.

**After execution completes, list every PNG written** with absolute paths so the user can open them directly. Use `ls <CKPT_DIR>/plots/*.png` and quote each path verbatim — don't summarize as "plots are in `<dir>`". Format:

```
saved plots:
  <CKPT_DIR>/plots/hydrograph_<gage>_<year>.png
  <CKPT_DIR>/plots/<other>.png
```

## Defaults and conventions

These match DDR's `plots.py` and `evaluate.ipynb` so notebooks look familiar:

- **Warmup**: drop the first 3 timesteps from hydrographs (DDR's `plot_time_series` default).
- **Metric clipping**: NSE/KGE clipped to `[-1, 1]` before plotting (matches `evaluate.ipynb`).
- **Basemap**: CartoDB.Positron, alpha 0.6, attribution off (matches `param_plot`).
- **CONUS bounds** for full-CONUS maps: `xlim=(-125, -66)`, `ylim=(24, 53)`.
- **Colormaps**: `plasma_r` for Manning's n (high n = rough = red), `viridis` for p/q, `Blues` for depth/width, `bamako` or `plasma` for NSE.
- **Backend**: `matplotlib.use("Agg")` only if running headlessly; in a notebook, leave default.
- **Save kwargs**: `dpi=300, bbox_inches="tight", facecolor="white"` for publication-quality PNGs.

## When NOT to use this skill

- The user is plotting DDR-Python output directly — point them at DDR's own notebooks (this skill is for ddrs).
- The user wants `examples/benchmark_hydrograph.rs` (the 10-reach synthetic chain) — that's a Rust example, not a trained model.
- The user is debugging gradient parity against DDR — use `examples/compare_ddr_sandbox.rs`, not plots.

## Files in this skill

- `SKILL.md` — this file
- `references/hydrograph.md` — single-gauge predictions vs. observations
- `references/parameter_map.md` — learned KAN parameters over MERIT polygons
- `references/metrics.md` — NSE/KGE/bias distributions across gauges
- `scripts/load_ddrs_predictions.py` — **always use this** to open predictions/baseline zarrs. Handles two pitfalls every notebook hits otherwise:
  1. ddrs writes zarr v3 with `_ARRAY_DIMENSIONS` but no `dimension_names`, so `xr.open_zarr` raises `KeyError`.
  2. `gage_ids` is stored as `(G, W) uint8` (W = longest ID, min 8; widened 2026-06-12 so global `Provider__GageId` names are no longer truncated), not 1D bytes/string — naïve `.decode()` won't work.

  Both reference templates already import it. Don't reinvent the loading code.
- `evals/evals.json` — test prompts
