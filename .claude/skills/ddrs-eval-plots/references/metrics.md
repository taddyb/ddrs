# Reference: metrics distribution plots

Cross-gauge distributions of model performance — NSE, KGE, bias, RMSE, FHV, FLV — rendered as box plots, CDF, and a basemap-backed gauge scatter. Direct port of `~/projects/ddr/examples/eval/evaluate.ipynb`, adapted to ddrs's predictions zarr schema.

## Inputs

### Predictions zarr (from `cargo run --bin eval`)

Same schema as the hydrograph reference: `predictions (G,T)`, `observations (G,T)`, `gage_ids (G,)`, `time (T,)`.

### Optional baseline zarr

Summed Q' baseline (no routing) — DDR's standard "is routing pulling its weight" comparison. Build with DDR's `scripts/summed_q_prime.py`. If provided, plots compare ddrs vs. baseline.

### Optional gauges CSV

Required only for the drainage-area boxplot and the gauge map. Path lives in the training YAML under `data_sources.gages` — e.g. `~/projects/ddr/data/camels_670.csv`. Columns expected: `STAID` (str, zero-padded to 8), `LAT_GAGE`, `LNG_GAGE`, `DRAIN_SQKM`.

## Metric helper

DDR's `ddr.validation.Metrics` (`~/projects/ddr/src/ddr/validation/metrics.py`) is a pydantic model that computes all metrics in one shot:

```python
from ddr.validation import Metrics
m = Metrics(pred=ds.predictions.values, target=ds.observations.values)
# fields: bias, mae, rmse, ub_rmse, fdc_rmse, corr, corr_spearman, r2,
#         nse, flv, fhv, pbias, pbias_mid, kge   (all shape (G,))
```

Reuse it — don't roll metrics by hand. The notebook runs in **ddrs-py's** `uv` venv (after `cd ./ddrs-py && uv sync --extra plots`), which installs DDR as a local-path dependency so `from ddr.validation import ...` works without leaving the ddrs project.

## Notebook template

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import sys
import yaml
from ddr.validation import Metrics, plot_box_fig, plot_cdf, plot_drainage_area_boxplots, plot_gauge_map

# Bundled loader handles ddrs's zarr v3 layout + (G, W) uint8 gage_ids.
SKILL_SCRIPTS = Path("/home/tbindas/projects/ddrs/.claude/worktrees/plot-predictions-notebook/.claude/skills/ddrs-eval-plots/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from load_ddrs_predictions import load_predictions_zarr, load_baseline_zarr

# --- USER INPUTS ---------------------------------------------------------
PRED_ZARR     = Path("/home/tbindas/projects/ddrs/output/predictions_latest.zarr")
BASELINE_ZARR = None  # Path("...summed_q_prime.zarr") if comparing; else None
CKPT_DIR      = Path("/home/tbindas/projects/ddrs/output/saved_models_1")
TRAINING_YAML = Path("/home/tbindas/projects/ddrs/config/merit_training.yaml")
RUN_LABEL     = "ddrs latest"
# -------------------------------------------------------------------------

PLOT_DIR = CKPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Resolve the gauges CSV from the training config. It lives at
# `data_sources.gages` and is whatever the model was actually trained on
# (e.g., gages_3000.csv, camels_670.csv) — don't hardcode the filename.
GAUGES_CSV = None
if TRAINING_YAML.exists():
    cfg = yaml.safe_load(TRAINING_YAML.read_text())
    gpath = cfg.get("data_sources", {}).get("gages")
    if gpath:
        GAUGES_CSV = Path(gpath).expanduser()

ds = load_predictions_zarr(PRED_ZARR)

results = [Metrics(pred=ds.predictions.values, target=ds.observations.values)]
labels  = [RUN_LABEL]

if BASELINE_ZARR is not None:
    dsb = load_baseline_zarr(BASELINE_ZARR)
    # Intersect axes — neither zarr is guaranteed a strict subset of the
    # other. Past bugs: time axes drifted by 1-2 days; ddrs gauge set is a
    # subset of baseline but the other direction also happens.
    shared_gids = np.intersect1d(ds.gage_ids.values, dsb.gage_ids.values)
    shared_time = np.intersect1d(ds.time.values, dsb.time.values)
    ds  = ds.sel(gage_ids=shared_gids, time=shared_time)
    dsb = dsb.sel(gage_ids=shared_gids, time=shared_time)
    results = [Metrics(pred=ds.predictions.values, target=ds.observations.values)]
    results.insert(0, Metrics(pred=dsb.predictions.values, target=dsb.observations.values))
    labels.insert(0, r"$\sum$ Q' baseline")

# --- Box plot of 6 metrics ----------------------------------------------
keys   = ["bias", "rmse", "fhv", "flv", "nse", "kge"]
xlabel = [r"Bias (m$^3$/s)", "RMSE", "FHV", "FLV", "NSE", "KGE"]
data_box = []
for k in keys:
    row = []
    for r in results:
        v = dict(r)[k]
        if k in ("nse", "kge"):
            v = np.clip(v, -1, 1)
        v = v[~np.isnan(v)]
        row.append(v)
    data_box.append(row)

fig = plot_box_fig(data=data_box, xlabel_list=xlabel, legend_labels=labels,
                   sharey=False, figsize=(20, 8))
fig.patch.set_facecolor("white")
fig.suptitle("Metrics across evaluation gauges", fontsize=24)
fig.savefig(PLOT_DIR / "metrics_boxplot.png", dpi=200, bbox_inches="tight")

# --- CDF of NSE ----------------------------------------------------------
fig, ax = plot_cdf(
    data_list=[np.clip(dict(r)["nse"], 0, None) for r in results],
    title="NSE Cumulative Distribution",
    legend_labels=labels, figsize=(10, 6),
    xlabel="NSE", ylabel="Cumulative frequency", reference_line=None,
)
fig.savefig(PLOT_DIR / "metrics_nse_cdf.png", dpi=200, bbox_inches="tight")

# --- Drainage-area boxplots + gauge map (require gauges CSV) ------------
if GAUGES_CSV is not None and GAUGES_CSV.exists():
    g = pd.read_csv(GAUGES_CSV)
    g["STAID"] = g["STAID"].astype(str).str.zfill(8)
    # Only join gauges that are in BOTH the predictions zarr and the CSV.
    # Missing rows would otherwise produce NaN-padded metric columns.
    common_gids = np.intersect1d(g["STAID"].values, ds.gage_ids.values)
    g = g.set_index("STAID").loc[common_gids].reset_index()

    # CRITICAL: per-gauge metric arrays follow `ds.gage_ids` row order
    # (which is whatever order .sel() produced — sorted, if .sel was fed
    # intersect1d output). The gauges DataFrame `g` is now in `common_gids`
    # order, which may be a STRICT SUBSET of `ds.gage_ids`. Joining
    # `results[i].nse` directly would mis-row (length mismatch or worse,
    # silently wrong correspondence). Build a positional reindex.
    gid_to_idx = {gid: i for i, gid in enumerate(ds.gage_ids.values)}
    positions = np.array([gid_to_idx[gid] for gid in common_gids])

    for i, lbl in enumerate(labels):
        col = f"{lbl}_NSE".replace(" ", "_")
        g[col] = np.clip(dict(results[i])["nse"][positions], 0, 1)
    metric_cols = [f"{l}_NSE".replace(" ", "_") for l in labels]

    DRAINAGE_BINS = np.array([0, 1000, 5000, 10000, 30000, 50000])
    fig = plot_drainage_area_boxplots(
        gages=g, metrics=metric_cols, model_names=labels,
        bins=DRAINAGE_BINS,
        path=None,  # save after we annotate per-panel medians
    )
    # Annotate each bin with its median NSE in the top-right corner of the
    # panel. DDR's plot_drainage_area_boxplots draws all bins on a SINGLE
    # Axes (plots.py:465), so we place text in data coordinates at each
    # bin's right edge.
    n_bins  = len(DRAINAGE_BINS) - 1
    bin_w   = 5      # matches plots.py:471 (`bin_width`)
    y_upper = 1.0    # matches default `y_limits=(0.0, 1.0)`
    ax = fig.axes[0]
    bin_assignment = pd.cut(g["DRAIN_SQKM"], DRAINAGE_BINS, labels=False)
    for bi in range(n_bins):
        in_bin = bin_assignment == bi
        if not in_bin.any():
            continue
        if len(metric_cols) == 1:
            med  = float(np.nanmedian(g.loc[in_bin, metric_cols[0]].values))
            text = f"median = {med:.3f}"
        else:
            text = "\n".join(
                f"{lbl}: {float(np.nanmedian(g.loc[in_bin, col].values)):.3f}"
                for col, lbl in zip(metric_cols, labels)
            )
        ax.text(
            (bi + 1) * bin_w - 0.25, y_upper - 0.02, text,
            ha="right", va="top", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#666", alpha=0.9),
        )
    fig.savefig(PLOT_DIR / "metrics_drainage_area.png",
                dpi=200, bbox_inches="tight")

    fig = plot_gauge_map(gages=g, metric_column=metric_cols[-1],
                         title=RUN_LABEL, colormap="plasma",
                         figsize=(16, 8), point_size=30,
                         path=PLOT_DIR / "metrics_gauge_map.png")

print(f"saved metrics plots to {PLOT_DIR}")
```

## Notes

- **Why clip NSE to [-1, 1]?** Without clipping, a handful of catastrophic gauges (`NSE < -100`) make the box-plot scale unreadable. DDR's `evaluate.ipynb` does the same (`np.clip(data, -1, 1)` for NSE/KGE only).
- **Why drop NaN per metric (not per gauge)?** Different metrics fail for different reasons (e.g., zero-variance observations break `nse` but not `bias`). Filtering per metric preserves more data than dropping any gauge with any NaN.
- **FHV / FLV** (High-Flow Volume / Low-Flow Volume biases) — percent errors in the top/bottom flow regimes; closer to zero is better. From `ddr.validation.Metrics`.
- **The CDF clips NSE to `[0, ∞)`** to emphasize where useful skill begins. The boxplot uses `[-1, 1]`. This asymmetry is inherited from DDR.
- **Axis intersection, not subset.** Neither zarr is guaranteed a strict subset of the other — ddrs's predictions zarr and DDR's summed-Q' baseline have been observed to differ by 1-2 days at the time-axis edges, and the gauge sets can be different sizes. Compute the intersection on both axes before metrics, otherwise `.sel(...)` raises `KeyError`. Same rule for the gauges-CSV join — intersect with the predictions zarr `gage_ids` before `.loc[...]`.
- **`gage_ids` storage gotchas — handled by `load_predictions_zarr`.** ddrs writes `gage_ids` as `(G, W) uint8` with `_dtype_hint: |S<W>` (W = longest ID, min 8; stores written before 2026-06-12 truncated global Provider__GageId names to 8 bytes — per-gauge joins on such stores are unreliable), NOT as 1D bytes/string. The zarr v3 store also lacks the `dimension_names` metadata `xr.open_zarr` requires. The bundled helper opens with raw `zarr`, decodes the 2D uint8 layout per-row, casts `time` from `int64` nanoseconds to `datetime64[ns]`, and assembles a clean `xarray.Dataset`. Use it for both the ddrs zarr and the DDR summed-Q' baseline (same schema).
- **`GAUGES_CSV` comes from the training YAML.** The path lives at `data_sources.gages` and tracks whatever the model was actually trained on. Hardcoding `camels_670.csv` or `gages_3000.csv` breaks the moment training switches to a different gauge set.
- **Positional reindex when joining metrics to a smaller gauges DataFrame.** `Metrics(...).nse` is shape `(G_ds,)` keyed positionally by `ds.gage_ids`. If you then intersect with a gauges CSV that lacks some of those gauges, you have `G_csv < G_ds`. Writing `g[col] = m.nse` either errors (length mismatch) or — worse, in some pandas versions — silently broadcasts and writes the wrong gauge's NSE to each row. Always reindex with a `gid_to_idx` lookup before the join. This was iter-2's silent bug.
- **The gauge map uses the last (rightmost) result in `labels`.** If you want to map the baseline instead, swap the index.
- **Drainage-area panel medians are positioned in data coordinates, not axes coordinates.** DDR's `plot_drainage_area_boxplots` draws all bins on a single Axes with bin centers at `bin_positions[i] + bin_width/2` (data x in `[0, 25]` for the default 5 bins × `bin_width=5`). Iterating `fig.axes[:n_bins]` returns ONE axis and puts every annotation on top of itself — use `ax.text(right_edge_x, y_upper - eps, ...)` in data coords instead. If DDR's defaults change (different bin count or `bin_width`), update both numbers — they're encoded in `plots.py:471` and `plots.py:485-491`.
