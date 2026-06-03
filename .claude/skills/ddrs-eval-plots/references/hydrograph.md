# Reference: hydrograph plot

Single-gauge predicted-vs-observed streamflow over a user-selected year.
Direct port of DDR's `plot_time_series` (`~/projects/ddr/src/ddr/validation/plots.py:18-92`) adapted to ddrs's predictions zarr schema.

## Inputs

### Predictions zarr (from `cargo run --bin eval`)

Required schema (set by `write_predictions_zarr`):
- group `predictions` — `(G, T)` float, m³/s
- group `observations` — `(G, T)` float, m³/s, NaN where unobserved
- group `gage_ids` — `(G,)` string, USGS STAIDs (e.g., `"01013500"`)
- group `time` — `(T,)` datetime64

### User-supplied selection

- **gauge id** — one entry from `gage_ids`. If user passes a name like "Allagash River", join through `data/camels_670.csv` (column `STAID`, `STANAME`).
- **year** — water year or calendar year (clarify if ambiguous). Default: full time range in the zarr.
- **optional comparison series** — e.g., DDR-Python predictions at the same gauge, or summed Q' baseline. Drawn as extra lines.

## Notebook template

```python
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Bundled loader handles ddrs's zarr v3 layout (missing dimension_names) and
# the (G, 8) uint8 gage_ids encoding. Without this, `xr.open_zarr` raises
# KeyError and `.sel(gage_ids="01013500")` silently misses.
SKILL_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" if "__file__" in dir() else Path("/home/tbindas/projects/ddrs/.claude/worktrees/plot-predictions-notebook/.claude/skills/ddrs-eval-plots/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from load_ddrs_predictions import load_predictions_zarr

# --- USER INPUTS ---------------------------------------------------------
PRED_ZARR = Path("/home/tbindas/projects/ddrs/output/predictions_latest.zarr")
CKPT_DIR  = Path("/home/tbindas/projects/ddrs/output/saved_models_1")
GAGE_ID   = "01013500"          # USGS STAID
YEAR      = 2000                 # calendar year; for water year, set WATER_YEAR=True below
WATER_YEAR = False
WARMUP    = 3                    # drop first N timesteps (DDR default)
# -------------------------------------------------------------------------

PLOT_DIR = CKPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

ds = load_predictions_zarr(PRED_ZARR)
g = ds.sel(gage_ids=GAGE_ID)

# Time slice
if WATER_YEAR:
    start, end = f"{YEAR-1}-10-01", f"{YEAR}-09-30"
else:
    start, end = f"{YEAR}-01-01", f"{YEAR}-12-31"
g = g.sel(time=slice(start, end))

pred = g.predictions.values[WARMUP:]
obs  = g.observations.values[WARMUP:]
time = pd.to_datetime(g.time.values[WARMUP:])

# Metrics on the slice
mask = np.isfinite(obs) & np.isfinite(pred)
if mask.sum() > 1:
    obs_m, pred_m = obs[mask], pred[mask]
    nse = 1 - np.sum((pred_m - obs_m) ** 2) / np.sum((obs_m - obs_m.mean()) ** 2)
    bias = (pred_m - obs_m).mean()
    obs_mass = float(np.nansum(obs))
    pred_mass = float(np.nansum(pred))
else:
    nse = bias = obs_mass = pred_mass = float("nan")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, obs,  label=f"USGS [ΣQ={obs_mass:.1f}]")
ax.plot(time, pred, label=f"ddrs [ΣQ={pred_mass:.1f}, NSE={nse:.4f}]")
ax.set_xlabel("Date")
ax.set_ylabel(r"Discharge (m$^3$/s)")
ax.set_title(f"Gauge {GAGE_ID} — {'water year' if WATER_YEAR else 'calendar year'} {YEAR}")
ax.legend()
fig.autofmt_xdate(rotation=30)

out = PLOT_DIR / f"hydrograph_{GAGE_ID}_{YEAR}.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
```

## Notes

- **Why warmup=3?** Matches DDR's default. The first few timesteps of MC routing have not yet propagated through the network, so the predicted hydrograph is artificially flat at hotstart.
- **NSE inside the slice, not over full zarr.** When the user asks for a single year, recomputing NSE for that window is more informative than reusing the whole-period NSE.
- **For multiple gauges**: loop over `GAGE_ID` and save one PNG per gauge. Don't put multiple gauges on one axis — discharge magnitudes vary by orders of magnitude across basins.
- **Adding a comparison line (e.g., summed Q' baseline)**: open the baseline zarr, slice the same time window, plot with `linestyle="--"` and add to the legend.
- **`gage_ids` dtype gotcha — handled by `load_predictions_zarr`.** ddrs writes `gage_ids` as `(G, 8) uint8` with `_dtype_hint: |S8`, NOT as a 1D bytes/string array. Combined with the missing `dimension_names` metadata in the zarr v3 store, `xr.open_zarr` plus a naïve decode fails in two different places. The bundled `scripts/load_ddrs_predictions.py` handles both: opens with the raw `zarr` library, assembles an `xarray.Dataset`, decodes the 2D uint8 layout into a clean 1D string axis.
- **`obs <= 0` is a sentinel for "missing".** DDR's convention is to treat non-positive observations as unobserved before computing NSE/bias. If your zarr was produced from a USGS source that uses this convention, mask `obs[obs <= 0] = np.nan` before metrics.
