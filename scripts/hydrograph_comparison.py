"""HUC-grouped hydrograph comparison across the 2026-07-16 AORC2F/LSTM Q'
campaign's 4 runs: one randomly-selected gauge per USGS major-basin region
(first 2 digits of STAID, a HUC2-like grouping), all 4 models + observations
overlaid for the eval-window year with the best observation coverage.

Run from ddrs-py's venv:
    cd ddrs-py && uv run python ../scripts/hydrograph_comparison.py
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".claude/skills/ddrs-eval-plots/scripts"))
from load_ddrs_predictions import load_predictions_zarr  # noqa: E402

GAGES_CSV = REPO.parent / "ddr/references/gage_info/gages_3000.csv"
OUT_DIR = REPO / "output/2026-07-16-wave-comparison/hydrographs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    {"label": "AORC2F distributed", "pred_zarr": REPO / ".ddrs/runs/2026-07-16T02-22-14Z-train-and-test/eval/predictions.zarr", "color": "#1f4fd8"},
    {"label": "AORC2F lumped", "pred_zarr": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/eval/predictions.zarr", "color": "#d81f6a"},
    {"label": "daily-lstm", "pred_zarr": REPO / ".ddrs/runs/2026-07-16T11-31-50Z-train-and-test/eval/predictions.zarr", "color": "#1fb01f"},
    {"label": "hourly-lstm", "pred_zarr": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/eval/predictions.zarr", "color": "#d88a1f"},
]

# --- Region grouping + random gauge selection ------------------------------
rows = list(csv.DictReader(open(GAGES_CSV)))
valid = [r for r in rows if r["DA_VALID"] == "True"]
regions = defaultdict(list)
for r in valid:
    regions[r["STAID"][:2]].append(r["STAID"])

# --- Load all 4 runs' predictions/observations -----------------------------
datasets = []
for run in RUNS:
    ds = load_predictions_zarr(run["pred_zarr"])
    datasets.append(ds)

# Union of gauges present in at least one run's eval gauge subset (the eval
# set is a filtered subset of gages_3000.csv — DA_VALID plus gages_adjacency
# headwater/coverage drops — so not every DA_VALID gauge is actually usable).
present = set()
for ds in datasets:
    present |= set(ds.gage_ids.values.tolist())

random.seed(42)
selected = {}
for region, staids in sorted(regions.items()):
    candidates = [s for s in staids if s in present]
    if not candidates:
        print(f"region {region}: no DA_VALID gauge present in any run's eval set, skipping region")
        continue
    selected[region] = random.choice(candidates)
print(f"selected {len(selected)} gauges (one per region, restricted to gauges present in the eval set):")
for region, staid in selected.items():
    print(f"  region {region}: {staid}")

# --- Plot one hydrograph per selected gauge --------------------------------
for region, staid in selected.items():
    # Find the gauge in each run's dataset; skip runs missing it (shouldn't happen, same gauge set)
    series = []
    obs_ref = None
    for run, ds in zip(RUNS, datasets):
        if staid not in ds.gage_ids.values:
            series.append(None)
            continue
        pred = ds.sel(gage_ids=staid).predictions.values
        obs = ds.sel(gage_ids=staid).observations.values
        time = ds.time.values
        series.append((time, pred))
        if obs_ref is None:
            obs_ref = (time, obs)

    if obs_ref is None or all(s is None for s in series):
        print(f"region {region} ({staid}): no data in any run, skipping")
        continue

    time_obs, obs = obs_ref
    df_obs = pd.Series(obs, index=pd.to_datetime(time_obs))
    # Pick the water-year (Oct-Sep) with the most non-NaN observation coverage.
    water_year = df_obs.index.year + (df_obs.index.month >= 10).astype(int)
    coverage = df_obs.notna().groupby(water_year).sum()
    best_wy = coverage.idxmax()
    start, end = f"{best_wy - 1}-10-01", f"{best_wy}-09-30"

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    obs_window = df_obs.loc[start:end]
    ax.plot(obs_window.index, obs_window.values, color="black", lw=1.2, label="Observed", zorder=5)

    for run, s in zip(RUNS, series):
        if s is None:
            continue
        time, pred = s
        pred_series = pd.Series(pred, index=pd.to_datetime(time)).loc[start:end]
        ax.plot(pred_series.index, pred_series.values, color=run["color"], lw=1.0, alpha=0.85, label=run["label"])

    ax.set_title(f"USGS {staid} (region {region}) — water year {best_wy} (best coverage in eval window)")
    ax.set_ylabel("Q (m³/s)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out = OUT_DIR / f"hydrograph_region{region}_{staid}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
