"""Cross-run NSE/KGE comparison plots for the 2026-07-16 AORC2F + LSTM wave 1/2
campaign (docs/2026-07-16-aorc2f-wave1-findings.md,
docs/2026-07-16-wave2-cross-wave-findings.md).

Run from ddrs-py's venv:
    cd ddrs-py && uv run python ../scripts/wave_comparison_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".claude/skills/ddrs-eval-plots/scripts"))
from load_ddrs_predictions import load_predictions_zarr  # noqa: E402

from ddr.validation import Metrics, plot_box_fig, plot_cdf  # noqa: E402

OUT_DIR = REPO / "output" / "2026-07-16-wave-comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Run-ID-to-experiment mapping: .ddrs/README.md (all 4 runs and their
# baseline cache keys were consolidated from separate parallel-launch
# workspaces into the main .ddrs/ workspace after the campaign finished).
RUNS = [
    {
        "label": "AORC2F distributed",
        "pred_zarr": REPO / ".ddrs/runs/2026-07-16T02-22-14Z-train-and-test/eval/predictions.zarr",
        "baseline_dir": REPO / ".ddrs/baselines/86b907e4b17de998",
    },
    {
        "label": "AORC2F lumped",
        "pred_zarr": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/eval/predictions.zarr",
        "baseline_dir": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/baseline",
    },
    {
        "label": "daily-lstm",
        "pred_zarr": REPO / ".ddrs/runs/2026-07-16T11-31-50Z-train-and-test/eval/predictions.zarr",
        "baseline_dir": REPO / ".ddrs/baselines/7ffd117fc9161734",
    },
    {
        "label": "hourly-lstm",
        "pred_zarr": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/eval/predictions.zarr",
        "baseline_dir": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/baseline",
    },
]


def load_baseline_f32(bdir: Path):
    man = json.load(open(bdir / "manifest.json"))
    g, t = man["n_gauges"], man["n_days"]
    pred = np.fromfile(bdir / "predictions.f32", dtype="<f4").reshape(g, t)
    obs = np.fromfile(bdir / "observations.f32", dtype="<f4").reshape(g, t)
    return pred, obs


results = []
labels = []
for run in RUNS:
    ds = load_predictions_zarr(run["pred_zarr"])
    results.append(Metrics(pred=ds.predictions.values, target=ds.observations.values))
    labels.append(run["label"])

for run in RUNS:
    pred, obs = load_baseline_f32(run["baseline_dir"])
    results.append(Metrics(pred=pred, target=obs))
    labels.append(f"{run['label']} baseline")

# --- Box plot (NSE, KGE) --------------------------------------------------
keys = ["nse", "kge"]
xlabel = ["NSE", "KGE"]
data_box = []
for k in keys:
    row = []
    for r in results:
        v = np.clip(dict(r)[k], -1, 1)
        v = v[~np.isnan(v)]
        row.append(v)
    data_box.append(row)

fig = plot_box_fig(
    data=data_box, xlabel_list=xlabel, legend_labels=labels,
    sharey=False, figsize=(20, 9),
)
fig.patch.set_facecolor("white")
fig.suptitle("NSE/KGE across 4 trained arms + their summed-Q' baselines (2026-07-16)", fontsize=18)
fig.savefig(OUT_DIR / "wave_comparison_boxplot.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT_DIR / 'wave_comparison_boxplot.png'}")

# --- CDF (NSE) -------------------------------------------------------------
fig, ax = plot_cdf(
    data_list=[np.clip(dict(r)["nse"], 0, None) for r in results],
    title="NSE CDF across 4 trained arms + summed-Q' baselines (2026-07-16)",
    legend_labels=labels, figsize=(11, 7),
    xlabel="NSE", ylabel="Cumulative frequency", reference_line=None,
)
fig.savefig(OUT_DIR / "wave_comparison_nse_cdf.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT_DIR / 'wave_comparison_nse_cdf.png'}")

# --- CDF (KGE) --------------------------------------------------------------
fig, ax = plot_cdf(
    data_list=[np.clip(dict(r)["kge"], -1, 1) for r in results],
    title="KGE CDF across 4 trained arms + summed-Q' baselines (2026-07-16)",
    legend_labels=labels, figsize=(11, 7),
    xlabel="KGE", ylabel="Cumulative frequency", reference_line=None,
)
fig.savefig(OUT_DIR / "wave_comparison_kge_cdf.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT_DIR / 'wave_comparison_kge_cdf.png'}")

# --- Summary table -----------------------------------------------------
print()
print(f"{'label':<28}{'median NSE':>12}{'median KGE':>12}")
for lbl, r in zip(labels, results):
    d = dict(r)
    nse = np.nanmedian(d["nse"])
    kge = np.nanmedian(d["kge"])
    print(f"{lbl:<28}{nse:>12.4f}{kge:>12.4f}")
