"""Generates and executes per-run parameter_map + metrics_distribution
notebooks for the 2026-07-16 AORC2F/LSTM Q' campaign (4 runs), following
the ddrs-eval-plots skill conventions.

Run from ddrs-py's venv:
    cd ddrs-py && uv run python ../scripts/generate_run_notebooks.py
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
SHAPEFILE = REPO.parent / "ddr/data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
ATTRS_NC = REPO.parent / "ddr/data/merit_global_attributes_v2.nc"

RUNS = [
    {
        "label": "AORC2F distributed",
        "run_dir": REPO / ".ddrs/runs/2026-07-16T02-22-14Z-train-and-test",
        "baseline_dir": REPO / ".ddrs/baselines/86b907e4b17de998",
    },
    {
        "label": "AORC2F lumped",
        "run_dir": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test",
        "baseline_dir": REPO / ".ddrs/runs/2026-07-16T02-23-20Z-train-and-test/baseline",
    },
    {
        "label": "daily-lstm",
        "run_dir": REPO / ".ddrs/runs/2026-07-16T11-31-50Z-train-and-test",
        "baseline_dir": REPO / ".ddrs/baselines/7ffd117fc9161734",
    },
    {
        "label": "hourly-lstm",
        "run_dir": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test",
        "baseline_dir": REPO / ".ddrs/runs/2026-07-16T11-31-52Z-train-and-test/baseline",
    },
]

SKILL_SCRIPTS = REPO / ".claude/skills/ddrs-eval-plots/scripts"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(src: str):
    return nbf.v4.new_code_cell(src)


def build_parameter_map_nb(run) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# Parameter map — {run['label']}\n\n"
           f"Checkpoint: `{run['run_dir']}/checkpoints/epoch_5_mb_35`\n\n"
           f"Params: `{run['run_dir']}/kan_parameters.nc`\n\n"
           f"Generated: {datetime.utcnow().strftime('%Y-%m-%d')} (auto, ddrs-eval-plots skill)"),
        code(f"""
from pathlib import Path
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

PARAMS_NC = Path("{run['run_dir']}/kan_parameters.nc")
RUN_DIR = Path("{run['run_dir']}")
SHAPEFILE = Path("{SHAPEFILE}")
ATTRS_NC = Path("{ATTRS_NC}")
LABEL = "{run['label']}"
BBOX = (-125, 24, -66, 53)

PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

ds = xr.open_dataset(PARAMS_NC)
gdf_full = gpd.read_file(SHAPEFILE).set_index("COMID")
print(f"{{len(ds.COMID):,}} params, {{len(gdf_full):,}} shapefile reaches")
"""),
        code("""
PLOT_CONFIGS = {
    "n":         {"title": "Manning's Roughness", "unit": "m$^{-1/3}$ s", "cmap": "plasma_r", "vmax": 0.2, "range": (0.015, 0.25)},
    "q_spatial": {"title": "Width-Depth Exponent (q)", "unit": "–", "cmap": "viridis", "vmax": None, "range": (0.0, 1.0)},
    "p_spatial": {"title": "Width Coefficient (p)",   "unit": "–", "cmap": "viridis", "vmax": None, "range": (1.0, 200.0)},
}

def make_map(var):
    cfg = PLOT_CONFIGS[var]
    gdf = gdf_full.copy()
    shared = np.intersect1d(gdf.index.values, ds.COMID.values)
    gdf.loc[shared, var] = ds.sel(COMID=shared)[var].values
    gdf = gdf.set_crs(epsg=4326)
    xmin, ymin, xmax, ymax = BBOX
    gdf = gdf.cx[xmin:xmax, ymin:ymax]
    gdf_clean = gdf.dropna(subset=[var]).sort_values(var, ascending=True)

    vmin = float(np.nanmin(gdf_clean[var]))
    vmax = cfg["vmax"] if cfg["vmax"] is not None else float(np.nanmax(gdf_clean[var]))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    gdf_clean.plot(ax=ax, column=var, cmap=cfg["cmap"], linewidth=0.3, vmin=vmin, vmax=vmax, zorder=1)
    try:
        cx.add_basemap(ax, crs=gdf_clean.crs, source=cx.providers.CartoDB.Positron, alpha=0.6, zorder=0, attribution=False)
    except Exception as e:
        print(f"basemap skipped: {e}")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{cfg['title']} — {LABEL} (CONUS)", fontsize=13)
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cfg["cmap"]); sm.set_array([]); sm.set_clim(vmin, vmax)
    fig.colorbar(sm, cax=cax).set_label(f"{var} ({cfg['unit']})")
    out = PLOT_DIR / f"parameter_map_{var}_conus.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
    return cfg

for var in ["n", "q_spatial", "p_spatial"]:
    make_map(var)
"""),
        md("## Distribution histograms (full CONUS population, x-axis anchored to declared parameter_ranges)"),
        code("""
for var in ["n", "q_spatial", "p_spatial"]:
    cfg = PLOT_CONFIGS[var]
    v_all = ds[var].values
    v_finite = v_all[np.isfinite(v_all)]
    vmin_hist, vmax_hist = cfg["range"]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.hist(v_finite, bins=80, range=(vmin_hist, vmax_hist), color="#6c2178", edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.nanmedian(v_finite)), color="black", linestyle="--", linewidth=1.5, label=f"median = {float(np.nanmedian(v_finite)):.4f}")
    ax.axvline(float(np.nanmean(v_finite)), color="#c63", linestyle=":", linewidth=1.5, label=f"mean = {float(np.nanmean(v_finite)):.4f}")
    ax.set_xlabel(f"{var} ({cfg['unit']})")
    ax.set_ylabel(f"reach count (total = {len(v_finite):,})")
    ax.set_title(f"Distribution of learned {var} — {LABEL}")
    ax.set_xlim(vmin_hist, vmax_hist)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    out = PLOT_DIR / f"parameter_hist_{var}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
"""),
        md("## Parameter vs log10(drainage area) hexbin (sanity check against the KAN's own input)"),
        code("""
attrs = xr.open_dataset(ATTRS_NC)
shared_a = np.intersect1d(attrs.COMID.values, ds.COMID.values)
print(f"joined {len(shared_a):,} COMIDs")
attrs_s = attrs.sel(COMID=shared_a)
ds_s = ds.sel(COMID=shared_a)
log_da = attrs_s["log10_uparea"].values

for var in ["n", "q_spatial", "p_spatial"]:
    cfg = PLOT_CONFIGS[var]
    vmin_hist, vmax_hist = cfg["range"]
    y_vals = ds_s[var].values
    mask = np.isfinite(log_da) & np.isfinite(y_vals)
    lx, ly = log_da[mask], y_vals[mask]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    hb = ax.hexbin(lx, ly, gridsize=80, cmap="viridis", mincnt=1, extent=(lx.min(), lx.max(), vmin_hist, vmax_hist))
    fig.colorbar(hb, ax=ax, label="reach count per hex")
    ax.set_xlabel(r"$\\log_{10}$(drainage area, km$^2$)")
    ax.set_ylabel(f"learned {var} ({cfg['unit']})")
    ax.set_title(f"{var} vs drainage area — {LABEL} ({len(ly):,} reaches)")
    ax.set_ylim(vmin_hist, vmax_hist)
    ax.grid(alpha=0.3)
    bin_edges = np.linspace(lx.min(), lx.max(), 21)
    bin_idx = np.digitize(lx, bin_edges) - 1
    med = np.array([np.nanmedian(ly[bin_idx == b]) if np.any(bin_idx == b) else np.nan for b in range(len(bin_edges) - 1)])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(bin_centers, med, color="#ff4500", lw=2.0, label=f"median {var} per bin")
    ax.legend(loc="upper right", frameon=True)
    out = PLOT_DIR / f"parameter_scatter_{var}_vs_log10_uparea.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
"""),
    ]
    return nb


def build_metrics_nb(run) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# Metrics distribution — {run['label']}\n\n"
           f"Run: `{run['run_dir']}`\n\n"
           f"Baseline: `{run['baseline_dir']}`\n\n"
           f"Generated: {datetime.utcnow().strftime('%Y-%m-%d')} (auto, ddrs-eval-plots skill)"),
        code(f"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "{SKILL_SCRIPTS}")
from load_ddrs_predictions import load_predictions_zarr
from ddr.validation import Metrics, plot_box_fig, plot_cdf

RUN_DIR = Path("{run['run_dir']}")
BASELINE_DIR = Path("{run['baseline_dir']}")
LABEL = "{run['label']}"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

def load_baseline_f32(bdir):
    man = json.load(open(bdir / "manifest.json"))
    g, t = man["n_gauges"], man["n_days"]
    pred = np.fromfile(bdir / "predictions.f32", dtype="<f4").reshape(g, t)
    obs = np.fromfile(bdir / "observations.f32", dtype="<f4").reshape(g, t)
    return pred, obs

ds = load_predictions_zarr(RUN_DIR / "eval/predictions.zarr")
result_run = Metrics(pred=ds.predictions.values, target=ds.observations.values)
pred_b, obs_b = load_baseline_f32(BASELINE_DIR)
result_baseline = Metrics(pred=pred_b, target=obs_b)

results = [result_baseline, result_run]
labels = [f"{{LABEL}} baseline", LABEL]
"""),
        code("""
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

fig = plot_box_fig(data=data_box, xlabel_list=xlabel, legend_labels=labels, sharey=False, figsize=(10, 6))
fig.patch.set_facecolor("white")
fig.suptitle(f"NSE/KGE — {LABEL} vs own baseline", fontsize=14)
out = PLOT_DIR / "metrics_boxplot.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
"""),
        code("""
fig, ax = plot_cdf(
    data_list=[np.clip(dict(r)["nse"], 0, None) for r in results],
    title=f"NSE CDF — {LABEL} vs own baseline",
    legend_labels=labels, figsize=(9, 6), xlabel="NSE", ylabel="Cumulative frequency", reference_line=None,
)
out = PLOT_DIR / "metrics_nse_cdf.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")

fig, ax = plot_cdf(
    data_list=[np.clip(dict(r)["kge"], -1, 1) for r in results],
    title=f"KGE CDF — {LABEL} vs own baseline",
    legend_labels=labels, figsize=(9, 6), xlabel="KGE", ylabel="Cumulative frequency", reference_line=None,
)
out = PLOT_DIR / "metrics_kge_cdf.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"saved {out}")
"""),
        code("""
print(f"{'series':<20}{'median NSE':>12}{'median KGE':>12}")
for lbl, r in zip(labels, results):
    d = dict(r)
    print(f"{lbl:<20}{np.nanmedian(d['nse']):>12.4f}{np.nanmedian(d['kge']):>12.4f}")
"""),
    ]
    return nb


def main():
    for run in RUNS:
        run_dir = run["run_dir"]
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        pm_nb = build_parameter_map_nb(run)
        pm_path = plots_dir / "parameter_map_conus.ipynb"
        nbf.write(pm_nb, pm_path)
        print(f"wrote {pm_path}")

        met_nb = build_metrics_nb(run)
        met_path = plots_dir / "metrics_distribution.ipynb"
        nbf.write(met_nb, met_path)
        print(f"wrote {met_path}")


if __name__ == "__main__":
    main()
