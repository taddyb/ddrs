#!/usr/bin/env python
"""Figures for the adjoint influence-map study.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/adjoint/plots.py <experiment out dir>
        [--fabric /mnt/ssd1/data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp]

Reads only the study output directory (manifest.json, gauges.csv,
<arm>/gauges/<staid>.nc) and writes PNGs to <out>/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

DEFAULT_FABRIC = "/mnt/ssd1/data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
SEASON = {0: "Oct-Dec", 92: "Jan-Mar", 182: "Apr-Jun", 273: "Jul-Sep"}


def load(out: Path):
    manifest = json.loads((out / "manifest.json").read_text())
    arms = [a["name"] for a in manifest["arms"]]
    gauges = pd.read_csv(out / "gauges.csv", dtype={"staid": str, "upstream_staids": str})
    data: dict[tuple[str, str], xr.Dataset] = {}
    for arm in arms:
        for staid in gauges["staid"]:
            p = out / arm / "gauges" / f"{staid}.nc"
            if p.exists():
                data[(arm, staid)] = xr.open_dataset(p, decode_timedelta=False)
    return manifest, arms, gauges, data


def arm_colors(arms):
    cmap = plt.get_cmap("tab10")
    return {a: cmap(i) for i, a in enumerate(arms)}


def fig_kernel_by_lag(out, arms, gauges, data, colors):
    staids = list(gauges["staid"])
    fig, axes = plt.subplots(len(staids), 2, figsize=(11, 3.4 * len(staids)), squeeze=False, sharex=True)
    for r, staid in enumerate(staids):
        for c, kind in enumerate(["high", "low"]):
            ax = axes[r, c]
            for arm in arms:
                ds = data.get((arm, staid))
                if ds is None or "kernel_hourly" not in ds:
                    continue
                sel = ds["anchor_is_high"].values == (1 if kind == "high" else 0)
                if not sel.any():
                    continue
                k = ds["kernel_hourly"].values[sel].mean(axis=0)
                lag_days = np.arange(k.size) / 24.0
                ax.plot(lag_days, k, color=colors[arm], label=arm, lw=1.4)
            role = gauges.loc[gauges.staid == staid, "role"].iloc[0]
            ax.set_title(f"{staid} ({role}) — {kind}-flow anchors")
            ax.set_ylabel("dQ_g(t0)/dq'  summed over reaches")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlim(0, None)
            if r == len(staids) - 1:
                ax.set_xlabel("lag (days before anchor)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Bias-propagation kernel by lag, per inflow-source arm", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "figures" / "kernel_by_lag.png", dpi=150)
    plt.close(fig)


def fig_kernel_vs_distance(out, arms, gauges, data, colors):
    staids = list(gauges["staid"])
    fig, axes = plt.subplots(len(staids), 2, figsize=(11, 3.6 * len(staids)), squeeze=False)
    for r, staid in enumerate(staids):
        for arm in arms:
            ds = data.get((arm, staid))
            if ds is None or "kernel_mass" not in ds:
                continue
            hi = ds["anchor_is_high"].values == 1
            dist_km = ds["dist_to_gauge_m"].values / 1000.0
            mass = ds["kernel_mass"].values[hi].mean(axis=0) if hi.any() else ds["kernel_mass"].values.mean(axis=0)
            lag = ds["kernel_mean_lag_days"].values[hi].mean(axis=0) if hi.any() else ds["kernel_mean_lag_days"].values.mean(axis=0)
            size = 6 + 40 * np.sqrt(np.clip(ds["q_prime_mean"].values, 0, None) / max(np.nanmax(ds["q_prime_mean"].values), 1e-9))
            axes[r, 0].scatter(dist_km, mass, s=size, color=colors[arm], alpha=0.6, label=arm, edgecolor="none")
            axes[r, 1].scatter(dist_km, lag, s=size, color=colors[arm], alpha=0.6, label=arm, edgecolor="none")
        axes[r, 0].set_title(f"{staid}: kernel mass (high-flow anchors)")
        axes[r, 0].set_ylabel("Σ_lag kernel")
        axes[r, 1].set_title(f"{staid}: kernel mean lag")
        axes[r, 1].set_ylabel("days")
        for ax in axes[r]:
            ax.set_xlabel("along-channel distance to gauge (km)")
    axes[0, 0].legend(fontsize=8, markerscale=1.5)
    fig.suptitle("Per-reach kernel vs distance (marker size ∝ √ mean inflow)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "figures" / "kernel_vs_distance.png", dpi=150)
    plt.close(fig)


def load_polygons(fabric: Path, comids: np.ndarray):
    import geopandas as gpd

    comids = [int(c) for c in comids]
    where = "COMID IN (" + ",".join(str(c) for c in comids) + ")"
    try:
        gdf = gpd.read_file(fabric, where=where, engine="pyogrio")
    except Exception as e:  # noqa: BLE001
        print(f"  pyogrio where= failed ({e}); reading whole fabric")
        gdf = gpd.read_file(fabric)
        gdf = gdf[gdf["COMID"].isin(comids)]
    return gdf


def fig_influence_maps(out, arms, gauges, data, fabric: Path):
    if not fabric.exists():
        print(f"fabric {fabric} not found; skipping influence maps")
        return
    for staid in gauges["staid"]:
        any_ds = next((data[(a, staid)] for a in arms if (a, staid) in data), None)
        if any_ds is None:
            continue
        gdf = load_polygons(fabric, any_ds["COMID"].values)
        if gdf.empty:
            print(f"  no polygons found for {staid}; skipping map")
            continue
        fields = [f for f in ["residual_attr", "volume_sens"] if f in any_ds]
        fig, axes = plt.subplots(len(fields), len(arms), figsize=(3.6 * len(arms), 3.6 * len(fields)), squeeze=False)
        # symmetric residual scale shared across arms
        res_abs = max(
            (np.nanmax(np.abs(data[(a, staid)]["residual_attr"].values)) for a in arms if (a, staid) in data and "residual_attr" in data[(a, staid)]),
            default=1.0,
        )
        for c, arm in enumerate(arms):
            ds = data.get((arm, staid))
            for r, field in enumerate(fields):
                ax = axes[r, c]
                ax.set_axis_off()
                if ds is None or field not in ds:
                    ax.set_title(f"{arm}: n/a")
                    continue
                vals = pd.Series(ds[field].values, index=ds["COMID"].values.astype(int))
                g = gdf.copy()
                g["v"] = g["COMID"].astype(int).map(vals)
                if field == "residual_attr":
                    kw = dict(cmap="RdBu_r", vmin=-res_abs, vmax=res_abs)
                else:
                    kw = dict(cmap="viridis", vmin=0.0, vmax=1.2)
                g.plot(column="v", ax=ax, legend=(c == len(arms) - 1), edgecolor="none", **kw,
                       legend_kwds={"shrink": 0.6, "label": field})
                gauge_comid = int(ds["COMID"].values[ds["is_gauge_reach"].values == 1][0])
                gg = g[g["COMID"].astype(int) == gauge_comid]
                if not gg.empty:
                    pt = gg.geometry.representative_point().iloc[0]
                    ax.plot(pt.x, pt.y, marker="^", color="k", ms=8)
                ax.set_title(f"{arm}\n{field}", fontsize=9)
        fig.suptitle(f"Influence map — gauge {staid} (▲ gauge reach)", y=1.0)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"influence_map_{staid}.png", dpi=150)
        plt.close(fig)


def fig_inherited_vs_local(out, arms, gauges, data, colors):
    downs = gauges[gauges.role == "downstream"]
    for _, row in downs.iterrows():
        down = row["staid"]
        ups = [u for u in str(row["upstream_staids"]).split(";") if u and u != "nan"]
        if not ups:
            continue
        up = ups[0]
        fig, axes = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 3.8), squeeze=False, sharey=True)
        for c, arm in enumerate(arms):
            ax = axes[0, c]
            dd = data.get((arm, down))
            du = data.get((arm, up))
            if dd is None or du is None or "residual_gauge_mean" not in dd or "residual_gauge_mean" not in du:
                ax.set_title(f"{arm}: n/a")
                continue
            up_comid = int(du["COMID"].values[du["is_gauge_reach"].values == 1][0])
            idx = np.where(dd["COMID"].values.astype(int) == up_comid)[0]
            transfer = float(dd["volume_sens"].values[idx[0]]) if idx.size and "volume_sens" in dd else np.nan
            total = dd["residual_gauge_mean"].values
            inherited = du["residual_gauge_mean"].values * transfer
            local = total - inherited
            starts = dd["residual_window_start_day"].values
            labels = [SEASON.get(int(s - starts[0]), str(s)) for s in starts]
            x = np.arange(len(total))
            w = 0.27
            ax.bar(x - w, total, w, color="k", alpha=0.75, label="downstream bias")
            ax.bar(x, inherited, w, color=colors[arm], label=f"inherited from {up} × {transfer:.2f}")
            ax.bar(x + w, local, w, color=colors[arm], alpha=0.4, hatch="//", label="local remainder")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(arm, fontsize=10)
            if c == 0:
                ax.set_ylabel("mean(pred − obs), m³/s")
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"Downstream {down} bias: inherited from upstream gauge {up} vs local (WY windows)", y=1.02)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"inherited_vs_local_{up}_{down}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def fig_volume_sens(out, arms, gauges, data, colors):
    staids = list(gauges["staid"])
    fig, axes = plt.subplots(len(staids), 2, figsize=(11, 3.4 * len(staids)), squeeze=False)
    for r, staid in enumerate(staids):
        for arm in arms:
            ds = data.get((arm, staid))
            if ds is None or "volume_sens" not in ds:
                continue
            v = ds["volume_sens"].values
            axes[r, 0].hist(v, bins=40, histtype="step", color=colors[arm], label=f"{arm} (median {np.nanmedian(v):.3f})", lw=1.4)
            axes[r, 1].scatter(ds["dist_to_gauge_m"].values / 1000.0, v, s=8, color=colors[arm], alpha=0.6, label=arm, edgecolor="none")
        for ax in axes[r]:
            ax.axhline(1.0, color="k", lw=0.6, ls="--") if ax is axes[r, 1] else ax.axvline(1.0, color="k", lw=0.6, ls="--")
        axes[r, 0].set_title(f"{staid}: volume sensitivity (expect ≈1)")
        axes[r, 0].set_xlabel("d[ΣQ_g]/dq' (time-mean)")
        axes[r, 1].set_title(f"{staid}: volume sensitivity vs distance")
        axes[r, 1].set_xlabel("along-channel distance to gauge (km)")
        axes[r, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "volume_sens.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--fabric", type=Path, default=Path(DEFAULT_FABRIC))
    args = ap.parse_args()
    out = args.out
    (out / "figures").mkdir(exist_ok=True)
    manifest, arms, gauges, data = load(out)
    arms = [a for a in arms if any(k[0] == a for k in data)]
    if not data:
        raise SystemExit("no gauge netCDFs found under the output directory")
    colors = arm_colors(arms)
    print(f"arms: {arms}; gauges: {list(gauges.staid)}; datasets: {len(data)}")
    fig_kernel_by_lag(out, arms, gauges, data, colors)
    fig_kernel_vs_distance(out, arms, gauges, data, colors)
    fig_volume_sens(out, arms, gauges, data, colors)
    fig_inherited_vs_local(out, arms, gauges, data, colors)
    fig_influence_maps(out, arms, gauges, data, args.fabric)
    for p in sorted((out / "figures").glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
