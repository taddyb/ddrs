#!/usr/bin/env python
"""Figures for the adjoint influence-map study.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/adjoint/plots.py <experiment out dir>
        [--fabric /mnt/ssd1/data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp]
        [--maps N]   # influence maps for the N largest gauges (default 2 in population mode)

Reads only the study output directory (manifest.json, gauges.csv,
<arm>/gauges/<staid>.nc) and writes PNGs to <out>/figures/. With ≤ 6 gauges
the per-gauge figures are drawn; with more, population summaries replace them.
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


# ----------------------------------------------------------------------------- io
def load(out: Path):
    manifest = json.loads((out / "manifest.json").read_text())
    arms = [a["name"] for a in manifest["arms"]]
    gauges = pd.read_csv(out / "gauges.csv", dtype={"staid": str, "upstream_staids": str}, keep_default_na=False)
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


def upstream_list(row) -> list[str]:
    return [u for u in str(row["upstream_staids"]).split(";") if u]


def inherited_decomposition(arms, gauges, data):
    """Per (arm, downstream gauge): total, inherited (summed over upstream gauges), local — per window."""
    rows = []
    for _, row in gauges[gauges.role == "downstream"].iterrows():
        down = row["staid"]
        ups = upstream_list(row)
        for arm in arms:
            dd = data.get((arm, down))
            if dd is None or "residual_gauge_mean" not in dd or "volume_sens" not in dd:
                continue
            total = dd["residual_gauge_mean"].values.astype(float)
            inherited = np.zeros_like(total)
            transfers = []
            ok = True
            for up in ups:
                du = data.get((arm, up))
                if du is None or "residual_gauge_mean" not in du:
                    ok = False
                    break
                up_comid = int(du["COMID"].values[du["is_gauge_reach"].values == 1][0])
                idx = np.where(dd["COMID"].values.astype(int) == up_comid)[0]
                if idx.size == 0:
                    ok = False
                    break
                tr = float(dd["volume_sens"].values[idx[0]])
                transfers.append(tr)
                inherited = inherited + du["residual_gauge_mean"].values.astype(float) * tr
            if not ok:
                continue
            starts = dd["residual_window_start_day"].values
            labels = [SEASON.get(int(s - starts[0]), str(s)) for s in starts]
            for w in range(len(total)):
                rows.append(dict(arm=arm, staid=down, window=labels[w], total=total[w], inherited=inherited[w],
                                 local=total[w] - inherited[w], n_up=len(ups), transfer_mean=float(np.mean(transfers))))
    return pd.DataFrame(rows)


def per_gauge_celerity(arms, gauges, data):
    """Per (arm, gauge, kind): origin-fit slope of mean lag vs distance → effective celerity (m/s)."""
    rows = []
    for staid in gauges["staid"]:
        for arm in arms:
            ds = data.get((arm, staid))
            if ds is None or "kernel_mean_lag_days" not in ds:
                continue
            dist = ds["dist_to_gauge_m"].values.astype(float)
            for kind, flag in [("high", 1), ("low", 0)]:
                sel = ds["anchor_is_high"].values == flag
                if not sel.any():
                    continue
                lag_d = np.nanmean(ds["kernel_mean_lag_days"].values[sel], axis=0) * 86400.0  # seconds
                m = np.isfinite(lag_d) & np.isfinite(dist) & (dist > 0) & (lag_d > 0)
                if m.sum() < 5:
                    continue
                slope = float(np.sum(dist[m] * lag_d[m]) / np.sum(dist[m] ** 2))  # s per m
                rows.append(dict(arm=arm, staid=staid, kind=kind, celerity_m_s=1.0 / slope if slope > 0 else np.nan,
                                 n_reach=int(m.sum()), max_dist_km=float(dist[m].max() / 1000.0),
                                 kernel_mass_median=float(np.nanmedian(ds["kernel_mass"].values[sel])),
                                 volume_sens_median=float(np.nanmedian(ds["volume_sens"].values)) if "volume_sens" in ds else np.nan,
                                 frac_vol_lt_half=float(np.mean(ds["volume_sens"].values < 0.5)) if "volume_sens" in ds else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- per-gauge figures
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
                ax.plot(np.arange(k.size) / 24.0, k, color=colors[arm], label=arm, lw=1.4)
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


def fig_influence_maps(out, arms, gauges, data, fabric: Path, staids):
    if not fabric.exists():
        print(f"fabric {fabric} not found; skipping influence maps")
        return
    for staid in staids:
        any_ds = next((data[(a, staid)] for a in arms if (a, staid) in data), None)
        if any_ds is None:
            continue
        gdf = load_polygons(fabric, any_ds["COMID"].values)
        if gdf.empty:
            print(f"  no polygons found for {staid} (outside the pfaf-7 fabric?); skipping map")
            continue
        fields = [f for f in ["residual_attr", "volume_sens"] if f in any_ds]
        fig, axes = plt.subplots(len(fields), len(arms), figsize=(3.6 * len(arms), 3.6 * len(fields)), squeeze=False)
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
                kw = dict(cmap="RdBu_r", vmin=-res_abs, vmax=res_abs) if field == "residual_attr" else dict(cmap="viridis", vmin=0.0, vmax=1.2)
                g.plot(column="v", ax=ax, legend=(c == len(arms) - 1), edgecolor="none", **kw, legend_kwds={"shrink": 0.6, "label": field})
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
    dec = inherited_decomposition(arms, gauges, data)
    for down, sub in dec.groupby("staid"):
        ups = upstream_list(gauges[gauges.staid == down].iloc[0])
        fig, axes = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 3.8), squeeze=False, sharey=True)
        for c, arm in enumerate(arms):
            ax = axes[0, c]
            s = sub[sub.arm == arm]
            if s.empty:
                ax.set_title(f"{arm}: n/a")
                continue
            x = np.arange(len(s))
            w = 0.27
            ax.bar(x - w, s.total, w, color="k", alpha=0.75, label="downstream bias")
            ax.bar(x, s.inherited, w, color=colors[arm], label=f"inherited from {'+'.join(ups)} × {s.transfer_mean.iloc[0]:.2f}")
            ax.bar(x + w, s.local, w, color=colors[arm], alpha=0.4, hatch="//", label="local remainder")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(list(s.window), fontsize=8)
            ax.set_title(arm, fontsize=10)
            if c == 0:
                ax.set_ylabel("mean(pred − obs), m³/s")
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"Downstream {down} bias: inherited from upstream gauge(s) {', '.join(ups)} vs local (WY windows)", y=1.02)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"inherited_vs_local_{down}.png", dpi=150, bbox_inches="tight")
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
        axes[r, 0].axvline(1.0, color="k", lw=0.6, ls="--")
        axes[r, 1].axhline(1.0, color="k", lw=0.6, ls="--")
        axes[r, 0].set_title(f"{staid}: volume sensitivity (expect ≈1)")
        axes[r, 0].set_xlabel("d[ΣQ_g]/dq' (time-mean)")
        axes[r, 1].set_title(f"{staid}: volume sensitivity vs distance")
        axes[r, 1].set_xlabel("along-channel distance to gauge (km)")
        axes[r, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "volume_sens.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- population figures
def _box_by_arm(ax, df, col, arms, colors, ylabel, title, hline=None):
    vals = [df.loc[df.arm == a, col].dropna().values for a in arms]
    bp = ax.boxplot(vals, tick_labels=arms, patch_artist=True, showfliers=True, widths=0.6)
    for patch, a in zip(bp["boxes"], arms):
        patch.set_facecolor(colors[a])
        patch.set_alpha(0.55)
    for i, v in enumerate(vals):
        ax.scatter(np.full(len(v), i + 1) + np.random.uniform(-0.15, 0.15, len(v)), v, s=8, color="k", alpha=0.5, zorder=3)
        ax.text(i + 1, ax.get_ylim()[1], f"n={len(v)}", ha="center", va="top", fontsize=7)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.6, ls="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)


def fig_population(out, arms, gauges, data, colors):
    cel = per_gauge_celerity(arms, gauges, data)
    dec = inherited_decomposition(arms, gauges, data)
    cel.to_csv(out / "figures" / "population_celerity.csv", index=False)
    dec.to_csv(out / "figures" / "population_inherited.csv", index=False)

    # 1. effective celerity per gauge, by arm and flow state
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _box_by_arm(axes[0], cel[cel.kind == "high"], "celerity_m_s", arms, colors, "m/s", "Effective celerity per gauge — high-flow anchors")
    _box_by_arm(axes[1], cel[cel.kind == "low"], "celerity_m_s", arms, colors, "m/s", "Effective celerity per gauge — low-flow anchors")
    _box_by_arm(axes[2], cel[cel.kind == "low"], "kernel_mass_median", arms, colors, "Σ_lag kernel (median reach)", "Kernel mass — low-flow anchors", hline=1.0)
    fig.suptitle(f"Population: {gauges.staid.nunique()} gauges ({(gauges.role == 'downstream').sum()} GAGES-II Ref downstream + nested upstream)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "figures" / "population_celerity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. pooled mean lag vs distance, all gauges, per arm (low-flow anchors)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for c, (kind, flag) in enumerate([("high", 1), ("low", 0)]):
        ax = axes[c]
        for arm in arms:
            xs, ys = [], []
            for staid in gauges["staid"]:
                ds = data.get((arm, staid))
                if ds is None or "kernel_mean_lag_days" not in ds:
                    continue
                sel = ds["anchor_is_high"].values == flag
                if not sel.any():
                    continue
                xs.append(ds["dist_to_gauge_m"].values / 1000.0)
                ys.append(np.nanmean(ds["kernel_mean_lag_days"].values[sel], axis=0))
            if not xs:
                continue
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            m = np.isfinite(x) & np.isfinite(y) & (x > 0)
            ax.scatter(x[m], y[m], s=4, color=colors[arm], alpha=0.35, edgecolor="none")
            slope = np.sum(x[m] * y[m]) / np.sum(x[m] ** 2)
            xx = np.linspace(0, np.nanmax(x[m]), 50)
            ax.plot(xx, slope * xx, color=colors[arm], lw=2, label=f"{arm}: {1000.0 / (slope * 86400.0):.2f} m/s")
        ax.set_title(f"Kernel mean lag vs distance, all reaches — {kind}-flow anchors")
        ax.set_xlabel("along-channel distance to gauge (km)")
        ax.set_ylabel("mean lag (days)")
        ax.set_ylim(-1, None)
        ax.legend(fontsize=8, title="pooled origin fit")
    fig.tight_layout()
    fig.savefig(out / "figures" / "population_lag_vs_distance.png", dpi=150)
    plt.close(fig)

    # 3. volume sensitivity per gauge
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vs = cel[cel.kind == "low"] if (cel.kind == "low").any() else cel
    _box_by_arm(axes[0], vs, "volume_sens_median", arms, colors, "median d[ΣQ_g]/dq'", "Volume sensitivity per gauge (expect ≈1)", hline=1.0)
    _box_by_arm(axes[1], vs, "frac_vol_lt_half", arms, colors, "fraction of reaches", "Reaches with volume sensitivity < 0.5 (mass lost)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "population_volume_sens.png", dpi=150)
    plt.close(fig)

    # 4. inherited vs local across downstream gauges
    if not dec.empty:
        share = (
            dec.assign(abs_in=dec.inherited.abs(), abs_tot=dec.total.abs())
            .groupby(["arm", "staid"], as_index=False)
            .agg(abs_in=("abs_in", "sum"), abs_tot=("abs_tot", "sum"), n_up=("n_up", "first"))
        )
        share["inherited_share"] = share.abs_in / share.abs_tot.replace(0, np.nan)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        _box_by_arm(axes[0], share, "inherited_share", arms, colors, "Σ|inherited| / Σ|total| over 4 windows",
                    "Share of downstream bias inherited from upstream gauges", hline=1.0)
        for arm in arms:
            s = dec[dec.arm == arm]
            axes[1].scatter(s.total, s.inherited, s=14, color=colors[arm], alpha=0.7, label=arm, edgecolor="none")
        lim = np.nanmax(np.abs(dec[["total", "inherited"]].values)) * 1.05 if not dec.empty else 1
        axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        axes[1].axhline(0, color="k", lw=0.4)
        axes[1].axvline(0, color="k", lw=0.4)
        axes[1].set_xlabel("downstream mean(pred − obs), m³/s")
        axes[1].set_ylabel("inherited from upstream gauge(s), m³/s")
        axes[1].set_title("Per window: total vs inherited (1:1 = fully inherited)", fontsize=10)
        axes[1].legend(fontsize=7)
        gm = dec.groupby(["arm", "staid"], as_index=False).agg(total=("total", "mean"))
        _box_by_arm(axes[2], gm, "total", arms, colors, "m³/s", "Downstream mean bias per gauge (WY2000 mean)", hline=0.0)
        fig.tight_layout()
        fig.savefig(out / "figures" / "population_inherited.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--fabric", type=Path, default=Path(DEFAULT_FABRIC))
    ap.add_argument("--maps", type=int, default=None, help="influence maps for the N largest gauges")
    args = ap.parse_args()
    out = args.out
    (out / "figures").mkdir(exist_ok=True)
    manifest, arms, gauges, data = load(out)
    arms = [a for a in arms if any(k[0] == a for k in data)]
    if not data:
        raise SystemExit("no gauge netCDFs found under the output directory")
    colors = arm_colors(arms)
    population = gauges.staid.nunique() > 6
    print(f"arms: {arms}; gauges: {gauges.staid.nunique()}; datasets: {len(data)}; mode: {'population' if population else 'per-gauge'}")
    np.random.seed(0)
    if population:
        fig_population(out, arms, gauges, data, colors)
        n_maps = 2 if args.maps is None else args.maps
    else:
        fig_kernel_by_lag(out, arms, gauges, data, colors)
        fig_kernel_vs_distance(out, arms, gauges, data, colors)
        fig_volume_sens(out, arms, gauges, data, colors)
        n_maps = len(gauges) if args.maps is None else args.maps
    fig_inherited_vs_local(out, arms, gauges, data, colors)
    # maps for the largest gauges (by reach count) that have data
    sizes = {}
    for staid in gauges["staid"]:
        ds = next((data[(a, staid)] for a in arms if (a, staid) in data), None)
        if ds is not None:
            sizes[staid] = ds.sizes["reach"]
    largest = [s for s, _ in sorted(sizes.items(), key=lambda kv: -kv[1])][:n_maps]
    fig_influence_maps(out, arms, gauges, data, args.fabric, largest)
    for p in sorted((out / "figures").glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
