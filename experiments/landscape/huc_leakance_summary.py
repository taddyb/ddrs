#!/usr/bin/env python
"""Per-region read of the leakance-frame curvature census (landscape-huc-leakance).

For every gauge netCDF under the sharded run dirs: the trained-point Hessian
diagonal in (n, K_D, d_gw), the same at the local optimum, the n:K_D and n:d_gw
stiffness ratios, corr(n, d_gw), where alpha*_K_D went, and the loss plateau
width (fraction of the K_D x d_gw grid within 5 % of the trained loss).
Grouped by HUC2 with a table and one summary figure.

Usage:
    python huc_leakance_summary.py <run-dir-glob> --sel huc_selection.csv --out <dir>
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

BOX = 2.3025851
HUC_NAMES = {
    "01": "New England", "02": "Mid-Atlantic", "03": "South Atlantic-Gulf", "04": "Great Lakes",
    "05": "Ohio", "06": "Tennessee", "07": "Upper Mississippi", "08": "Lower Mississippi",
    "09": "Souris-Red-Rainy", "10": "Missouri", "11": "Arkansas-White-Red", "12": "Texas-Gulf",
    "13": "Rio Grande", "14": "Upper Colorado", "15": "Lower Colorado", "16": "Great Basin",
    "17": "Pacific Northwest", "18": "California",
}


def collect(run_glob: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{run_glob}/*/gauges/*.nc")):
        ds = xr.open_dataset(p, decode_timedelta=False)
        H0 = ds["hess0"].values.astype(float)
        Hs = ds["hess_star"].values.astype(float)
        d = np.sqrt(np.abs(np.diag(H0)))
        C = H0 / np.outer(d, d)
        a = ds["alpha_star"].values.astype(float)
        names = ds.attrs["plane_names"].split(",")
        L = ds["grid_loss"].values.astype(float)
        cl = ds["grid_clamped"].values.astype(float)
        k = names.index("kd-dgw")
        Lk = np.where(cl[k] > 0.05, np.nan, L[k])
        loss0 = float(ds["loss0"])
        plateau = float(np.nanmean(np.abs(Lk - loss0) <= 0.05 * loss0))
        rows.append(dict(
            staid=ds.attrs["staid"], n_reach=int(ds.attrs["n_reach"]),
            nse0=float(ds["nse0"]), nse_star=float(ds["nse_star"]),
            h0_n=H0[0, 0], h0_kd=H0[1, 1], h0_dgw=H0[2, 2],
            hs_n=Hs[0, 0], hs_kd=Hs[1, 1], hs_dgw=Hs[2, 2],
            corr_n_dgw=C[0, 2], corr_n_kd=C[0, 1],
            a_n=a[0], a_kd=a[1], a_dgw=a[2],
            kd_at_wall=abs(abs(a[1]) - BOX) < 0.01,
            plateau_kd_dgw=plateau,
            clamped_frac_n_planes=float(np.mean([(cl[i] > 0.05).mean() for i in range(3) if names[i] != "kd-dgw"])),
        ))
        ds.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_glob")
    ap.add_argument("--sel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-nse0", type=float, default=None, help="keep only gauges with trained NSE >= this (drop broken fits)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = collect(args.run_glob)
    if args.min_nse0 is not None:
        df = df[df.nse0 >= args.min_nse0]
    sel = pd.read_csv(args.sel, dtype={"staid": str, "huc2": str})
    df = df.merge(sel[["staid", "huc2", "AGGECOREGI", "STATE", "STANAME"]], on="staid", how="left")
    df["ratio_n_kd"] = df.h0_n / df.h0_kd.abs()
    df["ratio_n_dgw"] = df.h0_n / df.h0_dgw.abs()
    df.to_csv(args.out / "huc_leakance_gauges.csv", index=False)

    g = df.groupby("huc2")
    tab = pd.DataFrame({
        "region": [HUC_NAMES.get(h, "") for h in g.size().index],
        "gauges": g.size(),
        "median |H_n| trained": g.h0_n.apply(lambda s: np.median(np.abs(s))),
        "median |H_KD| trained": g.h0_kd.apply(lambda s: np.median(np.abs(s))),
        "median |H_dgw| trained": g.h0_dgw.apply(lambda s: np.median(np.abs(s))),
        "median n:K_D": g.ratio_n_kd.median(),
        "median n:d_gw": g.ratio_n_dgw.median(),
        "K_D curv < 0": g.h0_kd.apply(lambda s: int((s < 0).sum())),
        "K_D at wall": g.kd_at_wall.sum(),
        "median corr(n,d_gw)": g.corr_n_dgw.median(),
        "median plateau frac": g.plateau_kd_dgw.median(),
        "median dNSE to opt": (g.nse_star.median() - g.nse0.median()),
    })
    cols = list(tab.columns)
    lines = ["| HUC2 | " + " | ".join(cols) + " |", "|---|" + "|".join("---" for _ in cols) + "|"]
    for h, r in tab.iterrows():
        lines.append(f"| {h} | " + " | ".join(str(v) if isinstance(v, str) else f"{v:.3g}" for v in r.values) + " |")
    (args.out / "HUC_LEAKANCE_TABLE.md").write_text("\n".join(lines) + "\n")
    print(tab.to_string(float_format=lambda x: f"{x:.3g}"))

    # figure: per-region distributions
    order = sorted(df.huc2.unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (col, title, log) in zip(axes, [
        ("ratio_n_kd", "stiffness of n relative to K/D at the trained point", True),
        ("ratio_n_dgw", "stiffness of n relative to d_gw at the trained point", True),
        ("plateau_kd_dgw", "fraction of the K/D x d_gw box within 5 % of the trained loss", False),
    ]):
        data = [df.loc[df.huc2 == h, col].replace([np.inf, -np.inf], np.nan).dropna().values for h in order]
        bp = ax.boxplot(data, positions=range(len(order)), widths=0.6, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor="#a9cbf2", edgecolor="#2a78d6", linewidth=1)
        for med in bp["medians"]:
            med.set(color="#0d3a73", linewidth=2)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d, s=10, color="#2a78d6", alpha=0.6, zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8)
        ax.set_xlabel("HUC2 region")
        if log:
            ax.set_yscale("log")
            ax.axhline(1, color="#52514e", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Leakance-frame curvature by region: two-way leakance arm, WY2000, nse-batch, ten gauges per HUC2"
                 + (f" (trained NSE >= {args.min_nse0:g} only, {len(df)} gauges)" if args.min_nse0 is not None else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "huc_leakance_summary.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"{len(df)} gauges -> {args.out}")


if __name__ == "__main__":
    main()
