#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "xarray", "netCDF4", "matplotlib"]
# ///
"""The (n_0, gamma) loss plane at each gauge of a landscape-n0-gamma study.

One panel per gauge: NSE over the n-gamma plane (alpha space, log multipliers
on the trained fields; the third slot, fixed q, is inert), the trained point
(alpha = 0) as a black dot, the per-gauge optimum as a red star, and the
curvature ratio |H_gg| / |H_nn| at the trained point in the title. A flat
valley along gamma shows as contours running parallel to the gamma axis.

    experiments/stage_roughness/n_gamma_plane.py <study-dir> [<study-dir> ...] --out <png>
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plane", default="n-gamma")
    ap.add_argument("--three-d", action="store_true", help="3-D surfaces (NSE as height) instead of contour panels")
    ap.add_argument("--elev", type=float, default=32.0)
    ap.add_argument("--azim", type=float, default=-135.0)
    a = ap.parse_args()
    files = sorted(f for d in a.study_dirs for f in glob.glob(f"{d}/*/gauges/*.nc"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(files)
    cols = 4
    rows = int(np.ceil(n / cols))
    if a.three_d:
        fig = plt.figure(figsize=(5.2 * cols, 4.6 * rows))
        axes = [fig.add_subplot(rows, cols, i + 1, projection="3d") for i in range(rows * cols)]
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows))
        axes = list(np.atleast_1d(axes).ravel())
    for ax, f in zip(axes, files):
        ds = xr.open_dataset(f)
        planes = ds.attrs["plane_names"].split(",")
        if a.plane not in planes:
            ax.set_title(f"{ds.attrs['staid']}: no {a.plane} plane"); ax.set_axis_off(); continue
        k = planes.index(a.plane)
        ga, gb = ds["grid_axis_a"].values[k], ds["grid_axis_b"].values[k]
        nse = ds["grid_nse"].values[k]  # (ga, gb): a = slot 0 (n), b = slot 1 (gamma)
        A, B = np.meshgrid(ga, gb, indexing="ij")
        ast = ds["alpha_star"].values
        h = ds["hess0"].values
        ratio = abs(h[1, 1]) / max(abs(h[0, 0]), 1e-12)
        title = f"{ds.attrs['staid']}  ({int(ds.attrs['n_reach'])} reaches)\nNSE {float(ds['nse0']):.3f} -> {float(ds['nse_star']):.3f}   |H_gg/H_nn| = {ratio:.3f}"
        if a.three_d:
            zmax = np.nanmax(nse); zmin = max(np.nanmin(nse), zmax - 0.3)
            z = np.clip(nse, zmin, zmax)
            # x = n_0 multiplier, y = gamma multiplier (log axes drawn as alpha), z = NSE
            ax.plot_surface(A, B, z, cmap="viridis", vmin=zmin, vmax=zmax, linewidth=0, antialiased=True, alpha=0.95)
            def zat(x, y):
                return float(z[int(np.argmin(np.abs(ga - x))), int(np.argmin(np.abs(gb - y)))])
            ax.plot([0, 0], [0, 0], [zmin, zat(0, 0)], color="k", lw=1.5)
            ax.plot([0], [0], [zat(0, 0)], "ko", ms=6)
            ax.plot([ast[0], ast[0]], [ast[1], ast[1]], [zmin, zat(ast[0], ast[1])], color="red", lw=1.5)
            ax.plot([ast[0]], [ast[1]], [zat(ast[0], ast[1])], "r*", ms=12)
            ticks = [-1.0986, -0.6931, 0.0, 0.6931, 1.0986]
            ax.set_xticks(ticks); ax.set_xticklabels(["x1/3", "x1/2", "x1", "x2", "x3"], fontsize=7)
            ax.set_yticks(ticks); ax.set_yticklabels(["x1/3", "x1/2", "x1", "x2", "x3"], fontsize=7)
            ax.set_xlabel("n_0 multiplier", fontsize=8); ax.set_ylabel("gamma multiplier", fontsize=8); ax.set_zlabel("NSE", fontsize=8)
            ax.set_zlim(zmin, zmax + 0.02)
            ax.view_init(elev=a.elev, azim=a.azim)
            ax.set_title(title, fontsize=8)
            continue
        vmax = np.nanmax(nse); vmin = max(np.nanmin(nse), vmax - 0.3)
        if not np.isfinite(vmax) or vmax - vmin < 1e-4:
            vmin, vmax = (vmax - 0.01, vmax + 0.01) if np.isfinite(vmax) else (0.0, 1.0)
        cf = ax.contourf(np.exp(A), np.exp(B), nse, levels=np.linspace(vmin, vmax, 16), cmap="viridis")
        lv = [l for l in (vmax - 0.05, vmax - 0.01) if l > vmin]
        if lv:
            ax.contour(np.exp(A), np.exp(B), nse, levels=lv, colors="w", linewidths=0.8)
        ax.plot(1.0, 1.0, "ko", ms=6, label="trained")
        ax.plot(np.exp(ast[0]), np.exp(ast[1]), "r*", ms=11, label="optimum")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_0 multiplier"); ax.set_ylabel("gamma multiplier")
        ax.set_title(title, fontsize=9)
        fig.colorbar(cf, ax=ax, shrink=0.8, label="NSE")
    for ax in axes[n:]:
        ax.set_axis_off()
    if not a.three_d:
        axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("(n_0, gamma) loss surface per gauge, n_0 + gamma head with p = 21 and q = 0.65 fixed; black = trained point, red star = per-gauge optimum"
                 + ("" if a.three_d else "; white contours 0.01 and 0.05 NSE below the plane maximum"), fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor="white")
    print(f"wrote {a.out} ({n} gauges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
