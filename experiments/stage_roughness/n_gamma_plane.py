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
    a = ap.parse_args()
    files = sorted(f for d in a.study_dirs for f in glob.glob(f"{d}/*/gauges/*.nc"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(files)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, files):
        ds = xr.open_dataset(f)
        planes = ds.attrs["plane_names"].split(",")
        if a.plane not in planes:
            ax.set_title(f"{ds.attrs['staid']}: no {a.plane} plane"); ax.set_axis_off(); continue
        k = planes.index(a.plane)
        ga, gb = ds["grid_axis_a"].values[k], ds["grid_axis_b"].values[k]
        nse = ds["grid_nse"].values[k]  # (ga, gb): a = slot 0 (n), b = slot 1 (gamma)
        A, B = np.meshgrid(ga, gb, indexing="ij")
        vmax = np.nanmax(nse); vmin = max(np.nanmin(nse), vmax - 0.3)
        if not np.isfinite(vmax) or vmax - vmin < 1e-4:
            vmin, vmax = (vmax - 0.01, vmax + 0.01) if np.isfinite(vmax) else (0.0, 1.0)
        cf = ax.contourf(np.exp(A), np.exp(B), nse, levels=np.linspace(vmin, vmax, 16), cmap="viridis")
        lv = [l for l in (vmax - 0.05, vmax - 0.01) if l > vmin]
        if lv:
            ax.contour(np.exp(A), np.exp(B), nse, levels=lv, colors="w", linewidths=0.8)
        ast = ds["alpha_star"].values
        ax.plot(1.0, 1.0, "ko", ms=6, label="trained")
        ax.plot(np.exp(ast[0]), np.exp(ast[1]), "r*", ms=11, label="optimum")
        h = ds["hess0"].values
        ratio = abs(h[1, 1]) / max(abs(h[0, 0]), 1e-12)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n_0 multiplier"); ax.set_ylabel("gamma multiplier")
        ax.set_title(f"{ds.attrs['staid']}  ({int(ds.attrs['n_reach'])} reaches)\nNSE {float(ds['nse0']):.3f} -> {float(ds['nse_star']):.3f}   |H_gg/H_nn| = {ratio:.3f}", fontsize=9)
        fig.colorbar(cf, ax=ax, shrink=0.8, label="NSE")
    for ax in axes[n:]:
        ax.set_axis_off()
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("(n_0, gamma) loss plane per gauge, n_0 + gamma head with p = 21 and q = 0.65 fixed; white contours 0.01 and 0.05 NSE below the plane maximum", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130, facecolor="white")
    print(f"wrote {a.out} ({n} gauges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
