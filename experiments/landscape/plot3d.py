#!/usr/bin/env python
"""3-D alpha-landscape figure for one gauge (src/experiment/landscape).

Reads a single per-gauge netCDF written by the landscape study and renders
the three orthogonal slices (n-p at q=q*, n-q at p=p*, p-q at n=n*) as one
3-D scene: a static matplotlib PNG (two viewpoints) plus a self-contained
interactive plotly HTML.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/plot3d.py <gauge.nc> --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    import plotly.graph_objects as go

    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

ALPHA_NAMES = ["n", "p", "q"]
CLAMP_THRESHOLD = 0.05
ARROW_CLIP = 2.0
EIG_STYLES = [
    ("solid", "v1 (stiff)"),
    ("dotted", "v2"),
    ("dashed", "v3 (sloppy)"),
]


def load(path: Path):
    return xr.open_dataset(path, decode_timedelta=False)


def plane_meshes(ds):
    """Build absolute-alpha (X, Y, Z), NSE and clamped-mask grids for the
    three physical planes n-p, n-q, p-q (skip the stiff-sloppy plane)."""
    plane_names = ds.attrs["plane_names"].split(",")
    alpha_star = ds["alpha_star"].values.astype(np.float64)
    meshes = {}
    for idx, name in enumerate(plane_names):
        if name == "stiff-sloppy":
            continue
        axis_a = ds["grid_axis_a"].values[idx]
        axis_b = ds["grid_axis_b"].values[idx]
        basis_a = ds["basis_a"].values[idx]
        basis_b = ds["basis_b"].values[idx]
        nse = ds["grid_nse"].values[idx]
        clamped = ds["grid_clamped"].values[idx]
        A, B = np.meshgrid(axis_a, axis_b, indexing="ij")
        X = alpha_star[0] + A * basis_a[0] + B * basis_b[0]
        Y = alpha_star[1] + A * basis_a[1] + B * basis_b[1]
        Z = alpha_star[2] + A * basis_a[2] + B * basis_b[2]
        meshes[name] = dict(X=X, Y=Y, Z=Z, nse=nse, clamped=clamped)
    return meshes


def eig_arrows(ds):
    """Return list of (vec_from_alpha_star, raw_hw, clipped_hw, style, label)
    for the three eigenvectors at the 10% tolerance."""
    eigvec = ds["eigvec_star"].values  # (component, k)
    half_width = ds["half_width"].values  # (tol, k)
    tolerances = ds["tolerances"].values
    tol_idx = int(np.argmin(np.abs(tolerances - 0.1)))
    arrows = []
    for k in range(3):
        v = eigvec[:, k]
        hw = float(half_width[tol_idx, k])
        hw_clipped = min(hw, ARROW_CLIP)
        style, label = EIG_STYLES[k]
        if hw > ARROW_CLIP:
            label = f"{label}, hw={hw:.2f} (clipped to {ARROW_CLIP:.1f})"
        else:
            label = f"{label}, hw={hw:.2f}"
        arrows.append((v * hw_clipped, hw, hw_clipped, style, label))
    return arrows


def shared_norm_cmap(meshes):
    all_nse = np.concatenate([m["nse"].ravel() for m in meshes.values()])
    norm = Normalize(vmin=float(np.nanmin(all_nse)), vmax=float(np.nanmax(all_nse)))
    cmap = plt.get_cmap("viridis")
    return norm, cmap


# --------------------------------------------------------------------- matplotlib
def plot_mpl(ds, meshes, arrows, out_png: Path):
    staid = ds.attrs["staid"]
    nse0 = float(ds["nse0"].values)
    nse_star = float(ds["nse_star"].values)
    alpha_star = ds["alpha_star"].values

    norm, cmap = shared_norm_cmap(meshes)
    grey = np.array([0.6, 0.6, 0.6, 0.9])

    fig = plt.figure(figsize=(16, 7.5))
    views = [(25, -60), (25, 30)]
    legend_handles = None
    for panel, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        for name, m in meshes.items():
            colors = cmap(norm(m["nse"]))
            mask = m["clamped"] > CLAMP_THRESHOLD
            colors[mask] = grey
            ax.plot_surface(
                m["X"], m["Y"], m["Z"], facecolors=colors, shade=False, alpha=0.9,
                rstride=1, cstride=1, linewidth=0, antialiased=False,
            )
        ax.plot([0.0], [0.0], [0.0], marker="x", color="k", ms=14, mew=3, zorder=5)
        ax.plot(
            [alpha_star[0]], [alpha_star[1]], [alpha_star[2]],
            marker="*", color="red", ms=16, zorder=5,
        )
        for vec, _hw, _hwc, style, _label in arrows:
            end = alpha_star + vec
            ax.plot(
                [alpha_star[0], end[0]], [alpha_star[1], end[1]], [alpha_star[2], end[2]],
                color="k", lw=1.8, linestyle=style, zorder=6,
            )
        ax.set_xlabel("ln multiplier n")
        ax.set_ylabel("ln multiplier p")
        ax.set_zlabel("ln multiplier q")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=9)
        if legend_handles is None:
            legend_handles = [
                Line2D([0], [0], marker="x", color="k", linestyle="None", ms=10, mew=2.5,
                       label="trained (alpha=0)"),
                Line2D([0], [0], marker="*", color="red", linestyle="None", ms=13,
                       label="optimum (alpha*)"),
            ] + [
                Line2D([0], [0], color="k", lw=1.8, linestyle=style, label=label)
                for _vec, _hw, _hwc, style, label in arrows
            ]

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=fig.axes, shrink=0.6, pad=0.03, label="NSE")
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{staid}  NSE trained {nse0:.3f} -> optimum {nse_star:.3f}", fontsize=12)
    fig.subplots_adjust(bottom=0.14, top=0.9)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- plotly
def mpl_colorscale(cmap, n=32):
    return [[i / (n - 1), matplotlib.colors.rgb2hex(cmap(i / (n - 1)))] for i in range(n)]


def plot_plotly(ds, meshes, arrows, out_html: Path):
    staid = ds.attrs["staid"]
    nse0 = float(ds["nse0"].values)
    nse_star = float(ds["nse_star"].values)
    alpha_star = ds["alpha_star"].values

    norm, cmap = shared_norm_cmap(meshes)
    vmin, vmax = norm.vmin, norm.vmax
    span = vmax - vmin if vmax > vmin else 1.0
    # Extend the color range above vmax with a solid grey band for clamped cells.
    grey_frac = 0.85
    cmax_ext = vmax + span * (1.0 / grey_frac - 1.0)
    colorscale = [[f * grey_frac, c] for f, c in mpl_colorscale(cmap)]
    colorscale += [[grey_frac, "rgb(153,153,153)"], [1.0, "rgb(153,153,153)"]]
    sentinel = vmax + span * 0.15

    fig = go.Figure()
    for name, m in meshes.items():
        surfacecolor = np.where(m["clamped"] > CLAMP_THRESHOLD, sentinel, m["nse"])
        fig.add_trace(
            go.Surface(
                x=m["X"], y=m["Y"], z=m["Z"], surfacecolor=surfacecolor,
                cmin=vmin, cmax=cmax_ext, colorscale=colorscale,
                showscale=(name == "n-p"),
                colorbar=dict(title="NSE") if name == "n-p" else None,
                opacity=0.9, name=name,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0], mode="markers",
            marker=dict(symbol="x", size=6, color="black"),
            name="trained (alpha=0)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[alpha_star[0]], y=[alpha_star[1]], z=[alpha_star[2]], mode="markers",
            marker=dict(symbol="diamond", size=7, color="red"),
            name="optimum (alpha*)",
        )
    )
    dash_map = {"solid": "solid", "dotted": "dot", "dashed": "dash"}
    for vec, _hw, _hwc, style, label in arrows:
        end = alpha_star + vec
        fig.add_trace(
            go.Scatter3d(
                x=[alpha_star[0], end[0]], y=[alpha_star[1], end[1]], z=[alpha_star[2], end[2]],
                mode="lines", line=dict(color="black", width=5, dash=dash_map[style]),
                name=label,
            )
        )

    fig.update_layout(
        title=f"{staid}  NSE trained {nse0:.3f} -> optimum {nse_star:.3f}",
        scene=dict(
            xaxis_title="ln multiplier n",
            yaxis_title="ln multiplier p",
            zaxis_title="ln multiplier q",
        ),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gauge_nc", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = load(args.gauge_nc)
    staid = ds.attrs["staid"]
    args.out.mkdir(parents=True, exist_ok=True)

    meshes = plane_meshes(ds)
    arrows = eig_arrows(ds)

    out_png = args.out / f"landscape3d_{staid}.png"
    plot_mpl(ds, meshes, arrows, out_png)
    print(f"wrote {out_png}")

    if HAVE_PLOTLY:
        out_html = args.out / f"landscape3d_{staid}.html"
        plot_plotly(ds, meshes, arrows, out_html)
        print(f"wrote {out_html}")
    else:
        print("plotly not importable; skipping HTML output")


if __name__ == "__main__":
    main()
