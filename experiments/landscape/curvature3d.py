#!/usr/bin/env python
"""Curvature landscape: the loss surface on each parameter plane, and beneath
it the LOCAL curvature of that surface, differenced from the 25x25 loss grid.

Row 1  loss L(a, b)                        height = loss
Row 2  stiff local curvature lambda_1(a,b)  height = largest eigenvalue of the
                                            2x2 Hessian of L at that grid cell
Row 3  sloppy local curvature lambda_2(a,b) smallest eigenvalue; negative means
                                            the surface is a saddle there

The point curvature the study reports (hess_star) is one number per gauge at
alpha*. This shows how that curvature varies across the whole box.

Usage:
    python curvature3d.py <gauge.nc> --out <dir> [--label text]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

CLAMP_THRESHOLD = 0.05
# palette.md: sequential = one hue light->dark; diverging = two hues + gray midpoint
CMAP_LOSS = LinearSegmentedColormap.from_list("loss", ["#efe9ff", "#9085e9", "#4a3aa7", "#1d1350"])
CMAP_STIFF = LinearSegmentedColormap.from_list("stiff", ["#e8f1fc", "#8fbdf0", "#2a78d6", "#0d3a73"])
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f6c9b8", "#d9d9d6", "#a9cbf2", "#2a78d6"])


def axis_names(ds):
    raw = ds.attrs.get("param_names")
    names = raw.split(",") if raw else ["n", "p_spatial", "q_spatial"]
    return [x.replace("_spatial", "") for x in names]


def local_hessian_eigs(L, da, db):
    """Eigenvalues of the 2x2 Hessian of L at every interior cell, by central
    differences. Returns (lam1, lam2), lam1 >= lam2, NaN on the border."""
    lam1 = np.full_like(L, np.nan)
    lam2 = np.full_like(L, np.nan)
    Laa = (L[2:, 1:-1] - 2 * L[1:-1, 1:-1] + L[:-2, 1:-1]) / da**2
    Lbb = (L[1:-1, 2:] - 2 * L[1:-1, 1:-1] + L[1:-1, :-2]) / db**2
    Lab = (L[2:, 2:] - L[2:, :-2] - L[:-2, 2:] + L[:-2, :-2]) / (4 * da * db)
    tr = Laa + Lbb
    disc = np.sqrt(np.maximum((Laa - Lbb) ** 2 / 4 + Lab**2, 0.0))
    lam1[1:-1, 1:-1] = tr / 2 + disc
    lam2[1:-1, 1:-1] = tr / 2 - disc
    return lam1, lam2


def planes(ds):
    names = ds.attrs["plane_names"].split(",")
    labels = axis_names(ds)
    out = []
    for idx, name in enumerate(names):
        if name == "stiff-sloppy":
            continue
        a = ds["grid_axis_a"].values[idx].astype(float)
        b = ds["grid_axis_b"].values[idx].astype(float)
        L = ds["grid_loss"].values[idx].astype(float)
        clamped = ds["grid_clamped"].values[idx].astype(float)
        L = np.where(clamped > CLAMP_THRESHOLD, np.nan, L)
        lam1, lam2 = local_hessian_eigs(L, a[1] - a[0], b[1] - b[0])
        ia, ib = [int(np.argmax(np.abs(ds["basis_a"].values[idx]))), int(np.argmax(np.abs(ds["basis_b"].values[idx])))]
        out.append(dict(name=name, a=a, b=b, L=L, lam1=lam1, lam2=lam2, xl=labels[ia], yl=labels[ib]))
    return out


def surface(ax, A, B, Z, cmap, norm, zlabel):
    ax.plot_surface(A, B, Z, facecolors=cmap(norm(Z)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=4)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=28, azim=-55)


def figure(ds, out_png: Path, label: str):
    P = planes(ds)
    staid = ds.attrs["staid"]
    a_star = ds["alpha_star"].values.astype(float)
    center = np.zeros(3) if ds.attrs.get("slice_center") == "trained" else a_star
    eig_star = ds["eigval_star"].values.astype(float)

    all_L = np.concatenate([p["L"].ravel() for p in P])
    n_loss = Normalize(np.nanmin(all_L), np.nanmax(all_L))
    all_l1 = np.concatenate([p["lam1"].ravel() for p in P])
    n_stiff = SymLogNorm(linthresh=1e-3, vmin=0.0, vmax=np.nanmax(all_l1))
    all_l2 = np.concatenate([p["lam2"].ravel() for p in P])
    m = np.nanmax(np.abs(all_l2))
    n_div = SymLogNorm(linthresh=1e-4, vmin=-m, vmax=m)

    fig = plt.figure(figsize=(15, 13))
    fig.patch.set_facecolor("#fcfcfb")
    rows = [("loss  L(a, b)", "L", CMAP_LOSS, n_loss),
            ("stiff local curvature  lambda_1", "lam1", CMAP_STIFF, n_stiff),
            ("sloppy local curvature  lambda_2  (orange: saddle)", "lam2", CMAP_DIV, n_div)]
    for r, (title, key, cmap, norm) in enumerate(rows):
        for c, p in enumerate(P):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1, projection="3d")
            A, B = np.meshgrid(p["a"], p["b"], indexing="ij")
            surface(ax, A, B, p[key], cmap, norm, key)
            ax.set_xlabel(f"ln mult {p['xl']} (from {'trained' if center is a_star else 'trained'})", fontsize=7)
            ax.set_xlabel(f"ln multiplier {p['xl']}", fontsize=7)
            ax.set_ylabel(f"ln multiplier {p['yl']}", fontsize=7)
            if r == 0:
                ax.set_title(f"plane {p['name']}", fontsize=9)
            # mark the slice centre (alpha* or trained point) as a stem
            zc = np.nanmin(p[key]) if np.isfinite(np.nanmin(p[key])) else 0
            ax.plot([0], [0], [zc], marker="o", color="#0b0b0b", ms=4)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([0.93, 0.70 - 0.31 * r, 0.012, 0.22])
        fig.colorbar(mappable, cax=cax).set_label(title, fontsize=8)
    fig.suptitle(
        f"{label}   gauge {staid}, {ds.attrs['n_reach']} reaches   "
        f"NSE trained {float(ds['nse0']):.3f} -> optimum {float(ds['nse_star']):.3f}   "
        f"point Hessian at alpha*: eig {eig_star[0]:.3g}, {eig_star[1]:.3g}, {eig_star[2]:.3g}\n"
        "each plane is a slice through the slice centre (black dot); axes are log multipliers on the trained field; "
        "grid cells with >5% clamped reaches are blank",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.02, right=0.90, top=0.92, bottom=0.03, wspace=0.05, hspace=0.12)
    fig.savefig(out_png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def html(ds, out_html: Path, label: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    P = planes(ds)
    fig = make_subplots(rows=2, cols=3, specs=[[{"type": "surface"}] * 3] * 2,
                        subplot_titles=[f"loss, plane {p['name']}" for p in P] + [f"stiff local curvature, plane {p['name']}" for p in P],
                        horizontal_spacing=0.02, vertical_spacing=0.08)
    for c, p in enumerate(P):
        A, B = np.meshgrid(p["a"], p["b"], indexing="ij")
        fig.add_trace(go.Surface(x=A, y=B, z=p["L"], colorscale=[[0, "#efe9ff"], [0.5, "#9085e9"], [1, "#1d1350"]],
                                 showscale=(c == 0), colorbar=dict(title="loss", x=0.30, y=0.78, len=0.4),
                                 hovertemplate=f"{p['xl']} %{{x:.2f}}<br>{p['yl']} %{{y:.2f}}<br>loss %{{z:.4f}}<extra></extra>"),
                      row=1, col=c + 1)
        l1 = p["lam1"]
        fig.add_trace(go.Surface(x=A, y=B, z=l1, colorscale=[[0, "#e8f1fc"], [0.5, "#2a78d6"], [1, "#0d3a73"]],
                                 showscale=(c == 0), colorbar=dict(title="lambda_1", x=0.30, y=0.22, len=0.4),
                                 hovertemplate=f"{p['xl']} %{{x:.2f}}<br>{p['yl']} %{{y:.2f}}<br>lambda_1 %{{z:.3g}}<extra></extra>"),
                      row=2, col=c + 1)
        for r in (1, 2):
            fig.update_scenes(dict(xaxis_title=f"ln mult {p['xl']}", yaxis_title=f"ln mult {p['yl']}",
                                   zaxis_title="loss" if r == 1 else "lambda_1"), row=r, col=c + 1)
    fig.update_layout(title=f"{label}   gauge {ds.attrs['staid']}, {ds.attrs['n_reach']} reaches: loss (top) and its local stiff curvature (bottom)",
                      height=900, width=1500, paper_bgcolor="#fcfcfb")
    fig.write_html(out_html, include_plotlyjs="cdn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nc", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(args.nc, decode_timedelta=False)
    stem = f"curvature3d_{args.tag + '_' if args.tag else ''}{ds.attrs['staid']}"
    figure(ds, args.out / f"{stem}.png", args.label)
    html(ds, args.out / f"{stem}.html", args.label)
    print(args.out / f"{stem}.png")


if __name__ == "__main__":
    main()
