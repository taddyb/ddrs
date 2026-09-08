#!/usr/bin/env python
"""Li-et-al.-2018-style 3-D terrain render of one landscape plane.

Reads a single per-gauge landscape netCDF (src/experiment/landscape/output.rs)
and, for one plane (or all four), renders the loss surface as a shaded 3-D
terrain: matplotlib PNG (two viewpoints) + an interactive plotly HTML.

Grid geometry (verified against src/experiment/landscape/mod.rs:311-370, and
matching experiments/landscape/plots.py's plot_plane, which reads the same
file). The `slice_center` global attribute ("optimum", the default, or
"trained") records which convention centred the slices; absent on files from
before this attribute existed, in which case it defaults to "optimum":
  - Planes "n-p", "n-q", "p-q": grid_axis_a/b are always ABSOLUTE alpha
    values for the two swept log-multiplier components; the third component
    is pinned at alpha_star ("optimum") or at 0, the trained value
    ("trained"). Either way the trained point (alpha = 0) sits at (0, 0) on
    these axes and the optimum (alpha_star) sits at (alpha_star[i],
    alpha_star[j]) -- the marker math is unaffected by slice_center, only
    which value the third (unplotted) component is pinned at changes.
  - Plane "stiff-sloppy": grid_axis_a/b are offsets (s, t) along the Hessian
    eigenvectors v1 (stiff), v3 (sloppy) of H(alpha_star), from a centre that
    depends on slice_center. With "optimum" (default) the centre is
    alpha_star: the optimum sits at (0, 0), the trained point sits at
    (coord_trained[0], coord_trained[2]). With "trained" the centre is
    alpha = 0: the trained point sits at (0, 0), the optimum sits at
    (-coord_trained[0], -coord_trained[2]).
Note this means the two conventions differ (the axis-aligned planes are NOT
offsets from alpha_star), even though both are "the grid_axis_a/b variables".

Axis-aligned planes (n-p, n-q, p-q) are labelled in physical parameter values
by default: physical = median(x0) * exp(alpha), where x0 is the per-reach
trained field (n0/p0/q0); pass --alpha-axes to restore the old ln-multiplier
labelling. The stiff-sloppy plane always keeps its eigen-coordinate axes.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/surface.py \\
        <gauge.nc> --plane all --out <dir> [--log] [--zmax <loss>] [--alpha-axes]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

ALPHA_INDEX = {"n": 0, "p": 1, "q": 2}
CLAMP_THRESHOLD = 0.05
UPSAMPLE_MAX_SIDE = 15
UPSAMPLE_FACTOR = 4
VERT_EXAG = 2.0
GREY = np.array([0.55, 0.55, 0.55, 1.0])

# Nice physical-unit tick multipliers per decade, used to label axis-aligned
# planes (n-p, n-q, p-q) in physical parameter values instead of ln
# multipliers: physical = median(x0) * exp(alpha). n/p use the standard
# {1, 2, 5} log-tick pattern; q uses a denser {1, 2, 3, 5, 7} pattern so the
# axis reads as roughly linear over its narrower physical range.
PHYSICAL_TICK_SUBS = {
    "n": (1.0, 2.0, 5.0),
    "p": (1.0, 2.0, 5.0),
    "q": (1.0, 2.0, 3.0, 5.0, 7.0),
}
PHYSICAL_AXIS_LABEL = {
    "n": "Manning n (basin median x multiplier)",
    "p": "width coefficient p",
    "q": "width exponent q",
}
PARAM_DISPLAY_NAME = {"n": "Manning n", "p": "width coefficient p", "q": "width exponent q"}


def axis_labels(plane_name: str) -> tuple[str, str]:
    if plane_name == "stiff-sloppy":
        return "offset along v1 (stiff)", "offset along v3 (sloppy)"
    a_name, b_name = plane_name.split("-")
    return f"ln multiplier {a_name} (0 = trained)", f"ln multiplier {b_name} (0 = trained)"


def field_medians(ds) -> dict[str, float]:
    """Per-reach trained-field medians (n0, p0, q0) -> physical alpha=0 value."""
    return {k: float(np.median(ds[f"{k}0"].values)) for k in ("n", "p", "q")}


def active_mask(ds) -> np.ndarray:
    """Bool (n, p, q) mask of which alpha components are learned model
    parameters vs fixed at a constant default. Defaults to all-active for
    netCDFs written before the active/active_params schema existed."""
    if "active" in ds.variables:
        return ds["active"].values.astype(bool)
    return np.array([True, True, True])


def _nice_physical_ticks(median: float, alpha_lo: float, alpha_hi: float, param: str) -> tuple[list[float], list[str]]:
    """Nice physical-value tick locations, expressed as alpha (so callers can
    place them directly on the alpha grid), plus their labels. physical =
    median * exp(alpha); ticks are clipped to [alpha_lo, alpha_hi]."""
    lo, hi = (alpha_lo, alpha_hi) if alpha_lo <= alpha_hi else (alpha_hi, alpha_lo)
    vmin, vmax = median * np.exp(lo), median * np.exp(hi)
    if vmin <= 0 or vmax <= vmin:
        return [], []
    e_lo = int(np.floor(np.log10(vmin)))
    e_hi = int(np.ceil(np.log10(vmax)))
    values = sorted({
        s * 10.0 ** e
        for e in range(e_lo, e_hi + 1)
        for s in PHYSICAL_TICK_SUBS[param]
        if vmin - 1e-9 <= s * 10.0 ** e <= vmax + 1e-9
    })
    locs = [float(np.log(v / median)) for v in values]
    return locs, [f"{v:g}" for v in values]


def axis_ticks(ds, plane_name: str, ga: np.ndarray, gb: np.ndarray, alpha_axes: bool):
    """Return (xlabel, ylabel, xlocs, xlabels, ylocs, ylabels). The tick lists
    are None when the old ln-multiplier axes are in effect (--alpha-axes, or
    the stiff-sloppy plane, which always keeps its eigen-coordinate axes)."""
    if alpha_axes or plane_name == "stiff-sloppy":
        xlabel, ylabel = axis_labels(plane_name)
        return xlabel, ylabel, None, None, None, None
    a_name, b_name = plane_name.split("-")
    medians = field_medians(ds)
    xlocs, xlabels = _nice_physical_ticks(medians[a_name], float(ga.min()), float(ga.max()), a_name)
    ylocs, ylabels = _nice_physical_ticks(medians[b_name], float(gb.min()), float(gb.max()), b_name)
    return PHYSICAL_AXIS_LABEL[a_name], PHYSICAL_AXIS_LABEL[b_name], xlocs, xlabels, ylocs, ylabels


def title_param_suffix(ds, plane_name: str, alpha_axes: bool) -> str:
    """'n 0.103 -> 0.037, q 0.35 -> 0.22' style physical trained->optimum
    suffix for the two plotted parameters, empty for stiff-sloppy or when
    --alpha-axes is in effect."""
    if alpha_axes or plane_name == "stiff-sloppy":
        return ""
    a_name, b_name = plane_name.split("-")
    i, j = ALPHA_INDEX[a_name], ALPHA_INDEX[b_name]
    medians = field_medians(ds)
    alpha_star = ds["alpha_star"].values
    ta, oa = medians[a_name], medians[a_name] * float(np.exp(alpha_star[i]))
    tb, ob = medians[b_name], medians[b_name] * float(np.exp(alpha_star[j]))
    return f"  [{a_name} {ta:.3g} -> {oa:.3g}, {b_name} {tb:.3g} -> {ob:.3g}]"


def pinned_third_note(ds, plane_name: str, alpha_axes: bool) -> str:
    """'; third parameter pinned at <name> = <value> (<trained|optimum>)'
    for the unplotted third parameter of an axis-aligned plane."""
    if alpha_axes or plane_name == "stiff-sloppy":
        return ""
    a_name, b_name = plane_name.split("-")
    pinned = ({"n", "p", "q"} - {a_name, b_name}).pop()
    median = field_medians(ds)[pinned]
    if not active_mask(ds)[ALPHA_INDEX[pinned]]:
        # Fixed parameter: alpha_star is exactly 0 (Newton never moves it),
        # so it's pinned at the trained/default value regardless of
        # slice_center -- label it as such instead of "(optimum)", which
        # would misleadingly imply it was fitted there.
        return f"; third parameter pinned at {PARAM_DISPLAY_NAME[pinned]} = {median:.3g} (fixed, not learned)"
    slice_center = ds.attrs.get("slice_center", "optimum")
    if slice_center == "trained":
        val = median
    else:
        val = median * float(np.exp(ds["alpha_star"].values[ALPHA_INDEX[pinned]]))
    return f"; third parameter pinned at {PARAM_DISPLAY_NAME[pinned]} = {val:.3g} ({slice_center})"


def marker_points(ds, plane_name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ((trained_x, trained_y), (star_x, star_y)) for this plane."""
    if plane_name == "stiff-sloppy":
        coord_trained = ds["coord_trained"].values
        # Last real eigen-slot: 2 when all three components are active
        # (unchanged behavior), or the last ACTIVE index when one component
        # is fixed (its coord_trained slot at index 2 is NaN, not a real
        # sloppy coordinate -- see active_mask).
        last_k = int(active_mask(ds).sum()) - 1
        slice_center = ds.attrs.get("slice_center", "optimum")
        if slice_center == "trained":
            return (0.0, 0.0), (-float(coord_trained[0]), -float(coord_trained[last_k]))
        return (float(coord_trained[0]), float(coord_trained[last_k])), (0.0, 0.0)
    a_name, b_name = plane_name.split("-")
    i, j = ALPHA_INDEX[a_name], ALPHA_INDEX[b_name]
    alpha_star = ds["alpha_star"].values
    return (0.0, 0.0), (float(alpha_star[i]), float(alpha_star[j]))


def nearest_z(ga: np.ndarray, gb: np.ndarray, z: np.ndarray, x: float, y: float) -> float:
    ia = int(np.argmin(np.abs(ga - x)))
    ib = int(np.argmin(np.abs(gb - y)))
    return float(z[ia, ib])


def load_plane(ds, plane_idx: int, log: bool, zmax_arg: float | None):
    """Return dict with ga, gb, z (transformed+clipped), clamped, note, and the
    raw (untransformed) loss array, all at the plane's native grid resolution."""
    ga = ds["grid_axis_a"].values[plane_idx].astype(np.float64)
    gb = ds["grid_axis_b"].values[plane_idx].astype(np.float64)
    raw = ds["grid_loss"].values[plane_idx].astype(np.float64)
    clamped = np.nan_to_num(ds["grid_clamped"].values[plane_idx].astype(np.float64), nan=0.0)

    zmax_used = zmax_arg if zmax_arg is not None else float(np.nanpercentile(raw, 99))
    clipped = np.clip(raw, None, zmax_used)
    z = np.log10(np.clip(clipped, 1e-12, None)) if log else clipped

    note = f"z = {'log10(loss)' if log else 'loss'}; loss clipped at {zmax_used:.4g}"
    note += " (given)" if zmax_arg is not None else " (99th pct of grid)"
    return dict(ga=ga, gb=gb, z=z, clamped=clamped, note=note, raw=raw)


def upsample_for_display(ga, gb, z, clamped):
    n_a, n_b = z.shape
    if max(n_a, n_b) > UPSAMPLE_MAX_SIDE:
        return ga, gb, z, clamped, False
    ga_d = np.linspace(ga.min(), ga.max(), n_a * UPSAMPLE_FACTOR)
    gb_d = np.linspace(gb.min(), gb.max(), n_b * UPSAMPLE_FACTOR)
    z_d = zoom(z, UPSAMPLE_FACTOR, order=3)
    clamped_d = zoom(clamped, UPSAMPLE_FACTOR, order=0)
    return ga_d, gb_d, z_d, clamped_d, True


def shaded_facecolors(z_disp: np.ndarray, clamped_disp: np.ndarray) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=45)
    z_safe = np.nan_to_num(z_disp, nan=np.nanmax(z_disp))
    rgba = ls.shade(z_safe, cmap=cm.coolwarm, vert_exag=VERT_EXAG, blend_mode="soft")
    facecolors = rgba[:-1, :-1, :].copy()
    mask = clamped_disp[:-1, :-1] > CLAMP_THRESHOLD
    facecolors[mask] = 0.5 * facecolors[mask] + 0.5 * GREY
    return facecolors


def plot_mpl(ds, plane_name: str, plane: dict, out_png: Path, log: bool, alpha_axes: bool = False):
    staid = ds.attrs["staid"]
    loss0 = float(ds["loss0"].values)
    loss_star = float(ds["loss_star"].values)
    nse0 = float(ds["nse0"].values)
    nse_star = float(ds["nse_star"].values)

    ga, gb, z, clamped, note = plane["ga"], plane["gb"], plane["z"], plane["clamped"], plane["note"]
    ga_d, gb_d, z_d, clamped_d, upsampled = upsample_for_display(ga, gb, z, clamped)
    if upsampled:
        note = note + f"; display-interpolated x{UPSAMPLE_FACTOR}"
    note = note + pinned_third_note(ds, plane_name, alpha_axes)

    X, Y = np.meshgrid(ga_d, gb_d, indexing="ij")
    facecolors = shaded_facecolors(z_d, clamped_d)

    (tx, ty), (sx, sy) = marker_points(ds, plane_name)
    tz = nearest_z(ga, gb, z, tx, ty)
    sz = nearest_z(ga, gb, z, sx, sy)
    zfloor = float(np.nanmin(z_d))

    xlabel, ylabel, xlocs, xlabels, ylocs, ylabels = axis_ticks(ds, plane_name, ga, gb, alpha_axes)
    zlabel = "log10(loss)" if log else "loss"

    fig = plt.figure(figsize=(18, 8))
    views = [(35, -50), (20, 40)]
    for panel, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        ax.plot_surface(
            X, Y, z_d,
            facecolors=facecolors,
            rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False,
        )
        ax.plot([tx, tx], [ty, ty], [zfloor, tz], color="k", lw=2.0, zorder=10)
        ax.plot([tx], [ty], [tz], marker="x", color="k", ms=12, mew=3, zorder=11)
        ax.plot([sx, sx], [sy, sy], [zfloor, sz], color="red", lw=2.0, zorder=10)
        ax.plot([sx], [sy], [sz], marker="*", color="red", ms=16, zorder=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_zlabel(zlabel, fontsize=9)
        if xlocs:
            ax.set_xticks(xlocs)
            ax.set_xticklabels(xlabels, fontsize=7)
        if ylocs:
            ax.set_yticks(ylocs)
            ax.set_yticklabels(ylabels, fontsize=7)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=9)

    fig.suptitle(
        f"{staid}  {plane_name}  loss trained {loss0:.4f}  optimum {loss_star:.4f}  "
        f"(NSE {nse0:.3f} -> {nse_star:.3f})"
        f"{title_param_suffix(ds, plane_name, alpha_axes)}",
        fontsize=12,
    )
    fig.text(0.01, 0.01, note, fontsize=7, ha="left", va="bottom")
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.02, right=0.98, wspace=0.05)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_plotly(ds, plane_name: str, plane: dict, out_html: Path, log: bool, alpha_axes: bool = False):
    staid = ds.attrs["staid"]
    loss0 = float(ds["loss0"].values)
    loss_star = float(ds["loss_star"].values)
    nse0 = float(ds["nse0"].values)
    nse_star = float(ds["nse_star"].values)

    ga, gb, z, clamped, note = plane["ga"], plane["gb"], plane["z"], plane["clamped"], plane["note"]
    note = note + pinned_third_note(ds, plane_name, alpha_axes)
    (tx, ty), (sx, sy) = marker_points(ds, plane_name)
    tz = nearest_z(ga, gb, z, tx, ty)
    sz = nearest_z(ga, gb, z, sx, sy)
    zfloor = float(np.nanmin(z))

    xlabel, ylabel, xlocs, xlabels, ylocs, ylabels = axis_ticks(ds, plane_name, ga, gb, alpha_axes)
    zlabel = "log10(loss)" if log else "loss"

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=ga, y=gb, z=z.T, colorscale="RdBu_r",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.6),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            colorbar=dict(title=zlabel),
        )
    )
    for (x, y, ztop, color, symbol, name) in [
        (tx, ty, tz, "black", "x", "trained (alpha=0)"),
        (sx, sy, sz, "red", "diamond", "optimum (alpha*)"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[x, x], y=[y, y], z=[zfloor, ztop], mode="lines",
            line=dict(color=color, width=6), showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[ztop], mode="markers",
            marker=dict(symbol=symbol, size=6, color=color), name=name,
        ))

    xaxis = dict(title=xlabel)
    if xlocs:
        xaxis.update(tickvals=xlocs, ticktext=xlabels)
    yaxis = dict(title=ylabel)
    if ylocs:
        yaxis.update(tickvals=ylocs, ticktext=ylabels)

    fig.update_layout(
        title=(
            f"{staid}  {plane_name}  loss trained {loss0:.4f}  optimum {loss_star:.4f}  "
            f"(NSE {nse0:.3f} -> {nse_star:.3f})"
            f"{title_param_suffix(ds, plane_name, alpha_axes)}<br><sup>{note}</sup>"
        ),
        scene=dict(
            xaxis=xaxis, yaxis=yaxis, zaxis_title=zlabel,
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6),
        ),
    )
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gauge_nc", type=Path)
    ap.add_argument("--plane", default="all", help="plane name (e.g. n-p) or 'all'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", action="store_true", help="plot z = log10(loss) instead of loss")
    ap.add_argument("--zmax", type=float, default=None, help="clip loss (raw, not log) at this value instead of the 99th percentile")
    ap.add_argument("--alpha-axes", action="store_true", help="label axis-aligned planes in ln multiplier (alpha) space instead of physical parameter values")
    args = ap.parse_args()

    ds = xr.open_dataset(args.gauge_nc, decode_timedelta=False)
    staid = ds.attrs["staid"]
    plane_names = ds.attrs["plane_names"].split(",")
    if not plane_names or plane_names == [""]:
        raise SystemExit(f"{args.gauge_nc} has no grid slices (grid: 0 study)")

    if args.plane == "all":
        selected = list(enumerate(plane_names))
    else:
        if args.plane not in plane_names:
            raise SystemExit(f"unknown plane {args.plane!r}; available: {plane_names}")
        selected = [(plane_names.index(args.plane), args.plane)]

    args.out.mkdir(parents=True, exist_ok=True)
    for plane_idx, plane_name in selected:
        plane = load_plane(ds, plane_idx, args.log, args.zmax)
        out_png = args.out / f"surface_{staid}_{plane_name}.png"
        out_html = args.out / f"surface_{staid}_{plane_name}.html"
        plot_mpl(ds, plane_name, plane, out_png, args.log, args.alpha_axes)
        print(f"wrote {out_png}")
        plot_plotly(ds, plane_name, plane, out_html, args.log, args.alpha_axes)
        print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
