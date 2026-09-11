#!/usr/bin/env python
"""Per-gauge landscape report: (n, q) loss terrain next to observed vs.
routed hydrographs, for every gauge netCDF in every arm of a `landscape`
study run.

Reads `<run_dir>/<arm>/gauges/<staid>.nc` (written by
`src/experiment/landscape/mod.rs` / `output.rs`). Requires
`landscape.series: true` (the `obs_daily`/`routed_daily_*`/
`summed_qprime_daily` variables) and the "n-q" plane in `landscape.planes`;
gauges missing either are skipped with a warning.

Accepts either one merged run directory (`scripts/landscape_merge.py`
output) or a list of unmerged shard run directories from the same bundle —
in the latter case gauge netCDFs are unioned across shards (first occurrence
wins on a duplicate staid, same rule as `landscape_merge.py`).

For each (arm, staid):
  - report_<staid>.png (or <out>/<arm>/report_<staid>.png when more than one
    arm is present, to avoid collisions): left, the n-q loss terrain exactly
    as experiments/landscape/surface.py draws it (physical axes, trained x
    and optimum star markers); right, two stacked daily hydrograph panels —
    the whole window, and a 90-day zoom on the window's wettest 90 days by
    observed volume.
  - REPORT.md: one row per (arm, staid) with nse0/nse_star, kge0/kge_star,
    window length, and the optimum's physical multipliers (mult_n, mult_p,
    mult_q = exp(alpha_star)).
  - report_overview.png: the wettest-90-day zoom for every (arm, staid) in
    one grid figure.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/gauge_report.py \\
        <run_dir> [<run_dir_2> ...] [--out <run_dir>/figures]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surface  # noqa: E402

ZOOM_DAYS = 90
COLOR_OBS = "black"
COLOR_SUMMED = "grey"
COLOR_TRAINED = "tab:blue"
COLOR_STAR = "red"


# ------------------------------------------------------------------- discovery
def discover_netcdfs(run_dirs: list[Path]) -> dict[str, dict[str, Path]]:
    """{arm: {staid: path}} unioned across run_dirs, first occurrence wins on
    a duplicate staid (mirrors scripts/landscape_merge.py's rule)."""
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    arm_names = [a["name"] for a in manifest["arms"]]
    out: dict[str, dict[str, Path]] = {arm: {} for arm in arm_names}
    for d in run_dirs:
        for arm in arm_names:
            gauges_dir = d / arm / "gauges"
            if not gauges_dir.is_dir():
                continue
            for nc in sorted(gauges_dir.glob("*.nc")):
                staid = nc.stem
                if staid in out[arm]:
                    print(f"warning: staid {staid!r} appears in more than one shard for arm {arm!r}; keeping the first occurrence")
                    continue
                out[arm][staid] = nc
    return out


def has_series(ds: xr.Dataset) -> bool:
    return "obs_daily" in ds.variables and "day" in ds.dims


def has_nq_plane(ds: xr.Dataset) -> bool:
    plane_names = ds.attrs.get("plane_names", "").split(",")
    return "n-q" in plane_names


# --------------------------------------------------------------------- series
def series_dates(ds: xr.Dataset) -> pd.DatetimeIndex:
    axis_start = pd.Timestamp(ds.attrs["axis_start_date"])
    window_start_day = int(ds["window_start_day"].values[0])
    n_days = ds.sizes["day"]
    return axis_start + pd.to_timedelta(window_start_day + np.arange(n_days), unit="D")


def wettest_zoom_slice(obs: np.ndarray, zoom_days: int) -> slice:
    """Index slice of length min(zoom_days, len(obs)) with the largest sum of
    observed volume (NaN treated as 0 for ranking only)."""
    n = len(obs)
    w = min(zoom_days, n)
    filled = np.nan_to_num(obs, nan=0.0)
    roll = pd.Series(filled).rolling(w, min_periods=1).sum()
    end = int(roll.iloc[w - 1 :].idxmax())
    start = max(0, end - w + 1)
    return slice(start, start + w)


def plot_hydro_panel(ax, dates, obs, summed, trained, star, title: str, legend: bool):
    ax.plot(dates, summed, color=COLOR_SUMMED, lw=1.0, label="summed Q' (no routing)")
    ax.plot(dates, trained, color=COLOR_TRAINED, lw=1.1, label="routed (trained)")
    ax.plot(dates, star, color=COLOR_STAR, lw=1.1, ls="--", label="routed (optimum)")
    ax.plot(dates, obs, color=COLOR_OBS, lw=1.1, label="observed")
    ax.set_ylabel("discharge (m3/s)")
    ax.set_title(title, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    if legend:
        ax.legend(fontsize=7, ncol=4, loc="upper right")


# --------------------------------------------------------------------- terrain
def plot_terrain_panel(ax, ds: xr.Dataset):
    """The n-q loss terrain, single view, drawn with the same functions and
    conventions as surface.py's plot_mpl (physical axes, trained x / optimum
    star markers)."""
    plane_names = ds.attrs["plane_names"].split(",")
    plane_idx = plane_names.index("n-q")
    plane = surface.load_plane(ds, plane_idx, log=False, zmax_arg=None)
    ga, gb, z, clamped, note = plane["ga"], plane["gb"], plane["z"], plane["clamped"], plane["note"]
    ga_d, gb_d, z_d, clamped_d, upsampled = surface.upsample_for_display(ga, gb, z, clamped)
    X, Y = np.meshgrid(ga_d, gb_d, indexing="ij")
    facecolors = surface.shaded_facecolors(z_d, clamped_d)

    (tx, ty), (sx, sy) = surface.marker_points(ds, "n-q")
    tz = surface.nearest_z(ga, gb, z, tx, ty)
    sz = surface.nearest_z(ga, gb, z, sx, sy)
    zfloor = float(np.nanmin(z_d))

    xlabel, ylabel, xlocs, xlabels, ylocs, ylabels = surface.axis_ticks(ds, "n-q", ga, gb, alpha_axes=False)

    ax.plot_surface(X, Y, z_d, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax.plot([tx, tx], [ty, ty], [zfloor, tz], color="k", lw=2.0, zorder=10)
    ax.plot([tx], [ty], [tz], marker="x", color="k", ms=12, mew=3, zorder=11)
    ax.plot([sx, sx], [sy, sy], [zfloor, sz], color="red", lw=2.0, zorder=10)
    ax.plot([sx], [sy], [sz], marker="*", color="red", ms=16, zorder=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_zlabel("loss", fontsize=9)
    if xlocs:
        ax.set_xticks(xlocs)
        ax.set_xticklabels(xlabels, fontsize=7)
    if ylocs:
        ax.set_yticks(ylocs)
        ax.set_yticklabels(ylabels, fontsize=7)
    ax.view_init(elev=35, azim=-50)
    title = "n-q loss terrain" + surface.title_param_suffix(ds, "n-q", False)
    ax.set_title(title, fontsize=10)
    note = note + surface.pinned_third_note(ds, "n-q", False)
    return note


# ---------------------------------------------------------------------- report
def gauge_stats(ds: xr.Dataset) -> dict:
    alpha_star = ds["alpha_star"].values.astype(float)
    mult = np.exp(alpha_star)
    return dict(
        staid=ds.attrs["staid"],
        arm=ds.attrs["arm"],
        objective=ds.attrs["objective"],
        window_days=int(ds.attrs["window_days"]),
        nse0=float(ds["nse0"].values),
        nse_star=float(ds["nse_star"].values),
        kge0=float(ds["kge0"].values),
        kge_star=float(ds["kge_star"].values),
        mult_n=float(mult[0]),
        mult_p=float(mult[1]),
        mult_q=float(mult[2]),
        hit_range_bound=bool(ds.attrs.get("hit_range_bound", 0)),
    )


def fig_gauge_report(ds: xr.Dataset, out_png: Path) -> dict:
    stats = gauge_stats(ds)
    dates = series_dates(ds)
    obs = ds["obs_daily"].values
    trained = ds["routed_daily_trained"].values
    star = ds["routed_daily_star"].values
    summed = ds["summed_qprime_daily"].values
    zoom = wettest_zoom_slice(obs, ZOOM_DAYS)
    zoom_len = zoom.stop - zoom.start

    fig = plt.figure(figsize=(18, 10))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18, left=0.03, right=0.98, top=0.86, bottom=0.08)
    ax3d = fig.add_subplot(outer[0, 0], projection="3d")
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    ax_full = fig.add_subplot(right[0, 0])
    ax_zoom = fig.add_subplot(right[1, 0])

    note = plot_terrain_panel(ax3d, ds)
    plot_hydro_panel(ax_full, dates, obs, summed, trained, star, f"full window ({len(dates)} d)", legend=True)
    plot_hydro_panel(
        ax_zoom, dates[zoom], obs[zoom], summed[zoom], trained[zoom], star[zoom],
        f"{zoom_len}-day zoom (wettest by observed volume)", legend=False,
    )

    bound_note = "  [hit range bound: q* pinned at a domain edge]" if stats["hit_range_bound"] else ""
    fig.suptitle(
        f"{stats['staid']}   arm={stats['arm']}   objective={stats['objective']}   window={stats['window_days']} d{bound_note}\n"
        f"NSE {stats['nse0']:.3f} -> {stats['nse_star']:.3f}     "
        f"KGE {stats['kge0']:.3f} -> {stats['kge_star']:.3f}     "
        f"mult (n, p, q) = ({stats['mult_n']:.3g}, {stats['mult_p']:.3g}, {stats['mult_q']:.3g})",
        fontsize=13,
    )
    fig.text(0.01, 0.01, note, fontsize=7, ha="left", va="bottom")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return stats


def fig_overview(entries: list[tuple[dict, xr.Dataset]], out_png: Path) -> None:
    n = len(entries)
    if n == 0:
        return
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows), squeeze=False)
    for idx, (stats, ds) in enumerate(entries):
        ax = axes[idx // cols][idx % cols]
        dates = series_dates(ds)
        obs = ds["obs_daily"].values
        trained = ds["routed_daily_trained"].values
        star = ds["routed_daily_star"].values
        summed = ds["summed_qprime_daily"].values
        zoom = wettest_zoom_slice(obs, ZOOM_DAYS)
        title = f"{stats['staid']} ({stats['arm']})  NSE {stats['nse0']:.2f}->{stats['nse_star']:.2f}"
        plot_hydro_panel(ax, dates[zoom], obs[zoom], summed[zoom], trained[zoom], star[zoom], title, legend=(idx == 0))
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"wettest {ZOOM_DAYS}-day zoom, all gauges", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_report_md(rows: list[dict], run_dirs: list[Path], out_dir: Path) -> None:
    cols = ["staid", "arm", "objective", "window_days", "nse0", "nse_star", "kge0", "kge_star", "mult_n", "mult_p", "mult_q"]
    lines = ["# Gauge landscape report", ""]
    lines.append("Source run(s): " + ", ".join(str(d) for d in run_dirs))
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'REPORT.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path, help="merged run dir, or a list of unmerged shard run dirs from the same bundle")
    ap.add_argument("--out", type=Path, default=None, help="default: <first run_dir>/figures")
    args = ap.parse_args()

    run_dirs = [d.resolve() for d in args.run_dirs]
    out = (args.out or (run_dirs[0] / "figures")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    by_arm = discover_netcdfs(run_dirs)
    arms = [a for a in by_arm if by_arm[a]]
    single_arm = len(arms) == 1

    all_rows: list[dict] = []
    overview_entries: list[tuple[dict, xr.Dataset]] = []
    for arm in arms:
        arm_out = out if single_arm else out / arm
        arm_out.mkdir(parents=True, exist_ok=True)
        for staid, path in sorted(by_arm[arm].items()):
            ds = xr.open_dataset(path, decode_timedelta=False)
            if not has_series(ds):
                print(f"skipping {arm}/{staid}: no daily series in {path} (landscape.series: true required)")
                continue
            if not has_nq_plane(ds):
                print(f"skipping {arm}/{staid}: no n-q plane in {path} (landscape.planes must include \"n-q\")")
                continue
            out_png = arm_out / f"report_{staid}.png"
            stats = fig_gauge_report(ds, out_png)
            print(f"wrote {out_png}")
            all_rows.append(stats)
            overview_entries.append((stats, ds))

    if not all_rows:
        raise SystemExit("no gauges had both series and the n-q plane; nothing to report")

    write_report_md(all_rows, run_dirs, out)
    if single_arm:
        fig_overview(overview_entries, out / "report_overview.png")
        print(f"wrote {out / 'report_overview.png'}")
    else:
        for arm in arms:
            entries = [(s, d) for s, d in overview_entries if s["arm"] == arm]
            if entries:
                fig_overview(entries, out / arm / "report_overview.png")
                print(f"wrote {out / arm / 'report_overview.png'}")


if __name__ == "__main__":
    main()
