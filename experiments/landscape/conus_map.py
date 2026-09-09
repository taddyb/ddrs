#!/usr/bin/env python
"""CONUS map of trained-vs-optimal Manning n, per gauge, from a landscape
census run.

For each gauge, alpha_n_star = ln(n_optimal / n_trained) (a basin-uniform
log-multiplier on the trained n field, from the census run's 2-D/3-D Newton
search for that gauge's own optimum). This script plots |alpha_n_star| (how
far training left the gauge from its own optimum) and signed alpha_n_star
(which way it wants to move) at the gauge's lat/lon, so the user can judge at
a glance where training is doing well (blue, near the optimum) vs poorly (red,
far from it).

Reads <census_run_dir>/<arm>/summary.csv, <census_run_dir>/manifest.json (to
find the arm's run_dir -> config.yaml -> data_sources.gages, and one gauge
netCDF for the active_params attribute), and the gages_3000-style CSV named
there (STAID, LAT_GAGE, LNG_GAGE, DRAIN_SQKM). Writes conus_n_gap.png and
conus_n_gap.csv under --out.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/conus_map.py <census_run_dir> \\
        [--arm <name>] [--out <census_run_dir>/figures] [--nse-min 0.3]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402

LN3 = math.log(3.0)


# ----------------------------------------------------------------------------- io
def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def pick_arm(manifest: dict, arm: str | None) -> dict:
    arms = manifest["arms"]
    if arm is None:
        if len(arms) != 1:
            names = [a["name"] for a in arms]
            raise SystemExit(f"run has {len(arms)} arms {names}; pass --arm to pick one")
        return arms[0]
    for a in arms:
        if a["name"] == arm:
            return a
    names = [a["name"] for a in arms]
    raise SystemExit(f"arm {arm!r} not found; run has arms {names}")


def load_gage_csv(run_dir: Path, arm_entry: dict) -> pd.DataFrame:
    # manifest paths are recorded relative to the ddrs repo root (cwd at run
    # time), not relative to the census run_dir -- resolve from cwd first.
    candidates = [Path(arm_entry["config_path"]), Path.cwd() / arm_entry["config_path"]]
    config_path = next((p for p in candidates if p.exists()), candidates[0])
    config = yaml.safe_load(config_path.read_text())
    gages_path = Path(config["data_sources"]["gages"])
    df = pd.read_csv(gages_path, dtype={"STAID": str})
    df["STAID"] = df["STAID"].str.zfill(8)
    return df[["STAID", "LAT_GAGE", "LNG_GAGE", "DRAIN_SQKM"]].drop_duplicates("STAID")


def active_p_dim(run_dir: Path, arm: str) -> str:
    """'2-D' if every gauge netCDF for this arm has p inactive, else '3-D'.
    Reads one netCDF's active_params attr (spec: read one, they're uniform
    per training config)."""
    files = sorted(glob.glob(str(run_dir / arm / "gauges" / "*.nc")))
    if not files:
        return "3-D"
    ds = xr.open_dataset(files[0], decode_timedelta=False)
    active_params = ds.attrs.get("active_params", "n,p_spatial,q_spatial")
    return "2-D" if "p_spatial" not in active_params.split(",") else "3-D"


def window_info(run_dir: Path, arm: str) -> tuple[int | None, float | None]:
    """Reads window_days (attr) and window_start_day[0] (var) from one gauge
    netCDF (spec: uniform per training config, like active_p_dim)."""
    files = sorted(glob.glob(str(run_dir / arm / "gauges" / "*.nc")))
    if not files:
        return None, None
    ds = xr.open_dataset(files[0], decode_timedelta=False)
    window_days = ds.attrs.get("window_days")
    window_start_day = None
    if "window_start_day" in ds.variables and ds["window_start_day"].size:
        window_start_day = float(ds["window_start_day"].values[0])
    return (int(window_days) if window_days is not None else None), window_start_day


# ------------------------------------------------------------------------ build
def build_df(
    run_dir: Path, arm: str, gage_csv: pd.DataFrame, nse_min: float, window_days: int | None
) -> pd.DataFrame:
    summary = pd.read_csv(run_dir / arm / "summary.csv", dtype={"staid": str})
    summary["staid"] = summary["staid"].str.zfill(8)
    df = summary.merge(gage_csv, left_on="staid", right_on="STAID", how="inner")
    missing = set(summary["staid"]) - set(df["staid"])
    if missing:
        print(f"warning: {len(missing)} gauge(s) in summary.csv have no match in the gage CSV: {sorted(missing)}")
    df["well_fit"] = df["nse0"] >= nse_min
    out = pd.DataFrame({
        "staid": df["staid"],
        "lat": df["LAT_GAGE"],
        "lon": df["LNG_GAGE"],
        "area_km2": df["DRAIN_SQKM"],
        "n_reach": df["n_reach"],
        "nse0": df["nse0"],
        "nse_star": df["nse_star"],
        "gain": df["nse_star"] - df["nse0"],
        "alpha_n_star": df["alpha_n_star"],
        "mult_n": df["mult_n"],
        "hit_range_bound": df["hit_range_bound"],
        "well_fit": df["well_fit"],
        "window_days": window_days,
    })
    return out


# ----------------------------------------------------------------------- figure
def marker_sizes(area_km2: np.ndarray) -> np.ndarray:
    area = np.where(np.isfinite(area_km2) & (area_km2 > 0), area_km2, np.nan)
    med = np.nanmedian(area) if np.isfinite(np.nanmedian(area)) else 1.0
    return 18.0 * np.sqrt(area / med) if np.isfinite(med) and med > 0 else np.full(len(area_km2), 18.0)


def draw_frame(ax):
    """Plain CONUS lon/lat frame with light gridlines -- used when no offline
    state-boundary source is available on this machine."""
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    # Auto (not equal-degree) aspect: fills the panel instead of letterboxing
    # a wide/short CONUS extent inside a taller subplot box. No basemap is
    # drawn, so the mild east-west stretch costs nothing but a bit of shape
    # accuracy the reader isn't using anyway.
    ax.set_aspect("auto")
    ax.grid(True, color="0.85", lw=0.5, zorder=0)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.text(
        0.02, 0.02, "no offline state-boundary layer found; plain lon/lat frame",
        transform=ax.transAxes, fontsize=7, color="0.4", ha="left", va="bottom",
    )


def alpha_colorbar_ticks():
    """Tick positions/labels at multiplier factors 1, 1.5, 2, 3 on the
    |ln(n_opt/n_trained)| axis (ticks placed at ln(factor))."""
    factors = [1, 1.5, 2, 3]
    return [math.log(f) for f in factors], factors


def fig_conus(
    out: Path,
    df: pd.DataFrame,
    arm: str,
    run_id: str,
    dim_note: str,
    title_suffix: str | None,
    window_days: int | None,
    window_start_day: float | None,
) -> Path:
    sizes = marker_sizes(df["area_km2"].to_numpy(dtype=float))
    df = df.assign(_size=sizes)
    well = df[df["well_fit"]]
    poor = df[~df["well_fit"]]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(27, 7), dpi=150)

    # ---- panel (a): distance from optimum
    cmap_a = plt.get_cmap("RdBu_r")
    for ax in (ax_a, ax_b, ax_c):
        draw_frame(ax)

    poor_a = poor
    if not poor_a.empty:
        ax_a.scatter(poor_a["lon"], poor_a["lat"], s=poor_a["_size"], facecolors="none",
                     edgecolors="0.6", linewidths=0.8, zorder=3)
    bound = well[well["hit_range_bound"] == 1]
    unbound = well[well["hit_range_bound"] != 1]
    sc_a = ax_a.scatter(unbound["lon"], unbound["lat"], s=unbound["_size"],
                         c=unbound["alpha_n_star"].abs(), cmap=cmap_a, vmin=0, vmax=LN3,
                         edgecolors="none", zorder=5)
    ax_a.scatter(bound["lon"], bound["lat"], s=bound["_size"],
                 c=bound["alpha_n_star"].abs(), cmap=cmap_a, vmin=0, vmax=LN3,
                 edgecolors="k", linewidths=0.8, zorder=6)
    cbar_a = fig.colorbar(sc_a, ax=ax_a, shrink=0.85, pad=0.02)
    cbar_a.set_label(
        "|ln(n_optimal / n_trained)|\n(0 = trained at optimum; ln 3 = factor 3 off)",
        fontsize=8,
    )
    tick_pos, tick_lab = alpha_colorbar_ticks()
    cbar_a.set_ticks(tick_pos)
    cbar_a.set_ticklabels([f"{f:g}x" for f in tick_lab])
    ax_a.set_title("(a) how far trained n is from the gauge optimum")

    # ---- panel (b): direction
    cmap_b = plt.get_cmap("coolwarm")
    poor_b = poor
    if not poor_b.empty:
        ax_b.scatter(poor_b["lon"], poor_b["lat"], s=poor_b["_size"], facecolors="none",
                     edgecolors="0.6", linewidths=0.8, zorder=3)
    bound_b = well[well["hit_range_bound"] == 1]
    unbound_b = well[well["hit_range_bound"] != 1]
    sc_b = ax_b.scatter(unbound_b["lon"], unbound_b["lat"], s=unbound_b["_size"],
                         c=unbound_b["alpha_n_star"], cmap=cmap_b, vmin=-LN3, vmax=LN3,
                         edgecolors="none", zorder=5)
    ax_b.scatter(bound_b["lon"], bound_b["lat"], s=bound_b["_size"],
                 c=bound_b["alpha_n_star"], cmap=cmap_b, vmin=-LN3, vmax=LN3,
                 edgecolors="k", linewidths=0.8, zorder=6)
    cbar_b = fig.colorbar(sc_b, ax=ax_b, shrink=0.85, pad=0.02)
    cbar_b.set_label(
        "signed ln(n_optimal / n_trained)\n(blue = wants higher n / slower; red = wants lower n / faster)",
        fontsize=8,
    )
    ax_b.set_title("(b) direction the gauge wants to move n")

    # ---- panel (c): actionable gain at the gauge's own optimum. Panel (a)
    # says how far n is from the optimum, but where the landscape is flat a
    # large distance costs nothing -- (c) is the "how well is training doing"
    # view: NSE the gauge could still gain by moving to alpha_star.
    cmap_c = plt.get_cmap("RdBu_r")
    poor_c = poor
    if not poor_c.empty:
        ax_c.scatter(poor_c["lon"], poor_c["lat"], s=poor_c["_size"], facecolors="none",
                     edgecolors="0.6", linewidths=0.8, zorder=3)
    bound_c = well[well["hit_range_bound"] == 1]
    unbound_c = well[well["hit_range_bound"] != 1]
    sc_c = ax_c.scatter(unbound_c["lon"], unbound_c["lat"], s=unbound_c["_size"],
                         c=unbound_c["gain"], cmap=cmap_c, vmin=0, vmax=0.10,
                         edgecolors="none", zorder=5)
    ax_c.scatter(bound_c["lon"], bound_c["lat"], s=bound_c["_size"],
                 c=bound_c["gain"], cmap=cmap_c, vmin=0, vmax=0.10,
                 edgecolors="k", linewidths=0.8, zorder=6)
    cbar_c = fig.colorbar(sc_c, ax=ax_c, shrink=0.85, pad=0.02)
    cbar_c.set_label("NSE gain available at the gauge optimum (0 = none)", fontsize=8)
    ax_c.set_title("(c) NSE the gauge could gain at its own optimum")

    n_gauges = len(df)
    n_well = int(well.shape[0])
    med_abs = float(np.median(well["alpha_n_star"].abs())) if n_well else float("nan")
    frac_1p25 = float((well["alpha_n_star"].abs() <= math.log(1.25)).mean()) if n_well else float("nan")
    med_gain = float(np.median(well["gain"])) if n_well else float("nan")
    frac_gain = float((well["gain"] > 0.02).mean()) if n_well else float("nan")
    frac_slower = float((well["mult_n"] > 1.25).mean()) if n_well else float("nan")
    frac_faster = float((well["mult_n"] < 0.8).mean()) if n_well else float("nan")
    window_note = ""
    if window_days is not None:
        window_note = f"; window {window_days} d"
        if window_start_day is not None:
            window_note += f" starting at day {window_start_day:g}"
    title = (
        f"arm={arm}  run={run_id}"
        + (f"  ({title_suffix})" if title_suffix else "")
        + "\n"
        f"N gauges={n_gauges}  N well-fit (nse0>=nse_min)={n_well}  "
        f"median |ln(n_opt/n_trained)| (well-fit)={med_abs:.3f}  "
        f"share within factor 1.25={frac_1p25:.1%}\n"
        f"median NSE gain (well-fit)={med_gain:.3f}  share gain>0.02={frac_gain:.1%}  "
        f"share wanting slower (mult_n>1.25)={frac_slower:.1%}  "
        f"share wanting faster (mult_n<0.8)={frac_faster:.1%}\n"
        f"corner note: optimum = {dim_note} Newton over basin-uniform "
        f"{'(n, q)' if dim_note == '2-D' else '(n, p, q)'} multipliers"
        + (", p fixed" if dim_note == "2-D" else "")
        + window_note
    )
    fig.suptitle(title, fontsize=10)
    fig.text(
        0.5, 0.005,
        f"hollow grey = nse0 < nse_min (optimum not meaningful, n={int((~df['well_fit']).sum())}); "
        f"black edge = range-bound (optimum is a bound, n={int((df['hit_range_bound'] == 1).sum())}); "
        f"marker size ~ sqrt(drainage area)",
        ha="center", fontsize=8, color="0.3",
    )
    # subplots_adjust (not tight_layout) here: tight_layout's colorbar
    # accounting leaves a large unused vertical band above the axes for this
    # wide/short CONUS extent -- fixed margins fill the figure correctly.
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.025, right=0.99, wspace=0.30)
    p = out / "conus_n_gap.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census_run_dir", type=Path)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--nse-min", type=float, default=0.3)
    ap.add_argument("--title-suffix", default=None, help='e.g. "WY2000, 365 d", appended to the header')
    args = ap.parse_args()

    run_dir = args.census_run_dir
    out = args.out if args.out is not None else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_dir)
    arm_entry = pick_arm(manifest, args.arm)
    arm = arm_entry["name"]
    run_id = arm_entry["run_id"]

    gage_csv = load_gage_csv(run_dir, arm_entry)
    window_days, window_start_day = window_info(run_dir, arm)
    df = build_df(run_dir, arm, gage_csv, args.nse_min, window_days)
    dim_note = active_p_dim(run_dir, arm)

    csv_path = out / "conus_n_gap.csv"
    df.to_csv(csv_path, index=False)
    png_path = fig_conus(
        out, df, arm, run_id, dim_note, args.title_suffix, window_days, window_start_day
    )

    n_well = int(df["well_fit"].sum())
    med_abs = float(np.median(df.loc[df["well_fit"], "alpha_n_star"].abs())) if n_well else float("nan")
    frac_1p25 = float((df.loc[df["well_fit"], "alpha_n_star"].abs() <= math.log(1.25)).mean()) if n_well else float("nan")
    med_gain = float(np.median(df.loc[df["well_fit"], "gain"])) if n_well else float("nan")
    frac_gain = float((df.loc[df["well_fit"], "gain"] > 0.02).mean()) if n_well else float("nan")
    frac_slower = float((df.loc[df["well_fit"], "mult_n"] > 1.25).mean()) if n_well else float("nan")
    frac_faster = float((df.loc[df["well_fit"], "mult_n"] < 0.8).mean()) if n_well else float("nan")
    print(f"arm={arm} run={run_id}")
    print(f"N gauges={len(df)}  N well-fit(nse0>={args.nse_min})={n_well}")
    print(f"median |ln(n_opt/n_trained)| over well-fit gauges = {med_abs:.4f}")
    print(f"share of well-fit gauges within factor 1.25 of optimum = {frac_1p25:.1%}")
    print(f"median NSE gain over well-fit gauges = {med_gain:.4f}")
    print(f"share of well-fit gauges with gain > 0.02 = {frac_gain:.1%}")
    print(f"share of well-fit gauges wanting slower (mult_n>1.25) = {frac_slower:.1%}")
    print(f"share of well-fit gauges wanting faster (mult_n<0.8) = {frac_faster:.1%}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
