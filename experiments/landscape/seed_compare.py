#!/usr/bin/env python
"""Compare two trained-seed arms of the alpha-landscape study (src/experiment/landscape).

Each seed's alpha is relative to its OWN trained per-reach fields (n0, p0, q0),
so this script places the "other" seed's trained point inside the "reference"
seed's landscape by first measuring, per reach, how far the other seed's
trained fields sit from the reference seed's trained fields
(d_i = ln(field_other_i / field_ref_i)), averaging that basin-uniformly into a
log-multiplier vector Delta-alpha, and then projecting Delta-alpha into the
reference seed's eigenbasis at its own optimum.

Reads only <run_dir>/gauges.csv and <run_dir>/<arm>/gauges/<staid>.nc for the
two named arms; writes <out>/SEED_COMPARE.md, <out>/seed_compare.csv, and one
<out>/seed_compare_<staid>.png per gauge present in both arms.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/seed_compare.py <run_dir> \\
        --ref uh-seed42 --other uh-seed43 [--out <run_dir>/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

COMPONENTS = ["n", "p", "q"]
FIELD_VARS = ["n0", "p0", "q0"]


# ----------------------------------------------------------------------------- io
def load_gauge(run_dir: Path, arm: str, staid: str) -> xr.Dataset | None:
    p = run_dir / arm / "gauges" / f"{staid}.nc"
    if not p.exists():
        return None
    return xr.open_dataset(p, decode_timedelta=False)


def align_by_comid(ds_ref: xr.Dataset, ds_other: xr.Dataset):
    """Return (n_common, idx_ref, idx_other) pairing reaches by comid."""
    comid_ref = ds_ref["comid"].values
    comid_other = ds_other["comid"].values
    common, idx_ref, idx_other = np.intersect1d(comid_ref, comid_other, return_indices=True)
    return common, idx_ref, idx_other


def tol_index(ds: xr.Dataset, tol_target: float) -> int:
    tolerances = ds["tolerances"].values
    return int(np.argmin(np.abs(tolerances - tol_target)))


# --------------------------------------------------------------------- per-gauge
def compare_gauge(run_dir: Path, ref_arm: str, other_arm: str, staid: str, tol_target: float = 0.05):
    ds_ref = load_gauge(run_dir, ref_arm, staid)
    ds_other = load_gauge(run_dir, other_arm, staid)
    if ds_ref is None or ds_other is None:
        return None

    common, idx_ref, idx_other = align_by_comid(ds_ref, ds_other)
    n_ref = int(ds_ref.sizes["reach"])
    n_other = int(ds_other.sizes["reach"])
    n_common = int(len(common))

    # 1. per-reach log ratio of trained fields, basin-uniform mean and spread
    d_per_reach = {}
    for var in FIELD_VARS:
        ref_v = ds_ref[var].values[idx_ref]
        other_v = ds_other[var].values[idx_other]
        d_per_reach[var] = np.log(other_v / ref_v)
    delta_alpha = np.array([d_per_reach[v].mean() for v in FIELD_VARS])
    delta_alpha_std = np.array([d_per_reach[v].std() for v in FIELD_VARS])

    # 2. other seed's trained point in the reference eigenbasis at the reference optimum
    alpha_star_ref = ds_ref["alpha_star"].values
    eigvec_star_ref = ds_ref["eigvec_star"].values  # [component, k]
    coord_trained_ref = ds_ref["coord_trained"].values  # ref's own trained point, c_k
    ti = tol_index(ds_ref, tol_target)
    half_width = ds_ref["half_width"].values[ti]  # [k]
    tol_used = float(ds_ref["tolerances"].values[ti])

    x = delta_alpha - alpha_star_ref
    coord_other = np.einsum("ik,i->k", eigvec_star_ref, x)  # c_k(other)

    ratio_ref = np.abs(coord_trained_ref) / half_width
    ratio_other = np.abs(coord_other) / half_width

    # 3. the two seeds' physical optima
    alpha_star_other = ds_other["alpha_star"].values
    ln_mult_star = delta_alpha + alpha_star_other - alpha_star_ref
    mult_star = np.exp(ln_mult_star)

    row = {
        "staid": staid,
        "n_reach_ref": n_ref,
        "n_reach_other": n_other,
        "n_common": n_common,
    }
    for k, comp in enumerate(COMPONENTS):
        row[f"delta_alpha_{comp}"] = float(delta_alpha[k])
        row[f"delta_alpha_std_{comp}"] = float(delta_alpha_std[k])
    for k in range(3):
        row[f"c{k + 1}_ref"] = float(coord_trained_ref[k])
        row[f"c{k + 1}_other"] = float(coord_other[k])
        row[f"hw{k + 1}"] = float(half_width[k])
        row[f"ratio{k + 1}_ref"] = float(ratio_ref[k])
        row[f"ratio{k + 1}_other"] = float(ratio_other[k])
    for comp in COMPONENTS:
        row[f"mult_star_{comp}"] = float(mult_star[COMPONENTS.index(comp)])
    row["nse_star_ref"] = float(ds_ref["nse_star"].values)
    row["nse_star_other"] = float(ds_other["nse_star"].values)
    row["loss_star_ref"] = float(ds_ref["loss_star"].values)
    row["loss_star_other"] = float(ds_other["loss_star"].values)
    row["clamped_frac_star_ref"] = float(ds_ref.attrs["clamped_frac_star"])
    row["clamped_frac_star_other"] = float(ds_other.attrs["clamped_frac_star"])
    row["tol_used"] = tol_used

    extra = {
        "ds_ref": ds_ref,
        "ds_other": ds_other,
        "delta_alpha": delta_alpha,
        "coord_other": coord_other,
        "coord_trained_ref": coord_trained_ref,
        "alpha_star_ref": alpha_star_ref,
    }
    return row, extra


# --------------------------------------------------------------------------- figure
def plot_filled_nse(ax, axis_a, axis_b, nse):
    nan_min = np.nanmin(nse)
    nan_max = np.nanmax(nse)
    vmin = max(-1.0, nan_min)
    vmax = nan_max if np.isfinite(nan_max) and nan_max > vmin else vmin + 1e-6
    levels = np.linspace(vmin, vmax, 21)
    z = np.clip(nse, -1.0, None).T
    return ax.contourf(axis_a, axis_b, z, levels=levels, cmap="viridis")


def fig_seed_compare(out: Path, staid: str, extra: dict) -> Path:
    ds_ref = extra["ds_ref"]
    planes = ds_ref.attrs["plane_names"].split(",")
    delta_alpha = extra["delta_alpha"]
    coord_other = extra["coord_other"]
    coord_trained_ref = extra["coord_trained_ref"]
    alpha_star_ref = extra["alpha_star_ref"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # --- panel 1: stiff-sloppy plane, axes already offsets from alpha_star along (v1, v3)
    pi = planes.index("stiff-sloppy")
    axis_a = ds_ref["grid_axis_a"].values[pi]
    axis_b = ds_ref["grid_axis_b"].values[pi]
    nse = ds_ref["grid_nse"].values[pi]
    cf = plot_filled_nse(axes[0], axis_a, axis_b, nse)
    fig.colorbar(cf, ax=axes[0], shrink=0.9, pad=0.03, label="NSE")
    ref_xy = (float(coord_trained_ref[0]), float(coord_trained_ref[2]))
    other_xy = (float(coord_other[0]), float(coord_other[2]))
    axes[0].plot(*ref_xy, marker="x", color="k", ms=10, mew=2.2, zorder=5, label=f"{extra['ref_arm']} trained")
    axes[0].plot(*other_xy, marker="^", color="magenta", ms=9, zorder=5, label=f"{extra['other_arm']} trained")
    axes[0].plot(0.0, 0.0, marker="*", color="red", ms=13, zorder=5, label=f"{extra['ref_arm']} optimum")
    axes[0].set_xlabel("offset s along v1 (stiff)")
    axes[0].set_ylabel("offset t along v3 (sloppy)")
    axes[0].set_title(f"{staid}: stiff-sloppy plane ({extra['ref_arm']} eigenbasis)", fontsize=9)
    axes[0].legend(fontsize=6.5, loc="best")

    # --- panel 2: n-p plane, re-expressed as offsets from alpha_star
    pi = planes.index("n-p")
    axis_a = ds_ref["grid_axis_a"].values[pi] - alpha_star_ref[0]
    axis_b = ds_ref["grid_axis_b"].values[pi] - alpha_star_ref[1]
    nse = ds_ref["grid_nse"].values[pi]
    cf = plot_filled_nse(axes[1], axis_a, axis_b, nse)
    fig.colorbar(cf, ax=axes[1], shrink=0.9, pad=0.03, label="NSE")
    ref_xy = (-float(alpha_star_ref[0]), -float(alpha_star_ref[1]))
    other_xy = (float(delta_alpha[0] - alpha_star_ref[0]), float(delta_alpha[1] - alpha_star_ref[1]))
    axes[1].plot(*ref_xy, marker="x", color="k", ms=10, mew=2.2, zorder=5)
    axes[1].plot(*other_xy, marker="^", color="magenta", ms=9, zorder=5)
    axes[1].plot(0.0, 0.0, marker="*", color="red", ms=13, zorder=5)
    axes[1].set_xlabel("alpha_n - alpha_star_n")
    axes[1].set_ylabel("alpha_p - alpha_star_p")
    axes[1].set_title(f"{staid}: n-p plane, offsets from {extra['ref_arm']} optimum", fontsize=9)

    fig.suptitle(
        "black x: reference trained point   magenta triangle: other seed's trained point   red star: reference optimum",
        fontsize=8, y=1.02,
    )
    fig.tight_layout()
    p = out / f"seed_compare_{staid}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# -------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Path, df: pd.DataFrame, ref_arm: str, other_arm: str, run_dir: Path):
    lines = [f"# Seed comparison: {other_arm} vs {ref_arm}", ""]
    lines.append(f"Run directory: {run_dir}")
    lines.append("")
    lines.append(
        "Each seed's alpha is relative to its own trained fields. Delta-alpha is the "
        "basin-uniform mean, over reaches paired by comid, of ln(field_other / field_ref) "
        "for n, p, and q. delta_alpha_std is the across-reach standard deviation of that "
        "same per-reach log ratio; the basin-uniform projection below is trustworthy only "
        "where delta_alpha_std is small relative to |delta_alpha|."
    )
    lines.append("")
    lines.append("## Table: per-gauge seed comparison")
    lines.append("")
    cols = [
        "staid", "n_reach_ref", "n_reach_other", "n_common",
        "delta_alpha_n", "delta_alpha_p", "delta_alpha_q",
        "delta_alpha_std_n", "delta_alpha_std_p", "delta_alpha_std_q",
        "c1_ref", "c2_ref", "c3_ref",
        "c1_other", "c2_other", "c3_other",
        "hw1", "hw2", "hw3",
        "ratio1_ref", "ratio2_ref", "ratio3_ref",
        "ratio1_other", "ratio2_other", "ratio3_other",
        "mult_star_n", "mult_star_p", "mult_star_q",
        "nse_star_ref", "nse_star_other",
        "loss_star_ref", "loss_star_other",
        "clamped_frac_star_ref", "clamped_frac_star_other",
        "tol_used",
    ]
    if df.empty:
        lines.append("(no gauge available in both arms)")
    else:
        lines.append(md_table(df, cols))
    lines.append("")
    lines.append(
        "ratio*_ref is |c_k| / half_width(tol) for the reference seed's own trained point in "
        "its eigenbasis (should be at or under 1 by construction, since it is the reference "
        "arm's own diagnostic). ratio*_other is the same ratio for the other seed's trained "
        "point projected via Delta-alpha; a value at or under 1 means the other seed's trained "
        "fields fall inside the reference seed's own behavioural set at that tolerance, i.e. "
        "the two seeds are not distinguishable by the reference loss landscape alone. "
        "mult_star_* is exp(Delta-alpha + alpha_star_other - alpha_star_ref), the ratio of the "
        "two seeds' physical optima (other / ref) per component."
    )
    lines.append("")
    if not df.empty:
        worst = df.loc[df[["delta_alpha_std_n", "delta_alpha_std_p", "delta_alpha_std_q"]].abs().max(axis=1).idxmax()]
        lines.append(
            f"Largest per-reach spread: gauge {worst['staid']}, "
            f"delta_alpha_std = ({worst['delta_alpha_std_n']:.3g}, {worst['delta_alpha_std_p']:.3g}, "
            f"{worst['delta_alpha_std_q']:.3g}) against delta_alpha = ({worst['delta_alpha_n']:.3g}, "
            f"{worst['delta_alpha_p']:.3g}, {worst['delta_alpha_q']:.3g})."
        )
    lines.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "SEED_COMPARE.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ref", required=True, help="reference arm name (e.g. uh-seed42)")
    ap.add_argument("--other", required=True, help="other arm name (e.g. uh-seed43)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <run_dir>/figures)")
    ap.add_argument("--tol", type=float, default=0.05, help="behavioural tolerance to report (default 0.05)")
    args = ap.parse_args()

    run_dir = args.run_dir
    out = args.out if args.out is not None else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    gauges = pd.read_csv(run_dir / "gauges.csv", dtype={"staid": str}, keep_default_na=False)

    rows = []
    written = []
    for staid in gauges["staid"]:
        result = compare_gauge(run_dir, args.ref, args.other, staid, tol_target=args.tol)
        if result is None:
            print(f"skip {staid}: missing netCDF for one or both arms")
            continue
        row, extra = result
        extra["ref_arm"] = args.ref
        extra["other_arm"] = args.other
        rows.append(row)
        written.append(fig_seed_compare(out, staid, extra))

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "seed_compare.csv", index=False)
    write_report(out, df, args.ref, args.other, run_dir)

    if not df.empty:
        print(md_table(df, ["staid", "delta_alpha_n", "delta_alpha_p", "delta_alpha_q",
                             "delta_alpha_std_n", "delta_alpha_std_p", "delta_alpha_std_q",
                             "ratio1_other", "ratio2_other", "ratio3_other",
                             "mult_star_n", "mult_star_p", "mult_star_q",
                             "nse_star_ref", "nse_star_other"]))
    else:
        print("no gauges available in both arms")

    for p in written:
        print("wrote", p)
    print("wrote", out / "seed_compare.csv")
    print("wrote", out / "SEED_COMPARE.md")


if __name__ == "__main__":
    main()
