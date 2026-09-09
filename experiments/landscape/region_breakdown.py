#!/usr/bin/env python
"""Regional breakdown of the trained-vs-optimal Manning n landscape census
(conus_map.py), joined against GAGES-II basin attributes.

Reuses conus_map.build_df (per-gauge nse0/nse_star/alpha_n_star/mult_n/gain)
and joins it to the GAGES-II table
(/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf) on zero-padded
STAID, to see whether the "how far is training from the gauge optimum" and
"which way does it want to move" patterns from conus_map.py cluster by
region. Writes region_summary.csv, REGION.md, and
conus_n_gap_by_region.png under --out.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/region_breakdown.py \\
        <census_run_dir> [--arm <name>] [--region {huc02,ecoregion,pur7}] \\
        [--out <census_run_dir>/figures/region-<region>] [--nse-min 0.3]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from conus_map import LN3, build_df, load_gage_csv, load_manifest, pick_arm, window_info  # noqa: E402

GAGES_II_DBF = Path("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf")

# TODO: this is a provisional geography-only grouping of HUC02 codes, standing
# in for the published Feng et al. 2021 PUR (physiographic/unit-response)
# region definition until that table is available -- replace it wholesale
# when it is.
PUR7 = {
    "NE": ["01", "02"],
    "SE": ["03", "06"],
    "Midwest": ["04", "05", "07"],
    "Plains": ["09", "10U", "10L", "11"],
    "South": ["08", "12", "13"],
    "Mountain": ["14", "15", "16"],
    "Pacific": ["17", "18"],
}


# ----------------------------------------------------------------------------- io
def load_gages_ii(dbf_path: Path) -> pd.DataFrame:
    gdf = gpd.read_file(dbf_path)
    df = pd.DataFrame(gdf[["STAID", "HUC02", "AGGECOREGI", "STATE"]])
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    return df


def huc02_to_pur7(huc02: str) -> str:
    for name, codes in PUR7.items():
        if huc02 in codes:
            return name
    return "Other"


def assign_region(df: pd.DataFrame, gages_ii: pd.DataFrame, region_kind: str) -> pd.DataFrame:
    merged = df.merge(gages_ii, left_on="staid", right_on="STAID", how="left")
    missing = int(merged["HUC02"].isna().sum())
    if missing:
        print(f"warning: {missing} gauge(s) have no match in the GAGES-II table; dropped from the regional breakdown")
    merged = merged.dropna(subset=["HUC02"]).copy()
    if region_kind == "huc02":
        merged["region"] = merged["HUC02"]
    elif region_kind == "ecoregion":
        merged["region"] = merged["AGGECOREGI"]
    elif region_kind == "pur7":
        print(
            "warning: --region pur7 uses a PROVISIONAL geography-only HUC02 grouping "
            "(see the PUR7 dict TODO in this script), not the published Feng et al. 2021 "
            "PUR region definition -- treat this breakdown as a placeholder"
        )
        merged["region"] = merged["HUC02"].map(huc02_to_pur7)
    else:
        raise ValueError(f"unknown region kind {region_kind!r}")
    return merged


# ------------------------------------------------------------------------ summary
def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per region plus a trailing 'all' row, sorted by median_gain
    descending (the 'all' row is appended last, not sorted in)."""

    def stats(sub: pd.DataFrame, name: str) -> dict:
        well = sub[sub["well_fit"]]
        n_well = len(well)
        return {
            "region": name,
            "n_gauges": len(sub),
            "n_well_fit": n_well,
            "median_nse0_well_fit": float(np.median(well["nse0"])) if n_well else float("nan"),
            "median_abs_ln_ratio_well_fit": float(np.median(well["alpha_n_star"].abs())) if n_well else float("nan"),
            "share_within_1p25": float((well["alpha_n_star"].abs() <= math.log(1.25)).mean()) if n_well else float("nan"),
            "share_wants_slower": float((well["mult_n"] > 1.25).mean()) if n_well else float("nan"),
            "share_wants_faster": float((well["mult_n"] < 0.8).mean()) if n_well else float("nan"),
            "median_gain": float(np.median(well["gain"])) if n_well else float("nan"),
            "share_gain_gt_0p02": float((well["gain"] > 0.02).mean()) if n_well else float("nan"),
        }

    rows = [stats(sub, region) for region, sub in df.groupby("region")]
    out = pd.DataFrame(rows).sort_values("median_gain", ascending=False, na_position="last").reset_index(drop=True)
    all_row = pd.DataFrame([stats(df, "all")])
    return pd.concat([out, all_row], ignore_index=True)


MD_COLUMNS = [
    ("region", "Region", "s"),
    ("n_gauges", "N gauges", "d"),
    ("n_well_fit", "N well-fit", "d"),
    ("median_nse0_well_fit", "Median NSE0 (well-fit)", "f"),
    ("median_abs_ln_ratio_well_fit", "Median |ln(n_opt/n_trained)|", "f"),
    ("share_within_1p25", "Share within 1.25x", "%"),
    ("share_wants_slower", "Wants slower (mult_n>1.25)", "%"),
    ("share_wants_faster", "Wants faster (mult_n<0.8)", "%"),
    ("median_gain", "Median gain", "f"),
    ("share_gain_gt_0p02", "Share gain>0.02", "%"),
]


def region_md_table(summary: pd.DataFrame) -> str:
    header = [label for _, label, _ in MD_COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in summary.iterrows():
        cells = []
        for col, _, kind in MD_COLUMNS:
            v = row[col]
            if kind == "s":
                cells.append(str(v))
            elif kind == "d":
                cells.append(f"{int(v)}")
            elif kind == "%":
                cells.append(f"{v:.1%}" if np.isfinite(v) else "nan")
            else:
                cells.append(f"{v:.4g}" if np.isfinite(v) else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------- figure
def region_colors(regions: list[str]) -> dict:
    cmap = plt.get_cmap("tab20" if len(regions) > 10 else "tab10")
    return {r: cmap(i % cmap.N) for i, r in enumerate(regions)}


def draw_mini_frame(ax):
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.8")
    ax.grid(True, color="0.92", lw=0.4, zorder=0)


def box_strip(ax, df: pd.DataFrame, regions: list[str], colors: dict, col: str, xlabel: str):
    data = [df.loc[(df["region"] == r) & df["well_fit"], col].dropna().to_numpy() for r in regions]
    positions = np.arange(len(regions))
    bp = ax.boxplot(data, positions=positions, vert=False, widths=0.6, showfliers=False, patch_artist=True)
    for i, r in enumerate(regions):
        c = colors[r]
        bp["boxes"][i].set(edgecolor=c, facecolor="none", linewidth=1.6)
        bp["medians"][i].set(color=c, linewidth=1.6)
        for whisk in bp["whiskers"][2 * i:2 * i + 2]:
            whisk.set(color=c)
        for cap in bp["caps"][2 * i:2 * i + 2]:
            cap.set(color=c)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.08, size=len(vals))
        ax.scatter(vals, positions[i] + jitter, s=6, color=colors[regions[i]], alpha=0.35, zorder=5, linewidths=0)
    ax.axvline(0, color="0.3", lw=0.8, ls="--", zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(regions, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, axis="x", color="0.9", lw=0.5, zorder=0)


def fig_region_grid(
    out: Path, df: pd.DataFrame, regions: list[str], colors: dict, region_kind: str, arm: str, run_id: str,
) -> Path:
    n = len(regions)
    ncols = max(1, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(18, 14), dpi=150)
    gs = fig.add_gridspec(
        nrows + 2, ncols + 1,
        width_ratios=[*([1] * ncols), 0.06],
        height_ratios=[*([1] * nrows), 1.2, 1.2],
        hspace=0.55, wspace=0.15,
    )

    cmap = plt.get_cmap("coolwarm_r")  # reversed: negative alpha (faster) -> red, positive (slower) -> blue
    mini_axes = []
    sm = None
    for i, region in enumerate(regions):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        draw_mini_frame(ax)
        other = df[df["region"] != region]
        mine = df[df["region"] == region]
        ax.scatter(other["lon"], other["lat"], s=3, c="0.85", edgecolors="none", zorder=1)
        poor = mine[~mine["well_fit"]]
        well = mine[mine["well_fit"]]
        if not poor.empty:
            ax.scatter(poor["lon"], poor["lat"], s=10, facecolors="none", edgecolors="0.5", linewidths=0.6, zorder=3)
        if not well.empty:
            sm = ax.scatter(
                well["lon"], well["lat"], s=10, c=well["alpha_n_star"], cmap=cmap, vmin=-LN3, vmax=LN3,
                edgecolors="none", zorder=4,
            )
        ax.set_title(f"{region}\nn={len(mine)}, well-fit={len(well)}", fontsize=8)
        mini_axes.append(ax)

    cax = fig.add_subplot(gs[:nrows, ncols])
    if sm is not None:
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("signed ln(n_opt/n_trained)\n(blue = slower/higher n, red = faster/lower n)", fontsize=8)
    else:
        cax.axis("off")

    ax_alpha = fig.add_subplot(gs[nrows, :ncols])
    box_strip(ax_alpha, df, regions, colors, "alpha_n_star", "signed ln(n_opt/n_trained), well-fit gauges")
    ax_gain = fig.add_subplot(gs[nrows + 1, :ncols])
    box_strip(ax_gain, df, regions, colors, "gain", "NSE gain at optimum, well-fit gauges")

    fig.suptitle(
        f"arm={arm}  run={run_id}  region={region_kind}\n"
        "small multiples: focal region coloured by signed ln(n_opt/n_trained) "
        "(hollow = not well-fit), other regions as grey context dots; "
        "bottom: per-region distributions among well-fit gauges, region order = median gain descending",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.07, right=0.97)
    p = out / "conus_n_gap_by_region.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# -------------------------------------------------------------------------- report
def write_report(
    out: Path, summary: pd.DataFrame, md: str, region_kind: str, arm: str, run_id: str, nse_min: float,
    n_dropped: int,
) -> None:
    lines = [f"# Regional breakdown ({region_kind})", ""]
    lines.append(f"arm={arm}  run={run_id}  nse_min={nse_min}")
    lines.append("")
    if region_kind == "pur7":
        lines.append(
            "**Note:** the pur7 grouping is a provisional geography-only bucketing of HUC02 "
            "codes (see the `PUR7` dict TODO in `region_breakdown.py`), not the published "
            "Feng et al. 2021 PUR region definition. Treat this table as a placeholder."
        )
        lines.append("")
    if n_dropped:
        lines.append(f"{n_dropped} gauge(s) had no GAGES-II match and were dropped from this breakdown.")
        lines.append("")
    lines.append(md)
    lines.append("")
    lines.append(
        "Rows are sorted by median gain (NSE the gauge could still gain at its own optimum) "
        "descending; the `all` row is the whole-population baseline from conus_map.py."
    )
    (out / "REGION.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census_run_dir", type=Path)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--region", choices=["huc02", "ecoregion", "pur7"], default="ecoregion")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--nse-min", type=float, default=0.3)
    ap.add_argument("--gages-ii-dbf", type=Path, default=GAGES_II_DBF)
    args = ap.parse_args()

    run_dir = args.census_run_dir
    out = args.out if args.out is not None else run_dir / "figures" / f"region-{args.region}"
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(run_dir)
    arm_entry = pick_arm(manifest, args.arm)
    arm = arm_entry["name"]
    run_id = arm_entry["run_id"]

    gage_csv = load_gage_csv(run_dir, arm_entry)
    window_days, _ = window_info(run_dir, arm)
    df = build_df(run_dir, arm, gage_csv, args.nse_min, window_days)

    gages_ii = load_gages_ii(args.gages_ii_dbf)
    n_before = len(df)
    df = assign_region(df, gages_ii, args.region)
    n_dropped = n_before - len(df)

    summary = region_summary(df)
    csv_path = out / "region_summary.csv"
    summary.to_csv(csv_path, index=False)

    regions = summary.loc[summary["region"] != "all", "region"].tolist()
    colors = region_colors(regions)

    md = region_md_table(summary)
    write_report(out, summary, md, args.region, arm, run_id, args.nse_min, n_dropped)

    png_path = fig_region_grid(out, df, regions, colors, args.region, arm, run_id)

    print(md)
    print(f"wrote {csv_path}")
    print(f"wrote {out / 'REGION.md'}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
