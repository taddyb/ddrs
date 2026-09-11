#!/usr/bin/env python
"""Compare two fixed-p=21 arms that differ only in training population.

Run A trained on gages_3000 (2,859 gauges after filtering); Run B trained on
gages_2000_area_balanced (1,841 gauges). Both are p=21, p-fixed landscape
study arms (src/experiment/landscape); each gauge netCDF carries its own
basin-median trained Manning n (n0), width exponent q (q0), the trained-point
NSE (nse0), and the local-optimum log-multiplier vector (alpha_star) in the
(n, p_spatial, q_spatial) alpha ordering.

Two hypotheses this script is built to separate:
  H-in:  gauges IN a run's training set get n near their own optimum
         (alpha_star_n small in that run, regardless of the other run).
  H-pop: the wider training population shifts trained n everywhere, roughly
         alike whether or not the gauge itself was in that population.

Reads only <runA-dir>/gauges/<staid>.nc and <runB-dir>/gauges/<staid>.nc for
gauges present in BOTH directories (Run B may still be training and have
fewer gauges written -- re-running this script later picks up whatever is
new). Writes <out>/population_compare_per_gauge.csv, <out>/POPULATION_COMPARE.md,
and <out>/population_compare.png.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/population_compare.py \\
        <runA-dir> <runB-dir> [--out <runA-dir>/figures]

Example:
    ~/projects/ddr/.venv/bin/python experiments/landscape/population_compare.py \\
        .ddrs/experiments/landscape-p21-census41/2026-09-08T18-06-55Z/p21-conus \\
        .ddrs/experiments/landscape-pfixed-census41/2026-09-08T20-45-04Z/uh-pfixed21
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

# Training-population CSVs named in the task. STAID is already zero-padded
# to 8 digits as text in both files; read with dtype=str to keep it that way
# and match gauge netCDF filenames / attrs["staid"] verbatim.
GAGES_A_CSV = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
GAGES_B_CSV = Path("/home/tbindas/projects/ddr/references/gage_info/gages_2000_area_balanced.csv")
STAID_COL = "STAID"


# ----------------------------------------------------------------------------- io
def load_gauge(arm_dir: Path, staid: str) -> xr.Dataset | None:
    p = arm_dir / "gauges" / f"{staid}.nc"
    if not p.exists():
        return None
    return xr.open_dataset(p, decode_timedelta=False)


def common_staids(a_dir: Path, b_dir: Path) -> tuple[list[str], list[str], list[str]]:
    a_ids = sorted(p.stem for p in (a_dir / "gauges").glob("*.nc"))
    b_ids = sorted(p.stem for p in (b_dir / "gauges").glob("*.nc"))
    common = sorted(set(a_ids) & set(b_ids))
    return common, a_ids, b_ids


def load_training_staids(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path, dtype=str)
    return set(df[STAID_COL].str.zfill(8))


def find_area_var(ds: xr.Dataset) -> str | None:
    """Look for an upstream-area-like variable/attr; return its name or None."""
    for name in list(ds.variables) + list(ds.attrs):
        if "area" in name.lower():
            return name
    return None


# --------------------------------------------------------------------- per-gauge
def per_gauge_row(staid: str, ds_a: xr.Dataset, ds_b: xr.Dataset, a_train: set[str], b_train: set[str]) -> dict:
    n_reach_a = int(ds_a.attrs["n_reach"])
    n_reach_b = int(ds_b.attrs["n_reach"])

    n0_a, n0_b = float(np.median(ds_a["n0"].values)), float(np.median(ds_b["n0"].values))
    q0_a, q0_b = float(np.median(ds_a["q0"].values)), float(np.median(ds_b["q0"].values))
    nse0_a, nse0_b = float(ds_a["nse0"].values), float(ds_b["nse0"].values)

    alpha_star_n_a = float(ds_a["alpha_star"].values[0])
    alpha_star_n_b = float(ds_b["alpha_star"].values[0])
    mult_n_a = float(np.exp(alpha_star_n_a))
    mult_n_b = float(np.exp(alpha_star_n_b))

    in_a = staid in a_train
    in_b = staid in b_train
    if in_a and in_b:
        cls = "both"
    elif in_a:
        cls = "A only"
    elif in_b:
        cls = "B only"
    else:
        cls = "neither"

    return {
        "staid": staid,
        "n_reach": n_reach_a,
        "n_reach_mismatch": n_reach_a != n_reach_b,
        "inA_train": in_a,
        "inB_train": in_b,
        "class": cls,
        "nA": n0_a,
        "nB": n0_b,
        "ln_nA_over_nB": float(np.log(n0_a / n0_b)),
        "qA": q0_a,
        "qB": q0_b,
        "nse0A": nse0_a,
        "nse0B": nse0_b,
        "alpha_star_n_A": alpha_star_n_a,
        "alpha_star_n_B": alpha_star_n_b,
        "mult_n_A": mult_n_a,
        "mult_n_B": mult_n_b,
    }


def build_df(common: list[str], a_dir: Path, b_dir: Path, a_train: set[str], b_train: set[str]) -> pd.DataFrame:
    rows = []
    for staid in common:
        ds_a = load_gauge(a_dir, staid)
        ds_b = load_gauge(b_dir, staid)
        if ds_a is None or ds_b is None:
            continue
        rows.append(per_gauge_row(staid, ds_a, ds_b, a_train, b_train))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- summaries
def group_median(df: pd.DataFrame, mask: pd.Series, col: str) -> tuple[float, int]:
    sub = df.loc[mask, col].dropna()
    if len(sub) == 0:
        return float("nan"), 0
    return float(sub.median()), len(sub)


def summarize(df: pd.DataFrame) -> dict:
    both_med, both_n = group_median(df, df["class"] == "both", "ln_nA_over_nB")
    a_only_med, a_only_n = group_median(df, df["class"] == "A only", "ln_nA_over_nB")

    df = df.assign(abs_ln_mult_n_B=df["alpha_star_n_B"].abs())
    in_b_med, in_b_n = group_median(df, df["inB_train"], "abs_ln_mult_n_B")
    not_in_b_med, not_in_b_n = group_median(df, ~df["inB_train"], "abs_ln_mult_n_B")

    return {
        "both_median_ln_ratio": both_med, "both_n": both_n,
        "a_only_median_ln_ratio": a_only_med, "a_only_n": a_only_n,
        "in_b_median_abs_ln_mult_n_b": in_b_med, "in_b_n": in_b_n,
        "not_in_b_median_abs_ln_mult_n_b": not_in_b_med, "not_in_b_n": not_in_b_n,
    }


# ----------------------------------------------------------------------- figure
CLASS_COLORS = {"both": "tab:blue", "A only": "tab:orange", "B only": "tab:green", "neither": "0.5"}


def make_figure(out: Path, df: pd.DataFrame) -> Path:
    fig, (ax_scatter, ax_strip) = plt.subplots(1, 2, figsize=(12, 5.5))

    for cls, sub in df.groupby("class"):
        ax_scatter.scatter(sub["nA"], sub["nB"], color=CLASS_COLORS.get(cls, "k"),
                            label=f"{cls} (n={len(sub)})", s=45, edgecolor="k", linewidth=0.4, zorder=5)
    if len(df):
        lo = min(df["nA"].min(), df["nB"].min()) * 0.8
        hi = max(df["nA"].max(), df["nB"].max()) * 1.25
        ax_scatter.plot([lo, hi], [lo, hi], color="0.6", ls="--", lw=1, label="1:1", zorder=1)
        ax_scatter.set_xlim(lo, hi)
        ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("nA (basin-median trained n, run A: gages_3000)")
    ax_scatter.set_ylabel("nB (basin-median trained n, run B: gages_2000_area_balanced)")
    ax_scatter.set_title("trained n, run A vs run B")
    ax_scatter.legend(fontsize=8)

    classes = [c for c in ["both", "A only", "B only", "neither"] if c in df["class"].unique()]
    rng = np.random.default_rng(0)
    for i, cls in enumerate(classes):
        sub = df[df["class"] == cls]
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        ax_strip.scatter(np.full(len(sub), i) + jitter, sub["ln_nA_over_nB"],
                          color=CLASS_COLORS.get(cls, "k"), s=40, edgecolor="k", linewidth=0.4, zorder=5)
        if len(sub):
            med = sub["ln_nA_over_nB"].median()
            ax_strip.plot([i - 0.25, i + 0.25], [med, med], color="k", lw=2, zorder=6)
    ax_strip.axhline(0, color="0.7", lw=0.8)
    ax_strip.set_xticks(range(len(classes)))
    ax_strip.set_xticklabels([f"{c}\n(n={len(df[df['class'] == c])})" for c in classes])
    ax_strip.set_ylabel("ln(nA / nB)")
    ax_strip.set_title("ln(nA/nB) by training-set membership class\n(black bar = class median)")

    fig.suptitle("Run A (gages_3000) vs run B (gages_2000_area_balanced), p fixed at 21", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "population_compare.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# -------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4g}" if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Path, df: pd.DataFrame, summary: dict, a_dir: Path, b_dir: Path,
                  a_ids: list[str], b_ids: list[str], common: list[str],
                  a_train_csv: Path, b_train_csv: Path, area_var: str | None):
    lines = ["# Population compare: run A (gages_3000) vs run B (gages_2000_area_balanced), p fixed at 21", ""]
    lines.append(f"Run A dir: `{a_dir}` ({len(a_ids)} gauge netCDFs found).")
    lines.append(f"Run B dir: `{b_dir}` ({len(b_ids)} gauge netCDFs found; run may still be in progress).")
    lines.append(f"Gauges present in both: {len(common)} of {len(a_ids)} run-A gauges.")
    lines.append("")
    lines.append(
        f"Training-set membership matched on column `{STAID_COL}` in `{a_train_csv}` "
        f"(run A, {len(load_training_staids(a_train_csv))} rows) and `{b_train_csv}` "
        f"(run B, {len(load_training_staids(b_train_csv))} rows), zero-padded to 8 digits."
    )
    if len(df) < 5:
        lines.append("")
        lines.append(
            f"**Note: only {len(df)} gauges are present in both runs right now** (run B has "
            f"{len(b_ids)} gauge netCDFs written so far). The summaries below are computed on "
            "whatever overlap exists; re-run this script as run B writes more gauges."
        )
    lines.append("")

    lines.append("## Table 1: per-gauge comparison")
    lines.append("")
    t1_cols = ["staid", "n_reach", "inA_train", "inB_train", "nA", "nB", "ln_nA_over_nB", "qA", "qB", "nse0A", "nse0B"]
    if df.empty:
        lines.append("(no gauges present in both runs yet)")
    else:
        lines.append(md_table(df, t1_cols))
        if df["n_reach_mismatch"].any():
            mism = df.loc[df["n_reach_mismatch"], "staid"].tolist()
            lines.append("")
            lines.append(
                f"**Warning:** n_reach (upstream reach count) disagrees between run A and run B for "
                f"{mism} -- n_reach above is run A's value. Check the two arms share the same adjacency."
            )
    lines.append("")

    lines.append("## Summary 1: H-pop vs H-in, ln(nA/nB) by membership class")
    lines.append("")
    lines.append(
        f"- Gauges in BOTH training sets (n={summary['both_n']}): median ln(nA/nB) = "
        f"{summary['both_median_ln_ratio']:.4g}" if summary["both_n"] else
        "- Gauges in BOTH training sets: none in the current overlap."
    )
    lines.append(
        f"- Gauges in run A's training set ONLY (n={summary['a_only_n']}): median ln(nA/nB) = "
        f"{summary['a_only_median_ln_ratio']:.4g}" if summary["a_only_n"] else
        "- Gauges in run A's training set ONLY: none in the current overlap."
    )
    lines.append("")
    if summary["both_n"] and summary["a_only_n"]:
        lines.append(
            "Under H-pop, these two medians should be similar (the wider population shifts n "
            "everywhere, whether or not the gauge itself was in the training set). Under H-in, "
            "the 'A only' group should show a systematically larger |ln(nA/nB)| than the 'both' "
            "group, since those gauges are pulled toward run A's optimum but not run B's."
        )
    else:
        lines.append(
            "Not enough gauges in one or both classes yet to compare the two medians -- "
            "re-run once run B has written more gauges."
        )
    lines.append("")

    lines.append("## Summary 2: does run B's training membership predict nearness to B's own optimum")
    lines.append("")
    lines.append(
        f"- Gauges IN run B's training set (n={summary['in_b_n']}): median |ln(mult_n_B)| "
        f"(= |alpha_star_n| at run B's own optimum) = {summary['in_b_median_abs_ln_mult_n_b']:.4g}"
        if summary["in_b_n"] else
        "- Gauges IN run B's training set: none in the current overlap."
    )
    lines.append(
        f"- Gauges NOT in run B's training set (n={summary['not_in_b_n']}): median |ln(mult_n_B)| = "
        f"{summary['not_in_b_median_abs_ln_mult_n_b']:.4g}" if summary["not_in_b_n"] else
        "- Gauges NOT in run B's training set: none in the current overlap."
    )
    lines.append("")
    lines.append(
        "Smaller |ln(mult_n_B)| means run B's trained n at that gauge sits closer to that gauge's "
        "own local optimum. If the in-training-set group is systematically smaller than the "
        "not-in-training-set group, that supports H-in for run B."
    )
    lines.append("")

    lines.append("## 84-gauge run-A-only view: nA vs upstream area")
    lines.append("")
    if area_var is None:
        lines.append(
            "Skipped: no upstream-area variable or attribute was found in the gauge netCDFs "
            "(checked variable names and attrs for anything containing 'area'; only n0/p0/q0 per "
            "reach and comid are stored, no drainage-area field). This panel is not in "
            "population_compare.png."
        )
    else:
        lines.append(f"Found area-like field `{area_var}` -- see population_compare.png for the panel.")
    lines.append("")

    lines.append("## Input files")
    lines.append("")
    lines.append(f"- {a_dir / 'gauges'}")
    lines.append(f"- {b_dir / 'gauges'}")
    lines.append(f"- {a_train_csv}")
    lines.append(f"- {b_train_csv}")
    lines.append("")
    (out / "POPULATION_COMPARE.md").write_text("\n".join(lines))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runA_dir", type=Path)
    ap.add_argument("runB_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a_dir = args.runA_dir
    b_dir = args.runB_dir
    out = args.out if args.out is not None else a_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    common, a_ids, b_ids = common_staids(a_dir, b_dir)
    print(f"run A gauges: {len(a_ids)}; run B gauges: {len(b_ids)}; present in both: {len(common)}")
    if len(b_ids) < 5:
        print(f"note: run B has only {len(b_ids)} gauge netCDFs -- it may still be running; "
              "proceeding with whatever exists, re-run later for more.")

    a_train = load_training_staids(GAGES_A_CSV)
    b_train = load_training_staids(GAGES_B_CSV)

    df = build_df(common, a_dir, b_dir, a_train, b_train)

    area_var = None
    if common:
        ds0 = load_gauge(a_dir, common[0])
        area_var = find_area_var(ds0) if ds0 is not None else None

    written = []
    if not df.empty:
        df.to_csv(out / "population_compare_per_gauge.csv", index=False)
        written.append(out / "population_compare_per_gauge.csv")
        written.append(make_figure(out, df))

    summary = summarize(df) if not df.empty else {
        "both_median_ln_ratio": float("nan"), "both_n": 0,
        "a_only_median_ln_ratio": float("nan"), "a_only_n": 0,
        "in_b_median_abs_ln_mult_n_b": float("nan"), "in_b_n": 0,
        "not_in_b_median_abs_ln_mult_n_b": float("nan"), "not_in_b_n": 0,
    }

    write_report(out, df, summary, a_dir, b_dir, a_ids, b_ids, common, GAGES_A_CSV, GAGES_B_CSV, area_var)
    written.append(out / "POPULATION_COMPARE.md")

    if not df.empty:
        print(md_table(df, ["staid", "n_reach", "inA_train", "inB_train", "nA", "nB",
                             "ln_nA_over_nB", "qA", "qB", "nse0A", "nse0B"]))
        print()
        print("summary:", summary)
    else:
        print("no gauges present in both runs yet")

    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
