#!/usr/bin/env python
"""Displacement census for the alpha-landscape study (spec section 7.1,
docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md).

Per gauge per arm: the gain available from the gauge's own optimum, the
optimum direction alpha_star and its physical multipliers, the stiff-
coordinate ratio |c1| / half_width(tol=0.05, k=1), and the clamp diagnostics.
Across gauges (per arm): the mean resultant length R of the unit alpha_star
directions -- R near 1 means every gauge wants the same move, R near 0 means
the directions scatter. Across arms (per gauge, only when exactly two arms
are present): the angle between the two arms' alpha_star vectors and the
difference of their gains -- the seed-to-seed agreement the design calls the
"noise on each gauge's alpha_star".

Reads only <run_dir>/manifest.json, <run_dir>/gauges.csv, and
<run_dir>/<arm>/gauges/<staid>.nc. Writes <out>/CENSUS.md,
<out>/census_per_gauge.csv, and <out>/census.png.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/census.py <run_dir> \\
        [--out <run_dir>/figures] [--tol 0.05]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

ALPHA_INDEX = {"n": 0, "p": 1, "q": 2}
COMPONENTS = ["n", "p", "q"]


# ----------------------------------------------------------------------------- io
def load(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    arms = [a["name"] for a in manifest["arms"]]
    gauges = pd.read_csv(run_dir / "gauges.csv", dtype={"staid": str}, keep_default_na=False)
    data: dict[tuple[str, str], xr.Dataset] = {}
    input_files = ["manifest.json", "gauges.csv"]
    for arm in arms:
        for staid in gauges["staid"]:
            p = run_dir / arm / "gauges" / f"{staid}.nc"
            if p.exists():
                data[(arm, staid)] = xr.open_dataset(p, decode_timedelta=False)
                input_files.append(str(p.relative_to(run_dir)))
    return manifest, arms, gauges, data, input_files


def tol_index(ds: xr.Dataset, tol_target: float) -> int:
    tolerances = ds["tolerances"].values
    return int(np.argmin(np.abs(tolerances - tol_target)))


def active_mask(ds: xr.Dataset) -> np.ndarray:
    """Bool (n, p, q) mask of which alpha components are learned model
    parameters vs fixed at a constant default. Defaults to all-active for
    netCDFs written before the active/active_params schema existed."""
    if "active" in ds.variables:
        return ds["active"].values.astype(bool)
    return np.array([True, True, True])


def study_active(data: dict) -> dict[str, bool]:
    """Per-component active flag for the whole study: a component counts as
    active only if it is active in every gauge/arm netCDF present. Used to
    decide which n-p/n-q/p-q panels are meaningful to draw."""
    mask = np.array([True, True, True])
    for ds in data.values():
        mask = mask & active_mask(ds)
    return {c: bool(mask[i]) for i, c in enumerate(COMPONENTS)}


def arm_colors(arms):
    cmap = plt.get_cmap("tab10")
    return {a: cmap(i) for i, a in enumerate(arms)}


def arm_markers(arms):
    shapes = ["o", "s", "^", "D", "v", "P", "X"]
    return {a: shapes[i % len(shapes)] for i, a in enumerate(arms)}


# --------------------------------------------------------------------- per-gauge
def per_gauge_row(arm: str, staid: str, ds: xr.Dataset, tol_target: float) -> dict:
    alpha_star = ds["alpha_star"].values.astype(float)
    norm = float(np.linalg.norm(alpha_star))
    unit = alpha_star / norm if norm > 0 else np.full(3, np.nan)
    mult = np.exp(alpha_star)
    coord_trained = ds["coord_trained"].values
    ti = tol_index(ds, tol_target)
    hw = ds["half_width"].values[ti]
    stiff_ratio = float(np.abs(coord_trained[0]) / hw[0]) if np.isfinite(hw[0]) and hw[0] > 0 else np.nan
    hit_range_bound = ds.attrs.get("hit_range_bound", np.nan)
    return {
        "arm": arm,
        "staid": staid,
        "n_reach": int(ds.attrs["n_reach"]),
        "gain": float(ds["nse_star"].values) - float(ds["nse0"].values),
        "nse0": float(ds["nse0"].values),
        "nse_star": float(ds["nse_star"].values),
        "alpha_star_n": float(alpha_star[0]),
        "alpha_star_p": float(alpha_star[1]),
        "alpha_star_q": float(alpha_star[2]),
        "alpha_star_norm": norm,
        "unit_n": float(unit[0]),
        "unit_p": float(unit[1]),
        "unit_q": float(unit[2]),
        "mult_n": float(mult[0]),
        "mult_p": float(mult[1]),
        "mult_q": float(mult[2]),
        "stiff_ratio": stiff_ratio,
        "tol_used": float(ds["tolerances"].values[ti]),
        "hit_range_bound": hit_range_bound,
        "clamped_frac_star": float(ds.attrs["clamped_frac_star"]),
    }


def build_df(arms, gauges, data, tol_target: float) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for staid in gauges["staid"]:
            ds = data.get((arm, staid))
            if ds is None:
                continue
            rows.append(per_gauge_row(arm, staid, ds, tol_target))
    return pd.DataFrame(rows)


# ----------------------------------------------------------- direction consistency
def direction_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Mean resultant length R of the unit alpha_star vectors, per arm."""
    rows = []
    for arm, sub in df.groupby("arm"):
        u = sub[["unit_n", "unit_p", "unit_q"]].to_numpy(dtype=float)
        valid = np.isfinite(u).all(axis=1)
        n_used = int(valid.sum())
        n_excluded = int((~valid).sum())
        if n_used == 0:
            rows.append(dict(arm=arm, n_gauges=n_used, n_excluded=n_excluded,
                              R=np.nan, mean_dir_n=np.nan, mean_dir_p=np.nan, mean_dir_q=np.nan))
            continue
        mean_vec = u[valid].mean(axis=0)
        R = float(np.linalg.norm(mean_vec))
        mean_dir = mean_vec / R if R > 0 else np.full(3, np.nan)
        rows.append(dict(
            arm=arm, n_gauges=n_used, n_excluded=n_excluded, R=R,
            mean_dir_n=float(mean_dir[0]), mean_dir_p=float(mean_dir[1]), mean_dir_q=float(mean_dir[2]),
        ))
    return pd.DataFrame(rows)


def pairwise_agreement(arms: list[str], gauges: pd.DataFrame, data: dict) -> pd.DataFrame:
    """Angle between the two arms' alpha_star vectors and the gain difference,
    per gauge. Only meaningful (and only computed) when exactly two arms are
    present in the run."""
    if len(arms) != 2:
        return pd.DataFrame()
    a0, a1 = arms
    rows = []
    for staid in gauges["staid"]:
        ds0, ds1 = data.get((a0, staid)), data.get((a1, staid))
        if ds0 is None or ds1 is None:
            continue
        v0 = ds0["alpha_star"].values.astype(float)
        v1 = ds1["alpha_star"].values.astype(float)
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if n0 > 0 and n1 > 0:
            cos = np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos)))
        else:
            angle_deg = np.nan
        gain0 = float(ds0["nse_star"].values) - float(ds0["nse0"].values)
        gain1 = float(ds1["nse_star"].values) - float(ds1["nse0"].values)
        rows.append(dict(
            staid=staid, arm_a=a0, arm_b=a1,
            angle_deg=angle_deg,
            alpha_star_norm_a=float(n0), alpha_star_norm_b=float(n1),
            gain_a=gain0, gain_b=gain1, gain_diff=gain1 - gain0,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- figure
def fig_census(out: Path, df: pd.DataFrame, arms: list[str], active: dict[str, bool] | None = None) -> Path:
    if active is None:
        active = {"n": True, "p": True, "q": True}
    colors_map = arm_markers(arms)  # marker shape by arm
    gains = df["gain"].to_numpy(dtype=float)
    vmin, vmax = float(np.nanmin(gains)), float(np.nanmax(gains))
    if not np.isfinite(vmax - vmin) or vmax <= vmin:
        vmax = vmin + 1e-6
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.1, 1.0])
    ax_np = fig.add_subplot(gs[0, 0:2])
    ax_nq = fig.add_subplot(gs[0, 2:4])
    ax_pq = fig.add_subplot(gs[0, 4:6])
    ax_gain = fig.add_subplot(gs[1, 0:3])
    ax_ratio = fig.add_subplot(gs[1, 3:6])

    panels_all = [("n", "p", ax_np), ("n", "q", ax_nq), ("p", "q", ax_pq)]
    # A pair panel only shows a real direction when both its components are
    # learned; drop panels touching a fixed parameter (e.g. n-p and p-q when
    # p is fixed) rather than plotting it as if it were a free axis.
    panels = [(a, b, ax) for a, b, ax in panels_all if active[a] and active[b]]
    for a_name, b_name, ax in panels_all:
        if not (active[a_name] and active[b_name]):
            ax.set_visible(False)
    for a_name, b_name, ax in panels:
        i, j = ALPHA_INDEX[a_name], ALPHA_INDEX[b_name]
        for _, row in df.iterrows():
            a, b = row[f"alpha_star_{a_name}"], row[f"alpha_star_{b_name}"]
            color = cmap(norm(row["gain"]))
            ax.annotate("", xy=(a, b), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.9))
            ax.scatter([a], [b], marker=colors_map[row["arm"]], color=color,
                       s=55, edgecolor="k", linewidth=0.4, zorder=5)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.plot(0, 0, marker="x", color="k", ms=8, mew=1.8, zorder=6)
        ax.set_xlabel(f"alpha_star_{a_name}")
        ax.set_ylabel(f"alpha_star_{b_name}")
        ax.set_title(f"{a_name}-{b_name}", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax for _, _, ax in panels], shrink=0.8, pad=0.02, label="gain (NSE* - NSE0)")

    staids = sorted(df["staid"].unique())
    x = np.arange(len(staids))
    n_arm = max(len(arms), 1)
    width = 0.8 / n_arm
    arm_c = arm_colors(arms)
    for ai, arm in enumerate(arms):
        sub = df[df["arm"] == arm].set_index("staid").reindex(staids)
        xpos = x + (ai - (n_arm - 1) / 2) * width
        ax_gain.bar(xpos, sub["gain"], width * 0.9, color=arm_c[arm], label=arm)
        ax_ratio.bar(xpos, sub["stiff_ratio"], width * 0.9, color=arm_c[arm], label=arm)
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(staids, rotation=30, ha="right")
    ax_gain.set_ylabel("gain = NSE* - NSE0")
    ax_gain.set_title("per-gauge gain available from own optimum")
    ax_gain.axhline(0, color="k", lw=0.6)
    ax_gain.legend(fontsize=7)

    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(staids, rotation=30, ha="right")
    ax_ratio.set_yscale("log")
    ax_ratio.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_ratio.set_title("stiff coordinate ratio |c1| / half_width(tol=0.05, k=1)")
    ax_ratio.legend(fontsize=7)

    fig.suptitle(
        "arrow: trained point (origin) -> alpha_star, colour by gain, marker shape by arm",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "census.png"
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


def write_report(out: Path, df: pd.DataFrame, cons_df: pd.DataFrame, pair_df: pd.DataFrame,
                  arms: list[str], gauges: pd.DataFrame, input_files: list[str], tol_target: float):
    lines = ["# Landscape census (spec section 7.1)", ""]
    present = set(df["staid"]) if not df.empty else set()
    missing = [s for s in gauges["staid"] if s not in present]
    lines.append(f"Arms: {arms}. Gauges with data: {sorted(present)}. Gauges not yet available: {missing}.")
    if not df.empty and df["hit_range_bound"].isna().all():
        lines.append(
            "Note: `hit_range_bound` is absent from every gauge netCDF in this run "
            "(older ddrs binary, predates that attribute) -- reported as nan below."
        )
    lines.append("")

    lines.append("## Table 1: per-gauge per-arm census")
    lines.append("")
    t1_cols = [
        "arm", "staid", "n_reach", "gain", "nse0", "nse_star",
        "alpha_star_n", "alpha_star_p", "alpha_star_q", "alpha_star_norm",
        "unit_n", "unit_p", "unit_q",
        "mult_n", "mult_p", "mult_q",
        "stiff_ratio", "tol_used", "hit_range_bound", "clamped_frac_star",
    ]
    if df.empty:
        lines.append("(no gauge netCDFs available yet)")
    else:
        lines.append(md_table(df, t1_cols))
    lines.append("")
    lines.append(
        "gain is NSE(alpha_star) - NSE(0): how much the gauge could improve by moving to its own "
        "optimum. unit_* is alpha_star normalised to a unit vector (nan if alpha_star = 0). "
        "mult_* = exp(alpha_star) is the physical optimum's multiplier on the trained n, p, q "
        f"fields. stiff_ratio uses tolerance {tol_target} and k=1 (the stiffest eigenvector); "
        "values at or above 1 mean the trained point sits outside the behavioural set along the "
        "stiff direction."
    )
    lines.append("")

    lines.append("## Table 2: direction consistency across gauges, per arm")
    lines.append("")
    if cons_df.empty:
        lines.append("(no gauge netCDFs available yet)")
    else:
        lines.append(md_table(cons_df, ["arm", "n_gauges", "n_excluded", "R", "mean_dir_n", "mean_dir_p", "mean_dir_q"]))
    lines.append("")
    lines.append(
        "R is the mean resultant length of the unit alpha_star vectors across gauges (R near 1: "
        "every gauge wants the same move; R near 0: directions scatter). mean_dir is the unit "
        "vector of that mean direction (undefined, reported nan, when R = 0). n_excluded counts "
        "gauges with alpha_star = 0 (no displacement, undefined direction)."
    )
    lines.append("")

    lines.append("## Table 3: pairwise seed / arm agreement per gauge")
    lines.append("")
    if len(arms) != 2:
        lines.append(f"Skipped: this run has {len(arms)} arm(s), not 2. Pairwise agreement needs exactly two.")
    elif pair_df.empty:
        lines.append("(no gauge available in both arms yet)")
    else:
        lines.append(md_table(pair_df, ["staid", "arm_a", "arm_b", "angle_deg",
                                         "alpha_star_norm_a", "alpha_star_norm_b",
                                         "gain_a", "gain_b", "gain_diff"]))
        lines.append("")
        lines.append(
            "angle_deg is the angle between the two arms' alpha_star vectors (nan if either is "
            "zero); gain_diff is gain_b - gain_a. Small angles and small gain_diff indicate the "
            "census finding is stable across the two seeds/arms, i.e. it reflects the batch "
            "training rather than seed noise."
        )
    lines.append("")

    lines.append("## Input files")
    lines.append("")
    for f in input_files:
        lines.append(f"- {f}")
    lines.append("")
    (out / "CENSUS.md").write_text("\n".join(lines))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = args.run_dir
    out = args.out if args.out is not None else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    manifest, arms, gauges, data, input_files = load(run_dir)
    print(f"arms: {arms}; gauges: {list(gauges['staid'])}; datasets found: {len(data)}")

    df = build_df(arms, gauges, data, args.tol)
    cons_df = direction_consistency(df) if not df.empty else pd.DataFrame()
    pair_df = pairwise_agreement(arms, gauges, data)

    written = []
    if not df.empty:
        df.to_csv(out / "census_per_gauge.csv", index=False)
        written.append(out / "census_per_gauge.csv")
        written.append(fig_census(out, df, arms, study_active(data)))

    write_report(out, df, cons_df, pair_df, arms, gauges, input_files, args.tol)
    written.append(out / "CENSUS.md")

    if not df.empty:
        print(md_table(df, ["arm", "staid", "gain", "alpha_star_n", "alpha_star_p", "alpha_star_q",
                             "stiff_ratio", "clamped_frac_star"]))
    else:
        print("no gauges available yet")

    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
