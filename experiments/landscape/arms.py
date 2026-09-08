#!/usr/bin/env python
"""Cross-arm (inflow input) figures for the alpha-landscape study (spec
section 7.3, docs/superpowers/specs/2026-09-07-adjoint-landscape-design.md).

Arms here are different inflow inputs -- separate runs, so the trained
fields (n0, p0, q0) differ per reach AND per arm. For a chosen reference arm
(the first arm listed in the run's experiment.yaml, unless --ref overrides
it), every other arm's trained point and per-gauge optimum are expressed in
the reference arm's physical log space, reach-paired by comid:

    delta_alpha(arm)      = mean over common reaches of ln(field_arm / field_ref)
    optimum_offset(arm)   = delta_alpha(arm) + alpha_star(arm) - alpha_star(ref)

delta_alpha(arm) is arm's trained point, expressed as a log-multiplier
offset from the reference arm's trained point. optimum_offset(arm) is the
same for arm's own per-gauge optimum. The reference arm's own row is the
zero vector for both by construction.

Discriminating statistic per gauge per component: spread (max - min, in log
units) of optimum_offset across arms, divided by spread of delta_alpha
across arms. Near 0: only the batch (trained) solution moves with the
input, the constrained optimum does not. Near 1: the optimum moves with the
input as much as the trained point does (the channel parameters are
absorbing inflow bias).

Reads only <run_dir>/manifest.json, <run_dir>/gauges.csv, and
<run_dir>/<arm>/gauges/<staid>.nc. Writes <out>/ARMS.md,
<out>/arms_per_gauge.csv, and <out>/arms.png.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/arms.py <run_dir> \\
        [--out <run_dir>/figures] [--ref <arm-name>]
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

FIELD_VARS = ["n0", "p0", "q0"]
COMPONENTS = ["n", "p", "q"]


# ----------------------------------------------------------------------------- io
def load(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # "the run's experiment.yaml" order -- manifest["spec"]["arms"] is the
    # experiment.yaml arms list verbatim (before run-id resolution).
    spec_arms = [a["name"] for a in manifest["spec"]["arms"]]
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
    return manifest, spec_arms, arms, gauges, data, input_files


def align_by_comid(ds_a: xr.Dataset, ds_b: xr.Dataset):
    comid_a = ds_a["comid"].values
    comid_b = ds_b["comid"].values
    common, idx_a, idx_b = np.intersect1d(comid_a, comid_b, return_indices=True)
    return common, idx_a, idx_b


def arm_colors(arms):
    cmap = plt.get_cmap("tab10")
    return {a: cmap(i % 10) for i, a in enumerate(arms)}


# --------------------------------------------------------------------- per-gauge
def arm_row(ref_arm: str, arm: str, staid: str, ds_ref: xr.Dataset, ds_arm: xr.Dataset) -> dict:
    if arm == ref_arm:
        delta_alpha = np.zeros(3)
        delta_alpha_std = np.zeros(3)
        n_common = int(ds_ref.sizes["reach"])
    else:
        common, idx_arm, idx_ref = align_by_comid(ds_arm, ds_ref)
        n_common = int(len(common))
        d_per_reach = {}
        for var in FIELD_VARS:
            v_arm = ds_arm[var].values[idx_arm]
            v_ref = ds_ref[var].values[idx_ref]
            d_per_reach[var] = np.log(v_arm / v_ref)
        delta_alpha = np.array([d_per_reach[v].mean() for v in FIELD_VARS])
        delta_alpha_std = np.array([d_per_reach[v].std() for v in FIELD_VARS])

    alpha_star_arm = ds_arm["alpha_star"].values.astype(float)
    alpha_star_ref = ds_ref["alpha_star"].values.astype(float)
    optimum_offset = delta_alpha + alpha_star_arm - alpha_star_ref

    row = {
        "arm": arm,
        "staid": staid,
        "is_ref": arm == ref_arm,
        "n_reach_arm": int(ds_arm.attrs["n_reach"]),
        "n_reach_ref": int(ds_ref.attrs["n_reach"]),
        "n_common": n_common,
        "nse_star": float(ds_arm["nse_star"].values),
        "hit_range_bound": ds_arm.attrs.get("hit_range_bound", np.nan),
        "clamped_frac_star": float(ds_arm.attrs["clamped_frac_star"]),
    }
    for k, comp in enumerate(COMPONENTS):
        row[f"delta_alpha_{comp}"] = float(delta_alpha[k])
        row[f"delta_alpha_std_{comp}"] = float(delta_alpha_std[k])
        row[f"optimum_offset_{comp}"] = float(optimum_offset[k])
    return row


def build_df(ref_arm: str, arms, gauges, data) -> pd.DataFrame:
    rows = []
    for staid in gauges["staid"]:
        ds_ref = data.get((ref_arm, staid))
        if ds_ref is None:
            continue
        for arm in arms:
            ds_arm = data.get((arm, staid))
            if ds_arm is None:
                continue
            rows.append(arm_row(ref_arm, arm, staid, ds_ref, ds_arm))
    return pd.DataFrame(rows)


def build_spread_df(df: pd.DataFrame) -> pd.DataFrame:
    """Per gauge per component: spread of delta_alpha (trained points) and of
    optimum_offset (optima) across arms, and their ratio."""
    rows = []
    for staid, sub in df.groupby("staid"):
        for comp in COMPONENTS:
            trained = sub[f"delta_alpha_{comp}"].to_numpy(dtype=float)
            optima = sub[f"optimum_offset_{comp}"].to_numpy(dtype=float)
            spread_trained = float(trained.max() - trained.min())
            spread_optima = float(optima.max() - optima.min())
            ratio = spread_optima / spread_trained if spread_trained != 0 else np.nan
            rows.append(dict(staid=staid, component=comp, n_arms=int(len(sub)),
                              spread_trained=spread_trained, spread_optima=spread_optima, ratio=ratio))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- figure
def fig_arms(out: Path, df: pd.DataFrame, spread_df: pd.DataFrame, ref_arm: str, arms) -> Path:
    staids = sorted(df["staid"].unique())
    n_g = len(staids)
    colors = arm_colors(arms)
    arm_x = {a: i for i, a in enumerate(arms)}

    fig = plt.figure(figsize=(13, 3.6 * n_g + 3.6))
    gs = fig.add_gridspec(2 * n_g + 1, 3, height_ratios=[1] * (2 * n_g) + [1.3], hspace=0.7, wspace=0.35)

    top_left_axes = []  # (staid, ax) pairs, used to place row labels AFTER layout is finalized
    for gi, staid in enumerate(staids):
        sub = df[df["staid"] == staid].set_index("arm").reindex(arms)
        row_top = 2 * gi
        row_bot = 2 * gi + 1
        for ci, comp in enumerate(COMPONENTS):
            ax_top = fig.add_subplot(gs[row_top, ci])
            ax_bot = fig.add_subplot(gs[row_bot, ci])
            if ci == 0:
                top_left_axes.append((staid, ax_top))
            xs = [arm_x[a] for a in arms]
            cs = [colors[a] for a in arms]
            ax_top.scatter(xs, sub[f"delta_alpha_{comp}"], marker="o", c=cs, s=60, edgecolor="k", linewidth=0.4)
            ax_bot.scatter(xs, sub[f"optimum_offset_{comp}"], marker="*", c=cs, s=110, edgecolor="k", linewidth=0.4)
            for ax in (ax_top, ax_bot):
                ax.axhline(0.0, color="0.7", lw=0.7)
                ax.set_xticks(xs)
                ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
            ax_top.set_title(f"{comp} trained point", fontsize=8)
            ax_bot.set_title(f"{comp} optimum", fontsize=8)
            if ci == 0:
                ax_top.set_ylabel(f"delta_alpha\n(vs {ref_arm})", fontsize=7)
                ax_bot.set_ylabel(f"optimum offset\n(vs {ref_arm})", fontsize=7)

    ax_sum = fig.add_subplot(gs[2 * n_g, :])
    x = np.arange(n_g)
    width = 0.25
    comp_colors = {"n": "tab:blue", "p": "tab:orange", "q": "tab:green"}
    for ci, comp in enumerate(COMPONENTS):
        vals = [spread_df[(spread_df["staid"] == s) & (spread_df["component"] == comp)]["ratio"].iloc[0]
                if not spread_df[(spread_df["staid"] == s) & (spread_df["component"] == comp)].empty else np.nan
                for s in staids]
        ax_sum.bar(x + (ci - 1) * width, vals, width * 0.9, color=comp_colors[comp], label=comp)
    ax_sum.axhline(0.0, color="k", lw=0.6)
    ax_sum.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(staids, rotation=0, ha="center")
    ax_sum.set_ylabel("spread(optima) /\nspread(trained)", fontsize=9)
    ax_sum.set_title("(summary) near 0: optimum input-independent; near 1: optimum moves with input", fontsize=9)
    ax_sum.legend(fontsize=8)

    fig.suptitle(
        f"Cross-arm (inflow input) displacement, relative to reference arm '{ref_arm}'. "
        "circle: trained point offset; star: per-gauge optimum offset (both in log units).",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for staid, ax in top_left_axes:  # placed after tight_layout so positions are final
        pos = ax.get_position()
        fig.text(0.01, pos.y1 + 0.008, f"gauge {staid}", ha="left", va="bottom",
                  fontsize=10, fontweight="bold", transform=fig.transFigure)
    p = out / "arms.png"
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


def write_report(out: Path, df: pd.DataFrame, spread_df: pd.DataFrame, ref_arm, arms, gauges,
                  input_files, is_smoke: bool):
    lines = ["# Landscape cross-arm (inflow input) census (spec section 7.3)", ""]
    if is_smoke:
        lines.append(
            "**This run is a synthetic smoke test, not a real multi-input run.** It exercises the "
            "pipeline end to end; the numbers have no scientific meaning."
        )
        lines.append("")
    present = set(df["staid"]) if not df.empty else set()
    missing = [s for s in gauges["staid"] if s not in present]
    lines.append(f"Arms: {arms}. Reference arm: {ref_arm}.")
    lines.append(f"Gauges with data: {sorted(present)}. Gauges not yet available: {missing}.")
    lines.append("")

    lines.append("## Table 1: per-gauge per-arm displacement")
    lines.append("")
    t1_cols = [
        "arm", "staid", "is_ref", "n_common", "n_reach_arm", "n_reach_ref",
        "delta_alpha_n", "delta_alpha_p", "delta_alpha_q",
        "delta_alpha_std_n", "delta_alpha_std_p", "delta_alpha_std_q",
        "optimum_offset_n", "optimum_offset_p", "optimum_offset_q",
        "nse_star", "hit_range_bound", "clamped_frac_star",
    ]
    if df.empty:
        lines.append("(no arm netCDFs available yet)")
    else:
        lines.append(md_table(df, t1_cols))
    lines.append("")
    lines.append(
        f"delta_alpha(arm) is arm's trained point expressed as a log-multiplier offset from {ref_arm}'s "
        "trained point (reach-paired by comid); optimum_offset(arm) is the same for arm's own per-gauge "
        f"optimum: delta_alpha(arm) + alpha_star(arm) - alpha_star({ref_arm}). Both are exactly zero for "
        f"{ref_arm} itself by construction."
    )
    lines.append("")

    lines.append("## Table 2: spread across arms and the discriminating ratio")
    lines.append("")
    if spread_df.empty:
        lines.append("(no arm netCDFs available yet)")
    else:
        lines.append(md_table(spread_df, ["staid", "component", "n_arms", "spread_trained", "spread_optima", "ratio"]))
    lines.append("")
    lines.append(
        "spread_trained and spread_optima are max - min (log units) of delta_alpha and optimum_offset "
        "across arms, per gauge per component. ratio = spread_optima / spread_trained: near 0 means the "
        "channel-parameter optimum for that gauge and component is input-independent (only the batch "
        "solution moved with the inflow); near 1 means the optimum moved with the input as much as the "
        "trained point did, i.e. the channel parameters are absorbing inflow bias. A nan ratio means "
        "spread_trained was exactly zero (no displacement to normalise by)."
    )
    lines.append("")
    lines.append("## Input files")
    lines.append("")
    for f in input_files:
        lines.append(f"- {f}")
    lines.append("")
    (out / "ARMS.md").write_text("\n".join(lines))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ref", type=str, default=None, help="reference arm name (default: first arm in experiment.yaml)")
    ap.add_argument("--smoke-test", action="store_true",
                     help="label the report as a synthetic smoke test (no scientific meaning)")
    args = ap.parse_args()

    run_dir = args.run_dir
    out = args.out if args.out is not None else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    manifest, spec_arms, arms, gauges, data, input_files = load(run_dir)
    ref_arm = args.ref if args.ref is not None else spec_arms[0]
    if ref_arm not in arms:
        raise SystemExit(f"reference arm {ref_arm!r} not found among run arms {arms}")
    print(f"arms: {arms}; reference arm: {ref_arm}; gauges: {list(gauges['staid'])}; datasets found: {len(data)}")

    df = build_df(ref_arm, arms, gauges, data)
    spread_df = build_spread_df(df) if not df.empty else pd.DataFrame()

    written = []
    if not df.empty:
        df.to_csv(out / "arms_per_gauge.csv", index=False)
        written.append(out / "arms_per_gauge.csv")
        written.append(fig_arms(out, df, spread_df, ref_arm, arms))

    write_report(out, df, spread_df, ref_arm, arms, gauges, input_files, args.smoke_test)
    written.append(out / "ARMS.md")

    if not spread_df.empty:
        print(md_table(spread_df, ["staid", "component", "spread_trained", "spread_optima", "ratio"]))
    else:
        print("no arms available yet")

    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
