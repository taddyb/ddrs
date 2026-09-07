#!/usr/bin/env python
"""Figures for the alpha-landscape study (src/experiment/landscape).

Reads only the study output directory (manifest.json, gauges.csv,
<arm>/gauges/<staid>.nc) and writes PNGs + LANDSCAPE.md to <run_dir>/figures/.
Gauges whose netCDF has not been written yet (the run is still in progress)
are skipped.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/plots.py <run_dir>
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


def arm_colors(arms):
    cmap = plt.get_cmap("tab10")
    return {a: cmap(i) for i, a in enumerate(arms)}


# --------------------------------------------------------------- per-gauge figure
def plot_plane(ax, ds, plane_idx, plane_name):
    axis_a = ds["grid_axis_a"].values[plane_idx]
    axis_b = ds["grid_axis_b"].values[plane_idx]
    nse = ds["grid_nse"].values[plane_idx]
    clamped = ds["grid_clamped"].values[plane_idx]
    nse_star = float(ds["nse_star"].values)
    alpha_star = ds["alpha_star"].values
    coord_trained = ds["coord_trained"].values

    nan_min = np.nanmin(nse)
    nan_max = np.nanmax(nse)
    vmin = max(-1.0, nan_min)
    vmax = nan_max if np.isfinite(nan_max) and nan_max > vmin else vmin + 1e-6
    levels = np.linspace(vmin, vmax, 21)
    z = np.clip(nse, -1.0, None).T
    cf = ax.contourf(axis_a, axis_b, z, levels=levels, cmap="viridis")
    thr = nse_star - 0.02
    if vmin < thr < vmax:
        ax.contour(axis_a, axis_b, z, levels=[thr], colors="white", linewidths=1.1)

    clamped_mask = (clamped.T > 0.05).astype(float)
    if clamped_mask.max() > 0:
        ax.contourf(axis_a, axis_b, clamped_mask, levels=[0.5, 1.5], colors="none", hatches=["////"])

    if plane_name == "stiff-sloppy":
        trained_xy = (float(coord_trained[0]), float(coord_trained[2]))
        star_xy = (0.0, 0.0)
        ax.set_xlabel("offset s along v1 (stiff)")
        ax.set_ylabel("offset t along v3 (sloppy)")
    else:
        a_name, b_name = plane_name.split("-")
        i, j = ALPHA_INDEX[a_name], ALPHA_INDEX[b_name]
        trained_xy = (0.0, 0.0)
        star_xy = (float(alpha_star[i]), float(alpha_star[j]))
        newton_alpha = ds["newton_alpha"].values
        ax.plot(newton_alpha[:, i], newton_alpha[:, j], color="0.5", lw=0.9, zorder=4)
        eigvec = ds["eigvec_star"].values
        half_width = ds["half_width"].values
        v1, v3 = eigvec[:, 0], eigvec[:, 2]
        w1 = min(float(half_width[0, 0]), 1.5)
        w3 = min(float(half_width[0, 2]), 1.5)
        ax.annotate("", xy=(star_xy[0] + v1[i] * w1, star_xy[1] + v1[j] * w1), xytext=star_xy,
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.3, linestyle="-"))
        ax.annotate("", xy=(star_xy[0] + v3[i] * w3, star_xy[1] + v3[j] * w3), xytext=star_xy,
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.3, linestyle="--"))
        ax.set_xlabel(f"ln multiplier {a_name}")
        ax.set_ylabel(f"ln multiplier {b_name}")

    ax.plot(*trained_xy, marker="x", color="k", ms=8, mew=2, zorder=5)
    ax.plot(*star_xy, marker="*", color="red", ms=13, zorder=5)
    ax.set_title(plane_name, fontsize=9)
    return cf


def fig_landscape_gauge(out: Path, arms, staid: str, data):
    rows = [a for a in arms if (a, staid) in data]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 4.2 * len(rows) + 0.6), squeeze=False)
    row_labels = []
    for r, arm in enumerate(rows):
        ds = data[(arm, staid)]
        planes = ds.attrs["plane_names"].split(",")
        nse0 = float(ds["nse0"].values)
        nse_star = float(ds["nse_star"].values)
        n_reach = int(ds.attrs["n_reach"])
        for c, plane_name in enumerate(planes):
            ax = axes[r, c]
            cf = plot_plane(ax, ds, c, plane_name)
            fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03, label="NSE" if c == 3 else None)
        row_labels.append(f"{staid}  arm={arm}  NSE {nse0:.3f} -> {nse_star:.3f}  n_reach={n_reach}")
    fig.suptitle(
        "black x: trained point (alpha=0)   red star: optimum   "
        "solid arrow: v1 (stiff)   dashed arrow: v3 (sloppy)",
        fontsize=8, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86 if len(rows) == 1 else 0.92))
    for r, label in enumerate(row_labels):
        pos = axes[r, 0].get_position()
        fig.text(pos.x0, pos.y1 + 0.03, label, ha="left", va="bottom", fontsize=9, fontweight="bold")
    p = out / "figures" / f"landscape_{staid}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# --------------------------------------------------------------------- gauge stats
def build_stats_df(arms, gauges, data) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for staid in gauges["staid"]:
            ds = data.get((arm, staid))
            if ds is None:
                continue
            alpha_star = ds["alpha_star"].values
            eigval = ds["eigval_star"].values
            eigvec = ds["eigvec_star"].values
            coord_trained = ds["coord_trained"].values
            half_width = ds["half_width"].values
            celerity_dir = ds["celerity_dir"].values
            hess_star = ds["hess_star"].values
            newton_grad_norm = ds["newton_grad_norm"].values
            v1 = eigvec[:, 0]
            denom = np.linalg.norm(v1) * np.linalg.norm(celerity_dir)
            cos_v1_cel = float(abs(np.dot(v1, celerity_dir) / denom)) if denom > 0 else np.nan
            ck_wk = np.abs(coord_trained) / half_width[0]
            asym = np.abs(hess_star - hess_star.T)
            hmax = np.abs(hess_star).max()
            hess_asym_ratio = float(asym.max() / hmax) if hmax > 0 else np.nan
            min_eigval = float(eigval.min())
            grad_ratio = (
                float(newton_grad_norm[-1] / newton_grad_norm[0])
                if newton_grad_norm[0] != 0
                else np.nan
            )
            rows.append(dict(
                arm=arm, staid=staid,
                n_reach=int(ds.attrs["n_reach"]),
                sigma=float(ds.attrs["sigma_obs_training"]),
                loss0=float(ds["loss0"].values), loss_star=float(ds["loss_star"].values),
                nse0=float(ds["nse0"].values), nse_star=float(ds["nse_star"].values),
                alpha_star_n=float(alpha_star[0]), alpha_star_p=float(alpha_star[1]), alpha_star_q=float(alpha_star[2]),
                mult_n=float(np.exp(alpha_star[0])), mult_p=float(np.exp(alpha_star[1])), mult_q=float(np.exp(alpha_star[2])),
                lambda1=float(eigval[0]), lambda2=float(eigval[1]), lambda3=float(eigval[2]),
                v1_n=float(v1[0]), v1_p=float(v1[1]), v1_q=float(v1[2]),
                cos_v1_cel=cos_v1_cel,
                coord1=float(coord_trained[0]), coord2=float(coord_trained[1]), coord3=float(coord_trained[2]),
                hw1=float(half_width[0, 0]), hw2=float(half_width[0, 1]), hw3=float(half_width[0, 2]),
                ck_wk1=float(ck_wk[0]), ck_wk2=float(ck_wk[1]), ck_wk3=float(ck_wk[2]),
                newton_iters=int(ds.sizes["newton"] - 1),
                grad_ratio=grad_ratio,
                clamped_frac_star=float(ds.attrs["clamped_frac_star"]),
                hess_asym_ratio=hess_asym_ratio,
                min_eigval=min_eigval,
                min_eigval_sign=("positive" if min_eigval > 0 else ("negative" if min_eigval < 0 else "zero")),
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- summary
def fig_summary(out: Path, df: pd.DataFrame, colors):
    staids = sorted(df["staid"].unique())
    arms = sorted(df["arm"].unique())
    x = np.arange(len(staids))
    n_arm = max(len(arms), 1)
    width = 0.8 / n_arm

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))

    ax = axes[0]
    for ai, arm in enumerate(arms):
        sub = df[df["arm"] == arm].set_index("staid").reindex(staids)
        xpos = x + (ai - (n_arm - 1) / 2) * width
        ax.bar(xpos - width * 0.22, sub["loss0"], width * 0.42, color=colors[arm], label=f"{arm} loss0")
        ax.bar(xpos + width * 0.22, sub["loss_star"], width * 0.42, color=colors[arm], alpha=0.5, hatch="//", label=f"{arm} loss*")
    ax.set_xticks(x)
    ax.set_xticklabels(staids, rotation=30, ha="right")
    ax.set_ylabel("loss")
    ax.set_title("loss0 vs loss_star per gauge")
    ax.legend(fontsize=6)

    ax = axes[1]
    markers = ["o", "s", "^"]
    for ai, arm in enumerate(arms):
        sub = df[df["arm"] == arm].set_index("staid").reindex(staids)
        xpos = x + (ai - (n_arm - 1) / 2) * width
        for k, col in enumerate(["lambda1", "lambda2", "lambda3"]):
            ax.scatter(xpos, sub[col], marker=markers[k], color=colors[arm],
                       label=f"{arm} lambda{k + 1}" if True else None, s=30)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(staids, rotation=30, ha="right")
    ax.set_title("Hessian eigenvalues at alpha_star")
    ax.set_ylabel("eigenvalue (log)")
    ax.legend(fontsize=5.5, ncol=1)

    ax = axes[2]
    for ai, arm in enumerate(arms):
        sub = df[df["arm"] == arm].set_index("staid").reindex(staids)
        xpos = x + (ai - (n_arm - 1) / 2) * width
        for k, col in enumerate(["ck_wk1", "ck_wk2", "ck_wk3"]):
            ax.scatter(xpos, sub[col], marker=markers[k], color=colors[arm], s=30)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(staids, rotation=30, ha="right")
    ax.set_title("|c_k| / half_width(tol0) : below 1 is inside the behavioural set")

    ax = axes[3]
    for ai, arm in enumerate(arms):
        sub = df[df["arm"] == arm].set_index("staid").reindex(staids)
        xpos = x + (ai - (n_arm - 1) / 2) * width
        ax.bar(xpos, sub["cos_v1_cel"], width * 0.8, color=colors[arm], label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels(staids, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("|cos(v1, celerity_dir)|")
    ax.legend(fontsize=7)

    fig.tight_layout()
    p = out / "figures" / "landscape_summary.png"
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
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out: Path, df: pd.DataFrame, arms, gauges, input_files):
    lines = ["# Landscape report", ""]
    present = set(df["staid"]) if not df.empty else set()
    missing = [s for s in gauges["staid"] if s not in present]
    lines.append(f"Gauges with data: {sorted(present)}. Gauges not yet available: {missing}.")
    lines.append("")
    lines.append("## Table 1: per-gauge landscape summary")
    lines.append("")
    t1_cols = [
        "arm", "staid", "n_reach", "sigma", "loss0", "loss_star", "nse0", "nse_star",
        "alpha_star_n", "alpha_star_p", "alpha_star_q",
        "mult_n", "mult_p", "mult_q",
        "lambda1", "lambda2", "lambda3",
        "v1_n", "v1_p", "v1_q",
        "cos_v1_cel",
        "coord1", "coord2", "coord3",
        "hw1", "hw2", "hw3",
        "ck_wk1", "ck_wk2", "ck_wk3",
        "newton_iters", "grad_ratio", "clamped_frac_star",
    ]
    if df.empty:
        lines.append("(no gauge netCDFs available yet)")
    else:
        lines.append(md_table(df, t1_cols))
    lines.append("")
    n_arms = df["arm"].nunique() if not df.empty else 0
    n_gauges = df["staid"].nunique() if not df.empty else 0
    lines.append(f"Table 1 has {len(df)} row(s) spanning {n_arms} arm(s) and {n_gauges} gauge(s).")
    lines.append(
        "Each row reports the trained-point and per-gauge-optimum loss and NSE, the optimum in "
        "log-multiplier space and as multipliers, the Hessian eigenvalues and leading eigenvector "
        "at alpha_star, the trained point's coordinates in that eigenbasis, the behavioural "
        "half-widths at the first tolerance, and the resulting normalized offsets."
    )
    lines.append("")
    lines.append("## Table 2: Hessian symmetry and curvature sign")
    lines.append("")
    t2_cols = ["arm", "staid", "hess_asym_ratio", "min_eigval", "min_eigval_sign"]
    if df.empty:
        lines.append("(no gauge netCDFs available yet)")
    else:
        lines.append(md_table(df, t2_cols))
    lines.append("")
    if not df.empty:
        n_pos = int((df["min_eigval_sign"] == "positive").sum())
        n_neg = int((df["min_eigval_sign"] == "negative").sum())
        n_zero = int((df["min_eigval_sign"] == "zero").sum())
        lines.append(
            f"Table 2 reports the relative Hessian asymmetry |H - H^T|_max / |H|_max at alpha_star "
            f"and the sign of the smallest eigenvalue for the same {len(df)} row(s) as Table 1."
        )
        lines.append(f"{n_pos} row(s) have a positive minimum eigenvalue, {n_neg} negative, and {n_zero} zero.")
    else:
        lines.append("No rows are available yet, so no symmetry check has been computed.")
        lines.append("This table will populate once at least one gauge netCDF has been written.")
    lines.append("")
    lines.append("## Input files")
    lines.append("")
    for f in input_files:
        lines.append(f"- {f}")
    lines.append("")
    (out / "figures" / "LANDSCAPE.md").write_text("\n".join(lines))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run_dir = args.run_dir
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    manifest, arms, gauges, data, input_files = load(run_dir)
    print(f"arms: {arms}; gauges: {list(gauges['staid'])}; datasets found: {len(data)}")

    colors = arm_colors(arms)
    written = []
    for staid in gauges["staid"]:
        p = fig_landscape_gauge(run_dir, arms, staid, data)
        if p is not None:
            written.append(p)

    df = build_stats_df(arms, gauges, data)
    if not df.empty:
        written.append(fig_summary(run_dir, df, colors))

    write_report(run_dir, df, arms, gauges, input_files)
    written.append(run_dir / "figures" / "LANDSCAPE.md")

    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
