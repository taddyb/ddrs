#!/usr/bin/env python
"""Seed-to-seed noise floor for the adjoint influence-map statistics.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/adjoint/seed_noise.py \
        <seed-run-dir> <cross-arm-run-dir> [--out figures]

Reads two adjoint study outputs:
  - a two-seed run (arms uh-seed42 / uh-seed43, same model/config, different
    training seed), the noise floor.
  - the cross-arm reference run (5 arms, different inflow sources), what the
    noise floor is being judged against.

Reuses experiments/adjoint/plots.py's exact functions for every quantity it
already defines (per_gauge_celerity, inherited_decomposition, vol_var,
upstream_list, load) rather than re-deriving them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import inherited_decomposition, load, per_gauge_celerity  # noqa: E402


# ----------------------------------------------------------------- new metric
def kernel_mass_at_gauge(arms, gauges, data):
    """Per (arm, staid, kind): kernel_mass AT the gauge's own reach, mean over
    anchors of that kind. Not in plots.py (which only pools kernel_mass over
    all reaches for the population figure), so defined once here."""
    rows = []
    for staid in gauges["staid"]:
        for arm in arms:
            ds = data.get((arm, staid))
            if ds is None or "kernel_mass" not in ds or "is_gauge_reach" not in ds:
                continue
            gidx = np.where(ds["is_gauge_reach"].values == 1)[0]
            if gidx.size == 0:
                continue
            gi = int(gidx[0])
            for kind, flag in [("high", 1), ("low", 0)]:
                sel = ds["anchor_is_high"].values == flag
                if not sel.any():
                    continue
                val = float(np.nanmean(ds["kernel_mass"].values[sel, gi]))
                rows.append(dict(arm=arm, staid=staid, kind=kind, kernel_mass_gauge=val))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- seed-run table
def build_seed_table(seed_dir: Path):
    manifest, arms, gauges, data = load(seed_dir)
    arms = [a for a in arms if any(k[0] == a for k in data)]
    if len(arms) != 2:
        raise SystemExit(f"expected exactly 2 seed arms with data, got {arms}")

    cel = per_gauge_celerity(arms, gauges, data)
    kmg = kernel_mass_at_gauge(arms, gauges, data)
    dec = inherited_decomposition(arms, gauges, data)

    # volume_sens_median (prefers volume_sens_wet via vol_var) is identical
    # across kind rows in per_gauge_celerity's output; take one per gauge.
    vol = cel[["arm", "staid", "volume_sens_median"]].drop_duplicates(subset=["arm", "staid"])

    if not dec.empty:
        share = (
            dec.assign(abs_in=dec.inherited.abs(), abs_tot=dec.total.abs())
            .groupby(["arm", "staid"], as_index=False)
            .agg(abs_in=("abs_in", "sum"), abs_tot=("abs_tot", "sum"), transfer_mean=("transfer_mean", "first"))
        )
        share["inherited_share"] = share.abs_in / share.abs_tot.replace(0, np.nan)
        share = share[["arm", "staid", "transfer_mean", "inherited_share"]]
    else:
        share = pd.DataFrame(columns=["arm", "staid", "transfer_mean", "inherited_share"])

    cel_p = cel.pivot_table(index=["arm", "staid"], columns="kind", values="celerity_m_s").reset_index()
    cel_p = cel_p.rename(columns={"low": "celerity_low", "high": "celerity_high"})
    kmg_p = kmg.pivot_table(index=["arm", "staid"], columns="kind", values="kernel_mass_gauge").reset_index()
    kmg_p = kmg_p.rename(columns={"low": "kernel_mass_low", "high": "kernel_mass_high"})

    merged = cel_p.merge(kmg_p, on=["arm", "staid"], how="outer")
    merged = merged.merge(vol, on=["arm", "staid"], how="outer")
    merged = merged.merge(share, on=["arm", "staid"], how="left")
    return arms, gauges, merged


RATIO_COLS = ["celerity_low", "celerity_high"]
DIFF_COLS = ["kernel_mass_low", "kernel_mass_high", "volume_sens_median", "transfer_mean", "inherited_share"]


def seed_deltas(arms, merged):
    seed42, seed43 = arms  # manifest order: uh-seed42, uh-seed43
    p1 = merged[merged.arm == seed42].set_index("staid")
    p2 = merged[merged.arm == seed43].set_index("staid")
    staids = sorted(set(p1.index) & set(p2.index))
    rows = []
    for s in staids:
        r1, r2 = p1.loc[s], p2.loc[s]
        row = dict(staid=s)
        for col in RATIO_COLS:
            v1, v2 = r1.get(col), r2.get(col)
            row[f"{seed42}_{col}"] = v1
            row[f"{seed43}_{col}"] = v2
            row[f"{col}_ratio_43_over_42"] = (
                v2 / v1 if pd.notna(v1) and pd.notna(v2) and v1 > 0 and v2 > 0 else np.nan
            )
        for col in DIFF_COLS:
            v1, v2 = r1.get(col), r2.get(col)
            row[f"{seed42}_{col}"] = v1
            row[f"{seed43}_{col}"] = v2
            row[f"{col}_diff_43_minus_42"] = v2 - v1 if pd.notna(v1) and pd.notna(v2) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- cross-arm
def cross_arm_celerity_ratio(cross_dir: Path):
    """Per gauge, per kind: max/min celerity_m_s across the cross-arm run's
    arms, the same per_gauge_celerity fit plots.py's population mode uses,
    just reduced to a max/min ratio instead of a boxplot."""
    manifest, arms, gauges, data = load(cross_dir)
    arms = [a for a in arms if any(k[0] == a for k in data)]
    cel = per_gauge_celerity(arms, gauges, data)
    rows = []
    for kind in ["low", "high"]:
        sub = cel[cel.kind == kind]
        for staid, g in sub.groupby("staid"):
            vals = g["celerity_m_s"].dropna()
            vals = vals[vals > 0]
            if len(vals) < 2:
                continue
            rows.append(dict(staid=staid, kind=kind, ratio=float(vals.max() / vals.min()), n_arms=int(len(vals))))
    return pd.DataFrame(rows), arms


# --------------------------------------------------------------- summaries
def summarize_ratio(s: pd.Series) -> dict:
    s = pd.Series(s).dropna()
    s = s[s > 0]
    n = len(s)
    if n == 0:
        return dict(n=0, median=np.nan, iqr=np.nan, max_abs_log_ratio=np.nan)
    q1, q3 = np.percentile(s, [25, 75])
    return dict(n=n, median=float(np.median(s)), iqr=float(q3 - q1), max_abs_log_ratio=float(np.max(np.abs(np.log10(s)))))


def summarize_diff(s: pd.Series) -> dict:
    s = pd.Series(s).dropna()
    n = len(s)
    if n == 0:
        return dict(n=0, median=np.nan, iqr=np.nan, max_abs=np.nan)
    q1, q3 = np.percentile(s, [25, 75])
    return dict(n=n, median=float(np.median(s)), iqr=float(q3 - q1), max_abs=float(np.max(np.abs(s))))


def build_markdown_table(deltas: pd.DataFrame, cross_ratio: pd.DataFrame) -> tuple[str, dict]:
    lines = [
        "| statistic | N | median | IQR | max \\|extreme\\| |",
        "|---|---|---|---|---|",
    ]
    stats = {}

    def add_ratio_row(label, series):
        d = summarize_ratio(series)
        stats[label] = d
        lines.append(
            f"| {label} | {d['n']} | {d['median']:.4g} | {d['iqr']:.4g} | "
            f"{d['max_abs_log_ratio']:.4g} (max \\|log10 ratio\\|) |"
            if d["n"] > 0
            else f"| {label} | 0 | - | - | - |"
        )

    def add_diff_row(label, series):
        d = summarize_diff(series)
        stats[label] = d
        lines.append(
            f"| {label} | {d['n']} | {d['median']:.4g} | {d['iqr']:.4g} | {d['max_abs']:.4g} (max \\|diff\\|) |"
            if d["n"] > 0
            else f"| {label} | 0 | - | - | - |"
        )

    add_ratio_row("celerity_low ratio (seed43/seed42)", deltas["celerity_low_ratio_43_over_42"])
    add_ratio_row("celerity_high ratio (seed43/seed42)", deltas["celerity_high_ratio_43_over_42"])
    add_ratio_row(
        "celerity_low cross-arm max/min ratio (5 arms)",
        cross_ratio.loc[cross_ratio.kind == "low", "ratio"],
    )
    add_ratio_row(
        "celerity_high cross-arm max/min ratio (5 arms)",
        cross_ratio.loc[cross_ratio.kind == "high", "ratio"],
    )
    add_diff_row("kernel_mass_low @ gauge diff (seed43-seed42)", deltas["kernel_mass_low_diff_43_minus_42"])
    add_diff_row("kernel_mass_high @ gauge diff (seed43-seed42)", deltas["kernel_mass_high_diff_43_minus_42"])
    add_diff_row("median volume_sens_wet diff (seed43-seed42)", deltas["volume_sens_median_diff_43_minus_42"])
    add_diff_row("transfer diff (seed43-seed42, downstream gauges)", deltas["transfer_mean_diff_43_minus_42"])
    add_diff_row("inherited_share diff (seed43-seed42, downstream gauges)", deltas["inherited_share_diff_43_minus_42"])

    return "\n".join(lines), stats


# ---------------------------------------------------------------- figure
def make_figure(out_dir: Path, arms, merged, deltas, cross_ratio):
    seed42, seed43 = arms
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) scatter seed42 vs seed43 celerity, both anchors, log-log, 1:1 line
    ax = axes[0]
    lo = deltas.dropna(subset=[f"{seed42}_celerity_low", f"{seed43}_celerity_low"])
    hi = deltas.dropna(subset=[f"{seed42}_celerity_high", f"{seed43}_celerity_high"])
    ax.scatter(lo[f"{seed42}_celerity_low"], lo[f"{seed43}_celerity_low"], s=22, color="tab:blue", alpha=0.75, label="low anchor")
    ax.scatter(hi[f"{seed42}_celerity_high"], hi[f"{seed43}_celerity_high"], s=22, color="tab:orange", alpha=0.75, label="high anchor")
    allv = pd.concat(
        [lo[f"{seed42}_celerity_low"], lo[f"{seed43}_celerity_low"], hi[f"{seed42}_celerity_high"], hi[f"{seed43}_celerity_high"]]
    )
    allv = allv[allv > 0]
    lim = (float(allv.min()) * 0.8, float(allv.max()) * 1.25) if len(allv) else (0.1, 10)
    ax.plot(lim, lim, "k--", lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"celerity, {seed42} (m/s)")
    ax.set_ylabel(f"celerity, {seed43} (m/s)")
    ax.set_title("Per-gauge celerity: seed42 vs seed43")
    ax.legend(fontsize=8)

    # (b) histogram of log10 seed ratio (both anchors) vs log10 cross-arm max/min ratio
    ax = axes[1]
    seed_log_ratio = np.log10(
        pd.concat(
            [
                deltas["celerity_low_ratio_43_over_42"].dropna(),
                deltas["celerity_high_ratio_43_over_42"].dropna(),
            ]
        )
    )
    cross_log_ratio = np.log10(cross_ratio["ratio"].dropna())
    bins = np.linspace(
        min(seed_log_ratio.min(), cross_log_ratio.min()) if len(seed_log_ratio) and len(cross_log_ratio) else -1,
        max(seed_log_ratio.max(), cross_log_ratio.max()) if len(seed_log_ratio) and len(cross_log_ratio) else 1,
        30,
    )
    ax.hist(seed_log_ratio, bins=bins, histtype="step", color="tab:blue", lw=1.6, label=f"seed43/seed42 (n={len(seed_log_ratio)})")
    ax.hist(cross_log_ratio, bins=bins, histtype="step", color="tab:red", lw=1.6, label=f"cross-arm max/min (n={len(cross_log_ratio)})")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("log10 celerity ratio")
    ax.set_ylabel("count (gauges x anchor kind)")
    ax.set_title("Seed noise vs cross-arm spread in celerity ratio")
    ax.legend(fontsize=8)

    # (c) scatter seed42 vs seed43 median volume_sens_wet, 1:1 line
    ax = axes[2]
    vv = deltas.dropna(subset=[f"{seed42}_volume_sens_median", f"{seed43}_volume_sens_median"])
    ax.scatter(vv[f"{seed42}_volume_sens_median"], vv[f"{seed43}_volume_sens_median"], s=22, color="tab:green", alpha=0.75)
    vals = pd.concat([vv[f"{seed42}_volume_sens_median"], vv[f"{seed43}_volume_sens_median"]])
    lim = (float(vals.min()) - 0.05, float(vals.max()) + 0.05) if len(vals) else (0, 1.2)
    ax.plot(lim, lim, "k--", lw=0.9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"median volume_sens_wet, {seed42}")
    ax.set_ylabel(f"median volume_sens_wet, {seed43}")
    ax.set_title("Per-gauge median volume sensitivity: seed42 vs seed43")

    fig.suptitle("Seed-to-seed noise floor, adjoint influence-map statistics", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "seed_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_dir", type=Path)
    ap.add_argument("cross_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arms, gauges, merged = build_seed_table(args.seed_dir)
    deltas = seed_deltas(arms, merged)
    cross_ratio, cross_arms = cross_arm_celerity_ratio(args.cross_dir)

    # gauges present in gauges.csv but missing from deltas (e.g. celerity fit failed)
    all_staids = set(gauges["staid"])
    skipped = sorted(all_staids - set(deltas["staid"]))

    table, stats = build_markdown_table(deltas, cross_ratio)
    header = (
        f"# Seed noise floor\n\n"
        f"Seed run: `{args.seed_dir}` (arms: {', '.join(arms)})\n\n"
        f"Cross-arm reference run: `{args.cross_dir}` (arms: {', '.join(cross_arms)})\n\n"
    )
    md = header + table + "\n"
    (args.out / "SEED_NOISE.md").write_text(md + "\n")
    print(md)

    deltas.to_csv(args.out / "seed_noise_per_gauge.csv", index=False)
    make_figure(args.out, arms, merged, deltas, cross_ratio)

    print(f"\nwrote {args.out / 'SEED_NOISE.md'}")
    print(f"wrote {args.out / 'seed_noise_per_gauge.csv'}")
    print(f"wrote {args.out / 'seed_noise.png'}")
    if skipped:
        print(f"\ngauges skipped from the seed table (no valid data in one/both arms): {skipped}")
    else:
        print("\nno gauges skipped from the seed table")


if __name__ == "__main__":
    main()
