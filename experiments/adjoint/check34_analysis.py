#!/usr/bin/env python
"""CHECK 3 / CHECK 4 analysis for the adjoint uh-retro w90 vs w180 experiment pair.

CHECK 3: truncation of the volume functional (short eval window clips long lags).
CHECK 4: mechanism of residual mass loss via the clamp floor (floor_frac).

Usage:
    check34_analysis.py <run_dir_w90> <run_dir_w180>

Writes PNGs and CHECK34.md into <run_dir_w180>/figures/.

Run with: /home/tbindas/projects/ddr/.venv/bin/python check34_analysis.py <w90> <w180>
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ARM = "uh-retro"
STAIDS_C4B = ("06354000", "06447000")


# ---------------------------------------------------------------------------
# small numeric helpers (nan-safe, empty-safe)
# ---------------------------------------------------------------------------

def safe_nanmedian(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmedian(x))


def safe_frac(mask):
    mask = np.asarray(mask)
    if mask.size == 0:
        return np.nan
    return float(np.mean(mask))


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))


# ---------------------------------------------------------------------------
# per-gauge netCDF loading
# ---------------------------------------------------------------------------

def compute_lag_low_days(ds):
    """nanmean over anchors with anchor_is_high == 0 of kernel_mean_lag_days, per reach."""
    anchor_is_high = ds["anchor_is_high"].values
    kmld = ds["kernel_mean_lag_days"].values  # (anchor, reach)
    low_idx = np.where(anchor_is_high == 0)[0]
    n_reach = kmld.shape[1]
    if low_idx.size == 0:
        return np.full(n_reach, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(kmld[low_idx, :], axis=0)


def build_reach_frame(ds):
    n = ds["COMID"].values.shape[0]
    return pd.DataFrame(
        {
            "reach_row": np.arange(n),
            "COMID": ds["COMID"].values.astype(np.int64),
            "dist_to_gauge_m": ds["dist_to_gauge_m"].values.astype(float),
            "q_prime_mean": ds["q_prime_mean"].values.astype(float),
            "is_gauge_reach": ds["is_gauge_reach"].values.astype(int),
            "volume_sens": ds["volume_sens"].values.astype(float),
            "volume_sens_lag_aware": ds["volume_sens_lag_aware"].values.astype(float),
            "floor_frac": ds["floor_frac"].values.astype(float),
            "lag_low_days": compute_lag_low_days(ds),
        }
    )


def load_gauges(run90, run180):
    """Read gauges.csv (from run180), open each staid's nc in both runs, merge on COMID.

    Returns (all_df, present_staids, notes) where notes is a list of strings
    describing skipped gauges and why.
    """
    gauges_csv = run180 / "gauges.csv"
    gauges_df = pd.read_csv(gauges_csv, dtype={"staid": str})
    staids = gauges_df["staid"].tolist()

    frames = []
    present = []
    notes = []
    for staid in staids:
        p90 = run90 / ARM / "gauges" / f"{staid}.nc"
        p180 = run180 / ARM / "gauges" / f"{staid}.nc"
        if not p90.exists() or not p180.exists():
            missing_in = []
            if not p90.exists():
                missing_in.append("w90")
            if not p180.exists():
                missing_in.append("w180")
            notes.append(f"{staid}: skipped, not yet present in {', '.join(missing_in)}")
            continue
        try:
            ds90 = xr.open_dataset(p90, decode_timedelta=False)
        except Exception as e:
            notes.append(f"{staid}: skipped, failed to open w90 file ({e})")
            continue
        try:
            ds180 = xr.open_dataset(p180, decode_timedelta=False)
        except Exception as e:
            ds90.close()
            notes.append(f"{staid}: skipped, failed to open w180 file ({e})")
            continue
        try:
            df90 = build_reach_frame(ds90)
            df180 = build_reach_frame(ds180)
        except Exception as e:
            notes.append(f"{staid}: skipped, failed to read variables ({e})")
            ds90.close()
            ds180.close()
            continue
        finally:
            ds90.close()
            ds180.close()

        merge_cols = [
            "COMID",
            "reach_row",
            "q_prime_mean",
            "is_gauge_reach",
            "volume_sens",
            "volume_sens_lag_aware",
            "floor_frac",
            "lag_low_days",
        ]
        merged = df90.merge(df180[merge_cols], on="COMID", suffixes=("_w90", "_w180"))
        if len(merged) != len(df90) or len(merged) != len(df180):
            notes.append(
                f"{staid}: COMID merge dropped reaches "
                f"(w90 n={len(df90)}, w180 n={len(df180)}, merged n={len(merged)})"
            )
        merged.insert(0, "staid", staid)
        frames.append(merged)
        present.append(staid)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
    else:
        all_df = pd.DataFrame()
    return all_df, present, notes


# ---------------------------------------------------------------------------
# markdown table writer (no tabulate)
# ---------------------------------------------------------------------------

def df_to_markdown(df):
    if df is None or df.empty:
        return "(no rows)\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("NaN")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CHECK 3: Table C3
# ---------------------------------------------------------------------------

def c3_row(label, sub):
    vs90 = sub["volume_sens_w90"].values
    vsla90 = sub["volume_sens_lag_aware_w90"].values
    vs180 = sub["volume_sens_w180"].values
    lag90 = sub["lag_low_days_w90"].values

    n = len(sub)
    fin_la = np.isfinite(vsla90)
    recov_mask = (vs90 < 0.5) & (lag90 < 30)
    recov_n = int(np.sum(recov_mask))
    recov_frac = safe_frac(vs180[recov_mask] >= 0.9) if recov_n > 0 else np.nan

    return {
        "staid": label,
        "n_reach": n,
        "median_vs_w90": safe_nanmedian(vs90),
        "median_vsla_w90": safe_nanmedian(vsla90),
        "median_vs_w180": safe_nanmedian(vs180),
        "frac_lt05_vs_w90": safe_frac(vs90 < 0.5),
        "frac_lt05_vsla_w90": safe_frac(vsla90[fin_la] < 0.5) if fin_la.any() else np.nan,
        "frac_lt05_vs_w180": safe_frac(vs180 < 0.5),
        "recovery_n": recov_n,
        "recovery_frac_ge09": recov_frac,
    }


def build_table_c3(all_df):
    rows = [c3_row(g, sub) for g, sub in all_df.groupby("staid", sort=False)]
    rows.append(c3_row("TOTAL", all_df))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CHECK 4: Table C4 (per window) and pooled 2x2 contingency
# ---------------------------------------------------------------------------

def c4_reach_flags(sub, suffix):
    vs = sub[f"volume_sens_{suffix}"].values
    vsla = sub[f"volume_sens_lag_aware_{suffix}"].values
    ff = sub[f"floor_frac_{suffix}"].values
    qpm = sub[f"q_prime_mean_{suffix}"].values
    lag = sub[f"lag_low_days_{suffix}"].values

    fin = np.isfinite(vsla)
    vs_or_la = np.where(fin, vsla, vs)
    loss = np.where(fin, vsla < 0.5, vs < 0.5)
    wet = qpm >= 1e-3
    quick = lag < 3
    unexplained = loss & wet & quick
    return vs_or_la, ff, loss, wet, quick, unexplained


def c4_row(label, sub, suffix):
    vs_or_la, ff, loss, wet, quick, unexplained = c4_reach_flags(sub, suffix)

    nonloss = ~loss
    rho = spearman(vs_or_la[wet], ff[wet])

    row = {
        "staid": label,
        "n_reach": len(sub),
        "n_loss": int(loss.sum()),
        "n_unexplained": int(unexplained.sum()),
        "median_floor_loss": safe_nanmedian(ff[loss]),
        "median_floor_nonloss": safe_nanmedian(ff[nonloss]),
        "frac_unexplained_floor_gt05": (
            safe_frac(ff[unexplained] > 0.5) if unexplained.sum() > 0 else np.nan
        ),
        "frac_nonloss_floor_gt05": (
            safe_frac(ff[nonloss] > 0.5) if nonloss.sum() > 0 else np.nan
        ),
        "spearman_rho_wet": rho,
    }

    loss_w = loss[wet]
    floor_w = ff[wet] > 0.5
    cont = {
        "loss_yes_floor_yes": int(np.sum(loss_w & floor_w)),
        "loss_yes_floor_no": int(np.sum(loss_w & ~floor_w)),
        "loss_no_floor_yes": int(np.sum(~loss_w & floor_w)),
        "loss_no_floor_no": int(np.sum(~loss_w & ~floor_w)),
    }
    return row, cont


def build_table_c4(all_df, suffix):
    rows = []
    cont_total = {"loss_yes_floor_yes": 0, "loss_yes_floor_no": 0,
                  "loss_no_floor_yes": 0, "loss_no_floor_no": 0}
    for g, sub in all_df.groupby("staid", sort=False):
        row, cont = c4_row(g, sub, suffix)
        rows.append(row)
        for k in cont_total:
            cont_total[k] += cont[k]
    return pd.DataFrame(rows), cont_total


def contingency_to_df(cont):
    return pd.DataFrame(
        {
            "": ["loss = yes", "loss = no"],
            "floor_frac>0.5 = yes": [cont["loss_yes_floor_yes"], cont["loss_no_floor_yes"]],
            "floor_frac>0.5 = no": [cont["loss_yes_floor_no"], cont["loss_no_floor_no"]],
        }
    )


# ---------------------------------------------------------------------------
# Table C4b: unexplained reaches at named gauges (w180)
# ---------------------------------------------------------------------------

def build_table_c4b(all_df, suffix="w180", staids=STAIDS_C4B):
    frames = []
    absent = []
    empty_for = []
    for s in staids:
        sub = all_df[all_df["staid"] == s]
        if sub.empty:
            absent.append(s)
            continue
        _, _, _, _, _, unexplained = c4_reach_flags(sub, suffix)
        rows = sub.loc[unexplained]
        if rows.empty:
            empty_for.append(s)
            continue
        tbl = pd.DataFrame(
            {
                "staid": rows["staid"].values,
                "reach_row": rows[f"reach_row_{suffix}"].values,
                "COMID": rows["COMID"].values,
                "dist_km": rows["dist_to_gauge_m"].values / 1000.0,
                "q_prime_mean": rows[f"q_prime_mean_{suffix}"].values,
                "lag_low_days": rows[f"lag_low_days_{suffix}"].values,
                "volume_sens": rows[f"volume_sens_{suffix}"].values,
                "volume_sens_lag_aware": rows[f"volume_sens_lag_aware_{suffix}"].values,
                "floor_frac": rows[f"floor_frac_{suffix}"].values,
            }
        )
        frames.append(tbl)

    cols = ["staid", "reach_row", "COMID", "dist_km", "q_prime_mean", "lag_low_days",
            "volume_sens", "volume_sens_lag_aware", "floor_frac"]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return result, absent, empty_for


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_check3(all_df, c3_df, present, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    x = all_df["volume_sens_w90"].values
    y = all_df["volume_sens_w180"].values
    c = all_df["lag_low_days_w90"].values
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=12)
    finite_vals = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if finite_vals.size:
        lo, hi = float(np.min(finite_vals)), float(np.max(finite_vals))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("w90 volume_sens")
    ax.set_ylabel("w180 volume_sens")
    ax.set_title("(a) w90 vs w180 volume_sens")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("lag_low_days (w90)")

    ax = axes[1]
    mask = np.isfinite(all_df["volume_sens_lag_aware_w90"].values)
    x2 = all_df["volume_sens_w90"].values[mask]
    y2 = all_df["volume_sens_lag_aware_w90"].values[mask]
    c2 = all_df["dist_to_gauge_m"].values[mask] / 1000.0
    sc2 = ax.scatter(x2, y2, c=c2, cmap="plasma", s=12)
    finite_vals2 = np.concatenate([x2, y2]) if x2.size else np.array([])
    if finite_vals2.size:
        lo, hi = float(np.min(finite_vals2)), float(np.max(finite_vals2))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("w90 raw volume_sens")
    ax.set_ylabel("w90 lag-aware volume_sens")
    ax.set_title("(b) raw vs lag-aware, w90")
    cb2 = plt.colorbar(sc2, ax=ax)
    cb2.set_label("dist_to_gauge_m / 1000 (km)")

    ax = axes[2]
    gauge_rows = c3_df[c3_df["staid"] != "TOTAL"]
    labels = gauge_rows["staid"].tolist()
    xpos = np.arange(len(labels))
    width = 0.25
    ax.bar(xpos - width, gauge_rows["frac_lt05_vs_w90"], width, label="raw w90 frac<0.5")
    ax.bar(xpos, gauge_rows["frac_lt05_vsla_w90"], width, label="lag-aware w90 frac<0.5")
    ax.bar(xpos + width, gauge_rows["frac_lt05_vs_w180"], width, label="raw w180 frac<0.5")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("fraction of reaches")
    ax.set_title("(c) frac of reaches with sensitivity < 0.5")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_check4(all_df, c4_w180_df, present, out_path):
    rho_by_staid = dict(zip(c4_w180_df["staid"], c4_w180_df["spearman_rho_wet"]))
    n = len(present)
    ncols = min(3, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

    for i, g in enumerate(present):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = all_df[all_df["staid"] == g]
        vs_or_la, ff, loss, wet, quick, unexplained = c4_reach_flags(sub, "w180")
        qpm = sub["q_prime_mean_w180"].values
        sizes = 15 + 8 * (np.log10(np.clip(qpm[wet], 1e-3, None)) + 3)
        ax.scatter(ff[wet], vs_or_la[wet], s=sizes, alpha=0.6, edgecolor="none")
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("floor_frac")
        ax.set_ylabel("volume_sens (lag-aware, fallback raw)")
        rho = rho_by_staid.get(g, np.nan)
        rho_str = f"{rho:.3f}" if pd.notna(rho) else "NaN"
        ax.set_title(f"{g} (rho={rho_str})")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <run_dir_w90> <run_dir_w180>", file=sys.stderr)
        sys.exit(1)

    run90 = Path(sys.argv[1])
    run180 = Path(sys.argv[2])
    figdir = run180 / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    md_path = figdir / "CHECK34.md"

    gauges_csv_path = run180 / "gauges.csv"
    gauges_df_full = pd.read_csv(gauges_csv_path, dtype={"staid": str})
    n_listed = len(gauges_df_full)

    all_df, present, notes = load_gauges(run90, run180)

    if all_df.empty:
        lines = [
            "# CHECK 3 / CHECK 4 analysis: adjoint uh-retro, w90 vs w180",
            "",
            f"No gauges from {gauges_csv_path} were readable in both runs yet "
            f"({n_listed} listed).",
            "",
            "## Notes",
            *[f"- {n}" for n in notes],
            "",
            "## Input files",
            f"- {run90 / 'gauges.csv'}",
            f"- {run180 / 'gauges.csv'}",
        ]
        md_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote {md_path} (no gauges available yet)")
        return

    # ---- CHECK 3 ----
    c3_df = build_table_c3(all_df)
    fig3_path = figdir / "check3_truncation.png"
    fig_check3(all_df, c3_df, present, fig3_path)

    # ---- CHECK 4 ----
    c4_w180_df, cont_w180 = build_table_c4(all_df, "w180")
    c4_w90_df, cont_w90 = build_table_c4(all_df, "w90")
    cont_w180_df = contingency_to_df(cont_w180)
    cont_w90_df = contingency_to_df(cont_w90)
    c4b_df, c4b_absent, c4b_empty = build_table_c4b(all_df, "w180", STAIDS_C4B)

    fig4_path = figdir / "check4_floor.png"
    fig_check4(all_df, c4_w180_df, present, fig4_path)

    # ---- write CHECK34.md ----
    lines = []
    lines.append("# CHECK 3 / CHECK 4 analysis: adjoint uh-retro, w90 vs w180")
    lines.append("")
    lines.append(
        f"Gauges present in both runs: {len(present)} of {n_listed} listed in "
        f"{gauges_csv_path.name} ({', '.join(present)})."
    )
    if notes:
        lines.append("")
        lines.append("Skipped gauges:")
        for n in notes:
            lines.append(f"- {n}")
    lines.append("")

    lines.append("## CHECK 3: truncation of the volume functional")
    lines.append("")
    lines.append("### Table C3")
    lines.append("")
    lines.append(df_to_markdown(c3_df))
    lines.append(
        "Each row is one gauge's reaches; TOTAL pools all reaches across all present "
        "gauges. Recovery is computed over reaches with w90 volume_sens < 0.5 and "
        "w90 lag_low_days < 30, and reports the fraction and count of those reaches "
        "whose w180 volume_sens reaches at least 0.9."
    )
    lines.append("")

    lines.append("### Figure: check3_truncation.png")
    lines.append("")
    lines.append(
        "Panel (a) plots w90 volume_sens against w180 volume_sens for every reach at "
        "every gauge, colored by w90 lag_low_days, with a 1:1 reference line. Panel (b) "
        "plots w90 raw volume_sens against w90 lag-aware volume_sens, colored by "
        "distance to gauge in km, with a 1:1 reference line. Panel (c) shows, per "
        "gauge, the fraction of reaches with sensitivity below 0.5 for raw w90, "
        "lag-aware w90, and raw w180."
    )
    lines.append("")

    lines.append("## CHECK 4: mechanism of residual mass loss via the clamp floor")
    lines.append("")
    lines.append("### Table C4 (w180)")
    lines.append("")
    lines.append(df_to_markdown(c4_w180_df))
    lines.append(
        "loss is volume_sens_lag_aware < 0.5 where finite, else volume_sens < 0.5; "
        "wet is q_prime_mean >= 1e-3; quick is lag_low_days < 3; unexplained is "
        "loss and wet and quick. spearman_rho_wet is the Spearman correlation "
        "between the (lag-aware, fallback raw) sensitivity and floor_frac over wet "
        "reaches."
    )
    lines.append("")

    lines.append("### Table C4 (w90)")
    lines.append("")
    lines.append(df_to_markdown(c4_w90_df))
    lines.append(
        "Same definitions as Table C4 (w180), computed entirely from w90 quantities."
    )
    lines.append("")

    lines.append("### Pooled 2x2 contingency, loss x floor_frac>0.5, over wet reaches (w180)")
    lines.append("")
    lines.append(df_to_markdown(cont_w180_df))
    lines.append("Counts are pooled across all present gauges' wet reaches, w180 quantities.")
    lines.append("")

    lines.append("### Pooled 2x2 contingency, loss x floor_frac>0.5, over wet reaches (w90)")
    lines.append("")
    lines.append(df_to_markdown(cont_w90_df))
    lines.append("Counts are pooled across all present gauges' wet reaches, w90 quantities.")
    lines.append("")

    lines.append("### Table C4b: unexplained reaches at gauges 06354000 and 06447000 (w180)")
    lines.append("")
    lines.append(df_to_markdown(c4b_df))
    detail_bits = []
    if c4b_absent:
        detail_bits.append(f"not present in either run yet: {', '.join(c4b_absent)}")
    if c4b_empty:
        detail_bits.append(f"present but zero unexplained reaches: {', '.join(c4b_empty)}")
    if detail_bits:
        lines.append("; ".join(detail_bits) + ".")
    else:
        lines.append(
            "Both gauges were present with at least one unexplained reach listed above."
        )
    lines.append("")

    lines.append("### Figure: check4_floor.png")
    lines.append("")
    lines.append(
        "One subplot per present gauge, w180 quantities, wet reaches only. x axis is "
        "floor_frac, y axis is volume_sens_lag_aware with fallback to raw volume_sens "
        "where lag-aware is undefined, marker size scaled to log10(q_prime_mean). "
        "Reference lines are drawn at y=0.5 and x=0.5. Each subplot title reports the "
        "gauge id and the Spearman rho between y and floor_frac over its wet reaches."
    )
    lines.append("")

    lines.append("## Input files")
    lines.append(f"- {run90 / 'gauges.csv'}")
    lines.append(f"- {run180 / 'gauges.csv'}")
    for staid in present:
        lines.append(f"- {run90 / ARM / 'gauges' / (staid + '.nc')}")
        lines.append(f"- {run180 / ARM / 'gauges' / (staid + '.nc')}")
    lines.append(f"- {fig3_path}")
    lines.append(f"- {fig4_path}")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    print(f"Wrote {fig3_path}")
    print(f"Wrote {fig4_path}")
    print(f"Gauges present: {present}")


if __name__ == "__main__":
    main()
