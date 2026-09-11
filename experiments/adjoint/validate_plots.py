#!/usr/bin/env python
"""Method-validation figures for the adjoint-uh-validate study.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/adjoint/validate_plots.py <run_dir>

Reads, under <run_dir>:
  - gauges.csv
  - manifest.json
  - <arm>/validation/<staid>_<high|low>_day<N>.csv   (finite-difference checks)
  - <arm>/gauges/<staid>.nc                          (kernel vs hydraulic lag)

Writes PNGs and a Markdown report to <run_dir>/figures/. Tolerates a partially
complete run directory: any input file that is missing or unreadable is
skipped with a printed "skipping: ..." line, and the script still produces
whatever figures/tables the data on hand supports.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

# Full target validation-CSV schema (some columns may be absent in
# partially-written / older-schema files; missing ones are filled with NaN).
VALIDATION_COLUMNS = [
    "arm", "staid", "anchor_day_idx", "anchor_kind", "t0", "q_gauge_t0",
    "noise_floor_m3s", "reach_row", "comid", "dist_to_gauge_m", "lag_hours",
    "delta_m3s", "dq_predicted", "dq_actual", "dq_total_pred_m3s",
    "dq_total_actual_m3s", "abs_err_m3s", "rel_err", "unresolvable", "passed",
    "sweep_deltas", "sweep_dq_actual", "sweep_dq_forward", "sweep_dq_backward",
    "sweep_rel_err",
]
SWEEP_COLUMNS = ["sweep_deltas", "sweep_dq_actual", "sweep_dq_forward", "sweep_dq_backward", "sweep_rel_err"]
FNAME_RE = re.compile(r"^(?P<staid>.+)_(?P<kind>high|low)_day(?P<day>\d+)\.csv$")

REL_ERR_LINE_DEFAULT = 0.02
LIMIT_PASS_REL_TOL = 0.05
KERNEL_MASS_MIN = 0.5
DIST_MIN_M = 5000.0
RATIO_BAND = 0.15


def log_skip(msg: str) -> None:
    print(f"skipping: {msg}")


# --------------------------------------------------------------------------- io
def load_manifest(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if not p.exists():
        log_skip(f"manifest.json not found at {p}")
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        log_skip(f"manifest.json unreadable ({e})")
        return {}


def load_gauges_csv(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "gauges.csv"
    cols = ["staid", "role", "pair", "comid", "class", "upstream_staids"]
    if not p.exists():
        log_skip(f"gauges.csv not found at {p}")
        return pd.DataFrame(columns=cols)
    try:
        return pd.read_csv(p, dtype={"staid": str}, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        log_skip(f"gauges.csv unreadable ({e})")
        return pd.DataFrame(columns=cols)


def discover_arms(run_dir: Path, manifest: dict) -> list[str]:
    arms = [a["name"] for a in manifest.get("arms", []) if "name" in a]
    if arms:
        return arms
    # Fallback: any immediate subdirectory that looks like an arm output dir.
    found = []
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and ((d / "validation").exists() or (d / "gauges").exists()):
            found.append(d.name)
    if found:
        log_skip("manifest.json had no usable 'arms' list; discovered arm dirs by scanning: " + ", ".join(found))
    return found


def parse_sweep(val) -> np.ndarray:
    if val is None:
        return np.array([], dtype=float)
    if isinstance(val, float) and np.isnan(val):
        return np.array([], dtype=float)
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return np.array([], dtype=float)
    try:
        return np.array([float(x) for x in s.split(";") if x != ""], dtype=float)
    except ValueError:
        return np.array([], dtype=float)


def compute_limit_pass_and_kink(row) -> tuple:
    """limit_pass / kink per the small-delta-limit and kink definitions.

    Both are only defined for failed rows (passed == 0) that carry sweep data;
    otherwise returns (nan, nan).
    """
    if row["passed"] == 1:
        return (np.nan, np.nan)
    deltas = parse_sweep(row.get("sweep_deltas"))
    if deltas.size == 0:
        return (np.nan, np.nan)
    order = np.argsort(deltas)
    deltas_s = deltas[order]

    dq_actual_sw = parse_sweep(row.get("sweep_dq_actual"))
    rel_err_sw = parse_sweep(row.get("sweep_rel_err"))
    fwd_sw = parse_sweep(row.get("sweep_dq_forward"))
    bwd_sw = parse_sweep(row.get("sweep_dq_backward"))

    limit_pass = np.nan
    if dq_actual_sw.size == deltas.size and rel_err_sw.size == deltas.size:
        dq_actual_s = dq_actual_sw[order]
        rel_err_s = rel_err_sw[order]
        noise_floor = row.get("noise_floor_m3s")
        idx_chosen = 0
        if noise_floor is not None and not (isinstance(noise_floor, float) and np.isnan(noise_floor)):
            resolved = np.abs(dq_actual_s * 2.0 * deltas_s) >= 2.0 * noise_floor
            hits = np.where(resolved)[0]
            if hits.size > 0:
                idx_chosen = int(hits[0])
        limit_pass = bool(rel_err_s[idx_chosen] <= LIMIT_PASS_REL_TOL)

    kink = np.nan
    if fwd_sw.size == deltas.size and bwd_sw.size == deltas.size:
        grad = row.get("dq_predicted")
        if grad is not None and not (isinstance(grad, float) and np.isnan(grad)):
            fwd0 = fwd_sw[order][0]
            bwd0 = bwd_sw[order][0]
            gabs = max(abs(grad), 1e-12)
            cond_diff = abs(fwd0 - bwd0) > 0.1 * gabs
            cond_match = (abs(grad - fwd0) <= 0.05 * abs(grad)) or (abs(grad - bwd0) <= 0.05 * abs(grad))
            kink = bool(cond_diff and cond_match)
    return (limit_pass, kink)


def load_validation_df(run_dir: Path, arms: list[str]) -> pd.DataFrame:
    frames = []
    for arm in arms:
        vdir = run_dir / arm / "validation"
        if not vdir.exists():
            log_skip(f"no validation dir for arm '{arm}' at {vdir}")
            continue
        files = sorted(vdir.glob("*.csv"))
        if not files:
            log_skip(f"no validation csvs under {vdir}")
            continue
        for f in files:
            m = FNAME_RE.match(f.name)
            if not m:
                log_skip(f"filename does not match <staid>_<high|low>_day<N>.csv: {f}")
                continue
            try:
                df = pd.read_csv(f, dtype={"staid": str})
            except Exception as e:  # noqa: BLE001
                log_skip(f"failed to read {f} ({e})")
                continue
            if df.empty:
                log_skip(f"{f} has no rows")
                continue
            for col in VALIDATION_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            if "staid" not in df.columns or df["staid"].isna().all():
                df["staid"] = m.group("staid")
            df["staid"] = df["staid"].astype(str)
            if "anchor_kind" not in df.columns or df["anchor_kind"].isna().all():
                df["anchor_kind"] = m.group("kind")
            if "arm" not in df.columns or df["arm"].isna().all():
                df["arm"] = arm
            df["_source_file"] = str(f.relative_to(run_dir))
            frames.append(df[VALIDATION_COLUMNS + ["_source_file"]])
    if not frames:
        return pd.DataFrame(columns=VALIDATION_COLUMNS + ["_source_file"])
    out = pd.concat(frames, ignore_index=True, sort=False)

    out["unresolvable"] = pd.to_numeric(out["unresolvable"], errors="coerce").fillna(0).astype(int)
    out["passed"] = pd.to_numeric(out["passed"], errors="coerce").fillna(0).astype(int)
    for col in ["dq_predicted", "dq_actual", "rel_err", "dist_to_gauge_m", "lag_hours",
                "noise_floor_m3s", "q_gauge_t0"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["resolvable"] = out["unresolvable"] == 0

    limit_kink = out.apply(compute_limit_pass_and_kink, axis=1, result_type="expand")
    out["limit_pass"] = limit_kink[0]
    out["kink"] = limit_kink[1]
    return out


def find_gauge_datasets(run_dir: Path, arms: list[str], staids: list[str]) -> dict:
    data = {}
    for arm in arms:
        gdir = run_dir / arm / "gauges"
        if not gdir.exists():
            log_skip(f"no gauges dir for arm '{arm}' at {gdir}")
            continue
        for staid in staids:
            p = gdir / f"{staid}.nc"
            if not p.exists():
                log_skip(f"no gauge netCDF for {arm}/{staid} at {p}")
                continue
            try:
                data[(arm, staid)] = xr.open_dataset(p, decode_timedelta=False)
            except Exception as e:  # noqa: BLE001
                log_skip(f"failed to open {p} ({e})")
    return data


# --------------------------------------------------------------------- helpers
def make_grid(n: int, ncols: int = 3, panel_w: float = 4.2, panel_h: float = 3.8):
    n = max(n, 1)
    ncols_use = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols_use))
    fig, axes = plt.subplots(nrows, ncols_use, figsize=(panel_w * ncols_use, panel_h * nrows), squeeze=False)
    flat = axes.flatten()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def color_map_for(values) -> dict:
    uniq = sorted({v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))})
    cmap = plt.get_cmap("viridis")
    if len(uniq) <= 1:
        return {v: cmap(0.5) for v in uniq}
    return {v: cmap(i / (len(uniq) - 1)) for i, v in enumerate(uniq)}


def empty_axis_note(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------- figure 1
def fig_fd_vs_adjoint(figdir: Path, val_df: pd.DataFrame) -> bool:
    if val_df.empty:
        log_skip("check1_fd_vs_adjoint: no validation rows available")
        return False
    groups = list(val_df.groupby(["staid", "anchor_kind"], sort=True))
    if not groups:
        return False
    lag_colors = color_map_for(val_df["lag_hours"].tolist())
    fig, axes = make_grid(len(groups))
    for ax, ((staid, kind), g) in zip(axes, groups):
        res = g[g["resolvable"]]
        for lag, color in lag_colors.items():
            sub = res[res["lag_hours"] == lag]
            passed = sub[sub["passed"] == 1]
            failed = sub[sub["passed"] == 0]
            if not passed.empty:
                ax.scatter(passed["dq_predicted"], passed["dq_actual"], color=color, s=22,
                           label=f"{lag:g} h", zorder=3)
            if not failed.empty:
                ax.scatter(failed["dq_predicted"], failed["dq_actual"], facecolors="none",
                           edgecolors=color, s=32, linewidths=1.2, zorder=3)
        finite_vals = pd.concat([res["dq_predicted"], res["dq_actual"]]).replace([np.inf, -np.inf], np.nan).dropna()
        lim = float(np.abs(finite_vals).max()) * 1.1 if not finite_vals.empty else 1.0
        lim = max(lim, 1e-2)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, zorder=1)
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("dq_predicted (adjoint), m3/s per unit inflow")
        ax.set_ylabel("dq_actual (central FD), m3/s per unit inflow")
        ax.set_title(f"{staid} - {kind}-flow anchors", fontsize=10)

        n_checks = len(g)
        n_pass = int((g["passed"] == 1).sum())
        n_unres = int((g["unresolvable"] == 1).sum())
        n_failed = int((g["passed"] == 0).sum())
        failed_g = g[g["passed"] == 0]
        n_limit = int((failed_g["limit_pass"] == True).sum())  # noqa: E712
        n_kink = int((failed_g["kink"] == True).sum())  # noqa: E712
        ax.text(0.02, 0.98, f"{n_pass} pass / {n_checks} checks ({n_unres} unresolvable)\n"
                             f"limit-pass {n_limit} / kinks {n_kink} (of {n_failed} failed)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", alpha=0.75, lw=0.4))
        ax.legend(fontsize=6.5, loc="lower right", title="lag", title_fontsize=6.5)
    fig.suptitle("Check 1: finite-difference vs adjoint-kernel dq (resolvable rows; hollow = failed)", y=1.0)
    fig.tight_layout()
    out = figdir / "check1_fd_vs_adjoint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return True


# --------------------------------------------------------------------- figure 2
def fig_relerr_vs_distance(figdir: Path, val_df: pd.DataFrame, rel_tol_line: float) -> bool:
    if val_df.empty:
        log_skip("check1_relerr_vs_distance: no validation rows available")
        return False
    kinds = [k for k in ["high", "low"] if (val_df["anchor_kind"] == k).any()]
    if not kinds:
        return False
    lag_colors = color_map_for(val_df["lag_hours"].tolist())
    fig, axes = make_grid(len(kinds), ncols=2)
    for ax, kind in zip(axes, kinds):
        g = val_df[(val_df["anchor_kind"] == kind) & val_df["resolvable"]]
        for lag, color in lag_colors.items():
            sub = g[g["lag_hours"] == lag]
            sub = sub[sub["rel_err"] > 0]
            if sub.empty:
                continue
            ax.scatter(sub["dist_to_gauge_m"] / 1000.0, sub["rel_err"], color=color, s=16,
                       alpha=0.8, label=f"{lag:g} h")
        ax.axhline(rel_tol_line, color="k", lw=0.8, ls="--", label=f"rel_tol={rel_tol_line:g}")
        ax.set_yscale("log")
        ax.set_xlabel("distance to gauge (km)")
        ax.set_ylabel("rel_err")
        ax.set_title(f"{kind}-flow anchors, pooled over gauges", fontsize=10)
        ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("Check 1: relative error vs along-channel distance (resolvable rows)", y=1.0)
    fig.tight_layout()
    out = figdir / "check1_relerr_vs_distance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return True


# ------------------------------------------------------------- gauge-nc helpers
def group_averages(ds: xr.Dataset) -> dict:
    """Per-anchor-kind, per-reach average kernel/hydraulic lag over reaches passing
    kernel_mass > KERNEL_MASS_MIN and dist_to_gauge_m > DIST_MIN_M (mask applied
    per anchor, then averaged over the valid anchors of that kind, per reach)."""
    required = ["kernel_mass", "kernel_mean_lag_days", "hydraulic_lag_days",
                "hydraulic_lag_t0_days", "dist_to_gauge_m", "anchor_is_high"]
    if any(v not in ds for v in required):
        return {}
    kernel_mass = ds["kernel_mass"].values.astype(float)
    kernel_lag = ds["kernel_mean_lag_days"].values.astype(float)
    hyd_lag = ds["hydraulic_lag_days"].values.astype(float)
    hyd_t0 = ds["hydraulic_lag_t0_days"].values.astype(float)
    dist = ds["dist_to_gauge_m"].values.astype(float)
    anchor_is_high = ds["anchor_is_high"].values.astype(bool)
    n_reach = kernel_mass.shape[1]
    result = {}
    for kind, is_high in [("low", False), ("high", True)]:
        anchor_idx = np.where(anchor_is_high == is_high)[0]
        if anchor_idx.size == 0:
            result[kind] = None
            continue
        mask = (kernel_mass[anchor_idx, :] > KERNEL_MASS_MIN) & (dist[None, :] > DIST_MIN_M)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            k_masked = np.where(mask, kernel_lag[anchor_idx, :], np.nan)
            h_masked = np.where(mask, hyd_lag[anchor_idx, :], np.nan)
            ht0_masked = np.where(mask, hyd_t0[anchor_idx, :], np.nan)
            kernel_avg = np.nanmean(k_masked, axis=0) if k_masked.size else np.full(n_reach, np.nan)
            hyd_avg = np.nanmean(h_masked, axis=0) if h_masked.size else np.full(n_reach, np.nan)
            hyd_t0_avg = np.nanmean(ht0_masked, axis=0) if ht0_masked.size else np.full(n_reach, np.nan)
        valid = mask.sum(axis=0) > 0
        result[kind] = dict(valid=valid, kernel_avg=kernel_avg, hyd_avg=hyd_avg,
                             hyd_t0_avg=hyd_t0_avg, dist=dist)
    return result


def ratio_stats(kernel_avg, ref_avg, valid) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ratio = kernel_avg / ref_avg
    ratio = np.where(valid & np.isfinite(ratio), ratio, np.nan)
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return dict(median=np.nan, iqr=np.nan, frac_within=np.nan, n=0, ratio=ratio)
    q25, q50, q75 = np.nanpercentile(finite, [25, 50, 75])
    frac_within = float(np.mean(np.abs(finite - 1.0) <= RATIO_BAND))
    return dict(median=q50, iqr=q75 - q25, frac_within=frac_within, n=int(finite.size), ratio=ratio)


# --------------------------------------------------------------------- figure 3
def fig_kernel_vs_hydraulic(figdir: Path, gauge_data: dict) -> tuple:
    """Returns (wrote_bool, table_b_rows)."""
    table_rows = []
    keys = sorted(gauge_data.keys())
    if not keys:
        log_skip("check2_kernel_vs_hydraulic: no gauge netCDFs available")
        return False, table_rows
    fig, axes = make_grid(len(keys), ncols=3)
    for ax, (arm, staid) in zip(axes, keys):
        ds = gauge_data[(arm, staid)]
        avgs = group_averages(ds)
        if not avgs:
            empty_axis_note(ax, "missing required variables")
            log_skip(f"check2_kernel_vs_hydraulic: {arm}/{staid} missing required variables")
            continue
        n_reach = ds.sizes.get("reach", 0)
        row = dict(arm=arm, staid=staid, n_reach=n_reach)
        all_vals = []
        for kind, marker, filled in [("low", "o", True), ("high", "o", False)]:
            g = avgs.get(kind)
            if g is None:
                row[f"{kind}_median_ratio"] = np.nan
                row[f"{kind}_iqr_ratio"] = np.nan
                row[f"{kind}_frac_within_15pct"] = np.nan
                row[f"{kind}_median_ratio_t0"] = np.nan
                row[f"{kind}_n"] = 0
                continue
            stats_meank = ratio_stats(g["kernel_avg"], g["hyd_avg"], g["valid"])
            stats_t0 = ratio_stats(g["kernel_avg"], g["hyd_t0_avg"], g["valid"])
            row[f"{kind}_median_ratio"] = stats_meank["median"]
            row[f"{kind}_iqr_ratio"] = stats_meank["iqr"]
            row[f"{kind}_frac_within_15pct"] = stats_meank["frac_within"]
            row[f"{kind}_median_ratio_t0"] = stats_t0["median"]
            row[f"{kind}_n"] = stats_meank["n"]
            v = g["valid"]
            x = g["hyd_avg"][v]
            y = g["kernel_avg"][v]
            all_vals.extend([x, y])
            if filled:
                ax.scatter(x, y, s=18, color="tab:blue", alpha=0.75, label="low-flow anchors", zorder=3)
            else:
                ax.scatter(x, y, s=26, facecolors="none", edgecolors="tab:red", linewidths=1.0,
                           label="high-flow anchors", zorder=3)
                x0 = g["hyd_t0_avg"][v]
                ax.scatter(x0, y, marker="x", s=20, color="tab:red", alpha=0.8, linewidths=0.9,
                           label="high-flow, t0 hydraulic ref", zorder=2)
                all_vals.append(x0)
        finite_all = np.concatenate([a[np.isfinite(a)] for a in all_vals]) if all_vals else np.array([])
        if finite_all.size >= 2 and finite_all.min() > 0:
            lo, hi = finite_all.min(), finite_all.max()
            span_decades = np.log10(hi / lo) if lo > 0 else 0
            lo_p, hi_p = lo * 0.9, hi * 1.1
            xx = np.linspace(lo_p, hi_p, 50)
            ax.plot(xx, xx, "k--", lw=0.8, zorder=1)
            ax.fill_between(xx, xx * (1 - RATIO_BAND), xx * (1 + RATIO_BAND), color="k", alpha=0.08, zorder=0)
            if span_decades > 1.5:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlim(lo_p, hi_p)
            ax.set_ylim(lo_p, hi_p)
        ax.set_xlabel("hydraulic lag (days)")
        ax.set_ylabel("kernel mean lag (days)")
        ax.set_title(f"{staid} ({arm})", fontsize=10)
        med_low = row.get("low_median_ratio", np.nan)
        frac_low = row.get("low_frac_within_15pct", np.nan)
        med_high = row.get("high_median_ratio", np.nan)
        frac_high = row.get("high_frac_within_15pct", np.nan)
        ax.text(0.02, 0.98,
                f"low: median ratio {med_low:.3g}, {frac_low * 100:.0f}% within 15%\n"
                f"high: median ratio {med_high:.3g}, {frac_high * 100:.0f}% within 15%"
                if np.isfinite(med_low) or np.isfinite(med_high) else "no reaches pass filter",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", alpha=0.75, lw=0.4))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = dict(zip(labels, handles))
            ax.legend(seen.values(), seen.keys(), fontsize=6, loc="lower right")
        table_rows.append(row)
    fig.suptitle(f"Check 2: kernel mean lag vs hydraulic lag "
                 f"(kernel_mass > {KERNEL_MASS_MIN:g}, dist > {DIST_MIN_M/1000:g} km)", y=1.0)
    fig.tight_layout()
    out = figdir / "check2_kernel_vs_hydraulic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return True, table_rows


# --------------------------------------------------------------------- figure 4
def fig_ratio_vs_distance(figdir: Path, gauge_data: dict) -> bool:
    keys = sorted(gauge_data.keys())
    if not keys:
        log_skip("check2_ratio_vs_distance: no gauge netCDFs available")
        return False
    gauge_colors = color_map_for([staid for _, staid in keys])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    any_plotted = False
    for arm, staid in keys:
        ds = gauge_data[(arm, staid)]
        avgs = group_averages(ds)
        g = avgs.get("low") if avgs else None
        if g is None:
            continue
        stats_meank = ratio_stats(g["kernel_avg"], g["hyd_avg"], g["valid"])
        ratio = stats_meank["ratio"]
        finite = np.isfinite(ratio)
        if not finite.any():
            continue
        ax.scatter(g["dist"][finite] / 1000.0, ratio[finite], s=16, alpha=0.75,
                   color=gauge_colors[staid], label=staid)
        any_plotted = True
    if not any_plotted:
        log_skip("check2_ratio_vs_distance: no reaches pass the kernel_mass/distance filter for any gauge")
        plt.close(fig)
        return False
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.axhline(1.0 + RATIO_BAND, color="k", lw=0.6, ls=":")
    ax.axhline(1.0 - RATIO_BAND, color="k", lw=0.6, ls=":")
    ax.set_xlabel("distance to gauge (km)")
    ax.set_ylabel("kernel / hydraulic lag ratio (mean-K, low-flow anchors)")
    ax.set_title("Check 2: kernel/hydraulic ratio vs distance, pooled over gauges")
    ax.legend(fontsize=7, title="gauge", loc="best")
    fig.tight_layout()
    out = figdir / "check2_ratio_vs_distance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return True


# --------------------------------------------------------------------- tables
def build_table_a(val_df: pd.DataFrame) -> pd.DataFrame:
    if val_df.empty:
        return pd.DataFrame()
    rows = []
    for (staid, kind), g in val_df.groupby(["staid", "anchor_kind"], sort=True):
        n_checks = len(g)
        n_passed = int((g["passed"] == 1).sum())
        n_unresolvable = int((g["unresolvable"] == 1).sum())
        n_failed = int((g["passed"] == 0).sum())
        n_limit_pass = int((g["limit_pass"] == True).sum())  # noqa: E712
        n_kink = int((g["kink"] == True).sum())  # noqa: E712
        not_limit_pass = ~(g["limit_pass"] == True)  # noqa: E712  (NaN -> True, i.e. included)
        candidate = g[g["resolvable"] & not_limit_pass]
        worst_rel_err = candidate["rel_err"].max() if not candidate.empty else np.nan
        rows.append(dict(
            staid=staid, anchor_kind=kind, n_checks=n_checks, n_passed=n_passed,
            n_unresolvable=n_unresolvable, n_failed=n_failed, n_limit_pass=n_limit_pass,
            n_kink=n_kink, worst_rel_err_resolvable_not_limit_pass=worst_rel_err,
            q_gauge_t0=g["q_gauge_t0"].mean(), noise_floor_m3s=g["noise_floor_m3s"].mean(),
        ))
    return pd.DataFrame(rows)


def build_table_b(table_b_rows: list) -> pd.DataFrame:
    if not table_b_rows:
        return pd.DataFrame()
    cols = ["arm", "staid", "n_reach",
            "low_n", "low_median_ratio", "low_iqr_ratio", "low_frac_within_15pct", "low_median_ratio_t0",
            "high_n", "high_median_ratio", "high_iqr_ratio", "high_frac_within_15pct", "high_median_ratio_t0"]
    df = pd.DataFrame(table_b_rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    if df.empty:
        return "*(no data)*\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if pd.isna(v) else float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(figdir: Path, run_dir: Path, table_a: pd.DataFrame, table_b: pd.DataFrame,
                  input_files: list, wrote: dict) -> None:
    lines = []
    lines.append(f"# Adjoint UH-validation report\n")
    lines.append(f"Run directory: `{run_dir}`\n")
    lines.append("## Figures\n")
    for name, ok in wrote.items():
        lines.append(f"- {name}: {'written' if ok else 'skipped (no data)'}")
    lines.append("")
    lines.append("## Table A: finite-difference check summary, per gauge and anchor kind\n")
    lines.append("Each row pools all anchor-day validation files sharing the same (staid, anchor_kind); "
                 "n_checks counts individual (reach, lag) rows, n_limit_pass/n_kink are counted among "
                 "failed rows, and worst_rel_err_resolvable_not_limit_pass is the maximum rel_err over "
                 "resolvable rows excluding failed rows that pass in the small-delta limit.\n")
    lines.append(df_to_markdown(table_a))
    lines.append("")
    lines.append("## Table B: kernel vs hydraulic lag ratio summary, per gauge\n")
    lines.append("Reaches are filtered to kernel_mass > 0.5 and dist_to_gauge_m > 5000 m per anchor, "
                 "averaged over anchors of the same kind, before computing the kernel/hydraulic ratio; "
                 "_t0 columns use hydraulic_lag_t0_days as the reference instead of the mean-K hydraulic lag.\n")
    lines.append(df_to_markdown(table_b))
    lines.append("")
    lines.append("## Input files used\n")
    for f in input_files:
        lines.append(f"- `{f}`")
    lines.append("")
    out = figdir / "VALIDATION.md"
    out.write_text("\n".join(lines))
    print("wrote", out)


# --------------------------------------------------------------------- main
def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <run_dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)
    figdir = run_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    input_files = []
    manifest = load_manifest(run_dir)
    if (run_dir / "manifest.json").exists():
        input_files.append("manifest.json")
    gauges = load_gauges_csv(run_dir)
    if (run_dir / "gauges.csv").exists():
        input_files.append("gauges.csv")

    arms = discover_arms(run_dir, manifest)
    print(f"arms: {arms}")

    val_df = load_validation_df(run_dir, arms)
    input_files.extend(sorted(val_df["_source_file"].unique().tolist()) if not val_df.empty else [])
    print(f"validation rows loaded: {len(val_df)}")

    rel_tol_line = REL_ERR_LINE_DEFAULT
    try:
        rel_tol_line = float(manifest["spec"]["adjoint"]["validation"]["rel_tol"])
    except Exception:  # noqa: BLE001
        pass

    staid_universe = sorted(set(val_df["staid"].unique().tolist()) | set(gauges["staid"].unique().tolist()))
    gauge_data = find_gauge_datasets(run_dir, arms, staid_universe)
    for arm, staid in sorted(gauge_data.keys()):
        input_files.append(f"{arm}/gauges/{staid}.nc")
    print(f"gauge netCDFs loaded: {len(gauge_data)}")

    wrote = {}
    wrote["check1_fd_vs_adjoint.png"] = fig_fd_vs_adjoint(figdir, val_df)
    wrote["check1_relerr_vs_distance.png"] = fig_relerr_vs_distance(figdir, val_df, rel_tol_line)
    wrote_k, table_b_rows = fig_kernel_vs_hydraulic(figdir, gauge_data)
    wrote["check2_kernel_vs_hydraulic.png"] = wrote_k
    wrote["check2_ratio_vs_distance.png"] = fig_ratio_vs_distance(figdir, gauge_data)

    table_a = build_table_a(val_df)
    table_b = build_table_b(table_b_rows)
    write_report(figdir, run_dir, table_a, table_b, input_files, wrote)

    for ds in gauge_data.values():
        ds.close()


if __name__ == "__main__":
    main()
