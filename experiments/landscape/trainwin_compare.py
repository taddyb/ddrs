#!/usr/bin/env python
"""Compare the per-gauge loss gradient at the trained parameter point
(grad0_n) between two landscape censuses of the same trained model: one
scored on the TESTING window, one scored on the TRAINING window. This is
the control for a convergence test -- if training actually converged, the
per-gauge gradients on the training window should be small and should
roughly cancel across gauges (batch-level stationarity), not all point the
same way.

Inputs are two "merged" landscape run directories produced by
scripts/landscape_merge.py (see covariates.py for the same directory
layout: <merged>/<arm>/summary.csv, <merged>/<arm>/gauges/<staid>.nc,
manifest.json with a single arm).

For each window this loads staid/n_reach/nse0/grad0_n/grad0_q/DRAIN_SQKM,
preferring an existing <merged>/figures/covariates.csv (as already built
for the testing window) and falling back to summary.csv + per-gauge
gauges/<staid>.nc + a GAGES-II join when covariates.csv has not been
generated yet (the training window, freshly merged).

Usage:
    uv run --with numpy --with pandas python experiments/landscape/trainwin_compare.py \\
        <testing-merged-dir> <training-merged-dir> --out <out-dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

GAGES_II_DBF = Path("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf")
COLS = ["staid", "n_reach", "nse0", "grad0_n", "grad0_q", "DRAIN_SQKM"]
NSE0_MIN = 0.3

# ----------------------------------------------------------------------------- io
def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())

def pick_arm(manifest: dict) -> str:
    arms = manifest["arms"]
    if len(arms) != 1:
        names = [a["name"] for a in arms]
        raise SystemExit(f"{manifest.get('bundle_dir', '?')} has {len(arms)} arms {names}; this script assumes a single-arm bundle")
    return arms[0]["name"]

def load_gages_ii_drain_sqkm(dbf_path: Path) -> pd.DataFrame | None:
    if gpd is None or not dbf_path.exists():
        return None
    gdf = gpd.read_file(dbf_path)
    df = pd.DataFrame(gdf[["STAID", "DRAIN_SQKM"]])
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    return df.rename(columns={"STAID": "staid"})

def load_window(merged_dir: Path) -> pd.DataFrame:
    """staid/n_reach/nse0/grad0_n/grad0_q/DRAIN_SQKM table for one merged
    landscape-census directory."""
    cov_path = merged_dir / "figures" / "covariates.csv"
    if cov_path.exists():
        df = pd.read_csv(cov_path, dtype={"staid": str})
        return df[COLS].copy()

    if xr is None:
        raise SystemExit(f"{merged_dir}: no {cov_path} and xarray is not importable to read grad0_n/grad0_q from gauges/<staid>.nc")

    manifest = load_manifest(merged_dir)
    arm = pick_arm(manifest)
    summary = pd.read_csv(merged_dir / arm / "summary.csv", dtype={"staid": str})[["staid", "n_reach", "nse0"]]

    grad_rows = []
    for staid in summary["staid"]:
        nc_path = merged_dir / arm / "gauges" / f"{staid}.nc"
        if not nc_path.exists():
            grad_rows.append((staid, np.nan, np.nan))
            continue
        ds = xr.open_dataset(nc_path, decode_timedelta=False)
        try:
            grad0 = ds["grad0"].values.astype(float)  # rows n=0, p=1, q=2
            grad_rows.append((staid, float(grad0[0]), float(grad0[2])))
        finally:
            ds.close()
    grad_df = pd.DataFrame(grad_rows, columns=["staid", "grad0_n", "grad0_q"])

    df = summary.merge(grad_df, on="staid", how="left")

    gages_ii = load_gages_ii_drain_sqkm(GAGES_II_DBF)
    if gages_ii is not None:
        df = df.merge(gages_ii, on="staid", how="left")
    else:
        df["DRAIN_SQKM"] = np.nan
        print(f"warning: GAGES-II table unavailable at {GAGES_II_DBF}; DRAIN_SQKM left as NaN")

    return df[COLS].copy()

# ------------------------------------------------------------------------- stats
def well_fit(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["nse0"] > NSE0_MIN) & np.isfinite(df["grad0_n"])].copy()

def alignment(g: np.ndarray) -> float:
    """|mean(g)| / mean(|g|). 0 = per-gauge gradients cancel (a converged
    batch compromise); 1 = they all point the same way (descent unfinished)."""
    g = g[np.isfinite(g)]
    if len(g) == 0 or np.mean(np.abs(g)) == 0:
        return float("nan")
    return float(abs(np.mean(g)) / np.mean(np.abs(g)))

def three_line_block(g: np.ndarray, indent: str) -> list[str]:
    g = g[np.isfinite(g)]
    n = len(g)
    if n == 0:
        return [f"{indent}n=0"]
    med, mean = np.median(g), np.mean(g)
    sd = np.std(g, ddof=1) if n > 1 else float("nan")
    frac_neg = float(np.mean(g < 0))
    lines = [
        f"{indent}n={n}  median={med:.6g}  mean={mean:.6g}  sd={sd:.6g}",
        f"{indent}frac(grad0_n < 0) = {frac_neg:.4f}",
        f"{indent}alignment |mean(g)|/mean(|g|) = {alignment(g):.4f}",
    ]
    return lines


SIZE_BINS = [
    ("n_reach <= 50", lambda n: n <= 50),
    ("51 < n_reach <= 200", lambda n: (n > 50) & (n <= 200)),
    ("n_reach > 200", lambda n: n > 200),
]

def window_report(label: str, path: Path, df: pd.DataFrame) -> list[str]:
    wf = well_fit(df)
    lines = [f"=== {label} ({path}) ===", f"  well-fit gauges: {len(wf)} / {len(df)} total"]
    lines.append("  overall:")
    lines += three_line_block(wf["grad0_n"].to_numpy(dtype=float), "    ")
    for bin_label, pred in SIZE_BINS:
        sub = wf[pred(wf["n_reach"].to_numpy(dtype=float))]
        lines.append(f"  {bin_label}:")
        lines += three_line_block(sub["grad0_n"].to_numpy(dtype=float), "    ")
    lines.append("")
    return lines, wf

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])

def joint_report(wf_test: pd.DataFrame, wf_train: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    shared = wf_test.merge(
        wf_train[["staid", "grad0_n"]], on="staid", how="inner", suffixes=("_test", "_train")
    )
    shared = shared[["staid", "n_reach", "nse0", "grad0_n_test", "grad0_n_train"]]

    lines = [f"=== joint: gauges well-fit in BOTH windows ===", f"  shared gauges: {len(shared)}"]
    if len(shared) == 0:
        lines.append("  (no shared gauges -- skipping correlation/sign stats)")
        return lines, shared

    a = shared["grad0_n_test"].to_numpy(dtype=float)
    b = shared["grad0_n_train"].to_numpy(dtype=float)
    lines.append(f"  alignment, testing window  = {alignment(a):.4f}")
    lines.append(f"  alignment, training window = {alignment(b):.4f}")
    lines.append(f"  pearson  r(grad0_n_test, grad0_n_train)  = {np.corrcoef(a, b)[0, 1]:.4f}")
    lines.append(f"  spearman r(grad0_n_test, grad0_n_train)  = {spearman_corr(a, b):.4f}")
    sign_agree = float(np.mean(np.sign(a) == np.sign(b)))
    lines.append(f"  sign agreement (sign(grad0_n_test) == sign(grad0_n_train)) = {sign_agree:.4f}")
    lines.append("")
    return lines, shared

# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("testing_dir", type=Path, help="merged landscape-census dir scored on the testing window")
    ap.add_argument("training_dir", type=Path, help="merged landscape-census dir scored on the training window")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df_test = load_window(args.testing_dir)
    df_train = load_window(args.training_dir)

    lines = [
        "trainwin_compare: per-gauge loss-gradient alignment, testing window vs training window",
        f"testing window : {args.testing_dir}",
        f"training window: {args.training_dir}",
        f"well-fit filter: nse0 > {NSE0_MIN} and grad0_n finite",
        "",
    ]
    test_lines, wf_test = window_report("testing window", args.testing_dir, df_test)
    train_lines, wf_train = window_report("training window", args.training_dir, df_train)
    lines += test_lines + train_lines

    joint_lines, shared = joint_report(wf_test, wf_train)
    lines += joint_lines

    report = "\n".join(lines)
    print(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "trainwin_compare.txt").write_text(report + "\n")
    shared.to_csv(args.out / "trainwin_compare.csv", index=False)
    print(f"wrote {args.out / 'trainwin_compare.txt'}")
    print(f"wrote {args.out / 'trainwin_compare.csv'} ({len(shared)} rows)")


if __name__ == "__main__":
    main()
