#!/usr/bin/env python
"""Per-gauge covariate table for "why is the trained roughness not at the
gauge's own optimum" analysis.

Joins, for every gauge in a landscape census run: the census run's own
displacement/curvature diagnostics (<census_dir>/<arm>/summary.csv and
<census_dir>/<arm>/gauges/<staid>.nc), optionally a second diagnostic run's
daily series and 5-yr gradient/Hessian (--diag, same layout, written with
`landscape.series: true`), GAGES-II basin attributes
(/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf), MERIT reach
attributes at the gauge's own COMID
(~/projects/ddr/data/merit_global_attributes_v2.nc), and the census run's own
eval-window NSE/KGE (<run_dir>/eval/predictions.zarr, run_id read from
manifest.json).

Assumes the census (and diag, if given) manifest has exactly one arm -- pass
a single-arm merged bundle. "routed" series in --diag metrics means
routed_daily_trained (the trained-point series), not routed_daily_star.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/landscape/covariates.py \\
        <census-merged-dir> [--diag <diag-merged-dir>] \\
        [--out <census-dir>/figures/covariates.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import zarr
except ImportError:
    zarr = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from surface import gauge_depth_and_width  # noqa: E402
except ImportError:
    def gauge_depth_and_width(n: np.ndarray, p, q: np.ndarray, discharge: float, slope: float):
        """Fallback copy of surface.py's formula (kept identical) for when
        plotly/scipy aren't installed and surface.py can't be imported."""
        q_eps = q + 1e-6
        numerator = discharge * n * (q_eps + 1.0)
        denominator = p * np.sqrt(slope)
        ratio = numerator / (denominator + 1e-8)
        exponent = 3.0 / (q_eps * 3.0 + 5.0)
        depth = ratio ** exponent
        width = p * depth ** q_eps
        return depth, width

GAGES_II_DBF = Path("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.dbf")
MERIT_ATTR_NC = Path("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
GEOM_MIN_SLOPE = 1e-3  # matches surface.py's depth-axis slope floor

DIAG_COLS = [
    "vol_ratio_qprime", "vol_ratio_routed", "lowflow_frac", "zeroflow_frac",
    "routed_floor_frac", "rb_flashiness", "lag_days_qprime", "lag_days_routed",
    "spring_frac", "peak_ratio",
]
DIAG_GRAD_COLS = ["grad0_n_diag", "grad0_q_diag", "H_nn_diag", "H_qq_diag"]


# ----------------------------------------------------------------------------- io
def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def pick_arm(manifest: dict) -> dict:
    arms = manifest["arms"]
    if len(arms) != 1:
        names = [a["name"] for a in arms]
        raise SystemExit(f"{manifest.get('bundle_dir', '?')} has {len(arms)} arms {names}; this script assumes a single-arm bundle")
    return arms[0]


def alpha_max_of(manifest: dict) -> float:
    return float(manifest["spec"]["landscape"]["alpha_max"])


# --------------------------------------------------------------------- census
def load_census_summary(run_dir: Path, arm: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / arm / "summary.csv", dtype={"staid": str})
    df["gain"] = df["nse_star"] - df["nse0"]
    keep = [
        "staid", "n_reach", "sigma", "nse0", "nse_star", "gain",
        "alpha_n_star", "alpha_q_star", "mult_n", "mult_q",
        "hit_range_bound", "newton_iters", "used_gradient_fallback",
        "lambda1", "lambda2",
    ]
    return df[keep]


def extract_census_nc_row(path: Path, alpha_max: float) -> dict:
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        alpha_star = ds["alpha_star"].values.astype(float)
        grad0 = ds["grad0"].values.astype(float)
        hess0 = ds["hess0"].values.astype(float)  # (alpha, k), rows/cols n=0,p=1,q=2
        n0 = ds["n0"].values.astype(float)
        p0 = ds["p0"].values.astype(float)
        q0 = ds["q0"].values.astype(float)
        slope = ds["slope"].values.astype(float)
        length = ds["length"].values.astype(float)
        comid = ds["comid"].values
        gauge_row = int(ds.attrs["gauge_reach_row"])

        kge0 = float(ds["kge0"].values) if "kge0" in ds.variables else np.nan
        kge_star = float(ds["kge_star"].values) if "kge_star" in ds.variables else np.nan

        alpha_n_star, alpha_q_star = float(alpha_star[0]), float(alpha_star[2])
        box_edge = int(abs(alpha_n_star) >= 0.999 * alpha_max or abs(alpha_q_star) >= 0.999 * alpha_max)

        q0_med = float(np.median(q0))
        H_nn = float(hess0[0, 0])
        H_qq = float(hess0[2, 2])
        H_nq = float(hess0[0, 2])
        H_qq_natural = H_qq / q0_med ** 2 if q0_med != 0 else np.nan
        H_nq_natural = H_nq / q0_med if q0_med != 0 else np.nan

        n_gauge, p_gauge, q_gauge = float(n0[gauge_row]), float(p0[gauge_row]), float(q0[gauge_row])
        slope_gauge = float(slope[gauge_row])
        obs_mean_q_m3s = float(ds.attrs.get("obs_mean_q_m3s", np.nan))
        depth_gauge, width_gauge = gauge_depth_and_width(
            n_gauge, p_gauge, q_gauge, obs_mean_q_m3s, max(slope_gauge, GEOM_MIN_SLOPE)
        )
        total_length_km = float(np.sum(length)) / 1000.0
        n_reach = len(length)

        return dict(
            comid_gauge=int(comid[gauge_row]),
            kge0=kge0, kge_star=kge_star, box_edge=box_edge,
            grad0_n=float(grad0[0]), grad0_q=float(grad0[2]),
            H_nn=H_nn, H_qq=H_qq, H_nq=H_nq,
            H_qq_natural=H_qq_natural, H_nq_natural=H_nq_natural,
            n0_med=float(np.median(n0)), q0_med=q0_med,
            slope_gauge=slope_gauge,
            obs_mean_q_m3s=obs_mean_q_m3s,
            obs_n_valid_days=float(ds.attrs.get("obs_n_valid_days", np.nan)),
            depth_gauge_mean_flow=float(depth_gauge), width_gauge=float(width_gauge),
            total_length_km=total_length_km,
            mean_reach_length_km=total_length_km / n_reach if n_reach else np.nan,
        )
    finally:
        ds.close()


# ------------------------------------------------------------------------ diag
def best_lag(x: np.ndarray, y: np.ndarray, valid: np.ndarray, max_lag: int = 10) -> float:
    """Lag (days) maximizing corr(x[t], y[t+lag]) over +-max_lag. Positive
    lag means x leads y (x happens first)."""
    n = len(x)
    best_l, best_c = np.nan, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            xs, ys, v = x[:-lag], y[lag:], valid[:-lag] & valid[lag:]
        elif lag < 0:
            xs, ys, v = x[-lag:], y[:lag], valid[-lag:] & valid[:lag]
        else:
            xs, ys, v = x, y, valid
        if v.sum() < 10:
            continue
        xv, yv = xs[v], ys[v]
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        c = np.corrcoef(xv, yv)[0, 1]
        if np.isfinite(c) and c > best_c:
            best_c, best_l = c, lag
    return float(best_l)


def series_metrics(obs: np.ndarray, routed: np.ndarray, qprime: np.ndarray, axis_start_date: str, window_start_day: float) -> dict:
    valid = np.isfinite(obs) & np.isfinite(routed) & np.isfinite(qprime)
    if valid.sum() < 2:
        return {k: np.nan for k in DIAG_COLS}

    o, r, q = obs[valid], routed[valid], qprime[valid]
    n_valid = len(o)
    sum_obs = float(np.sum(o))
    vol_ratio_qprime = float(np.sum(q) / sum_obs) if sum_obs != 0 else np.nan
    vol_ratio_routed = float(np.sum(r) / sum_obs) if sum_obs != 0 else np.nan
    mean_obs = float(np.mean(o))
    lowflow_frac = float(np.mean(o < 0.1 * mean_obs))
    zeroflow_frac = float(np.mean(o <= 1e-3))
    routed_floor_frac = float(np.mean(r <= 1e-3))

    d_obs = np.diff(obs)
    pair_valid = valid[1:] & valid[:-1]
    den = float(np.sum(obs[1:][pair_valid]))
    rb_flashiness = float(np.sum(np.abs(d_obs[pair_valid])) / den) if den != 0 else np.nan

    lag_days_qprime = best_lag(qprime, obs, valid)
    lag_days_routed = best_lag(routed, obs, valid)

    dates = pd.Timestamp(axis_start_date) + pd.to_timedelta(window_start_day + np.arange(len(obs)), unit="D")
    spring_mask = valid & dates.month.isin([3, 4, 5, 6])
    spring_frac = float(np.sum(obs[spring_mask]) / sum_obs) if sum_obs != 0 else np.nan

    k = max(1, int(round(0.01 * n_valid)))
    top_r = float(np.mean(np.sort(r)[-k:]))
    top_o = float(np.mean(np.sort(o)[-k:]))
    peak_ratio = float(top_r / top_o) if top_o != 0 else np.nan

    return dict(
        vol_ratio_qprime=vol_ratio_qprime, vol_ratio_routed=vol_ratio_routed,
        lowflow_frac=lowflow_frac, zeroflow_frac=zeroflow_frac,
        routed_floor_frac=routed_floor_frac, rb_flashiness=rb_flashiness,
        lag_days_qprime=lag_days_qprime, lag_days_routed=lag_days_routed,
        spring_frac=spring_frac, peak_ratio=peak_ratio,
    )


def extract_diag_row(path: Path) -> dict:
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        row = {}
        if "grad0" in ds.variables:
            grad0 = ds["grad0"].values.astype(float)
            row["grad0_n_diag"], row["grad0_q_diag"] = float(grad0[0]), float(grad0[2])
        else:
            row["grad0_n_diag"] = row["grad0_q_diag"] = np.nan
        if "hess0" in ds.variables:
            hess0 = ds["hess0"].values.astype(float)
            row["H_nn_diag"], row["H_qq_diag"] = float(hess0[0, 0]), float(hess0[2, 2])
        else:
            row["H_nn_diag"] = row["H_qq_diag"] = np.nan

        if "obs_daily" in ds.variables and "routed_daily_trained" in ds.variables and "summed_qprime_daily" in ds.variables:
            obs = ds["obs_daily"].values.astype(float)
            routed = ds["routed_daily_trained"].values.astype(float)
            qprime = ds["summed_qprime_daily"].values.astype(float)
            axis_start_date = ds.attrs.get("axis_start_date")
            window_start_day = float(ds["window_start_day"].values[0]) if "window_start_day" in ds.variables else 0.0
            row.update(series_metrics(obs, routed, qprime, axis_start_date, window_start_day))
        else:
            row.update({k: np.nan for k in DIAG_COLS})
        return row
    finally:
        ds.close()


# --------------------------------------------------------------------- gages-ii
def load_gages_ii(dbf_path: Path) -> pd.DataFrame | None:
    if gpd is None or not dbf_path.exists():
        return None
    gdf = gpd.read_file(dbf_path)
    cols = ["STAID", "DRAIN_SQKM", "HUC02", "AGGECOREGI", "CLASS", "STATE", "LAT_GAGE", "LNG_GAGE"]
    df = pd.DataFrame(gdf[cols])
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    return df


# --------------------------------------------------------------------- merit
MERIT_VARS = [
    "log10_uparea", "meanslope", "aridity", "snow_fraction", "snowfall_fraction",
    "meanP", "meanTa", "NDVI", "permeability", "Porosity", "catchsize",
]


def load_merit_attrs(comids: np.ndarray, nc_path: Path) -> pd.DataFrame:
    """One row per entry of `comids`, same order (duplicates in `comids`
    preserved 1:1, not fanned out) -- caller assigns positionally, never
    merges on comid_gauge (gauge-reach COMIDs are not unique across gauges,
    e.g. nested USGS sites)."""
    if not nc_path.exists() or len(comids) == 0:
        return pd.DataFrame({v: np.full(len(comids), np.nan) for v in MERIT_VARS})
    ds = xr.open_dataset(nc_path, decode_timedelta=False)
    try:
        sub = ds[MERIT_VARS].reindex(COMID=comids)
        return sub.to_dataframe()[MERIT_VARS].reset_index(drop=True)
    finally:
        ds.close()


# --------------------------------------------------------------------- eval
def nse_kge(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0)
    if valid.sum() < 2:
        return np.nan, np.nan
    o, s = obs[valid], sim[valid]
    denom = float(np.sum((o - o.mean()) ** 2))
    nse = float(1.0 - np.sum((s - o) ** 2) / denom) if denom != 0 else np.nan
    if np.std(o) == 0 or np.std(s) == 0 or o.mean() == 0:
        kge = np.nan
    else:
        r = float(np.corrcoef(s, o)[0, 1])
        alpha = float(np.std(s) / np.std(o))
        beta = float(np.mean(s) / np.mean(o))
        kge = float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return nse, kge


def load_eval(census_run_dir: Path, staids: list[str]) -> pd.DataFrame:
    cols = ["staid", "eval_nse_15yr", "eval_kge_15yr"]
    if zarr is None:
        print("warning: zarr not importable; eval_nse_15yr/eval_kge_15yr left as NaN")
        return pd.DataFrame({"staid": staids, "eval_nse_15yr": np.nan, "eval_kge_15yr": np.nan})[cols]
    candidates = [census_run_dir / "eval" / "predictions.zarr", Path.cwd() / census_run_dir / "eval" / "predictions.zarr"]
    zarr_path = next((c for c in candidates if c.exists()), None)
    if zarr_path is None:
        print(f"warning: no eval predictions.zarr found at {candidates[0]}; eval_nse_15yr/eval_kge_15yr left as NaN")
        return pd.DataFrame({"staid": staids, "eval_nse_15yr": np.nan, "eval_kge_15yr": np.nan})[cols]

    z = zarr.open(str(zarr_path), mode="r")
    gage_ids = [bytes(row).decode("ascii", errors="ignore").strip("\x00") for row in z["gage_ids"][:]]
    predictions = z["predictions"][:]
    observations = z["observations"][:]
    row_of = {g: i for i, g in enumerate(gage_ids)}

    rows = []
    for staid in staids:
        i = row_of.get(staid)
        if i is None:
            rows.append((staid, np.nan, np.nan))
            continue
        nse, kge = nse_kge(predictions[i], observations[i])
        rows.append((staid, nse, kge))
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------- main
def build(census_dir: Path, diag_dir: Path | None) -> pd.DataFrame:
    manifest = load_manifest(census_dir)
    arm = pick_arm(manifest)["name"]
    alpha_max = alpha_max_of(manifest)
    run_id = pick_arm(manifest)["run_id"]
    census_run_dir = Path(pick_arm(manifest)["run_dir"])

    gauges = pd.read_csv(census_dir / "gauges.csv", dtype={"staid": str}, keep_default_na=False)
    staids = gauges["staid"].tolist()

    summary = load_census_summary(census_dir, arm)

    nc_rows = []
    n_missing_nc = 0
    for staid in staids:
        p = census_dir / arm / "gauges" / f"{staid}.nc"
        if not p.exists():
            n_missing_nc += 1
            continue
        row = extract_census_nc_row(p, alpha_max)
        row["staid"] = staid
        nc_rows.append(row)
    nc_df = pd.DataFrame(nc_rows)

    df = pd.DataFrame({"staid": staids}).merge(summary, on="staid", how="left").merge(nc_df, on="staid", how="left")
    n_missing_summary = int(df["nse0"].isna().sum())

    df["lambda1"] = df["lambda1"].astype(float)
    df["lambda2"] = df["lambda2"].astype(float)
    df["anisotropy"] = df["lambda1"] / df["lambda2"]

    # ------------------------------------------------------------- diag
    if diag_dir is not None:
        diag_manifest = load_manifest(diag_dir)
        diag_arm = pick_arm(diag_manifest)["name"]
        diag_rows = []
        n_missing_diag = 0
        for staid in staids:
            p = diag_dir / diag_arm / "gauges" / f"{staid}.nc"
            if not p.exists():
                n_missing_diag += 1
                row = {k: np.nan for k in DIAG_COLS + DIAG_GRAD_COLS}
            else:
                row = extract_diag_row(p)
            row["staid"] = staid
            diag_rows.append(row)
        diag_df = pd.DataFrame(diag_rows)
        df = df.merge(diag_df, on="staid", how="left")
        if n_missing_diag:
            print(f"diag: {n_missing_diag}/{len(staids)} gauges have no {diag_dir}/{diag_arm}/gauges/<staid>.nc")

    # -------------------------------------------------------------- gages-ii
    gages_ii = load_gages_ii(GAGES_II_DBF)
    gages_ii_cols = ["DRAIN_SQKM", "HUC02", "AGGECOREGI", "CLASS", "STATE", "LAT_GAGE", "LNG_GAGE"]
    if gages_ii is not None:
        before = len(df)
        df = df.merge(gages_ii, left_on="staid", right_on="STAID", how="left")
        df = df.drop(columns=["STAID"])
        n_missing_gages_ii = int(df["HUC02"].isna().sum())
        assert len(df) == before
    else:
        for c in gages_ii_cols:
            df[c] = np.nan
        n_missing_gages_ii = len(df)
        print(f"warning: GAGES-II table unavailable at {GAGES_II_DBF}; {gages_ii_cols} left as NaN")

    # -------------------------------------------------------------------- merit
    # Assigned positionally (not merged on comid_gauge): gauge-reach COMIDs
    # are not unique across gauges (nested USGS sites), so a value-merge
    # fans rows out.
    comid_arr = df["comid_gauge"].to_numpy(dtype=float)
    valid_comid = np.isfinite(comid_arr)
    merit_query = comid_arr[valid_comid].astype(np.int64)
    merit_valid = load_merit_attrs(merit_query, MERIT_ATTR_NC)
    for c in MERIT_VARS:
        df[c] = np.nan
        df.loc[valid_comid, c] = merit_valid[c].to_numpy()
    n_missing_merit = int(df["log10_uparea"].isna().sum())
    df = df.drop(columns=["comid_gauge"])

    # ---------------------------------------------------------------------- eval
    eval_df = load_eval(census_run_dir, staids)
    df = df.merge(eval_df, on="staid", how="left")

    column_order = [
        "staid", "n_reach", "sigma", "nse0", "nse_star", "gain", "kge0", "kge_star",
        "alpha_n_star", "alpha_q_star", "mult_n", "mult_q",
        "hit_range_bound", "newton_iters", "used_gradient_fallback", "box_edge",
        "grad0_n", "grad0_q", "H_nn", "H_qq", "H_nq", "H_qq_natural", "H_nq_natural",
        "lambda1", "lambda2", "anisotropy",
        "n0_med", "q0_med", "slope_gauge", "obs_mean_q_m3s", "obs_n_valid_days",
        "depth_gauge_mean_flow", "width_gauge", "total_length_km", "mean_reach_length_km",
    ]
    if diag_dir is not None:
        column_order += DIAG_COLS + DIAG_GRAD_COLS
    column_order += gages_ii_cols + MERIT_VARS + ["eval_nse_15yr", "eval_kge_15yr"]
    df = df[column_order]

    print(f"census gauges: {len(staids)}")
    print(f"  missing summary.csv row: {n_missing_summary}")
    print(f"  missing gauges/<staid>.nc: {n_missing_nc}")
    print(f"  missing GAGES-II match: {n_missing_gages_ii}")
    print(f"  missing MERIT attribute match: {n_missing_merit}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("census_dir", type=Path)
    ap.add_argument("--diag", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out if args.out is not None else args.census_dir / "figures" / "covariates.csv"
    df = build(args.census_dir, args.diag)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {len(df.columns)} columns)")

    print("\ncolumn coverage (non-NaN counts):")
    for c in df.columns:
        print(f"  {c}: {df[c].notna().sum()}")
    print("\nfirst 5 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(5))


if __name__ == "__main__":
    main()
