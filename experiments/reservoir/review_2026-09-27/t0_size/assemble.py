"""Assemble one table per dam COMID: learned T0 (two seeds), raw NID size measures, release-head inputs,
mean-flow proxies (nearest-gauge specific discharge; GRanD via ISTARF), residence times, and independent
T estimates (offline bucket fits on routed inflow; ResOpsUS fits; HydroLAKES Res_time)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size")
RUNS = {
    42: Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-55Z-train-and-test"),
    43: Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T10-31-50Z-train-and-test"),
}
FEAT = Path("/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/reservoir/release_head/dam_features.csv")
RES = Path("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir")
NID = RES / "nid" / "nid_dams_in_eval_network.csv"
GAGES = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
ISTARF = Path("/mnt/ssd1/data/resops/istarf_conus/ISTARF-CONUS.csv")
G2M = Path("/mnt/ssd1/data/resops/derived/grand_to_merit_comid.csv")


def nansum_or_nan(x):
    return float(x.sum()) if x.notna().any() else float("nan")


def aggregate_nid() -> pd.DataFrame:
    """Same aggregation as build_dam_features.py, plus lat/lon and raw storages of the largest dam."""
    nid = pd.read_csv(NID)
    keep = nid[(nid.storage_mcm >= 10.0) & nid.snap_class.isin({"A", "B", "C", "D"})].copy()
    rows = []
    for comid, g in keep.groupby("COMID", sort=True):
        big = g.sort_values("storage_mcm", ascending=False).iloc[0]
        rows.append(dict(
            COMID=int(comid), n_dams=len(g), name=big["name"], nid_id=big.nid_id,
            storage_mcm=float(g.storage_mcm.sum()),
            storage_normal_mcm=nansum_or_nan(g.storage_normal_mcm),
            storage_max_mcm=nansum_or_nan(g.storage_max_mcm),
            storage_nid_mcm=nansum_or_nan(g.storage_nid_mcm),
            surface_km2=nansum_or_nan(g.surface_km2),
            max_discharge_m3s=nansum_or_nan(g.max_discharge_m3s),
            height_m=float(g.height_m.max()),
            da_km2=big.da_km2, reach_uparea_km2=big.reach_uparea_km2,
            year=big.year, primary_purpose=big.primary_purpose if isinstance(big.primary_purpose, str) else "Unknown",
            lat=big.lat, lon=big.lon, snap_class=big.snap_class,
            on_a_gauge_reach=bool(big.on_a_gauge_reach),
        ))
    a = pd.DataFrame(rows)
    a["da_eff_km2"] = a.da_km2.where(a.da_km2 > 0, a.reach_uparea_km2)
    return a


def gauge_specific_discharge() -> pd.DataFrame:
    """Mean observed discharge per gauge over the baseline window (WY1996-2010), from the no-dam-agnostic
    baseline cache (observations.f32 is raw USGS daily obs, NaN = missing)."""
    b = RUNS[42] / "baseline"
    m = json.load(open(b / "manifest.json"))
    n_g, n_d = m["n_gauges"], m["n_days"]
    obs = np.fromfile(b / "observations.f32", dtype=np.float32).reshape(n_g, n_d)
    mean_obs = np.nanmean(obs, axis=1)
    n_valid = np.isfinite(obs).sum(axis=1)
    g = pd.DataFrame(dict(STAID=[str(s).zfill(8) for s in m["gage_ids"]], mean_obs_m3s=mean_obs, n_valid_days=n_valid))
    meta = pd.read_csv(GAGES, dtype={"STAID": str})
    meta["STAID"] = meta.STAID.str.zfill(8)
    g = g.merge(meta[["STAID", "DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "COMID"]], on="STAID", how="left")
    g = g[(g.n_valid_days > 365) & (g.DRAIN_SQKM > 0) & np.isfinite(g.mean_obs_m3s)].copy()
    g["q_spec"] = g.mean_obs_m3s / g.DRAIN_SQKM  # m3/s per km2
    return g


def nearest_gauge_q(dams: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Specific discharge from the gauge on the dam's own reach when there is one, else the nearest gauge
    (great-circle) whose drainage area is within a factor 30 of the dam's; also record distance."""
    glat = np.radians(g.LAT_GAGE.to_numpy()); glon = np.radians(g.LNG_GAGE.to_numpy())
    garea = g.DRAIN_SQKM.to_numpy(); gq = g.q_spec.to_numpy(); gcomid = g.COMID.to_numpy(); gid = g.STAID.to_numpy()
    out = []
    for r in dams.itertuples():
        same = np.where(gcomid == r.COMID)[0]
        if len(same):
            j = same[0]; d = 0.0; how = "on_reach"
        else:
            lat, lon = np.radians(r.lat), np.radians(r.lon)
            dist = 6371.0 * np.arccos(np.clip(np.sin(lat) * np.sin(glat) + np.cos(lat) * np.cos(glat) * np.cos(lon - glon), -1, 1))
            ok = (garea > r.da_eff_km2 / 30) & (garea < r.da_eff_km2 * 30)
            dist_ok = np.where(ok, dist, np.inf)
            j = int(np.argmin(dist_ok)); d = float(dist_ok[j]); how = "nearest"
            if not np.isfinite(d):
                j = int(np.argmin(dist)); d = float(dist[j]); how = "nearest_any"
        out.append(dict(COMID=r.COMID, q_gauge=gid[j], q_gauge_dist_km=d, q_gauge_how=how, q_spec=gq[j],
                        q_gauge_area_km2=garea[j]))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dams = aggregate_nid()
    for seed, run in RUNS.items():
        p = pd.read_csv(run / "release_params.csv")
        p = p.rename(columns={c: f"{c}_s{seed}" for c in p.columns if c != "COMID"})
        dams = dams.merge(p, on="COMID", how="left")
    feat = pd.read_csv(FEAT)
    feat_cols = [c for c in feat.columns if c not in ("COMID", "n_dams", "largest_nid_id", "largest_name", "storage_mcm", "year_completed")]
    dams = dams.merge(feat[["COMID"] + feat_cols + ["year_completed"]].rename(columns={c: f"f_{c}" for c in feat_cols}), on="COMID", how="left")

    # mean-flow proxy A: gauge specific discharge x dam drainage area
    g = gauge_specific_discharge()
    q = nearest_gauge_q(dams, g)
    dams = dams.merge(q, on="COMID", how="left")
    dams["qmean_gauge_m3s"] = dams.q_spec * dams.da_eff_km2
    # mean-flow proxy B: GRanD mean flow (HydroSHEDS-derived) via ISTARF-CONUS, joined on COMID
    ist = pd.read_csv(ISTARF, usecols=["GRanD_ID", "GRanD_NAME", "GRanD_CAP_MCM", "GRanD_MEANFLOW_CUMECS", "Obs_MEANFLOW_CUMECS"])
    g2m = pd.read_csv(G2M)
    ist = ist.merge(g2m[["GRAND_ID", "COMID"]], left_on="GRanD_ID", right_on="GRAND_ID", how="inner")
    ist = ist.sort_values("GRanD_CAP_MCM", ascending=False).drop_duplicates("COMID")
    ist["qmean_grand_m3s"] = pd.to_numeric(ist.GRanD_MEANFLOW_CUMECS, errors="coerce")
    ist["qmean_grand_obs_m3s"] = pd.to_numeric(ist.Obs_MEANFLOW_CUMECS, errors="coerce")
    dams = dams.merge(ist[["COMID", "GRanD_ID", "GRanD_NAME", "GRanD_CAP_MCM", "qmean_grand_m3s", "qmean_grand_obs_m3s"]], on="COMID", how="left")

    # residence times (days) = storage / mean flow
    for src in ("gauge", "grand"):
        qm = dams[f"qmean_{src}_m3s"]
        dams[f"res_time_{src}_d"] = dams.storage_mcm * 1e6 / (qm * 86400.0)
    dams["storage_per_area_m"] = dams.storage_mcm / dams.reach_uparea_km2  # MCM/km2 = m of runoff depth

    # independent T estimates
    e = pd.read_csv(RES / "smoke" / "expected_release_fit.csv", dtype={"STAID": str})
    e = e[e.dam_COMID.notna()].copy(); e["COMID"] = e.dam_COMID.astype(int)
    e = e.sort_values(["on_reach", "nse_seas_test"], ascending=[False, False]).drop_duplicates("COMID")
    e = e[["COMID", "STAID", "on_reach", "lin_T0", "seas_T0", "seas_T0_lo", "seas_T0_hi", "lin_T0_lo", "lin_T0_hi",
           "nse_seas_test", "nse_lin_test", "nse_pass_test", "d_seas", "d_lin"]].rename(
        columns={"STAID": "off_STAID", "on_reach": "off_on_reach", "lin_T0": "off_lin_T0", "seas_T0": "off_seas_T0",
                 "seas_T0_lo": "off_seas_lo", "seas_T0_hi": "off_seas_hi", "lin_T0_lo": "off_lin_lo", "lin_T0_hi": "off_lin_hi",
                 "nse_seas_test": "off_nse_seas", "nse_lin_test": "off_nse_lin", "nse_pass_test": "off_nse_pass",
                 "d_seas": "off_d_seas", "d_lin": "off_d_lin"})
    dams = dams.merge(e, on="COMID", how="left")
    r = pd.read_csv(RES / "benchmark" / "reservoir_T_fits.csv")
    r = r[r.status == "fitted"][["COMID", "LAKE_NAME", "T_days", "T_lo", "T_hi", "on_upper_wall", "nse_fit", "mean_inflow", "inflow_over_release"]]
    r = r.rename(columns={"LAKE_NAME": "resops_name", "T_days": "resops_T", "T_lo": "resops_T_lo", "T_hi": "resops_T_hi",
                          "on_upper_wall": "resops_wall", "nse_fit": "resops_nse", "mean_inflow": "resops_mean_inflow",
                          "inflow_over_release": "resops_in_over_out"}).drop_duplicates("COMID")
    dams = dams.merge(r, on="COMID", how="left")
    hb = pd.read_csv(RES / "benchmark" / "dam_benchmark.csv")[["COMID", "Res_time", "Vol_total", "DOR_PC"]].drop_duplicates("COMID")
    dams = dams.merge(hb.rename(columns={"Res_time": "hydrolakes_res_time_d", "Vol_total": "hydrolakes_vol_mcm", "DOR_PC": "grand_dor_pc"}), on="COMID", how="left")

    dams.to_csv(OUT / "t0_size_table.csv", index=False)
    summary = dict(
        n_comids=int(len(dams)),
        n_T0_s42=int(dams.T0_days_s42.notna().sum()), n_T0_s43=int(dams.T0_days_s43.notna().sum()),
        n_features=int(dams.f_log10_storage.notna().sum()),
        q_gauge_how=dams.q_gauge_how.value_counts().to_dict(),
        q_gauge_dist_km_quantiles=dams.q_gauge_dist_km.quantile([.1, .5, .9]).round(1).to_dict(),
        n_grand_meanflow=int(dams.qmean_grand_m3s.notna().sum()),
        n_grand_obs_meanflow=int(dams.qmean_grand_obs_m3s.notna().sum()),
        n_offline_fit=int(dams.off_seas_T0.notna().sum()), n_offline_on_reach=int((dams.off_on_reach == True).sum()),
        n_resops=int(dams.resops_T.notna().sum()), n_hydrolakes=int(dams.hydrolakes_res_time_d.notna().sum()),
        purposes=dams.primary_purpose.value_counts().to_dict(),
    )
    # agreement of the two mean-flow proxies
    both = dams[dams.qmean_grand_m3s.notna() & dams.qmean_gauge_m3s.notna()]
    lr = np.log10(both.qmean_gauge_m3s / both.qmean_grand_m3s)
    from scipy.stats import spearmanr
    summary["qmean_gauge_vs_grand"] = dict(n=int(len(both)), spearman=float(spearmanr(both.qmean_gauge_m3s, both.qmean_grand_m3s).statistic),
                                          log10_ratio_median=float(lr.median()), log10_ratio_iqr=[float(lr.quantile(.25)), float(lr.quantile(.75))])
    json.dump(summary, open(OUT / "assemble_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
