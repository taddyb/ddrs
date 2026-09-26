#!/usr/bin/env python
"""Reservoir benchmark: up to ten dams per HUC2, each paired with the evaluation gauge just below it.

Design (user, 2026-09-26; findings `research/findings/2026-09-26-dam-benchmark.md`):
  pool       GRanD dams in the NWM / RFC-DA reservoir table that map to a MERIT COMID
             (`/mnt/ssd1/data/resops/derived/grand_to_merit_comid.csv`, 2,177 dams)
  gauge      the nearest downstream gauge of the 2,365-gauge eval population (the gauge whose
             upstream `order` contains the dam COMID with the fewest reaches), with
             DRAIN_SQKM(gauge) / HydroLAKES Wshd_area(dam) <= MAX_AREA_RATIO
  regulated  the gauge carries NWIS peak-flow qualification code 6 ("affected by regulation or
             diversion") in at least one peak of WY1996-2010
  one dam per gauge: when several dams share a gauge, keep the one nearest it (area ratio closest
             to 1), whose release the gauge records; the largest upstream volume is a tag
  HUC2       HUC02 of the gauge (GAGES-II)
  selection  per HUC2, fill up to TARGET: tier 1 = ResOpsUS dams with >= 5 years of overlapping
             inflow and outflow in WY1996-2010, tier 2 = other ResOpsUS dams, tier 3 = the rest;
             within a tier take the needed count evenly spaced in gauge drainage-area rank
             (deterministic, the regional census's convention). Regions short of TARGET keep all.
  tags       release data per dam (ResOpsUS coverage, ISTARF-CONUS fit, GloFAS Qf, GRanD purpose),
             gauge DOR and code-6 years, trained-model and summed-Q' NSE/KGE at the gauge.

Inputs off-repo: /mnt/ssd1/data/resops/, /mnt/ssd1/data/usgs_regulation/derived/, the gauges
adjacency store, GAGES-II, and `output/dam_sandbox/regulation_by_gauge.csv` from
`experiments/reservoir/regulation_sizing.py` (run that first).
Outputs: experiments/reservoir/benchmark/dam_benchmark.csv, dam_benchmark_summary.json, and the
full candidate table output/dam_sandbox/dam_benchmark_candidates.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import zarr

TARGET = 10
MAX_AREA_RATIO = 1.5
MIN_OVERLAP_DAYS = 5 * 365
W0, W1 = pd.Timestamp("1995-10-01"), pd.Timestamp("2010-09-30")

HERE = Path(__file__).resolve().parent
RES = Path("/mnt/ssd1/data/resops")
REG = Path("/mnt/ssd1/data/usgs_regulation/derived")
RUN = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test")
SIZING = Path("/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv")
CANDIDATES = Path("/home/tbindas/projects/ddrs/output/dam_sandbox/dam_benchmark_candidates.csv")

# ---- dams and their attributes
dams = pd.read_csv(RES / "derived/grand_to_merit_comid.csv")
hl = pyogrio.read_dataframe("/home/tbindas/projects/ddr/data/hydrolakes/HydroLAKES_polys_v10.shp",
                            columns=["Hylak_id", "Wshd_area", "Vol_total", "Res_time", "Dis_avg", "Lake_type"],
                            read_geometry=False)
dams = dams.merge(hl, left_on="HYLAK_ID", right_on="Hylak_id", how="left").drop(columns="Hylak_id")
ist = pd.read_csv(RES / "istarf_conus/ISTARF-CONUS.csv")[["GRanD_ID", "fit"]].rename(
    columns={"GRanD_ID": "GRAND_ID", "fit": "istarf_fit"})
dams = dams.merge(ist, on="GRAND_ID", how="left")
cars = RES / "resopsus_cars/v1.0/attributes"
glofas = pd.read_csv(cars / "glofas.csv")[["GRAND_ID", "Qf", "Qn", "Qmin"]].rename(
    columns={"Qf": "glofas_Qf", "Qn": "glofas_Qn", "Qmin": "glofas_Qmin"})
dams = dams.merge(glofas, on="GRAND_ID", how="left")
grand = pd.read_csv(cars / "grand.csv")
purpose_cols = [f"MAIN_{p}" for p in ["ELEC", "FCON", "FISH", "IRRI", "NAVI", "OTHR", "RECR", "SUPP"]]
flags = grand[purpose_cols].fillna(0)
grand["main_purpose"] = flags.idxmax(axis=1).str.replace("MAIN_", "").where(flags.max(axis=1) > 0)
dams = dams.merge(grand[["GRAND_ID", "CAP_MCM", "DOR_PC", "YEAR", "main_purpose"]], on="GRAND_ID", how="left")

overlap = {}
for g in dams.loc[dams.IN_RESOPSUS.astype(bool), "GRAND_ID"]:
    f = RES / f"resopsus/ResOpsUS/time_series_all/ResOpsUS_{g}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f, na_values=["NA"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[(d.date >= W0) & (d.date <= W1)]
    overlap[g] = int((d.inflow.notna() & d.outflow.notna()).sum())
dams["resops_overlap_days"] = dams.GRAND_ID.map(overlap).fillna(0).astype(int)

# ---- gauges
g2 = pyogrio.read_dataframe("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.shp",
                            columns=["STAID", "STANAME", "HUC02", "DRAIN_SQKM", "CLASS"], read_geometry=False)
g2["STAID"] = g2.STAID.astype(str).str.zfill(8)
g2 = g2.set_index("STAID")
z = zarr.open(str(RUN / "eval/predictions.zarr"), mode="r")
eval_ids = set(bytes(r).decode().strip("\x00") for r in z["gage_ids"][:])
skill = pd.read_csv(SIZING, dtype={"STAID": str}).set_index("STAID")
peaks = pd.read_csv(REG / "peak_regulation_summary.csv", dtype={"STAID": str}).set_index("STAID")

# ---- nearest downstream eval gauge per dam
adj = zarr.open("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
dcom = dams.COMID.to_numpy()
below = {int(c): [] for c in dcom}
for k in adj.group_keys():
    if k not in eval_ids:
        continue
    o = adj[k]["order"][:]
    for c in o[np.isin(o, dcom)]:
        below[int(c)].append((len(o), k))
dams["gauge"] = [min(below[int(c)])[1] if below[int(c)] else None for c in dams.COMID]
cand = dams.dropna(subset=["gauge"]).copy()
cand["gauge_name"] = cand.gauge.map(g2.STANAME)
cand["huc2"] = cand.gauge.map(g2.HUC02).astype(str).str[:2]
cand["gauge_area_km2"] = cand.gauge.map(g2.DRAIN_SQKM)
cand["gauge_class"] = cand.gauge.map(g2.CLASS)
cand["area_ratio"] = cand.gauge_area_km2 / cand.Wshd_area
cand["code6_years_wy1996_2010"] = cand.gauge.map(peaks.n_years_code6_wy1996_2010).fillna(0).astype(int)
cand["peak_years_wy1996_2010"] = cand.gauge.map(peaks.n_peak_years_wy1996_2010).fillna(0).astype(int)
for col in ["dor", "nse_trained", "nse_base", "kge_trained", "kge_base", "beta_trained"]:
    cand[f"gauge_{col}"] = cand.gauge.map(skill[col])
gii = pd.read_csv(REG / "gagesii_regulation.csv", dtype={"STAID": str})
gii["STAID"] = gii.STAID.str.zfill(8)
gii = gii.set_index("STAID")
for col in ["STOR_NOR_2009", "MAJ_NDAMS_2009", "RAW_DIS_NEAREST_MAJ_DAM", "HYDRO_DISTURB_INDX"]:
    cand[f"gii_{col}"] = cand.gauge.map(gii[col])
cand["n_dams_to_gauge"] = cand.groupby("gauge").GRAND_ID.transform("size")
cand["eligible"] = (cand.area_ratio <= MAX_AREA_RATIO) & (cand.code6_years_wy1996_2010 >= 1)
CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
cand.to_csv(CANDIDATES, index=False)

# one dam per gauge: the dam nearest the gauge (area ratio closest to 1), whose release the gauge
# records; the largest upstream volume among the gauge's candidate dams is kept as a tag
cand["max_vol_to_gauge"] = cand.groupby("gauge").Vol_total.transform("max")
elig = cand[cand.eligible].sort_values("area_ratio").drop_duplicates("gauge")
elig["tier"] = np.select([elig.resops_overlap_days >= MIN_OVERLAP_DAYS, elig.IN_RESOPSUS.astype(bool)], [1, 2], 3)


def spaced(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """k rows evenly spaced in gauge drainage-area rank (all rows when k >= len)."""
    df = df.sort_values("gauge_area_km2")
    if k >= len(df):
        return df
    idx = np.unique(np.round(np.linspace(0, len(df) - 1, k)).astype(int))
    return df.iloc[idx]


picked = []
for h, s in elig.groupby("huc2"):
    need = TARGET
    for t in (1, 2, 3):
        if need == 0:
            break
        take = spaced(s[s.tier == t], need)
        picked.append(take)
        need -= len(take)
bench = pd.concat(picked).sort_values(["huc2", "tier", "gauge_area_km2"])
cols = ["huc2", "tier", "GRAND_ID", "LAKE_NAME", "COMID", "gauge", "gauge_name", "gauge_area_km2", "Wshd_area",
        "area_ratio", "Vol_total", "Res_time", "Lake_type", "main_purpose", "CAP_MCM", "DOR_PC", "YEAR",
        "IN_RESOPSUS", "resops_overlap_days", "istarf_fit", "glofas_Qf", "glofas_Qn", "glofas_Qmin",
        "gauge_class", "code6_years_wy1996_2010", "peak_years_wy1996_2010", "n_dams_to_gauge", "max_vol_to_gauge",
        "gii_STOR_NOR_2009", "gii_MAJ_NDAMS_2009", "gii_RAW_DIS_NEAREST_MAJ_DAM", "gii_HYDRO_DISTURB_INDX", "gauge_dor",
        "gauge_nse_trained", "gauge_nse_base", "gauge_kge_trained", "gauge_kge_base", "gauge_beta_trained"]
bench[cols].to_csv(HERE / "dam_benchmark.csv", index=False)

# ---- summary: median NSE per HUC2 with its sample size and a bootstrap 95 % interval on the median
rng = np.random.default_rng(42)


def median_ci(x: pd.Series, n_boot: int = 2000) -> tuple[float, float]:
    x = x.dropna().to_numpy()
    if len(x) < 2:
        return (np.nan, np.nan)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


per_huc = bench.groupby("huc2").agg(n=("gauge", "size"), tier1=("tier", lambda t: int((t == 1).sum())),
                                    nse_trained=("gauge_nse_trained", "median"), nse_base=("gauge_nse_base", "median"),
                                    kge_trained=("gauge_kge_trained", "median"), dor=("gauge_dor", "median"))
ci = {h: median_ci(s.gauge_nse_trained) for h, s in bench.groupby("huc2")}
per_huc["nse_trained_ci_lo"] = [ci[h][0] for h in per_huc.index]
per_huc["nse_trained_ci_hi"] = [ci[h][1] for h in per_huc.index]
by_tier = bench.groupby("tier").agg(n=("gauge", "size"), nse_trained=("gauge_nse_trained", "median"),
                                    nse_base=("gauge_nse_base", "median"))
funnel = {
    "dams_in_pool": int(len(dams)),
    "with_downstream_eval_gauge": int(len(cand)),
    "area_ratio_ok": int((cand.area_ratio <= MAX_AREA_RATIO).sum()),
    "area_ratio_ok_and_code6": int(cand.eligible.sum()),
    "unique_gauges_eligible": int(len(elig)),
    "benchmark_dams": int(len(bench)),
}
summary = {"design": {"target_per_huc2": TARGET, "max_area_ratio": MAX_AREA_RATIO, "window": "WY1996-2010",
                      "min_resops_overlap_days_tier1": MIN_OVERLAP_DAYS, "skill_window": "WY1997-2010",
                      "run": RUN.name},
           "funnel": funnel,
           "per_huc2": per_huc.reset_index().to_dict(orient="records"),
           "by_tier": by_tier.reset_index().to_dict(orient="records"),
           "overall": {"n": int(len(bench)), "nse_trained": float(bench.gauge_nse_trained.median()),
                       "nse_trained_ci": median_ci(bench.gauge_nse_trained),
                       "nse_base": float(bench.gauge_nse_base.median()),
                       "kge_trained": float(bench.gauge_kge_trained.median())}}
json.dump(summary, open(HERE / "dam_benchmark_summary.json", "w"), indent=1, default=float)

print("funnel:", funnel)
print(per_huc.to_string(float_format=lambda x: f"{x:.3f}"))
print(by_tier.to_string(float_format=lambda x: f"{x:.3f}"))
print("overall:", summary["overall"])
print("area ratio of benchmark pairs: min", round(bench.area_ratio.min(), 3), "median", round(bench.area_ratio.median(), 3))
