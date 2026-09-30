#!/usr/bin/env python
"""Which reservoir attributes the 121 benchmark dams have, and which each reservoir option lacks.

Per dam, from every source on the workstation:
  GRanD (via ResOpsUS-CARS `grand.csv`, which holds only ResOpsUS dams): purpose, year, capacity, DOR
  ISTARF-CONUS: GRanD capacity for all dams, starfit rule (fit = full / extrapolated / storage_only),
      Release_max / Release_min (standardized: release / mean inflow - 1), observed mean inflow
  HydroLAKES (benchmark CSV): Vol_total, Res_time (full-volume residence time), Lake_type
  NWM / RFC-DA level-pool table (`merit_reservoir_params.csv`): weir and orifice geometry
  GloFAS (ResOpsUS-CARS `glofas.csv`): Qmin, Qn, Qf, Vmin, Vn, Vf
  ResOpsUS v2 daily series: inflow, outflow, storage day counts, overall and inside WY1997-2010
  fit_reservoir_T.py: the fitted linear-reservoir T (option C)
What each option needs:
  C  T.  Fitted from paired inflow/outflow; where ResOpsUS has outflow and storage but no inflow,
     inflow = outflow + dS/dt could be reconstructed (not done here, counted only).
  D  T and Q_max.  Candidates: ISTARF Release_max where fit = full, GloFAS Qf, the observed
     99th-percentile ResOpsUS release.  No maximum-release or spillway dataset (NID) is on disk.
  E  a rule: ISTARF fit = full (extrapolated rules are borrowed from another dam).
  B  observed release inside the eval window (ResOpsUS outflow days in WY1997-2010).
Also checked, secondary: the KAN head's 10 input attributes on every reach of each gauge's subgraph
(ddrs fills a NaN with the batch row mean, `data::statistics::fill_nans`, so a gap is silent).
Writes attribute_audit.csv and attribute_audit_summary.json next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

HERE = Path(__file__).resolve().parent
D = Path("/home/tbindas/projects/ddr/data")
RES = Path("/mnt/ssd1/data/resops")
TS_DIR = RES / "resopsus/ResOpsUS/time_series_all"
CARS = RES / "resopsus_cars/v1.0/attributes"
KAN_INPUTS = ["SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI", "meanslope", "log10_uparea",
              "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity"]
YEARS3 = 3 * 365

bench = pd.read_csv(HERE / "dam_benchmark.csv", dtype={"huc2": str, "gauge": str})
fits = pd.read_csv(HERE / "reservoir_T_fits.csv", dtype={"huc2": str, "gauge": str}).set_index("GRAND_ID")
ist = pd.read_csv(RES / "istarf_conus/ISTARF-CONUS.csv", na_values=["NA"]).set_index("GRanD_ID")
grand = pd.read_csv(CARS / "grand.csv").set_index("GRAND_ID")
glofas = pd.read_csv(CARS / "glofas.csv").set_index("GRAND_ID")
nwm = pd.read_csv(D / "merit_reservoir_params.csv").set_index("COMID")


def series_counts(gid):
    f = TS_DIR / f"ResOpsUS_{gid}.csv"
    if not f.exists():
        return {}
    ts = pd.read_csv(f, parse_dates=["date"], na_values=["NA"])
    wy = ts.date.dt.year + (ts.date.dt.month >= 10)
    ev = (wy >= 1997) & (wy <= 2010)
    out = ts.outflow.to_numpy(float)
    return dict(
        inflow_days=int(ts.inflow.notna().sum()), outflow_days=int(ts.outflow.notna().sum()),
        storage_days=int(ts.storage.notna().sum()),
        outflow_and_storage_days=int((ts.outflow.notna() & ts.storage.notna()).sum()),
        outflow_days_eval=int((ts.outflow.notna() & ev).sum()),
        release_p99=float(np.nanpercentile(out, 99)) if np.isfinite(out).any() else np.nan,
    )


rows = []
for _, b in bench.iterrows():
    gid = int(b.GRAND_ID)
    f = fits.loc[gid]
    s = series_counts(gid)
    i = ist.loc[gid] if gid in ist.index else None
    istarf_fit = i.fit if i is not None else "none"
    mean_in = (i.Obs_MEANFLOW_CUMECS if i is not None and pd.notna(i.Obs_MEANFLOW_CUMECS)
               else (i.GRanD_MEANFLOW_CUMECS if i is not None else np.nan))
    rows.append(dict(
        huc2=b.huc2, tier=int(b.tier), gauge=b.gauge, GRAND_ID=gid, COMID=int(b.COMID), LAKE_NAME=b.LAKE_NAME,
        # descriptive attributes
        grand_purpose=gid in grand.index and bool(pd.notna(b.main_purpose)),
        grand_year=bool(pd.notna(b.YEAR)), grand_dor=bool(pd.notna(b.DOR_PC)),
        capacity_mcm=i.GRanD_CAP_MCM if i is not None else np.nan, hydrolakes_vol_mcm=b.Vol_total,
        hydrolakes_res_time_days=b.Res_time, nwm_level_pool=bool(int(b.COMID) in nwm.index),
        # option C
        T_status=f.status, T_days=f.get("T_days", np.nan),
        mass_balance_inflow_possible=bool(f.status != "fitted" and s.get("outflow_and_storage_days", 0) >= YEARS3),
        # option D
        istarf_Qmax=(mean_in * (1 + i.Release_max)) if i is not None and istarf_fit == "full" else np.nan,
        glofas_Qf=glofas.Qf.get(gid, np.nan), release_p99=s.get("release_p99", np.nan),
        # option E
        istarf_fit=istarf_fit, istarf_match=i.match if i is not None else np.nan,
        # option B
        outflow_days_eval=s.get("outflow_days_eval", 0),
        **{k: s.get(k, 0) for k in ("inflow_days", "outflow_days", "storage_days", "outflow_and_storage_days")},
    ))
df = pd.DataFrame(rows)
df["has_T"] = df.T_status == "fitted"
df["has_Qmax"] = df[["istarf_Qmax", "glofas_Qf", "release_p99"]].notna().any(axis=1)
df["has_Qmax_observed"] = df.istarf_Qmax.notna() | df.release_p99.notna()
df["has_rule_fitted"] = df.istarf_fit == "full"
df["has_release_eval_3y"] = df.outflow_days_eval >= YEARS3
df["option_C_ready"] = df.has_T
df["option_D_ready"] = df.has_T & df.has_Qmax_observed
df["option_E_ready"] = df.has_rule_fitted


def count(mask):
    return int(mask.sum())


summary = dict(
    n_dams=len(df),
    descriptive=dict(
        capacity_any=count(df.capacity_mcm.notna()), hydrolakes_volume=count(df.hydrolakes_vol_mcm.notna()),
        hydrolakes_res_time=count(df.hydrolakes_res_time_days.notna()), grand_purpose=count(df.grand_purpose),
        grand_year=count(df.grand_year), nwm_level_pool=count(df.nwm_level_pool),
        lake_name_blank=count(df.LAKE_NAME.isna()),
    ),
    option_C=dict(
        T_fitted=count(df.has_T), T_status=df.T_status.value_counts().to_dict(),
        mass_balance_inflow_possible=count(df.mass_balance_inflow_possible),
        no_source=count(~df.has_T & ~df.mass_balance_inflow_possible),
        hydrolakes_res_time_over_fitted_T_median=float((df.hydrolakes_res_time_days / df.T_days)[df.has_T].median()),
    ),
    option_D=dict(
        istarf_Qmax_full_fit=count(df.istarf_Qmax.notna()), glofas_Qf=count(df.glofas_Qf.notna()),
        observed_release_p99=count(df.release_p99.notna()), any_Qmax=count(df.has_Qmax),
        ready_T_and_observed_Qmax=count(df.option_D_ready),
    ),
    option_E=dict(istarf=df.istarf_fit.value_counts().to_dict(), ready=count(df.option_E_ready)),
    option_B=dict(release_in_eval_ge_3y=count(df.has_release_eval_3y),
                  release_in_eval_any=count(df.outflow_days_eval > 0)),
    by_huc2={h: dict(n=len(g), T=count(g.has_T), mass_balance=count(g.mass_balance_inflow_possible),
                     Qmax_obs=count(g.has_Qmax_observed), rule_full=count(g.has_rule_fitted),
                     release_eval=count(g.has_release_eval_3y)) for h, g in df.groupby("huc2")},
    by_tier={int(t): dict(n=len(g), T=count(g.has_T), mass_balance=count(g.mass_balance_inflow_possible),
                          rule_full=count(g.has_rule_fitted)) for t, g in df.groupby("tier")},
)

# ---- secondary: KAN input attributes over each gauge's subgraph
attrs = xr.open_dataset(D / "merit_global_attributes_v2.nc")[KAN_INPUTS].load()
a_idx = pd.Index(attrs.COMID.values.astype(np.int64))
A = np.stack([attrs[v].values for v in KAN_INPUTS], axis=1)
gadj = zarr.open(str(D / "merit_gages_conus_adjacency.zarr"), mode="r")
gages = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str}).set_index("STAID")
nan_reach, n_reach, dam_nan, on_gauge = [], [], [], []
for _, b in bench.iterrows():
    up = gadj[b.gauge]["order"][:].astype(np.int64)
    pos = a_idx.get_indexer(up)
    bad = (pos < 0) | np.isnan(A[np.clip(pos, 0, None)]).any(axis=1)
    nan_reach.append(int(bad.sum())); n_reach.append(len(up))
    dp = a_idx.get_indexer([int(b.COMID)])[0]
    dam_nan.append(";".join(v for j, v in enumerate(KAN_INPUTS) if dp < 0 or np.isnan(A[dp, j])))
    on_gauge.append(int(gages.loc[b.gauge, "COMID"]) == int(b.COMID))
df["subgraph_reaches"], df["subgraph_reaches_nan_kan_input"] = n_reach, nan_reach
df["dam_nan_kan_inputs"], df["dam_on_gauge_reach"] = dam_nan, on_gauge
summary["network_kan_inputs"] = dict(
    reaches=int(sum(n_reach)), reaches_nan=int(sum(nan_reach)),
    gauges_with_nan=int(sum(x > 0 for x in nan_reach)), dams_with_nan=int(sum(bool(x) for x in dam_nan)),
    dams_on_gauge_reach=int(sum(on_gauge)),
)

df.to_csv(HERE / "attribute_audit.csv", index=False)
json.dump(summary, open(HERE / "attribute_audit_summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
