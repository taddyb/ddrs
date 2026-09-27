#!/usr/bin/env python
"""Which NID dams sit inside the upstream network of each eval gauge, and what that changes.

Inputs:
  /mnt/ssd1/data/nid/derived/nid_merit_comid.csv   NID dams snapped to MERIT COMIDs
      (~/projects/remote_sensing_extraction/nid/snap_nid_to_merit.py; classes A, B = drainage-area matched, C = location only)
  ~/projects/ddr/data/merit_gages_conus_adjacency.zarr   per-gauge upstream COMIDs (`order`)
  ~/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv   the 2,365 eval gauges: observed mean flow, NWM-table
      reservoir count and DOR class (regulation_sizing.py)
  ~/projects/ddr/references/gage_info/gages_3000.csv   gauge COMID and drainage area
  ../benchmark/dam_benchmark.csv, ../benchmark/gauge_name_dam_flags.csv

Per gauge: NID dams upstream (all snapped, and >= 1 / >= 10 MCM), total storage (normal storage, NID storage where
normal is missing), NID degree of regulation = storage / mean annual flow volume, a dam on the gauge's own reach,
the largest dam and the nearest one (drainage area closest to the gauge's).
Writes nid_dams_by_gauge.csv, nid_dams_in_eval_network.csv (one row per dam inside any eval gauge's network: the
candidate dam table for training) and nid_gauge_summary.json next to this script. Run under ~/projects/ddr/.venv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
NID = Path("/mnt/ssd1/data/nid/derived/nid_merit_comid.csv")
ADJ = Path("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr")
REG = Path("/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv")
GAGES = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
BENCH = HERE.parent / "benchmark" / "dam_benchmark.csv"
NAMES = HERE.parent / "benchmark" / "gauge_name_dam_flags.csv"

nid = pd.read_csv(NID, low_memory=False)
nid = nid[nid.COMID.notna()].copy()
nid["COMID"] = nid.COMID.astype(np.int64)
nid["storage_mcm"] = nid.storage_normal_mcm.where(nid.storage_normal_mcm > 0, nid.storage_nid_mcm).fillna(0.0)
by_comid = nid.groupby("COMID")

reg = pd.read_csv(REG, dtype={"STAID": str}).set_index("STAID")
g3 = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")
bench = pd.read_csv(BENCH, dtype={"gauge": str})
names = pd.read_csv(NAMES, dtype={"STAID": str}).set_index("STAID")
adj = zarr.open(str(ADJ), mode="r")

rows, dam_rows = [], {}
for s in reg.index:
    up = adj[s]["order"][:].astype(np.int64) if s in adj else np.array([], np.int64)
    d = nid[nid.COMID.isin(up)]
    qmean = reg.qmean.get(s, np.nan)
    vol = d.storage_mcm.sum()
    gcomid = int(g3.COMID.get(s, -1))
    area = g3.DRAIN_SQKM.get(s, np.nan)
    big = d.sort_values("storage_mcm", ascending=False).head(1)
    dd = d[d.da_km2 > 0]
    nearest = dd.iloc[[int(np.argmin(np.abs(np.log(area / dd.da_km2.values))))]] if len(dd) else dd
    rows.append(dict(
        STAID=s, area_km2=area, qmean=qmean, n_upstream_reaches=len(up),
        n_nid=len(d), n_nid_ge1mcm=int((d.storage_mcm >= 1).sum()), n_nid_ge10mcm=int((d.storage_mcm >= 10).sum()),
        nid_storage_mcm=vol, nid_dor=vol * 1e6 / (qmean * 365.25 * 86400) if qmean > 0 else np.nan,
        nid_dor_max=d.storage_nid_mcm.fillna(0).sum() * 1e6 / (qmean * 365.25 * 86400) if qmean > 0 else np.nan,
        nid_on_gauge_reach=bool((d.COMID == gcomid).any()),
        largest_dam=big.name.iloc[0] if len(big) else None, largest_storage_mcm=big.storage_mcm.iloc[0] if len(big) else 0.0,
        nearest_dam=nearest.name.iloc[0] if len(nearest) else None,
        nearest_area_ratio=area / nearest.da_km2.iloc[0] if len(nearest) else np.nan,
        nwm_n_res=int(reg.n_res.get(s, 0)), nwm_dor=reg.dor.get(s, np.nan), nwm_cls=reg.cls.get(s, None),
        name_cat=names.cat.get(s, None), benchmark=s in set(bench.gauge), nse_trained=reg.nse_trained.get(s, np.nan)))
    for idx in d.index:
        dam_rows.setdefault(idx, []).append(s)
df = pd.DataFrame(rows)
df["nid_cls"] = np.select([df.n_nid == 0, df.nid_dor <= 0.1, df.nid_dor <= 0.5], ["none", "DOR<=0.1", "0.1<DOR<=0.5"], "DOR>0.5")
df.to_csv(HERE / "nid_dams_by_gauge.csv", index=False)

dams = nid.loc[list(dam_rows)].copy()
dams["n_eval_gauges_downstream"] = [len(dam_rows[i]) for i in dams.index]
dams["on_a_gauge_reach"] = dams.COMID.isin(set(g3.COMID.reindex(reg.index).dropna().astype(np.int64)))
dams.to_csv(HERE / "nid_dams_in_eval_network.csv", index=False)

hits = df[df.name_cat.isin(["below", "at dam / outlet"])]
bench_ok = [int(c) in set(nid.COMID) for c in bench.COMID]


def cnt(m):
    return int(np.asarray(m).sum())


summary = dict(
    n_eval_gauges=len(df),
    gauges_with_any_nid_dam=cnt(df.n_nid > 0), gauges_with_nid_ge1mcm=cnt(df.n_nid_ge1mcm > 0),
    gauges_with_nid_ge10mcm=cnt(df.n_nid_ge10mcm > 0), gauges_with_nwm_reservoir=cnt(df.nwm_n_res > 0),
    gauges_nid_ge10mcm_but_no_nwm=cnt((df.n_nid_ge10mcm > 0) & (df.nwm_n_res == 0)),
    gauges_with_dam_on_own_reach=cnt(df.nid_on_gauge_reach),
    dor_class_nid=df.nid_cls.value_counts().to_dict(), dor_class_nwm=df.nwm_cls.value_counts().to_dict(),
    dor_gt_05_by_max_storage=cnt(df.nid_dor_max > 0.5),
    dor_gt_05_nid_not_nwm=cnt((df.nid_cls == "DOR>0.5") & (df.nwm_cls != "DOR>0.5")),
    dor_gt_05_nwm_not_nid=cnt((df.nwm_cls == "DOR>0.5") & (df.nid_cls != "DOR>0.5")),
    median_nse_by_nid_cls=df.groupby("nid_cls").nse_trained.median().round(3).to_dict(),
    n_by_nid_cls=df.nid_cls.value_counts().to_dict(),
    dams_in_eval_networks=len(dams), dams_in_eval_networks_ge1mcm=cnt(dams.storage_mcm >= 1),
    dams_in_eval_networks_ge10mcm=cnt(dams.storage_mcm >= 10), dams_in_eval_networks_ge100mcm=cnt(dams.storage_mcm >= 100),
    dams_in_eval_networks_by_class=dams.snap_class.value_counts().to_dict(),
    dams_on_an_eval_gauge_reach=cnt(dams.on_a_gauge_reach),
    name_flagged_regulated=len(hits), name_flagged_with_nid_dam=cnt(hits.n_nid > 0),
    name_flagged_no_nwm_but_nid=cnt((hits.nwm_n_res == 0) & (hits.n_nid > 0)),
    name_flagged_no_nwm=cnt(hits.nwm_n_res == 0),
    benchmark_dam_comid_has_nid_dam=int(sum(bench_ok)), benchmark_n=len(bench),
)
json.dump(summary, open(HERE / "nid_gauge_summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
