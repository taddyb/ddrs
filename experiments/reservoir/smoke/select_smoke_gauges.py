#!/usr/bin/env python
"""Smoke-test gauge set for the learned dam release: a few gauges below NID dams in every HUC2, plus controls.

Built before any release code exists, so the implementation has a small, fast set with known expected results.
Per HUC2 (GAGES-II HUC02 of the gauge), from the 2,365 training / eval gauges:
  dam gauges (up to 2)  exactly one NID dam >= 10 MCM upstream (one clean signal); gauge drainage area <= 1.5x the
                        dam reach's upstream area (the gauge sees the release); dam completed by 1980 (it exists
                        through the 1981-1995 training window); snap class A, B or D (drainage-area checked);
                        gauge area <= 25,000 km2 (keeps the smoke network small); >= 80 % daily observations in
                        both WY1982-1995 and WY1996-2010. Ranked by NID degree of regulation (normal storage /
                        mean annual flow); the first, then the next with a different primary purpose (else the next).
  relaxed               a region with no dam gauge under those rules gets one with area ratio <= 10 (flagged).
  control (1)           no NID dam upstream, no NWIS peak code 6 in WY1996-2010, same coverage rule, drainage area
                        closest (log) to the region's dam gauges. Joint training can shift roughness everywhere, so
                        the smoke test must show undammed gauges do not get worse.
Writes, next to this script: smoke_gauges.csv (the set, one row per gauge), gages_smoke.csv (gages_3000.csv
format, for data_sources.gages), smoke_dams.csv (every NID dam >= 10 MCM in the smoke network, with the features a
release head would read), smoke_summary.json. Run under ~/projects/ddr/.venv.
"""
from __future__ import annotations

import json
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import pyogrio
import xarray as xr
import zarr

HERE = Path(__file__).resolve().parent
NIDDIR = HERE.parent / "nid"
ADJ = zarr.open("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
G3 = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str})
REG = pd.read_csv("/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
PEAK = pd.read_csv("/mnt/ssd1/data/usgs_regulation/derived/peak_regulation_summary.csv", dtype={"STAID": str}).set_index("STAID")
GII = pyogrio.read_dataframe("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.shp",
                             columns=["STAID", "HUC02"], read_geometry=False)
GII["STAID"] = GII.STAID.astype(str).str.zfill(8)
HUC = GII.set_index("STAID").HUC02.astype(str).str[:2]
PER_HUC, MIN_COVER, MAX_AREA = 2, 0.8, 25000.0

bygauge = pd.read_csv(NIDDIR / "nid_dams_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
dams = pd.read_csv(NIDDIR / "nid_dams_in_eval_network.csv", low_memory=False)
big = dams[dams.storage_mcm >= 10]

# observation coverage in both windows, from the store ddrs trains on
repo = icechunk.Repository.open(icechunk.local_filesystem_storage("/mnt/ssd1/data/icechunk/usgs_daily_observations"))
obs = xr.open_zarr(repo.readonly_session("main").store, consolidated=False).streamflow
ids = [s for s in REG.index if s in set(obs.gage_id.values)]
o = obs.sel(gage_id=ids)
wy = o.time.dt.year + (o.time.dt.month >= 10)
cov_tr = o.where((wy >= 1982) & (wy <= 1995)).notnull().sum("time").compute() / int(((wy >= 1982) & (wy <= 1995)).sum())
cov_te = o.where((wy >= 1996) & (wy <= 2010)).notnull().sum("time").compute() / int(((wy >= 1996) & (wy <= 2010)).sum())
cover = pd.DataFrame({"cover_train": cov_tr.to_pandas(), "cover_test": cov_te.to_pandas()})

rows = []
for s in REG.index:
    up = set(ADJ[s]["order"][:].astype(np.int64).tolist()) if s in ADJ else set()
    b = big[big.COMID.isin(up)]
    rows.append(dict(STAID=s, huc2=HUC.get(s), n_big=len(b), dam_index=b.index[0] if len(b) == 1 else -1))
cand = pd.DataFrame(rows).set_index("STAID")
cand = cand.join(cover).join(bygauge[["area_km2", "n_nid", "nid_dor", "nse_trained"]])
cand["code6"] = cand.index.map(lambda s: PEAK.n_years_code6_wy1996_2010.get(s, 0) > 0)
covered = (cand.cover_train >= MIN_COVER) & (cand.cover_test >= MIN_COVER)

d = cand[(cand.n_big == 1) & covered & (cand.area_km2 <= MAX_AREA)].copy()
dd = big.loc[d.dam_index.values]
d["dam_nid_id"], d["dam_name"], d["dam_COMID"] = dd.nid_id.values, dd.name.values, dd.COMID.values
d["dam_storage_mcm"], d["dam_da_km2"], d["dam_reach_uparea_km2"] = dd.storage_mcm.values, dd.da_km2.values, dd.reach_uparea_km2.values
d["dam_max_discharge_m3s"], d["dam_purpose"], d["dam_year"] = dd.max_discharge_m3s.values, dd.primary_purpose.values, dd.year.values
d["dam_snap_class"], d["dam_height_m"] = dd.snap_class.values, dd.height_m.values
d["area_ratio"] = d.area_km2 / d.dam_reach_uparea_km2
d_any = d[(d.dam_year <= 1980) & d.dam_snap_class.isin(["A", "B", "D"])]
d = d_any[d_any.area_ratio <= 1.5]

picked = []
for h, s in d.sort_values("nid_dor", ascending=False).groupby("huc2", sort=True):
    first = s.iloc[0]
    rest = s.iloc[1:]
    other = rest[rest.dam_purpose != first.dam_purpose]
    second = other.iloc[0] if len(other) else (rest.iloc[0] if len(rest) else None)
    picked += [first.name] + ([second.name] if second is not None else [])
dam_set = d.loc[picked].assign(role="dam", relaxed=False)
# Regions with no strict candidate (HUC 08, 09 on 2026-09-26): one "relaxed" dam gauge, the single-large-dam gauge
# with the smallest area ratio up to 10 (the dam is further up, so its release is diluted at the gauge).
missing = sorted(set(cand.huc2.dropna()) - set(dam_set.huc2))
relaxed = d_any[d_any.huc2.isin(missing) & (d_any.area_ratio <= 10)].sort_values("area_ratio").groupby("huc2").head(1)
dam_set = pd.concat([dam_set, relaxed.assign(role="dam", relaxed=True)])

ctrl_pool = cand[(cand.n_nid == 0) & ~cand.code6 & covered & (cand.area_km2 <= MAX_AREA)]
ctrl = []
for h, s in dam_set.groupby("huc2"):
    pool = ctrl_pool[ctrl_pool.huc2 == h]
    if len(pool):
        target = np.exp(np.log(s.area_km2).mean())
        ctrl.append((pool.area_km2.apply(np.log) - np.log(target)).abs().idxmin())
ctrl_set = cand.loc[ctrl].assign(role="control")

smoke = pd.concat([dam_set, ctrl_set]).sort_values(["huc2", "role"])
smoke["staname"] = smoke.index.map(G3.set_index("STAID").STANAME)
smoke.index.name = "STAID"
smoke.drop(columns=["dam_index"]).to_csv(HERE / "smoke_gauges.csv")
G3[G3.STAID.isin(smoke.index)].to_csv(HERE / "gages_smoke.csv", index=False)

net = set()
for s in smoke.index:
    net.update(ADJ[s]["order"][:].astype(np.int64).tolist())
sd = big[big.COMID.isin(net)]
sd.to_csv(HERE / "smoke_dams.csv", index=False)

huc_all = sorted(set(HUC.reindex(REG.index).dropna()))
summary = dict(
    dam_gauges=int((smoke.role == "dam").sum()), controls=int((smoke.role == "control").sum()),
    hucs_with_dam_gauge=sorted(dam_set.huc2.unique().tolist()), relaxed_gauges=dam_set.index[dam_set.relaxed].tolist(),
    hucs_without=sorted(set(huc_all) - set(dam_set.huc2)),
    candidates_per_huc=d.groupby("huc2").size().to_dict(),
    network_reaches=len(net), dams_ge10mcm_in_network=len(sd),
    median_nse_trained=dict(dam=float(dam_set.nse_trained.median()), control=float(ctrl_set.nse_trained.median())),
)
json.dump(summary, open(HERE / "smoke_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
pd.set_option("display.width", 250)
print(smoke[["huc2", "role", "staname", "area_km2", "dam_name", "dam_storage_mcm", "dam_purpose", "dam_year", "area_ratio",
             "nid_dor", "nse_trained"]].round(2).to_string())
