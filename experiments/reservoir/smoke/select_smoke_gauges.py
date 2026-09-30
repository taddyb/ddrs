#!/usr/bin/env python
"""Smoke-test gauge set for the learned dam release: every eval gauge just below a large NID dam, plus one matched
undammed control per dam gauge.

Built before any release code exists, so the implementation has a set with known expected results. From the 2,365
training / eval gauges (HUC2 = GAGES-II HUC02 of the gauge):
  dam gauges   at least one NID dam >= 10 MCM upstream; the NEAREST of them (upstream area closest to the gauge's)
               has gauge drainage area <= 3x its reach's upstream area (2026-09-26, user: 3x, up from 1.5x), so the gauge
               sees that dam's release (other
               large dams further up are allowed: the release law puts a bucket on every one of them); that dam was
               completed by 1980 (it exists through the 1981-1995 training window) and snapped by drainage-area match
               (class A, B or D); gauge area <= 25,000 km2 (keeps the network small); >= 80 % daily observations in
               both WY1982-1995 and WY1996-2010.
  relaxed      a region with no such gauge gets the one with the smallest area ratio up to 10 (flagged).
  controls     one per dam gauge: no NID dam upstream, no NWIS peak code 6 in WY1996-2010, same coverage rule, same
               HUC2, drainage area closest in log, drawn without replacement. When a region runs out, the unused
               gauge minimising |log area ratio| + distance / 1,000 km, from any region (flagged `control_cross_huc`).
               Joint training can shift roughness everywhere, and
               a release law fitted anywhere can smooth a flashy model, so dam gauges are judged against controls.
Writes, next to this script: smoke_gauges.csv (one row per gauge; dam columns describe the nearest large dam),
gages_smoke.csv (gages_3000.csv format, for data_sources.gages), smoke_dams.csv (every NID dam >= 10 MCM in the smoke
network, with the NID features a release head would read), smoke_summary.json. Run under ~/projects/ddr/.venv.
The 2026-09-26 first version (two dam gauges per region, 50 gauges) is commit 5a47623.
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
MIN_COVER, MAX_AREA, MAX_RATIO, RELAXED_RATIO = 0.8, 25000.0, 3.0, 10.0

bygauge = pd.read_csv(NIDDIR / "nid_dams_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
dams = pd.read_csv(NIDDIR / "nid_dams_in_eval_network.csv", low_memory=False)
big = dams[dams.storage_mcm >= 10]

repo = icechunk.Repository.open(icechunk.local_filesystem_storage("/mnt/ssd1/data/icechunk/usgs_daily_observations"))
obs = xr.open_zarr(repo.readonly_session("main").store, consolidated=False).streamflow
o = obs.sel(gage_id=[s for s in REG.index if s in set(obs.gage_id.values)])
wy = o.time.dt.year + (o.time.dt.month >= 10)
tr, te = (wy >= 1982) & (wy <= 1995), (wy >= 1996) & (wy <= 2010)
cover = pd.DataFrame({"cover_train": (o.where(tr).notnull().sum("time") / int(tr.sum())).compute().to_pandas(),
                      "cover_test": (o.where(te).notnull().sum("time") / int(te.sum())).compute().to_pandas()})

rows = []
for s in REG.index:
    up = set(ADJ[s]["order"][:].astype(np.int64).tolist()) if s in ADJ else set()
    b = big[big.COMID.isin(up)]
    area = bygauge.area_km2.get(s, np.nan)
    r = dict(STAID=s, huc2=HUC.get(s), n_big=len(b))
    if len(b):
        k = b.index[int(np.argmin(np.abs(np.log(area / b.reach_uparea_km2.values))))]
        n = big.loc[k]
        r.update(dam_nid_id=n.nid_id, dam_name=n["name"], dam_COMID=n.COMID, dam_storage_mcm=n.storage_mcm, dam_da_km2=n.da_km2,
                 dam_reach_uparea_km2=n.reach_uparea_km2, dam_max_discharge_m3s=n.max_discharge_m3s, dam_purpose=n.primary_purpose,
                 dam_year=n.year, dam_snap_class=n.snap_class, dam_height_m=n.height_m, area_ratio=area / n.reach_uparea_km2)
    rows.append(r)
cand = pd.DataFrame(rows).set_index("STAID").join(cover).join(bygauge[["area_km2", "n_nid", "nid_dor", "nse_trained"]])
cand["code6"] = cand.index.map(lambda s: PEAK.n_years_code6_wy1996_2010.get(s, 0) > 0)
covered = (cand.cover_train >= MIN_COVER) & (cand.cover_test >= MIN_COVER) & (cand.area_km2 <= MAX_AREA)
ok_dam = (cand.n_big > 0) & (cand.dam_year <= 1980) & cand.dam_snap_class.isin(["A", "B", "D"]) & covered

dam_set = cand[ok_dam & (cand.area_ratio <= MAX_RATIO)].assign(role="dam", relaxed=False)
missing = sorted(set(cand.huc2.dropna()) - set(dam_set.huc2))
relaxed = cand[ok_dam & cand.huc2.isin(missing) & (cand.area_ratio <= RELAXED_RATIO)].sort_values("area_ratio").groupby("huc2").head(1)
dam_set = pd.concat([dam_set, relaxed.assign(role="dam", relaxed=True)])
dam_set["cascade"] = dam_set.n_big > 1

pool = cand[(cand.n_nid == 0) & ~cand.code6 & covered].copy()
pool["la"] = np.log(pool.area_km2)
ll = G3.set_index("STAID")[["LAT_GAGE", "LNG_GAGE"]]
pool["lat"], pool["lon"] = np.radians(pool.index.map(ll.LAT_GAGE)), np.radians(pool.index.map(ll.LNG_GAGE))


def km(lat1, lon1, lat2, lon2):
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))


ctrl_rows, used = [], set()
for s, d in dam_set.sort_values("area_km2", ascending=False).iterrows():  # big basins first: fewer large controls
    la = np.log(d.area_km2)
    p = pool[~pool.index.isin(used)]
    same = p[p.huc2 == d.huc2]
    cross = len(same) == 0
    if not cross:
        pick = (same.la - la).abs().idxmin()
    else:  # a region ran out: trade area mismatch against distance, so borrowed controls stay nearby
        lat0, lon0 = np.radians(ll.LAT_GAGE[s]), np.radians(ll.LNG_GAGE[s])
        pick = ((p.la - la).abs() + km(lat0, lon0, p.lat, p.lon) / 1000.0).idxmin()
    used.add(pick)
    ctrl_rows.append(dict(STAID=pick, control_for=s, control_cross_huc=cross))
ctrl = pd.DataFrame(ctrl_rows).set_index("STAID")
ctrl_set = cand.loc[ctrl.index].join(ctrl).assign(role="control")

smoke = pd.concat([dam_set, ctrl_set]).sort_values(["huc2", "role", "STAID"])
smoke["staname"] = smoke.index.map(G3.set_index("STAID").STANAME)
smoke.index.name = "STAID"
smoke.to_csv(HERE / "smoke_gauges.csv")
G3[G3.STAID.isin(smoke.index)].to_csv(HERE / "gages_smoke.csv", index=False)

net = set()
for s in smoke.index:
    net.update(ADJ[s]["order"][:].astype(np.int64).tolist())
sd = big[big.COMID.isin(net)]
sd.to_csv(HERE / "smoke_dams.csv", index=False)
g3c = G3.set_index("STAID").COMID
on_reach = (dam_set.dam_COMID.astype("Int64").values == dam_set.index.map(g3c).astype("Int64").values)

summary = dict(
    dam_gauges=len(dam_set), relaxed=dam_set.index[dam_set.relaxed].tolist(), cascades=int(dam_set.cascade.sum()),
    dam_on_gauge_reach=int(on_reach.sum()), controls=len(ctrl_set), controls_cross_huc=int(ctrl_set.control_cross_huc.sum()),
    per_huc={h: dict(dam=int((dam_set.huc2 == h).sum()), control=int((ctrl_set.huc2 == h).sum()))
             for h in sorted(set(dam_set.huc2) | set(ctrl_set.huc2))},
    network_reaches=len(net), dams_ge10mcm_in_network=len(sd),
    median_nse_trained=dict(dam=float(dam_set.nse_trained.median()), control=float(ctrl_set.nse_trained.median())),
    purposes=dam_set.dam_purpose.value_counts().to_dict(),
)
json.dump(summary, open(HERE / "smoke_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
