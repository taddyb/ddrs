#!/usr/bin/env python
"""For each DOR > 0.5 eval gauge: is there another gauge between it and its largest upstream reservoir?

That intermediate gauge is where an observed-release boundary condition could be inserted, and the
DOR > 0.5 gauge would then be scored on routing below it. When the gauge itself is the first gauge
below its largest reservoir, a boundary condition cannot help it (it would be the boundary); it can
only be dropped or modelled. Candidate boundary gauges: every gauge in the gauges adjacency store
(the gages_3000 list with a subgraph), whether or not it is in the eval set. A gauge u is between
reservoir r and gauge g when r is in order(u) and u's own catchment is in order(g), u != g.
Needs regulation_sizing.py's output. Result: experiments/reservoir/results/release_gauge_coverage.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

D = Path("/home/tbindas/projects/ddr/data")
CSV = Path("/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv")
OUT = Path(__file__).resolve().parent / "results" / "release_gauge_coverage.json"

mp = pd.read_csv(D / "merit_reservoir_params.csv")
rfc = pd.read_csv(D / "hydrolakes_rfc_da.csv")
import pyogrio  # noqa: E402

j = mp.merge(rfc, left_on=["lake_area_m2", "weir_elevation", "weir_length"], right_on=["LkArea", "WeirE", "WeirL"])
hl = pyogrio.read_dataframe(D / "hydrolakes/HydroLAKES_polys_v10.shp", columns=["Hylak_id", "Vol_total"], read_geometry=False)
res = j[["COMID", "Hylak_id"]].astype({"Hylak_id": int}).merge(hl, on="Hylak_id")
rcomids = res.COMID.to_numpy()
rvol = dict(zip(res.COMID, res.Vol_total))

adj = zarr.open(str(D / "merit_gages_conus_adjacency.zarr"), mode="r")
keys = sorted(adj.group_keys())
gauge_comid, res_in = {}, {}
for k in keys:
    grp = adj[k]
    gauge_comid[k] = int(grp.attrs["gage_catchment"])
    o = grp["order"][:]
    res_in[k] = set(o[np.isin(o, rcomids)].tolist())

df = pd.read_csv(CSV, dtype={"STAID": str})
hi = df[df.dor > 0.5]
rows = []
for g in hi.STAID:
    if g not in res_in or not res_in[g]:
        continue
    order_g = set(adj[g]["order"][:].tolist())
    r = max(res_in[g], key=lambda c: rvol[c])
    between = [u for u in keys if u != g and r in res_in[u] and gauge_comid[u] in order_g]
    rows.append(dict(STAID=g, main_res=int(r), main_vol_mcm=rvol[r], n_between=len(between),
                     vol_share_main=rvol[r] / sum(rvol[c] for c in res_in[g])))
cov = pd.DataFrame(rows)
has = cov.n_between > 0
summary = dict(n_dor_gt_05=len(hi), n_with_mapped_res=len(cov), n_with_gauge_between=int(has.sum()),
               n_first_gauge_below=int((~has).sum()), median_main_share=float(cov.vol_share_main.median()),
               adjacency_gauges=len(keys))
print(json.dumps(summary, indent=1))
nse = dict(zip(df.STAID, df.nse_trained))
print(f"median trained NSE: gauge between -> {np.nanmedian([nse[s] for s in cov.STAID[has]]):.3f}, "
      f"first gauge below -> {np.nanmedian([nse[s] for s in cov.STAID[~has]]):.3f}")
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=1)
