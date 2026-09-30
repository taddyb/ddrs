#!/usr/bin/env python
"""Share of the training network's reaches that host an NID dam, or sit at or below one.

Network: the union of the upstream reach sets (`order` in merit_gages_conus_adjacency.zarr) of the 2,365 training /
eval gauges (regulation_by_gauge.csv, the gages_3000.csv population after the DA_VALID and headwater filters), and,
for comparison, of every gages_3000.csv gauge present in the adjacency store.
"Hosts a dam": the reach's COMID carries a snapped NID dam (/mnt/ssd1/data/nid/derived/nid_merit_comid.csv, classes
A, B, C, D). "Regulated": the reach hosts a dam or has one anywhere upstream inside the network, found by walking
each dam's reach down MERIT NextDownID. Shares by reach count and by reach length (MERIT lengthkm), for any snapped
dam and for dams of >= 1, 10, 100 MCM (normal storage, else NID storage).
Writes reach_dam_coverage.json next to this script. Run under ~/projects/ddr/.venv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import zarr

HERE = Path(__file__).resolve().parent
ADJ = zarr.open("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
REG = pd.read_csv("/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv", dtype={"STAID": str})
G3 = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str})
riv = pyogrio.read_dataframe("/mnt/ssd1/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp",
                             columns=["COMID", "NextDownID", "lengthkm"], read_geometry=False)
down = dict(zip(riv.COMID.values, riv.NextDownID.values))
length = dict(zip(riv.COMID.values, riv.lengthkm.values))
nid = pd.read_csv("/mnt/ssd1/data/nid/derived/nid_merit_comid.csv", low_memory=False)
nid = nid[nid.COMID.notna()].copy()
nid["COMID"] = nid.COMID.astype(np.int64)
nid["storage_mcm"] = nid.storage_normal_mcm.where(nid.storage_normal_mcm > 0, nid.storage_nid_mcm).fillna(0.0)


def union(staids):
    u = set()
    for s in staids:
        if s in ADJ:
            u.update(ADJ[s]["order"][:].astype(np.int64).tolist())
    return u


def coverage(net):
    L = np.array([length.get(c, 0.0) for c in net])
    total_len = L.sum()
    idx = {c: i for i, c in enumerate(net)}
    out = {"reaches": len(net), "length_km": float(total_len)}
    for lab, lo in [("any dam", -1.0), (">=1 MCM", 1.0), (">=10 MCM", 10.0), (">=100 MCM", 100.0)]:
        dams = set(nid.COMID[nid.storage_mcm >= lo].tolist()) & net
        host = np.zeros(len(net), bool)
        host[[idx[c] for c in dams]] = True
        reg = np.zeros(len(net), bool)
        for c in dams:  # walk down until leaving the network or joining an already-marked path
            while c in idx and not reg[idx[c]]:
                reg[idx[c]] = True
                c = down.get(c)
        out[lab] = dict(dam_reaches=len(dams), host_pct=round(100 * host.mean(), 2), host_len_pct=round(100 * L[host].sum() / total_len, 2),
                        regulated_pct=round(100 * reg.mean(), 1), regulated_len_pct=round(100 * L[reg].sum() / total_len, 1))
    return out


res = {"training_2365": coverage(union(REG.STAID)), "gages_3000_all": coverage(union(G3.STAID))}
json.dump(res, open(HERE / "reach_dam_coverage.json", "w"), indent=1)
print(json.dumps(res, indent=1))
