#!/usr/bin/env python
"""How many gauges sit below NID dams (>= 10 MCM) that did not exist for part of the train or test window, and how
did they fare with the learned release (which activates every dam in every year)?

Dam years: NID "Year Completed" (experiments/reservoir/nid/nid_dams_in_eval_network.csv). A dam finished in year Y is
treated as absent before water year Y+1 (NID gives a year only). Train window WY1982-1995, test WY1996-2010.
Per gauge: dams >= 10 MCM in its upstream network (merit_gages_conus_adjacency.zarr `order`). Paired changes from
paired_full_run.csv (seed 42) and paired_full_run_seed43.csv. Writes dam_age_check.json next to this script.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
dams = pd.read_csv(HERE.parent / "nid" / "nid_dams_in_eval_network.csv", low_memory=False)
big = dams[dams.storage_mcm >= 10]
adj = zarr.open("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
p42 = pd.read_csv(HERE / "paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
p43 = pd.read_csv(HERE / "paired_full_run_seed43.csv", dtype={"STAID": str}).set_index("STAID")

rows = []
for s in p42.index:
    up = set(adj[s]["order"][:].astype(np.int64).tolist())
    d = big[big.COMID.isin(up)]
    if len(d) == 0:
        continue
    yrs = d.year
    rows.append(dict(STAID=s, n_dams=len(d), n_after_1981=int((yrs > 1981).sum()), n_after_1995=int((yrs > 1995).sum()),
                     n_no_year=int(yrs.isna().sum()), d42=p42.dnse[s], d43=p43.dnse[s]))
g = pd.DataFrame(rows).set_index("STAID")
g["group"] = np.select([g.n_after_1995 > 0, g.n_after_1981 > 0], ["dam built during test", "dam built during train"],
                       "all dams built by 1981")


def summary(x):
    return dict(n=int(len(x)), median_d42=round(float(x.d42.median()), 4), median_d43=round(float(x.d43.median()), 4),
                up42=int((x.d42 > 0).sum()), up43=int((x.d43 > 0).sum()))


res = dict(
    dams_ge10mcm_in_eval_networks=len(big), built_after_1981=int((big.year > 1981).sum()),
    built_after_1995=int((big.year > 1995).sum()), no_year=int(big.year.isna().sum()),
    gauges_with_dam=len(g), gauges_with_any_no_year_dam=int((g.n_no_year > 0).sum()),
    by_group={k: summary(v) for k, v in g.groupby("group")},
)
json.dump(res, open(HERE / "dam_age_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
