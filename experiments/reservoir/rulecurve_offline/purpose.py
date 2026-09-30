"""Gain by dam purpose, per law and for the feature-limited variants (on-reach set; DOR>0.5 and all)."""
import json

import numpy as np
import pandas as pd

import rc
import summarise as S

df, dam, ctl = S.load()
pool = pd.read_csv(f"{rc.HERE}/pooled_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
feat = pd.read_csv(f"{rc.HERE}/features_on_reach_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
d = dam[dam.on_reach].join(pool).join(feat, rsuffix="_f")
c = ctl[dam.on_reach.values]
res = {}
for sub, idx in [("all", d.index), ("DOR>0.5", d.index[d.nid_dor > 0.5])]:
    x = d.loc[idx]; y = c.loc[idx]
    fl = x.dam_purpose == "Flood Risk Reduction"
    e = {}
    for nm in ["L1", "L2", "L3", "L3b", "L4", "L4pen", "L5", "cvL1", "cvL4", "cvL4pen", "pool_T0only", "pool_small_cal", "pool_small_rot", "pool_full_rot"]:
        g_d = x[f"{nm}_nse"] - x.L0_nse
        row = dict(flood=S.paired(g_d[fl]), other=S.paired(g_d[~fl]))
        if f"{nm}_nse" in y:
            g_c = (y[f"{nm}_nse"] - y.L0_nse).values
            row["flood_ctl"] = S.paired(g_c[fl.values])
            row["flood_did"] = S.paired(g_d[fl].values - g_c[fl.values])
        e[nm] = row
    res[sub] = e
    print(f"=== on-reach {sub}: flood n={int(fl.sum())}, other n={int((~fl).sum())}")
    for nm, row in e.items():
        print(f"  {nm:15s} flood {S.fmt(row['flood'])} | other {S.fmt(row['other'])}"
              + (f" | flood ctl {S.fmt(row['flood_ctl'])} | flood DiD {S.fmt(row['flood_did'])}" if 'flood_ctl' in row else ""))
# phase of the fitted rule curve at flood dams (doy of max storage S0, min S0)
dg = pd.read_csv(f"{rc.HERE}/diag_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
x = d.join(dg)
fl = x.dam_purpose == "Flood Risk Reduction"
for lab, m in [("flood", fl), ("other", ~fl)]:
    xx = x[m & (x.nid_dor > 0.5)]
    print(f"{lab} DOR>0.5 n={len(xx)}: S0 max doy q25/50/75 {np.percentile(xx.smax_doy, [25, 50, 75])}, "
          f"S0 min doy {np.percentile(xx.smin_doy, [25, 50, 75])}, swing/DOR {np.percentile(xx.swing_frac_annual / xx.nid_dor, [25, 50, 75]).round(3)}, "
          f"clim peak doy {np.percentile(xx.clim_peak_doy, [25, 50, 75])}")
    res[f"{lab}_phase_DOR>0.5"] = dict(n=len(xx), smax_doy=np.percentile(xx.smax_doy, [25, 50, 75]).tolist(),
                                       smin_doy=np.percentile(xx.smin_doy, [25, 50, 75]).tolist(),
                                       swing_over_dor=np.percentile(xx.swing_frac_annual / xx.nid_dor, [25, 50, 75]).round(3).tolist())
json.dump(res, open(f"{rc.HERE}/purpose.json", "w"), indent=1)
