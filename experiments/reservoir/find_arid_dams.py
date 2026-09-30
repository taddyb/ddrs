#!/usr/bin/env python
"""List dam-controlled gauges in arid regions of the training population, with observed-inflow
candidates (upstream gauges sharing the river name), observation coverage, baseline skill and
volume ratio over 1995-2010."""
import json, re, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, geopandas as gpd

R = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/baseline")
m = json.load(open(R / "manifest.json")); G, T = m["n_gauges"], m["n_days"]
P = np.fromfile(R / "predictions.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
O = np.fromfile(R / "observations.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
gid = [str(g) for g in m["gage_ids"]]
ok = np.isfinite(O)
obs_mean = np.nanmean(O, axis=1); cov = ok.mean(axis=1)
vol = np.array([P[i][ok[i]].sum() / max(O[i][ok[i]].sum(), 1e-9) for i in range(G)])
nse = np.array([1 - ((P[i][ok[i]] - O[i][ok[i]]) ** 2).sum() / max(((O[i][ok[i]] - O[i][ok[i]].mean()) ** 2).sum(), 1e-12) for i in range(G)])
base = pd.DataFrame(dict(STAID=gid, obs_mean=obs_mean, coverage=cov, vol_ratio=vol, base_nse=nse))

g = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str})
a = xr.open_dataset("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")[["aridity", "meanP"]].to_dataframe()
g = g.merge(a, left_on="COMID", right_index=True, how="left")
s = gpd.read_file("/mnt/ssd1/data/gage_shp_files/gagesII_9322_sept30_2011.shp").drop(columns="geometry")
s["huc2"] = s.HUC02.str[:2]
g = g.merge(s[["STAID", "huc2", "CLASS", "STATE"]], on="STAID", how="left").merge(base, on="STAID", how="left")
g["in_baseline"] = g.STAID.isin(gid)
g["dam"] = g.STANAME.str.upper().str.contains(r"\bDAM\b|RESERVOIR|\bRES\b|\bBLW\b|BELOW|\bLAKE\b|\bLK\b", regex=True)

arid = g[(g.dam) & (g.aridity >= 2.5)].copy()
print(f"dam-named gauges with aridity >= 2.5: {len(arid)} (of {int(g.dam.sum())} dam-named overall)")

def river_token(name):
    n = re.sub(r"\b(RIVER|RV|R|CREEK|CK|C|CR|WASH|FORK|FK|NEAR|NR|AT|BELOW|BLW|ABOVE|ABV|DAM|RESERVOIR|RES|LAKE|LK|THE|OF)\b", " ", name.upper())
    n = re.sub(r"[^A-Z ]", " ", n)
    return [w for w in n.split() if len(w) > 2][:2]

rows = []
for r in arid.itertuples():
    tok = river_token(r.STANAME)
    up = g[(g.STAID != r.STAID) & (g.DRAIN_SQKM < r.DRAIN_SQKM) & (g.DRAIN_SQKM > 0.1 * r.DRAIN_SQKM)]
    if tok:
        pat = r"\b" + tok[0] + r"\b"
        up = up[up.STANAME.str.upper().str.contains(pat, regex=True)]
        # crude proximity: within 1.5 degrees
        up = up[(np.abs(up.LAT_GAGE - r.LAT_GAGE) < 1.5) & (np.abs(up.LNG_GAGE - r.LNG_GAGE) < 1.5)]
    ups = "; ".join(f"{u.STAID} {u.STANAME.strip()[:34]} ({u.DRAIN_SQKM:.0f} km2, cov {u.coverage:.2f})" for u in up.sort_values("DRAIN_SQKM", ascending=False).head(3).itertuples())
    rows.append(dict(STAID=r.STAID, name=r.STANAME.strip()[:44], huc2=r.huc2, aridity=r.aridity, DA_km2=r.DRAIN_SQKM, obs_mean=r.obs_mean,
                     cov=r.coverage, vol_ratio=r.vol_ratio, base_nse=r.base_nse, in_base=r.in_baseline, upstream_gauges=ups))
df = pd.DataFrame(rows).sort_values(["huc2", "aridity"], ascending=[True, False])
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
df.to_csv("/home/tbindas/projects/ddrs/output/raystown_sandbox/arid_dam_candidates.csv", index=False)
