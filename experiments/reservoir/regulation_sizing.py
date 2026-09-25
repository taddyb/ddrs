#!/usr/bin/env python
"""How much of the eval population sits below reservoir storage, and what does it cost the trained model?

Degree of regulation (Lehner et al. 2011): upstream reservoir storage / mean annual flow volume at the
gauge. Reservoir -> MERIT COMID mapping is DDR's reverted #139 table (`merit_reservoir_params.csv`,
2,178 COMIDs), joined to `hydrolakes_rfc_da.csv` on (area, weir elevation, weir length) to recover
Hylak_id, then to HydroLAKES v1.0 for Vol_total (total lake volume, MCM). It is the NWM / RFC-DA
reservoir set: it misses dams outside it (Alamo, for one), so every regulated count here is a
lower bound. Upstream set = the gauge's `order` array in the gauges adjacency store.

Skill: trained run 2026-09-17T16-38-16Z-train-and-test, eval predictions WY1997-2010, and the
summed-Q' baseline over the same days. "vol-fixed" NSE rescales the prediction to the observed mean,
i.e. the NSE a perfect volume correction would reach with the timing unchanged.
Outputs: output/dam_sandbox/regulation_by_gauge.csv, experiments/reservoir/results/regulation_sizing.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import zarr

RUN = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test")
D = Path("/home/tbindas/projects/ddr/data")
CSV = Path("/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv")
OUT = Path(__file__).resolve().parent / "results" / "regulation_sizing.json"

mp = pd.read_csv(D / "merit_reservoir_params.csv")
rfc = pd.read_csv(D / "hydrolakes_rfc_da.csv")
j = mp.merge(rfc, left_on=["lake_area_m2", "weir_elevation", "weir_length"], right_on=["LkArea", "WeirE", "WeirL"], how="left")
assert len(j) == len(mp) and j.Hylak_id.notna().all(), "reservoir table join is not one-to-one"
hl = pyogrio.read_dataframe(D / "hydrolakes/HydroLAKES_polys_v10.shp",
                            columns=["Hylak_id", "Lake_type", "Grand_id", "Vol_total"], read_geometry=False)
res = j[["COMID", "Hylak_id"]].astype({"Hylak_id": int}).merge(hl, on="Hylak_id", how="left")
vol = dict(zip(res.COMID, res.Vol_total))
print(f"{len(res)} reservoir COMIDs; Lake_type {res.Lake_type.value_counts().to_dict()}; Grand_id > 0: {(res.Grand_id > 0).sum()}")

z = zarr.open(str(RUN / "eval/predictions.zarr"), mode="r")
ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
wy = np.array([d.year + (d.month >= 10) for d in t.astype(object)])
win = (wy >= 1997) & (wy <= 2010)
Pt, Ob, tw = z["predictions"][:][:, win], z["observations"][:][:, win], t[win]

bm = json.load(open(RUN / "baseline/manifest.json"))
bP = np.fromfile(RUN / "baseline/predictions.f32", dtype=np.float32).reshape(bm["n_gauges"], bm["n_days"])
bidx = {str(g): i for i, g in enumerate(bm["gage_ids"])}
off = int((tw[0] - np.datetime64(str(bm["time_range_daily"][0])[:10])).astype(int))


def metrics(p, o):
    m = np.isfinite(o) & np.isfinite(p)
    if m.sum() < 365:
        return dict(nse=np.nan, kge=np.nan, r=np.nan, alpha=np.nan, beta=np.nan, nse_volfix=np.nan)
    p, o = p[m], o[m]
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    a, b = p.std() / o.std(), p.mean() / o.mean()
    pv = p / b if b > 0 else p
    sst = ((o - o.mean()) ** 2).sum()
    return dict(nse=1 - ((p - o) ** 2).sum() / sst, kge=1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2),
                r=r, alpha=a, beta=b, nse_volfix=1 - ((pv - o) ** 2).sum() / sst)


gages = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str})
area = dict(zip(gages.STAID, gages.DRAIN_SQKM))
adj = zarr.open(str(D / "merit_gages_conus_adjacency.zarr"), mode="r")
rows = []
for i, g in enumerate(ids):
    up = adj[g]["order"][:].tolist() if g in adj else []
    rc = [c for c in up if c in vol]
    v = float(np.nansum([vol[c] for c in rc])) if rc else 0.0
    qmean = np.nanmean(Ob[i])
    mt = metrics(Pt[i], Ob[i])
    mb = metrics(bP[bidx[g], off:off + len(tw)].astype(np.float64), Ob[i]) if g in bidx else {}
    rows.append(dict(STAID=g, area_km2=area.get(g, np.nan), n_res=len(rc), vol_mcm=v, qmean=qmean,
                     dor=v * 1e6 / (qmean * 365.25 * 86400) if qmean > 0 else np.nan,
                     **{f"{k}_trained": x for k, x in mt.items()}, **{f"{k}_base": x for k, x in mb.items()}))
df = pd.DataFrame(rows)
df["cls"] = np.select([df.n_res == 0, df.dor <= 0.1, df.dor <= 0.5], ["none", "DOR<=0.1", "0.1<DOR<=0.5"], "DOR>0.5")
df["abin"] = pd.cut(np.log10(df.area_km2), [0, 2.5, 3, 3.5, 4, 7], labels=["<316", "316-1k", "1k-3.2k", "3.2k-10k", ">10k"])
CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV, index=False)

summary = {"n_gauges": len(df), "n_any_reservoir": int((df.n_res > 0).sum()), "classes": {}, "area_matched_nse": {}}
print(f"\n{'class':14s} {'n':>5s}  NSE tr/base   KGE tr/base   vol-fixed NSE   r      |1-beta|  alpha  share NSE<0")
for c in ["none", "DOR<=0.1", "0.1<DOR<=0.5", "DOR>0.5"]:
    s = df[df.cls == c]
    row = dict(n=len(s), nse=s.nse_trained.median(), nse_base=s.nse_base.median(), kge=s.kge_trained.median(),
               kge_base=s.kge_base.median(), nse_volfix=s.nse_volfix_trained.median(), r=s.r_trained.median(),
               abs_1_minus_beta=(1 - s.beta_trained).abs().median(), alpha=s.alpha_trained.median(),
               share_nse_neg=float((s.nse_trained < 0).mean()))
    summary["classes"][c] = row
    print(f"{c:14s} {row['n']:5d}  {row['nse']:.3f}/{row['nse_base']:.3f}  {row['kge']:.3f}/{row['kge_base']:.3f}  "
          f"{row['nse_volfix']:.3f}           {row['r']:.3f}  {row['abs_1_minus_beta']:.3f}     {row['alpha']:.3f}  {row['share_nse_neg']:.2f}")
print("\nmedian trained NSE by drainage area (count):")
for a, s in df.groupby("abin", observed=False):
    cells = {c: (s[s.cls == c].nse_trained.median(), int((s.cls == c).sum())) for c in ["none", "DOR<=0.1", "0.1<DOR<=0.5", "DOR>0.5"]}
    summary["area_matched_nse"][str(a)] = cells
    print(f"  {str(a):9s} " + "  ".join(f"{c} {m:.3f} ({n})" for c, (m, n) in cells.items()))
hi = df.dor > 0.5
summary["median_nse_all"] = df.nse_trained.median()
summary["median_nse_without_dor_gt_0.5"] = df[~hi].nse_trained.median()
print(f"\nDOR > 0.5: {hi.sum()} gauges ({100 * hi.mean():.1f} %); median NSE all {summary['median_nse_all']:.4f}, "
      f"without them {summary['median_nse_without_dor_gt_0.5']:.4f}")
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=1, default=float)
print("->", OUT, CSV)
