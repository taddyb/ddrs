"""Median over gauges of predicted/observed volume by water year, WY1982-2010 (replay for WY1982-1995, source run for
WY1996-2010), plus observed and predicted median flow index. Usage: volume_by_year.py <source run> <replay run>.
Writes volume_by_year_<source[:20]>.txt."""
import sys

import numpy as np

import common as C

src, rep = sys.argv[1], sys.argv[2]
ids, tt, PT, OT = C.load_preds(src)
rid, tr, PR, OR = C.load_preds(rep)
PR, OR = C.align(rid, PR, ids), C.align(rid, OR, ids)
out = ["WY    median pred/obs vol  share>1  median obs mean (m3/s, per-gauge relative to its 1982-2010 mean)"]
wyr, wyt = C.water_year(tr), C.water_year(tt)
allobs = []
per = {}
for y in range(1982, 2011):
    P, O = (PR[:, wyr == y], OR[:, wyr == y]) if y <= 1995 else (PT[:, wyt == y], OT[:, wyt == y])
    m = np.isfinite(P) & np.isfinite(O)
    ok = m.sum(1) >= 300
    r = np.where(ok, np.where(m, P, 0).sum(1) / np.where(m, O, 0).sum(1), np.nan)
    om = np.where(ok, np.where(m, O, 0).sum(1) / np.maximum(m.sum(1), 1), np.nan)
    per[y] = (r, om)
clim = np.nanmean(np.stack([v[1] for v in per.values()]), axis=0)
for y, (r, om) in per.items():
    out.append(f"{y}  {np.nanmedian(r):.4f}               {np.nanmean(r > 1):.3f}    {np.nanmedian(om / clim):.3f}")
open(C.HERE / f"volume_by_year_{src[:20]}.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
