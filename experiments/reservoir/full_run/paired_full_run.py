#!/usr/bin/env python
"""Paired per-gauge comparison of the two full-population arms of the learned dam release.

Arms (same config and seed, CPU, code 0ac6f2e on branch dam-release-head, train 1981-1995, test 1995-10-01..2010-09-30):
  off      .ddrs/runs/2026-09-27T07-29-47Z-train-and-test  (sr_n0_gamma recipe, no dams)
  learned  .ddrs/runs/2026-09-27T07-29-55Z-train-and-test  (same + learned release on 1,024 NID dam COMIDs >= 10 MCM)
Per gauge: NSE, KGE over the test window, as the run manifests score it; the summed-Q' baseline on the same gauges and
days. Groups from experiments/reservoir/nid/nid_dams_by_gauge.csv: any NID dam >= 10 MCM upstream vs none, NID degree
of regulation, dam on the gauge reach, and the 458 smoke-set dam gauges and their controls.
Writes experiments/reservoir/full_run/paired_full_run.{json,csv}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
OFF, LEARNED = "2026-09-27T07-29-47Z-train-and-test", "2026-09-27T07-29-55Z-train-and-test"
NID = HERE.parent / "nid" / "nid_dams_by_gauge.csv"
SMOKE = HERE.parent / "smoke" / "smoke_gauges.csv"


def load(run):
    z = zarr.open(str(RUNS / run / "eval/predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
    return pd.Index(ids), t, z["predictions"][:], z["observations"][:]


def metrics(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 365:
        return np.nan, np.nan
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)
    return nse, kge


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    b = np.median(rng.choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=float(np.median(d)), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                n_up=up, n_down=int(len(nz) - up), sign_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else None)


ids_o, t, P_off, O = load(OFF)
ids_l, t_l, P_l, O_l = load(LEARNED)
assert (ids_o == ids_l).all() and (t == t_l).all()
bm = json.load(open(RUNS / OFF / "baseline/manifest.json"))
B = np.fromfile(RUNS / OFF / "baseline/predictions.f32", dtype=np.float32).reshape(bm["n_gauges"], bm["n_days"])
bidx = {str(g): i for i, g in enumerate(bm["gage_ids"])}
off0 = int((t[0] - np.datetime64(str(bm["time_range_daily"][0])[:10])).astype(int))

rows = []
for i, s in enumerate(ids_o):
    n_o, k_o = metrics(P_off[i].astype(float), O[i].astype(float))
    n_l, k_l = metrics(P_l[i].astype(float), O[i].astype(float))
    n_b, k_b = (metrics(B[bidx[s], off0:off0 + len(t)].astype(float), O[i].astype(float)) if s in bidx else (np.nan, np.nan))
    rows.append(dict(STAID=s, nse_off=n_o, kge_off=k_o, nse_learned=n_l, kge_learned=k_l, nse_base=n_b, kge_base=k_b))
df = pd.DataFrame(rows).set_index("STAID")
nid = pd.read_csv(NID, dtype={"STAID": str}).set_index("STAID")
df = df.join(nid[["area_km2", "n_nid", "n_nid_ge10mcm", "nid_dor", "nid_on_gauge_reach", "nwm_cls"]])
sm = pd.read_csv(SMOKE, dtype={"STAID": str}).set_index("STAID")
df["smoke_role"] = df.index.map(sm.role)
df["dnse"], df["dkge"] = df.nse_learned - df.nse_off, df.kge_learned - df.kge_off
df.to_csv(HERE / "paired_full_run.csv")

dammed = df.n_nid_ge10mcm > 0
dor = pd.cut(df.nid_dor.where(dammed), [0, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
res = dict(
    arms=dict(off=OFF, learned=LEARNED), n=len(df),
    median=dict(nse=dict(off=float(df.nse_off.median()), learned=float(df.nse_learned.median()), base=float(df.nse_base.median())),
                kge=dict(off=float(df.kge_off.median()), learned=float(df.kge_learned.median()), base=float(df.kge_base.median()))),
    paired_nse=dict(all=paired(df.dnse), dammed_ge10mcm=paired(df.dnse[dammed]), undammed=paired(df.dnse[~dammed]),
                    dam_on_gauge_reach=paired(df.dnse[dammed & df.nid_on_gauge_reach.astype(bool)]),
                    smoke_dam=paired(df.dnse[df.smoke_role == "dam"]), smoke_control=paired(df.dnse[df.smoke_role == "control"])),
    paired_kge=dict(all=paired(df.dkge), dammed_ge10mcm=paired(df.dkge[dammed]), undammed=paired(df.dkge[~dammed])),
    by_dor={str(k): paired(v) for k, v in df.dnse[dammed].groupby(dor[dammed], observed=True)},
    median_by_group={g: dict(n=int(m.sum()), nse_off=float(df.nse_off[m].median()), nse_learned=float(df.nse_learned[m].median()),
                             nse_base=float(df.nse_base[m].median()))
                     for g, m in [("dammed_ge10mcm", dammed), ("undammed", ~dammed)]},
)
json.dump(res, open(HERE / "paired_full_run.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ["median", "paired_nse", "paired_kge", "by_dor", "median_by_group"]}, indent=1))
