#!/usr/bin/env python
"""Paired comparison of dam-release smoke arms against the smoke no-dam arm.

Usage: smoke_v2_paired.py --off <run dir> --arm NAME=<run dir> [--arm ...] [--out <json>]
Groups (smoke set, experiments/reservoir/smoke/): all 458 dam gauges, the 214 on-reach dams (on_reach in
expected_release_fit.csv), DOR bins, DOR > 0.5 pooled (all / on-reach), and the 458 matched controls. With
frozen routing the controls must be unchanged (max |dNSE| reported). DiD = dam gain minus its own control's gain.
"""
import argparse, json
import numpy as np, pandas as pd, zarr
from scipy.stats import binomtest

SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"
ap = argparse.ArgumentParser()
ap.add_argument("--off", required=True)
ap.add_argument("--arm", action="append", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()


def load(run):
    z = zarr.open(run.rstrip("/") + "/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
    return pd.Index(ids), t, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def met(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 365:
        return (np.nan,) * 5
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1]
    al, be = p.std() / o.std(), p.mean() / o.mean()
    return nse, 1 - np.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2), r, al, be


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {}
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    b = np.median(np.random.default_rng(42).choice(d, (2000, len(d))), axis=1)
    return dict(n=len(d), median=round(float(np.median(d)), 5),
                ci=[round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)],
                mean_clip=None, n_up=up, n_down=len(nz) - up,
                sign_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else None)


sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str, "control_for": str}).set_index("STAID")
fit = pd.read_csv(SM + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID")
sm["on_reach"] = fit.on_reach.astype(str).reindex(sm.index) == "True"
ctl_of = sm[sm.role == "control"].reset_index().set_index("control_for").STAID

ids, t, P0, O = load(a.off)
base = pd.DataFrame([met(P0[i], O[i]) for i in range(len(ids))], index=ids, columns=["nse", "kge", "r", "alpha", "beta"])
res = {"off": a.off, "arms": {}}
for spec in a.arm:
    name, run = spec.split("=", 1)
    ids2, t2, P, O2 = load(run)
    assert (ids2 == ids).all() and (t2 == t).all(), f"{name}: gauge/time axis differs from off arm"
    m = pd.DataFrame([met(P[i], O[i]) for i in range(len(ids))], index=ids, columns=["nse", "kge", "r", "alpha", "beta"])
    d = (m - base).join(sm[["role", "nid_dor", "on_reach"]])
    d["nse_off"], d["nse_arm"] = base.nse, m.nse
    dam = d[d.role == "dam"].copy()
    ctl = d[d.role == "control"]
    dam["dnse_ctl"] = [d.nse.get(ctl_of.get(s), np.nan) for s in dam.index]
    dam["did"] = dam.nse - dam.dnse_ctl
    out = {"controls_max_abs_dnse": float(ctl.nse.abs().max()), "controls": paired(ctl.nse)}
    groups = {"dam_all": dam, "on_reach": dam[dam.on_reach], "dor_gt_0.5": dam[dam.nid_dor > 0.5],
              "on_reach_dor_gt_0.5": dam[dam.on_reach & (dam.nid_dor > 0.5)]}
    for lo, hi, lab in [(0, 0.1, "<=0.1"), (0.1, 0.5, "0.1-0.5"), (0.5, 1, "0.5-1"), (1, 2, "1-2"), (2, 1e9, ">2")]:
        groups["dor " + lab] = dam[(dam.nid_dor > lo) & (dam.nid_dor <= hi)] if lo > 0 else dam[dam.nid_dor <= hi]
    for g, x in groups.items():
        e = {"dnse": paired(x.nse), "dkge": paired(x.kge), "did_nse": paired(x.did)}
        e["dnse"]["mean_clip"] = round(float((x.nse_arm.clip(-1) - x.nse_off.clip(-1)).mean()), 5) if len(x) else None
        e["dr"], e["dalpha"], e["dbeta"] = (round(float(x[c].median()), 5) for c in ["r", "alpha", "beta"])
        e["median_nse_off"], e["median_nse_arm"] = round(float(x.nse_off.median()), 4), round(float(x.nse_arm.median()), 4)
        out[g] = e
    res["arms"][name] = out
    print(f"\n=== {name}  controls max|dNSE| {out['controls_max_abs_dnse']:.2e}")
    for g in ["dam_all", "on_reach", "dor_gt_0.5", "on_reach_dor_gt_0.5", "dor <=0.1", "dor 0.1-0.5", "dor 0.5-1", "dor 1-2", "dor >2"]:
        e = out[g]
        print(f"  {g:22s} n={e['dnse'].get('n', 0):3d}  dNSE {e['dnse'].get('median', np.nan):+.4f} {e['dnse'].get('ci')} "
              f"up/down {e['dnse'].get('n_up')}/{e['dnse'].get('n_down')}  mean(clip) {e['dnse'].get('mean_clip')}  "
              f"DiD {e['did_nse'].get('median', np.nan):+.4f}  dKGE {e['dkge'].get('median', np.nan):+.4f}  "
              f"dr {e['dr']:+.4f} dalpha {e['dalpha']:+.4f}  NSE {e['median_nse_off']:.3f}->{e['median_nse_arm']:.3f}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
