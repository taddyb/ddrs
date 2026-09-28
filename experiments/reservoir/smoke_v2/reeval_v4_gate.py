#!/usr/bin/env python
"""Gate the v3 arms and the replays on the corrected created-water account (v4 re-evaluations).

For each re-evaluated run (zero-step resume, test phase only, binary ddrs-v4-56eb713): median test NSE/KGE, per-dam
created-share distribution (release_clamp.csv, created = storage forgiven + below-floor outflow, per dam), and the
paired dNSE against the smoke no-dam arm at on-reach smoke dams with DOR > 0.5, for all gauges and for the "clean"
gauges whose dam creates < 0.5 % of its inflow. Reads /home/tbindas/.claude/jobs/dacd6d8c/tmp/reeval_v4.status.
"""
import json
import numpy as np, pandas as pd, zarr

W = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/"
R = W + ".ddrs/runs/"
SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"
OFF = "2026-09-27T04-29-33Z-train-and-test"
status = [l.split() for l in open("/home/tbindas/.claude/jobs/dacd6d8c/tmp/reeval_v4.status") if l.strip()]


def load(run):
    z = zarr.open(R + run + "/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    return {s: i for i, s in enumerate(ids)}, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return 1 - ((p[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum()


def med_ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    b = np.median(np.random.default_rng(42).choice(d, (2000, len(d))), axis=1)
    return f"{np.median(d):+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}] n={len(d)}"


idx, P0, O = load(OFF)
sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
onr = pd.read_csv(SM + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID").on_reach.astype(str) == "True"
tgt = [s for s in sm.index[(sm.role == "dam") & (sm.nid_dor > 0.5)] if onr.get(s, False) and s in idx]
n0 = {s: nse(P0[idx[s]], O[idx[s]]) for s in tgt}
out = {}
for name, run, *_ in status:
    m = json.load(open(R + run + "/manifest.json"))["metrics"]
    c = pd.read_csv(R + run + "/release_clamp.csv").set_index("COMID")
    share = c.created_share
    _, P, _ = load(run)
    rows = []
    for s in tgt:
        comid = int(sm.dam_COMID[s])
        rows.append(dict(STAID=s, d=nse(P[idx[s]], O[idx[s]]) - n0[s],
                         share=float(share.get(comid, np.nan)) if comid in share.index else np.nan))
    D = pd.DataFrame(rows)
    clean = D[(D.share < 0.005) | D.share.isna()]
    out[name] = dict(run=run, nse=m["median_nse_finite"], kge=m["median_kge_finite"],
                     dams=len(c), ge_0p5=int((share >= 0.005).sum()), ge_5=int((share >= 0.05).sum()),
                     p90=float(share.quantile(0.9)), max=float(share.max()),
                     all=med_ci(D.d), clean=med_ci(clean.d))
    o = out[name]
    print(f"{name:12s} NSE {o['nse']:.4f} KGE {o['kge']:.4f} | dams {o['dams']}: created>=0.5% {o['ge_0p5']}, >=5% {o['ge_5']}, "
          f"p90 {o['p90']:.4%}, max {o['max']:.1%}\n              on-reach DOR>0.5 all: {o['all']}   clean(<0.5%): {o['clean']}")
json.dump(out, open(W + "experiments/reservoir/smoke_v2/reeval_v4_gate.json", "w"), indent=1)
