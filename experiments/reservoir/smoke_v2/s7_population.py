#!/usr/bin/env python
"""S7 follow-up: (1) S6base vs S7base isolates per_dam_l2 (both rho 180); (2) S7base's per-DOR-bin median gains at
the 458 smoke dam gauges transferred to the 917 dammed gauges of the 2,365-gauge population (as population_ceiling.py)."""
import numpy as np, pandas as pd, zarr

W = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/"
R = W + ".ddrs/runs/"
SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"
RUNS = dict(off="2026-09-27T04-29-33Z", S4v3="2026-09-27T23-36-04Z", S6base="2026-09-30T01-53-00Z",
            S7base="2026-09-30T02-46-12Z", S7pool="2026-09-30T02-46-23Z")


def load(run):
    z = zarr.open(R + run + "-train-and-test/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    return {s: i for i, s in enumerate(ids)}, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return 1 - ((p[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum()


idx, _, O = load(RUNS["off"])
N = {}
for k, r in RUNS.items():
    _, P, _ = load(r)
    N[k] = pd.Series({s: nse(P[i], O[i]) for s, i in idx.items()})
D = pd.DataFrame(N)
sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
onr = pd.read_csv(SM + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID").on_reach.astype(str) == "True"
dam = [s for s in sm.index[sm.role == "dam"] if s in D.index]
tgt = [s for s in dam if onr.get(s, False) and sm.nid_dor[s] > 0.5]
for a in ["S4v3", "S6base", "S7base", "S7pool"]:
    d = D[a] - D["off"]
    print(f"{a:7s} vs off: target {d[tgt].median():+.4f}  dam_all {d[dam].median():+.4f}  median NSE {D[a].median():.4f}")
print(f"L2 effect at rho 180 (S7base - S6base): target {(D.S7base - D.S6base)[tgt].median():+.4f}, dam_all {(D.S7base - D.S6base)[dam].median():+.4f}")
bins = [(-1, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 1e9)]
def binof(x):
    for j, (lo, hi) in enumerate(bins):
        if (lo < 0 or x > lo) and x <= hi:
            return j
    return len(bins) - 1
g = (D.S7base - D.off)[dam]
kb = pd.Series({s: binof(sm.nid_dor[s]) for s in dam})
gain = [float(g[kb == j].median()) for j in range(len(bins))]
print("S7base median gain by DOR bin:", [round(x, 4) for x in gain])
P = pd.read_csv("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/full_run/paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
base = P.nse_off
dammed = P.n_nid_ge10mcm > 0
k = P.nid_dor.fillna(0).map(binof)
new = base + np.where(dammed, [gain[j] for j in k], 0.0)
print(f"population median {base.median():.4f} -> {new.median():.4f} (change {new.median() - base.median():+.4f}); target +0.0300")
