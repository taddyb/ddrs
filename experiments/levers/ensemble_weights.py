"""Ensemble weights fitted on ROUTED training-year predictions (WY1986-1995, zero-step replays), scored on test years.

Members: UH (seed 42; optional seed 43), DIST, LSTM -- the set chosen by test-free forward selection (store_selection.py).
  equal     : 1/m each (no fitting)
  global    : one weight vector on the simplex (grid 0.05) maximizing the training-year median NSE
  per-HUC2  : the same within each HUC2
  per-gauge : non-negative least squares on the gauge's training years (weights free to sum != 1: gauged ceiling)
  + scalar  : the pre-registered global multiplier (median per-gauge LS k*) fitted on the ensemble's training years
Usage: ensemble_weights.py UH42=<replay> DIST=<replay> LSTM=<replay> [UH43=<replay>]
Writes ensemble_weights.txt / .json.
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import common as C

TEST = {"UH42": C.S42, "UH43": C.S43, "DIST": "2026-09-13T17-22-30Z-train-and-test",
        "LSTM": "2026-09-13T13-55-03Z-train-and-test"}
rep = dict(a.split("=", 1) for a in sys.argv[1:])
keys = [k for k in ["UH42", "UH43", "DIST", "LSTM"] if k in rep]
ids, tt, P42, O = C.load_preds(C.S42)
wy = C.water_year(tt)
tst = (wy >= 1996) & (wy <= 2010)
Ote = O[:, tst]
B = C.metrics(P42[:, tst], Ote)
bn, bk = B.nse.values, B.kge.values
Pte, Ptr = {}, {}
raw = {}
for k in keys:
    i, t, P, _ = C.load_preds(TEST[k])
    Pte[k] = C.align(i, P, ids)[:, tst]
    i, t, P, Ob = C.load_preds(rep[k])
    raw[k] = (t, C.align(i, P, ids), C.align(i, Ob, ids))
# common training days: WY1986-1995, from 1985-11-01 (skip the store replays' cold-start month), present in every replay
days = None
for k in keys:
    t = raw[k][0]
    s = set(t[(C.water_year(t) >= 1986) & (C.water_year(t) <= 1995) & (t >= pd.Timestamp("1985-11-01"))])
    days = s if days is None else days & s
ttr = pd.DatetimeIndex(sorted(days))
Otr = None
for k in keys:
    t, P, Ob = raw[k]
    ix = t.get_indexer(ttr)
    Ptr[k] = P[:, ix]
    if Otr is None:
        Otr = Ob[:, ix]
del raw
out, res = [], {}
say = out.append
say(f"members {keys}; training {ttr[0].date()}..{ttr[-1].date()} ({len(ttr)} d); test WY1996-2010")
for k in keys:
    say(f"   {k}: training median NSE {np.nanmedian(C.metrics(Ptr[k], Otr).nse):.4f}, test {np.nanmedian(C.metrics(Pte[k], Ote).nse):.4f}")


def ens(Pd, w):
    return sum(w[j] * Pd[k] for j, k in enumerate(keys))


def score(P, name, w=None):
    M = C.metrics(P, Ote)
    s, sk = C.summarize(name, M.nse.values, bn), C.summarize(name, M.kge.values, bk, "KGE")
    say(C.fmt(s))
    say(C.fmt(sk))
    res[name] = dict(nse=s, kge=sk, weights=None if w is None else [float(x) for x in np.atleast_1d(w)])
    return M


m = len(keys)
eq = np.full(m, 1.0 / m)
score(ens(Pte, eq), "equal weights")
grid = [np.array(c) / 20 for c in itertools.product(range(21), repeat=m) if sum(c) == 20]
tr_med = [(np.nanmedian(C.metrics(ens(Ptr, w), Otr).nse.values), tuple(w)) for w in grid]
v, wbest = max(tr_med)
wbest = np.array(wbest)
say(f"   global weights (max training median NSE {v:.4f}): " + ", ".join(f"{k} {x:.2f}" for k, x in zip(keys, wbest)))
score(ens(Pte, wbest), "global weights fitted on training years", wbest)
G = pd.read_csv(C.HERE / "gauges_table.csv", dtype={"STAID": str, "HUC02": str}).set_index("STAID").reindex(ids)
huc = G.HUC02.values
W = np.zeros((len(ids), m))
for h in sorted(set(huc)):
    r = huc == h
    best = max((np.nanmedian(C.metrics(sum(w[j] * Ptr[k][r] for j, k in enumerate(keys)), Otr[r]).nse.values), tuple(w))
               for w in grid)
    W[r] = best[1]
Ph = sum(W[:, [j]] * Pte[k] for j, k in enumerate(keys))
score(Ph, "per-HUC2 weights fitted on training years")
Wg = np.tile(eq, (len(ids), 1))
for g in range(len(ids)):
    X = np.stack([Ptr[k][g] for k in keys], 1)
    y = Otr[g]
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() > 365:
        Wg[g] = nnls(X[ok], y[ok])[0]
Pg = sum(Wg[:, [j]] * Pte[k] for j, k in enumerate(keys))
say(f"   per-gauge NNLS weights: median " + ", ".join(f"{k} {x:.2f}" for k, x in zip(keys, np.median(Wg, 0))) +
    f"; median sum {np.median(Wg.sum(1)):.3f}")
score(Pg, "per-gauge NNLS weights on training years (gauged ceiling)")
# pre-registered global multiplier on the equal-weight ensemble
Pe_tr = ens(Ptr, eq)
k = C.ls_scale(Pe_tr, Otr)
kg = float(np.nanmedian(k[np.isfinite(k) & (k > 0.2) & (k < 5)]))
say(f"   equal-weight ensemble, training median per-gauge LS k* = {kg:.4f}")
score(kg * ens(Pte, eq), f"equal weights x global scalar {kg:.4f}")
open(C.HERE / "ensemble_weights.txt", "w").write("\n".join(out) + "\n")
json.dump(res, open(C.HERE / "ensemble_weights.json", "w"), indent=1, default=float)
print("\n".join(out))
