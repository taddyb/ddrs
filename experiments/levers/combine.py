"""Combine the non-dam levers on per-gauge test predictions (WY1996-2010), against seed 42's 0.7391 / 0.7591.

Arms are equal-weight ensembles of runs; a runoff multiplier is fitted on the arm's own TRAINING-year predictions
(zero-step replays, WY1982-1995) and applied to its test predictions. Pre-registered primary multipliers (prereg.txt):
global = median per-gauge LS k*; regional = per-HUC2 median k*; ceiling = per-gauge k*.
Usage: combine.py key=testrun:trainreplay ...   e.g. s42=2026-09-27T07-29-47Z-train-and-test:2026-09-30T13-13-59Z-train-and-test
(trainreplay may be '-'). Writes combine.txt / combine.json.
"""
import json
import sys

import numpy as np
import pandas as pd

import common as C

runs = dict(a.split("=", 1) for a in sys.argv[1:])
runs = {k: tuple(v.split(":")) for k, v in runs.items()}
out, res = [], {}
say = out.append
ids, tt, P42, O = C.load_preds(C.S42)
wy = C.water_year(tt)
tst = (wy >= 1996) & (wy <= 2010)
Ote = O[:, tst]
B = C.metrics(P42[:, tst], Ote)
bn, bk = B.nse.values, B.kge.values
G = pd.read_csv(C.HERE / "gauges_table.csv", dtype={"STAID": str, "HUC02": str}).set_index("STAID").reindex(ids)
huc = G.HUC02.values
cache = {}


def test_pred(run):
    if run not in cache:
        i, t, P, _ = C.load_preds(run)
        assert (t == tt).all()
        cache[run] = C.align(i, P, ids)[:, tst]
    return cache[run]


def train_pred(run):
    k = ("tr", run)
    if k not in cache:
        i, t, P, Ob = C.load_preds(run)
        w = C.water_year(t)
        m = (w >= 1982) & (w <= 1995) & (t >= pd.Timestamp("1981-11-01"))
        cache[k] = (C.align(i, P, ids)[:, m], C.align(i, Ob, ids)[:, m])
    return cache[k]


def multipliers(Ptr, Otr):
    k = C.ls_scale(Ptr, Otr)
    ok = np.isfinite(k) & (k > 0.2) & (k < 5) & np.isfinite(C.metrics(Ptr, Otr).nse.values)
    glob = np.full(len(ids), np.nanmedian(k[ok]))
    reg = np.ones(len(ids))
    for h in set(huc):
        r = huc == h
        reg[r] = np.nanmedian(k[r & ok]) if (r & ok).any() else glob[0]
    gauge = np.where(ok, k, 1.0)
    return dict(none=np.ones(len(ids)), global_k=glob, huc2_k=reg, gauge_k=gauge)


ARMS = {
    "working recipe, seed 42": ["s42"],
    "working recipe, seed 43": ["s43"],
    "working recipe, two-seed ensemble": ["s42", "s43"],
    "gamma = 0, p and q learned (g0), seed 42": ["g0"],
    "g0 + working recipe s42 + s43 (3-member ensemble)": ["g0", "s42", "s43"],
    "n only (gamma = 0, channel fixed), seed 42": ["nonly"],
}
for name, keys in ARMS.items():
    if not all(k in runs for k in keys):
        continue
    Pte = np.mean([test_pred(runs[k][0]) for k in keys], axis=0)
    have_tr = all(runs[k][1] != "-" for k in keys)
    if have_tr:
        trs = [train_pred(runs[k][1]) for k in keys]
        Ptr = np.mean([a for a, _ in trs], axis=0)
        mults = multipliers(Ptr, trs[0][1])
    else:
        mults = dict(none=np.ones(len(ids)))
    say(f"\n== {name}  ({'+'.join(keys)})")
    for mk, m in mults.items():
        M = C.metrics(Pte * m[:, None], Ote)
        s = C.summarize(f"{name} | {mk}", M.nse.values, bn)
        sk = C.summarize(f"{name} | {mk}", M.kge.values, bk, "KGE")
        say(C.fmt(s))
        say(C.fmt(sk))
        res[f"{name} | {mk}"] = dict(nse=s, kge=sk, mult_median=float(np.median(m)))
open(C.HERE / "combine.txt", "w").write("\n".join(out) + "\n")
json.dump(res, open(C.HERE / "combine.json", "w"), indent=1, default=float)
print("\n".join(out))
