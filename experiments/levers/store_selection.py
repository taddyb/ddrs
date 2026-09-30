"""Test-free selection of Q' stores for an ensemble: greedy forward selection on the TRAINING years (WY1986-1995) using
the routing-free summed-Q' baselines, starting from UH; then the selected set scored on the test years, both as summed Q'
and as the routed working recipe (equal weights). 2,365 population gauges. Writes store_selection.txt."""
import json
import pathlib

import numpy as np
import pandas as pd

import common as C

BL = pathlib.Path("/home/tbindas/projects/ddrs/.ddrs/baselines")
TRAIN = {"UH": "abe830ef3781964b", "DIST": "196652afd33a6fda", "LSTM": "367784ef8974b46f", "LUMPED": "7b6fd0ba64be6ae2",
         "HYDRODL": "8e6b5af4534def3c"}
ROUTED = {"UH": C.S42, "DIST": "2026-09-13T17-22-30Z-train-and-test", "LSTM": "2026-09-13T13-55-03Z-train-and-test",
          "LUMPED": "2026-09-13T13-56-50Z-train-and-test", "HYDRODL": "2026-09-13T17-21-58Z-train-and-test"}
ids, tt, P42, O = C.load_preds(C.S42)


def load_bl(d):
    m = json.load(open(d / "manifest.json"))
    n, t = m["n_gauges"], m["n_days"]
    P = np.fromfile(d / "predictions.f32", dtype="<f4").reshape(n, t).astype(float)
    Ob = np.fromfile(d / "observations.f32", dtype="<f4").reshape(n, t).astype(float)
    g = [str(s) for s in m["gage_ids"]]
    return C.align(g, P, ids), C.align(g, Ob, ids), pd.DatetimeIndex(m["time_range_daily"])


out = []
say = out.append
TR = {k: load_bl(BL / v) for k, v in TRAIN.items()}
TE = {k: load_bl(C.RUNS / r / "baseline") for k, r in ROUTED.items()}
Otr, Ote = TR["UH"][1], TE["UH"][1]
for k in TR:
    assert (TR[k][2] == TR["UH"][2]).all() and (TE[k][2] == TE["UH"][2]).all()


def med(Pd, Ob, keys):
    return float(np.nanmedian(C.metrics(np.mean([Pd[k][0] for k in keys], axis=0), Ob).nse.values))


say(f"summed Q' (routing-free), population 2,365 gauges; training WY1986-1995 {TR['UH'][2][0].date()}..{TR['UH'][2][-1].date()}, "
    f"test {TE['UH'][2][0].date()}..{TE['UH'][2][-1].date()}")
say(f"{'set':28s} {'train NSE':>9s} {'test NSE':>9s}")
for keys in [["UH"], ["DIST"], ["LSTM"], ["LUMPED"], ["HYDRODL"], ["UH", "DIST"], ["UH", "LSTM"], ["UH", "LUMPED"],
             ["UH", "HYDRODL"], ["UH", "DIST", "LSTM"], list(TRAIN)]:
    say(f"{'+'.join(keys):28s} {med(TR, Otr, keys):9.4f} {med(TE, Ote, keys):9.4f}")
# greedy forward selection on training years
sel, best = ["UH"], med(TR, Otr, ["UH"])
while True:
    cands = [(med(TR, Otr, sel + [k]), k) for k in TRAIN if k not in sel]
    if not cands:
        break
    v, k = max(cands)
    if v <= best:
        break
    sel.append(k)
    best = v
    say(f"  forward selection adds {k}: training median {v:.4f}")
say(f"selected on training years: {'+'.join(sel)} (training summed-Q' median {best:.4f})")
# the selected set, routed, on test years
R = {}
for k in sel:
    i, t, P, _ = C.load_preds(ROUTED[k])
    R[k] = C.align(i, P, ids)
B = C.metrics(P42, O)
M = C.metrics(np.mean([R[k] for k in sel], axis=0), O)
say(C.fmt(C.summarize(f"routed, selected set {'+'.join(sel)}", M.nse.values, B.nse.values)))
say(C.fmt(C.summarize(f"routed, selected set {'+'.join(sel)}", M.kge.values, B.kge.values, "KGE")))
open(C.HERE / "store_selection.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
