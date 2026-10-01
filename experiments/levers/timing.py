"""Extra lever: a daily timing correction, fitted on TRAINING years, scored on TEST years.

p'(t) = (1 - |w|) p(t) + |w| p(t - sign(w))    w > 0 delays (blend with yesterday), w < 0 advances (blend with tomorrow)
Grid w in [-0.9, 0.9] step 0.1. Fitted: one global w (max training median NSE), per-HUC2 w, per-gauge w (gauged ceiling),
an attribute GBM on the per-gauge w (out of fold, 10-fold by gauge), and per-gauge (w, k) jointly with the LS scalar.
Usage: timing.py <seed> <replay run id>. Writes timing_s<seed>.txt / .json.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

import common as C

seed, replay = int(sys.argv[1]), sys.argv[2]
src = {42: C.S42, 43: C.S43}[seed]
out, res = [], {}
say = out.append
ids, tt, PT, OT = C.load_preds(src)
rid, tr, PR, OR = C.load_preds(replay)
PR, OR = C.align(rid, PR, ids), C.align(rid, OR, ids)
wy_t, wy_r = C.water_year(tt), C.water_year(tr)
import os
TWY = [int(x) for x in os.environ.get("LEVERS_TRAIN_WY", "1982,1995").split(",")]  # smoke-test override only
trn = (wy_r >= TWY[0]) & (wy_r <= TWY[1]) & (tr >= pd.Timestamp("1981-11-01"))
tst = (wy_t >= 1996) & (wy_t <= 2010)
W = np.round(np.arange(-0.9, 0.9001, 0.1), 2)


def shift(P, w):
    if w == 0:
        return P
    S = np.full_like(P, np.nan)
    if w > 0:
        S[:, 1:] = P[:, :-1]
    else:
        S[:, :-1] = P[:, 1:]
    S = np.where(np.isfinite(S), S, P)
    return (1 - abs(w)) * P + abs(w) * S


def windowed(P, O, mask):
    return P[:, mask], O[:, mask]


# shift on the full series, then cut the window (so the first test day uses the day before)
NTR = np.stack([C.metrics(*windowed(shift(PR, w), OR, trn)).nse.values for w in W])  # [w, g]
KTR = np.stack([C.ls_scale(*windowed(shift(PR, w), OR, trn)) for w in W])
TE = {w: shift(PT, w)[:, tst] for w in W}
Ote = OT[:, tst]
base = C.metrics(TE[0.0], Ote)
bn, bk = base.nse.values, base.kge.values
say(f"== seed {seed}: timing blend, fit WY1982-1995, score WY1996-2010; base test median NSE {np.nanmedian(bn):.4f}")


def score(wg, name, kg=None):
    wg = np.asarray(wg, dtype=float)
    P = np.empty_like(Ote)
    for w in W:
        rows = np.isclose(wg, w)
        if rows.any():
            P[rows] = TE[w][rows]
    if kg is not None:
        P = P * np.where(np.isfinite(kg), kg, 1.0)[:, None]
    M = C.metrics(P, Ote)
    s, sk = C.summarize(name, M.nse.values, bn), C.summarize(name, M.kge.values, bk, "KGE")
    say(C.fmt(s))
    say(C.fmt(sk))
    res[name] = dict(nse=s, kge=sk)


med = np.nanmedian(NTR, axis=1)
wg_glob = W[int(np.nanargmax(med))]
say(f"   training median NSE by w: " + " ".join(f"{w:+.1f}:{m:.4f}" for w, m in zip(W, med)))
score(np.full(len(ids), wg_glob), f"global w {wg_glob:+.1f}")
G = pd.read_csv(C.HERE / "gauges_table.csv", dtype={"STAID": str, "HUC02": str}).set_index("STAID").reindex(ids)
huc = G.HUC02.values
wh = np.zeros(len(ids))
for h in sorted(set(huc)):
    rows = huc == h
    wh[rows] = W[int(np.nanargmax(np.nanmedian(NTR[:, rows], axis=1)))]
score(wh, "per-HUC2 w")
ok = np.isfinite(NTR).all(0)
wg = np.where(ok, W[np.nanargmax(np.where(np.isfinite(NTR), NTR, -9), axis=0)], 0.0)
say(f"   per-gauge w: share delay {np.mean(wg > 0):.3f}, advance {np.mean(wg < 0):.3f}, none {np.mean(wg == 0):.3f}; "
    f"median |w| {np.median(np.abs(wg)):.2f}")
FEAT = ["up_aridity", "up_meanP", "up_meanslope", "out_log10_uparea", "up_snow_fraction", "up_seasonality_P",
        "longest_path_km", "n_reach", "LAT_GAGE", "LNG_GAGE", "up_NDVI", "up_Porosity"]
X = G[FEAT].values.astype(float)
X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
pred = np.zeros(len(ids))
for a, b in KFold(10, shuffle=True, random_state=0).split(X):
    fa = a[ok[a]]
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=40, random_state=0)
    m.fit(X[fa], wg[fa])
    pred[b] = m.predict(X[b])
wa = W[np.abs(pred[:, None] - W[None, :]).argmin(1)]
r2 = 1 - np.mean((pred[ok] - wg[ok]) ** 2) / np.var(wg[ok])
say(f"   attribute GBM on per-gauge w, out-of-fold R2 {r2:.3f}")
score(wa, "attribute GBM w (10-fold by gauge)")
score(wg, "per-gauge w fitted on training years (gauged ceiling)")
kj = KTR[np.nanargmax(np.where(np.isfinite(NTR), NTR, -9), axis=0), np.arange(len(ids))]
# joint (w, k) per gauge: choose w maximizing training NSE after its own LS scalar
NTRK = np.stack([C.metrics(*windowed(shift(PR, w) * KTR[i][:, None], OR, trn)).nse.values for i, w in enumerate(W)])
ij = np.nanargmax(np.where(np.isfinite(NTRK), NTRK, -9), axis=0)
wj = np.where(ok, W[ij], 0.0)
kj = np.where(ok, KTR[ij, np.arange(len(ids))], 1.0)
kj = np.where((kj > 0.2) & (kj < 5), kj, 1.0)
score(wj, "per-gauge (w, k) jointly, training years (gauged ceiling)", kg=kj)
open(C.HERE / f"timing_s{seed}.txt", "w").write("\n".join(out) + "\n")
json.dump(res, open(C.HERE / f"timing_s{seed}.json", "w"), indent=1, default=float)
print("\n".join(out))
