"""Why the multi-product blend scores higher: exact per-gauge ambiguity decomposition, test vs training years.

For weights w_i summing to 1 and blend pbar = sum w_i p_i, at every time step
    sum_i w_i (p_i - o)^2 = (pbar - o)^2 + sum_i w_i (p_i - pbar)^2,
so, dividing the time sums by sum (o - obar)^2,
    NSE_blend = sum_i w_i NSE_i + SPREAD,   SPREAD = sum_t sum_i w_i (p_i - pbar)^2 / sum_t (o - obar)^2 >= 0.
SPREAD is how much the members disagree, in NSE units; it is the whole gain over the weighted-mean member.
This is the ambiguity decomposition of Krogh & Vedelsby (1995, NIPS 7, "Neural network ensembles, cross validation,
and active learning") divided by the observed variance: a standard multi-model ensemble gain, not a routing result.
SPREAD is split by time scale on each member's deviation d_i = p_i - pbar: water-year mean (year-to-year volume),
30-day centred moving average of the rest (seasonal / monthly), and the remainder (event scale, < ~30 d).
Writes blend_mechanism.txt.
"""
import numpy as np
import pandas as pd

import common as C

TEST = {"UH": C.S42, "DIST": "2026-09-13T17-22-30Z-train-and-test", "LSTM": "2026-09-13T13-55-03Z-train-and-test"}
TRAIN = {"UH": "2026-09-30T13-13-59Z-train-and-test", "DIST": "2026-09-30T14-56-59Z-train-and-test",
         "LSTM": "2026-09-30T14-30-08Z-train-and-test"}
WEIGHTS = {"equal": {"UH": 1 / 3, "DIST": 1 / 3, "LSTM": 1 / 3}, "training-year": {"UH": 0.5, "DIST": 0.25, "LSTM": 0.25}}


def load(runs, t0, t1, ref_ids=None):
    """Members aligned to ref_ids on the days every run has inside [t0, t1]; observations from the UH run."""
    raw = {k: C.load_preds(r) for k, r in runs.items()}
    ref = ref_ids or raw["UH"][0]
    days = pd.DatetimeIndex(sorted(set.intersection(*[set(v[1]) for v in raw.values()])))
    days = days[(days >= t0) & (days <= t1)]
    P = {}
    for k, (i, t, Pr, Or) in raw.items():
        pos = t.get_indexer(days)
        P[k] = C.align(i, Pr, ref)[:, pos]
        if k == "UH":
            O = C.align(i, Or, ref)[:, pos]
    return ref, days, P, O


def bands(d, tt):
    """Split d[g, t] into (water-year mean, 30-day MA of the rest, remainder)."""
    wy = C.water_year(tt)
    low = np.empty_like(d)
    for y in np.unique(wy):
        s = wy == y
        low[:, s] = d[:, s].mean(1, keepdims=True)
    rest = d - low
    mid = pd.DataFrame(rest.T).rolling(30, center=True, min_periods=1).mean().to_numpy().T
    return low, mid, rest - mid


def decompose(P, O, tt, w):
    m = np.isfinite(O) & np.all([np.isfinite(P[k]) for k in P], axis=0)
    nv = m.sum(1)
    om = np.where(m, O, 0.0)
    mo = om.sum(1) / np.maximum(nv, 1)
    sst = (np.where(m, O - mo[:, None], 0.0) ** 2).sum(1)
    pbar = sum(w[k] * P[k] for k in P)
    nse = {k: 1 - (np.where(m, P[k] - O, 0.0) ** 2).sum(1) / sst for k in P}
    nse_blend = 1 - (np.where(m, pbar - O, 0.0) ** 2).sum(1) / sst
    spread = sum(w[k] * (np.where(m, P[k] - pbar, 0.0) ** 2).sum(1) for k in P) / sst
    parts = np.zeros((3, len(sst)))
    for k in P:
        for j, b in enumerate(bands(P[k] - pbar, tt)):
            parts[j] += w[k] * (np.where(m, b, 0.0) ** 2).sum(1) / sst
    wmean = sum(w[k] * nse[k] for k in P)
    ok = (nv >= 365) & np.isfinite(nse_blend)
    df = pd.DataFrame(dict(blend=nse_blend, wmean=wmean, spread=spread, low=parts[0], mid=parts[1], high=parts[2],
                           **{f"nse_{k}": nse[k] for k in P}))[ok]
    assert np.allclose(df.blend, df.wmean + df.spread, atol=1e-8), "identity failed"
    return df


out = []
ids, tt_te, P_te, O_te = load(TEST, pd.Timestamp("1995-10-01"), pd.Timestamp("2010-09-30"))
_, tt_tr, P_tr, O_tr = load(TRAIN, pd.Timestamp("1985-11-01"), pd.Timestamp("1995-09-30"), ids)
for period, (P, O, tt) in {"test WY1996-2010": (P_te, O_te, tt_te), "training WY1986-1995": (P_tr, O_tr, tt_tr)}.items():
    for wname, w in WEIGHTS.items():
        D = decompose(P, O, tt, w)
        med = D.median()
        tot = D[["low", "mid", "high"]].sum(1)
        share = (D[["low", "mid", "high"]].div(tot, axis=0)).median()
        out.append(f"{period}, {wname} weights, {len(D)} gauges (medians over gauges)")
        out.append(f"  member NSE: UH {med.nse_UH:.4f}  DIST {med.nse_DIST:.4f}  LSTM {med.nse_LSTM:.4f}")
        out.append(f"  weighted mean of member NSEs {med.wmean:.4f} + spread {med.spread:.4f} = blend; blend median "
                   f"{med.blend:.4f}; blend - UH: median {D.blend.median() - D.nse_UH.median():+.4f}, "
                   f"paired {np.median(D.blend - D.nse_UH):+.4f}")
        out.append(f"  paired per gauge: blend - UH = (wmean - UH) + spread: median {np.median(D.wmean - D.nse_UH):+.4f} "
                   f"+ {med.spread:+.4f}")
        out.append(f"  spread by time scale, median share: year-to-year volume {share.low:.2f}, seasonal/monthly "
                   f"{share.mid:.2f}, event (< 30 d) {share.high:.2f}; band sum / spread median "
                   f"{np.median(tot / D.spread):.3f}")
        out.append("")
txt = "\n".join(out)
(C.HERE / "blend_mechanism.txt").write_text(txt + "\n")
print(txt)
