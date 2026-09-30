"""Lever 1b: STATE-dependent runoff corrections, fitted on TRAINING years (WY1982-1995), scored on TEST years.

The constant multiplier fails because the Q' bias is dry-year over-prediction plus a post-2000 drift (volume_by_year).
These corrections use only the model's own predictions at test time (no observations), with parameters fitted on the
training years:
  power  : p' = a * pbar * (p / pbar)^b          pbar = the gauge's training-year mean prediction (global a, b; grid)
  state  : p' = p * (A(t) / pbar)^c               A(t) = trailing 365-day mean prediction (global c; grid)
  qmap   : per-gauge empirical quantile mapping, training-year predicted -> observed quantiles (gauged ceiling)
  qmap_g : global quantile mapping on p / pbar pooled over gauges
Usage: volume_state.py <seed> <replay run>. Writes volume_state_s<seed>.txt / .json.
"""
import json
import sys

import numpy as np
import pandas as pd

import common as C

seed, replay = int(sys.argv[1]), sys.argv[2]
src = {42: C.S42, 43: C.S43}[seed]
out, res = [], {}
say = out.append
ids, tt, PT, OT = C.load_preds(src)
rid, tr, PR, OR = C.load_preds(replay)
PR, OR = C.align(rid, PR, ids), C.align(rid, OR, ids)
wyr, wyt = C.water_year(tr), C.water_year(tt)
trn = (wyr >= 1982) & (wyr <= 1995) & (tr >= pd.Timestamp("1981-11-01"))
tst = (wyt >= 1996) & (wyt <= 2010)
Ptr, Otr = PR[:, trn], OR[:, trn]
Pte, Ote = PT[:, tst], OT[:, tst]
pbar = np.nanmean(Ptr, axis=1)
B = C.metrics(Pte, Ote)
bn, bk = B.nse.values, B.kge.values
say(f"== seed {seed}: state-dependent corrections, fit WY1982-1995, score WY1996-2010 (base {np.nanmedian(bn):.4f} / "
    f"{np.nanmedian(bk):.4f})")


def score(P, name):
    M = C.metrics(P, Ote)
    s, sk = C.summarize(name, M.nse.values, bn), C.summarize(name, M.kge.values, bk, "KGE")
    say(C.fmt(s))
    say(C.fmt(sk))
    res[name] = dict(nse=s, kge=sk)


# power law (global a, b), fitted to maximize the training median NSE
x_tr = Ptr / pbar[:, None]
x_te = Pte / pbar[:, None]
best = (-np.inf, 1.0, 1.0)
for b in np.round(np.arange(0.85, 1.1501, 0.025), 3):
    xb = np.power(np.maximum(x_tr, 0), b)
    for a in np.round(np.arange(0.90, 1.1001, 0.01), 3):
        v = np.nanmedian(C.metrics(a * pbar[:, None] * xb, Otr).nse.values)
        if v > best[0]:
            best = (v, a, b)
say(f"   power law fit: a {best[1]}, b {best[2]} (training median {best[0]:.4f} vs "
    f"{np.nanmedian(C.metrics(Ptr, Otr).nse.values):.4f})")
score(best[1] * pbar[:, None] * np.power(np.maximum(x_te, 0), best[2]), f"power law a={best[1]} b={best[2]}")


# trailing-year state multiplier, computed on the full series (test days see the preceding year, which for WY1996 lies
# in the replay's WY1995: use the replay there so the state is continuous)
def trailing(P, n=365):
    c = np.nancumsum(np.where(np.isfinite(P), P, 0), axis=1)
    k = np.cumsum(np.isfinite(P), axis=1)
    s = c.copy()
    s[:, n:] = c[:, n:] - c[:, :-n]
    kk = k.copy()
    kk[:, n:] = k[:, n:] - k[:, :-n]
    return s / np.maximum(kk, 1)


A_r = trailing(PR)
# test state: concatenate the replay's last 365 days before 1995-10-01 with the source run's test series
pre = (tr < pd.Timestamp("1995-10-01")) & (tr >= pd.Timestamp("1994-10-01"))
cat = np.concatenate([PR[:, pre], PT], axis=1)
A_t = trailing(cat)[:, pre.sum():][:, tst]
A_tr = A_r[:, trn]
best = (-np.inf, 0.0)
for c in np.round(np.arange(-0.6, 0.6001, 0.05), 3):
    m = np.power(np.clip(A_tr / pbar[:, None], 0.05, 20), c)
    v = np.nanmedian(C.metrics(Ptr * m, Otr).nse.values)
    if v > best[0]:
        best = (v, c)
say(f"   trailing-year state exponent c = {best[1]} (training median {best[0]:.4f})")
score(Pte * np.power(np.clip(A_t / pbar[:, None], 0.05, 20), best[1]), f"trailing-year state multiplier c={best[1]}")
for c in [0.2, 0.4]:
    score(Pte * np.power(np.clip(A_t / pbar[:, None], 0.05, 20), c), f"  sensitivity: state multiplier c={c} (not fitted)")

# quantile mapping
qs = np.linspace(0.005, 0.995, 199)


def qmap(p_fit, o_fit, p_apply):
    ok = np.isfinite(p_fit) & np.isfinite(o_fit)
    if ok.sum() < 365:
        return p_apply
    qp = np.quantile(p_fit[ok], qs)
    qo = np.quantile(o_fit[ok], qs)
    qp = np.maximum.accumulate(qp + np.arange(len(qs)) * 1e-12)
    y = np.interp(p_apply, qp, qo)
    # beyond the fitted range keep the edge ratio
    hi = p_apply > qp[-1]
    y[hi] = p_apply[hi] * qo[-1] / max(qp[-1], 1e-9)
    lo = p_apply < qp[0]
    y[lo] = p_apply[lo] * qo[0] / max(qp[0], 1e-9)
    return y


Pq = np.stack([qmap(Ptr[g], Otr[g], Pte[g]) for g in range(len(ids))])
score(Pq, "per-gauge quantile mapping (gauged ceiling)")
# global quantile mapping on normalized flow
xo_tr = Otr / pbar[:, None]
ok = np.isfinite(x_tr) & np.isfinite(xo_tr)
qp = np.quantile(x_tr[ok], qs)
qo = np.quantile(xo_tr[ok], qs)
say(f"   global normalized quantiles (pred -> obs): q05 {qp[9]:.3f}->{qo[9]:.3f}, q50 {qp[99]:.3f}->{qo[99]:.3f}, "
    f"q95 {qp[189]:.3f}->{qo[189]:.3f}")
Pg = pbar[:, None] * np.interp(x_te, qp, qo, right=np.nan)
Pg = np.where(np.isfinite(Pg), Pg, Pte * qo[-1] / qp[-1])
score(Pg, "global quantile mapping on p / pbar")
open(C.HERE / f"volume_state_s{seed}.txt", "w").write("\n".join(out) + "\n")
json.dump(res, open(C.HERE / f"volume_state_s{seed}.json", "w"), indent=1, default=float)
print("\n".join(out))
