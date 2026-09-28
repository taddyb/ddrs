"""Task 1, part 2: where the 30-120 d error of the per-dam bucket lives, and how much of it any law could reach.

Per gauge (test WY1996-2010), residual e = Q_L2 - O band-passed to 30-120 d (FFT mask). Reported per group:
  post_frac   share of the band-passed residual energy inside the 0-120 d windows after inflow peaks (diag.py events)
  time_frac   share of test days inside those windows (post_frac / time_frac > 1 means the error clusters after floods)
  phase_frac  share of the band-passed residual variance that is its day-of-year climatology (phase-locked, what a
              fixed rule curve can reach)
  coh_I       mean squared coherence of e with the model inflow I over 30-120 d periods (the share of that band's
              error linearly predictable from the inflow, an upper bound for any time-invariant linear law on I)
  coh_I_obs   the same with the observed-minus-no-dam signal (O - I) as the target instead (dam signal predictability)
Also the test-year volume ratio O/I quantiles at irrigation and water-supply dams vs controls.
"""
import numpy as np
import pandas as pd
from scipy.signal import coherence

import v6
from diag import series, events, sets, ctl_of, D, dam, ctl

te = D["test"]
ti = np.flatnonzero(te)


def bp(x, lo=30, hi=120):
    F = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x))
    per = np.where(f > 0, 1 / np.maximum(f, 1e-12), 1e12)
    F[(per < lo) | (per >= hi)] = 0
    return np.fft.irfft(F, len(x))


def stats(s):
    g, q2, q4 = series(s)
    O = g["O"][te]
    m = np.isfinite(O)
    if m.mean() < 0.8:
        return None
    x = np.arange(len(O))
    O = np.interp(x, x[m], O[m])
    I = g["I"][te]
    e = bp(q2[te] - O)
    ev = [t - ti[0] for t in events(g["I"], D["wy"]) if te[t]]
    post = np.zeros(len(O), bool)
    for t in ev:
        post[t:t + 121] = True
    E = e ** 2
    doy = np.minimum(D["doy"][te], 365) - 1
    clim = pd.Series(e).groupby(doy).transform("mean").values
    f, c = coherence(e, I - I.mean(), nperseg=1024)
    f2, c2 = coherence(bp(O - I), I - I.mean(), nperseg=1024)
    per = np.where(f > 0, 1 / np.maximum(f, 1e-12), 1e12)
    band = (per >= 30) & (per < 120)
    return dict(STAID=s, post_frac=E[post].sum() / E.sum(), time_frac=post.mean(),
                phase_frac=float(clim.var() / e.var()), coh_I=float(c[band].mean()), coh_I_obs=float(c2[band].mean()),
                vol_ratio=float(O.sum() / I.sum()))


lines = []
for lab, ids in list(sets.items()) + [("all on-reach DOR>0.5", dam[dam.on_reach & (dam.nid_dor > 0.5)].index)]:
    a = pd.DataFrame([r for r in map(stats, ids) if r]).set_index("STAID")
    b = pd.DataFrame([r for r in map(stats, [ctl_of[s] for s in ids]) if r]).set_index("STAID")
    lines.append(f"\n== {lab}: dams n={len(a)}, controls n={len(b)} (medians)")
    for col in ["post_frac", "time_frac", "phase_frac", "coh_I", "coh_I_obs"]:
        lines.append(f"  {col:11s} dam {a[col].median():.3f}   control {b[col].median():.3f}")
    lines.append(f"  post_frac/time_frac  dam {(a.post_frac / a.time_frac).median():.2f}   control {(b.post_frac / b.time_frac).median():.2f}")

for lab in ["Irrigation", "Water Supply"]:
    ids = dam[dam.dam_purpose == lab].index
    a = pd.DataFrame([r for r in map(stats, ids) if r]).set_index("STAID")
    b = pd.DataFrame([r for r in map(stats, [ctl_of[s] for s in ids]) if r]).set_index("STAID")
    qa, qb = a.vol_ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9]), b.vol_ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    lines.append(f"\n== {lab}: test-year observed/no-dam volume O/I, quantiles 10/25/50/75/90")
    lines.append("  dams     " + " ".join(f"{v:.3f}" for v in qa) + f"   share < 0.8: {(a.vol_ratio < 0.8).mean():.2f}")
    lines.append("  controls " + " ".join(f"{v:.3f}" for v in qb) + f"   share < 0.8: {(b.vol_ratio < 0.8).mean():.2f}")
    lines.append(f"  mean I/O (predicted/observed volume): dams {(1 / a.vol_ratio).mean():.3f}, controls {(1 / b.vol_ratio).mean():.3f}")
open(v6.HERE + "/diag2.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
