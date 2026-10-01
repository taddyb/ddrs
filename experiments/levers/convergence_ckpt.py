"""Direct convergence test: seed-42 checkpoints at epochs 10 / 25 / 40 / 50 scored on the same five test years
(WY1996-2000, cold start 1995-10-01 in every case; epoch 50 is the source run's own test phase cut to those years).
Usage: convergence_ckpt.py <e10 run> <e25 run> <e40 run>  (any may be '-' if missing). Writes convergence_ckpt.txt.
"""
import sys

import numpy as np

import common as C

out = []
say = out.append
ids, t, P50, O = C.load_preds(C.S42)
wy = C.water_year(t)
m5 = (wy >= 1996) & (wy <= 2000)
P50, O5 = P50[:, m5], O[:, m5]
M50 = C.metrics(P50, O5)
say(f"seed 42, WY1996-2000 ({m5.sum()} d): epoch 50 median NSE {np.nanmedian(M50.nse):.4f} KGE {np.nanmedian(M50.kge):.4f}")
arms = dict(zip([10, 25, 40], sys.argv[1:4]))
for ep, run in arms.items():
    if run == "-":
        continue
    i, tt, P, Ob = C.load_preds(run)
    t5 = t[m5]
    common_d = t5.intersection(tt)
    a, b = t5.get_indexer(common_d), tt.get_indexer(common_d)
    P = C.align(i, P, ids)[:, b]
    Oc = O5[:, a]
    M = C.metrics(P, Oc)
    M50c = C.metrics(P50[:, a], Oc)
    say(f"  common days {len(common_d)}; epoch 50 on them: median NSE {np.nanmedian(M50c.nse):.4f}")
    s = C.summarize(f"epoch {ep} vs epoch 50", M50c.nse.values, M.nse.values)
    sk = C.summarize(f"epoch {ep} vs epoch 50", M50c.kge.values, M.kge.values, "KGE")
    say(f"epoch {ep}: median NSE {np.nanmedian(M.nse):.4f} KGE {np.nanmedian(M.kge):.4f}; change from epoch {ep} to 50:")
    say(C.fmt(s))
    say(C.fmt(sk))
open(C.HERE / "convergence_ckpt.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
