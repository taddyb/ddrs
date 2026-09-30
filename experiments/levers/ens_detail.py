"""Equal-weight multi-store ensembles of the working recipe (n + gamma), test years WY1996-2010, vs seed 42 UH.
Reports NSE, KGE and the KGE components (r, alpha, beta) at the median, and robustness to the UH seed.
Writes ens_detail.txt."""
import numpy as np

import common as C

RUN = {"UH42": C.S42, "UH43": C.S43, "DIST": "2026-09-13T17-22-30Z-train-and-test",
       "LSTM": "2026-09-13T13-55-03Z-train-and-test"}
ids, tt, P42, O = C.load_preds(C.S42)
P = {"UH42": P42}
for k, r in RUN.items():
    if k not in P:
        i, t, Pr, _ = C.load_preds(r)
        assert (t == tt).all()
        P[k] = C.align(i, Pr, ids)
B = C.metrics(P42, O)
out = []
COMBOS = [["UH42"], ["UH43"], ["DIST"], ["LSTM"], ["UH42", "DIST"], ["UH42", "LSTM"], ["DIST", "LSTM"],
          ["UH42", "DIST", "LSTM"], ["UH43", "DIST", "LSTM"], ["UH42", "UH43", "DIST", "LSTM"]]
out.append(f"{'members':28s} {'NSE':>7s} {'KGE':>7s} {'r':>6s} {'alpha':>6s} {'beta':>6s}   dNSE [CI] paired up | dKGE paired up")
for c in COMBOS:
    Pm = np.mean([P[k] for k in c], axis=0)
    M = C.metrics(Pm, O)
    s = C.summarize("+".join(c), M.nse.values, B.nse.values)
    sk = C.summarize("+".join(c), M.kge.values, B.kge.values, "KGE")
    out.append(f"{'+'.join(c):28s} {np.nanmedian(M.nse):7.4f} {np.nanmedian(M.kge):7.4f} {np.nanmedian(M.r):6.3f} "
               f"{np.nanmedian(M.alpha):6.3f} {np.nanmedian(M.beta):6.3f}   {s['pop_change']:+.4f} [{s['ci_lo']:+.4f},"
               f"{s['ci_hi']:+.4f}] {s['paired_median']:+.4f} {100 * s['share_up']:.0f}% | {sk['pop_change']:+.4f} "
               f"{sk['paired_median']:+.4f} {100 * sk['share_up']:.0f}%")
# pairwise error correlation between members (median over gauges of corr of residuals)
for a, b in [("UH42", "DIST"), ("UH42", "LSTM"), ("DIST", "LSTM"), ("UH42", "UH43")]:
    ea, eb = P[a] - O, P[b] - O
    m = np.isfinite(ea) & np.isfinite(eb)
    cs = []
    for g in range(len(ids)):
        mm = m[g]
        if mm.sum() > 365:
            cs.append(np.corrcoef(ea[g, mm], eb[g, mm])[0, 1])
    out.append(f"median residual correlation {a} vs {b}: {np.nanmedian(cs):.3f}")
open(C.HERE / "ens_detail.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
