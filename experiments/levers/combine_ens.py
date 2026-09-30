"""Do the recipe lever and the multi-store ensemble stack? Replace the UH member of the selected UH+DIST+LSTM ensemble by
the gamma = 0 UH variants (single seed each), equal weights, test years WY1996-2010, vs seed 42 (0.7391 / 0.7591).
Writes combine_ens.txt."""
import numpy as np

import common as C

RUN = {"UH42": C.S42, "UH43": C.S43, "NONLY": "2026-09-12T23-39-06Z-train-and-test",
       "G0": "2026-09-12T03-53-34Z-train-and-test", "DIST": "2026-09-13T17-22-30Z-train-and-test",
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
for c in [["NONLY"], ["G0"], ["UH42", "DIST", "LSTM"], ["NONLY", "DIST", "LSTM"], ["G0", "DIST", "LSTM"],
          ["UH42", "UH43", "DIST", "LSTM"], ["G0", "NONLY", "DIST", "LSTM"]]:
    M = C.metrics(np.mean([P[k] for k in c], axis=0), O)
    out.append(C.fmt(C.summarize("+".join(c), M.nse.values, B.nse.values)))
    out.append(C.fmt(C.summarize("+".join(c), M.kge.values, B.kge.values, "KGE")))
open(C.HERE / "combine_ens.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
