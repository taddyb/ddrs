"""Cross-store ensembles without selection: the working recipe (n + gamma, seed 42, same config) was trained on five
other Q' stores on 2026-09-13. Report UH paired with EACH of them (equal weights), and UH with all of them, so the
dHBV2-distributed pairing is not chosen on test skill alone. Test years WY1996-2010, vs seed 42 (0.7391 / 0.7591).
Writes store_ensembles.txt."""
import numpy as np

import common as C

STORES = {
    "dhbv2 distributed aorc2f": "2026-09-13T17-22-30Z-train-and-test",
    "dhbv2 lumped aorc2f": "2026-09-13T13-56-50Z-train-and-test",
    "daily lstm": "2026-09-13T13-55-03Z-train-and-test",
    "hourly lstm": "2026-09-13T14-14-46Z-train-and-test",
    "hydrodl lstm aorc2f": "2026-09-13T17-21-58Z-train-and-test",
}
out = []
ids, tt, P42, O = C.load_preds(C.S42)
B = C.metrics(P42, O)
bn, bk = B.nse.values, B.kge.values
P = {}
for name, r in STORES.items():
    i, t, Pr, _ = C.load_preds(r)
    assert (t == tt).all()
    P[name] = C.align(i, Pr, ids)
    M = C.metrics(P[name], O)
    out.append(f"single {name:28s} NSE {np.nanmedian(M.nse):.4f} KGE {np.nanmedian(M.kge):.4f}")
for name in STORES:
    M = C.metrics((P42 + P[name]) / 2, O)
    out.append(C.fmt(C.summarize(f"UH + {name}", M.nse.values, bn)))
    out.append(C.fmt(C.summarize(f"UH + {name}", M.kge.values, bk, "KGE")))
M = C.metrics(np.mean([P42] + list(P.values()), axis=0), O)
out.append(C.fmt(C.summarize("UH + all five stores (6 members)", M.nse.values, bn)))
out.append(C.fmt(C.summarize("UH + all five stores (6 members)", M.kge.values, bk, "KGE")))
M = C.metrics(np.mean([P42, P["dhbv2 distributed aorc2f"], P["daily lstm"]], axis=0), O)
out.append(C.fmt(C.summarize("UH + dhbv2 distributed + daily lstm", M.nse.values, bn)))
open(C.HERE / "store_ensembles.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
