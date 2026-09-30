"""Check the ported fit_L4 against rulecurve_offline's stored L4 fits (unscaled flow)."""
import time
import numpy as np, pandas as pd
import v6
D = v6.load()
sg, dam, ctl = v6.gauges()
rcf = pd.read_csv(v6.RCFITS, dtype={"STAID": str}).set_index("STAID")
ids = list(dam[dam.on_reach & (dam.nid_dor > 0.5)].index[:12])
d = []
t = time.time()
for s in ids:
    i = D["ids"].get_loc(s)
    g = v6.prep(D["P"][i], D["O"][i], D["doy"], D["train"], D["test"])
    T4, c4, Ib = v6.fit_L4(g)
    q = v6.sim(g, T=T4, r=Ib * (c4 @ g["H"]), want=True)["Q"]
    d.append((v6.metrics(q, g["O"], g["test"])["nse"], rcf.loc[s, "L4_nse"], v6.metrics(q, g["O"], g["train"])["nse"], rcf.loc[s, "L4_train_nse"]))
d = np.array(d)
print(np.round(d, 4))
print("secs/gauge", (time.time() - t) / len(ids))
