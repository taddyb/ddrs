"""Check that v6.sim reproduces rulecurve_offline L0/L2/L4 test NSE from the stored parameters."""
import numpy as np, pandas as pd
import v6
D = v6.load()
rcf = pd.read_csv(v6.RCFITS, dtype={"STAID": str}).set_index("STAID")
d = []
for s in rcf.index[::7]:
    i = D["ids"].get_loc(s)
    g = v6.prep(D["P"][i], D["O"][i], D["doy"], D["train"], D["test"])
    p = rcf.loc[s]
    q2 = v6.sim(g, T=float(p.L2_p_T0), want=True)["Q"]
    q4 = v6.sim(g, T=float(p.L4_p_T0), r=v6.l4_flux(g, [p.L4_p_c1s, p.L4_p_c1c, p.L4_p_c2s, p.L4_p_c2c]), want=True)["Q"]
    d.append((v6.metrics(g["I"], g["O"], g["test"])["nse"] - p.L0_nse, v6.metrics(q2, g["O"], g["test"])["nse"] - p.L2_nse,
              v6.metrics(q4, g["O"], g["test"])["nse"] - p.L4_nse))
d = np.abs(np.array(d))
print("n", len(d), "max abs diff L0 L2 L4:", d.max(axis=0), "median", np.median(d, axis=0))
