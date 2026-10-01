"""Pick three representative flood-control gauges: interior optima in z and kc, at the 25th/50th/75th percentile of
the z curvature at the optimum."""
import numpy as np
import pandas as pd

import landscapes as L

df = pd.read_csv("per_gauge_slices.csv", dtype={"STAID": str})
z = df[(df.law == "FA") & (df.param == "z")].set_index("STAID")
kc = df[(df.law == "FA") & (df.param == "kc")].set_index("STAID")
ok = z[(~z.edge_opt) & (~kc.loc[z.index].edge_opt) & (z.z_raw < 120)]
qs = np.percentile(ok.H_q, [25, 50, 75])
pick = [ok.index[np.argmin(np.abs(ok.H_q - v))] for v in qs]
lf = L.LF
for s in pick:
    print(s, L.DAM.loc[s, "staname"], L.DAM.loc[s, "dam_name"], "DOR", round(L.DAM.loc[s, "nid_dor"], 2),
          "H_z", round(z.loc[s, "H_q"], 4), "z*", round(z.loc[s, "v_opt"], 2), "kc*", round(kc.loc[s, "v_opt"], 2),
          "FA-L2 test", round(lf.loc[s, "FA_nse"] - lf.loc[s, "L2_nse"], 4))
open("picked_gauges.txt", "w").write("\n".join(pick) + "\n")
