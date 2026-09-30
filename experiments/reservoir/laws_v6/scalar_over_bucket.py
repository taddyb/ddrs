"""Per-gauge runoff scalar on top of the per-dam bucket (L2Kg - L2), test years, by purpose; controls get the same
procedure; DiD. Writes scalar_over_bucket.txt."""
import numpy as np

import v6
from summarise import boot, fmt, ctl_of, dam, F

L = ["per-gauge scalar on top of the per-dam bucket (L2Kg - L2), test years; controls get the same; DiD"]
G = {"all 458 dams": dam.index, "on-reach DOR>0.5": dam[dam.on_reach & (dam.nid_dor > 0.5)].index}
for p in ["Flood Risk Reduction", "Hydroelectric", "Water Supply", "Irrigation", "Recreation"]:
    G[p] = dam[dam.dam_purpose == p].index
for g, ids in G.items():
    c = [ctl_of[s] for s in ids]
    a = F.loc[ids, "L2Kg_nse"].values - F.loc[ids, "L2_nse"].values
    b = F.loc[c, "L2Kg_nse"].values - F.loc[c, "L2_nse"].values
    k = F.loc[ids, "L2Kg_kge"].values - F.loc[ids, "L2_kge"].values
    L.append(f"  {g:22s} n={len(ids):3d} dam {fmt(boot(a))} {int((a > 0).sum())}/{int((a < 0).sum())} | ctl {fmt(boot(b))} | "
             f"DiD {fmt(boot(a - b))} | dKGE {np.median(k):+.4f} | k median {F.loc[ids, 'L2Kg_p_k'].median():.3f}")
open(v6.HERE + "/scalar_over_bucket.txt", "w").write("\n".join(L) + "\n")
