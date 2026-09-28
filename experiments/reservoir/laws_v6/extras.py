"""Small extra numbers for the report: train-vs-test gains of the new laws, W3 timing at irrigation dams,
volume-ratio behaviour of the per-gauge scalar at dams vs controls. Writes extras.txt."""
import numpy as np
import pandas as pd

import v6

F = pd.read_csv(v6.HERE + "/laws_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
sg, dam, ctl = v6.gauges()
ctl_of = ctl.reset_index().set_index("control_for").STAID
G = {"flood-control all": dam[dam.dam_purpose == "Flood Risk Reduction"].index,
     "on-reach DOR>0.5": dam[dam.on_reach & (dam.nid_dor > 0.5)].index,
     "irrigation": dam[dam.dam_purpose == "Irrigation"].index,
     "all dams": dam.index}
L = []
for g, ids in G.items():
    c = [ctl_of[s] for s in ids]
    L.append(f"== {g} (n={len(ids)}) median paired gain, train years | test years (dams), and controls test")
    for law, ref in [("FA", "L2"), ("FC", "L2"), ("FA4", "L4"), ("L4", "L2"), ("W1", "L2"), ("W3", "L2"), ("Kg", "L0"), ("L2Kg", "L2")]:
        tr = (F.loc[ids, f"{law}_train_nse"] - F.loc[ids, f"{ref}_train_nse"]).median()
        ts = (F.loc[ids, f"{law}_nse"] - F.loc[ids, f"{ref}_nse"]).median()
        ct = (F.loc[c, f"{law}_nse"] - F.loc[c, f"{ref}_nse"]).median()
        L.append(f"  {law:5s}-{ref:3s} train {tr:+.4f} | test {ts:+.4f} | ctl test {ct:+.4f}")
ir = dam[dam.dam_purpose == "Irrigation"].index
w3 = F.loc[ir]
L.append(f"\nW3 at irrigation dams: centre doy median {w3.W3_p_centre.median():.0f} [IQR {w3.W3_p_centre.quantile(.25):.0f},"
         f"{w3.W3_p_centre.quantile(.75):.0f}], width {w3.W3_p_width.median():.0f} d, withdrawn share {w3.W3_w_share.median():.3f}")
L.append(f"  share of irrigation dams whose W3 window is centred Oct-Apr (doy >= 274 or <= 120): "
         f"{((w3.W3_p_centre >= 274) | (w3.W3_p_centre <= 120)).mean():.2f}")
for nm, ids in [("dams", dam.index), ("controls", ctl.index)]:
    x = F.loc[ids]
    L.append(f"per-gauge scalar k, {nm}: median {x.Kg_p_k.median():.3f}, share k<0.9 {(x.Kg_p_k < 0.9).mean():.2f}, "
             f"share k>1.1 {(x.Kg_p_k > 1.1).mean():.2f}, median |log k| {np.abs(np.log(x.Kg_p_k)).median():.3f}, "
             f"test beta L0 median {x.L0_beta.median():.3f}, median |log beta| {np.abs(np.log(x.L0_beta)).median():.3f}")
open(v6.HERE + "/extras.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
