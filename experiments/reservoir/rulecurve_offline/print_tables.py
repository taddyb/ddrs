import json

import numpy as np
import pandas as pd

import summarise as S

res = json.load(open(f"{S.OUT}/summary.json"))
for setname in ["on_reach", "all_dams"]:
    print(f"\n##### {setname} bands at DOR>0.5 (median normalised MSE per band)")
    for law, e in res[setname]["bands_DOR>0.5"].items():
        print(f"  {law:4s} dam  " + " ".join(f"{k}:{v['dam']:.4f}" for k, v in e.items()))
        print(f"  {law:4s} ctl  " + " ".join(f"{k}:{v['control']:.4f}" for k, v in e.items()))
        print(f"  {law:4s} d-c  " + " ".join(f"{k}:{v['dam_minus_control']:+.4f}" for k, v in e.items()))
        print(f"  {law:4s} dam-L0 " + " ".join(f"{k}:{v['dam_change_vs_L0']:+.4f}" for k, v in e.items()))
    print(f"\n##### {setname} parameters (dam gauges; q25/q50/q75)")
    for b, e in res[setname]["params"]["dam"].items():
        print(f"  {b:8s} n={e['n']}")
        for k, v in e.items():
            if k != "n":
                print(f"     {k}: {v}")
    print(f"\n##### {setname} parameters (controls) DOR>0.5 / all")
    for b in ["DOR>0.5", "all"]:
        e = res[setname]["params"]["control"][b]
        print(f"  {b}: T0 L2 {e['T0_L2']} L3 {e['T0_L3']} L4 {e['T0_L4']} f_L3 {e['f_L3']} flux_amp_L4 {e['flux_amp_L4']} phi {e['phi_L3b']}")

df, dam, ctl = S.load()
o = dam[dam.on_reach & (dam.nid_dor > 0.5)]
oc = ctl.loc[o.index]
print("\nfloor share test, on-reach DOR>0.5 dams, L4: pct", np.percentile(o.L4_floor_test, [25, 50, 75, 90]).round(3),
      "frac>0.01", (o.L4_floor_test > 0.01).mean().round(3), "frac>0.1", (o.L4_floor_test > 0.1).mean().round(3))
print("  controls L4 floor pct", np.percentile(oc.L4_floor_test, [50, 75, 90]).round(3))
lo = o.L4_floor_test <= 0.01
for name, m in [("floor<=1%", lo), ("floor>1%", ~lo)]:
    g = o[m]
    print(f"  {name}: n={m.sum()} L4 dNSE median {(g.L4_nse - g.L0_nse).median():+.4f}, L1 {(g.L1_nse - g.L0_nse).median():+.4f}, "
          f"L3 {(g.L3_nse - g.L0_nse).median():+.4f}, L0 median nse {g.L0_nse.median():.3f}")
# observed low flow share: how often obs is near zero at these gauges
