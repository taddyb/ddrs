"""Train -> test transfer of each 1-D optimum: NSE gain from the engine init to the training optimum on training years
vs test years, share of gauges whose test gain is positive, and where the test-year optimum sits.
Writes transfer.txt."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "classification.csv"), dtype={"STAID": str})
lines = ["gain init -> training optimum along the slice (other parameters fitted): train vs test",
         "  key      median train gain  median test gain  ratio  test gain > 0   test NSE lost vs test-optimal"]
for k in ["L2:T0", "L4:T0", "L4:amp", "FA:T0", "FA:z", "FA:kc", "FA:phi"]:
    s = df[df.key == k]
    a, b = s.gain_tr.median(), s.gain_te.median()
    lines.append(f"  {k:7s}  {a:+.4f}            {b:+.4f}           {b / a:4.2f}   {(s.gain_te > 0).mean():.0%}"
                 f"            {s.te_cost.median():.4f}")
s = df[df.key == "L4:amp"]
lines.append(f"\nrule-curve amplitude: test-optimal s < 0.25 (test years prefer no rule curve) at {(s.x_teopt < 0.25).mean():.0%};"
             f" test-optimal s within 0.25 of 1 at {(np.abs(s.x_teopt - 1) <= 0.25).mean():.0%}; split-half optimum "
             f"s_h1 {s.x_h1.median():.2f}, s_h2 {s.x_h2.median():.2f}, |s_h1 - s_h2| median {np.abs(s.x_h1 - s.x_h2).median():.2f}")
z = df[df.key == "FA:z"]
lines.append(f"pool z: test-optimal z / training-optimal z, median {np.exp(np.median(z.x_teopt - z.x_opt)):.2f}, "
             f"|log10| median {np.median(np.abs(z.x_teopt - z.x_opt)) / np.log(10):.2f} decades; test optimum at the grid "
             f"top (120 d) {(np.isclose(np.exp(z.x_teopt), 120)).mean():.0%}")
kc = df[df.key == "FA:kc"]
lines.append(f"kc: test-optimal / training-optimal median {np.exp(np.median(kc.x_teopt - kc.x_opt)):.2f}, |log10| "
             f"median {np.median(np.abs(kc.x_teopt - kc.x_opt)) / np.log(10):.3f} decades; local minima per slice "
             f"(prominence 0.002) median {kc.nmin_002.median():.0f}, IQR {kc.nmin_002.quantile(.25):.0f}-{kc.nmin_002.quantile(.75):.0f}")
txt = "\n".join(lines)
print(txt)
open(os.path.join(HERE, "transfer.txt"), "w").write(txt + "\n")
