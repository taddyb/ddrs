"""Robust loss-curve slopes: per-epoch MEDIAN micro-batch loss (seed 43 has one exploding micro-batch in epochs 41-50),
OLS and Theil-Sen slope over epochs 41-50 and 31-40, and the change from epochs 6-10 to 41-50. Writes loss_robust.txt."""
import glob

import numpy as np
import pandas as pd
from scipy import stats

import common as C

out = []
for f in sorted(glob.glob(str(C.HERE / "loss_curve_*.csv"))):
    e = pd.read_csv(f).set_index("epoch")
    line = [f.split("loss_curve_")[1][:20]]
    for lo, hi in [(31, 40), (41, 50)]:
        x = e.loc[lo:hi]
        r = stats.linregress(x.index.values, x.loss_med.values)
        ts = stats.theilslopes(x.loss_med.values, x.index.values)
        line.append(f"ep{lo}-{hi}: level {x.loss_med.mean():.4f}, OLS slope {r.slope:+.5f}+-{r.stderr:.5f}/epoch, "
                    f"Theil-Sen {ts.slope:+.5f} [{ts.low_slope:+.5f},{ts.high_slope:+.5f}]")
    a, b = e.loc[6:10].loss_med.mean(), e.loc[41:50].loss_med.mean()
    line.append(f"median-loss level ep6-10 {a:.4f} -> ep41-50 {b:.4f} ({100 * (b - a) / a:+.1f}%); ep1 {e.loss_med.iloc[0]:.4f}")
    out.append("  " + "\n    ".join(line))
open(C.HERE / "loss_robust.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
