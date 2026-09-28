#!/usr/bin/env python
"""How far can a dam model move the MEDIAN NSE of the full 2,365-gauge evaluation population (gages_3000 filter)?

Uses the full-run seed-42 no-dam arm per-gauge NSE (paired_full_run.csv, branch reservoir-options) and adds
counterfactual per-gauge gains to the gauges with a NID dam >= 10 MCM upstream, by DOR bin:
  measured   : the best trained smoke arm's median gain in each DOR bin (all dam gauges), a realistic transfer
  close_gap  : each dammed gauge gains its DOR bin's full dam-minus-matched-control gap (smoke no-dam arm), i.e. a
               perfect dam model that makes every regulated gauge as good as its undammed twin
  ceiling    : every dammed gauge raised to at least the undammed population median (an absolute upper bound)
Reports the population median before and after, and how many gauges cross it.
"""
import numpy as np, pandas as pd

P = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/full_run/paired_full_run.csv"
d = pd.read_csv(P, dtype={"STAID": str}).set_index("STAID")
base = d.nse_off
dam = d.n_nid_ge10mcm > 0
bins = [(-1, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 1e9)]
labels = ["0-0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]
# smoke-set medians (overnight page data): best trained arm gain by DOR bin, all dam gauges; dam-minus-control gap
measured = [0.00195, 0.0066, 0.00695, 0.0193, 0.0076]
gap = [0.032, 0.085, 0.145, 0.199, 0.36]


def binof(x):
    for k, (lo, hi) in enumerate(bins):
        if (x > lo or lo < 0) and x <= hi:
            return k
    return len(bins) - 1


k = d.nid_dor.fillna(0).map(binof)
print(f"population n={len(d)}, median NSE {base.median():.4f}; dammed >=10 MCM n={int(dam.sum())}, "
      f"of which DOR>0.5 n={int((dam & (d.nid_dor > 0.5)).sum())}")
print("dammed gauges by DOR bin and position vs the population median:")
for j, lab in enumerate(labels):
    x = base[dam & (k == j)]
    print(f"  DOR {lab:8s} n={len(x):4d}  median NSE {x.median():.3f}  below pop median {(x < base.median()).mean():.0%}")
und_med = base[~dam].median()
for name, add in [("measured (best smoke arm by bin)", measured), ("close the whole dam gap", gap)]:
    new = base + np.where(dam, [add[j] for j in k], 0.0)
    print(f"{name:34s} median {new.median():.4f}  change {new.median() - base.median():+.4f}")
new = base.where(~dam, np.maximum(base, und_med))
print(f"{'ceiling: dammed raised to undammed median':34s} median {new.median():.4f}  change {new.median() - base.median():+.4f} "
      f"(undammed median {und_med:.4f})")
print(f"for reference: +0.03 means a population median of {base.median() + 0.03:.4f}")
