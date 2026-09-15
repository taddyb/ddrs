"""Close the open question from PR #42.

The derivative-loss model's 400-gauge stratified census gave a median signed
roughness displacement of +0.000 — gauges sitting exactly at their own optimum.
Without a matched control that number could not be attributed to the LOSS rather
than to the optimizer budget, since both models had 500 updates but only one had
the time-derivative term. This is the control: the same 400 gauges, same
objective, same window, same Newton search, on the nse-batch model.
"""
import glob

import numpy as np
import pandas as pd

ROOT = "/home/tbindas/projects/ddrs/.ddrs/experiments"
ARMS = {
    "nse-batch (control)": f"{ROOT}/landscape-p21b-strat400/2026-09-11T21-00-03Z-shard-*/*/summary.csv",
    "nse-deriv": f"{ROOT}/landscape-p21c-strat400/2026-09-11T12-36-26Z-shard-*/*/summary.csv",
}

frames = {}
for name, pat in ARMS.items():
    files = sorted(glob.glob(pat))
    if not files:
        print(f"{name}: no summary.csv found at {pat}")
        continue
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frames[name] = df
    print(f"{name}: {len(df)} gauges from {len(files)} shards")

if not frames:
    raise SystemExit(1)

print("\ncolumns:", list(next(iter(frames.values())).columns)[:18])

# Well-fit, and with a usable optimum away from the search-box edge.
def prep(df):
    d = df.copy()
    for c in ("nse0", "alpha_n_star", "clamped"):
        if c not in d.columns:
            print(f"  ! missing column {c}")
    d = d[np.isfinite(d.get("alpha_n_star", np.nan))]
    if "nse0" in d:
        d = d[d["nse0"] > 0.3]
    if "clamped" in d:
        d = d[d["clamped"] < 0.05]
    return d


print()
print(f"{'arm':<22} {'n':>5} {'median a*':>11} {'median |a*|':>12} {'|a*|<0.10':>10} {'|a*|<0.25':>10}")
for name, df in frames.items():
    d = prep(df)
    a = d["alpha_n_star"].to_numpy()
    print(
        f"{name:<22} {len(d):>5} {np.median(a):>+11.3f} {np.median(np.abs(a)):>12.3f} "
        f"{100*np.mean(np.abs(a) < 0.10):>9.1f}% {100*np.mean(np.abs(a) < 0.25):>9.1f}%"
    )

# Paired, on the gauges both arms resolved.
if len(frames) == 2:
    (n1, d1), (n2, d2) = [(k, prep(v)) for k, v in frames.items()]
    key = "staid" if "staid" in d1.columns else d1.columns[0]
    m = d1.merge(d2, on=key, suffixes=("_a", "_b"))
    if len(m):
        a, b = m["alpha_n_star_a"].to_numpy(), m["alpha_n_star_b"].to_numpy()
        print(f"\npaired on {len(m)} gauges resolved by both:")
        print(f"  {n1:<22} median a* = {np.median(a):+.3f}   median |a*| = {np.median(np.abs(a)):.3f}")
        print(f"  {n2:<22} median a* = {np.median(b):+.3f}   median |a*| = {np.median(np.abs(b)):.3f}")
        diff = b - a
        print(f"  median (deriv - control) = {np.median(diff):+.3f}")
        share = np.mean(np.abs(b) < np.abs(a))
        print(f"  deriv closer to its own optimum at {100*share:.1f}% of gauges")
        from scipy.stats import wilcoxon

        try:
            st = wilcoxon(np.abs(a), np.abs(b))
            print(f"  Wilcoxon on |a*|: p = {st.pvalue:.3g}")
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")
