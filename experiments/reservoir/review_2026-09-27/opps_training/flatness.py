"""Is the learned T0 inside the offline fit's flat band (within 0.02 NSE of the fit optimum)?"""
import numpy as np
import pandas as pd

ROOT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
fit = pd.read_csv(f"{ROOT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str})
fit = fit[(fit.role == "dam") & (fit.on_reach == True)].dropna(subset=["dam_COMID"])
fit["dam_COMID"] = fit.dam_COMID.astype(int)
r42 = pd.read_csv("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-55Z-train-and-test/release_params.csv").set_index("COMID")
r43 = pd.read_csv("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T10-31-50Z-train-and-test/release_params.csv").set_index("COMID")
sm = pd.read_csv("/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/reservoir/release_head/results/release_params_smoke_learned_fixed.csv").set_index("COMID")
m = fit.set_index("dam_COMID").join(r42[["T0_days"]].rename(columns={"T0_days": "t42"})).join(r43[["T0_days"]].rename(columns={"T0_days": "t43"})).join(sm[["T0_days"]].rename(columns={"T0_days": "tsm"}))
act = m[m.lin_T0 > 0.06]
print("on-reach dams with an active linear fit:", len(act))
w = np.log10(act.lin_T0_hi / act.lin_T0_lo)
print("0.02-NSE band width (decades): median %.2f, IQR %.2f-%.2f; fraction of bands reaching the 0.05 d floor: %.2f" % (
    w.median(), *np.percentile(w, [25, 75]), (act.lin_T0_lo <= 0.051).mean()))
for c in ("t42", "t43", "tsm"):
    inside = (act[c] >= act.lin_T0_lo) & (act[c] <= act.lin_T0_hi)
    below = act[c] < act.lin_T0_lo
    print(f"  {c}: inside band {inside.mean():.2f}, below band {below.mean():.2f}, above {(act[c] > act.lin_T0_hi).mean():.2f};"
          f" median log10(learned / fit optimum) {np.median(np.log10(act[c] / act.lin_T0)):+.2f}")
# where below the band: how far below (decades) and what fit T0 those dams have
for c in ("t42", "tsm"):
    b = act[act[c] < act.lin_T0_lo]
    print(f"  {c} below-band dams: n {len(b)}, fit T0 median {b.lin_T0.median():.1f} d, learned median {b[c].median():.2f} d, decades below lo: median {np.median(np.log10(b.lin_T0_lo / b[c])):.2f}")
# seasonal band
w2 = np.log10(act.seas_T0_hi / act.seas_T0_lo)
print("seasonal fit band width (decades) median %.2f" % w2.median())
