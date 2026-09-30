"""Is the high-DOR NSE deficit a dam effect or a regional Q' effect? Compare each smoke dam gauge with its
matched control (same HUC2, closest area, no dam), in the seed-42 off arm."""
import numpy as np, pandas as pd
WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/"
sm = pd.read_csv(WT + "smoke/smoke_gauges.csv", dtype={"STAID": str, "control_for": str})
D = pd.read_csv("/home/tbindas/.claude/jobs/dacd6d8c/tmp/decomp.csv", dtype={"STAID": str}).set_index("STAID")
ctl = sm[sm.role == "control"].set_index("control_for")
dam = sm[sm.role == "dam"].set_index("STAID")
dam = dam.join(ctl[["STAID"]].rename(columns={"STAID": "ctl"}))
cols = ["nse", "r", "alpha", "beta", "bias2", "corr", "clim", "anom"]
rows = []
for s, x in dam.iterrows():
    if s not in D.index or x.ctl not in D.index:
        continue
    a, b = D.loc[s], D.loc[x.ctl]
    rows.append(dict(STAID=s, dor=x.nid_dor, **{c: a[c] for c in cols}, **{c + "_c": b[c] for c in cols},
                     on_reach=(x.area_ratio <= 1.05)))
R = pd.DataFrame(rows)
R["bin"] = pd.cut(R.dor, [-1e-9, 0.1, 0.5, 1, 2, 1e9], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
g = R.groupby("bin", observed=True)
out = g.size().rename("n").to_frame()
for c in ["nse", "r", "beta", "corr", "clim", "anom"]:
    out[c + " dam"] = g[c].median().round(3)
    out[c + " ctl"] = g[c + "_c"].median().round(3)
print(out.to_string())
R["gap"] = R.nse - R.nse_c
print("\nmedian paired gap (dam - control) NSE by bin:", g.apply(lambda x: round((x.nse - x.nse_c).median(), 3)).to_dict())
hi = R[R.dor > 0.5]
print("DOR>0.5: n", len(hi), "dam NSE", round(hi.nse.median(), 3), "control", round(hi.nse_c.median(), 3),
      "paired gap", round((hi.nse - hi.nse_c).median(), 3))
