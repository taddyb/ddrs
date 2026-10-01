"""Does the offline engine-init gradient predict where the engine's trained pool parameters moved?
Trained values: S6pool run 2026-09-30T02-46-23Z (release_params.csv; per-dam pool from off, lr 0.02, l2 0, 200 steps).
Offline: joint_init.csv (pathwise gradient at the engine init) and year_grad.csv (per-year signs)."""
import os

import numpy as np
import pandas as pd

import landscapes as L

HERE = os.path.dirname(os.path.abspath(__file__))
rp = pd.read_csv(os.path.join(L.AG, ".ddrs/runs/2026-09-30T02-46-23Z-train-and-test/release_params.csv"))
fa = pd.read_csv(os.path.join(L.AG, "experiments/reservoir/smoke/fixed_FA.csv"), dtype={"STAID": str})
tr = fa[["COMID", "STAID"]].merge(rp, on="COMID").set_index("STAID")
jt = pd.read_csv(os.path.join(HERE, "joint_init.csv"), dtype={"STAID": str})
ji = jt[jt.point == "init"].set_index("STAID")
yg = pd.read_csv(os.path.join(HERE, "year_grad.csv"), dtype={"STAID": str})
yi = yg[yg.point == "init"].groupby("STAID").pg_z.agg(lambda v: float((v < 0).mean()))
lines = []
for p, init, col in (("z", 0.05, "pg_z"), ("kc", 3.0, "pg_kc"), ("phi", 0.5, "pg_phi")):
    moved = np.log(tr[p] / init)
    pred = -np.sign(ji.loc[tr.index, col])
    ok = np.abs(moved) > 1e-3
    agree = float((np.sign(moved[ok]) == pred[ok]).mean())
    lines.append(f"{p}: trained median {tr[p].median():.3f} (IQR {tr[p].quantile(.25):.3f}-{tr[p].quantile(.75):.3f}); "
                 f"moved at {ok.mean():.0%}; direction agrees with the offline init gradient at {agree:.0%} of movers "
                 f"(n={int(ok.sum())})")
yy = yi.loc[tr.index]
rho = pd.Series(np.log(tr.z / 0.05)).corr(yy, method="spearman")
lines.append(f"z: Spearman(ln z_trained/0.05, share of training years whose init gradient wants a larger pool) = {rho:.2f}")
lines.append(f"z: offline fitted z median {fa.z_offline_days.median():.1f} d; trained z > 1 d at {(tr.z > 1).mean():.0%}, "
             f"> 5 d at {(tr.z > 5).mean():.0%}")
lines.append(f"T0: Spearman(trained T0, offline FA T0) = {pd.Series(tr.T0_days.values).corr(pd.Series(fa.set_index('STAID').loc[tr.index].T_days.values), method='spearman'):.2f}")
txt = "\n".join(lines)
print(txt)
open(os.path.join(HERE, "engine_training_check.txt"), "w").write(txt + "\n")
