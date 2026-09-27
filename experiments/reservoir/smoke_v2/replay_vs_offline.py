"""Per-gauge: offline fitted gain vs in-engine replay gain (same parameters), on-reach DOR > 0.5 smoke gauges."""
import numpy as np, pandas as pd, zarr
from scipy.stats import spearmanr
W = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/"
R = W + ".ddrs/runs/"
SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"


def load(run):
    z = zarr.open(R + run + "/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    return {s: i for i, s in enumerate(ids)}, z["predictions"][:].astype(float), z["observations"][:].astype(float)


def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return 1 - ((p[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum()


idx, P0, O = load("2026-09-27T04-29-33Z-train-and-test")
_, P2, _ = load("2026-09-27T23-22-35Z-train-and-test")
_, P4, _ = load("2026-09-27T23-22-39Z-train-and-test")
sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
onr = pd.read_csv(SM + "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID").on_reach.astype(str) == "True"
fits = pd.read_csv(W + "experiments/reservoir/rulecurve_offline/fits_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
tab4 = pd.read_csv(W + "experiments/reservoir/smoke/fixed_L4.csv")
rows = []
for s in sm.index[(sm.role == "dam") & (sm.nid_dor > 0.5)]:
    if not onr.get(s, False) or s not in idx or s not in fits.index:
        continue
    i = idx[s]
    n0 = nse(P0[i], O[i])
    f = fits.loc[s]
    rows.append(dict(STAID=s, dam=sm.dam_COMID[s], purpose=sm.dam_purpose[s], n0_engine=n0, n0_offline=f.L0_nse,
                     eng_L2=nse(P2[i], O[i]) - n0, off_L2=f.L2_nse - f.L0_nse,
                     eng_L4=nse(P4[i], O[i]) - n0, off_L4=f.L4_nse - f.L0_nse, T0=f.L4_p_T0))
D = pd.DataFrame(rows)
print(f"n={len(D)}  no-dam NSE: engine median {D.n0_engine.median():.3f}, offline median {D.n0_offline.median():.3f}; "
      f"per-gauge Spearman of the two no-dam NSEs {spearmanr(D.n0_engine, D.n0_offline)[0]:.2f}")
for k in ["L2", "L4"]:
    e, o = D["eng_" + k], D["off_" + k]
    print(f"{k}: engine median {e.median():+.4f}, offline {o.median():+.4f}; Spearman(engine, offline) {spearmanr(e, o)[0]:+.2f}; "
          f"median ratio engine/offline where offline > 0.01: {(e / o)[o > 0.01].median():.2f}; "
          f"engine < -0.01 at {(e < -0.01).sum()} gauges (offline < -0.01 at {(o < -0.01).sum()})")
t4 = pd.read_csv(W + "experiments/reservoir/smoke/fixed_L4.csv").set_index("COMID")
D["cmax"] = [t4.loc[int(c), ["c1s", "c1c", "c2s", "c2c"]].abs().max() if int(c) in t4.index else np.nan for c in D.dam]
for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 9)]:
    x = D[(D.cmax > lo) & (D.cmax <= hi)]
    if len(x):
        print(f"max|c| in ({lo},{hi}]: n={len(x):3d}  engine L4 median {x.eng_L4.median():+.4f}  offline {x.off_L4.median():+.4f}  "
              f"engine L2 {x.eng_L2.median():+.4f}  engine<-0.01: {(x.eng_L4 < -0.01).sum()}")
x = D[D.cmax <= 1]
print(f"physical amplitudes (max|c| <= 1): n={len(x)} engine L4 {x.eng_L4.median():+.4f} vs offline {x.off_L4.median():+.4f}")
D["combo"] = np.where(D.cmax <= 1, D.eng_L4, D.eng_L2)
b = np.median(np.random.default_rng(42).choice(D.combo.values, (2000, len(D))), axis=1)
print(f"bounded estimate (rule curve where max|c| <= 1, else bucket): median {D.combo.median():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}], up {(D.combo > 0).sum()} down {(D.combo < 0).sum()}")
worst = D.sort_values("eng_L4").head(8)[["STAID", "purpose", "T0", "n0_engine", "eng_L2", "eng_L4", "off_L4"]]
print("\nworst engine L4 gauges:\n" + worst.round(3).to_string(index=False))
c = pd.read_csv(R + "2026-09-27T23-22-39Z-train-and-test/release_clamp.csv")
print(f"\nL4 replay clamp: created {c.created_m3.sum() / c.inflow_m3.sum():.4%} of dam inflow; clamp steps "
      f"{c.clamp_steps.sum() / c.steps.sum():.3%}; dams with created > 1% of their inflow: "
      f"{(c.created_m3 > 0.01 * c.inflow_m3).sum()} of {len(c)}")
c2 = pd.read_csv(R + "2026-09-27T23-22-35Z-train-and-test/release_clamp.csv")
print(f"L2 replay clamp: created {c2.created_m3.sum() / c2.inflow_m3.sum():.4%}; clamp steps {c2.clamp_steps.sum() / c2.steps.sum():.3%}")
