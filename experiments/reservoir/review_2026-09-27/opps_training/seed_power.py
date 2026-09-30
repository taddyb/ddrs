"""How many seeds resolve effects of this size? Per-gauge paired dNSE (seed 42 full run) and the offline fit's
relation to dam year and DOR on the smoke set."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

p = pd.read_csv("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/full_run/paired_full_run.csv", dtype={"STAID": str})
dam = p[p.n_nid_ge10mcm > 0]
und = p[p.n_nid_ge10mcm == 0]
for lab, t in [("dammed", dam), ("undammed", und), ("on-reach", p[p.nid_on_gauge_reach == True])]:
    d = t.dnse.values
    d = d[np.isfinite(d)]
    print(f"{lab:9s} n={len(d)} median {np.median(d):+.4f} mean {d.mean():+.4f} sd {d.std():.4f} IQR {np.percentile(d,25):+.4f}..{np.percentile(d,75):+.4f}; bootstrap SE of median {np.std([np.median(np.random.default_rng(i).choice(d, len(d))) for i in range(300)]):.5f}")
# seed-to-seed: the two seeds gave medians +0.0014 / +0.0039 (dammed), -0.0007 / -0.0004 (undammed); the no-dam
# seed pair differs by -0.0008 at undammed gauges. Treat the seed SD of a run's median as s.
for lab, meds in [("dammed", (0.0014, 0.0039)), ("undammed", (-0.0007, -0.0004))]:
    s = np.std(meds, ddof=1)
    eff = np.mean(meds)
    print(f"{lab}: two-seed medians {meds}, seed SD {s:.4f}, mean effect {eff:+.4f};")
    for n in (2, 3, 4, 6, 8, 12):
        se = s * np.sqrt(2.0 / n)  # paired arms, n seeds each, difference of arm means
        print(f"   n seeds/arm {n:2d}: SE of the arm difference {se:.4f}, t = {eff / se:.1f}")
print("undammed: the no-dam seed pair differs by -0.0008 at the same gauges, so a one-run median has seed SD about %.4f" % (0.0008 / np.sqrt(2)))

sm = pd.read_csv("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str})
d = sm[(sm.role == "dam")]
act = d[d.seas_T0 > 0.06]
print("\noffline seas_T0 vs dam_year: Spearman all %.3f, active %.3f" % (spearmanr(d.seas_T0, d.dam_year, nan_policy="omit").correlation, spearmanr(act.seas_T0, act.dam_year, nan_policy="omit").correlation))
print("offline seas_T0 vs dam_height: %.3f; vs max_discharge: %.3f; vs storage: %.3f; vs nid_dor %.3f; vs dam_da %.3f" % tuple(
    spearmanr(act.seas_T0, act[c], nan_policy="omit").correlation for c in ("dam_height_m", "dam_max_discharge_m3s", "dam_storage_mcm", "nid_dor", "dam_da_km2")))
print("active fraction by purpose:")
print(d.groupby("dam_purpose").apply(lambda t: pd.Series({"active": (t.seas_T0 > 0.06).mean(), "T0_med_active": t[t.seas_T0 > 0.06].seas_T0.median(), "n": len(t)})).sort_values("n", ascending=False).head(8))
# seasonal amplitude in the offline fit vs T0
act = act.copy()
act["amp"] = np.hypot(act.seas_a, act.seas_b)
print("offline amplitude: median %.2f; fraction amp>=1: %.2f; Spearman(amp, T0) %.3f; gain seasonal minus linear (d_seas - d_lin) median %.4f" % (
    act.amp.median(), (act.amp >= 1).mean(), spearmanr(act.amp, act.seas_T0).correlation, (act.d_seas - act.d_lin).median()))
print("offline phase (deg, atan2(b,a)) quartiles of active dams with amp>0.5:", np.percentile(np.degrees(np.arctan2(act.seas_b, act.seas_a))[act.amp > 0.5], [10, 25, 50, 75, 90]))
