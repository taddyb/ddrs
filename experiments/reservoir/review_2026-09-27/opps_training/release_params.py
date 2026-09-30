"""Per-dam release parameters, seed 42 vs 43, against NID features and the offline fits."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R42 = "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-55Z-train-and-test/release_params.csv"
R43 = "/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T10-31-50Z-train-and-test/release_params.csv"
FEAT = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/reservoir/release_head/dam_features.csv"
NID = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/nid/nid_dams_in_eval_network.csv"
SMOKE = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/expected_release_fit.csv"
SMOKE_LEARNED = "/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4/experiments/reservoir/release_head/results/release_params_smoke_learned_fixed.csv"

a = pd.read_csv(R42).set_index("COMID")
b = pd.read_csv(R43).set_index("COMID")
f = pd.read_csv(FEAT).set_index("COMID")
j = a.join(b, lsuffix="_42", rsuffix="_43").join(f)
print("dams:", len(j))
for c in ("T0_days", "a", "b"):
    r = spearmanr(j[c + "_42"], j[c + "_43"]).correlation
    print(f"seed42 vs seed43 Spearman {c}: {r:.3f}")
# phase
for s in ("42", "43"):
    j["amp_" + s] = np.hypot(j["a_" + s], j["b_" + s])
    j["phase_" + s] = np.degrees(np.arctan2(j["b_" + s], j["a_" + s]))
print("amplitude Spearman between seeds:", spearmanr(j.amp_42, j.amp_43).correlation)
d = (j.phase_42 - j.phase_43 + 180) % 360 - 180
print("phase difference seed42-43: median %.0f deg, IQR %.0f..%.0f" % (np.median(d), *np.percentile(d, [25, 75])))
print("T0 medians: seed42 %.3f d, seed43 %.3f d; amp medians %.3f %.3f" % (
    j.T0_days_42.median(), j.T0_days_43.median(), j.amp_42.median(), j.amp_43.median()))
print("a mean/std seed42 %.3f/%.3f seed43 %.3f/%.3f; b %.3f/%.3f %.3f/%.3f" % (
    j.a_42.mean(), j.a_42.std(), j.a_43.mean(), j.a_43.std(), j.b_42.mean(), j.b_42.std(), j.b_43.mean(), j.b_43.std()))
print("T0 ratio seed43/seed42: median %.2f, IQR %.2f..%.2f" % (
    (j.T0_days_43 / j.T0_days_42).median(), *np.percentile(j.T0_days_43 / j.T0_days_42, [25, 75])))

feats = [c for c in f.columns if c not in ("n_dams", "largest_nid_id", "largest_name", "storage_mcm", "year_completed")]
print("\nSpearman(feature, log T0) both seeds; and with a, b (seed 42):")
for c in feats:
    x = j[c]
    if x.nunique() < 2:
        continue
    print(f"  {c:32s} T0: {spearmanr(x, np.log(j.T0_days_42)).correlation:+.3f} {spearmanr(x, np.log(j.T0_days_43)).correlation:+.3f}"
          f"   a42 {spearmanr(x, j.a_42).correlation:+.3f} b42 {spearmanr(x, j.b_42).correlation:+.3f}  a43 {spearmanr(x, j.a_43).correlation:+.3f} b43 {spearmanr(x, j.b_43).correlation:+.3f}")

# linear R2 of log T0 on the features (how much of the head's output is a linear function of inputs)
X = j[feats].values
X = np.c_[np.ones(len(X)), X]
for s in ("42", "43"):
    y = np.log(j["T0_days_" + s].values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"linear R2 of log T0 on 19 features, seed {s}: {r2:.3f}; std log T0 {y.std():.3f}")
# unit-interval position of T0 in the log box
lo, hi = np.log(1 / 24), np.log(365)
u = (np.log(j.T0_days_42) - lo) / (hi - lo)
print("u(T0) seed42: median %.3f, 5-95%% %.3f-%.3f; init 0.166" % (u.median(), *np.percentile(u, [5, 95])))
u = (np.log(j.T0_days_43) - lo) / (hi - lo)
print("u(T0) seed43: median %.3f, 5-95%% %.3f-%.3f" % (u.median(), *np.percentile(u, [5, 95])))

# NID: storage, drainage, residence proxies
nid = pd.read_csv(NID)
nid = nid[nid.storage_mcm >= 10]
g = nid.groupby("COMID").agg(storage=("storage_mcm", "sum"), da=("da_km2", "max"), uparea=("reach_uparea_km2", "max"),
                             year_nid=("year", "max"), n=("nid_id", "count"), purpose=("primary_purpose", "first"))
jj = j.join(g, how="inner")
print("\njoined with NID:", len(jj))
print("Spearman log T0 vs log storage/uparea (seed42 / 43): %.3f / %.3f" % (
    spearmanr(np.log(jj.storage / jj.uparea), np.log(jj.T0_days_42)).correlation,
    spearmanr(np.log(jj.storage / jj.uparea), np.log(jj.T0_days_43)).correlation))
print("T0 by purpose (median d, seed42 / seed43 / n):")
print(jj.groupby("purpose").apply(lambda t: pd.Series({"T0_42": t.T0_days_42.median(), "T0_43": t.T0_days_43.median(), "amp42": t.amp_42.median(), "n": len(t)})).sort_values("n", ascending=False).head(12))
print("T0 by year-completed bin:")
jj["ybin"] = pd.cut(jj.year_nid, [0, 1940, 1960, 1975, 1981, 1995, 2030])
print(jj.groupby("ybin").apply(lambda t: pd.Series({"T0_42": t.T0_days_42.median(), "T0_43": t.T0_days_43.median(), "n": len(t)})))

# offline fit comparison on the smoke set: nearest dam on the gauge reach
s = pd.read_csv(SMOKE)
s = s[(s.role == "dam") & (s.on_reach == True)]
s = s.dropna(subset=["dam_COMID"])
s["dam_COMID"] = s.dam_COMID.astype(int)
m = s.set_index("dam_COMID").join(j, how="inner")
print("\non-reach smoke dams joined:", len(m))
for s_ in ("42", "43"):
    print(f"seed {s_}: Spearman(learned T0, seas_T0) {spearmanr(m['T0_days_' + s_], m.seas_T0).correlation:.3f};"
          f" median learned {m['T0_days_' + s_].median():.2f} d vs fit {m.seas_T0.median():.2f} d;"
          f" ratio median {np.median(m['T0_days_' + s_] / m.seas_T0):.2f}")
    act = m[m.seas_T0 > 0.06]
    print(f"   where fit active ({len(act)}): learned {act['T0_days_' + s_].median():.2f} vs fit {act.seas_T0.median():.2f};"
          f" ratio median {np.median(act['T0_days_' + s_] / act.seas_T0):.2f}, IQR {np.percentile(act['T0_days_' + s_] / act.seas_T0, 25):.2f}-{np.percentile(act['T0_days_' + s_] / act.seas_T0, 75):.2f}")
    # by fitted T0 bins
    act["bin"] = pd.cut(act.seas_T0, [0.06, 0.5, 2, 10, 50, 2000])
    print(act.groupby("bin").apply(lambda t: pd.Series({"fit": t.seas_T0.median(), "learned": t["T0_days_" + s_].median(), "n": len(t)})))
# seasonal: fit's a,b vs learned
print("fit |a|,|b| on +-2 edge:", ((m.seas_a.abs() >= 2) | (m.seas_b.abs() >= 2)).mean())
print("Spearman(fit a, learned a) s42 %.3f s43 %.3f; (fit b, learned b) %.3f %.3f" % (
    spearmanr(m.seas_a, m.a_42).correlation, spearmanr(m.seas_a, m.a_43).correlation,
    spearmanr(m.seas_b, m.b_42).correlation, spearmanr(m.seas_b, m.b_43).correlation))
print("fit phase median deg:", np.median(np.degrees(np.arctan2(m.seas_b, m.seas_a))[m.seas_T0 > 0.06]))

# smoke learned arm vs full learned arm (same seed): how does the loss population change T0?
try:
    sl = pd.read_csv(SMOKE_LEARNED).set_index("COMID")
    q = sl.join(j, how="inner")
    print("\nsmoke-learned vs full-learned seed42: Spearman T0 %.3f; medians %.2f vs %.2f d; ratio median %.2f" % (
        spearmanr(q.T0_days, q.T0_days_42).correlation, q.T0_days.median(), q.T0_days_42.median(),
        np.median(q.T0_days / q.T0_days_42)))
    q["amp"] = np.hypot(q.a, q.b)
    print("smoke-learned amp median %.3f, phase median %.0f deg" % (q.amp.median(), np.median(np.degrees(np.arctan2(q.b, q.a)))))
except Exception as e:
    print("smoke learned:", e)

print("\nrank-1 test: Spearman(log T0, a), (log T0, b) within seed 42: %.3f %.3f; seed 43: %.3f %.3f" % (
    spearmanr(np.log(j.T0_days_42), j.a_42).correlation, spearmanr(np.log(j.T0_days_42), j.b_42).correlation,
    spearmanr(np.log(j.T0_days_43), j.a_43).correlation, spearmanr(np.log(j.T0_days_43), j.b_43).correlation))
print("Spearman(a, b) within seed 42 %.3f, 43 %.3f" % (spearmanr(j.a_42, j.b_42).correlation, spearmanr(j.a_43, j.b_43).correlation))
# standardized linear coefficients per seed
Xs = (j[feats] - j[feats].mean()) / (j[feats].std() + 1e-9)
Xs = np.c_[np.ones(len(Xs)), Xs.values]
print("\nstandardized linear coefficients of log T0 (seed42, seed43):")
for s_ in ("42", "43"):
    y = np.log(j["T0_days_" + s_].values)
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    j["beta_" + s_] = 0
    globals()["beta" + s_] = beta
for i, c in enumerate(feats):
    print(f"  {c:32s} {beta42[i+1]:+.3f} {beta43[i+1]:+.3f}")
# DOR vs fitted T0 on the smoke set
sm = pd.read_csv(SMOKE)
dm = sm[(sm.role == "dam")]
act = dm[dm.seas_T0 > 0.06]
print("\nsmoke dam gauges: Spearman(seas_T0, nid_dor) all %.3f (n %d); active %.3f (n %d)" % (
    spearmanr(dm.seas_T0, dm.nid_dor).correlation, len(dm), spearmanr(act.seas_T0, act.nid_dor).correlation, len(act)))
print("Spearman(seas_T0, storage/uparea) active %.3f; (seas_T0, dam_storage) %.3f; (seas_T0, area_ratio) %.3f; cascade effect: median seas_T0 cascade %.2f vs single %.2f" % (
    spearmanr(act.seas_T0, act.dam_storage_mcm / act.dam_reach_uparea_km2).correlation,
    spearmanr(act.seas_T0, act.dam_storage_mcm).correlation, spearmanr(act.seas_T0, act.area_ratio).correlation,
    act[act.cascade == True].seas_T0.median(), act[act.cascade == False].seas_T0.median()))
print("fit at floor: fraction of on-reach dam gauges %.2f, off-reach %.2f" % (
    (dm[dm.on_reach == True].seas_T0 <= 0.06).mean(), (dm[dm.on_reach == False].seas_T0 <= 0.06).mean()))
