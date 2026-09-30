"""Completion-year rule: per-gauge change new learned arm (16-56-06Z, f95f298) minus old learned arm (07-29-55Z),
seed 42, grouped as dam_age_check.py does. Unaffected gauges (all dams built by 1981) are the noise reference
(the co-trained routing head differs between the two runs). Also: NSE on the pre-completion part of the test window
for gauges with a dam built during the test period."""
import numpy as np, pandas as pd, zarr
from scipy.stats import mannwhitneyu

FR = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/"
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs/"
old = pd.read_csv(FR + "full_run/paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
new = pd.read_csv(FR + "full_run/paired_full_run_cy_seed42.csv", dtype={"STAID": str}).set_index("STAID")
dams = pd.read_csv(FR + "nid/nid_dams_in_eval_network.csv", low_memory=False)
big = dams[dams.storage_mcm >= 10]
adj = zarr.open("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
rows = []
for s in old.index:
    up = set(adj[s]["order"][:].astype(np.int64).tolist())
    d = big[big.COMID.isin(up)]
    if len(d) == 0:
        continue
    y = d.year
    rows.append(dict(STAID=s, n_after_1981=int((y > 1981).sum()), n_after_1995=int((y > 1995).sum()),
                     first_test_year=(int(y[y > 1995].max()) if (y > 1995).any() else np.nan)))
g = pd.DataFrame(rows).set_index("STAID")
g["group"] = np.select([g.n_after_1995 > 0, g.n_after_1981 > 0], ["built during test", "built during train"], "all built by 1981")
g["d_nse"] = new.nse_learned - old.nse_learned
g["d_kge"] = new.kge_learned - old.kge_learned
g["dam_gain_old"] = old.dnse
g["dam_gain_new"] = new.dnse
ref = g[g.group == "all built by 1981"]
print("new minus old learned arm, per gauge (median [q25, q75]); gain vs off arm old -> new")
for k, x in g.groupby("group"):
    p = mannwhitneyu(x.d_nse, ref.d_nse).pvalue if k != "all built by 1981" else np.nan
    print(f"  {k:20s} n={len(x):3d}  dNSE {x.d_nse.median():+.4f} [{x.d_nse.quantile(.25):+.4f}, {x.d_nse.quantile(.75):+.4f}]"
          f"  |dNSE|>0.01: {(x.d_nse.abs() > 0.01).mean():.2f}  dKGE {x.d_kge.median():+.4f}  MWU p vs ref {p:.3f}"
          f"  gain {x.dam_gain_old.median():+.4f} -> {x.dam_gain_new.median():+.4f}")
undam = new.index.difference(g.index)
print(f"  undammed            n={len(undam)}  dNSE {(new.nse_learned - old.nse_learned)[undam].median():+.4f}")

# Pre-completion window for gauges whose dam was built during the test period.
def load(run):
    z = zarr.open(RUNS + run + "/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    return {s: i for i, s in enumerate(ids)}, t, z["predictions"], z["observations"]
io, t, Po, O = load("2026-09-27T07-29-47Z-train-and-test")
il, _, Pl, _ = load("2026-09-27T07-29-55Z-train-and-test")
ic, _, Pc, _ = load("2026-09-27T16-56-06Z-train-and-test")
def nse(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    return 1 - ((p[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum() if m.sum() > 180 else np.nan
rows = []
for s, x in g[g.group == "built during test"].iterrows():
    pre = t < pd.Timestamp(int(x.first_test_year), 1, 1)
    if pre.sum() < 365:
        continue
    o = O[io[s]][pre].astype(float)
    rows.append(dict(STAID=s, days=int(pre.sum()), off=nse(Po[io[s]][pre].astype(float), o),
                     old=nse(Pl[il[s]][pre].astype(float), o), new=nse(Pc[ic[s]][pre].astype(float), o)))
w = pd.DataFrame(rows)
print(f"\npre-completion window, gauges with a dam built during test: n={len(w)}, median days {w.days.median():.0f}")
print(f"  NSE off {w.off.median():.4f}  old learned {w.old.median():.4f}  new learned {w.new.median():.4f}")
print(f"  median (new - old) {(w.new - w.old).median():+.4f}; median |new - off| {(w.new - w.off).abs().median():.4f} "
      f"vs |old - off| {(w.old - w.off).abs().median():.4f}")
