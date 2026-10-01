"""Is the seed-to-seed skill difference the unidentified gamma wandering? Per gauge: basin-median (over the gauge's
upstream reaches) n and gamma for seed 42 and seed 43 (final KAN fields), against the per-gauge test NSE difference.
Also: the constant-gamma sweep (p, q learned, seed 42) as an external check of skill against gamma.
Writes variance_gamma.txt.
"""
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy import stats

import common as C

out = []
say = out.append
V = pd.read_csv(C.HERE / "variance_by_gauge.csv", dtype={"Unnamed: 0": str}).rename(columns={"Unnamed: 0": "STAID"})
V = V.set_index("STAID")
gz = zarr.open_group("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")


def field(run, f="kan_parameters.nc"):
    ds = xr.open_dataset(C.RUNS / run / "plot" / f)
    return pd.DataFrame(dict(n=ds.n.values.astype(float), gamma=ds.gamma.values.astype(float)), index=ds.COMID.values)


F42, F43 = field(C.S42, "kan_parameters_epoch50.nc"), field(C.S43)
rows = []
for s in V.index:
    order = np.asarray(gz[s]["order"][:], dtype=np.int64)
    a, b = F42.reindex(order), F43.reindex(order)
    rows.append((s, a.n.median(), b.n.median(), a.gamma.median(), b.gamma.median()))
B = pd.DataFrame(rows, columns=["STAID", "n42", "n43", "g42", "g43"]).set_index("STAID")
V = V.join(B)
V["dg"] = V.g43 - V.g42
V["dlnn"] = np.log(V.n43 / V.n42)
say(f"basin-median gamma: seed42 median {V.g42.median():.3f}, seed43 {V.g43.median():.3f}; share of gauges where seed43 "
    f"gamma is higher {np.mean(V.dg > 0):.3f}")
say(f"basin-median n: seed42 {V.n42.median():.4f}, seed43 {V.n43.median():.4f}")
for c in ["dg", "dlnn", "g43", "g42"]:
    ok = V[c].notna() & V.d43.notna()
    say(f"Spearman(seed43 - seed42 NSE, {c}) = {stats.spearmanr(V.d43[ok], V[c][ok]).correlation:+.3f} (n={ok.sum()})")
say("seed43 - seed42 NSE by tercile of the basin gamma difference (median d, share down):")
q = pd.qcut(V.dg, 3, labels=["low", "mid", "high"])
for k in ["low", "mid", "high"]:
    m = q == k
    say(f"   {k:4s} dg median {V.dg[m].median():+.3f}: d NSE median {V.d43[m].median():+.4f}, share down {np.mean(V.d43[m] < 0):.3f}")
say("by area bin: median basin gamma s42 | s43, and d NSE:")
for lo, hi in [(0, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, 1e7)]:
    m = (V.DRAIN_SQKM >= lo) & (V.DRAIN_SQKM < hi)
    say(f"   {lo:>6}-{hi:<9} {V.g42[m].median():.3f} | {V.g43[m].median():.3f}   {V.d43[m].median():+.4f}")
# external check: the constant-gamma sweep (seed 42, n+p+q learned), population median NSE vs gamma
sweep = {0.0: "2026-09-12T03-53-34Z-train-and-test", 0.1: "2026-09-12T20-38-08Z-train-and-test",
         0.183: "2026-09-12T20-36-00Z-train-and-test", 0.35: "2026-09-12T06-06-19Z-train-and-test"}
ids, t, P, O = C.load_preds(C.S42)
say("constant-gamma sweep (seed 42, n+p+q learned): gamma -> population median NSE / KGE")
for g, r in sweep.items():
    i, tt, Pr, _ = C.load_preds(r)
    M = C.metrics(C.align(i, Pr, ids), O)
    say(f"   {g:5.3f}: {np.nanmedian(M.nse):.4f} / {np.nanmedian(M.kge):.4f}")
V.to_csv(C.HERE / "variance_by_gauge.csv")
open(C.HERE / "variance_gamma.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
