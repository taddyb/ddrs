"""Lever 2: routing-head convergence and variance, from existing runs only (test years WY1996-2010, 2,365 gauges).

A. Loss curves from run.log: per-epoch gauge-weighted mean micro-batch loss (nse-batch), OLS slope over epochs 41-50
   (and 31-40) with its standard error; median n per epoch.
B. Parameter drift from plot/kan_parameters_epoch{25,50}.nc (reach level) against the seed-to-seed difference.
C. Ensembles (equal-weight mean of routed predictions, no fitted weights): two seeds of the working recipe; the working
   recipe on two Q' stores; every UH-store 50-epoch nse-batch variant. Scored on test years, paired against seed 42.
D. The gauges just below the population median (within 0.03): what the seed change does there and who they are.
Writes variance.txt and variance.json, per-gauge table variance_by_gauge.csv.
"""
import json
import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import common as C

out, res = [], {}
say = out.append

# ---------------- A. loss curves ----------------
RX_EP = re.compile(r"\] epoch (\d+) lr=([0-9.e-]+)")
RX_MI = re.compile(r"micro \d+/\d+ gauges=(\d+) loss=([0-9.eE+-]+) n=\d+ median_n=([0-9.eE+-]+)")


def curve(run):
    ep, rows = 0, []
    for line in open(C.RUNS / run / "run.log"):
        m = RX_EP.search(line)
        if m:
            ep = int(m.group(1))
            continue
        m = RX_MI.search(line)
        if m and ep:
            rows.append((ep, int(m.group(1)), float(m.group(2)), float(m.group(3))))
    d = pd.DataFrame(rows, columns=["epoch", "g", "loss", "median_n"])
    e = d.groupby("epoch").apply(lambda x: pd.Series(dict(loss=np.average(x.loss, weights=x.g), n_micro=len(x),
                                                          median_n=x.median_n.iloc[-1],
                                                          loss_med=x.loss.median())))
    return e


def slope(e, lo, hi, col="loss"):
    x = e.loc[lo:hi]
    r = stats.linregress(x.index.values.astype(float), x[col].values)
    return r.slope, r.stderr, x[col].mean()


say("== A. training loss (nse-batch, gauge-weighted mean over each epoch's micro-batches; random 90-day windows)")
CURVES = {"seed42 n0+gamma (07-29-47Z)": C.S42, "seed43 n0+gamma (10-31-30Z)": C.S43,
          "gamma=0, n+p+q (09-12T03-53-34Z)": "2026-09-12T03-53-34Z-train-and-test",
          "n only (09-12T23-39-06Z)": "2026-09-12T23-39-06Z-train-and-test"}
for name, run in CURVES.items():
    e = curve(run)
    s1, se1, m1 = slope(e, 41, 50)
    s2, se2, m2 = slope(e, 31, 40)
    s3, se3, m3 = slope(e, 11, 20)
    early = e.loss.loc[1:5].mean()
    say(f"  {name:36s} epochs {int(e.index.max())}: mean loss ep1-5 {early:.4f}, ep11-20 {m3:.4f}, ep31-40 {m2:.4f}, "
        f"ep41-50 {m1:.4f}")
    say(f"  {'':36s} slope/epoch ep11-20 {s3:+.5f}+-{se3:.5f}; ep31-40 {s2:+.5f}+-{se2:.5f}; ep41-50 {s1:+.5f}+-{se1:.5f} "
        f"(= {100 * s1 * 10 / m1:+.2f}% of the level over 10 epochs); epoch-to-epoch sd ep41-50 {e.loss.loc[41:50].std():.4f}")
    say(f"  {'':36s} median n by epoch: " + " ".join(f"e{k}:{e.median_n.get(k, np.nan):.4f}" for k in [1, 5, 10, 20, 30, 40, 45, 50]))
    res[f"curve::{name}"] = dict(slope_41_50=s1, se_41_50=se1, level_41_50=m1, slope_31_40=s2, se_31_40=se2,
                                 level_31_40=m2, slope_11_20=s3, level_11_20=m3, early=early)
    e.to_csv(C.HERE / f"loss_curve_{run[:20]}.csv")

# ---------------- A2. paired training loss on identical batches (seed-42 runs share gauge batches and windows) ----
def micro(run):
    ep, rows = 0, []
    for line in open(C.RUNS / run / "run.log"):
        m = RX_EP.search(line)
        if m:
            ep = int(m.group(1))
            continue
        m = RX_MI.search(line)
        if m and ep:
            nn = re.search(r" n=(\d+) ", line).group(1)
            rows.append((ep, int(m.group(1)), int(nn), float(m.group(2))))
    return pd.DataFrame(rows, columns=["epoch", "g", "nobs", "loss"])


say("\n== A2. paired training loss on identical micro-batches (seed-42 runs draw the same gauges and windows)")
ref = micro(C.S42)
for name, run in [("gamma=0, n+p+q (09-12T03-53-34Z)", "2026-09-12T03-53-34Z-train-and-test"),
                  ("gamma=0.1, n+p+q (09-12T20-38-08Z)", "2026-09-12T20-38-08Z-train-and-test"),
                  ("learned gamma, n+p+q (09-12T16-30-14Z)", "2026-09-12T16-30-14Z-train-and-test"),
                  ("n only (09-12T23-39-06Z)", "2026-09-12T23-39-06Z-train-and-test")]:
    o = micro(run)
    same = len(o) == len(ref) and (o.nobs.values == ref.nobs.values).all() and (o.g.values == ref.g.values).all()
    if not same:
        say(f"  {name}: batches differ, skipped")
        continue
    d = o.loss.values - ref.loss.values
    rel = d / ref.loss.values
    for lo, hi in [(1, 10), (11, 20), (21, 40), (41, 50)]:
        b = (ref.epoch >= lo) & (ref.epoch <= hi)
        say(f"  {name:40s} ep{lo:>2}-{hi:<2} mean(loss - s42) {d[b].mean():+.4f} +- {d[b].std() / np.sqrt(b.sum()):.4f}, "
            f"median rel {np.median(rel[b]):+.3f}, share lower {np.mean(d[b] < 0):.3f} (n={int(b.sum())})")

# ---------------- B. parameter drift ----------------
say("\n== B. reach-level parameter drift (KAN forward over all 346,321 CONUS reaches)")


def kp(run, ep=None):
    f = f"kan_parameters_epoch{ep}.nc" if ep else "kan_parameters.nc"
    ds = xr.open_dataset(C.RUNS / run / "plot" / f)
    return ds.n.values.astype(float), ds.gamma.values.astype(float), ds.COMID.values


n42_10, g42_10, c42 = kp(C.S42, 10)
n42_25, g42_25, _ = kp(C.S42, 25)
n42_50, g42_50, _ = kp(C.S42, 50)
n43_50, g43_50, c43 = kp(C.S43)  # seed 43 kept only its final field
assert (c42 == c43).all()


def mad_log(a, b):
    return float(np.median(np.abs(np.log(a / b))))


say(f"  seed42 median n e10 {np.median(n42_10):.4f} e25 {np.median(n42_25):.4f} e50 {np.median(n42_50):.4f}; "
    f"gamma e10 {np.median(g42_10):.4f} e25 {np.median(g42_25):.4f} e50 {np.median(g42_50):.4f}")
say(f"  seed43 median n e50 {np.median(n43_50):.4f}; gamma e50 {np.median(g43_50):.4f} (only the final field was kept)")
say(f"  median |ln n ratio|: seed42 e10->e25 {mad_log(n42_25, n42_10):.3f}, e25->e50 {mad_log(n42_50, n42_25):.3f}; "
    f"seed42 vs seed43 at e50 {mad_log(n42_50, n43_50):.3f}")
say(f"  median |gamma change|: seed42 e10->e25 {np.median(np.abs(g42_25 - g42_10)):.4f}, e25->e50 "
    f"{np.median(np.abs(g42_50 - g42_25)):.4f}; seed42 vs seed43 at e50 {np.median(np.abs(g42_50 - g43_50)):.4f}")
say(f"  Spearman(n) seed42 vs seed43 at e50 {stats.spearmanr(n42_50, n43_50).correlation:.3f}; "
    f"Spearman(gamma) {stats.spearmanr(g42_50, g43_50).correlation:.3f}")

# ---------------- C. ensembles ----------------
say("\n== C. ensembles on the test years (equal-weight mean of routed predictions), paired against seed 42 (0.7391)")
ids, t, P42, O = C.load_preds(C.S42)
M42 = C.metrics(P42, O)
base_n, base_k = M42.nse.values, M42.kge.values
BASE = float(np.nanmedian(base_n))
preds = {"s42": P42}


def get(run):
    if run not in preds:
        i, tt, P, _ = C.load_preds(run)
        assert len(tt) == len(t) and (tt == t).all()
        preds[run] = C.align(i, P, ids)
    return preds[run]


UH = ["2026-09-10T21-21-48Z-conus-train-and-test", "2026-09-11T06-38-49Z-conus-train-and-test",
      "2026-09-12T03-53-34Z-train-and-test", "2026-09-12T06-06-19Z-train-and-test", "2026-09-12T16-30-14Z-train-and-test",
      "2026-09-12T20-36-00Z-train-and-test", "2026-09-12T20-38-08Z-train-and-test", "2026-09-12T23-39-06Z-train-and-test",
      C.S42, C.S43]
singles = {}
for r in UH + ["2026-09-13T17-22-30Z-train-and-test"]:
    M = C.metrics(get(r) if r != C.S42 else P42, O)
    singles[r] = M
    s = C.summarize(f"single {r[:20]}", M.nse, base_n)
    sk = C.summarize(f"single {r[:20]}", M.kge, base_k, "KGE")
    say(C.fmt(s))
    say(C.fmt(sk))
    res[f"single::{r}"] = dict(nse=s, kge=sk)

ENS = {
    "two seeds of the working recipe (s42+s43)": [C.S42, C.S43],
    "working recipe on two Q' stores (UH + dhbv2-dist aorc2f)": [C.S42, "2026-09-13T17-22-30Z-train-and-test"],
    "two seeds + dhbv2-dist aorc2f (3 members)": [C.S42, C.S43, "2026-09-13T17-22-30Z-train-and-test"],
    "all 10 UH-store 50-epoch nse-batch variants": UH,
    "UH variants without the three constant-gamma arms (7)": [r for r in UH if r not in (
        "2026-09-12T06-06-19Z-train-and-test", "2026-09-12T20-36-00Z-train-and-test", "2026-09-12T20-38-08Z-train-and-test")],
}
say("")
for name, members in ENS.items():
    Pm = np.mean([get(r) if r != C.S42 else P42 for r in members], axis=0)
    M = C.metrics(Pm, O)
    s = C.summarize(f"ens {name}", M.nse, base_n)
    sk = C.summarize(f"ens {name}", M.kge, base_k, "KGE")
    say(C.fmt(s))
    say(C.fmt(sk))
    mean_single = np.mean([np.nanmedian(singles[r].nse) for r in members])
    say(f"      mean of the members' own medians {mean_single:.4f}; ensemble minus that {s['median_new'] - mean_single:+.4f}")
    res[f"ens::{name}"] = dict(nse=s, kge=sk, mean_member_median=mean_single)

# recipe effect vs seed noise, paired per gauge
d_seed = singles[C.S43].nse.values - base_n
d_rec = singles["2026-09-12T03-53-34Z-train-and-test"].nse.values - base_n
say("\n  per-gauge |dNSE| distribution: seed43-seed42 median |d| "
    f"{np.nanmedian(np.abs(d_seed)):.4f}, p90 {np.nanpercentile(np.abs(d_seed), 90):.4f}; "
    f"gamma0/n+p+q minus s42 median |d| {np.nanmedian(np.abs(d_rec)):.4f}, p90 {np.nanpercentile(np.abs(d_rec), 90):.4f}")
say(f"  sign test gamma0/n+p+q vs s42: up {np.mean(d_rec > 0):.3f}; seed43 vs s42: up {np.mean(d_seed > 0):.3f}")

# ---------------- D. near-median gauges ----------------
say("\n== D. gauges within 0.03 below the seed-42 population median")
G = pd.read_csv(C.HERE / "gauges_table.csv", dtype={"STAID": str, "HUC02": str}).set_index("STAID").reindex(ids)
near = (base_n < BASE) & (base_n >= BASE - 0.03)
M43 = singles[C.S43]
dn = M43.nse.values - base_n
ens2 = C.metrics(np.mean([P42, get(C.S43)], axis=0), O).nse.values
T = pd.DataFrame(dict(nse42=base_n, nse43=M43.nse.values, d43=dn, nse_ens2=ens2, kge42=base_k,
                      kge43=M43.kge.values, beta42=M42.beta.values, alpha42=M42.alpha.values, r42=M42.r.values),
                 index=ids).join(G)
T["near"] = near
T.to_csv(C.HERE / "variance_by_gauge.csv")
say(f"  n near = {int(near.sum())}; in them seed43-seed42 median {np.nanmedian(dn[near]):+.4f}, share down "
    f"{np.mean(dn[near] < 0):.3f}; |d| median {np.nanmedian(np.abs(dn[near])):.4f} (population |d| {np.nanmedian(np.abs(dn)):.4f})")
say(f"  crossings at the median: s42-below-and-near gauges that seed43 lifts above the s42 median: "
    f"{int((near & (M43.nse.values >= BASE)).sum())}; s42-above gauges seed43 drops below: "
    f"{int(((base_n >= BASE) & (M43.nse.values < BASE)).sum())}; s42-below that s43 lifts above: "
    f"{int(((base_n < BASE) & (M43.nse.values >= BASE)).sum())}")
cols = ["DRAIN_SQKM", "up_aridity", "up_snow_fraction", "tt_days_1ms", "longest_path_km", "n_reach", "nid_dor",
        "beta42", "alpha42", "r42"]
lose = near & (dn < np.nanpercentile(dn[near], 25))
say("  medians: population | near-median | near-median losing most (bottom quartile of seed43-seed42 there, n="
    f"{int(lose.sum())})")
for c in cols:
    say(f"    {c:18s} {T[c].median():10.3f} | {T.loc[near, c].median():10.3f} | {T.loc[lose, c].median():10.3f}")
say(f"  near-median share Ref class {np.mean(T.loc[near, 'CLASS'] == 'Ref'):.3f} vs population "
    f"{np.mean(T.CLASS == 'Ref'):.3f}; share with a NID dam >= 10 MCM {np.mean(T.loc[near, 'n_nid_ge10mcm'] > 0):.3f} "
    f"vs {np.mean(T.n_nid_ge10mcm > 0):.3f}")
say("  near-median HUC02 counts (top 6): " + ", ".join(f"{k}:{v}" for k, v in T.loc[near, 'HUC02'].value_counts().head(6).items()))
say("  near-median losing-most HUC02 counts: " + ", ".join(f"{k}:{v}" for k, v in T.loc[lose, 'HUC02'].value_counts().head(6).items()))
# what drives seed sensitivity, population-wide
say("  Spearman of |seed43-seed42 NSE| with attributes (population):")
for c in ["DRAIN_SQKM", "up_aridity", "up_snow_fraction", "tt_days_1ms", "n_reach", "nid_dor", "nse42"]:
    ok = np.isfinite(T[c]) & np.isfinite(dn)
    rho = stats.spearmanr(np.abs(dn[ok]), T[c][ok]).correlation
    say(f"    {c:18s} {rho:+.3f}")
say("  Spearman of signed (seed43-seed42) NSE with attributes (population):")
for c in ["DRAIN_SQKM", "up_aridity", "up_snow_fraction", "tt_days_1ms"]:
    ok = np.isfinite(T[c]) & np.isfinite(dn)
    say(f"    {c:18s} {stats.spearmanr(dn[ok], T[c][ok]).correlation:+.3f}")
# where the median-seed difference comes from: by area bin
say("  seed43-seed42 by drainage-area bin (median d, share down, n):")
for lo, hi in [(0, 300), (300, 1000), (1000, 3000), (3000, 10000), (10000, 1e7)]:
    b = (T.DRAIN_SQKM >= lo) & (T.DRAIN_SQKM < hi)
    say(f"    {lo:>6}-{hi:<8} {np.nanmedian(dn[b]):+.4f}  {np.mean(dn[b] < 0):.3f}  {int(b.sum())}")

open(C.HERE / "variance.txt", "w").write("\n".join(out) + "\n")
json.dump(res, open(C.HERE / "variance.json", "w"), indent=1, default=float)
print("\n".join(out))
