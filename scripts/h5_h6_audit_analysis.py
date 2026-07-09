"""Audit re-analysis of the H5/H6 registered raw CSVs (2026-07-09).

Reads ONLY the raw per-window / per-gauge CSVs produced by the registered
`probe_zeta_gradient --mode eval-loss` and `--mode landscape` runs — no Rust
re-execution. Produces every number cited in
docs/2026-07-09-h5-h6-equifinality-v2-findings.md:

  1. H5 paired per-window statistics (compositions share identical windows,
     so paired differences are the correct test — not mean vs window std).
  2. H5 registered split-half noise floors (even/odd + first/second windows).
  3. H5 per-gauge DA-stratified deltas and gauge-concentration of the penalty.
  4. H6 surface geometry: sublevel sets under both definitions, PCA aspect,
     profiles, valley-floor trace alpha*(beta), quadratic curvature ratio,
     split-half minima displacement, cross-forcing minima cross-evaluation.

Run: cd ~/projects/ddrs/ddrs-py && uv run python ../scripts/h5_h6_audit_analysis.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "/home/tbindas/projects/ddrs/output/equif"
GAGES_CSV = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"

RUNS = {
    "R1<-R3 (primary)": "r1_under_r1_donor_r3",
    "R3<-R1 (primary)": "r3_under_r3_donor_r1",
    "R1<-R2 (control)": "r1_under_r1_donor_r2",
    "R2<-R1 (control)": "r2_under_r2_donor_r1",
}

# ---------------- H5: paired per-window analysis ----------------
print("=" * 100)
print("H5 — PAIRED per-window analysis (identical windows across compositions)")
print("=" * 100)
for label, stem in RUNS.items():
    df = pd.read_csv(f"{BASE}/h5/registered/{stem}.csv")
    piv = df.pivot(index="window", columns="composition", values="mean_loss")
    d_n = piv["n-swap"] - piv["own"]
    d_g = piv["geo-swap"] - piv["own"]
    d_f = piv["full-swap"] - piv["own"]

    def stats(d):
        se = d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), d.std(ddof=1), se, d.mean() / se, (d > 0).mean()

    mn, sdn, sen, tn, fpos_n = stats(d_n)
    mg, sdg, seg, tg, fpos_g = stats(d_g)
    mf, _, _, tf, _ = stats(d_f)

    # registered split-half noise floor: penalties on disjoint window halves
    even = piv.index % 2 == 0
    floor_n = abs(d_n[even].mean() - d_n[~even].mean())
    floor_g = abs(d_g[even].mean() - d_g[~even].mean())
    h1 = piv.index < piv.index.max() / 2
    floor_n2 = abs(d_n[h1].mean() - d_n[~h1].mean())

    print(f"\n--- {label} ({len(piv)} windows) ---")
    print(f"  P_n   = {mn:+.4f}  (paired sd {sdn:.4f}, SE {sen:.4f}, t = {tn:+.2f}, frac windows >0: {fpos_n:.2f})")
    print(f"  P_geo = {mg:+.4f}  (paired sd {sdg:.4f}, SE {seg:.4f}, t = {tg:+.2f}, frac windows >0: {fpos_g:.2f})")
    print(f"  P_full= {mf:+.4f}  (t = {tf:+.2f});  additivity P_n+P_geo = {mn+mg:+.4f}")
    print(f"  split-half floors: even/odd P_n {floor_n:.4f}  P_geo {floor_g:.4f};  first/second P_n {floor_n2:.4f}")
    print(f"  registered bar: P_n >= 3x floor(even/odd)? {abs(mn) >= 3*floor_n}   ratio = {abs(mn)/floor_n:.1f}x")
    print(f"  f_n = {mn/(mn+mg):.3f}")

# ---------------- H5: per-gauge DA-stratified + concentration ----------------
print("\n" + "=" * 100)
print("H5 — per-gauge paired deltas, DA-stratified, and penalty concentration")
print("=" * 100)
gages = pd.read_csv(GAGES_CSV, dtype={"STAID": str})
gages["STAID"] = gages["STAID"].str.zfill(8)
da = gages.set_index("STAID")["DRAIN_SQKM"]

for label, stem in [("R1<-R3", "r1_under_r1_donor_r3"),
                    ("R3<-R1", "r3_under_r3_donor_r1"),
                    ("R1<-R2 ctrl", "r1_under_r1_donor_r2")]:
    pg = pd.read_csv(f"{BASE}/h5/registered/{stem}_per_gauge.csv", dtype={"staid": str})
    pg["staid"] = pg["staid"].str.zfill(8)
    piv = pg.pivot_table(index=["staid", "window"], columns="composition", values="gauge_loss").dropna()
    piv["d_n"] = piv["n-swap"] - piv["own"]
    piv["d_g"] = piv["geo-swap"] - piv["own"]

    per_gauge = piv.groupby("staid")[["d_n", "d_g"]].agg(["mean", "count"])
    contrib = per_gauge[("d_n", "mean")] * per_gauge[("d_n", "count")]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
    top10_share = top.sum() / contrib.sum()

    print(f"\n--- {label}: pooled mean d_n = {piv['d_n'].mean():+.4f} over {len(piv)} gauge-window pairs, {len(per_gauge)} gauges ---")
    print(f"  median per-gauge d_n = {per_gauge[('d_n','mean')].median():+.4f}, frac gauges hurt: {(per_gauge[('d_n','mean')]>0).mean():.2f}")
    print(f"  top 10 gauges carry {top10_share:.0%} of the total summed n-swap penalty:")
    print("  " + ", ".join(f"{s} ({v:+.1f})" for s, v in top.items()))

    pgda = per_gauge[("d_n", "mean")].rename("d_n").to_frame().join(da, how="inner")
    pgda["log10da"] = np.log10(pgda["DRAIN_SQKM"])
    rho, _ = spearmanr(pgda["log10da"], pgda["d_n"])
    quint = pgda.groupby(pd.qcut(pgda["log10da"], 5, labels=False))["d_n"].median()
    print(f"  {len(pgda)} gauges matched to DA; Spearman rho(log10DA, d_n) = {rho:+.3f}")
    print(f"  DA-quintile median d_n: " + "  ".join(f"{v:+.4f}" for v in quint.values))

# ---------------- H6: surface geometry from raw ----------------
print("\n" + "=" * 100)
print("H6 — surface re-analysis from raw per-window grid")
print("=" * 100)
mins = {}
for arm in ["r1", "r3"]:
    s = pd.read_csv(f"{BASE}/h6/{arm}_surface.csv")
    full = s.groupby(["log2_alpha", "log2_beta"])["mean_loss"].mean()
    grid = full.unstack()  # rows=alpha, cols=beta
    lo, hi = full.min(), full.max()
    mins[arm] = full.idxmin()
    a_star, b_star = mins[arm]

    sub_rel = full[full <= lo * 1.05]          # registered reading: within 5% of min VALUE
    sub_rng = full[full <= lo + 0.05 * (hi - lo)]  # alternative: within 5% of RANGE

    def aspect(sub):
        pts = np.array(list(sub.index), dtype=float)
        pts -= pts.mean(0)
        ev = np.linalg.eigvalsh(np.cov(pts.T))
        return np.sqrt(ev[-1] / max(ev[0], 1e-12)), len(sub)

    ar_rel, n_rel = aspect(sub_rel)
    ar_rng, n_rng = aspect(sub_rng)

    # valley floor trace and curvature anisotropy
    floor_a = grid.idxmin(axis=0)   # alpha*(beta)
    floor_L = grid.min(axis=0)      # L*(beta)
    prof_a = grid.min(axis=1)       # L*(alpha)
    slope = np.polyfit(floor_a.index, floor_a.values, 1)[0]
    c_stiff = np.polyfit(prof_a.index, prof_a.values, 2)[0]
    c_floor = np.polyfit(floor_L.index, floor_L.values, 2)[0]

    # registered split-half noise floor for the minimum location
    halves = {}
    for name, mask in [("even", s["window"] % 2 == 0), ("odd", s["window"] % 2 == 1)]:
        halves[name] = s[mask].groupby(["log2_alpha", "log2_beta"])["mean_loss"].mean().idxmin()
    d_half = np.hypot(halves["even"][0] - halves["odd"][0], halves["even"][1] - halves["odd"][1])

    print(f"\n--- {arm.upper()} forcing ---")
    print(f"  grid: min {lo:.4f} at (a={a_star}, b={b_star}), max {hi:.4f}, range {hi-lo:.4f} ({100*(hi-lo)/lo:.1f}% of min)")
    print(f"  sublevel [<= min*1.05]:     n={n_rel:3d}/121, PCA aspect = {ar_rel:.2f}   <- registered reading (saturates)")
    print(f"  sublevel [<= min+5% range]: n={n_rng:3d}/121, PCA aspect = {ar_rng:.2f}")
    print(f"  L*(beta) total range = {floor_L.max()-floor_L.min():.4f} over 8x p-scaling (sloppy axis)")
    print(f"  valley floor alpha*(beta): {list(floor_a.values)}  OLS slope {slope:+.3f}")
    print(f"  quadratic curvature: stiff L*(alpha) c={c_stiff:.4f}, floor L*(beta) c={c_floor:.5f}, ratio {c_stiff/c_floor:.0f}x")
    print(f"  split-half minima: even {halves['even']}, odd {halves['odd']}, displacement = {d_half:.3f} log2-units")
    print(f"  L(anchor 0,0) - L(min) = {full.loc[(0.0, 0.0)] - lo:+.4f}")

disp = np.hypot(mins["r1"][0] - mins["r3"][0], mins["r1"][1] - mins["r3"][1])
print(f"\ncross-forcing minima: R1 {mins['r1']}, R3 {mins['r3']}, displacement = {disp:.4f}")
print(f"  components: d_alpha = {abs(mins['r1'][0]-mins['r3'][0]):.2f} (stiff/n axis), "
      f"d_beta = {abs(mins['r1'][1]-mins['r3'][1]):.2f} (sloppy/p axis)")

print("\ncross-evaluation of minima, paired over windows:")
for arm, own_min, other_min in [("r1", mins["r1"], mins["r3"]), ("r3", mins["r3"], mins["r1"])]:
    s = pd.read_csv(f"{BASE}/h6/{arm}_surface.csv")
    piv = s.set_index(["log2_alpha", "log2_beta", "window"]).sort_index()["mean_loss"]
    d = piv.loc[other_min] - piv.loc[own_min]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  under {arm.upper()} forcing: penalty(other arm's min) = {d.mean():+.4f}  paired SE {se:.4f}  t = {d.mean()/se:+.2f}")

print("\nbarrier endpoint deltas (consistency check vs H5 P_full, independent window plans):")
for arm in ["r1", "r3"]:
    b = pd.read_csv(f"{BASE}/h6/{arm}_barrier.csv").groupby("t")["mean_loss"].mean()
    print(f"  {arm.upper()} forcing: L(t=1) - L(t=0) = {b.iloc[-1] - b.iloc[0]:+.4f}  "
          f"(interior max {b.iloc[1:-1].max():.4f} at t={b.iloc[1:-1].idxmax()})")
