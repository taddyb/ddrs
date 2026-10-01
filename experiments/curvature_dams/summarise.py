"""Phase 1 summaries: per-parameter identifiability table, routing comparison, by DOR and purpose, pool plateau.
Writes summary_tables.txt and summary.json."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LN10 = np.log(10.0)
ROUTING = "/home/tbindas/projects/ddrs/.ddrs/experiments/landscape-p21-all-5yr/merged/p21-conus/summary.csv"

df = pd.read_csv(os.path.join(HERE, "per_gauge_slices.csv"), dtype={"STAID": str})
df["key"] = df.law + ":" + df.param
LOGP = {"T0", "z", "kc"}
out = {}
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def q(x, p=(25, 50, 75)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, p)] if len(x) else [np.nan] * len(p)


def fmt(v, d=3):
    return "[" + ", ".join(f"{x:.{d}g}" for x in v) + "]"


# band widths in natural units: decades for T0/z/kc, s for amp, logit for phi
for d in ("0.02", "0.005"):
    for w in ("tr", "te"):
        c = f"band_{w}_{d}"
        df[c + "_u"] = np.where(df.param.isin(LOGP), df[c] / LN10, df[c])
df["shift_u"] = np.where(df.param.isin(LOGP), (df.x_opt - df.x_teopt) / LN10, df.x_opt - df.x_teopt)
df["te_cost"] = df.teNSE_max - df.teNSE_at_trainopt
df["gain_tr"] = df.trNSE_opt - df.trNSE_init
df["gain_te"] = df.teNSE_at_trainopt - df.teNSE_init
# quadratic-approximation full band widths from the curvature (loss ~ NSE units), for the routing comparison
for d in (0.02, 0.005):
    hw = np.sqrt(2 * d / np.maximum(df.H_q, 1e-12))
    df[f"qband_{d}"] = np.where(df.param.isin(LOGP), 2 * hw / LN10, 2 * hw)
# transfer: train-optimal point outside the test-optimum's 0.005 band half-width
df["nontransfer"] = np.abs(df.shift_u) > 0.5 * df["band_te_0.005_u"]
df["pg_ratio"] = np.abs(df.pg_init) / np.abs(df.pg_mid)
df["flat"] = df.f2_rel < 0.01
# split-half window stability: shift of the optimum between WY1983-1989 and WY1990-1995 (natural units)
df["half_shift_u"] = np.where(df.param.isin(LOGP), (df.x_h1 - df.x_h2) / LN10, df.x_h1 - df.x_h2)
df["half_unstable"] = np.abs(df.half_shift_u) > 0.5 * df["band_tr_0.005_u"]

ORDER = ["L2:T0", "L4:T0", "L4:amp", "FA:T0", "FA:z", "FA:kc", "FA:phi"]
UNIT = {"T0": "decades", "z": "decades", "kc": "decades", "amp": "s (x fitted coefs)", "phi": "logit"}

P("=" * 110)
P("PER-PARAMETER LANDSCAPE MEASURES, training loss = nse-batch on WY1983-1995, test = WY1996-2010")
P("medians [IQR] across gauges; curvature H = d2L/dx2 at the slice optimum, x = ln T0 / ln z / ln kc / logit phi / ln s")
P("=" * 110)
tab = []
for k in ORDER:
    s = df[df.key == k]
    name = k.split(":")[1]
    r = dict(key=k, n=len(s))
    r["H_q"] = q(s.H_q)
    r["H_ln"] = q(s.H_ln)
    r["f2_rel"] = q(s.f2_rel)
    r["flat_share"] = float(s.flat.mean())
    for d in ("0.02", "0.005"):
        r[f"band_tr_{d}"] = q(s[f"band_tr_{d}_u"])
        r[f"band_tr_{d}_open"] = float(s[f"band_tr_{d}_open"].mean())
        r[f"init_in_tr_{d}"] = float(s[f"init_in_tr_{d}"].mean())
        r[f"init_in_te_{d}"] = float(s[f"init_in_te_{d}"].mean())
        r[f"band_te_{d}"] = q(s[f"band_te_{d}_u"])
    r["edge_opt"] = float(s.edge_opt.mean())
    r["multimodal_002"] = float((s.nmin_002 >= 2).mean())
    r["multimodal_0005"] = float((s.nmin_0005 >= 2).mean())
    r["g_ratio_fd"] = q(s.g_ratio)
    r["g_init_toward_fd"] = float(s.g_init_toward.mean())
    if "pg_init" in s and s.pg_init.notna().any():
        r["pg_ratio"] = q(s.pg_ratio)
        r["pg_init_abs"] = q(np.abs(s.pg_init))
        r["pg_mid_abs"] = q(np.abs(s.pg_mid))
        r["pg_init_toward"] = float((np.sign(-s.pg_init) == np.sign(s.x_opt - s.x_init)).mean())
        r["pg_init_zero"] = float((np.abs(s.pg_init) < 1e-7).mean())
    r["shift_u"] = q(np.abs(s.shift_u))
    r["te_cost"] = q(s.te_cost)
    r["nontransfer"] = float(s.nontransfer.mean())
    r["gain_tr"] = q(s.gain_tr)
    r["gain_te"] = q(s.gain_te)
    r["half_shift"] = q(np.abs(s.half_shift_u))
    r["half_unstable"] = float(s.half_unstable.mean())
    r["half_cross"] = q(s.h_cross)
    if k == "FA:phi":
        r["phi_band_0.02"] = q(s["band_tr_0.02_phi"])
        r["phi_band_0.005"] = q(s["band_tr_0.005_phi"])
        r["phi_band_0.005_lo"] = q(s["band_tr_0.005_phi_lo"])
        r["phi_opt"] = q(s.v_opt)
        r["phi1_test_cost"] = q(s.teNSE_at_trainopt - s.teNSE_phi1)
    r["qband_0.02"] = q(s["qband_0.02"])
    tab.append(r)
    P(f"\n{k}  (n = {len(s)}; band unit: {UNIT[name]})")
    P(f"  curvature at optimum H            {fmt(r['H_q'])}   (ln-s units for amp: {fmt(r['H_ln'])})")
    P(f"  factor-2 move dL/L*               {fmt(r['f2_rel'])}   flat (< 1 %): {r['flat_share']:.0%}")
    P(f"  0.02 NSE band (train)             {fmt(r['band_tr_0.02'])}  open-ended {r['band_tr_0.02_open']:.0%};"
      f" init inside: train {r['init_in_tr_0.02']:.0%}, test {r['init_in_te_0.02']:.0%}")
    P(f"  0.005 NSE band (train)            {fmt(r['band_tr_0.005'])}  open-ended {r['band_tr_0.005_open']:.0%};"
      f" init inside: train {r['init_in_tr_0.005']:.0%}, test {r['init_in_te_0.005']:.0%}")
    P(f"  0.005 NSE band (test)             {fmt(r['band_te_0.005'])}")
    P(f"  optimum on grid edge {r['edge_opt']:.0%}; multimodal (prominence 0.002 / 0.0005): "
      f"{r['multimodal_002']:.0%} / {r['multimodal_0005']:.0%}")
    P(f"  |g(init)| / |g(halfway)| (FD h=0.05) {fmt(r['g_ratio_fd'])}; init gradient points to optimum "
      f"{r['g_init_toward_fd']:.0%}")
    if "pg_ratio" in r:
        P(f"  pathwise |g(init)| / |g(halfway)| {fmt(r['pg_ratio'])}; |g(init)| {fmt(r['pg_init_abs'])}; "
          f"|g(mid)| {fmt(r['pg_mid_abs'])}; toward {r['pg_init_toward']:.0%}; exactly zero {r['pg_init_zero']:.0%}")
    P(f"  train->test: |argmin shift| {fmt(r['shift_u'])}; test NSE lost at train-optimum {fmt(r['te_cost'])};"
      f" shift > half test 0.005 band: {r['nontransfer']:.0%}")
    P(f"  NSE gain init -> train-optimum: train {fmt(r['gain_tr'])}, test {fmt(r['gain_te'])}")
    P(f"  split-half training windows (WY83-89 vs WY90-95): |optimum shift| {fmt(r['half_shift'])}; shift > half the"
      f" 0.005 band {r['half_unstable']:.0%}; loss paid at the other half's optimum {fmt(r['half_cross'])}")
    if k == "FA:phi":
        P(f"  phi units: optimum {fmt(r['phi_opt'])}; 0.02 band {fmt(r['phi_band_0.02'])}, 0.005 band "
          f"{fmt(r['phi_band_0.005'])} (lower end {fmt(r['phi_band_0.005_lo'])}); test NSE given up by fixing phi = 1:"
          f" {fmt(r['phi1_test_cost'])}")
out["params"] = tab

# ------------------------------------------------------------------ by DOR and purpose
P("\n" + "=" * 110)
P("BY DOR AND PURPOSE: median H (curvature), median 0.02 train band, share flat, share non-transferable")
df["dor_bin"] = pd.cut(df.nid_dor, [0, 0.1, 0.5, 1.0, 1e9], labels=["<0.1", "0.1-0.5", "0.5-1", ">1"])
df["fc"] = np.where(df.dam_purpose == "Flood Risk Reduction", "flood", "other")
byg = []
for k in ["L4:T0", "L4:amp", "FA:z", "FA:kc", "FA:phi"]:
    s = df[df.key == k]
    for col in ("dor_bin", "fc"):
        for lev, t in s.groupby(col, observed=True):
            if len(t) < 5:
                continue
            row = dict(key=k, by=col, level=str(lev), n=len(t), H=float(t.H_q.median()),
                       band002=float(t["band_tr_0.02_u"].median()), flat=float(t.flat.mean()),
                       nontransfer=float(t.nontransfer.mean()), gain_te=float(t.gain_te.median()))
            byg.append(row)
            P(f"  {k:7s} {col:7s} {str(lev):8s} n={len(t):3d}  H={row['H']:.4g}  band0.02={row['band002']:.3g}  "
              f"flat={row['flat']:.0%}  nontransfer={row['nontransfer']:.0%}  test gain={row['gain_te']:+.4f}")
out["by_group"] = byg

# ------------------------------------------------------------------ routing comparison (5-yr census, well fit)
P("\n" + "=" * 110)
P("ROUTING PARAMETERS ON THE SAME SCALE (landscape-p21-all-5yr census, WY1996-2000, NSE > 0.3 at trained point)")
r = pd.read_csv(ROUTING, dtype={"staid": str})
r = r[r.nse0 > 0.3]
v1n, v1q = r.v1_n, r.v1_q
Hnn = r.lambda1 * v1n ** 2 + r.lambda2 * v1q ** 2
Hqq = r.lambda1 * v1q ** 2 + r.lambda2 * v1n ** 2
rt = []
for nm, H in (("routing n", Hnn), ("routing q", Hqq)):
    f2 = 0.5 * H * np.log(2) ** 2 / r.loss_star
    b02 = 2 * np.sqrt(2 * 0.02 / np.maximum(H, 1e-12)) / LN10
    b005 = 2 * np.sqrt(2 * 0.005 / np.maximum(H, 1e-12)) / LN10
    row = dict(key=nm, n=int(len(r)), H=q(H), f2_rel=q(f2), flat=float((f2 < 0.01).mean()), qband_002=q(b02),
               qband_0005=q(b005), H_le0=float((H <= 0).mean()))
    rt.append(row)
    P(f"  {nm}: n={len(r)}  H {fmt(row['H'])}  factor-2 dL/L* {fmt(row['f2_rel'])}  flat {row['flat']:.0%}  "
      f"quadratic 0.02 band {fmt(row['qband_002'])} decades, 0.005 band {fmt(row['qband_0005'])}; H <= 0 at "
      f"{row['H_le0']:.0%}")
out["routing"] = rt
P("  dam parameters, quadratic-approximation 0.02 band (decades or s/logit units) for the same comparison:")
for k in ORDER:
    s = df[df.key == k]
    P(f"    {k:7s} {fmt(q(s['qband_0.02']))}   H>0 at optimum: {(s.H_q > 0).mean():.0%}")

# ------------------------------------------------------------------ pool plateau: regime + joint init gradient
P("\n" + "=" * 110)
P("FLOOD POOL AT INIT: which constraint binds on training flood days (69 on-reach flood-control dams)")
pr = pd.read_csv(os.path.join(HERE, "pool_regime.csv"), dtype={"STAID": str})
pr["zr"] = pr.z.round(3)
for at, t in pr.groupby("at"):
    for z, u in t.groupby("zr"):
        if z not in (0.05, 0.2, 1.0, 5.0, 20.0):
            continue
        P(f"  {at:14s} z={z:6.2f} d: capture capacity-limited on {np.median(u.pool_full_share):.0%} of flood days "
          f"(median; mean {u.pool_full_share.mean():.0%}), evacuation pool-limited {np.median(u.evac_pool_share):.0%};"
          f" flood days/gauge {np.median(u.flood_days):.0f}")
jt = pd.read_csv(os.path.join(HERE, "joint_init.csv"), dtype={"STAID": str})
P("\nJOINT engine-init gradient, raw engine coordinates (delta = ln T0/T0_head, r_kc, r_phi, r_z), pathwise:")
jo = {}
for pt, t in jt.groupby("point"):
    row = {}
    for c in ("pg_T0", "pg_kc", "pg_phi", "pg_z"):
        row[c] = q(np.abs(t[c]))
        row[c + "_grow"] = float((t[c] < 0).mean())
    jo[pt] = row
    P(f"  {pt:5s} |dL/d delta| {fmt(row['pg_T0'])}  |dL/dr_kc| {fmt(row['pg_kc'])}  |dL/dr_phi| {fmt(row['pg_phi'])}"
      f"  |dL/dr_z| {fmt(row['pg_z'])};  share wanting larger z {row['pg_z_grow']:.0%}, larger phi "
      f"{row['pg_phi_grow']:.0%}, larger kc {row['pg_kc_grow']:.0%}")
ji = jt[jt.point == "init"].set_index("STAID")
jm = jt[jt.point == "mid"].set_index("STAID")
for c in ("pg_z", "pg_kc", "pg_phi", "pg_T0"):
    rat = np.abs(ji[c]) / np.abs(jm[c])
    P(f"  joint |g_init|/|g_mid| {c}: {fmt(q(rat))}")
out["joint"] = jo
# L2 break-even: per-dam L2 lambda sum r^2 vs data curvature per step: H_data_engine = H_gauge / G, G ~ 250
G = 250.0
lam = 1e-3
P(f"\nPER-DAM L2 (lambda = 1e-3, sum over dams) vs data curvature at init, engine step with G = {G:.0f} gauges:")
P("  L2 curvature per raw unit^2 = 2 lambda = 0.002; data curvature per raw unit^2 = H_gauge(raw) / G")
for c in ("H_z", "H_kc", "H_phi", "H_T0"):
    Hg = ji[c]
    P(f"  {c}: gauge curvature at init (raw units, FD h 0.02) {fmt(q(Hg))}; data/L2 ratio {fmt(q(Hg / G / (2 * lam)))};"
      f" L2 dominates at {(Hg / G < 2 * lam).mean():.0%}")
yg = pd.read_csv(os.path.join(HERE, "year_grad.csv"), dtype={"STAID": str})
P("\nPER-TRAINING-YEAR SIGN OF dL/d ln z (pathwise), 13 water years per gauge:")
ys = {}
for pt, t in yg.groupby("point"):
    agg = t.groupby("STAID").agg(frac_grow=("pg_z", lambda v: float((v < 0).mean())),
                                 tot=("pg_z", "sum"), sd=("pg_z", "std"), mu=("pg_z", "mean"))
    agg["snr"] = np.abs(agg.mu) / agg.sd * np.sqrt(13)
    ys[pt] = dict(frac_grow=q(agg.frac_grow), net_grow=float((agg.tot < 0).mean()), snr=q(agg.snr))
    P(f"  {pt:13s} share of years wanting a larger pool {fmt(ys[pt]['frac_grow'])}; gauges whose 13-yr sum wants "
      f"larger {ys[pt]['net_grow']:.0%}; |mean|/sd*sqrt(13) {fmt(ys[pt]['snr'])}")
out["year_sign"] = ys

# ------------------------------------------------------------------ 2-D planes
P("\n" + "=" * 110)
P("2-D PLANES (69 flood-control dams): quadratic fit at the grid optimum, ln units")
pl = pd.read_csv(os.path.join(HERE, "planes.csv"), dtype={"STAID": str})
for w, t in pl.groupby("plane"):
    ok = t[~t.edge]
    ani = ok.lam_max / np.maximum(np.abs(ok.lam_min), 1e-12)
    P(f"  {w}: interior optimum {len(ok)}/{len(t)}; H11 {fmt(q(ok.H11))} H22 {fmt(q(ok.H22))}; coupling "
      f"-H12/sqrt(H11 H22) {fmt(q(ok["corr"]))}; anisotropy {fmt(q(ani))}; indefinite {(ok.lam_min <= 0).mean():.0%};"
      f" test NSE lost at train-optimum {fmt(q(t.te_max - t.te_at_tropt))}")
    P(f"     train-opt vs test-opt axis-a shift (decades) {fmt(q(np.abs(np.log10(t.a_opt / t.a_teopt))))}, "
      f"axis-b {fmt(q(np.abs(np.log10(t.b_opt / t.b_teopt))))}")

json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
open(os.path.join(HERE, "summary_tables.txt"), "w").write("\n".join(lines) + "\n")
df.to_csv(os.path.join(HERE, "per_gauge_slices_derived.csv"), index=False)
