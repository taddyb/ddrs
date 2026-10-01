"""Phase 2: score the engine z-sweep replays at the 69 on-reach flood-control gauges against the fair no-pool twin
(2026-09-30T02-16-02Z: plain-bucket T0 = L2_T0_days, no pool), and put the offline analogue beside them
(offline FA(z; fitted T0, kc, phi) minus offline L2, WY1996-2010). Writes engine_z.csv and engine_summary.txt."""
import os
import re

import numpy as np
import pandas as pd
import zarr

import landscapes as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(L.AG, ".ddrs/runs")
TWIN = "2026-09-30T02-16-02Z-train-and-test"
FITTED = "2026-09-30T01-52-25Z-train-and-test"      # FA pool replay, fitted z
FA_NOPOOL = "2026-09-30T01-52-45Z-train-and-test"   # same table (FA T0), pool off


def run_of(log):
    m = re.search(r"run output → (\S+)", open(os.path.join(HERE, "engine", log)).read())
    return os.path.basename(m.group(1)) if m else None


def nse_by_gauge(run):
    z = zarr.open(os.path.join(RUNS, run, "eval/predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    P, O = z["predictions"][:], z["observations"][:]
    out = {}
    for i, s in enumerate(ids):
        if s not in L.FA69:
            continue
        m = np.isfinite(O[i]) & np.isfinite(P[i])
        o, p = O[i][m], P[i][m]
        out[s] = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    return pd.Series(out)


def boot_median(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


runs = {"twin": TWIN, "fitted": FITTED, "fa_nopool": FA_NOPOOL}
for tag, log in (("0.05", "z0p05.log"), ("1", "z1.log"), ("5", "z5.log"), ("20", "z20.log")):
    if os.path.exists(os.path.join(HERE, "engine", log)):
        r = run_of(log)
        if r and os.path.exists(os.path.join(RUNS, r, "eval/predictions.zarr")):
            runs["z" + tag] = r
E = pd.DataFrame({k: nse_by_gauge(v) for k, v in runs.items()})
# offline analogue at the same z values
rows = []
for s in L.FA69:
    g = L.ctx(s)
    f = L.fitted(s, "FA")
    base = {k: f[k] for k in ("T0", "kc", "phi")}
    l2 = L.evalQ(g, L.simQ(g, "L2", dict(T0=float(L.RCF.loc[s, "L2_p_T0"]))))
    r = dict(STAID=s, off_L2_te=l2[2], off_L2_tr=l2[1])
    for zt in (0.05, 1.0, 5.0, 20.0):
        e = L.evalQ(g, L.simQ(g, "FA", dict(base, z=zt)))
        r[f"off_z{zt:g}_te"], r[f"off_z{zt:g}_tr"] = e[2], e[1]
    e = L.evalQ(g, L.simQ(g, "FA", dict(base, z=f["z_raw"])))
    r["off_fitted_te"], r["off_fitted_tr"] = e[2], e[1]
    rows.append(r)
off = pd.DataFrame(rows).set_index("STAID")
T = E.join(off)
T.index.name = "STAID"
T.to_csv(os.path.join(HERE, "engine_z.csv"))
lines = [f"runs: {runs}", f"gauges: {len(T)}", "",
         "median paired dNSE vs the fair no-pool twin (engine) / vs offline L2 (offline), test WY1996-2010, [95% CI]"]
for tag in ("0.05", "1", "5", "20", "fitted"):
    k = "z" + tag if tag != "fitted" else "fitted"
    ok = f"off_z{tag}_te" if tag != "fitted" else "off_fitted_te"
    s_eng = boot_median(T[k] - T.twin) if k in T else (np.nan,) * 3
    s_off = boot_median(T[ok] - T.off_L2_te)
    up = int(((T[k] - T.twin) > 0).sum()) if k in T else -1
    dn = int(((T[k] - T.twin) < 0).sum()) if k in T else -1
    lines.append(f"  z = {tag:>6s}: engine {s_eng[0]:+.4f} [{s_eng[1]:+.4f}, {s_eng[2]:+.4f}] {up}/{dn}   "
                 f"offline {s_off[0]:+.4f} [{s_off[1]:+.4f}, {s_off[2]:+.4f}]")
if "fa_nopool" in T:
    s = boot_median(T.fa_nopool - T.twin)
    lines.append(f"  pool off at the FA T0 (run {FA_NOPOOL}): engine {s[0]:+.4f} [{s[1]:+.4f}, {s[2]:+.4f}]")
if "z0.05" in T:
    s = boot_median(T["z0.05"] - T.fa_nopool)
    lines.append(f"  z = 0.05 minus pool off (same T0): engine {s[0]:+.4f} [{s[1]:+.4f}, {s[2]:+.4f}], max |d| "
                 f"{np.abs(T['z0.05'] - T.fa_nopool).max():.4f}")
if "z0.05" in T:
    # the shelf in pool terms: what opening the pool from 0.05 d to z buys, same T0/kc/phi, paired per gauge
    lines += ["", "the pool's own contribution: NSE(z) minus NSE(z = 0.05 d), test WY1996-2010, median [95% CI], "
                  "share of gauges gaining < 0.005, median share of the fitted pool's gain reached"]
    for tag in ("1", "5", "20", "fitted"):
        k = "z" + tag if tag != "fitted" else "fitted"
        ok = f"off_z{tag}_te" if tag != "fitted" else "off_fitted_te"
        if k not in T:
            continue
        de, do = T[k] - T["z0.05"], T[ok] - T["off_z0.05_te"]
        fe = (de / (T.fitted - T["z0.05"])).where((T.fitted - T["z0.05"]) > 0.005)
        fo = (do / (T.off_fitted_te - T["off_z0.05_te"])).where((T.off_fitted_te - T["off_z0.05_te"]) > 0.005)
        se, so = boot_median(de), boot_median(do)
        lines.append(f"  z = {tag:>6s}: engine {se[0]:+.4f} [{se[1]:+.4f}, {se[2]:+.4f}] <0.005 {(de < 0.005).mean():.0%}"
                     f" reached {fe.median():.2f}   offline {so[0]:+.4f} [{so[1]:+.4f}, {so[2]:+.4f}] <0.005 "
                     f"{(do < 0.005).mean():.0%} reached {fo.median():.2f}")
    lines.append("  (reached = median over gauges whose fitted pool gains > 0.005 test NSE: engine "
                 f"n={int(((T.fitted - T['z0.05']) > 0.005).sum())}, offline "
                 f"n={int(((T.off_fitted_te - T['off_z0.05_te']) > 0.005).sum())})")
cols = [c for c in ("z0.05", "z1", "z5", "z20", "fitted") if c in T]
rho = [(T[c] - T.twin).corr(T[("off_" + c + "_te") if c != "fitted" else "off_fitted_te"] - T.off_L2_te, method="spearman")
       for c in cols]
lines.append("  per-gauge Spearman engine vs offline dNSE: " + ", ".join(f"{c} {r:.2f}" for c, r in zip(cols, rho)))
txt = "\n".join(lines)
print(txt)
open(os.path.join(HERE, "engine_summary.txt"), "w").write(txt + "\n")
