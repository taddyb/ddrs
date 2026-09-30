import numpy as np, pandas as pd, zarr
from scipy.stats import spearmanr
from pathlib import Path
OUT = Path("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_metrics")
w = pd.read_csv(OUT / "per_gauge_metrics.csv", dtype={"STAID": str}).set_index("STAID")
dammed = (w.n_nid_ge10mcm > 0).values
la = np.log10(w.area_km2.values)
print("area confound: Spearman(dNSE, log area)")
for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
    d = (w[f"{a}__nse"] - w[f"{b}__nse"]).values
    dd = (w[f"{a}__nse_diff"] - w[f"{b}__nse_diff"]).values
    for g, sel in [("dammed", dammed), ("undammed", ~dammed)]:
        ok = sel & np.isfinite(d) & np.isfinite(la)
        print(f" seed {s} {g}: rho(dNSE, logA)={spearmanr(d[ok], la[ok]).correlation:+.3f}  rho(dNSEdiff, logA)={spearmanr(dd[ok], la[ok]).correlation:+.3f}")
bins = np.nanpercentile(la[dammed], [0, 20, 40, 60, 80, 100])
for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
    d = (w[f"{a}__nse"] - w[f"{b}__nse"]).values
    parts = []
    for i in range(5):
        sel_d = dammed & (la >= bins[i]) & (la <= bins[i + 1])
        sel_u = ~dammed & (la >= bins[i]) & (la <= bins[i + 1])
        parts.append((i, int(sel_d.sum()), int(sel_u.sum()), np.nanmedian(d[sel_d]), np.nanmedian(d[sel_u])))
    print(f" seed {s} area-binned (bins {np.round(10**bins).astype(int)} km2) dammed/undammed medians and DiD:")
    for i, nd, nu, md, mu in parts:
        print(f"   bin{i}: n_dam={nd} n_undam={nu} dam={md:+.4f} undam={mu:+.4f} did={md-mu:+.4f}")
z = zarr.open("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-27T07-29-55Z-train-and-test/eval/predictions.zarr", mode="r")
print("zarr arrays:", list(z.array_keys()), dict(z.attrs))
P = z["predictions"][:]; O = z["observations"][:]
m = np.isfinite(P) & np.isfinite(O)
diffs = []
for i in range(P.shape[0]):
    p, o = P[i, m[i]], O[i, m[i]]
    if len(p) < 365:
        continue
    om32 = np.add.accumulate(o, dtype=np.float32)[-1] / np.float32(len(o))
    sse32 = np.add.accumulate(((p - o) * (p - o)).astype(np.float32), dtype=np.float32)[-1]
    sso32 = np.add.accumulate(((o - om32) * (o - om32)).astype(np.float32), dtype=np.float32)[-1]
    nse32 = 1 - sse32 / sso32
    p64, o64 = p.astype(np.float64), o.astype(np.float64)
    nse64 = 1 - ((p64 - o64) ** 2).sum() / ((o64 - o64.mean()) ** 2).sum()
    diffs.append((abs(nse32 - nse64), nse64, o64.mean()))
diffs = np.array(diffs)
print(f"f32 sequential vs f64 NSE: max abs diff {diffs[:,0].max():.2e}, 99th pct {np.percentile(diffs[:,0],99):.2e}, median {np.median(diffs[:,0]):.2e}; n > 1e-4: {(diffs[:,0]>1e-4).sum()}, > 1e-3: {(diffs[:,0]>1e-3).sum()}")
big = diffs[diffs[:, 0] > 1e-4]
print(" those gauges (diff, NSE, mean obs m3/s):", [(f"{d:.1e}", f"{n:.3f}", f"{q:.0f}") for d, n, q in big[:10]])
