import numpy as np, pandas as pd, zarr, sys
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs/"
PAIRED = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/full_run/paired_full_run.csv"
OUT = "/home/tbindas/.claude/jobs/dacd6d8c/tmp/decomp.csv"


def load(run):
    z = zarr.open(RUNS + run + "/eval/predictions.zarr", mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
    return pd.Index(ids), t, z["predictions"][:].astype(float), z["observations"][:].astype(float)


ids, t, P, O = load(sys.argv[1])
_, _, PL, _ = load(sys.argv[2])
mon = pd.DatetimeIndex(t).month.values
d = pd.read_csv(PAIRED, dtype={"STAID": str}).set_index("STAID")


def dec(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    mm = mon[m]
    so = o.std()
    a = p.std() / so
    r = np.corrcoef(p, o)[0, 1]
    bn = (p.mean() - o.mean()) / so
    cp = pd.Series(p).groupby(mm).transform("mean").values
    co = pd.Series(o).groupby(mm).transform("mean").values
    var = o.var()
    return dict(alpha=a, r=r, beta=p.mean() / o.mean(), bias2=bn**2, varmis=(a - r) ** 2, corr=1 - r**2,
                clim=((cp - co) ** 2).mean() / var, anom=(((p - cp) - (o - co)) ** 2).mean() / var,
                nse=1 - ((p - o) ** 2).mean() / var)


rows = []
for i, s in enumerate(ids):
    x = dec(P[i], O[i])
    y = dec(PL[i], O[i])
    rows.append(dict(STAID=s, **x, **{k + "_L": v for k, v in y.items()}))
R = pd.DataFrame(rows).set_index("STAID").join(d[["n_nid_ge10mcm", "nid_dor", "nid_on_gauge_reach"]])
dam = R.n_nid_ge10mcm > 0
R["bin"] = np.where(~dam, "undammed", pd.cut(R.nid_dor, [-1e-9, 0.1, 0.5, 1, 2, 1e9],
                                              labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]).astype(str))
R.to_csv(OUT)
cols = ["nse", "alpha", "r", "beta", "bias2", "varmis", "corr", "clim", "anom"]
print("OFF arm medians by bin (1-NSE = bias2+varmis+corr; also = clim+anom)")
print(R.groupby("bin")[cols].median().round(3).to_string())
for c in cols:
    R["d_" + c] = R[c + "_L"] - R[c]
print("\nLEARNED - OFF, median of per-gauge diffs")
print(R.groupby("bin")[["d_" + c for c in cols]].median().round(4).to_string())
comp = ["bias2", "varmis", "corr", "clim", "anom"]
print("\nMEAN of components (clipped at 3) by bin, off arm")
print(R.assign(**{c: R[c].clip(upper=3) for c in comp}).groupby("bin")[comp].mean().round(3).to_string())
hi = R[R.nid_dor > 0.5]
print("\nDOR>0.5: share beta>1.1", round((hi.beta > 1.1).mean(), 3), " beta<0.9", round((hi.beta < 0.9).mean(), 3))
print("undammed: share beta>1.1", round((R[~dam].beta > 1.1).mean(), 3), " beta<0.9", round((R[~dam].beta < 0.9).mean(), 3))
