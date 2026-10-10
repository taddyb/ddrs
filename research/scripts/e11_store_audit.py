"""E-11 store audit. Read-only. Run with /home/tbindas/projects/ddr/.venv/bin/python."""
import icechunk, zarr, numpy as np, xarray as xr, json, warnings
warnings.filterwarnings("ignore")
B = "/mnt/ssd1/data/icechunk/"
S = {
 "lumped_orig_0814": "daily_dhbv2_lumped_aorc2f_merit_unit_catchments.ic",
 "lumped_rechunk":   "daily_dhbv2_lumped_aorc2f_merit_unit_catchments_rechunked.ic",
 "distributed":      "daily_dhbv2_distributed_aorc2f_merit_unit_catchments_rechunked.ic",
 "UH_retro":         "merit_dhbv2_UH_retrospective.ic",
 "hydroDL_lstm":     "daily_hydrodl_lstm_aorc2f_merit_unit_catchments_rechunked.ic",
}
G = {k: zarr.open_group(icechunk.Repository.open(icechunk.local_filesystem_storage(B+v)).readonly_session("main").store, mode="r") for k, v in S.items()}
ids = {k: g["divide_id"][:] for k, g in G.items()}
for k in ids: print(k, "divide_id equal to lumped_rechunk:", bool((ids[k] == ids["lumped_rechunk"]).all()), "n", len(ids[k]))
t = G["lumped_rechunk"]["time"][:]
print("time", t[0], t[-1], "step==1:", bool((np.diff(t) == 1).all()), "n", len(t), "-> 1980-01-01 +", t[-1], "d =", np.datetime64("1980-01-01") + int(t[-1]))
for k in G: print(k, "time equal:", bool((G[k]["time"][:] == t).all()))
A = xr.open_dataset("/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
cs = A["catchsize"].sel(COMID=ids["lumped_rechunk"]).values.astype("f8")   # km2
print("catchsize km2 quantiles", np.nanquantile(cs, [0, .05, .5, .95, 1]), "NaN", int(np.isnan(cs).sum()))
UHPOS = np.full(int(max(ids["lumped_rechunk"].max(), ids["UH_retro"].max()))+1, -1, dtype=np.int64); print("UH id sets equal:", set(ids["UH_retro"].tolist()) == set(ids["lumped_rechunk"].tolist()))
UHPOS[ids["UH_retro"]] = np.arange(len(ids["UH_retro"]))
blocks = [3, 13, 23, 33, 43, 53]
tr = slice(5479*0, None)
rows = {k: [] for k in G}
eq_ok = True
res = {}
LAGMAX = 10
FLOOR = np.float32(1e-6)
agg = {k: dict(n=0, nan=0, floor=0, allfloor=0, sum=0.0) for k in G}
cmp = {k: dict(corr=[], lag=[], ratio=[]) for k in ["distributed", "UH_retro", "hydroDL_lstm"]}
depth_bins = np.array([0, 3, 10, 30, 100, 1e9])
dep = {k: [[] for _ in range(5)] for k in G}
shape_stats = {k: dict(cv=[], p98r=[]) for k in G}
orig_eq = 0; orig_n = 0
for b in blocks:
    s = slice(b*3080, (b+1)*3080)
    X = {k: g["Qr"][s, :] for k, g in G.items() if k != "UH_retro"}
    # UH_retro stores divides in a different row order: align by divide_id
    pos = UHPOS[ids["lumped_rechunk"][s]]
    assert (pos >= 0).all()
    lo, hi = pos.min(), pos.max()+1
    X["UH_retro"] = G["UH_retro"]["Qr"][lo:hi, :][pos-lo]
    orig_eq += int((X["lumped_orig_0814"] == X["lumped_rechunk"]).sum()); orig_n += X["lumped_rechunk"].size
    area = cs[s]
    for k, x in X.items():
        a = agg[k]; a["n"] += x.size; a["nan"] += int(np.isnan(x).sum()); a["floor"] += int((x == FLOOR).sum())
        a["allfloor"] += int((x.max(1) <= FLOOR).sum()); a["sum"] += float(np.nansum(x, dtype="f8"))
        m = np.nanmean(x, 1, dtype="f8"); depth = m*86.4/area
        bi = np.digitize(area, depth_bins[1:-1])
        for i in range(5):
            sel = bi == i
            if sel.any(): dep[k][i].append(depth[sel & np.isfinite(depth)])
    L = X["lumped_rechunk"].astype("f8"); mL = L.mean(1)
    ok = mL > 0.01
    for k in cmp:
        Y = X[k].astype("f8"); Ys = Y[ok]; Ls = L[ok]
        cmp[k]["ratio"].append(Ls.mean(1)/np.maximum(Ys.mean(1), 1e-12))
        Lz = Ls - Ls.mean(1, keepdims=True); Yz = Ys - Ys.mean(1, keepdims=True)
        den = np.sqrt((Lz**2).sum(1)*(Yz**2).sum(1)) + 1e-12
        cc = {}
        for lag in range(-LAGMAX, LAGMAX+1):   # lag>0: lumped leads (lumped[t] matches product[t+lag])
            if lag >= 0: num = (Lz[:, :Lz.shape[1]-lag]*Yz[:, lag:]).sum(1)
            else: num = (Lz[:, -lag:]*Yz[:, :Yz.shape[1]+lag]).sum(1)
            cc[lag] = num/den
        M = np.stack([cc[l] for l in range(-LAGMAX, LAGMAX+1)], 1)
        cmp[k]["corr"].append(cc[0]); cmp[k]["lag"].append(np.argmax(M, 1)-LAGMAX)
    for k, x in X.items():
        xs = x[ok].astype("f8")
        shape_stats[k]["cv"].append(xs.std(1)/xs.mean(1))
        shape_stats[k]["p98r"].append(np.quantile(xs, .98, axis=1)/xs.mean(1))
print("\noriginal(08-14 snapshot) vs rechunked identical values: %d / %d" % (orig_eq, orig_n))
print("\nper-store sample stats (6 row blocks = %d divides)" % (len(blocks)*3080))
for k, a in agg.items():
    print(f"{k:18s} mean {a['sum']/a['n']:.4f} m3/s  NaN {a['nan']}  ==1e-6 floor frac {a['floor']/a['n']:.4f}  divides entirely floor {a['allfloor']}")
print("\nlumped vs product, divides with lumped mean>0.01 m3/s")
for k, c in cmp.items():
    r = np.concatenate(c["ratio"]); cr = np.concatenate(c["corr"]); lg = np.concatenate(c["lag"])
    print(f"vs {k:13s} n={len(r)} mean-ratio median {np.median(r):.3f} [IQR {np.quantile(r,.25):.3f},{np.quantile(r,.75):.3f}]  daily corr median {np.median(cr):.3f}  best-lag distribution (days, + = lumped leads) " + str({int(l): round(float((lg == l).mean()), 3) for l in range(-LAGMAX, LAGMAX+1) if (lg == l).mean() > 0.005}))
print("\nrunoff depth (mm/day, mean over 1980-2020) by divide area class")
names = ["<3 km2", "3-10", "10-30", "30-100", ">100"]
for i, n in enumerate(names):
    print(f"{n:8s} " + "  ".join(f"{k}: {np.median(np.concatenate(dep[k][i])):.3f} (n={sum(len(z) for z in dep[k][i])})" for k in ["lumped_rechunk", "distributed", "UH_retro", "hydroDL_lstm"] if dep[k][i]))
print("\nper-divide flashiness (divides with lumped mean>0.01)")
for k, s_ in shape_stats.items():
    print(f"{k:18s} CV median {np.median(np.concatenate(s_['cv'])):.2f}  p98/mean median {np.median(np.concatenate(s_['p98r'])):.2f}")
