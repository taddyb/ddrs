"""Decompose where each learned term acts. KGE splits skill into three independent
error modes: r (timing and shape), alpha = sd_sim/sd_obs (amplitude), beta =
mean_sim/mean_obs (volume). Roughness conserves mass, so it can move r and alpha
but not beta. Leakance removes mass, so beta is the only mode it can reach that
roughness cannot. Comparing the same three numbers for the unrouted inflow, the
leakance-off arm and the leakance-on arm says which bias each term actually absorbed."""
import numpy as np, zarr, json
B="/home/tbindas/projects/ddrs/.ddrs/runs"
ON=f"{B}/2026-09-16T04-44-15Z-train-and-test"; OFF=f"{B}/2026-09-16T06-11-35Z-train-and-test"

def comps(sim, obs):
    out=[]
    for i in range(sim.shape[0]):
        s,o = sim[i], obs[i]
        m = np.isfinite(s)&np.isfinite(o)&(o>=0)
        if m.sum() < 365: out.append((np.nan,)*4); continue
        s,o = s[m], o[m]
        so, oo = s.std(), o.std()
        if oo <= 0 or so <= 0: out.append((np.nan,)*4); continue
        r = np.corrcoef(s,o)[0,1]; a = so/oo; b = s.mean()/o.mean()
        kge = 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)
        out.append((r,a,b,kge))
    return np.array(out)

def load_eval(run):
    g = zarr.open_group(f"{run}/eval/predictions.zarr", mode="r")
    return np.asarray(g["predictions"][:],float), np.asarray(g["observations"][:],float), np.asarray(g["gage_ids"][:])

res={}
for lab, run in [("leakance ON", ON), ("leakance OFF", OFF)]:
    sim, obs, ids = load_eval(run)
    if sim.shape[0] < sim.shape[1] and sim.shape[0] < 5000: pass
    else: sim, obs = sim.T, obs.T
    res[lab] = (comps(sim, obs), ids)
    print(f"{lab}: {sim.shape} gauges x days")

# unrouted summed inflow, from the shared baseline
bm = json.load(open(f"{ON}/baseline/manifest.json"))
ng = len(bm["gage_ids"]); nd = len(bm["time_range_daily"])
bp = np.fromfile(f"{ON}/baseline/predictions.f32", dtype=np.float32).reshape(ng, nd).astype(float)
bo = np.fromfile(f"{ON}/baseline/observations.f32", dtype=np.float32).reshape(ng, nd).astype(float)
res["summed inflow (no routing)"] = (comps(bp, bo), np.asarray(bm["gage_ids"]))
print(f"summed inflow: {bp.shape} gauges x days")

print(f"\n{'arm':28s} {'r':>8s} {'alpha':>8s} {'beta':>8s} {'KGE':>8s}   n")
order=["summed inflow (no routing)","leakance OFF","leakance ON"]
med={}
for k in order:
    c,_ = res[k]; ok=np.isfinite(c[:,3])
    med[k]=np.nanmedian(c[ok],axis=0)
    print(f"{k:28s} " + " ".join(f"{v:8.4f}" for v in med[k]) + f"   {ok.sum()}")

print("\nwhat routing moved (leakance OFF minus unrouted):")
d=med["leakance OFF"]-med["summed inflow (no routing)"]
print("   " + " ".join(f"{n}={v:+.4f}" for n,v in zip(["r","alpha","beta","KGE"],d)))
print("what leakance added on top (ON minus OFF):")
d2=med["leakance ON"]-med["leakance OFF"]
print("   " + " ".join(f"{n}={v:+.4f}" for n,v in zip(["r","alpha","beta","KGE"],d2)))
print("\nvolume bias |beta-1| at the median gauge:")
for k in order: print(f"   {k:28s} {abs(med[k][2]-1)*100:6.2f} %")
