"""Split the summed-Q' seasonal-cycle error into volume (bias^2) and shape parts per dam gauge and control."""
import json
import numpy as np, pandas as pd, zarr
OUT = "/home/tbindas/.claude/jobs/dacd6d8c/tmp/"
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs/"
OFF = "2026-09-27T07-29-47Z-train-and-test"
SM = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/"
z = zarr.open(RUNS + OFF + "/eval/predictions.zarr", mode="r")
ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
O = z["observations"][:].astype(float)
idx = {s: i for i, s in enumerate(ids)}
bm = json.load(open(RUNS + OFF + "/baseline/manifest.json"))
B = np.fromfile(RUNS + OFF + "/baseline/predictions.f32", dtype=np.float32).reshape(bm["n_gauges"], bm["n_days"])
bidx = {str(g): i for i, g in enumerate(bm["gage_ids"])}
off0 = int((t[0].to_datetime64().astype("datetime64[D]") - np.datetime64(str(bm["time_range_daily"][0])[:10])).astype(int))
mon = t.month.values


def split(g):
    p = B[bidx[g], off0:off0 + len(t)].astype(float)
    o = O[idx[g]]
    m = np.isfinite(p) & np.isfinite(o)
    p, o, mm = p[m], o[m], mon[m]
    var = o.var()
    cp = pd.Series(p).groupby(mm).transform("mean").values
    co = pd.Series(o).groupby(mm).transform("mean").values
    clim = ((cp - co) ** 2).mean() / var
    vol = (p.mean() - o.mean()) ** 2 / var
    return clim, vol, clim - vol, p.mean() / o.mean()


sm = pd.read_csv(SM + "smoke_gauges.csv", dtype={"STAID": str, "control_for": str})
ctl = sm[sm.role == "control"].set_index("control_for").STAID
D = pd.read_csv(OUT + "clim_by_dam.csv", dtype={"STAID": str, "huc2": str})
rows = []
for s in D.STAID:
    c = ctl[s]
    a, b = split(s), split(c)
    rows.append(dict(STAID=s, vol_x=a[1] - b[1], shape_x=a[2] - b[2], beta=a[3], beta_ctl=b[3]))
D = D.merge(pd.DataFrame(rows), on="STAID")
D.to_csv(OUT + "clim_by_dam.csv", index=False)
pd.set_option("display.width", 200)
for by in ["purp", "region", "size"]:
    g = D.groupby(by, observed=True)
    print(f"\n=== {by}: excess over matched control, medians (volume part + shape part ~ clim_excess)")
    print(g.size().rename("n").to_frame().join(g[["clim_excess", "vol_x", "shape_x", "beta", "beta_ctl"]].median()).round(3).to_string())
D["clim_tercile"] = pd.qcut(D.clim_excess, 3, labels=["low", "mid", "high"])
print("\nShare of gauges gaining > 0.01, by seasonal-error tercile:")
print(D.groupby("clim_tercile", observed=True)[["g_bucket_engine", "g_bucket_off", "g_rule_off"]]
      .agg(lambda v: (v > 0.01).mean()).round(2).to_string())
