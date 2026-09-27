"""Does dHBV's Q' already carry reservoir operations? Matched-pair gap (dam gauge minus its undammed control)
for the summed-Q' baseline (no routing) and for the routed no-dam model, seed 42, test WY1996-2010.
Also r and the seasonal (monthly climatology) error for the summed-Q' baseline."""
import json
import numpy as np, pandas as pd, zarr

RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs/"
WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/"
OFF = "2026-09-27T07-29-47Z-train-and-test"
p = pd.read_csv(WT + "full_run/paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
sm = pd.read_csv(WT + "smoke/smoke_gauges.csv", dtype={"STAID": str, "control_for": str})
ctl = sm[sm.role == "control"].set_index("control_for").STAID
dam = sm[sm.role == "dam"].set_index("STAID")

z = zarr.open(RUNS + OFF + "/eval/predictions.zarr", mode="r")
ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
O = z["observations"][:].astype(float)
P = z["predictions"][:].astype(float)
bm = json.load(open(RUNS + OFF + "/baseline/manifest.json"))
B = np.fromfile(RUNS + OFF + "/baseline/predictions.f32", dtype=np.float32).reshape(bm["n_gauges"], bm["n_days"])
bidx = {str(g): i for i, g in enumerate(bm["gage_ids"])}
off0 = int((t[0].to_datetime64().astype("datetime64[D]") - np.datetime64(str(bm["time_range_daily"][0])[:10])).astype(int))
idx = {s: i for i, s in enumerate(ids)}
mon = t.month.values


def stats(pr, o):
    m = np.isfinite(pr) & np.isfinite(o)
    pr, o, mm = pr[m], o[m], mon[m]
    var = o.var()
    cp = pd.Series(pr).groupby(mm).transform("mean").values
    co = pd.Series(o).groupby(mm).transform("mean").values
    return dict(nse=1 - ((pr - o) ** 2).mean() / var, r=np.corrcoef(pr, o)[0, 1], clim=((cp - co) ** 2).mean() / var,
                cv_ratio=(pr.std() / pr.mean()) / (o.std() / o.mean()))


rows = []
for s, x in dam.iterrows():
    c = ctl.get(s)
    if s not in idx or c not in idx or s not in bidx or c not in bidx:
        continue
    for who, g in [("dam", s), ("ctl", c)]:
        o = O[idx[g]]
        b = B[bidx[g], off0:off0 + len(t)].astype(float)
        rows.append(dict(pair=s, who=who, dor=x.nid_dor, **{"base_" + k: v for k, v in stats(b, o).items()},
                         **{"routed_" + k: v for k, v in stats(P[idx[g]], o).items()}))
R = pd.DataFrame(rows)
W = R.pivot(index="pair", columns="who")
dor = W[("dor", "dam")]
hi = W[dor > 0.5]
print(f"DOR > 0.5 pairs: {len(hi)}")
for k in ["nse", "r", "clim", "cv_ratio"]:
    for src in ["base", "routed"]:
        d, c = hi[(f"{src}_{k}", "dam")], hi[(f"{src}_{k}", "ctl")]
        print(f"  {src:6s} {k:8s} dam {d.median():.3f}  control {c.median():.3f}  paired gap {(d - c).median():+.3f}")
lo = W[dor <= 0.1]
print(f"DOR <= 0.1 pairs: {len(lo)}  base gap {(lo[('base_nse','dam')] - lo[('base_nse','ctl')]).median():+.3f}"
      f"  routed gap {(lo[('routed_nse','dam')] - lo[('routed_nse','ctl')]).median():+.3f}")
