"""Equifinality measurement: did the routing head change Manning n upstream of dams differently from elsewhere
when the release was learned? Uses the gauge subgraph COMID orders (merit_gages_conus_adjacency.zarr) for the
smoke set's on-reach dam gauges and their matched controls, and the per-reach n / gamma dumps of the four arms."""
import numpy as np
import pandas as pd
import zarr
import netCDF4 as nc

RUNS = {
    "off42": "2026-09-27T07-29-47Z-train-and-test",
    "learn42": "2026-09-27T07-29-55Z-train-and-test",
    "off43": "2026-09-27T10-31-30Z-train-and-test",
    "learn43": "2026-09-27T10-31-50Z-train-and-test",
}
par = {}
for k, r in RUNS.items():
    d = nc.Dataset(f"/home/tbindas/projects/ddrs/.ddrs/runs/{r}/plot/kan_parameters.nc")
    par[k] = pd.DataFrame({"n": d.variables["n"][:], "gamma": d.variables["gamma"][:]}, index=d.variables["COMID"][:])
    d.close()
allc = par["off42"].index
smoke = pd.read_csv("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir/smoke/smoke_gauges.csv",
                    dtype={"STAID": str, "control_for": str})
g = zarr.open_group("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
dam_on = smoke[(smoke.role == "dam") & (smoke.dam_COMID == smoke.dam_COMID)].copy()
# on-reach: dam COMID equals the gauge's COMID; need the gauge COMID: use gage_catchment attr
rows = []
for _, s in smoke.iterrows():
    if s.STAID not in g:
        continue
    gg = g[s.STAID]
    order = gg["order"][:]
    gc = gg.attrs.get("gage_catchment")
    on_reach = (s.role == "dam") and (gc == s.dam_COMID)
    sub = par["off42"].index.isin(order)
    rec = {"STAID": s.STAID, "role": s.role, "on_reach": on_reach, "n_reaches": int(sub.sum()), "cascade": s.cascade, "dor": s.nid_dor}
    for k in RUNS:
        p = par[k][sub]
        rec[f"logn_{k}"] = np.log(p.n).mean()
        rec[f"gamma_{k}"] = p.gamma.mean()
    rows.append(rec)
df = pd.DataFrame(rows)
df["d42"] = df.logn_learn42 - df.logn_off42
df["d43"] = df.logn_learn43 - df.logn_off43
df["dseed_off"] = df.logn_off43 - df.logn_off42
df["dg42"] = df.gamma_learn42 - df.gamma_off42
df["dg43"] = df.gamma_learn43 - df.gamma_off43
print("gauges:", len(df), " roles:", df.role.value_counts().to_dict(), " on-reach dam gauges:", int(df.on_reach.sum()))
for lab, m in [("on-reach dam", df.on_reach), ("dam (any)", df.role == "dam"), ("control", df.role == "control")]:
    t = df[m]
    print(f"{lab:14s} n={len(t):3d}  mean Δlog n (learned-off) seed42 {t.d42.mean():+.4f} (median {t.d42.median():+.4f}), seed43 {t.d43.mean():+.4f} (median {t.d43.median():+.4f});"
          f" off43-off42 {t.dseed_off.mean():+.4f}; Δgamma s42 {t.dg42.mean():+.4f} s43 {t.dg43.mean():+.4f}")
# paired: dam gauge minus its control
c = df[df.role == "control"].set_index("STAID")
d = df[df.role == "dam"].copy()
ctl = smoke[smoke.role == "control"].set_index("control_for").STAID
d["ctl"] = d.STAID.map(ctl)
d = d[d.ctl.isin(c.index)]
for col in ("d42", "d43", "dseed_off"):
    diff = d[col].values - c.loc[d.ctl, col].values
    on = d.on_reach.values
    print(f"dam minus matched control, {col}: all {np.mean(diff):+.4f} (median {np.median(diff):+.4f}, n {len(diff)}); on-reach {np.mean(diff[on]):+.4f} (n {on.sum()})")
# whole-CONUS reference
for k in ("off42", "learn42", "off43", "learn43"):
    print(k, "CONUS median n %.4f, mean log n %.4f, median gamma %.4f" % (par[k].n.median(), np.log(par[k].n).mean(), par[k].gamma.median()))
print("CONUS mean Δlog n learned-off: s42 %+.4f, s43 %+.4f; off43-off42 %+.4f" % (
    (np.log(par["learn42"].n) - np.log(par["off42"].n)).mean(), (np.log(par["learn43"].n) - np.log(par["off43"].n)).mean(),
    (np.log(par["off43"].n) - np.log(par["off42"].n)).mean()))
# does the change scale with regulation?
t = df[df.role == "dam"].copy()
t["dorbin"] = pd.cut(t.dor, [-1, 0.1, 0.5, 1, 2, 100])
print(t.groupby("dorbin")[["d42", "d43", "dseed_off"]].mean().assign(n=t.groupby("dorbin").size()))
df.to_csv("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_training/n_above_dams.csv", index=False)
