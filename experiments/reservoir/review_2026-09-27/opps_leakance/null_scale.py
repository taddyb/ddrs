"""Null model: is the leakance arm's per-gauge effect reproduced by a single global volume scale on the
no-leakance arm's routed predictions?  Plus: where does the extra dam-reach loss come from (parameters or
geometry), and the volume-bias split at dammed gauges by DOR."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import netCDF4
from scipy.stats import spearmanr


RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
WT = Path("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options")
AG = Path("/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4")
OUT = Path("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_leakance")
LEAK, BASE = "2026-09-17T16-38-16Z-train-and-test", "2026-09-12T23-39-03Z-train-and-test"


def load(run):
    z = zarr.open(str(RUNS / run / "eval" / "predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    return pd.Index(ids), z["predictions"][:].astype(float), z["observations"][:].astype(float)


def nse_rows(P, O):
    out = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        m = np.isfinite(P[i]) & np.isfinite(O[i])
        if m.sum() < 365:
            continue
        p, o = P[i][m], O[i][m]
        out[i] = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    return out


ids, P_b, O = load(BASE)
_, P_l, _ = load(LEAK)
nse_b = nse_rows(P_b, O)
nse_l = nse_rows(P_l, O)
res = {}
scales = {}
for s in [0.99, 0.985, 0.981, 0.975, 0.97, 0.96, 0.95]:
    n_s = nse_rows(P_b * s, O)
    d = n_s - nse_b
    rho = spearmanr(d, nse_l - nse_b, nan_policy="omit")[0]
    scales[s] = dict(median_nse=round(float(np.nanmedian(n_s)), 4), median_dnse=round(float(np.nanmedian(d)), 4),
                     up=int((d > 1e-9).sum()), down=int((d < -1e-9).sum()), spearman_vs_leak_dnse=round(float(rho), 3))
res["global_scale_null"] = scales
res["arms"] = dict(base_median=round(float(np.nanmedian(nse_b)), 4), leak_median=round(float(np.nanmedian(nse_l)), 4))
# per-gauge best scale on the base arm (what volume correction the gauge wants), and its relation to leak dVOL
per = pd.read_csv(OUT / "leak_vs_base_per_gauge.csv", dtype={"STAID": str}).set_index("STAID")
per = per.reindex(ids)
dl = (nse_l - nse_b)
# partial correlation: leak gain vs dam-release gain, controlling for vol_base and log area, dammed gauges only
d = per.copy()
d["dl"] = dl
d = d[(d.n_nid_ge10mcm > 0)].dropna(subset=["dl", "dnse_dam_release", "vol_base", "area_km2"])
X = np.column_stack([np.ones(len(d)), np.log(d.vol_base), np.log(d.area_km2), d.nse_base])
r1 = d.dl.values - X @ np.linalg.lstsq(X, d.dl.values, rcond=None)[0]
r2 = d.dnse_dam_release.values - X @ np.linalg.lstsq(X, d.dnse_dam_release.values, rcond=None)[0]
res["partial_spearman_leakgain_vs_damreleasegain_ctrl_vol_area_nse"] = dict(rho=round(float(spearmanr(r1, r2)[0]), 3), n=int(len(d)))
# vol_base split by DOR
dam = per[per.n_nid_ge10mcm > 0]
dor = pd.cut(dam.nid_dor, [0, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
res["vol_base_by_dor"] = {str(k): dict(n=int(len(g)), median_vol=round(float(g.vol_base.median()), 3),
                                        share_surplus=round(float((g.vol_base > 1.05).mean()), 3),
                                        share_deficit=round(float((g.vol_base < 0.95).mean()), 3),
                                        median_abs_log_vol=round(float(np.abs(np.log(g.vol_base)).median()), 3))
                          for k, g in dam.groupby(dor, observed=True)}
und = per[per.n_nid_ge10mcm == 0]
res["vol_base_undammed"] = dict(n=int(len(und)), median_vol=round(float(und.vol_base.median()), 3),
                                share_surplus=round(float((und.vol_base > 1.05).mean()), 3),
                                share_deficit=round(float((und.vol_base < 0.95).mean()), 3),
                                median_abs_log_vol=round(float(np.abs(np.log(und.vol_base)).median()), 3))

# learned parameters at dam reaches vs others (full CONUS dump), restricted to the eval network
ds = netCDF4.Dataset(RUNS / LEAK / "plot" / "kan_parameters.nc")
kp = pd.DataFrame({k: ds.variables[k][:] for k in ["COMID", "K_D", "d_gw", "leakance_factor", "n", "gamma"]}).set_index("COMID")
ds.close()
ds = netCDF4.Dataset(RUNS / LEAK / "kan_parameters.nc")
z = pd.DataFrame({k: ds.variables[k][:] for k in ["COMID_eval", "zeta_net", "depth_mean", "area_z_mean", "q_mean"]}).rename(
    columns={"COMID_eval": "COMID"}).set_index("COMID")
ds.close()
z = z.join(kp, how="inner")
dams = pd.read_csv(AG / "experiments/reservoir/nid/nid_dams_in_eval_network.csv")
big = dams[dams.storage_mcm >= 10].COMID.unique()
z["dam"] = z.index.isin(big)
z["qdec"] = pd.qcut(z.q_mean.rank(method="first"), 10, labels=False)
z["kfac"] = z.K_D * z.leakance_factor
z["area_per_q"] = z.area_z_mean / z.q_mean.clip(lower=1e-3)
z["hd"] = z.depth_mean - z.d_gw
rows = []
for dec, g in z.groupby("qdec"):
    a, b = g[g.dam], g[~g.dam]
    if len(a) < 20:
        continue
    rows.append(dict(qdec=int(dec), n_dam=int(len(a)),
                     K_D_dam=float(f"{a.K_D.median():.3g}"), K_D_other=float(f"{b.K_D.median():.3g}"),
                     factor_dam=round(float(a.leakance_factor.median()), 3), factor_other=round(float(b.leakance_factor.median()), 3),
                     d_gw_dam=round(float(a.d_gw.median()), 3), d_gw_other=round(float(b.d_gw.median()), 3),
                     head_dam=round(float(a.hd.median()), 3), head_other=round(float(b.hd.median()), 3),
                     area_per_q_dam=round(float(a.area_per_q.median()), 1), area_per_q_other=round(float(b.area_per_q.median()), 1),
                     n_dam_=round(float(a.n.median()), 4), n_other=round(float(b.n.median()), 4)))
res["dam_vs_other_by_q_decile"] = rows
res["field_spread"] = dict(K_D_p10_p50_p90=[float(f"{v:.3g}") for v in z.K_D.quantile([0.1, 0.5, 0.9])],
                           d_gw_p10_p50_p90=[round(float(v), 3) for v in z.d_gw.quantile([0.1, 0.5, 0.9])],
                           factor_p10_p50_p90=[round(float(v), 3) for v in z.leakance_factor.quantile([0.1, 0.5, 0.9])],
                           frac_p10_p50_p90=[round(float(v), 5) for v in (z.zeta_net / z.q_mean.clip(lower=1e-3)).quantile([0.1, 0.5, 0.9])])
json.dump(res, open(OUT / "null_scale.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
