"""Does the two-way leakance arm's skill gain concentrate at dammed gauges, and does the learned
zeta field sit on dam reaches?  Light read on existing outputs, no training.

Arms: leak  = 2026-09-17T16-38-16Z (sr_n0_gamma_leakance, gamma box [-0.2,0.85], two-way, alpha 0.25)
      base  = 2026-09-12T23-39-03Z (sr_n0_gamma, gamma box [0,0.5], no leakance)
Caveat: the pair differs in the gamma box as well as leakance (the matched control was never run).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import netCDF4
from scipy.stats import binomtest, spearmanr, mannwhitneyu

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
WT = Path("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options")
AG = Path("/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4")
OUT = Path("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_leakance")
LEAK, BASE = "2026-09-17T16-38-16Z-train-and-test", "2026-09-12T23-39-03Z-train-and-test"


def load(run):
    z = zarr.open(str(RUNS / run / "eval" / "predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
    return pd.Index(ids), t, z["predictions"][:], z["observations"][:]


def metrics(p, o):
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() < 365:
        return np.nan, np.nan, np.nan
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    kge = 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)
    return nse, kge, p.mean() / o.mean()


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(n=0)
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    b = np.median(rng.choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=round(float(np.median(d)), 4),
                ci=[round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                up=up, down=int(len(nz) - up), sign_p=float(f"{binomtest(up, len(nz)).pvalue:.2g}") if len(nz) else None)


ids_l, t_l, P_l, O_l = load(LEAK)
ids_b, t_b, P_b, O_b = load(BASE)
assert (ids_l == ids_b).all() and (t_l == t_b).all(), "gauge/time axes differ"
rows = []
for i, s in enumerate(ids_l):
    nl, kl, vl = metrics(P_l[i].astype(float), O_l[i].astype(float))
    nb, kb, vb = metrics(P_b[i].astype(float), O_b[i].astype(float))
    rows.append(dict(STAID=s, nse_leak=nl, kge_leak=kl, vol_leak=vl, nse_base=nb, kge_base=kb, vol_base=vb))
df = pd.DataFrame(rows).set_index("STAID")
nid = pd.read_csv(WT / "experiments/reservoir/nid/nid_dams_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
df = df.join(nid[["area_km2", "n_nid_ge10mcm", "nid_dor", "nid_on_gauge_reach"]])
sm = pd.read_csv(AG / "experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
df["smoke_role"] = df.index.map(sm.role)
pf = pd.read_csv(WT / "experiments/reservoir/full_run/paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
df["dnse_dam_release"] = pf.dnse
df["dnse"] = df.nse_leak - df.nse_base
df["dkge"] = df.kge_leak - df.kge_base
df["dvol"] = df.vol_leak - df.vol_base
df.to_csv(OUT / "leak_vs_base_per_gauge.csv")

dammed = df.n_nid_ge10mcm > 0
onreach = dammed & df.nid_on_gauge_reach.astype(bool)
surplus = df.vol_base > 1.05
deficit = df.vol_base < 0.95
res = {}
res["medians"] = dict(nse_base=round(df.nse_base.median(), 4), nse_leak=round(df.nse_leak.median(), 4),
                      kge_base=round(df.kge_base.median(), 4), kge_leak=round(df.kge_leak.median(), 4))
res["dNSE_leak_minus_base"] = dict(
    all=paired(df.dnse), dammed=paired(df.dnse[dammed]), undammed=paired(df.dnse[~dammed]),
    dam_on_reach=paired(df.dnse[onreach]),
    smoke_dam=paired(df.dnse[df.smoke_role == "dam"]), smoke_control=paired(df.dnse[df.smoke_role == "control"]),
    dammed_surplus=paired(df.dnse[dammed & surplus]), dammed_deficit=paired(df.dnse[dammed & deficit]),
    undammed_surplus=paired(df.dnse[~dammed & surplus]), undammed_deficit=paired(df.dnse[~dammed & deficit]),
)
res["dKGE_leak_minus_base"] = dict(all=paired(df.dkge), dammed=paired(df.dkge[dammed]), undammed=paired(df.dkge[~dammed]))
res["dVOL_leak_minus_base"] = dict(all=paired(df.dvol), dammed=paired(df.dvol[dammed]), undammed=paired(df.dvol[~dammed]),
                                   surplus=paired(df.dvol[surplus]), deficit=paired(df.dvol[deficit]))
dor = pd.cut(df.nid_dor.where(dammed), [0, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
res["dNSE_by_DOR"] = {str(k): paired(v) for k, v in df.dnse[dammed].groupby(dor[dammed], observed=True)}
res["vol_base_by_group"] = dict(dammed=round(df.vol_base[dammed].median(), 3), undammed=round(df.vol_base[~dammed].median(), 3),
                                dammed_share_surplus=round(float(surplus[dammed].mean()), 3),
                                undammed_share_surplus=round(float(surplus[~dammed].mean()), 3))
m = dammed & df.dnse_dam_release.notna()
rho, p = spearmanr(df.dnse[m], df.dnse_dam_release[m])
res["spearman_leakgain_vs_damreleasegain_dammed"] = dict(rho=round(rho, 3), p=float(f"{p:.2g}"), n=int(m.sum()))
m2 = df.dnse.notna() & df.vol_base.notna()
rho2, p2 = spearmanr(df.dnse[m2], np.abs(np.log(df.vol_base[m2])))
res["spearman_leakgain_vs_abs_log_volbias_base"] = dict(rho=round(rho2, 3), p=float(f"{p2:.2g}"), n=int(m2.sum()))
rho3, p3 = spearmanr(df.dnse[m2], df.vol_base[m2])
res["spearman_leakgain_vs_volratio_base"] = dict(rho=round(rho3, 3), p=float(f"{p3:.2g}"))
u = mannwhitneyu(df.dnse[dammed].dropna(), df.dnse[~dammed].dropna())
res["mannwhitney_dnse_dammed_vs_undammed_p"] = float(f"{u.pvalue:.2g}")

# ---- per-reach zeta field on dam reaches ----------------------------------------------
ds = netCDF4.Dataset(RUNS / LEAK / "kan_parameters.nc")
z = pd.DataFrame({k: ds.variables[k][:] for k in ["COMID_eval", "zeta", "zeta_net", "depth_mean", "area_z_mean", "q_mean"]})
ds.close()
z = z.rename(columns={"COMID_eval": "COMID"}).set_index("COMID")
z["frac"] = z.zeta_net / z.q_mean.clip(lower=1e-3)
z["absfrac"] = z.zeta / z.q_mean.clip(lower=1e-3)
dams = pd.read_csv(AG / "experiments/reservoir/nid/nid_dams_in_eval_network.csv")
big = dams[dams.storage_mcm >= 10].groupby("COMID").storage_mcm.sum()
z["dam"] = z.index.isin(big.index)
z["storage"] = big.reindex(z.index).values
zres = dict(n_reaches=int(len(z)), n_dam_reaches=int(z.dam.sum()),
            share_losing_all=round(float((z.zeta_net > 0).mean()), 3), share_losing_dam=round(float((z.zeta_net[z.dam] > 0).mean()), 3),
            median_frac_all=round(float(z.frac.median()), 5), median_frac_dam=round(float(z.frac[z.dam].median()), 5),
            median_absfrac_all=round(float(z.absfrac.median()), 5), median_absfrac_dam=round(float(z.absfrac[z.dam].median()), 5),
            sum_zeta_net_all=round(float(z.zeta_net.sum()), 1), sum_zeta_net_dam=round(float(z.zeta_net[z.dam].sum()), 1),
            sum_abs_zeta_all=round(float(z.zeta.sum()), 1), sum_abs_zeta_dam=round(float(z.zeta[z.dam].sum()), 1))
# size-matched: compare dam reaches against non-dam reaches in the same q_mean decile
z["qdec"] = pd.qcut(z.q_mean.rank(method="first"), 10, labels=False)
per = []
for d, g in z.groupby("qdec"):
    a, b = g[g.dam], g[~g.dam]
    if len(a) < 5:
        continue
    per.append(dict(qdec=int(d), q_med=round(float(g.q_mean.median()), 2), n_dam=int(len(a)),
                    frac_dam=round(float(a.frac.median()), 5), frac_other=round(float(b.frac.median()), 5),
                    losing_dam=round(float((a.zeta_net > 0).mean()), 3), losing_other=round(float((b.zeta_net > 0).mean()), 3),
                    mw_p=float(f"{mannwhitneyu(a.frac, b.frac).pvalue:.2g}")))
zres["by_q_decile"] = per
top = z[z.dam].sort_values("frac")
zres["dam_reach_frac_quantiles"] = {q: round(float(z.frac[z.dam].quantile(q)), 4) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
zres["other_reach_frac_quantiles"] = {q: round(float(z.frac[~z.dam].quantile(q)), 4) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
res["zeta_field"] = zres
json.dump(res, open(OUT / "leak_vs_dams.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
