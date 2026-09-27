"""Summarise fits_by_gauge.csv (the per-gauge output of fits.py)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
OUT = "/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release"


def paired(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    nz = d[np.abs(d) > 1e-9]; up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    bs = np.median(rng.choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=round(float(np.median(d)), 5), ci=[round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
                n_up=up, n_down=int(len(nz) - up), sign_p=round(float(binomtest(up, len(nz)).pvalue), 4) if len(nz) else None)


def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.percentile(x, [10, 25, 50, 75, 90])] if len(x) else None


df = pd.read_csv(f"{OUT}/fits_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
fit = pd.read_csv(f"{WT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str, "control_for": str}).set_index("STAID")
neutral = {"inflowT": 0.0, "storeT": 0.0, "loss": 0.0, "cap_nid": np.inf, "cap_mean": np.inf, "spill": np.inf}
res = {"n": dict(dam=int((df.role == "dam").sum()), control=int((df.role == "control").sum())), "laws": {}}
d_all = df[df.role == "dam"]; c_all = df[df.role == "control"]
for law in ["cap_nid", "cap_mean", "inflowT", "storeT", "loss", "spill"]:
    entry = {}
    for role, g in [("dam", d_all), ("control", c_all)]:
        if f"{law}_nse" not in g:
            continue
        g = g[g[f"{law}_nse"].notna()]
        th = g[f"{law}_theta"]
        entry[role] = dict(
            dnse_vs_seas=paired(g[f"{law}_nse"] - g["seas_nse"]),
            dkge_vs_seas=paired(g[f"{law}_kge"] - g["seas_kge"]),
            dalpha_vs_seas=paired(g[f"{law}_alpha"] - g["seas_alpha"]),
            dbeta_vs_seas=paired(g[f"{law}_beta"] - g["seas_beta"]),
            dnse_vs_pass=paired(g[f"{law}_nse"] - g["pass_nse"]),
            median_nse=round(float(g[f"{law}_nse"].median()), 4),
            theta_pct=pct(th.replace(np.inf, np.nan)),
            frac_theta_neutral=round(float((th == neutral[law]).mean()), 3),
            frac_dnse_gt_0_01=round(float(((g[f"{law}_nse"] - g["seas_nse"]) > 0.01).mean()), 3),
            frac_dnse_lt_m0_01=round(float(((g[f"{law}_nse"] - g["seas_nse"]) < -0.01).mean()), 3),
            train_gain_vs_seas_median=round(float((g[f"{law}_train_nse"] - g["seas_train_nse"]).median()), 4),
        )
    if f"{law}_nse" in d_all and f"{law}_nse" in c_all:
        cmap = pd.Series((c_all[f"{law}_nse"] - c_all["seas_nse"]).values, index=fit.loc[c_all.index, "control_for"].values)
        dd = (d_all[f"{law}_nse"] - d_all["seas_nse"]) - d_all.index.map(cmap).astype(float)
        entry["dam_minus_matched_control_vs_seas"] = paired(dd)
    res["laws"][law] = entry
res["seas_regrid"] = dict(
    dam_dnse_vs_pass=paired(d_all.seas_nse - d_all.pass_nse), dam_median_nse=round(float(d_all.seas_nse.median()), 4),
    dam_dkge_vs_pass=paired(d_all.seas_kge - d_all.pass_kge), dam_dalpha_vs_pass=paired(d_all.seas_alpha - d_all.pass_alpha),
    ctl_dnse_vs_pass=paired(c_all.seas_nse - c_all.pass_nse),
)
# Laws by regulation band and by purpose at dams
sg = pd.read_csv(f"{WT}/experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
dd = d_all.join(sg[["nid_dor", "dam_purpose", "dam_storage_mcm"]])
bands = pd.cut(dd.nid_dor, [0, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
res["by_dor"] = {law: {str(k): paired(v[f"{law}_nse"] - v["seas_nse"]) for k, v in dd.groupby(bands, observed=True)}
                 for law in ["cap_nid", "inflowT", "storeT", "loss", "spill"]}
res["by_purpose"] = {law: {str(k): paired(v[f"{law}_nse"] - v["seas_nse"]) for k, v in dd.groupby("dam_purpose") if len(v) >= 8}
                     for law in ["cap_nid", "inflowT", "storeT", "loss", "spill"]}
# Is the loss fraction wanted where pass-through beta > 1 only?
g = d_all
res["loss_vs_beta"] = dict(
    spearman_loss_theta_vs_pass_beta=round(float(pd.Series(g.loss_theta.values).corr(pd.Series(g.pass_beta.values), method="spearman")), 3),
    dam_beta_gt_1_1=paired((g.loss_nse - g.seas_nse)[g.pass_beta > 1.1]),
    dam_beta_le_1_1=paired((g.loss_nse - g.seas_nse)[g.pass_beta <= 1.1]),
    ctl_beta_gt_1_1=paired((c_all.loss_nse - c_all.seas_nse)[c_all.pass_beta > 1.1]),
)
# Best single law per gauge (train-selected): does picking by TRAIN NSE generalise?
laws = ["seas", "cap_nid", "inflowT", "storeT", "loss", "spill"]
tr = d_all[[f"{l}_train_nse" for l in laws]].to_numpy()
te = d_all[[f"{l}_nse" for l in laws]].to_numpy()
best = np.nanargmax(np.where(np.isfinite(tr), tr, -np.inf), axis=1)
picked = te[np.arange(len(best)), best]
res["train_selected_best_law"] = dict(dnse_vs_seas=paired(picked - d_all.seas_nse.values),
                                      counts={laws[i]: int((best == i).sum()) for i in range(len(laws))})
json.dump(res, open(f"{OUT}/fits.json", "w"), indent=1)
print(json.dumps(res, indent=1))
