import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

WT = Path("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options")
AG = Path("/home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4")
OUT = Path("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_leakance")
per = pd.read_csv(OUT / "leak_vs_base_per_gauge.csv", dtype={"STAID": str}).set_index("STAID")
s42 = pd.read_csv(WT / "experiments/reservoir/full_run/paired_full_run.csv", dtype={"STAID": str}).set_index("STAID")
s43 = pd.read_csv(WT / "experiments/reservoir/full_run/paired_full_run_seed43.csv", dtype={"STAID": str}).set_index("STAID")
per["dr42"] = s42.dnse
per["dr43"] = s43.dnse
dam = per.n_nid_ge10mcm > 0
res = {}
for nm, col in [("seed42_same_base", "dr42"), ("seed43_other_base", "dr43")]:
    for grp, m in [("dammed", dam), ("undammed", ~dam)]:
        d = per[m].dropna(subset=["dnse", col])
        rho, p = spearmanr(d.dnse, d[col])
        res[f"spearman_leakgain_vs_damrelease_{nm}_{grp}"] = dict(rho=round(float(rho), 3), p=float(f"{p:.2g}"), n=int(len(d)))
# seed42 vs seed43 dam-release gains themselves (how reproducible is the dam-release gain per gauge)
d = per.dropna(subset=["dr42", "dr43"])
res["spearman_damrelease_seed42_vs_seed43_all"] = round(float(spearmanr(d.dr42, d.dr43)[0]), 3)
res["spearman_damrelease_seed42_vs_seed43_dammed"] = round(float(spearmanr(d.dr42[dam[d.index]], d.dr43[dam[d.index]])[0]), 3)

# volume surplus: smoke dam gauges vs matched controls, by DOR of the dam gauge
sm = pd.read_csv(AG / "experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str, "control_for": str}).set_index("STAID")
dams = sm[sm.role == "dam"]
ctrl = sm[sm.role == "control"]
pairs = pd.DataFrame({"dam": ctrl.control_for.values, "ctrl": ctrl.index.values})
pairs["vol_dam"] = per.vol_base.reindex(pairs.dam).values
pairs["vol_ctrl"] = per.vol_base.reindex(pairs.ctrl).values
pairs["dor"] = sm.nid_dor.reindex(pairs.dam).values
pairs["dlogvol"] = np.log(pairs.vol_dam) - np.log(pairs.vol_ctrl)
pairs = pairs.dropna()
bins = pd.cut(pairs.dor, [-1e-9, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
out = {}
for k, g in pairs.groupby(bins, observed=True):
    out[str(k)] = dict(n=int(len(g)), median_vol_dam=round(float(g.vol_dam.median()), 3), median_vol_ctrl=round(float(g.vol_ctrl.median()), 3),
                       median_paired_dlogvol=round(float(g.dlogvol.median()), 3),
                       share_dam_more_surplus=round(float((g.dlogvol > 0).mean()), 3))
out["all"] = dict(n=int(len(pairs)), median_vol_dam=round(float(pairs.vol_dam.median()), 3), median_vol_ctrl=round(float(pairs.vol_ctrl.median()), 3),
                  median_paired_dlogvol=round(float(pairs.dlogvol.median()), 3), share_dam_more_surplus=round(float((pairs.dlogvol > 0).mean()), 3))
res["smoke_pairs_volume_ratio_base_arm"] = out
# leakance arm dNSE at smoke dam gauges vs their controls, paired, by surplus of the dam gauge
pairs["dnse_dam"] = per.dnse.reindex(pairs.dam).values
pairs["dnse_ctrl"] = per.dnse.reindex(pairs.ctrl).values
pairs["dd"] = pairs.dnse_dam - pairs.dnse_ctrl
res["leak_dnse_dam_minus_control_smoke"] = dict(n=int(pairs.dd.notna().sum()), median=round(float(pairs.dd.median()), 4),
                                               dam_ahead=int((pairs.dd > 0).sum()), ctrl_ahead=int((pairs.dd < 0).sum()))
hi = pairs[pairs.dor > 0.5]
res["leak_dnse_dam_minus_control_smoke_dor_gt_0.5"] = dict(n=int(len(hi)), median=round(float(hi.dd.median()), 4),
                                                          dam_ahead=int((hi.dd > 0).sum()), ctrl_ahead=int((hi.dd < 0).sum()))
json.dump(res, open(OUT / "artifact_checks.json", "w"), indent=1)
print(json.dumps(res, indent=1))
