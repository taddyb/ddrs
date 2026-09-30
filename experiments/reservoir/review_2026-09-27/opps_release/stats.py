"""Volume / variance audit of the smoke set and the learned release parameters.

Inputs (read-only):
  reservoir-options worktree: experiments/reservoir/smoke/expected_release_fit.csv, smoke_gauges.csv,
                              experiments/reservoir/nid/nid_dams_in_eval_network.csv
  run 2026-09-27T07-29-55Z: release_params.csv (learned, seed 42); 10-31-50Z (seed 43)
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs"
out = {}

fit = pd.read_csv(f"{WT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str, "huc2": str, "control_for": str})
dam = fit[fit.role == "dam"]
ctl = fit[fit.role == "control"]
onr = dam[dam.on_reach.astype(str) == "True"]


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 4) for v in np.percentile(x, [10, 25, 50, 75, 90])]


for name, g in [("dam", dam), ("on_reach_dam", onr), ("control", ctl)]:
    out[name] = dict(
        n=len(g),
        beta_pass_pct=q(g.beta_pass_test),
        frac_beta_pass_gt_1_1=round(float((g.beta_pass_test > 1.1).mean()), 3),
        frac_beta_pass_lt_0_9=round(float((g.beta_pass_test < 0.9).mean()), 3),
        alpha_pass_pct=q(g.alpha_pass_test),
        alpha_seas_pct=q(g.alpha_seas_test),
        d_alpha_seas_minus_pass_pct=q(g.alpha_seas_test - g.alpha_pass_test),
        r_pass_pct=q(g.r_pass_test),
        r_seas_pct=q(g.r_seas_test),
        d_r_seas_minus_pass_pct=q(g.r_seas_test - g.r_pass_test),
        dkge_seas_pct=q(g.dk_seas),
        dnse_seas_pct=q(g.d_seas),
        frac_alpha_pass_lt_1=round(float((g.alpha_pass_test < 1).mean()), 3),
    )

# Where does the seasonal bucket hurt KGE while helping NSE?
both = dam[(dam.d_seas > 0.01)]
out["dam_gauges_with_dNSE_gt_0.01"] = dict(
    n=len(both),
    median_dkge=round(float(both.dk_seas.median()), 4),
    median_dalpha=round(float((both.alpha_seas_test - both.alpha_pass_test).median()), 4),
    median_dr=round(float((both.r_seas_test - both.r_pass_test).median()), 4),
    frac_alpha_pass_lt_1=round(float((both.alpha_pass_test < 1).mean()), 3),
)

# Is the pass-through model's volume bias different at dams vs their matched controls?
ctl_beta = ctl.set_index("control_for").beta_pass_test
pair = dam.set_index("STAID")
pair_beta_ctl = pair.index.map(ctl_beta)
d = np.log(pair.beta_pass_test.values) - np.log(np.asarray(pair_beta_ctl, float))
d = d[np.isfinite(d)]
out["log_beta_dam_minus_matched_control"] = dict(n=int(len(d)), pct=q(d), frac_pos=round(float((d > 0).mean()), 3))

# NID max discharge availability and its ratio to mean flow at the on-reach dams
sg = pd.read_csv(f"{WT}/experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str, "huc2": str}).set_index("STAID")
onr_i = onr.set_index("STAID")
mx = sg.loc[onr_i.index, "dam_max_discharge_m3s"]
out["nid_max_discharge_on_reach"] = dict(n=len(mx), n_finite=int(mx.notna().sum()), pct_m3s=q(mx))

# Learned release parameters vs NID features (full population, two seeds)
nid = pd.read_csv(f"{WT}/experiments/reservoir/nid/nid_dams_in_eval_network.csv")
nid = nid[nid.storage_mcm >= 10]
agg = nid.groupby("COMID").agg(storage_mcm=("storage_mcm", "sum"), uparea=("reach_uparea_km2", "max"),
                               max_discharge=("max_discharge_m3s", "sum"), height=("height_m", "max"),
                               purpose=("primary_purpose", "first"), year=("year", "max"))
for seed, run in [(42, "2026-09-27T07-29-55Z-train-and-test"), (43, "2026-09-27T10-31-50Z-train-and-test")]:
    try:
        rp = pd.read_csv(f"{RUNS}/{run}/release_params.csv").set_index("COMID")
    except FileNotFoundError:
        out[f"seed{seed}"] = "no release_params.csv"
        continue
    j = rp.join(agg, how="inner")
    amp = np.sqrt(j.a ** 2 + j.b ** 2)
    phase = np.degrees(np.arctan2(j.a, j.b))
    res = dict(
        n=len(j),
        T0_hours_pct=q(j.T0_days * 24),
        frac_T0_within_10pct_of_init_4p5h=round(float((np.abs(j.T0_days * 24 / 4.5 - 1) < 0.1).mean()), 3),
        frac_T0_gt_1d=round(float((j.T0_days > 1).mean()), 3),
        amp_pct=q(amp),
        spearman_T0_log_storage=round(float(spearmanr(np.log10(j.storage_mcm), j.T0_days).correlation), 3),
        spearman_T0_log_storage_per_area=round(float(spearmanr(np.log10(j.storage_mcm / j.uparea), j.T0_days).correlation), 3),
        spearman_T0_height=round(float(spearmanr(j.height.fillna(j.height.median()), j.T0_days).correlation), 3),
        T0_hours_by_purpose={k: [int(len(v)), round(float(v.median() * 24), 2)] for k, v in j.groupby("purpose").T0_days if len(v) >= 10},
        # sign of the seasonal term: T largest at which day of year? phi where a sin w + b cos w is max: w* = atan2(a, b)
        doy_of_max_T_pct=q(((phase % 360) / 360 * 365.25)),
    )
    out[f"seed{seed}"] = res
    globals()[f"rp{seed}"] = rp
if "rp42" in globals() and "rp43" in globals():
    jj = rp42.join(rp43, lsuffix="_42", rsuffix="_43", how="inner")
    out["seed42_vs_43"] = dict(
        n=len(jj),
        spearman_T0=round(float(spearmanr(jj.T0_days_42, jj.T0_days_43).correlation), 3),
        median_abs_log_ratio_T0=round(float(np.median(np.abs(np.log(jj.T0_days_42 / jj.T0_days_43)))), 3),
        spearman_a=round(float(spearmanr(jj.a_42, jj.a_43).correlation), 3),
        spearman_b=round(float(spearmanr(jj.b_42, jj.b_43).correlation), 3),
        frac_amp_sign_agree_a=round(float((np.sign(jj.a_42) == np.sign(jj.a_43)).mean()), 3),
        frac_amp_sign_agree_b=round(float((np.sign(jj.b_42) == np.sign(jj.b_43)).mean()), 3),
    )

# Learned T0 (seed 42) against the offline seasonal fit's T0 at the nearest on-reach dam
if "rp42" in globals():
    o = onr_i.join(sg[["dam_COMID"]], rsuffix="_sg")
    o["learned_T0"] = o.dam_COMID.map(rp42.T0_days)
    o = o[o.learned_T0.notna()]
    active = o[o.seas_T0 > 0.051]
    out["on_reach_learned_vs_fit"] = dict(
        n=len(o), n_fit_active=len(active),
        median_fit_T0_active=round(float(active.seas_T0.median()), 3),
        median_learned_T0_active=round(float(active.learned_T0.median()), 3),
        median_log10_ratio_fit_over_learned=round(float(np.log10(active.seas_T0 / active.learned_T0).median()), 3),
        spearman=round(float(spearmanr(o.seas_T0, o.learned_T0).correlation), 3),
        frac_fit_T0_gt_5d=round(float((o.seas_T0 > 5).mean()), 3),
        frac_learned_T0_gt_5d=round(float((o.learned_T0 > 5).mean()), 3),
    )

print(json.dumps(out, indent=1))
json.dump(out, open("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release/stats.json", "w"), indent=1)
