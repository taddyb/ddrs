#!/usr/bin/env python
"""Across-seed comparison of the learned dam release: seed 42 and seed 43, both arms.

    replicate.py        (after analysis.py --tag s42 and --tag s43, and the seed-43 parameter dumps)

Per gauge, the dam effect in each seed d_s = NSE(dam_s) - NSE(off_s), and the seed noise within an arm
NSE(off_43) - NSE(off_42), NSE(dam_43) - NSE(dam_42). Reports: the paired effect in each seed and pooled over seeds
(mean of the two per-gauge effects), all four cross-seed pairings of dam and no-dam arms, the rank correlation of the
per-gauge effect between seeds (dammed and undammed separately), the within-arm seed spread, and the learned
parameter medians of all four runs. Writes output/reservoir_full_run/web/build/replicate_numbers.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binomtest, spearmanr

HERE = Path(__file__).resolve().parent
WEB = HERE.parents[3] / "output/reservoir_full_run/web"
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
DOR_NAMES = ["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]


def paired(d, seed=42):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    b = np.median(np.random.default_rng(seed).choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=float(np.median(d)), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                n_up=up, n_down=int(len(nz) - up), sign_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else None,
                median_abs=float(np.median(np.abs(d))))


r42 = json.load(open(WEB / "build/results_s42.json"))
r43 = json.load(open(WEB / "build/results_s43.json"))
g42 = pd.read_csv(WEB / "build/gauges_s42.csv", dtype={"STAID": str}).set_index("STAID")
g43 = pd.read_csv(WEB / "build/gauges_s43.csv", dtype={"STAID": str}).set_index("STAID")
assert (g42.index == g43.index).all()
dm = g42.dammed.values
groups = {"all": np.ones(len(g42), bool), "dammed": dm, "undammed": ~dm, "on_reach": g42.on_reach.values,
          "further_up": dm & ~g42.on_reach.values}
for b in DOR_NAMES:
    groups["dor_" + b] = dm & (g42.dor_bin.astype(str) == b).values

o42, d42, o43, d43 = g42.nse_off.values, g42.nse_dam.values, g43.nse_off.values, g43.nse_dam.values
k_o42, k_d42, k_o43, k_d43 = g42.kge_off.values, g42.kge_dam.values, g43.kge_off.values, g43.kge_dam.values
e42, e43 = d42 - o42, d43 - o43
out = dict(runs=dict(s42=r42["runs"], s43=r43["runs"]), n=int(len(g42)))
out["median_nse"] = {s: {k: r["median"]["all"][k] for k in ("nse_base", "nse_off", "nse_dam", "kge_off", "kge_dam")} for s, r in (("s42", r42), ("s43", r43))}
out["median_nse_groups"] = {s: {g: {k: r["median"][g][k] for k in ("nse_off", "nse_dam")} for g in ("dammed", "undammed", "on_reach")}
                            for s, r in (("s42", r42), ("s43", r43))}
out["effect"] = {}
for g, m in groups.items():
    out["effect"][g] = dict(
        s42=paired(e42[m]), s43=paired(e43[m]), pooled=paired(((e42 + e43) / 2)[m]),
        cross_d42_o43=paired((d42 - o43)[m]), cross_d43_o42=paired((d43 - o42)[m]),
        kge_s42=paired((k_d42 - k_o42)[m]), kge_s43=paired((k_d43 - k_o43)[m]),
        rho=float(spearmanr(e42[m], e43[m], nan_policy="omit").statistic),
        same_sign=float(np.mean(np.sign(e42[m]) == np.sign(e43[m]))),
        noise_off=paired((o43 - o42)[m]), noise_dam=paired((d43 - d42)[m]),
    )
# is the per-gauge effect larger than the seed noise? per-gauge |effect| vs |off43 - off42|
out["abs"] = {g: dict(eff42=float(np.nanmedian(np.abs(e42[m]))), eff43=float(np.nanmedian(np.abs(e43[m]))),
                      noise_off=float(np.nanmedian(np.abs((o43 - o42)[m]))), noise_dam=float(np.nanmedian(np.abs((d43 - d42)[m]))))
              for g, m in groups.items()}
# parameters of all four runs
pars = {}
for tag, run in (("off42", r42["runs"]["off"]), ("dam42", r42["runs"]["learned"]), ("off43", r43["runs"]["off"]), ("dam43", r43["runs"]["learned"])):
    ds = xr.open_dataset(RUNS / run / "plot/kan_parameters.nc")
    pars[tag] = {v: dict(median=float(np.nanmedian(ds[v].values)), q25=float(np.nanpercentile(ds[v].values, 25)),
                         q75=float(np.nanpercentile(ds[v].values, 75))) for v in ("n", "gamma")}
    pars[tag]["_vals"] = {v: ds[v].values for v in ("n", "gamma")}
cmp = {}
for a, b in (("dam42", "off42"), ("dam43", "off43"), ("off43", "off42"), ("dam43", "dam42")):
    cmp[f"{a}-{b}"] = {}
    for v in ("n", "gamma"):
        d = pars[a]["_vals"][v] - pars[b]["_vals"][v]
        iqr = pars["off42"][v]["q75"] - pars["off42"][v]["q25"]
        cmp[f"{a}-{b}"][v] = dict(median=float(np.nanmedian(d)), median_abs_over_iqr=float(np.nanmedian(np.abs(d)) / iqr),
                                  frac_pos=float(np.mean(d > 0)))
for t in pars:
    pars[t].pop("_vals")
out["params"], out["param_diff"] = pars, cmp
# release parameters across seeds
rp42 = pd.read_csv(RUNS / r42["runs"]["learned"] / "release_params.csv").set_index("COMID")
rp43 = pd.read_csv(RUNS / r43["runs"]["learned"] / "release_params.csv").set_index("COMID")
j = rp42.join(rp43, lsuffix="_42", rsuffix="_43", how="inner")
out["release"] = dict(n=int(len(j)), T0_median_42=float(j.T0_days_42.median()), T0_median_43=float(j.T0_days_43.median()),
                      rho_T0=float(spearmanr(j.T0_days_42, j.T0_days_43).statistic),
                      rho_peak=float(spearmanr(np.arctan2(j.a_42, j.b_42), np.arctan2(j.a_43, j.b_43)).statistic),
                      ratio_median=float(np.median(j.T0_days_43 / j.T0_days_42)))
out["release"]["s43"] = r43["release"]
json.dump(out, open(WEB / "build/replicate_numbers.json", "w"), indent=1)
# per-gauge values for the page scatter
json.dump({s: [round(float(a), 6) if np.isfinite(a) else None, round(float(b), 6) if np.isfinite(b) else None]
           for s, a, b in zip(g43.index, o43, d43)}, open(WEB / "build/gauges_s43_nse.json", "w"), separators=(",", ":"))


def show(p):
    return f"{p['median']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] {p['n_up']}/{p['n_down']}"


for g in ("all", "dammed", "undammed", "on_reach", "further_up") + tuple("dor_" + b for b in DOR_NAMES):
    e = out["effect"][g]
    print(f"{g:>12} s42 {show(e['s42'])} | s43 {show(e['s43'])} | pooled {show(e['pooled'])} | x42-43 {e['cross_d42_o43']['median']:+.4f} "
          f"x43-42 {e['cross_d43_o42']['median']:+.4f} | rho {e['rho']:.2f} same {e['same_sign']:.2f} | noise off {show(e['noise_off'])}")
print(json.dumps(out["abs"], indent=0))
print(json.dumps(out["median_nse"], indent=0))
print(json.dumps(out["params"], indent=0))
print(json.dumps(out["param_diff"], indent=0))
print(json.dumps({k: v for k, v in out["release"].items() if k != "s43"}, indent=0))
