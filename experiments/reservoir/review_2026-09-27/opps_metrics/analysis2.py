#!/usr/bin/env python
"""Follow-ups on analysis.py: NSE decomposition, signed FDC deltas, over-smoothing, dammed-vs-undammed tests, outliers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

OUT = Path(__file__).resolve().parent
w = pd.read_csv(OUT / "per_gauge_metrics.csv", dtype={"STAID": str}).set_index("STAID")
gages = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str}).set_index("STAID")
gages.index = gages.index.str.zfill(8)

dammed = (w.n_nid_ge10mcm > 0).values
onreach = dammed & w.nid_on_gauge_reach.astype(bool).values
dor = pd.cut(w.nid_dor.where(dammed), [0, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
groups = {"dammed": dammed, "undammed": ~dammed, "dam_on_reach": onreach, "smoke_dam": (w.smoke_role == "dam").values,
          "smoke_control": (w.smoke_role == "control").values}
for lab in ["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]:
    groups[f"dor {lab}"] = (dor == lab).values

md, tables = [], {}


def med(x):
    return float(np.nanmedian(x))


# 1. Gupta (2009) decomposition: NSE = 2*alpha*r - alpha^2 - bn^2, bn = (mu_p - mu_o)/sigma_o
#    dNSE = 2*alpha_o*dr + (2*r_l - alpha_l - alpha_o)*dalpha - d(bn^2)
md.append("## Gupta decomposition of the per-gauge NSE change: correlation term, variance-ratio term, bias term (medians)\n")
rows = []
for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
    r_o, r_l = w[f"{b}__r"].values, w[f"{a}__r"].values
    al_o, al_l = w[f"{b}__alpha"].values, w[f"{a}__alpha"].values
    be_o, be_l = w[f"{b}__beta"].values, w[f"{a}__beta"].values
    # bn^2 = ((beta-1)*mu_o/sigma_o)^2; NSE - (2 alpha r - alpha^2) gives -bn^2 directly
    bn2_o = (2 * al_o * r_o - al_o**2) - w[f"{b}__nse"].values
    bn2_l = (2 * al_l * r_l - al_l**2) - w[f"{a}__nse"].values
    term_r = 2 * al_o * (r_l - r_o)
    term_a = (2 * r_l - al_l - al_o) * (al_l - al_o)
    term_b = -(bn2_l - bn2_o)
    total = w[f"{a}__nse"].values - w[f"{b}__nse"].values
    assert np.nanmax(np.abs(term_r + term_a + term_b - total)) < 1e-6
    w[f"d{s}_term_r"], w[f"d{s}_term_alpha"], w[f"d{s}_term_bias"] = term_r, term_a, term_b
    for g in ["dammed", "undammed", "dam_on_reach", "smoke_dam", "dor 0.1-0.5", "dor 0.5-1", "dor 1-2", "dor >2"]:
        sel = groups[g]
        # share of gauges where alpha term dominates the gain among gauges that gained > 0.005
        gain = sel & (total > 0.005)
        rows.append(dict(seed=s, group=g, n=int(sel.sum()), dNSE=med(total[sel]), term_r=med(term_r[sel]), term_alpha=med(term_a[sel]),
                         term_bias=med(term_b[sel]), n_gain_gt_0p005=int(gain.sum()),
                         frac_gainers_alpha_dominant=float(np.mean(term_a[gain] > term_r[gain])) if gain.sum() else np.nan,
                         alpha_off=med(al_o[sel]), r_off=med(r_o[sel]), alpha_learned=med(al_l[sel]),
                         frac_alpha_below_r_off=float(np.nanmean(al_o[sel] < r_o[sel])), frac_alpha_below_r_learned=float(np.nanmean(al_l[sel] < r_l[sel]))))
df = pd.DataFrame(rows)
md.append(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
tables["gupta"] = rows

# 2. signed FHV / FLV / peak ratio / flash deltas (group medians and paired medians)
md.append("\n\n## Signed FDC deltas (learned minus off): paired median of signed change, plus group medians\n")
rows = []
for k in ["fhv", "flv", "fms", "alpha", "peak_ratio", "flash_ratio", "recession_ratio", "ac1_pred"]:
    for g in ["dammed", "undammed", "dam_on_reach", "dor 0.5-1", "dor 1-2", "dor >2"]:
        sel = groups[g]
        row = dict(metric=k, group=g)
        for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
            d = w[f"{a}__{k}"].values - w[f"{b}__{k}"].values
            row[f"paired_{s}"] = med(d[sel])
            row[f"group_off_{s}"] = med(w[f"{b}__{k}"].values[sel])
            row[f"group_l_{s}"] = med(w[f"{a}__{k}"].values[sel])
        rows.append(row)
md.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
tables["signed"] = rows

# 3. over-smoothing: learned lag-1 autocorrelation above the observed
md.append("\n\n## Over-smoothing: fraction of gauges whose prediction is smoother than the observation (ac1_pred > ac1_obs), and NSE change there\n")
rows = []
for g in ["dammed", "undammed", "dam_on_reach", "dor 0.1-0.5", "dor 0.5-1", "dor 1-2", "dor >2"]:
    sel = groups[g]
    row = dict(group=g, n=int(sel.sum()))
    for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
        over_off = w[f"{b}__ac1_pred"].values > w[f"{a}__ac1_obs"].values
        over_l = w[f"{a}__ac1_pred"].values > w[f"{a}__ac1_obs"].values
        d = w[f"{a}__nse"].values - w[f"{b}__nse"].values
        row[f"frac_over_off_{s}"] = float(np.mean(over_off[sel]))
        row[f"frac_over_learned_{s}"] = float(np.mean(over_l[sel]))
        row[f"dNSE_where_over_l_{s}"] = med(d[sel & over_l])
        row[f"dNSE_where_not_over_l_{s}"] = med(d[sel & ~over_l])
        row[f"n_became_over_{s}"] = int((sel & ~over_off & over_l).sum())
        row[f"dNSE_became_over_{s}"] = med(d[sel & ~over_off & over_l])
    rows.append(row)
md.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
tables["oversmooth"] = rows

# 4. dammed vs undammed delta distributions: Mann-Whitney (the null that respects the trajectory confound)
md.append("\n\n## Dammed vs undammed per-gauge deltas: Mann-Whitney p (two-sided), medians\n")
rows = []
for k in ["nse", "kge", "nse_diff", "nse_anomaly", "nse_seasonal", "r", "alpha", "flash_ratio", "fhv", "flv"]:
    for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
        d = w[f"{a}__{k}"].values - w[f"{b}__{k}"].values
        x, y = d[dammed & np.isfinite(d)], d[~dammed & np.isfinite(d)]
        rows.append(dict(metric=k, seed=s, dammed=med(x), undammed=med(y), mwu_p=float(mannwhitneyu(x, y).pvalue),
                         onreach=med(d[onreach]), mwu_p_onreach_vs_undammed=float(mannwhitneyu(d[onreach & np.isfinite(d)], y).pvalue)))
md.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.4g}"))
tables["mwu"] = rows

# 5. heavy tails: fraction of gauges with |dNSE| > thresholds; contribution of top gauges to the mean
md.append("\n\n## Heavy tails of the NSE delta\n")
rows = []
for g in ["dammed", "undammed"]:
    sel = groups[g]
    for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
        d = w[f"{a}__nse"].values[sel] - w[f"{b}__nse"].values[sel]
        d = d[np.isfinite(d)]
        srt = np.sort(d)
        rows.append(dict(group=g, seed=s, n=len(d), frac_gt_0p01=float(np.mean(d > 0.01)), frac_lt_m0p01=float(np.mean(d < -0.01)),
                         frac_gt_0p05=float(np.mean(d > 0.05)), frac_lt_m0p05=float(np.mean(d < -0.05)),
                         mean=float(d.mean()), mean_without_top1=float(np.delete(d, np.argmax(np.abs(d))).mean()),
                         mean_clip_pm1=float(np.clip(d, -1, 1).mean()), median=float(np.median(d)),
                         hodges_lehmann=float(np.median((srt[:, None] + srt[None, :])[np.triu_indices(len(srt))])) if len(srt) < 1500 else np.nan))
md.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
tables["tails"] = rows

# 6. outlier gauges
md.append("\n\n## Largest |dNSE| gauges (either seed)\n")
d42 = w["l42__nse"] - w["off42__nse"]
d43 = w["l43__nse"] - w["off43__nse"]
big = pd.DataFrame(dict(d42=d42, d43=d43, off42=w["off42__nse"], off43=w["off43__nse"], base=w["base__nse"], dor=w.nid_dor, on_reach=w.nid_on_gauge_reach,
                        area=w.area_km2, name=gages.STANAME.reindex(w.index)))
big["maxabs"] = np.maximum(big.d42.abs(), big.d43.abs())
md.append(big.sort_values("maxabs", ascending=False).head(15).to_string(float_format=lambda v: f"{v:+.3f}"))

# 7. seed-consistency of the sub-daily metrics too
md.append("\n\n## Seed consistency (Spearman of per-gauge deltas d42 vs d43) by metric, dammed vs undammed\n")
rows = []
for k in ["nse", "kge", "nse_diff", "nse_anomaly", "r", "alpha", "flash_ratio", "fhv", "flv", "peak_ratio", "recession_ratio"]:
    da = (w[f"l42__{k}"] - w[f"off42__{k}"]).values
    db = (w[f"l43__{k}"] - w[f"off43__{k}"]).values
    row = dict(metric=k)
    for g in ["dammed", "undammed", "dam_on_reach", "smoke_dam"]:
        sel = groups[g] & np.isfinite(da) & np.isfinite(db)
        row[g] = float(spearmanr(da[sel], db[sel]).correlation)
    rows.append(row)
md.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
tables["seed_spearman"] = rows

# 8. dNSE vs learned T0 at on-reach dams, all 377, via smoke set nearest dam (only for smoke dam gauges we know the dam COMID) done in analysis.py
(OUT / "tables2.md").write_text("\n".join(md))
json.dump(tables, open(OUT / "tables2.json", "w"), indent=1, default=float)
print("\n".join(md))
