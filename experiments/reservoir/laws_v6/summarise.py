"""Paired summaries of laws_by_gauge.csv: medians with 2,000-draw bootstrap CIs, sign counts, clipped means,
controls (same law fitted at the matched control), difference-in-differences, KGE and r/alpha/beta, period bands.
Writes summary.txt and summary.json."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import v6

CSV = sys.argv[1] if len(sys.argv) > 1 else v6.HERE + "/laws_by_gauge.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else v6.HERE + "/summary"
F = pd.read_csv(CSV, dtype={"STAID": str}).set_index("STAID")
sg, dam, ctl = v6.gauges()
ctl_of = ctl.reset_index().set_index("control_for").STAID
rng = np.random.default_rng(0)


def boot(x, B=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def clip(v):
    return np.maximum(v, -1.0)


def paired(ids, law, ref, metric="nse"):
    ids = [s for s in ids if s in F.index and ctl_of.get(s) in F.index]
    c = [ctl_of[s] for s in ids]
    a = F.loc[ids, f"{law}_{metric}"].values - F.loc[ids, f"{ref}_{metric}"].values
    b = F.loc[c, f"{law}_{metric}"].values - F.loc[c, f"{ref}_{metric}"].values
    ac = clip(F.loc[ids, f"{law}_{metric}"].values) - clip(F.loc[ids, f"{ref}_{metric}"].values)
    up, dn = int((a > 0).sum()), int((a < 0).sum())
    p = binomtest(up, up + dn).pvalue if up + dn else np.nan
    return dict(n=len(ids), med=boot(a), up=up, dn=dn, p=p, cmean=float(ac.mean()), ctl=boot(b), did=boot(a - b),
                ctl_up=int((b > 0).sum()), ctl_dn=int((b < 0).sum()))


GROUPS = {
    "flood-control on-reach DOR>0.5": dam[(dam.dam_purpose == "Flood Risk Reduction") & dam.on_reach & (dam.nid_dor > 0.5)].index,
    "all on-reach DOR>0.5": dam[dam.on_reach & (dam.nid_dor > 0.5)].index,
    "flood-control DOR>0.5 (all)": dam[(dam.dam_purpose == "Flood Risk Reduction") & (dam.nid_dor > 0.5)].index,
    "flood-control, all": dam[dam.dam_purpose == "Flood Risk Reduction"].index,
    "irrigation, all": dam[dam.dam_purpose == "Irrigation"].index,
    "irrigation, on-reach": dam[(dam.dam_purpose == "Irrigation") & dam.on_reach].index,
    "water supply, all": dam[dam.dam_purpose == "Water Supply"].index,
    "all 458 dams": dam.index,
}


def fmt(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"


def row(ids, law, ref):
    r = paired(ids, law, ref)
    k = paired(ids, law, ref, "kge")
    out = (f"  {law:5s} vs {ref:4s} {fmt(r['med'])} {r['up']:3d}/{r['dn']:<3d} p={r['p']:.0e} cmean {r['cmean']:+.4f} | "
           f"ctl {r['ctl'][0]:+.4f} ({r['ctl_up']}/{r['ctl_dn']}) | DiD {fmt(r['did'])} | dKGE {k['med'][0]:+.4f}")
    extra = []
    for m in ["r", "alpha", "beta"]:
        extra.append(f"d{m} {paired(ids, law, ref, m)['med'][0]:+.4f}")
    return out + " " + " ".join(extra), dict(nse=r, kge=k)


def bands(ids, laws):
    ids = [s for s in ids if s in F.index]
    c = [ctl_of[s] for s in ids if ctl_of.get(s) in F.index]
    lines = [f"  mean normalised MSE by period band {v6.BAND_LAB} (test years)"]
    for law in laws:
        a = [F.loc[ids, f"{law}_band_{b}"].mean() for b in v6.BAND_LAB]
        b = [F.loc[c, f"{law}_band_{b}"].mean() for b in v6.BAND_LAB]
        lines.append(f"    {law:5s} dam " + " ".join(f"{x:.3f}" for x in a) + " | ctl " + " ".join(f"{x:.3f}" for x in b))
    return lines


FLOOD = ["FA", "FB", "FC", "FA4", "FC4"]
WITH = ["W1", "W2", "W3"]
NULL = ["Kg", "Kh", "Kc", "L2Kg", "L2Kh", "L2Kc", "L2Kj"]
res, lines = {}, []
for gname, ids in GROUPS.items():
    lines.append(f"\n=========== {gname} (n={len(ids)})")
    lev = F.loc[[s for s in ids if s in F.index]]
    lines.append("  median test NSE: " + " ".join(f"{l} {lev[l + '_nse'].median():.3f}" for l in ["L0", "L2", "L4", "FA", "FB", "FC", "FA4", "W1", "L2Kg"]))
    res[gname] = {}
    pairs = [("L2", "L0"), ("L4", "L0"), ("L4", "L2")] + [(l, "L0") for l in FLOOD] + [(l, "L2") for l in FLOOD] + \
            [("FA4", "L4"), ("FC4", "L4")]
    if "irrigation" in gname or "water" in gname or "all 458" in gname:
        pairs += [(l, "L2") for l in WITH] + [(l, "L0") for l in NULL] + [("W1", "L2Kg"), ("W2", "L2Kg"), ("W3", "L2Kg"),
                                                                         ("W1", "L2Kj"), ("W3", "L2Kj")]
    for law, ref in pairs:
        s, d = row(ids, law, ref)
        lines.append(s)
        res[gname][f"{law}-{ref}"] = d
    lines += bands(ids, ["L0", "L2", "L4", "FA", "FB", "FC", "FA4", "FC4"] + (WITH + ["L2Kg"] if "irrig" in gname else []))
    # fitted parameters and fluxes
    for law in ["FA", "FB", "FC"]:
        lines.append(f"  {law} params median [IQR]: " + "; ".join(
            f"{k} {lev[f'{law}_p_{k}'].median():.3g} [{lev[f'{law}_p_{k}'].quantile(.25):.3g},{lev[f'{law}_p_{k}'].quantile(.75):.3g}]"
            for k in ["T0", "kc", "phi", "z"]) + f"; captured share of test inflow {lev[f'{law}_cap_share'].median():.4f}"
                     f" (p90 {lev[f'{law}_cap_share'].quantile(.9):.4f}); max pool {lev[f'{law}_Fmax_days'].median():.2f} d of Ibar")
    for law in WITH:
        lines.append(f"  {law} w median {lev[f'{law}_p_w'].median():.3f} [IQR {lev[f'{law}_p_w'].quantile(.25):.3f},"
                     f"{lev[f'{law}_p_w'].quantile(.75):.3f}], share w=0 {(lev[f'{law}_p_w'] < 1e-3).mean():.2f}, "
                     f"withdrawn share of test inflow {lev[f'{law}_w_share'].median():.4f}")
    lines.append(f"  per-gauge scalar Kg median {lev.Kg_p_k.median():.3f} [IQR {lev.Kg_p_k.quantile(.25):.3f},{lev.Kg_p_k.quantile(.75):.3f}]")
open(OUT + ".txt", "w").write("\n".join(lines) + "\n")
json.dump(res, open(OUT + ".json", "w"), indent=1, default=float)
print("\n".join(lines))
