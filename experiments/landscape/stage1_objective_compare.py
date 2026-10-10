#!/usr/bin/env python
"""Side-by-side of the KGE-objective and NSE-objective Stage 1 censuses.

Usage: stage1_objective_compare.py <kge-merged> <nse-merged> --out DIR
Reads <merged>/figures/stage1/{stage1_per_gauge.csv,stage1_tables.json}.
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("kge", type=Path); ap.add_argument("nse", type=Path)
ap.add_argument("--out", type=Path, required=True); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
T = {k: json.loads((p / "figures/stage1/stage1_tables.json").read_text()) for k, p in (("KGE", a.kge), ("NSE", a.nse))}
D = {k: pd.read_csv(p / "figures/stage1/stage1_per_gauge.csv", dtype={"staid": str}) for k, p in (("KGE", a.kge), ("NSE", a.nse))}
pops = ["nse0_gt_0.3", "all", "kge0 KGE < 0", "kge0 0 to 0.3", "kge0 0.3 to 0.6", "kge0 >= 0.6"]
keys = ["n_gauges", "flat_q_0.5pct", "flat_q_1pct", "flat_q_2pct", "flat_p_1pct", "flat_n_1pct", "H0_concave_n", "H0_concave_p", "H0_concave_q",
        "Hs_concave_n", "Hs_concave_p", "Hs_concave_q", "edge_p", "edge_q", "ratio_p_to_n_trained_median", "ratio_q_to_n_trained_median",
        "ratio_p_to_n_optimum_median", "ratio_q_to_n_optimum_median", "indef_trained", "indef_star", "lam1_le0_star",
        "v1_n_median", "v1_p_median", "v1_q_median", "stiff_b_over_a_median", "stiff_c_over_a_median", "stiff_n_dominant_share", "hit_bound"]
rows = []
for k in keys:
    r = {"quantity": k}
    for pop in pops:
        for o in ("KGE", "NSE"):
            r[f"{pop} | {o}"] = T[o][pop].get(k)
    rows.append(r)
pd.DataFrame(rows).to_csv(a.out / "stage1_objective_compare.csv", index=False)
# extra: stiff angle, same-gauge agreement
for o, d in D.items():
    ok = d[(d.lam1 > 0) & (d.nse0 > 0.3)]
    print(o, "stiff angle median deg (well fit)", round(float(np.degrees(np.arctan2(ok.v1_p, ok.v1_n)).median()), 1))
m = D["KGE"].merge(D["NSE"], on="staid", suffixes=("_k", "_n"))
wf = m[m.nse0_k > 0.3]
print("same-gauge: n", len(wf), "flat_q agree", float(((wf.dL2_q_k < .01) == (wf.dL2_q_n < .01)).mean()),
      "spearman dL2_q", float(wf.dL2_q_k.corr(wf.dL2_q_n, method="spearman")),
      "median loss0 KGE/NSE", float(wf.loss0_k.median()), float(wf.loss0_n.median()),
      "alpha_q* same sign", float((np.sign(wf.alpha_q_k) == np.sign(wf.alpha_q_n)).mean()),
      "spearman alpha_n*", float(wf.alpha_n_k.corr(wf.alpha_n_n, method="spearman")),
      "spearman alpha_q*", float(wf.alpha_q_k.corr(wf.alpha_q_n, method="spearman")))
# absolute loss change for factor two in q: dL2*loss0
for o, d in D.items():
    wf2 = d[d.nse0 > 0.3]; ab = wf2.dL2_q * wf2.loss0
    print(o, "abs dL (factor-2 in q) median", float(ab.median()), "share abs<0.001", float((ab < 1e-3).mean()), "share abs<0.0025",float((ab<2.5e-3).mean()), "loss0 median", float(wf2.loss0.median()))
# figure: CDF of dL2_q by objective (well fit) and flat share by KGE band
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for o, c in (("KGE", "#2166ac"), ("NSE", "#b2182b")):
    d = D[o]; wf2 = d[d.nse0 > 0.3]
    for nm, ls in (("q", "-"), ("p", "--"), ("n", ":")):
        r = np.sort(wf2[f"dL2_{nm}"].values); ax[0].plot(r, np.arange(1, len(r) + 1) / len(r), color=c, ls=ls, label=f"{nm}, {o}")
ax[0].set_xscale("log"); ax[0].axvline(.01, color="k", lw=.7); ax[0].set_xlabel("relative loss change, factor-two move"); ax[0].set_ylabel("cumulative share (NSE>0.3 gauges)"); ax[0].legend(fontsize=7); ax[0].set_title("how flat, by objective")
bl = ["kge0 KGE < 0", "kge0 0 to 0.3", "kge0 0.3 to 0.6", "kge0 >= 0.6", "nse0_gt_0.3"]
x = np.arange(len(bl))
for i, (o, c) in enumerate((("KGE", "#2166ac"), ("NSE", "#b2182b"))):
    ax[1].bar(x + i * .38 - .19, [T[o][b]["flat_q_1pct"] for b in bl], .38, color=c, label=o)
ax[1].axhline(.85, color="k", ls="--", lw=.8); ax[1].text(4.45, .86, "old 85 %", ha="right", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(["KGE<0", "0-0.3", "0.3-0.6", ">=0.6", "NSE>0.3"], fontsize=8); ax[1].set_ylabel("share with q flat (1 %)"); ax[1].legend(); ax[1].set_title("q flat share by band")
sc = ax[2].scatter(wf.dL2_q_k, wf.dL2_q_n, s=5, alpha=.4); ax[2].set_xscale("log"); ax[2].set_yscale("log")
lo, hi = 1e-6, 10; ax[2].plot([lo, hi], [lo, hi], "k", lw=.6); ax[2].axvline(.01, color="gray", lw=.5); ax[2].axhline(.01, color="gray", lw=.5)
ax[2].set_xlabel("KGE objective"); ax[2].set_ylabel("NSE objective"); ax[2].set_title("per-gauge q flatness, same gauge")
fig.tight_layout(); fig.savefig(a.out / "fig_objective_compare.png", dpi=130)
