#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "xarray", "zarr>=3", "netCDF4", "scipy", "matplotlib"]
# ///
"""E-08: n_0 level vs spatial pattern across the six fixed-channel models, per reach.

Fields as cross_arm_fields.py loads them (plot/kan_parameters.nc: `n` is n_0, `gamma` is the stage exponent),
restricted to reaches with a finite log10_uparea in the attribute table, as there. Two reach sets:
  all       every such reach (the section 38.2 population, so its numbers can be reproduced);
  filtered  reaches covered by every product (COMID on each store's divide_id axis, so ddrs read real Q' and
            not its 0.001 m3/s fill) with median daily Q' >= 1e-3 m3/s over WY1996-2010 in every product
            (qprime_medians.py).
Per pair of products: Spearman rho of n_0 and of gamma, median per-reach ratio of n_0 (and the ratio of the
medians), median per-reach difference of gamma. Per reach: range-normalised cross-product spread (paper Eq.
normspread as cross_arm_fields.py implements it: (max - min) / box width, box n_0 [0.015, 0.25], gamma
[0, 0.5]). Plus a two-way decomposition of log n_0 over (reach, product): product main effect (level), reach
main effect (pattern shared by all products), and the interaction (product-specific pattern).

    experiments/cross_product/e08_fields.py
Writes experiments/cross_product/e08_fields.txt and research/figures/cross-product/n0_rank_matrix.png.
"""
import io
import itertools

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import spearmanr

import cp_common as C

FLOOR = 1e-3


def summarise(name, F, idx, say):
    N = np.stack([F[k].n0.reindex(idx).to_numpy() for k in C.KEYS])
    G = np.stack([F[k].gamma.reindex(idx).to_numpy() for k in C.KEYS])
    say(f"\n== {name}: {idx.size:,} reaches")
    say(f"{'product':>20} {'n0 median':>10} {'n0 p10-p90':>14} {'at n0 ceiling':>13} {'gamma median':>12} {'gamma = 0':>9}")
    for i, k in enumerate(C.KEYS):
        say(f"{C.LABEL[k]:>20} {np.median(N[i]):10.4f} {np.percentile(N[i], 10):6.3f}-{np.percentile(N[i], 90):.3f} "
            f"{100 * np.mean(N[i] >= 0.2495):12.1f}% {np.median(G[i]):12.3f} {100 * np.mean(G[i] < 0.001):8.1f}%")
    rows = []
    say(f"\n{'pair':>42} {'rho n0':>7} {'rho gamma':>9} {'median n0 ratio':>15} {'ratio of medians':>16} {'median gamma diff':>17}")
    for i, j in itertools.combinations(range(len(C.KEYS)), 2):
        a, b = C.KEYS[i], C.KEYS[j]
        rn = spearmanr(N[i], N[j]).statistic
        rg = spearmanr(G[i], G[j]).statistic
        mr = np.median(N[i] / N[j])
        rm = np.median(N[i]) / np.median(N[j])
        gd = np.median(G[i] - G[j])
        rows.append(dict(a=a, b=b, rho_n0=rn, rho_gamma=rg, med_ratio=mr, ratio_med=rm, gamma_diff=gd))
        say(f"{C.LABEL[a] + ' / ' + C.LABEL[b]:>42} {rn:+7.3f} {rg:+9.3f} {mr:15.3f} {rm:16.3f} {gd:+17.3f}")
    R = pd.DataFrame(rows)
    sn = (N.max(0) - N.min(0)) / (C.N_BOX[1] - C.N_BOX[0])
    sg = (G.max(0) - G.min(0)) / (C.G_BOX[1] - C.G_BOX[0])
    say(f"\nrange-normalised cross-product spread per reach: n0 median {np.median(sn):.3f} (p10 {np.percentile(sn, 10):.3f}, "
        f"p90 {np.percentile(sn, 90):.3f}); gamma median {np.median(sg):.3f} (p10 {np.percentile(sg, 10):.3f}, "
        f"p90 {np.percentile(sg, 90):.3f})")
    say(f"pairwise rho n0: median {R.rho_n0.median():.3f}, range {R.rho_n0.min():+.3f}..{R.rho_n0.max():+.3f}; "
        f"rho gamma: median {R.rho_gamma.median():.3f}, range {R.rho_gamma.min():+.3f}..{R.rho_gamma.max():+.3f}")
    # level vs pattern: two-way decomposition of log n0
    L = np.log(N)
    grand = L.mean()
    prod_eff = L.mean(1, keepdims=True) - grand
    reach_eff = L.mean(0, keepdims=True) - grand
    inter = L - grand - prod_eff - reach_eff
    ss_tot = ((L - grand) ** 2).sum()
    ss_p = (prod_eff ** 2).sum() * L.shape[1]
    ss_r = (reach_eff ** 2).sum() * L.shape[0]
    ss_i = (inter ** 2).sum()
    say(f"log n0 two-way decomposition (shares of total SS): product level {ss_p / ss_tot:.3f}, shared reach pattern "
        f"{ss_r / ss_tot:.3f}, product-specific pattern {ss_i / ss_tot:.3f} (sum {(ss_p + ss_r + ss_i) / ss_tot:.3f})")
    # same with rank-transformed fields per product: how much of the pattern is shared once level is removed entirely
    rk = np.stack([pd.Series(x).rank().to_numpy() for x in N])
    rk = (rk - rk.mean(1, keepdims=True))
    shared = rk.mean(0, keepdims=True)
    say(f"rank-pattern share explained by the cross-product mean rank: "
        f"{1 - ((rk - shared) ** 2).sum() / (rk ** 2).sum():.3f}")
    return R, N, G, sn, sg


def main():
    buf = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=buf)

    at = Dataset(C.ATTRS)
    ac = np.asarray(at["COMID"][:], dtype=np.int64)
    ua = np.asarray(at["log10_uparea"][:], dtype=np.float64)
    la = pd.Series(ua, index=ac)
    F = {}
    for k in C.KEYS:
        f = C.load_field(k)
        ok = np.isfinite(la.reindex(f.index).to_numpy())
        F[k] = f[ok]
    idx_all = F[C.KEYS[0]].index
    for k in C.KEYS[1:]:
        idx_all = idx_all.intersection(F[k].index)

    cov = idx_all
    say("coverage and wetness filter, per product (on the all-reach set):")
    for k in C.KEYS:
        z = np.load(C.OUT / f"qprime_median_{k}.npz")
        med = pd.Series(z["median"], index=z["divide_id"])
        on = idx_all.isin(med.index)
        wet = on & (med.reindex(idx_all).to_numpy() >= FLOOR)
        say(f"{C.LABEL[k]:>20}: store has {med.size:,} divides; {on.sum():,} of {idx_all.size:,} reaches covered; "
            f"{wet.sum():,} covered with median Q' >= {FLOOR:g} m3/s")
        cov = cov[(cov.isin(med.index)) & (med.reindex(cov).to_numpy() >= FLOOR)]
    covered_all = idx_all
    for k in C.KEYS:
        z = np.load(C.OUT / f"qprime_median_{k}.npz")
        covered_all = covered_all[covered_all.isin(z["divide_id"])]
    say(f"covered by every product: {covered_all.size:,}; and wet in every product: {cov.size:,}")

    Ra, *_ = summarise("all reaches (section 38.2 population)", F, idx_all, say)
    Rc, *_ = summarise("covered by every product, no flow floor", F, covered_all, say)
    Rf, N, G, sn, sg = summarise(f"filtered: covered by every product, median Q' >= {FLOOR:g} m3/s in all six", F, cov, say)

    # what the filter changed
    say("\n== What the filter changed (filtered minus all-reach)")
    m = Ra.merge(Rf, on=["a", "b"], suffixes=("_all", "_filt"))
    for _, r in m.iterrows():
        say(f"{C.LABEL[r.a] + ' / ' + C.LABEL[r.b]:>42} rho n0 {r.rho_n0_all:+.3f} -> {r.rho_n0_filt:+.3f}   "
            f"rho gamma {r.rho_gamma_all:+.3f} -> {r.rho_gamma_filt:+.3f}   median n0 ratio {r.med_ratio_all:.3f} -> {r.med_ratio_filt:.3f}")

    # the reaches the filter removes: what are they?
    drop = idx_all.difference(cov)
    say(f"\nreaches removed by the filter: {drop.size:,}; median log10 upstream area removed {np.median(la.reindex(drop)):.2f} "
        f"vs kept {np.median(la.reindex(cov)):.2f}")
    (C.HERE / "e08_fields.txt").write_text(buf.getvalue())
    pd.concat([Ra.assign(set="all"), Rc.assign(set="covered"), Rf.assign(set="filtered")]).to_csv(
        C.OUT / "e08_pairs.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(C.KEYS)
    M = np.full((K, K), np.nan)
    for _, r in Rf.iterrows():
        i, j = C.KEYS.index(r.a), C.KEYS.index(r.b)
        M[i, j] = r.rho_n0
        M[j, i] = r.rho_gamma
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(K):
        for j in range(K):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    labs = [C.LABEL[k] for k in C.KEYS]
    ax.set_xticks(range(K), labs, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(K), labs, fontsize=8.5)
    ax.set_title(f"per-reach Spearman rho between products, {cov.size:,} covered wet reaches\n"
                 "upper triangle: n0;  lower triangle: gamma", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()
    C.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG / "n0_rank_matrix.png", dpi=150, facecolor="white")
    print(f"wrote {C.FIG / 'n0_rank_matrix.png'}")


if __name__ == "__main__":
    main()
