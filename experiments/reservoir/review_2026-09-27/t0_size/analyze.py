"""T0 vs dam size: Spearman with bootstrap CI (both seeds), purpose breakdown, surrogate fits from the
release-head inputs, agreement with independent T estimates, seed stability. Writes results.json,
tables (CSV) and three figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

OUT = Path("/home/tbindas/.claude/jobs/dacd6d8c/tmp/t0_size")
RNG = np.random.default_rng(0)
SEEDS = (42, 43)
COL = {42: "#2A6EBB", 43: "#D9782D", "grey": "#6B6B6B"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})

SIZE_VARS = [
    ("storage_mcm", "normal storage [MCM]"),
    ("storage_max_mcm", "max storage [MCM]"),
    ("surface_km2", "surface area [km2]"),
    ("height_m", "dam height [m]"),
    ("da_eff_km2", "drainage area [km2]"),
    ("max_discharge_m3s", "max discharge [m3/s]"),
    ("res_time_gauge_d", "residence time, gauge-flow proxy [d]"),
    ("res_time_grand_d", "residence time, GRanD mean flow [d]"),
    ("res_time_best_d", "residence time, GRanD where available else gauge proxy [d]"),
    ("storage_per_area_m", "storage / upstream area [m]"),
    ("year", "year completed"),
]


def boot_spearman(x, y, n=2000):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0 if np.nanmin(x) >= 0 else True)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 8:
        return dict(n=int(len(x)), rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    r = spearmanr(x, y)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    # rank-based bootstrap
    rs = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    return dict(n=int(len(x)), rho=float(r.statistic), lo=float(np.nanpercentile(rs, 2.5)),
                hi=float(np.nanpercentile(rs, 97.5)), p=float(r.pvalue))


def main():
    d = pd.read_csv(OUT / "t0_size_table.csv")
    d["res_time_best_d"] = d.res_time_grand_d.fillna(d.res_time_gauge_d)
    res = {}

    # ---------- 1. T0 vs size, both seeds ----------
    rows = []
    for var, label in SIZE_VARS:
        for s in SEEDS:
            b = boot_spearman(d[var].to_numpy(float), d[f"T0_days_s{s}"].to_numpy(float), n=1000)
            rows.append(dict(variable=var, label=label, seed=s, **b))
    size_tab = pd.DataFrame(rows)
    size_tab.to_csv(OUT / "table1_spearman_size.csv", index=False)
    res["spearman_size"] = size_tab.to_dict("records")

    # seed stability
    st = boot_spearman(d.T0_days_s42.to_numpy(), d.T0_days_s43.to_numpy(), n=1000)
    res["seed_rank_agreement"] = st
    lr = np.log10(d.T0_days_s43 / d.T0_days_s42)
    res["seed_log10_ratio_43_over_42"] = dict(median=float(lr.median()), q25=float(lr.quantile(.25)), q75=float(lr.quantile(.75)),
                                             frac_within_factor2=float((lr.abs() < np.log10(2)).mean()))

    # partial Spearman: residence time controlling for storage (rank residuals), and storage controlling for drainage
    def partial_rho(x, y, z):
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0) & (z > 0)
        rx, ry, rz = [pd.Series(v[m]).rank().to_numpy() for v in (x, y, z)]
        def resid(a, c):
            A = np.c_[np.ones_like(c), c]
            return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
        return float(spearmanr(resid(rx, rz), resid(ry, rz)).statistic), int(m.sum())
    res["partial"] = {}
    for s in SEEDS:
        t = d[f"T0_days_s{s}"].to_numpy(float)
        res["partial"][f"s{s}"] = dict(
            restime_given_storage=partial_rho(d.res_time_gauge_d.to_numpy(float), t, d.storage_mcm.to_numpy(float)),
            storage_given_restime=partial_rho(d.storage_mcm.to_numpy(float), t, d.res_time_gauge_d.to_numpy(float)),
            storage_given_drainage=partial_rho(d.storage_mcm.to_numpy(float), t, d.da_eff_km2.to_numpy(float)),
            drainage_given_storage=partial_rho(d.da_eff_km2.to_numpy(float), t, d.storage_mcm.to_numpy(float)),
            height_given_storage=partial_rho(d.height_m.to_numpy(float), t, d.storage_mcm.to_numpy(float)),
        )
    # storage vs drainage collinearity and vs residence time
    res["collinearity"] = dict(
        storage_vs_drainage=float(spearmanr(np.log10(d.storage_mcm), np.log10(d.da_eff_km2)).statistic),
        storage_vs_restime=float(spearmanr(np.log10(d.storage_mcm), np.log10(d.res_time_gauge_d)).statistic),
        storage_vs_height=float(spearmanr(np.log10(d.storage_mcm), d.height_m).statistic),
        restime_gauge_vs_grand=boot_spearman(d.res_time_gauge_d.to_numpy(float), d.res_time_grand_d.to_numpy(float), 500),
        restime_gauge_vs_hydrolakes=boot_spearman(d.res_time_gauge_d.to_numpy(float), d.hydrolakes_res_time_d.to_numpy(float), 500),
    )

    # by purpose
    prow = []
    for purp, g in d.groupby("primary_purpose"):
        if len(g) < 15:
            continue
        for s in SEEDS:
            b = boot_spearman(g.storage_mcm.to_numpy(float), g[f"T0_days_s{s}"].to_numpy(float), n=500)
            b2 = boot_spearman(g.res_time_gauge_d.to_numpy(float), g[f"T0_days_s{s}"].to_numpy(float), n=500)
            prow.append(dict(purpose=purp, seed=s, n=len(g), T0_median_d=float(g[f"T0_days_s{s}"].median()),
                             T0_q25=float(g[f"T0_days_s{s}"].quantile(.25)), T0_q75=float(g[f"T0_days_s{s}"].quantile(.75)),
                             storage_median_mcm=float(g.storage_mcm.median()), restime_median_d=float(g.res_time_gauge_d.median()),
                             rho_storage=b["rho"], rho_storage_lo=b["lo"], rho_storage_hi=b["hi"],
                             rho_restime=b2["rho"], rho_restime_lo=b2["lo"], rho_restime_hi=b2["hi"]))
    ptab = pd.DataFrame(prow).sort_values(["seed", "T0_median_d"], ascending=[True, False])
    ptab.to_csv(OUT / "table2_by_purpose.csv", index=False)
    res["by_purpose"] = ptab.to_dict("records")

    # ---------- 2. surrogates from the release-head inputs ----------
    fcols = [c for c in d.columns if c.startswith("f_")]
    X = d[fcols].to_numpy(float)
    names = [c[2:] for c in fcols]
    sur = {}
    kf = KFold(5, shuffle=True, random_state=0)
    for s in SEEDS:
        y = np.log10(d[f"T0_days_s{s}"].to_numpy(float))
        out = {}
        # linear (ridge) on the 19 normalised inputs
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 25))
        yhat = cross_val_predict(ridge, X, y, cv=kf)
        out["ridge_cv_r2"] = float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        ridge.fit(X, y)
        out["ridge_train_r2"] = float(ridge.score(X, y))
        out["ridge_coef"] = {n: float(c) for n, c in sorted(zip(names, ridge.coef_), key=lambda t: -abs(t[1]))}
        # linear on the four raw log size variables only
        Z = np.c_[np.log10(d.storage_mcm), np.log10(d.res_time_gauge_d), d.height_m, np.log10(d.da_eff_km2)]
        m = np.isfinite(Z).all(1)
        lin = LinearRegression()
        yh = cross_val_predict(lin, Z[m], y[m], cv=kf)
        out["lin4_cv_r2"] = float(1 - ((y[m] - yh) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())
        for name, cols in [("storage_only", [0]), ("restime_only", [1]), ("storage+restime", [0, 1]), ("storage+drainage", [0, 3])]:
            yh = cross_val_predict(LinearRegression(), Z[m][:, cols], y[m], cv=kf)
            out[f"lin_{name}_cv_r2"] = float(1 - ((y[m] - yh) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())
        # small tree ensemble
        gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
        yh = cross_val_predict(gbm, X, y, cv=kf)
        out["gbm_cv_r2"] = float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        gbm.fit(X, y)
        out["gbm_train_r2"] = float(gbm.score(X, y))
        pi = permutation_importance(gbm, X, y, n_repeats=10, random_state=0, n_jobs=1)
        out["gbm_perm_importance"] = {n: float(v) for n, v in sorted(zip(names, pi.importances_mean), key=lambda t: -t[1])}
        out["gbm_perm_importance_std"] = {n: float(v) for n, v in zip(names, pi.importances_std)}
        sur[f"s{s}"] = out
        sur[f"s{s}_gbm"] = gbm
    res["surrogate"] = {k: v for k, v in sur.items() if not k.endswith("_gbm")}

    # ---------- 3. agreement with independent estimates ----------
    agree = {}
    comps = [
        ("offline seasonal fit (dam on gauge reach)", d[d.off_on_reach == True], "off_seas_T0"),
        ("offline linear fit (dam on gauge reach)", d[d.off_on_reach == True], "off_lin_T0"),
        ("offline seasonal fit (all, nearest large dam)", d, "off_seas_T0"),
        ("offline seasonal fit (on reach, offline NSE > 0.5)", d[(d.off_on_reach == True) & (d.off_nse_seas > 0.5)], "off_seas_T0"),
        ("ResOpsUS fit (own inflow/release)", d[d.resops_T.notna() & (d.resops_wall != True)], "resops_T"),
        ("ResOpsUS fit (own inflow/release, incl. wall)", d, "resops_T"),
        ("HydroLAKES residence time", d, "hydrolakes_res_time_d"),
        ("residence time (gauge proxy)", d, "res_time_gauge_d"),
    ]
    arow = []
    for label, sub, col in comps:
        for s in SEEDS:
            t = sub[f"T0_days_s{s}"].to_numpy(float); ind = sub[col].to_numpy(float)
            b = boot_spearman(ind, t, n=1000)
            m = np.isfinite(ind) & np.isfinite(t) & (ind > 0)
            ratio = np.log10(t[m] / ind[m])
            arow.append(dict(comparison=label, indep=col, seed=s, n=b["n"], rho=b["rho"], lo=b["lo"], hi=b["hi"],
                             median_learned_d=float(np.median(t[m])) if m.any() else np.nan,
                             median_indep_d=float(np.median(ind[m])) if m.any() else np.nan,
                             log10_ratio_median=float(np.median(ratio)) if m.any() else np.nan,
                             log10_ratio_q25=float(np.quantile(ratio, .25)) if m.any() else np.nan,
                             log10_ratio_q75=float(np.quantile(ratio, .75)) if m.any() else np.nan,
                             frac_within_factor2=float((np.abs(ratio) < np.log10(2)).mean()) if m.any() else np.nan))
    atab = pd.DataFrame(arow)
    atab.to_csv(OUT / "table3_independent.csv", index=False)
    res["independent"] = atab.to_dict("records")

    # worst disagreements (offline on-reach, seed 42), with context
    sub = d[(d.off_on_reach == True)].copy()
    sub["log10_ratio_s42"] = np.log10(sub.T0_days_s42 / sub.off_seas_T0)
    sub["log10_ratio_s43"] = np.log10(sub.T0_days_s43 / sub.off_seas_T0)
    cols = ["COMID", "name", "primary_purpose", "storage_mcm", "res_time_gauge_d", "T0_days_s42", "T0_days_s43",
            "off_seas_T0", "off_seas_lo", "off_seas_hi", "off_nse_seas", "off_STAID", "log10_ratio_s42", "log10_ratio_s43"]
    worst = pd.concat([sub.nsmallest(8, "log10_ratio_s42")[cols], sub.nlargest(8, "log10_ratio_s42")[cols]])
    worst.to_csv(OUT / "table4_worst_offline.csv", index=False)
    rs = d[d.resops_T.notna()].copy()
    rs["log10_ratio_s42"] = np.log10(rs.T0_days_s42 / rs.resops_T); rs["log10_ratio_s43"] = np.log10(rs.T0_days_s43 / rs.resops_T)
    rcols = ["COMID", "name", "resops_name", "primary_purpose", "storage_mcm", "res_time_gauge_d", "T0_days_s42", "T0_days_s43",
             "resops_T", "resops_T_lo", "resops_T_hi", "resops_nse", "resops_wall", "log10_ratio_s42", "log10_ratio_s43"]
    rs.sort_values("log10_ratio_s42")[rcols].to_csv(OUT / "table5_resops.csv", index=False)
    # does the disagreement scale with residence time / storage?
    res["disagreement_drivers"] = {}
    for s in SEEDS:
        res["disagreement_drivers"][f"s{s}"] = dict(
            offline_ratio_vs_restime=boot_spearman(sub.res_time_gauge_d.to_numpy(float), sub[f"log10_ratio_s{s}"].to_numpy(float), 500),
            offline_ratio_vs_storage=boot_spearman(sub.storage_mcm.to_numpy(float), sub[f"log10_ratio_s{s}"].to_numpy(float), 500),
            resops_ratio_vs_resopsT=boot_spearman(rs.resops_T.to_numpy(float), rs[f"log10_ratio_s{s}"].to_numpy(float), 500),
        )
    # top learned T0 dams
    top = d.nlargest(12, "T0_days_s42")[["COMID", "name", "primary_purpose", "storage_mcm", "da_eff_km2", "res_time_gauge_d",
                                           "T0_days_s42", "T0_days_s43", "off_seas_T0", "off_on_reach", "resops_T", "hydrolakes_res_time_d"]]
    top.to_csv(OUT / "table6_top_T0.csv", index=False)

    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)

    # ---------- figures ----------
    # Fig 1: T0 vs size measures, both seeds, log-log; last panel by purpose
    panels = [("storage_mcm", "Normal storage [MCM]", True), ("res_time_gauge_d", "Residence time = storage / mean flow [d]", True),
              ("height_m", "Dam height [m]", True), ("da_eff_km2", "Drainage area [km2]", True),
              ("storage_per_area_m", "Storage per upstream area [m]", True)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    axes = axes.ravel()
    for ax, (var, label, logx) in zip(axes, panels):
        for s, mk in zip(SEEDS, ("o", "^")):
            ax.scatter(d[var], d[f"T0_days_s{s}"], s=9, alpha=0.45, c=COL[s], marker=mk, edgecolors="none", label=f"seed {s}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(label); ax.set_ylabel("Learned T0 [d]")
        txt = []
        for s in SEEDS:
            r = size_tab[(size_tab.variable == var) & (size_tab.seed == s)].iloc[0]
            txt.append(f"seed {s}: rho={r.rho:.2f} [{r.lo:.2f}, {r.hi:.2f}]")
        ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
    # purpose panel
    ax = axes[5]
    pp = ptab[ptab.seed == 42].sort_values("T0_median_d", ascending=True)
    order = list(pp.purpose)
    data42 = [np.log10(d[d.primary_purpose == p].T0_days_s42) for p in order]
    data43 = [np.log10(d[d.primary_purpose == p].T0_days_s43) for p in order]
    pos = np.arange(len(order))
    for data, s, off in ((data42, 42, -0.18), (data43, 43, 0.18)):
        bp = ax.boxplot(data, positions=pos + off, widths=0.32, vert=False, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        for b in bp["boxes"]:
            b.set(facecolor=COL[s], alpha=0.55, edgecolor=COL[s])
    ax.set_yticks(pos); ax.set_yticklabels([f"{p} (n={int(pp[pp.purpose == p].n.iloc[0])})" for p in order], fontsize=7.5)
    ax.set_xlabel("log10 learned T0 [d]"); ax.set_title("By primary purpose (largest dam on reach)", fontsize=9)
    ax.axvline(np.log10(1), color="#bbbbbb", lw=0.8, ls="--")
    fig.suptitle("Learned dam response time T0 against dam size (1,024 dam reaches, two seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig1_t0_vs_size.png", dpi=200); plt.close(fig)

    # Fig 2: surrogate importance + partial dependence
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    imp42 = sur["s42"]["gbm_perm_importance"]; imp43 = sur["s43"]["gbm_perm_importance"]
    keys = list(imp42.keys())[:10]
    y = np.arange(len(keys))
    ax.barh(y - 0.18, [imp42[k] for k in keys], height=0.34, color=COL[42], label=f"seed 42 (CV R2={sur['s42']['gbm_cv_r2']:.2f})")
    ax.barh(y + 0.18, [imp43[k] for k in keys], height=0.34, color=COL[43], label=f"seed 43 (CV R2={sur['s43']['gbm_cv_r2']:.2f})")
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (drop in R2 of log10 T0)"); ax.set_title("Gradient-boosting surrogate: which inputs the head uses")
    ax.legend(fontsize=7.5, frameon=False)
    for ax, feat in zip(axes[1:], ["log10_storage_max", "log10_storage_per_area"]):
        j = names.index(feat)
        for s in SEEDS:
            pdp = partial_dependence(sur[f"s{s}_gbm"], X, [j], grid_resolution=40, kind="average")
            gx = pdp["grid_values"][0]; gy = pdp["average"][0]
            ax.plot(gx, gy, color=COL[s], lw=2, label=f"seed {s}")
        ax.set_xlabel(f"{feat} (z-scored head input)"); ax.set_ylabel("partial dependence of log10 T0 [d]")
        ax.set_title(f"Partial dependence: {feat}")
        ax.legend(fontsize=7.5, frameon=False)
        # rug of data
        ax.plot(X[:, j], np.full(len(X), ax.get_ylim()[0]), "|", color=COL["grey"], alpha=0.3, ms=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_surrogate.png", dpi=200); plt.close(fig)

    # Fig 3: learned vs independent
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    specs = [
        (axes[0], d[d.off_on_reach == True], "off_seas_T0", "Offline seasonal bucket fit on routed inflow, T0 [d]\n(dam on the gauge reach)", "offline seasonal fit (dam on gauge reach)"),
        (axes[1], d[d.resops_T.notna()], "resops_T", "ResOpsUS fit from own inflow and release, T [d]", "ResOpsUS fit (own inflow/release, incl. wall)"),
        (axes[2], d, "res_time_gauge_d", "Residence time = storage / mean flow [d]", "residence time (gauge proxy)"),
    ]
    for ax, sub, col, xlabel, label in specs:
        for s, mk in zip(SEEDS, ("o", "^")):
            ax.scatter(sub[col], sub[f"T0_days_s{s}"], s=14, alpha=0.6, c=COL[s], marker=mk, edgecolors="none", label=f"seed {s}")
        lo = np.nanmin([sub[col].min(), 0.05]); hi = np.nanmax([sub[col].max(), 80])
        ax.plot([lo, hi], [lo, hi], color=COL["grey"], lw=1, ls="--", label="1:1")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("Learned T0 [d]")
        txt = []
        for s in SEEDS:
            r = atab[(atab.comparison == label) & (atab.seed == s)].iloc[0]
            txt.append(f"seed {s}: n={r.n}, rho={r.rho:.2f} [{r.lo:.2f}, {r.hi:.2f}], median ratio={10**r.log10_ratio_median:.2f}")
        ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.suptitle("Learned T0 against independent time-scale estimates", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig3_independent.png", dpi=200); plt.close(fig)

    print(size_tab.round(3).to_string())
    print(ptab.round(3).to_string())
    print(atab.round(3).to_string())
    print(json.dumps({k: res[k] for k in ("seed_rank_agreement", "seed_log10_ratio_43_over_42", "partial", "collinearity", "disagreement_drivers")}, indent=1, default=str))
    for s in SEEDS:
        o = sur[f"s{s}"]
        print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if "coef" not in k and "importance" not in k})
        print(" ridge coef:", {k: round(v, 3) for k, v in list(o["ridge_coef"].items())[:8]})
        print(" gbm perm:", {k: round(v, 3) for k, v in list(o["gbm_perm_importance"].items())[:8]})
    print(worst.round(3).to_string())
    print(top.round(3).to_string())


if __name__ == "__main__":
    main()
