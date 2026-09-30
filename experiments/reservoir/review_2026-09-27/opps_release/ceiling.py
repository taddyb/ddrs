"""Where is the ceiling? Offline, on the smoke set's on-reach dams (214) and their matched controls.

 a. ddrs-learned (T0, a, b) from release_params.csv (seeds 42, 43), simulated OFFLINE on the routed no-dam flow:
    if this reproduces ddrs's +0.006, the learned parameters (not the joint routing) are the limit.
 b. Feature-limited ceiling: 5-fold CV prediction of the per-gauge fit's (log T0, a, b) from NID features
    (random forest), simulated offline: what a perfect feature->parameter head could reach.
 c. Per-gauge fit gain split by fitted T0 (short vs long): how much of the ceiling sits in T0 > 5 d.
 d. Nash cascade (N = 2, 3 buckets in series, each T_seas/N; T0 re-gridded x0.25..x4): does a cascade buy
    timing (r) at less variance (alpha) cost than one bucket?
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest, spearmanr

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
RUNS = "/home/tbindas/projects/ddrs/.ddrs/runs"
OUT = "/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release"
T0_FACTORS = np.array([0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0])


def sim(I, sinw, cosw, T0, a, b, n, N=1, obs=None, mask=None):
    T0 = np.asarray(T0, float); a = np.asarray(a, float); b = np.asarray(b, float)
    T = np.maximum(T0 * np.exp(a * sinw[0] + b * cosw[0]), 0.05) / N
    S = [T * I[0] for _ in range(N)]
    sse = np.zeros_like(T0) if obs is not None else None
    Q = np.empty(n) if obs is None else None
    for t in range(n):
        T = np.maximum(T0 * np.exp(a * sinw[t] + b * cosw[t]), 0.05) / N
        inflow = I[t]
        for k in range(N):
            avail = S[k] + inflow
            q = avail / (T + 1.0)
            S[k] = avail - q
            inflow = q
        if obs is None:
            Q[t] = q[0]
        elif mask[t]:
            sse += (q - obs[t]) ** 2
    return sse if obs is not None else Q


def metrics(p, o, m):
    m = m & np.isfinite(o) & np.isfinite(p)
    p, o = p[m], o[m]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    al, be = p.std() / o.std(), p.mean() / o.mean()
    return dict(nse=float(nse), kge=float(1 - np.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2)), r=float(r), alpha=float(al), beta=float(be))


def job(args):
    s, role, I, O, sinw, cosw, train, test, params = args
    I = np.where(np.isfinite(I), I, np.nanmean(I))
    r = {"STAID": s, "role": role}
    for k, v in metrics(I, O, test).items():
        r[f"pass_{k}"] = v
    # fixed-parameter laws (learned seeds, CV prediction, the per-gauge seasonal fit itself)
    for name, (t0, a, b) in params.items():
        if not np.isfinite(t0):
            continue
        q = sim(I, sinw, cosw, [t0], [a], [b], len(I))
        for k, v in metrics(q, O, test).items():
            r[f"{name}_{k}"] = v
    # Nash cascades, T0 re-gridded, a, b from the per-gauge fit
    t0f, af, bf = params["fit"]
    n_fit = int(np.flatnonzero(train)[-1]) + 1
    m = train & np.isfinite(O); otr = O[m]; den = ((otr - otr.mean()) ** 2).sum()
    for N in (1, 2, 3):
        T0 = t0f * T0_FACTORS
        nse = 1 - sim(I, sinw, cosw, T0, np.full(7, af), np.full(7, bf), n_fit, N, obs=O, mask=m) / den
        k = int(np.argmax(nse))
        q = sim(I, sinw, cosw, T0[k:k + 1], [af], [bf], len(I), N)
        for kk, v in metrics(q, O, test).items():
            r[f"nash{N}_{kk}"] = v
        r[f"nash{N}_T0"] = float(T0[k]); r[f"nash{N}_train"] = float(nse[k])
    return r


def paired(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) == 0:
        return None
    nz = d[np.abs(d) > 1e-9]; up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    bs = np.median(rng.choice(d, (2000, len(d))), axis=1)
    return dict(n=int(len(d)), median=round(float(np.median(d)), 5), ci=[round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
                n_up=up, n_down=int(len(nz) - up), sign_p=round(float(binomtest(up, len(nz)).pvalue), 4) if len(nz) else None)


if __name__ == "__main__":
    z = zarr.open(f"{WT}/output/reservoir_smoke/pred_1981_2010.zarr", mode="r")
    ids = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    P, O = z["predictions"][:].astype(float), z["observations"][:].astype(float)
    wy = np.asarray(t.year + (t.month >= 10))
    train, test = (wy >= 1983) & (wy <= 1995), (wy >= 1996) & (wy <= 2010)
    w = 2 * np.pi * np.asarray(t.dayofyear) / 365.25
    sinw, cosw = np.sin(w), np.cos(w)
    fit = pd.read_csv(f"{WT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str, "control_for": str}).set_index("STAID")
    sg = pd.read_csv(f"{WT}/experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
    onr = fit[(fit.role == "dam") & (fit.on_reach.astype(str) == "True")].join(sg[["dam_COMID", "dam_storage_mcm", "dam_da_km2", "dam_reach_uparea_km2",
                                                                                   "dam_max_discharge_m3s", "dam_height_m", "dam_year", "dam_purpose"]], rsuffix="_sg")
    ctl = fit[(fit.role == "control") & fit.control_for.isin(onr.index)]
    rp = {s: pd.read_csv(f"{RUNS}/{run}/release_params.csv").set_index("COMID") for s, run in
          [(42, "2026-09-27T07-29-55Z-train-and-test"), (43, "2026-09-27T10-31-50Z-train-and-test")]}

    # b. CV feature model of the per-gauge fit
    X = pd.DataFrame(index=onr.index)
    X["ls"] = np.log10(onr.dam_storage_mcm)
    X["lda"] = np.log10(onr.dam_da_km2.fillna(onr.dam_reach_uparea_km2))
    X["lup"] = np.log10(onr.dam_reach_uparea_km2)
    X["lspa"] = X.ls - X.lup
    X["lmq"] = np.log10(onr.dam_max_discharge_m3s).replace(-np.inf, np.nan)
    X["mq_missing"] = X.lmq.isna().astype(float); X["lmq"] = X.lmq.fillna(X.lmq.median())
    X["h"] = onr.dam_height_m.fillna(onr.dam_height_m.median())
    X["yr"] = onr.dam_year.fillna(onr.dam_year.median())
    for p in ["Flood Risk Reduction", "Hydroelectric", "Water Supply", "Irrigation", "Recreation", "Navigation"]:
        X[f"p_{p[:5]}"] = (onr.dam_purpose == p).astype(float)
    Y = np.column_stack([np.log10(onr.seas_T0), onr.seas_a, onr.seas_b])
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import KFold
        pred = np.zeros_like(Y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(300, min_samples_leaf=5, random_state=0).fit(X.values[tr], Y[tr])
            pred[te] = rf.predict(X.values[te])
        cv_ok = True
    except ImportError:
        pred = np.tile(Y.mean(0), (len(Y), 1)); cv_ok = False
    cv = pd.DataFrame(pred, index=onr.index, columns=["lt0", "a", "b"])
    r2 = 1 - ((Y[:, 0] - pred[:, 0]) ** 2).sum() / ((Y[:, 0] - Y[:, 0].mean()) ** 2).sum()

    jobs = []
    for s, row in onr.iterrows():
        i = ids.get_loc(s)
        params = {"fit": (row.seas_T0, row.seas_a, row.seas_b),
                  "cv": (10 ** cv.loc[s, "lt0"], cv.loc[s, "a"], cv.loc[s, "b"]),
                  "cv_T0_only": (10 ** cv.loc[s, "lt0"], 0.0, 0.0)}
        for seed in (42, 43):
            if row.dam_COMID in rp[seed].index:
                q = rp[seed].loc[row.dam_COMID]
                params[f"learned{seed}"] = (q.T0_days, q.a, q.b)
                params[f"learned{seed}_x3"] = (q.T0_days * 3.0, q.a, q.b)
        jobs.append((s, "dam", P[i], O[i], sinw, cosw, train, test, params))
    for s, row in ctl.iterrows():
        i = ids.get_loc(s)
        jobs.append((s, "control", P[i], O[i], sinw, cosw, train, test, {"fit": (row.seas_T0, row.seas_a, row.seas_b)}))
    with Pool(4) as pool:
        rows = pool.map(job, jobs)
    df = pd.DataFrame(rows).set_index("STAID")
    df.to_csv(f"{OUT}/ceiling_by_gauge.csv")
    d = df[df.role == "dam"]; c = df[df.role == "control"]
    res = {"n_dam": len(d), "n_control": len(c), "cv_feature_model": dict(sklearn=cv_ok, r2_log_T0_5fold=round(float(r2), 3),
                                                                       spearman_pred_vs_fit_T0=round(float(spearmanr(cv.lt0, np.log10(onr.seas_T0)).correlation), 3))}
    for name in ["fit", "cv", "cv_T0_only", "learned42", "learned43", "learned42_x3", "learned43_x3", "nash1", "nash2", "nash3"]:
        if f"{name}_nse" not in d:
            continue
        res[name] = dict(dnse_vs_pass=paired(d[f"{name}_nse"] - d.pass_nse), dkge_vs_pass=paired(d[f"{name}_kge"] - d.pass_kge),
                         dalpha_vs_pass=paired(d[f"{name}_alpha"] - d.pass_alpha), dr_vs_pass=paired(d[f"{name}_r"] - d.pass_r),
                         median_nse=round(float(d[f"{name}_nse"].median()), 4))
    for N in (2, 3):
        res[f"nash{N}_vs_nash1"] = dict(dnse=paired(d[f"nash{N}_nse"] - d.nash1_nse), dkge=paired(d[f"nash{N}_kge"] - d.nash1_kge),
                                       dalpha=paired(d[f"nash{N}_alpha"] - d.nash1_alpha), dr=paired(d[f"nash{N}_r"] - d.nash1_r),
                                       control_dnse=paired(c[f"nash{N}_nse"] - c.nash1_nse),
                                       median_T0_ratio=round(float(np.median(d[f"nash{N}_T0"] / d.nash1_T0)), 3))
    # c. gain by fitted T0 class
    bins = pd.cut(onr.seas_T0, [0, 0.051, 0.5, 2, 5, 20, 2000], labels=["floor", "<=0.5d", "0.5-2d", "2-5d", "5-20d", ">20d"])
    res["fit_gain_by_fitted_T0"] = {str(k): dict(fit=paired((d.fit_nse - d.pass_nse)[v.index]), learned42=paired((d.learned42_nse - d.pass_nse)[v.index]),
                                                  n=len(v)) for k, v in onr.groupby(bins, observed=True)}
    # share of total (sum) fit gain by class
    tot = (d.fit_nse - d.pass_nse).clip(-0.5, 0.5).sum()
    res["share_of_summed_fit_gain_by_T0_class"] = {str(k): round(float((d.fit_nse - d.pass_nse).clip(-0.5, 0.5)[v.index].sum() / tot), 3) for k, v in onr.groupby(bins, observed=True)}
    json.dump(res, open(f"{OUT}/ceiling.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
