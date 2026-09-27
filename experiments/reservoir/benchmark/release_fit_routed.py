#!/usr/bin/env python
"""Tune a parametric reservoir release to the gauge below each benchmark dam, driven only by ROUTED inflow.

No dam data enters: no observed inflow, no observed release, no ResOpsUS, no capacity or release caps.
The inflow is the trained model's routed prediction at the benchmark gauge with no reservoir (its eval
zarr), and the gauge observation is used only as the fitting target on training years, as when a
hydrograph is calibrated. Test years score it.

Release models, daily implicit Euler on storage, same-day inflow (`avail = S + I; Q = min(avail/(T+1), Qmax);
S = avail - Q`), nested:
  pass      Q = I                                   the trained model as it is (0 parameters)
  lin       T                                       linear reservoir, S = T Q (Zoch 1934)
  lin_cap   T, Qmax                                 + release cap; Qmax = m x mean train inflow, m in [1.5, 50] or inf
  lin_seas  T0, a, b                                T_t = T0 exp(a sin w_t + b cos w_t), w_t = 2 pi doy / 365.25
  lin_cap_seas  T0, m, a, b
Fit: grid search, best train NSE (WY1997-2001, WY1996 is spin-up), scored on WY2002-2010.

Placement: at 80 of 121 benchmark dams the gauge is on the dam's reach, so the routed prediction there is the
dam's inflow and this equals a reservoir row inside ddrs. At the other 41 (gauge area <= 1.5x the dam's
watershed) the release model also acts on the local inflow between dam and gauge; they are reported apart.

Control: every benchmark gauge is paired with the unregulated gauge of nearest drainage area (no NWM reservoir
upstream, no NWIS peak code 6 in WY1996-2010, no major GAGES-II dam), sampled without replacement. If the release
models lift the controls as much as the dams, the gain is generic smoothing of the model, not dam behaviour.

Usage: release_fit_routed.py [--run RUN_ID] [--procs N]
Writes experiments/reservoir/results/release_fit_routed_<run>.json and
experiments/reservoir/benchmark/release_fit_routed_<run>.csv.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="2026-09-17T16-38-16Z-train-and-test")
ap.add_argument("--procs", type=int, default=16)
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
REG = Path("/home/tbindas/projects/ddrs/output/dam_sandbox/regulation_by_gauge.csv")  # regulation_sizing.py
PEAK = Path("/mnt/ssd1/data/usgs_regulation/derived/peak_regulation_summary.csv")
GII = Path("/mnt/ssd1/data/usgs_regulation/derived/gagesii_regulation.csv")
short = args.run[:20]

T_GRID = np.logspace(np.log10(0.05), np.log10(1000), 30)
T_FINE = np.logspace(np.log10(0.05), np.log10(1000), 80)
M_GRID = np.r_[np.logspace(np.log10(1.5), np.log10(50), 20), np.inf]
AB_GRID = np.linspace(-2.0, 2.0, 9)


def grid(model):
    t = T_FINE if model == "lin" else T_GRID
    m = M_GRID if "cap" in model else np.array([np.inf])
    ab = AB_GRID if "seas" in model else np.array([0.0])
    T, M, A, B = np.meshgrid(t, m, ab, ab if "seas" in model else np.array([0.0]), indexing="ij")
    return T.ravel(), M.ravel(), A.ravel(), B.ravel()


def simulate(I, sinw, cosw, T0, m, a, b, qref, n, obs=None, mask=None):
    """Vectorised over parameter cells. With obs/mask returns train SSE per cell over the first n days;
    without, returns the release series of a single cell."""
    Qmax = m * qref
    S = T0 * np.exp(a * sinw[0] + b * cosw[0]) * I[0]
    sse = np.zeros_like(T0) if obs is not None else None
    Q = np.empty(n) if obs is None else None
    for t in range(n):
        T = T0 * np.exp(a * sinw[t] + b * cosw[t]) if a.any() or b.any() else T0
        avail = S + I[t]
        q = np.minimum(avail / (T + 1.0), Qmax)
        S = avail - q
        if obs is None:
            Q[t] = q[0]
        elif mask[t]:
            sse += (q - obs[t]) ** 2
    return sse if obs is not None else Q


def metrics(p, o, mask):
    mm = mask & np.isfinite(o) & np.isfinite(p)
    p, o = p[mm], o[mm]
    nse = 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    a, b = p.std() / o.std(), p.mean() / o.mean()
    return dict(nse=float(nse), kge=float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)),
                r=float(r), alpha=float(a), beta=float(b))


def fit_one(job):
    gauge, I, O, doy, train, test, n_fit = job
    w = 2 * np.pi * doy / 365.25
    sinw, cosw = np.sin(w), np.cos(w)
    I = np.where(np.isfinite(I), I, np.nanmean(I))
    qref = float(I[train].mean())
    otr = O[train & np.isfinite(O)]
    sst = ((otr - otr.mean()) ** 2).sum()
    out = dict(gauge=gauge, pass_train=metrics(I, O, train), pass_test=metrics(I, O, test))
    for model in ["lin", "lin_cap", "lin_seas", "lin_cap_seas"]:
        T0, M, A, B = grid(model)
        sse = simulate(I, sinw, cosw, T0, M, A, B, qref, n_fit, obs=O, mask=train & np.isfinite(O))
        nse_tr = 1 - sse / sst
        k = int(np.argmax(nse_tr))
        q = simulate(I, sinw, cosw, T0[k:k + 1], M[k:k + 1], A[k:k + 1], B[k:k + 1], qref, len(I))
        near = nse_tr >= nse_tr[k] - 0.02
        out[model] = dict(T0=float(T0[k]), m=float(M[k]), a=float(A[k]), b=float(B[k]),
                          T0_band=[float(T0[near].min()), float(T0[near].max())],
                          train=metrics(q, O, train), test=metrics(q, O, test))
    return out


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    b = np.median(rng.choice(d, (2000, len(d))), axis=1) if len(d) > 1 else np.array([np.nan])
    return dict(n=int(len(d)), median=float(np.median(d)) if len(d) else np.nan,
                ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                n_up=up, n_down=int(len(nz) - up), sign_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else np.nan)


if __name__ == "__main__":
    z = zarr.open(str(RUNS / args.run / "eval/predictions.zarr"), mode="r")
    ids = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    P, O = z["predictions"][:], z["observations"][:]
    wy = t.year + (t.month >= 10)
    train = np.asarray((wy >= 1997) & (wy <= 2001))
    test = np.asarray((wy >= 2002) & (wy <= 2010))
    n_fit = int(np.flatnonzero(train)[-1]) + 1
    doy = np.asarray(t.dayofyear, float)

    bench = pd.read_csv(HERE / "dam_benchmark.csv", dtype={"huc2": str, "gauge": str})
    audit = pd.read_csv(HERE / "attribute_audit.csv", dtype={"gauge": str}).set_index("gauge")
    reg = pd.read_csv(REG, dtype={"STAID": str}).set_index("STAID")
    peak = pd.read_csv(PEAK, dtype={"STAID": str}).set_index("STAID")
    gii = pd.read_csv(GII, dtype={"STAID": str}).set_index("STAID")
    pool = reg[(reg.cls == "none") & reg.index.isin(ids)].copy()
    pool = pool[pool.index.map(lambda s: peak.n_years_code6_wy1996_2010.get(s, 0) == 0)]
    pool = pool[pool.index.map(lambda s: gii.MAJ_NDAMS_2009.get(s, 0) == 0)]
    la = np.log(pool.area_km2)
    controls, used = [], set()
    for _, b in bench.sort_values("gauge_area_km2").iterrows():
        order = (la - np.log(b.gauge_area_km2)).abs().sort_values().index
        c = next(s for s in order if s not in used)
        used.add(c)
        controls.append((b.gauge, c))
    ctrl_of = dict(controls)

    gauges = list(bench.gauge) + [c for _, c in controls]
    jobs = [(g, P[ids.get_loc(g)].astype(float), O[ids.get_loc(g)].astype(float), doy, train, test, n_fit) for g in gauges]
    with Pool(args.procs) as p:
        fits = {r["gauge"]: r for r in p.map(fit_one, jobs)}

    rows = []
    MODELS = ["lin", "lin_cap", "lin_seas", "lin_cap_seas"]
    for g in gauges:
        f = fits[g]
        is_dam = g in set(bench.gauge)
        r = dict(gauge=g, role="dam" if is_dam else "control",
                 pair=g if is_dam else next(k for k, v in ctrl_of.items() if v == g),
                 nse_pass_train=f["pass_train"]["nse"], nse_pass_test=f["pass_test"]["nse"],
                 kge_pass_test=f["pass_test"]["kge"], alpha_pass_test=f["pass_test"]["alpha"])
        for mdl in MODELS:
            r[f"nse_{mdl}_train"] = f[mdl]["train"]["nse"]
            r[f"nse_{mdl}_test"] = f[mdl]["test"]["nse"]
            r[f"kge_{mdl}_test"] = f[mdl]["test"]["kge"]
            r[f"alpha_{mdl}_test"] = f[mdl]["test"]["alpha"]
            for k in ["T0", "m", "a", "b"]:
                r[f"{mdl}_{k}"] = f[mdl][k]
            r[f"{mdl}_T0_lo"], r[f"{mdl}_T0_hi"] = f[mdl]["T0_band"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df = df.merge(bench[["gauge", "huc2", "tier", "LAKE_NAME", "main_purpose", "gauge_area_km2", "gauge_dor"]],
                  left_on="pair", right_on="gauge", how="left", suffixes=("", "_b")).drop(columns="gauge_b")
    df["dam_on_gauge_reach"] = df.pair.map(audit.dam_on_gauge_reach)
    df.to_csv(HERE / f"release_fit_routed_{short}.csv", index=False)

    dam, ctl = df[df.role == "dam"].set_index("pair"), df[df.role == "control"].set_index("pair")
    res = dict(run=args.run, train="WY1997-2001", test="WY2002-2010", n_dams=len(dam), n_controls=len(ctl),
               median_test_nse={}, paired_test_dnse={}, paired_test_dkge={}, dam_minus_control={}, by_placement={},
               by_huc2={}, lin_T0={})
    res["median_test_nse"]["pass"] = dict(dam=float(dam.nse_pass_test.median()), control=float(ctl.nse_pass_test.median()))
    for mdl in MODELS:
        dd = dam[f"nse_{mdl}_test"] - dam.nse_pass_test
        dc = ctl[f"nse_{mdl}_test"] - ctl.nse_pass_test
        res["median_test_nse"][mdl] = dict(dam=float(dam[f"nse_{mdl}_test"].median()), control=float(ctl[f"nse_{mdl}_test"].median()))
        res["paired_test_dnse"][mdl] = dict(dam=paired(dd), control=paired(dc))
        res["paired_test_dkge"][mdl] = dict(dam=paired(dam[f"kge_{mdl}_test"] - dam.kge_pass_test),
                                            control=paired(ctl[f"kge_{mdl}_test"] - ctl.kge_pass_test))
        res["dam_minus_control"][mdl] = paired(dd - dc.reindex(dd.index))
        res["by_placement"][mdl] = {("on_reach" if k else "off_reach"): paired(dd[dam.dam_on_gauge_reach == k]) for k in (True, False)}
        res["by_huc2"][mdl] = {h: dict(n=int(len(v)), median=float(v.median()), up=int((v > 0).sum()))
                               for h, v in dd.groupby(dam.huc2)}
    res["lin_T0"] = dict(dam_median=float(dam.lin_T0.median()), control_median=float(ctl.lin_T0.median()),
                         dam_quartiles=[float(x) for x in dam.lin_T0.quantile([0.25, 0.75])],
                         control_quartiles=[float(x) for x in ctl.lin_T0.quantile([0.25, 0.75])])
    res["alpha_test_median"] = {m_: dict(dam=float(dam[f"alpha_{m_}_test"].median()), control=float(ctl[f"alpha_{m_}_test"].median()))
                                for m_ in ["pass"] + MODELS}
    out = HERE.parent / "results" / f"release_fit_routed_{short}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ["median_test_nse", "paired_test_dnse", "dam_minus_control", "by_placement", "lin_T0",
                                          "alpha_test_median"]}, indent=1, default=float))
    print("->", out)
