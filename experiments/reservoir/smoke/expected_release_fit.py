#!/usr/bin/env python
"""Expected results for the dam-release smoke set, before any release code exists in ddrs.

Input: output/reservoir_smoke/pred_1981_2010.zarr from run_smoke_eval.sh (the no-dam sr_n0_gamma head routed over
1981-10-01..2010-09-30 on the smoke gauges, CPU). The routed prediction at each gauge stands in for the dam's inflow
(the nearest large dam is within 1.5x the gauge's drainage area); observations are the fitting target only.
Release laws, daily implicit Euler on storage with same-day inflow, as in ../benchmark/release_fit_routed.py:
  pass      Q = I, the no-dam model as it is
  lin       S = T0 Q                                        1 parameter
  lin_seas  T_t = T0 exp(a sin w_t + b cos w_t)            3 parameters
(The cap is left out: with routed inflow it overfit in the benchmark screen.)
Windows: WY1982 spin-up, fit WY1983-1995 (best NSE on a grid), score WY1996-2010, the ddrs train / test split.
Controls are fitted the same way; each dam gauge's gain is also read against its own matched control's gain.

Writes expected_release_fit.csv / .json next to this script, and for the results page output/reservoir_smoke/web/:
index.json (per-gauge rows, summary, layout) and series/huc<NN>.bin (per region: for each gauge in index order the
observed, no-dam and tuned-release daily series over WY1996-2010, uint16 little-endian, code = round((log10(q) - LO)
/ (HI - LO) * 65534) with q <= 10**LO -> 0 and missing -> 65535).
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "output" / "reservoir_smoke"
T_GRID = np.logspace(np.log10(0.05), np.log10(1000), 40)
T_FINE = np.logspace(np.log10(0.05), np.log10(1000), 100)
AB = np.linspace(-2.0, 2.0, 9)
LO, HI = -4.0, 5.0


def simulate(I, sinw, cosw, T0, a, b, n, obs=None, mask=None):
    """Vectorised over parameter cells; with obs returns train SSE, else the release of cell 0."""
    T = T0 * np.exp(a * sinw[0] + b * cosw[0])
    S = T * I[0]
    sse = np.zeros_like(T0) if obs is not None else None
    Q = np.empty(n) if obs is None else None
    seasonal = bool(np.any(a) or np.any(b))
    for t in range(n):
        if seasonal:
            T = T0 * np.exp(a * sinw[t] + b * cosw[t])
        avail = S + I[t]
        q = avail / (T + 1.0)
        S = avail - q
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
    a, b = p.std() / o.std(), p.mean() / o.mean()
    return dict(nse=float(nse), kge=float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)), r=float(r),
                alpha=float(a), beta=float(b))


def fit(I, O, sinw, cosw, train, seasonal):
    t0 = T_GRID if seasonal else T_FINE
    ab = AB if seasonal else np.array([0.0])
    T0, A, B = (x.ravel() for x in np.meshgrid(t0, ab, ab, indexing="ij"))
    n_fit = int(np.flatnonzero(train)[-1]) + 1
    m = train & np.isfinite(O)
    otr = O[m]
    nse = 1 - simulate(I, sinw, cosw, T0, A, B, n_fit, obs=O, mask=m) / ((otr - otr.mean()) ** 2).sum()
    k = int(np.argmax(nse))
    near = nse >= nse[k] - 0.02
    q = simulate(I, sinw, cosw, T0[k:k + 1], A[k:k + 1], B[k:k + 1], len(I))
    return q, dict(T0=float(T0[k]), a=float(A[k]), b=float(B[k]), T0_lo=float(T0[near].min()), T0_hi=float(T0[near].max()))


def fit_gauge(job):
    s, I, Ob, sinw, cosw, train, test = job
    I = np.where(np.isfinite(I), I, np.nanmean(I))
    q_lin, p_lin = fit(I, Ob, sinw, cosw, train, seasonal=False)
    q_sea, p_sea = fit(I, Ob, sinw, cosw, train, seasonal=True)
    r = {"STAID": s}
    for lab, q in [("pass", I), ("lin", q_lin), ("seas", q_sea)]:
        for k, v in metrics(q, Ob, test).items():
            r[f"{k}_{lab}_test"] = v
        r[f"nse_{lab}_train"] = metrics(q, Ob, train)["nse"]
    r.update({f"lin_{k}": v for k, v in p_lin.items()})
    r.update({f"seas_{k}": v for k, v in p_sea.items()})
    return r, np.stack([Ob[test], I[test], q_sea[test]])


def encode(x):
    c = np.full(x.shape, 65535, dtype="<u2")
    ok = np.isfinite(x)
    lg = np.log10(np.maximum(x[ok], 10 ** LO))
    c[ok] = np.clip(np.round((lg - LO) / (HI - LO) * 65534), 0, 65534).astype("<u2")
    return c


def finite(o):
    """NaN / inf -> None, so the JSON is strict (the page's fetch().json() rejects a bare NaN)."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    b = np.median(rng.choice(d, (2000, len(d))), axis=1) if len(d) > 1 else np.array([np.nan])
    return dict(n=int(len(d)), median=float(np.median(d)) if len(d) else float("nan"),
                ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], n_up=up, n_down=int(len(nz) - up),
                sign_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else float("nan"))


if __name__ == "__main__":
    z = zarr.open(str(OUT / "pred_1981_2010.zarr"), mode="r")
    ids = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    P, O = z["predictions"][:].astype(float), z["observations"][:].astype(float)
    wy = np.asarray(t.year + (t.month >= 10))
    train, test = (wy >= 1983) & (wy <= 1995), (wy >= 1996) & (wy <= 2010)
    w = 2 * np.pi * np.asarray(t.dayofyear) / 365.25
    sinw, cosw = np.sin(w), np.cos(w)
    smoke = pd.read_csv(HERE / "smoke_gauges.csv", dtype={"STAID": str, "huc2": str, "control_for": str}).set_index("STAID")
    g3c = pd.read_csv("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv", dtype={"STAID": str}).set_index("STAID").COMID

    jobs = [(s, P[ids.get_loc(s)], O[ids.get_loc(s)], sinw, cosw, train, test) for s in smoke.index]
    with Pool(16) as pool:
        out = pool.map(fit_gauge, jobs)
    fits = pd.DataFrame([r for r, _ in out]).set_index("STAID")
    series = {r["STAID"]: x for r, x in out}
    df = smoke.join(fits)
    df["relaxed"] = df.relaxed.astype(str) == "True"
    df["cascade"] = df.cascade.astype(str) == "True"
    is_dam = (df.role == "dam").values
    gauge_comid = df.index.map(g3c).to_numpy(dtype=float)
    df["on_reach"] = is_dam & (df.dam_COMID.to_numpy(dtype=float) == gauge_comid)
    for lab in ["lin", "seas"]:
        df[f"d_{lab}"] = df[f"nse_{lab}_test"] - df.nse_pass_test
        df[f"dk_{lab}"] = df[f"kge_{lab}_test"] - df.kge_pass_test
    ctl_gain = df[df.role == "control"].set_index("control_for").d_seas
    df["d_seas_minus_control"] = np.where(df.role == "dam", df.d_seas - df.index.map(ctl_gain).astype(float), np.nan)
    df.to_csv(HERE / "expected_release_fit.csv")

    dam, ctl = df[df.role == "dam"], df[df.role == "control"]
    dor_bins = pd.cut(dam.nid_dor, [0, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
    res = dict(
        head="2026-09-12T23-39-03Z-train-and-test (sr_n0_gamma) @ epoch_50_mb_9, CPU, no dams",
        windows=dict(spin_up="WY1982", fit="WY1983-1995", test="WY1996-2010"),
        n=dict(dam=len(dam), control=len(ctl), on_reach=int(dam.on_reach.sum()), cascade=int(dam.cascade.sum())),
        median_test_nse={lab: dict(dam=float(dam[f"nse_{lab}_test"].median()), control=float(ctl[f"nse_{lab}_test"].median()))
                         for lab in ["pass", "lin", "seas"]},
        median_test_kge={lab: dict(dam=float(dam[f"kge_{lab}_test"].median()), control=float(ctl[f"kge_{lab}_test"].median()))
                         for lab in ["pass", "lin", "seas"]},
        paired_dnse={lab: dict(dam=paired(dam[f"d_{lab}"]), control=paired(ctl[f"d_{lab}"])) for lab in ["lin", "seas"]},
        paired_dkge={lab: dict(dam=paired(dam[f"dk_{lab}"]), control=paired(ctl[f"dk_{lab}"])) for lab in ["lin", "seas"]},
        dam_minus_matched_control=paired(dam.d_seas_minus_control),
        by_placement={"on_reach": paired(dam.d_seas[dam.on_reach]), "off_reach": paired(dam.d_seas[~dam.on_reach])},
        by_cascade={"single": paired(dam.d_seas[~dam.cascade]), "cascade": paired(dam.d_seas[dam.cascade])},
        by_dor={str(k): paired(v) for k, v in dam.d_seas.groupby(dor_bins, observed=True)},
        by_huc2={h: dict(dam=paired(g[g.role == "dam"].d_seas), control=paired(g[g.role == "control"].d_seas)) for h, g in df.groupby("huc2")},
        fitted_T0_days=dict(dam=[float(x) for x in dam.seas_T0.quantile([0.25, 0.5, 0.75])],
                            control=[float(x) for x in ctl.seas_T0.quantile([0.25, 0.5, 0.75])],
                            dam_at_upper_wall=int((dam.seas_T0 >= 999).sum()), dam_at_floor=int((dam.seas_T0 <= 0.051).sum()),
                            control_at_floor=int((ctl.seas_T0 <= 0.051).sum())),
    )
    res = finite(res)
    json.dump(res, open(HERE / "expected_release_fit.json", "w"), indent=1, allow_nan=False)

    web = OUT / "web"
    (web / "series").mkdir(parents=True, exist_ok=True)
    layout = {}
    for h, g in df.groupby("huc2"):
        order = list(g.index)
        np.concatenate([encode(series[s]).ravel() for s in order]).tofile(web / "series" / f"huc{h}.bin")
        layout[h] = dict(file=f"series/huc{h}.bin", gauges=order)
    keep = ["huc2", "role", "relaxed", "cascade", "on_reach", "control_for", "control_cross_huc", "staname", "dam_name",
            "dam_storage_mcm", "dam_purpose", "dam_year", "n_big", "area_km2", "area_ratio", "nid_dor", "nse_pass_test",
            "nse_lin_test", "nse_seas_test", "kge_pass_test", "kge_seas_test", "d_lin", "d_seas", "d_seas_minus_control",
            "seas_T0", "seas_a", "seas_b", "lin_T0"]
    meta = json.loads(df[keep].reset_index().round(4).to_json(orient="records"))
    d0 = t[test][0].strftime("%Y-%m-%d")
    json.dump(dict(start=d0, n_days=int(test.sum()), lo=LO, hi=HI, layout=layout, meta=meta, summary=res),
              open(web / "index.json", "w"), separators=(",", ":"), allow_nan=False)
    print(json.dumps({k: res[k] for k in ["n", "median_test_nse", "paired_dnse", "dam_minus_matched_control", "by_placement",
                                          "by_cascade", "by_dor", "fitted_T0_days"]}, indent=1))
