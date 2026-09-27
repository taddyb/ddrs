#!/usr/bin/env python
"""Expected results for the dam-release smoke set, before any release code exists in ddrs.

Input: output/reservoir_smoke/pred_1981_2010.zarr from run_smoke_eval.sh (the no-dam sr_n0_gamma head routed over
1981-10-01..2010-09-30 on the smoke gauges, CPU). The routed prediction at each gauge is the dam's inflow (the
smoke dams sit near their gauges); observations are the fitting target only.
Release laws, daily implicit Euler on storage with same-day inflow, as in ../benchmark/release_fit_routed.py:
  pass      Q = I, the no-dam model as it is
  lin       S = T0 Q                                        1 parameter
  lin_seas  T_t = T0 exp(a sin w_t + b cos w_t)            3 parameters
(The cap is left out: with routed inflow it overfit in the benchmark screen.)
Windows: WY1982 spin-up, fit WY1983-1995 (best NSE on a grid), score WY1996-2010, the ddrs train / test split.
Controls (no dam) are fitted the same way: a law that only helps dams leaves them near pass-through.

Writes expected_release_fit.csv and expected_release_fit.json next to this script, and
output/reservoir_smoke/smoke_series.json (daily test-period series for the artifact's hydrograph viewer).
"""
from __future__ import annotations

import json
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


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    return dict(n=int(len(d)), median=float(np.median(d)), n_up=up, n_down=int(len(nz) - up),
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
    smoke = pd.read_csv(HERE / "smoke_gauges.csv", dtype={"STAID": str, "huc2": str}).set_index("STAID")

    rows, series = [], {}
    for s, g in smoke.iterrows():
        I, Ob = P[ids.get_loc(s)], O[ids.get_loc(s)]
        I = np.where(np.isfinite(I), I, np.nanmean(I))
        q_lin, p_lin = fit(I, Ob, sinw, cosw, train, seasonal=False)
        q_sea, p_sea = fit(I, Ob, sinw, cosw, train, seasonal=True)
        r = dict(STAID=s, huc2=g.huc2, role=g.role, relaxed=str(g.get("relaxed")) == "True",
                 staname=g.staname, dam_name=g.dam_name, dam_storage_mcm=g.dam_storage_mcm, dam_purpose=g.dam_purpose,
                 dam_year=g.dam_year, area_km2=g.area_km2, area_ratio=g.area_ratio, nid_dor=g.nid_dor)
        for lab, q in [("pass", I), ("lin", q_lin), ("seas", q_sea)]:
            for k, v in metrics(q, Ob, test).items():
                r[f"{k}_{lab}_test"] = v
            r[f"nse_{lab}_train"] = metrics(q, Ob, train)["nse"]
        r.update({f"lin_{k}": v for k, v in p_lin.items()})
        r.update({f"seas_{k}": v for k, v in p_sea.items()})
        rows.append(r)
        te = np.flatnonzero(test)
        fmt = lambda x: [None if not np.isfinite(v) else float(f"{v:.3g}") for v in x[te]]
        series[s] = dict(obs=fmt(Ob), nodam=fmt(I), release=fmt(q_sea))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "expected_release_fit.csv", index=False)

    dam, ctl = df[df.role == "dam"], df[df.role == "control"]
    res = dict(
        head="2026-09-12T23-39-03Z-train-and-test (sr_n0_gamma) @ epoch_50_mb_9, CPU, no dams",
        windows=dict(spin_up="WY1982", fit="WY1983-1995", test="WY1996-2010"),
        n=dict(dam=len(dam), control=len(ctl)),
        median_test_nse={lab: dict(dam=float(dam[f"nse_{lab}_test"].median()), control=float(ctl[f"nse_{lab}_test"].median()))
                         for lab in ["pass", "lin", "seas"]},
        median_test_kge={lab: dict(dam=float(dam[f"kge_{lab}_test"].median()), control=float(ctl[f"kge_{lab}_test"].median()))
                         for lab in ["pass", "lin", "seas"]},
        paired_test_dnse={lab: dict(dam=paired(dam[f"nse_{lab}_test"] - dam.nse_pass_test),
                                    control=paired(ctl[f"nse_{lab}_test"] - ctl.nse_pass_test)) for lab in ["lin", "seas"]},
        paired_test_dkge={lab: dict(dam=paired(dam[f"kge_{lab}_test"] - dam.kge_pass_test),
                                    control=paired(ctl[f"kge_{lab}_test"] - ctl.kge_pass_test)) for lab in ["lin", "seas"]},
        fitted_T0_days=dict(lin_dam_median=float(dam.lin_T0.median()), lin_control_median=float(ctl.lin_T0.median()),
                            seas_dam_median=float(dam.seas_T0.median()), seas_control_median=float(ctl.seas_T0.median())),
    )
    json.dump(res, open(HERE / "expected_release_fit.json", "w"), indent=1)
    dates = [d.strftime("%Y-%m-%d") for d in t[test]]
    keep = ["STAID", "huc2", "role", "relaxed", "staname", "dam_name", "dam_storage_mcm", "dam_purpose", "dam_year", "area_km2",
            "area_ratio", "nse_pass_test", "nse_lin_test", "nse_seas_test", "kge_pass_test", "kge_seas_test", "lin_T0",
            "seas_T0", "seas_a", "seas_b"]
    meta = json.loads(df[keep].round(4).to_json(orient="records"))
    json.dump(dict(start=dates[0], n_days=len(dates), meta=meta, gauges=series, summary=res),
              open(OUT / "smoke_series.json", "w"), separators=(",", ":"))
    print(json.dumps(res, indent=1))
    pd.set_option("display.width", 250)
    print(df[["huc2", "role", "STAID", "dam_name", "nse_pass_test", "nse_lin_test", "nse_seas_test", "kge_pass_test",
              "kge_seas_test", "lin_T0", "seas_T0", "seas_a", "seas_b"]].round(3).to_string(index=False))
