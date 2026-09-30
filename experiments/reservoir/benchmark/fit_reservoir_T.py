#!/usr/bin/env python
"""Fit a linear-reservoir residence time T per benchmark dam from ResOpsUS, for option C in ddrs.

The model is what a ddrs dam row computes under `params.use_reservoirs`: Muskingum with K = T,
X = 0, trapezoid rule, Q_t = c1 I_t + c2 I_{t-1} + c3 Q_{t-1} (same-day inflow). Here it runs on
the daily ResOpsUS step; ddrs runs it hourly with the same T. Each contiguous run of paired
inflow/outflow days (>= 60 days) starts at the observed release.

Fit window: ResOpsUS days OUTSIDE the ddrs eval window WY1997-2010 when they hold at least three
years of paired days, so the ddrs score at the benchmark gauge is out of sample for T. Otherwise
all paired days, flagged `fit_window = all`. T grid: 200 log points in [1/24, 3000] days (1/24 is
the floor `read_reservoir_table` accepts). Dams with no paired inflow/outflow get no fit.

Tables written for ddrs (`data_sources.reservoirs`, header COMID,T_days):
  reservoirs_T_fit.csv    fitted dams only
  reservoirs_T_prior.csv  fitted dams, plus every other benchmark dam at the median fitted T
Per-dam detail: reservoir_T_fits.csv. Run under DDR's venv:
  ~/projects/ddr/.venv/bin/python experiments/reservoir/benchmark/fit_reservoir_T.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
RESOPS = Path("/mnt/ssd1/data/resops/resopsus/ResOpsUS/time_series_all")
MIN_SEG, MIN_FIT_DAYS = 60, 3 * 365
TS = np.logspace(np.log10(1 / 24), np.log10(3000), 200)

bench = pd.read_csv(HERE / "dam_benchmark.csv", dtype={"huc2": str, "gauge": str})


def segments(ok):
    """Start/stop indices of runs of True at least MIN_SEG long."""
    d = np.diff(np.r_[0, ok.astype(int), 0])
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(starts, stops) if b - a >= MIN_SEG]


def simulate(I, O, segs, T):
    """Trapezoid X = 0 per segment, initialised at the observed release; NaN outside segments."""
    den = 2 * T + 1.0
    c1 = c2 = 1.0 / den
    c3 = (2 * T - 1.0) / den
    Q = np.full(len(I), np.nan)
    for a, b in segs:
        x = I[a:b]
        # lfilter's transposed state z gives y[0] = c1 x[0] + z, so this starts at Q[a] = O[a]
        y, _ = lfilter([c1, c2], [1.0, -c3], x, zi=np.array([O[a] - c1 * x[0]]))
        Q[a:b] = y
    return Q


def nse(sim, obs, mask):
    m = mask & np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < MIN_SEG:
        return np.nan
    s, o = sim[m], obs[m]
    return 1.0 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


rows = []
for _, r in bench.iterrows():
    base = dict(huc2=r.huc2, tier=int(r.tier), GRAND_ID=int(r.GRAND_ID), COMID=int(r.COMID), gauge=r.gauge,
                LAKE_NAME=r.LAKE_NAME, main_purpose=r.main_purpose, in_resopsus=bool(r.IN_RESOPSUS))
    f = RESOPS / f"ResOpsUS_{int(r.GRAND_ID)}.csv"
    if not f.exists():
        rows.append(dict(base, status="not in ResOpsUS"))
        continue
    ts = pd.read_csv(f, parse_dates=["date"], na_values=["NA"])
    ts = ts.set_index("date").asfreq("D")
    I, O = ts.inflow.to_numpy(float), ts.outflow.to_numpy(float)
    n_in, n_out = int(np.isfinite(I).sum()), int(np.isfinite(O).sum())
    both = np.isfinite(I) & np.isfinite(O)
    if both.sum() < MIN_SEG:
        rows.append(dict(base, status="no paired inflow/outflow", n_inflow_days=n_in, n_outflow_days=n_out))
        continue
    wy = ts.index.year + (ts.index.month >= 10)
    in_eval = (wy >= 1997) & (wy <= 2010)
    segs = segments(both)
    seg_mask = np.zeros(len(I), bool)
    for a, b in segs:
        seg_mask[a:b] = True
    out_mask = seg_mask & ~in_eval
    fit_mask, window = (out_mask, "outside WY1997-2010") if out_mask.sum() >= MIN_FIT_DAYS else (seg_mask, "all")
    if fit_mask.sum() < MIN_SEG:
        rows.append(dict(base, status="no paired segment >= 60 d", n_inflow_days=n_in, n_outflow_days=n_out))
        continue
    Ii = np.where(both, I, 0.0)
    fits = np.array([nse(simulate(Ii, O, segs, T), O, fit_mask) for T in TS])
    k = int(np.nanargmax(fits))
    band = TS[fits >= fits[k] - 0.02]
    q = simulate(Ii, O, segs, TS[k])
    rows.append(dict(
        base, status="fitted", fit_window=window, fit_days=int(fit_mask.sum()),
        paired_days=int(both.sum()), paired_days_eval=int((seg_mask & in_eval).sum()),
        T_days=float(TS[k]), T_lo=float(band.min()), T_hi=float(band.max()), on_upper_wall=bool(k == len(TS) - 1),
        nse_fit=float(fits[k]), nse_pass_fit=nse(Ii, O, fit_mask),
        nse_eval_obs_inflow=nse(q, O, seg_mask & in_eval), nse_pass_eval=nse(Ii, O, seg_mask & in_eval),
        inflow_over_release=float(Ii[fit_mask].sum() / O[fit_mask].sum()),
        mean_inflow=float(Ii[fit_mask].mean()), buffer_mcm=float(TS[k] * 86400 * Ii[fit_mask].mean() / 1e6),
        n_inflow_days=n_in, n_outflow_days=n_out))

df = pd.DataFrame(rows)
df.to_csv(HERE / "reservoir_T_fits.csv", index=False)
fit = df[df.status == "fitted"]
T_med = float(fit.T_days.median())
fit[["COMID", "T_days"]].to_csv(HERE / "reservoirs_T_fit.csv", index=False)
prior = df[["COMID"]].assign(T_days=df.T_days.fillna(T_med))
prior.to_csv(HERE / "reservoirs_T_prior.csv", index=False)

summary = dict(
    n_dams=len(df), status=df.status.value_counts().to_dict(),
    fit_window=fit.fit_window.value_counts().to_dict(), T_median_days=T_med,
    T_quartiles_days=[float(x) for x in fit.T_days.quantile([0.25, 0.75])],
    n_T_on_upper_wall=int(fit.on_upper_wall.sum()),
    median_nse_fit=float(fit.nse_fit.median()), median_nse_pass_fit=float(fit.nse_pass_fit.median()),
    median_nse_eval_obs_inflow=float(fit.nse_eval_obs_inflow.median()),
    median_nse_pass_eval=float(fit.nse_pass_eval.median()),
    n_with_eval_overlap=int(fit.nse_eval_obs_inflow.notna().sum()),
    by_tier={int(t): g.status.value_counts().to_dict() for t, g in df.groupby("tier")},
)
json.dump(summary, open(HERE / "reservoir_T_fits_summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
pd.set_option("display.width", 220)
print(fit[["huc2", "tier", "LAKE_NAME", "main_purpose", "fit_window", "T_days", "T_lo", "T_hi", "nse_fit", "nse_pass_fit",
           "nse_eval_obs_inflow", "inflow_over_release"]].round(3).to_string())
