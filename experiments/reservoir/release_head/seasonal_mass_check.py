#!/usr/bin/env python
"""Why check 2 fails at long T: a Muskingum row with a time-varying K does not conserve storage.

At the on-reach smoke gauges of check 2, route the no-dam daily gauge flow (repeat-24 to hourly) through the seasonal
bucket with three discretisations and compare their daily means with ddrs's check-2 output and with the offline fit:

  muskingum   K := T(t+1), X := 0 (what ddrs does):  Q1 = [dt/2 (I0 + I1) + (T1 - dt/2) Q0] / (T1 + dt/2)
  storage     S = T Q conserved (trapezoid):          Q1 = [dt/2 (I0 + I1) + (T0 - dt/2) Q0] / (T1 + dt/2)
  offline     daily implicit Euler on S (expected_release_fit.simulate)

The two hourly forms differ only in c3's numerator (T of the step's start vs its end). With T varying, the
Muskingum form carries Q across a change in T, so S = T Q jumps: a source of Q dT/dt, large when T0 is long and the
seasonal amplitude is large.

Usage: seasonal_mass_check.py <no-dam pred zarr> <ddrs check-2 zarr> <fit_dams.csv> [staid ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "smoke"))
from expected_release_fit import simulate  # noqa: E402

DT = 1.0 / 24.0  # days


def load(p):
    z = zarr.open(p, mode="r")
    return pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]), \
        pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]")), z["predictions"][:]


def nse(s, r):
    m = np.isfinite(s) & np.isfinite(r)
    s, r = s[m], r[m]
    return float(1 - ((s - r) ** 2).sum() / ((r - r.mean()) ** 2).sum())


def hourly(inflow_daily, t0, a, b, days, conserve):
    hours = np.arange(len(inflow_daily) * 24)
    doy = np.repeat(np.asarray(days.dayofyear, dtype=float), 24) + (hours % 24) / 24.0
    w = 2 * np.pi * doy / 365.25
    T = np.maximum(t0 * np.exp(a * np.sin(w) + b * np.cos(w)), DT)
    I = np.repeat(inflow_daily, 24)
    q = np.empty_like(I)
    q[0] = I[0]
    for h in range(1, len(I)):
        tprev = T[h - 1] if conserve else T[h]
        q[h] = (DT / 2 * (I[h - 1] + I[h]) + (tprev - DT / 2) * q[h - 1]) / (T[h] + DT / 2)
    return q.reshape(-1, 24).mean(axis=1)


def main():
    ref_p, ddrs_p, fit_p = sys.argv[1:4]
    ids, t, ref = load(ref_p)
    _, _, dd = load(ddrs_p)
    fits = pd.read_csv(fit_p, dtype={"STAID": str}).set_index("STAID")
    staids = sys.argv[4:] or list(fits.index)
    wy = np.asarray(t.year + (t.month >= 10))
    test = (wy >= 1996) & (wy <= 2010)
    w = 2 * np.pi * np.asarray(t.dayofyear) / 365.25
    rows = []
    for s in staids:
        i = ids.get_loc(s)
        f = fits.loc[s]
        I = ref[i].astype(float)
        I = np.where(np.isfinite(I), I, np.nanmean(I))
        off = simulate(I, np.sin(w), np.cos(w), np.array([f.T_days]), np.array([f.a]), np.array([f.b]), len(I))
        mus = hourly(I, f.T_days, f.a, f.b, t, conserve=False)
        sto = hourly(I, f.T_days, f.a, f.b, t, conserve=True)
        d = dd[i].astype(float)
        rows.append(dict(staid=s, T0=f.T_days, a=f.a, b=f.b,
                         ddrs_vs_muskingum=nse(d[test], mus[test]), ddrs_vs_storage=nse(d[test], sto[test]),
                         offline_vs_storage=nse(off[test], sto[test]), offline_vs_muskingum=nse(off[test], mus[test]),
                         ddrs_vs_offline=nse(d[test], off[test]),
                         mass_ddrs=d[test].mean() / I[test].mean(), mass_storage=sto[test].mean() / I[test].mean()))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(out.drop(columns="staid").median().round(4).to_string())


if __name__ == "__main__":
    main()
