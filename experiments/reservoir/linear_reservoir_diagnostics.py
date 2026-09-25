#!/usr/bin/env python
"""What the fitted linear reservoir of dam_sandbox.py actually buys, at the four sandbox dams.

dam_sandbox.py drives its linear reservoir with the previous day's inflow (`inflow[t - 1]`) and
compares it with a pass-through that uses the same day's inflow. This script separates:
  pass-through lagged 0..4 days                      (pure delay)
  implicit-Euler linear reservoir, inflow lag 0..3   (attenuation, with and without the delay)
  trapezoidal linear reservoir = Muskingum X = 0     (what an MC row with K = T, X = 0 computes)
  Muskingum (K, X) fitted, X in [0, 0.5]            (the most one MC row can do at the dam)
  linear reservoir x a volume factor                 (whether mass balance is what is missing)
plus the NSE profile over T (the band within 0.02 of the optimum) and the largest test events.

Train WY1997-2001, test WY2002-2010, daily NSE; inflow = upstream gauges observed + summed Q' of
the ungauged remainder, as in dam_sandbox.py. Results: experiments/reservoir/results/linear_diagnostics.json
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, lfilter_zi

R = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/baseline")
OUT = Path(__file__).resolve().parent / "results" / "linear_diagnostics.json"

m = json.load(open(R / "manifest.json"))
G, T = m["n_gauges"], m["n_days"]
P = np.fromfile(R / "predictions.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
O = np.fromfile(R / "observations.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
gid = [str(g) for g in m["gage_ids"]]
t0 = dt.date.fromisoformat(str(m["time_range_daily"][0])[:10])
dates = np.array([t0 + dt.timedelta(days=i) for i in range(T)])
wy = np.array([d.year + (1 if d.month >= 10 else 0) for d in dates])
train = (wy >= 1997) & (wy <= 2001)
test = (wy >= 2002) & (wy <= 2010)

# name: (release gauge, inflow gauges, assumed capacity MCM as in the findings docs)
DAMS = {
    "Raystown": ("01563200", ["01562000"], 940),
    "Abiquiu": ("08287000", ["08286500"], 1480),
    "Alamo": ("09426000", ["09424450", "09424900"], 1290),
    "Santa Rosa": ("08382830", ["08382650"], 550),
}


def nse(sim, obs, mask):
    mm = mask & np.isfinite(obs) & np.isfinite(sim)
    s, o = sim[mm], obs[mm]
    return 1.0 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def kge_parts(sim, obs, mask):
    mm = mask & np.isfinite(obs) & np.isfinite(sim)
    s, o = sim[mm], obs[mm]
    return np.corrcoef(s, o)[0, 1], s.std() / o.std(), s.mean() / o.mean()


def lag(x, L):
    if L == 0:
        return x.copy()
    y = np.empty_like(x)
    y[:L] = x[0]
    y[L:] = x[:-L]
    return y


def linear_ie(I, Tres, L=0):
    """Implicit Euler, Q_t = a Q_{t-1} + (1 - a) I_{t-L}, a = T / (T + 1), T in days, steady start.
    L = 1 is dam_sandbox.py's simulate_linear."""
    a = Tres / (Tres + 1.0)
    x = lag(I, L)
    b_, a_ = [1 - a], [1, -a]
    return lfilter(b_, a_, x, zi=lfilter_zi(b_, a_) * x[0])[0]


def muskingum(I, K, X):
    """Q_t = c1 I_t + c2 I_{t-1} + c3 Q_{t-1}, daily step, K in days; same c1..c3 as src/routing/mmc.rs."""
    den = 2 * K * (1 - X) + 1.0
    c1, c2, c3 = (1.0 - 2 * K * X) / den, (1.0 + 2 * K * X) / den, (2 * K * (1 - X) - 1.0) / den
    b_, a_ = [c1, c2], [1, -c3]
    return lfilter(b_, a_, I, zi=lfilter_zi(b_, a_) * I[0])[0], (c1, c2, c3)


Ts = np.logspace(np.log10(0.05), np.log10(3000), 200)
Xs = np.linspace(0.0, 0.5, 26)
out = {}
for name, (below, above, s0) in DAMS.items():
    ib = gid.index(below)
    ias = [gid.index(s) for s in above]
    up_obs = np.nansum(np.stack([O[i] for i in ias]), axis=0)
    local = np.clip(P[ib] - np.stack([P[i] for i in ias]).sum(axis=0), 0.0, None)
    I = up_obs + local
    Q = O[ib]
    ok = np.isfinite(Q)
    r = {"mean_inflow": I.mean(), "mean_release": np.nanmean(Q), "inflow_over_release_volume": I[ok].sum() / Q[ok].sum(),
         "capacity_mcm": s0}
    r["pass_through_lag"] = {L: dict(train=nse(lag(I, L), Q, train), test=nse(lag(I, L), Q, test)) for L in range(5)}

    r["linear_ie"] = {}
    for L in range(4):
        tr = np.array([nse(linear_ie(I, Tt, L), Q, train) for Tt in Ts])
        k = int(np.nanargmax(tr))
        near = Ts[tr >= tr[k] - 0.02]
        r["linear_ie"][L] = dict(train=tr[k], test=nse(linear_ie(I, Ts[k], L), Q, test), T_days=Ts[k],
                                 T_band_days=(near.min(), near.max()))

    tr = np.array([nse(muskingum(I, Tt, 0.0)[0], Q, train) for Tt in Ts])
    k = int(np.nanargmax(tr))
    r["trapezoid_X0"] = dict(train=tr[k], test=nse(muskingum(I, Ts[k], 0.0)[0], Q, test), T_days=Ts[k])

    best = max(((nse(muskingum(I, K, X)[0], Q, train), K, X) for K in Ts for X in Xs), key=lambda b: b[0])
    q_kx, c = muskingum(I, best[1], best[2])
    r["muskingum_KX"] = dict(train=best[0], test=nse(q_kx, Q, test), K_days=best[1], X=best[2], c1=c[0], c2=c[1], c3=c[2])

    Lb = max(r["linear_ie"], key=lambda L: r["linear_ie"][L]["train"])
    q = linear_ie(I, r["linear_ie"][Lb]["T_days"], Lb)
    mm = train & ok
    beta = (q[mm] * Q[mm]).sum() / (q[mm] ** 2).sum()
    r["linear_x_volume"] = dict(lag=Lb, beta=beta, train=nse(beta * q, Q, train), test=nse(beta * q, Q, test))
    r["buffer_mcm"] = r["linear_ie"][Lb]["T_days"] * 86400 * I.mean() / 1e6

    Lp = max(r["pass_through_lag"], key=lambda L: r["pass_through_lag"][L]["train"])
    r["kge_parts_test"] = {"pass_lag0": kge_parts(I, Q, test), f"pass_lag{Lp}": kge_parts(lag(I, Lp), Q, test),
                           f"linear_lag{Lb}": kge_parts(q, Q, test)}

    # eight largest test inflow peaks, 15 days apart; max release over the peak day + 5
    It = np.where(test, I, -np.inf)
    peaks = []
    for idx in np.argsort(It)[::-1]:
        if all(abs(idx - p) >= 15 for p in peaks):
            peaks.append(int(idx))
        if len(peaks) == 8:
            break
    r["test_events"] = [dict(date=str(dates[p]), inflow=I[p], release_obs=np.nanmax(Q[p:p + 6]), release_linear=q[p:p + 6].max())
                        for p in sorted(peaks)]
    out[name] = r

    print(f"\n==== {name}: inflow {r['mean_inflow']:.2f}, release {r['mean_release']:.2f} m3/s, "
          f"inflow/release volume {r['inflow_over_release_volume']:.2f}")
    print("  pass-through, lag L:", ", ".join(f"{L}: {v['train']:.3f}/{v['test']:.3f}" for L, v in r["pass_through_lag"].items()))
    for L, v in r["linear_ie"].items():
        tag = "  <- dam_sandbox.py" if L == 1 else ""
        print(f"  linear (implicit Euler), inflow lag {L}: {v['train']:.3f}/{v['test']:.3f}  T {v['T_days']:.2f} d "
              f"[{v['T_band_days'][0]:.2f}, {v['T_band_days'][1]:.2f}]{tag}")
    v = r["trapezoid_X0"]
    print(f"  linear (trapezoid = Muskingum X=0): {v['train']:.3f}/{v['test']:.3f}  T {v['T_days']:.2f} d")
    v = r["muskingum_KX"]
    print(f"  Muskingum (K, X): {v['train']:.3f}/{v['test']:.3f}  K {v['K_days']:.2f} d  X {v['X']:.2f}")
    v = r["linear_x_volume"]
    print(f"  linear x volume factor {v['beta']:.3f}: {v['train']:.3f}/{v['test']:.3f}")
    print(f"  implied buffer {r['buffer_mcm']:.2f} MCM = {100 * r['buffer_mcm'] / s0:.2f} % of assumed capacity")
    for e in r["test_events"]:
        print(f"    {e['date']}  inflow {e['inflow']:7.1f}  release obs {e['release_obs']:7.1f}  linear {e['release_linear']:7.1f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("->", OUT)
