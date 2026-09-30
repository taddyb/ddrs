#!/usr/bin/env python
"""Threshold linear reservoir at the four sandbox dams: which single operating feature closes the gap?

    avail = S + I_t;  w = min(E, avail)
    Q = min( max(avail - w - S_d, 0) / (T + 1) , Q_max )      implicit Euler on the linear part, dt = 1 d
    S <- avail - w - Q

  T     release timescale above the pool (days)                  attenuation
  Q_max maximum release, e.g. downstream channel capacity         capped peaks, flat-topped evacuation
  S_d   conservation pool, no release below it                    zero-release spells (needs E to recur)
  E     constant withdrawal from storage (evaporation, diversion) drains the pool between events

Storage starts at the pool, so S_d only matters through the S >= 0 floor once E drains it: with E = 0
the pool is a no-op by construction. Nested families are fitted on a (T, Q_max, S_d, E) grid on train
NSE (WY1997-2001) and scored on test (WY2002-2010); "band" is the range of the parameter over cells
within 0.02 NSE of the family optimum. Same-day inflow, as the lag-0 linear reservoir in
linear_reservoir_diagnostics.py.

--inflow observed (default): upstream gauges observed + local summed Q', as dam_sandbox.py.
--inflow modelled: the trained model's prediction at the upstream gauges + local summed Q', i.e.
what a reservoir node inside ddrs would see.
Figure: output/dam_sandbox/threshold_hydrographs[_modelled].png
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inflow", choices=["observed", "modelled"], default="observed")
args = ap.parse_args()
sfx = "" if args.inflow == "observed" else "_modelled"

RUN = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test")
R = RUN / "baseline"
OUT = Path(__file__).resolve().parent / "results" / f"threshold_reservoir{sfx}.json"
FIG = Path(f"/home/tbindas/projects/ddrs/output/dam_sandbox/threshold_hydrographs{sfx}.png")

m = json.load(open(R / "manifest.json"))
G, NT = m["n_gauges"], m["n_days"]
P = np.fromfile(R / "predictions.f32", dtype=np.float32).reshape(G, NT).astype(np.float64)
O = np.fromfile(R / "observations.f32", dtype=np.float32).reshape(G, NT).astype(np.float64)
gid = [str(g) for g in m["gage_ids"]]
t0 = dt.date.fromisoformat(str(m["time_range_daily"][0])[:10])
dates = np.array([t0 + dt.timedelta(days=i) for i in range(NT)])
wy = np.array([d.year + (1 if d.month >= 10 else 0) for d in dates])
train = (wy >= 1997) & (wy <= 2001)
test = (wy >= 2002) & (wy <= 2010)

if args.inflow == "modelled":
    Z = zarr.open(str(RUN / "eval/predictions.zarr"), mode="r")
    zids = [bytes(r).decode().strip("\x00") for r in Z["gage_ids"][:]]
    z0 = (dt.date.fromisoformat(str(Z["time"][:1].astype("datetime64[ns]").astype("datetime64[D]")[0])) - t0).days
    zn = Z["predictions"].shape[1]


def upstream_inflow(above):
    if args.inflow == "observed":
        return np.nansum(np.stack([O[gid.index(s)] for s in above]), axis=0)
    up = np.zeros(NT)
    for s in above:
        up[z0:z0 + zn] += Z["predictions"][zids.index(s), :]
    up[:z0] = up[z0]
    up[z0 + zn:] = up[z0 + zn - 1]
    return up


DAMS = {
    "Raystown": ("01563200", ["01562000"], (2003, 2004)),
    "Abiquiu": ("08287000", ["08286500"], (2005, 2005)),
    "Alamo": ("09426000", ["09424450", "09424900"], (2005, 2005)),
    "Santa Rosa": ("08382830", ["08382650"], (2004, 2005)),
}


def simulate(I, T, Qmax, Sd, E, obs=None, masks=()):
    """Vectorised over parameter cells. With obs, returns NSE per mask without materialising Q;
    without, returns the (cells, time) release matrix."""
    S = Sd.copy()
    sse = [np.zeros_like(T) for _ in masks]
    Q = None if obs is not None else np.empty((T.shape[0], I.shape[0]))
    for t in range(I.shape[0]):
        avail = S + I[t]
        w = np.minimum(E, avail)
        q = np.minimum(np.maximum(avail - w - Sd, 0.0) / (T + 1.0), Qmax)
        S = avail - w - q
        if Q is not None:
            Q[:, t] = q
        elif np.isfinite(obs[t]):
            for j, mk in enumerate(masks):
                if mk[t]:
                    sse[j] += (q - obs[t]) ** 2
    if Q is not None:
        return Q
    out = []
    for j, mk in enumerate(masks):
        o = obs[mk & np.isfinite(obs)]
        out.append(1.0 - sse[j] / ((o - o.mean()) ** 2).sum())
    return out


summary, series = {}, {}
for name, (below, above, _) in DAMS.items():
    ib = gid.index(below)
    ias = [gid.index(s) for s in above]
    local = np.clip(P[ib] - np.stack([P[i] for i in ias]).sum(axis=0), 0.0, None)
    I = upstream_inflow(above) + local
    obs = O[ib]
    mu = I.mean()
    Tg = np.logspace(np.log10(0.1), np.log10(1000), 24)
    Qg = np.concatenate([np.logspace(np.log10(0.5), np.log10(300), 40) * mu, [np.inf]])
    Sg = np.concatenate([[0.0], np.logspace(np.log10(1), np.log10(1000), 10) * mu])
    Eg = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5]) * mu
    TT, QQ, SS, EE = (a.ravel() for a in np.meshgrid(Tg, Qg, Sg, Eg, indexing="ij"))
    tr, te = simulate(I, TT, QQ, SS, EE, obs=obs, masks=[train, test])
    fam = {
        "linear": (SS == 0) & np.isinf(QQ) & (EE == 0),
        "linear + cap": (SS == 0) & (EE == 0),
        "linear + withdrawal": np.isinf(QQ) & (SS == 0),
        "linear + cap + withdrawal": SS == 0,
        "linear + pool + cap + withdrawal": np.ones_like(tr, bool),
    }
    res, best = {"mean_inflow": mu}, {}
    print(f"\n==== {name}: mean inflow {mu:.2f} m3/s")
    for f, sel in fam.items():
        k = np.flatnonzero(sel)[np.nanargmax(tr[sel])]
        near = sel & (tr >= tr[k] - 0.02)
        band = lambda a: (float(a[near].min()), float(a[near].max()))
        res[f] = dict(train=float(tr[k]), test=float(te[k]), T_days=float(TT[k]), Qmax=float(QQ[k]),
                      Sd_mcm=float(SS[k] * 0.0864), E=float(EE[k]), T_band=band(TT), Qmax_band=band(QQ),
                      Sd_band_mcm=tuple(x * 0.0864 for x in band(SS)), E_band=band(EE), n_near=int(near.sum()))
        best[f] = k
        print(f"  {f:34s} {tr[k]:.3f}/{te[k]:.3f}  T {TT[k]:7.2f} d [{band(TT)[0]:.2f}, {band(TT)[1]:.2f}]"
              f"  Qmax {QQ[k]:7.1f} [{band(QQ)[0]:.1f}, {band(QQ)[1]:.1f}]  E {EE[k]:.2f} [{band(EE)[0]:.2f}, {band(EE)[1]:.2f}]")
    summary[name] = res
    ks = [best["linear"], best["linear + cap"]]
    Qs = simulate(I, TT[ks], QQ[ks], SS[ks], EE[ks])
    series[name] = (I, obs, Qs, res)

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=1)

fig, axes = plt.subplots(len(DAMS), 1, figsize=(13, 3.2 * len(DAMS)))
fig.patch.set_facecolor("#fcfcfb")
for ax, (name, (_, _, wyr)) in zip(axes, DAMS.items()):
    I, obs, Qs, res = series[name]
    sel = (wy >= wyr[0]) & (wy <= wyr[1])
    x = dates[sel]
    ax.plot(x, I[sel], color="#c3c2b7", lw=1.1, label=f"inflow (upstream {args.inflow} + local Q')")
    ax.plot(x, obs[sel], color="#0b0b0b", lw=1.4, label="observed release")
    lin, cap = res["linear"], res["linear + cap"]
    ax.plot(x, Qs[0][sel], color="#2a78d6", lw=1.2, ls="--", label=f"linear, T {lin['T_days']:.2f} d (test NSE {lin['test']:.2f})")
    ax.plot(x, Qs[1][sel], color="#eb6834", lw=1.4,
            label=f"linear + cap, T {cap['T_days']:.2f} d, Q_max {cap['Qmax']:.0f} (test NSE {cap['test']:.2f})")
    ax.set_ylabel("m3/s")
    ax.set_title(f"{name}, test WY{wyr[0]}" + (f"-{wyr[1]}" if wyr[1] != wyr[0] else ""), fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
fig.tight_layout()
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print("->", OUT, FIG)
