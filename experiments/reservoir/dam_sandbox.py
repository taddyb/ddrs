#!/usr/bin/env python
"""Generic dam sandbox: fit the dMC storage law at a dam with observed inflow (upstream gauges +
summed-Q' of the ungauged remainder) and observed release, against pass-through and linear
reservoirs, with a capacity sweep. See raystown_sandbox.py for the derivation.

Usage: dam_sandbox.py --below STAID --above STAID[,STAID...] --s0-mcm S0 --name "Alamo Dam"
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

ap = argparse.ArgumentParser()
ap.add_argument("--below", required=True)
ap.add_argument("--above", required=True, help="comma-separated upstream gauges")
ap.add_argument("--s0-mcm", type=float, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--plot-wy", type=int, nargs=2, default=[2004, 2005])
args = ap.parse_args()

R = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/baseline")
OUT = Path("/home/tbindas/projects/ddrs/output/dam_sandbox") / args.tag
OUT.mkdir(parents=True, exist_ok=True)
DT = 86400.0

m = json.load(open(R / "manifest.json"))
G, T = m["n_gauges"], m["n_days"]
P = np.fromfile(R / "predictions.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
O = np.fromfile(R / "observations.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
gid = [str(g) for g in m["gage_ids"]]
ib = gid.index(args.below)
ias = [gid.index(s) for s in args.above.split(",")]
t0 = dt.date.fromisoformat(str(m["time_range_daily"][0])[:10])
dates = np.array([t0 + dt.timedelta(days=i) for i in range(T)])
wy = np.array([d.year + (1 if d.month >= 10 else 0) for d in dates])

up_obs = np.nansum(np.stack([O[i] for i in ias]), axis=0)
up_qp = np.stack([P[i] for i in ias]).sum(axis=0)
local = np.clip(P[ib] - up_qp, 0.0, None)
inflow = up_obs + local
outflow = O[ib]
gauged_frac = up_qp.mean() / P[ib].mean()
print(f"{args.name}: below {args.below}, above {args.above}; gauged share of summed Q' {gauged_frac:.2f}")
print(f"mean inflow {inflow.mean():.2f} m3/s (gauged {up_obs.mean():.2f} + local Q' {local.mean():.2f}); mean release {np.nanmean(outflow):.2f}; "
      f"summed-Q' volume ratio at dam gauge {P[ib].sum() / np.nansum(outflow):.2f}; inflow/release volume ratio {inflow.sum() / np.nansum(outflow):.2f}")

train = (wy >= 1997) & (wy <= 2001)
test = (wy >= 2002) & (wy <= 2010)
warm = wy <= 1996
ok = np.isfinite(outflow)


def nse(sim, obs, mask):
    mm = mask & np.isfinite(obs)
    s, o = sim[mm], obs[mm]
    return 1.0 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def simulate_law(theta, b, S0, q_ref, inflow, s_init_frac=0.5):
    theta = np.asarray(theta, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    K = theta.shape[0]; q0 = theta * q_ref
    S = np.full(K, S0 * s_init_frac); Q = np.zeros((K, T)); Sout = np.zeros((K, T))
    Q[:, 0] = q0 * (S / S0) ** b; Sout[:, 0] = S
    for t in range(1, T):
        i_t = inflow[t - 1]
        frac = np.clip(S / S0, 1e-6, None)
        q_calc = q0 * frac ** b
        dS_dQ = np.clip(S0 / (b * q0) * frac ** (1.0 - b), 1e-3, None)
        eta = dS_dQ / DT
        q_next = np.maximum((eta * q_calc + i_t) / (eta + 1.0), 0.0)
        S = np.maximum(S + DT * (i_t - q_next), 0.0)
        Q[:, t] = q_next; Sout[:, t] = S
    return Q, Sout


def simulate_linear(Tres, S0, inflow, s_init_frac=0.5):
    S = S0 * s_init_frac; Q = np.zeros(T)
    for t in range(1, T):
        S = (S + DT * inflow[t - 1]) / (1.0 + DT / Tres); Q[t] = S / Tres
    return Q


q_ref = float(np.median(inflow)) if np.median(inflow) > 0 else float(inflow.mean())
res = {"name": args.name, "below": args.below, "above": args.above, "gauged_frac": gauged_frac,
       "mean_inflow": float(inflow.mean()), "mean_release": float(np.nanmean(outflow)), "q_ref": q_ref}
res["pass-through"] = dict(train=nse(inflow, outflow, train), test=nse(inflow, outflow, test))
S0 = args.s0_mcm * 1e6
T_fixed = S0 / inflow.mean()
q_lin = simulate_linear(T_fixed, S0, inflow)
res["linear, T = S0/mean inflow"] = dict(train=nse(q_lin, outflow, train), test=nse(q_lin, outflow, test), T_days=T_fixed / DT)
Ts = np.logspace(np.log10(0.5), np.log10(3000), 70) * DT
lin_train = [nse(simulate_linear(Tt, S0, inflow), outflow, train) for Tt in Ts]
kb = int(np.argmax(lin_train)); q_lin_fit = simulate_linear(Ts[kb], S0, inflow)
res["linear, T fitted"] = dict(train=lin_train[kb], test=nse(q_lin_fit, outflow, test), T_days=Ts[kb] / DT)

th = np.linspace(0.3, 3.0, 28); bb = np.logspace(np.log10(0.01), np.log10(3.0), 28)
TH, BB = np.meshgrid(th, bb, indexing="ij")


def fit_law(S0v):
    Q, S = simulate_law(TH.ravel(), BB.ravel(), S0v, q_ref, inflow)
    gtr = np.array([nse(Q[k], outflow, train) for k in range(Q.shape[0])]).reshape(TH.shape)
    gte = np.array([nse(Q[k], outflow, test) for k in range(Q.shape[0])]).reshape(TH.shape)
    i, j = np.unravel_index(np.nanargmax(gtr), gtr.shape)
    near = gtr >= gtr[i, j] - 0.02
    return dict(train=float(gtr[i, j]), test=float(gte[i, j]), theta=float(TH[i, j]), b=float(BB[i, j]),
                on_wall=bool(i in (0, len(th) - 1) or j in (0, len(bb) - 1)), near_frac=float(near.mean()),
                resp_days=float(S0v / (BB[i, j] * TH[i, j] * q_ref) / DT)), gtr, gte, (i, j), Q[i * len(bb) + j], S[i * len(bb) + j]


law, gtr, gte, ij, q_best, s_best = fit_law(S0)
res["storage law at assumed S0"] = law
sweep = {}
for f in (0.001, 0.01, 0.1, 0.3, 1.0, 3.0):
    l, *_ = fit_law(S0 * f)
    sweep[f"S0 x {f}"] = dict(S0_mcm=args.s0_mcm * f, **l)
res["capacity sweep"] = sweep
json.dump(res, open(OUT / "result.json", "w"), indent=2, default=float)

print(f"\n== {args.name}, assumed S0 = {args.s0_mcm:.0f} MCM ==")
for k in ["pass-through", "linear, T = S0/mean inflow", "linear, T fitted", "storage law at assumed S0"]:
    print(f"  {k:32s} {res[k]}")
print("  capacity sweep (best on grid):")
for k, v in sweep.items():
    print(f"    {k:10s} S0 {v['S0_mcm']:8.1f} MCM  train {v['train']:.3f} test {v['test']:.3f}  (theta {v['theta']:.2f}, b {v['b']:.2f}, wall {v['on_wall']}, resp {v['resp_days']:.1f} d)")

# figures
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.patch.set_facecolor("#fcfcfb")
sel = (wy >= args.plot_wy[0]) & (wy <= args.plot_wy[1]); x = dates[sel]
ax = axes[0]
ax.plot(x, inflow[sel], color="#c3c2b7", lw=1.2, label="inflow (upstream obs + local Q')")
ax.plot(x, outflow[sel], color="#0b0b0b", lw=1.4, label="observed release below dam")
ax.plot(x, q_best[sel], color="#eb6834", lw=1.4, label=f"storage law, fitted at S0 (theta {law['theta']:.2f}, b {law['b']:.2f})")
ax.plot(x, q_lin_fit[sel], color="#2a78d6", lw=1.2, ls="--", label=f"linear reservoir, T fitted = {res['linear, T fitted']['T_days']:.1f} d")
ax.set_ylabel("m3/s"); ax.set_title(f"{args.name}: test years WY{args.plot_wy[0]}-{args.plot_wy[1]}", fontsize=10); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax = axes[1]
vmax = np.nanmax(gtr); vmin = max(np.nanmin(gtr), vmax - 0.8)
cs = ax.contourf(th, bb, gtr.T, levels=np.linspace(vmin, vmax, 25), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("s", ["#e8f1fc", "#2a78d6", "#0d3a73"]))
ax.contour(th, bb, gtr.T, levels=[vmax - 0.02], colors="#0b0b0b", linewidths=1)
ax.plot(th[ij[0]], bb[ij[1]], "o", color="#eb6834", ms=8, label="train optimum")
ax.plot(1.5, 0.8, "s", color="#0b0b0b", ms=6, label="dMC default")
ax.set_yscale("log"); ax.set_xlabel("theta (full-pool release / median inflow)"); ax.set_ylabel("b (storage_n x alpha)")
ax.set_title(f"train NSE landscape at S0 = {args.s0_mcm:.0f} MCM (black: within 0.02 of optimum)", fontsize=10); ax.legend(fontsize=8, loc="lower right")
fig.colorbar(cs, ax=ax, label="NSE")
fig.tight_layout(); fig.savefig(OUT / "sandbox.png", dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
print("->", OUT)
