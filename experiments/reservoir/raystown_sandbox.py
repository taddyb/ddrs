#!/usr/bin/env python
"""Raystown Dam sandbox: can downstream discharge alone identify a storage-release law?

Inflow  = observed Raystown Branch at Saxton (01562000) + summed-Q' of the ungauged area
          between Saxton and the dam (baseline(01563200) - baseline(01562000)).
Outflow = observed Raystown Branch below Raystown Dam (01563200).
Law     = dMC-dev `_first_order_euler_storage`: Q = Q0 (S/S0)^b, Q0 = theta * Q_ref,
          linearised implicit Euler, daily step, storage spun up from S0/2 with a one-year
          warm-up that is never scored.  The dMC pair (storage_n, alpha) enters only as
          b = storage_n * alpha, so it is fitted as one parameter.
Baselines: pass-through (Q = I); zero-parameter linear reservoir with residence time
          T = S0 / mean inflow; one-parameter linear reservoir with T fitted.
Train WY1997-2001, test WY2002-2010.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

R = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/baseline")
OUT = Path("/home/tbindas/projects/ddrs/output/raystown_sandbox")
OUT.mkdir(parents=True, exist_ok=True)
ABOVE, BELOW = "01562000", "01563200"
S0_MCM = 940.0            # Raystown Lake capacity, million m^3 (GRanD value from memory; see sensitivity)
DT = 86400.0

m = json.load(open(R / "manifest.json"))
G, T = m["n_gauges"], m["n_days"]
P = np.fromfile(R / "predictions.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
O = np.fromfile(R / "observations.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
gid = [str(g) for g in m["gage_ids"]]
ia, ib = gid.index(ABOVE), gid.index(BELOW)
t0 = dt.date.fromisoformat(str(m["time_range_daily"][0])[:10])
dates = np.array([t0 + dt.timedelta(days=i) for i in range(T)])
wy = np.array([d.year + (1 if d.month >= 10 else 0) for d in dates])

local = np.clip(P[ib] - P[ia], 0.0, None)          # summed Q' between Saxton and the dam
inflow_obs = O[ia] + local
inflow = np.where(np.isfinite(inflow_obs), inflow_obs, P[ia] + local)  # fall back to Q' on missing Saxton days
outflow = O[ib]
print(f"days {T}, Saxton obs missing {int((~np.isfinite(O[ia])).sum())}, below-dam obs missing {int((~np.isfinite(outflow)).sum())}")
print(f"mean inflow {inflow.mean():.1f} m3/s (Saxton {np.nanmean(O[ia]):.1f} + local Q' {local.mean():.1f}); mean release {np.nanmean(outflow):.1f} m3/s")
print(f"summed-Q' volume ratio at the dam gauge over the record: {P[ib].sum() / np.nansum(outflow):.3f}")

train = (wy >= 1997) & (wy <= 2001)
test = (wy >= 2002) & (wy <= 2010)
warm = wy <= 1996
ok = np.isfinite(outflow)


def nse(sim, obs, mask):
    mm = mask & np.isfinite(obs)
    s, o = sim[mm], obs[mm]
    return 1.0 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


# --------------------------------------------------------------------- the law, vectorised over a parameter grid
def simulate_law(theta, b, S0, q_ref, inflow, s_init_frac=0.5):
    """theta, b: arrays of shape [K]. Returns Q [K, T], S [K, T]. Mirrors _first_order_euler_storage."""
    theta = np.asarray(theta, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = theta.shape[0]
    q0 = theta * q_ref
    S = np.full(K, S0 * s_init_frac)
    Q = np.zeros((K, T))
    Sout = np.zeros((K, T))
    Q[:, 0] = q0 * (S / S0) ** b
    Sout[:, 0] = S
    eps = 1e-6
    for t in range(1, T):
        i_t = inflow[t - 1]
        frac = np.clip(S / S0, eps, None)
        q_calc = q0 * frac ** b
        dS_dQ = np.clip(S0 / (b * q0) * frac ** (1.0 - b), 1e-3, None)
        eta = dS_dQ / DT
        q_next = np.maximum((eta * q_calc + i_t) / (eta + 1.0), 0.0)
        S = np.maximum(S + DT * (i_t - q_next), 0.0)
        Q[:, t] = q_next
        Sout[:, t] = S
    return Q, Sout


def simulate_linear(Tres, S0, inflow, s_init_frac=0.5):
    S = S0 * s_init_frac
    Q = np.zeros(T)
    for t in range(1, T):
        S = (S + DT * inflow[t - 1]) / (1.0 + DT / Tres)
        Q[t] = S / Tres
    return Q


def run(S0_mcm, tag):
    S0 = S0_mcm * 1e6
    q_ref = float(np.median(inflow))          # dMC uses the median summed Q'
    res = {}
    # baselines
    res["pass-through"] = dict(train=nse(inflow, outflow, train), test=nse(inflow, outflow, test))
    T_fixed = S0 / inflow.mean()
    q_lin = simulate_linear(T_fixed, S0, inflow)
    res["linear reservoir, T = S0/mean inflow (0 params)"] = dict(train=nse(q_lin, outflow, train), test=nse(q_lin, outflow, test), T_days=T_fixed / DT)
    Ts = np.logspace(np.log10(0.5), np.log10(2000), 60) * DT
    lin_train = [nse(simulate_linear(Tt, S0, inflow), outflow, train) for Tt in Ts]
    kbest = int(np.argmax(lin_train))
    q_lin_fit = simulate_linear(Ts[kbest], S0, inflow)
    res["linear reservoir, T fitted (1 param)"] = dict(train=lin_train[kbest], test=nse(q_lin_fit, outflow, test), T_days=Ts[kbest] / DT)
    # the law on a grid
    th = np.linspace(0.3, 3.0, 46)
    bb = np.logspace(np.log10(0.01), np.log10(3.0), 46)
    TH, BB = np.meshgrid(th, bb, indexing="ij")
    Q, S = simulate_law(TH.ravel(), BB.ravel(), S0, q_ref, inflow)
    grid_train = np.array([nse(Q[k], outflow, train) for k in range(Q.shape[0])]).reshape(TH.shape)
    grid_test = np.array([nse(Q[k], outflow, test) for k in range(Q.shape[0])]).reshape(TH.shape)
    i, j = np.unravel_index(np.nanargmax(grid_train), grid_train.shape)
    res["storage law, (theta, b) fitted (2 params)"] = dict(train=float(grid_train[i, j]), test=float(grid_test[i, j]), theta=float(TH[i, j]), b=float(BB[i, j]))
    # one-param law: theta = 1 (full-pool release = median inflow)
    jt = int(np.argmin(np.abs(th - 1.0)))
    j1 = int(np.nanargmax(grid_train[jt]))
    res["storage law, theta = 1, b fitted (1 param)"] = dict(train=float(grid_train[jt, j1]), test=float(grid_test[jt, j1]), b=float(BB[jt, j1]))
    # how flat: fraction of the grid within 0.02 NSE of the optimum, and the ridge along theta*b
    near = grid_train >= grid_train[i, j] - 0.02
    res["_grid"] = dict(near_frac=float(near.mean()), theta_range_near=[float(TH[near].min()), float(TH[near].max())], b_range_near=[float(BB[near].min()), float(BB[near].max())])
    # gradient and Hessian at the optimum and at the dMC default (theta 1.5, storage_n 0.4 * alpha 2 -> b 0.8) via autograd
    def loss_torch(p):
        theta_t, logb_t = p[0], p[1]
        b_t = torch.exp(logb_t)
        q0 = theta_t * q_ref
        S_t = torch.tensor(S0 * 0.5, dtype=torch.float64)
        S0_t = torch.tensor(S0, dtype=torch.float64)
        qs = []
        for t in range(1, T):
            frac = torch.clamp(S_t / S0_t, min=1e-6)
            q_calc = q0 * frac ** b_t
            dS_dQ = torch.clamp(S0_t / (b_t * q0) * frac ** (1.0 - b_t), min=1e-3)
            eta = dS_dQ / DT
            q_next = torch.relu((eta * q_calc + inflow[t - 1]) / (eta + 1.0))
            S_t = torch.relu(S_t + DT * (inflow[t - 1] - q_next))
            qs.append(q_next)
        q = torch.stack([torch.tensor(0.0, dtype=torch.float64)] + qs)
        mm = torch.tensor(train & ok)
        o = torch.tensor(outflow)[mm]
        return ((q[mm] - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()   # 1 - NSE
    for name, p0 in [("dMC default (theta 1.5, b 0.8)", [1.5, np.log(0.8)]), ("grid optimum", [float(TH[i, j]), float(np.log(BB[i, j]))])]:
        p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
        L = loss_torch(p)
        g = torch.autograd.grad(L, p)[0]
        H = torch.autograd.functional.hessian(loss_torch, p.detach())
        ev = torch.linalg.eigvalsh(H)
        res[f"_grad at {name}"] = dict(loss=float(L), grad=[float(x) for x in g], hess_eig=[float(x) for x in ev])
    json.dump(res, open(OUT / f"raystown_{tag}.json", "w"), indent=2)
    return res, (th, bb, grid_train, grid_test, (i, j)), Q[i * len(bb) + j], S[i * len(bb) + j], q_lin, q_lin_fit


res, grid, q_best, s_best, q_lin, q_lin_fit = run(S0_MCM, "S0_940")
print("\n== Raystown, S0 = 940 MCM ==")
for k, v in res.items():
    print(f"{k:55s} {v}")
for f in (0.5, 2.0):
    r2, *_ = run(S0_MCM * f, f"S0_{int(S0_MCM * f)}")
    print(f"\n-- sensitivity S0 x {f}: law 2p train {r2['storage law, (theta, b) fitted (2 params)']['train']:.3f} test {r2['storage law, (theta, b) fitted (2 params)']['test']:.3f}; "
          f"linear fixed train {r2['linear reservoir, T = S0/mean inflow (0 params)']['train']:.3f} test {r2['linear reservoir, T = S0/mean inflow (0 params)']['test']:.3f}; "
          f"linear fitted test {r2['linear reservoir, T fitted (1 param)']['test']:.3f}; near_frac {r2['_grid']['near_frac']:.2f}")

# --------------------------------------------------------------------- figures
th, bb, gtr, gte, (i, j) = grid
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#fcfcfb")
for ax, g, title in zip(axes, [gtr, gte], ["train NSE, WY1997-2001", "test NSE, WY2002-2010"]):
    vmax = np.nanmax(g); vmin = max(np.nanmin(g), vmax - 0.6)
    cs = ax.contourf(th, bb, g.T, levels=np.linspace(vmin, vmax, 25), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("s", ["#e8f1fc", "#2a78d6", "#0d3a73"]))
    ax.contour(th, bb, g.T, levels=[vmax - 0.02], colors="#0b0b0b", linewidths=1)
    ax.plot(th[i], bb[j], "o", color="#eb6834", ms=8, label=f"train optimum theta={th[i]:.2f}, b={bb[j]:.2f}")
    ax.plot(1.5, 0.8, "s", color="#0b0b0b", ms=6, label="dMC default (1.5, 0.8)")
    ax.set_yscale("log"); ax.set_xlabel("theta  (full-pool release / median inflow)"); ax.set_ylabel("b  (storage_n x alpha)")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="lower right")
    fig.colorbar(cs, ax=ax, label="NSE (black line: within 0.02 of optimum)")
fig.suptitle(f"Raystown Dam storage-law landscape, S0 = {S0_MCM:.0f} MCM, storage spun up from S0/2, daily step", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "raystown_landscape.png", dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.patch.set_facecolor("#fcfcfb")
sel = (wy >= 2003) & (wy <= 2004)
x = dates[sel]
ax = axes[0]
ax.plot(x, inflow[sel], color="#c3c2b7", lw=1.2, label="inflow (Saxton obs + local Q')")
ax.plot(x, outflow[sel], color="#0b0b0b", lw=1.4, label="observed release below dam")
ax.plot(x, q_best[sel], color="#eb6834", lw=1.4, label=f"storage law, fitted (theta {res['storage law, (theta, b) fitted (2 params)']['theta']:.2f}, b {res['storage law, (theta, b) fitted (2 params)']['b']:.2f})")
ax.plot(x, q_lin[sel], color="#2a78d6", lw=1.2, ls="--", label=f"linear reservoir, T = S0/mean inflow = {res['linear reservoir, T = S0/mean inflow (0 params)']['T_days']:.0f} d")
ax.set_ylabel("m3/s"); ax.set_title("test years WY2003-2004", fontsize=10); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(dates, s_best / 1e6, color="#eb6834", lw=1.2, label="storage under the fitted law")
ax.axhline(S0_MCM, color="#52514e", ls="--", lw=0.8); ax.text(dates[30], S0_MCM * 1.02, "S0", fontsize=8)
ax.axvspan(dates[warm][0], dates[warm][-1], color="#d9d9d6", alpha=0.5, label="warm-up (unscored)")
ax.axvspan(dates[train][0], dates[train][-1], color="#a9cbf2", alpha=0.3, label="train")
ax.set_ylabel("MCM"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "raystown_hydrograph.png", dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
print("figures ->", OUT)
