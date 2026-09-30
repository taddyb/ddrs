#!/usr/bin/env python
"""Verify the Raystown sandbox implementation of the dMC storage law.

1. Equivalence: run dMC-dev's `_first_order_euler_storage` (copied verbatim from
   dMC/physics_models/methods.py at the PR #79 merge) step by step and compare with the
   sandbox's numpy loop on identical inputs.
2. Step response: constant inflow, then a step; measure the e-folding time of the release
   and compare with the analytic linearised timescale dS/dQ at equilibrium.
3. Capacity sweep: as S0 -> 0 the law must collapse to pass-through (NSE -> 0.65 train).
"""
import numpy as np
import torch
from typing import Tuple



# ---- dMC-dev function, verbatim (methods.py at eb3bc7e5), minus the jit decorator
def _first_order_euler_storage(dt, _exponent, i_t, q_0, S_t, S_0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = 1e-6
    S_0_safe = torch.clamp(S_0, min=epsilon)
    q_0_safe = torch.clamp(q_0, min=epsilon)
    _exponent_safe = torch.clamp(_exponent, min=epsilon)
    S_over_S0 = torch.div(S_t, S_0_safe)
    S_over_S0 = torch.clamp(S_over_S0, min=epsilon)
    power_term = 1 - 1 / (_exponent_safe + epsilon)
    power_result = torch.pow(S_over_S0, power_term)
    dS_dQ = (_exponent_safe / q_0_safe) * S_0_safe * power_result
    dS_dQ = torch.clamp(dS_dQ, min=1e-3)
    eta = torch.div(1.0, dt) * dS_dQ
    q_power = torch.div(1.0, _exponent_safe)
    q_calc_t = q_0_safe * torch.pow(S_over_S0, q_power)
    denominator = torch.clamp(eta + 1.0, min=epsilon)
    q_t1 = torch.relu(torch.div(eta * q_calc_t + i_t, denominator))
    S_t1 = torch.relu(S_t + dt * (i_t - q_t1))
    return q_t1, q_calc_t, S_t1


# ---- sandbox loop (copied from raystown_sandbox.py, single parameter set)
def sandbox_loop(theta, b, S0, q_ref, inflow, DT, s_init_frac=0.5):
    q0 = theta * q_ref
    S = S0 * s_init_frac
    T = len(inflow)
    Q = np.zeros(T); Sout = np.zeros(T)
    Q[0] = q0 * (S / S0) ** b; Sout[0] = S
    eps = 1e-6
    for t in range(1, T):
        i_t = inflow[t - 1]
        frac = max(S / S0, eps)
        q_calc = q0 * frac ** b
        dS_dQ = max(S0 / (b * q0) * frac ** (1.0 - b), 1e-3)
        eta = dS_dQ / DT
        q_next = max((eta * q_calc + i_t) / (eta + 1.0), 0.0)
        S = max(S + DT * (i_t - q_next), 0.0)
        Q[t] = q_next; Sout[t] = S
    return Q, Sout


# ---- 1. equivalence on real inflow
import json, datetime as dt
from pathlib import Path
R = Path("/home/tbindas/projects/ddrs/.ddrs/runs/2026-09-17T16-38-16Z-train-and-test/baseline")
m = json.load(open(R / "manifest.json")); G, T = m["n_gauges"], m["n_days"]
P = np.fromfile(R / "predictions.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
O = np.fromfile(R / "observations.f32", dtype=np.float32).reshape(G, T).astype(np.float64)
gid = [str(g) for g in m["gage_ids"]]; ia, ib = gid.index("01562000"), gid.index("01563200")
inflow = O[ia] + np.clip(P[ib] - P[ia], 0, None)
outflow = O[ib]
t0 = dt.date.fromisoformat(str(m["time_range_daily"][0])[:10])
wy = np.array([(t0 + dt.timedelta(days=i)).year + (1 if (t0 + dt.timedelta(days=i)).month >= 10 else 0) for i in range(T)])
train = (wy >= 1997) & (wy <= 2001)
def nse(sim, obs, mask):
    s, o = sim[mask], obs[mask]; return 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()

DT = 86400.0; S0 = 940e6; q_ref = float(np.median(inflow))
print(f"median inflow (q_ref) {q_ref:.1f} m3/s, mean {inflow.mean():.1f}")
for theta, b in [(1.5, 0.8), (3.0, 3.0), (1.0, 0.1)]:
    Qs, Ss = sandbox_loop(theta, b, S0, q_ref, inflow, DT)
    # dMC verbatim, torch float64, same initial state, exponent = 1/b
    dtt = torch.tensor(DT, dtype=torch.float64); ex = torch.tensor(1.0 / b, dtype=torch.float64)
    q0t = torch.tensor(theta * q_ref, dtype=torch.float64); S0t = torch.tensor(S0, dtype=torch.float64)
    St = torch.tensor(S0 * 0.5, dtype=torch.float64); Qd = np.zeros(T); Qd[0] = Qs[0]
    for t in range(1, T):
        qt1, _, St = _first_order_euler_storage(dtt, ex, torch.tensor(inflow[t - 1], dtype=torch.float64), q0t, St, S0t)
        Qd[t] = float(qt1)
    print(f"  theta {theta} b {b}: max |sandbox - dMC verbatim| = {np.abs(Qs - Qd).max():.3e} m3/s over {T} days; NSE train sandbox {nse(Qs, outflow, train):.4f} dMC {nse(Qd, outflow, train):.4f}")

# ---- 2. step response vs analytic timescale
print("\nstep response (constant 20 m3/s for 5 y, then 60 m3/s):")
for theta, b in [(1.5, 0.8), (3.0, 3.0)]:
    n1 = 365 * 5; step = np.concatenate([np.full(n1, 20.0), np.full(365 * 6, 60.0)])
    Q, S = sandbox_loop(theta, b, S0, q_ref, step, DT, s_init_frac=0.5)
    q_before = Q[n1 - 1]; q_final = Q[-1]
    target = q_before + (1 - np.exp(-1)) * (q_final - q_before)
    k = n1 + int(np.argmax(Q[n1:] >= target))
    tau_sim_days = k - n1
    Seq = S[-1]; q0 = theta * q_ref
    dS_dQ = S0 / (b * q0) * (Seq / S0) ** (1 - b)
    print(f"  theta {theta} b {b}: release {q_before:.1f} -> {q_final:.1f} m3/s (equilibrium settled: {abs(q_final-60)<0.5}); "
          f"e-folding time simulated {tau_sim_days} d vs analytic dS/dQ at equilibrium {dS_dQ/DT:.0f} d; equilibrium fill {Seq/S0:.2f}")

# ---- 3. capacity sweep with the grid fit
print("\ncapacity sweep, best (theta, b) on the grid, train NSE (pass-through = %.3f):" % nse(inflow, outflow, train))
th = np.linspace(0.3, 3.0, 12); bb = np.logspace(-2, np.log10(3), 12)
for S0m in [1, 3, 10, 30, 100, 300, 940]:
    best = (-9, None)
    for t_ in th:
        for b_ in bb:
            Q, _ = sandbox_loop(t_, b_, S0m * 1e6, q_ref, inflow, DT)
            v = nse(Q, outflow, train)
            if v > best[0]: best = (v, (t_, b_))
    print(f"  S0 = {S0m:4d} MCM: best train NSE {best[0]:.3f} at theta {best[1][0]:.2f}, b {best[1][1]:.2f}; response time S0/(b*theta*q_ref) = {S0m*1e6/(best[1][1]*best[1][0]*q_ref)/DT:.1f} d")
