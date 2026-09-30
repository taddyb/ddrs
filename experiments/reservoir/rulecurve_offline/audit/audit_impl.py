"""Independent minimal implementation of L0 / L1 / L4 (does NOT import rc.py).

Law (from the definition): daily implicit-Euler linear reservoir fed with an effective inflow
    x_t = I_t - r_t,   r_t = Ibar * sum_{k=1,2} (alpha_k sin k w_t + beta_k cos k w_t),   w_t = 2 pi doy_t / 365.25
    storage balance S_t = S_{t-1} + x_t - Q_t,  Q_t = S_t / T  ->  Q_t = (S_{t-1} + x_t) / (T + 1)
    floor: Q_t = max(Q_t, 0), S_t = S_{t-1} + x_t - Q_t (deficit kept in storage)
    initial storage S = T * x_0 (steady state for the first day's effective inflow)
Ibar = mean model inflow over WY1983-1995 only. Observations enter only the training SSE (WY1983-1995 days).

Fitting (deliberately different optimiser from rc.py):
  L1: harness grid (T0 40 log pts 0.05..1000 d, a, b in linspace(-2, 2, 9)), no T floor in the fit;
      scored with T >= 0.05 (as ceiling.py / rc.py) and also without it (consistent with the fit).
  L4: stage 1, T0 on a 60-pt log grid 0.05..365, bounded linear LS (lsq_linear, [-3, 3]) for the 4 coefficients
      on the unfloored bucket (closed-form lfilter); stage 2, scipy least_squares (TRF, bounds) on the FLOORED
      training residuals over (log T0, 4 coefficients) from the 3 best stage-1 cells (ranked by floored SSE).
  L4lin: same but no floor anywhere.
Score WY1996-2010, same mask for every law: test & finite(obs).
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import zarr
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import lfilter

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
OUT = "/home/tbindas/.claude/jobs/dacd6d8c/tmp/rulecurve_audit"
T_LO, T_HI = 0.05, 365.0
G = {}


def load():
    z = zarr.open(f"{WT}/output/reservoir_smoke/pred_1981_2010.zarr", mode="r")
    G["ids"] = {bytes(r).decode().strip("\x00"): k for k, r in enumerate(z["gage_ids"][:])}
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    G["P"] = np.asarray(z["predictions"][:], dtype=np.float64)
    G["O"] = np.asarray(z["observations"][:], dtype=np.float64)
    wy = np.asarray(t.year + (t.month >= 10))
    G["train"] = (wy >= 1983) & (wy <= 1995)
    G["test"] = (wy >= 1996) & (wy <= 2010)
    w = 2 * np.pi * np.asarray(t.dayofyear, dtype=np.float64) / 365.25
    G["w"] = w
    G["H"] = np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    G["nfit"] = int(np.flatnonzero(G["train"])[-1]) + 1


def bucket_py(x, T, floor):
    """Pure-Python mass-balance loop. T scalar or array. Returns Q, floored flags."""
    n = len(x)
    Tarr = np.broadcast_to(np.asarray(T, float), (n,))
    Q = np.empty(n)
    fl = np.zeros(n, bool)
    S = float(Tarr[0]) * float(x[0])
    xs = x.tolist(); Ts = Tarr.tolist()
    for t in range(n):
        Tt = Ts[t]
        avail = S + xs[t]
        q = avail / (Tt + 1.0)
        if floor and q < 0.0:
            q = 0.0
            fl[t] = True
        S = avail - q
        Q[t] = q
    return Q, fl


def lin_resp(x, T):
    """Unfloored constant-T bucket as a first-order IIR filter: q_t = a q_{t-1} + (1-a) x_t, q_0 = x_0."""
    a = T / (T + 1.0)
    return lfilter([1.0 - a], [1.0, -a], x, zi=[a * x[0]])[0]


def nse(p, o, m):
    m = m & np.isfinite(o) & np.isfinite(p)
    p, o = p[m], o[m]
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def kge(p, o, m):
    m = m & np.isfinite(o) & np.isfinite(p)
    p, o = p[m], o[m]
    r = np.corrcoef(p, o)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2))


# ------------------------------------------------------------------ L1 (harness grid)
def fit_L1(I, O, w, train, nfit):
    T0g = np.logspace(np.log10(0.05), np.log10(1000), 40)
    ab = np.linspace(-2.0, 2.0, 9)
    T0, A, B = (v.ravel() for v in np.meshgrid(T0g, ab, ab, indexing="ij"))
    m = train & np.isfinite(O)
    sw, cw = np.sin(w), np.cos(w)
    S = None
    sse = np.zeros(T0.size)
    for t in range(nfit):
        T = T0 * np.exp(A * sw[t] + B * cw[t])
        if S is None:
            S = T * I[0]
        avail = S + I[t]
        q = avail / (T + 1.0)
        S = avail - q
        if m[t]:
            sse += (q - O[t]) ** 2
    k = int(np.argmin(sse))
    return float(T0[k]), float(A[k]), float(B[k])


def L1_series(I, w, T0, a, b, tfloor):
    T = T0 * np.exp(a * np.sin(w) + b * np.cos(w))
    if tfloor:
        T = np.maximum(T, 0.05)
    return bucket_py(I, T, floor=False)[0]


# ------------------------------------------------------------------ L4
def flux(Ibar, H, c):
    return Ibar * (np.asarray(c) @ H)


def fit_L4(I, O, H, Ibar, train, nfit, floor=True, cbound=3.0, thi=T_HI):
    n = nfit
    m = (train & np.isfinite(O))[:n]
    o = O[:n][m]
    Ii, Hh = I[:n], H[:, :n]

    def resid(theta):
        T0 = np.exp(theta[0])
        x = Ii - flux(Ibar, Hh, theta[1:])
        q = bucket_py(x, T0, floor)[0] if floor else lin_resp(x, T0)
        return q[m] - o

    cells = []
    for T0 in np.logspace(np.log10(T_LO), np.log10(thi), 60):
        base = lin_resp(Ii, T0)
        Z = np.stack([Ibar * lin_resp(Hh[j], T0) for j in range(4)])  # response to +Ibar*H_j
        # q = base - c @ Z  ->  minimise || (base - o) - c @ Z ||
        y = base[m] - o
        sol = lsq_linear(Z[:, m].T, y, bounds=(-cbound, cbound))
        th = np.concatenate([[np.log(T0)], sol.x])
        r = resid(th)
        cells.append((float(r @ r), th))
    cells.sort(key=lambda c: c[0])
    lb = np.array([np.log(T_LO)] + [-cbound] * 4)
    ub = np.array([np.log(thi)] + [cbound] * 4)
    best = None
    for sse0, th0 in cells[:3]:
        th0 = np.clip(th0, lb + 1e-9, ub - 1e-9)
        r = least_squares(resid, th0, bounds=(lb, ub), method="trf", x_scale=np.array([1.0, .1, .1, .1, .1]),
                          diff_step=1e-5, max_nfev=300)
        s = float(r.fun @ r.fun)
        if best is None or s < best[0]:
            best = (s, r.x)
        if sse0 < best[0]:
            best = (sse0, th0)
    th = best[1]
    return float(np.exp(th[0])), [float(v) for v in th[1:]]


def L4_series(I, H, Ibar, T0, c, floor=True):
    x = I - flux(Ibar, H, c)
    return bucket_py(x, T0, floor)


def gauge(s):
    t0 = time.time()
    k = G["ids"][s]
    I, O = G["P"][k].copy(), G["O"][k].copy()
    assert np.all(np.isfinite(I))
    train, test, w, H, nfit = G["train"], G["test"], G["w"], G["H"], G["nfit"]
    Ibar = float(I[train].mean())
    r = dict(STAID=s, Ibar=Ibar)
    r["a_L0_nse"], r["a_L0_kge"] = nse(I, O, test), kge(I, O, test)
    r["a_L0_train"] = nse(I, O, train)
    T0, a, b = fit_L1(I, O, w, train, nfit)
    r.update(a_L1_T0=T0, a_L1_a=a, a_L1_b=b)
    q = L1_series(I, w, T0, a, b, True)
    r["a_L1_nse"], r["a_L1_kge"], r["a_L1_train"] = nse(q, O, test), kge(q, O, test), nse(q, O, train)
    r["a_L1_nse_noTfloor"] = nse(L1_series(I, w, T0, a, b, False), O, test)
    for lab, fl in [("L4", True), ("L4lin", False)]:
        T0, c = fit_L4(I, O, H, Ibar, train, nfit, floor=fl)
        q, flg = L4_series(I, H, Ibar, T0, c, fl)
        r[f"a_{lab}_T0"] = T0
        for j, kk in enumerate(["c1s", "c1c", "c2s", "c2c"]):
            r[f"a_{lab}_{kk}"] = c[j]
        r[f"a_{lab}_nse"], r[f"a_{lab}_kge"], r[f"a_{lab}_train"] = nse(q, O, test), kge(q, O, test), nse(q, O, train)
        r[f"a_{lab}_floor_test"] = float(flg[test].mean())
    r["secs"] = time.time() - t0
    return r


if __name__ == "__main__":
    which = sys.argv[1]  # csv of STAIDs
    outname = sys.argv[2]
    ids = pd.read_csv(which, dtype=str).iloc[:, 0].tolist()
    t0 = time.time()
    with Pool(8, initializer=load) as pool:
        rows = pool.map(gauge, ids, chunksize=1)
    pd.DataFrame(rows).set_index("STAID").to_csv(f"{OUT}/{outname}")
    print(f"{len(rows)} gauges in {time.time() - t0:.0f} s", flush=True)
