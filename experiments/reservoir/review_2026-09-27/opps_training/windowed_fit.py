"""Mechanism test, offline, on the smoke store: does fitting a linear bucket on 90-day hotstarted windows (what ddrs
training sees) recover a shorter T0 than fitting on the continuous series (what the offline fit and the test phase
see)? Per on-reach dam gauge, the no-dam routed flow is the inflow, the obs the target, daily implicit Euler, same
law as expected_release_fit.py. Objectives:
  cont           : continuous WY1983-1995 (spin-up WY1982), the offline fit
  win(rho,warm)  : K random rho-day windows starting in the training period, S0 = T * I(t0) (the hotstart analogue),
                   scored from day `warm`, SSE summed over windows
  oracle(90,5)   : same windows, S0 = the continuous run's storage at t0 under the same T (a converged state cache)
"""
import sys
import numpy as np
import pandas as pd
import zarr

ROOT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
z = zarr.open(f"{ROOT}/output/reservoir_smoke/pred_1981_2010.zarr", mode="r")
ids = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
P, O = z["predictions"][:].astype(float), z["observations"][:].astype(float)
wy = np.asarray(t.year + (t.month >= 10))
train = (wy >= 1983) & (wy <= 1995)
spin = wy >= 1982
tr_idx = np.flatnonzero(train)
fit = pd.read_csv(f"{ROOT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID")
dam = fit[(fit.role == "dam") & (fit.on_reach == True)]
T_GRID = np.logspace(np.log10(0.05), np.log10(1000), 40)
rng = np.random.default_rng(0)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def run_cont(I, T):
    """Continuous run from spin-up start; returns Q (n, cells) and S (n, cells)."""
    n = len(I)
    S = T * I[0]
    Q = np.empty((n, len(T)))
    Ss = np.empty((n, len(T)))
    for k in range(n):
        avail = S + I[k]
        q = avail / (T + 1.0)
        S = avail - q
        Q[k] = q
        Ss[k] = S
    return Q, Ss


def sse_windows(I, Obs, T, starts, rho, warm, S0=None):
    """Windows vectorised: (K, cells). S0 None -> hotstart S = T * I(t0)."""
    Kw = len(starts)
    S = np.broadcast_to(T[None, :] * I[starts][:, None], (Kw, len(T))).copy() if S0 is None else S0.copy()
    sse = np.zeros(len(T))
    for d in range(rho):
        idx = starts + d
        avail = S + I[idx][:, None]
        q = avail / (T[None, :] + 1.0)
        S = avail - q
        if d >= warm:
            o = Obs[idx]
            m = np.isfinite(o)
            if m.any():
                sse += (((q[m] - o[m][:, None]) ** 2)).sum(axis=0)
    return sse


rows = []
variants = [(90, 5), (90, 30), (120, 30), (180, 30), (365, 30), (365, 5)]
for s in dam.index:
    gi = ids.get_loc(s)
    I = np.where(np.isfinite(P[gi]), P[gi], np.nanmean(P[gi]))
    Ob = O[gi]
    # continuous: spin-up from WY1982 start
    s0 = int(np.flatnonzero(spin)[0])
    Ic, Oc = I[s0:], Ob[s0:]
    trc = train[s0:]
    Q, Ss = run_cont(Ic, T_GRID)
    m = trc & np.isfinite(Oc)
    sse_c = ((Q[m] - Oc[m][:, None]) ** 2).sum(axis=0)
    r = {"STAID": s, "T_cont": T_GRID[np.argmin(sse_c)], "lin_T0_csv": dam.loc[s, "lin_T0"], "seas_T0_csv": dam.loc[s, "seas_T0"]}
    # windows within the training period
    for rho, warm in variants:
        lo, hi = tr_idx[0], tr_idx[-1] - rho
        starts = rng.integers(lo, hi, size=K)
        sse_w = sse_windows(I, Ob, T_GRID, starts, rho, warm)
        r[f"T_win{rho}_{warm}"] = T_GRID[np.argmin(sse_w)]
        if (rho, warm) == (90, 5):
            # oracle state at t0 from the continuous run (index shift s0)
            S0 = Ss[starts - s0 - 1]  # storage after day t0-1
            sse_o = sse_windows(I, Ob, T_GRID, starts, rho, warm, S0=S0)
            r["T_oracle90_5"] = T_GRID[np.argmin(sse_o)]
    rows.append(r)

df = pd.DataFrame(rows).set_index("STAID")
df.to_csv("/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_training/windowed_fit.csv")
print("gauges:", len(df), "windows per gauge:", K)
print("sanity: Spearman(T_cont, lin_T0 csv) = %.3f; median T_cont %.2f vs csv %.2f" % (
    df[["T_cont", "lin_T0_csv"]].corr(method="spearman").iloc[0, 1], df.T_cont.median(), df.lin_T0_csv.median()))
act = df[df.T_cont > 0.06]
print(f"active (T_cont > floor): {len(act)} of {len(df)}")
cols = [c for c in df.columns if c.startswith("T_win") or c.startswith("T_oracle")]
print("medians over active gauges (days):  T_cont %.2f" % act.T_cont.median())
for c in cols:
    ratio = act[c] / act.T_cont
    print(f"  {c:16s} median {act[c].median():6.2f}  ratio to cont: median {np.median(ratio):.2f}, IQR {np.percentile(ratio,25):.2f}-{np.percentile(ratio,75):.2f};"
          f" below half: {(ratio < 0.5).mean():.2f}; at floor: {(act[c] <= 0.06).mean():.2f}; Spearman with cont {act[[c,'T_cont']].corr(method='spearman').iloc[0,1]:.2f}")
# by T_cont bins
act = act.copy()
act["bin"] = pd.cut(act.T_cont, [0.06, 0.5, 2, 10, 50, 2000])
print(act.groupby("bin")[["T_cont", "T_win90_5", "T_oracle90_5", "T_win90_30", "T_win180_30", "T_win365_30"]].median().assign(n=act.groupby("bin").size()))
