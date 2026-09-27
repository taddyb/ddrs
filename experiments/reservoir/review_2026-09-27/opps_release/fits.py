"""One-parameter extensions of the seasonal bucket, offline, on the smoke set's routed no-dam flow.

Same machinery as experiments/reservoir/smoke/expected_release_fit.py (daily implicit Euler on storage,
same-day inflow, fit WY1983-1995 by NSE, score WY1996-2010). Each law starts from the gauge's fitted
(seas_T0, seas_a, seas_b) and adds ONE parameter on a small grid, re-gridding T0 over a factor-of-4 band
around the seasonal fit so the new parameter cannot win merely by re-tuning T0. Laws:

  seas      T_t = T0 exp(a sin w + b cos w);  Q = (S + I)/(T + 1)                 (reference; re-gridded T0)
  cap_nid   Q = min(bucket, f * Qmax_NID)      f in a log grid, inf = off        (needs NID max discharge)
  cap_mean  Q = min(bucket, f * mean(I))       f in a log grid, inf = off        (a free cap, the benchmark's)
  inflowT   T_t = T_seas(t) * (I_{t-1}/mean I)^c,  c in [-0.75, 1]               (lagged-inflow-driven T)
  storeT    T_t = T_seas(t) * (S_{t-1}/(T0 mean I))^c,  c in [-0.75, 1]          (storage-state-driven T)
  loss      Q = (1 - f) * bucket,  f in [0, 0.4]                                 (evaporation / withdrawal)
  spill     Q = bucket + max(S - k * mean(I) * 1 d, 0) / T_fast (T_fast = 0.1 d), k in a log grid, inf = off

Run: ~/projects/ddr/.venv/bin/python fits.py  -> fits.json, fits_by_gauge.csv
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
OUT = "/home/tbindas/.claude/jobs/dacd6d8c/tmp/opps_release"
T0_FACTORS = np.array([0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0])
CAP_F = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf])
CAP_MEAN_F = np.array([1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, np.inf])
C_GRID = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
LOSS_F = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
SPILL_K = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, np.inf])
T_FAST = 0.1


def simulate(I, sinw, cosw, T0, a, b, n, law, theta, Imean, obs=None, mask=None):
    """Vectorised over parameter cells (T0, a, b, theta all same length)."""
    T0 = np.asarray(T0, float); a = np.asarray(a, float); b = np.asarray(b, float); theta = np.asarray(theta, float)
    Ts = T0 * np.exp(a * sinw[0] + b * cosw[0])
    S = Ts * I[0]
    Iprev = I[0]
    sse = np.zeros_like(T0) if obs is not None else None
    Q = np.empty(n) if obs is None else None
    for t in range(n):
        Ts = T0 * np.exp(a * sinw[t] + b * cosw[t])
        if law == "inflowT":
            T = Ts * np.power(max(Iprev, 1e-3) / Imean, theta)
        elif law == "storeT":
            T = Ts * np.power(np.maximum(S, 1e-6) / (T0 * Imean), theta)
        else:
            T = Ts
        T = np.maximum(T, 0.05)
        avail = S + I[t]
        q = avail / (T + 1.0)
        if law == "spill":
            Sfull = theta * Imean
            S_after = avail - q
            sp = np.maximum(S_after - Sfull, 0.0) / (T_FAST + 1.0)
            q = q + sp
        S = avail - q
        if law in ("cap_nid", "cap_mean"):
            q_rel = np.minimum(q, theta)
            S = S + (q - q_rel)   # what the cap holds back stays in storage
            q = q_rel
        elif law == "loss":
            q = (1.0 - theta) * q  # the withheld fraction leaves the system
        Iprev = I[t]
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
    al, be = p.std() / o.std(), p.mean() / o.mean()
    return dict(nse=float(nse), kge=float(1 - np.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2)),
                r=float(r), alpha=float(al), beta=float(be))


def fit_law(I, O, sinw, cosw, train, law, t0s, a, b, grid, Imean, qmax_nid):
    if law == "cap_nid":
        if not np.isfinite(qmax_nid) or qmax_nid <= 0:
            return None, None
        thetas = grid * qmax_nid
    elif law == "cap_mean":
        thetas = grid * Imean
    elif law == "spill":
        thetas = grid  # in units of mean inflow * 1 day, applied inside simulate
    else:
        thetas = grid
    T0, TH = (x.ravel() for x in np.meshgrid(t0s, thetas, indexing="ij"))
    A = np.full_like(T0, a); B = np.full_like(T0, b)
    n_fit = int(np.flatnonzero(train)[-1]) + 1
    m = train & np.isfinite(O)
    otr = O[m]
    nse = 1 - simulate(I, sinw, cosw, T0, A, B, n_fit, law, TH, Imean, obs=O, mask=m) / ((otr - otr.mean()) ** 2).sum()
    k = int(np.argmax(nse))
    q = simulate(I, sinw, cosw, T0[k:k + 1], A[k:k + 1], B[k:k + 1], len(I), law, TH[k:k + 1], Imean)
    theta_raw = grid.ravel()[k % len(grid)] if law != "spill" else grid[k % len(grid)]
    return q, dict(T0=float(T0[k]), theta=float(theta_raw), train_nse=float(nse[k]))


def fit_gauge(job):
    s, role, I, O, sinw, cosw, train, test, t0, a, b, qmax_nid = job
    I = np.where(np.isfinite(I), I, np.nanmean(I))
    Imean = float(np.mean(I[train]))
    t0s = t0 * T0_FACTORS
    r = {"STAID": s, "role": role}
    laws = [("seas", np.array([0.0])), ("cap_nid", CAP_F), ("cap_mean", CAP_MEAN_F), ("inflowT", C_GRID),
            ("storeT", C_GRID), ("loss", LOSS_F), ("spill", SPILL_K)]
    for law, grid in laws:
        q, p = fit_law(I, O, sinw, cosw, train, law, t0s, a, b, grid, Imean, qmax_nid)
        if q is None:
            continue
        for k, v in metrics(q, O, test).items():
            r[f"{law}_{k}"] = v
        r[f"{law}_T0"] = p["T0"]; r[f"{law}_theta"] = p["theta"]; r[f"{law}_train_nse"] = p["train_nse"]
    for k, v in metrics(I, O, test).items():
        r[f"pass_{k}"] = v
    return r


def paired(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-9]; up = int((nz > 0).sum())
    rng = np.random.default_rng(42)
    bs = np.median(rng.choice(d, (2000, len(d))), axis=1) if len(d) > 1 else np.array([np.nan])
    return dict(n=int(len(d)), median=round(float(np.median(d)), 5) if len(d) else None,
                ci=[round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)],
                n_up=up, n_down=int(len(nz) - up), sign_p=round(float(binomtest(up, len(nz)).pvalue), 4) if len(nz) else None)


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    z = zarr.open(f"{WT}/output/reservoir_smoke/pred_1981_2010.zarr", mode="r")
    ids = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    P, O = z["predictions"][:].astype(float), z["observations"][:].astype(float)
    wy = np.asarray(t.year + (t.month >= 10))
    train, test = (wy >= 1983) & (wy <= 1995), (wy >= 1996) & (wy <= 2010)
    w = 2 * np.pi * np.asarray(t.dayofyear) / 365.25
    sinw, cosw = np.sin(w), np.cos(w)
    fit = pd.read_csv(f"{WT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str, "huc2": str, "control_for": str}).set_index("STAID")
    onr = fit[(fit.role == "dam") & (fit.on_reach.astype(str) == "True")]
    ctl = fit[(fit.role == "control") & fit.control_for.isin(onr.index)]
    sel = pd.concat([onr, ctl])
    jobs = []
    for s, row in sel.iterrows():
        i = ids.get_loc(s)
        jobs.append((s, row.role, P[i], O[i], sinw, cosw, train, test, float(row.seas_T0), float(row.seas_a), float(row.seas_b),
                     float(row.dam_max_discharge_m3s) if row.role == "dam" else np.nan))
    print(f"{len(onr)} on-reach dam gauges, {len(ctl)} matched controls, {nproc} procs", flush=True)
    with Pool(nproc) as pool:
        rows = pool.map(fit_gauge, jobs)
    df = pd.DataFrame(rows).set_index("STAID")
    df.to_csv(f"{OUT}/fits_by_gauge.csv")
    res = {"n": dict(dam=len(onr), control=len(ctl)), "laws": {}}
    for law in ["cap_nid", "cap_mean", "inflowT", "storeT", "loss", "spill"]:
        entry = {}
        for role in ["dam", "control"]:
            g = df[df.role == role]
            if f"{law}_nse" not in g:
                continue
            ok = g[f"{law}_nse"].notna()
            g = g[ok]
            entry[role] = dict(
                dnse_vs_seas=paired(g[f"{law}_nse"] - g["seas_nse"]),
                dkge_vs_seas=paired(g[f"{law}_kge"] - g["seas_kge"]),
                dalpha_vs_seas=paired(g[f"{law}_alpha"] - g["seas_alpha"]),
                dnse_vs_pass=paired(g[f"{law}_nse"] - g["pass_nse"]),
                median_nse=round(float(g[f"{law}_nse"].median()), 4),
                theta_pct=[round(float(v), 3) for v in np.nanpercentile(g[f"{law}_theta"].replace(np.inf, np.nan), [10, 25, 50, 75, 90])],
                frac_theta_neutral=round(float((g[f"{law}_theta"] == (0.0 if law in ("inflowT", "storeT", "loss") else np.inf)).mean()), 3),
                train_gain_vs_seas=round(float((g[f"{law}_train_nse"] - g["seas_train_nse"]).median()), 4),
            )
        # dam minus matched control, vs seas
        d = df[df.role == "dam"]; c = df[df.role == "control"]
        if f"{law}_nse" in d and f"{law}_nse" in c:
            cg = (c[f"{law}_nse"] - c["seas_nse"]); cg.index = ctl.loc[cg.index, "control_for"] if False else cg.index
            cmap = pd.Series((c[f"{law}_nse"] - c["seas_nse"]).values, index=fit.loc[c.index, "control_for"].values)
            dd = (d[f"{law}_nse"] - d["seas_nse"]) - d.index.map(cmap).astype(float)
            entry["dam_minus_matched_control_vs_seas"] = paired(dd)
        res["laws"][law] = entry
    g = df[df.role == "dam"]
    res["seas_regrid"] = dict(dam_dnse_vs_pass=paired(g.seas_nse - g.pass_nse), dam_median_nse=round(float(g.seas_nse.median()), 4),
                              dam_dkge_vs_pass=paired(g.seas_kge - g.pass_kge), dam_dalpha_vs_pass=paired(g.seas_alpha - g.pass_alpha))
    json.dump(res, open(f"{OUT}/fits.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
