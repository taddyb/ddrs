"""Head-like feature-limited ceiling for the harmonic rule curve: fit the coefficient MAP (features -> 4 coefficients)
by minimising the pooled training-years normalised SSE over dams directly (what a head trained on the loss does),
instead of regressing on noisy per-gauge parameter fits.

For fixed T0_i the release is linear in the coefficients: Q_i = B_T(I_i) - Z_i^T c_i, c_i = W x_i. The pooled
objective sum_i ||y_i - Z_i^T W x_i||^2 / den_i (y = B_T(I) - O over training days, den_i = training obs SS) is
ridge least squares in vec(W), built from per-dam 4x4 Gram matrices. T0_i is the 5-fold RF prediction of the
dam's fitted L4 T0 (features.py, same folds). Frames: 'cal' (calendar harmonics sin/cos k w) and 'rot' (harmonics
rotated into the dam's MODEL-inflow phase psi1, so a constant W row means 'fill k days before the inflow peak').
Feature sets: const (a single universal rule curve in that frame), small (log DOR, log storage/area, clim amp,
inflow CV, purposes), full (19 NID + log DOR + 7 shape). Ridge lambda by inner 5-fold CV on the training dams.
Outer 5-fold CV over dams; held-out dams simulated on test years (floor with feedback, and engine clamp).

Run: ~/projects/ddr/.venv/bin/python pooled.py [nproc]  -> pooled_by_gauge.csv, pooled.json, pooled.txt
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import features as F
import rc
import run_fits as R
import summarise as S

LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e4]
SMALL = ["log10_storage_per_area", "purpose_flood", "purpose_hydro", "purpose_supply", "purpose_irrigation",
         "purpose_recreation"]


def basis(g, frame, psi):
    w = 2 * np.pi * np.asarray(g["doy"]) / 365.25
    if frame == "cal":
        return np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    return np.stack([np.sin(w - psi), np.cos(w - psi), np.sin(2 * (w - psi)), np.cos(2 * (w - psi))])


def to_cal(c, frame, psi):
    """Rotated-frame coefficients (sin k(w-psi), cos k(w-psi)) -> calendar (c1s, c1c, c2s, c2c)."""
    if frame == "cal":
        return c
    out = np.empty(4)
    for k, (i_s, i_c) in enumerate([(0, 1), (2, 3)], start=1):
        q, p = c[i_s], c[i_c]  # q sin k(w-psi) + p cos k(w-psi)
        out[i_s] = q * np.cos(k * psi) + p * np.sin(k * psi)
        out[i_c] = p * np.cos(k * psi) - q * np.sin(k * psi)
    return out


def gram_job(args):
    s, T0, psi = args
    i = R._G["ids"].get_loc(s)
    g = rc.prep(R._G["P"][i], R._G["O"][i], R._G["doy"], R._G["train"], R._G["test"])
    g["doy"] = R._G["doy"]
    n = g["n_fit"]
    m = g["mtr"][:n].astype(bool)
    T = np.full(n, T0)
    y = rc._lin(g, g["I"][:n], T, n)[m] - g["obs0"][:n][m]
    out = {"STAID": s, "yy": float(y @ y), "den": g["den"]}
    for frame in ("cal", "rot"):
        H = basis(g, frame, psi)
        Z = np.stack([g["Ibar"] * rc._lin(g, H[j, :n], T, n)[m] for j in range(4)])
        out[f"G_{frame}"] = Z @ Z.T
        out[f"h_{frame}"] = Z @ y
    return out


def sim_job(args):
    s, sets = args
    i = R._G["ids"].get_loc(s)
    g = rc.prep(R._G["P"][i], R._G["O"][i], R._G["doy"], R._G["train"], R._G["test"])
    r = {"STAID": s}
    for name, (law, p) in sets.items():
        sc = rc.score(g, law, p)
        r.update({k.replace(f"{law}_", f"{name}_", 1): v for k, v in sc.items() if "_band_" not in k})
    return r


def solve(idx, X, G, h, den, lam):
    Fn = X.shape[1]
    A = np.zeros((4 * Fn, 4 * Fn)); b = np.zeros(4 * Fn)
    for i in idx:
        A += np.kron(np.outer(X[i], X[i]), G[i]) / den[i]
        b += np.kron(X[i], h[i]) / den[i]
    A += lam * np.eye(4 * Fn)
    return np.linalg.solve(A, b).reshape(Fn, 4)  # row f = coefficients of feature f


def sse(idx, X, W, G, h, yy, den):
    tot = 0.0
    for i in idx:
        c = X[i] @ W
        tot += (yy[i] - 2 * c @ h[i] + c @ G[i] @ c) / den[i]
    return tot


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    df, dam, ctl = S.load()
    fe = pd.read_csv(f"{rc.HERE}/dam_features.csv").set_index("COMID")
    d = dam[dam.on_reach].copy()
    with Pool(nproc, initializer=R._init) as pool:
        psis = dict(pool.map(F.shape_job, list(d.index)))
    d["psi1"] = [psis[s] for s in d.index]
    d["psi1_sin"], d["psi1_cos"] = np.sin(d.psi1), np.cos(d.psi1)
    nid = fe.loc[d.dam_COMID.astype(np.int64)]
    full = np.column_stack([nid[F.NID].values, np.log10(d.nid_dor.values), d[F.SHAPE].values])
    small = np.column_stack([nid[SMALL].values, np.log10(d.nid_dor.values), d[["clim_amp", "inflow_cv"]].values])
    # T0: out-of-fold RF prediction of the fitted L4 T0 (same folds / seeds as features.py)
    folds = list(KFold(5, shuffle=True, random_state=0).split(full))
    lt0 = np.log(d.L4_p_T0.values); T0p = np.zeros(len(d))
    for tr, te in folds:
        T0p[te] = np.exp(RandomForestRegressor(400, min_samples_leaf=5, random_state=0, n_jobs=4).fit(full[tr], lt0[tr]).predict(full[te]))
    T0p = np.clip(T0p, rc.T_MIN, rc.T_MAX)
    with Pool(nproc, initializer=R._init) as pool:
        gr = pool.map(gram_job, [(s, T0p[i], d.psi1.iloc[i]) for i, s in enumerate(d.index)])
    gr = {r["STAID"]: r for r in gr}
    yy = np.array([gr[s]["yy"] for s in d.index]); den = np.array([gr[s]["den"] for s in d.index])
    preds = {s: {} for s in d.index}
    info = {"T0_pred_median": float(np.median(T0p)), "lambda": {}}
    for fs_name, Xraw in [("const", np.zeros((len(d), 0))), ("small", small), ("full", full)]:
        for frame in ("cal", "rot"):
            G = np.array([gr[s][f"G_{frame}"] for s in d.index]); h = np.array([gr[s][f"h_{frame}"] for s in d.index])
            name = f"pool_{fs_name}_{frame}"
            lams = []
            for tr, te in folds:
                mu, sd = Xraw[tr].mean(0), Xraw[tr].std(0) + 1e-9
                X = np.column_stack([np.ones(len(d)), (Xraw - mu) / sd])
                best = None
                for lam in LAMBDAS:  # inner CV on the training dams
                    tot = 0.0
                    for itr, ite in KFold(5, shuffle=True, random_state=1).split(tr):
                        W = solve(tr[itr], X, G, h, den, lam)
                        tot += sse(tr[ite], X, W, G, h, yy, den)
                    if best is None or tot < best[0]:
                        best = (tot, lam)
                lams.append(best[1])
                W = solve(tr, X, G, h, den, best[1])
                for i in te:
                    c = to_cal(X[i] @ W, frame, d.psi1.iloc[i])
                    p = dict(T0=float(T0p[i]), **{k: float(np.clip(v, -3, 3)) for k, v in zip(rc.CK, c)})
                    preds[d.index[i]][name] = ("L4", p)
                    preds[d.index[i]][name + "ec"] = ("L4ec", p)
                    if fs_name == "const" and frame == "cal":
                        preds[d.index[i]]["pool_T0only"] = ("L2", dict(T0=float(T0p[i])))
            info["lambda"][name] = lams
    with Pool(nproc, initializer=R._init) as pool:
        out = pool.map(sim_job, list(preds.items()), chunksize=4)
    pr = pd.DataFrame(out).set_index("STAID")
    pr.to_csv(f"{rc.HERE}/pooled_by_gauge.csv")
    dd = d.join(pr)
    res, lines = {"info": info}, [f"T0 = RF-CV prediction of L4 T0 (median {np.median(T0p):.3f} d); lambdas {info['lambda']}"]
    names = ["pool_T0only"] + [f"pool_{a}_{b}{e}" for a in ("const", "small", "full") for b in ("cal", "rot") for e in ("", "ec")]
    for nm in names:
        res[nm] = {}
        lines.append(f"-- {nm}")
        for sub, idx in S.subsets(dd, None):
            x = dd.loc[idx]
            e = dict(dam=S.paired(x[f"{nm}_nse"] - x.L0_nse), dkge=S.paired(x[f"{nm}_kge"] - x.L0_kge),
                     mean_dnse_clip=round(float((x[f"{nm}_nse"].clip(lower=-1) - x.L0_nse.clip(lower=-1)).mean()), 4),
                     floor_mean=round(float(x[f"{nm}_floor_test"].mean()), 4))
            res[nm][sub] = e
            lines.append(f"  {sub:8s} n={len(idx):3d} dam {S.fmt(e['dam'])} mean(clip) {e['mean_dnse_clip']:+.4f} "
                         f"dKGE {e['dkge']['median']:+.4f} floor {e['floor_mean']:.3f}")
    json.dump(res, open(f"{rc.HERE}/pooled.json", "w"), indent=1)
    open(f"{rc.HERE}/pooled.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
