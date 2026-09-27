"""Feature-limited ceiling: can a head predict the per-dam rule-curve parameters from allowed inputs?

5-fold CV random forests (one per target, 400 trees, min_samples_leaf 5) predict each dam gauge's fitted parameters
from the 19 normalised NID head features (dam_features.csv, keyed by dam COMID), log10 DOR, and shape features of
the MODEL inflow climatology on training years (amplitude (Cmax - Cmin)/(2 Cbar), std(C)/Cbar, doy of C's peak as
sin/cos, first-harmonic phase psi1 as sin/cos, coefficient of variation of daily I). No observations, no observed
dam data. Targets:
  L1   log T0, a, b                         (the seasonal bucket, for comparison with ceiling.py's +0.0168)
  L3   log T0, f
  L3b  log T0, f, phi
  L4   log T0, c1s, c1c, c2s, c2c            (calendar frame)
  L4rot log T0 + the four coefficients rotated into the inflow's own phase frame (k=1 by psi1, k=2 by 2 psi1),
       predicted there and rotated back per dam
  L4pen same targets as L4 but from the floor-penalised fit
Placebo: each dam's matched control is simulated with parameters the same fold's forest predicts from the dam's NID
features and DOR but the control's own inflow-shape features; DiD = dam gain - placebo gain.

Run: ~/projects/ddr/.venv/bin/python features.py [nproc] -> features_by_gauge.csv, features.json
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import rc
import run_fits as R
import summarise as S

NID = ['log10_storage', 'log10_storage_max', 'log10_surface', 'log10_drainage', 'log10_storage_per_area',
       'log10_max_discharge', 'height', 'year', 'log10_storage_max_missing', 'log10_surface_missing',
       'log10_max_discharge_missing', 'year_missing', 'purpose_flood', 'purpose_hydro', 'purpose_supply',
       'purpose_irrigation', 'purpose_recreation', 'purpose_navigation', 'purpose_other']
SHAPE = ["clim_amp", "clim_std", "clim_peak_sin", "clim_peak_cos", "psi1_sin", "psi1_cos", "inflow_cv"]
TARGETS = {"L1": ["T0", "a", "b"], "L3": ["T0", "f"], "L3b": ["T0", "f", "phi"], "L4": ["T0"] + rc.CK,
           "L4rot": ["T0"] + rc.CK, "L4pen": ["T0"] + rc.CK}
BOUNDS = {"T0": (rc.T_MIN, rc.T_MAX), "a": (-2, 2), "b": (-2, 2), "f": (0, 1.5), "phi": (-90, 90),
          **{k: (-3, 3) for k in rc.CK}}


def psi1(g):
    """Phase of the first harmonic of the model climatology C(doy): C - Cbar ~ A cos(w - psi1)."""
    w = 2 * np.pi * np.arange(1, 366) / 365.25
    u = 2 * np.mean((g["C"] - g["Cbar"]) * np.sin(w))
    v = 2 * np.mean((g["C"] - g["Cbar"]) * np.cos(w))
    return float(np.arctan2(u, v))


def rotate(c, ang):
    """c = (c1s, c1c, c2s, c2c) as R_k cos(k w - th_k); rotate th_1 by -ang and th_2 by -2 ang."""
    out = np.empty_like(c)
    for k, (i_s, i_c) in enumerate([(0, 1), (2, 3)], start=1):
        R_ = np.hypot(c[..., i_s], c[..., i_c]); th = np.arctan2(c[..., i_s], c[..., i_c]) - k * ang
        out[..., i_s], out[..., i_c] = R_ * np.sin(th), R_ * np.cos(th)
    return out


_W = {}


def _init():
    R._init()


def prep_one(s):
    i = R._G["ids"].get_loc(s)
    return rc.prep(R._G["P"][i], R._G["O"][i], R._G["doy"], R._G["train"], R._G["test"])


def shape_job(s):
    g = prep_one(s)
    return s, psi1(g)


def sim_job(args):
    s, sets = args  # sets: {name: (law, params)}
    g = prep_one(s)
    r = {"STAID": s}
    for name, (law, p) in sets.items():
        sc = rc.score(g, law, p)
        r.update({k.replace(f"{law}_", f"{name}_", 1): v for k, v in sc.items() if not k.startswith(f"{law}_band")})
    return r


def cv_predict(Xd, Xc, Y, seed=0):
    """Out-of-fold predictions for dams (Xd) and placebo predictions for their controls (Xc, same fold's model)."""
    pd_, pc = np.zeros_like(Y), np.zeros_like(Y)
    r2 = []
    for j in range(Y.shape[1]):
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xd):
            rf = RandomForestRegressor(400, min_samples_leaf=5, random_state=seed, n_jobs=4).fit(Xd[tr], Y[tr, j])
            pd_[te, j] = rf.predict(Xd[te]); pc[te, j] = rf.predict(Xc[te])
        r2.append(1 - ((Y[:, j] - pd_[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum())
    return pd_, pc, r2


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    df, dam, ctl = S.load()
    fe = pd.read_csv(f"{rc.HERE}/dam_features.csv").set_index("COMID")
    with Pool(nproc, initializer=_init) as pool:
        ps = dict(pool.map(shape_job, list(dam.index) + list(ctl.STAID)))
    for g in (dam, ctl):
        key = g.index if g is dam else g.STAID
        g["psi1"] = [ps[s] for s in key]
        g["psi1_sin"], g["psi1_cos"] = np.sin(g.psi1), np.cos(g.psi1)
    nidX = fe.loc[dam.dam_COMID.astype(np.int64), NID].values
    res = {}
    rows = {}
    for setname, sel in [("on_reach", dam.on_reach.values), ("all_dams", np.ones(len(dam), bool))]:
        d, c = dam[sel], ctl[sel]
        Xd = np.column_stack([nidX[sel], np.log10(d.nid_dor.values), d[SHAPE].values])
        Xc = np.column_stack([nidX[sel], np.log10(d.nid_dor.values), c[SHAPE].values])  # placebo: dam's NID + DOR, control's inflow
        res[setname] = {"r2": {}}
        preds = {}
        for tgt, keys in TARGETS.items():
            law = "L4" if tgt == "L4rot" else tgt
            Y = np.column_stack([np.log(d[f"{law}_p_{k}"]) if k == "T0" else d[f"{law}_p_{k}"] for k in keys])
            if tgt == "L4rot":
                Y[:, 1:] = rotate(Y[:, 1:], d.psi1.values)
            pdm, pcm, r2 = cv_predict(Xd, Xc, Y)
            if tgt == "L4rot":
                pdm[:, 1:] = rotate(pdm[:, 1:], -d.psi1.values)
                pcm[:, 1:] = rotate(pcm[:, 1:], -c.psi1.values)  # placebo rotated back in the control's own frame
            res[setname]["r2"][tgt] = dict(zip(keys, [round(float(x), 3) for x in r2]))
            for who, P_, idx in [("dam", pdm, d.index), ("ctl", pcm, c.STAID.values)]:
                for i, s in enumerate(idx):
                    p = {}
                    for j, k in enumerate(keys):
                        v = float(np.exp(P_[i, j])) if k == "T0" else float(P_[i, j])
                        p[k] = float(np.clip(v, *BOUNDS[k]))
                    if law == "L3":
                        p["phi"] = 0.0
                    preds.setdefault(s, {})[f"cv{tgt}"] = (law, p)
                    if tgt in ("L4", "L4rot"):
                        preds[s][f"cv{tgt}ec"] = ("L4ec", p)
        with Pool(nproc, initializer=_init) as pool:
            out = pool.map(sim_job, list(preds.items()), chunksize=4)
        rows[setname] = pd.DataFrame(out).set_index("STAID")
        rows[setname].to_csv(f"{rc.HERE}/features_{setname}_by_gauge.csv")
    json.dump(res, open(f"{rc.HERE}/features_r2.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
