"""Hybrid ceiling: T0 from features (5-fold RF prediction of the L4 T0, as pooled.py), rule-curve coefficients as
per-dam free parameters calibrated on the gauge's training years (what an NID head for T0 plus a per-dam
coefficient table trained on the gauge loss could reach at gauged dams). On-reach set.

Run: ~/projects/ddr/.venv/bin/python hybrid.py -> hybrid_by_gauge.csv
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

import features as F
import rc
import run_fits as R
import summarise as S


def job(args):
    s, T0 = args
    i = R._G["ids"].get_loc(s)
    g = rc.prep(R._G["P"][i], R._G["O"][i], R._G["doy"], R._G["train"], R._G["test"])
    p = rc.fit_L4(g, fixed_T0=T0)  # LS + Nelder-Mead on the 4 coefficients, T0 held
    r = {"STAID": s}
    for name, law in [("hyb", "L4"), ("hybec", "L4ec")]:
        r.update({k.replace(f"{law}_", f"{name}_", 1): v for k, v in rc.score(g, law, p).items() if "_band_" not in k})
    r["hyb_T0"] = T0
    return r


if __name__ == "__main__":
    df, dam, ctl = S.load()
    fe = pd.read_csv(f"{rc.HERE}/dam_features.csv").set_index("COMID")
    d = dam[dam.on_reach].copy()
    with Pool(12, initializer=R._init) as pool:
        psis = dict(pool.map(F.shape_job, list(d.index)))
    d["psi1"] = [psis[s] for s in d.index]
    d["psi1_sin"], d["psi1_cos"] = np.sin(d.psi1), np.cos(d.psi1)
    nid = fe.loc[d.dam_COMID.astype(np.int64)]
    full = np.column_stack([nid[F.NID].values, np.log10(d.nid_dor.values), d[F.SHAPE].values])
    lt0 = np.log(d.L4_p_T0.values); T0p = np.zeros(len(d))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(full):
        T0p[te] = np.exp(RandomForestRegressor(400, min_samples_leaf=5, random_state=0, n_jobs=4).fit(full[tr], lt0[tr]).predict(full[te]))
    T0p = np.clip(T0p, rc.T_MIN, rc.T_MAX)
    with Pool(12, initializer=R._init) as pool:
        out = pd.DataFrame(pool.map(job, list(zip(d.index, T0p)))).set_index("STAID")
    out.to_csv(f"{rc.HERE}/hybrid_by_gauge.csv")
    x = d.join(out)
    for sub, idx in S.subsets(x, None):
        y = x.loc[idx]
        fl = y.dam_purpose == "Flood Risk Reduction"
        print(f"{sub:8s} n={len(idx):3d} hyb {S.fmt(S.paired(y.hyb_nse - y.L0_nse))} | hybec {S.fmt(S.paired(y.hybec_nse - y.L0_nse))} "
              f"| dKGE {S.paired(y.hyb_kge - y.L0_kge)['median']:+.4f} | floor {y.hyb_floor_test.mean():.3f} "
              f"| flood {S.fmt(S.paired((y.hyb_nse - y.L0_nse)[fl]))}")
