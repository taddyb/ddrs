"""Fit every v6 law per gauge on the 916 smoke gauges (458 dam gauges + 458 matched controls).

Train WY1983-1995 (WY1982 spin-up), score WY1996-2010. L2 / L4 parameters are the rulecurve_offline fits
(same protocol; v6.sim reproduces their test NSE to 4e-14, check_repro.py).
Run: ~/projects/ddr/.venv/bin/python run_laws.py [nproc]  -> laws_by_gauge.csv, kpool.json
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import v6

_G = {}


def _init(kh, kc, huc):
    _G.update(v6.load())
    _G["kh"], _G["kc"], _G["huc"] = kh, kc, huc
    _G["rcf"] = pd.read_csv(v6.RCFITS, dtype={"STAID": str}).set_index("STAID")


def gauge(s):
    t0 = time.time()
    i = _G["ids"].get_loc(s)
    g = v6.prep(_G["P"][i], _G["O"][i], _G["doy"], _G["train"], _G["test"])
    p = _G["rcf"].loc[s]
    r = {"STAID": s, "Ibar": g["Ibar"]}
    T2 = float(p.L2_p_T0)
    c4 = [p.L4_p_c1s, p.L4_p_c1c, p.L4_p_c2s, p.L4_p_c2c]
    r4 = v6.l4_flux(g, c4)
    r.update(v6.score(g, "L0", v6.sim(g, mode=-1, T=0.0, want=True)))
    r.update(v6.score(g, "L2", v6.sim(g, T=T2, want=True), dict(T0=T2)))
    r.update(v6.score(g, "L4", v6.sim(g, T=float(p.L4_p_T0), r=r4, want=True), dict(T0=float(p.L4_p_T0))))

    # flood-pool laws on top of the bucket (T0 refitted jointly)
    for name, mode in [("FA", 0), ("FB", 1), ("FC", 2)]:
        x = v6.fit_flood(g, mode, T2)
        kw = v6.flood_kw(g, mode, *x)
        r.update(v6.score(g, name, v6.sim(g, want=True, **kw),
                          dict(T0=kw["T"], kc=float(np.exp(x[1])), phi=float(x[2]), z=float(np.exp(x[3])))))
    # flood law + the L4 rule-curve flux (L4 coefficients held at their fitted values, flood + T0 refitted)
    for name, mode in [("FA4", 0), ("FC4", 2)]:
        x = v6.fit_flood(g, mode, float(p.L4_p_T0), r=r4)
        kw = v6.flood_kw(g, mode, *x)
        r.update(v6.score(g, name, v6.sim(g, r=r4, want=True, **kw),
                          dict(T0=kw["T"], kc=float(np.exp(x[1])), phi=float(x[2]), z=float(np.exp(x[3])))))

    # withdrawal laws on top of the bucket
    for name, shp in [("W1", v6.shape_W1(g)), ("W2", v6.shape_W2(g))]:
        x = v6.fit_withdraw(g, shp, T2)
        T = float(np.exp(x[0]))
        r.update(v6.score(g, name, v6.sim(g, T=T, Wstar=v6.withdraw_star(g, shp, x[1]), want=True),
                          dict(T0=T, w=float(x[1]))))
    x = v6.fit_withdraw3(g, T2)
    T = float(np.exp(x[0]))
    shp = v6.shape_W3(g, x[2], x[3])
    r.update(v6.score(g, "W3", v6.sim(g, T=T, Wstar=v6.withdraw_star(g, shp, x[1]), want=True),
                      dict(T0=T, w=float(x[1]), centre=float(x[2]), width=float(x[3]))))

    # runoff-rescaling nulls: per gauge / per HUC2 / global scalar on the no-dam flow, alone and under a bucket
    m = g["mtr"].astype(bool)
    kg = v6.k_closed(g["I"], g["obs0"], m)
    for name, k in [("Kg", kg), ("Kh", _G["kh"].get(_G["huc"][s], _G["kc"])), ("Kc", _G["kc"])]:
        Ik = k * g["I"]
        r.update(v6.score(g, name, v6.sim(g, mode=-1, T=0.0, I=Ik, want=True), dict(k=k)))
        T = v6.fit_T0(g, I=Ik)
        r.update(v6.score(g, "L2" + name, v6.sim(g, T=T, I=Ik, want=True), dict(T0=T, k=k)))
    # per-gauge scalar fitted jointly with T0 (bucket is linear on I >= 0, so k is closed form per T0)
    best = (np.inf, None)
    for T0 in v6.TGRID:
        q = v6.sim(g, n=g["n_fit"], T=T0, want=True)["Q"]
        k = v6.k_closed(q, g["obs0"][:g["n_fit"]], m[:g["n_fit"]])
        sse = float((((k * q - g["obs0"][:g["n_fit"]]) ** 2)[m[:g["n_fit"]]]).sum())
        if sse < best[0]:
            best = (sse, (T0, k))
    T0, k = best[1]
    r.update(v6.score(g, "L2Kj", v6.sim(g, T=T0, I=k * g["I"], want=True), dict(T0=T0, k=k)))
    r["secs"] = time.time() - t0
    return r


def pooled_k(D, sg):
    """Per-HUC2 and global scalars maximising the mean training NSE over the smoke gauges in the group."""
    rows = []
    for s, h in zip(sg.STAID, sg.huc2):
        i = D["ids"].get_loc(s)
        g = v6.prep(D["P"][i], D["O"][i], D["doy"], D["train"], D["test"])
        m = g["mtr"].astype(bool)
        rows.append(dict(STAID=s, huc2=h, a=float((g["I"][m] * g["O"][m]).sum() / g["den"]),
                         b=float((g["I"][m] ** 2).sum() / g["den"])))
    P = pd.DataFrame(rows)
    kc = float(P.a.sum() / P.b.sum())
    kh = (P.groupby("huc2").a.sum() / P.groupby("huc2").b.sum()).to_dict()
    nh = P.groupby("huc2").size().to_dict()
    kh = {h: (v if nh[h] >= 6 else kc) for h, v in kh.items()}  # tiny HUC2 groups fall back to the global scalar
    return kh, kc, nh


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sg, dam, ctl = v6.gauges()
    D = v6.load()
    kh, kc, nh = pooled_k(D, sg)
    json.dump(dict(kh=kh, kc=kc, n_by_huc2=nh), open(v6.HERE + "/kpool.json", "w"), indent=1)
    print("global k", round(kc, 4), "per-HUC2", {h: round(v, 3) for h, v in kh.items()}, flush=True)
    del D
    huc = dict(zip(sg.STAID, sg.huc2))
    t0 = time.time()
    with Pool(nproc, initializer=_init, initargs=(kh, kc, huc)) as pool:
        rows = pool.map(gauge, list(sg.STAID), chunksize=2)
    df = pd.DataFrame(rows).set_index("STAID")
    df.to_csv(v6.HERE + "/laws_by_gauge.csv")
    print(f"{len(df)} gauges in {time.time() - t0:.0f} s, median {df.secs.median():.1f} s/gauge", flush=True)
