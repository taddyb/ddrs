"""Dam laws refitted on the per-HUC2-rescaled no-dam flow (k_h from kpool.json, fitted on training years), so that
runoff rescaling and the dam law set can be combined without double counting.
Run: ~/projects/ddr/.venv/bin/python run_kh.py [nproc] [kh|kg] -> laws_kh_by_gauge.csv / laws_kg_by_gauge.csv
(kg: the per-gauge scalar fitted on training years instead of the per-HUC2 one; the "Kh" columns then hold Kg.)"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import v6

_G = {}


def _init(kh, kc, huc, mode):
    _G.update(v6.load())
    _G["mode"] = mode
    _G["kh"], _G["kc"], _G["huc"] = kh, kc, huc


def gauge(s):
    i = _G["ids"].get_loc(s)
    g = v6.prep(_G["P"][i], _G["O"][i], _G["doy"], _G["train"], _G["test"])
    if _G["mode"] == "kg":  # per-gauge scalar fitted on training years
        k = v6.k_closed(g["I"], g["obs0"], g["mtr"].astype(bool))
    else:
        k = _G["kh"].get(_G["huc"][s], _G["kc"])
    I = k * g["I"]
    r = {"STAID": s, "k": k}
    r.update(v6.score(g, "Kh", v6.sim(g, mode=-1, T=0.0, I=I, want=True)))
    T2 = v6.fit_T0(g, I=I)
    r.update(v6.score(g, "L2", v6.sim(g, T=T2, I=I, want=True), dict(T0=T2)))
    T4, c4, Ib = v6.fit_L4(g, I=I)
    r4 = Ib * (c4 @ g["H"])
    r.update(v6.score(g, "L4", v6.sim(g, T=T4, I=I, r=r4, want=True), dict(T0=T4)))
    for name, mode, rr, Ts in [("FA", 0, None, T2), ("FC", 2, None, T2), ("FA4", 0, r4, T4), ("FC4", 2, r4, T4)]:
        x = v6.fit_flood(g, mode, Ts, r=rr, I=I)
        kw = v6.flood_kw(g, mode, *x)
        r.update(v6.score(g, name, v6.sim(g, I=I, r=rr, want=True, **kw),
                          dict(T0=kw["T"], kc=float(np.exp(x[1])), phi=float(x[2]), z=float(np.exp(x[3])))))
    return r


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    mode = sys.argv[2] if len(sys.argv) > 2 else "kh"
    sg, dam, ctl = v6.gauges()
    kp = json.load(open(v6.HERE + "/kpool.json"))
    t0 = time.time()
    with Pool(nproc, initializer=_init, initargs=(kp["kh"], kp["kc"], dict(zip(sg.STAID, sg.huc2)), mode)) as pool:
        rows = pool.map(gauge, list(sg.STAID), chunksize=2)
    df = pd.DataFrame(rows).set_index("STAID")
    df.to_csv(v6.HERE + ("/laws_kg_by_gauge.csv" if mode == "kg" else "/laws_kh_by_gauge.csv"))
    print(f"{len(df)} gauges in {time.time() - t0:.0f} s", flush=True)
