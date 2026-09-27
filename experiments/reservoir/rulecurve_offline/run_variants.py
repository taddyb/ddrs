"""Floor variants of the rule-curve laws (does the gain need the zero-release floor?).

  L4ec / L5ec    fitted and scored with an engine-like clamp: Q := max(Q, 0) with the closed form S = T Q kept, so
                 a floored day forgets its deficit (mass is created), what ddrs's post-solve clamp_min does today
  L4lin          no floor at all (pure linear; negative releases allowed, a formal score only)
  L4pen / L5pen  floor with feedback, fitted with the training floored-day share held <= 1 % (penalty)
  L4asec         L4's parameters (fitted with the feedback floor) re-scored under the engine clamp, no refit
Also: created-mass share of test inflow for the ec variants.

Run: ~/projects/ddr/.venv/bin/python run_variants.py [nproc]  -> fits_variants.csv
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import rc
import run_fits as R

VLAWS = ["L4ec", "L4lin", "L4pen", "L5ec", "L5pen"]
_P = {}


def _init():
    R._init()
    _P["fits"] = pd.read_csv(f"{rc.HERE}/fits_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")


def created_share(g, law, p):
    """Mass created by the engine clamp over the test years, as a share of test inflow."""
    Q, fl = rc.run(g, law, p)
    r = rc.flux(g, law, p)
    x = g["I"] - (r if r is not None else 0.0)
    T = rc.Tseries(g, law, p, g["n"])
    te = np.flatnonzero(g["test"])
    a, b = te[0], te[-1]
    created = Q[a:b + 1].sum() + T[b] * Q[b] - T[a - 1] * Q[a - 1] - x[a:b + 1].sum()
    return float(created / g["I"][a:b + 1].sum())


def gauge(s):
    t0 = time.time()
    i = R._G["ids"].get_loc(s)
    g = rc.prep(R._G["P"][i], R._G["O"][i], R._G["doy"], R._G["train"], R._G["test"])
    params = rc.fit_all(g, VLAWS)
    r = {"STAID": s}
    for law, p in params.items():
        r.update(rc.score(g, law, p))
        for k, v in p.items():
            r[f"{law}_p_{k}"] = v
        if law.endswith("ec"):
            r[f"{law}_created"] = created_share(g, law, p)
    f = _P["fits"].loc[s]
    p4 = {k: float(f[f"L4_p_{k}"]) for k in ["T0"] + rc.CK}
    sc = rc.score(g, "L4ec", p4)
    r.update({k.replace("L4ec_", "L4asec_"): v for k, v in sc.items()})
    r["L4asec_created"] = created_share(g, "L4ec", p4)
    r["secs"] = time.time() - t0
    return r


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sg = pd.read_csv(f"{R.WT}/experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str})
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        rows = pool.map(gauge, list(sg.STAID), chunksize=1)
    df = pd.DataFrame(rows).set_index("STAID")
    df.to_csv(f"{rc.HERE}/fits_variants.csv")
    print(f"{len(df)} gauges in {time.time() - t0:.0f} s", flush=True)
