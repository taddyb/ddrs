"""Fairness check: L1 (seasonal bucket) refined by Nelder-Mead from its harness grid optimum, like the new laws.

Run: ~/projects/ddr/.venv/bin/python run_l1ref.py -> fits_l1ref.csv, then prints the on-reach comparison.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import rc
import run_fits as R
import summarise as S

_F = {}


def _init():
    R._init()
    _F["fits"] = pd.read_csv(f"{rc.HERE}/fits_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")


def job(s):
    i = R._G["ids"].get_loc(s)
    g = rc.prep(R._G["P"][i], R._G["O"][i], R._G["doy"], R._G["train"], R._G["test"])
    f = _F["fits"].loc[s]
    x0 = np.array([np.log(f.L1_p_T0), f.L1_p_a, f.L1_p_b])
    p, _ = rc._nm(g, "L1", ["T0", "a", "b"], [x0], [(np.log(0.05), np.log(1000.0)), (-2, 2), (-2, 2)], {})
    r = {"STAID": s}
    r.update({k.replace("L1_", "L1ref_", 1): v for k, v in rc.score(g, "L1", p).items() if "_band_" not in k})
    r.update({f"L1ref_p_{k}": v for k, v in p.items()})
    return r


if __name__ == "__main__":
    sg = pd.read_csv(f"{R.WT}/experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str})
    with Pool(12, initializer=_init) as pool:
        out = pd.DataFrame(pool.map(job, list(sg.STAID), chunksize=4)).set_index("STAID")
    out.to_csv(f"{rc.HERE}/fits_l1ref.csv")
    df, dam, ctl = S.load()
    d = dam[dam.on_reach].join(out)
    c = ctl[dam.on_reach.values]
    oc = out.reindex(c.STAID.values); oc.index = c.index; c = c.join(oc)
    for sub, idx in [("all", d.index), ("DOR>0.5", d.index[d.nid_dor > 0.5])]:
        x, y = d.loc[idx], c.loc[idx]
        gd, gc = x.L1ref_nse - x.L0_nse, (y.L1ref_nse - y.L0_nse).values
        print(sub, "L1ref dam", S.fmt(S.paired(gd)), "| ctl", S.fmt(S.paired(gc)), "| DiD", S.fmt(S.paired(gd.values - gc)),
              "| dKGE", S.paired(x.L1ref_kge - x.L0_kge)["median"], "| L4 - L1ref", S.fmt(S.paired(x.L4_nse - x.L1ref_nse)),
              "| T0 median", round(float(x.L1ref_p_T0.median()), 3))
