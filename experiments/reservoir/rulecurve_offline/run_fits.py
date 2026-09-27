"""Fit every law per gauge on the 916 smoke gauges (458 dam gauges + 458 matched controls).

Run: ~/projects/ddr/.venv/bin/python run_fits.py [nproc] [limit]  -> fits_by_gauge.csv
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import zarr

import rc

WT = "/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options"
OUT = rc.HERE

_G = {}


def _init():
    z = zarr.open(f"{WT}/output/reservoir_smoke/pred_1981_2010.zarr", mode="r")
    _G["ids"] = pd.Index([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    _G["P"], _G["O"] = z["predictions"][:].astype(float), z["observations"][:].astype(float)
    wy = np.asarray(t.year + (t.month >= 10))
    _G["train"], _G["test"] = (wy >= 1983) & (wy <= 1995), (wy >= 1996) & (wy <= 2010)
    _G["doy"] = np.asarray(t.dayofyear)


def gauge(s):
    t0 = time.time()
    i = _G["ids"].get_loc(s)
    g = rc.prep(_G["P"][i], _G["O"][i], _G["doy"], _G["train"], _G["test"])
    params = rc.fit_all(g)
    r = {"STAID": s}
    r.update(rc.score(g, "L0", {}))
    for law, p in params.items():
        r.update(rc.score(g, law, p))
        for k, v in p.items():
            r[f"{law}_p_{k}"] = v
    r.update(rc.shape_features(g))
    r["secs"] = time.time() - t0
    return r


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sg = pd.read_csv(f"{WT}/experiments/reservoir/smoke/smoke_gauges.csv", dtype={"STAID": str, "control_for": str, "huc2": str})
    todo = list(sg.STAID)
    if limit:
        fit = pd.read_csv(f"{WT}/experiments/reservoir/smoke/expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID")
        onr = fit[(fit.role == "dam") & (fit.on_reach.astype(str) == "True")].index
        todo = list(onr[:limit])
    t0 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        rows = pool.map(gauge, todo, chunksize=1)
    df = pd.DataFrame(rows).set_index("STAID")
    name = "fits_by_gauge.csv" if not limit else "fits_probe.csv"
    df.to_csv(f"{OUT}/{name}")
    print(f"{len(df)} gauges in {time.time() - t0:.0f} s, median {df.secs.median():.1f} s/gauge", flush=True)
