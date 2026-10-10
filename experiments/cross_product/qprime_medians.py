#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "xarray", "zarr>=3", "icechunk>=2,<3", "netCDF4", "scipy", "matplotlib"]
# ///
"""Per-divide coverage and median daily lateral inflow Q' of each product over WY1996-2010.

E-08 keeps reaches covered by every product (the COMID is on the product store's divide_id axis, so ddrs
reads real Q' rather than its 0.001 m3/s fill) whose median daily Q' over the test water years is at least
1e-3 m3/s in every product. Daily stores are read directly; the hourly MTS-LSTM store is averaged to
UTC calendar days first, so all six medians are medians of daily-mean flow.

    experiments/cross_product/qprime_medians.py
Writes <OUT>/qprime_median_<key>.npz (divide_id, median, nan_frac).
"""
import time

import icechunk
import numpy as np
import pandas as pd
import zarr

import cp_common as C


def open_store(path):
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    return zarr.open_group(repo.readonly_session(branch="main").store, mode="r")


def window(g):
    units = g["time"].attrs["units"]
    step, origin = units.split(" since ")
    t0 = pd.Timestamp(origin)
    n_days = (C.TEST1 - C.TEST0).days + 1
    d0 = (C.TEST0 - t0).days
    if step == "hours":
        return 24, d0 * 24, n_days
    assert step == "days", units
    return 1, d0, n_days


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    for k in C.KEYS:
        dst = C.OUT / f"qprime_median_{k}.npz"
        if dst.exists():
            print(f"{k}: exists, skipping")
            continue
        tic = time.time()
        g = open_store(C.STORE[k])
        q = g["Qr"]
        dims = tuple(q.metadata.dimension_names or q.attrs.get("_ARRAY_DIMENSIONS"))
        assert dims == ("divide_id", "time"), dims
        ids = np.asarray(g["divide_id"][:], dtype=np.int64)
        per_day, s0, n_days = window(g)
        cd = q.chunks[0]
        med = np.empty(ids.size, dtype=np.float64)
        nanf = np.empty(ids.size, dtype=np.float64)
        tchunk = q.chunks[1] if per_day == 24 else n_days
        for a in range(0, ids.size, cd):
            b = min(a + cd, ids.size)
            if per_day == 1:
                daily = q[a:b, s0:s0 + n_days].astype(np.float64)
            else:
                daily = np.empty((b - a, n_days))
                for d in range(0, n_days, tchunk // 24):
                    e = min(d + tchunk // 24, n_days)
                    x = q[a:b, s0 + 24 * d:s0 + 24 * e].astype(np.float64)
                    daily[:, d:e] = x.reshape(b - a, e - d, 24).mean(2)
            nanf[a:b] = np.isnan(daily).mean(1)
            med[a:b] = np.nanmedian(daily, axis=1)
        np.savez(dst, divide_id=ids, median=med, nan_frac=nanf)
        print(f"{k}: {ids.size:,} divides, median of medians {np.nanmedian(med):.4g} m3/s, "
              f"{np.mean(med >= 1e-3):.3f} at >= 1e-3, any NaN {np.mean(nanf > 0):.4f}, {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
