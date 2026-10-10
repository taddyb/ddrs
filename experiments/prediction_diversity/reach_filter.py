"""Per-reach coverage and median local inflow for every inflow product, test years WY1996-2010.

The reach filter for the n_0 comparisons (the E-08 filter in the paper outline): a reach counts when its
divide has a Q' series in every product's store ("covered by every product") and its median local Q' over
the test window is at least 1e-3 m3/s in every product. Reaches without a Q' series are read by ddrs as the
1e-3 fill, so their learned n_0 never meets real water.

The hourly MTS-LSTM store is reduced to daily means before the median (the routing is scored daily).

Writes <cache>/reach_flow_medians.npz: COMID (the 346,321-reach parameter order of plot/kan_parameters.nc),
and per product a float32 median (NaN where the store has no series for that reach).

    /home/tbindas/projects/ddr/.venv/bin/python reach_filter.py
"""
import pathlib
import re
import time

import icechunk
import numpy as np
import zarr
from netCDF4 import Dataset

RUNS = pathlib.Path("/home/tbindas/projects/ddrs/.ddrs/runs")
CACHE = pathlib.Path("/home/tbindas/projects/ddrs/.ddrs/experiments/prediction-diversity")
T0, T1 = np.datetime64("1995-10-01"), np.datetime64("2010-10-01")
PRODUCTS = {  # product label -> a seed-42 run that read it (store path comes from that run's config snapshot)
    "uh": "2026-09-12T23-39-03Z-train-and-test",
    "dhbv2_lumped": "2026-09-13T13-56-50Z-train-and-test",
    "dhbv2_dist": "2026-09-13T17-22-30Z-train-and-test",
    "hydrodl_lstm": "2026-09-13T17-21-58Z-train-and-test",
    "nh_daily_lstm": "2026-09-13T13-55-03Z-train-and-test",
    "nh_hourly_lstm": "2026-09-13T14-14-46Z-train-and-test",
}


def store_path(run):
    m = re.search(r"^\s+streamflow:\s*(\S+)", (RUNS / run / "config.yaml").read_text(), re.M)
    return m.group(1)


def median_local(path):
    """(divide_id, median over the test window of daily local Q') for one store."""
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    g = zarr.open_group(repo.readonly_session("main").store, mode="r")
    a = g["Qr"]
    div = np.asarray(g["divide_id"][:], dtype=np.int64)
    units = g["time"].attrs["units"]
    step = "h" if units.startswith("hours") else "D"
    origin = np.datetime64(units.split("since")[-1].strip().split()[0])
    tt = origin + np.asarray(g["time"][:]).astype(f"timedelta64[{step}]")
    i0, i1 = int(np.searchsorted(tt, T0)), int(np.searchsorted(tt, T1))
    assert a.shape[0] == div.size, "expected divide-major Qr"
    nday = (i1 - i0) // 24 if step == "h" else i1 - i0
    assert nday == 5479, nday
    med = np.empty(div.size, dtype=np.float32)
    cb = a.chunks[0]
    for s in range(0, div.size, cb):
        x = np.asarray(a[s:s + cb, i0:i1], dtype=np.float32)
        if step == "h":
            x = x.reshape(x.shape[0], nday, 24).mean(2)
        med[s:s + cb] = np.nanmedian(x, axis=1)
    return div, med


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    comid = np.asarray(Dataset(RUNS / PRODUCTS["uh"] / "plot" / "kan_parameters.nc")["COMID"][:], dtype=np.int64)
    pos = {c: i for i, c in enumerate(comid)}
    out = {"COMID": comid}
    for lab, run in PRODUCTS.items():
        t = time.time()
        path = store_path(run)
        div, med = median_local(path)
        col = np.full(comid.size, np.nan, dtype=np.float32)
        ix = np.array([pos.get(int(d), -1) for d in div])
        col[ix[ix >= 0]] = med[ix >= 0]
        out[lab] = col
        print(f"{lab:16s} {path}: {div.size:,} divides, {int((ix >= 0).sum()):,} in the network, "
              f"median of medians {np.nanmedian(col):.4f} m3/s, {time.time() - t:.0f} s", flush=True)
    np.savez(CACHE / "reach_flow_medians.npz", **out)
    M = np.stack([out[k] for k in PRODUCTS])
    covered = np.isfinite(M).all(0)
    keep = covered & (np.nan_to_num(M, nan=0.0) >= 1e-3).all(0)
    print(f"covered by every product: {int(covered.sum()):,} of {comid.size:,}; "
          f"and median local Q' >= 1e-3 m3/s in every product: {int(keep.sum()):,}")


if __name__ == "__main__":
    main()
