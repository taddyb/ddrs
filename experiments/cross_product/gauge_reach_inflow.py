#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "xarray", "zarr>=3", "icechunk>=2,<3", "netCDF4", "scipy", "matplotlib"]
# ///
"""Daily lateral inflow of each gauge's own reach, per product, over WY1996-2010.

Why: the routed prediction scales the gauge reach's lateral inflow by FLOW_SCALE, the fraction of that unit
catchment upstream of the gauge (src/data/collate.rs::build_flow_scale, mirroring DDR readers.py), but the
summed-Q' baseline (src/baseline/summed_q_prime.rs) sums the whole unit catchment. Part of the routed-minus-
unrouted volume difference is therefore this bookkeeping, not routing. With the gauge reach's own inflow q_g,
the like-for-like unrouted series is S - (1 - FLOW_SCALE) q_g.

    experiments/cross_product/gauge_reach_inflow.py
Writes <OUT>/gauge_reach_inflow_<key>.npz (gauge, comid, flow_scale, days, q[g, t]); a COMID missing from the
product store gets the reader's 0.001 m3/s fill, as in ddrs.
"""
import icechunk
import numpy as np
import pandas as pd
import zarr

import cp_common as C
from qprime_medians import open_store, window


def main():
    meta = pd.read_csv(C.GAGES, dtype={"STAID": str}).set_index("STAID")
    ev = C.load_predictions_zarr(C.RUNS / C.RUN["uh"] / "eval" / "predictions.zarr")
    gauges = ev.gage_ids.values
    comid = meta["COMID"].reindex(gauges).to_numpy(dtype=np.int64)
    fs = meta["FLOW_SCALE"].reindex(gauges).to_numpy(dtype=float)
    days = pd.date_range(C.TEST0, C.TEST1, freq="D")
    for k in C.KEYS:
        dst = C.OUT / f"gauge_reach_inflow_{k}.npz"
        if dst.exists():
            print(f"{k}: exists")
            continue
        g = open_store(C.STORE[k])
        q = g["Qr"]
        ids = np.asarray(g["divide_id"][:], dtype=np.int64)
        pos = pd.Index(ids).get_indexer(comid)
        per_day, s0, n_days = window(g)
        out = np.full((gauges.size, n_days), 0.001)
        cd = q.chunks[0]
        have = np.where(pos >= 0)[0]
        for a in range(0, ids.size, cd):
            sel = have[(pos[have] >= a) & (pos[have] < a + cd)]
            if sel.size == 0:
                continue
            b = min(a + cd, ids.size)
            if per_day == 1:
                x = q[a:b, s0:s0 + n_days].astype(np.float64)
            else:
                x = np.empty((b - a, n_days))
                step = q.chunks[1] // 24
                for d in range(0, n_days, step):
                    e = min(d + step, n_days)
                    x[:, d:e] = q[a:b, s0 + 24 * d:s0 + 24 * e].astype(np.float64).reshape(b - a, e - d, 24).mean(2)
            out[sel] = x[pos[sel] - a]
        np.savez(dst, gauge=gauges, comid=comid, flow_scale=fs, days=days.values, q=out)
        print(f"{k}: {have.size}/{gauges.size} gauge reaches covered; median FLOW_SCALE {np.nanmedian(fs):.3f}")


if __name__ == "__main__":
    main()
