#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "netCDF4", "xarray", "zarr>=3", "scipy", "matplotlib", "icechunk>=2"]
# ///
"""Learned (n_0, gamma) fields of the blend's three members, recomputed from their checkpoint dumps.

Two parts.

1. Runs `experiments/stage_roughness/cross_arm_fields.py` unchanged on the member
   runs. That script reads `<run>/plot/kan_parameters.nc`; the UH fixed-channel run
   2026-09-27T07-29-47Z only has `plot/kan_parameters_epoch50.nc` (same final
   checkpoint, `epoch_50_mb_9`), so the runs are presented through a read-only
   symlink farm under --out and the script's RUNS constant is pointed at it. No
   run directory is written.

2. The two reach sets the prediction is stated on:
     (a) every CONUS reach with a finite log10 drainage area (cross_arm_fields' set);
     (b) the reaches whose whole upstream divide set, the reach itself included, is
         in all three blend stores, so the blend reads exactly 0.50/0.25/0.25 on every
         divide that drains to them.
   For each member: n_0 and gamma median (with p25, p75), fraction of reaches at the
   n_0 ceiling, Spearman rho(n_0, gamma), and medians by drainage-area class.

    uv run experiments/blend_prediction/member_fields.py --out <dir> [--add blend=<run-id>]

The primary statistic of research/specs/2026-09-30-blend-arm-n0-prediction.md is the
`n0_med` cell of row (label, b_full_cover) in member_fields.csv.
"""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import zarr
from netCDF4 import Dataset
from scipy.stats import spearmanr

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
HERE = Path(__file__).resolve().parent
CROSS = HERE.parent / "stage_roughness" / "cross_arm_fields.py"
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
CONUS = "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr"
STORES = [
    "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic",
    "/mnt/ssd1/data/icechunk/daily_dhbv2_distributed_aorc2f_merit_unit_catchments_rechunked.ic",
    "/mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic",
]
# label, run id, parameter dump inside the run (all are the epoch_50_mb_9 head)
MEMBERS = [
    ("uh_0927", "2026-09-27T07-29-47Z-train-and-test", "plot/kan_parameters_epoch50.nc"),
    ("uh_0912", "2026-09-12T23-39-03Z-train-and-test", "plot/kan_parameters.nc"),
    ("dist", "2026-09-13T17-22-30Z-train-and-test", "plot/kan_parameters.nc"),
    ("lstm", "2026-09-13T13-55-03Z-train-and-test", "plot/kan_parameters.nc"),
]
SIZE = [(-np.inf, 1), (1, 2), (2, 3), (3, 4), (4, np.inf)]  # log10 km2
SIZE_LAB = ["<10", "10-100", "100-1e3", "1e3-1e4", ">1e4"]
N_CEIL = 0.2495


def store_divides(path):
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    g = zarr.open_group(repo.readonly_session("main").store, mode="r")
    return np.asarray(g["divide_id"][:], dtype=np.int64)


def upstream_full_cover():
    """Per CONUS reach: is every divide upstream of it (itself included) in all three stores?"""
    c = zarr.open_group(CONUS, mode="r")
    order = np.asarray(c["order"][:], dtype=np.int64)
    rows = np.asarray(c["indices_0"][:], dtype=np.int64)  # downstream position
    cols = np.asarray(c["indices_1"][:], dtype=np.int64)  # upstream position
    assert np.all(rows >= cols), "adjacency not lower-triangular"
    own = np.ones(order.size, dtype=bool)
    for p in STORES:
        own &= np.isin(order, store_divides(p))
    full = own.copy()
    # positions are topologically ordered, so processing edges by increasing
    # downstream position settles every upstream reach before its receiver
    for k in np.argsort(rows, kind="stable"):
        if not full[cols[k]]:
            full[rows[k]] = False
    return pd.Series(own, index=order), pd.Series(full, index=order)


def summarize(df, lab, set_name):
    r = {"member": lab, "set": set_name, "n_reaches": len(df),
         "n0_med": df.n0.median(), "n0_p25": df.n0.quantile(.25), "n0_p75": df.n0.quantile(.75),
         "n0_ceil_pct": 100 * np.mean(df.n0 >= N_CEIL),
         "gamma_med": df.gamma.median(), "gamma_p25": df.gamma.quantile(.25), "gamma_p75": df.gamma.quantile(.75),
         "gamma_zero_pct": 100 * np.mean(df.gamma < 0.001),
         "rho_n0_gamma": spearmanr(df.n0, df.gamma).statistic}
    for (lo, hi), sl in zip(SIZE, SIZE_LAB):
        m = (df.la >= lo) & (df.la < hi)
        r[f"n0_{sl}"] = df.n0[m].median(); r[f"gamma_{sl}"] = df.gamma[m].median(); r[f"n_{sl}"] = int(m.sum())
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--add", action="append", default=[],
                    help="extra run scored the same way, label=<run-id> (dump plot/kan_parameters.nc); "
                         "for the blend read-out, after the run has finished")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for x in args.add:
        lab, rid, *dump = x.split("=")  # label=<run-id>[=<absolute dump path, when the run has no plot/ dump>]
        MEMBERS.append((lab, rid, dump[0] if dump else "plot/kan_parameters.nc"))

    # Part 1: cross_arm_fields.py, unchanged, through a symlink farm.
    farm = out / "runs_view"
    for _, rid, nc in MEMBERS:
        d = farm / rid; (d / "plot").mkdir(parents=True, exist_ok=True)
        for sub in ("baseline", "eval"):
            if not (d / sub).exists():
                (d / sub).symlink_to(RUNS / rid / sub)
        if not (d / "plot" / "kan_parameters.nc").exists():
            (d / "plot" / "kan_parameters.nc").symlink_to(RUNS / rid / nc)  # absolute nc overrides RUNS
    spec = importlib.util.spec_from_file_location("cross_arm_fields", CROSS)
    caf = importlib.util.module_from_spec(spec); spec.loader.exec_module(caf)
    caf.RUNS = farm
    import sys
    argv = sys.argv
    sys.argv = [str(CROSS)] + [f"{lab}={rid}" for lab, rid, _ in MEMBERS] + ["--out", str(out / "cross_arm_fields")]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        caf.main()
    sys.argv = argv
    (out / "cross_arm_fields.txt").write_text(buf.getvalue())
    print(buf.getvalue())

    # Part 2: the two reach sets.
    own, full = upstream_full_cover()
    print(f"CONUS reaches: {own.size:,}; own divide in all three stores {own.sum():,}; "
          f"whole upstream set in all three stores {full.sum():,}")
    at = Dataset(ATTRS)
    la_by = pd.Series(np.asarray(at["log10_uparea"][:], dtype=np.float64),
                      index=np.asarray(at["COMID"][:], dtype=np.int64))
    rows = []
    for lab, rid, nc in MEMBERS:
        ds = Dataset(RUNS / rid / nc)
        assert ds.getncattr("checkpoint").endswith("epoch_50_mb_9/head"), ds.getncattr("checkpoint")
        c = np.asarray(ds["COMID"][:], dtype=np.int64)
        df = pd.DataFrame({"n0": np.asarray(ds["n"][:], dtype=np.float64),
                           "gamma": np.asarray(ds["gamma"][:], dtype=np.float64),
                           "la": la_by.reindex(c).values}, index=c)
        df = df[np.isfinite(df.la)]
        rows.append(summarize(df, lab, "a_all"))
        rows.append(summarize(df[full.reindex(df.index, fill_value=False).values], lab, "b_full_cover"))
    t = pd.DataFrame(rows)
    t.to_csv(out / "member_fields.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(t.round(4).to_string(index=False))
    np.save(out / "comids_full_upstream_cover.npy", full.index.values[full.values])
    (out / "member_fields.json").write_text(json.dumps({
        "n_conus": int(own.size), "n_own_cover": int(own.sum()), "n_full_upstream_cover": int(full.sum()),
        "members": [{"label": l, "run": r, "dump": n} for l, r, n in MEMBERS]}, indent=2))


if __name__ == "__main__":
    main()
