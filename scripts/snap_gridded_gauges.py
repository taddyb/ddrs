"""Build the CONUS gauge CSV for ddrs's gridded (ISIMIP DDM30) routing path.

Gridded counterpart of the gages_3000 gauge selection used by the MERIT-fabric
path: takes `gages_3000.csv`, keeps gauges with DRAIN_SQKM above a minimum
(default 2,000 km2 -- a 0.5 degree DDM30 cell is ~2,350 km2, so 2,416 of the
3,211 gages_3000 gauges are sub-cell and cannot be represented by a single
routing element), snaps each surviving gauge to a DDM30 cell via
`ddr_benchmarks.gridded.snap_gauges` (drainage-area-matched 3x3 neighborhood
search against per-cell upstream area from the sub-reach adjacency zarr), then
keeps gauges present in the USGS observation store with daily coverage above a
minimum (default 0.9) over 1981-10-01..1997-09-30 (train_conus.py measures coverage
from `--train-start` to its `--test-end` default), which yields DDR's 620 gauges.
Mirrors `~/projects/ddr/examples/juniata_gridded/train_conus.py::build_network`
(lines 51-91) exactly, including its default thresholds.

Output columns: STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID,da_ratio
`COMID` is the snapped DDM30 cell id (ddrs's gauge reader joins on `COMID`);
`da_ratio` is snap_gauges's cell-upstream-area / gauge-DA ratio, informational.

Run under DDR's venv (needs `ddr_benchmarks` and `ddr` importable):
    cd ~/projects/ddr && \
    ~/projects/ddr/.venv/bin/python ~/projects/ddrs/scripts/snap_gridded_gauges.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

SUBREACH = Path("/home/tbindas/projects/ddr/data/ddm30/ddm30_subreach_adjacency.zarr")
OBS = Path("/mnt/ssd1/data/icechunk/usgs_daily_observations")
GAGES = Path("/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv")
OUT_DEFAULT = "/home/tbindas/projects/ddr/references/gage_info/ddm30_conus_gauges.csv"

MIN_DA_KM2 = 2000.0
MIN_COVERAGE = 0.9
START = "1981-10-01"
END = "1997-09-30"  # train_conus.py measures coverage over train_start..test_end (default 1997-09-30)

log = logging.getLogger("snap_gridded_gauges")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--min-da-km2", type=float, default=MIN_DA_KM2)
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from ddr_benchmarks.gridded import cell_areas_km2, snap_gauges, topo_accumulate

    from ddr.io.readers import read_ic

    g = zarr.open_group(str(SUBREACH), mode="r")
    parent = g["parent_cell"][:]
    node_ids = g["order"][:]
    rows, cols = g["indices_0"][:], g["indices_1"][:]
    n_all = len(node_ids)
    dn = np.full(n_all, -1, dtype=np.int64)
    dn[cols] = rows

    _, inv, cnt = np.unique(parent, return_inverse=True, return_counts=True)
    k_per_node = cnt[inv]
    uparea = topo_accumulate(cell_areas_km2(g["lat"][:]) / k_per_node, dn)

    # the cell's outlet is its most downstream sub-reach
    sub_idx = node_ids % 1000
    outlet_of_cell: dict[int, int] = {}
    for i in range(n_all):
        c = int(parent[i])
        if c not in outlet_of_cell or sub_idx[i] > sub_idx[outlet_of_cell[c]]:
            outlet_of_cell[c] = i
    ua_cell = {c: float(uparea[i]) for c, i in outlet_of_cell.items()}

    gauges = pd.read_csv(GAGES, dtype={"STAID": str})
    gauges["STAID"] = gauges.STAID.str.zfill(8)
    gauges = snap_gauges(gauges[gauges.DRAIN_SQKM > args.min_da_km2], ua_cell)
    obs_ds = read_ic(str(OBS))
    gauges = gauges[gauges.STAID.isin(set(obs_ds.gage_id.values.astype(str)))]
    cov = (
        obs_ds["streamflow"]
        .sel(gage_id=gauges.STAID.tolist(), time=slice(START, END))
        .notnull()
        .mean("time")
        .values
    )
    gauges = gauges[cov > args.min_coverage].reset_index(drop=True)
    log.info("%d gauges after DA match and observation coverage", len(gauges))

    out = gauges[["STAID", "STANAME", "DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "cell", "da_ratio"]].rename(
        columns={"cell": "COMID"}
    )
    out.to_csv(args.out, index=False, quoting=1)  # quoting=1 == csv.QUOTE_ALL, keeps STAID's leading zeros
    log.info("wrote %d gauges to %s", len(out), args.out)


if __name__ == "__main__":
    main()
