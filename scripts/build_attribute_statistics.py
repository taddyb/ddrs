"""Build the attribute-statistics sidecar JSON that ddrs reads at data-load time.

ddrs never recomputes normalization statistics itself (see
`src/data/dataset.rs::stats_path_from_attrs`): it expects a
`<attrs_dir>/statistics/merit_attribute_statistics_<attrs_filename>.json`
file to already exist next to any attributes NetCDF it opens, MERIT or
DDM30-gridded alike (the `merit_` prefix is a fixed template, not a claim
about the network type). This script produces that sidecar, mirroring DDR's
`src/ddr/io/statistics.py::set_statistics` but without needing a DDR `Config`
object, so it can run standalone under DDR's venv against any attributes file.

Usage:
    python build_attribute_statistics.py <attributes.nc> [--out-dir <dir>]

Default `--out-dir` is `<attributes dir>/statistics`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attributes", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.attributes.parent / "statistics")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"merit_attribute_statistics_{args.attributes.name}.json"

    ds = xr.open_dataset(args.attributes)
    stats = {}
    for attr in list(ds.data_vars.keys()):
        data = ds[attr].values
        stats[attr] = {
            "min": float(np.nanmin(data, axis=0)),
            "max": float(np.nanmax(data, axis=0)),
            "mean": float(np.nanmean(data, axis=0)),
            "std": float(np.nanstd(data, axis=0)),
            "p10": float(np.nanpercentile(data, 10, axis=0)),
            "p90": float(np.nanpercentile(data, 90, axis=0)),
        }

    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
