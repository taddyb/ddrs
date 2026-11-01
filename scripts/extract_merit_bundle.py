"""Extract a self-contained MERIT bundle for any gauge set.

Generalizes ~/projects/ddr/examples/juniata/extract_bundle.py, which reads
gauge groups from DDR's `merit_gages_conus_adjacency.zarr` and so only
handles the gages_3000 gauges.

Split of work:
  1. `ddrs plan` (managed adjacency builder, src/adjacency/gauges.rs::
     build_gauge_subgraphs) cuts one subgraph per gauge CSV row from the raw
     MERIT fabric. Output: a CONUS store and a gauges store, both in CONUS
     positional space.
  2. This script re-indexes the domain into a compact 0..N-1 space (members
     sorted by CONUS position, so the matrix stays lower-triangular) and
     slices the Q', observation and attribute stores.

Run under the DDR venv (icechunk, xarray, zarr, netCDF4, ddr.io.readers):

    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/extract_merit_bundle.py \
        --out OUT --name juniata --gages 01567000 --gage-csv GAGE.csv \
        --conus-zarr BUILT_CONUS.zarr --gages-zarr BUILT_GAGES.zarr \
        --outlet-gage 01567000 --time 1980-01-01:2010-12-31

Never run it from inside a directory that holds downloaded data.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import icechunk as ic
import numpy as np
import xarray as xr
import zarr

from ddr.io.readers import read_ic

ATTRS_NC = Path("~/projects/ddr/data/merit_global_attributes_v2.nc").expanduser()
QPRIME_IC = "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic"
OBS_IC = "/mnt/ssd1/data/icechunk/usgs_daily_observations"

KAN_INPUT_VARS = [
    "SoilGrids1km_clay",
    "aridity",
    "meanelevation",
    "meanP",
    "NDVI",
    "meanslope",
    "log10_uparea",
    "SoilGrids1km_sand",
    "ETPOT_Hargr",
    "Porosity",
]


def upstream_set(start_pos: int, rows: np.ndarray, cols: np.ndarray) -> set[int]:
    """Return CONUS positions of `start_pos` and everything upstream of it.

    Edge k means reach rows[k] (downstream) receives from cols[k] (upstream).
    """
    ups: dict[int, list[int]] = {}
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        ups.setdefault(r, []).append(c)
    seen = {start_pos}
    stack = [start_pos]
    while stack:
        for u in ups.get(stack.pop(), ()):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


def write_local_ic(path: Path, ds: xr.Dataset, message: str) -> None:
    """Create a local icechunk store at path and commit ds into it."""
    storage = ic.local_filesystem_storage(str(path))
    repo = ic.Repository.create(storage)
    session = repo.writable_session("main")
    ds.to_zarr(session.store, consolidated=False, mode="w")
    session.commit(message)


def attribute_statistics(ds: xr.Dataset) -> dict:
    """Per-variable statistics, as ddr.io.statistics.set_statistics computes them."""
    out = {}
    for attr in ds.data_vars:
        data = ds[attr].values
        out[attr] = {
            "min": float(np.nanmin(data, axis=0)),
            "max": float(np.nanmax(data, axis=0)),
            "mean": float(np.nanmean(data, axis=0)),
            "std": float(np.nanstd(data, axis=0)),
            "p10": float(np.nanpercentile(data, 10, axis=0)),
            "p90": float(np.nanpercentile(data, 90, axis=0)),
        }
    return out


def split_list(text: str | None) -> list[str]:
    """Split a comma-separated CLI value; empty or None gives []."""
    return [t.strip() for t in text.split(",") if t.strip()] if text else []


def main() -> None:
    """Run the extraction."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", required=True, help="file prefix, e.g. juniata")
    p.add_argument("--gages", required=True, help="comma-separated STAIDs, one group each")
    p.add_argument("--gage-csv", type=Path, required=True)
    p.add_argument("--conus-zarr", type=Path, required=True, help="built CONUS adjacency store")
    p.add_argument("--gages-zarr", type=Path, required=True, help="built gauges adjacency store")
    p.add_argument("--outlet-gage", required=True)
    p.add_argument("--domain-from-comid", type=int, default=None)
    p.add_argument("--time", required=True, help="START:END, e.g. 2001-01-01:2019-12-31")
    p.add_argument("--extra-obs", type=Path, default=None, help="CSV date,gage_id,q_m3s")
    p.add_argument("--nan-obs-gages", default=None, help="comma-separated STAIDs")
    p.add_argument("--qprime-ic", default=QPRIME_IC)
    p.add_argument("--obs-ic", default=OBS_IC)
    p.add_argument("--attributes", type=Path, default=ATTRS_NC)
    args = p.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    gages = split_list(args.gages)
    nan_gages = split_list(args.nan_obs_gages)
    t0, t1 = args.time.split(":")
    time_slice = slice(t0, t1)

    # --- 1. Domain and re-indexing (sorted CONUS position = topological order) ---
    conus = zarr.open_group(str(args.conus_zarr), mode="r")
    c_order = conus["order"][:]
    c_rows = conus["indices_0"][:]
    c_cols = conus["indices_1"][:]
    c_length = conus["length_m"][:]
    c_slope = conus["slope"][:]
    comid_to_pos = {int(c): i for i, c in enumerate(c_order)}

    gz_in = zarr.open_group(str(args.gages_zarr), mode="r")
    groups = {}
    for g in gages:
        grp = gz_in[g]
        attrs = dict(grp.attrs)
        gidx = int(attrs["gage_idx"])
        # The builder's cut and our own walk must agree (silent disagreement guard).
        members = {int(comid_to_pos[int(c)]) for c in grp["order"][:]}
        walk = upstream_set(gidx, c_rows, c_cols)
        if members != walk:
            raise SystemExit(
                f"gauge {g}: builder cut ({len(members)} reaches) differs from "
                f"upstream walk ({len(walk)} reaches) from position {gidx}"
            )
        groups[g] = {
            "attrs": attrs,
            "gidx": gidx,
            "rows": grp["indices_0"][:],
            "cols": grp["indices_1"][:],
            "members": members,
        }

    if args.domain_from_comid is not None:
        domain = upstream_set(comid_to_pos[args.domain_from_comid], c_rows, c_cols)
    else:
        if args.outlet_gage not in groups:
            raise SystemExit(f"--outlet-gage {args.outlet_gage} is not in --gages")
        domain = set(groups[args.outlet_gage]["members"])
    for g, info in groups.items():
        if not info["members"] <= domain:
            raise SystemExit(f"gauge {g}: upstream set is not inside the domain")

    member_pos = np.array(sorted(domain))
    old_to_new = {int(o): n for n, o in enumerate(member_pos)}
    n = len(member_pos)
    comids = c_order[member_pos].astype(np.int32)
    in_domain = np.isin(c_rows, member_pos)
    if not np.all(np.isin(c_cols[in_domain], member_pos)):
        raise SystemExit("domain is not upstream-closed")
    d_rows = np.array([old_to_new[int(i)] for i in c_rows[in_domain]], dtype=np.int32)
    d_cols = np.array([old_to_new[int(i)] for i in c_cols[in_domain]], dtype=np.int32)
    if not np.all(d_rows >= d_cols):
        raise SystemExit("re-indexed matrix is not lower-triangular")

    # --- 2. <name>_conus_adjacency.zarr ---
    cz = zarr.open_group(str(out / f"{args.name}_conus_adjacency.zarr"), mode="w")
    cz.create_array("indices_0", data=d_rows)
    cz.create_array("indices_1", data=d_cols)
    cz.create_array("order", data=comids)
    cz.create_array("values", data=np.ones(len(d_rows), dtype=np.uint8))
    cz.create_array("length_m", data=c_length[member_pos])
    cz.create_array("slope", data=c_slope[member_pos])
    cz.attrs.update({"format": "COO", "shape": [n, n], "geodataset": "merit"})

    # --- 3. <name>_gages_adjacency.zarr ---
    gz = zarr.open_group(str(out / f"{args.name}_gages_adjacency.zarr"), mode="w")
    for g, info in groups.items():
        sub_pos = np.array(sorted(info["members"]))
        sub_new = {int(o): old_to_new[int(o)] for o in sub_pos}
        r = np.array([sub_new[int(i)] for i in info["rows"]], dtype=np.int32)
        c = np.array([sub_new[int(i)] for i in info["cols"]], dtype=np.int32)
        grp = gz.create_group(g)
        grp.create_array("indices_0", data=r)
        grp.create_array("indices_1", data=c)
        grp.create_array("values", data=np.ones(len(r), dtype=np.uint8))
        grp.create_array("order", data=c_order[sub_pos].astype(np.int32))
        grp.attrs.update(
            {
                "format": "COO",
                "shape": [n, n],
                "geodataset": "merit",
                "gage_catchment": int(info["attrs"]["gage_catchment"]),
                "gage_idx": int(old_to_new[info["gidx"]]),
                "data_types": {"indices_0": "int32", "indices_1": "int32", "values": "uint8"},
            }
        )

    # --- 4. Q' icechunk store ---
    q_sub = read_ic(args.qprime_ic).sel(divide_id=comids, time=time_slice).compute()
    write_local_ic(out / f"{args.name}_qprime.ic", q_sub, f"Q' subset ({args.name})")

    # --- 5. Obs icechunk store ---
    ods = read_ic(args.obs_ic)
    store_ids = {str(x) for x in ods.gage_id.values}
    extra = None
    if args.extra_obs is not None:
        import pandas as pd

        extra = pd.read_csv(args.extra_obs, dtype={"gage_id": str}, parse_dates=["date"])
    extra_ids = set(extra.gage_id) if extra is not None else set()
    present = [g for g in gages if g in store_ids and g not in nan_gages]
    o_sub = ods.sel(gage_id=present, time=time_slice).compute()
    wanted = [g for g in gages if g in present or g in nan_gages or g in extra_ids]
    missing = [g for g in gages if g not in wanted]
    # Gauges with no source keep out of the obs store; ddrs filters by index.
    if missing:
        print(f"obs: no series for {missing}; not in the obs store")
    o_sub = o_sub.reindex(gage_id=wanted)
    if extra is not None:
        for gid, df in extra.groupby("gage_id"):
            s = df.set_index("date")["q_m3s"]
            s = s[(s.index >= o_sub.time.values[0]) & (s.index <= o_sub.time.values[-1])]
            idx = int(np.where(o_sub.gage_id.values == gid)[0][0])
            tpos = o_sub.indexes["time"].get_indexer(s.index)
            if (tpos < 0).any():
                raise SystemExit(f"extra obs for {gid} has dates outside the time axis")
            vals = o_sub["streamflow"].values.copy()
            vals[idx, tpos] = s.values
            o_sub["streamflow"] = (("gage_id", "time"), vals)
    for g in nan_gages:
        idx = int(np.where(o_sub.gage_id.values == g)[0][0])
        vals = o_sub["streamflow"].values.copy()
        vals[idx, :] = np.nan
        o_sub["streamflow"] = (("gage_id", "time"), vals)
    for var in ("streamflow", "gage_id", "time"):  # drop whole-store chunking
        o_sub[var].encoding.pop("chunks", None)
        o_sub[var].encoding.pop("preferred_chunks", None)
    write_local_ic(out / f"{args.name}_obs.ic", o_sub, f"Obs subset ({args.name})")

    # --- 6. Attributes NetCDF and statistics ---
    ads = xr.open_mfdataset(str(args.attributes))
    a_sub = ads[KAN_INPUT_VARS].sel(COMID=comids).compute()
    a_sub.to_netcdf(out / f"{args.name}_attributes.nc")
    stats_dir = out / "statistics"
    stats_dir.mkdir(exist_ok=True)
    with open(stats_dir / f"merit_attribute_statistics_{args.name}_attributes.nc.json", "w") as f:
        json.dump(attribute_statistics(a_sub), f, indent=2)

    # --- 7. Gage CSV ---
    with open(args.gage_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = {r[0]: r for r in reader}
    dest = out / f"{args.name}_gage.csv"
    if dest.resolve() == args.gage_csv.resolve():
        print("gage csv: input is the output path; not rewritten")
    else:
        with open(dest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for g in gages:
                w.writerow(rows[g])

    size = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
    print(f"bundle: {out} ({size})")
    print(f"reaches: {n}, edges: {len(d_rows)}, gauges: {len(groups)}")
    print(f"q' time: {q_sub.time.values[0]} .. {q_sub.time.values[-1]}")


if __name__ == "__main__":
    main()
