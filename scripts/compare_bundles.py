"""Compare two MERIT bundles written by scripts/extract_merit_bundle.py.

Usage (DDR venv, outside any data directory):
    ~/projects/ddr/.venv/bin/python -I scripts/compare_bundles.py REFERENCE_DIR CANDIDATE_DIR

Rules (the engine and the managed builder order edges differently):
  order, gage_idx, gage_catchment, Q', obs, attributes, statistics JSON and
  the gage CSV: equal exactly. (indices_0, indices_1): equal as a sorted edge
  set. length_m and slope: equal within 1e-6 relative where both sides are
  finite; rows where either side is NaN or inf are printed, not compared.
Prints `bundles match` on success; exits 1 on the first difference.
"""

import sys
from pathlib import Path

import icechunk as ic
import numpy as np
import xarray as xr
import zarr


def fail(msg: str) -> None:
    """Print a difference and stop."""
    print(f"DIFFERENCE: {msg}")
    sys.exit(1)


def open_ic(path: Path) -> xr.Dataset:
    """Open a local icechunk store read-only."""
    repo = ic.Repository.open(ic.local_filesystem_storage(str(path)))
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False).compute()


def prefix_of(d: Path) -> str:
    """Return the bundle prefix from `<name>_conus_adjacency.zarr`."""
    hits = sorted(d.glob("*_conus_adjacency.zarr"))
    if len(hits) != 1:
        fail(f"{d}: expected one *_conus_adjacency.zarr, found {len(hits)}")
    return hits[0].name.removesuffix("_conus_adjacency.zarr")


def edge_set(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return edges as a lexicographically sorted (k, 2) array."""
    e = np.stack([rows, cols], axis=1)
    return e[np.lexsort((e[:, 1], e[:, 0]))]


def compare_physical(name: str, a: np.ndarray, b: np.ndarray) -> None:
    """Compare length_m or slope; print non-finite rows without comparing them."""
    if a.shape != b.shape:
        fail(f"{name}: shape {a.shape} vs {b.shape}")
    bad = ~(np.isfinite(a) & np.isfinite(b))
    for i in np.where(bad)[0]:
        print(f"  {name}[{i}] not compared: ref={a[i]} cand={b[i]}")
    ok = ~bad
    if not np.allclose(a[ok], b[ok], rtol=1e-6, atol=0.0):
        i = np.where(ok)[0][~np.isclose(a[ok], b[ok], rtol=1e-6, atol=0.0)][0]
        fail(f"{name}[{i}]: ref={a[i]} cand={b[i]}")
    print(f"{name}: {int(ok.sum())} rows within 1e-6 relative, {int(bad.sum())} not compared")


def compare_ic(name: str, var: str, dim: str, ra: Path, rb: Path) -> None:
    """Compare an icechunk store elementwise; report the first differing id and date."""
    a, b = open_ic(ra), open_ic(rb)
    if not np.array_equal(a[dim].values, b[dim].values):
        fail(f"{name}: {dim} axis differs")
    if not np.array_equal(a.time.values, b.time.values):
        fail(f"{name}: time axis differs")
    x, y = a[var].values, b[var].values
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    if not same.all():
        i, j = np.argwhere(~same)[0]
        fail(f"{name}: first {dim}={a[dim].values[i]} time={a.time.values[j]}: {x[i, j]} vs {y[i, j]}")
    print(f"{name}: {x.shape} equal")


def main() -> None:
    """Compare the two bundle directories."""
    ref, cand = Path(sys.argv[1]), Path(sys.argv[2])
    na, nb = prefix_of(ref), prefix_of(cand)
    if na != nb:
        fail(f"prefix {na} vs {nb}")

    # CONUS store
    ca = zarr.open_group(str(ref / f"{na}_conus_adjacency.zarr"), mode="r")
    cb = zarr.open_group(str(cand / f"{na}_conus_adjacency.zarr"), mode="r")
    if dict(ca.attrs) != dict(cb.attrs):
        fail(f"conus attrs {dict(ca.attrs)} vs {dict(cb.attrs)}")
    if not np.array_equal(ca["order"][:], cb["order"][:]):
        fail("conus order differs")
    if not np.array_equal(edge_set(ca["indices_0"][:], ca["indices_1"][:]),
                          edge_set(cb["indices_0"][:], cb["indices_1"][:])):
        fail("conus edge set differs")
    print(f"conus order and {ca['indices_0'].shape[0]} edges equal")
    compare_physical("length_m", ca["length_m"][:], cb["length_m"][:])
    compare_physical("slope", ca["slope"][:], cb["slope"][:])

    # Gauge groups
    ga = zarr.open_group(str(ref / f"{na}_gages_adjacency.zarr"), mode="r")
    gb = zarr.open_group(str(cand / f"{na}_gages_adjacency.zarr"), mode="r")
    if sorted(ga.group_keys()) != sorted(gb.group_keys()):
        fail(f"gauge groups {sorted(ga.group_keys())} vs {sorted(gb.group_keys())}")
    for k in sorted(ga.group_keys()):
        a, b = ga[k], gb[k]
        for attr in ("gage_idx", "gage_catchment", "shape"):
            if a.attrs[attr] != b.attrs[attr]:
                fail(f"gauge {k} attr {attr}: {a.attrs[attr]} vs {b.attrs[attr]}")
        if not np.array_equal(a["order"][:], b["order"][:]):
            fail(f"gauge {k} order differs")
        if not np.array_equal(edge_set(a["indices_0"][:], a["indices_1"][:]),
                              edge_set(b["indices_0"][:], b["indices_1"][:])):
            fail(f"gauge {k} edge set differs")
        print(f"gauge {k}: order, gage_idx, gage_catchment, edge set equal")

    # Value stores
    compare_ic("qprime", "Qr", "divide_id", ref / f"{na}_qprime.ic", cand / f"{na}_qprime.ic")
    compare_ic("obs", "streamflow", "gage_id", ref / f"{na}_obs.ic", cand / f"{na}_obs.ic")
    aa = xr.open_dataset(ref / f"{na}_attributes.nc")
    ab = xr.open_dataset(cand / f"{na}_attributes.nc")
    if not np.array_equal(aa.COMID.values, ab.COMID.values):
        fail("attributes COMID differs")
    for v in aa.data_vars:
        if not np.array_equal(aa[v].values, ab[v].values, equal_nan=True):
            i = int(np.where(aa[v].values != ab[v].values)[0][0])
            fail(f"attributes {v} COMID={aa.COMID.values[i]}")
    print(f"attributes: {len(aa.data_vars)} vars equal")

    # Bytes
    sj = f"statistics/merit_attribute_statistics_{na}_attributes.nc.json"
    for rel in (sj, f"{na}_gage.csv"):
        if (ref / rel).read_bytes() != (cand / rel).read_bytes():
            fail(f"{rel} differs byte for byte")
        print(f"{rel}: bytes equal")
    print("bundles match")


if __name__ == "__main__":
    main()
