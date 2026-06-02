"""Load ddrs predictions/observations zarr stores into an xarray.Dataset.

Why this exists: ddrs writes zarr v3 via the `zarrs` Rust crate. Each array
carries `_ARRAY_DIMENSIONS` (xarray's pre-v3 convention) but NOT zarr v3's
native `dimension_names` metadata, so `xr.open_zarr` raises:

    KeyError: 'Zarr object is missing the `dimension_names` metadata which
    is required for xarray to determine variable dimensions.'

In addition, `gage_ids` is stored as `(G, 8) uint8` with `_dtype_hint: |S8`
rather than as a 1D string/bytes array, so even after a successful open
the gauge axis would still need manual decoding.

This helper handles both: it reads the store with the raw `zarr` library
and assembles an `xarray.Dataset` with sensible coords (decoded gauge IDs,
`datetime64[ns]` time axis). Root attributes are preserved on the Dataset.

Usage:
    from load_ddrs_predictions import load_predictions_zarr
    ds = load_predictions_zarr("output/predictions_latest.zarr")
    ds.sel(gage_ids="01013500")          # decoded ASCII works
    ds.sel(time=slice("2000-01-01", "2000-12-31"))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def _decode_gage_ids(arr: np.ndarray) -> np.ndarray:
    """Normalize gage_ids to a 1D array of Python strings.

    ddrs writes (G, 8) uint8 (bytes spread across a `char` axis). Older
    zarr/xarray versions stored as 1D |S8 bytes or object dtype. Handle all
    three so the same downstream code works regardless of store vintage.
    """
    if arr.ndim == 2 and arr.dtype == np.uint8:
        # (G, 8) uint8 → strip trailing NULs and decode ASCII per row
        return np.array(
            [bytes(row).rstrip(b"\x00").decode("ascii") for row in arr]
        )
    if arr.dtype.kind == "S":
        return np.array([x.decode("ascii") for x in arr])
    if arr.dtype.kind == "O":
        return np.array(
            [x.decode("ascii") if isinstance(x, bytes) else str(x) for x in arr]
        )
    return arr


def load_predictions_zarr(path: str | Path) -> xr.Dataset:
    """Open a ddrs predictions zarr as an xarray.Dataset.

    Returns a Dataset with:
        data_vars:  predictions(gage_ids, time), observations(gage_ids, time)
        coords:     gage_ids (str), time (datetime64[ns])
        attrs:      copied from the zarr root group (start time, end time,
                    version, model, evaluation basins file, ...)

    The `time` axis is reconstructed from the zarr's `nanoseconds since
    1970-01-01` int64 storage to `datetime64[ns]`.
    """
    z = zarr.open(str(path), mode="r")

    # Decode coords
    gage_ids = _decode_gage_ids(np.asarray(z["gage_ids"]))
    time_ns = np.asarray(z["time"])
    time = np.array(time_ns, dtype="datetime64[ns]")

    # Build data vars. Each predictions/observations is (gage_ids, time)
    # per its `_ARRAY_DIMENSIONS` attr.
    preds = np.asarray(z["predictions"])
    obs = np.asarray(z["observations"])

    ds = xr.Dataset(
        data_vars={
            "predictions": (("gage_ids", "time"), preds),
            "observations": (("gage_ids", "time"), obs),
        },
        coords={"gage_ids": gage_ids, "time": time},
        attrs=dict(z.attrs),
    )

    # Preserve per-variable attrs (units etc) when present
    for var in ("predictions", "observations"):
        try:
            ds[var].attrs.update({k: v for k, v in z[var].attrs.items()
                                  if not k.startswith("_")})
        except Exception:
            pass

    return ds


def load_baseline_zarr(path: str | Path) -> xr.Dataset:
    """Load a DDR summed-Q' baseline zarr.

    DDR's baseline stores follow the same schema as ddrs (same producer
    convention), so the loader is identical. Kept as a separate name for
    readability at call sites.
    """
    return load_predictions_zarr(path)
