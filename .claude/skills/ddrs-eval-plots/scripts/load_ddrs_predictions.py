"""Load ddrs predictions/observations zarr stores into an xarray.Dataset.

Why this exists: ddrs writes zarr v3 via the `zarrs` Rust crate. Each array
carries `_ARRAY_DIMENSIONS` (xarray's pre-v3 convention) but NOT zarr v3's
native `dimension_names` metadata, so `xr.open_zarr` raises:

    KeyError: 'Zarr object is missing the `dimension_names` metadata which
    is required for xarray to determine variable dimensions.'

In addition, `gage_ids` is stored as `(G, W) uint8` with `_dtype_hint: |S<W>`
(W = longest ID; exactly 8 in stores written before 2026-06-12, which
truncated global Provider__GageId names)
rather than as a 1D string/bytes array, so even after a successful open
the gauge axis would still need manual decoding.

This helper handles both: it reads the store with the raw `zarr` library
and assembles an `xarray.Dataset` with sensible coords (decoded gauge IDs,
`datetime64[ns]` time axis). Root attributes are preserved on the Dataset.

The companion `load_baseline_f32` reads the summed-Q' baseline, which is
NOT a zarr — `src/baseline/cache.rs` writes raw little-endian f32 planes
plus a JSON manifest. It returns the same `(gage_ids, time)` Dataset shape
so the two series can be intersected and scored side by side.

Usage:
    from load_ddrs_predictions import load_predictions_zarr, load_baseline_f32
    RUN = Path(".ddrs/runs/<id>")
    ds  = load_predictions_zarr(RUN / "eval" / "predictions.zarr")
    dsb = load_baseline_f32(RUN / "baseline")
    ds.sel(gage_ids="01013500")          # decoded ASCII works
    ds.sel(time=slice("2000-01-01", "2000-12-31"))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def _decode_gage_ids(arr: np.ndarray) -> np.ndarray:
    """Normalize gage_ids to a 1D array of Python strings.

    ddrs writes (G, W) uint8 (bytes spread across a `char` axis). Older
    zarr/xarray versions stored as 1D |S8 bytes or object dtype. Handle all
    three so the same downstream code works regardless of store vintage.
    """
    if arr.ndim == 2 and arr.dtype == np.uint8:
        # (G, W) uint8 → strip trailing NULs and decode ASCII per row
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


def load_baseline_f32(baseline_dir: str | Path) -> xr.Dataset:
    """Open a ddrs summed-Q' baseline directory as an xarray.Dataset.

    The baseline is NOT a zarr. `src/baseline/cache.rs` writes three files
    into `<run_dir>/baseline/` (copied from `.ddrs/baselines/<key>/`):

        predictions.f32   raw little-endian f32, row-major (n_gauges, n_days)
        observations.f32  same shape/dtype
        manifest.json     key, n_gauges, n_days, gage_ids, time_range_daily
                          (ISO `YYYY-MM-DD` strings), metrics, sources

    Returns the same structure as `load_predictions_zarr` — data_vars
    `predictions(gage_ids, time)` / `observations(gage_ids, time)`, coords
    `gage_ids` (str) and `time` (datetime64[ns]) — so the run and the
    baseline can be intersected on both axes and scored with one metric
    implementation.

    Root attrs carry the cache `key` and the `sources` provenance block.
    The manifest's precomputed per-gauge `metrics` (nse/rmse/kge/bias/fhv/flv,
    NaN encoded as JSON `null`) are attached as `ds.attrs["metrics"]` for
    reference, but prefer recomputing both series from the raw arrays so the
    comparison uses identical conventions.
    """
    bdir = Path(baseline_dir)
    manifest = json.loads((bdir / "manifest.json").read_text())

    n_gauges, n_days = manifest["n_gauges"], manifest["n_days"]
    pred = np.fromfile(bdir / "predictions.f32", dtype="<f4").reshape(n_gauges, n_days)
    obs = np.fromfile(bdir / "observations.f32", dtype="<f4").reshape(n_gauges, n_days)

    gage_ids = np.array([str(g) for g in manifest["gage_ids"]])
    time = np.array(manifest["time_range_daily"], dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )

    ds = xr.Dataset(
        data_vars={
            "predictions": (("gage_ids", "time"), pred),
            "observations": (("gage_ids", "time"), obs),
        },
        coords={"gage_ids": gage_ids, "time": time},
        attrs={
            "key": manifest.get("key"),
            "sources": manifest.get("sources"),
            "metrics": manifest.get("metrics"),
        },
    )
    ds["predictions"].attrs["units"] = "m^3/s"
    ds["observations"].attrs["units"] = "m^3/s"
    return ds
