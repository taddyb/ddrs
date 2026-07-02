"""Write tiny icechunk Qr fixture stores for ddrs integration tests.

Run under DDR's uv venv (it has icechunk + xarray + zarr):

    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/make_streamflow_fixtures.py

Layout matches the DDR Q' store contract (docs/nh-qprime-store-contract.md):
Qr(divide_id, time) f32 m^3/s, divide_id int64, CF int64 time axis.

Deterministic values so tests can assert exact elements:
  qr_daily.ic   : 4 divides x 10 days,   Qr[j, t] = (j+1)*100  + t
  qr_hourly.ic  : 4 divides x 240 hours, Qr[j, h] = (j+1)*1000 + h
  qr_minutes.ic : sniff-rejection fixture (units "minutes since ...")

Note: xarray normalises "hours since 1981-01-01 00:00:00" to drop the time
component on write.  We patch the zarr attr back to the full string after
to_zarr() so the on-disk CF units string is exactly as documented above.
"""
from pathlib import Path
import shutil

import icechunk
import numpy as np
import xarray as xr
import zarr

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DIVIDES = np.array([101, 102, 103, 104], dtype=np.int64)


def write_store(
    path: Path,
    times: np.ndarray,
    qr: np.ndarray,
    time_units: str,
    *,
    time_units_on_disk: str | None = None,
) -> None:
    shutil.rmtree(path, ignore_errors=True)
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    ds = xr.Dataset(
        data_vars={
            "Qr": (["divide_id", "time"], qr.astype(np.float32), {"units": "m^3/s"}),
        },
        coords={
            "divide_id": ("divide_id", DIVIDES),
            "time": ("time", times),
        },
        attrs={"units": "m^3/s", "source": "ddrs test fixture"},
    )
    ds.to_zarr(
        session.store,
        mode="w",
        encoding={"time": {"units": time_units, "dtype": "int64"}},
    )
    # Patch the on-disk units attr if xarray normalised it (e.g. strips
    # " 00:00:00" from "hours since 1981-01-01 00:00:00").
    # Note: icechunk places the prior repo ref in overwritten/ on ANY write; this is normal.
    if time_units_on_disk is not None:
        z = zarr.open_group(session.store, mode="r+")
        z["time"].attrs["units"] = time_units_on_disk
    session.commit("fixture")
    print(f"wrote {path}")


def main() -> None:
    n_days = 10
    daily_times = np.datetime64("1981-01-01") + np.arange(n_days).astype("timedelta64[D]")
    daily = (np.arange(4)[:, None] + 1) * 100 + np.arange(n_days)[None, :]
    write_store(FIXTURES / "qr_daily.ic", daily_times, daily, "days since 1981-01-01")

    n_hours = n_days * 24
    hourly_times = np.datetime64("1981-01-01T00") + np.arange(n_hours).astype("timedelta64[h]")
    hourly = (np.arange(4)[:, None] + 1) * 1000 + np.arange(n_hours)[None, :]
    write_store(
        FIXTURES / "qr_hourly.ic", hourly_times, hourly,
        "hours since 1981-01-01 00:00:00",
        time_units_on_disk="hours since 1981-01-01 00:00:00",
    )

    # Same data, unsupported units string — exercises the sniff hard-error.
    write_store(
        FIXTURES / "qr_minutes.ic", hourly_times[:48], hourly[:, :48],
        "minutes since 1981-01-01",
    )


if __name__ == "__main__":
    main()
