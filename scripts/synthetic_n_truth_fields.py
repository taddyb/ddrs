"""Prescribed truth-n fields for the synthetic-n recoverability experiment
(docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md §1).

Combines each prescribed n field with the fixed consensus geometry
(scripts/synthetic_n_consensus_geometry.py) into a single donor NetCDF per
variant, in the dump_parameters::write_netcdf schema (COMID dim, f32 vars)
that probe_zeta_gradient's --mode teacher --donor-params-nc reads.

Run from ddrs-py's venv, AFTER synthetic_n_consensus_geometry.py:
    cd ddrs-py && uv run python ../scripts/synthetic_n_truth_fields.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "output/synthetic_n"
ATTRS = REPO.parent / "ddr/data/merit_global_attributes_v2.nc"

N_LO, N_HI = 0.015, 0.15
N_CENTER = 0.08
SEED = 42


def leopold_maddock_n(log10_uparea: np.ndarray) -> np.ndarray:
    """Decreasing power law: n = clip(N_CENTER * (uparea/uparea_median)^-b, N_LO, N_HI).

    b is calibrated so the field spans roughly [N_LO, N_HI] across the real
    CONUS log10_uparea distribution (see design spec §1 footnote — a tuning
    detail, not a design fork).
    """
    median = np.median(log10_uparea)
    b = 0.15
    n = N_CENTER * 10.0 ** (-b * (log10_uparea - median))
    return np.clip(n, N_LO, N_HI).astype(np.float32)


def gaussian_noise_n(n_reaches: int) -> np.ndarray:
    """IID Gaussian field, no spatial structure — the null control."""
    rng = np.random.default_rng(SEED)
    spread = (N_HI - N_LO) / 4.0  # ~2 std devs to each bound from N_CENTER
    n = rng.normal(loc=N_CENTER, scale=spread, size=n_reaches)
    return np.clip(n, N_LO, N_HI).astype(np.float32)


def main() -> None:
    geom = xr.open_dataset(OUT_DIR / "consensus_geometry.nc")
    attrs = xr.open_dataset(ATTRS)
    attrs_by_comid = attrs.set_index(COMID="COMID").sel(COMID=geom["COMID"].values)
    log10_uparea = attrs_by_comid["log10_uparea"].values.astype(np.float64)

    variants = {
        "truth_leopold_maddock.nc": leopold_maddock_n(log10_uparea),
        "truth_gaussian.nc": gaussian_noise_n(len(geom["COMID"])),
    }

    for filename, n_vals in variants.items():
        out = OUT_DIR / filename
        xr.Dataset(
            {
                "n": ("COMID", n_vals),
                "q_spatial": ("COMID", geom["q_spatial"].values),
                "p_spatial": ("COMID", geom["p_spatial"].values),
            },
            coords={"COMID": geom["COMID"].values},
        ).to_netcdf(out)
        print(
            f"{out}: n range [{n_vals.min():.4f}, {n_vals.max():.4f}], "
            f"median {np.median(n_vals):.4f} ({len(n_vals)} reaches)"
        )


if __name__ == "__main__":
    main()
