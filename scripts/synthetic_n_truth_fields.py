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
    """Decreasing power law, calibrated against the REAL CONUS log10_uparea
    distribution so the field actually spans [N_LO, N_HI] (not just
    approximately): n falls linearly in log-log space from N_HI at the 1st
    percentile of log10_uparea (smallest headwaters) to N_LO at the 99th
    percentile (largest rivers). Anchoring on the 1st/99th percentile rather
    than the true min/max avoids a handful of extreme-tail reaches
    compressing the realized range for everyone else — log10_uparea here is
    right-skewed (median much closer to the min than the max), so a fixed
    exponent centered on the median (an earlier version of this function)
    undershoots both bounds; this anchors directly to the bounds instead.
    Reaches beyond the 1st/99th percentile are clipped to N_HI/N_LO.
    """
    lo_x, hi_x = np.percentile(log10_uparea, [1, 99])
    log_n = np.log10(N_HI) + (log10_uparea - lo_x) * (
        np.log10(N_LO) - np.log10(N_HI)
    ) / (hi_x - lo_x)
    n = 10.0**log_n
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
