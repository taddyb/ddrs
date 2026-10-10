"""Shared table, loaders and statistics for the cross-product analysis (E-10, E-08).

The six fixed-channel models: the same n_0 + gamma head, channel (p = 21, q = 0.65), 2,365 gauges,
network, seed and 500-update recipe; only the unit-catchment inflow product differs. Run ids are the
outline's section 6 (`~/papers/ddr_equifinality/outline_2026-09-25.md`), unless the E-01 lineage doc
re-points them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
from load_ddrs_predictions import load_baseline_f32, load_predictions_zarr  # noqa: E402

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
OUT = Path("/home/tbindas/projects/ddrs/.ddrs/experiments/cross_product/2026-09-30")
HERE = Path(__file__).resolve().parent
FIG = HERE.parents[1] / "research" / "figures" / "cross-product"
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"
GAGES_ADJ = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
IC = "/mnt/ssd1/data/icechunk/"

# key, descriptive label, run id, unit-catchment Q' store (from each run's config.yaml data_sources.streamflow)
MODELS = [
    ("uh", "UH fixed-channel", "2026-09-12T23-39-03Z-train-and-test", IC + "merit_dhbv2_UH_retrospective.ic"),
    ("dhbv2_lumped", "dHBV2 lumped", "2026-09-13T13-56-50Z-train-and-test",
     IC + "daily_dhbv2_lumped_aorc2f_merit_unit_catchments_rechunked.ic"),
    ("dhbv2_dist", "dHBV2 distributed", "2026-09-13T17-22-30Z-train-and-test",
     IC + "daily_dhbv2_distributed_aorc2f_merit_unit_catchments_rechunked.ic"),
    ("hydrodl_lstm", "hydroDL LSTM", "2026-09-13T17-21-58Z-train-and-test",
     IC + "daily_hydrodl_lstm_aorc2f_merit_unit_catchments_rechunked.ic"),
    ("nh_daily_lstm", "NH daily LSTM", "2026-09-13T13-55-03Z-train-and-test", IC + "daily_lstm_merit_unit_catchments.ic"),
    ("nh_hourly_mts", "NH hourly MTS-LSTM", "2026-09-13T14-14-46Z-train-and-test", IC + "hourly_lstm_merit_unit_catchments.ic"),
]
KEYS = [m[0] for m in MODELS]
LABEL = {m[0]: m[1] for m in MODELS}
RUN = {m[0]: m[2] for m in MODELS}
STORE = {m[0]: m[3] for m in MODELS}

TEST0, TEST1 = pd.Timestamp("1995-10-01"), pd.Timestamp("2010-09-30")  # WY1996-2010
AREA_CLASSES = [("< 300", 0, 300), ("300-3,000", 300, 3000), ("3,000-30,000", 3000, 30000), ("> 30,000", 30000, np.inf)]
N_BOX = (0.015, 0.25)
G_BOX = (0.0, 0.5)


def load_series():
    """Routed, unrouted (summed Q') and observed daily series on one gauge x day grid for all six models.

    Returns (gauges, days, R[k] [g, t], S[k] [g, t], O [g, t]). Gauges and days are the intersection of every
    model's eval predictions and baseline. Observations come from the UH fixed-channel run and are checked
    identical (NaN-aware) in every other run, both in the eval zarr and in the baseline.
    """
    ev = {k: load_predictions_zarr(RUNS / RUN[k] / "eval" / "predictions.zarr") for k in KEYS}
    bl = {k: load_baseline_f32(RUNS / RUN[k] / "baseline") for k in KEYS}
    g = ev[KEYS[0]].gage_ids.values
    for k in KEYS:
        g = np.intersect1d(g, ev[k].gage_ids.values)
        g = np.intersect1d(g, bl[k].gage_ids.values)
    t = ev[KEYS[0]].time.values
    for k in KEYS:
        t = np.intersect1d(t, ev[k].time.values)
        t = np.intersect1d(t, bl[k].time.values)
    R, S = {}, {}
    O = ev[KEYS[0]].observations.sel(gage_ids=g, time=t).values.astype(np.float64)
    for k in KEYS:
        R[k] = ev[k].predictions.sel(gage_ids=g, time=t).values.astype(np.float64)
        S[k] = bl[k].predictions.sel(gage_ids=g, time=t).values.astype(np.float64)
        for name, oo in (("eval", ev[k].observations), ("baseline", bl[k].observations)):
            ok = oo.sel(gage_ids=g, time=t).values.astype(np.float64)
            same = np.array_equal(np.isnan(ok), np.isnan(O)) and np.allclose(ok[np.isfinite(ok)], O[np.isfinite(O)], rtol=1e-6)
            assert same, f"observations differ: {k} {name}"
    return g, pd.DatetimeIndex(t), R, S, O


def gauge_area(gauges):
    meta = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")
    return meta["DRAIN_SQKM"].reindex(gauges).to_numpy(dtype=float)


def basin_comids(gauges):
    """Upstream COMIDs per gauge from the gauge subgraph store the six runs trained on."""
    import zarr
    gz = zarr.open_group(GAGES_ADJ, mode="r")
    out = {}
    for s in gauges:
        try:
            out[s] = np.asarray(gz[str(s)]["order"][:], dtype=np.int64)
        except KeyError:
            pass
    return out


def load_field(key):
    """Per-reach learned n_0 and gamma (plot/kan_parameters.nc), indexed by COMID, as cross_arm_fields.py reads them."""
    from netCDF4 import Dataset
    ds = Dataset(RUNS / RUN[key] / "plot" / "kan_parameters.nc")
    c = np.asarray(ds["COMID"][:], dtype=np.int64)
    return pd.DataFrame({"n0": np.asarray(ds["n"][:], dtype=np.float64),
                         "gamma": np.asarray(ds["gamma"][:], dtype=np.float64)}, index=c)


def water_year(t):
    return np.asarray(t.year + (t.month >= 10))


def bands(d, tt):
    """Split d[g, t] into (water-year mean, 30-day MA of the rest, remainder).

    Copied from `bands()` in experiments/levers/blend_mechanism.py on branch multi-product-forcing
    (worktree ~/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4, file untracked there as of
    2026-09-30). One change: d may hold NaN on days without an observation, so the water-year mean is a
    nanmean and the moving average skips NaN (pandas rolling mean counts only finite values toward
    min_periods), and all three bands are NaN on the days d is NaN. With no NaN the result is identical to the
    original.
    """
    wy = water_year(tt)
    low = np.full_like(d, np.nan)
    for y in np.unique(wy):
        s = wy == y
        with np.errstate(invalid="ignore"):
            low[:, s] = np.nanmean(d[:, s], axis=1, keepdims=True)
    rest = d - low
    mid = pd.DataFrame(rest.T).rolling(30, center=True, min_periods=1).mean().to_numpy().T
    ok = np.isfinite(d)
    mid = np.where(ok, mid, np.nan)
    return np.where(ok, low, np.nan), mid, rest - mid


def boot_median(x, B=2000, seed=0, stat=np.nanmedian):
    """Statistic of x with a gauge-bootstrap 95 % percentile CI. Returns (est, lo, hi, n)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    bs = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(x.size)


def fmt_ci(t, nd=3):
    est, lo, hi, n = t
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
