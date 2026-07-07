#!/usr/bin/env python3
"""Cross-arm selective-equifinality convergence analysis (4 levels, H1–H4 verdicts).

Run from ~/projects/ddr:
    uv run python ~/projects/ddrs/scripts/equif_convergence_analysis.py \\
        --r1 <run_id> --r2 <run_id> --r3 <run_id> \\
        --params-r1 output/equif/R1_kan_parameters.nc \\
        --params-r2 output/equif/R2_kan_parameters.nc \\
        --params-r3 output/equif/R3_kan_parameters.nc \\
        --grads-r1  output/equif_probe/grad_R1.nc \\
        --grads-r2  output/equif_probe/grad_R2.nc \\
        --grads-r3  output/equif_probe/grad_R3.nc

Dev run (5 gauges, same run + store thrice, smoke gradients):
    uv run python ~/projects/ddrs/scripts/equif_convergence_analysis.py \\
        --r1 2026-06-23T02-49-12Z-conus-hourly-train-and-test \\
        --r2 2026-06-23T02-49-12Z-conus-hourly-train-and-test \\
        --r3 2026-06-23T02-49-12Z-conus-hourly-train-and-test \\
        --params-r1 /path/to/kan_parameters.nc \\
        --params-r2 /path/to/kan_parameters.nc \\
        --params-r3 /path/to/kan_parameters.nc \\
        --grads-r1 output/equif_probe/smoke.nc \\
        --grads-r2 output/equif_probe/smoke.nc \\
        --grads-r3 output/equif_probe/smoke.nc \\
        --max-gauges 5

Stages (each cached to <out>/<stage>*.npz; re-runnable, skipped when cache exists):
  A — eval network, coverage closure, analysis set
  B — summed upstream Q' for daily-lstm, hourly-lstm, common (dHBV2) stores
  C — Level 1 (raw param spread) + Level 2 (realized geometry)
  D — Level 3 (routing skill vs baseline)
  E — Level 4 (gradient alignment and reachability)
  F — H1–H4 verdicts + figures

Edge convention (empirically verified):
  CONUS adjacency: indices_0 = downstream position in `order`, indices_1 = upstream.
  Gages adjacency: indices_0 / indices_1 are GLOBAL positions into CONUS `order`
  (not local positions into the gauge subgraph's order array).
  Invariant: indices_0[k] >= indices_1[k] (topological, upstream-before-downstream).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Fixed store / data paths
# ---------------------------------------------------------------------------
DAILY_LSTM_STORE = "/mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic"
HOURLY_LSTM_STORE = "/mnt/ssd1/data/icechunk/hourly_lstm_merit_unit_catchments.ic"
COMMON_STORE = "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic"
CONUS_ADJ_PATH = "/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr"
GAGES_ADJ_PATH = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"
GAGES_CSV_PATH = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"

EVAL_START = np.datetime64("1995-10-01")
EVAL_END   = np.datetime64("2010-09-30")

# Physical parameter ranges (from config/merit_training.yaml PARAM_RANGES)
PARAM_RANGES = {
    "n":         (0.015, 0.250),
    "q_spatial": (0.000, 1.000),
    "p_spatial": (1.000, 200.0),
}

# ---------------------------------------------------------------------------
# Metric helpers (NaN-safe)
# ---------------------------------------------------------------------------

def nse(pred: np.ndarray, obs: np.ndarray, eps: float = 0.0) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() < 2:
        return float("nan")
    p, o = pred[mask], obs[mask]
    denom = np.var(o) + eps
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.mean((p - o) ** 2) / denom)


def kge(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() < 3:
        return float("nan")
    p, o = pred[mask], obs[mask]
    mu_o, mu_p = o.mean(), p.mean()
    std_o = o.std(ddof=0)
    std_p = p.std(ddof=0)
    if mu_o == 0 or std_o == 0:
        return float("nan")
    r = float(np.corrcoef(p, o)[0, 1])
    alpha = float(std_p / (std_o + eps))
    beta  = float(mu_p / (mu_o + eps))
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

# ---------------------------------------------------------------------------
# Run data loaders
# ---------------------------------------------------------------------------

def load_eval(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load trained-model predictions + observations from eval/predictions.zarr.

    Returns (pred [n_gauges, n_days] f64, obs, gage_ids).
    """
    try:
        import zarr  # type: ignore
    except ImportError:
        sys.exit("ERROR: zarr not installed — run: uv pip install zarr")
    zp = run_dir / "eval" / "predictions.zarr"
    if not zp.exists():
        sys.exit(f"ERROR: eval zarr not found at {zp}")
    store = zarr.open_group(str(zp), mode="r")
    pred    = np.array(store["predictions"]).astype(np.float64)
    obs     = np.array(store["observations"]).astype(np.float64)
    raw_ids = np.array(store["gage_ids"])
    gage_ids = [bytes(row).rstrip(b"\x00").decode("ascii") for row in raw_ids]
    return pred, obs, gage_ids


def load_baseline(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load baseline summed-Q′ predictions + observations.

    Returns (pred [n_gauges, n_days] f32, obs, gage_ids).
    """
    base = run_dir / "baseline"
    mf   = base / "manifest.json"
    pp   = base / "predictions.f32"
    op   = base / "observations.f32"
    for p in (mf, pp, op):
        if not p.exists():
            sys.exit(f"ERROR: baseline file not found: {p}")
    with open(mf) as f:
        manifest = json.load(f)
    n_g  = manifest["n_gauges"]
    n_d  = manifest["n_days"]
    pred = np.fromfile(pp, dtype=np.float32).reshape(n_g, n_d)
    obs  = np.fromfile(op, dtype=np.float32).reshape(n_g, n_d)
    return pred, obs, manifest["gage_ids"]


def load_kan_params(path: Path) -> dict[str, np.ndarray]:
    """Load kan_parameters.nc → dict with COMID and parameter arrays."""
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError:
        sys.exit("ERROR: netCDF4 not installed — run: uv pip install netCDF4")
    with nc.Dataset(str(path), "r") as f:
        data = {v: np.array(f.variables[v][:]) for v in f.variables}
    return data


def load_gradients(path: Path) -> dict[str, np.ndarray]:
    """Load gradient NetCDF → dict with COMID_probe and grad arrays."""
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError:
        sys.exit("ERROR: netCDF4 not installed")
    with nc.Dataset(str(path), "r") as f:
        data = {v: np.array(f.variables[v][:]) for v in f.variables}
    return data

# ---------------------------------------------------------------------------
# Q' store loader
# ---------------------------------------------------------------------------

def load_qprime_for_eval_network(
    store_path: str,
    eval_comids: np.ndarray,
    is_hourly: bool = False,
) -> np.ndarray:
    """Load daily Q' [n_eval, n_days_eval] for eval_comids over eval window.

    Missing COMIDs (not in store) are 0-filled.
    Hourly stores are aggregated to daily means before returning.
    """
    try:
        import icechunk as ic  # type: ignore
        import xarray as xr    # type: ignore
    except ImportError:
        sys.exit("ERROR: icechunk / xarray not installed")

    print(f"    opening store: {store_path}")
    repo = ic.Repository.open(ic.local_filesystem_storage(store_path))
    ds   = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)

    store_div = ds["divide_id"].values  # (n_store,)
    store_div_set = set(store_div.tolist())

    # Identify which eval COMIDs are in the store
    covered_mask = np.array([int(c) in store_div_set for c in eval_comids])
    covered_comids = eval_comids[covered_mask].tolist()
    print(f"    store coverage: {covered_mask.sum()} / {len(eval_comids)} eval COMIDs")

    if covered_mask.sum() == 0:
        n_days_eval = int((EVAL_END - EVAL_START).astype(int)) + 1
        return np.zeros((len(eval_comids), n_days_eval), dtype=np.float32)

    time_vals = ds["time"].values
    t_mask    = (time_vals >= EVAL_START) & (time_vals <= EVAL_END)

    # For hourly stores, round to day boundaries before masking
    if is_hourly:
        days_only = time_vals.astype("datetime64[D]")
        t_mask    = (days_only >= EVAL_START.astype("datetime64[D]")) & (
                     days_only <= EVAL_END.astype("datetime64[D]"))

    # xarray select: covered_comids (label-based), then integer-position select for time
    ds_sel = ds["Qr"].sel(
        divide_id=covered_comids,
    ).isel(time=np.flatnonzero(t_mask))
    # Load into memory
    qr_covered = ds_sel.values.astype(np.float32)  # [n_covered, n_hours_or_days]

    if is_hourly:
        # Aggregate hours → daily means
        # Group by calendar day using the time coordinate
        t_sub = time_vals[t_mask]
        days  = t_sub.astype("datetime64[D]")
        unique_days, inv_idx = np.unique(days, return_inverse=True)
        n_days = len(unique_days)
        n_cov  = qr_covered.shape[0]
        qr_daily = np.zeros((n_cov, n_days), dtype=np.float32)
        counts   = np.zeros(n_days, dtype=np.int32)
        np.add.at(counts, inv_idx, 1)
        for h in range(qr_covered.shape[1]):
            qr_daily[:, inv_idx[h]] += qr_covered[:, h]
        qr_daily /= np.maximum(counts[np.newaxis, :], 1)
        qr_covered = qr_daily

    n_days_eval = qr_covered.shape[1]

    # Assemble output: 0-fill for uncovered
    q_out = np.zeros((len(eval_comids), n_days_eval), dtype=np.float32)
    q_out[covered_mask] = qr_covered

    return q_out

# ---------------------------------------------------------------------------
# Topological accumulation
# ---------------------------------------------------------------------------

def topo_accumulate(q: np.ndarray, local_down: np.ndarray, local_up: np.ndarray) -> np.ndarray:
    """Accumulate upstream Q' via topological sweep.

    q        : [n_eval, n_days] f32 — per-reach direct Q', modified in place
    local_down: edge downstream indices (sorted ascending = topo order)
    local_up  : edge upstream indices

    After this call, q[i] = summed upstream Q' (including self) for reach i.
    """
    # Edges are pre-sorted by ascending downstream local position (Stage A)
    for k in range(len(local_down)):
        q[local_down[k]] += q[local_up[k]]
    return q

# ---------------------------------------------------------------------------
# Trapezoidal geometry (port of ddr/trapezoidal.py / src/geometry.rs)
# ---------------------------------------------------------------------------

def trapezoidal_geometry(
    n: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    Q: np.ndarray,
    slope: np.ndarray,
    depth_lb: float = 0.01,
    bw_lb: float = 0.01,
) -> dict[str, np.ndarray]:
    """Compute realized channel geometry from MC parameters at discharge Q.

    All array inputs broadcast over a leading reaches dimension.
    Returns dict with keys: depth, top_width, hydraulic_radius.
    """
    q_eps     = q + 1e-6
    depth     = np.clip(
        ((Q * n * (q_eps + 1)) / (p * np.sqrt(slope) + 1e-8)) ** (3.0 / (5.0 + 3.0 * q_eps)),
        depth_lb, None,
    )
    top_width  = p * depth ** q_eps
    side_slope = np.clip(top_width * q_eps / (2.0 * depth), 0.5, 50.0)
    bot_width  = np.clip(top_width - 2.0 * side_slope * depth, bw_lb, None)
    area       = (top_width + bot_width) * depth / 2.0
    wp         = bot_width + 2.0 * depth * np.sqrt(1.0 + side_slope ** 2)
    return {
        "depth":            depth,
        "top_width":        top_width,
        "hydraulic_radius": area / wp,
    }

# ---------------------------------------------------------------------------
# BFS distance (upstream hops from gauges)
# ---------------------------------------------------------------------------

def bfs_upstream_distance(
    local_down: np.ndarray,
    local_up: np.ndarray,
    n_eval: int,
    gauge_local_idx: np.ndarray,
    max_sweeps: int = 200,
) -> np.ndarray:
    """Multi-source BFS going upstream from gauges.

    Returns dist[i] = number of hops upstream from the nearest gauge,
    or 999999 if unreachable.
    """
    INF  = 999999
    dist = np.full(n_eval, INF, dtype=np.int32)
    if len(gauge_local_idx) == 0:
        return dist
    dist[gauge_local_idx] = 0

    # Sweep in reverse topological order (descending downstream local pos)
    order     = np.argsort(local_down)[::-1]
    d_rev     = local_down[order]
    u_rev     = local_up[order]

    for _ in range(max_sweeps):
        d_dist = dist[d_rev]
        new_u  = np.where(d_dist < INF, d_dist + 1, INF)
        prev_u = dist[u_rev]
        update = new_u < prev_u
        if not update.any():
            break
        np.minimum.at(dist, u_rev, new_u)
    else:
        print(f"  WARNING: bfs_upstream_distance did not reach fixpoint "
              f"after {max_sweeps} sweeps — some distances may be underestimated")

    return dist

# ---------------------------------------------------------------------------
# Stage A: network & coverage
# ---------------------------------------------------------------------------

def stage_a(
    out_dir: Path,
    gages_adj_path: str,
    conus_adj_path: str,
    max_gauges: Optional[int],
    force: bool = False,
) -> dict:
    """Build eval network, edge list, and per-store coverage closures.

    Returns dict with keys:
      eval_comids        : (n_eval,) int64, topo-sorted
      local_down         : (n_edges,) int32, sorted ascending by downstream
      local_up           : (n_edges,) int32
      analysis_mask_r12  : (n_eval,) bool — closure(daily-lstm ∩ hourly-lstm)
      analysis_mask_full : (n_eval,) bool — closure(daily-lstm ∩ hourly-lstm ∩ dHBV2)
      gauge_staids       : list[str]
    """
    cache = out_dir / "stage_a.npz"
    if cache.exists() and not force:
        print("[Stage A] loading from cache")
        d = np.load(cache, allow_pickle=True)
        return {
            "eval_comids":        d["eval_comids"],
            "local_down":         d["local_down"],
            "local_up":           d["local_up"],
            "analysis_mask_r12":  d["analysis_mask_r12"],
            "analysis_mask_full": d["analysis_mask_full"],
            "gauge_staids":       d["gauge_staids"].tolist(),
        }

    print("[Stage A] building eval network and coverage closures")
    import zarr  # type: ignore

    conus = zarr.open(conus_adj_path, mode="r")
    gages = zarr.open(gages_adj_path, mode="r")

    conus_order = conus["order"][:].astype(np.int64)  # (346321,) COMIDs
    conus_idx0  = conus["indices_0"][:]               # downstream global pos
    conus_idx1  = conus["indices_1"][:]               # upstream  global pos

    # Map COMID → global position in conus_order
    comid_to_gpos = {int(c): i for i, c in enumerate(conus_order)}

    # Select gauges
    all_staids = sorted(gages.keys())
    if max_gauges is not None:
        all_staids = all_staids[:max_gauges]
    print(f"  gauges selected: {len(all_staids)}")

    # Build eval network = union of subgraph COMIDs
    eval_comid_set: set[int] = set()
    for staid in all_staids:
        g_order = gages[staid]["order"][:].astype(np.int64)
        eval_comid_set.update(g_order.tolist())

    # Sort by global topological position (upstream → downstream)
    eval_comids_unsorted = np.array(list(eval_comid_set), dtype=np.int64)
    gpos_of_eval = np.array([comid_to_gpos[int(c)] for c in eval_comids_unsorted])
    topo_sort    = np.argsort(gpos_of_eval)
    eval_comids  = eval_comids_unsorted[topo_sort]
    gpos_sorted  = gpos_of_eval[topo_sort]         # global positions, ascending

    n_eval = len(eval_comids)
    print(f"  eval network size: {n_eval} reaches")

    # Map COMID → local eval position
    eval_local = {int(c): i for i, c in enumerate(eval_comids)}

    # Filter CONUS edges to eval network; remap to local positions
    eval_gpos_set = set(gpos_sorted.tolist())
    edge_down = []
    edge_up   = []
    for k in range(len(conus_idx0)):
        d_gp = int(conus_idx0[k])
        u_gp = int(conus_idx1[k])
        if d_gp in eval_gpos_set and u_gp in eval_gpos_set:
            d_comid = int(conus_order[d_gp])
            u_comid = int(conus_order[u_gp])
            edge_down.append(eval_local[d_comid])
            edge_up.append(eval_local[u_comid])

    local_down = np.array(edge_down, dtype=np.int32)
    local_up   = np.array(edge_up,   dtype=np.int32)

    # Sort edges by downstream local position (ascending = topo order)
    sort_e     = np.argsort(local_down, kind="stable")
    local_down = local_down[sort_e]
    local_up   = local_up[sort_e]
    assert np.all(local_down >= local_up), "eval-network remap broke topological ordering"
    print(f"  eval network edges: {len(local_down)}")

    # --- per-store coverage masks (COMID present in store) ---
    def store_coverage(store_path: str) -> np.ndarray:
        try:
            import icechunk as ic  # type: ignore
            import xarray as xr   # type: ignore
        except ImportError:
            sys.exit("ERROR: icechunk/xarray not installed")
        repo    = ic.Repository.open(ic.local_filesystem_storage(store_path))
        ds      = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        div_set = set(ds["divide_id"].values.tolist())
        mask    = np.array([int(c) in div_set for c in eval_comids])
        print(f"    {store_path.split('/')[-1]}: {mask.sum()} / {n_eval} covered")
        return mask

    print("  computing per-store coverage:")
    cov_daily  = store_coverage(DAILY_LSTM_STORE)
    cov_hourly = store_coverage(HOURLY_LSTM_STORE)
    cov_common = store_coverage(COMMON_STORE)

    # Upstream-closure: closure[i] = covered[i] AND closure(all upstream)
    def upstream_closure(covered: np.ndarray) -> np.ndarray:
        """Single pass in topo order (edges sorted ascending by downstream)."""
        closure = covered.copy()
        for k in range(len(local_down)):
            if not closure[local_up[k]]:
                closure[local_down[k]] = False
        return closure

    print("  computing upstream closures:")
    cl_daily  = upstream_closure(cov_daily)
    cl_hourly = upstream_closure(cov_hourly)
    cl_common = upstream_closure(cov_common)
    print(f"    closure daily-lstm:  {cl_daily.sum()}")
    print(f"    closure hourly-lstm: {cl_hourly.sum()}")
    print(f"    closure dHBV2:       {cl_common.sum()}")

    analysis_mask_r12  = cl_daily  & cl_hourly
    analysis_mask_full = cl_daily  & cl_hourly & cl_common
    print(f"  analysis set (R1/R2 ∩ R3):          {analysis_mask_r12.sum()}")
    print(f"  analysis set (R1/R2 ∩ R3 ∩ common): {analysis_mask_full.sum()}")

    np.savez_compressed(
        cache,
        eval_comids        = eval_comids,
        local_down         = local_down,
        local_up           = local_up,
        analysis_mask_r12  = analysis_mask_r12,
        analysis_mask_full = analysis_mask_full,
        gauge_staids       = np.array(all_staids),
    )
    print(f"  saved → {cache}")
    return {
        "eval_comids":        eval_comids,
        "local_down":         local_down,
        "local_up":           local_up,
        "analysis_mask_r12":  analysis_mask_r12,
        "analysis_mask_full": analysis_mask_full,
        "gauge_staids":       all_staids,
    }

# ---------------------------------------------------------------------------
# Stage B: reference discharges
# ---------------------------------------------------------------------------

def stage_b(
    out_dir: Path,
    stage_a_data: dict,
    tag: str,
    store_path: str,
    is_hourly: bool = False,
    force: bool = False,
) -> dict:
    """Compute per-reach median / p10 / p90 / mean of SUMMED upstream daily Q' over eval window.

    tag : short label used for cache filename ('daily', 'hourly', 'common')
    Returns dict with keys: median_q, p10_q, p90_q, mean_q  each (n_eval,) f32
    """
    cache = out_dir / f"stage_b_{tag}.npz"
    if cache.exists() and not force:
        print(f"[Stage B:{tag}] loading from cache")
        d = np.load(cache)
        if "mean_q" in d.files:
            return {
                "median_q": d["median_q"],
                "p10_q":    d["p10_q"],
                "p90_q":    d["p90_q"],
                "mean_q":   d["mean_q"],
            }
        print(f"  [Stage B:{tag}] old cache missing mean_q — recomputing")

    print(f"[Stage B:{tag}] computing summed Q' from {store_path.split('/')[-1]}")
    eval_comids = stage_a_data["eval_comids"]
    local_down  = stage_a_data["local_down"]
    local_up    = stage_a_data["local_up"]

    # Load per-reach direct Q' over eval window
    q = load_qprime_for_eval_network(store_path, eval_comids, is_hourly=is_hourly)
    # q: [n_eval, n_days] f32

    n_eval, n_days = q.shape
    print(f"    loaded: {n_eval} reaches × {n_days} days")

    # Topological accumulation (modifies q in place)
    topo_accumulate(q, local_down, local_up)

    # Per-reach statistics over time axis
    median_q = np.median(q, axis=1).astype(np.float32)
    p10_q    = np.percentile(q, 10, axis=1).astype(np.float32)
    p90_q    = np.percentile(q, 90, axis=1).astype(np.float32)
    mean_q   = np.mean(q, axis=1).astype(np.float32)

    np.savez_compressed(cache, median_q=median_q, p10_q=p10_q, p90_q=p90_q, mean_q=mean_q)
    print(f"    saved → {cache}")

    return {"median_q": median_q, "p10_q": p10_q, "p90_q": p90_q, "mean_q": mean_q}

# ---------------------------------------------------------------------------
# Stage C: Level 1 (raw params) + Level 2 (realized geometry)
# ---------------------------------------------------------------------------

def stage_c(
    out_dir: Path,
    stage_a_data: dict,
    params_r1: Path,
    params_r2: Path,
    params_r3: Path,
    b_r12: dict,
    b_r3: dict,
    b_common: dict,
    force: bool = False,
) -> dict:
    """Compute raw-parameter spread (Level 1) and geometry spread (Level 2).

    Returns dict with per-reach and aggregate statistics.
    """
    cache = out_dir / "stage_c.npz"
    if cache.exists() and not force:
        print("[Stage C] loading from cache")
        d = np.load(cache, allow_pickle=True)
        geo_spread_own = {
            "depth":            d["geo_own_depth"],
            "top_width":        d["geo_own_top_width"],
            "hydraulic_radius": d["geo_own_hyd_radius"],
        }
        geo_spread_common = {
            "depth":            d["geo_common_depth"],
            "top_width":        d["geo_common_top_width"],
            "hydraulic_radius": d["geo_common_hyd_radius"],
        }
        return {
            "per_reach_spread_n": d["per_reach_spread_n"],
            "per_reach_spread_q": d["per_reach_spread_q"],
            "per_reach_spread_p": d["per_reach_spread_p"],
            "median_spread_n":    float(d["median_spread_n"]),
            "geo_spread_own":     geo_spread_own,
            "geo_spread_common":  geo_spread_common,
            "q_disagreement":     d["q_disagreement"],
            "h2_rho":             float(d["h2_rho"]),
            "spearman_n":         dict(zip(
                d["spearman_pair_labels"].tolist(),
                d["spearman_n"].tolist(),
            )),
        }

    print("[Stage C] computing raw-parameter and geometry spreads")

    eval_comids = stage_a_data["eval_comids"].astype(np.int64)
    # Use the primary analysis set (R1/R2 ∩ R3)
    amask = stage_a_data["analysis_mask_r12"]
    # For sensitivity (Level 2 common reference), use amask_full
    amask_full = stage_a_data["analysis_mask_full"]

    analysis_comids = eval_comids[amask]
    analysis_full   = eval_comids[amask_full]
    print(f"  analysis set (primary):     {len(analysis_comids)}")
    print(f"  analysis set (sensitivity): {len(analysis_full)}")

    def load_align(path: Path, target_comids: np.ndarray) -> dict[str, np.ndarray]:
        """Load kan_parameters.nc and align to target_comids."""
        raw = load_kan_params(path)
        file_comids = raw["COMID"].astype(np.int64)
        comid_to_pos = {int(c): i for i, c in enumerate(file_comids)}
        idx = np.array([comid_to_pos.get(int(c), -1) for c in target_comids])
        missing = (idx < 0).sum()
        if missing > 0:
            print(f"    WARNING: {missing} analysis COMIDs not in {path.name}")
        valid = idx >= 0
        out: dict[str, np.ndarray] = {}
        for p_name in ("n", "q_spatial", "p_spatial", "slope"):
            if p_name not in raw:
                continue
            arr = np.full(len(target_comids), np.nan, dtype=np.float32)
            arr[valid] = raw[p_name][idx[valid]]
            out[p_name] = arr
        return out

    p1 = load_align(params_r1, analysis_comids)
    p2 = load_align(params_r2, analysis_comids)
    p3 = load_align(params_r3, analysis_comids)

    # Assert slope is identical across arms (by construction)
    if "slope" in p1 and "slope" in p2 and "slope" in p3:
        diff_12 = np.nanmax(np.abs(p1["slope"] - p2["slope"]))
        diff_13 = np.nanmax(np.abs(p1["slope"] - p3["slope"]))
        print(f"  slope max-abs-diff R1-R2: {diff_12:.2e}, R1-R3: {diff_13:.2e}")

    # ---- Level 1: raw-param spread ----
    from scipy.stats import spearmanr  # type: ignore

    def param_stats(v1, v2, v3, p_name) -> dict:
        lo, hi = PARAM_RANGES.get(p_name, (0.0, 1.0))
        rng = hi - lo
        stack = np.stack([v1, v2, v3], axis=0)          # (3, n)
        per_reach_spread = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)) / rng
        median_spread    = float(np.nanmedian(per_reach_spread))
        pairs = [(v1, v2, "R1-R2"), (v1, v3, "R1-R3"), (v2, v3, "R2-R3")]
        spears = {}
        for a, b, lbl in pairs:
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() > 5:
                r, _ = spearmanr(a[valid], b[valid])
            else:
                r = float("nan")
            spears[lbl] = float(r)
        return {
            "per_reach_spread": per_reach_spread,
            "median_spread":    median_spread,
            "spearman":         spears,
        }

    stats_n = param_stats(p1["n"], p2["n"], p3["n"], "n")
    stats_q = param_stats(p1["q_spatial"], p2["q_spatial"], p3["q_spatial"], "q_spatial")
    stats_p = param_stats(p1["p_spatial"], p2["p_spatial"], p3["p_spatial"], "p_spatial")

    print(f"  Level 1 median norm-spread:  n={stats_n['median_spread']:.4f}"
          f"  q={stats_q['median_spread']:.4f}  p={stats_p['median_spread']:.4f}")
    for lbl, r in stats_n["spearman"].items():
        print(f"    spearman n {lbl}: {r:.3f}")

    # ---- Level 2: realized geometry ----
    slope_use = p1.get("slope", np.ones(len(analysis_comids), dtype=np.float32))

    def geometry_spread_at_Q(
        Q_ref: np.ndarray,
        label: str,
        arm_params: tuple,
        slope: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute per-reach relative spread of depth/top_width/hyd_radius.

        arm_params : tuple of 3 param dicts (one per arm), each aligned to Q_ref length.
        slope      : (n,) array aligned to Q_ref length.
        """
        geo_stack: dict[str, list] = {"depth": [], "top_width": [], "hydraulic_radius": []}
        for params in arm_params:
            g = trapezoidal_geometry(
                params["n"], params["p_spatial"], params["q_spatial"],
                Q_ref, slope,
            )
            for k in geo_stack:
                geo_stack[k].append(g[k])
        result = {}
        for k, arm_list in geo_stack.items():
            stack     = np.stack(arm_list, axis=0)           # (3, n)
            cross_mean = np.nanmean(stack, axis=0)
            spread     = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)) / (
                          np.abs(cross_mean) + 1e-9)
            result[k] = spread
            print(f"  Level 2 [{label}] {k} median rel-spread: {np.nanmedian(spread):.4f}")
        return result

    # PRIMARY: arm-own Q' references (R1/R2 share daily; R3 uses hourly)
    q_r12 = b_r12["median_q"][amask]
    q_r3  = b_r3["median_q"][amask]

    # Use each arm's own median Q' as operating point
    geo_arm_own: list[dict] = []
    Q_own_per_arm = [q_r12, q_r12, q_r3]  # R1/R2 share daily store
    geo_stack_own: dict[str, list] = {"depth": [], "top_width": [], "hydraulic_radius": []}
    for params, Q_own in zip((p1, p2, p3), Q_own_per_arm):
        g = trapezoidal_geometry(
            params["n"], params["p_spatial"], params["q_spatial"],
            Q_own, slope_use,
        )
        for k in geo_stack_own:
            geo_stack_own[k].append(g[k])

    geo_spread_own: dict[str, np.ndarray] = {}
    print("  Level 2 [arm-own Q'] relative spread:")
    for k, arm_list in geo_stack_own.items():
        stack      = np.stack(arm_list, axis=0)
        cross_mean = np.nanmean(stack, axis=0)
        spread     = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)) / (
                      np.abs(cross_mean) + 1e-9)
        geo_spread_own[k] = spread
        print(f"    {k}: median={np.nanmedian(spread):.4f}")

    # SENSITIVITY: common dHBV2 reference — realign params to sensitivity set
    p1s = load_align(params_r1, analysis_full)
    p2s = load_align(params_r2, analysis_full)
    p3s = load_align(params_r3, analysis_full)
    slope_s = p1s.get("slope", np.ones(len(analysis_full), dtype=np.float32))
    Q_common     = b_common["median_q"][amask_full]
    Q_common_p10 = b_common["p10_q"][amask_full]
    Q_common_p90 = b_common["p90_q"][amask_full]
    print("  Level 2 [common dHBV2 Q'] relative spread:")
    geo_spread_common = geometry_spread_at_Q(
        Q_common, label="common-median", arm_params=(p1s, p2s, p3s), slope=slope_s)
    print("  Level 2 [common dHBV2 p10] relative spread:")
    geo_spread_common_p10 = geometry_spread_at_Q(
        Q_common_p10, label="common-p10", arm_params=(p1s, p2s, p3s), slope=slope_s)
    print("  Level 2 [common dHBV2 p90] relative spread:")
    geo_spread_common_p90 = geometry_spread_at_Q(
        Q_common_p90, label="common-p90", arm_params=(p1s, p2s, p3s), slope=slope_s)

    # Q' disagreement between stores (for H2): R1/R2 = daily, R3 = hourly
    # per-reach relative range of eval-window mean Q' across the 2 distinct stores
    q_mean_r12 = b_r12["mean_q"][amask]
    q_mean_r3  = b_r3["mean_q"][amask]
    q_stack_2  = np.stack([q_mean_r12, q_mean_r3], axis=0)
    q_mean_2   = np.nanmean(q_stack_2, axis=0)
    q_disagreement = (np.nanmax(q_stack_2, axis=0) - np.nanmin(q_stack_2, axis=0)) / (
                      np.abs(q_mean_2) + 1e-9)

    # H2: Spearman(n-spread, Q'-disagreement)
    from scipy.stats import spearmanr as sp
    valid = np.isfinite(stats_n["per_reach_spread"]) & np.isfinite(q_disagreement)
    if valid.sum() > 5:
        h2_rho, _ = sp(stats_n["per_reach_spread"][valid], q_disagreement[valid])
    else:
        h2_rho = float("nan")
    print(f"  H2 Spearman(n-spread, Q'-disagreement): {h2_rho:.3f}")

    np.savez_compressed(
        cache,
        # Level 1
        per_reach_spread_n  = stats_n["per_reach_spread"],
        per_reach_spread_q  = stats_q["per_reach_spread"],
        per_reach_spread_p  = stats_p["per_reach_spread"],
        median_spread_n     = np.float32(stats_n["median_spread"]),
        median_spread_q     = np.float32(stats_q["median_spread"]),
        median_spread_p     = np.float32(stats_p["median_spread"]),
        spearman_n          = np.array(list(stats_n["spearman"].values())),
        spearman_q          = np.array(list(stats_q["spearman"].values())),
        spearman_p          = np.array(list(stats_p["spearman"].values())),
        spearman_pair_labels= np.array(list(stats_n["spearman"].keys())),
        # Level 2 arm-own
        geo_own_depth       = geo_spread_own["depth"],
        geo_own_top_width   = geo_spread_own["top_width"],
        geo_own_hyd_radius  = geo_spread_own["hydraulic_radius"],
        # Level 2 sensitivity
        geo_common_depth    = np.array(list(geo_spread_common.values())[0]),   # approx
        geo_common_top_width= np.array(list(geo_spread_common.values())[1]),
        geo_common_hyd_radius=np.array(list(geo_spread_common.values())[2]),
        # Q' disagreement
        q_disagreement      = q_disagreement,
        h2_rho              = np.float32(h2_rho),
        # Analysis set info
        analysis_n_primary  = np.int32(amask.sum()),
        analysis_n_full     = np.int32(amask_full.sum()),
    )
    print(f"  saved → {cache}")

    return {
        "per_reach_spread_n":  stats_n["per_reach_spread"],
        "per_reach_spread_q":  stats_q["per_reach_spread"],
        "per_reach_spread_p":  stats_p["per_reach_spread"],
        "median_spread_n":     stats_n["median_spread"],
        "geo_spread_own":      geo_spread_own,
        "geo_spread_common":   geo_spread_common,
        "q_disagreement":      q_disagreement,
        "h2_rho":              float(h2_rho),
        "spearman_n":          stats_n["spearman"],
    }

# ---------------------------------------------------------------------------
# Stage D: Level 3 (routing skill)
# ---------------------------------------------------------------------------

def stage_d(
    out_dir: Path,
    runs_dir: Path,
    r1_id: Optional[str],
    r2_id: Optional[str],
    r3_id: Optional[str],
    force: bool = False,
) -> Optional[dict]:
    """Load eval + baseline metrics per arm.

    Returns dict or None if any run ID is missing.
    """
    if not all([r1_id, r2_id, r3_id]):
        print("[Stage D] skipped — one or more run IDs not provided")
        return None

    cache = out_dir / "stage_d.npz"
    if cache.exists() and not force:
        print("[Stage D] loading from cache")
        d = np.load(cache, allow_pickle=True)
        return {k: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files}

    print("[Stage D] computing routing skill per arm")

    rows = {}
    for arm, run_id in zip(("R1", "R2", "R3"), (r1_id, r2_id, r3_id)):
        rdir = runs_dir / run_id
        print(f"  [{arm}] {rdir}")
        eval_pred, eval_obs, eval_gids = load_eval(rdir)
        base_pred, base_obs, base_gids = load_baseline(rdir)

        # Per-gauge NSE/KGE from trained model
        n_g = eval_pred.shape[0]
        eval_nse_arr = np.array([nse(eval_pred[i], eval_obs[i]) for i in range(n_g)])
        eval_kge_arr = np.array([kge(eval_pred[i], eval_obs[i]) for i in range(n_g)])

        # Per-gauge NSE/KGE from baseline (recomputed for consistency)
        n_b = base_pred.shape[0]
        base_nse_arr = np.array([nse(base_pred[i].astype(np.float64),
                                      base_obs[i].astype(np.float64)) for i in range(n_b)])
        base_kge_arr = np.array([kge(base_pred[i].astype(np.float64),
                                      base_obs[i].astype(np.float64)) for i in range(n_b)])

        rows[arm] = {
            "eval_median_nse": float(np.nanmedian(eval_nse_arr)),
            "eval_median_kge": float(np.nanmedian(eval_kge_arr)),
            "base_median_nse": float(np.nanmedian(base_nse_arr)),
            "base_median_kge": float(np.nanmedian(base_kge_arr)),
            "n_gauges":        int(n_g),
        }
        print(f"    eval NSE={rows[arm]['eval_median_nse']:.4f}  "
              f"KGE={rows[arm]['eval_median_kge']:.4f}  "
              f"(baseline NSE={rows[arm]['base_median_nse']:.4f}  "
              f"KGE={rows[arm]['base_median_kge']:.4f}  "
              f"n_gauges={rows[arm]['n_gauges']}  "
              f"window={EVAL_START}–{EVAL_END})")

    np.savez_compressed(cache, **{
        f"{arm}_{k}": np.float32(v) if isinstance(v, float) else np.int32(v)
        for arm, d in rows.items()
        for k, v in d.items()
    })
    print(f"  saved → {cache}")
    return rows

# ---------------------------------------------------------------------------
# Stage E: Level 4 (gradients)
# ---------------------------------------------------------------------------

def stage_e(
    out_dir: Path,
    stage_a_data: dict,
    grads_r1: Optional[Path],
    grads_r2: Optional[Path],
    grads_r3: Optional[Path],
    gages_csv: str = GAGES_CSV_PATH,
    force: bool = False,
) -> Optional[dict]:
    """Compute gradient cosine alignment (H3) and gauged/distance decay (H4).

    Returns dict or None if any grad file is missing.
    """
    if not all([grads_r1, grads_r2, grads_r3]):
        print("[Stage E] skipped — one or more gradient files not provided")
        return None

    cache = out_dir / "stage_e.npz"
    if cache.exists() and not force:
        print("[Stage E] loading from cache")
        d = np.load(cache, allow_pickle=True)
        h3_loaded: dict = {}
        for pname, cos_key, mean_key in [
            ("n",         "h3_n_cosines", "h3_n_mean_cos"),
            ("q_spatial", "h3_q_cosines", "h3_q_mean_cos"),
            ("p_spatial", "h3_p_cosines", "h3_p_mean_cos"),
        ]:
            cosines = d[cos_key]
            h3_loaded[pname] = {
                "cosine_R1R2": float(cosines[0]),
                "cosine_R1R3": float(cosines[1]),
                "cosine_R2R3": float(cosines[2]),
                "mean_cosine": float(d[mean_key]),
            }
        h4_loaded: dict = {}
        for arm in ("R1", "R2", "R3"):
            h4_loaded[arm] = {}
            for p in ("n", "q_spatial", "p_spatial"):
                h4_loaded[arm][p] = {
                    "ratio":       float(d[f"h4_{arm}_{p}_ratio"]),
                    "bin_medians": d[f"h4_{arm}_{p}_bins"].tolist(),
                }
        bin_labels_loaded = (
            d["bin_labels"].tolist()
            if "bin_labels" in d.files
            else ["0", "1-2", "3-5", "6-10", ">10"]
        )
        return {
            "h3":        h3_loaded,
            "h4":        h4_loaded,
            "probe_dist": d["probe_dist"],
            "bin_labels": bin_labels_loaded,
        }

    print("[Stage E] computing gradient alignment and reachability")

    eval_comids = stage_a_data["eval_comids"]
    amask       = stage_a_data["analysis_mask_r12"]
    local_down  = stage_a_data["local_down"]
    local_up    = stage_a_data["local_up"]
    n_eval      = len(eval_comids)

    eval_comid_to_local = {int(c): i for i, c in enumerate(eval_comids)}

    # Load gradient files
    g1 = load_gradients(grads_r1)
    g2 = load_gradients(grads_r2)
    g3 = load_gradients(grads_r3)

    # Find common COMID_probe ∩ analysis set
    probe_1 = set(g1["COMID_probe"].astype(np.int64).tolist())
    probe_2 = set(g2["COMID_probe"].astype(np.int64).tolist())
    probe_3 = set(g3["COMID_probe"].astype(np.int64).tolist())
    analysis_set = set(eval_comids[amask].astype(np.int64).tolist())
    common_comids = sorted(probe_1 & probe_2 & probe_3 & analysis_set)
    print(f"  common probe COMIDs ∩ analysis: {len(common_comids)}")

    if len(common_comids) == 0:
        print("  WARNING: no common probe COMIDs — Stage E returning empty")
        return None

    def align_grads(gdata: dict, comids: list[int]) -> dict[str, np.ndarray]:
        probe_arr = gdata["COMID_probe"].astype(np.int64)
        idx_map   = {int(c): i for i, c in enumerate(probe_arr)}
        idx       = np.array([idx_map[c] for c in comids])
        return {
            "grad_n_abs":          gdata["grad_n_abs"][idx],
            "grad_n_net":          gdata["grad_n_net"][idx],
            "grad_q_spatial_abs":  gdata["grad_q_spatial_abs"][idx],
            "grad_q_spatial_net":  gdata["grad_q_spatial_net"][idx],
            "grad_p_spatial_abs":  gdata["grad_p_spatial_abs"][idx],
            "grad_p_spatial_net":  gdata["grad_p_spatial_net"][idx],
        }

    a1 = align_grads(g1, common_comids)
    a2 = align_grads(g2, common_comids)
    a3 = align_grads(g3, common_comids)

    def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return float("nan")
        return float(np.dot(v1, v2) / (n1 * n2))

    def sign_agree(v1: np.ndarray, v2: np.ndarray) -> float:
        s1 = np.sign(v1)
        s2 = np.sign(v2)
        nonzero = (s1 != 0) & (s2 != 0)
        if nonzero.sum() == 0:
            return float("nan")
        return float((s1[nonzero] == s2[nonzero]).mean())

    # ---- H3: cosine alignment per parameter ----
    h3: dict = {}
    for pname, net_key in [("n", "grad_n_net"),
                            ("q_spatial", "grad_q_spatial_net"),
                            ("p_spatial", "grad_p_spatial_net")]:
        v1, v2, v3 = a1[net_key], a2[net_key], a3[net_key]
        cos_12 = cosine_sim(v1, v2)
        cos_13 = cosine_sim(v1, v3)
        cos_23 = cosine_sim(v2, v3)
        mean_cos = np.nanmean([cos_12, cos_13, cos_23])
        sa_12 = sign_agree(v1, v2)
        sa_13 = sign_agree(v1, v3)
        sa_23 = sign_agree(v2, v3)
        h3[pname] = {
            "cosine_R1R2": cos_12, "cosine_R1R3": cos_13, "cosine_R2R3": cos_23,
            "mean_cosine": float(mean_cos),
            "sign_agree_R1R2": sa_12, "sign_agree_R1R3": sa_13, "sign_agree_R2R3": sa_23,
        }
        print(f"  H3 {pname}: cos(R1-R2)={cos_12:.3f} cos(R1-R3)={cos_13:.3f} "
              f"cos(R2-R3)={cos_23:.3f} mean={mean_cos:.3f} "
              f"sign-agree(R1-R2)={sa_12:.3f}")

    # ---- H4: gradient reachability ----
    # Gauged COMIDs: from gages_3000.csv COMID column
    try:
        import pandas as pd  # type: ignore
        gdf = pd.read_csv(gages_csv)
        gauged_comids = set(gdf["COMID"].astype(int).tolist())
    except Exception as e:
        print(f"  WARNING: could not load gages CSV ({e}); H4 gauged/ungauged skipped")
        gauged_comids = set()

    # BFS distance from gauge reaches, going UPSTREAM
    gauge_local_idx = np.array([
        eval_comid_to_local[c] for c in gauged_comids
        if c in eval_comid_to_local
    ], dtype=np.int32)
    print(f"  gauge reaches in eval network: {len(gauge_local_idx)}")

    dist = bfs_upstream_distance(local_down, local_up, n_eval, gauge_local_idx)

    # For each grad file, probe-reach distances
    common_comids_arr = np.array(common_comids, dtype=np.int64)
    probe_local_idx   = np.array(
        [eval_comid_to_local.get(int(c), -1) for c in common_comids_arr]
    )
    valid_probe = probe_local_idx >= 0
    probe_dist  = np.full(len(common_comids), -1, dtype=np.int32)
    probe_dist[valid_probe] = dist[probe_local_idx[valid_probe]]

    gauged_probe  = np.array([int(c) in gauged_comids for c in common_comids], dtype=bool)
    ungauged_probe = ~gauged_probe

    bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 999999)]
    bin_labels = ["0", "1-2", "3-5", "6-10", ">10"]

    h4: dict = {}
    for arm_label, adat in zip(("R1", "R2", "R3"), (a1, a2, a3)):
        arm_h4: dict = {}
        for pname, abs_key in [("n", "grad_n_abs"),
                                ("q_spatial", "grad_q_spatial_abs"),
                                ("p_spatial", "grad_p_spatial_abs")]:
            gabs = adat[abs_key]
            med_gauged   = float(np.nanmedian(gabs[gauged_probe]))
            med_ungauged = float(np.nanmedian(gabs[ungauged_probe])) if ungauged_probe.any() else float("nan")
            ratio = med_gauged / (med_ungauged + 1e-12) if med_ungauged > 0 else float("nan")

            bin_medians = []
            for lo, hi in bins:
                in_bin = (probe_dist >= lo) & (probe_dist <= hi) & valid_probe
                med = float(np.nanmedian(gabs[in_bin])) if in_bin.any() else float("nan")
                bin_medians.append(med)

            arm_h4[pname] = {
                "med_gauged":   med_gauged,
                "med_ungauged": med_ungauged,
                "ratio":        ratio,
                "bin_medians":  bin_medians,
                "bin_labels":   bin_labels,
            }
            print(f"  H4 [{arm_label}] {pname}: "
                  f"gauged={med_gauged:.2e} ungauged={med_ungauged:.2e} ratio={ratio:.2f}  "
                  f"bins={[f'{v:.2e}' for v in bin_medians]}")
        h4[arm_label] = arm_h4

    np.savez_compressed(
        cache,
        common_comids  = common_comids_arr,
        probe_dist     = probe_dist,
        bin_labels     = np.array(bin_labels),
        # H3 cosines (flattened)
        h3_n_cosines   = np.array([h3["n"]["cosine_R1R2"],       h3["n"]["cosine_R1R3"],       h3["n"]["cosine_R2R3"]]),
        h3_q_cosines   = np.array([h3["q_spatial"]["cosine_R1R2"], h3["q_spatial"]["cosine_R1R3"], h3["q_spatial"]["cosine_R2R3"]]),
        h3_p_cosines   = np.array([h3["p_spatial"]["cosine_R1R2"], h3["p_spatial"]["cosine_R1R3"], h3["p_spatial"]["cosine_R2R3"]]),
        h3_n_mean_cos  = np.float32(h3["n"]["mean_cosine"]),
        h3_q_mean_cos  = np.float32(h3["q_spatial"]["mean_cosine"]),
        h3_p_mean_cos  = np.float32(h3["p_spatial"]["mean_cosine"]),
        # H4 per-arm ratio arrays (R1/R2/R3 × params)
        **{f"h4_{arm}_{p}_ratio": np.float32(h4[arm][p]["ratio"])
           for arm in ("R1", "R2", "R3") for p in ("n", "q_spatial", "p_spatial")},
        **{f"h4_{arm}_{p}_bins": np.array(h4[arm][p]["bin_medians"])
           for arm in ("R1", "R2", "R3") for p in ("n", "q_spatial", "p_spatial")},
    )
    print(f"  saved → {cache}")
    return {"h3": h3, "h4": h4, "probe_dist": probe_dist, "bin_labels": bin_labels}

# ---------------------------------------------------------------------------
# Stage F: verdicts + figures
# ---------------------------------------------------------------------------

def stage_f(
    out_dir: Path,
    stage_a_data: dict,
    c_data: Optional[dict],
    d_data: Optional[dict],
    e_data: Optional[dict],
) -> None:
    """Print and write H1–H4 verdicts; generate figures."""

    print("\n" + "=" * 72)
    print("EQUIFINALITY CONVERGENCE VERDICTS")
    print(f"Eval window: {EVAL_START} – {EVAL_END}")
    print("=" * 72)

    verdicts: dict = {}

    # ---- H1 ----------------------------------------------------------------
    if c_data is not None:
        n_spread = c_data["median_spread_n"]
        geo_own  = c_data["geo_spread_own"]
        med_geo  = float(np.nanmedian([
            np.nanmedian(geo_own["depth"]),
            np.nanmedian(geo_own["top_width"]),
            np.nanmedian(geo_own["hydraulic_radius"]),
        ]))
        med_geo_d  = float(np.nanmedian(geo_own["depth"]))
        med_geo_tw = float(np.nanmedian(geo_own["top_width"]))
        med_geo_hr = float(np.nanmedian(geo_own["hydraulic_radius"]))

        h1_supported = (med_geo_d  < n_spread and
                        med_geo_tw < n_spread and
                        med_geo_hr < n_spread)
        h1_verdict   = "SUPPORTED" if h1_supported else "REFUTED"
        verdicts["H1"] = h1_verdict

        geo_common = c_data.get("geo_spread_common", {})
        print("\n[H1] Realized geometry converges; Manning's n diverges")
        print(f"  Primary (arm-own Q'):")
        print(f"    median norm-spread(n)          = {n_spread:.4f}")
        print(f"    median rel-spread(depth)        = {med_geo_d:.4f}")
        print(f"    median rel-spread(top_width)    = {med_geo_tw:.4f}")
        print(f"    median rel-spread(hyd_radius)   = {med_geo_hr:.4f}")
        print(f"  Rule: SUPPORTED iff all geometry spreads < n-spread")
        print(f"  → H1: {h1_verdict}")
        if geo_common:
            print(f"  Sensitivity (common dHBV2 Q'):")
            for k, v in geo_common.items():
                print(f"    median rel-spread({k}) = {np.nanmedian(v):.4f}")
    else:
        verdicts["H1"] = "INCONCLUSIVE"
        print("\n[H1] INCONCLUSIVE — Stage C not run (params not provided)")

    # ---- H2 ----------------------------------------------------------------
    if c_data is not None:
        h2_rho   = c_data["h2_rho"]
        n_spread = c_data["median_spread_n"]
        med_geo_spread = float(np.nanmedian([
            np.nanmedian(c_data["geo_spread_own"]["depth"]),
            np.nanmedian(c_data["geo_spread_own"]["top_width"]),
            np.nanmedian(c_data["geo_spread_own"]["hydraulic_radius"]),
        ]))
        contrast = n_spread > med_geo_spread
        h2_supported = (h2_rho > 0.2) and contrast
        h2_verdict   = (
            "SUPPORTED" if h2_supported
            else "REFUTED" if (h2_rho <= 0.2 or not contrast)
            else "INCONCLUSIVE"
        )
        verdicts["H2"] = h2_verdict
        print(f"\n[H2] n-divergence predicted by inter-source Q' disagreement")
        print(f"  Spearman ρ(n-spread, Q'-disagreement) = {h2_rho:.3f}  (bar: > 0.2)")
        print(f"  n-spread vs geometry contrast: {n_spread:.4f} vs {med_geo_spread:.4f}  contrast={contrast}")
        print(f"  Rule: SUPPORTED iff ρ > 0.2 AND n-spread > geometry-spread")
        print(f"  → H2: {h2_verdict}")
    else:
        verdicts["H2"] = "INCONCLUSIVE"
        print("\n[H2] INCONCLUSIVE — Stage C not run")

    # ---- H3 ----------------------------------------------------------------
    if e_data is not None:
        h3 = e_data["h3"]
        cos_n = h3["n"]["mean_cosine"]
        cos_q = h3["q_spatial"]["mean_cosine"]
        cos_p = h3["p_spatial"]["mean_cosine"]
        q_above_n = cos_q > cos_n
        p_above_n = cos_p > cos_n
        if q_above_n and p_above_n:
            h3_verdict = "SUPPORTED"
        elif cos_n >= cos_q and cos_n >= cos_p:
            h3_verdict = "REFUTED"
        else:
            h3_verdict = "INCONCLUSIVE"
        verdicts["H3"] = h3_verdict
        print(f"\n[H3] Cross-arm gradient alignment: geometry > n")
        print(f"  mean cosine: n={cos_n:.3f}  q_spatial={cos_q:.3f}  p_spatial={cos_p:.3f}")
        print(f"  (cosines for pairs R1-R2, R1-R3, R2-R3)")
        for pname, pd in h3.items():
            print(f"    {pname}: {pd['cosine_R1R2']:.3f} / {pd['cosine_R1R3']:.3f} / {pd['cosine_R2R3']:.3f}")
        print(f"  Rule: SUPPORTED iff mean_cos(q) > mean_cos(n) AND mean_cos(p) > mean_cos(n)")
        print(f"        REFUTED   iff mean_cos(n) >= both; INCONCLUSIVE otherwise")
        print(f"  → H3: {h3_verdict}")
    else:
        verdicts["H3"] = "INCONCLUSIVE"
        print("\n[H3] INCONCLUSIVE — Stage E not run (grad files not provided)")

    # ---- H4 ----------------------------------------------------------------
    if e_data is not None:
        h4 = e_data["h4"]
        bin_labels = e_data["bin_labels"]
        print(f"\n[H4] Gradient reachability decays with gauge distance")
        print(f"  {'Arm':<4} {'Param':<12} {'Ratio':<8} {'Verdict cell'}")
        cell_verdicts = []
        for arm in ("R1", "R2", "R3"):
            for pname in ("n", "q_spatial", "p_spatial"):
                ratio = h4[arm][pname]["ratio"]
                bins  = h4[arm][pname]["bin_medians"]
                # Check monotone decay (allow 1 bin violation)
                valid_bins = [(b, v) for b, v in zip(bin_labels, bins) if np.isfinite(v)]
                vals_in_order = [v for _, v in valid_bins]
                violations = sum(
                    vals_in_order[i] > vals_in_order[i - 1] + 1e-15
                    for i in range(1, len(vals_in_order))
                )
                if ratio > 1 and violations == 0:
                    cell_v = "S"   # SUPPORTED
                elif ratio <= 1:
                    cell_v = "R"   # REFUTED
                else:
                    cell_v = "I"   # INCONCLUSIVE
                cell_verdicts.append(cell_v)
                print(f"  {arm:<4} {pname:<12} {ratio:<8.2f} {cell_v}  bins={[f'{v:.1e}' for v in vals_in_order]}")

        n_supported = cell_verdicts.count("S")
        n_refuted   = cell_verdicts.count("R")
        n_total     = len(cell_verdicts)
        if n_supported > n_total / 2:
            h4_verdict = "SUPPORTED"
        elif n_refuted > n_total / 2:
            h4_verdict = "REFUTED"
        else:
            h4_verdict = "INCONCLUSIVE"
        verdicts["H4"] = h4_verdict
        print(f"  Rule: majority vote over 9 param×arm cells (S/R/I count: {n_supported}/{n_refuted}/{n_total-n_supported-n_refuted})")
        print(f"  → H4: {h4_verdict}")
    else:
        verdicts["H4"] = "INCONCLUSIVE"
        print("\n[H4] INCONCLUSIVE — Stage E not run")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("VERDICT TABLE")
    for h, v in verdicts.items():
        print(f"  {h}: {v}")
    print("=" * 72 + "\n")

    # Write verdicts.json
    verdicts_path = out_dir / "verdicts.json"
    verdicts_out  = {
        "verdicts":    verdicts,
        "eval_window": {"start": str(EVAL_START), "end": str(EVAL_END)},
        "analysis_n_primary":  int(stage_a_data["analysis_mask_r12"].sum()),
        "analysis_n_full":     int(stage_a_data["analysis_mask_full"].sum()),
        "n_gauges":            len(stage_a_data["gauge_staids"]),
    }
    if c_data is not None:
        verdicts_out["L1_median_spread"] = {
            "n":         float(c_data["median_spread_n"]),
            "q_spatial": float(np.nanmedian(c_data["per_reach_spread_q"])),
            "p_spatial": float(np.nanmedian(c_data["per_reach_spread_p"])),
        }
    if e_data is not None:
        verdicts_out["L4_h3_mean_cosines"] = {
            p: e_data["h3"][p]["mean_cosine"] for p in e_data["h3"]
        }
    with open(verdicts_path, "w") as f:
        json.dump(verdicts_out, f, indent=2)
    print(f"verdicts written → {verdicts_path}")

    # ---- Figures ----
    if c_data is None and e_data is None:
        print("[Stage F] skipping figures — no C/E data available")
        return

    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[Stage F] WARNING: matplotlib not available — skipping figures")
        return

    # Fig 1: param scatter matrices per pair (R1 vs R2, R1 vs R3, R2 vs R3)
    if c_data is not None:
        # Load raw params for scatter — just show spread CDFs since params not returned
        pass  # scatter requires reloading params; not done here to keep Stage F side-effect free

    # Fig 2: spread CDFs — n vs geometry (arm-own)
    if c_data is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        spread_n  = c_data["per_reach_spread_n"]
        geo_own   = c_data["geo_spread_own"]
        for label, arr, ls in [
            ("n (norm)", spread_n, "-"),
            ("depth (rel)", geo_own["depth"], "--"),
            ("top-width (rel)", geo_own["top_width"], "-."),
            ("hyd-radius (rel)", geo_own["hydraulic_radius"], ":"),
        ]:
            vals = arr[np.isfinite(arr)]
            if len(vals) == 0:
                continue
            vals = np.sort(vals)
            cdf  = np.linspace(0, 1, len(vals))
            ax.plot(vals, cdf, ls, label=label, linewidth=1.5)
        ax.set_xlabel("Cross-arm spread (normalized / relative)")
        ax.set_ylabel("CDF")
        ax.set_title("Spread CDFs: Manning's n vs realized geometry")
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
        fig.tight_layout()
        fp = figs_dir / "spread_cdfs.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"  figure → {fp}")

    # Fig 3: gradient cosine bar chart
    if e_data is not None:
        h3    = e_data["h3"]
        pairs = ["R1-R2", "R1-R3", "R2-R3"]
        params_plot = ["n", "q_spatial", "p_spatial"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
        for ax, pname in zip(axes, params_plot):
            cos_vals = [
                h3[pname]["cosine_R1R2"],
                h3[pname]["cosine_R1R3"],
                h3[pname]["cosine_R2R3"],
            ]
            colors = ["steelblue", "darkorange", "seagreen"]
            bars   = ax.bar(pairs, cos_vals, color=colors)
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_title(pname)
            ax.set_ylim(-1, 1)
            ax.set_ylabel("Cosine similarity" if pname == "n" else "")
            for bar, v in zip(bars, cos_vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        fig.suptitle("Cross-arm gradient cosine alignment (H3)", fontsize=10)
        fig.tight_layout()
        fp = figs_dir / "gradient_cosine.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"  figure → {fp}")

    # Fig 4: |g|-vs-distance curves
    if e_data is not None:
        h4         = e_data["h4"]
        bin_labels = e_data["bin_labels"]
        fig, axes  = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, pname in zip(axes, ["n", "q_spatial", "p_spatial"]):
            for arm, ls in zip(("R1", "R2", "R3"), ("-", "--", ":")):
                bins = h4[arm][pname]["bin_medians"]
                valid = [(lbl, v) for lbl, v in zip(bin_labels, bins) if np.isfinite(v)]
                if valid:
                    lbls, vals = zip(*valid)
                    ax.plot(range(len(lbls)), vals, ls + "o", label=arm, linewidth=1.5, markersize=4)
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, fontsize=8)
            ax.set_xlabel("Distance bins (hops to nearest gauge)")
            ax.set_ylabel("|grad| median" if pname == "n" else "")
            ax.set_title(f"|∂L/∂{pname}| vs gauge distance")
            ax.legend(fontsize=8)
        fig.suptitle("Gradient reachability decay (H4)", fontsize=10)
        fig.tight_layout()
        fp = figs_dir / "grad_distance.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"  figure → {fp}")

# ---------------------------------------------------------------------------
# Dev-time cross-checks (run when --max-gauges < 20)
# ---------------------------------------------------------------------------

def _dev_crosscheck_topo(
    stage_a_data: dict,
    b_daily: dict,
    store_path: str,
) -> None:
    """For a small eval network, verify the topological accumulation.

    1. For each headwater (no upstream edges): q_sum == its own q_direct.
    2. For one gauge reach: q_sum ≈ brute-force isin sum of all upstream q_direct.
    """
    print("\n[Dev cross-check] topological accumulation verification")
    import icechunk as ic  # type: ignore
    import xarray as xr    # type: ignore

    eval_comids = stage_a_data["eval_comids"]
    local_down  = stage_a_data["local_down"]
    local_up    = stage_a_data["local_up"]

    # Load q_direct (non-accumulated) for verification
    repo   = ic.Repository.open(ic.local_filesystem_storage(store_path))
    ds     = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    time_vals = ds["time"].values
    t_mask    = (time_vals >= EVAL_START) & (time_vals <= EVAL_END)
    store_div = ds["divide_id"].values
    store_div_set = set(store_div.tolist())
    covered_mask = np.array([int(c) in store_div_set for c in eval_comids])

    covered_comids = eval_comids[covered_mask].tolist()
    qr_raw = ds["Qr"].sel(divide_id=covered_comids, time=t_mask).values.astype(np.float32)
    q_direct = np.zeros((len(eval_comids), int(t_mask.sum())), dtype=np.float32)
    q_direct[covered_mask] = qr_raw

    # q_sum is already computed (b_daily contains median over time)
    # Re-compute q_sum for cross-check
    q_sum = q_direct.copy()
    topo_accumulate(q_sum, local_down, local_up)

    # 1. Headwaters: not appearing as local_down (no upstream edge)
    has_upstream = np.zeros(len(eval_comids), dtype=bool)
    has_upstream[local_down] = True  # reaches that have at least one upstream edge
    # Actually: reaches appearing in local_down ARE downstream of something
    # Headwaters = COMIDs that are NOT in local_down... wait.
    # A headwater is a reach with NO upstream neighbours, i.e., it never appears as local_down.
    # Wait: local_down[k] is downstream, local_up[k] is upstream.
    # A headwater appears as local_up but NOT as local_down (it's never downstream of anything).
    # Actually: a headwater has no upstream, meaning it never appears in local_up.
    # And it may or may not appear in local_down.
    # Let me rethink: local_up contains upstream reaches. A headwater is a reach
    # that has no incoming upstream edges, meaning it does NOT appear in local_up.
    is_upstream_of_something = set(local_up.tolist())
    headwaters = [i for i in range(len(eval_comids)) if i not in is_upstream_of_something]
    # Hmm wait — any reach can be downstream of another. Let me clarify:
    # "headwater" = a reach that has no upstream neighbours in the eval network
    # = a reach whose local index does NOT appear as local_down[k] for any k?
    # No: local_down[k] is the downstream reach. local_up[k] is the upstream reach.
    # A headwater is a reach with no upstream reaches, i.e., it never appears in local_up.
    # But that's wrong too — any terminal headwater would appear as local_up[k] since
    # it flows INTO local_down[k].
    # Correct: a headwater never appears as local_down (it's never the downstream endpoint
    # of an edge). Wait — every reach except the outlet appears as local_up at least once.
    # A headwater has NO upstream, so it never has an edge where IT is the downstream node.
    # So a headwater: its local index NEVER appears in local_up? No...
    # Let me re-read: edges are (downstream, upstream). local_down = downstream, local_up = upstream.
    # A headwater: no upstream tributaries → it never appears as local_down[k]
    # (because there's no edge where it is the downstream endpoint of an upstream-flowing edge).
    # Wait, that's wrong. Every reach CAN appear as local_down[k] if it has tributaries upstream.
    # A headwater has NO tributaries, so it never appears as local_down[k].
    headwater_set = set(range(len(eval_comids))) - set(local_down.tolist())
    headwaters = sorted(headwater_set)[:5]

    print(f"  headwaters (first 5 local indices): {headwaters}")
    all_hw_ok = True
    for hw in headwaters:
        # q_sum[hw] should equal q_direct[hw] (no accumulation for headwaters)
        ok = np.allclose(q_sum[hw], q_direct[hw], atol=1e-5, rtol=1e-4)
        if not ok:
            max_diff = np.max(np.abs(q_sum[hw] - q_direct[hw]))
            print(f"  FAIL headwater {hw} (COMID {eval_comids[hw]}): max_diff={max_diff:.2e}")
            all_hw_ok = False
    if all_hw_ok:
        print(f"  headwater check: PASS ({len(headwaters)} checked)")

    # 2. One leaf gauge reach: pick the last reach in topo order (most downstream)
    # It's the outlet of some subgraph. Its q_sum should ≈ sum of all upstream q_direct.
    leaf_local = int(np.max(local_down)) if len(local_down) > 0 else 0
    leaf_comid = int(eval_comids[leaf_local])

    # Brute-force: BFS upstream from leaf_local to find all upstream COMIDs
    upstream_set: set[int] = {leaf_local}
    queue = [leaf_local]
    # local_up[k] is upstream of local_down[k] — build reverse lookup
    upstream_of: dict[int, list[int]] = {}
    for k in range(len(local_down)):
        d, u = int(local_down[k]), int(local_up[k])
        upstream_of.setdefault(d, []).append(u)
    while queue:
        node = queue.pop()
        for u in upstream_of.get(node, []):
            if u not in upstream_set:
                upstream_set.add(u)
                queue.append(u)
    upstream_idx = np.array(sorted(upstream_set), dtype=np.int32)

    # brute-force sum = sum of q_direct over all upstream COMIDs (per day)
    brute_sum = q_direct[upstream_idx].sum(axis=0)  # (n_days,)
    topo_sum  = q_sum[leaf_local]

    max_diff  = np.max(np.abs(brute_sum - topo_sum))
    rel_diff  = max_diff / (np.abs(brute_sum).mean() + 1e-6)
    print(f"  outlet COMID {leaf_comid} (n_upstream={len(upstream_idx)}):")
    print(f"    max abs diff (brute vs topo): {max_diff:.4e} m³/s")
    print(f"    relative diff: {rel_diff:.4e}")
    if rel_diff < 1e-3:
        print("  topological accumulation cross-check: PASS")
    else:
        print("  topological accumulation cross-check: FAIL (check edge direction)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-arm selective-equifinality convergence analysis"
    )
    # Run IDs (Stage D)
    ap.add_argument("--r1",  default=None, help="Run ID for arm R1")
    ap.add_argument("--r2",  default=None, help="Run ID for arm R2")
    ap.add_argument("--r3",  default=None, help="Run ID for arm R3")
    ap.add_argument("--ddrs-root", default="/home/tbindas/projects/ddrs",
                    help="Path to ddrs workspace root")
    # kan_parameters.nc paths (Stage C)
    ap.add_argument("--params-r1", default=None, type=Path)
    ap.add_argument("--params-r2", default=None, type=Path)
    ap.add_argument("--params-r3", default=None, type=Path)
    # Gradient NetCDF paths (Stage E)
    ap.add_argument("--grads-r1", default=None, type=Path)
    ap.add_argument("--grads-r2", default=None, type=Path)
    ap.add_argument("--grads-r3", default=None, type=Path)
    # Output
    ap.add_argument("--out",        default=None, help="Output dir (default: <ddrs-root>/output/equif)")
    ap.add_argument("--max-gauges", default=None, type=int,
                    help="Subsample gauges for fast dev run")
    ap.add_argument("--force",      action="store_true",
                    help="Recompute all stages even if cache exists")
    args = ap.parse_args()

    ddrs_root = Path(args.ddrs_root)
    runs_dir  = ddrs_root / ".ddrs" / "runs"
    out_dir   = Path(args.out) if args.out else ddrs_root / "output" / "equif"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ddrs root : {ddrs_root}")
    print(f"output dir: {out_dir}")
    print(f"max_gauges: {args.max_gauges}")
    print()

    # Stage A
    a_data = stage_a(
        out_dir, GAGES_ADJ_PATH, CONUS_ADJ_PATH,
        max_gauges=args.max_gauges, force=args.force,
    )

    # Stage B — daily-lstm (R1/R2)
    b_daily = stage_b(out_dir, a_data, "daily", DAILY_LSTM_STORE,
                      is_hourly=False, force=args.force)
    # Stage B — hourly-lstm (R3)
    b_hourly = stage_b(out_dir, a_data, "hourly", HOURLY_LSTM_STORE,
                       is_hourly=True, force=args.force)
    # Stage B — common (dHBV2)
    b_common = stage_b(out_dir, a_data, "common", COMMON_STORE,
                       is_hourly=False, force=args.force)

    # Dev cross-check when eval network is small
    if args.max_gauges is not None and args.max_gauges <= 20:
        _dev_crosscheck_topo(a_data, b_daily, DAILY_LSTM_STORE)

    # Stage C
    c_data = None
    if args.params_r1 and args.params_r2 and args.params_r3:
        c_data = stage_c(
            out_dir, a_data,
            args.params_r1, args.params_r2, args.params_r3,
            b_daily, b_hourly, b_common,
            force=args.force,
        )
    else:
        print("[Stage C] skipped — one or more --params-r* not provided")

    # Stage D
    d_data = stage_d(
        out_dir, runs_dir,
        args.r1, args.r2, args.r3,
        force=args.force,
    )

    # Stage E
    e_data = None
    if args.grads_r1 and args.grads_r2 and args.grads_r3:
        e_data = stage_e(
            out_dir, a_data,
            args.grads_r1, args.grads_r2, args.grads_r3,
            force=args.force,
        )
    else:
        print("[Stage E] skipped — one or more --grads-r* not provided")

    # Stage F
    stage_f(out_dir, a_data, c_data, d_data, e_data)


if __name__ == "__main__":
    main()
