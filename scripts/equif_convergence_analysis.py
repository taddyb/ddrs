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
MERIT_ATTRS_PATH = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"

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
    hourly_block_days: int = 30,
) -> np.ndarray:
    """Load daily Q' [n_eval, n_days_eval] for eval_comids over eval window.

    Missing COMIDs (not in store) are 0-filled.
    Hourly stores are aggregated to daily means via chunked time iteration
    (hourly_block_days at a time) to cap peak RSS — loading the full eval
    window at once (132k reaches × 131k hours) exceeds 70 GB.
    Peak extra memory per block: n_cov × (block_days*24) × 4 bytes
    = 132336 × 720 × 4 ≈ 0.38 GB at the default block size of 30 days.
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

    if is_hourly:
        # --- Chunked hourly → daily aggregation ---
        # Compute day-grouping metadata once, then iterate hour-blocks.
        days_only = time_vals.astype("datetime64[D]")
        t_mask    = (days_only >= EVAL_START.astype("datetime64[D]")) & (
                     days_only <= EVAL_END.astype("datetime64[D]"))
        hour_indices = np.flatnonzero(t_mask)
        t_sub        = time_vals[hour_indices]
        days_sub     = t_sub.astype("datetime64[D]")
        unique_days, inv_idx = np.unique(days_sub, return_inverse=True)
        n_days = len(unique_days)
        n_cov  = covered_mask.sum()
        print(f"    hourly eval window: {len(hour_indices)} hours → {n_days} days"
              f"  (block_days={hourly_block_days})")

        # Persistent NaN-safe accumulators
        qr_daily   = np.zeros((n_cov, n_days), dtype=np.float32)
        fin_counts = np.zeros((n_cov, n_days), dtype=np.int32)

        # Pre-select by divide_id once to avoid repeated label lookup per block
        qr_da       = ds["Qr"].sel(divide_id=covered_comids)
        block_hours = hourly_block_days * 24
        n_hours     = len(hour_indices)
        block_num   = 0
        h0 = 0
        while h0 < n_hours:
            h1        = min(h0 + block_hours, n_hours)
            h_slice   = hour_indices[h0:h1]
            inv_local = inv_idx[h0:h1]
            block_num += 1
            if block_num % 12 == 1 or h1 == n_hours:
                print(f"      block {block_num}: hours [{h0}, {h1}) / {n_hours}"
                      f"  days [{inv_local[0]}, {inv_local[-1]}]")
                sys.stdout.flush()

            block   = qr_da.isel(time=h_slice).values.astype(np.float32)
            qr_safe = np.where(np.isfinite(block), block, 0.0)
            fin_blk = np.isfinite(block).astype(np.int32)
            del block

            np.add.at(qr_daily.T,   inv_local, qr_safe.T)
            np.add.at(fin_counts.T, inv_local, fin_blk.T)
            del qr_safe, fin_blk

            h0 = h1

        qr_daily  /= np.maximum(fin_counts, 1)
        qr_covered  = qr_daily
        n_days_eval = n_days

    else:
        # --- Daily path: load entire eval window at once (fine for daily stores) ---
        t_mask     = (time_vals >= EVAL_START) & (time_vals <= EVAL_END)
        qr_covered = ds["Qr"].sel(
            divide_id=covered_comids,
        ).isel(time=np.flatnonzero(t_mask)).values.astype(np.float32)
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
# Cosine / gradient helpers (module-level; shared by stage_e and stage_e_ext)
# ---------------------------------------------------------------------------

def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    return float(np.dot(v1, v2) / (n1 * n2))


def _sign_agree(v1: np.ndarray, v2: np.ndarray) -> float:
    s1 = np.sign(v1)
    s2 = np.sign(v2)
    nonzero = (s1 != 0) & (s2 != 0)
    if nonzero.sum() == 0:
        return float("nan")
    return float((s1[nonzero] == s2[nonzero]).mean())


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
    hourly_block_days: int = 30,
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
    q = load_qprime_for_eval_network(store_path, eval_comids, is_hourly=is_hourly,
                                     hourly_block_days=hourly_block_days)
    # q: [n_eval, n_days] f32

    n_eval, n_days = q.shape
    print(f"    loaded: {n_eval} reaches × {n_days} days")

    # Topological accumulation (modifies q in place)
    topo_accumulate(q, local_down, local_up)

    # Per-reach statistics over time axis (NaN-safe: Q' stores can carry NaN)
    median_q = np.nanmedian(q, axis=1).astype(np.float32)
    p10_q    = np.nanpercentile(q, 10, axis=1).astype(np.float32)
    p90_q    = np.nanpercentile(q, 90, axis=1).astype(np.float32)
    mean_q   = np.nanmean(q, axis=1).astype(np.float32)

    np.savez_compressed(cache, median_q=median_q, p10_q=p10_q, p90_q=p90_q, mean_q=mean_q)
    print(f"    saved → {cache}")

    return {"median_q": median_q, "p10_q": p10_q, "p90_q": p90_q, "mean_q": mean_q}

# ---------------------------------------------------------------------------
# Stage B2: inter-store timing disagreement
# ---------------------------------------------------------------------------

def stage_b2(
    out_dir: Path,
    stage_a_data: dict,
    force: bool = False,
    hourly_block_days: int = 30,
) -> dict:
    """Compute per-reach inter-store TIMING disagreement (daily-lstm vs hourly-lstm).

    Loads summed upstream daily hydrographs for both stores and computes per-reach:
      pearson_r       : Pearson correlation between the two daily time series
      flashiness_diff : |std(diff(qa))/mean(qa) - std(diff(qb))/mean(qb)|

    Both arrays are aligned to stage_a_data["analysis_mask_r12"].
    Memory: 2 × n_eval × n_days × 4 bytes ≈ 5.8 GB at full scale — acceptable.
    """
    cache = out_dir / "stage_b2.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        if "pearson_r" in d.files and "flashiness_diff" in d.files:
            print("[Stage B2] loading from cache")
            return {"pearson_r": d["pearson_r"], "flashiness_diff": d["flashiness_diff"]}
        print("[Stage B2] cache missing keys — recomputing")

    print("[Stage B2] computing inter-store timing disagreement")
    eval_comids = stage_a_data["eval_comids"]
    local_down  = stage_a_data["local_down"]
    local_up    = stage_a_data["local_up"]
    amask       = stage_a_data["analysis_mask_r12"]

    print("  loading daily-lstm summed series ...")
    q_daily = load_qprime_for_eval_network(
        DAILY_LSTM_STORE, eval_comids, is_hourly=False
    )
    topo_accumulate(q_daily, local_down, local_up)

    print("  loading hourly-lstm summed series (→ daily aggregation) ...")
    q_hourly = load_qprime_for_eval_network(
        HOURLY_LSTM_STORE, eval_comids, is_hourly=True,
        hourly_block_days=hourly_block_days,
    )
    topo_accumulate(q_hourly, local_down, local_up)

    # Align time axes (should both be 5479 days but be defensive)
    n_days = min(q_daily.shape[1], q_hourly.shape[1])
    if n_days != q_daily.shape[1] or n_days != q_hourly.shape[1]:
        print(f"  WARNING: time-axis length mismatch — daily={q_daily.shape[1]}"
              f" hourly={q_hourly.shape[1]}; truncating to {n_days}")

    # Plumbing self-check: if both store paths are identical, correlation must be ≈ 1
    if DAILY_LSTM_STORE == HOURLY_LSTM_STORE:
        print("  [Stage B2] same store for both arms — self-correlation should be ≈ 1.0")

    # Restrict to analysis set BEFORE computing (saves 5x memory on analysis fraction)
    qa = q_daily[amask, :n_days].astype(np.float32)
    qb = q_hourly[amask, :n_days].astype(np.float32)
    del q_daily, q_hourly
    n_reach = qa.shape[0]
    print(f"  analysis reaches: {n_reach}  eval days: {n_days}")

    # Vectorized NaN-aware Pearson correlation per reach
    qa_mu  = np.nanmean(qa, axis=1, keepdims=True)
    qb_mu  = np.nanmean(qb, axis=1, keepdims=True)
    qa_dm  = np.where(np.isfinite(qa), qa - qa_mu, 0.0).astype(np.float32)
    qb_dm  = np.where(np.isfinite(qb), qb - qb_mu, 0.0).astype(np.float32)
    cov    = (qa_dm * qb_dm).sum(axis=1)
    std_a  = np.sqrt((qa_dm ** 2).sum(axis=1))
    std_b  = np.sqrt((qb_dm ** 2).sum(axis=1))
    pearson_r = np.clip(cov / (std_a * std_b + 1e-12), -1.0, 1.0).astype(np.float32)
    del qa_dm, qb_dm, cov, std_a, std_b

    if DAILY_LSTM_STORE == HOURLY_LSTM_STORE:
        med_r = float(np.nanmedian(pearson_r))
        assert med_r > 0.99, (
            f"[Stage B2] self-correlation plumbing check failed: median r={med_r:.4f} (expected >0.99)"
        )
        print(f"  self-correlation median r={med_r:.6f} — plumbing OK")

    # Flashiness: |std(diff(qa))/mean(qa) - std(diff(qb))/mean(qb)|  (Richards-Baker style)
    diff_a = np.diff(np.where(np.isfinite(qa), qa, np.nan), axis=1)
    diff_b = np.diff(np.where(np.isfinite(qb), qb, np.nan), axis=1)
    mean_qa = np.nanmean(qa, axis=1)
    mean_qb = np.nanmean(qb, axis=1)
    flash_a = np.nanstd(diff_a, axis=1) / (np.abs(mean_qa) + 1e-6)
    flash_b = np.nanstd(diff_b, axis=1) / (np.abs(mean_qb) + 1e-6)
    flashiness_diff = np.abs(flash_a - flash_b).astype(np.float32)
    del diff_a, diff_b, qa, qb

    print(f"  median Pearson r:       {np.nanmedian(pearson_r):.4f}")
    print(f"  median flashiness-diff: {np.nanmedian(flashiness_diff):.4f}")

    np.savez_compressed(cache, pearson_r=pearson_r, flashiness_diff=flashiness_diff)
    print(f"  saved → {cache}")
    return {"pearson_r": pearson_r, "flashiness_diff": flashiness_diff}

# ---------------------------------------------------------------------------
# Shared N-arm-ready helpers (spread, geometry, percentiles, DA-conditioning)
# ---------------------------------------------------------------------------

def load_align(path: Path, target_comids: np.ndarray) -> dict[str, np.ndarray]:
    """Load kan_parameters.nc and align to target_comids (module-level; shared
    by stage_c and stage_g_level1_5)."""
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


def _rel_spread(stack: np.ndarray) -> np.ndarray:
    """(max-min across arms, axis=0) / (|cross-arm mean| + eps) — the v2-primary
    relative-to-mean normalization, k-agnostic (stack is (k, n))."""
    cross_mean = np.nanmean(stack, axis=0)
    return (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)) / (
        np.abs(cross_mean) + 1e-9
    )


def _arm_own_geometry(
    params_list: list[dict],
    Q_own_per_arm: list[np.ndarray],
    slope: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Per-arm realized geometry using each arm's OWN Q' reference (k-agnostic).

    params_list   : list of k per-arm param dicts (each aligned to the same reaches).
    Q_own_per_arm : list of k per-arm reference discharge arrays (same length as params).
    Returns dict: quantity -> list of k per-arm arrays (depth/top_width/hydraulic_radius).
    """
    geo_stack: dict[str, list] = {"depth": [], "top_width": [], "hydraulic_radius": []}
    for params, Q_own in zip(params_list, Q_own_per_arm):
        g = trapezoidal_geometry(
            params["n"], params["p_spatial"], params["q_spatial"], Q_own, slope,
        )
        for k in geo_stack:
            geo_stack[k].append(g[k])
    return geo_stack


def _percentiles(values: np.ndarray, pcts: tuple = (5, 25, 50, 75, 95)) -> dict[str, float]:
    """Percentiles of a single array, NaN-safe."""
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {f"p{p}": float("nan") for p in pcts}
    return {f"p{p}": float(np.percentile(v, p)) for p in pcts}


def _decile_bins(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Percentile-based decile bin index (0..n_bins-1) per element of x."""
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges = edges.copy()
    edges[0]  -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(x, edges[1:-1], right=False)


def _binned_profile(log_da: np.ndarray, values: np.ndarray, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Median `values` and median log_da within decile bins of log_da (NaN-safe)."""
    valid = np.isfinite(log_da) & np.isfinite(values)
    log_da_v, values_v = log_da[valid], values[valid]
    bin_da  = np.full(n_bins, np.nan)
    bin_val = np.full(n_bins, np.nan)
    bin_n   = np.zeros(n_bins, dtype=np.int64)
    if len(log_da_v) < n_bins:
        return {"bin_da_median": bin_da, "bin_value_median": bin_val, "bin_count": bin_n}
    bin_idx = _decile_bins(log_da_v, n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        bin_n[b] = m.sum()
        if m.any():
            bin_da[b]  = np.median(log_da_v[m])
            bin_val[b] = np.median(values_v[m])
    return {"bin_da_median": bin_da, "bin_value_median": bin_val, "bin_count": bin_n}


def _loglog_slope(log_da: np.ndarray, values: np.ndarray) -> tuple[float, int]:
    """OLS slope of ln(values) vs log_da (exploratory, unweighted np.polyfit).

    Drops non-finite and non-positive values (ln undefined) before fitting;
    returns (slope, n_dropped).
    """
    finite   = np.isfinite(log_da) & np.isfinite(values)
    positive = values > 0
    keep     = finite & positive
    n_dropped = int((~keep).sum())
    if keep.sum() < 10:
        return float("nan"), n_dropped
    slope, _intercept = np.polyfit(log_da[keep], np.log(values[keep]), 1)
    return float(slope), n_dropped


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
        if "per_reach_rel_spread_n" not in d.files:
            print("  [Stage C] old cache missing audit keys (rel-spread) — recomputing")
        else:
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
                "per_reach_spread_n":     d["per_reach_spread_n"],
                "per_reach_spread_q":     d["per_reach_spread_q"],
                "per_reach_spread_p":     d["per_reach_spread_p"],
                "median_spread_n":        float(d["median_spread_n"]),
                "geo_spread_own":         geo_spread_own,
                "geo_spread_common":      geo_spread_common,
                "q_disagreement":         d["q_disagreement"],
                "h2_rho":                 float(d["h2_rho"]),
                "spearman_n":             dict(zip(
                    d["spearman_pair_labels"].tolist(),
                    d["spearman_n"].tolist(),
                )),
                # Audit keys (rel-to-mean spread + per-arm medians)
                "per_reach_rel_spread_n": d["per_reach_rel_spread_n"],
                "per_reach_rel_spread_q": d["per_reach_rel_spread_q"],
                "per_reach_rel_spread_p": d["per_reach_rel_spread_p"],
                "rel_spread_n":           float(d["rel_spread_n"]),
                "rel_spread_q":           float(d["rel_spread_q"]),
                "rel_spread_p":           float(d["rel_spread_p"]),
                "arm_median_n":           {
                    "R1": float(d["arm_median_n_R1"]),
                    "R2": float(d["arm_median_n_R2"]),
                    "R3": float(d["arm_median_n_R3"]),
                },
                "arm_median_q":           {
                    "R1": float(d["arm_median_q_R1"]),
                    "R2": float(d["arm_median_q_R2"]),
                    "R3": float(d["arm_median_q_R3"]),
                },
                "arm_median_p":           {
                    "R1": float(d["arm_median_p_R1"]),
                    "R2": float(d["arm_median_p_R2"]),
                    "R3": float(d["arm_median_p_R3"]),
                },
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

    p1 = load_align(params_r1, analysis_comids)
    p2 = load_align(params_r2, analysis_comids)
    p3 = load_align(params_r3, analysis_comids)

    # Guard: q_spatial and p_spatial must be present — fail fast with actionable message
    for pname in ("q_spatial", "p_spatial"):
        for arm_label, pdata, ppath in (("R1", p1, params_r1), ("R2", p2, params_r2), ("R3", p3, params_r3)):
            if pname not in pdata:
                sys.exit(
                    f"ERROR: required parameter '{pname}' not found in {ppath} (arm {arm_label}); "
                    f"re-run ddrs eval with a config that includes '{pname}' in kan_head.learnable_parameters"
                )

    # Assert slope is identical across arms (by construction)
    if "slope" in p1 and "slope" in p2 and "slope" in p3:
        diff_12 = np.nanmax(np.abs(p1["slope"] - p2["slope"]))
        diff_13 = np.nanmax(np.abs(p1["slope"] - p3["slope"]))
        print(f"  slope max-abs-diff R1-R2: {diff_12:.2e}, R1-R3: {diff_13:.2e}")

    # ---- Level 1: raw-param spread ----
    from scipy.stats import spearmanr  # type: ignore

    def param_stats(values: list[np.ndarray], p_name) -> dict:
        """N-arm-ready: values is a list of k per-arm arrays (k=3 here)."""
        lo, hi = PARAM_RANGES.get(p_name, (0.0, 1.0))
        rng = hi - lo
        stack = np.stack(values, axis=0)                # (k, n)
        per_reach_spread = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)) / rng
        median_spread    = float(np.nanmedian(per_reach_spread))
        # Relative-to-mean spread (audit: like-for-like comparison with geometry normalization)
        per_reach_rel_spread = _rel_spread(stack)
        median_rel_spread = float(np.nanmedian(per_reach_rel_spread))
        k = len(values)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        spears = {}
        for i, j in pairs:
            a, b, lbl = values[i], values[j], f"R{i + 1}-R{j + 1}"
            valid = np.isfinite(a) & np.isfinite(b)
            a_v, b_v = a[valid], b[valid]
            # Skip if too few points or either input is constant (would produce NaN/warning)
            if valid.sum() > 5 and np.unique(a_v).size >= 2 and np.unique(b_v).size >= 2:
                r, _ = spearmanr(a_v, b_v)
            else:
                r = float("nan")
            spears[lbl] = float(r)
        return {
            "per_reach_spread":     per_reach_spread,
            "median_spread":        median_spread,
            "per_reach_rel_spread": per_reach_rel_spread,
            "median_rel_spread":    median_rel_spread,
            "spearman":             spears,
        }

    stats_n = param_stats([p1["n"], p2["n"], p3["n"]], "n")
    stats_q = param_stats([p1["q_spatial"], p2["q_spatial"], p3["q_spatial"]], "q_spatial")
    stats_p = param_stats([p1["p_spatial"], p2["p_spatial"], p3["p_spatial"]], "p_spatial")

    print(f"  Level 1 median norm-spread:  n={stats_n['median_spread']:.4f}"
          f"  q={stats_q['median_spread']:.4f}  p={stats_p['median_spread']:.4f}")
    print(f"  Level 1 median rel-spread:   n={stats_n['median_rel_spread']:.4f}"
          f"  q={stats_q['median_rel_spread']:.4f}  p={stats_p['median_rel_spread']:.4f}")
    for lbl, r in stats_n["spearman"].items():
        print(f"    spearman n {lbl}: {r:.3f}")

    # Per-arm medians (for audit level-disagreement reporting)
    arm_median_n = {
        "R1": float(np.nanmedian(p1["n"])),
        "R2": float(np.nanmedian(p2["n"])),
        "R3": float(np.nanmedian(p3["n"])),
    }
    arm_median_q = {
        "R1": float(np.nanmedian(p1["q_spatial"])),
        "R2": float(np.nanmedian(p2["q_spatial"])),
        "R3": float(np.nanmedian(p3["q_spatial"])),
    }
    arm_median_p = {
        "R1": float(np.nanmedian(p1["p_spatial"])),
        "R2": float(np.nanmedian(p2["p_spatial"])),
        "R3": float(np.nanmedian(p3["p_spatial"])),
    }
    print(f"  Per-arm median n: R1={arm_median_n['R1']:.4f}  R2={arm_median_n['R2']:.4f}  R3={arm_median_n['R3']:.4f}")
    print(f"  Per-arm median q: R1={arm_median_q['R1']:.4f}  R2={arm_median_q['R2']:.4f}  R3={arm_median_q['R3']:.4f}")
    print(f"  Per-arm median p: R1={arm_median_p['R1']:.4f}  R2={arm_median_p['R2']:.4f}  R3={arm_median_p['R3']:.4f}")

    # ---- Level 2: realized geometry ----
    slope_use = p1.get("slope", np.ones(len(analysis_comids), dtype=np.float32))

    def geometry_spread_at_Q(
        Q_ref: np.ndarray,
        label: str,
        arm_params: list[dict],
        slope: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute per-reach relative spread of depth/top_width/hyd_radius.

        arm_params : list of k param dicts (one per arm), each aligned to Q_ref length,
                     SAME Q_ref for every arm (N-arm-ready via _rel_spread).
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
            spread = _rel_spread(np.stack(arm_list, axis=0))
            result[k] = spread
            print(f"  Level 2 [{label}] {k} median rel-spread: {np.nanmedian(spread):.4f}")
        return result

    # PRIMARY: arm-own Q' references (R1/R2 share daily; R3 uses hourly)
    q_r12 = b_r12["median_q"][amask]
    q_r3  = b_r3["median_q"][amask]

    # Use each arm's own median Q' as operating point (N-arm-ready: list, not tuple)
    Q_own_per_arm = [q_r12, q_r12, q_r3]  # R1/R2 share daily store
    geo_stack_own = _arm_own_geometry([p1, p2, p3], Q_own_per_arm, slope_use)

    geo_spread_own: dict[str, np.ndarray] = {}
    print("  Level 2 [arm-own Q'] relative spread:")
    for k, arm_list in geo_stack_own.items():
        spread = _rel_spread(np.stack(arm_list, axis=0))
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
        Q_common, label="common-median", arm_params=[p1s, p2s, p3s], slope=slope_s)
    print("  Level 2 [common dHBV2 p10] relative spread:")
    geo_spread_common_p10 = geometry_spread_at_Q(
        Q_common_p10, label="common-p10", arm_params=[p1s, p2s, p3s], slope=slope_s)
    print("  Level 2 [common dHBV2 p90] relative spread:")
    geo_spread_common_p90 = geometry_spread_at_Q(
        Q_common_p90, label="common-p90", arm_params=[p1s, p2s, p3s], slope=slope_s)

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
    v1, v2 = stats_n["per_reach_spread"][valid], q_disagreement[valid]
    if valid.sum() > 5 and np.unique(v1).size >= 2 and np.unique(v2).size >= 2:
        h2_rho, _ = sp(v1, v2)
    else:
        h2_rho = float("nan")
    print(f"  H2 Spearman(n-spread, Q'-disagreement): {h2_rho:.3f}")

    np.savez_compressed(
        cache,
        # Level 1 (registered keys — unchanged)
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
        # Level 2 sensitivity (named access — dict order is not guaranteed stable)
        geo_common_depth     = geo_spread_common["depth"],
        geo_common_top_width = geo_spread_common["top_width"],
        geo_common_hyd_radius= geo_spread_common["hydraulic_radius"],
        # Q' disagreement
        q_disagreement      = q_disagreement,
        h2_rho              = np.float32(h2_rho),
        # Analysis set info
        analysis_n_primary  = np.int32(amask.sum()),
        analysis_n_full     = np.int32(amask_full.sum()),
        # Audit keys: relative-to-mean spread (like-for-like with geometry normalization)
        per_reach_rel_spread_n = stats_n["per_reach_rel_spread"],
        per_reach_rel_spread_q = stats_q["per_reach_rel_spread"],
        per_reach_rel_spread_p = stats_p["per_reach_rel_spread"],
        rel_spread_n        = np.float32(stats_n["median_rel_spread"]),
        rel_spread_q        = np.float32(stats_q["median_rel_spread"]),
        rel_spread_p        = np.float32(stats_p["median_rel_spread"]),
        # Audit keys: per-arm medians
        arm_median_n_R1     = np.float32(arm_median_n["R1"]),
        arm_median_n_R2     = np.float32(arm_median_n["R2"]),
        arm_median_n_R3     = np.float32(arm_median_n["R3"]),
        arm_median_q_R1     = np.float32(arm_median_q["R1"]),
        arm_median_q_R2     = np.float32(arm_median_q["R2"]),
        arm_median_q_R3     = np.float32(arm_median_q["R3"]),
        arm_median_p_R1     = np.float32(arm_median_p["R1"]),
        arm_median_p_R2     = np.float32(arm_median_p["R2"]),
        arm_median_p_R3     = np.float32(arm_median_p["R3"]),
    )
    print(f"  saved → {cache}")

    return {
        "per_reach_spread_n":     stats_n["per_reach_spread"],
        "per_reach_spread_q":     stats_q["per_reach_spread"],
        "per_reach_spread_p":     stats_p["per_reach_spread"],
        "median_spread_n":        stats_n["median_spread"],
        "geo_spread_own":         geo_spread_own,
        "geo_spread_common":      geo_spread_common,
        "q_disagreement":         q_disagreement,
        "h2_rho":                 float(h2_rho),
        "spearman_n":             stats_n["spearman"],
        # Audit fields
        "per_reach_rel_spread_n": stats_n["per_reach_rel_spread"],
        "per_reach_rel_spread_q": stats_q["per_reach_rel_spread"],
        "per_reach_rel_spread_p": stats_p["per_reach_rel_spread"],
        "rel_spread_n":           stats_n["median_rel_spread"],
        "rel_spread_q":           stats_q["median_rel_spread"],
        "rel_spread_p":           stats_p["median_rel_spread"],
        "arm_median_n":           arm_median_n,
        "arm_median_q":           arm_median_q,
        "arm_median_p":           arm_median_p,
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
        # H3 sign-agreement fractions (per pair, same order as cosines)
        h3_n_sign_agree = np.array([h3["n"]["sign_agree_R1R2"],         h3["n"]["sign_agree_R1R3"],         h3["n"]["sign_agree_R2R3"]]),
        h3_q_sign_agree = np.array([h3["q_spatial"]["sign_agree_R1R2"], h3["q_spatial"]["sign_agree_R1R3"], h3["q_spatial"]["sign_agree_R2R3"]]),
        h3_p_sign_agree = np.array([h3["p_spatial"]["sign_agree_R1R2"], h3["p_spatial"]["sign_agree_R1R3"], h3["p_spatial"]["sign_agree_R2R3"]]),
        # H4 per-arm ratio arrays (R1/R2/R3 × params)
        **{f"h4_{arm}_{p}_ratio": np.float32(h4[arm][p]["ratio"])
           for arm in ("R1", "R2", "R3") for p in ("n", "q_spatial", "p_spatial")},
        **{f"h4_{arm}_{p}_bins": np.array(h4[arm][p]["bin_medians"])
           for arm in ("R1", "R2", "R3") for p in ("n", "q_spatial", "p_spatial")},
    )
    print(f"  saved → {cache}")
    return {"h3": h3, "h4": h4, "probe_dist": probe_dist, "bin_labels": bin_labels}

# ---------------------------------------------------------------------------
# Stage E extension: common-mode-removed H3 + within-arm noise ceiling
# ---------------------------------------------------------------------------

def stage_e_ext(
    out_dir: Path,
    grads_r1: Optional[Path],
    grads_r2: Optional[Path],
    grads_r3: Optional[Path],
    grads_r1_rep: Optional[Path] = None,
    grads_r2_rep: Optional[Path] = None,
    grads_r3_rep: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Common-mode-removed H3 cosines + within-arm noise ceiling (audit, POST-HOC).

    Common-mode removal: for each param, subtract the cross-arm mean gradient field
    from each arm's net-gradient, then recompute pairwise cosines on residuals.

    Noise ceiling (--grads-r*-rep): per arm per param, cosine(seed-42, seed-123).
    Skips gracefully when rep files are absent or don't exist on disk.

    Cache: stage_e_ext.npz (schema-tolerant; --force regenerates).
    Returns empty dict if grad files not provided or stage_e.npz not found.
    """
    cache = out_dir / "stage_e_ext.npz"
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=True)
        if "cm_n_R1R2" in d.files:
            print("[Stage E ext] loading from cache")
            result: dict = {}
            for pname in ("n", "q_spatial", "p_spatial"):
                result[f"cm_{pname}"] = {
                    "R1R2": float(d[f"cm_{pname}_R1R2"]),
                    "R1R3": float(d[f"cm_{pname}_R1R3"]),
                    "R2R3": float(d[f"cm_{pname}_R2R3"]),
                }
            ceiling: dict = {}
            for arm in ("R1", "R2", "R3"):
                arm_ceil: dict = {}
                for pname in ("n", "q_spatial", "p_spatial"):
                    key = f"ceil_{arm}_{pname}"
                    if key in d.files:
                        arm_ceil[pname] = float(d[key])
                if arm_ceil:
                    ceiling[arm] = arm_ceil
            result["ceiling"] = ceiling
            return result
        print("[Stage E ext] cache missing keys — recomputing")

    if not all([grads_r1, grads_r2, grads_r3]):
        print("[Stage E ext] skipped — one or more --grads-r* not provided")
        return {}

    # Need common_comids from stage_e cache
    e_cache = out_dir / "stage_e.npz"
    if not e_cache.exists():
        print("[Stage E ext] stage_e.npz not found — run Stage E first; skipping")
        return {}
    common_comids_arr = np.load(e_cache)["common_comids"].astype(np.int64)
    common_list       = common_comids_arr.tolist()

    print("[Stage E ext] computing common-mode-removed cosines and noise ceiling")

    def _load_align(path: Path) -> dict[str, np.ndarray]:
        """Load gradient NetCDF, align to common_list. Returns net-grad arrays."""
        gdata     = load_gradients(path)
        probe_arr = gdata["COMID_probe"].astype(np.int64)
        probe_set = set(probe_arr.tolist())
        valid_c   = [c for c in common_list if c in probe_set]
        missing   = len(common_list) - len(valid_c)
        if missing:
            print(f"    WARNING: {missing} common COMIDs not in {path.name}")
        idx_map = {int(c): i for i, c in enumerate(probe_arr)}
        idx     = np.array([idx_map[c] for c in valid_c], dtype=np.int64)
        out: dict[str, np.ndarray] = {}
        for net_key in ("grad_n_net", "grad_q_spatial_net", "grad_p_spatial_net"):
            out[net_key] = gdata[net_key][idx]
        out["_n"] = len(valid_c)
        return out

    a1 = _load_align(grads_r1)
    a2 = _load_align(grads_r2)
    a3 = _load_align(grads_r3)

    save_dict: dict = {}
    result: dict    = {}

    # ---- Common-mode removal ----
    print("  Common-mode-removed cosines:")
    for pname, net_key in [
        ("n",         "grad_n_net"),
        ("q_spatial", "grad_q_spatial_net"),
        ("p_spatial", "grad_p_spatial_net"),
    ]:
        v1 = a1[net_key]
        v2 = a2[net_key]
        v3 = a3[net_key]
        n_min = min(len(v1), len(v2), len(v3))
        v1, v2, v3 = v1[:n_min], v2[:n_min], v3[:n_min]

        mean_g = (v1 + v2 + v3) / 3.0
        r1_cm  = v1 - mean_g
        r2_cm  = v2 - mean_g
        r3_cm  = v3 - mean_g

        cos_12 = _cosine_sim(r1_cm, r2_cm)
        cos_13 = _cosine_sim(r1_cm, r3_cm)
        cos_23 = _cosine_sim(r2_cm, r3_cm)
        print(f"    {pname}: R1-R2={cos_12:.3f}  R1-R3={cos_13:.3f}  R2-R3={cos_23:.3f}")

        save_dict[f"cm_{pname}_R1R2"] = np.float32(cos_12)
        save_dict[f"cm_{pname}_R1R3"] = np.float32(cos_13)
        save_dict[f"cm_{pname}_R2R3"] = np.float32(cos_23)
        result[f"cm_{pname}"] = {"R1R2": cos_12, "R1R3": cos_13, "R2R3": cos_23}

    # ---- Noise ceiling (replicate grads, seed 123) ----
    ceiling: dict = {}
    rep_paths = {
        "R1": (grads_r1, grads_r1_rep),
        "R2": (grads_r2, grads_r2_rep),
        "R3": (grads_r3, grads_r3_rep),
    }
    seed42_data = {"R1": a1, "R2": a2, "R3": a3}
    any_rep = False
    for arm_label, (g42_path, g123_path) in rep_paths.items():
        if g123_path is None or not Path(g123_path).exists():
            reason = "not provided" if g123_path is None else f"not found at {g123_path}"
            print(f"  noise ceiling [{arm_label}] SKIPPED — rep file {reason}")
            continue
        print(f"  noise ceiling [{arm_label}] loading {Path(g123_path).name} ...")
        a123 = _load_align(g123_path)
        a42  = seed42_data[arm_label]
        arm_ceil: dict = {}
        for pname, net_key in [
            ("n",         "grad_n_net"),
            ("q_spatial", "grad_q_spatial_net"),
            ("p_spatial", "grad_p_spatial_net"),
        ]:
            v42  = a42[net_key]
            v123 = a123[net_key]
            n_min = min(len(v42), len(v123))
            ceil_val = _cosine_sim(v42[:n_min], v123[:n_min])
            arm_ceil[pname] = ceil_val
            save_dict[f"ceil_{arm_label}_{pname}"] = np.float32(ceil_val)
            print(f"    ceiling [{arm_label}] {pname}: {ceil_val:.3f}")
        ceiling[arm_label] = arm_ceil
        any_rep = True
    if not any_rep:
        print("  noise ceiling: all rep files absent — ceiling section will be skipped in report")
    result["ceiling"] = ceiling

    np.savez_compressed(cache, **save_dict)
    print(f"  saved → {cache}")
    return result

# ---------------------------------------------------------------------------
# Stage H2-network: network-scale H2 variant (per-gauge integrated upstream)
# ---------------------------------------------------------------------------

def stage_h2_network(
    out_dir: Path,
    stage_a_data: dict,
    c_data: dict,
    b2_data: dict,
    gages_adj_path: str = GAGES_ADJ_PATH,
    min_reaches: int = 3,
    force: bool = False,
) -> dict:
    """Network-scale H2 variant (v2 spec, pre-registered).

    For each gauge g with upstream network U(g) (gauge adjacency store, same
    zarr access pattern stage_a already uses):
      n_disagreement(g)      = median of per-reach n rel-spread over U(g)
      timing_disagreement(g) = median of per-reach (1 - pearson_r) over U(g)
      (flashiness_diff reported alongside as the network mean)
    then Spearman rho across gauges between n_disagreement and timing_disagreement.

    Gauges whose upstream set has fewer than `min_reaches` valid (finite)
    reaches are skipped (too few reaches to form a meaningful median).
    """
    cache = out_dir / "stage_h2_network.npz"
    if cache.exists() and not force:
        print("[Stage H2-network] loading from cache")
        d = np.load(cache, allow_pickle=True)
        return {
            "gauge_staids":        d["gauge_staids"].tolist(),
            "n_disagreement":      d["n_disagreement"],
            "timing_disagreement": d["timing_disagreement"],
            "flashiness_mean":     d["flashiness_mean"],
            "n_upstream_reaches":  d["n_upstream_reaches"],
            "rho":                 float(d["rho"]),
            "n_gauges_used":       int(d["n_gauges_used"]),
        }

    print("[Stage H2-network] computing per-gauge network-scale H2 aggregation")
    import zarr  # type: ignore
    from scipy.stats import spearmanr

    gages  = zarr.open(gages_adj_path, mode="r")
    staids = stage_a_data["gauge_staids"]

    eval_comids = stage_a_data["eval_comids"].astype(np.int64)
    amask       = stage_a_data["analysis_mask_r12"]
    analysis_comids = eval_comids[amask]
    comid_to_idx = {int(c): i for i, c in enumerate(analysis_comids)}

    n_rel_spread = c_data["per_reach_rel_spread_n"]
    pearson_r    = b2_data["pearson_r"]
    flashiness   = b2_data["flashiness_diff"]

    n_dis_list: list = []
    timing_dis_list: list = []
    flash_list: list = []
    n_up_list: list = []
    used_staids: list = []
    for staid in staids:
        g_order = gages[staid]["order"][:].astype(np.int64)
        idxs = [comid_to_idx[c] for c in g_order.tolist() if c in comid_to_idx]
        if len(idxs) < min_reaches:
            continue
        idxs = np.array(idxs, dtype=np.int64)
        n_vals = n_rel_spread[idxs]
        r_vals = pearson_r[idxs]
        f_vals = flashiness[idxs]
        valid_n = np.isfinite(n_vals)
        valid_r = np.isfinite(r_vals)
        if valid_n.sum() < min_reaches or valid_r.sum() < min_reaches:
            continue
        n_dis_list.append(float(np.nanmedian(n_vals)))
        timing_dis_list.append(float(np.median(1.0 - r_vals[valid_r])))
        flash_list.append(float(np.nanmean(f_vals)))
        n_up_list.append(len(idxs))
        used_staids.append(staid)

    n_disagreement      = np.array(n_dis_list, dtype=np.float64)
    timing_disagreement = np.array(timing_dis_list, dtype=np.float64)
    flashiness_mean      = np.array(flash_list, dtype=np.float64)
    n_upstream_reaches   = np.array(n_up_list, dtype=np.int32)

    print(f"  gauges used (>= {min_reaches} valid upstream reaches):"
          f" {len(used_staids)} / {len(staids)}")

    valid = np.isfinite(n_disagreement) & np.isfinite(timing_disagreement)
    if (valid.sum() > 5 and np.unique(n_disagreement[valid]).size >= 2
            and np.unique(timing_disagreement[valid]).size >= 2):
        rho, _ = spearmanr(n_disagreement[valid], timing_disagreement[valid])
        rho = float(rho)
    else:
        rho = float("nan")
    n_gauges_used = int(valid.sum())

    print(f"  network-scale H2: Spearman rho(n_disagreement, timing_disagreement)"
          f" = {rho:.3f}  (n_gauges={n_gauges_used})")

    np.savez_compressed(
        cache,
        gauge_staids        = np.array(used_staids),
        n_disagreement      = n_disagreement,
        timing_disagreement = timing_disagreement,
        flashiness_mean      = flashiness_mean,
        n_upstream_reaches   = n_upstream_reaches,
        rho                  = np.float32(rho),
        n_gauges_used        = np.int32(n_gauges_used),
    )
    print(f"  saved → {cache}")
    return {
        "gauge_staids":        used_staids,
        "n_disagreement":      n_disagreement,
        "timing_disagreement": timing_disagreement,
        "flashiness_mean":     flashiness_mean,
        "n_upstream_reaches":  n_upstream_reaches,
        "rho":                 rho,
        "n_gauges_used":       n_gauges_used,
    }

# ---------------------------------------------------------------------------
# Stage G: Level 1.5 — per-arm distributions + drainage-area conditioning
# ---------------------------------------------------------------------------

def stage_g_level1_5(
    out_dir: Path,
    stage_a_data: dict,
    c_data: dict,
    params_r1: Path,
    params_r2: Path,
    params_r3: Path,
    b_r12: dict,
    b_r3: dict,
    attrs_path: str = MERIT_ATTRS_PATH,
    force: bool = False,
) -> dict:
    """Level 1.5 (v2 spec): per-arm percentiles + drainage-area conditioning.

    (a) per-arm percentiles (p5/p25/p50/p75/p95) of n, q_spatial, p_spatial
    (b) join analysis-set COMIDs to log10_uparea (MERIT global attributes)
    (c) per-arm, per-quantity (n, q_spatial, p_spatial + arm-own realized
        depth/top_width/hydraulic_radius): decile-binned median vs log10(DA)
        and OLS slope of ln(quantity) vs log10(DA)
    (d) spread-vs-DA profile: median cross-arm rel-spread (from stage_c, NOT
        box-normalized) within each log10(DA) decile bin

    Descriptive only — no falsification bar (per v2 spec).
    """
    qnames = ("n", "q_spatial", "p_spatial", "depth", "top_width", "hydraulic_radius")
    pct_keys = ("p5", "p25", "p50", "p75", "p95")

    cache = out_dir / "stage_g.npz"
    if cache.exists() and not force:
        print("[Stage G] loading from cache")
        d = np.load(cache, allow_pickle=True)
        percentiles: dict = {}
        for arm_label in ("R1", "R2", "R3"):
            percentiles[arm_label] = {}
            for pname in ("n", "q_spatial", "p_spatial"):
                percentiles[arm_label][pname] = {
                    k: float(d[f"pct_{arm_label}_{pname}_{k}"]) for k in pct_keys
                }
        da_slopes: dict = {}
        da_bins: dict = {}
        for qname in qnames:
            da_slopes[qname] = {}
            da_bins[qname] = {}
            for arm_label in ("R1", "R2", "R3"):
                da_slopes[qname][arm_label] = {
                    "slope":     float(d[f"slope_{qname}_{arm_label}"]),
                    "n_dropped": int(d[f"dropped_{qname}_{arm_label}"]),
                }
                da_bins[qname][arm_label] = {
                    "bin_da_median":    d[f"bin_da_{qname}_{arm_label}"],
                    "bin_value_median": d[f"bin_val_{qname}_{arm_label}"],
                    "bin_count":        d[f"bin_n_{qname}_{arm_label}"],
                }
        spread_vs_da: dict = {}
        for qname in qnames:
            spread_vs_da[qname] = {
                "bin_da_median":    d[f"spread_bin_da_{qname}"],
                "bin_value_median": d[f"spread_bin_val_{qname}"],
                "bin_count":        d[f"spread_bin_n_{qname}"],
            }
        return {
            "percentiles":  percentiles,
            "da_slopes":    da_slopes,
            "da_bins":      da_bins,
            "spread_vs_da": spread_vs_da,
            "log_da":       d["log_da"],
        }

    print("[Stage G] computing Level 1.5 (distributions + drainage-area conditioning)")

    eval_comids = stage_a_data["eval_comids"].astype(np.int64)
    amask       = stage_a_data["analysis_mask_r12"]
    analysis_comids = eval_comids[amask]
    n_analysis = len(analysis_comids)

    p1 = load_align(params_r1, analysis_comids)
    p2 = load_align(params_r2, analysis_comids)
    p3 = load_align(params_r3, analysis_comids)
    arms = {"R1": p1, "R2": p2, "R3": p3}

    # ---- (a) per-arm percentiles ----
    percentiles: dict = {}
    for arm_label, pdict in arms.items():
        percentiles[arm_label] = {
            pname: _percentiles(pdict[pname]) for pname in ("n", "q_spatial", "p_spatial")
        }
    print("  Per-arm percentiles (p5/p25/p50/p75/p95):")
    for arm_label, pd_ in percentiles.items():
        for pname, pcts in pd_.items():
            print(f"    [{arm_label}] {pname}: "
                  + " / ".join(f"{k}={v:.4f}" for k, v in pcts.items()))

    # ---- (b) join log10_uparea (exact-COMID; MERIT CONUS is a strict subset) ----
    import xarray as xr  # type: ignore
    ds_attrs = xr.open_dataset(attrs_path)
    attr_comid  = ds_attrs["COMID"].values.astype(np.int64)
    attr_log_da = ds_attrs["log10_uparea"].values.astype(np.float64)
    comid_to_da = dict(zip(attr_comid.tolist(), attr_log_da.tolist()))
    log_da = np.array(
        [comid_to_da.get(int(c), np.nan) for c in analysis_comids], dtype=np.float64,
    )
    n_missing = int(np.sum(~np.isfinite(log_da)))
    assert n_missing == 0, (
        f"[Stage G] {n_missing} analysis COMIDs missing log10_uparea in {attrs_path}"
        f" — expected 0 (MERIT CONUS eval network must be a strict subset of the"
        f" global attributes file); this is a bug, not expected data loss"
    )
    print(f"  log10_uparea joined: {n_analysis - n_missing} / {n_analysis} (missing={n_missing})")

    # ---- (c) per-arm realized geometry (arm-own Q' reference, same as stage_c) ----
    q_r12 = b_r12["median_q"][amask]
    q_r3  = b_r3["median_q"][amask]
    Q_own_per_arm = [q_r12, q_r12, q_r3]
    slope_use = p1.get("slope", np.ones(n_analysis, dtype=np.float32))
    geo_stack_own = _arm_own_geometry([p1, p2, p3], Q_own_per_arm, slope_use)

    arm_order = ("R1", "R2", "R3")
    quantities: dict[str, dict[str, np.ndarray]] = {
        "n":         {a: arms[a]["n"] for a in arm_order},
        "q_spatial": {a: arms[a]["q_spatial"] for a in arm_order},
        "p_spatial": {a: arms[a]["p_spatial"] for a in arm_order},
        "depth":            {a: geo_stack_own["depth"][i] for i, a in enumerate(arm_order)},
        "top_width":        {a: geo_stack_own["top_width"][i] for i, a in enumerate(arm_order)},
        "hydraulic_radius": {a: geo_stack_own["hydraulic_radius"][i] for i, a in enumerate(arm_order)},
    }

    da_slopes: dict = {}
    da_bins: dict = {}
    print("  DA-conditioned OLS slopes (ln(quantity) ~ log10_uparea):")
    for qname, arm_vals in quantities.items():
        da_slopes[qname] = {}
        da_bins[qname] = {}
        for arm_label, vals in arm_vals.items():
            slope, n_dropped = _loglog_slope(log_da, vals.astype(np.float64))
            da_slopes[qname][arm_label] = {"slope": slope, "n_dropped": n_dropped}
            da_bins[qname][arm_label] = _binned_profile(log_da, vals.astype(np.float64))
            print(f"    {qname:<18} [{arm_label}] slope={slope:+.4f}  (dropped={n_dropped})")

    # ---- (d) spread-vs-DA profile (rel-to-mean spread, NOT box-normalized, from stage_c) ----
    spread_arrays = {
        "n":                c_data["per_reach_rel_spread_n"],
        "q_spatial":        c_data["per_reach_rel_spread_q"],
        "p_spatial":        c_data["per_reach_rel_spread_p"],
        "depth":            c_data["geo_spread_own"]["depth"],
        "top_width":        c_data["geo_spread_own"]["top_width"],
        "hydraulic_radius": c_data["geo_spread_own"]["hydraulic_radius"],
    }
    spread_vs_da: dict = {}
    print("  Spread-vs-DA profile (median rel-spread per decile bin):")
    for qname, spr in spread_arrays.items():
        profile = _binned_profile(log_da, spr.astype(np.float64))
        spread_vs_da[qname] = profile
        finite_vals = profile["bin_value_median"][np.isfinite(profile["bin_value_median"])]
        if len(finite_vals):
            print(f"    {qname:<18} bin medians: "
                  + " / ".join(f"{v:.3f}" for v in finite_vals))

    # ---- save cache (flatten nested dicts for npz) ----
    save_dict: dict = {"analysis_comids": analysis_comids, "log_da": log_da}
    for arm_label, pd_ in percentiles.items():
        for pname, pcts in pd_.items():
            for k, v in pcts.items():
                save_dict[f"pct_{arm_label}_{pname}_{k}"] = np.float64(v)
    for qname, arm_slopes in da_slopes.items():
        for arm_label, sd in arm_slopes.items():
            save_dict[f"slope_{qname}_{arm_label}"]   = np.float64(sd["slope"])
            save_dict[f"dropped_{qname}_{arm_label}"] = np.int64(sd["n_dropped"])
    for qname, arm_profiles in da_bins.items():
        for arm_label, prof in arm_profiles.items():
            save_dict[f"bin_da_{qname}_{arm_label}"]  = prof["bin_da_median"]
            save_dict[f"bin_val_{qname}_{arm_label}"] = prof["bin_value_median"]
            save_dict[f"bin_n_{qname}_{arm_label}"]   = prof["bin_count"]
    for qname, prof in spread_vs_da.items():
        save_dict[f"spread_bin_da_{qname}"]  = prof["bin_da_median"]
        save_dict[f"spread_bin_val_{qname}"] = prof["bin_value_median"]
        save_dict[f"spread_bin_n_{qname}"]   = prof["bin_count"]

    np.savez_compressed(cache, **save_dict)
    print(f"  saved → {cache}")

    return {
        "percentiles":  percentiles,
        "da_slopes":    da_slopes,
        "da_bins":      da_bins,
        "spread_vs_da": spread_vs_da,
        "log_da":       log_da,
    }

# ---------------------------------------------------------------------------
# Stage F: verdicts + figures
# ---------------------------------------------------------------------------

def stage_f(
    out_dir: Path,
    stage_a_data: dict,
    c_data: Optional[dict],
    d_data: Optional[dict],
    e_data: Optional[dict],
    b2_data: Optional[dict] = None,
    e_ext_data: Optional[dict] = None,
    h2n_data: Optional[dict] = None,
    g_data: Optional[dict] = None,
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

        # ---- [H1-audit] like-for-like relative-to-mean comparison ----
        if "rel_spread_n" in c_data:
            rel_n = c_data["rel_spread_n"]
            rel_q = c_data["rel_spread_q"]
            rel_p = c_data["rel_spread_p"]
            geo_own_d  = float(np.nanmedian(geo_own["depth"]))
            geo_own_tw = float(np.nanmedian(geo_own["top_width"]))
            geo_own_hr = float(np.nanmedian(geo_own["hydraulic_radius"]))
            geo_com_d  = float(np.nanmedian(geo_common["depth"]))  if geo_common else float("nan")
            geo_com_tw = float(np.nanmedian(geo_common["top_width"])) if geo_common else float("nan")
            geo_com_hr = float(np.nanmedian(geo_common["hydraulic_radius"])) if geo_common else float("nan")

            all_geo = [geo_own_d, geo_own_tw, geo_own_hr]
            if geo_common:
                all_geo += [geo_com_d, geo_com_tw, geo_com_hr]
            h1_reverses = all(rel_n > g for g in all_geo if np.isfinite(g))

            print(f"\n[H1-audit] Like-for-like (relative-to-mean) comparison"
                  f" — POST-HOC, discovered 2026-07-07 audit:")
            print(f"  rel-spread(n)={rel_n:.4f}  rel-spread(q)={rel_q:.4f}"
                  f"  rel-spread(p)={rel_p:.4f}")
            print(f"  geometry (arm-own): depth {geo_own_d:.4f} / top_width {geo_own_tw:.4f}"
                  f" / Rh {geo_own_hr:.4f}")
            if geo_common:
                print(f"  geometry (common):  depth {geo_com_d:.4f} / top_width {geo_com_tw:.4f}"
                      f" / Rh {geo_com_hr:.4f}")
            if h1_reverses:
                print("  Under this metric the H1 direction REVERSES:"
                      " n spreads more than every geometry quantity.")
            else:
                print("  Under this metric n does NOT exceed all geometry quantities"
                      " — H1 direction is CONSISTENT with pre-registered metric.")
            am_n = c_data.get("arm_median_n", {})
            am_q = c_data.get("arm_median_q", {})
            am_p = c_data.get("arm_median_p", {})
            if am_n:
                print(f"  per-arm median n: R1={am_n.get('R1', float('nan')):.4f}"
                      f"  R2={am_n.get('R2', float('nan')):.4f}"
                      f"  R3={am_n.get('R3', float('nan')):.4f}")
            if am_q:
                print(f"  per-arm median q: R1={am_q.get('R1', float('nan')):.4f}"
                      f"  R2={am_q.get('R2', float('nan')):.4f}"
                      f"  R3={am_q.get('R3', float('nan')):.4f}")
            if am_p:
                print(f"  per-arm median p: R1={am_p.get('R1', float('nan')):.4f}"
                      f"  R2={am_p.get('R2', float('nan')):.4f}"
                      f"  R3={am_p.get('R3', float('nan')):.4f}")
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
        # NaN rho → INCONCLUSIVE (too few data points or constant input vector)
        h2_supported = np.isfinite(h2_rho) and (h2_rho > 0.2) and contrast
        h2_verdict   = (
            "SUPPORTED"         if h2_supported
            else "INCONCLUSIVE" if not np.isfinite(h2_rho)
            else "REFUTED"
        )
        verdicts["H2"] = h2_verdict
        print(f"\n[H2] n-divergence predicted by inter-source Q' disagreement")
        print(f"  Spearman ρ(n-spread, Q'-disagreement) = {h2_rho:.3f}  (bar: > 0.2)")
        print(f"  n-spread vs geometry contrast: {n_spread:.4f} vs {med_geo_spread:.4f}  contrast={contrast}")
        print(f"  Rule: SUPPORTED iff ρ > 0.2 AND n-spread > geometry-spread")
        print(f"  → H2: {h2_verdict}")

        # ---- [H2-audit] timing-axis disagreement ----
        if b2_data is not None and "per_reach_rel_spread_n" in c_data:
            from scipy.stats import spearmanr as _sp
            n_spr = c_data["per_reach_rel_spread_n"]
            p_r   = b2_data["pearson_r"]
            p_fd  = b2_data["flashiness_diff"]
            one_minus_r = (1.0 - p_r).astype(np.float64)

            h2_rho_corr  = float("nan")
            h2_rho_flash = float("nan")
            valid1 = np.isfinite(n_spr) & np.isfinite(one_minus_r)
            if valid1.sum() > 5:
                h2_rho_corr = float(_sp(n_spr[valid1], one_minus_r[valid1])[0])
            valid2 = np.isfinite(n_spr) & np.isfinite(p_fd)
            if valid2.sum() > 5:
                h2_rho_flash = float(_sp(n_spr[valid2], p_fd[valid2])[0])

            print(f"\n[H2-audit] Timing-axis H2 (Spearman ρ on timing disagreement)"
                  f" — POST-HOC, 2026-07-07 audit:")
            print(f"  registered volume-based ρ(n-spread, Q'-disagreement)"
                  f" = {h2_rho:.3f}  (bar: > 0.2)")
            print(f"  timing-based: ρ(n-rel-spread, 1−pearson_r) = {h2_rho_corr:.3f}")
            print(f"                ρ(n-rel-spread, flashiness-diff) = {h2_rho_flash:.3f}")
            print(f"  n can only compensate timing, not volume;"
                  f" timing axis is the physically relevant test.")
        elif b2_data is None:
            print("\n[H2-audit] SKIPPED — Stage B2 not run")
    else:
        verdicts["H2"] = "INCONCLUSIVE"
        print("\n[H2] INCONCLUSIVE — Stage C not run")

    # ---- [H2-network-audit] network-scale H2 variant (v2 pre-registered) --
    h2n_verdict = "INCONCLUSIVE"
    if h2n_data is not None and c_data is not None:
        rho   = h2n_data["rho"]
        n_g   = h2n_data["n_gauges_used"]
        n_tot = len(stage_a_data["gauge_staids"])
        rel_n = c_data.get("rel_spread_n", float("nan"))
        geo_own = c_data.get("geo_spread_own", {})
        med_geo_rel = (
            float(np.nanmedian([
                np.nanmedian(geo_own["depth"]),
                np.nanmedian(geo_own["top_width"]),
                np.nanmedian(geo_own["hydraulic_radius"]),
            ]))
            if geo_own else float("nan")
        )
        contrast = (np.isfinite(rel_n) and np.isfinite(med_geo_rel)
                    and (rel_n > med_geo_rel))
        if not np.isfinite(rho):
            h2n_verdict = "INCONCLUSIVE"
        elif rho > 0.2 and contrast:
            h2n_verdict = "SUPPORTED"
        else:
            h2n_verdict = "REFUTED"

        print(f"\n[H2-network-audit] Network-scale H2 (per-gauge integrated"
              f" upstream response) — v2 pre-registered, 2026-07-07:")
        print(f"  n_gauges used: {n_g} / {n_tot}"
              f"  (skipped if upstream set has < min_reaches valid reaches)")
        print(f"  Spearman ρ(n_disagreement, timing_disagreement) = {rho:.3f}  (bar: > 0.2)")
        print(f"  n rel-spread vs geometry rel-spread contrast:"
              f" {rel_n:.4f} vs {med_geo_rel:.4f}  contrast={contrast}")
        print(f"  Rule: SUPPORTED iff ρ > 0.2 AND n-rel-spread > geometry-rel-spread;"
              f" REFUTED iff ρ ≤ 0.2 or contrast fails; INCONCLUSIVE if ρ undefined")
        print(f"  → H2_network: {h2n_verdict}  (own v2 verdict — NOT folded into"
              f" the registered H2 REFUTED verdict above)")
    else:
        print("\n[H2-network-audit] SKIPPED — Stage H2-network not run")

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

        # ---- [H3-audit] common-mode-removed cosines + noise ceiling ----
        if e_ext_data:
            print(f"\n[H3-audit] Common-mode-removed cosines (CM = subtract cross-arm mean"
                  f" gradient field) — POST-HOC, 2026-07-07 audit:")
            cm_mean: dict = {}
            for pname in ("n", "q_spatial", "p_spatial"):
                cm_key = f"cm_{pname}"
                if cm_key not in e_ext_data:
                    continue
                cm = e_ext_data[cm_key]
                # Original (pre-CM) cosines for reference
                orig_key_map = {"n": "h3_n_cosines", "q_spatial": "h3_q_cosines",
                                 "p_spatial": "h3_p_cosines"}
                orig_mean = h3[pname]["mean_cosine"]
                cm_vals = [cm["R1R2"], cm["R1R3"], cm["R2R3"]]
                cm_mean_val = float(np.nanmean([v for v in cm_vals if np.isfinite(v)]))
                cm_mean[pname] = cm_mean_val
                print(f"  {pname}: R1-R2={cm['R1R2']:.3f} / R1-R3={cm['R1R3']:.3f}"
                      f" / R2-R3={cm['R2R3']:.3f}  (pre-CM mean: {orig_mean:.3f}"
                      f" → CM mean: {cm_mean_val:.3f})")

            # Assessment. With k=3 arms, CM removal forces residuals to sum to
            # zero per reach, so the NULL pairwise residual cosine is
            # -1/(k-1) = -0.5, not 0. Judge each pair against that null; the
            # R1-R3 pair (distinct stores) is the informative one. Note: CM
            # removal cannot distinguish shared-init common descent from a
            # genuinely source-independent signal — both are common mode.
            print("  NOTE: null for CM-removed pairwise cosines is -1/(k-1) = -0.5"
                  " (residuals sum to zero); values are read as deviations from -0.5,"
                  " and R1-R3 (distinct stores) is the informative pair.")
            n_r13 = e_ext_data.get("cm_n", {}).get("R1R3", float("nan"))
            q_r13 = e_ext_data.get("cm_q_spatial", {}).get("R1R3", float("nan"))
            p_r13 = e_ext_data.get("cm_p_spatial", {}).get("R1R3", float("nan"))
            print(f"  R1-R3 residual structure beyond common mode:"
                  f" n={n_r13:.3f}  q_spatial={q_r13:.3f}  p_spatial={p_r13:.3f}"
                  f"  (≈0 ⇒ that parameter's pre-CM alignment was entirely common mode)")

            # Noise ceiling
            ceiling = e_ext_data.get("ceiling", {})
            if ceiling:
                print("  Within-arm noise ceiling (seed-42 vs seed-123 gradients):")
                for arm in ("R1", "R2", "R3"):
                    if arm not in ceiling:
                        print(f"    [{arm}] SKIPPED — rep file absent")
                        continue
                    ac = ceiling[arm]
                    n_c  = ac.get("n",         float("nan"))
                    q_c  = ac.get("q_spatial",  float("nan"))
                    p_c  = ac.get("p_spatial",  float("nan"))
                    print(f"    [{arm}] n={n_c:.3f}  q_spatial={q_c:.3f}"
                          f"  p_spatial={p_c:.3f}")
                    # Note cross-arm vs ceiling for n (registered cosine)
                    cross_n_mean = h3["n"]["mean_cosine"]
                    if np.isfinite(n_c):
                        print(f"      cross-arm n={cross_n_mean:.3f} vs ceiling={n_c:.3f}"
                              f" (ceiling is interpretability floor)")
            else:
                print("  Within-arm noise ceiling: SKIPPED — rep grad files not provided"
                      " (pass --grads-r*-rep when seed-123 probes are ready)")
        elif e_ext_data is not None:
            pass  # empty dict means stage skipped
        else:
            print("\n[H3-audit] SKIPPED — Stage E ext not run")
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

    # ---- [Level 1.5] distributional + drainage-area conditioning (descriptive) ----
    if g_data is not None:
        print(f"\n[Level 1.5] Per-arm distributions + drainage-area conditioning"
              f" — v2 pre-registered, DESCRIPTIVE ONLY (no falsification bar):")
        print("  Per-arm percentiles (p5/p25/p50/p75/p95):")
        for arm_label in ("R1", "R2", "R3"):
            for pname in ("n", "q_spatial", "p_spatial"):
                pcts = g_data["percentiles"][arm_label][pname]
                print(f"    [{arm_label}] {pname}: "
                      + " / ".join(f"{k}={v:.4f}" for k, v in pcts.items()))
        print("  DA-conditioned ln(quantity) ~ log10(uparea) OLS slopes:")
        for qname, arm_slopes in g_data["da_slopes"].items():
            row = "  ".join(
                f"{arm}={sd['slope']:+.4f}(dropped={sd['n_dropped']})"
                for arm, sd in arm_slopes.items()
            )
            print(f"    {qname:<18} {row}")
        print("  Spread-vs-DA profile (median rel-spread per log10(uparea) decile bin,"
              " headwater→outlet):")
        for qname, prof in g_data["spread_vs_da"].items():
            vals = prof["bin_value_median"]
            finite = vals[np.isfinite(vals)]
            if len(finite):
                print(f"    {qname:<18} first-bin(headwater)={finite[0]:.4f}"
                      f"  last-bin(outlet)={finite[-1]:.4f}"
                      f"  range=[{np.nanmin(finite):.4f}, {np.nanmax(finite):.4f}]")
    else:
        print("\n[Level 1.5] SKIPPED — Stage G not run")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("VERDICT TABLE")
    for h, v in verdicts.items():
        print(f"  {h}: {v}")
    print("=" * 72 + "\n")

    # Write verdicts.json
    verdicts_path = out_dir / "verdicts.json"
    verdicts_out: dict = {
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

    # ---- audit key (separate from registered verdicts — do NOT edit verdicts above) ----
    audit: dict = {}
    if c_data is not None and "rel_spread_n" in c_data:
        geo_own  = c_data["geo_spread_own"]
        geo_com  = c_data.get("geo_spread_common", {})
        audit["H1"] = {
            "rel_spread":       {
                "n": float(c_data["rel_spread_n"]),
                "q": float(c_data["rel_spread_q"]),
                "p": float(c_data["rel_spread_p"]),
            },
            "geo_arm_own_median": {
                "depth":        float(np.nanmedian(geo_own["depth"])),
                "top_width":    float(np.nanmedian(geo_own["top_width"])),
                "hyd_radius":   float(np.nanmedian(geo_own["hydraulic_radius"])),
            },
            "geo_common_median": {
                "depth":        float(np.nanmedian(geo_com["depth"]))     if geo_com else None,
                "top_width":    float(np.nanmedian(geo_com["top_width"])) if geo_com else None,
                "hyd_radius":   float(np.nanmedian(geo_com["hydraulic_radius"])) if geo_com else None,
            },
            "arm_median_n": c_data.get("arm_median_n", {}),
            "arm_median_q": c_data.get("arm_median_q", {}),
            "arm_median_p": c_data.get("arm_median_p", {}),
        }

    if c_data is not None and b2_data is not None and "per_reach_rel_spread_n" in c_data:
        from scipy.stats import spearmanr as _sp2
        n_spr = c_data["per_reach_rel_spread_n"]
        p_r   = b2_data["pearson_r"]
        p_fd  = b2_data["flashiness_diff"]
        one_minus_r = (1.0 - p_r).astype(np.float64)
        h2_rho_corr  = float("nan")
        h2_rho_flash = float("nan")
        valid1 = np.isfinite(n_spr) & np.isfinite(one_minus_r)
        if valid1.sum() > 5:
            h2_rho_corr = float(_sp2(n_spr[valid1], one_minus_r[valid1])[0])
        valid2 = np.isfinite(n_spr) & np.isfinite(p_fd)
        if valid2.sum() > 5:
            h2_rho_flash = float(_sp2(n_spr[valid2], p_fd[valid2])[0])
        audit["H2"] = {
            "volume_rho":           float(c_data["h2_rho"]),
            "timing_rho_pearson":   h2_rho_corr,
            "timing_rho_flashiness": h2_rho_flash,
            "median_pearson_r":     float(np.nanmedian(b2_data["pearson_r"])),
            "median_flashiness_diff": float(np.nanmedian(b2_data["flashiness_diff"])),
        }

    if e_ext_data:
        h3_audit: dict = {"cm_cosines": {}}
        for pname in ("n", "q_spatial", "p_spatial"):
            cm_key = f"cm_{pname}"
            if cm_key in e_ext_data:
                h3_audit["cm_cosines"][pname] = e_ext_data[cm_key]
        ceiling = e_ext_data.get("ceiling", {})
        if ceiling:
            h3_audit["noise_ceiling"] = ceiling
        audit["H3"] = h3_audit

    if h2n_data is not None:
        audit["H2_network"] = {
            "rho":            h2n_data["rho"],
            "verdict":        h2n_verdict,
            "n_gauges_used":  h2n_data["n_gauges_used"],
            "n_gauges_total": len(stage_a_data["gauge_staids"]),
        }

    if g_data is not None:
        audit["Level1_5"] = {
            "percentiles": g_data["percentiles"],
            "da_slopes": {
                qname: {arm: sd["slope"] for arm, sd in arm_slopes.items()}
                for qname, arm_slopes in g_data["da_slopes"].items()
            },
            "da_slopes_n_dropped": {
                qname: {arm: sd["n_dropped"] for arm, sd in arm_slopes.items()}
                for qname, arm_slopes in g_data["da_slopes"].items()
            },
            "spread_vs_da_bin_medians": {
                qname: prof["bin_value_median"].tolist()
                for qname, prof in g_data["spread_vs_da"].items()
            },
            "spread_vs_da_bin_log_da": {
                qname: prof["bin_da_median"].tolist()
                for qname, prof in g_data["spread_vs_da"].items()
            },
        }

    if audit:
        verdicts_out["audit"] = audit

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

    # 1. Headwaters: reaches with no upstream tributaries.
    # Edge convention: local_down[k] receives from local_up[k]; a headwater has no
    # incoming tributaries so its index never appears as local_down[k].
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
    # Gradient NetCDF paths (Stage E / Stage E ext seed-42)
    ap.add_argument("--grads-r1", default=None, type=Path)
    ap.add_argument("--grads-r2", default=None, type=Path)
    ap.add_argument("--grads-r3", default=None, type=Path)
    # Replicate gradient paths (Stage E ext seed-123 noise ceiling; optional)
    ap.add_argument("--grads-r1-rep", default=None, type=Path,
                    help="Replicate (seed-123) gradient NetCDF for R1 — noise ceiling")
    ap.add_argument("--grads-r2-rep", default=None, type=Path,
                    help="Replicate (seed-123) gradient NetCDF for R2 — noise ceiling")
    ap.add_argument("--grads-r3-rep", default=None, type=Path,
                    help="Replicate (seed-123) gradient NetCDF for R3 — noise ceiling")
    # Output
    ap.add_argument("--out",        default=None, help="Output dir (default: <ddrs-root>/output/equif)")
    ap.add_argument("--max-gauges", default=None, type=int,
                    help="Subsample gauges for fast dev run")
    ap.add_argument("--force",      action="store_true",
                    help="Recompute all stages even if cache exists")
    ap.add_argument("--hourly-block-days", default=30, type=int,
                    help="Block size in days for chunked hourly Q' loading (default: 30)."
                         " Use a large value (e.g. 9999) to load in one shot for testing.")
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
                       is_hourly=True, force=args.force,
                       hourly_block_days=args.hourly_block_days)
    # Stage B — common (dHBV2)
    b_common = stage_b(out_dir, a_data, "common", COMMON_STORE,
                       is_hourly=False, force=args.force)

    # Stage B2 — inter-store timing disagreement (audit; new stage)
    b2_data = stage_b2(out_dir, a_data, force=args.force,
                       hourly_block_days=args.hourly_block_days)

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

    # Stage H2-network — network-scale H2 variant (v2 spec; depends on B2 + C)
    h2n_data = None
    if c_data is not None:
        h2n_data = stage_h2_network(out_dir, a_data, c_data, b2_data, force=args.force)
    else:
        print("[Stage H2-network] skipped — Stage C not run")

    # Stage G — Level 1.5 distributions + drainage-area conditioning (v2 spec; depends on C)
    g_data = None
    if c_data is not None:
        g_data = stage_g_level1_5(
            out_dir, a_data, c_data,
            args.params_r1, args.params_r2, args.params_r3,
            b_daily, b_hourly,
            force=args.force,
        )
    else:
        print("[Stage G] skipped — Stage C not run")

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

    # Stage E ext — common-mode-removed H3 + noise ceiling (audit; new stage)
    e_ext_data = stage_e_ext(
        out_dir,
        args.grads_r1, args.grads_r2, args.grads_r3,
        grads_r1_rep=args.grads_r1_rep,
        grads_r2_rep=args.grads_r2_rep,
        grads_r3_rep=args.grads_r3_rep,
        force=args.force,
    )

    # Stage F
    stage_f(out_dir, a_data, c_data, d_data, e_data, b2_data=b2_data, e_ext_data=e_ext_data,
            h2n_data=h2n_data, g_data=g_data)


if __name__ == "__main__":
    main()
