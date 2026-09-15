#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "netCDF4", "matplotlib", "pillow", "scipy", "zarr>=3", "icechunk", "xarray", "pandas"]
# ///
"""One basin's n(d) breathing next to its gauge hydrograph, one frame per day.

Top panel: every reach draining to the gauge, x = log10 drainage area, y =
Manning's n(d) that day (n(d) = n_0·(d/d_ref)^(−gamma), gamma per reach when
learned), coloured by that day's discharge. Bottom panel: the water-year
hydrograph at the gauge — USGS observed, summed Q' (no routing) and the run's
routed flow — with a red dot on the frame's day.

Depth is recomputed from the closed form the solver inverts, from Q' summed
over the reach's upstream set (same approximation as animate_n_of_d.py: not
the routed per-reach flow, which is not exported).

    experiments/stage_roughness/animate_basin_n_of_d.py <run-id> --gauge 01563500 --water-year 2000
"""
import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
from load_ddrs_predictions import load_baseline_f32, load_predictions_zarr  # noqa: E402

spec = importlib.util.spec_from_file_location("anim", HERE / "animate_n_of_d.py")
anim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anim)

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"


def upstream_set(adj_path: str, comid: int):
    """Row indices (in `order`) of every reach draining to `comid`, inclusive."""
    g = zarr.open_group(adj_path, mode="r")
    order = np.asarray(g["order"][:])
    rows, cols = np.asarray(g["indices_0"][:]), np.asarray(g["indices_1"][:])
    pos = {int(x): i for i, x in enumerate(order)}
    up = defaultdict(list)
    for a, b in zip(rows, cols):
        up[a].append(b)
    seen, stack = {pos[comid]}, [pos[comid]]
    while stack:
        x = stack.pop()
        for y in up[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    keep = np.array(sorted(seen))
    sub_edges = [(a, b) for a, b in zip(rows, cols) if a in seen and b in seen]
    return order[keep], keep, sub_edges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--gauge", default="01563500")
    ap.add_argument("--water-year", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=366)
    a = ap.parse_args()
    run = RUNS / a.run_id
    cfg_text = (run / "config.yaml").read_text()
    meta = {r["STAID"]: r for r in csv.DictReader(open(GAGES))}
    gauge_comid = int(meta[a.gauge]["COMID"])
    name = meta[a.gauge]["STANAME"]

    import re
    adj_path = re.search(r"^\s+conus_adjacency:\s*(\S+)", cfg_text, re.M).group(1)
    comids, _rows, edges = upstream_set(adj_path, gauge_comid)
    n = comids.size
    print(f"{a.gauge} {name}: {n} reaches drain to COMID {gauge_comid}")

    ds = Dataset(run / "plot" / "kan_parameters.nc")
    all_comid = np.asarray(ds["COMID"][:], dtype=np.int64)
    idx = {int(c): i for i, c in enumerate(all_comid)}
    sel = np.array([idx[int(c)] for c in comids])
    n0 = np.asarray(ds["n"][:], dtype=np.float64)[sel]
    p = np.asarray(ds["p_spatial"][:], dtype=np.float64)[sel]
    qs = np.asarray(ds["q_spatial"][:], dtype=np.float64)[sel]
    slope = np.asarray(ds["slope"][:], dtype=np.float64)[sel]
    if "gamma" in ds.variables:
        gamma = np.asarray(ds["gamma"][:], dtype=np.float64)[sel]
        d_ref = 1.0
        glabel = f"gamma learned per reach (basin median {np.median(gamma):.3f})"
    else:
        gamma, d_ref = anim.read_config_scalars(cfg_text)
        glabel = f"gamma = {gamma}"
    log_area = anim.load_log10_uparea(cfg_text, comids)

    start, end = f"{a.water_year - 1}-10-01", f"{a.water_year}-10-01"
    q_local = anim.load_qprime(cfg_text, comids, start, end)  # (T, n)
    # Accumulate within the basin: q_up = (I - A)^-1 q_local on the sub-network.
    from scipy.sparse import csr_matrix, eye
    from scipy.sparse.linalg import spsolve_triangular
    loc = {int(r): i for i, r in enumerate(_rows)}
    er = [loc[x] for x, _ in edges]
    ec = [loc[y] for _, y in edges]
    A = csr_matrix((np.ones(len(er)), (er, ec)), shape=(n, n))
    # rows are in topological (upstream-first) order because `order` is
    # topologically sorted and we kept its ordering, so I - A is lower triangular.
    M = (eye(n, format="csr") - A).tocsr()
    q = np.vstack([spsolve_triangular(M, q_local[t], lower=True, unit_diagonal=True) for t in range(q_local.shape[0])])
    depth = anim.depth_from_discharge(q, n0, p, qs, slope, gamma, d_ref)
    n_t = anim.manning_n(depth, n0, gamma, d_ref)
    print(f"n(d): min {n_t.min():.4f} median {np.median(n_t):.4f} max {n_t.max():.4f}; "
          f"per-reach breathing median {np.median(n_t.max(0) / n_t.min(0)):.2f}x")

    # Hydrograph at the gauge over the water year.
    pz = load_predictions_zarr(run / "eval" / "predictions.zarr").sel(gage_ids=a.gauge, time=slice(start, f"{a.water_year}-09-30"))
    bz = load_baseline_f32(run / "baseline").sel(gage_ids=a.gauge, time=slice(start, f"{a.water_year}-09-30"))
    t = pd.to_datetime(pz.time.values)
    obs = np.where(pz.observations.values <= 0, np.nan, pz.observations.values)
    routed = pz.predictions.values
    summed = bz.predictions.values
    tb = pd.to_datetime(bz.time.values)
    days = pd.to_datetime(pd.date_range(start, periods=q.shape[0], freq="D"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out = run / "plots" / f"basin_{a.gauge}_n_of_d_wy{a.water_year}.gif"
    ylo, yhi = n_t.min() * 0.95, n_t.max() * 1.05
    qlo, qhi = np.log10(np.percentile(np.maximum(q, 1e-3), [1, 99]))
    xlo, xhi = np.nanmin(log_area) - 0.1, np.nanmax(log_area) + 0.1
    step = max(1, int(np.ceil(q.shape[0] / a.max_frames)))
    frames = []
    for k in range(0, q.shape[0], step):
        fig, (ax, hx) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [1.15, 1]})
        sc = ax.scatter(log_area, n_t[k], c=np.log10(np.maximum(q[k], 1e-3)), s=28, cmap="viridis",
                        vmin=qlo, vmax=qhi, edgecolor="k", linewidth=0.3)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("log10 drainage area (km²)"); ax.set_ylabel("Manning's n(d)")
        ax.set_title(f"{a.gauge} {name}: {n} reaches, water year {a.water_year}, "
                     f"{days[k].strftime('%Y-%m-%d')}\n{glabel}, discharge = summed Q' within the basin", fontsize=10)
        fig.colorbar(sc, ax=ax, label="log10 discharge today (m³/s)")
        hx.plot(t, obs, "k-", lw=1.2, label="USGS observed")
        hx.plot(tb, summed, color="tab:orange", ls="--", lw=0.9, label="summed Q' (no routing)")
        hx.plot(t, routed, color="tab:blue", lw=0.9, label="routed")
        d = days[k]
        j = int(np.searchsorted(t.values, np.datetime64(d)))
        if j < routed.size:
            hx.plot([d], [routed[j]], "o", color="red", ms=8, zorder=5)
            hx.axvline(d, color="red", lw=0.8, alpha=0.6)
        hx.set_ylabel("discharge (m³/s)"); hx.set_xlabel(f"water year {a.water_year}")
        hx.legend(fontsize=8, loc="upper right"); hx.grid(alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0)
    print(f"wrote {out} ({len(frames)} frames)")
    small = out.with_name(out.stem + "_small.gif")
    sm = [f.resize((int(f.width * 0.7), int(f.height * 0.7)), Image.LANCZOS).quantize(colors=128) for f in frames[::2]]
    sm[0].save(small, save_all=True, append_images=sm[1:], duration=int(2000 / a.fps), loop=0, optimize=True)
    print(f"wrote {small} ({len(sm)} frames, {small.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
