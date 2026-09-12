#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "netCDF4", "matplotlib", "pillow", "scipy", "zarr>=3", "icechunk"]
# ///
"""Animate stage-dependent Manning roughness over a water year.

With `params.stage_roughness`, Manning's n is no longer a fixed per-reach
number. It breathes with the flow:

    n(d) = n_0 · (d / d_ref)^(−gamma)

so every reach is rougher at low flow and smoother in flood. This renders that
as a GIF: one frame per day, the network coloured by n(d).

**Where the depth comes from, and the one approximation.** Nothing exports
per-reach daily depth, so it is recomputed here from the same closed form the
solver inverts (`src/geometry.rs`):

    d = ( Q · n_0 · (q+1) · d_ref^gamma / (p · √S) ) ^ ( 3 / (5 + 3q + 3·gamma) )

`Q` is the discharge each reach carries. The honest version accumulates the
upstream Q' through the network; `--discharge local` uses each reach's own Q'
instead, which is much faster and fine for small networks where routing barely
redistributes, but understates main-stem flow badly on CONUS. The mode used is
stamped on every frame so a figure cannot lose its caveat.

`gamma` is per reach when the run learned it (`gamma` variable in
`plot/kan_parameters.nc`, written by `dump_parameters` for heads that emit it),
else the global `params.stage_roughness.gamma` from the config snapshot.

Run it on a run directory that has `plot/kan_parameters.nc` (i.e. one trained
with `--plot`):

    experiments/stage_roughness/animate_n_of_d.py <run-dir> --water-year 2000
    experiments/stage_roughness/animate_n_of_d.py <run-dir> --traces      # no map

Views (`--view`):

    area     (default) x = log10 drainage area, y = n(d), one frame per day,
             points coloured by discharge, with the per-area-bin median n(d)
             drawn on top. Reads `log10_uparea` from the attributes NetCDF.
    scatter  the older (log10 median discharge, n_0) layout coloured by n(d).
    3d       a static surface: x = log10 drainage area (binned), y = day of
             water year, z = median n(d) in the bin. PNG, not a GIF.

`--traces` skips the map and plots n(d) through time for a handful of reaches,
which is the cheaper and often more legible view of the same thing.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def read_config_scalars(cfg_text: str) -> tuple[float, float]:
    """`(gamma, d_ref)` from a run's config snapshot. Absent block ⇒ (0, 1)."""
    if "stage_roughness:" not in cfg_text:
        return 0.0, 1.0
    block = cfg_text.split("stage_roughness:", 1)[1]
    g = re.search(r"^\s+gamma:\s*([0-9.eE+-]+)", block, re.M)
    d = re.search(r"^\s+d_ref:\s*([0-9.eE+-]+)", block, re.M)
    return (float(g.group(1)) if g else 0.0, float(d.group(1)) if d else 1.0)


def depth_from_discharge(q, n0, p, qs, slope, gamma, d_ref, depth_lb=0.01):
    """Mirrors `compute_trapezoidal_geometry_gamma` in src/geometry.rs.

    `gamma` is a scalar or a per-reach array (broadcast along the time axis).
    """
    q_eps = qs + 1e-6
    num = np.maximum(q, 1e-4) * n0 * (q_eps + 1.0)
    if np.any(gamma != 0.0) and d_ref != 1.0:
        num = num * d_ref**gamma
    den = p * np.sqrt(np.maximum(slope, 1e-8)) + 1e-8
    expo = 3.0 / (5.0 + 3.0 * q_eps + 3.0 * gamma)
    return np.maximum((num / den) ** expo, depth_lb)


def manning_n(depth, n0, gamma, d_ref):
    if not np.any(gamma != 0.0):
        return np.broadcast_to(n0, depth.shape)
    return n0 * (depth / d_ref) ** (-gamma)


def load_log10_uparea(cfg_text: str, comids: np.ndarray) -> np.ndarray | None:
    """`log10_uparea` per COMID from the attributes NetCDF named in the config."""
    m = re.search(r"^\s+attributes:\s*(\S+)", cfg_text, re.M)
    if not m:
        return None
    try:
        ds = Dataset(m.group(1))
        ac = np.asarray(ds["COMID"][:], dtype=np.int64)
        ua = np.asarray(ds["log10_uparea"][:], dtype=np.float64)
        pos = {int(c): i for i, c in enumerate(ac)}
        idx = np.array([pos.get(int(c), -1) for c in comids])
        out = np.full(comids.size, np.nan)
        out[idx >= 0] = ua[idx[idx >= 0]]
        return out
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not read log10_uparea: {e}")
        return None


def binned_median(x, y, edges):
    """Median of `y` in each `x` bin; NaN where a bin is empty."""
    which = np.digitize(x, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        sel = which == b
        if sel.sum() >= 20:
            out[b] = np.median(y[sel])
    return out


def load_qprime(cfg_text: str, comids: np.ndarray, start: str, end: str) -> np.ndarray | None:
    """Daily Q' for `comids` over [start, end), shape (T, N). None if unreadable."""
    m = re.search(r"^\s+streamflow:\s*(\S+)", cfg_text, re.M)
    if not m:
        return None
    path = m.group(1)
    try:
        import icechunk
        import zarr

        repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
        grp = zarr.open_group(repo.readonly_session("main").store, mode="r")
        arr = grp["Qr"]
        divide = np.asarray(grp["divide_id"][:])
        time = grp["time"]
        units = time.attrs.get("units", "days since 1980-01-01")
        origin = np.datetime64(units.split("since")[-1].strip().split()[0])
        days = origin + np.asarray(time[:]).astype("timedelta64[D]")
        i0 = int(np.searchsorted(days, np.datetime64(start)))
        i1 = int(np.searchsorted(days, np.datetime64(end)))
        pos = {c: i for i, c in enumerate(divide)}
        cols = np.array([pos.get(int(c), -1) for c in comids])
        ok = cols >= 0
        out = np.full((i1 - i0, comids.size), 1e-3, dtype=np.float64)
        # Qr is (divide_id, time) or (time, divide_id); sniff by shape.
        block = arr[:, i0:i1] if arr.shape[0] == divide.size else arr[i0:i1, :].T
        out[:, ok] = np.asarray(block)[cols[ok], :].T
        return out
    except Exception as e:  # noqa: BLE001 - a missing store is a skip, not a crash
        print(f"  ! could not read Q' from {path}: {e}")
        return None


def accumulate_upstream(q_local: np.ndarray, cfg_text: str, comids: np.ndarray) -> np.ndarray:
    """Route Q' downstream by topological accumulation. Falls back to local."""
    m = re.search(r"^\s+conus_adjacency:\s*(\S+)", cfg_text, re.M)
    if not m:
        print("  ! no conus_adjacency in config; using LOCAL discharge")
        return q_local
    try:
        import zarr
        from scipy.sparse import csr_matrix, eye
        from scipy.sparse.linalg import spsolve_triangular

        g = zarr.open_group(m.group(1), mode="r")
        order = np.asarray(g["order"][:])
        rows = np.asarray(g["indices_0"][:])
        cols = np.asarray(g["indices_1"][:])
        n = order.size
        pos = {int(c): i for i, c in enumerate(order)}
        keep = np.array([pos.get(int(c), -1) for c in comids])
        if (keep < 0).any():
            print("  ! parameter COMIDs not all in adjacency; using LOCAL")
            return q_local
        a = csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
        mat = (eye(n, format="csr") - a).tocsr()
        full = np.full((q_local.shape[0], n), 1e-3)
        full[:, keep] = q_local
        out = np.empty_like(full)
        for t in range(full.shape[0]):
            out[t] = spsolve_triangular(mat, full[t], lower=True, unit_diagonal=True)
        return out[:, keep]
    except Exception as e:  # noqa: BLE001
        print(f"  ! accumulation failed ({e}); using LOCAL discharge")
        return q_local


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--water-year", type=int, default=1996)
    ap.add_argument("--discharge", choices=["accumulated", "local"], default="accumulated")
    ap.add_argument("--traces", action="store_true", help="line plot instead of a map")
    ap.add_argument("--view", choices=["area", "scatter", "3d"], default="area")
    ap.add_argument("--n-traces", type=int, default=6)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=366, help="subsample days beyond this")
    ap.add_argument("--max-points", type=int, default=80_000, help="reaches drawn per frame")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    run = a.run_dir
    nc = run / "plot" / "kan_parameters.nc"
    if not nc.exists():
        print(f"no {nc} — retrain with `--plot`")
        return 1
    cfg_text = (run / "config.yaml").read_text()
    ds = Dataset(nc)
    comids = np.asarray(ds["COMID"][:], dtype=np.int64)
    n0 = np.asarray(ds["n"][:], dtype=np.float64)
    p = np.asarray(ds["p_spatial"][:], dtype=np.float64)
    qs = np.asarray(ds["q_spatial"][:], dtype=np.float64)
    slope = np.asarray(ds["slope"][:], dtype=np.float64)
    print(f"{comids.size:,} reaches")

    if "gamma" in ds.variables:
        gamma = np.asarray(ds["gamma"][:], dtype=np.float64)
        d_ref = 1.0  # learned gamma requires no stage_roughness block, so d_ref is 1
        gamma_label = (f"gamma learned per reach (median {np.median(gamma):.3f}, "
                       f"p10 {np.percentile(gamma, 10):.3f}, p90 {np.percentile(gamma, 90):.3f})")
    else:
        gamma, d_ref = read_config_scalars(cfg_text)
        gamma_label = f"gamma = {gamma}"
        if gamma == 0.0:
            print("  NOTE gamma = 0, so n(d) is constant in time and the animation")
            print("       will show a still frame. That is the correct answer for")
            print("       this run, not a bug.")
    print(f"{gamma_label}, d_ref = {d_ref}")

    start, end = f"{a.water_year - 1}-10-01", f"{a.water_year}-10-01"
    q_local = load_qprime(cfg_text, comids, start, end)
    if q_local is None:
        print("no Q' available; cannot animate")
        return 1
    q = q_local if a.discharge == "local" else accumulate_upstream(q_local, cfg_text, comids)
    print(f"{q.shape[0]} days, discharge mode = {a.discharge}")

    # ~42k CONUS fabric reaches have no Q' prediction and read as the 1e-3
    # fill (CLAUDE.md, data sources). Their discharge never moves, so their
    # n(d) is a flat line: including them puts dead reaches in the traces and
    # drags the breathing ratio toward 1. Keep only reaches whose discharge
    # actually varies.
    # Ask the question directly: does this reach's flow actually move over the
    # year? A relative-std threshold is too weak — numerical noise from the
    # accumulation solve sneaks dead reaches through at std/mean ~ 1e-9.
    # Two conditions, both needed. (1) Does the flow actually move over the
    # year? A relative-std threshold is too weak — noise from the accumulation
    # solve sneaks dead reaches through at std/mean ~ 1e-9. (2) Is the flow
    # physically meaningful? Reaches carrying 1e-9 m3/s sit pinned at the depth
    # floor, so their n(d) is an artefact of the clamp, not of hydraulics.
    Q_FLOOR = 1e-3  # m3/s, one litre per second
    live = ((q.max(axis=0) / np.maximum(q.min(axis=0), 1e-30)) > 1.01) & (
        np.median(q, axis=0) > Q_FLOOR
    )
    n_dead = int((~live).sum())
    if n_dead:
        print(f"  excluding {n_dead:,} reaches ({100 * n_dead / live.size:.1f}%) with no Q' "
              f"prediction or median flow below {Q_FLOOR} m3/s")

    depth = depth_from_discharge(q, n0, p, qs, slope, gamma, d_ref)
    n_t = manning_n(depth, n0, gamma, d_ref)
    nl = n_t[:, live]
    print(f"n(d) over live reaches: min {nl.min():.4f}  median {np.median(nl):.4f}  max {nl.max():.4f}")
    print(f"  ratio of network-median n at its highest vs lowest day: "
          f"{np.median(nl, axis=1).max() / np.median(nl, axis=1).min():.3f}x")
    per_reach = nl.max(axis=0) / np.maximum(nl.min(axis=0), 1e-12)
    print(f"  per-reach breathing, median {np.median(per_reach):.3f}x, "
          f"p90 {np.percentile(per_reach, 90):.3f}x")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = a.out or run / "plots" / (
        f"n_of_d_wy{a.water_year}" + ("_traces.png" if a.traces else f"_{a.view}.gif")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    days = np.arange(q.shape[0])

    if a.traces:
        # Pick LIVE reaches spanning the discharge range, so the spread is
        # visible and no flat sentinel lines get in.
        live_idx = np.flatnonzero(live)
        rank = live_idx[np.argsort(np.median(q[:, live_idx], axis=0))]
        pick = rank[np.linspace(0, rank.size - 1, a.n_traces).astype(int)]
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for i in pick:
            ax[0].plot(days, q[:, i], lw=0.9, label=f"COMID {comids[i]}")
            ax[1].plot(days, n_t[:, i], lw=0.9)
        ax[0].set_yscale("log")
        ax[0].set_ylabel("discharge (m³/s)")
        ax[0].legend(fontsize=7, ncol=2)
        ax[1].set_ylabel("Manning's n(d)")
        ax[1].set_xlabel(f"day of water year {a.water_year}")
        ax[0].set_title(
            f"roughness breathing with flow — {gamma_label}, "
            f"discharge = {a.discharge}"
        )
        for x in ax:
            x.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150, facecolor="white")
        print(f"wrote {out}")
        return 0

    from PIL import Image

    live_idx = np.flatnonzero(live)
    step = max(1, int(np.ceil(q.shape[0] / a.max_frames)))
    frame_days = range(0, q.shape[0], step)

    if a.view in ("area", "3d"):
        log_area = load_log10_uparea(cfg_text, comids)
        if log_area is None:
            print("no attributes NetCDF with log10_uparea; falling back to --view scatter")
            a.view = "scatter"
        else:
            ok = np.isfinite(log_area[live_idx])
            live_idx = live_idx[ok]
            print(f"{live_idx.size:,} live reaches with a drainage area")

    if a.view == "3d":
        # Surface of median n(d): x = log10 drainage area (bins), y = day of
        # water year, z = median n(d) over the reaches in that bin on that day.
        # A median per bin is what the eye can read; the raw cloud is 3e5
        # points per day and reads as noise in 3D.
        la = log_area[live_idx]
        edges = np.linspace(np.percentile(la, 1), np.percentile(la, 99), 25)
        centers = 0.5 * (edges[1:] + edges[:-1])
        surf = np.array([binned_median(la, n_t[t, live_idx], edges) for t in days])  # (T, bins)
        out = a.out or run / "plots" / f"n_of_d_wy{a.water_year}_3d.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        xx, yy = np.meshgrid(centers, days)
        ax.plot_surface(xx, yy, surf, cmap="plasma_r", linewidth=0, antialiased=True, alpha=0.95)
        ax.set_xlabel("log10 drainage area (km²)")
        ax.set_ylabel(f"day of water year {a.water_year}")
        ax.set_zlabel("median Manning's n(d) in bin")
        ax.set_title(f"n(d) by river size through the year\n{gamma_label}, discharge = {a.discharge}")
        ax.view_init(elev=28, azim=-135)
        fig.tight_layout()
        fig.savefig(out, dpi=150, facecolor="white")
        print(f"wrote {out}")
        # Also the flat version, which is easier to read numbers off.
        out2 = out.with_name(out.stem + "_heatmap.png")
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.pcolormesh(centers, days, surf, cmap="plasma_r", shading="nearest")
        ax.set_xlabel("log10 drainage area (km²)")
        ax.set_ylabel(f"day of water year {a.water_year}")
        ax.set_title(f"median n(d) per drainage-area bin — {gamma_label}, discharge = {a.discharge}")
        fig.colorbar(im, label="median Manning's n(d)")
        fig.tight_layout()
        fig.savefig(out2, dpi=150, facecolor="white")
        print(f"wrote {out2}")
        return 0

    if a.view == "area":
        # x = log10 drainage area, y = n(d) on day t. Fixed axes across frames
        # so the motion is the signal; colour = discharge that day; the black
        # line is the per-bin median, the thing to watch.
        rng = np.random.default_rng(0)
        draw = live_idx if live_idx.size <= a.max_points else rng.choice(live_idx, a.max_points, replace=False)
        la = log_area[draw]
        la_all = log_area[live_idx]
        edges = np.linspace(np.percentile(la_all, 1), np.percentile(la_all, 99), 30)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ylo, yhi = np.percentile(n_t[:, live_idx], [0.5, 99.5])
        qlo, qhi = np.log10(np.percentile(np.maximum(q[:, draw], 1e-3), [1, 99]))
        env_lo = binned_median(la_all, n_t[:, live_idx].min(axis=0), edges)
        env_hi = binned_median(la_all, n_t[:, live_idx].max(axis=0), edges)
        frames = []
        for t in frame_days:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            sc = ax.scatter(la, n_t[t, draw], c=np.log10(np.maximum(q[t, draw], 1e-3)),
                            s=2, alpha=0.35, cmap="viridis", vmin=qlo, vmax=qhi, rasterized=True)
            ax.fill_between(centers, env_lo, env_hi, color="0.6", alpha=0.25, lw=0,
                            label="year envelope of per-bin median (driest .. wettest day)")
            ax.plot(centers, binned_median(la_all, n_t[t, live_idx], edges), "k-", lw=2,
                    label="per-bin median n(d) today")
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(ylo, yhi)
            ax.set_xlabel("log10 drainage area (km²)")
            ax.set_ylabel("Manning's n(d)")
            ax.set_title(f"n(d) vs river size — water year {a.water_year}, day {t:3d}\n"
                         f"{gamma_label}, discharge = {a.discharge}")
            ax.legend(loc="upper right", fontsize=8)
            fig.colorbar(sc, label="log10 discharge today (m³/s)")
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / a.fps), loop=0)
        print(f"wrote {out}  ({len(frames)} frames)")
        return 0

    # Scatter view: the reaches are laid out by (discharge rank, n_0), coloured
    # by n(d); reads as a network gradient and needs no shapefile.
    vmin, vmax = np.percentile(n_t[:, live], [2, 98])
    frames = []
    x = np.log10(np.maximum(np.median(q[:, live_idx], axis=0), 1e-3))
    y = n0[live_idx]
    for t in frame_days:
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(x, y, c=n_t[t, live_idx], s=6, cmap="plasma_r", vmin=vmin, vmax=vmax)
        ax.set_xlabel("log10 median discharge (m³/s)")
        ax.set_ylabel("n₀ (roughness at d_ref)")
        ax.set_title(
            f"n(d), water year {a.water_year}, day {t}\n"
            f"{gamma_label}, discharge = {a.discharge}"
        )
        fig.colorbar(sc, label="Manning's n(d)")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=int(1000 / a.fps), loop=0,
    )
    print(f"wrote {out}  ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
