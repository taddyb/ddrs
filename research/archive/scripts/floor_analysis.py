#!/usr/bin/env python3
"""Phase B floor curves: transient noise floor vs warmup, by rho and basin size.

Pre-registered (spec §4): the fix must reach floor <= 0.25 mean L1.
Decision rule: if any (warmup <= 60, rho <= 180) cell reaches the bar ->
option A (config-only). Else -> STOP; option B (state-cache hotstart) gets
its own plan informed by the decay curves printed here.

The 58 planted-reach signals contribute ~0.0076 background (measured
continuously); irrelevant at the 0.25 bar but printed for honesty.

ADVANCE-FINDING EXTENSIONS (beyond the plan script):
  - Median-by-day plotted alongside mean-by-day (dashed) on the decay plot.
  - Large-uparea stratum: day-mean printed at days {0, 5, 30, 60, 87} so we
    can see whether big rivers decay within the 88-day window.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

OUT = Path("/home/tbindas/projects/ddrs/output/floor_fix")
BAR = 0.25
WARMUPS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150]

# Days at which we probe the large-stratum daily mean (rho90 index space; rho180
# probed at same absolute days where available).
LARGE_PROBE_DAYS = [0, 5, 30, 60, 87]


def curves(path):
    ds = xr.open_dataset(path)
    r = ds["abs_residual"].values          # (win, slot, day)
    ua = ds["uparea"].values               # (win, slot)
    n_days = r.shape[2]
    rows = []
    # uparea terciles over finite entries (falls back gracefully if uparea all-NaN)
    finite_ua = ua[np.isfinite(ua)]
    edges = (np.percentile(finite_ua, [33, 66]) if finite_ua.size else [np.inf, np.inf])
    strata = {
        "all": np.ones_like(ua, dtype=bool),
        "small": ua <= edges[0],
        "mid": (ua > edges[0]) & (ua <= edges[1]),
        "large": ua > edges[1],
    }
    for w in WARMUPS:
        if w >= n_days - 5:
            continue
        for name, m in strata.items():
            sel = r[:, :, w:][m[:, :, None] & np.ones((1, 1, n_days - w), bool)]
            rows.append((n_days, w, name, float(np.nanmean(sel)),
                         float(np.nanmedian(sel))))
    return rows, edges, ua, r, n_days


all_rows = []
datasets = {}
for f in ["floor_rho90.nc", "floor_rho180.nc"]:
    rows, edges, ua, r, n_days = curves(OUT / f)
    all_rows += rows
    datasets[f] = {"edges": edges, "ua": ua, "r": r, "n_days": n_days}

print(f"{'rho_days':>8} {'warmup':>6} {'stratum':>7} {'mean_L1':>9} {'median_L1':>9}")
passing = []
for n_days, w, s, mean, med in all_rows:
    flag = " <-- PASSES BAR" if (s == "all" and mean <= BAR) else ""
    print(f"{n_days:>8} {w:>6} {s:>7} {mean:>9.4f} {med:>9.4f}{flag}")
    if s == "all" and mean <= BAR:
        passing.append((n_days, w, mean))

# ---------------------------------------------------------------------------
# Large-stratum day-probe: do big rivers decay at all within 88 days?
# ---------------------------------------------------------------------------
print()
print("=== Large-stratum day-probe (mean |residual| at selected days) ===")
for fname, info in datasets.items():
    r = info["r"]
    ua = info["ua"]
    n_days = info["n_days"]
    edges = info["edges"]
    mask_large = ua > edges[1]
    days_to_probe = [d for d in LARGE_PROBE_DAYS if d < n_days]
    r_large = r.copy()
    r_large[~(mask_large[:, :, np.newaxis] * np.ones((1, 1, n_days), dtype=bool))] = np.nan
    day_mean_large = np.nanmean(r_large, axis=(0, 1))
    uparea_min = ua[mask_large].min() if mask_large.any() else float("nan")
    uparea_max = ua[mask_large].max() if mask_large.any() else float("nan")
    print(f"  {fname} (large = uparea > {edges[1]:.0f} km², range {uparea_min:.0f}–{uparea_max:.0f} km²):")
    for d in days_to_probe:
        print(f"    day {d:3d}: {day_mean_large[d]:.4f} m³/s")

# ---------------------------------------------------------------------------
# Per-day decay curves: mean AND median (dashed), semilogy, both rho values.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"floor_rho90.nc": "tab:blue", "floor_rho180.nc": "tab:orange"}
for f, label in [("floor_rho90.nc", "rho 90"), ("floor_rho180.nc", "rho 180")]:
    r = datasets[f]["r"]
    day_mean = np.nanmean(r, axis=(0, 1))
    day_med = np.nanmedian(r, axis=(0, 1))
    c = colors[f]
    ax.semilogy(day_mean, color=c, label=f"{label} mean")
    ax.semilogy(day_med, color=c, linestyle="--", label=f"{label} median")
ax.axhline(BAR, color="r", ls="--", label=f"bar {BAR}")
ax.axhline(0.0076, color="g", ls=":", label="continuous residual (plants)")
ax.set(xlabel="post-trim day in window", ylabel="mean / median |residual| m³/s",
       title="Hotstart transient decay — mean (solid) vs median (dashed)\n"
             "teacher weights, self-generated obs")
ax.legend()
fig.savefig(OUT / "floor_decay.png", dpi=200, bbox_inches="tight")
print(f"\nplot -> {OUT / 'floor_decay.png'}")

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if passing:
    n_days, w, mean = min(passing, key=lambda t: (t[0], t[1]))
    # effective loss-days per window after trim:
    print(f"DECISION: OPTION A — rho yielding {n_days} post-trim days with "
          f"warmup {w} reaches floor {mean:.4f} <= {BAR}")
    print(f"loss-days per window: {n_days - w} (sample-efficiency note for Phase C)")
else:
    best = min((x for x in all_rows if x[2] == "all"), key=lambda t: t[3])
    print(f"DECISION: OPTION B REQUIRED — best config-only floor is "
          f"{best[3]:.4f} (rho-days {best[0]}, warmup {best[1]}) > {BAR}. "
          f"STOP: write the state-cache plan using the decay curves above.")
print(f"plot -> {OUT / 'floor_decay.png'}")
