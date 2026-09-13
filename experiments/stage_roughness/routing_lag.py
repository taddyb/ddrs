#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "xarray", "zarr>=3", "matplotlib", "pandas"]
# ///
"""Is there a reasonable lag between the summed Q' and the routed flow?

Routing should delay and attenuate the summed lateral inflow; the question is
whether the delay the trained model produces is of the size the observations
ask for. For every eval gauge this measures three lags by cross-correlating
daily anomalies over the eval window, in days, positive = the second series
is later:

    lag(summed Q' -> routed)    what the router adds
    lag(summed Q' -> observed)  what the gauge asks for
    lag(routed    -> observed)  what is left over

and plots them against drainage area, plus three example hydrographs (small,
medium, large basin) over one water year with all three series on one axis.

    experiments/stage_roughness/routing_lag.py <run-id> [--water-year 2000]

Writes `<run>/plots/routing_lag.png`, `<run>/plots/routing_lag_examples.png`
and prints a summary table. The lag is resolved to whole days because the
eval output is daily; a sub-day lag reads as 0.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
from load_ddrs_predictions import load_baseline_f32, load_predictions_zarr  # noqa: E402

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"
LAGS = np.arange(-3, 11)


def best_lag(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    """Lag (days) that maximises corr(a(t), b(t + lag)); NaN-safe."""
    best, best_r = 0, -np.inf
    for lag in LAGS:
        if lag >= 0:
            x, y = a[: a.size - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: b.size + lag]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 365:
            continue
        x, y = x[m], y[m]
        r = np.corrcoef(x, y)[0, 1]
        if r > best_r:
            best, best_r = int(lag), float(r)
    return best, best_r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--water-year", type=int, default=2000)
    a = ap.parse_args()
    run = RUNS / a.run_id
    ds = load_predictions_zarr(run / "eval" / "predictions.zarr")
    dsb = load_baseline_f32(run / "baseline")
    gauges = np.intersect1d(ds.gage_ids.values, dsb.gage_ids.values)
    t0, t1 = max(ds.time.values[0], dsb.time.values[0]), min(ds.time.values[-1], dsb.time.values[-1])
    ds = ds.sel(gage_ids=gauges, time=slice(t0, t1))
    dsb = dsb.sel(gage_ids=gauges, time=slice(t0, t1))
    meta = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")
    area = meta["DRAIN_SQKM"].reindex(gauges).values
    print(f"{gauges.size} gauges, {ds.time.size} days, {str(t0)[:10]}..{str(t1)[:10]}")

    routed = ds.predictions.values
    obs = ds.observations.values
    obs = np.where(obs <= 0, np.nan, obs)
    summed = dsb.predictions.values
    rows = []
    for i in range(gauges.size):
        l_sr, r_sr = best_lag(summed[i], routed[i])
        l_so, r_so = best_lag(summed[i], obs[i])
        l_ro, r_ro = best_lag(routed[i], obs[i])
        rows.append((gauges[i], area[i], l_sr, r_sr, l_so, r_so, l_ro, r_ro))
    df = pd.DataFrame(rows, columns=["gauge", "area", "lag_sum_routed", "r_sum_routed",
                                     "lag_sum_obs", "r_sum_obs", "lag_routed_obs", "r_routed_obs"])
    df = df[np.isfinite(df.area)]
    df["la"] = np.log10(df.area)
    df.to_csv(run / "plots" / "routing_lag.csv", index=False) if (run / "plots").exists() else None

    print("\nlag in days (median [p10, p90]) by drainage-area class:")
    print(f"{'area km2':>16} {'n':>5} {'sum->routed':>14} {'sum->obs':>14} {'routed->obs':>14}  "
          f"{'r(sum,obs)':>10} {'r(routed,obs)':>13}")
    edges = [0, 300, 1000, 3000, 10000, 30000, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = df[(df.area >= lo) & (df.area < hi)]
        if d.empty:
            continue
        q = lambda c: f"{d[c].median():.0f} [{d[c].quantile(.1):.0f}, {d[c].quantile(.9):.0f}]"  # noqa: E731
        print(f"{f'{lo:.0f}-{hi:.0f}':>16} {len(d):>5} {q('lag_sum_routed'):>14} {q('lag_sum_obs'):>14} "
              f"{q('lag_routed_obs'):>14}  {d.r_sum_obs.median():>10.3f} {d.r_routed_obs.median():>13.3f}")
    same = (df.lag_sum_routed == df.lag_sum_obs).mean()
    print(f"\nrouter lag equals the lag the gauge asks for at {100 * same:.1f}% of gauges; "
          f"routed->obs residual lag is 0 at {100 * (df.lag_routed_obs == 0).mean():.1f}%")
    print(f"median corr with obs: summed Q' {df.r_sum_obs.median():.3f} -> routed {df.r_routed_obs.median():.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run / "plots"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for k, (col, title) in enumerate([("lag_sum_routed", "summed Q' -> routed (what routing adds)"),
                                      ("lag_sum_obs", "summed Q' -> observed (what the gauge asks)"),
                                      ("lag_routed_obs", "routed -> observed (residual)")]):
        jitter = np.random.default_rng(0).normal(0, 0.08, len(df))
        ax[k].scatter(df.la, df[col] + jitter, s=5, alpha=0.4)
        b = np.linspace(df.la.min(), df.la.max(), 12)
        which = np.digitize(df.la, b) - 1
        med = [df[col][which == j].median() if (which == j).sum() > 10 else np.nan for j in range(len(b) - 1)]
        ax[k].plot(0.5 * (b[1:] + b[:-1]), med, "r-", lw=2, label="binned median")
        ax[k].set_title(title, fontsize=10)
        ax[k].set_xlabel("log10 drainage area (km²)")
        ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("lag (days), positive = later")
    ax[0].legend()
    fig.suptitle(f"{a.run_id}: cross-correlation lag of daily anomalies, {str(t0)[:10]}..{str(t1)[:10]}")
    fig.tight_layout()
    fig.savefig(out / "routing_lag.png", dpi=150, facecolor="white")
    print(f"wrote {out / 'routing_lag.png'}")

    # Example hydrographs: pick well-observed gauges near the 10th, 50th, 90th
    # area percentiles.
    good = df[df.r_routed_obs > 0.6].sort_values("area")
    picks = [good.iloc[int(p * (len(good) - 1))] for p in (0.1, 0.5, 0.9)]
    s, e = f"{a.water_year - 1}-10-01", f"{a.water_year}-09-30"
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for axx, row in zip(axes, picks):
        g = str(row.gauge)
        w = ds.sel(gage_ids=g, time=slice(s, e))
        wb = dsb.sel(gage_ids=g, time=slice(s, e))
        t = pd.to_datetime(w.time.values)
        o = np.where(w.observations.values <= 0, np.nan, w.observations.values)
        axx.plot(t, o, "k-", lw=1.2, label="USGS observed")
        axx.plot(pd.to_datetime(wb.time.values), wb.predictions.values, color="tab:orange", lw=0.9, ls="--", label="summed Q' (no routing)")
        axx.plot(t, w.predictions.values, color="tab:blue", lw=0.9, label="routed")
        axx.set_title(f"{g} {meta.STANAME.get(g, '')}  {row.area:.0f} km²  "
                      f"lag sum->routed {row.lag_sum_routed:.0f} d, sum->obs {row.lag_sum_obs:.0f} d, "
                      f"r(routed,obs) {row.r_routed_obs:.2f}", fontsize=9)
        axx.set_ylabel("m³/s")
        axx.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel(f"water year {a.water_year}")
    fig.tight_layout()
    fig.savefig(out / f"routing_lag_examples_wy{a.water_year}.png", dpi=150, facecolor="white")
    print(f"wrote {out / f'routing_lag_examples_wy{a.water_year}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
