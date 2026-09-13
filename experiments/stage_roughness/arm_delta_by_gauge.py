#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "xarray", "zarr>=3", "matplotlib", "pandas"]
# ///
"""Where does a stage-roughness arm help or hurt, gauge by gauge?

The medians in findings §35/§36 hide the distribution. For each arm against
the control this computes per-gauge NSE, KGE, peak-flow bias (FHV, top 2% of
days), timing (lag of routed vs observed, days) and the routed/observed
standard-deviation ratio (KGE's alpha), then reports the deltas by drainage
area class and the fraction of gauges each arm improves.

    experiments/stage_roughness/arm_delta_by_gauge.py <control-run> <arm-run> [<arm-run> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
from load_ddrs_predictions import load_predictions_zarr  # noqa: E402

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"
WARMUP = 3


def metrics(pred, obs):
    m = np.isfinite(pred) & np.isfinite(obs) & (obs > 0)
    if m.sum() < 365:
        return dict(nse=np.nan, kge=np.nan, fhv=np.nan, alpha=np.nan, lag=np.nan)
    p, o = pred[m], obs[m]
    nse = 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)
    r = np.corrcoef(p, o)[0, 1]
    alpha = p.std() / o.std()
    beta = p.mean() / o.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    k = max(1, int(0.02 * o.size))
    top = np.argsort(o)[-k:]
    fhv = 100 * (p[top].sum() - o[top].sum()) / o[top].sum()
    best, br = 0, -np.inf
    for lag in range(-3, 6):
        a, b = (p[:p.size - lag], o[lag:]) if lag >= 0 else (p[-lag:], o[:o.size + lag])
        rr = np.corrcoef(a - a.mean(), b - b.mean())[0, 1]
        if rr > br:
            best, br = lag, rr
    return dict(nse=nse, kge=kge, fhv=fhv, alpha=alpha, lag=best)


def per_gauge(rid):
    ds = load_predictions_zarr(RUNS / rid / "eval" / "predictions.zarr")
    rows = {}
    for g in ds.gage_ids.values:
        w = ds.sel(gage_ids=g)
        rows[str(g)] = metrics(w.predictions.values[WARMUP:], w.observations.values[WARMUP:])
    return pd.DataFrame(rows).T


def main():
    ctrl, arms = sys.argv[1], sys.argv[2:]
    meta = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")
    c = per_gauge(ctrl)
    edges = [0, 300, 1000, 3000, 10000, 1e9]
    for arm in arms:
        a = per_gauge(arm).reindex(c.index)
        d = (a - c)
        d["area"] = meta["DRAIN_SQKM"].reindex(c.index).values
        print(f"\n##### {arm}  minus  {ctrl}   ({len(d)} gauges)")
        print(f"{'area km2':>14} {'n':>5} {'dNSE med':>9} {'frac NSE up':>12} {'dKGE med':>9} {'dFHV med':>9} {'alpha ctrl->arm':>16} {'dlag med':>9}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = d[(d.area >= lo) & (d.area < hi)]
            if s.empty:
                continue
            ca = c.loc[s.index, "alpha"].median(); aa = a.loc[s.index, "alpha"].median()
            print(f"{f'{lo:.0f}-{hi:.0f}':>14} {len(s):>5} {s.nse.median():>+9.4f} {100 * (s.nse > 0).mean():>11.1f}% "
                  f"{s.kge.median():>+9.4f} {s.fhv.median():>+9.2f} {ca:>7.3f}->{aa:<7.3f} {s.lag.median():>+9.2f}")
        print(f"{'ALL':>14} {len(d):>5} {d.nse.median():>+9.4f} {100 * (d.nse > 0).mean():>11.1f}% "
              f"{d.kge.median():>+9.4f} {d.fhv.median():>+9.2f} {c.alpha.median():>7.3f}->{a.alpha.median():<7.3f} {d.lag.median():>+9.2f}")
        big = d.nse.abs() > 0.05
        print(f"  gauges with |dNSE| > 0.05: {big.sum()} ({100 * big.mean():.1f}%), of which improved {int((d.nse[big] > 0).sum())}")
        print(f"  control medians: NSE {c.nse.median():.4f} KGE {c.kge.median():.4f} FHV {c.fhv.median():+.2f}% alpha {c.alpha.median():.3f} lag {c.lag.median():+.2f}")
        print(f"  arm medians:     NSE {a.nse.median():.4f} KGE {a.kge.median():.4f} FHV {a.fhv.median():+.2f}% alpha {a.alpha.median():.3f} lag {a.lag.median():+.2f}")
        (RUNS / arm / "plots").mkdir(exist_ok=True)
        d.to_csv(RUNS / arm / "plots" / f"delta_vs_{ctrl[:19]}.csv")


if __name__ == "__main__":
    main()
