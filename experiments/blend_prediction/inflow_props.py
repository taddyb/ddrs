"""Inflow properties of the blend and its members, by drainage-area class, before any blend training.

Reads the per-gauge summed inflow written by blend_inflow.py (exact route) and, for
the members only, their routed test predictions. Population: the 2,365 gauges the
member runs were scored on (eval/predictions.zarr), over the eval window
1995-10-02..2010-09-29 (WY1996-2010 less the two edge days the eval drops).

Per gauge and series (blend, UH retrospective, dHBV2 distributed, NH daily LSTM):
  flag_*    the same lags refined below one day by a parabola through the correlation
            at the best whole-day lag and its two neighbours
  lag_obs   cross-correlation lag of summed inflow against observed flow, in days,
            positive = observed is later, `best_lag` imported from
            experiments/stage_roughness/routing_lag.py (window -3..10 d, >= 365 pairs)
  fhv_obsday  100 (sum q on the top 2 % observed days - sum obs there) / sum obs there,
            the definition cross_arm_fields.py uses (linear in q, so the blend's value
            is the weighted mean of its members' exactly)
  fhv_fdc   the same on the flow-duration curve: top 2 % of sorted q against top 2 %
            of sorted obs (Yilmaz et al. 2008), which does see peak flattening
  alpha     KGE alpha, std(q) / std(obs)
  vol       sum q / sum obs
  r, nse    correlation and NSE of the summed inflow
Members also get lag_routed = lag(summed -> routed), what their trained router adds.

    /home/tbindas/projects/ddr/.venv/bin/python experiments/blend_prediction/inflow_props.py --data <dir>
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
from load_ddrs_predictions import load_predictions_zarr  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("routing_lag", HERE.parent / "stage_roughness" / "routing_lag.py")
rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)
best_lag = rl.best_lag

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"
SERIES = {"blend": "S_blend", "uh": "S_uh", "dist": "S_dist", "lstm": "S_lstm"}
ROUTED = {"uh": "2026-09-27T07-29-47Z-train-and-test", "uh_0912": "2026-09-12T23-39-03Z-train-and-test",
          "dist": "2026-09-13T17-22-30Z-train-and-test", "lstm": "2026-09-13T13-55-03Z-train-and-test"}
EDGES = [0, 300, 3000, 30000, np.inf]
CLASSES = ["<300", "300-3k", "3k-30k", ">30k"]


def corr_at(a, b, lag):
    x, y = (a[: a.size - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: b.size + lag])
    m = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() >= 365 else np.nan


def frac_lag(a, b, lag):
    """Sub-day lag: parabola through the cross-correlation at lag-1, lag, lag+1."""
    r0, r1, r2 = corr_at(a, b, lag - 1), corr_at(a, b, lag), corr_at(a, b, lag + 1)
    den = r0 - 2 * r1 + r2
    if not np.isfinite(den) or den >= 0:
        return float(lag)
    return lag + float(np.clip(0.5 * (r0 - r2) / den, -0.5, 0.5))


def metrics(q, o):
    m = np.isfinite(o) & (o > 0) & np.isfinite(q)
    if m.sum() < 365:
        return dict(fhv_obsday=np.nan, fhv_fdc=np.nan, alpha=np.nan, vol=np.nan, r=np.nan, nse=np.nan)
    q, o = q[m], o[m]
    k = max(1, int(0.02 * o.size))
    top = np.argsort(o)[-k:]
    qs, os_ = np.sort(q)[-k:], np.sort(o)[-k:]
    return dict(fhv_obsday=100 * (q[top].sum() - o[top].sum()) / o[top].sum(),
                fhv_fdc=100 * (qs.sum() - os_.sum()) / os_.sum(),
                alpha=q.std() / o.std(), vol=q.sum() / o.sum(), r=np.corrcoef(q, o)[0, 1],
                nse=1 - np.sum((q - o) ** 2) / np.sum((o - o.mean()) ** 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True)
    d = Path(ap.parse_args().data)
    z = np.load(d / "blend_inflow.npz")
    gauges = z["gauges"].astype(str)
    t_all = np.datetime64(str(z["start"])) + np.arange(int(z["n_days"]))
    ev = load_predictions_zarr(RUNS / ROUTED["uh"] / "eval" / "predictions.zarr")
    eg = ev.gage_ids.values.astype(str)
    te = ev.time.values.astype("datetime64[D]")
    gi = pd.Index(gauges).get_indexer(eg); assert (gi >= 0).all()
    ti = pd.Index(t_all).get_indexer(te); assert (ti >= 0).all()
    obs_b = z["obs"][gi][:, ti].astype(np.float64)
    obs_e = ev.observations.values.astype(np.float64)
    assert np.allclose(np.nan_to_num(obs_b, nan=-1), np.nan_to_num(obs_e, nan=-1), rtol=1e-6), "obs mismatch"
    obs = np.where(obs_e <= 0, np.nan, obs_e)
    area = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")["DRAIN_SQKM"].reindex(eg).values
    full = z["full_cover"][gi]
    print(f"{eg.size} eval gauges, {te.size} days {te[0]}..{te[-1]}; full three-store cover at {full.sum()}")

    routed = {}
    for lab, rid in ROUTED.items():
        e = load_predictions_zarr(RUNS / rid / "eval" / "predictions.zarr")
        assert (e.gage_ids.values.astype(str) == eg).all() and (e.time.values.astype("datetime64[D]") == te).all()
        routed[lab] = e.predictions.values.astype(np.float64)

    rows = []
    for s, key in SERIES.items():
        S = z[key][gi][:, ti].astype(np.float64)
        for i in range(eg.size):
            r = dict(series=s, gauge=eg[i], area=area[i], full_cover=bool(full[i]))
            r["lag_obs"], r["r_lag_obs"] = best_lag(S[i], obs[i])
            r["flag_obs"] = frac_lag(S[i], obs[i], r["lag_obs"])
            r.update(metrics(S[i], obs[i]))
            if s in routed:
                r["lag_routed"], _ = best_lag(S[i], routed[s][i])
                r["flag_routed"] = frac_lag(S[i], routed[s][i], r["lag_routed"])
            if s == "uh":
                r["lag_routed_0912"], _ = best_lag(S[i], routed["uh_0912"][i])
            rows.append(r)
    df = pd.DataFrame(rows)
    df["cls"] = pd.cut(df.area, EDGES, labels=CLASSES, right=False)
    df.to_csv(d / "inflow_props_per_gauge.csv", index=False)

    def block(g):
        return pd.Series({
            "n": len(g),
            "lag_obs_med": g.lag_obs.median(), "lag_obs_mean": g.lag_obs.mean(),
            "lead_ge1_pct": 100 * (g.lag_obs >= 1).mean(),
            "flag_obs_med": g.flag_obs.median(), "flag_obs_mean": g.flag_obs.mean(),
            "fhv_obsday": g.fhv_obsday.median(), "fhv_fdc": g.fhv_fdc.median(),
            "alpha": g.alpha.median(), "vol": g.vol.median(), "r": g.r.median(), "nse": g.nse.median(),
            "lag_routed_med": g.lag_routed.median() if "lag_routed" in g and g.lag_routed.notna().any() else np.nan,
            "lag_routed_mean": g.lag_routed.mean() if "lag_routed" in g and g.lag_routed.notna().any() else np.nan,
            "flag_routed_med": g.flag_routed.median() if "flag_routed" in g and g.flag_routed.notna().any() else np.nan,
        })
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    tab = df.groupby(["series", "cls"], observed=True).apply(block, include_groups=False)
    tot = df.groupby("series").apply(block, include_groups=False)
    tot.index = pd.MultiIndex.from_tuples([(s, "all") for s in tot.index], names=["series", "cls"])
    tab = pd.concat([tab, tot]).sort_index()
    tab.to_csv(d / "inflow_props_by_class.csv")
    print(tab.round(3).to_string())
    u = df[df.series == "uh"]
    print(f"\nUH router-added lag, run 0927 vs 0912, 3k-30k km2: "
          f"median {u[u.cls == '3k-30k'].lag_routed.median()} / {u[u.cls == '3k-30k'].lag_routed_0912.median()}, "
          f"mean {u[u.cls == '3k-30k'].lag_routed.mean():.3f} / {u[u.cls == '3k-30k'].lag_routed_0912.mean():.3f}")
    # blend vs weighted member mean, per gauge: how much does averaging flatten?
    w = {"uh": 0.5, "dist": 0.25, "lstm": 0.25}
    p = df.pivot(index="gauge", columns="series", values=["alpha", "fhv_fdc", "fhv_obsday", "vol"])
    for m in ["alpha", "fhv_fdc", "fhv_obsday", "vol"]:
        wm = sum(w[k] * p[(m, k)] for k in w)
        print(f"{m:>10}: blend median {p[(m, 'blend')].median():.3f}, weighted member mean (per gauge) median {wm.median():.3f}, "
              f"blend below UH at {100 * (p[(m, 'blend')] < p[(m, 'uh')]).mean():.1f}% of gauges")


if __name__ == "__main__":
    main()
