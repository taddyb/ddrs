"""Read-out of the three-product blend model against research/specs/2026-09-30-blend-arm-n0-prediction.md.

Everything here is read-only on the run directories. Scores the pieces of the prediction document that are
not the n_0 median on set (b) (that is member_fields.py), plus the skill comparison:

  1. validity gates of the prediction doc section 5: recipe diff, baseline/predictions.f32 vs offline S_blend,
     fraction of set-(b) reaches at the n_0 bounds, NaN check;
  2. secondary: router-added lag at 3,000-30,000 km2 (best_lag of routing_lag.py, summed inflow -> routed flow);
  3. secondary: peak-shape diagnostic ratio from bias_absorber_per_gauge.csv;
  4. skill: blend model vs UH fixed-channel model (seeds 42 and 43), gauge-bootstrap CIs (levers/common.py), and
     engine blend vs the offline output blend of the three member runs.

    /home/tbindas/projects/ddr/.venv/bin/python experiments/blend_prediction/blend_readout.py \
        --pred-data <blend-prediction dir> --readout <blend-readout dir>
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "levers"))
sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
import common as C  # noqa: E402
from load_ddrs_predictions import load_baseline_f32  # noqa: E402

spec = importlib.util.spec_from_file_location("routing_lag", HERE.parent / "stage_roughness" / "routing_lag.py")
rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)
spec = importlib.util.spec_from_file_location("inflow_props", HERE / "inflow_props.py")
ip = importlib.util.module_from_spec(spec); spec.loader.exec_module(ip)

BLEND = "2026-10-05T13-32-00Z-train-and-test"
DIST = "2026-09-13T17-22-30Z-train-and-test"
LSTM = "2026-09-13T13-55-03Z-train-and-test"
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"
N_FLOOR, N_CEIL = 0.015, 0.2495


def say(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-data", required=True)
    ap.add_argument("--readout", required=True)
    a = ap.parse_args()
    pdat, ro = Path(a.pred_data), Path(a.readout)

    # ---- 1. validity gates
    say("== 1. validity gates (prediction doc section 5)")
    z = np.load(pdat / "blend_inflow.npz")
    bl = load_baseline_f32(C.RUNS / BLEND / "baseline")
    gauges = z["gauges"].astype(str)
    t_all = np.datetime64(str(z["start"])) + np.arange(int(z["n_days"]))
    same_g = (bl.gage_ids.values.astype(str) == gauges).all()
    tb = bl.time.values.astype("datetime64[D]")
    ti = pd.Index(t_all).get_indexer(tb)
    say(f"baseline gauges {bl.gage_ids.size}, days {tb.size}; gauge order equals offline npz: {same_g}; "
        f"all baseline days inside npz span: {(ti >= 0).all()}")
    run_S = bl.predictions.values.astype(np.float64)
    off_S = z["S_blend"][:, ti[ti >= 0]].astype(np.float64)
    run_S = run_S[:, ti >= 0]
    rel = np.abs(run_S - off_S) / np.maximum(np.abs(off_S), 1e-9)
    say(f"run baseline summed inflow vs offline S_blend: relative diff p50 {np.percentile(rel, 50):.2e} "
        f"p99 {np.percentile(rel, 99):.2e} p99.9 {np.percentile(rel, 99.9):.2e} max {rel.max():.2e}; "
        f"threshold p99 <= 1e-4 -> {'PASS' if np.percentile(rel, 99) <= 1e-4 else 'FAIL'}")

    ds = netCDF4.Dataset(ro / "dump" / "blend_kan_parameters.nc")
    comid = np.asarray(ds["COMID"][:], dtype=np.int64)
    n = np.asarray(ds["n"][:], dtype=np.float64)
    full = np.load(pdat / "fields" / "comids_full_upstream_cover.npy")
    mb = np.isin(comid, full)
    say(f"n_0 finite everywhere: {np.isfinite(n).all()}; set (b) reaches {mb.sum():,}; "
        f"at floor (<= {N_FLOOR}): {100 * np.mean(n[mb] <= N_FLOOR + 1e-6):.3f} %, at ceiling (>= {N_CEIL}): "
        f"{100 * np.mean(n[mb] >= N_CEIL):.3f} %; min {n[mb].min():.4f} max {n[mb].max():.4f}")

    # ---- eval predictions
    ids, tt, Pb, O = C.load_preds(BLEND)
    ids42, t42, P42, O42 = C.load_preds(C.S42)
    assert (t42 == tt).all()
    P42 = C.align(ids42, P42, ids); O42 = C.align(ids42, O42, ids)
    assert np.allclose(np.nan_to_num(O, nan=-1), np.nan_to_num(O42, nan=-1))
    say(f"eval gauges {len(ids)}, days {len(tt)} ({tt[0].date()}..{tt[-1].date()}); NaN in blend predictions: "
        f"{int(np.isnan(Pb).sum())}; non-finite: {int((~np.isfinite(Pb)).sum())}")
    i43, t43, P43, _ = C.load_preds(C.S43); assert (t43 == tt).all(); P43 = C.align(i43, P43, ids)
    iD, tD, PD, _ = C.load_preds(DIST); assert (tD == tt).all(); PD = C.align(iD, PD, ids)
    iL, tL, PL, _ = C.load_preds(LSTM); assert (tL == tt).all(); PL = C.align(iL, PL, ids)

    # ---- 2. router-added lag
    say("\n== 2. router-added lag, run's own summed inflow -> its routed flow (best_lag, window -3..10 d)")
    area = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")["DRAIN_SQKM"].reindex(ids).values
    bidx = pd.Index(bl.gage_ids.values.astype(str)).get_indexer(ids)
    bt = pd.Index(tb).get_indexer(tt.values.astype("datetime64[D]"))
    assert (bidx >= 0).all() and (bt >= 0).all()
    Sb = bl.predictions.values[bidx][:, bt].astype(np.float64)
    lags, flags = [], []
    for i in range(len(ids)):
        l, _ = rl.best_lag(Sb[i], Pb[i]); lags.append(l); flags.append(ip.frac_lag(Sb[i], Pb[i], l))
    lag = pd.DataFrame(dict(gauge=ids, area=area, lag=lags, flag=flags))
    lag["cls"] = pd.cut(lag.area, ip.EDGES, labels=ip.CLASSES, right=False)
    t = lag.groupby("cls", observed=True).agg(n=("lag", "size"), median=("lag", "median"), mean=("lag", "mean"),
                                              sub_day_median=("flag", "median"))
    say(t.round(3).to_string())
    lag.to_csv(ro / "router_lag_per_gauge.csv", index=False)

    # ---- 3. peak-shape diagnostic
    say("\n== 3. peak-shape diagnostic (median over gauges of the basin-median n_0, >= 3 reaches)")
    ba = pd.read_csv(ro / "fields" / "cross_arm_fields" / "bias_absorber_per_gauge.csv", dtype={"gauge": str})
    ba = ba[ba.n_reach >= 3]
    ar = pd.Series(area, index=ids)
    out = {}
    for lo, hi, name in [(300, 3000, "300-3000"), (3000, 30000, "3000-30000")]:
        g = [x for x in ids if lo <= ar[x] < hi]
        u = ba[(ba.arm == "uh_0927") & ba.gauge.isin(g)].n0_basin
        b = ba[(ba.arm == "blend") & ba.gauge.isin(g)].n0_basin
        out[name] = (b.median() / u.median(), u.median(), b.median(), len(u), len(b))
        say(f"{name:>10}: UH median {u.median():.4f} ({len(u)} gauges), blend median {b.median():.4f} "
            f"({len(b)} gauges), ratio {b.median() / u.median():.4f}")
    r1, r2 = out["300-3000"][0], out["3000-30000"][0]
    say(f"ratio(300-3000) {r1:.4f} < ratio(3000-30000) - 0.05 = {r2 - 0.05:.4f}? -> {r1 < r2 - 0.05}")

    # ---- 4. skill
    say("\n== 4. skill on the test window (WY1996-2010), 2,365 gauges; bootstrap = paired over gauges, 2,000 draws")
    M = {k: C.metrics(P, O) for k, P in {"blend": Pb, "uh42": P42, "uh43": P43, "dist": PD, "lstm": PL,
                                         "off_w": 0.5 * P42 + 0.25 * PD + 0.25 * PL,
                                         "off_eq": (P42 + PD + PL) / 3}.items()}
    for k, m in M.items():
        say(f"  {k:7s} NSE median {m.nse.median():.4f}  KGE median {m.kge.median():.4f}  "
            f"(finite NSE {int(m.nse.notna().sum())}, KGE {int(m.kge.notna().sum())})")
    rows = []
    for new, base in [("blend", "uh42"), ("blend", "uh43"), ("uh43", "uh42"), ("off_w", "uh42"), ("off_eq", "uh42"),
                      ("blend", "off_w"), ("blend", "off_eq"), ("blend", "dist"), ("blend", "lstm")]:
        for met in ("nse", "kge"):
            s = C.summarize(f"{new} vs {base}", M[new][met].values, M[base][met].values, met.upper())
            rows.append(s); say(C.fmt(s))
    pd.DataFrame(rows).to_csv(ro / "skill_summary.csv", index=False)
    pd.DataFrame({"gauge": ids, "area": area, **{f"nse_{k}": m.nse.values for k, m in M.items()},
                  **{f"kge_{k}": m.kge.values for k, m in M.items()}}).to_csv(ro / "skill_per_gauge.csv", index=False)
    # size-class split of the blend-vs-UH(42) median change
    say("\n  blend minus UH42 median NSE by drainage-area class:")
    d = pd.DataFrame({"area": area, "b": M["blend"].nse.values, "u": M["uh42"].nse.values, "ow": M["off_w"].nse.values})
    d["cls"] = pd.cut(d.area, ip.EDGES, labels=ip.CLASSES, right=False)
    for c, g in d.groupby("cls", observed=True):
        say(f"    {c:>7} n={len(g):4d} blend {g.b.median():.4f} UH {g.u.median():.4f} offline {g.ow.median():.4f}")


if __name__ == "__main__":
    main()
