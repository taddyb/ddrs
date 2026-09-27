#!/usr/bin/env python
"""Full-population comparison of the learned dam release against its no-dam arm.

  --off      run dir of config/experiments/dam_release_full_off.yaml
  --learned  run dir of config/experiments/dam_release_full_learned.yaml
  --out      JSON summary path (a per-gauge CSV is written next to it)

Per gauge, test WY1996-2010 NSE and KGE from each run's eval/predictions.zarr (same metrics code as the smoke
checks), paired learned minus off: median with a 2,000-resample bootstrap 95 % interval (seed 42) and the sign test,
for all gauges, gauges with a NID dam >= 10 MCM upstream (../nid/nid_dams_by_gauge.csv, n_nid_ge10mcm > 0) and
gauges without. Median NSE / KGE per arm from the run manifests, and the summed-Q' baseline the run copied into
<run>/baseline/. The learned T0 / a / b distribution from <learned>/release_params.csv, all table dams and the dams
inside the eval network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "smoke"))
from expected_release_fit import metrics, paired  # noqa: E402


def per_gauge(run: Path) -> pd.DataFrame:
    z = zarr.open(str(run / "eval" / "predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    p, o = z["predictions"][:].astype(float), z["observations"][:].astype(float)
    rows = [dict(STAID=s, **{k: v for k, v in metrics(p[i], o[i], np.ones(p.shape[1], bool)).items()
                             if k in ("nse", "kge")}) for i, s in enumerate(ids)]
    return pd.DataFrame(rows).set_index("STAID")


def manifest_metrics(run: Path) -> dict:
    m = json.loads((run / "manifest.json").read_text())["metrics"]
    return {k: m.get(k) for k in ["median_nse_finite", "median_kge_finite", "n_gauges_total",
                                  "phase1_seconds", "phase2_seconds"]}


def baseline(run: Path, gauges: pd.Index) -> dict:
    """Summed-Q' medians on `gauges` only: the cached baseline may cover a larger population (2,698 gauges for
    gages_3000 against the 2,365 the run evaluates), and a baseline on other gauges is not a bar for this run."""
    b = json.loads((run / "baseline" / "manifest.json").read_text())
    df = pd.DataFrame({"nse": b["metrics"]["nse"], "kge": b["metrics"]["kge"]}, index=b["gage_ids"], dtype=float)
    df = df[df.index.isin(gauges)]
    return dict(n=int(len(df)), median_nse=float(df.nse.median()), median_kge=float(df.kge.median()))


def dist(x: pd.Series) -> dict:
    return {q: float(x.quantile(v)) for q, v in [("q05", .05), ("q25", .25), ("median", .5), ("q75", .75),
                                                 ("q95", .95)]} | dict(min=float(x.min()), max=float(x.max()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--learned", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    off, lrn, out = Path(a.off), Path(a.learned), Path(a.out)

    nid = pd.read_csv(HERE.parent / "nid" / "nid_dams_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
    g_off, g_lrn = per_gauge(off), per_gauge(lrn)
    df = g_off.add_suffix("_off").join(g_lrn.add_suffix("_learned"), how="inner")
    df["dam"] = df.index.map(nid.n_nid_ge10mcm).fillna(0).astype(int) > 0
    df["d_nse"] = df.nse_learned - df.nse_off
    df["d_kge"] = df.kge_learned - df.kge_off
    df.to_csv(out.with_suffix(".per_gauge.csv"))

    grp = {"all": df, "nid_dam_ge10mcm_upstream": df[df.dam], "no_nid_dam_ge10mcm": df[~df.dam]}
    res = dict(
        runs=dict(off=off.name, learned=lrn.name),
        manifest=dict(off=manifest_metrics(off), learned=manifest_metrics(lrn)),
        summed_q_prime_baseline=baseline(off, df.index),
        n_gauges=dict({k: int(len(v)) for k, v in grp.items()}),
        median_test_nse={k: dict(off=float(v.nse_off.median()), learned=float(v.nse_learned.median()))
                         for k, v in grp.items()},
        median_test_kge={k: dict(off=float(v.kge_off.median()), learned=float(v.kge_learned.median()))
                         for k, v in grp.items()},
        paired_dnse={k: paired(v.d_nse) for k, v in grp.items()},
        paired_dkge={k: paired(v.d_kge) for k, v in grp.items()},
    )
    rp = pd.read_csv(lrn / "release_params.csv")
    try:
        # Dams inside the eval network: the ones the test phase matched (run.log line).
        log = (lrn / "run.log").read_text()
        res["reservoir_match_line"] = [l.split("] ", 1)[-1] for l in log.splitlines()
                                       if "table COMIDs are in the network" in l]
    except OSError:
        pass
    res["learned_release"] = dict(
        n_dams=int(len(rp)),
        T0_days=dist(rp.T0_days), a=dist(rp.a), b=dist(rp.b),
        amplitude=dist(np.sqrt(rp.a ** 2 + rp.b ** 2)),
        T_min_days=dist(rp.T_min_days), T_max_days=dist(rp.T_max_days),
        n_T0_above_1d=int((rp.T0_days > 1).sum()), n_T0_above_10d=int((rp.T0_days > 10).sum()),
        n_amplitude_above_0_5=int((np.sqrt(rp.a ** 2 + rp.b ** 2) > 0.5).sum()),
    )
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
