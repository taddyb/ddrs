#!/usr/bin/env python
"""Score dam-release smoke checks 4 (learning) and 5 (beat the controls) from two train-and-test runs.

  --off      run dir of config/experiments/dam_release_smoke_off.yaml
  --learned  run dir of config/experiments/dam_release_smoke_learned.yaml
  --out      JSON summary path (a per-gauge CSV and a per-dam CSV are written next to it)

Check 4: the learned per-dam T0 (<learned>/release_params.csv) against the offline seasonal fit's T0 at each dam
gauge's nearest dam (../smoke/expected_release_fit.csv): T0 should leave its 4.5 h init where the fit found storage
(seas_T0 above the grid floor) and stay low where it found none (seas_T0 at the floor, 0.05 d). Also the release
head's median T0 per mini-batch from the training log.

Check 5: per-gauge test NSE (WY1996-2010, both runs' eval/predictions.zarr), paired learned minus off. Dam gauges:
median with a 2,000-resample bootstrap 95 % interval (seed 42, as expected_release_fit.paired), sign test. Dam gain
minus its matched control's gain. Controls' median test NSE in each arm. The offline fit's numbers are the ceiling.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "smoke"))
from expected_release_fit import metrics, paired  # noqa: E402

INIT_T0 = 4.5 / 24.0
FIT_FLOOR = 0.051


def per_gauge(run: Path) -> pd.DataFrame:
    z = zarr.open(str(run / "eval" / "predictions.zarr"), mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    p, o = z["predictions"][:].astype(float), z["observations"][:].astype(float)
    rows = []
    for i, s in enumerate(ids):
        m = metrics(p[i], o[i], np.ones(p.shape[1], dtype=bool))
        rows.append(dict(STAID=s, nse=m["nse"], kge=m["kge"]))
    return pd.DataFrame(rows).set_index("STAID")


def t0_trajectory(run: Path) -> list[float]:
    vals = []
    for line in (run / "run.log").read_text().splitlines():
        m = re.search(r"release_T0_median=([0-9.eE+-]+)d", line)
        if m:
            vals.append(float(m.group(1)))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--learned", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    off_dir, lrn_dir, out = Path(a.off), Path(a.learned), Path(a.out)

    smoke = pd.read_csv(HERE.parent / "smoke" / "smoke_gauges.csv",
                        dtype={"STAID": str, "control_for": str, "huc2": str}).set_index("STAID")
    fit = pd.read_csv(HERE.parent / "smoke" / "expected_release_fit.csv",
                      dtype={"STAID": str, "control_for": str, "huc2": str}).set_index("STAID")
    res: dict = {}

    # ---- check 4 ----
    rp = pd.read_csv(lrn_dir / "release_params.csv").set_index("COMID")
    dam = fit[fit.role == "dam"].copy()
    dam["dam_COMID"] = dam.dam_COMID.astype(int)
    dam = dam[dam.dam_COMID.isin(rp.index)]
    dam["learned_T0"] = rp.loc[dam.dam_COMID, "T0_days"].to_numpy()
    dam["learned_a"] = rp.loc[dam.dam_COMID, "a"].to_numpy()
    dam["learned_b"] = rp.loc[dam.dam_COMID, "b"].to_numpy()
    per_dam = dam.drop_duplicates("dam_COMID")
    storage = per_dam[per_dam.seas_T0 > FIT_FLOOR]
    none = per_dam[per_dam.seas_T0 <= FIT_FLOOR]
    moved = lambda g: float((np.abs(np.log(g.learned_T0 / INIT_T0)) > np.log(1.1)).mean())  # noqa: E731
    rho, p_rho = spearmanr(per_dam.learned_T0, per_dam.seas_T0)
    traj = t0_trajectory(lrn_dir)
    res["check4_learning"] = dict(
        n_dams_learned=int(len(rp)),
        n_nearest_dams=int(len(per_dam)),
        learned_T0_days_all=dict(q25=float(rp.T0_days.quantile(0.25)), median=float(rp.T0_days.median()),
                                 q75=float(rp.T0_days.quantile(0.75)), max=float(rp.T0_days.max())),
        fit_found_storage=dict(n=int(len(storage)), median_learned_T0=float(storage.learned_T0.median()),
                               median_fit_T0=float(storage.seas_T0.median()),
                               frac_moved_10pct=moved(storage),
                               frac_above_init=float((storage.learned_T0 > INIT_T0).mean())),
        fit_found_none=dict(n=int(len(none)), median_learned_T0=float(none.learned_T0.median()),
                            frac_moved_10pct=moved(none),
                            frac_above_init=float((none.learned_T0 > INIT_T0).mean())),
        spearman_learned_vs_fit_T0=dict(rho=float(rho), p=float(p_rho)),
        batch_median_T0_first_last=[traj[0], traj[-1]] if traj else None,
        batch_median_T0_quartiles_of_log=[traj[len(traj) * k // 4] for k in range(4)] + [traj[-1]] if traj else None,
    )
    per_dam[["dam_COMID", "dam_name", "dam_storage_mcm", "on_reach", "seas_T0", "seas_a", "seas_b",
             "learned_T0", "learned_a", "learned_b"]].to_csv(out.with_suffix(".per_dam.csv"))

    # ---- check 5 ----
    g_off, g_lrn = per_gauge(off_dir), per_gauge(lrn_dir)
    df = smoke[["role", "control_for", "huc2"]].join(g_off.add_suffix("_off")).join(g_lrn.add_suffix("_learned"))
    df = df.join(fit[["nse_pass_test", "nse_seas_test", "d_seas", "on_reach"]])
    df["d_nse"] = df.nse_learned - df.nse_off
    df["d_kge"] = df.kge_learned - df.kge_off
    ctl_gain = df[df.role == "control"].set_index("control_for").d_nse
    df["d_minus_control"] = np.where(df.role == "dam", df.d_nse - df.index.map(ctl_gain).astype(float), np.nan)
    df.to_csv(out.with_suffix(".per_gauge.csv"))
    dm, ct = df[df.role == "dam"], df[df.role == "control"]
    on = dm.on_reach.astype(str) == "True"
    res["check5_beat_controls"] = dict(
        median_test_nse=dict(dam_off=float(dm.nse_off.median()), dam_learned=float(dm.nse_learned.median()),
                             control_off=float(ct.nse_off.median()), control_learned=float(ct.nse_learned.median())),
        median_test_kge=dict(dam_off=float(dm.kge_off.median()), dam_learned=float(dm.kge_learned.median()),
                             control_off=float(ct.kge_off.median()), control_learned=float(ct.kge_learned.median())),
        paired_dnse=dict(dam=paired(dm.d_nse), control=paired(ct.d_nse),
                         dam_on_reach=paired(dm.d_nse[on]), dam_off_reach=paired(dm.d_nse[~on])),
        paired_dkge=dict(dam=paired(dm.d_kge), control=paired(ct.d_kge)),
        dam_minus_matched_control=paired(dm.d_minus_control),
        control_median_within_001=bool(abs(ct.nse_learned.median() - ct.nse_off.median()) < 0.01),
        offline_ceiling=dict(dam=paired(dm.d_seas), control=paired(ct.d_seas)),
    )
    c5 = res["check5_beat_controls"]
    c5["passed"] = dict(
        dam_median_above_zero_ci_clear=bool(c5["paired_dnse"]["dam"]["median"] > 0 and c5["paired_dnse"]["dam"]["ci"][0] > 0),
        dam_minus_control_positive=bool(c5["dam_minus_matched_control"]["median"] > 0),
        controls_within_001=c5["control_median_within_001"],
    )
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
