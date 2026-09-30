#!/usr/bin/env python
"""Score dam-release smoke checks 1 and 2 (README of ../smoke, 'What the implementation must show').

Inputs (zarr stores written by the legacy test-phase binary on the configs from smoke_check_tables.py):
  --ref           the no-dam reference run, output/reservoir_smoke/pred_1981_2010.zarr (reservoir-options worktree)
  --off           this branch, use_reservoirs off                     (check 1a: bitwise identical to --ref)
  --passthrough   this branch, every smoke dam at T = 1/24 d, a = b = 0 (check 1b: NSE > 0.999 at every gauge)
  --seasonal      this branch, on-reach dams at the offline seasonal fit (check 2: NSE > 0.99 vs the fit's series)
  --fit-dams      fit_dams.csv from smoke_check_tables.py (which gauge's fit each dam carries)

Check 2's reference series is recomputed with expected_release_fit.simulate (daily implicit Euler on the no-dam
gauge series, spin-up from WY1982) at each on-reach gauge whose fit is in the table. Scored on WY1996-2010 (the
fit's test window) and WY1983-2010. A gauge with another table dam upstream is reported separately ("cascade"):
its dam inflow in ddrs includes that dam's bucket, the fit's does not. Upstream sets come from the gauge subgraphs
in the gages adjacency zarr.

Writes a JSON summary to --out and prints it.
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
from expected_release_fit import simulate  # noqa: E402

GAGES_ADJ = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"


def load(path: str):
    z = zarr.open(path, mode="r")
    ids = [bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]]
    t = pd.DatetimeIndex(z["time"][:].astype("datetime64[ns]"))
    return pd.Index(ids), t, z["predictions"][:], z["observations"][:]


def nse(sim: np.ndarray, ref: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(ref)
    s, r = sim[m].astype(float), ref[m].astype(float)
    den = ((r - r.mean()) ** 2).sum()
    return float(1 - ((s - r) ** 2).sum() / den) if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ["ref", "off", "passthrough", "seasonal", "fit-dams", "out"]:
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()

    ids, t, ref, obs = load(a.ref)
    res: dict = {}

    # ---- 1a: off is bitwise identical ----
    ids_off, t_off, off, _ = load(a.off)
    same_axes = list(ids_off) == list(ids) and (t_off == t).all()
    diff = off.view(np.uint32) != ref.view(np.uint32) if off.dtype == np.float32 else off != ref
    res["check1a_off_bitwise"] = dict(
        same_gauges_and_dates=bool(same_axes),
        n_values=int(ref.size),
        n_bit_differences=int(diff.sum()),
        max_abs_diff=float(np.nanmax(np.abs(off.astype(float) - ref.astype(float)))),
        passed=bool(same_axes and diff.sum() == 0),
    )

    # ---- 1b: one-hour buckets are nearly pass-through ----
    _, _, pas, _ = load(a.passthrough)
    per = np.array([nse(pas[i], ref[i]) for i in range(len(ids))])
    res["check1b_passthrough"] = dict(
        n_gauges=len(ids),
        min_nse=float(np.nanmin(per)),
        median_nse=float(np.nanmedian(per)),
        n_below_0999=int((per <= 0.999).sum()),
        worst=[dict(staid=ids[i], nse=float(per[i])) for i in np.argsort(per)[:5]],
        n_gauges_changed=int(sum((pas[i] != ref[i]).any() for i in range(len(ids)))),
        passed=bool(np.nanmin(per) > 0.999),
    )

    # ---- 2: engine matches the offline seasonal fit at on-reach dams ----
    _, _, sea, _ = load(a.seasonal)
    fit_dams = pd.read_csv(a.fit_dams, dtype={"STAID": str})
    fits = pd.read_csv(HERE.parent / "smoke" / "expected_release_fit.csv", dtype={"STAID": str}).set_index("STAID")
    w = 2 * np.pi * np.asarray(t.dayofyear) / 365.25
    sinw, cosw = np.sin(w), np.cos(w)
    wy = np.asarray(t.year + (t.month >= 10))
    test = (wy >= 1996) & (wy <= 2010)
    post = wy >= 1983
    table_comids = set(int(c) for c in fit_dams.COMID)
    adj = zarr.open(GAGES_ADJ, mode="r")
    rows = []
    for _, d in fit_dams.iterrows():
        s = d.STAID
        i = ids.get_loc(s)
        inflow = ref[i].astype(float)
        inflow = np.where(np.isfinite(inflow), inflow, np.nanmean(inflow))
        f = fits.loc[s]
        q_fit = simulate(inflow, sinw, cosw, np.array([f.seas_T0]), np.array([f.seas_a]), np.array([f.seas_b]),
                         len(inflow))
        upstream = set(int(c) for c in adj[s]["order"][:]) - {int(d.COMID)}
        rows.append(dict(
            staid=s, comid=int(d.COMID), T0=float(f.seas_T0), a=float(f.seas_a), b=float(f.seas_b),
            cascade=bool(upstream & table_comids),
            nse_test_window=nse(sea[i][test], q_fit[test]),
            nse_1983_2010=nse(sea[i][post], q_fit[post]),
            nse_ddrs_vs_nodam_test=nse(sea[i][test], ref[i][test]),
            nse_fit_vs_nodam_test=nse(q_fit[test], ref[i][test]),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out).with_suffix(".check2.csv"), index=False)

    def summ(g: pd.DataFrame) -> dict:
        x = g.nse_test_window
        return dict(n=len(g), median=float(x.median()), min=float(x.min()),
                    n_above_099=int((x > 0.99).sum()), frac_above_099=float((x > 0.99).mean()),
                    median_1983_2010=float(g.nse_1983_2010.median()))

    iso = df[~df.cascade]
    res["check2_engine_vs_offline_fit"] = dict(
        all_on_reach=summ(df),
        isolated=summ(iso),
        cascade=summ(df[df.cascade]),
        # How far each is from the no-dam series: a tiny T gives NSE ~ 1 trivially.
        median_nse_fit_vs_nodam_test=float(df.nse_fit_vs_nodam_test.median()),
        worst_isolated=iso.nsmallest(8, "nse_test_window")[["staid", "T0", "a", "b", "nse_test_window"]]
        .to_dict("records"),
        passed_isolated=bool((iso.nse_test_window > 0.99).all()),
    )
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
