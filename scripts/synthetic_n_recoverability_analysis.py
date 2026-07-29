"""Pre-registered verdicts S1-S5 for the synthetic-n recoverability
experiment (docs/superpowers/specs/2026-07-22-synthetic-n-recoverability-design.md §3).

Run from ddrs-py's venv, after all 4 students' dump_parameters outputs exist:
    cd ddrs-py && uv run python ../scripts/synthetic_n_recoverability_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "output/synthetic_n"
ATTRS = REPO.parent / "ddr/data/merit_global_attributes_v2.nc"

ARMS = ["distributed", "lumped", "daily_lstm", "hourly_lstm"]

# Mean daily Q' volume ratio of each real store vs the standard benchmark
# store, for S5 — filled in manually from `ddrs import --dry-run` /
# icechunk inspection once available; None disables S5 for that arm. S5 is
# WIRED IN below (pearsonr against n_errors) but only computed when at
# least 3 arms have a non-None ratio — with only 4 arms total, fewer points
# than that isn't a meaningful correlation, so it's reported as
# "not computed", never silently skipped without saying why.
VOLUME_RATIO_VS_TRUTH: dict[str, float | None] = {
    "distributed": None,
    "lumped": None,
    "daily_lstm": None,
    "hourly_lstm": None,
}


def median_abs_error(recovered: np.ndarray, truth: np.ndarray) -> float:
    return float(np.median(np.abs(recovered - truth)))


def fitted_slope(log10_uparea: np.ndarray, n: np.ndarray) -> float:
    return float(np.polyfit(log10_uparea, n, 1)[0])


def main() -> None:
    truth = xr.open_dataset(OUT_DIR / "truth_leopold_maddock.nc")
    truth_comids = truth["COMID"].values
    truth_n = truth["n"].values
    truth_q = truth["q_spatial"].values
    truth_p = truth["p_spatial"].values

    attrs = xr.open_dataset(ATTRS).set_index(COMID="COMID").sel(COMID=truth_comids)
    log10_uparea = attrs["log10_uparea"].values.astype(np.float64)
    true_slope = fitted_slope(log10_uparea, truth_n)

    rows = []
    n_errors, geom_errors = {}, {}
    for arm in ARMS:
        rec = xr.open_dataset(OUT_DIR / f"recovered_{arm}.nc").set_index(COMID="COMID").sel(
            COMID=truth_comids
        )
        n_err = median_abs_error(rec["n"].values, truth_n)
        q_err = median_abs_error(rec["q_spatial"].values, truth_q)
        p_err = median_abs_error(rec["p_spatial"].values, truth_p)
        slope = fitted_slope(log10_uparea, rec["n"].values)
        n_errors[arm] = n_err
        geom_errors[arm] = (q_err + p_err) / 2.0
        rows.append(
            {
                "arm": arm,
                "n_median_abs_error": n_err,
                "q_median_abs_error": q_err,
                "p_median_abs_error": p_err,
                "recovered_n_slope": slope,
                "true_n_slope": true_slope,
                "slope_sign_flipped": bool(slope > 0 and true_slope < 0),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "recoverability_rows.csv"
    df.to_csv(csv_path, index=False)

    n_spread = max(n_errors.values()) - min(n_errors.values())
    geom_spread = max(geom_errors.values()) - min(geom_errors.values())
    s4_ratio = n_spread / geom_spread if geom_spread > 0 else float("inf")

    any_flip = df["slope_sign_flipped"].any()

    # S5: pearsonr of n_errors against VOLUME_RATIO_VS_TRUTH, restricted to
    # arms with a filled-in ratio. Needs >=3 points to be worth reporting at
    # all (spec §3: "only 4 data points" — 2 points is a line, not a
    # correlation). Prints an explicit reason when it can't run, rather than
    # silently doing nothing.
    filled = {a: r for a, r in VOLUME_RATIO_VS_TRUTH.items() if r is not None}
    if len(filled) >= 3:
        arms_with_ratio = list(filled.keys())
        s5_r, s5_p = pearsonr(
            [n_errors[a] for a in arms_with_ratio],
            [filled[a] for a in arms_with_ratio],
        )
        s5_line = f"  [S5] pearson r={s5_r:.3f} (p={s5_p:.3f}) over {len(filled)} arms: {filled}"
    else:
        s5_r = s5_p = None
        s5_line = (
            f"  [S5] not computed — only {len(filled)}/{len(ARMS)} arms have a filled-in "
            "VOLUME_RATIO_VS_TRUTH (need >=3). Fill in the dict from `ddrs import --dry-run` "
            "/ icechunk mean-daily-volume inspection per arm to enable this."
        )

    print(df.to_string(index=False))
    print()
    print("========================================================================")
    print("VERDICTS (bars pre-registered in the design spec)")
    print("========================================================================")
    print(f"  [S1] n median-abs-error per arm: {n_errors}")
    print(f"  [S2] true slope={true_slope:.5f}; any arm sign-flipped positive: {any_flip}")
    print(f"  [S3] geometry median-abs-error per arm: {geom_errors}")
    print(f"  [S4 {'PASS' if s4_ratio >= 3 else 'FAIL'}] n-spread/geom-spread = {s4_ratio:.2f} (bar: >=3)")
    print(s5_line)
    print(f"  HEADLINE: {'PASS' if (s4_ratio >= 3 and any_flip) else 'FAIL'} "
          "(requires S4>=3x AND at least one slope sign flip; S5 is supporting evidence only)")
    print()
    print(f"per-arm rows -> {csv_path}")


if __name__ == "__main__":
    main()
