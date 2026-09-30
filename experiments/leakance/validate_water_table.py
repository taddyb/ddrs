#!/usr/bin/env python
"""Does the learned groundwater offset recover a water table it never saw?

The question this answers is NOT "does the map look plausible". A map of any
learned field looks plausible, because the head is a smooth function of
attributes that are themselves spatially smooth. The question is whether the
field carries information about the water table BEYOND what its own inputs
already encode.

Two tests, and the second only became possible when two-way exchange was
enabled on 2026-09-17:

  MAGNITUDE  partial correlation of learned `d_gw` against the independent
             water table, controlling for all ten head inputs. A raw
             correlation that collapses under the control means the model
             reproduced a projection of its predictors, not the water table.
             This is what the 2026-09-16 arms did: raw -0.36, controlled +0.05.

  SIGN       whether the model puts each reach on the right SIDE. A reach gains
             when `d_gw > depth`, so the sign of `depth - d_gw` is a regime
             classification the reference can score directly. Under
             `leakance_losing_only: true` this test was vacuous: every reach was
             forced losing. It is the sharper of the two, because a magnitude
             can be fitted by any monotone function of size while the sign
             cannot.

Usage:
  validate_water_table.py <run-id> [--attrs ...] [--channel ...]
"""
import argparse
import sys

import numpy as np
import netCDF4 as nc

HEAD_INPUTS = [
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]


def resid(y, X):
    """Residual of y after least-squares regression on X (intercept included)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def pcorr(a, b, controls=None):
    if controls is not None and controls.shape[1] > 0:
        a, b = resid(a, controls), resid(b, controls)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--workspace", default="/home/tbindas/projects/ddrs/.ddrs")
    ap.add_argument("--attrs", default="/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc")
    ap.add_argument("--channel", default="/home/tbindas/projects/ddr/data/merit_channel_attributes_v1.nc")
    a = ap.parse_args()

    run = f"{a.workspace}/runs/{a.run_id}"
    params = nc.Dataset(f"{run}/plot/kan_parameters.nc")
    comid = np.asarray(params["COMID"][:]).astype(np.int64)
    d_gw = np.asarray(params["d_gw"][:], dtype=float)

    # Routed depth comes from the EVAL dump, which covers the eval network only.
    ev = nc.Dataset(f"{run}/kan_parameters.nc")
    comid_e = np.asarray(ev["COMID_eval"][:]).astype(np.int64)
    depth_e = np.asarray(ev["depth_mean"][:], dtype=float)

    ch = nc.Dataset(a.channel)
    comid_c = np.asarray(ch["COMID"][:]).astype(np.int64)
    # `channel_wtd_bed_rel` is POSITIVE where the table sits BELOW the bed, so
    # the reference in our sign convention (positive = above bed) is its negation.
    ref_d_gw = -np.asarray(ch["channel_wtd_bed_rel"][:], dtype=float)

    at = nc.Dataset(a.attrs)
    comid_at = np.asarray(at["COMID"][:]).astype(np.int64)
    inputs = {v: np.asarray(at[v][:], dtype=float) for v in HEAD_INPUTS if v in at.variables}
    missing = [v for v in HEAD_INPUTS if v not in inputs]
    if missing:
        print(f"WARNING: head inputs absent from the attribute file: {missing}", file=sys.stderr)
        print("The control is INCOMPLETE and the partial correlation will be an", file=sys.stderr)
        print("UPPER bound on the independent signal.", file=sys.stderr)

    ic = {c: i for i, c in enumerate(comid_c)}
    ia = {c: i for i, c in enumerate(comid_at)}
    ie = {c: i for i, c in enumerate(comid_e)}

    rows = []
    for k, c in enumerate(comid):
        j, m, e = ic.get(c, -1), ia.get(c, -1), ie.get(c, -1)
        if j < 0 or m < 0:
            continue
        rows.append((k, j, m, e))
    rows = np.array(rows)
    k_, j_, m_, e_ = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]

    learned = d_gw[k_]
    ref = ref_d_gw[j_]
    X = np.column_stack([inputs[v][m_] for v in inputs])

    ok = np.isfinite(learned) & np.isfinite(ref) & np.isfinite(X).all(axis=1)
    learned, ref, X = learned[ok], ref[ok], X[ok]
    e_ok = e_[ok]

    print(f"\n{'='*74}\n  LEARNED d_gw vs AN INDEPENDENT WATER TABLE — run {a.run_id}\n{'='*74}")
    print(f"  reaches matched: {len(learned):,}   controls: {X.shape[1]} of {len(HEAD_INPUTS)} head inputs\n")
    print(f"  learned d_gw   p10={np.percentile(learned,10):7.2f}  p50={np.percentile(learned,50):7.2f}  p90={np.percentile(learned,90):7.2f}")
    print(f"  reference      p10={np.percentile(ref,10):7.2f}  p50={np.percentile(ref,50):7.2f}  p90={np.percentile(ref,90):7.2f}")

    print(f"\n  --- MAGNITUDE ---")
    print(f"  {'raw correlation':<52} {pcorr(learned, ref):+.3f}")
    size = X[:, [list(inputs).index('log10_uparea')]] if 'log10_uparea' in inputs else None
    if size is not None:
        print(f"  {'controlling for river size alone':<52} {pcorr(learned, ref, size):+.3f}")
    print(f"  {'controlling for ALL head inputs':<52} {pcorr(learned, ref, X):+.3f}")
    print("\n  A raw correlation that survives the full control is evidence the field")
    print("  carries water-table information the inputs do not. One that collapses")
    print("  is a projection of the predictors. The 2026-09-16 arms gave -0.36 raw,")
    print("  +0.05 controlled, i.e. nothing.")

    # --- SIGN test, only meaningful with two-way exchange -------------------
    has_depth = e_ok >= 0
    if has_depth.sum() > 0:
        d = depth_e[e_ok[has_depth]]
        lg, rf, Xs = learned[has_depth], ref[has_depth], X[has_depth]
        fin = np.isfinite(d)
        d, lg, rf, Xs = d[fin], lg[fin], rf[fin], Xs[fin]
        pred_gain = (d - lg) < 0        # model says gaining
        true_gain = (d - rf) < 0        # reference says gaining
        acc = (pred_gain == true_gain).mean()
        base = max(true_gain.mean(), 1 - true_gain.mean())
        # Matthews correlation: chance-corrected, robust to the class imbalance.
        tp = np.sum(pred_gain & true_gain); tn = np.sum(~pred_gain & ~true_gain)
        fp = np.sum(pred_gain & ~true_gain); fn = np.sum(~pred_gain & true_gain)
        den = np.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn - fp*fn)/den if den > 0 else np.nan
        print(f"\n  --- SIGN (gaining vs losing regime) ---")
        print(f"  reaches with a routed depth: {len(d):,}")
        print(f"  reference says gaining:              {true_gain.mean()*100:5.1f}%")
        print(f"  model says gaining:                  {pred_gain.mean()*100:5.1f}%")
        print(f"  agreement:                           {acc*100:5.1f}%")
        print(f"  majority-class baseline:             {base*100:5.1f}%   <- beat this or it is vacuous")
        print(f"  Matthews correlation:                {mcc:+.3f}")
        print(f"  partial corr of the two regimes,")
        print(f"    controlling for all head inputs:   "
              f"{pcorr(pred_gain.astype(float), true_gain.astype(float), Xs):+.3f}")
        print("\n  Under `leakance_losing_only: true` this test is VACUOUS — every reach")
        print("  is forced losing, so the model 'predicts' one class by construction.")
    else:
        print("\n  --- SIGN --- skipped: no routed depth in the eval dump.")
    print(f"{'='*74}\n")


if __name__ == "__main__":
    main()
