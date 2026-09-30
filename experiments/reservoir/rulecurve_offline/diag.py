"""Diagnostics of the harmonic rule curve (L4) fits: shape, storage swing vs NID capacity, negative-release share of
the pure-linear variant, purpose breakdown, additive (mean) band split.

Run: ~/projects/ddr/.venv/bin/python diag.py -> diag.json, diag.txt
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import rc
import run_fits as R
import summarise as S


def job(args):
    s, p4, plin = args
    i = R._G["ids"].get_loc(s)
    g = rc.prep(R._G["P"][i], R._G["O"][i], R._G["doy"], R._G["train"], R._G["test"])
    # one representative (non-leap) year of the flux, per doy 1..365
    w = 2 * np.pi * np.arange(1, 366) / 365.25
    c = np.array([p4[k] for k in rc.CK])
    r = g["Ibar"] * (c[0] * np.sin(w) + c[1] * np.cos(w) + c[2] * np.sin(2 * w) + c[3] * np.cos(2 * w))
    S0 = np.cumsum(r)  # m3/s * day
    swing_days = (S0.max() - S0.min()) / g["Ibar"] if g["Ibar"] > 0 else np.nan  # in days of mean inflow
    ca = g["C"] - g["Cbar"]
    corr = float(np.corrcoef(r, ca)[0, 1]) if r.std() > 0 and ca.std() > 0 else np.nan
    # pure-linear variant: share of test days with a negative release
    Q, _ = rc.run(g, "L4lin", plin)
    neg = float((Q[g["test"]] < 0).mean())
    return dict(STAID=s, fill_peak_doy=int(np.argmax(r) + 1), draw_peak_doy=int(np.argmin(r) + 1),
                smax_doy=int(np.argmax(S0) + 1), smin_doy=int(np.argmin(S0) + 1),
                clim_peak_doy=int(np.argmax(g["C"]) + 1), corr_flux_clim=corr,
                swing_frac_annual=float(swing_days / 365.25), L4lin_neg_test=neg)


def circ_diff(a, b):
    d = (a - b + 182) % 365 - 182
    return d


if __name__ == "__main__":
    df, dam, ctl = S.load()
    jobs = []
    for s in list(dam.index) + list(ctl.STAID):
        row = df.loc[s]
        jobs.append((s, {k: float(row[f"L4_p_{k}"]) for k in rc.CK},
                     {"T0": float(row["L4lin_p_T0"]), **{k: float(row[f"L4lin_p_{k}"]) for k in rc.CK}}))
    with Pool(12, initializer=R._init) as pool:
        out = pd.DataFrame(pool.map(job, jobs)).set_index("STAID")
    out.to_csv(f"{rc.HERE}/diag_by_gauge.csv")
    res, lines = {}, []
    for setname, sel in [("on_reach", dam.on_reach.values), ("all_dams", np.ones(len(dam), bool))]:
        d = dam[sel].join(out)
        c = ctl[sel]
        oc = out.reindex(c.STAID.values); oc.index = c.index; c = c.join(oc)
        e = {}
        for sub, idx in [("all", d.index), ("DOR>0.5", d.index[d.nid_dor > 0.5])]:
            x, y = d.loc[idx], c.loc[idx]
            lag = circ_diff(x.fill_peak_doy.values, x.clim_peak_doy.values)
            e[sub] = dict(
                n=len(idx),
                corr_flux_vs_model_climatology=[round(float(v), 3) for v in np.nanpercentile(x.corr_flux_clim, [25, 50, 75])],
                frac_corr_gt_0p5=round(float((x.corr_flux_clim > 0.5).mean()), 3),
                fill_peak_minus_clim_peak_days=[round(float(v), 1) for v in np.percentile(lag, [25, 50, 75])],
                swing_frac_of_annual_inflow=[round(float(v), 4) for v in np.nanpercentile(x.swing_frac_annual, [25, 50, 75, 90])],
                swing_over_dor=[round(float(v), 3) for v in np.nanpercentile(x.swing_frac_annual / x.nid_dor, [25, 50, 75, 90])],
                frac_swing_gt_dor=round(float((x.swing_frac_annual > x.nid_dor).mean()), 3),
                control_swing_frac=[round(float(v), 4) for v in np.nanpercentile(y.swing_frac_annual, [25, 50, 75, 90])],
                L4lin_neg_test_dam=dict(mean=round(float(x.L4lin_neg_test.mean()), 4), median=round(float(x.L4lin_neg_test.median()), 4),
                                        frac_gt_1pct=round(float((x.L4lin_neg_test > 0.01).mean()), 3)),
                L4lin_neg_test_ctl_mean=round(float(y.L4lin_neg_test.mean()), 4),
                gain_by_swing_le_dor=dict(
                    le=S.paired((x.L4_nse - x.L0_nse)[x.swing_frac_annual <= x.nid_dor]),
                    gt=S.paired((x.L4_nse - x.L0_nse)[x.swing_frac_annual > x.nid_dor])),
                by_purpose={str(k): S.paired(v.L4_nse - v.L0_nse) for k, v in x.groupby(x.dam_purpose.fillna("unknown")) if len(v) >= 8},
                band_means={lab: dict(dam_L0=round(float(x[f"L0_band_{lab}"].mean()), 4), dam_L4=round(float(x[f"L4_band_{lab}"].mean()), 4),
                                      dam_L1=round(float(x[f"L1_band_{lab}"].mean()), 4),
                                      ctl_L0=round(float(y[f"L0_band_{lab}"].mean()), 4), ctl_L4=round(float(y[f"L4_band_{lab}"].mean()), 4))
                            for lab in rc.BAND_LAB},
            )
        res[setname] = e
        lines.append(f"===== {setname}")
        lines.append(json.dumps(e, indent=1))
    json.dump(res, open(f"{rc.HERE}/diag.json", "w"), indent=1)
    open(f"{rc.HERE}/diag.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
