#!/usr/bin/env python
"""Extended metrics for the learned dam release full runs (two seeds), paired per gauge.

Arms: off42 / l42 (07-29-47Z / 07-29-55Z), off43 / l43 (10-31-30Z / 10-31-50Z), base (summed Q').
Writes per_gauge_metrics.csv, tables.json, tables.md under this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import binomtest, spearmanr, wilcoxon

OUT = Path(__file__).resolve().parent
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
EXP = Path("/home/tbindas/projects/ddrs/.claude/worktrees/reservoir-options/experiments/reservoir")
ARMS = {
    "off42": "2026-09-27T07-29-47Z-train-and-test",
    "l42": "2026-09-27T07-29-55Z-train-and-test",
    "off43": "2026-09-27T10-31-30Z-train-and-test",
    "l43": "2026-09-27T10-31-50Z-train-and-test",
}
FLOOR = 1e-3


def _to_md(self, index=False, floatfmt=".4f"):
    df = self.reset_index() if index else self
    cols = list(df.columns)
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


pd.DataFrame.to_markdown = _to_md


def load(run):
    z = zarr.open(str(RUNS / run / "eval/predictions.zarr"), mode="r")
    ids = np.array([bytes(r).decode().strip("\x00") for r in z["gage_ids"][:]])
    t = z["time"][:].astype("datetime64[ns]").astype("datetime64[D]")
    return ids, t, z["predictions"][:].astype(np.float64), z["observations"][:].astype(np.float64)


def masked_corr(a, b, m):
    n = m.sum(1)
    a = np.where(m, a, 0.0)
    b = np.where(m, b, 0.0)
    am = a.sum(1) / n
    bm = b.sum(1) / n
    ac = np.where(m, a - am[:, None], 0.0)
    bc = np.where(m, b - bm[:, None], 0.0)
    sa = np.sqrt((ac**2).sum(1))
    sb = np.sqrt((bc**2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ac * bc).sum(1) / (sa * sb)


def metrics(P, O, t):
    G, T = P.shape
    m = np.isfinite(P) & np.isfinite(O)
    n = m.sum(1).astype(float)
    Pm = np.where(m, P, 0.0)
    Om = np.where(m, O, 0.0)
    pmean = Pm.sum(1) / n
    omean = Om.sum(1) / n
    pc = np.where(m, P - pmean[:, None], 0.0)
    oc = np.where(m, O - omean[:, None], 0.0)
    sse = (np.where(m, P - O, 0.0) ** 2).sum(1)
    sso = (oc**2).sum(1)
    nse = 1 - sse / sso
    pstd = np.sqrt((pc**2).sum(1) / n)
    ostd = np.sqrt((oc**2).sum(1) / n)
    r = (pc * oc).sum(1) / (n * pstd * ostd)
    alpha = pstd / ostd
    beta = pmean / omean
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    out = dict(nse=nse, kge=kge, r=r, alpha=alpha, beta=beta, n_days=n)

    # daily-difference NSE, flashiness (Richards-Baker) ratio
    md = m[:, 1:] & m[:, :-1]
    nd = md.sum(1)
    dP = np.where(md, P[:, 1:] - P[:, :-1], 0.0)
    dO = np.where(md, O[:, 1:] - O[:, :-1], 0.0)
    dOm = dO.sum(1) / nd
    out["nse_diff"] = 1 - ((dP - dO) ** 2).sum(1) / (np.where(md, dO - dOm[:, None], 0.0) ** 2).sum(1)
    rb_o = np.abs(dO).sum(1) / np.where(md, O[:, 1:], 0.0).sum(1)
    rb_p = np.abs(dP).sum(1) / np.where(md, P[:, 1:], 0.0).sum(1)
    out["flash_ratio"] = rb_p / rb_o  # < 1: prediction smoother than observed
    # lag-1 autocorrelation of each series
    out["ac1_obs"] = masked_corr(O[:, 1:], O[:, :-1], md)
    out["ac1_pred"] = masked_corr(P[:, 1:], P[:, :-1], md)

    # cross-correlation best lag in -5..5 (positive: prediction late)
    lags = list(range(-5, 6))
    R = np.full((G, len(lags)), np.nan)
    for j, k in enumerate(lags):
        if k >= 0:
            a, b, mm = P[:, k:], O[:, : T - k], m[:, k:] & m[:, : T - k]
        else:
            a, b, mm = P[:, : T + k], O[:, -k:], m[:, : T + k] & m[:, -k:]
        R[:, j] = masked_corr(a, b, mm)
    best = np.nanargmax(R, axis=1)
    out["xcorr_lag"] = np.array(lags)[best].astype(float)
    out["r_best"] = R[np.arange(G), best]
    out["r_gain_at_best_lag"] = out["r_best"] - r

    # seasonal cycle: monthly climatology NSE and anomaly NSE
    month = pd.DatetimeIndex(t).month.values - 1
    clim_p = np.zeros((G, 12))
    clim_o = np.zeros((G, 12))
    for k in range(12):
        mk = m & (month == k)[None, :]
        nk = mk.sum(1)
        clim_p[:, k] = np.where(mk, P, 0.0).sum(1) / nk
        clim_o[:, k] = np.where(mk, O, 0.0).sum(1) / nk
    co_m = clim_o.mean(1)
    out["nse_seasonal"] = 1 - ((clim_p - clim_o) ** 2).sum(1) / ((clim_o - co_m[:, None]) ** 2).sum(1)
    Pa = P - clim_p[:, month]
    Oa = O - clim_o[:, month]
    Oa_c = np.where(m, Oa - (np.where(m, Oa, 0).sum(1) / n)[:, None], 0.0)
    out["nse_anomaly"] = 1 - (np.where(m, Pa - Oa, 0.0) ** 2).sum(1) / (Oa_c**2).sum(1)
    out["seasonal_var_frac_obs"] = ((clim_o - co_m[:, None]) ** 2).mean(1) / (ostd**2)

    # FDC-based: FHV (top 2 %), FLV (bottom 30 %), FMS (20-70 % exceedance, log), Q95 low-flow log ratio
    fhv = np.full(G, np.nan)
    flv = np.full(G, np.nan)
    fms = np.full(G, np.nan)
    q95 = np.full(G, np.nan)
    for i in range(G):
        mi = m[i]
        if mi.sum() < 365:
            continue
        ps = np.sort(P[i, mi])
        os_ = np.sort(O[i, mi])
        nn = len(ps)
        lo, hi = int(round(0.3 * nn)), int(round(0.98 * nn))
        if os_[:lo].sum() > 0:
            flv[i] = 100 * (ps[:lo] - os_[:lo]).sum() / os_[:lo].sum()
        fhv[i] = 100 * (ps[hi:] - os_[hi:]).sum() / os_[hi:].sum()
        lp = np.log(np.maximum(ps, FLOOR))
        lo_ = np.log(np.maximum(os_, FLOOR))
        p20, p70 = np.percentile(lp, [80, 30])
        o20, o70 = np.percentile(lo_, [80, 30])
        if o20 - o70 > 0:
            fms[i] = 100 * ((p20 - p70) - (o20 - o70)) / (o20 - o70)
        q95[i] = np.percentile(lp, 5) - np.percentile(lo_, 5)
    out.update(fhv=fhv, flv=flv, fms=fms, q95_logratio=q95)

    # annual (water-year) peaks: timing lag and magnitude ratio
    wy = pd.DatetimeIndex(t)
    wy = wy.year.values + (wy.month.values >= 10)
    years = np.unique(wy)
    lag_med = np.full(G, np.nan)
    lag_abs = np.full(G, np.nan)
    pk_ratio = np.full(G, np.nan)
    pk_hit1 = np.full(G, np.nan)
    Pn = np.where(m, P, -np.inf)
    On = np.where(m, O, -np.inf)
    for i in range(G):
        lags_i, ratios_i = [], []
        for y in years:
            sel = wy == y
            if m[i, sel].sum() < 300:
                continue
            idx = np.where(sel)[0]
            io = idx[np.argmax(On[i, sel])]
            ip = idx[np.argmax(Pn[i, sel])]
            lags_i.append(ip - io)
            w = slice(max(io - 3, 0), min(io + 4, T))
            ratios_i.append(np.nanmax(P[i, w]) / O[i, io] if O[i, io] > 0 else np.nan)
        if len(lags_i) >= 5:
            la = np.array(lags_i, float)
            lag_med[i] = np.median(la)
            lag_abs[i] = np.mean(np.abs(la))
            pk_hit1[i] = np.mean(np.abs(la) <= 1)
            pk_ratio[i] = np.nanmedian(ratios_i)
    out.update(peak_lag_median=lag_med, peak_abs_lag=lag_abs, peak_ratio=pk_ratio, peak_within1d=pk_hit1)

    # recession: on observed falling days above the observed median, median log-decline ratio
    rec = np.full(G, np.nan)
    for i in range(G):
        mi = md[i]
        o0, o1 = O[i, :-1], O[i, 1:]
        p0, p1 = P[i, :-1], P[i, 1:]
        med = np.nanmedian(O[i, m[i]])
        sel = mi & (o1 < o0) & (o0 > med) & (o1 > 0) & (p1 > 0) & (p0 > 0)
        if sel.sum() < 50:
            continue
        ko = np.median(np.log(o0[sel] / o1[sel]))
        kp = np.median(np.log(p0[sel] / p1[sel]))
        rec[i] = kp / ko if ko > 0 else np.nan
    out["recession_ratio"] = rec  # <1: prediction recedes slower than observed
    return pd.DataFrame(out)


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return dict(n=int(len(d)))
    nz = d[np.abs(d) > 1e-9]
    up = int((nz > 0).sum())
    rng = np.random.default_rng(0)
    b = np.median(rng.choice(d, (2000, len(d))), axis=1)
    try:
        wp = float(wilcoxon(nz).pvalue) if len(nz) > 10 else None
    except ValueError:
        wp = None
    return dict(n=int(len(d)), median=float(np.median(d)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                up=up, down=int(len(nz) - up), sign_p=float(binomtest(up, len(nz)).pvalue) if len(nz) else None,
                wilcoxon_p=wp, mean=float(d.mean()), trim10=float(np.mean(np.sort(d)[int(0.1 * len(d)):int(0.9 * len(d))])))


def fmt(p):
    if "median" not in p:
        return f"n={p['n']}"
    return f"{p['median']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] {p['up']}/{p['down']} (n={p['n']})"


# ---------------------------------------------------------------- load
ids, t, P, O = load(ARMS["off42"])
series = {"off42": P}
for k in ["l42", "off43", "l43"]:
    i2, t2, P2, O2 = load(ARMS[k])
    assert (i2 == ids).all() and (t2 == t).all()
    assert np.array_equal(np.isfinite(O2), np.isfinite(O))
    series[k] = P2
bm = json.load(open(RUNS / ARMS["off42"] / "baseline/manifest.json"))
B = np.fromfile(RUNS / ARMS["off42"] / "baseline/predictions.f32", dtype=np.float32).reshape(bm["n_gauges"], bm["n_days"]).astype(np.float64)
bidx = {str(g): i for i, g in enumerate(bm["gage_ids"])}
off0 = int((t[0] - np.datetime64(str(bm["time_range_daily"][0])[:10])).astype(int))
Bal = np.full_like(P, np.nan)
for i, s in enumerate(ids):
    if s in bidx:
        Bal[i] = B[bidx[s], off0:off0 + len(t)]
series["base"] = Bal

M = {k: metrics(v, O, t) for k, v in series.items()}
wide = pd.concat({k: v for k, v in M.items()}, axis=1)
wide.columns = [f"{a}__{b}" for a, b in wide.columns]
wide.index = ids
wide.index.name = "STAID"

nid = pd.read_csv(EXP / "nid/nid_dams_by_gauge.csv", dtype={"STAID": str}).set_index("STAID")
sm = pd.read_csv(EXP / "smoke/smoke_gauges.csv", dtype={"STAID": str}).set_index("STAID")
meta = nid[["area_km2", "qmean", "n_nid_ge10mcm", "nid_dor", "nid_on_gauge_reach", "nid_cls"]].reindex(ids)
meta["smoke_role"] = sm.role.reindex(ids)
wide = wide.join(meta)
wide.to_csv(OUT / "per_gauge_metrics.csv")

dammed = (wide.n_nid_ge10mcm > 0).values
onreach = dammed & wide.nid_on_gauge_reach.astype(bool).values
dor = pd.cut(wide.nid_dor.where(dammed), [0, 0.1, 0.5, 1, 2, np.inf], labels=["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"])
groups = {"all": np.ones(len(ids), bool), "dammed": dammed, "undammed": ~dammed, "dam_on_reach": onreach,
          "smoke_dam": (wide.smoke_role == "dam").values, "smoke_control": (wide.smoke_role == "control").values}
for lab in ["<=0.1", "0.1-0.5", "0.5-1", "1-2", ">2"]:
    groups[f"dor {lab}"] = (dor == lab).values

# metrics oriented so that a NEGATIVE delta of the "error" version is an improvement
ERR = {"fhv": "abs", "flv": "abs", "fms": "abs", "q95_logratio": "abs", "peak_abs_lag": "id", "xcorr_lag": "abs",
       "flash_ratio": "log1", "recession_ratio": "log1", "peak_ratio": "log1", "alpha": "log1", "beta": "log1"}
SKILL = ["nse", "kge", "r", "nse_diff", "nse_seasonal", "nse_anomaly", "peak_within1d", "r_best"]


def err(df, k):
    v = df[k].values.astype(float)
    how = ERR[k]
    if how == "abs":
        return np.abs(v)
    if how == "log1":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.abs(np.log(v))
    return v


def delta(k, a, b):
    if k in ERR:
        return err(M[a], k) - err(M[b], k)
    return M[a][k].values - M[b][k].values


tables = {}
md = []
md.append("## Levels: medians per arm (test WY1996-2010)\n")
lvl_rows = []
for g in ["all", "dammed", "undammed", "dam_on_reach", "dor 0.5-1", "dor 1-2", "dor >2"]:
    sel = groups[g]
    for k in ["nse", "kge", "r", "alpha", "beta", "nse_diff", "flash_ratio", "ac1_pred", "ac1_obs", "recession_ratio", "peak_ratio", "peak_abs_lag", "peak_within1d", "xcorr_lag", "r_gain_at_best_lag", "nse_seasonal", "nse_anomaly", "seasonal_var_frac_obs", "fhv", "flv", "fms", "q95_logratio"]:
        row = dict(group=g, metric=k, n=int(sel.sum()))
        for a in ["base", "off42", "l42", "off43", "l43"]:
            row[a] = float(np.nanmedian(M[a][k].values[sel]))
        lvl_rows.append(row)
lvl = pd.DataFrame(lvl_rows)
tables["levels"] = lvl.to_dict("records")
md.append(lvl.to_markdown(index=False, floatfmt=".4f"))

md.append("\n\n## Paired deltas learned minus off, per seed (median [95 % bootstrap], up/down). Error metrics are |error|, so negative = better\n")
pd_rows = []
for k in SKILL + list(ERR):
    for g in groups:
        sel = groups[g]
        d42 = delta(k, "l42", "off42")[sel]
        d43 = delta(k, "l43", "off43")[sel]
        noise = delta(k, "off43", "off42")[sel]
        p42, p43, pn = paired(d42), paired(d43), paired(noise)
        pd_rows.append(dict(metric=k, group=g, seed42=fmt(p42), seed43=fmt(p43), seed_noise_off43_minus_off42=fmt(pn),
                            noise_median_abs=float(np.nanmedian(np.abs(noise)))))
        tables.setdefault("paired", []).append(dict(metric=k, group=g, seed42=p42, seed43=p43, noise=pn))
pdf = pd.DataFrame(pd_rows)
md.append(pdf.to_markdown(index=False))

# KGE decomposition at dammed gauges: signed component deltas
md.append("\n\n## KGE components at dammed gauges, signed deltas (learned minus off)\n")
rows = []
for k in ["r", "alpha", "beta", "kge", "nse"]:
    for g in ["dammed", "undammed", "dam_on_reach", "smoke_dam"]:
        sel = groups[g]
        rows.append(dict(metric=k, group=g, seed42=fmt(paired(M["l42"][k].values[sel] - M["off42"][k].values[sel])),
                         seed43=fmt(paired(M["l43"][k].values[sel] - M["off43"][k].values[sel])),
                         level_off42=float(np.nanmedian(M["off42"][k].values[sel]))))
md.append(pd.DataFrame(rows).to_markdown(index=False))
tables["kge_components"] = rows

# seed consistency of per-gauge deltas
md.append("\n\n## Seed consistency of the per-gauge NSE delta (d42 = l42-off42, d43 = l43-off43)\n")
rows = []
for g in ["dammed", "undammed", "dam_on_reach", "dor 1-2", "smoke_dam", "smoke_control"]:
    sel = groups[g]
    d42 = delta("nse", "l42", "off42")[sel]
    d43 = delta("nse", "l43", "off43")[sel]
    ok = np.isfinite(d42) & np.isfinite(d43)
    rho = spearmanr(d42[ok], d43[ok]).correlation
    same = np.mean(np.sign(d42[ok]) == np.sign(d43[ok]))
    both_up = np.mean((d42[ok] > 0) & (d43[ok] > 0))
    avg = paired((d42 + d43) / 2)
    rows.append(dict(group=g, n=int(ok.sum()), spearman_d42_d43=float(rho), frac_same_sign=float(same), frac_both_up=float(both_up),
                     seed_mean_delta=fmt(avg), sd_d42=float(np.nanstd(d42)), sd_d43=float(np.nanstd(d43)),
                     sd_off_noise=float(np.nanstd(delta("nse", "off43", "off42")[sel])),
                     mad_d42=float(np.nanmedian(np.abs(d42))), mad_noise=float(np.nanmedian(np.abs(delta("nse", "off43", "off42")[sel])))))
md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
tables["seed_consistency"] = rows

# difference in differences per seed
md.append("\n\n## Difference in differences (dammed paired median minus undammed paired median), per seed\n")
rows = []
for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
    for k in ["nse", "kge", "nse_diff", "r", "peak_abs_lag"]:
        d = delta(k, a, b)
        rows.append(dict(seed=s, metric=k, dammed=float(np.nanmedian(d[dammed])), undammed=float(np.nanmedian(d[~dammed])),
                         did=float(np.nanmedian(d[dammed]) - np.nanmedian(d[~dammed])),
                         dammed_minus_undammed_mean=float(np.nanmean(d[dammed]) - np.nanmean(d[~dammed]))))
md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".5f"))
tables["did"] = rows

# outliers and mean-vs-median
md.append("\n\n## Outliers: mean vs median vs 10 % trimmed mean of the NSE delta\n")
rows = []
for g in ["all", "dammed", "undammed"]:
    sel = groups[g]
    for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
        d = delta("nse", a, b)[sel]
        p = paired(d)
        worst = wide.index[sel][np.nanargmin(d)]
        best = wide.index[sel][np.nanargmax(d)]
        rows.append(dict(group=g, seed=s, median=p["median"], mean=p["mean"], trim10=p["trim10"], min=float(np.nanmin(d)), min_gauge=worst,
                         max=float(np.nanmax(d)), max_gauge=best, n_abs_gt_0p05=int((np.abs(d) > 0.05).sum())))
md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
tables["outliers"] = rows
da = wide.loc["08390800", [c for c in wide.columns if c.endswith("__nse") or c.endswith("__kge")]]
md.append("\n\nDiamond A (08390800): " + ", ".join(f"{k}={v:.3f}" for k, v in da.items()))

# group median vs paired median disagreement: deltas by off-NSE tercile
md.append("\n\n## Why group medians and paired medians disagree: NSE delta by tercile of off-arm NSE, dammed gauges\n")
rows = []
for s, (a, b) in {"42": ("l42", "off42"), "43": ("l43", "off43")}.items():
    base_nse = M[b]["nse"].values
    d = delta("nse", a, b)
    for g in ["dammed", "undammed"]:
        sel = groups[g]
        q = np.nanpercentile(base_nse[sel], [33.3, 66.7])
        for lab, lo, hi in [("low", -np.inf, q[0]), ("mid", q[0], q[1]), ("high", q[1], np.inf)]:
            ss = sel & (base_nse > lo) & (base_nse <= hi)
            rows.append(dict(seed=s, group=g, tercile=lab, n=int(ss.sum()), off_median=float(np.nanmedian(base_nse[ss])),
                             delta_median=float(np.nanmedian(d[ss])), delta_mean=float(np.nanmean(d[ss])),
                             group_median_shift=float(np.nanmedian(M[a]["nse"].values[ss]) - np.nanmedian(base_nse[ss]))))
md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
tables["tercile"] = rows

# release parameters across seeds
rp42 = pd.read_csv(RUNS / ARMS["l42"] / "release_params.csv").set_index("COMID")
rp43 = pd.read_csv(RUNS / ARMS["l43"] / "release_params.csv").set_index("COMID")
j = rp42.join(rp43, lsuffix="_42", rsuffix="_43")
amp42 = np.sqrt(j.a_42**2 + j.b_42**2)
amp43 = np.sqrt(j.a_43**2 + j.b_43**2)
rel = dict(n=int(len(j)), spearman_T0=float(spearmanr(j.T0_days_42, j.T0_days_43).correlation),
           spearman_amp=float(spearmanr(amp42, amp43).correlation), spearman_a=float(spearmanr(j.a_42, j.a_43).correlation),
           spearman_b=float(spearmanr(j.b_42, j.b_43).correlation),
           median_T0_42=float(j.T0_days_42.median()), median_T0_43=float(j.T0_days_43.median()),
           median_abs_log_ratio_T0=float(np.median(np.abs(np.log(j.T0_days_42 / j.T0_days_43)))),
           frac_T0_gt_1d_42=float((j.T0_days_42 > 1).mean()), frac_T0_gt_1d_43=float((j.T0_days_43 > 1).mean()),
           median_amp_42=float(amp42.median()), median_amp_43=float(amp43.median()),
           corr_sign_a=float(np.mean(np.sign(j.a_42) == np.sign(j.a_43))), corr_sign_b=float(np.mean(np.sign(j.b_42) == np.sign(j.b_43))))
md.append("\n\n## Learned release parameters, seed 42 vs seed 43 (1,024 dams)\n\n" + json.dumps(rel, indent=1))
tables["release_params"] = rel

# does the gauge-level NSE delta track the learned T0 of its nearest / largest dam? (dam on reach only, via smoke set)
smk = sm[sm.role == "dam"].copy()
smk["dam_COMID"] = smk.dam_COMID.astype("Int64")
smk = smk.join(rp42[["T0_days"]].rename(columns={"T0_days": "T0_42"}), on="dam_COMID").join(rp43[["T0_days"]].rename(columns={"T0_days": "T0_43"}), on="dam_COMID")
smk["d42"] = pd.Series(delta("nse", "l42", "off42"), index=ids).reindex(smk.index)
smk["d43"] = pd.Series(delta("nse", "l43", "off43"), index=ids).reindex(smk.index)
ok = smk.T0_42.notna() & smk.d42.notna()
rows = dict(n=int(ok.sum()), spearman_dnse_T0_seed42=float(spearmanr(smk.d42[ok], np.log(smk.T0_42[ok])).correlation),
            spearman_dnse_T0_seed43=float(spearmanr(smk.d43[ok], np.log(smk.T0_43[ok])).correlation))
for lab, lo, hi in [("T0<0.3d", 0, 0.3), ("0.3-1d", 0.3, 1), ("1-3d", 1, 3), (">3d", 3, 1e9)]:
    s42 = ok & (smk.T0_42 > lo) & (smk.T0_42 <= hi)
    s43 = ok & (smk.T0_43 > lo) & (smk.T0_43 <= hi)
    rows[lab] = dict(n42=int(s42.sum()), d42=fmt(paired(smk.d42[s42])), n43=int(s43.sum()), d43=fmt(paired(smk.d43[s43])))
md.append("\n\n## Smoke dam gauges: NSE delta against the learned T0 of the gauge's nearest dam\n\n" + json.dumps(rows, indent=1))
tables["dnse_vs_T0"] = rows

# power: seeds needed
d42d = delta("nse", "l42", "off42")
d43d = delta("nse", "l43", "off43")
did = [float(np.nanmedian(d42d[dammed]) - np.nanmedian(d42d[~dammed])), float(np.nanmedian(d43d[dammed]) - np.nanmedian(d43d[~dammed]))]
sd_did = float(np.std(did, ddof=1))
pw = dict(did_per_seed=did, sd_between_seeds=sd_did, mean=float(np.mean(did)),
          seeds_for_se_0p0005=float((sd_did / 0.0005) ** 2), seeds_for_se_0p001=float((sd_did / 0.001) ** 2),
          per_gauge_sd_noise_dammed=float(np.nanstd(delta("nse", "off43", "off42")[dammed])),
          per_gauge_sd_effect_dammed=float(np.nanstd(d42d[dammed])),
          bootstrap_se_of_median_dammed_one_seed=float((tables["paired"][0]["seed42"]["hi"] - tables["paired"][0]["seed42"]["lo"]) / 3.92))
md.append("\n\n## Power\n\n" + json.dumps(pw, indent=1))
tables["power"] = pw

json.dump(tables, open(OUT / "tables.json", "w"), indent=1, default=float)
(OUT / "tables.md").write_text("\n".join(md))
print("\n".join(md))
