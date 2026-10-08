#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "xarray", "zarr>=3", "netCDF4", "scipy", "matplotlib"]
# ///
"""E-10: bias decomposition per inflow product, the router's added lag, n_0 vs timing lead, and error by time scale.

Six fixed-channel models (cp_common.MODELS), test water years 1996-2010, the 2,365 eval gauges.

Per gauge and product, for the unrouted summed inflow (the run's summed-Q' baseline) and for the routed flow,
each against USGS observations:
  volume ratio     sum(sim) / sum(obs) on jointly valid days
  FHV              DDR/ddrs flow-duration peak bias: sort each series, top 2 %, 100 sum(s - o) / sum(o)
                   (src/training/metrics.rs::Metrics::compute). Timing-free.
  event peak bias  the section 38.2 variant: sim on the observed top-2 % days (cross_arm_fields.py). Carries timing.
  NSE, KGE (2009)
  lags             best cross-correlation lag (days, -3..10, positive = second series later) as in
                   experiments/stage_roughness/routing_lag.py::best_lag: summed -> obs (timing lead of the
                   product), routed -> obs (residual), summed -> routed (what the router adds).
Medians with gauge-bootstrap 95 % CIs, overall and by drainage-area class.

n_0 vs lead: across products (n = 6, exploratory) and within each product (Spearman over gauges, basin-median
n_0 over the gauge's upstream reaches).

Error by time scale: each product's routed residual r_k = routed_k - obs split with bands() (water-year mean,
30-day centred MA of the rest, remainder) and into the part common to all six products c = mean_k r_k and the
product-specific part s_k = r_k - c. Per gauge, SS(r_k) = SS(c) + SS(s_k) + 2<c, s_k> exactly, and each part
splits into its three bands plus band cross terms, so the nine shares sum to one per gauge.

    experiments/cross_product/e10_bias.py
Writes <OUT>/e10_per_gauge.csv, <OUT>/e10_bands_per_gauge.csv, experiments/cross_product/e10_bias.txt and
research/figures/cross-product/{n0_vs_lead,band_shares}.png.
"""
import io
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cp_common as C

LAGS = np.arange(-3, 11)


def best_lag(a, b, min_days=365):
    """Vectorised over gauges: lag maximising corr(a(t), b(t + lag)), NaN-safe.

    Same lag range, sign convention, 365-day minimum and tie rule (first lag reaching the maximum wins,
    scanning -3..10) as routing_lag.py::best_lag, which loops one gauge at a time.
    """
    G, T = a.shape
    best = np.zeros(G, dtype=float)
    best_r = np.full(G, -np.inf)
    rall = np.full((G, LAGS.size), np.nan)
    for j, lag in enumerate(LAGS):
        if lag >= 0:
            x, y = a[:, :T - lag], b[:, lag:]
        else:
            x, y = a[:, -lag:], b[:, :T + lag]
        m = np.isfinite(x) & np.isfinite(y)
        n = m.sum(1)
        xs, ys = np.where(m, x, 0.0), np.where(m, y, 0.0)
        mx, my = xs.sum(1) / np.maximum(n, 1), ys.sum(1) / np.maximum(n, 1)
        dx, dy = np.where(m, x - mx[:, None], 0.0), np.where(m, y - my[:, None], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (dx * dy).sum(1) / np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
        r = np.where(n >= min_days, r, np.nan)
        rall[:, j] = r
        upd = np.isfinite(r) & (r > best_r)
        best[upd], best_r[upd] = lag, r[upd]
    best[~np.isfinite(best_r)] = np.nan
    # sub-day refinement (not in routing_lag.py): vertex of the parabola through the peak and its two neighbours
    j = np.searchsorted(LAGS, np.nan_to_num(best, nan=LAGS[0])).clip(1, LAGS.size - 2)
    rm, r0, rp = rall[np.arange(G), j - 1], rall[np.arange(G), j], rall[np.arange(G), j + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        off = 0.5 * (rm - rp) / (rm - 2 * r0 + rp)
    interior = (best > LAGS[0]) & (best < LAGS[-1])
    frac = np.where(interior & np.isfinite(off) & (np.abs(off) <= 0.5), LAGS[j] + off, best)
    frac[~np.isfinite(best)] = np.nan
    return best, best_r, frac


def skill(P, O, min_days=365):
    """Per-gauge volume ratio, FHV (flow-duration), event peak bias, NSE, KGE on jointly finite days."""
    G = P.shape[0]
    out = np.full((G, 5), np.nan)
    for i in range(G):
        m = np.isfinite(P[i]) & np.isfinite(O[i])
        if m.sum() < min_days:
            continue
        p, o = P[i, m], O[i, m]
        so = np.sum((o - o.mean()) ** 2)
        nse = 1 - np.sum((p - o) ** 2) / so if so > 0 else np.nan
        r = np.corrcoef(p, o)[0, 1]
        alpha, beta = p.std() / o.std(), p.mean() / o.mean()
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        ps, os_ = np.sort(p), np.sort(o)
        h = int(round(0.98 * p.size))
        fhv = 100 * np.sum(ps[h:] - os_[h:]) / np.sum(os_[h:])
        k = max(1, int(0.02 * o.size))
        top = np.argsort(o)[-k:]
        epb = 100 * (p[top].sum() - o[top].sum()) / o[top].sum()
        out[i] = (p.sum() / o.sum(), fhv, epb, nse, kge)
    return out


def ss(x):
    return np.nansum(x ** 2, axis=1)


def dot(x, y):
    return np.nansum(x * y, axis=1)


def band_split(r, tt):
    """Per gauge: SS of the three bands and twice their summed cross products, for residual r[g, t]."""
    lo, mi, hi = C.bands(r, tt)
    return ss(lo), ss(mi), ss(hi), 2 * (dot(lo, mi) + dot(lo, hi) + dot(mi, hi))


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.FIG.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=buf)

    gauges, tt, R, S, O = C.load_series()
    area = C.gauge_area(gauges)
    say(f"{gauges.size} gauges x {tt.size} days, {tt[0].date()}..{tt[-1].date()}; "
        f"observations identical across the six runs (eval zarr and baseline)")
    say(f"gauges with drainage area: {np.isfinite(area).sum()}; classes: " +
        ", ".join(f"{n} km2 {np.sum((area >= lo) & (area < hi))}" for n, lo, hi in C.AREA_CLASSES))
    Olag = np.where(O <= 0, np.nan, O)  # routing_lag.py treats obs <= 0 as missing

    # basin-median n_0 and gamma per gauge and product
    comids = C.basin_comids(gauges)
    rows = []
    for k in C.KEYS:
        f = C.load_field(k)
        sk_s, sk_r = skill(S[k], O), skill(R[k], O)
        l_so, r_so, f_so = best_lag(S[k], Olag)
        l_ro, r_ro, f_ro = best_lag(R[k], Olag)
        l_sr, _, f_sr = best_lag(S[k], R[k])
        for i, g in enumerate(gauges):
            c = comids.get(g)
            sub = f.reindex(c).dropna() if c is not None else f.iloc[:0]
            rows.append(dict(model=k, gauge=g, area=area[i],
                             vol_sum=sk_s[i, 0], fhv_sum=sk_s[i, 1], epb_sum=sk_s[i, 2], nse_sum=sk_s[i, 3], kge_sum=sk_s[i, 4],
                             vol_routed=sk_r[i, 0], fhv_routed=sk_r[i, 1], epb_routed=sk_r[i, 2], nse_routed=sk_r[i, 3],
                             kge_routed=sk_r[i, 4], lead=l_so[i], r_sum_obs=r_so[i], lag_routed_obs=l_ro[i],
                             r_routed_obs=r_ro[i], lag_router=l_sr[i], lead_frac=f_so[i], lag_router_frac=f_sr[i],
                             lag_routed_obs_frac=f_ro[i],
                             n0_basin=sub.n0.median() if len(sub) >= 3 else np.nan,
                             gamma_basin=sub.gamma.median() if len(sub) >= 3 else np.nan, n_reach=len(sub)))
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "e10_per_gauge.csv", index=False)

    # ---- tables: unrouted and routed skill / bias, by area class
    metrics = [("vol", "volume ratio", 3), ("fhv", "FHV %", 1), ("epb", "event peak bias %", 1),
               ("nse", "NSE", 3), ("kge", "KGE", 3)]
    classes = [("all", 0, np.inf)] + C.AREA_CLASSES
    say("\n== Unrouted summed inflow vs observed, and routed vs observed: median [gauge-bootstrap 95 % CI]")
    for col, name, nd in metrics:
        say(f"\n-- {name}")
        say(f"{'product':>20} {'area km2':>13} {'n':>5}   {'unrouted':>24}   {'routed':>24}   {'routed - unrouted (paired)':>26}")
        for k in C.KEYS:
            d = df[df.model == k]
            for cname, lo, hi in classes:
                dd = d[(d.area >= lo) & (d.area < hi)] if cname != "all" else d
                a, b = dd[f"{col}_sum"].to_numpy(), dd[f"{col}_routed"].to_numpy()
                say(f"{C.LABEL[k]:>20} {cname:>13} {np.isfinite(b).sum():>5}   {C.fmt_ci(C.boot_median(a), nd):>24}   "
                    f"{C.fmt_ci(C.boot_median(b), nd):>24}   {C.fmt_ci(C.boot_median(b - a), nd):>26}")

    say("\n== Lags (days, positive = later): timing lead of the summed inflow (summed -> obs), lag the router adds "
        "(summed -> routed), residual (routed -> obs). median [CI] / mean [CI]")
    lagcols = [("lead", "summed->obs"), ("lag_router", "summed->routed"), ("lag_routed_obs", "routed->obs")]
    say(f"{'product':>20} {'area km2':>13} {'n':>5}  " + "  ".join(f"{n:>34}" for _, n in lagcols) + "  router = lead")
    for k in C.KEYS:
        d = df[df.model == k]
        for cname, lo, hi in classes:
            dd = d[(d.area >= lo) & (d.area < hi)] if cname != "all" else d
            cells = []
            for c, _ in lagcols:
                x = dd[c].to_numpy()
                med, mean = C.boot_median(x), C.boot_median(x, stat=np.nanmean)
                cells.append(f"{med[0]:.0f} [{med[1]:.0f},{med[2]:.0f}] / {C.fmt_ci(mean, 2)}")
            ok = np.isfinite(dd.lead) & np.isfinite(dd.lag_router)
            say(f"{C.LABEL[k]:>20} {cname:>13} {ok.sum():>5}  " + "  ".join(f"{c:>34}" for c in cells) +
                f"  {100 * np.mean(dd.lead[ok] == dd.lag_router[ok]):5.1f}%")
    say("\n-- the same lags refined to sub-day by a parabola through the correlation peak (not in routing_lag.py): "
        "median [CI]")
    lagf = [("lead_frac", "summed->obs"), ("lag_router_frac", "summed->routed"), ("lag_routed_obs_frac", "routed->obs")]
    say(f"{'product':>20} {'area km2':>13} {'n':>5}  " + "  ".join(f"{n:>22}" for _, n in lagf))
    for k in C.KEYS:
        d = df[df.model == k]
        for cname, lo, hi in classes:
            dd = d[(d.area >= lo) & (d.area < hi)] if cname != "all" else d
            say(f"{C.LABEL[k]:>20} {cname:>13} {np.isfinite(dd.lead_frac).sum():>5}  " +
                "  ".join(f"{C.fmt_ci(C.boot_median(dd[c].to_numpy()), 2):>22}" for c, _ in lagf))

    SFS = {}
    # ---- volume bookkeeping: the routed run scales the gauge reach's own inflow by FLOW_SCALE, the baseline does not
    say("\n== Volume: how much of routed/unrouted is the FLOW_SCALE bookkeeping (gauge_reach_inflow.py), and what is "
        "left for the router. median [CI]")
    say(f"{'product':>20} {'area km2':>13} {'n':>5} {'routed / unrouted':>24} {'unrouted(FLOW_SCALE) / unrouted':>32} "
        f"{'routed / unrouted(FLOW_SCALE)':>30} {'vol ratio, unrouted(FS)':>24}")
    for k in C.KEYS:
        z = np.load(C.OUT / f"gauge_reach_inflow_{k}.npz", allow_pickle=True)
        gi = pd.Index(z["gauge"]).get_indexer(gauges)
        pos = pd.DatetimeIndex(z["days"]).get_indexer(tt)
        assert (gi >= 0).all() and (pos >= 0).all()
        Sfs = S[k] - (1 - z["flow_scale"][gi, None]) * z["q"][gi][:, pos]
        SFS[k] = Sfs
        m = np.isfinite(O) & np.isfinite(R[k]) & np.isfinite(S[k])
        vr = np.where(m, R[k], 0).sum(1) / np.where(m, S[k], 0).sum(1)
        vf = np.where(m, Sfs, 0).sum(1) / np.where(m, S[k], 0).sum(1)
        vrf = np.where(m, R[k], 0).sum(1) / np.where(m, Sfs, 0).sum(1)
        vo = np.where(m, Sfs, 0).sum(1) / np.where(m, O, 0).sum(1)
        enough = m.sum(1) >= 365
        df.loc[df.model == k, "vol_sum_fs"] = np.where(enough, vo, np.nan)
        df.loc[df.model == k, "router_vol"] = np.where(enough, vrf, np.nan)
        for cname, lo, hi in classes:
            s = enough & (((area >= lo) & (area < hi)) if cname != "all" else True)
            say(f"{C.LABEL[k]:>20} {cname:>13} {s.sum():>5} {C.fmt_ci(C.boot_median(vr[s])):>24} "
                f"{C.fmt_ci(C.boot_median(vf[s])):>32} {C.fmt_ci(C.boot_median(vrf[s])):>30} {C.fmt_ci(C.boot_median(vo[s])):>24}")
    df.to_csv(C.OUT / "e10_per_gauge.csv", index=False)
    say("\nDoes the router absorb volume bias? If it did, its volume change (routed / like-for-like unrouted) would fall "
        "where the inflow over-predicts. Spearman over gauges:")
    say(f"{'product':>20} {'n':>5} {'rho(router vol, inflow vol ratio)':>34} {'rho(router vol, basin n0)':>26} "
        f"{'rho(router vol, area)':>22} {'median router vol':>18} {'median inflow vol ratio (FS)':>29}")
    for k in C.KEYS:
        d = df[(df.model == k) & np.isfinite(df.router_vol) & np.isfinite(df.vol_sum_fs) & np.isfinite(df.n0_basin)]
        say(f"{C.LABEL[k]:>20} {len(d):5d} {spearmanr(d.router_vol, d.vol_sum_fs).statistic:+34.3f} "
            f"{spearmanr(d.router_vol, d.n0_basin).statistic:+26.3f} {spearmanr(d.router_vol, d.area).statistic:+22.3f} "
            f"{d.router_vol.median():18.3f} {d.vol_sum_fs.median():29.3f}")
    pv = df.groupby("model")[["router_vol", "vol_sum_fs", "n0_basin"]].median()
    say(f"  across products (n = 6): rho(median router vol, median inflow vol ratio) "
        f"{spearmanr(pv.router_vol, pv.vol_sum_fs).statistic:+.3f}; rho(median router vol, median basin n0) "
        f"{spearmanr(pv.router_vol, pv.n0_basin).statistic:+.3f}")

    # ---- population check on the summed-Q' baseline medians quoted in findings section 38.1
    say("\n== Baseline population: the manifest's precomputed summed-Q' medians over all baseline gauges vs the eval set")
    import json
    for k in C.KEYS:
        man = json.loads((C.RUNS / C.RUN[k] / "baseline" / "manifest.json").read_text())
        ids = np.array(man["gage_ids"])
        nse = np.array([np.nan if v is None else v for v in man["metrics"]["nse"]], float)
        kge = np.array([np.nan if v is None else v for v in man["metrics"]["kge"]], float)
        sel = np.isin(ids, gauges)
        say(f"{C.LABEL[k]:>20}: all {ids.size} gauges NSE {np.nanmedian(nse):.3f} KGE {np.nanmedian(kge):.3f}   "
            f"eval {sel.sum()} gauges NSE {np.nanmedian(nse[sel]):.3f} KGE {np.nanmedian(kge[sel]):.3f}")

    # ---- n_0 vs timing lead
    say("\n== n_0 vs timing lead")
    prod = []
    for k in C.KEYS:
        d = df[df.model == k]
        ok = np.isfinite(d.n0_basin) & np.isfinite(d.lead)
        prod.append(dict(model=k, n0_med=d.n0_basin.median(), lead_med=d.lead.median(), lead_mean=d.lead.mean(),
                         router_mean=d.lag_router.mean(), vol_med=d.vol_sum.median(), fhv_med=d.fhv_sum.median(), epb_med=d.epb_sum.median(),
                         n0_net=C.load_field(k).n0.median(), n=int(ok.sum()),
                         rho_lead=spearmanr(d.n0_basin[ok], d.lead[ok]).statistic,
                         lead_frac_med=d.lead_frac.median(), lead_frac_mean=d.lead_frac.mean(),
                         rho_lead_frac=spearmanr(d.n0_basin[ok], d.lead_frac[ok], nan_policy="omit").statistic,
                         rho_router=spearmanr(d.n0_basin[ok & np.isfinite(d.lag_router)], d.lag_router[ok & np.isfinite(d.lag_router)]).statistic,
                         rho_vol=spearmanr(d.n0_basin[ok], d.vol_sum[ok], nan_policy="omit").statistic,
                         rho_fhv=spearmanr(d.n0_basin[ok], d.fhv_sum[ok], nan_policy="omit").statistic,
                         rho_epb=spearmanr(d.n0_basin[ok], d.epb_sum[ok], nan_policy="omit").statistic))
    P = pd.DataFrame(prod).set_index("model")
    P["abs_logvol"] = np.abs(np.log(P.vol_med))
    say("Across products (n = 6, EXPLORATORY: six points, Spearman needs |rho| >= 0.89 for p < 0.05):")
    say(f"{'product':>20} {'basin n0 med':>12} {'network n0 med':>14} {'lead median':>11} {'lead mean':>9} "
        f"{'router mean':>11} {'vol ratio med':>13} {'FHV med %':>9} {'event pk %':>9}")
    for k in C.KEYS:
        p = P.loc[k]
        say(f"{C.LABEL[k]:>20} {p.n0_med:12.3f} {p.n0_net:14.3f} {p.lead_med:11.0f} {p.lead_mean:9.2f} "
            f"{p.router_mean:11.2f} {p.vol_med:13.3f} {p.fhv_med:9.1f} {p.epb_med:9.1f}")
    say(f"  sub-day lead (median / mean): " + ", ".join(f"{C.LABEL[k]} {P.loc[k].lead_frac_med:.2f} / {P.loc[k].lead_frac_mean:.2f}" for k in C.KEYS))
    for xc, nm in (("lead_frac_med", "median sub-day lead"), ("lead_frac_mean", "mean sub-day lead"), ("lead_mean", "mean lead"), ("router_mean", "mean router lag"), ("vol_med", "median volume ratio"),
                   ("fhv_med", "median FHV"), ("epb_med", "median event peak bias"),
                   ("abs_logvol", "|log median volume ratio|")):
        say(f"  rho(basin-median n0, {nm}) across products = {spearmanr(P.n0_med, P[xc]).statistic:+.3f}   "
            f"Pearson {np.corrcoef(P.n0_med, P[xc])[0, 1]:+.3f}")
    say(f"  with network-median n0 instead: rho(n0, mean lead) {spearmanr(P.n0_net, P.lead_mean).statistic:+.3f}, "
        f"rho(n0, median sub-day lead) {spearmanr(P.n0_net, P.lead_frac_med).statistic:+.3f}, "
        f"rho(n0, mean sub-day lead) {spearmanr(P.n0_net, P.lead_frac_mean).statistic:+.3f}, "
        f"rho(n0, median volume ratio) {spearmanr(P.n0_net, P.vol_med).statistic:+.3f}")
    say(f"  the confound: rho(mean lead, median volume ratio) across products {spearmanr(P.lead_mean, P.vol_med).statistic:+.3f}, "
        f"rho(median sub-day lead, median volume ratio) {spearmanr(P.lead_frac_med, P.vol_med).statistic:+.3f}, "
        f"rho(mean sub-day lead, median volume ratio) {spearmanr(P.lead_frac_mean, P.vol_med).statistic:+.3f}")
    say("Within each product (Spearman over gauges of basin-median n0 against the product's own per-gauge error):")
    say(f"{'product':>20} {'n':>5} {'rho(n0, lead)':>13} {'rho(n0, router lag)':>19} {'rho(n0, vol)':>12} "
        f"{'rho(n0, FHV)':>12} {'rho(n0, event peak)':>19} {'rho(n0, sub-day lead)':>21}")
    for k in C.KEYS:
        p = P.loc[k]
        say(f"{C.LABEL[k]:>20} {int(p.n):5d} {p.rho_lead:+13.3f} {p.rho_router:+19.3f} {p.rho_vol:+12.3f} {p.rho_fhv:+12.3f} {p.rho_epb:+19.3f} {p.rho_lead_frac:+21.3f}")
    ok = np.isfinite(df.n0_basin) & np.isfinite(df.lead) & np.isfinite(df.fhv_sum)
    say(f"  pooled over products and gauges (n = {ok.sum()}): rho(n0, lead) {spearmanr(df.n0_basin[ok], df.lead[ok]).statistic:+.3f}  "
        f"rho(n0, FHV) {spearmanr(df.n0_basin[ok], df.fhv_sum[ok]).statistic:+.3f}  "
        f"rho(n0, event peak) {spearmanr(df.n0_basin[ok], df.epb_sum[ok], nan_policy='omit').statistic:+.3f}  "
        f"rho(n0, vol) {spearmanr(df.n0_basin[ok], df.vol_sum[ok], nan_policy='omit').statistic:+.3f}")
    # demeaned within product: does a gauge that leads more than its product's typical gauge get a rougher basin?
    dm = df[ok].copy()
    for c in ("n0_basin", "lead"):
        dm[c + "_dm"] = dm[c] - dm.groupby("model")[c].transform("median")
    say(f"  pooled after removing each product's median (within-product only): rho(n0, lead) "
        f"{spearmanr(dm.n0_basin_dm, dm.lead_dm).statistic:+.3f}")
    say(f"  pooled sub-day: rho(n0, sub-day lead) {spearmanr(df.n0_basin, df.lead_frac, nan_policy='omit').statistic:+.3f}")

    # ---- error by time scale
    say("\n== Error by time scale: routed residual r = routed - obs, bands = water-year mean / 30-day MA / remainder")
    Rr = {k: np.where(np.isfinite(O), R[k] - O, np.nan) for k in C.KEYS}
    Sr = {k: np.where(np.isfinite(O), SFS[k] - O, np.nan) for k in C.KEYS}  # like-for-like unrouted (FLOW_SCALE applied)
    nvalid = np.isfinite(O).sum(1)
    keep = nvalid >= 365
    common = np.mean([Rr[k] for k in C.KEYS], axis=0)
    cb = band_split(common, tt)
    ssc = ss(common)
    frames = []
    for k in C.KEYS:
        spec = Rr[k] - common
        sb = band_split(spec, tt)
        rb = band_split(Rr[k], tt)
        ub = band_split(Sr[k], tt)
        tot, tot_u, sss = ss(Rr[k]), ss(Sr[k]), ss(spec)
        cross_cs = 2 * dot(common, spec)
        frames.append(pd.DataFrame(dict(
            model=k, gauge=gauges, area=area, ss_routed=tot, ss_unrouted=tot_u,
            r_low=rb[0] / tot, r_mid=rb[1] / tot, r_high=rb[2] / tot, r_x=rb[3] / tot,
            u_low=ub[0] / tot, u_mid=ub[1] / tot, u_high=ub[2] / tot,
            c_low=cb[0] / tot, c_mid=cb[1] / tot, c_high=cb[2] / tot, c_x=cb[3] / tot,
            s_low=sb[0] / tot, s_mid=sb[1] / tot, s_high=sb[2] / tot, s_x=sb[3] / tot, cs_x=cross_cs / tot,
            cl_of_c=cb[0] / ssc, cm_of_c=cb[1] / ssc, ch_of_c=cb[2] / ssc,
            sl_of_s=sb[0] / sss, sm_of_s=sb[1] / sss, sh_of_s=sb[2] / sss))[keep])
    B = pd.concat(frames, ignore_index=True)
    ident = B[["c_low", "c_mid", "c_high", "c_x", "s_low", "s_mid", "s_high", "s_x", "cs_x"]].sum(1)
    assert np.allclose(ident, 1.0, atol=1e-6), f"identity failed: {ident.min()} {ident.max()}"
    say(f"gauges with >= 365 observed days: {keep.sum()}; per-gauge identity SS(r_k) = sum of nine parts holds to 1e-6")
    B.to_csv(C.OUT / "e10_bands_per_gauge.csv", index=False)

    say("\n-- Band shares of each product's own routed MSE (median [CI] over gauges); band cross terms make the "
        "medians not sum to 1, the band-sum column is median (low+mid+high)")
    say(f"{'product':>20} {'n':>5} {'water-year mean':>22} {'30-day MA':>22} {'event (< ~30 d)':>22} {'band sum':>8}")
    for k in C.KEYS:
        d = B[B.model == k]
        say(f"{C.LABEL[k]:>20} {len(d):5d} {C.fmt_ci(C.boot_median(d.r_low)):>22} {C.fmt_ci(C.boot_median(d.r_mid)):>22} "
            f"{C.fmt_ci(C.boot_median(d.r_high)):>22} {np.median(d.r_low + d.r_mid + d.r_high):8.3f}")
    say("\n-- What routing changed, per band: median over gauges of SS_band(unrouted residual) / SS_band(routed residual); "
        "unrouted = summed inflow with the gauge reach scaled by FLOW_SCALE, so only the router differs")
    say(f"{'product':>20} {'WY mean unrouted/routed':>24} {'30-d unrouted/routed':>21} {'event unrouted/routed':>22}  "
        f"{'MSE unrouted/routed':>20}")
    for k in C.KEYS:
        d = B[B.model == k]
        say(f"{C.LABEL[k]:>20} {C.fmt_ci(C.boot_median(d.u_low / d.r_low)):>24} {C.fmt_ci(C.boot_median(d.u_mid / d.r_mid)):>21} "
            f"{C.fmt_ci(C.boot_median(d.u_high / d.r_high)):>22}  {C.fmt_ci(C.boot_median(d.ss_unrouted / d.ss_routed)):>20}")
    say("   share of gauges where routing reduced the band's SS (WY mean / 30-d / event): " + "; ".join(
        f"{C.LABEL[k]} {np.mean(B[B.model == k].u_low > B[B.model == k].r_low):.2f} / {np.mean(B[B.model == k].u_mid > B[B.model == k].r_mid):.2f}"
        f" / {np.mean(B[B.model == k].u_high > B[B.model == k].r_high):.2f}" for k in C.KEYS))

    # exact average over the six products: mean_k SS(r_k) = SS(c) + mean_k SS(s_k), no cross term
    say("\n-- The average product (exact, no cross term): mean_k SS(r_k) = SS(common) + mean_k SS(specific_k). "
        "Shares of mean_k SS(r_k), median [CI] over gauges")
    tot_avg = np.mean([ss(Rr[k]) for k in C.KEYS], axis=0)
    spec_b = [band_split(Rr[k] - common, tt) for k in C.KEYS]
    spec_avg = [np.mean([b[j] for b in spec_b], axis=0) for j in range(4)]
    A = pd.DataFrame(dict(gauge=gauges, area=area, common=ssc / tot_avg, specific=np.mean([ss(Rr[k] - common) for k in C.KEYS], axis=0) / tot_avg,
                          c_low=cb[0] / tot_avg, c_mid=cb[1] / tot_avg, c_high=cb[2] / tot_avg, c_x=cb[3] / tot_avg,
                          s_low=spec_avg[0] / tot_avg, s_mid=spec_avg[1] / tot_avg, s_high=spec_avg[2] / tot_avg,
                          s_x=spec_avg[3] / tot_avg))[keep]
    assert np.allclose(A.common + A.specific, 1.0, atol=1e-6)
    say(f"  common {C.fmt_ci(C.boot_median(A.common))}   specific {C.fmt_ci(C.boot_median(A.specific))}   (n = {len(A)})")
    say(f"  common:   WY mean {C.fmt_ci(C.boot_median(A.c_low))}  30-d {C.fmt_ci(C.boot_median(A.c_mid))}  event {C.fmt_ci(C.boot_median(A.c_high))}")
    say(f"  specific: WY mean {C.fmt_ci(C.boot_median(A.s_low))}  30-d {C.fmt_ci(C.boot_median(A.s_mid))}  event {C.fmt_ci(C.boot_median(A.s_high))}")
    say("  means over gauges (additive): " + "  ".join(f"{c} {A[c].mean():.3f}" for c in ("c_low", "c_mid", "c_high", "c_x", "s_low", "s_mid", "s_high", "s_x")))
    for cname, lo, hi in C.AREA_CLASSES:
        aa = A[(A.area >= lo) & (A.area < hi)]
        say(f"  {cname:>13} km2 (n = {len(aa)}): common {np.median(aa.common):.3f}, specific {np.median(aa.specific):.3f}, "
            f"specific event {np.median(aa.s_high):.3f}, common event {np.median(aa.c_high):.3f}")
    A.to_csv(C.OUT / "e10_bands_average_product.csv", index=False)

    say("\n-- Common vs product-specific: share of each product's routed MSE (mean over gauges, additive; the nine "
        "columns sum to 1), then medians")
    cols = ["c_low", "c_mid", "c_high", "c_x", "s_low", "s_mid", "s_high", "s_x", "cs_x"]
    say(f"{'product':>20} " + " ".join(f"{c:>7}" for c in cols) + f" {'common':>7} {'specific':>8} {'2<c,s>':>7}")
    for k in C.KEYS:
        d = B[B.model == k]
        mu = d[cols].mean()
        say(f"{C.LABEL[k]:>20} " + " ".join(f"{mu[c]:7.3f}" for c in cols) +
            f" {mu[['c_low', 'c_mid', 'c_high', 'c_x']].sum():7.3f} {mu[['s_low', 's_mid', 's_high', 's_x']].sum():8.3f} {mu.cs_x:7.3f}")
    mu = B[cols].mean()
    say(f"{'mean of six':>20} " + " ".join(f"{mu[c]:7.3f}" for c in cols))
    say("medians [CI] of the headline parts, share of each product's routed MSE:")
    say(f"{'product':>20} {'common, all bands':>22} {'specific, all bands':>22} {'specific event':>22} {'common event':>22}")
    for k in C.KEYS:
        d = B[B.model == k]
        cs = d[["c_low", "c_mid", "c_high", "c_x"]].sum(1)
        sp = d[["s_low", "s_mid", "s_high", "s_x"]].sum(1)
        say(f"{C.LABEL[k]:>20} {C.fmt_ci(C.boot_median(cs)):>22} {C.fmt_ci(C.boot_median(sp)):>22} "
            f"{C.fmt_ci(C.boot_median(d.s_high)):>22} {C.fmt_ci(C.boot_median(d.c_high)):>22}")
    say("\n-- Band composition within each part (median over gauges of SS_band / SS_part)")
    d = B[B.model == C.KEYS[0]]
    say(f"{'common part':>20} WY mean {C.fmt_ci(C.boot_median(d.cl_of_c))}  30-d {C.fmt_ci(C.boot_median(d.cm_of_c))}  "
        f"event {C.fmt_ci(C.boot_median(d.ch_of_c))}  (n = {len(d)})")
    for k in C.KEYS:
        d = B[B.model == k]
        say(f"{'specific, ' + C.LABEL[k]:>20} WY mean {C.fmt_ci(C.boot_median(d.sl_of_s.astype(float)))}  "
            f"30-d {C.fmt_ci(C.boot_median(d.sm_of_s.astype(float)))}  event {C.fmt_ci(C.boot_median(d.sh_of_s.astype(float)))}")
    say("\n-- by area class: median share of routed MSE that is product-specific event-scale / common (all bands)")
    say(f"{'product':>20} " + " ".join(f"{c[0]:>24}" for c in C.AREA_CLASSES))
    for k in C.KEYS:
        d = B[B.model == k]
        cells = []
        for _, lo, hi in C.AREA_CLASSES:
            dd = d[(d.area >= lo) & (d.area < hi)]
            cells.append(f"{np.median(dd.s_high):.2f} / {np.median(dd[['c_low', 'c_mid', 'c_high', 'c_x']].sum(1)):.2f} (n={len(dd)})")
        say(f"{C.LABEL[k]:>20} " + " ".join(f"{c:>24}" for c in cells))

    (C.HERE / "e10_bias.txt").write_text(buf.getvalue())
    figures(df, P, B, A)


def figures(df, P, B, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(C.KEYS, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]))
    # Figure 1: n_0 vs timing lead (sub-day lead)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    edges = np.arange(-1.25, 4.26, 0.5)
    centres = 0.5 * (edges[1:] + edges[:-1])
    for k in C.KEYS:
        d = df[(df.model == k) & np.isfinite(df.lead_frac) & np.isfinite(df.n0_basin)]
        ax[0].scatter(d.lead_frac, d.n0_basin, s=3, alpha=0.12, color=colors[k], rasterized=True)
        b = np.digitize(d.lead_frac, edges) - 1
        inb = (b >= 0) & (b < centres.size)
        g = d[inb].groupby(b[inb]).n0_basin
        med, cnt = g.median(), g.size()
        med = med[cnt >= 25].sort_index()
        ax[0].plot(centres[med.index], med.values, "-o", ms=3, color=colors[k], lw=1.8,
                   label=f"{C.LABEL[k]} (within-product rho {P.loc[k].rho_lead_frac:+.2f})")
    ax[0].set_xlabel("timing lead of the unrouted summed inflow over observed (days, sub-day cross-correlation lag)")
    ax[0].set_ylabel("basin-median learned n0")
    ax[0].set_xlim(-1.5, 4.5)
    ax[0].set_title("within each product: gauges (points), median n0 per half-day bin with >= 25 gauges (lines)", fontsize=10)
    ax[0].legend(fontsize=7.5, loc="upper right")
    ax[0].grid(alpha=0.3)
    offs = {"uh": (8, -12), "dhbv2_lumped": (-60, 8), "dhbv2_dist": (8, -6), "hydrodl_lstm": (-40, 10),
            "nh_daily_lstm": (8, 6), "nh_hourly_mts": (-30, 12)}
    for k in C.KEYS:
        p = P.loc[k]
        ax[1].scatter(p.lead_frac_mean, p.n0_med, s=90, color=colors[k], zorder=3, edgecolor="k")
        ax[1].annotate(C.LABEL[k], (p.lead_frac_mean, p.n0_med), textcoords="offset points", xytext=offs[k], fontsize=8)
    rho = spearmanr(P.n0_med, P.lead_frac_mean).statistic
    ax[1].set_xlabel("mean sub-day timing lead over gauges (days)")
    ax[1].set_ylabel("median basin n0 over gauges")
    ax[1].set_title(f"across products, n = 6 (exploratory): Spearman {rho:+.2f}", fontsize=10)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG / "n0_vs_lead.png", dpi=150, facecolor="white")
    plt.close(fig)

    # Figure 2: per product, the common and the product-specific part of the routed MSE, each split by time scale
    bands_ = [("low", "water-year mean"), ("mid", "30-day"), ("high", "event (< ~30 d)"), ("x", "band cross terms")]
    shade = {"c": ["#08306b", "#2171b5", "#6baed6", "#c6dbef"], "s": ["#7f2704", "#d94801", "#fd8d3c", "#fdd0a2"]}
    groups = [(C.LABEL[k], B[B.model == k]) for k in C.KEYS] + [("average product\n(exact, no cross term)", A)]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    w = 0.38
    for gi_, (name, d) in enumerate(groups):
        for part, dx in (("c", -w / 2), ("s", w / 2)):
            bottom = 0.0
            for bi, (b, bname) in enumerate(bands_):
                v = d[f"{part}_{b}"].mean()
                lab = (f"{'common' if part == 'c' else 'product-specific'}: {bname}") if gi_ == 0 else None
                ax.bar(gi_ + dx, v, w, bottom=bottom, color=shade[part][bi], edgecolor="white", lw=0.5, label=lab)
                bottom += v
        if "cs_x" in d:
            ax.plot(gi_, d.cs_x.mean(), "kD", ms=6, label="2<common, specific> (cross term)" if gi_ == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("share of the routed MSE (mean over gauges)")
    ax.set_title("left bar: error common to all six products (mean residual); right bar: product-specific remainder", fontsize=10)
    ax.legend(fontsize=7.5, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG / "band_shares.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {C.FIG / 'n0_vs_lead.png'} and {C.FIG / 'band_shares.png'}")


if __name__ == "__main__":
    sys.exit(main())
