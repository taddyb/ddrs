"""E-07 seed-pair read-out: how far skill, n_0 and gamma move when only the training seed changes.

Three fixed-channel models (n_0 + gamma head, p = 21, q = 0.65, 2,365 gauges, 500 Adam updates), each
trained at seed 42 and seed 43 on the same binary, config and inflow store:

  UH fixed-channel model     2026-09-12T23-39-03Z (canonical seed 42 per E-01) / 2026-09-27T10-31-30Z
  dHBV2 distributed model    2026-09-13T17-22-30Z / 2026-10-01T02-33-37Z
  dHBV2 lumped model         2026-09-13T13-56-50Z / 2026-10-01T02-33-01Z

Skill. Every number is recomputed from eval/predictions.zarr (loader from the ddrs-eval-plots skill) in the
ddrs convention E-01 uses (paper/e01-lineage@e8e7d3b, research/findings/2026-09-30-paper-model-lineage.md):
drop the first experiment.warmup = 5 days, then score with ddr.validation.Metrics (NSE; KGE 2009 with
r, sd ratio, mean ratio). The recomputed medians are checked against each manifest's median_nse_finite /
median_kge_finite. Per pair:
  - difference of medians (seed 43 minus seed 42), with a paired gauge bootstrap (B = 2000, resampling
    gauges jointly for both seeds) 95 % percentile CI;
  - median of the per-gauge difference (same bootstrap), the share of gauges where seed 43 is better, and
    the spread of |per-gauge difference|.
Residuals. Per gauge, corr(p42 - o, p43 - o) over days where o and both predictions are finite, gauges with
>= 365 such days (prompt 05's convention, experiments/prediction_diversity/pair_diversity.py on
paper/prediction-diversity@8d8000c, there on the full series).
Fields. n_0 (`n`) and gamma from plot/kan_parameters.nc. Network medians over all 346,321 reaches and over
E-08's filtered reaches; per-reach Spearman rho of n_0 and of gamma on the filtered reaches (and on all
reaches, the section 38.2 population, for comparison); rho of n_0 by drainage-area class; the share of
reaches at the n_0 ceiling (>= 0.2495) and at gamma = 0 (< 0.001), as e08_fields.py counts them.
Reach filter (E-08, paper/e10-e08-cross-product@283140e, experiments/cross_product/qprime_medians.py and
e08_fields.py): finite log10_uparea, COMID on every one of the six product stores' divide_id axes, and
median daily Q' over WY1996-2010 >= 1e-3 m3/s in all six products. Per-product medians read from that
commit's cache, /home/tbindas/projects/ddrs/.ddrs/experiments/cross_product/2026-09-30/. Expected 107,225.

    /home/tbindas/projects/ddr/.venv/bin/python experiments/seed_replicates/seed_pairs.py

Writes seed_pairs.txt and seed_pairs.csv next to this script, and per-gauge tables under
/home/tbindas/projects/ddrs/.ddrs/experiments/seed_replicates/<run date>/.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import spearmanr

sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
from load_ddrs_predictions import load_predictions_zarr  # noqa: E402
from ddr.validation import Metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
QMED = Path("/home/tbindas/projects/ddrs/.ddrs/experiments/cross_product/2026-09-30")
OUT = Path("/home/tbindas/projects/ddrs/.ddrs/experiments/seed_replicates") / date.today().isoformat()
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
PRODUCTS = ["uh", "dhbv2_lumped", "dhbv2_dist", "hydrodl_lstm", "nh_daily_lstm", "nh_hourly_mts"]
WARMUP = 5
B, BOOT_SEED = 2000, 0
MIN_DAYS = 365
N_CEIL, G_ZERO = 0.2495, 0.001

PAIRS = [
    ("UH fixed-channel model", "2026-09-12T23-39-03Z-train-and-test", "2026-09-27T10-31-30Z-train-and-test"),
    ("dHBV2 distributed model", "2026-09-13T17-22-30Z-train-and-test", "2026-10-01T02-33-37Z-train-and-test"),
    ("dHBV2 lumped model", "2026-09-13T13-56-50Z-train-and-test", "2026-10-01T02-33-01Z-train-and-test"),
]
# E-01: bitwise twin of the canonical UH seed-42 run; checked here, not used further.
UH_TWIN = "2026-09-27T07-29-47Z-train-and-test"
AREA_BINS = [(0, 2), (2, 3), (3, 4), (4, 9)]


def load_eval(run):
    ds = load_predictions_zarr(RUNS / run / "eval" / "predictions.zarr")
    return (np.asarray(ds.gage_ids.values), np.asarray(ds.time.values),
            ds.predictions.values.astype(np.float64), ds.observations.values.astype(np.float64))


def load_fields(run):
    ds = Dataset(RUNS / run / "plot" / "kan_parameters.nc")
    return (np.asarray(ds["COMID"][:], dtype=np.int64), np.asarray(ds["n"][:], dtype=np.float64),
            np.asarray(ds["gamma"][:], dtype=np.float64))


def reach_filter(comid):
    """E-08's filtered reach set, as a boolean mask over `comid`."""
    at = Dataset(ATTRS)
    la = pd.Series(np.asarray(at["log10_uparea"][:], dtype=np.float64),
                   index=np.asarray(at["COMID"][:], dtype=np.int64)).reindex(comid).to_numpy()
    keep = np.isfinite(la)
    covered = keep.copy()
    for k in PRODUCTS:
        z = np.load(QMED / f"qprime_median_{k}.npz")
        med = pd.Series(z["median"], index=z["divide_id"])
        on = np.isin(comid, z["divide_id"])
        covered &= on
        keep &= on & (med.reindex(comid).to_numpy() >= 1e-3)
    return keep, covered, la


def paired_boot(a, b):
    """Difference of medians and median of differences, each with a paired gauge-bootstrap 95 % CI."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    idx = np.random.default_rng(BOOT_SEED).integers(0, a.size, size=(B, a.size))
    dm = np.median(b[idx], axis=1) - np.median(a[idx], axis=1)
    md = np.median((b - a)[idx], axis=1)
    return dict(
        n=int(a.size), med_a=float(np.median(a)), med_b=float(np.median(b)),
        diff_of_medians=float(np.median(b) - np.median(a)),
        dom_lo=float(np.percentile(dm, 2.5)), dom_hi=float(np.percentile(dm, 97.5)),
        median_of_diff=float(np.median(b - a)),
        mod_lo=float(np.percentile(md, 2.5)), mod_hi=float(np.percentile(md, 97.5)),
        share_b_better=float(np.mean(b > a)),
        absdiff_p50=float(np.median(np.abs(b - a))), absdiff_p90=float(np.percentile(np.abs(b - a), 90)),
        share_absdiff_gt_005=float(np.mean(np.abs(b - a) > 0.05)),
    )


def resid_corr(Pa, Pb, O):
    m = np.isfinite(O) & np.isfinite(Pa) & np.isfinite(Pb)
    nv = m.sum(1)
    ea, eb = np.where(m, Pa - O, 0.0), np.where(m, Pb - O, 0.0)
    ma, mb = ea.sum(1) / np.maximum(nv, 1), eb.sum(1) / np.maximum(nv, 1)
    da, db = np.where(m, ea - ma[:, None], 0.0), np.where(m, eb - mb[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
    rc[nv < MIN_DAYS] = np.nan
    return rc


def boot_median(x):
    x = x[np.isfinite(x)]
    idx = np.random.default_rng(BOOT_SEED).integers(0, x.size, size=(B, x.size))
    m = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(x.size)


def main():
    buf = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=buf)

    OUT.mkdir(parents=True, exist_ok=True)

    # UH twin check (E-01 says bitwise identical)
    g0, t0, P0, O0 = load_eval(PAIRS[0][1])
    g1, t1, P1, O1 = load_eval(UH_TWIN)
    assert np.array_equal(g0, g1) and np.array_equal(t0, t1)
    c0, n0a, ga0 = load_fields(PAIRS[0][1])
    c1, n0b, gb1 = load_fields(UH_TWIN)
    assert np.array_equal(c0, c1)
    say(f"UH seed-42 twin check, {PAIRS[0][1][:20]} vs {UH_TWIN[:20]}: max |dQ| {np.nanmax(np.abs(P0 - P1)):.3g} m3/s, "
        f"max |d n_0| {np.max(np.abs(n0a - n0b)):.3g}, max |d gamma| {np.max(np.abs(ga0 - gb1)):.3g}")

    comid = c0
    keep, covered, la = reach_filter(comid)
    say(f"reaches: {comid.size:,} in the network, {int(covered.sum()):,} with finite log10_uparea and covered by every "
        f"product store, {int(keep.sum()):,} also with median daily Q' >= 1e-3 m3/s in all six (E-08 filter)")
    assert int(keep.sum()) == 107_225, "E-08 filter count changed"

    rows, gauge_tabs = [], []
    for name, r42, r43 in PAIRS:
        say(f"\n== {name}: seed 42 {r42[:20]}, seed 43 {r43[:20]}")
        ga, ta, Pa, Oa = load_eval(r42)
        gb, tb, Pb, Ob = load_eval(r43)
        assert np.array_equal(ta, tb), "time axes differ"
        assert set(ga) == set(gb), "gauge sets differ"
        ix = {s: i for i, s in enumerate(gb)}
        order = [ix[s] for s in ga]
        Pb, Ob = Pb[order], Ob[order]
        assert np.array_equal(np.isnan(Oa), np.isnan(Ob)) and np.allclose(Oa[np.isfinite(Oa)], Ob[np.isfinite(Ob)])
        Pa, Pb, O = Pa[:, WARMUP:], Pb[:, WARMUP:], Oa[:, WARMUP:]
        ma, mb = Metrics(pred=Pa, target=O), Metrics(pred=Pb, target=O)
        for run, m in ((r42, ma), (r43, mb)):
            man = json.loads((RUNS / run / "manifest.json").read_text())["metrics"]
            say(f"  {run[:20]}: recomputed median NSE {np.nanmedian(m.nse):.4f} KGE {np.nanmedian(m.kge):.4f}; "
                f"manifest {man['median_nse_finite']:.4f} / {man['median_kge_finite']:.4f}; "
                f"{int(np.isfinite(m.nse).sum())} gauges with finite NSE")
        s_nse, s_kge = paired_boot(ma.nse, mb.nse), paired_boot(ma.kge, mb.kge)
        for lab, s in (("NSE", s_nse), ("KGE", s_kge)):
            say(f"  {lab} ({s['n']} gauges): median {s['med_a']:.4f} -> {s['med_b']:.4f}, difference of medians "
                f"{s['diff_of_medians']:+.4f} [{s['dom_lo']:+.4f}, {s['dom_hi']:+.4f}]; median per-gauge difference "
                f"{s['median_of_diff']:+.4f} [{s['mod_lo']:+.4f}, {s['mod_hi']:+.4f}]; seed 43 better at "
                f"{100 * s['share_b_better']:.1f} %; |difference| p50 {s['absdiff_p50']:.4f}, p90 {s['absdiff_p90']:.4f}, "
                f"> 0.05 at {100 * s['share_absdiff_gt_005']:.1f} %")
        rc = resid_corr(Pa, Pb, O)
        rmed, rlo, rhi, rn = boot_median(rc)
        say(f"  routed residual correlation per gauge ({rn} gauges with >= {MIN_DAYS} days): median {rmed:.4f} "
            f"[{rlo:.4f}, {rhi:.4f}], p10 {np.nanpercentile(rc, 10):.4f}, p90 {np.nanpercentile(rc, 90):.4f}, "
            f"min {np.nanmin(rc):.4f}")
        # full-series residual correlation, prompt 05's window, for the cross-check
        rc_full = resid_corr(load_eval(r42)[2], load_eval(r43)[2][order], Oa)
        say(f"  same on the full series (no warmup drop): median {np.nanmedian(rc_full):.4f}")

        ca, na, gma = load_fields(r42)
        cb, nb, gmb = load_fields(r43)
        assert np.array_equal(ca, comid) and np.array_equal(cb, comid)
        fld = dict(
            n0_med_all_a=np.median(na), n0_med_all_b=np.median(nb),
            n0_med_f_a=np.median(na[keep]), n0_med_f_b=np.median(nb[keep]),
            g_med_all_a=np.median(gma), g_med_all_b=np.median(gmb),
            g_med_f_a=np.median(gma[keep]), g_med_f_b=np.median(gmb[keep]),
            rho_n0_f=spearmanr(na[keep], nb[keep]).statistic, rho_g_f=spearmanr(gma[keep], gmb[keep]).statistic,
            rho_n0_all=spearmanr(na, nb).statistic, rho_g_all=spearmanr(gma, gmb).statistic,
            ceil_f_a=np.mean(na[keep] >= N_CEIL), ceil_f_b=np.mean(nb[keep] >= N_CEIL),
            g0_f_a=np.mean(gma[keep] < G_ZERO), g0_f_b=np.mean(gmb[keep] < G_ZERO),
            dln_f=np.median(np.abs(np.log(na[keep]) - np.log(nb[keep]))),
            g_p10_f_a=np.percentile(gma[keep], 10), g_p90_f_a=np.percentile(gma[keep], 90),
            g_p10_f_b=np.percentile(gmb[keep], 10), g_p90_f_b=np.percentile(gmb[keep], 90),
            n0_p10_f_a=np.percentile(na[keep], 10), n0_p90_f_a=np.percentile(na[keep], 90),
            n0_p10_f_b=np.percentile(nb[keep], 10), n0_p90_f_b=np.percentile(nb[keep], 90),
        )
        fld = {k: float(v) for k, v in fld.items()}
        say(f"  n_0 network median, all {comid.size:,} reaches: {fld['n0_med_all_a']:.4f} -> {fld['n0_med_all_b']:.4f}; "
            f"filtered {int(keep.sum()):,}: {fld['n0_med_f_a']:.4f} -> {fld['n0_med_f_b']:.4f} "
            f"(ratio {max(fld['n0_med_f_a'], fld['n0_med_f_b']) / min(fld['n0_med_f_a'], fld['n0_med_f_b']):.3f}); "
            f"filtered p10-p90 {fld['n0_p10_f_a']:.3f}-{fld['n0_p90_f_a']:.3f} -> {fld['n0_p10_f_b']:.3f}-{fld['n0_p90_f_b']:.3f}")
        say(f"  gamma network median, all: {fld['g_med_all_a']:.4f} -> {fld['g_med_all_b']:.4f}; filtered: "
            f"{fld['g_med_f_a']:.4f} -> {fld['g_med_f_b']:.4f}; filtered p10-p90 {fld['g_p10_f_a']:.3f}-{fld['g_p90_f_a']:.3f} -> "
            f"{fld['g_p10_f_b']:.3f}-{fld['g_p90_f_b']:.3f}")
        say(f"  filtered reaches at the n_0 ceiling: {100 * fld['ceil_f_a']:.1f} % -> {100 * fld['ceil_f_b']:.1f} %; "
            f"at gamma = 0: {100 * fld['g0_f_a']:.1f} % -> {100 * fld['g0_f_b']:.1f} %")
        say(f"  per-reach Spearman rho, filtered: n_0 {fld['rho_n0_f']:+.3f}, gamma {fld['rho_g_f']:+.3f}; all reaches: "
            f"n_0 {fld['rho_n0_all']:+.3f}, gamma {fld['rho_g_all']:+.3f}; median |d ln n_0| (filtered) {fld['dln_f']:.3f}")
        by = []
        for lo, hi in AREA_BINS:
            s = keep & (la >= lo) & (la < hi)
            by.append(f"10^{lo}-10^{hi} km2 ({int(s.sum()):,}) {spearmanr(na[s], nb[s]).statistic:+.2f}")
        say("  rho n_0 by drainage area (filtered): " + ", ".join(by))

        rows.append(dict(model=name, run_s42=r42, run_s43=r43, n_reaches_filtered=int(keep.sum()),
                         **{f"nse_{k}": v for k, v in s_nse.items()}, **{f"kge_{k}": v for k, v in s_kge.items()},
                         resid_r_median=rmed, resid_r_lo=rlo, resid_r_hi=rhi, resid_r_n=rn, **fld))
        gauge_tabs.append(pd.DataFrame(dict(model=name, gauge=ga, nse_s42=ma.nse, nse_s43=mb.nse, kge_s42=ma.kge,
                                            kge_s43=mb.kge, resid_r=rc)))

    # Product pairs among the three models, at matched seeds (42/42, 43/43) and crossed seeds (42/43, 43/42).
    # All six product models in section 38.2 / E-08 share seed 42, i.e. the same head initialisation and gauge
    # batch order; the crossed pairs show how much of their per-reach agreement that shared seed carries.
    say("\n== product pairs among these three models, matched vs crossed seeds (filtered reaches)")
    say(f"{'pair':48s} {'seeds':>6s} {'rho n0':>7s} {'rho gamma':>9s} {'n0 ratio':>8s} {'resid r':>8s}")
    runs = {name: {42: r42, 43: r43} for name, r42, r43 in PAIRS}
    F = {(name, s): load_fields(r)[1:] for name, d in runs.items() for s, r in d.items()}
    E = {}
    for name, d in runs.items():
        for s, r in d.items():
            g, _, P, Oo = load_eval(r)
            ix = {x: i for i, x in enumerate(g)}
            E[(name, s)] = P[[ix[x] for x in g0]][:, WARMUP:]
    Oref = O0[:, WARMUP:]
    xrows = []
    names = [p[0] for p in PAIRS]
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = names[i], names[j]
            for sa, sb in ((42, 42), (43, 43), (42, 43), (43, 42)):
                (na, ga_), (nb, gb_) = F[(a, sa)], F[(b, sb)]
                ma_, mb_ = np.median(na[keep]), np.median(nb[keep])
                rcx = np.nanmedian(resid_corr(E[(a, sa)], E[(b, sb)], Oref))
                r = dict(a=a, b=b, seed_a=sa, seed_b=sb, rho_n0=spearmanr(na[keep], nb[keep]).statistic,
                         rho_gamma=spearmanr(ga_[keep], gb_[keep]).statistic,
                         n0_ratio=max(ma_, mb_) / min(ma_, mb_), resid_r=rcx)
                xrows.append(r)
                say(f"{a + ' / ' + b:48s} {sa}/{sb:<3d} {r['rho_n0']:+7.3f} {r['rho_gamma']:+9.3f} {r['n0_ratio']:8.3f} "
                    f"{r['resid_r']:8.4f}")
    pd.DataFrame(xrows).to_csv(HERE / "product_pairs_by_seed.csv", index=False, float_format="%.6g")

    pd.DataFrame(rows).to_csv(HERE / "seed_pairs.csv", index=False, float_format="%.6g")
    pd.concat(gauge_tabs).to_csv(OUT / "seed_pairs_per_gauge.csv", index=False, float_format="%.6g")
    (HERE / "seed_pairs.txt").write_text(buf.getvalue())
    print(f"wrote {HERE / 'seed_pairs.txt'}, {HERE / 'seed_pairs.csv'}, {OUT / 'seed_pairs_per_gauge.csv'}")


if __name__ == "__main__":
    main()
