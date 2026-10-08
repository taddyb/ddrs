"""Seed pairs vs inflow-product pairs: parameter distance against prediction diversity.

For every pair of trained fixed-channel models (n_0 + gamma head, p = 21, q = 0.65), on the test years
WY1996-2010 and the gauges common to both:

prediction diversity
  resid_corr   median over gauges of corr(p_a - o, p_b - o), the per-gauge correlation of routed residuals
  gain         NSE of the equal-weight average minus the members' mean NSE, per gauge. For weights 1/2,
               sum_i w_i (p_i - o)^2 = (pbar - o)^2 + sum_i w_i (p_i - pbar)^2 at every step, so
               gain = sum_t sum_i w_i (p_i - pbar)^2 / sum_t (o - obar)^2 >= 0: the ambiguity (spread) term of
               Krogh & Vedelsby (1995) divided by var(obs). The identity is asserted per gauge, as in
               experiments/levers/blend_mechanism.py (multi-product-forcing worktree). Median over gauges with a
               paired gauge-bootstrap 95 % CI.
  band shares  the gain split by time scale with blend_mechanism.py's bands(): water-year mean (year-to-year
               volume), 30-day centred moving average of the rest (seasonal), remainder (event scale, < ~30 d).
parameter distance
  n0_ratio     larger / smaller network-median n_0, over the filtered reaches
  rho_n0       per-reach Spearman rho of n_0 over the filtered reaches (covered by every product, median
               local Q' >= 1e-3 m3/s in every product; reach_filter.py)

Seed pairs whose seed-43 run has no eval output yet are written as PENDING rows.
Seed crossing: for the products with a seed-43 twin, every product pair is also scored at all four seed
combinations, to test whether a shared seed inflates product-pair pattern agreement (also the basin-median
n_0 rank correlation over gauges).
Writes pairs.csv, pairs.txt and prediction_diversity.png next to this script.

    /home/tbindas/projects/ddr/.venv/bin/python pair_diversity.py
"""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "levers"))
import common as C  # noqa: E402  (load_preds, align, water_year)

RUNS = C.RUNS
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
GAGES_ADJ = "/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr"  # every run's gages_adjacency
CACHE = pathlib.Path("/home/tbindas/projects/ddrs/.ddrs/experiments/prediction-diversity")
B, SEED = 2000, 1

# Descriptive name -> (run id, inflow product, seed). Product runs from the paper's lineage table.
MODELS = {
    "UH fixed-channel model": ("2026-09-12T23-39-03Z-train-and-test", "uh", 42),
    "dHBV2 lumped model": ("2026-09-13T13-56-50Z-train-and-test", "dhbv2_lumped", 42),
    "dHBV2 distributed model": ("2026-09-13T17-22-30Z-train-and-test", "dhbv2_dist", 42),
    "hydroDL LSTM model": ("2026-09-13T17-21-58Z-train-and-test", "hydrodl_lstm", 42),
    "NH daily LSTM model": ("2026-09-13T13-55-03Z-train-and-test", "nh_daily_lstm", 42),
    "NH hourly MTS-LSTM model": ("2026-09-13T14-14-46Z-train-and-test", "nh_hourly_lstm", 42),
    # seed pairs: the seed-42 member shares a binary with its seed-43 twin
    "UH fixed-channel model, seed 42 (09-27 rerun)": ("2026-09-27T07-29-47Z-train-and-test", "uh", 42),
    "UH fixed-channel model, seed 43": ("2026-09-27T10-31-30Z-train-and-test", "uh", 43),
}
SHORT = {"UH fixed-channel model": "UH", "dHBV2 lumped model": "dHBV2 lumped",
         "dHBV2 distributed model": "dHBV2 dist", "hydroDL LSTM model": "hydroDL LSTM",
         "NH daily LSTM model": "NH daily LSTM", "NH hourly MTS-LSTM model": "NH hourly LSTM",
         "UH fixed-channel model, seed 42 (09-27 rerun)": "UH s42", "UH fixed-channel model, seed 43": "UH s43"}
PRODUCT_MODELS = list(MODELS)[:6]
# E-07 (branch paper/e07-seed43) seed-43 replicates of the two dHBV2 models, release build of 70f6457 (the seed-42
# commit), finished 2026-10-01.
# seed42 = None keeps the 09-13 product run as the seed-42 member; set it if E-07 retrained seed 42 on a new binary.
E07 = {
    "dHBV2 distributed model": dict(product="dhbv2_dist", seed42=None, seed43="2026-10-01T02-33-37Z-train-and-test"),
    "dHBV2 lumped model": dict(product="dhbv2_lumped", seed42=None, seed43="2026-10-01T02-33-01Z-train-and-test"),
}
SEED_PAIRS = [("UH fixed-channel model, seed 42 (09-27 rerun)", "UH fixed-channel model, seed 43", "UH seeds 42/43")]
for _name, _e in E07.items():
    _short = SHORT[_name]
    _a = _name
    if _e["seed42"]:
        _a = f"{_name}, seed 42 (E-07 rerun)"
        MODELS[_a] = (_e["seed42"], _e["product"], 42)
        SHORT[_a] = f"{_short} s42"
    _b = None
    if _e["seed43"]:
        _b = f"{_name}, seed 43"
        MODELS[_b] = (_e["seed43"], _e["product"], 43)
        SHORT[_b] = f"{_short} s43"
    SEED_PAIRS.append((_a, _b, f"{_short} seeds 42/43"))
NOT_AORC = {"uh"}  # every other product is AORC-forced


def bands(d, tt):
    """Split d[g, t] into (water-year mean, 30-day MA of the rest, remainder).

    Copied verbatim from experiments/levers/blend_mechanism.py (multi-product-forcing worktree,
    /home/tbindas/projects/ddrs/.claude/worktrees/agent-a92e512a7c47c97b4, uncommitted there on 2026-09-30).
    """
    wy = C.water_year(tt)
    low = np.empty_like(d)
    for y in np.unique(wy):
        s = wy == y
        low[:, s] = d[:, s].mean(1, keepdims=True)
    rest = d - low
    mid = pd.DataFrame(rest.T).rolling(30, center=True, min_periods=1).mean().to_numpy().T
    return low, mid, rest - mid


def boot_median(x, rng_seed=SEED):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    idx = np.random.default_rng(rng_seed).integers(0, x.size, size=(B, x.size))
    m = np.median(x[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_fields(run):
    ds = Dataset(RUNS / run / "plot" / "kan_parameters.nc")
    return (np.asarray(ds["COMID"][:], dtype=np.int64), np.asarray(ds["n"][:], dtype=np.float64),
            np.asarray(ds["gamma"][:], dtype=np.float64))


def pair_row(a, b, kind, P, O, tt, F, keep):
    Pa, Pb = P[a], P[b]
    m = np.isfinite(O) & np.isfinite(Pa) & np.isfinite(Pb)
    nv = m.sum(1)
    om = np.where(m, O, 0.0)
    mo = om.sum(1) / np.maximum(nv, 1)
    sst = (np.where(m, O - mo[:, None], 0.0) ** 2).sum(1)
    pbar = 0.5 * (Pa + Pb)
    nse_a = 1 - (np.where(m, Pa - O, 0.0) ** 2).sum(1) / sst
    nse_b = 1 - (np.where(m, Pb - O, 0.0) ** 2).sum(1) / sst
    nse_avg = 1 - (np.where(m, pbar - O, 0.0) ** 2).sum(1) / sst
    spread = 0.5 * ((np.where(m, Pa - pbar, 0.0) ** 2).sum(1) + (np.where(m, Pb - pbar, 0.0) ** 2).sum(1)) / sst
    gain = nse_avg - 0.5 * (nse_a + nse_b)
    ok = (nv >= 365) & np.isfinite(nse_avg)
    assert np.allclose(gain[ok], spread[ok], atol=1e-8), "ambiguity identity failed"
    # residual correlation per gauge
    ea, eb = np.where(m, Pa - O, 0.0), np.where(m, Pb - O, 0.0)
    ma, mb = ea.sum(1) / np.maximum(nv, 1), eb.sum(1) / np.maximum(nv, 1)
    da, db = np.where(m, ea - ma[:, None], 0.0), np.where(m, eb - mb[:, None], 0.0)
    rc = (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
    rc[~ok] = np.nan
    # spread by time scale (deviation of a from the average; b's is its negative, same energy)
    parts = np.stack([(np.where(m, x, 0.0) ** 2).sum(1) / sst for x in bands(Pa - pbar, tt)])
    share = parts / parts.sum(0)
    g = gain[ok]
    glo, ghi = boot_median(g)
    rlo, rhi = boot_median(rc[ok])
    # parameters
    (na, ga), (nb, gb) = F[a], F[b]
    meda, medb = np.median(na[keep]), np.median(nb[keep])
    med_a, med_b, med_avg = np.median(nse_a[ok]), np.median(nse_b[ok]), np.median(nse_avg[ok])
    pa, pb = MODELS[a][1], MODELS[b][1]
    return dict(
        kind=kind, a=a, b=b, run_a=MODELS[a][0][:20], run_b=MODELS[b][0][:20],
        forcing=("" if kind.startswith("seed") else
                 "both AORC" if pa not in NOT_AORC and pb not in NOT_AORC else "UH vs AORC"),
        n_gauges=int(ok.sum()),
        nse_a=med_a, nse_b=med_b, nse_avg=med_avg, avg_minus_better=med_avg - max(med_a, med_b),
        resid_corr=float(np.median(rc[ok])), resid_corr_lo=rlo, resid_corr_hi=rhi,
        gain=float(np.median(g)), gain_lo=glo, gain_hi=ghi, gain_mean=float(np.mean(g)),
        share_year=float(np.median(share[0, ok])), share_season=float(np.median(share[1, ok])),
        share_event=float(np.median(share[2, ok])),
        n_reaches=int(keep.sum()), n0_med_a=meda, n0_med_b=medb, n0_ratio=max(meda, medb) / min(meda, medb),
        rho_n0=float(spearmanr(na[keep], nb[keep]).statistic),
        rho_gamma=float(spearmanr(ga[keep], gb[keep]).statistic),
        rho_n0_all=float(spearmanr(na, nb).statistic),
        n0_ratio_all=max(np.median(na), np.median(nb)) / min(np.median(na), np.median(nb)),
    )


def main():
    # predictions, aligned to the UH model's gauge order and time axis
    ref_ids, tt, P0, O = C.load_preds(MODELS["UH fixed-channel model"][0])
    P, F = {}, {}
    comid0 = None
    for name, (run, _, _) in MODELS.items():
        ids, t, Pr, Or = C.load_preds(run)
        assert (t == tt).all() and set(ids) == set(ref_ids)
        assert np.array_equal(C.align(ids, Or, ref_ids), O, equal_nan=True)
        P[name] = C.align(ids, Pr, ref_ids)
        c, n0, g = load_fields(run)
        comid0 = c if comid0 is None else comid0
        assert np.array_equal(c, comid0)
        F[name] = (n0, g)
    # the two seed-42 UH runs (different binaries, same src for routing) are the same model
    dup_p = float(np.nanmax(np.abs(P["UH fixed-channel model"] - P["UH fixed-channel model, seed 42 (09-27 rerun)"])))
    dup_n = float(np.max(np.abs(F["UH fixed-channel model"][0] - F["UH fixed-channel model, seed 42 (09-27 rerun)"][0])))
    lines = [f"UH 2026-09-12T23-39-03Z vs 2026-09-27T07-29-47Z: max |dQ| {dup_p:.3g} m3/s, max |dn_0| {dup_n:.3g}"]

    fl = np.load(CACHE / "reach_flow_medians.npz")
    assert np.array_equal(fl["COMID"], comid0)
    prods = ["uh", "dhbv2_lumped", "dhbv2_dist", "hydrodl_lstm", "nh_daily_lstm", "nh_hourly_lstm"]
    M = np.stack([fl[k] for k in prods])
    covered = np.isfinite(M).all(0)
    keep = covered & (np.nan_to_num(M, nan=0.0) >= 1e-3).all(0)
    lines.append(f"reaches: {comid0.size:,} in the network, {int(covered.sum()):,} covered by every product, "
                 f"{int(keep.sum()):,} also with median local Q' >= 1e-3 m3/s in every product")

    import zarr
    gz = zarr.open_group(GAGES_ADJ, mode="r")
    pos = {c: i for i, c in enumerate(comid0)}
    basin = [np.array([pos[c] for c in np.asarray(gz[s]["order"][:], dtype=np.int64) if c in pos]) for s in ref_ids]

    def basin_rho(a, b):
        """Spearman over gauges of the basin-median n_0 (median over each gauge's upstream reaches)."""
        bma = np.array([np.median(F[a][0][ix]) for ix in basin])
        bmb = np.array([np.median(F[b][0][ix]) for ix in basin])
        return float(spearmanr(bma, bmb).statistic), bma, bmb

    rows = []
    for a, b, lab in SEED_PAIRS:
        if b is None:
            rows.append(dict(kind="seed (PENDING)", a=a, b=None, label=lab))
            continue
        rows.append(dict(pair_row(a, b, "seed", P, O, tt, F, keep), label=lab))
    for a, b in itertools.combinations(PRODUCT_MODELS, 2):
        rows.append(pair_row(a, b, "product", P, O, tt, F, keep))
    for b in PRODUCT_MODELS[1:]:  # robustness: the UH member swapped for its seed-43 twin
        rows.append(pair_row("UH fixed-channel model, seed 43", b, "product (UH seed 43)", P, O, tt, F, keep))
    # seed crossing: the three products with a seed-43 twin, every pair of products at every seed combination.
    # If a shared seed inflates the product-pair pattern agreement, same-seed rows (42/42, 43/43) beat
    # cross-seed rows (42/43, 43/42) for every product pair.
    twins = {"UH": ("UH fixed-channel model", "UH fixed-channel model, seed 43")}
    for _name in E07:
        if E07[_name]["seed43"]:
            twins[SHORT[_name]] = (_name if not E07[_name]["seed42"] else f"{_name}, seed 42 (E-07 rerun)",
                                   f"{_name}, seed 43")
    for pa, pb in itertools.combinations(twins, 2):
        for sa, sb in [(0, 0), (1, 1), (0, 1), (1, 0)]:
            a, b = twins[pa][sa], twins[pb][sb]
            r = pair_row(a, b, "seed cross", P, O, tt, F, keep)
            r.update(label=f"{pa} s{42 + sa} / {pb} s{42 + sb}", products=f"{pa} / {pb}",
                     same_seed=sa == sb, rho_basin=basin_rho(a, b)[0])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "pairs.csv", index=False, float_format="%.6g")

    done = df[df.n_gauges.notna()]
    hdr = (f"{'pair':44s} {'kind':9s} {'gauges':>6s} {'NSE a/b':>13s} {'avg-best':>8s} {'resid r':>7s} "
           f"{'gain [95% CI]':>24s} {'yr/ssn/evt':>14s} {'reaches':>7s} {'n0 a/b':>13s} {'ratio':>5s} {'rho n0':>6s}")
    lines += ["", hdr]
    for _, r in df.iterrows():
        if not np.isfinite(r.get("n_gauges", np.nan)):
            lines.append(f"{r.label:44s} PENDING   seed-43 run from E-07 (branch paper/e07-seed43) not finished")
            continue
        if r.kind == "seed cross":
            continue
        nm = f"{SHORT[r.a]} / {SHORT[r.b]}"
        lines.append(
            f"{nm:44s} {r.kind[:9]:9s} {int(r.n_gauges):6d} {r.nse_a:.3f}/{r.nse_b:.3f} {r.avg_minus_better:+8.4f} "
            f"{r.resid_corr:7.3f} {r.gain:+.4f} [{r.gain_lo:+.4f},{r.gain_hi:+.4f}] "
            f"{r.share_year:.2f}/{r.share_season:.2f}/{r.share_event:.2f} {int(r.n_reaches):7d} "
            f"{r.n0_med_a:.3f}/{r.n0_med_b:.3f} {r.n0_ratio:5.2f} {r.rho_n0:+6.3f}")
    prod = done[done.kind == "product"]
    seed = done[done.kind == "seed"]
    lines += ["",
              f"product pairs (n = {len(prod)}): resid r {prod.resid_corr.min():.3f}-{prod.resid_corr.max():.3f}, "
              f"gain {prod.gain.min():+.4f} to {prod.gain.max():+.4f}, n0 ratio {prod.n0_ratio.min():.2f}-"
              f"{prod.n0_ratio.max():.2f}, rho n0 {prod.rho_n0.min():+.3f} to {prod.rho_n0.max():+.3f}",
              f"seed pairs done (n = {len(seed)}): resid r {', '.join(f'{x:.3f}' for x in seed.resid_corr)}, "
              f"gain {', '.join(f'{x:+.4f}' for x in seed.gain)}, n0 ratio "
              f"{', '.join(f'{x:.3f}' for x in seed.n0_ratio)}, rho n0 {', '.join(f'{x:+.3f}' for x in seed.rho_n0)}"]
    for lab, d in prod.groupby("forcing"):
        lines.append(f"  {lab} (n = {len(d)}): median resid r {d.resid_corr.median():.3f}, median gain "
                     f"{d.gain.median():+.4f}")
    rs = spearmanr(prod.n0_ratio, prod.gain).statistic
    rr = spearmanr(prod.rho_n0, prod.resid_corr).statistic
    rq = spearmanr(np.minimum(prod.nse_a, prod.nse_b), prod.gain).statistic
    lines.append(f"across the 15 product pairs (exploratory, pairs share members): Spearman(n0 ratio, gain) {rs:+.2f}, "
                 f"Spearman(rho n0, resid r) {rr:+.2f}, Spearman(weaker member's NSE, gain) {rq:+.2f}")
    s43 = done[done.kind == "product (UH seed 43)"].set_index("b")
    s42 = prod[prod.a == "UH fixed-channel model"].set_index("b")
    dd = (s43[["resid_corr", "gain", "rho_n0"]] - s42.loc[s43.index, ["resid_corr", "gain", "rho_n0"]]).abs()
    lines.append(f"UH member swapped to seed 43, 5 product pairs: max |d resid r| {dd.resid_corr.max():.4f}, "
                 f"max |d gain| {dd.gain.max():.4f}, max |d rho n0| {dd.rho_n0.max():.4f}")
    sx = done[done.kind == "seed cross"]
    if len(sx):
        lines += ["", "seed crossing (products with a seed-43 twin): same seed vs different seed, per product pair",
                  f"  {'pair':34s} {'rho n0':>7s} {'basin rho':>9s} {'resid r':>7s} {'gain':>8s} {'ratio':>5s}"]
        for _, r in sx.iterrows():
            lines.append(f"  {r.label:34s} {r.rho_n0:+7.3f} {r.rho_basin:+9.3f} {r.resid_corr:7.3f} {r.gain:+8.4f} "
                         f"{r.n0_ratio:5.2f}")
        g = sx.groupby(["products", "same_seed"])[["rho_n0", "rho_basin", "resid_corr", "gain"]].mean().unstack()
        for pp in g.index:
            lines.append(f"  {pp:34s} mean same-seed minus mean cross-seed: rho n0 "
                         f"{g.loc[pp, ('rho_n0', True)] - g.loc[pp, ('rho_n0', False)]:+.3f}, basin rho "
                         f"{g.loc[pp, ('rho_basin', True)] - g.loc[pp, ('rho_basin', False)]:+.3f}, resid r "
                         f"{g.loc[pp, ('resid_corr', True)] - g.loc[pp, ('resid_corr', False)]:+.4f}, gain "
                         f"{g.loc[pp, ('gain', True)] - g.loc[pp, ('gain', False)]:+.4f}")
    lines += ["", "per-reach n_0 agreement by drainage area (filtered reaches); log n_0 differences vs its spread"]
    at = Dataset(ATTRS)
    la = pd.Series(np.asarray(at["log10_uparea"][:], dtype=np.float64),
                   index=np.asarray(at["COMID"][:], dtype=np.int64)).reindex(comid0).values
    detail = [(a, b) for a, b, _ in SEED_PAIRS if b is not None]
    detail += [("UH fixed-channel model", "dHBV2 distributed model"), ("UH fixed-channel model, seed 43", "dHBV2 distributed model"),
               ("UH fixed-channel model", "dHBV2 lumped model"), ("UH fixed-channel model, seed 43", "dHBV2 lumped model"),
               ("NH daily LSTM model", "NH hourly MTS-LSTM model")]
    for a, b in detail:
        (na, ga), (nb, gb) = F[a], F[b]
        _, bma, bmb = basin_rho(a, b)
        lx, ly = np.log(na[keep]), np.log(nb[keep])
        iqr = np.subtract(*np.percentile(np.r_[lx, ly], [75, 25]))
        by = []
        for lo, hi in [(0, 2), (2, 3), (3, 4), (4, 9)]:
            s = keep & (la >= lo) & (la < hi)
            by.append(f"10^{lo}-10^{hi} km2 ({int(s.sum()):,}) {spearmanr(na[s], nb[s]).statistic:+.2f}")
        lines.append(f"  {SHORT[a]} / {SHORT[b]}: rho n0 by area " + ", ".join(by) +
                     f"; median |d ln n0| {np.median(np.abs(lx - ly)):.3f} = {np.median(np.abs(lx - ly)) / iqr:.2f} "
                     f"of the pooled IQR of ln n0; rho gamma {spearmanr(ga[keep], gb[keep]).statistic:+.2f}; "
                     f"basin-median n0 over {len(ref_ids):,} gauges: rho {spearmanr(bma, bmb).statistic:+.2f}, "
                     f"median |d ln| {np.median(np.abs(np.log(bma / bmb))):.3f}")
    txt = "\n".join(lines)
    (HERE / "pairs.txt").write_text(txt + "\n")
    print(txt)
    plot(done)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    # pairs named on the figure; the rest are identified in pairs.txt and the findings table
    named = {("UH fixed-channel model", "dHBV2 distributed model"): "UH / dHBV2 distributed",
             ("NH daily LSTM model", "NH hourly MTS-LSTM model"): "NH daily / NH hourly LSTM",
             ("UH fixed-channel model", "dHBV2 lumped model"): "UH / dHBV2 lumped",
             ("dHBV2 lumped model", "NH daily LSTM model"): "dHBV2 lumped / NH daily LSTM",
             ("dHBV2 lumped model", "hydroDL LSTM model"): "dHBV2 lumped / hydroDL LSTM"}
    OFFSETS = {"UH seeds 42/43": [(9, -2), (9, -2)],
               "dHBV2 dist seeds 42/43": [(9, -4), (9, -4)],
               "dHBV2 lumped seeds 42/43": [(9, -6), (9, -4)],
               "UH / dHBV2 lumped": [(-20, -14), (6, 4)],
               "dHBV2 lumped / NH daily LSTM": [None, (-125, 9)],
               "dHBV2 lumped / hydroDL LSTM": [(6, -10), (6, -10)],
               "NH daily / NH hourly LSTM": [(6, 4), (6, 6)]}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    style = {"seed": dict(marker="*", s=300, c="black", label="seed pair: same inflow product, seed 42 vs 43", zorder=6),
             "UH vs AORC": dict(marker="o", s=70, c="#d95f02", label="product pair: UH retrospective vs an AORC product"),
             "both AORC": dict(marker="s", s=60, c="#1b9e77", label="product pair: two AORC-forced products")}
    used = set()
    sub = df[df.kind.isin(["seed", "product"])]
    for _, r in sub.iterrows():
        key = "seed" if r.kind == "seed" else r.forcing
        st = dict(style[key])
        lab = st.pop("label")
        lab, _ = (None, used) if lab in used else (lab, used.add(lab))
        xy = [(r.n0_ratio, r.gain), (1 - r.rho_n0, 1 - r.resid_corr)]
        ax[0].errorbar(r.n0_ratio, r.gain, yerr=[[r.gain - r.gain_lo], [r.gain_hi - r.gain]], fmt="none",
                       ecolor=st["c"], alpha=0.6, zorder=4)
        ax[0].scatter(*xy[0], label=lab, edgecolor="white", linewidth=0.6, **st)
        ax[1].scatter(*xy[1], edgecolor="white", linewidth=0.6, **st)
        name = r.label if r.kind == "seed" else named.get((r.a, r.b))
        if name:
            off = OFFSETS.get(name, [(6, 4), (6, 4)])
            for k in range(2):
                if off[k] is not None:  # None: too crowded on that panel, named in the table only
                    ax[k].annotate(name, xy[k], xytext=off[k], textcoords="offset points", fontsize=7.5, color="0.15")
    seed = sub[sub.kind == "seed"]
    for _, r in seed.iterrows():  # the seed floor on both parameter axes
        ax[0].axvline(r.n0_ratio, color="0.45", ls="--", lw=0.7)
        ax[1].axvline(1 - r.rho_n0, color="0.45", ls="--", lw=0.7)
    ax[1].text(1 - seed.rho_n0.max(), 0.75, " dashed: the seed\n pairs' distances", fontsize=7.5, color="0.3", va="top")
    ax[0].text(seed.n0_ratio.max(), 0.25, " dashed: the seed\n pairs' distances", fontsize=7.5, color="0.3", va="top")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].xaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4]))
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].set_xlabel("level distance: ratio of the two models' network-median $n_0$ (larger / smaller)")
    ax[0].set_ylabel("averaging gain: NSE of the two-model average minus\nthe members' mean NSE (median over gauges, 95 % CI)")
    ax[0].set_title("roughness level vs averaging gain")
    ax[1].set_xlabel("pattern distance: 1 - per-reach Spearman $\\rho$ of $n_0$")
    ax[1].set_ylabel("residual disagreement: 1 - median per-gauge\ncorrelation of routed residuals")
    ax[1].set_yscale("log")
    ax[1].set_title("roughness pattern vs residual disagreement")
    ax[0].set_ylim(5e-5, 0.3)
    ax[1].set_ylim(1e-4, 1)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Fixed-channel routing models, test WY1996-2010, 2,365 gauges, 107,225 reaches: "
                 "seed pairs vs inflow-product pairs", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "prediction_diversity.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {HERE / 'prediction_diversity.png'}")


if __name__ == "__main__":
    main()
