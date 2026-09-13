#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "netCDF4", "xarray", "zarr>=3", "scipy", "matplotlib"]
# ///
"""Roughness as a bias absorber: the learned (n_0, gamma) fields across inflow arms.

Same head, network, gauges, channel and recipe; only the unit-catchment inflow
product differs. Per arm: n_0 and gamma medians by river size, fraction of
reaches at a box bound, rho(n_0, gamma), and skill against the product's own
summed-Q' baseline. Across arms: per-reach rank agreement of n_0 and gamma,
and the range-normalised spread (paper Eq. normspread). Per gauge: the
product's own inflow error (volume ratio and peak bias of summed Q' vs obs)
against the roughness the routing learned on it, pooled over arms, which is
the bias-absorber regression of the methods section.

    experiments/stage_roughness/cross_arm_fields.py <label>=<run-id> [<label>=<run-id> ...] --out <dir>
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from netCDF4 import Dataset
from scipy.stats import spearmanr

sys.path.insert(0, "/home/tbindas/projects/ddrs/.claude/skills/ddrs-eval-plots/scripts")
from load_ddrs_predictions import load_baseline_f32, load_predictions_zarr  # noqa: E402

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
GAGES = "/home/tbindas/projects/ddr/references/gage_info/gages_3000.csv"


def gauge_inflow_error(run):
    """Per gauge: summed-Q' volume ratio, peak bias (FHV) and NSE vs obs, and the routed NSE."""
    dsb = load_baseline_f32(run / "baseline"); dsp = load_predictions_zarr(run / "eval" / "predictions.zarr")
    g = np.intersect1d(dsb.gage_ids.values, dsp.gage_ids.values)
    rows = []
    for s in g:
        b = dsb.sel(gage_ids=s); p = dsp.sel(gage_ids=s)
        o = b.observations.values; q = b.predictions.values; m = np.isfinite(o) & (o > 0) & np.isfinite(q)
        if m.sum() < 365: continue
        o, q = o[m], q[m]; k = max(1, int(0.02 * o.size)); top = np.argsort(o)[-k:]
        po = p.observations.values; pp = p.predictions.values; mm = np.isfinite(po) & (po > 0) & np.isfinite(pp)
        rnse = 1 - np.sum((pp[mm] - po[mm]) ** 2) / np.sum((po[mm] - po[mm].mean()) ** 2) if mm.sum() > 365 else np.nan
        rows.append((s, q.sum() / o.sum(), 100 * (q[top].sum() - o[top].sum()) / o[top].sum(), 1 - np.sum((q - o) ** 2) / np.sum((o - o.mean()) ** 2), rnse))
    return pd.DataFrame(rows, columns=["gauge", "vol_ratio", "fhv", "nse_sum", "nse_routed"]).set_index("gauge")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("arms", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = dict(x.split("=", 1) for x in a.arms)
    at = Dataset(ATTRS); ac = np.asarray(at["COMID"][:], dtype=np.int64); ua = np.asarray(at["log10_uparea"][:], dtype=np.float64)
    la_by = pd.Series(ua, index=ac)
    fields = {}
    print(f"{'arm':>16} {'n0 med':>7} {'n0 ceil%':>8} {'gamma med':>9} {'gamma@0%':>8} {'rho(n0,g)':>9} | n0 by size 10^1-2 .. 10^4+ | gamma by size")
    for lab, rid in arms.items():
        ds = Dataset(RUNS / rid / "plot" / "kan_parameters.nc")
        c = np.asarray(ds["COMID"][:], dtype=np.int64); n = np.asarray(ds["n"][:], dtype=np.float64); g = np.asarray(ds["gamma"][:], dtype=np.float64)
        la = la_by.reindex(c).values; ok = np.isfinite(la)
        df = pd.DataFrame({"n0": n, "gamma": g, "la": la}, index=c)[ok]
        fields[lab] = df
        bins = [(1, 2), (2, 3), (3, 4), (4, 7)]
        nsz = " ".join(f"{df.n0[(df.la >= lo) & (df.la < hi)].median():.3f}" for lo, hi in bins)
        gsz = " ".join(f"{df.gamma[(df.la >= lo) & (df.la < hi)].median():.3f}" for lo, hi in bins)
        print(f"{lab:>16} {df.n0.median():7.4f} {100*np.mean(df.n0 >= 0.2495):8.1f} {df.gamma.median():9.3f} {100*np.mean(df.gamma < 0.001):8.1f} {spearmanr(df.n0, df.gamma).statistic:+9.3f} | {nsz} | {gsz}")
    labs = list(fields)
    print("\nper-reach rank agreement across arms (Spearman rho), n_0 / gamma:")
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            A, B = fields[labs[i]], fields[labs[j]]; idx = A.index.intersection(B.index)
            print(f"  {labs[i]:>16} vs {labs[j]:<16}: n_0 {spearmanr(A.n0[idx], B.n0[idx]).statistic:+.3f}   gamma {spearmanr(A.gamma[idx], B.gamma[idx]).statistic:+.3f}")
    idx = fields[labs[0]].index
    for l in labs[1:]: idx = idx.intersection(fields[l].index)
    N = np.stack([fields[l].n0[idx].values for l in labs]); G = np.stack([fields[l].gamma[idx].values for l in labs])
    print(f"\nrange-normalised cross-arm spread per reach (median): n_0 {np.median((N.max(0) - N.min(0)) / (0.25 - 0.015)):.3f}   gamma {np.median((G.max(0) - G.min(0)) / 0.5):.3f}   ({len(labs)} arms, {idx.size:,} reaches)")

    # bias absorber: per-gauge inflow error vs learned roughness, pooled over arms
    meta = pd.read_csv(GAGES, dtype={"STAID": str}).set_index("STAID")
    import zarr
    gz = zarr.open_group("/home/tbindas/projects/ddr/data/merit_gages_conus_adjacency.zarr", mode="r")
    rows = []
    for lab, rid in arms.items():
        err = gauge_inflow_error(RUNS / rid)
        f = fields[lab]
        for s, r in err.iterrows():
            try:
                comids = np.asarray(gz[str(s)]["order"][:], dtype=np.int64)
            except Exception:
                continue
            sub = f.reindex(comids).dropna()
            if len(sub) < 3: continue
            rows.append((lab, s, float(r.vol_ratio), float(r.fhv), float(r.nse_sum), float(r.nse_routed), float(sub.n0.median()), float(sub.gamma.median()), len(sub)))
    df = pd.DataFrame(rows, columns=["arm", "gauge", "vol_ratio", "fhv", "nse_sum", "nse_routed", "n0_basin", "gamma_basin", "n_reach"])
    df.to_csv(out / "bias_absorber_per_gauge.csv", index=False)
    print("\nbias absorber, per gauge and arm: Spearman rho of basin-median n_0 with the product's inflow error")
    for lab in labs:
        d = df[df.arm == lab]
        print(f"  {lab:>16}: rho(n_0, volume ratio) {spearmanr(d.n0_basin, d.vol_ratio).statistic:+.3f}   rho(n_0, FHV) {spearmanr(d.n0_basin, d.fhv).statistic:+.3f}   rho(n_0, NSE of summed Q') {spearmanr(d.n0_basin, d.nse_sum).statistic:+.3f}   (n={len(d)})")
    print(f"  {'pooled':>16}: rho(n_0, FHV) {spearmanr(df.n0_basin, df.fhv).statistic:+.3f}   rho(n_0, vol ratio) {spearmanr(df.n0_basin, df.vol_ratio).statistic:+.3f}   arm medians: " + ", ".join(f"{l} FHV {df[df.arm==l].fhv.median():+.1f}% n0 {df[df.arm==l].n0_basin.median():.3f}" for l in labs))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for lab in labs:
        d = df[df.arm == lab]; ax[0].scatter(d.fhv, d.n0_basin, s=6, alpha=0.4, label=lab)
    ax[0].set_xlabel("peak bias of the product's summed Q' vs observed (FHV, %)"); ax[0].set_ylabel("basin-median learned n_0"); ax[0].legend(fontsize=8); ax[0].set_xlim(-100, 200); ax[0].grid(alpha=.3)
    ax[0].set_title("rougher channel where the inflow over-predicts peaks?")
    for lab in labs:
        f = fields[lab]; b = np.floor(f.la * 4) / 4; m = f.groupby(b)["n0"].median(); ax[1].plot(m.index + 0.125, m.values, "-o", ms=3, label=lab)
    ax[1].set_xlabel("log10 drainage area (km²)"); ax[1].set_ylabel("median n_0"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("n_0 by river size, per inflow product")
    for lab in labs:
        f = fields[lab]; b = np.floor(f.la * 4) / 4; m = f.groupby(b)["gamma"].median(); ax[2].plot(m.index + 0.125, m.values, "-o", ms=3, label=lab)
    ax[2].set_xlabel("log10 drainage area (km²)"); ax[2].set_ylabel("median gamma"); ax[2].grid(alpha=.3); ax[2].set_title("gamma by river size, per inflow product")
    fig.tight_layout(); fig.savefig(out / "cross_arm_fields.png", dpi=150, facecolor="white"); print(f"wrote {out / 'cross_arm_fields.png'}")


if __name__ == "__main__":
    main()
