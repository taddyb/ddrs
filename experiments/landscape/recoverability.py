#!/usr/bin/env python
"""How much of the per-gauge displacement gain is actually recoverable.

Write-up: research/findings/2026-09-08-landscape-hypothesis-tests-findings.md, §20.

Both sections below use the per-gauge quadratic surrogate in the
log-multiplier `c` on Manning's n,

    NSE(c) = nse_star - k (c - a*)^2 ,   k = (nse_star - nse0) / a*^2

which is exact at the two points the census measures -- the trained point
`c = 0` and the Newton optimum `c = a*` -- and interpolates between them.

Section 1 (always run): sweeps a single global log-multiplier `c` applied to
every reach's trained Manning's n and asks how much of the per-gauge gain one
global number can recover. Reads `<census>/figures/covariates.csv`.

Section 2 (only with --compare-census): compares the per-gauge Newton
optimum `a*` estimated on two different windows -- correlation, sign
agreement, and how much of the in-sample gain transfers when the optimum
from one window is applied to the other.

Usage:
    uv run --with numpy --with pandas python experiments/landscape/recoverability.py \\
        --census <census-merged-dir> [--compare-census <compare-merged-dir>] \\
        [--out <dir>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CENSUS = Path(".ddrs/experiments/landscape-p21-all-5yr/merged")


# --------------------------------------------------------------- section 1
def global_shift(census_dir: Path, min_abs_astar: float, out: list[str]) -> None:
    f = census_dir / "figures" / "covariates.csv"
    d = pd.read_csv(f, dtype={"staid": str, "HUC02": str})
    out.append(f"rows {len(d)}")
    m = (d.nse0 > 0.3) & d.alpha_n_star.notna() & d.nse_star.notna() & (d.box_edge == 0)
    d = d[m].copy()
    out.append(f"well-fit, not on box edge: {len(d)}")

    d = d[d.alpha_n_star.abs() >= min_abs_astar].copy()
    out.append(f"min |a*| filter: {min_abs_astar:g}  -> {len(d)} gauges")

    # per-gauge quadratic in log-multiplier c around the gauge optimum:
    #   NSE(c) = nse_star - k (c - a*)^2 ,  k = (nse_star - nse0) / a*^2   (>=0)
    a = d.alpha_n_star.values
    k = np.where(np.abs(a) > 1e-6, (d.nse_star.values - d.nse0.values) / np.maximum(a**2, 1e-12), 0.0)
    k = np.clip(k, 0, None)
    nstar = d.nse_star.values
    n0 = d.nse0.values

    def med_nse(c):
        return np.median(nstar - k * (c - a) ** 2)

    grid = np.linspace(-1.5, 1.5, 601)
    vals = np.array([med_nse(c) for c in grid])
    best = grid[np.argmax(vals)]
    out.append("")
    out.append(f"median NSE at trained point (c=0)      : {np.median(n0):.4f}")
    out.append(f"median NSE at each gauge's own optimum : {np.median(nstar):.4f}   (gain {np.median(nstar)-np.median(n0):+.4f})")
    out.append(f"best SINGLE global log-multiplier c*   : {best:+.3f}  (n x {np.exp(best):.2f})")
    out.append(f"median NSE at that global c*           : {med_nse(best):.4f}   (gain {med_nse(best)-np.median(n0):+.4f})")
    tot = np.median(nstar) - np.median(n0)
    if tot > 0:
        out.append(f"share of the per-gauge gain captured by one global number: {(med_nse(best)-np.median(n0))/tot*100:.0f} %")
    out.append("")
    out.append("median NSE vs global multiplier:")
    for c in [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        out.append(f"   c={c:+.2f}  n x{np.exp(c):5.2f}   median NSE {med_nse(c):.4f}   mean NSE {np.mean(nstar-k*(c-a)**2):.4f}")
    out.append("")
    out.append("surrogate-free: who ends up nearer their own optimum")
    for c in [0.25, 0.50, 0.75, 1.00]:
        closer = (np.abs(a - c) < np.abs(a)).mean() * 100
        further = 100 - closer
        out.append(f"   c={c:.2f}  moved closer {closer:.0f} %   moved further {further:.0f} %")
    out.append("")
    out.append("k diagnostics (see section 20.4)")
    out.append(f"   median k : {np.median(k):.4f}")
    out.append(f"   p90 k    : {np.percentile(k, 90):.4f}")
    out.append(f"   p99 k    : {np.percentile(k, 99):.4f}")
    out.append(f"   p99.9 k  : {np.percentile(k, 99.9):.4f}")
    out.append(f"   max k    : {np.max(k):.4f}")
    order = np.argsort(k)[::-1]
    top1 = max(1, int(np.ceil(len(k) * 0.01)))
    out.append(f"   share of total k held by the top 1 % of gauges: {k[order[:top1]].sum() / k.sum() * 100:.1f} %")
    out.append("")
    out.append("distribution of the per-gauge optimum log-multiplier a*:")
    for q in [5, 25, 50, 75, 95]:
        out.append(f"   p{q:>2}: {np.percentile(a,q):+.3f}  (n x{np.exp(np.percentile(a,q)):.2f})")
    out.append(f"   share a* > 0 (gauge wants MORE roughness): {(a>0).mean()*100:.1f} %")
    out.append("")
    out.append("by basin size (n_reach):")
    for lab, sel in [("<=50", d.n_reach <= 50), ("51-200", (d.n_reach > 50) & (d.n_reach <= 200)), (">200", d.n_reach > 200)]:
        s = sel.values
        if s.sum() < 5:
            continue
        ai, ki, nsi, n0i = a[s], k[s], nstar[s], n0[s]
        g = np.linspace(-1.5, 1.5, 601)
        v = np.array([np.median(nsi - ki * (c - ai) ** 2) for c in g])
        b = g[np.argmax(v)]
        out.append(
            f"   {lab:>7}  n={s.sum():4d}  median a* {np.median(ai):+.3f}  best global c* {b:+.3f}"
            f"  median NSE {np.median(n0i):.3f} -> {v.max():.3f} (global) -> {np.median(nsi):.3f} (per-gauge)"
        )


# --------------------------------------------------------------- section 2
def astar_stability(census_dir: Path, compare_census_dir: Path, min_abs_astar: float, out: list[str]) -> None:
    c1 = pd.read_csv(compare_census_dir / "figures" / "covariates.csv", dtype={"staid": str})
    c5 = pd.read_csv(census_dir / "figures" / "covariates.csv", dtype={"staid": str})
    k = ["staid", "alpha_n_star", "nse0", "nse_star", "box_edge", "n_reach", "gain"]
    d = c1[k].merge(c5[k], on="staid", suffixes=("_1y", "_5y"))
    d = d[(d.nse0_5y > 0.3) & (d.nse0_1y > 0.3) & (d.box_edge_1y == 0) & (d.box_edge_5y == 0)]
    d = d.dropna(subset=["alpha_n_star_1y", "alpha_n_star_5y"])
    d = d[d.alpha_n_star_5y.abs() >= min_abs_astar].copy()
    out.append(f"min |a*| filter (five-year, primary census): {min_abs_astar:g}  -> {len(d)} gauges")
    a1 = d.alpha_n_star_1y.values
    a5 = d.alpha_n_star_5y.values
    out.append(f"gauges compared: {len(d)}   (WY2000 alone vs WY1996-2000, the 1y window is nested in the 5y)")
    out.append(f"Pearson  r(a*_1y, a*_5y) = {np.corrcoef(a1,a5)[0,1]:.3f}")
    r1 = pd.Series(a1).rank()
    r5 = pd.Series(a5).rank()
    out.append(f"Spearman r               = {np.corrcoef(r1,r5)[0,1]:.3f}")
    out.append(f"sign agreement           = {((a1>0)==(a5>0)).mean()*100:.1f} %")
    out.append(f"median a*: 1y {np.median(a1):+.3f}   5y {np.median(a5):+.3f}")
    out.append(
        f"median |a*_1y - a*_5y|   = {np.median(np.abs(a1-a5)):.3f} log units"
        f"   (median |a*_5y| = {np.median(np.abs(a5)):.3f})"
    )
    # transfer test: apply the WY2000 optimum to the 5-year quadratic
    k5 = np.clip(np.where(np.abs(a5) > 1e-6, (d.nse_star_5y.values - d.nse0_5y.values) / np.maximum(a5**2, 1e-12), 0.0), 0, None)
    nse_at_a1 = d.nse_star_5y.values - k5 * (a1 - a5) ** 2
    out.append("")
    out.append(f"median 5y NSE at trained point        : {np.median(d.nse0_5y):.4f}")
    out.append(f"median 5y NSE at the 5y optimum       : {np.median(d.nse_star_5y):.4f}  (in-sample gain {np.median(d.nse_star_5y)-np.median(d.nse0_5y):+.4f})")
    out.append(f"median 5y NSE at the WY2000 optimum   : {np.median(nse_at_a1):.4f}  (transfer gain {np.median(nse_at_a1)-np.median(d.nse0_5y):+.4f})")
    out.append(f"share of gauges where the WY2000 move IMPROVES the 5y score: {(nse_at_a1>d.nse0_5y.values).mean()*100:.1f} %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, default=DEFAULT_CENSUS, help="primary census merged-dir (default: the 5-year WY1996-2000 census)")
    ap.add_argument("--compare-census", type=Path, default=None, help="second census merged-dir; when given, also runs the cross-window stability and transfer analysis")
    ap.add_argument("--out", type=Path, default=None, help="directory to write the report to as recoverability.txt")
    ap.add_argument("--min-abs-astar", type=float, default=0.0, help="drop gauges with |alpha_n_star| below this from both sections (five-year a* for the transfer section); default 0.0 drops nothing")
    args = ap.parse_args()

    out: list[str] = []
    out.append("== global shift ==")
    global_shift(args.census, args.min_abs_astar, out)

    if args.compare_census is not None:
        out.append("")
        out.append("== a* stability across windows ==")
        astar_stability(args.census, args.compare_census, args.min_abs_astar, out)

    report = "\n".join(out)
    print(report)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "recoverability.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
