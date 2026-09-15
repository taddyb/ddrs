#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "netCDF4"]
# ///
"""Does "the trunk is rank 1" contradict the parameter fields varying in space?

No, and this script is the check. "Rank 1" does not mean one VALUE everywhere.
It means one independent PATTERN: the fields vary richly across reaches, but
every parameter is a fixed function of the same single spatial pattern, so
knowing one tells you the other with almost no residual freedom.

Reported:

  variation     how much each field actually varies across CONUS, so the rank
                claim cannot be mistaken for "the field is flat".
  vs attributes each field's rank correlation with the input attributes,
                including log10_uparea. This is the spatial structure you SEE
                on a map.
  independence  the part that matters. After removing the best fit of q on n
                (on the pre-sigmoid scale where the relation is affine), how
                much of q is left over? That residual is the only place a
                second independent pattern could live.

Usage:
    experiments/head_arch/spatial_structure.py <run-id>
"""

import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.stats import spearmanr

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
STATS = (
    "/home/tbindas/projects/ddr/data/statistics/"
    "merit_attribute_statistics_merit_global_attributes_v2.nc.json"
)
RANGES = {"n": (0.015, 0.25), "q_spatial": (0.0, 1.0), "p_spatial": (1.0, 200.0)}


def logit_of(v, lo, hi, log_space=False):
    v = np.asarray(v, dtype=np.float64)
    if log_space:
        lo, hi, v = np.log(lo), np.log(hi), np.log(np.clip(v, 1e-30, None))
    u = np.clip((v - lo) / (hi - lo), 1e-12, 1 - 1e-12)
    return np.log(u / (1 - u))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run = sys.argv[1]
    nc = RUNS / run / "plot" / "kan_parameters.nc"
    ds = Dataset(nc)
    comids = np.asarray(ds["COMID"][:], dtype=np.int64)
    fields = {k: np.asarray(ds[k][:], dtype=np.float64) for k in RANGES if k in ds.variables}
    cfg = (RUNS / run / "config.yaml").read_text()
    names = [
        ln.strip()[2:]
        for ln in cfg.split("input_var_names:")[1].split("learnable_parameters:")[0].splitlines()
        if ln.strip().startswith("- ")
    ]

    a = Dataset(ATTRS)
    stats = json.loads(Path(STATS).read_text())
    acom = np.asarray(a["COMID"][:], dtype=np.int64)
    pos = {c: i for i, c in enumerate(acom)}
    idx = np.array([pos.get(c, -1) for c in comids])
    ok = idx >= 0
    attrs = {}
    for n in names:
        v = np.asarray(a[n][:], dtype=np.float64)
        col = np.full(comids.size, np.nan)
        col[ok] = v[idx[ok]]
        col[~np.isfinite(col)] = stats[n]["mean"]
        attrs[n] = col

    print(f"trained fields — {run}")
    print(f"{comids.size:,} reaches, {ok.sum():,} matched to attributes\n")

    print("VARIATION — the fields are not flat")
    print(f"  {'field':<12} {'min':>10} {'p10':>10} {'median':>10} {'p90':>10} {'max':>10} {'distinct':>10}")
    for k, v in fields.items():
        q = np.percentile(v, [10, 50, 90])
        print(
            f"  {k:<12} {v.min():>10.4f} {q[0]:>10.4f} {q[1]:>10.4f} {q[2]:>10.4f} "
            f"{v.max():>10.4f} {np.unique(v).size:>10,d}"
        )
    print()

    print("SPATIAL STRUCTURE — Spearman rho of each field against each input attribute")
    keys = list(fields)
    print(f"  {'attribute':<22} " + " ".join(f"{k:>12}" for k in keys))
    order = sorted(
        names,
        key=lambda n: -abs(spearmanr(attrs[n], fields[keys[0]]).statistic),
    )
    for n in order:
        cells = " ".join(
            f"{spearmanr(attrs[n], fields[k]).statistic:>+12.3f}" for k in keys
        )
        print(f"  {n:<22} {cells}")
    print()

    if "n" in fields and "q_spatial" in fields:
        ln = logit_of(fields["n"], *RANGES["n"])
        lq = logit_of(fields["q_spatial"], *RANGES["q_spatial"])
        r = np.corrcoef(ln, lq)[0, 1]
        slope = np.cov(ln, lq, bias=True)[0, 1] / ln.var()
        resid = lq - (lq.mean() + slope * (ln - ln.mean()))
        print("INDEPENDENCE — how much of the width exponent is NOT the roughness field?")
        print(f"  logit(q) = {slope:+.3f} * logit(n) {lq.mean() - slope * ln.mean():+.3f}")
        print(f"  variance of logit(q) explained by logit(n):  {100 * r * r:.2f} %")
        print(f"  left over:                                   {100 * (1 - r * r):.2f} %")
        print(f"  sd of logit(q)          = {lq.std():.4f}")
        print(f"  sd of the residual      = {resid.std():.4f}")
        print()
        print("  Does the leftover carry its own spatial signal, or is it noise?")
        print(f"    {'attribute':<22} {'rho with residual':>18}")
        for n in sorted(names, key=lambda n: -abs(spearmanr(attrs[n], resid).statistic))[:4]:
            print(f"    {n:<22} {spearmanr(attrs[n], resid).statistic:>+18.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
