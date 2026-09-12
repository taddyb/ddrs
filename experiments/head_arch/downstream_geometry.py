#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "netCDF4"]
# ///
"""Fitted downstream hydraulic geometry of a trained model.

Leopold & Maddock's downstream relations are `w ~ Q^b` (b ~ 0.50) and
`d ~ Q^f` (f ~ 0.40), fitted ACROSS reaches at a common flow frequency. The
model reaches them through

    b = beta + q * f          beta = dlog(p) / dlog(Q)

so with `p_spatial` pinned at 21 the first term is zero and `b` is capped at
`q*f`. That cap was the stated reason to make `p` learnable
(findings §32.5, §33). This measures whether it actually worked.

Mirrors `.claude/skills/ddrs-eval-plots/references/channel_geometry.md`: widths
and depths at a specified baseflow specific discharge, then a log-log fit. The
fitted exponents are invariant to Q_SPEC (it is a constant inside the fit), so
only absolute widths and depths depend on that choice.

Usage:
    experiments/head_arch/downstream_geometry.py <run-id> [<run-id> ...]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
ATTRS = "/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc"
Q_SPEC = 0.005  # m3/s per km2 — stated assumption, see channel_geometry.md


def fit(x_log, y_log):
    m = np.isfinite(x_log) & np.isfinite(y_log)
    return float(np.polyfit(x_log[m], y_log[m], 1)[0])


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    a = Dataset(ATTRS)
    acom = np.asarray(a["COMID"][:], dtype=np.int64)
    aupa = np.asarray(a["log10_uparea"][:], dtype=np.float64)
    pos = {c: i for i, c in enumerate(acom)}

    print(f"Q_SPEC = {Q_SPEC} m3/s/km2   (exponents are invariant to this)")
    print(f"Leopold & Maddock downstream: b ~ 0.50, f ~ 0.40\n")
    print(f"{'run':<46} {'median q':>9} {'median p':>9} {'beta':>7} {'q*f':>7} {'b':>7} {'f':>7}")

    for rid in sys.argv[1:]:
        nc = RUNS / rid / "plot" / "kan_parameters.nc"
        if not nc.exists():
            print(f"{rid:<46} no plot/kan_parameters.nc")
            continue
        ds = Dataset(nc)
        comid = np.asarray(ds["COMID"][:], dtype=np.int64)
        n = np.asarray(ds["n"][:], dtype=np.float64)
        p = np.asarray(ds["p_spatial"][:], dtype=np.float64)
        q = np.asarray(ds["q_spatial"][:], dtype=np.float64)
        s = np.maximum(np.asarray(ds["slope"][:], dtype=np.float64), 1e-3)

        idx = np.array([pos.get(int(c), -1) for c in comid])
        ok = idx >= 0
        upa = np.full(comid.size, np.nan)
        upa[ok] = aupa[idx[ok]]
        area = 10.0**upa
        Q = Q_SPEC * area

        # Stage roughness, if this run used it: a per-reach field when the head
        # learned it (`gamma` variable in the NetCDF), else the config scalar.
        cfg = (RUNS / rid / "config.yaml").read_text()
        g = 0.0
        if "gamma" in ds.variables:
            g = np.asarray(ds["gamma"][:], dtype=np.float64)
        elif "stage_roughness:" in cfg:
            m = re.search(r"^\s+gamma:\s*([0-9.eE+-]+)", cfg.split("stage_roughness:", 1)[1], re.M)
            g = float(m.group(1)) if m else 0.0

        qe = q + 1e-6
        depth = np.maximum(
            (Q * n * (qe + 1.0) / (p * np.sqrt(s) + 1e-8)) ** (3.0 / (5.0 + 3.0 * qe + 3.0 * g)),
            0.01,
        )
        width = p * depth**qe

        lq = np.log10(np.where(Q > 0, Q, np.nan))
        b = fit(lq, np.log10(width))
        f = fit(lq, np.log10(depth))
        beta = fit(lq, np.log10(p)) if np.unique(p).size > 1 else 0.0
        print(
            f"{rid:<46} {np.median(q):>9.4f} {np.median(p):>9.3f} "
            f"{beta:>+7.3f} {np.median(qe) * f:>7.3f} {b:>7.3f} {f:>7.3f}"
        )

    print("\n  beta is how p scales with discharge. It is the ONLY term that can lift")
    print("  b above q*f, which is why p was made learnable. A NEGATIVE beta means p")
    print("  shrinks as rivers grow, which pushes b DOWN and refutes that rationale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
