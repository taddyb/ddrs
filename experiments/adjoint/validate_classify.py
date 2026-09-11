#!/usr/bin/env python
"""Classify every failed full-map finite-difference check into one of five classes.

Usage:
    ~/projects/ddr/.venv/bin/python experiments/adjoint/validate_classify.py <run_dir> [--arm uh-retro]

Reads <run_dir>/<arm>/validation/*.csv (written by `ddrs experiment` with an
`adjoint.validation:` block) and writes <run_dir>/figures/CHECK1_CLASSES.md.

Classes, evaluated in order at the smallest-δ sweep point (fwd/bwd = one-sided
slopes, grad = adjoint kernel entry, noise = f32 noise floor in m3/s):
  limit_pass    sweep rel_err ≤ 0.05: FD converges to the gradient as δ → 0
  clamp_floor   grad == 0 and bwd == 0 and |fwd| > 0: inflow at the model floor,
                flat on one side
  quantization  |grad|·2δ < 5·noise: no linear δ can resolve the response
  kink          fwd ≠ bwd and grad lies between them (± 5 %): non-differentiable
                base state, adjoint returns a sub-gradient
  unexplained   everything else (report individually)
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    """Minimal Markdown table (no tabulate dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def classify_row(r) -> tuple[str, dict]:
    sd = [float(x) for x in str(r.sweep_deltas).split(";") if x and x != "nan"]
    if not sd:
        return "no_sweep", {}
    fw = [float(x) for x in str(r.sweep_dq_forward).split(";")][0]
    bw = [float(x) for x in str(r.sweep_dq_backward).split(";")][0]
    cen = [float(x) for x in str(r.sweep_dq_actual).split(";")][0]
    e = [float(x) for x in str(r.sweep_rel_err).split(";")][0]
    g = float(r.dq_predicted)
    noise = float(r.noise_floor_m3s)
    d = sd[0]
    info = dict(fwd=fw, bwd=bw, cen=cen, dmin=d, rel_min=e)
    if e <= 0.05:
        return "limit_pass", info
    if g == 0.0 and (bw == 0.0 or np.isnan(bw)) and abs(fw) > 0:
        return "clamp_floor", info
    if abs(g) * 2 * d < 5 * noise:
        return "quantization", info
    if np.isfinite(fw) and np.isfinite(bw):
        scale = max(abs(g), abs(fw), abs(bw), 1e-12)
        lo, hi = min(fw, bw) - 0.05 * abs(g), max(fw, bw) + 0.05 * abs(g)
        if abs(fw - bw) > 0.1 * scale and lo <= g <= hi:
            return "kink", info
    return "unexplained", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(str(a.run_dir / (a.arm or "*") / "validation" / "*.csv")))
    if not files:
        raise SystemExit("no validation CSVs found")
    frames = [pd.read_csv(f, dtype={"staid": str}) for f in files]
    tot = pd.concat(frames, ignore_index=True)
    tot["staid"] = tot["staid"].astype(str).str.zfill(8)
    classes = ["limit_pass", "clamp_floor", "quantization", "kink", "unexplained", "no_sweep"]
    rows = []
    for _, r in tot[tot.passed == 0].iterrows():
        cls, info = classify_row(r)
        rows.append(dict(arm=r.arm, staid=r.staid, kind=r.anchor_kind, q_gauge=r.q_gauge_t0, reach=r.reach_row,
                         dist_km=r.dist_to_gauge_m / 1000.0, lag_h=r.lag_hours, grad=r.dq_predicted, cls=cls, **info))
    R = pd.DataFrame(rows)
    summ = tot.groupby(["arm", "staid", "anchor_kind"]).agg(
        n_checks=("passed", "size"), n_passed=("passed", "sum"), n_unresolvable=("unresolvable", "sum"),
        q_gauge_t0=("q_gauge_t0", "first"), noise=("noise_floor_m3s", "first"))
    if not R.empty:
        cnt = R.groupby(["arm", "staid", "kind"]).cls.value_counts().unstack(fill_value=0)
        cnt.index.names = ["arm", "staid", "anchor_kind"]
        for c in classes:
            if c not in cnt.columns:
                cnt[c] = 0
        summ = summ.join(cnt[classes], how="left").fillna(0)
        for c in classes:
            summ[c] = summ[c].astype(int)
    out = a.run_dir / "figures" / "CHECK1_CLASSES.md"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as w:
        w.write("# Check 1: full-map finite-difference failures by class\n\n")
        w.write(f"Total checks {len(tot)}, passed {int(tot.passed.sum())}, unresolvable (below f32 noise floor, counted as pass) "
                f"{int(tot.unresolvable.sum())}, failed {int((tot.passed == 0).sum())}.\n\n")
        if not R.empty:
            w.write("Failures by class: " + ", ".join(f"{k} {v}" for k, v in R.cls.value_counts().items()) + "\n\n")
        w.write(md_table(summ.reset_index()) + "\n\n")
        w.write("Classes (evaluated at the smallest sweep δ): limit_pass = FD converges to the gradient; clamp_floor = inflow at the "
                "model floor (grad 0, flat on the minus side); quantization = predicted ΔQ within 5 f32 noise floors, unresolvable at any "
                "linear δ; kink = one-sided slopes differ and the gradient lies between them (non-differentiable base state); "
                "unexplained = none of the above.\n\n")
        if not R.empty:
            ux = R[R.cls.isin(["unexplained", "kink"])]
            w.write("## Kink and unexplained rows\n\n")
            w.write(md_table(ux) + "\n")
    print(out)
    print(summ.reset_index().to_string())
    if not R.empty:
        print(R.cls.value_counts().to_string())


if __name__ == "__main__":
    main()
