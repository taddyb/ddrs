#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "netCDF4"]
# ///
"""Compare trained head-topology arms on output collapse and on skill.

The screen (`analyze.py`, findings §32) established that all seven topologies
CAN emit independent parameter fields, so anything that collapses after training
is the routing gradient's doing, not the architecture's. These are the arms that
test what the gradient actually does.

Reported per arm, over all 346,321 CONUS reaches:

  collapse   Spearman rho and affine R^2 between each parameter pair, on the
             pre-sigmoid scale where the §31 relation was measured
             (`logit(q) = 3.979 * logit(n) + 2.752`, R^2 = 0.987). The registered
             prediction is that the shared control reproduces this and the wide
             and split arms do not.
  bounds     how much of each declared range the field actually occupies, and
             what fraction sits within 1% of either end. §29 found q pinned at a
             box edge at 62.5% of gauges; §31 explained it as n's field
             amplified ~4x through a shared read-out.
  geometry   the DOWNSTREAM width exponent b, where w ~ Q^b, against Leopold &
             Maddock's b ~ 0.50. This is the only physical plausibility check
             available, since the attributes carry no width or depth to validate
             against.

             NOTE q is NOT b. q is the width-DEPTH exponent in w = p*d^q;
             b is the width-DISCHARGE exponent. At constant p they are related
             by b = 3q/(5+3q), so the declared range q in [0,1] caps b at 0.375
             and cannot reach 0.50 at all (needs q = 5/3). An earlier version of
             this script scored q itself against a "0.1-0.6 band", which is a
             conflation of the two: in b terms that band is [0.057, 0.257]. See
             .claude/skills/ddrs-eval-plots/references/channel_geometry.md and
             docs/superpowers/specs/2026-09-12-leopold-maddock-q-prior-design.md.
  skill      median NSE and KGE from each run's own manifest, plus its own
             baseline. Read arms against EACH OTHER: they all learn three
             parameters where the 0.7376/0.7600 reference learned two, so the
             shared-linear arm is the matched control, not that number.

Usage:
    experiments/head_arch/compare_arms.py <run-id> [<run-id> ...]
    experiments/head_arch/compare_arms.py --auto   # newest run per arm config
"""

import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.stats import spearmanr

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")
PAIRS = [("n", "q_spatial"), ("n", "p_spatial"), ("p_spatial", "q_spatial")]
# Declared ranges from params.parameter_ranges in the arm configs.
RANGES = {"n": (0.015, 0.25), "q_spatial": (0.0, 1.0), "p_spatial": (1.0, 200.0)}
# Leopold & Maddock downstream width exponent w ~ Q^b. Plausible band in b,
# NOT in q — see the docstring.
LM_B_BAND = (0.4, 0.6)


def b_from_q(q):
    """Downstream width exponent implied by the width-depth exponent, at constant p.

    w = p*d^q and d ~ Q^(3/(5+3q)) give w ~ Q^(3q/(5+3q)). This is the p-CONSTANT
    part of b only. In general b = beta + q*f with beta = dlog(p)/dlog(Q), and
    beta can have either sign, so q*f is NOT a bound in either direction: the
    p-learnable run 2026-09-11T23-24-04Z has q*f = 0.150 and an actual b of
    0.004, because beta = -0.144. Use experiments/head_arch/downstream_geometry.py
    for the real fitted b whenever p varies."""
    q = np.asarray(q, dtype=np.float64)
    return 3.0 * q / (5.0 + 3.0 * q)


def logit_of(v: np.ndarray, lo: float, hi: float, log_space: bool) -> np.ndarray:
    """Recover the head's pre-sigmoid column from a denormalised field.

    The head emits sigmoid(z) in (0,1) and `denormalize` maps that onto
    [lo, hi], in log space for p_spatial. Inverting both recovers z up to the
    affine scaling that a correlation is invariant to."""
    v = np.asarray(v, dtype=np.float64)
    if log_space:
        lo, hi = np.log(lo), np.log(hi)
        v = np.log(np.clip(v, 1e-30, None))
    u = (v - lo) / (hi - lo)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return np.log(u / (1 - u))


def affine_r2(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """R^2, slope and intercept of an OLS fit of b on a."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    slope = float(np.cov(a, b, bias=True)[0, 1] / a.var())
    return r * r, slope, float(b.mean() - slope * a.mean())


def load_arm(run_id: str) -> dict | None:
    d = RUNS / run_id
    nc = d / "plot" / "kan_parameters.nc"
    if not nc.exists():
        print(f"  ! {run_id}: no plot/kan_parameters.nc (was --plot passed?)")
        return None
    cfg = (d / "config.yaml").read_text()
    ds = Dataset(nc)
    fields = {k: np.asarray(ds[k][:], dtype=np.float64) for k in RANGES if k in ds.variables}
    man = json.loads((d / "manifest.json").read_text())
    return {
        "run": run_id,
        "label": next(
            (
                line.split("#", 1)[1].strip()
                for line in cfg.splitlines()[:3]
                if line.startswith("# head_")
            ),
            run_id,
        ),
        "fields": fields,
        "manifest": man,
        "config": cfg,
        "log_space": [k for k in RANGES if f"    - {k}" in cfg.split("log_space_parameters:")[-1][:200]],
    }


def metric(man: dict, *names):
    """Run manifests have moved metric keys around; try several."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)) and any(n in k.lower() for n in names):
                    yield k, v
                yield from walk(v)
    return dict(walk(man))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args == ["--auto"]:
        # Newest train-and-test run per arm, identified by its config banner.
        found = {}
        for d in sorted(RUNS.glob("*-train-and-test")):
            c = d / "config.yaml"
            if not c.exists() or d.name.startswith("ABANDONED"):
                continue
            head = c.read_text().splitlines()[0]
            if head.startswith("# head_"):
                found[head[2:].strip()] = d.name
        args = list(found.values())
        print(f"--auto found {len(args)} arm run(s): {', '.join(args)}\n")

    arms = [a for a in (load_arm(r) for r in args) if a]
    if not arms:
        print("no arms with parameter dumps found")
        return 1

    print("=" * 100)
    print("OUTPUT COLLAPSE — Spearman rho and, on the pre-sigmoid scale, the affine relation")
    print("  §31 on the two-parameter shared head: rho(n,q) = 0.9967, affine R^2 = 0.987,")
    print("  logit(q) = 3.979 * logit(n) + 2.752. A gain far from 1 is what pins q to its bounds.")
    print("=" * 100)
    for pa, pb in PAIRS:
        print(f"\n  {pa} vs {pb}")
        print(f"    {'arm':<22} {'rho':>8} {'affine R2':>10} {'gain':>8} {'offset':>9}")
        for a in arms:
            if pa not in a["fields"] or pb not in a["fields"]:
                continue
            x, y = a["fields"][pa], a["fields"][pb]
            la = logit_of(x, *RANGES[pa], pa in a["log_space"])
            lb = logit_of(y, *RANGES[pb], pb in a["log_space"])
            r2, slope, off = affine_r2(la, lb)
            print(
                f"    {a['label']:<22} {spearmanr(x, y).statistic:>+8.4f} "
                f"{r2:>10.4f} {slope:>+8.3f} {off:>+9.3f}"
            )

    print("\n" + "=" * 100)
    print("BOUND SATURATION — does the field use its declared range, or pile up at the ends?")
    print("=" * 100)
    for key, (lo, hi) in RANGES.items():
        if not any(key in a["fields"] for a in arms):
            continue
        print(f"\n  {key}  declared [{lo}, {hi}]")
        print(
            f"    {'arm':<22} {'min':>10} {'max':>10} {'span %':>8} "
            f"{'at lo %':>8} {'at hi %':>8} {'distinct':>9}"
        )
        for a in arms:
            if key not in a["fields"]:
                continue
            v = a["fields"][key]
            tol = 0.01 * (hi - lo)
            print(
                f"    {a['label']:<22} {v.min():>10.5f} {v.max():>10.5f} "
                f"{100 * (v.max() - v.min()) / (hi - lo):>8.1f} "
                f"{100 * (v < lo + tol).mean():>8.2f} {100 * (v > hi - tol).mean():>8.2f} "
                f"{np.unique(v).size:>9d}"
            )

    print("\n" + "=" * 100)
    print("CHANNEL GEOMETRY — downstream width exponent b (w ~ Q^b) vs Leopold & Maddock")
    print("  L&M b ~ 0.50. q in [0,1] caps b at 0.375, so the box cannot reach it.")
    print("  b below is the p-CONSTANT part only (q*f). When p varies the real b is")
    print("  beta + q*f and beta has either sign — run downstream_geometry.py for it.")
    print("=" * 100)
    print(
        f"  {'arm':<22} {'median q':>10} {'q*f only':>10} "
        f"{'b in 0.4-0.6 %':>15} {'q at box top %':>15} {'median p':>10}"
    )
    for a in arms:
        if "q_spatial" not in a["fields"]:
            continue
        q = a["fields"]["q_spatial"]
        b = b_from_q(q)
        inb = 100 * ((b >= LM_B_BAND[0]) & (b <= LM_B_BAND[1])).mean()
        qhi, qlo = RANGES["q_spatial"][1], RANGES["q_spatial"][0]
        at_top = 100 * (q > qhi - 0.01 * (qhi - qlo)).mean()
        pmed = np.median(a["fields"]["p_spatial"]) if "p_spatial" in a["fields"] else float("nan")
        print(
            f"  {a['label']:<22} {np.median(q):>10.4f} {b_from_q(np.median(q)):>10.4f} "
            f"{inb:>15.1f} {at_top:>15.1f} {pmed:>10.3f}"
        )

    print("\n" + "=" * 100
          + "\nSKILL — arms are comparable to EACH OTHER; all learn three parameters,")
    print("  where the 0.7376 / 0.7600 reference learned two. shared_linear is the control.")
    print("=" * 100)
    for a in arms:
        ms = metric(a["manifest"], "nse", "kge")
        keep = {k: v for k, v in ms.items() if "median" in k.lower()}
        print(f"  {a['label']:<22} {keep if keep else ms}")

    out = Path("/home/tbindas/projects/ddrs/.ddrs/experiments/head-arch/arms_compare.json")
    out.write_text(
        json.dumps(
            [
                {
                    "run": a["run"],
                    "label": a["label"],
                    "rho": {
                        f"{pa}|{pb}": float(spearmanr(a["fields"][pa], a["fields"][pb]).statistic)
                        for pa, pb in PAIRS
                        if pa in a["fields"] and pb in a["fields"]
                    },
                    "metrics": metric(a["manifest"], "nse", "kge"),
                }
                for a in arms
            ],
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
