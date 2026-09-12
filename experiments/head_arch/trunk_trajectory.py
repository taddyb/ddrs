#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Trunk rank and output coupling across a training run's checkpoints.

The point of this, from findings §32: the head starts with an effective trunk
rank of 4.36 of 21 and ends at 1.38, so training destroys three of the four
directions it began with. The architecture was never the wall; the gradient
contracts a capable trunk. This shows the shape of that contraction, which
decides whether it is an early collapse (an initialisation/optimisation problem)
or a steady grind (the objective faithfully reporting that daily discharge
constrains one combination of channel parameters — §30).

Usage:
    experiments/head_arch/trunk_trajectory.py <trajectory-dir>
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def spectrum(h: np.ndarray) -> tuple[float, float, int]:
    hc = h - h.mean(axis=0, keepdims=True)
    v = np.linalg.svd(hc, compute_uv=False) ** 2
    tot = v.sum()
    if tot <= 0:
        return float("nan"), float("nan"), 0
    frac = v / tot
    return (
        float(tot**2 / (v**2).sum()),
        float(frac[0]),
        int(np.searchsorted(np.cumsum(frac), 0.90) + 1),
    )


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    rows = []
    for d in sorted(root.glob("epoch_*"), key=lambda p: int(re.sub(r"\D", "", p.name))):
        man_path = d / "manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text())
        params = man["params"]
        info = man["arms"]["trained"]
        h = np.fromfile(d / "trained.trunk.bin", dtype=np.float32).reshape(
            -1, info["trunk_shape"][1]
        )
        f = np.fromfile(d / "trained.init.bin", dtype=np.float32).reshape(-1, len(params))
        i_n, i_q = params.index("n"), params.index("q_spatial")
        pr, pc1, n90 = spectrum(h)
        r = np.corrcoef(logit(f[:, i_n]), logit(f[:, i_q]))[0, 1]
        rows.append(
            {
                "epoch": int(re.sub(r"\D", "", d.name)),
                "eff_rank": pr,
                "pc1": pc1,
                "n90": n90,
                "rho": float(spearmanr(f[:, i_n], f[:, i_q]).statistic),
                "affine_r2": float(r * r),
                "q_lo": float((f[:, i_q] < 0.01).mean()),
                "q_hi": float((f[:, i_q] > 0.99).mean()),
            }
        )

    if not rows:
        print(f"no epoch_* dumps under {root}")
        return 1

    print(f"trunk collapse trajectory — {root}")
    print("  at init the same topology measures eff.rank 4.36, PC1 0.412 (findings §32.2)")
    print()
    print(
        f"{'epoch':>6} {'eff.rank':>9} {'PC1':>7} {'dims 90%':>9} "
        f"{'rho(n,q)':>9} {'affine R2':>10} {'q<.01':>7} {'q>.99':>7}"
    )
    for r in rows:
        print(
            f"{r['epoch']:>6} {r['eff_rank']:>9.2f} {r['pc1']:>7.3f} {r['n90']:>9d} "
            f"{r['rho']:>+9.4f} {r['affine_r2']:>10.4f} "
            f"{r['q_lo']:>7.3f} {r['q_hi']:>7.3f}"
        )
    print()
    first, last = rows[0], rows[-1]
    print(
        f"epoch {first['epoch']} -> {last['epoch']}: "
        f"rank {first['eff_rank']:.2f} -> {last['eff_rank']:.2f}, "
        f"affine R2 {first['affine_r2']:.4f} -> {last['affine_r2']:.4f}"
    )
    half = next(
        (r["epoch"] for r in rows if r["eff_rank"] <= (4.36 + last["eff_rank"]) / 2), None
    )
    if half is not None:
        print(f"half of the collapse (from the init 4.36) is done by epoch {half}")

    (root / "trajectory.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {root / 'trajectory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
