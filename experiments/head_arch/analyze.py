#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Statistics for `head_arch_screen`: does a head topology emit one latent
direction relabelled, or genuinely independent parameters?

Reads the raw f32 fields the Rust binary writes and reports, per arm:

  init      rho(n, q) as the head starts, before any training. The shared head
            measured 0.727 here at epoch 1 (§31), so part of the collapse is
            inherited from initialisation rather than learned.
  trunk     the singular spectrum of the penultimate activations h. This is the
            direct version of the rank check §31 could only infer from outputs.
            `pr` is the participation ratio (sum(s^2)^2 / sum(s^4)), an
            effective rank that does not need an arbitrary cutoff.
  capacity  the positive control. Two targets that are uncorrelated by
            construction and exactly recoverable from the attributes are fitted
            supervised. `fit_rho` near 0 means the topology CAN decorrelate two
            outputs; near 1 means it structurally cannot, and no objective will
            rescue it.

Usage:
    experiments/head_arch/analyze.py <out-dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read(path: Path, ncol: int) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(-1, ncol)


def logit(p: np.ndarray) -> np.ndarray:
    """Recover the pre-sigmoid column. The head's outputs are open on (0,1),
    but f32 saturation can still land exactly on a bound, so clip first."""
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def affine_r2(a: np.ndarray, b: np.ndarray) -> float:
    """R^2 of an OLS fit of b on a. This is the §31 statistic: 0.987 means
    98.7% of one parameter's pre-activation is a linear function of the
    other's, i.e. the two are the same direction with a gain."""
    r = pearson(a, b)
    return float("nan") if not np.isfinite(r) else r * r


def spectrum(h: np.ndarray) -> dict:
    hc = h - h.mean(axis=0, keepdims=True)
    s = np.linalg.svd(hc, compute_uv=False)
    v = s**2
    tot = v.sum()
    if tot <= 0:
        return {"pr": float("nan"), "pc1_frac": float("nan"), "n_90pct": 0}
    frac = v / tot
    pr = float(tot**2 / (v**2).sum())
    cum = np.cumsum(frac)
    return {
        "pr": pr,
        "pc1_frac": float(frac[0]),
        "pc2_frac": float(frac[1]) if len(frac) > 1 else float("nan"),
        "n_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "width": int(h.shape[1]),
    }


def bound_stats(p: np.ndarray) -> dict:
    return {
        "frac_lt_0.01": float((p < 0.01).mean()),
        "frac_gt_0.99": float((p > 0.99).mean()),
        "distinct": int(np.unique(p).size),
        "min": float(p.min()),
        "max": float(p.max()),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    meta = json.loads((out / "manifest.json").read_text())
    params = meta["params"]
    np_ = len(params)
    i_n = params.index("n") if "n" in params else 0
    i_q = params.index("q_spatial") if "q_spatial" in params else 1
    sup = params[:2]  # the two the capacity control supervises

    print(f"head topology screen — {out}")
    print(f"parameters: {params}   supervised in capacity control: {sup}")
    print(f"synthetic target correlation: {meta['target_corr']:+.2e} (0 by construction)")
    print(f"reaches: {meta['n_reaches']:,}   subsample: {meta['subsample_len']:,}")
    print()

    rows = []
    for name, info in meta["arms"].items():
        init = read(out / f"{name}.init.bin", np_)
        fitted = read(out / f"{name}.fitted.bin", np_)
        trunk = read(out / f"{name}.trunk.bin", info["trunk_shape"][1])

        ln, lq = logit(init[:, i_n]), logit(init[:, i_q])
        spec = spectrum(trunk)

        # In the capacity control the two supervised columns are the ones that
        # were told to be uncorrelated.
        f0, f1 = fitted[:, 0], fitted[:, 1]

        rows.append(
            {
                "arm": name,
                "rationale": info["rationale"],
                "init_rho_nq": spearmanr(init[:, i_n], init[:, i_q]).statistic,
                "init_affine_r2": affine_r2(ln, lq),
                "trunk_pr": spec["pr"],
                "trunk_pc1": spec["pc1_frac"],
                "trunk_n90": spec["n_90pct"],
                "trunk_width": spec["width"],
                "fit_loss0": info["fit_loss_first"],
                "fit_loss1": info["fit_loss_last"],
                "fit_rho": spearmanr(f0, f1).statistic,
                "fit_affine_r2": affine_r2(logit(f0), logit(f1)),
                "q_bounds": bound_stats(init[:, i_q]),
                "n_bounds": bound_stats(init[:, i_n]),
            }
        )

    w = max(len(r["arm"]) for r in rows) + 1
    print("AT INIT — what the topology inherits before any training")
    print(f"{'arm':<{w}} {'rho(n,q)':>9} {'affine R2':>10} {'q<.01':>7} {'q>.99':>7}")
    for r in rows:
        print(
            f"{r['arm']:<{w}} {r['init_rho_nq']:>+9.3f} {r['init_affine_r2']:>10.4f} "
            f"{r['q_bounds']['frac_lt_0.01']:>7.3f} {r['q_bounds']['frac_gt_0.99']:>7.3f}"
        )
    print()

    print("TRUNK RANK — how many directions the latent actually carries")
    print(f"{'arm':<{w}} {'width':>6} {'eff.rank':>9} {'PC1 frac':>9} {'dims for 90%':>13}")
    for r in rows:
        print(
            f"{r['arm']:<{w}} {r['trunk_width']:>6d} {r['trunk_pr']:>9.2f} "
            f"{r['trunk_pc1']:>9.4f} {r['trunk_n90']:>13d}"
        )
    print()

    print("CAPACITY CONTROL — can it decorrelate two outputs when TOLD the answer?")
    print(f"{'arm':<{w}} {'loss0':>9} {'loss1':>9} {'fit rho':>9} {'affine R2':>10}")
    for r in rows:
        print(
            f"{r['arm']:<{w}} {r['fit_loss0']:>9.5f} {r['fit_loss1']:>9.5f} "
            f"{r['fit_rho']:>+9.3f} {r['fit_affine_r2']:>10.4f}"
        )
    print()

    print("READING IT")
    print("  A topology whose capacity |fit rho| stays near 1 while its loss barely")
    print("  falls is structurally incapable of emitting two independent parameters:")
    print("  no objective and no amount of training will separate them.")
    print("  A topology that reaches |fit rho| near 0 with a low loss is capable, so")
    print("  if its ROUTING-trained fields still collapse, the cause is upstream of")
    print("  the architecture — the attributes, or what daily discharge can identify.")

    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
