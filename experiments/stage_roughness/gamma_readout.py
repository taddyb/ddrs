#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "netCDF4", "scipy", "matplotlib"]
# ///
"""Read out a learned per-reach gamma field against the registered prediction.

`config/experiments/sr_gamma_learned.yaml` registered, before the run:
rho(n, gamma) > 0.9, i.e. gamma comes out as another relabelled copy of n.
This prints the gamma distribution, the Spearman correlations among the four
learned fields over live CONUS reaches, and how gamma varies with river size,
and draws one figure. Uses `plot/kan_parameters.nc` (needs `--plot`).

    experiments/stage_roughness/gamma_readout.py <run-id>
"""
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.stats import spearmanr

RUNS = Path("/home/tbindas/projects/ddrs/.ddrs/runs")


def main() -> int:
    rid = sys.argv[1]
    run = RUNS / rid
    ds = Dataset(run / "plot" / "kan_parameters.nc")
    if "gamma" not in ds.variables:
        print("no per-reach gamma in this run's NetCDF (gamma was not learned)")
        return 1
    comid = np.asarray(ds["COMID"][:], dtype=np.int64)
    f = {k: np.asarray(ds[k][:], dtype=np.float64) for k in ("n", "q_spatial", "p_spatial", "gamma")}
    cfg = (run / "config.yaml").read_text()
    m = re.search(r"^\s+attributes:\s*(\S+)", cfg, re.M)
    at = Dataset(m.group(1))
    ac = np.asarray(at["COMID"][:], dtype=np.int64)
    ua = np.asarray(at["log10_uparea"][:], dtype=np.float64)
    pos = {int(c): i for i, c in enumerate(ac)}
    idx = np.array([pos.get(int(c), -1) for c in comid])
    la = np.full(comid.size, np.nan)
    la[idx >= 0] = ua[idx[idx >= 0]]
    live = np.isfinite(la)
    lo, hi = 0.0, 0.5
    m_ = re.search(r"^\s+gamma:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]", cfg, re.M)
    if m_:
        lo, hi = float(m_.group(1)), float(m_.group(2))

    g = f["gamma"][live]
    pct = lambda p: np.percentile(g, p)  # noqa: E731
    print(f"{rid}: {live.sum():,} reaches with a drainage area")
    print(f"gamma  min {g.min():.4f}  p10 {pct(10):.4f}  median {np.median(g):.4f}  p90 {pct(90):.4f}  max {g.max():.4f}")
    print(f"       frac at floor (<= lo + 0.1% of range) {100 * (g <= lo + 1e-3 * (hi - lo)).mean():.1f}%   "
          f"frac at ceiling {100 * (g >= hi - 1e-3 * (hi - lo)).mean():.1f}%   box [{lo}, {hi}], sigmoid centre {0.5 * (lo + hi)}")
    print("\nSpearman rho over live reaches (registered prediction: rho(n, gamma) > 0.9):")
    keys = ["n", "q_spatial", "p_spatial", "gamma"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            r = spearmanr(f[a][live], f[b][live]).statistic
            flag = "  <-- registered prediction" if {a, b} == {"n", "gamma"} else ""
            print(f"  rho({a:>9}, {b:>9}) = {r:+.3f}{flag}")
    print("\ngamma by drainage-area class (median [p10, p90]):")
    edges = [-1, 1, 2, 3, 4, 7]
    for a, b in zip(edges[:-1], edges[1:]):
        sel = live & (la >= a) & (la < b)
        if sel.sum() < 50:
            continue
        gg = f["gamma"][sel]
        print(f"  10^{a}..10^{b} km2  n={sel.sum():>7,}  {np.median(gg):.3f} [{np.percentile(gg, 10):.3f}, {np.percentile(gg, 90):.3f}]"
              f"   n_0 median {np.median(f['n'][sel]):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run / "plots"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    ax[0, 0].hist(g, bins=80, color="tab:purple")
    ax[0, 0].axvline(0.5 * (lo + hi), color="k", ls="--", lw=1, label="sigmoid centre (init)")
    ax[0, 0].axvline(0.183, color="tab:red", ls=":", lw=1, label="0.183 (at-a-station L&M)")
    ax[0, 0].set_xlabel("learned gamma"); ax[0, 0].set_ylabel("reaches"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].hexbin(la[live], g, gridsize=60, bins="log", cmap="viridis")
    ax[0, 1].set_xlabel("log10 drainage area (km²)"); ax[0, 1].set_ylabel("gamma")
    ax[1, 0].hexbin(f["n"][live], g, gridsize=60, bins="log", cmap="viridis")
    ax[1, 0].set_xlabel("n_0 (roughness at d_ref)"); ax[1, 0].set_ylabel("gamma")
    ax[1, 0].set_title(f"rho(n, gamma) = {spearmanr(f['n'][live], g).statistic:+.3f}", fontsize=10)
    ax[1, 1].hexbin(f["q_spatial"][live], g, gridsize=60, bins="log", cmap="viridis")
    ax[1, 1].set_xlabel("q_spatial"); ax[1, 1].set_ylabel("gamma")
    ax[1, 1].set_title(f"rho(q, gamma) = {spearmanr(f['q_spatial'][live], g).statistic:+.3f}", fontsize=10)
    fig.suptitle(f"{rid}: learned stage-roughness exponent")
    fig.tight_layout()
    fig.savefig(out / "gamma_readout.png", dpi=150, facecolor="white")
    print(f"\nwrote {out / 'gamma_readout.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
