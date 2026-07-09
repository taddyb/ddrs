# Reference: H6 loss-landscape overlay (`--mode landscape`)

**Status as of 2026-07-08: the Rust grid-scan infrastructure EXISTS** —
`probe_zeta_gradient --mode landscape` (Stage 7 in
`src/bin/probe_zeta_gradient.rs`) implements the 11×11 (α, β) surface scan,
the 21-point linear barrier, and the single-point re-check. No registered H6
run has been executed yet: the registered protocol
(`docs/superpowers/specs/2026-07-08-landscape-hypotheses-h5-h6-draft.md`,
"H6 — Forcing-indexed valley") is a detached/overnight job that hasn't been
launched. If a user asks for H6 plots before real surface CSVs exist under
`output/equif/h6/`, point them at the invocation below instead.

## What H6 measures

For each forcing arm X ∈ {R1, R3}, a 2D grid over global multiplicative
scaling factors (log2 α = n-scale, log2 β = p-scale, both in [-1.5, 1.5],
11×11), evaluating the training loss `L_X(α, β)` at the arm-mean parameter
field scaled by (α, β) — q_spatial is held at the anchor value at every grid
point (only 2 axes are scanned). Plus: the grid's minimum `(α*_X, β*_X)`, a
5%-sublevel contour (degenerate-valley detection), a linear barrier scan
between the two arms' own parameter fields, and 1D profiles
`L*_X(α) = min_β L_X(α, β)` (and symmetric in β).

## How to produce the data (real CLI)

One invocation per forcing arm; BOTH arms' `dump_parameters` NetCDFs are
always passed (they define the shared anchor / barrier endpoints — the
anchor is identical across both invocations):

```bash
cargo run --release --bin probe_zeta_gradient -- \
    --mode landscape --backend cpu \
    --config .ddrs/runs/<arm-run-id>/config.yaml \
    --checkpoint .ddrs/runs/<arm-run-id>/checkpoints/epoch_5_mb_35 \
    --params-nc-a output/equif/R1_kan_parameters.nc \
    --params-nc-b output/equif/R3_kan_parameters.nc \
    --windows 16 --seed 42 \
    --surface-output output/equif/h6/r1_surface.csv \
    --barrier-output output/equif/h6/r1_barrier.csv
```

- `--windows 16` is the REGISTERED H6 sample size and must be passed
  explicitly (the CLI default is 32, inherited from the older grad probe —
  same trap as H5's `--windows 96`).
- `--grid-n` (default 11) and `--barrier-points` (default 21) are the
  registered resolutions; smaller values exist for smoke tests only.
- Split-half noise floor: re-invoke the surface with `--seed 123` (same
  `--windows 16`) and diff the two surfaces' minima in the Python analysis.
  Alternatively, the per-window rows in one CSV support a within-run
  split-half (windows 0–7 vs 8–15) without a second 121-point sweep.
- Full-96-window re-check of the grid argmin:
  `--single-point "<log2_alpha>,<log2_beta>" --windows 96`
  (surface CSV with a single grid point; requires `--surface-output`).
- 256-gauge stratified subsample: point the config's `data_sources.gages`
  at the pre-generated 256-gauge CSV (no CLI flag — same mechanism as H5's
  registered-protocol note).

## Artifact schema (as built)

Per-forcing-arm CSVs, tidy long format with a `window` column (one row per
grid/barrier point per window — deviates from the earlier draft's
window-averaged sketch so that per-window statistics and within-run
split-half are possible; take `groupby(["log2_alpha","log2_beta"]).mean()`
to recover the surface):

```
log2_alpha,log2_beta,window,mean_loss
-1.5,-1.5,0,12.34
-1.5,-1.2,0,11.98
...
```

```
t,window,mean_loss
0,0,7.13
0.05,0,7.15
...
```

Derived scalars per arm (`alpha_star`, `beta_star`, `sublevel_aspect_ratio`,
`minima_displacement`, barrier `B_X`, noise-floor displacement) are computed
by the Python analysis step, NOT in Rust — the binary prints an
informational barrier statistic at the end of a `--barrier-output` run, but
the authoritative statistics/verdicts live in the analysis script.

## Notebook sketch (once data exists)

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- USER INPUTS ----------------------------------------------------------
GRIDS = {
    "R1 forcing": Path("/home/tbindas/projects/ddrs/output/equif/h6/r1_surface.csv"),
    "R3 forcing": Path("/home/tbindas/projects/ddrs/output/equif/h6/r3_surface.csv"),
}
PLOT_DIR = Path("/home/tbindas/projects/ddrs/output/equif/h6/plots")
# -------------------------------------------------------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, len(GRIDS), figsize=(6 * len(GRIDS), 5), sharex=True, sharey=True)
minima = {}
for ax, (label, path) in zip(axes, GRIDS.items()):
    raw = pd.read_csv(path)
    # Per-window rows -> window-mean surface before pivoting.
    df = raw.groupby(["log2_alpha", "log2_beta"], as_index=False)["mean_loss"].mean()
    pivot = df.pivot(index="log2_beta", columns="log2_alpha", values="mean_loss")
    assert pivot.shape == (11, 11), f"expected a full 11x11 grid, got {pivot.shape}"
    im = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=20, cmap="viridis")
    fig.colorbar(im, ax=ax, label="mean L1 loss")
    # Minimum + 5% sublevel contour.
    min_val = df["mean_loss"].min()
    star = df.loc[df["mean_loss"].idxmin()]
    minima[label] = (star["log2_alpha"], star["log2_beta"])
    ax.scatter([star["log2_alpha"]], [star["log2_beta"]], color="red", marker="*", s=200, label="minimum")
    ax.contour(pivot.columns, pivot.index, pivot.values, levels=[min_val * 1.05],
               colors="red", linestyles="--")
    ax.set_title(label)
    ax.set_xlabel(r"$\log_2 \alpha$ (n-scale)")
    ax.set_ylabel(r"$\log_2 \beta$ (p-scale)")
    ax.legend()
fig.suptitle("H6: loss surface per forcing arm (red star = minimum, dashed = 5% sublevel)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "h6_surfaces.png", dpi=300, bbox_inches="tight", facecolor="white")

# Minima displacement (log-coord distance) — compare against the registered
# split-half noise floor before reading this as forcing-dependence.
labels = list(minima.keys())
if len(labels) == 2:
    (a1, b1), (a2, b2) = minima[labels[0]], minima[labels[1]]
    displacement = float(np.hypot(a1 - a2, b1 - b2))
    print(f"minima displacement ({labels[0]} vs {labels[1]}): {displacement:.4f} (log-coord units)")

print(f"saved H6 plots to {PLOT_DIR}")
```

## Implementation notes (for whoever analyzes/extends this)

- The mode reuses H5's `eval-loss` plumbing (`RoutingParamOverride`,
  `physical_to_normalized`, the single-sampled-plan-replayed pattern via the
  shared `sample_window_plan`/`eval_window_filtered` helpers) — H6 is the
  same "inject parameters, measure loss" primitive scanning a continuous
  (α, β) grid.
- Anchor arm-mean is geometric for config-log-space fields, arithmetic
  otherwise (read from `params.log_space_parameters` — in the R1/R3 configs
  only `p_spatial` is log-space). The BARRIER is log-space interpolation for
  all three fields regardless of that flag — the registered formula.
- Grid coordinates are computed in f64 then narrowed to f32, so CSV strings
  are clean decimals (`-0.9`, not `-0.29999995`) — but match grid points by
  float value, not string, in analysis code.
- Log2-not-linear axes are load-bearing (Dinh et al. 2017 reparameterization
  caveat, cited in the spec) — don't relabel axes as linear scale factors
  without also fixing the tick labels.
