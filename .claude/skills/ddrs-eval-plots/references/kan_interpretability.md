# Reference: KAN interpretability (sensitivity sweep)

Visualizes how one of ddrs's KAN-based heads (the disaggregation head,
`src/nn/disagg_head.rs`, or the routing head, `src/nn/kan_head.rs`) maps an
input to its output — inspired by pykan's signature per-edge spline plots
(`kan/MultKAN.py::plot()`), adapted to what's actually feasible here.

## Two methods, pick by effort/fidelity tradeoff

### 1. Black-box sensitivity sweep (implemented, always feasible)

Freeze every input but one at a representative value, sweep that one input
over its observed range, plot the resulting output response — a
partial-dependence-style plot. Needs only forward-pass evaluation, no access
to `rskan`'s internals. This is what `examples/kan_sensitivity_sweep.rs` +
`output/disagg_verification/plot_sensitivity.py` implement for the disagg
head: sweep precip intensity at a fixed hour (daily Q' held fixed), plot the
resulting 24-hour disaggregated shape at each intensity.

**Rust side** (`examples/kan_sensitivity_sweep.rs`): builds a `DisaggHead`,
runs `forward()` once per intensity value in a sweep, writes a tidy CSV
(`precip_intensity,hour,hourly_value`).

```bash
cargo run --release --example kan_sensitivity_sweep -- \
    --output output/disagg_verification/precip_sensitivity.csv
```

**Python side** (notebook template):

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path("/home/tbindas/projects/ddrs/output/disagg_verification/precip_sensitivity.csv")
PLOT_DIR = CSV.parent

df = pd.read_csv(CSV)
intensities = sorted(df["precip_intensity"].unique())

fig, ax = plt.subplots(figsize=(8, 5.5))
cmap = plt.cm.viridis
for i, intensity in enumerate(intensities):
    sub = df[df["precip_intensity"] == intensity].sort_values("hour")
    ax.plot(sub["hour"], sub["hourly_value"], marker="o", markersize=3,
            color=cmap(i / (len(intensities) - 1)),
            label=f"precip={intensity:g} mm/hr")
ax.set_xlabel("hour of day")
ax.set_ylabel("disaggregated hourly value")
ax.legend(fontsize=8)
fig.savefig(PLOT_DIR / "kan_precip_sensitivity.png", dpi=300, bbox_inches="tight", facecolor="white")
```

Reusable for the routing `KanHead` too: sweep one attribute (e.g. `aridity`)
across its z-scored range while holding the others at their dataset mean,
plot the resulting `n` / `q_spatial` / `p_spatial` response per learnable
parameter.

### 2. True per-edge spline plot (feasible, not yet built — the pykan-faithful method)

`rskan::KanLayer`'s fields are **public** (`grid`, `coef`, `scale_base`,
`scale_sp`, `mask` — verified against
`~/.cargo/git/checkouts/rskan-.../rskan/src/layer.rs`), and its forward pass
is exactly pykan's formula before the sum over inputs:

```rust
// KanLayer::forward, per (input i, output o) edge:
// y[b,i,o] = mask[i,o] * (scale_base[i,o] * SiLU(x[b,i]) + scale_sp[i,o] * spline_{i,o}(x[b,i]))
// (rskan sums over i internally; the per-edge y[b,i,o] tensor itself is what
// pykan's plot() renders — see rskan/src/layer.rs:191-220)
```

This means a genuine per-edge curve (not just a black-box sweep) is possible
by evaluating `coef2curve` (`rskan::spline::coef2curve`, also used internally
by `KanLayer::forward`) at a swept range of `x` values for a chosen
`(i, o)` edge, using that layer's own `grid`/`coef`/`k`, then combining with
`scale_base`/`scale_sp` exactly as `forward` does. **Not yet built** — needs
a small Rust dump (one edge, or all edges of one layer, over a swept input
range) since this computation isn't exposed as a standalone function in
`rskan`'s public API (only the full `forward` reduction is). Building this
would give a much closer analog to pykan's actual network-diagram plot
(one small curve per edge, arranged by layer) than the sensitivity sweep.
Check `rskan`'s version (`Cargo.toml`'s `rskan` tag) hasn't changed the
field visibility before relying on this.

## Notes

- **Not a real-checkpoint requirement.** Like the H6 verification plot, this
  is fine with synthetic/freshly-initialized heads (`DisaggHeadConfig::new(seed).init()`)
  when the goal is understanding the *mechanism*, not a trained model's
  learned behavior. For a trained checkpoint's actual response, swap in
  `load_kan_head` against a real `.mpk` and sweep real attribute/precip
  ranges (e.g. from the training data's z-score stats) instead of synthetic
  ones.
- **Held-fixed values matter.** A sensitivity sweep's shape depends on what
  the OTHER inputs are frozen at — pick a representative point (dataset mean/
  median), not an arbitrary one, and say so in the plot title/caption.
- **This family has no fixed input schema** (unlike hydrograph/parameter_map/
  metrics/parameter_swap/loss_landscape_h6) — the sweep script is
  purpose-built per head/parameter being inspected. Treat
  `examples/kan_sensitivity_sweep.rs` as the template to copy and adapt, not
  a fixed CLI to invoke unchanged.
