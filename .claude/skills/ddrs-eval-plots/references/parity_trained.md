# Layer 2 — DDR ↔ DDRS trained-`n` parity comparison

Counterpart to `parity_init.md`. Where `parity_init.md` compares the
INIT distributions, this compares the TRAINED distributions on the
exact same CONUS reaches.

## When to use

After running DDRS's `train-and-test` workflow at `seed=42` AND DDR's
`scripts/train_and_test.py` at the same `seed=42`. Both runs must use
identical hyperparameters (`grid=50, k=2`, the same 10 attribute
names in the same order, the same 3 learnable params, 5 epochs).

## Inputs

| File | Producer |
|------|----------|
| `<RUN_DIR>/kan_parameters.nc` | `cargo run --release --bin dump_parameters -- --config <RUN_DIR>/config.yaml --checkpoint <CKPT> --output <RUN_DIR>/kan_parameters.nc` |
| `/tmp/kan_params_trained_ddr.nc` | `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_trained_params.py --checkpoint <DDR_CKPT> --conus-comids <RUN_DIR>/kan_parameters.nc --out /tmp/kan_params_trained_ddr.nc` |

## Notebook cells

Place at `<RUN_DIR>/plots/parity_trained.ipynb`.

### Cell 1 — load
```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ks_2samp, spearmanr

DDRS_NC = Path("../kan_parameters.nc")
DDR_NC  = Path("/tmp/kan_params_trained_ddr.nc")
PLOT_DIR = Path(".").resolve()

ddrs = xr.open_dataset(DDRS_NC)
ddr  = xr.open_dataset(DDR_NC)
assert (ddrs["COMID"].values == ddr["COMID"].values).all(), "COMID order must match"
print(f"{ddrs.sizes['COMID']:,} CONUS reaches in both files")
```

### Cell 2 — per-distribution stats
```python
rows = []
for k in ["n", "q_spatial", "p_spatial"]:
    a, b = ddrs[k].values, ddr[k].values
    ks = ks_2samp(a, b).statistic
    sp = spearmanr(a, b).correlation
    rows.append({
        "param": k,
        "ddrs_med":  float(np.median(a)),
        "ddr_med":   float(np.median(b)),
        "ddrs_p5":   float(np.percentile(a, 5)),
        "ddr_p5":    float(np.percentile(b, 5)),
        "ddrs_p95":  float(np.percentile(a, 95)),
        "ddr_p95":   float(np.percentile(b, 95)),
        "KS":        float(ks),
        "Spearman":  float(sp),
    })
pd.DataFrame(rows).set_index("param")
```

### Cell 3 — side-by-side histograms
```python
RANGES = {"n": (0.015, 0.25), "q_spatial": (0, 1), "p_spatial": (1, 200)}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key in zip(axes, ["n", "q_spatial", "p_spatial"]):
    lo, hi = RANGES[key]
    bins = np.linspace(lo, hi, 80)
    ax.hist(ddr[key].values,  bins=bins, alpha=0.5, label="DDR  trained", density=True)
    ax.hist(ddrs[key].values, bins=bins, alpha=0.5, label="DDRS trained", density=True)
    ks = ks_2samp(ddr[key].values, ddrs[key].values).statistic
    ax.set_title(f"{key}  KS={ks:.3f}")
    ax.set_xlabel(key)
    ax.legend()
fig.suptitle("DDR ↔ DDRS trained-parameter distributions — seed=42, 5 epochs", fontsize=16)
fig.tight_layout()
fig.savefig(PLOT_DIR / "parity_trained_hist.png", dpi=200, bbox_inches="tight")
plt.show()
```

### Cell 4 — per-gauge scatter (for `n`)
```python
fig, ax = plt.subplots(figsize=(8, 8))
ax.hexbin(ddr["n"].values, ddrs["n"].values, gridsize=80, cmap="viridis", mincnt=1)
lo, hi = 0.015, 0.25
ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
sp = spearmanr(ddr["n"].values, ddrs["n"].values).correlation
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("DDR trained n"); ax.set_ylabel("DDRS trained n")
ax.set_title(f"Per-reach DDR vs DDRS trained n   Spearman = {sp:.3f}")
ax.legend()
fig.savefig(PLOT_DIR / "parity_trained_scatter.png", dpi=200, bbox_inches="tight")
plt.show()
```

### Cell 5 — verdict
```python
KS_PASS  = 0.10   # spec A4
SP_PASS  = 0.90   # spec A5
KS_FAIL  = 0.20
SP_FAIL  = 0.70

print("DDR ↔ DDRS trained-distribution parity (Layer 2):\n")
for k in ["n", "q_spatial", "p_spatial"]:
    a, b = ddrs[k].values, ddr[k].values
    ks = float(ks_2samp(a, b).statistic)
    sp = float(spearmanr(a, b).correlation)
    if   ks <= KS_PASS and sp >= SP_PASS: v = "✓ same saturation"
    elif ks >= KS_FAIL  or  sp <= SP_FAIL: v = "✗ real divergence"
    else:                                  v = "~ borderline"
    print(f"  {k:12s}  KS={ks:.4f}  Spearman={sp:+.4f}   {v}")
```

## Pass criterion

The `n` row of Cell 5 must print `✓ same saturation` for the trained
parity verdict to land. `q_spatial` and `p_spatial` are reported for
completeness — they were not the original symptom.

If `n` lands `~ borderline`: bring to user. If `n` lands `✗`: a new
spec localizing the `src/training/` divergence is warranted.
