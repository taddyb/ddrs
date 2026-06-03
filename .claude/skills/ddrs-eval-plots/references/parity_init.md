# Layer 4 — DDR ↔ DDRS init-distribution side-by-side

## When to use

After running both `examples/dump_init_params` and
`scripts/dump_ddr_init_params.py`. The two NetCDFs they produce are the
inputs to this plot family.

## Inputs

| File | Producer |
|------|----------|
| `/tmp/kan_init_params_ddrs.nc` | `cargo run --release --example dump_init_params` |
| `/tmp/kan_init_params_ddr.nc`  | `cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_init_params.py` |

## Notebook cells

Place the notebook at `<RUN_DIR>/plots/parity_init.ipynb` even though
this plot is run-independent — keeps the artefacts colocated with
whichever run is being investigated.

### Cell 1 — load
```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DDRS_NC = Path("/tmp/kan_init_params_ddrs.nc")
DDR_NC  = Path("/tmp/kan_init_params_ddr.nc")
PLOT_DIR = Path(".").resolve()  # set to <RUN_DIR>/plots/ if needed

dssrs = xr.open_dataset(DDRS_NC)
ddr_full = xr.open_dataset(DDR_NC)

# DDR's NetCDF covers the full MERIT global attribute set (~2.94M reaches);
# DDRS's NetCDF covers only CONUS (346k). Intersect to the CONUS subset.
ddr = ddr_full.sel(COMID=dssrs["COMID"].values)
assert (ddr["COMID"].values == dssrs["COMID"].values).all()
print(f"{dssrs.sizes['COMID']:,} CONUS reaches in both files")
```

### Cell 2 — histograms
```python
from scipy.stats import ks_2samp

PARAMS = ["n", "q_spatial", "p_spatial"]
RANGES = {"n": (0.015, 0.25), "q_spatial": (0, 1), "p_spatial": (1, 200)}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key in zip(axes, PARAMS):
    lo, hi = RANGES[key]
    bins = np.linspace(lo, hi, 60)
    ax.hist(ddr[key].values,   bins=bins, alpha=0.5, label="DDR",  density=True)
    ax.hist(dssrs[key].values, bins=bins, alpha=0.5, label="DDRS", density=True)
    ks = ks_2samp(ddr[key].values, dssrs[key].values)
    ax.set_title(f"{key}  KS={ks.statistic:.3f} (p={ks.pvalue:.2g})")
    ax.set_xlabel(key)
    ax.legend()
fig.suptitle("DDR ↔ DDRS init distributions — seed=42, no training", fontsize=16)
fig.tight_layout()
fig.savefig(PLOT_DIR / "parity_init.png", dpi=200, bbox_inches="tight")
plt.show()
```

### Cell 3 — pass/fail
```python
TOL = 0.05  # spec A5
verdict = {}
for key in PARAMS:
    ks = ks_2samp(ddr[key].values, dssrs[key].values).statistic
    verdict[key] = ("✓" if ks < TOL else "✗", float(ks))
print("DDR ↔ DDRS init parity (Layer 4):")
for k, (v, ks) in verdict.items():
    print(f"  {k:12s}  {v}  KS={ks:.4f}")
```

## Pass criterion

KS < 0.05 on all three parameters. If `n` fails specifically, the
saturation symptom is a head-init divergence; if all three fail, look
upstream (denormalize, attribute normalisation). If all three pass yet
the trained `n` saturates after 5 epochs, the bug is in training
dynamics (optimizer / batching / loss) — outside the scope of the
parity plan.
