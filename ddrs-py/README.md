# ddrs-py

Python bindings for the [ddrs](../) Muskingum-Cunge routing crate. Read-only
inference path on CPU (`NdArray` backend). Intended for notebook-based
interpretation, not training.

## Setup

Requires the same vendored `cubecl` + `burn` fork clones as the main `ddrs`
crate — see `[patch.crates-io]` in `Cargo.toml`. If those paths don't exist
on your machine, the build will fail with "no matching package found".

```bash
cd ddrs-py
uv venv
uv pip install maturin pytest numpy
uv run maturin develop --release
uv run pytest tests/smoke.py -v
```

`--release` is required — debug builds of PyO3 + BURN are unusably slow for
any real-sized attribute batch.

`patchelf` must be installed for `maturin develop` to update the venv copy:

```bash
# Arch:
sudo pacman -S patchelf
# Debian/Ubuntu:
sudo apt install patchelf
# Or via pip (less robust):
uv pip install patchelf
```

Without `patchelf`, `maturin develop` writes the `.so` only under
`target/maturin/` and leaves the stale venv copy. The workaround is a manual
`cp target/maturin/libddrs_py.so .venv/lib/python3.X/site-packages/ddrs_py/ddrs_py.abi3.so`
after each build.

## API

```python
import ddrs_py
import numpy as np

# 1. Inspect parameter bounds defined by the training YAML.
bounds = ddrs_py.parameter_bounds("config/merit_training.yaml")
# {"n": ((0.015, 0.25), True), "q_spatial": ((0.0, 1.0), False), ...}

# 2. Denormalize MLP outputs to physical units.
physical = ddrs_py.denormalize(
    np.array([0.0, 0.5, 1.0], dtype=np.float32),
    bounds=(1.0, 200.0),
    log_space=False,
)

# 3. Load a trained MLP checkpoint.
model = ddrs_py.load_mlp(
    checkpoint="output/saved_models/epoch_1_mb_35",   # NO .mpk suffix
    config_path="config/merit_training.yaml",
)

# 4. Inference on an arbitrary (R, F=10) attrs batch.
out = model.forward(attrs)
# {"n": np.ndarray (R,), "q_spatial": ..., "p_spatial": ...}

# 5. Full CONUS parameter map — drives the SP-10c choropleth.
result = ddrs_py.run_inference_over_conus(
    attrs_nc="/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc",
    conus_adjacency_zarr="/home/tbindas/projects/ddr/data/merit_conus_adjacency.zarr",
    checkpoint="output/saved_models/epoch_1_mb_35",
    config_path="config/merit_training.yaml",
)
# {"comid": (N,) int64, "n": (N,) float32, ...}
```

## Out of scope for 10a

- GPU inference (use the `eval` binary in the parent crate for that).
- Full Muskingum-Cunge forward pass from Python — coming in 10b.
- DDR `.pt` checkpoint loading — never supported (different runtime).
