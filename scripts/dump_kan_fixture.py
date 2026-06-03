"""Dump a full DDR KAN init+forward+backward fixture for DDRS parity tests.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_kan_fixture.py

Output: ~/projects/ddrs/tests/fixtures/kan_head_init_seed42.npz

Contents:
    inputs              [64, 10] float32 — sampled with seed=0 in this script
    expected_n          [64]     float32 — DDR forward output for `n`
    expected_q_spatial  [64]     float32
    expected_p_spatial  [64]     float32
    input_weight        [21, 10] float32
    input_bias          [21]     float32
    output_weight       [3, 21]  float32
    output_bias         [3]      float32

    block_0_grid        [21, knots]      float32  (knots = grid+1+2k = 55)
    block_0_coef        [21, 21, n_basis] float32 (n_basis = grid+k = 52)
    block_0_scale_base  [21, 21] float32
    block_0_scale_sp    [21, 21] float32
    block_0_mask        [21, 21] float32
    block_1_*           (same shapes)

    grad_input_weight   [21, 10] float32
    grad_input_bias     [21]     float32
    grad_output_weight  [3, 21]  float32
    grad_output_bias    [3]      float32
    grad_block_<b>_<f>  (only for the trainable params per layer:
                         coef, scale_base, scale_sp; not grid or mask)

    meta                json blob (version, hyperparams)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 42
INPUTS_SEED = 0
BATCH = 64
INPUT_VAR_NAMES = [
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]
LEARNABLE = ["n", "q_spatial", "p_spatial"]
HIDDEN_SIZE = 21
NUM_HIDDEN_LAYERS = 2
GRID = 50
K = 2

OUT_NPZ = Path("~/projects/ddrs/tests/fixtures/kan_head_init_seed42.npz").expanduser()


def main() -> None:
    sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
    from ddr.nn.kan import kan as DdrKan  # type: ignore

    model = DdrKan(
        input_var_names=INPUT_VAR_NAMES,
        learnable_parameters=LEARNABLE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        grid=GRID,
        k=K,
        seed=SEED,
        device="cpu",
    )
    model.eval()  # disable dropout etc., though kan.py has none

    torch.manual_seed(INPUTS_SEED)
    inputs = torch.randn(BATCH, len(INPUT_VAR_NAMES), dtype=torch.float32)

    # Forward (with grad to capture the backward later).
    inputs_v = inputs.detach().clone().requires_grad_(False)
    out = model(inputs=inputs_v)
    expected = {k: v.detach().cpu().numpy().astype("float32") for k, v in out.items()}

    # Backward: scalar loss = sum of all three output parameters.
    loss = sum(out[k].sum() for k in LEARNABLE)
    loss.backward()

    # Param dict (named state).
    payload: dict[str, np.ndarray] = {
        "inputs": inputs.numpy().astype("float32"),
        **{f"expected_{k}": v for k, v in expected.items()},

        "input_weight":  model.input.weight.detach().cpu().numpy().astype("float32"),
        "input_bias":    model.input.bias.detach().cpu().numpy().astype("float32"),
        "output_weight": model.output.weight.detach().cpu().numpy().astype("float32"),
        "output_bias":   model.output.bias.detach().cpu().numpy().astype("float32"),

        "grad_input_weight":  model.input.weight.grad.detach().cpu().numpy().astype("float32"),
        "grad_input_bias":    model.input.bias.grad.detach().cpu().numpy().astype("float32"),
        "grad_output_weight": model.output.weight.grad.detach().cpu().numpy().astype("float32"),
        "grad_output_bias":   model.output.bias.grad.detach().cpu().numpy().astype("float32"),
    }

    for block_idx, layer in enumerate(model.layers):
        # MultKAN.act_fun is a ModuleList; for width=[H, H] it has length 1.
        inner = layer.act_fun[0]
        prefix = f"block_{block_idx}"
        payload[f"{prefix}_grid"]       = inner.grid.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_coef"]       = inner.coef.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_scale_base"] = inner.scale_base.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_scale_sp"]   = inner.scale_sp.detach().cpu().numpy().astype("float32")
        payload[f"{prefix}_mask"]       = inner.mask.detach().cpu().numpy().astype("float32")

        # Gradients only for trainable tensors (coef, scale_base, scale_sp).
        # grid and mask carry requires_grad=False.
        payload[f"grad_{prefix}_coef"]       = inner.coef.grad.detach().cpu().numpy().astype("float32")
        payload[f"grad_{prefix}_scale_base"] = inner.scale_base.grad.detach().cpu().numpy().astype("float32")
        payload[f"grad_{prefix}_scale_sp"]   = inner.scale_sp.grad.detach().cpu().numpy().astype("float32")

    meta = {
        "version": 1,
        "seed": SEED,
        "inputs_seed": INPUTS_SEED,
        "batch": BATCH,
        "in": len(INPUT_VAR_NAMES),
        "hidden": HIDDEN_SIZE,
        "out": len(LEARNABLE),
        "grid": GRID,
        "k": K,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "learnable_parameters": LEARNABLE,
    }
    payload["meta"] = np.array(json.dumps(meta), dtype=object)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **payload)
    print(f"wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size/1024:.1f} KiB)")
    print(f"  keys = {sorted(payload.keys())}")


if __name__ == "__main__":
    main()
