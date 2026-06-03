"""Dump per-tensor init statistics for DDR's `ddr.nn.kan.kan` head.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_kan_init_stats.py

Output: ~/projects/ddrs/tests/fixtures/kan_init_stats_ddr.csv
        with one row per parameter tensor.

Schema: name,shape,mean,std,min,max,abs_mean

Note: pykan's `KAN([H, H], seed=seed)` calls `torch.manual_seed(seed)`,
`np.random.seed(seed)`, and `random.seed(seed)` (MultKAN.__init__) BEFORE
the two outer Linears are initialised. That global-state side effect is
why DDR's `input.weight` and `output.weight` end up reproducible at fixed
seed — we preserve the same construction order here.
"""

import csv
from pathlib import Path
import sys

# Match DDRS's config/merit_training.yaml exactly.
SEED = 42
INPUT_VAR_NAMES = [
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]
LEARNABLE = ["n", "q_spatial", "p_spatial"]
HIDDEN_SIZE = 21
NUM_HIDDEN_LAYERS = 2
GRID = 50
K = 2

OUT_CSV = Path("~/projects/ddrs/tests/fixtures/kan_init_stats_ddr.csv").expanduser()


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

    rows: list[dict[str, object]] = []
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy().astype("float32")
        rows.append({
            "name":    name,
            "shape":   "x".join(str(d) for d in arr.shape),
            "mean":    float(arr.mean()),
            "std":     float(arr.std()),
            "min":     float(arr.min()),
            "max":     float(arr.max()),
            "abs_mean": float(abs(arr).mean()),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows → {OUT_CSV}")
    for r in rows:
        print(f"  {r['name']:40s} shape={r['shape']:20s} mean={r['mean']:+.4e} std={r['std']:.4e}")


if __name__ == "__main__":
    main()
