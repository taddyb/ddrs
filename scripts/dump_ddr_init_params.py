"""DDR-side mirror of `examples/dump_init_params.rs`.

Builds DDR's `ddr.nn.kan.kan` at seed=42 with identical hyperparameters,
sweeps all CONUS MERIT reaches through it (with the same z-score
normalization + NaN-fill that DDRS applies), and writes per-COMID
denormalised parameters to a NetCDF.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_ddr_init_params.py \
        --out /tmp/kan_init_params_ddr.nc
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

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

# Verbatim from config/merit_training.yaml::params.parameter_ranges
PARAM_RANGES = {
    "n":         (0.015, 0.25),
    "q_spatial": (0.0, 1.0),
    "p_spatial": (1.0, 200.0),
}
LOG_SPACE = {"n"}
ATTRS_NC = Path("~/projects/ddr/data/merit_global_attributes_v2.nc").expanduser()
STATS_JSON = Path(
    "~/projects/ddr/data/statistics/merit_attribute_statistics_merit_global_attributes_v2.nc.json"
).expanduser()


def denormalize(sigmoid_out: np.ndarray, lo: float, hi: float, log_space: bool) -> np.ndarray:
    if log_space:
        # Matches src/config.rs::denormalize log branch (eps = 1e-6 on lo).
        lo_eff = np.log(lo + 1e-6)
        hi_eff = np.log(hi)
        return np.exp(sigmoid_out * (hi_eff - lo_eff) + lo_eff)
    return sigmoid_out * (hi - lo) + lo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/kan_init_params_ddr.nc"))
    args = parser.parse_args()

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
    model.eval()

    # Load z-score statistics (same file DDRS reads via AttrStats::open).
    with open(STATS_JSON) as f:
        stats = json.load(f)
    means = np.array([stats[v]["mean"] for v in INPUT_VAR_NAMES], dtype="float32")
    stds  = np.array([stats[v]["std"]  for v in INPUT_VAR_NAMES], dtype="float32")

    with xr.open_dataset(ATTRS_NC) as ds:
        comids = ds["COMID"].values.astype("int64")
        attr_block = np.stack(
            [ds[name].values.astype("float32") for name in INPUT_VAR_NAMES],
            axis=1,
        )  # shape: (n_reaches, n_attrs)

    # NaN-fill with per-feature row-means (mirrors DDRS fill_nans).
    col_means = np.nanmean(attr_block, axis=0)
    nan_mask = np.isnan(attr_block)
    attr_block[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Z-score normalize.
    attr_block = (attr_block - means[None, :]) / stds[None, :]

    n_reaches = attr_block.shape[0]
    print(f"n_reaches = {n_reaches}", flush=True)

    with torch.no_grad():
        out = model(inputs=torch.from_numpy(attr_block))
        raw = {k: v.cpu().numpy().astype("float32") for k, v in out.items()}

    denorm = {
        k: denormalize(raw[k], *PARAM_RANGES[k], k in LOG_SPACE)
        for k in LEARNABLE
    }

    xr.Dataset(
        {k: ("COMID", denorm[k]) for k in LEARNABLE},
        coords={"COMID": comids},
        attrs={"seed": SEED, "source": "ddr.nn.kan.kan @ seed=42 (z-score normalized)"},
    ).to_netcdf(args.out)
    print(f"wrote {n_reaches} reaches → {args.out}")


if __name__ == "__main__":
    main()
