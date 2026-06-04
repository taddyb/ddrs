"""Dump DDR's *trained* KAN parameters (n / q_spatial / p_spatial) over the
CONUS subset of MERIT reaches.

Mirrors scripts/dump_ddr_init_params.py from the previous parity plan but
loads a trained checkpoint instead of building a fresh head.

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_ddr_trained_params.py \
        --checkpoint <path.pt> \
        --conus-comids /home/tbindas/projects/ddrs/.ddrs/runs/<RUN>/kan_parameters.nc \
        --out /tmp/kan_params_trained_ddr.nc
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

ATTRS_NC = Path("~/projects/ddr/data/merit_global_attributes_v2.nc").expanduser()
STATS_JSON = Path(
    "~/projects/ddr/data/statistics/"
    "merit_attribute_statistics_merit_global_attributes_v2.nc.json"
).expanduser()
DDR_YAML = Path("~/projects/ddr/config/merit_training_config.yaml").expanduser()


def _load_state_dict(checkpoint_path: Path, device: str) -> dict:
    """Load a DDR checkpoint, auto-unwrapping the nested model_state_dict.

    DDR's training script saves a dict like
        {"model_state_dict": {...}, "optimizer_state_dict": {...}, ...}
    Older checkpoints sometimes saved the plain state_dict at the top level.
    Handle both layouts transparently.
    """
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help=".pt file produced by ddr/scripts/train.py")
    parser.add_argument(
        "--conus-comids", type=Path, required=True,
        help="NetCDF whose COMID coord defines the CONUS subset (typically "
             "DDRS's kan_parameters.nc).",
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Output NetCDF path (e.g. /tmp/...nc)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda:0 — keep cpu for reproducibility.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path.home() / "projects" / "ddr" / "src"))
    from ddr.nn.kan import kan as DdrKan
    from ddr.routing.utils import denormalize

    # ── 1. Load DDR's hyperparameters from its YAML ─────────────────────
    # DDR's merit_training_config.yaml does not store parameter_ranges or
    # log_space_parameters — those live only in the Python-class defaults
    # (ddr.validation.configs.Params).  Defaults are:
    #   parameter_ranges: n=[0.015,0.25], q_spatial=[0.0,1.0], p_spatial=[1.0,200.0]
    #   log_space_parameters: ["p_spatial"]   (n is LINEAR in DDR)
    # Task 2's Layer-0.5 inline verification confirmed median n = 0.0744 using
    # linear denormalization for n — this is the ground truth we match.
    _DEFAULT_RANGES = {
        "n":         [0.015, 0.25],
        "q_spatial": [0.0, 1.0],
        "p_spatial": [1.0, 200.0],
    }
    _DEFAULT_LOG_SPACE = {"p_spatial"}  # DDR denormalizes n in LINEAR space

    y = yaml.safe_load(DDR_YAML.read_text())
    input_vars = y["kan"]["input_var_names"]
    learnable  = y["kan"]["learnable_parameters"]
    ranges     = y.get("params", {}).get("parameter_ranges", _DEFAULT_RANGES)
    log_space  = set(y.get("params", {}).get("log_space_parameters", list(_DEFAULT_LOG_SPACE)))

    # ── 2. Build the KAN architecture + load trained weights ────────────
    model = DdrKan(
        input_var_names=input_vars,
        learnable_parameters=learnable,
        hidden_size=y["kan"]["hidden_size"],
        num_hidden_layers=y["kan"]["num_hidden_layers"],
        grid=y["kan"]["grid"],
        k=y["kan"]["k"],
        seed=y["seed"],
        device=args.device,
    )
    model.load_state_dict(_load_state_dict(args.checkpoint, args.device))
    model.eval()

    # ── 3. Load global attributes + z-score normalize ───────────────────
    ds_attrs = xr.open_dataset(ATTRS_NC)
    global_comids = ds_attrs["COMID"].values.astype("int64")
    spatial = torch.tensor(
        ds_attrs[input_vars].to_array("variable").values,
        dtype=torch.float32,
        device=args.device,
    )
    # NaN-fill with per-feature mean (matches DDR's _predict_kan_params).
    for r in range(spatial.shape[0]):
        row_mean = torch.nanmean(spatial[r])
        spatial[r, torch.isnan(spatial[r])] = row_mean

    stats = json.loads(STATS_JSON.read_text())
    means = torch.tensor(
        [stats[v]["mean"] for v in input_vars],
        dtype=torch.float32, device=args.device,
    )
    stds = torch.tensor(
        [stats[v]["std"] for v in input_vars],
        dtype=torch.float32, device=args.device,
    )
    normalized = ((spatial - means.unsqueeze(1)) / stds.unsqueeze(1)).T

    # ── 4. Batched KAN inference over the global set ────────────────────
    batch_size = 50_000
    raw_parts: dict[str, list[np.ndarray]] = {k: [] for k in learnable}
    for start in range(0, normalized.shape[0], batch_size):
        chunk = normalized[start:start + batch_size].to(args.device)
        with torch.no_grad():
            out = model(inputs=chunk)
        for k in learnable:
            raw_parts[k].append(out[k].cpu().numpy())

    raw_concat = {k: np.concatenate(parts, axis=0) for k, parts in raw_parts.items()}

    denorm = {}
    for k in learnable:
        v_t = torch.from_numpy(raw_concat[k])
        denorm[k] = denormalize(v_t, ranges[k], k in log_space).numpy().astype("float32")

    # ── 5. Intersect to the CONUS COMID order from --conus-comids ───────
    conus_ds = xr.open_dataset(args.conus_comids)
    conus_comids = conus_ds["COMID"].values.astype("int64")

    global_pos = {c: i for i, c in enumerate(global_comids)}
    missing = [c for c in conus_comids if c not in global_pos]
    if missing:
        raise SystemExit(
            f"{len(missing)} CONUS COMIDs missing from global attributes; "
            f"first 5: {missing[:5]}"
        )
    select = np.array([global_pos[c] for c in conus_comids], dtype=np.int64)

    out_vars = {k: denorm[k][select] for k in learnable}

    # ── 6. Write the NetCDF in CONUS order ──────────────────────────────
    xr.Dataset(
        {k: (("COMID",), out_vars[k]) for k in learnable},
        coords={"COMID": conus_comids},
        attrs={
            "source": f"ddr.nn.kan + checkpoint={args.checkpoint.name}",
            "seed": y["seed"],
            "grid": y["kan"]["grid"],
            "k": y["kan"]["k"],
            "ddrs_companion": str(args.conus_comids),
        },
    ).to_netcdf(args.out)
    print(f"wrote {len(conus_comids)} CONUS reaches → {args.out}")


if __name__ == "__main__":
    main()
