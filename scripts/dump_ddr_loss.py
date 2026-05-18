"""Compute DDR's reference per-batch loss for SP-4 V1/V2 verification.

Usage (from ddrs repo root, using DDR's uv venv):
    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_loss.py --variant v1
    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_loss.py --variant v2

Writes ~/projects/ddrs/fixtures/sp4/{variant}_ddr_loss.json.

The FROZEN_* constants MUST match
~/projects/ddrs/src/training/forward.rs::{FROZEN_N, FROZEN_Q_SPATIAL,
FROZEN_P_SPATIAL}. If you change one, change the other.

Config loading strategy
-----------------------
DDR's Pydantic Config has extra="forbid", so we cannot feed it ddrs's YAML
directly (ddrs has `mlp` instead of `kan`, is missing `name`,
`data_sources.geospatial_fabric_gpkg`, and `data_sources.statistics`, and
has `experiment.grad_clip_max_norm` which DDR doesn't know).

Instead we load DDR's own merit_training_config.yaml as the base (which
already satisfies the schema), then override the routing-physics values from
ddrs's YAML (parameter_ranges, log_space_parameters, defaults) and set
cfg.params.save_path to /tmp so no disk write is attempted.

log_space_parameters note: ddrs uses ["n"] (not DDR's default ["p_spatial"]).
This override is explicit below and is what Rust's physical_to_normalized
uses — keep these in sync.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from ddr import dmc, streamflow
from ddr.io.functions import downsample
from ddr.validation import validate_config

# ---------------------------------------------------------------------------
# Frozen physical constants — must mirror src/training/forward.rs
# ---------------------------------------------------------------------------
FROZEN_N = 0.05
FROZEN_Q_SPATIAL = 0.5
FROZEN_P_SPATIAL = 21.0

OUTPUT_DIR = Path.home() / "projects/ddrs/fixtures/sp4"
DDR_ROOT = Path.home() / "projects/ddr"
DDRS_ROOT = Path.home() / "projects/ddrs"


def load_cfg():
    # Load DDR's own config as the base: it satisfies DDR's strict schema.
    ddr_yaml_path = DDR_ROOT / "config/merit_training_config.yaml"
    with ddr_yaml_path.open() as f:
        raw = yaml.safe_load(f)

    # Remove Hydra `defaults` key — OmegaConf/Pydantic doesn't need it.
    raw.pop("defaults", None)

    # Override routing-physics values from ddrs's YAML.
    # ddrs uses log_space_parameters: [n] (not DDR's default [p_spatial]).
    # parameter_ranges and defaults match between repos, but be explicit.
    ddrs_yaml_path = DDRS_ROOT / "config/merit_training.yaml"
    with ddrs_yaml_path.open() as f:
        ddrs_raw = yaml.safe_load(f)

    ddrs_params = ddrs_raw.get("params", {})
    raw.setdefault("params", {})
    if "parameter_ranges" in ddrs_params:
        raw["params"]["parameter_ranges"] = ddrs_params["parameter_ranges"]
    if "log_space_parameters" in ddrs_params:
        raw["params"]["log_space_parameters"] = ddrs_params["log_space_parameters"]
    if "defaults" in ddrs_params:
        raw["params"]["defaults"] = ddrs_params["defaults"]
    if "attribute_minimums" in ddrs_params:
        raw["params"]["attribute_minimums"] = ddrs_params["attribute_minimums"]

    # Redirect save_path to /tmp — we're not doing a full training run.
    raw["params"]["save_path"] = "/tmp/dump_ddr_loss_run"

    # Force CPU; the YAML uses GPU device index 0 by default.
    raw["device"] = "cpu"

    cfg = OmegaConf.create(raw)
    return validate_config(cfg, save_config=False)


def pick_batch(dataset, variant: str, seed: int, batch_size: int, rho: int):
    """Select a fixed set of gage IDs and set a deterministic date window."""
    if variant == "v1":
        gen = torch.Generator().manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            data_source=dataset, generator=gen
        )
        all_idx = list(sampler)[:batch_size]
        staids = dataset.gage_ids[all_idx].tolist()
    elif variant == "v2":
        staids = list(dataset.gage_ids)
    else:
        raise ValueError(variant)

    # Deterministic date window: same RNG as Rust's SequentialSampler seed path.
    sample_size = len(dataset.dates.daily_time_range)
    rng = np.random.default_rng(seed)
    start_day_idx = int(rng.integers(0, sample_size - rho))
    chunk = np.arange(start_day_idx, start_day_idx + rho)
    dataset.dates.set_date_range(chunk)
    return staids, start_day_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["v1", "v2"], required=True)
    args = parser.parse_args()

    cfg = load_cfg()

    # Instantiate the MERIT dataset (loads adjacency, gage list, observations).
    from ddr.geodatazoo.merit import Merit
    dataset = Merit(cfg=cfg)

    if args.variant == "v1":
        staids, start_day_idx = pick_batch(dataset, "v1", seed=42, batch_size=8, rho=90)
    else:
        staids, start_day_idx = pick_batch(dataset, "v2", seed=42, batch_size=len(dataset.gage_ids), rho=90)

    routing_dataclass = dataset._collate_gages(np.array(staids))
    n_active = routing_dataclass.spatial_attributes.shape[1]
    num_gauges = len(routing_dataclass.outflow_idx)

    # Normalize frozen physical parameters → [0, 1] using the inverse of
    # DDR's denormalize(). The +1e-6 epsilon for log-space MUST match both
    # DDR's denormalize() at ~/projects/ddr/src/ddr/routing/utils.py:180 and
    # Rust's physical_to_normalized in src/training/forward.rs:71.
    pr = cfg.params.parameter_ranges
    log_params = set(cfg.params.log_space_parameters)

    def physical_to_normalized(physical, lo, hi, log_space):
        if log_space:
            log_lo = float(np.log(lo + 1e-6))
            log_hi = float(np.log(hi))
            return (float(np.log(physical)) - log_lo) / (log_hi - log_lo)
        return (physical - lo) / (hi - lo)

    n_norm = physical_to_normalized(FROZEN_N,         pr["n"][0],         pr["n"][1],         "n" in log_params)
    q_norm = physical_to_normalized(FROZEN_Q_SPATIAL, pr["q_spatial"][0], pr["q_spatial"][1], "q_spatial" in log_params)
    p_norm = physical_to_normalized(FROZEN_P_SPATIAL, pr["p_spatial"][0], pr["p_spatial"][1], "p_spatial" in log_params)

    # Use string device ("cpu") — DDR's triangular_sparse_solve does an
    # equality check `if device == "cpu"` which fails with torch.device("cpu").
    device_str = cfg.device if isinstance(cfg.device, str) else str(cfg.device)
    device = torch.device(device_str)
    spatial_params = {
        "n":         torch.full((n_active,), float(n_norm), device=device, dtype=torch.float32),
        "q_spatial": torch.full((n_active,), float(q_norm), device=device, dtype=torch.float32),
        "p_spatial": torch.full((n_active,), float(p_norm), device=device, dtype=torch.float32),
    }

    flow_reader = streamflow(cfg)
    routing_model = dmc(cfg=cfg, device=device_str)
    routing_model.set_progress_info(epoch=0, mini_batch=0)

    streamflow_predictions = flow_reader(
        routing_dataclass=routing_dataclass, device=device_str, dtype=torch.float32
    )

    dmc_kwargs = {
        "routing_dataclass": routing_dataclass,
        "spatial_parameters": spatial_params,
        "streamflow": streamflow_predictions,
        "carry_state": False,
    }
    with torch.no_grad():
        dmc_output = routing_model(**dmc_kwargs)

    # Trim tau offset and downsample hourly → daily.
    # Mirrors src/training/loss.rs: slice [13+tau : -11+tau] then downsample.
    tau = cfg.params.tau
    sliced = dmc_output["runoff"][:, (13 + tau) : (-11 + tau)]
    num_days = sliced.shape[1] // 24
    daily_runoff = downsample(sliced, rho=num_days).numpy()  # (G, T_days)

    # Observations: xr.Dataset with dims (gage_id, time); trim outer day.
    obs = routing_dataclass.observations.streamflow.values  # (G, T_days_full)
    obs_trimmed = obs[:, 1:-1]
    assert obs_trimmed.shape[1] == daily_runoff.shape[1], (
        f"obs/pred T mismatch: {obs_trimmed.shape} vs {daily_runoff.shape}"
    )

    # NaN mask: drop gauges that have any NaN in observations.
    nan_mask = np.isnan(obs_trimmed).any(axis=1)
    keep_mask = ~nan_mask

    # Warmup: skip first N days (routing starts from dry conditions).
    warmup = cfg.experiment.warmup
    pred_kept = daily_runoff[keep_mask][:, warmup:]
    obs_kept = obs_trimmed[keep_mask][:, warmup:]

    loss = float(np.mean(np.abs(pred_kept - obs_kept)))

    out = {
        "variant": args.variant,
        "seed": 42,
        "batch_size": len(staids),
        "rho": 90,
        "start_day_idx": start_day_idx,
        "staids": list(map(str, staids)),
        "start_time": str(routing_dataclass.dates.batch_daily_time_range[0]),
        "frozen_n": FROZEN_N,
        "frozen_q_spatial": FROZEN_Q_SPATIAL,
        "frozen_p_spatial": FROZEN_P_SPATIAL,
        "n_active": int(n_active),
        "num_gauges": int(num_gauges),
        "loss": loss,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.variant}_ddr_loss.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}: loss={loss:.6f}")


if __name__ == "__main__":
    main()
