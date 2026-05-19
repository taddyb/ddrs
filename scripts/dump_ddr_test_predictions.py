"""Compute DDR's reference test-period predictions for SP-5 V4 verification.

Usage (under DDR's uv venv):
    cd ~/projects/ddr && uv run python ~/projects/ddrs/scripts/dump_ddr_test_predictions.py

Writes ~/projects/ddrs/fixtures/sp5/v4_ddr_test.zarr/ with predictions +
observations + gage_ids + time, matching the layout that ddrs's
write_predictions_zarr produces.

Runs DDR's _test loop with batch_size = n_days_total (single batch, full
window) so the reference is unambiguous regardless of any DDR multi-batch
shape semantics.

The FROZEN_* constants MUST match
~/projects/ddrs/src/training/forward.rs::{FROZEN_N, FROZEN_Q_SPATIAL,
FROZEN_P_SPATIAL}.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler

from ddr import dmc, streamflow
from ddr.scripts_utils import compute_daily_runoff
from ddr.validation import validate_config

FROZEN_N = 0.05
FROZEN_Q_SPATIAL = 0.5
FROZEN_P_SPATIAL = 21.0

DDR_ROOT = Path.home() / "projects/ddr"
DDRS_ROOT = Path.home() / "projects/ddrs"
OUTPUT_DIR = DDRS_ROOT / "fixtures/sp5"


def physical_to_normalized(physical, lo, hi, log_space):
    """Inverse of DDR's denormalize (src/ddr/routing/utils.py:166-182).

    Log-space branch uses lo + 1e-6 to mirror DDR exactly. Same as
    src/training/forward.rs::physical_to_normalized.
    """
    if log_space:
        log_lo = float(np.log(lo + 1e-6))
        log_hi = float(np.log(hi))
        return (float(np.log(physical)) - log_lo) / (log_hi - log_lo)
    return (physical - lo) / (hi - lo)


def load_cfg():
    """Compose config: load DDR's own yaml (satisfies extra='forbid' schema),
    override routing-physics from ddrs's yaml, then apply testing overrides.
    Same composition strategy as scripts/dump_ddr_loss.py.
    """
    ddr_yaml_path = DDR_ROOT / "config/merit_training_config.yaml"
    with ddr_yaml_path.open() as f:
        raw = yaml.safe_load(f)

    # Remove Hydra `defaults` key — OmegaConf/Pydantic doesn't need it.
    raw.pop("defaults", None)

    # Override routing-physics values from ddrs's YAML.
    # ddrs uses log_space_parameters: [n] (not DDR's default [p_spatial]).
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

    # Test-mode overrides — mirror ddrs's testing: section.
    test = ddrs_raw.get("testing", {})
    raw["experiment"]["start_time"] = test.get("start_time", "1995/10/01")
    raw["experiment"]["end_time"] = test.get("end_time", "2010/09/30")
    raw["experiment"]["rho"] = None  # full window in test
    # batch_size will be overridden below to n_days_total (single batch).
    raw["experiment"]["batch_size"] = 1  # placeholder; set after dataset init

    # Switch mode to testing — DDR uses _init_inference, which pre-builds
    # routing_dataclass for all gages (no per-batch gage collating needed).
    raw["mode"] = "testing"

    # Redirect save_path to /tmp — not doing a full training run.
    raw["params"]["save_path"] = "/tmp/dump_ddr_test_run"

    # Force CPU; the YAML uses GPU device index 0 by default.
    raw["device"] = "cpu"

    cfg = OmegaConf.create(raw)
    return validate_config(cfg, save_config=False)


def main():
    cfg = load_cfg()

    # Instantiate the MERIT dataset (loads adjacency, gage list, observations,
    # and pre-builds routing_dataclass for all gages via _init_inference).
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    assert dataset.routing_dataclass is not None, "routing_dataclass not set (inference init failed)"
    assert dataset.routing_dataclass.observations is not None, "observations not set"

    n_days_total = len(dataset.dates.daily_time_range)
    all_gage_ids = dataset.routing_dataclass.observations.gage_id.values
    num_gauges = len(all_gage_ids)
    n_active = dataset.routing_dataclass.spatial_attributes.shape[1]
    print(f"n_days_total = {n_days_total}, n_active reaches = {n_active}, num_gauges = {num_gauges}")

    # Normalize frozen physical parameters → [0, 1] using the inverse of
    # DDR's denormalize(). The +1e-6 epsilon for log-space MUST match both
    # DDR's denormalize() and Rust's physical_to_normalized in
    # src/training/forward.rs:71.
    pr = cfg.params.parameter_ranges
    log_params = set(cfg.params.log_space_parameters)
    n_norm = physical_to_normalized(FROZEN_N,         pr["n"][0],         pr["n"][1],         "n" in log_params)
    q_norm = physical_to_normalized(FROZEN_Q_SPATIAL, pr["q_spatial"][0], pr["q_spatial"][1], "q_spatial" in log_params)
    p_norm = physical_to_normalized(FROZEN_P_SPATIAL, pr["p_spatial"][0], pr["p_spatial"][1], "p_spatial" in log_params)
    print(f"n_norm={n_norm:.6f}, q_norm={q_norm:.6f}, p_norm={p_norm:.6f}")

    # Build frozen spatial params. Use string "cpu" (not torch.device("cpu")) —
    # DDR's triangular_sparse_solve does `if device == "cpu"` string compare.
    # See scripts/dump_ddr_loss.py for the same workaround.
    spatial_params = {
        "n":         torch.full((n_active,), float(n_norm), dtype=torch.float32),
        "q_spatial": torch.full((n_active,), float(q_norm), dtype=torch.float32),
        "p_spatial": torch.full((n_active,), float(p_norm), dtype=torch.float32),
    }

    flow_reader = streamflow(cfg)
    routing_model = dmc(cfg=cfg, device="cpu")
    routing_model.set_progress_info(epoch=0, mini_batch=0)

    # Single batch covering the whole window — unambiguous reference.
    # batch_size = n_days_total means the DataLoader yields exactly one batch.
    cfg.experiment.batch_size = n_days_total

    # Accumulator matching train_and_test.py:72 — shape (G, T_hourly).
    predictions = np.zeros([num_gauges, len(dataset.dates.hourly_time_range)])

    sampler = SequentialSampler(data_source=dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.experiment.batch_size,
        num_workers=0,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    with torch.no_grad():
        for i, routing_dataclass in enumerate(dataloader, start=0):
            routing_model.set_progress_info(epoch=0, mini_batch=i)

            streamflow_predictions = flow_reader(
                routing_dataclass=routing_dataclass, device="cpu", dtype=torch.float32
            )

            dmc_kwargs = {
                "routing_dataclass": routing_dataclass,
                "spatial_parameters": spatial_params,
                "streamflow": streamflow_predictions,
                "carry_state": i > 0,
            }
            dmc_output = routing_model(**dmc_kwargs)

            # Accumulate into the full hourly array, indexed by hourly_indices
            # (mirrors train_and_test.py:89).
            predictions[:, dataset.dates.hourly_indices] = dmc_output["runoff"].cpu().numpy()

    print(f"predictions hourly shape: {predictions.shape}")

    # Downsample hourly → daily with tau-dependent boundary trim.
    # Mirrors train_and_test.py:97 and scripts_utils.compute_daily_runoff.
    daily_runoff = compute_daily_runoff(torch.tensor(predictions), cfg.params.tau)
    print(f"daily_runoff shape: {daily_runoff.shape}, mean={daily_runoff.mean():.4f}")

    # Observations and time range — mirrors train_and_test.py:98-99.
    observations = dataset.routing_dataclass.observations.streamflow.values
    obs_trimmed = observations[:, 1:-1]
    time_range = dataset.dates.daily_time_range[1:-1]

    assert obs_trimmed.shape[1] == daily_runoff.shape[1], \
        f"obs/pred shape mismatch: {obs_trimmed.shape} vs {daily_runoff.shape}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_zarr = OUTPUT_DIR / "v4_ddr_test.zarr"
    if out_zarr.exists():
        shutil.rmtree(out_zarr)

    ds = xr.Dataset(
        data_vars={
            "predictions": (("gage_ids", "time"),
                            daily_runoff.astype(np.float64),
                            {"units": "m3/s", "long_name": "Streamflow"}),
            "observations": (("gage_ids", "time"),
                             obs_trimmed.astype(np.float64),
                             {"units": "m3/s", "long_name": "Observed Streamflow"}),
        },
        coords={
            "gage_ids": all_gage_ids,
            "time": time_range,
        },
        attrs={
            "description": "Predictions and obs for time period",
            "start time": "1995-10-01",
            "end time": "2010-09-30",
            "version": "sp5-v4-ref",
            "evaluation basins file": str(cfg.data_sources.gages),
            "model": "frozen",
        },
    )
    ds.to_zarr(out_zarr, mode="w")
    print(f"wrote {out_zarr}")


if __name__ == "__main__":
    main()
