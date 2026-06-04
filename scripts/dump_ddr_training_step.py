"""Dump DDR's per-stage training-step state for one mini-batch.

Produces 6 .npz files + manifest.json under
tests/fixtures/training_step/ for the smallest CONUS gauge with a
meaningful subgraph (STAID=10336740, 19 reaches).

Run under DDR's uv venv:

    cd ~/projects/ddr && uv run python \
        ~/projects/ddrs/scripts/dump_ddr_training_step.py

Consumed by tests/training_step_layer_{b,c,d}.rs in subsequent commits.

Weight layout note
------------------
The KAN fixture (tests/fixtures/kan_head_init_seed42.npz) stores Linear
weights in burn's [in, out] convention.  PyTorch's nn.Linear expects
[out, in].  We therefore TRANSPOSE input_weight and output_weight before
loading into the DDR model.  KanLayer fields (coef, scale_base, scale_sp,
grid, mask) are stored in the same layout that pykan uses, so no
transposition is needed there.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from scipy import sparse

# ---------------------------------------------------------------------------
# Fixture identity
# ---------------------------------------------------------------------------
STAID = "10336740"          # smallest CONUS gauge with n_reaches > 1
COMID = 77006074            # gage_catchment attr from merit_gages_conus_adjacency
DRAIN_SQKM = 5.537          # from gages_3000.csv

# Time window — fixed, deterministic.
START_TIME = "1990/01/01"
RHO = 90                    # days  (matches DDR's experiment.rho)
# Actual hourly window: DDR's Dates uses inclusive='left' from daily[0] to
# daily[-1], giving (rho-1)*24 = 2136 hours (not 2160). Computed at runtime.

# Matching the merit_training_config
TAU = 3
WARMUP_DAYS = 5

INPUT_VAR_NAMES = [
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]
LEARNABLE = ["n", "q_spatial", "p_spatial"]
HIDDEN_SIZE = 21
NUM_HIDDEN_LAYERS = 2
GRID = 50
K = 2
KAN_SEED = 42

DDR_ROOT = Path("~/projects/ddr").expanduser()
DDRS_ROOT = Path("~/projects/ddrs").expanduser()
KAN_FIXTURE = DDRS_ROOT / "tests/fixtures/kan_head_init_seed42.npz"
OUT_DIR = DDRS_ROOT / "tests/fixtures/training_step"

# ---------------------------------------------------------------------------
# Config helpers — mirror dump_ddr_loss.py pattern
# ---------------------------------------------------------------------------

def load_cfg():
    """Build a DDR Config from merit_training_config.yaml on CPU."""
    ddr_yaml_path = DDR_ROOT / "config/merit_training_config.yaml"
    with ddr_yaml_path.open() as f:
        raw = yaml.safe_load(f)

    raw.pop("defaults", None)

    # Load ddrs params overrides (log_space_parameters, etc.)
    ddrs_yaml_path = DDRS_ROOT / "config/merit_training.yaml"
    with ddrs_yaml_path.open() as f:
        ddrs_raw = yaml.safe_load(f)

    ddrs_params = ddrs_raw.get("params", {})
    raw.setdefault("params", {})
    for key in ("parameter_ranges", "log_space_parameters", "defaults", "attribute_minimums"):
        if key in ddrs_params:
            raw["params"][key] = ddrs_params[key]

    # Override time window to our fixture window.
    raw["experiment"]["start_time"] = START_TIME
    # end_time just needs to be >= START_TIME + RHO days
    import pandas as pd
    end_dt = pd.Timestamp(START_TIME) + pd.Timedelta(days=RHO + 1)
    raw["experiment"]["end_time"] = end_dt.strftime("%Y/%m/%d")
    raw["experiment"]["rho"] = RHO

    # CPU only; no Hydra save.
    raw["device"] = "cpu"
    raw["params"]["save_path"] = "/tmp/dump_ddr_training_step"

    sys.path.insert(0, str(DDR_ROOT / "src"))
    from ddr.validation import validate_config  # type: ignore
    cfg = OmegaConf.create(raw)
    return validate_config(cfg, save_config=False)


# ---------------------------------------------------------------------------
# KAN fixture loader — transposes Linear weights from burn [in,out] → [out,in]
# ---------------------------------------------------------------------------

def load_kan_from_fixture(cfg) -> "torch.nn.Module":
    """Load the KAN head from kan_head_init_seed42.npz into a DDR kan instance.

    The npz stores Linear weights in burn's convention [in, out].
    PyTorch nn.Linear.weight has shape [out, in], so we transpose.
    KanLayer parameters (coef, scale_base, scale_sp, grid, mask) use the
    same layout in both pykan and rskan — no transposition needed.
    """
    from ddr.nn.kan import kan as DdrKan  # type: ignore

    model = DdrKan(
        input_var_names=INPUT_VAR_NAMES,
        learnable_parameters=LEARNABLE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        grid=GRID,
        k=K,
        seed=KAN_SEED,
        device="cpu",
    )

    npz = np.load(KAN_FIXTURE)

    # Build state_dict from the fixture arrays.
    # dump_kan_fixture.py stored model.input.weight directly from PyTorch, so
    # input_weight is [out=21, in=10] and output_weight is [out=3, in=21] —
    # already in PyTorch [out, in] convention.  No transposition needed.
    # (The burn [in, out] ↔ PyTorch [out, in] transpose only matters when
    # loading burn tensors; the fixture was dumped from PyTorch so it's fine.)
    state = {
        "input.weight": torch.tensor(npz["input_weight"], dtype=torch.float32),
        "input.bias":   torch.tensor(npz["input_bias"],   dtype=torch.float32),
        "output.weight": torch.tensor(npz["output_weight"], dtype=torch.float32),
        "output.bias":   torch.tensor(npz["output_bias"],   dtype=torch.float32),
    }

    for b in range(NUM_HIDDEN_LAYERS):
        prefix = f"block_{b}"
        layer_prefix = f"layers.{b}.act_fun.0"
        state[f"{layer_prefix}.grid"]       = torch.tensor(npz[f"{prefix}_grid"],       dtype=torch.float32)
        state[f"{layer_prefix}.coef"]       = torch.tensor(npz[f"{prefix}_coef"],       dtype=torch.float32)
        state[f"{layer_prefix}.scale_base"] = torch.tensor(npz[f"{prefix}_scale_base"], dtype=torch.float32)
        state[f"{layer_prefix}.scale_sp"]   = torch.tensor(npz[f"{prefix}_scale_sp"],   dtype=torch.float32)
        state[f"{layer_prefix}.mask"]       = torch.tensor(npz[f"{prefix}_mask"],       dtype=torch.float32)

    # strict=False: pykan has extra internal state (node_bias, symbolic_fun, etc.)
    # not present in the fixture. These extras are not used in the forward pass.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in state_dict: {unexpected}")
    print(f"  load_state_dict: {len(missing)} missing (expected: pykan internal buffers)")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Subgraph construction — mirrors Merit._collate_gages for a single gauge
# ---------------------------------------------------------------------------

def build_subgraph(cfg):
    """Build the compressed CSR subgraph for the fixture gauge.

    Returns
    -------
    csr : scipy.sparse.csr_matrix   — compressed CSR (n_active × n_active)
    active_indices : np.ndarray[int64] — CONUS-level row indices (ascending)
    comid_order : np.ndarray[int64]    — MERIT COMIDs in topological order
    gage_idx_compressed : int          — compressed index of the gauge reach
    """
    import zarr  # type: ignore

    subsets = zarr.open(str(DDR_ROOT / "data/merit_gages_conus_adjacency.zarr"), mode="r")
    conus_adjacency = zarr.open(str(DDR_ROOT / "data/merit_conus_adjacency.zarr"), mode="r")

    gauge_root = subsets[STAID]
    attrs = dict(gauge_root.attrs)
    raw_rows = gauge_root["indices_0"][:].astype(np.int64)
    raw_cols = gauge_root["indices_1"][:].astype(np.int64)
    gage_idx_global = int(attrs["gage_idx"])

    # Deduplicate edges (set union) — mirrors Merit._collate_gages line
    # that uses a `set()` of (row, col) pairs.
    coord_set = set(zip(raw_rows.tolist(), raw_cols.tolist()))
    if coord_set:
        rows_arr, cols_arr = zip(*coord_set)
        rows_np = np.array(list(rows_arr), dtype=np.int64)
        cols_np = np.array(list(cols_arr), dtype=np.int64)
    else:
        rows_np = np.array([], dtype=np.int64)
        cols_np = np.array([], dtype=np.int64)

    # Active indices: edge nodes ∪ gage index, sorted ascending.
    edge_indices = np.unique(np.concatenate([rows_np, cols_np])) if len(rows_np) > 0 else np.array([], dtype=np.int64)
    gage_indices = np.array([gage_idx_global], dtype=np.int64)
    active_indices = np.unique(np.concatenate([edge_indices, gage_indices]))
    index_mapping = {orig: compr for compr, orig in enumerate(active_indices)}

    if len(rows_np) > 0:
        comp_rows = np.array([index_mapping[r] for r in rows_np], dtype=np.int64)
        comp_cols = np.array([index_mapping[c] for c in cols_np], dtype=np.int64)
    else:
        comp_rows = np.array([], dtype=np.int64)
        comp_cols = np.array([], dtype=np.int64)

    n_active = len(active_indices)
    coo = sparse.coo_matrix(
        (np.ones(len(comp_rows), dtype=np.float32), (comp_rows, comp_cols)),
        shape=(n_active, n_active),
    )
    csr = coo.tocsr()

    # MERIT COMIDs in topological order (from conus_adjacency["order"]).
    merit_ids_all = conus_adjacency["order"][:]
    comid_order = merit_ids_all[active_indices].astype(np.int64)

    gage_idx_compressed = index_mapping[gage_idx_global]

    return csr, active_indices, comid_order, gage_idx_compressed


# ---------------------------------------------------------------------------
# Streamflow reader — delegates to DDR's StreamflowReader via routing_dataclass
# ---------------------------------------------------------------------------
# (No standalone reader needed — Stage 3 uses StreamflowReader directly.)




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading DDR config...")
    cfg = load_cfg()

    print(f"Loading KAN fixture from {KAN_FIXTURE}...")
    model = load_kan_from_fixture(cfg)

    print(f"Building subgraph for STAID={STAID} (COMID={COMID})...")
    csr, active_indices, comid_order, gage_idx_compressed = build_subgraph(cfg)
    n_reaches = len(comid_order)
    print(f"  n_reaches={n_reaches}, gage_idx_compressed={gage_idx_compressed}")

    # -------------------------------------------------------------------
    # Stage 1: Dump subgraph
    # -------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coo_subgraph = csr.tocoo()
    subgraph_file = f"subgraph_{COMID}.npz"
    np.savez(
        OUT_DIR / subgraph_file,
        rows=coo_subgraph.row.astype(np.int64),
        cols=coo_subgraph.col.astype(np.int64),
        vals=coo_subgraph.data.astype(np.float32),
        comid_order=comid_order,
    )
    print(f"  wrote {subgraph_file}: {(OUT_DIR / subgraph_file).stat().st_size} bytes")

    # -------------------------------------------------------------------
    # Stages 2 & 3: Build routing_dataclass, read streamflow, then compute
    # hot-start and run MC forward.
    # We use DDR's own Merit._collate_gages and StreamflowReader to ensure
    # the subgraph, attributes, and streamflow all come from the same pipeline.
    # -------------------------------------------------------------------
    print("Building routing_dataclass via Merit._collate_gages...")

    from ddr import streamflow as StreamflowReader  # type: ignore
    from ddr.geodatazoo.merit import Merit  # type: ignore
    import pandas as pd

    # We need a proper RoutingDataclass for the MC engine.
    # Use Merit._collate_gages with our fixture gauge.
    dataset = Merit(cfg=cfg)

    # Set the batch time range to our fixture window
    start_ts = pd.Timestamp(START_TIME)
    chunk_indices = np.arange(
        dataset.dates.daily_time_range.get_loc(start_ts),
        dataset.dates.daily_time_range.get_loc(start_ts) + RHO,
    )
    dataset.dates.set_date_range(chunk_indices)

    routing_dataclass = dataset._collate_gages(np.array([STAID]))

    n_active = routing_dataclass.spatial_attributes.shape[1]
    print(f"  n_active (from routing_dataclass)={n_active}")

    # KAN forward: normalized_spatial_attributes → sigmoid outputs [0,1]
    # No torch.no_grad() here — we need the autograd tape for loss.backward().
    norm_attrs = routing_dataclass.normalized_spatial_attributes.to("cpu")  # (n_active, 10)
    kan_out = model(inputs=norm_attrs)  # dict of {n, q_spatial, p_spatial}

    # Read streamflow for the full routing window via the StreamflowReader
    flow_reader = StreamflowReader(cfg)
    q_prime_full = flow_reader(
        routing_dataclass=routing_dataclass, device="cpu", dtype=torch.float32
    )  # (RHO_HOURS, n_active)

    # -------------------------------------------------------------------
    # Stage 2: Hot-start
    # Computed from q_prime[0] via the PatternMapper (same as MC engine does
    # internally in _init_discharge_state / carry_state=False).
    # -------------------------------------------------------------------
    from ddr.routing.mmc import compute_hotstart_discharge, MuskingumCunge  # type: ignore

    adj_tensor = routing_dataclass.adjacency_matrix.to("cpu")
    discharge_lb = torch.tensor(cfg.params.attribute_minimums["discharge"], dtype=torch.float32)

    routing_engine_hs = MuskingumCunge(cfg, device="cpu")
    routing_engine_hs.network = adj_tensor
    mapper_hs, _, _ = routing_engine_hs.create_pattern_mapper()

    hotstart = compute_hotstart_discharge(
        q_prime_t0=q_prime_full[0],
        mapper=mapper_hs,
        discharge_lb=discharge_lb,
        device="cpu",
    )
    hotstart_np = hotstart.detach().numpy().astype(np.float32)

    hotstart_file = f"hotstart_{COMID}.npz"
    np.savez(
        OUT_DIR / hotstart_file,
        hotstart=hotstart_np,
        q_prime_t0=q_prime_full[0].detach().numpy().astype(np.float32),
    )
    print(f"  wrote {hotstart_file}: shape={hotstart_np.shape}, "
          f"{(OUT_DIR / hotstart_file).stat().st_size} bytes")

    # MC routing forward with gradients enabled (for loss.backward() later).
    from ddr import dmc as DmcModule  # type: ignore
    routing_model = DmcModule(cfg=cfg, device="cpu")
    routing_model.set_progress_info(epoch=0, mini_batch=0)

    dmc_out = routing_model(
        routing_dataclass=routing_dataclass,
        spatial_parameters=kan_out,
        streamflow=q_prime_full,
        carry_state=False,
    )
    runoff = dmc_out["runoff"]  # (n_gauges, RHO_HOURS) — scatter-mode output

    # Grab denormalized params from the routing engine for the fixture.
    n_param_np = routing_model.routing_engine.n.detach().numpy().astype(np.float32)
    q_spatial_np = routing_model.routing_engine.q_spatial.detach().numpy().astype(np.float32)
    p_spatial_np = routing_model.routing_engine.p_spatial.detach().numpy().astype(np.float32)

    mc_forward_file = f"mc_forward_{COMID}.npz"
    np.savez(
        OUT_DIR / mc_forward_file,
        Q=runoff.detach().numpy().astype(np.float32),
        n_param=n_param_np,
        q_spatial_param=q_spatial_np,
        p_spatial_param=p_spatial_np,
        q_prime_full=q_prime_full.detach().numpy().astype(np.float32),
    )
    print(f"  wrote {mc_forward_file}: shape={runoff.shape}, "
          f"{(OUT_DIR / mc_forward_file).stat().st_size} bytes")

    # -------------------------------------------------------------------
    # Stage 4: Tau-trim + daily downsample
    # Mirrors train.py:78-81: runoff[:, 13:(-11+tau)] then downsample
    # -------------------------------------------------------------------
    from ddr.io.functions import downsample  # type: ignore

    tau = cfg.params.tau  # 3
    trimmed = runoff[:, 13 : (-11 + tau)]
    n_days = trimmed.shape[1] // 24
    daily_q = downsample(trimmed, rho=n_days)  # (n_gauges, n_days)

    daily_q_file = f"daily_q_{COMID}.npz"
    np.savez(
        OUT_DIR / daily_q_file,
        daily_q=daily_q.detach().numpy().astype(np.float32),
        tau=np.int32(tau),
        n_days=np.int32(n_days),
    )
    print(f"  wrote {daily_q_file}: shape={daily_q.shape}, "
          f"{(OUT_DIR / daily_q_file).stat().st_size} bytes")

    # -------------------------------------------------------------------
    # Stage 5: Loss + gradients
    # Mirrors train.py:84-102
    # -------------------------------------------------------------------
    # Observations come from routing_dataclass.observations (same IcechunkUSGSReader
    # path that DDR's training loop uses).
    # They're shape (n_gauges, RHO) in the dataset. Trim outer day: obs[:, 1:-1]
    obs_ds = routing_dataclass.observations
    obs_arr = obs_ds.streamflow.values.astype(np.float32)  # (n_gauges, RHO)
    obs_trimmed = obs_arr[:, 1:-1]  # (n_gauges, RHO-2)

    # NaN-gauge filter (train.py:84-86).
    nan_mask = np.isnan(obs_trimmed).any(axis=1)
    np_nan_mask = nan_mask
    if np_nan_mask.all():
        print("WARNING: all gauges have NaN observations — fixture may not be valid.")

    filtered_obs = torch.tensor(obs_trimmed[~np_nan_mask], dtype=torch.float32)  # (kept, RHO-2)
    filtered_pred = daily_q[~np_nan_mask]  # (kept, n_days)

    warmup = cfg.experiment.warmup
    pred_pw = filtered_pred.transpose(0, 1)[warmup:].unsqueeze(2)  # (T-warmup, kept, 1)
    obs_pw  = filtered_obs.transpose(0, 1)[warmup:].unsqueeze(2)   # (T-warmup, kept, 1)

    loss = torch.nn.functional.l1_loss(input=pred_pw, target=obs_pw)
    print(f"  L1 loss = {loss.item():.6f}")

    loss.backward()
    clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  post-clip grad norm = {clip_norm.item():.6f}")

    # obs_post_warmup and pred_post_warmup for the fixture (first kept gauge = our gauge).
    obs_post_warmup = filtered_obs[0, warmup:].detach().numpy().astype(np.float32)
    pred_post_warmup = filtered_pred[0, warmup:].detach().numpy().astype(np.float32)

    loss_file = f"loss_and_grads_{COMID}.npz"
    grad_payload: dict[str, np.ndarray] = {
        "loss": np.float32(loss.item()),
        "post_clip_grad_norm": np.float32(clip_norm.item()),
        # Linear weights — PyTorch layout [out, in]
        "grad_input_weight":  model.input.weight.grad.detach().cpu().numpy().astype(np.float32),
        "grad_input_bias":    model.input.bias.grad.detach().cpu().numpy().astype(np.float32),
        "grad_output_weight": model.output.weight.grad.detach().cpu().numpy().astype(np.float32),
        "grad_output_bias":   model.output.bias.grad.detach().cpu().numpy().astype(np.float32),
        "obs_post_warmup":    obs_post_warmup,
        "pred_post_warmup":   pred_post_warmup,
    }
    for b in range(NUM_HIDDEN_LAYERS):
        inner = model.layers[b].act_fun[0]
        grad_payload[f"grad_block_{b}_coef"]       = inner.coef.grad.detach().cpu().numpy().astype(np.float32)
        grad_payload[f"grad_block_{b}_scale_base"] = inner.scale_base.grad.detach().cpu().numpy().astype(np.float32)
        grad_payload[f"grad_block_{b}_scale_sp"]   = inner.scale_sp.grad.detach().cpu().numpy().astype(np.float32)

    np.savez(OUT_DIR / loss_file, **grad_payload)
    print(f"  wrote {loss_file}: {(OUT_DIR / loss_file).stat().st_size} bytes")

    # -------------------------------------------------------------------
    # Stage 6: Adam step
    # -------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    # Grads from loss.backward() above are still on the params.
    optimizer.step()

    adam_payload: dict[str, np.ndarray] = {
        # Post-step params — PyTorch [out, in] for Linear weights
        "input_weight":  model.input.weight.detach().cpu().numpy().astype(np.float32),
        "input_bias":    model.input.bias.detach().cpu().numpy().astype(np.float32),
        "output_weight": model.output.weight.detach().cpu().numpy().astype(np.float32),
        "output_bias":   model.output.bias.detach().cpu().numpy().astype(np.float32),
    }
    for b in range(NUM_HIDDEN_LAYERS):
        inner = model.layers[b].act_fun[0]
        pfx = f"block_{b}"
        adam_payload[f"{pfx}_coef"]       = inner.coef.detach().cpu().numpy().astype(np.float32)
        adam_payload[f"{pfx}_scale_base"] = inner.scale_base.detach().cpu().numpy().astype(np.float32)
        adam_payload[f"{pfx}_scale_sp"]   = inner.scale_sp.detach().cpu().numpy().astype(np.float32)

    # Adam state (first and second moments)
    for param, name in [
        (model.input.weight,  "input_weight"),
        (model.input.bias,    "input_bias"),
        (model.output.weight, "output_weight"),
        (model.output.bias,   "output_bias"),
    ]:
        state = optimizer.state[param]
        adam_payload[f"moment1_{name}"] = state["exp_avg"].detach().cpu().numpy().astype(np.float32)
        adam_payload[f"moment2_{name}"] = state["exp_avg_sq"].detach().cpu().numpy().astype(np.float32)

    for b in range(NUM_HIDDEN_LAYERS):
        inner = model.layers[b].act_fun[0]
        for attr, aname in [
            (inner.coef, f"block_{b}_coef"),
            (inner.scale_base, f"block_{b}_scale_base"),
            (inner.scale_sp, f"block_{b}_scale_sp"),
        ]:
            state = optimizer.state[attr]
            adam_payload[f"moment1_{aname}"] = state["exp_avg"].detach().cpu().numpy().astype(np.float32)
            adam_payload[f"moment2_{aname}"] = state["exp_avg_sq"].detach().cpu().numpy().astype(np.float32)

    adam_file = f"adam_step_{COMID}.npz"
    np.savez(OUT_DIR / adam_file, **adam_payload)
    print(f"  wrote {adam_file}: {(OUT_DIR / adam_file).stat().st_size} bytes")

    # -------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------
    try:
        ddr_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(DDR_ROOT), text=True
        ).strip()
    except Exception:
        ddr_commit = "unknown"

    try:
        sys.path.insert(0, str(DDR_ROOT / "src"))
        from ddr._version import __version__ as ddr_version  # type: ignore
    except Exception:
        ddr_version = "unknown"

    fixture_files = [
        subgraph_file,
        hotstart_file,
        mc_forward_file,
        daily_q_file,
        loss_file,
        adam_file,
    ]

    manifest = {
        "version": 1,
        "fixture_kan_seed": KAN_SEED,
        "fixture_kan_source": "tests/fixtures/kan_head_init_seed42.npz",
        "gauge": {
            "staid": STAID,
            "comid": COMID,
            "drain_sqkm": DRAIN_SQKM,
            "n_reaches": int(n_reaches),
            "gage_idx_compressed": int(gage_idx_compressed),
        },
        "time_window": {
            "start": START_TIME,
            # rho_hours_actual: DDR's Dates class produces inclusive='left' hourly range
            # from daily[0] to daily[-1], giving (rho-1)*24 hours = 2136 for rho=90.
            "rho_hours_actual": int(q_prime_full.shape[0]),
            "rho_days": RHO,
            "warmup_days": WARMUP_DAYS,
            "tau": int(tau),
            "n_days_post_trim": int(n_days),
            "n_days_post_warmup": int(n_days - WARMUP_DAYS),
        },
        "files": fixture_files,
        "ddr_git_commit": ddr_commit,
        "ddr_version": ddr_version,
    }

    manifest_path = OUT_DIR / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
