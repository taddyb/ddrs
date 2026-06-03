"""End-to-end smoke test for the ddrs_py bridge.

Run from the ddrs-py directory with a venv that has ddrs_py installed:
    uv run pytest tests/smoke.py -v
"""

import os

import numpy as np
import pytest

import ddrs_py


REPO_ROOT = "/home/tbindas/projects/ddrs"
CONFIG_PATH = f"{REPO_ROOT}/config/merit_training.yaml"


def test_parameter_bounds_from_merit_training_yaml():
    bounds = ddrs_py.parameter_bounds(CONFIG_PATH)
    assert set(bounds.keys()) == {"n", "q_spatial", "p_spatial"}

    n_bounds, n_log = bounds["n"]
    assert n_bounds == (pytest.approx(0.015), pytest.approx(0.25))
    assert n_log is True  # log_space_parameters lists "n" in this yaml

    q_bounds, q_log = bounds["q_spatial"]
    assert q_bounds == (pytest.approx(0.0), pytest.approx(1.0))
    assert q_log is False

    p_bounds, p_log = bounds["p_spatial"]
    assert p_bounds == (pytest.approx(1.0), pytest.approx(200.0))
    assert p_log is False


def test_denormalize_linear_matches_formula():
    # bounds (1.0, 200.0), log_space=False  →  v · (200 - 1) + 1
    values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    result = ddrs_py.denormalize(values, (1.0, 200.0), False)
    expected = np.array([1.0, 100.5, 200.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_denormalize_log_matches_formula():
    # bounds (0.015, 0.25), log_space=True  →  exp(v · (ln(0.25) - ln(0.015+1e-6)) + ln(0.015+1e-6))
    values = np.array([0.0, 1.0], dtype=np.float32)
    result = ddrs_py.denormalize(values, (0.015, 0.25), True)
    lo_eff = np.log(0.015 + 1e-6)
    expected = np.array([np.exp(lo_eff), 0.25], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_denormalize_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        ddrs_py.denormalize(np.zeros((2, 2), dtype=np.float32), (0.0, 1.0), False)


def test_denormalize_rejects_log_space_with_nonpositive_hi():
    with pytest.raises(ValueError, match="hi > 0"):
        ddrs_py.denormalize(
            np.array([0.5], dtype=np.float32),
            (0.015, 0.0),
            True,
        )


CHECKPOINT_DIR = f"{REPO_ROOT}/output/saved_models"


def _first_available_checkpoint() -> str | None:
    """Return base path (no .mpk extension) of any checkpoint in the saved_models dir."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    mpks = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".mpk"))
    if not mpks:
        return None
    return f"{CHECKPOINT_DIR}/{mpks[0][:-len('.mpk')]}"


def test_load_kan_head_returns_pykanhead_with_param_names():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip(f"no checkpoints in {CHECKPOINT_DIR}; train one first")

    model = ddrs_py.load_kan_head(checkpoint=ckpt, config_path=CONFIG_PATH)
    assert model.learnable_parameters == ["n", "q_spatial", "p_spatial"]
    assert model.input_var_names_len == 10  # matches merit_training.yaml


def test_forward_returns_param_dict_in_unit_interval():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint")
    model = ddrs_py.load_kan_head(checkpoint=ckpt, config_path=CONFIG_PATH)

    rng = np.random.default_rng(seed=0)
    attrs = rng.standard_normal((7, model.input_var_names_len)).astype(np.float32)
    out = model.forward(attrs)

    assert set(out.keys()) == {"n", "q_spatial", "p_spatial"}
    for key, arr in out.items():
        assert arr.dtype == np.float32, f"{key} dtype is {arr.dtype}"
        assert arr.shape == (7,), f"{key} shape is {arr.shape}"
        assert np.all((arr >= 0.0) & (arr <= 1.0)), f"{key} values out of [0,1]: min={arr.min()} max={arr.max()}"


def test_forward_rejects_wrong_feature_count():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint")
    model = ddrs_py.load_kan_head(checkpoint=ckpt, config_path=CONFIG_PATH)
    bad = np.zeros((4, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatches"):
        model.forward(bad)


DDR_DATA = "/home/tbindas/projects/ddr/data"
ATTRS_NC = f"{DDR_DATA}/merit_global_attributes_v2.nc"
CONUS_ZARR = f"{DDR_DATA}/merit_conus_adjacency.zarr"


def test_run_inference_over_conus_returns_per_comid_params():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint")
    if not os.path.exists(ATTRS_NC) or not os.path.exists(CONUS_ZARR):
        pytest.skip(f"MERIT data not present at {DDR_DATA}")

    result = ddrs_py.run_inference_over_conus(
        attrs_nc=ATTRS_NC,
        conus_adjacency_zarr=CONUS_ZARR,
        checkpoint=ckpt,
        config_path=CONFIG_PATH,
    )

    assert set(result.keys()) == {"comid", "n", "q_spatial", "p_spatial"}
    n_reaches = result["comid"].shape[0]
    assert n_reaches > 100_000, f"expected ~346k MERIT reaches, got {n_reaches}"

    # COMID column is int64.
    assert result["comid"].dtype == np.int64

    # Parameter columns are float32, same length as comid, physical-units range.
    for key in ("n", "q_spatial", "p_spatial"):
        arr = result[key]
        assert arr.dtype == np.float32
        assert arr.shape == (n_reaches,)
        assert np.all(np.isfinite(arr)), f"{key} has non-finite values"

    # Physical bounds sanity per merit_training.yaml's parameter_ranges:
    #   n in [0.015, 0.25] (log space → exp() output strictly positive)
    #   q_spatial in [0.0, 1.0]
    #   p_spatial in [1.0, 200.0]
    # Log-space n is bounded below by lo + 1e-6 ≈ 0.015, not just > 0.
    assert result["n"].min() >= 0.015 - 1e-3 and result["n"].max() <= 0.25 + 1e-3
    assert result["q_spatial"].min() >= 0.0 and result["q_spatial"].max() <= 1.0 + 1e-5
    assert result["p_spatial"].min() >= 1.0 - 1e-3 and result["p_spatial"].max() <= 200.0 + 1e-3
