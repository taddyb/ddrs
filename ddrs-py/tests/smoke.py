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


def test_load_mlp_returns_pymlp_with_param_names():
    ckpt = _first_available_checkpoint()
    if ckpt is None:
        pytest.skip(f"no checkpoints in {CHECKPOINT_DIR}; train one first")

    model = ddrs_py.load_mlp(checkpoint=ckpt, config_path=CONFIG_PATH)
    assert model.learnable_parameters == ["n", "q_spatial", "p_spatial"]
    assert model.input_var_names_len == 10  # matches merit_training.yaml
