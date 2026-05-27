"""End-to-end smoke test for the ddrs_py bridge.

Run from the ddrs-py directory with a venv that has ddrs_py installed:
    uv run pytest tests/smoke.py -v
"""

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
