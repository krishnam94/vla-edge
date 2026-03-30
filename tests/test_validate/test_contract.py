"""Tests for @safety_contract decorator."""

import numpy as np
import pytest

from vla_edge.validate.contract import (
    SafetyViolationError,
    clear_violation_log,
    get_violation_log,
    safety_contract,
)


class FakeModel:
    """Minimal model for testing contracts."""

    @safety_contract(action_range=[-1.0, 1.0])
    def predict_bounded(self, image, instruction, state=None):
        return np.array([0.5, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0])

    @safety_contract(action_range=[-1.0, 1.0])
    def predict_unsafe(self, image, instruction, state=None):
        return np.array([5.0, -5.0, 0.3, 0.0, 0.0, 0.0, 0.0])

    @safety_contract(action_range=[-1.0, 1.0], on_violation="raise")
    def predict_strict(self, image, instruction, state=None):
        return np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @safety_contract(joint_velocity_max=0.1)
    def predict_with_velocity(self, image, instruction, state=None):
        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @safety_contract(
        action_range=[-2.0, 2.0],
        workspace_bounds=[[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.8]],
    )
    def predict_workspace(self, image, instruction, state=None):
        return np.array([1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture(autouse=True)
def clear_log():
    """Clear violation log before each test."""
    clear_violation_log()
    yield
    clear_violation_log()


def test_safe_actions_pass_through():
    """Actions within bounds pass through unchanged."""
    m = FakeModel()
    action = m.predict_bounded(None, "test")
    np.testing.assert_array_almost_equal(action, [0.5, -0.5, 0.3, 0, 0, 0, 0])
    assert len(get_violation_log()) == 0


def test_unsafe_actions_clipped():
    """Actions outside bounds are clipped."""
    m = FakeModel()
    action = m.predict_unsafe(None, "test")
    assert action[0] == 1.0  # Clipped from 5.0
    assert action[1] == -1.0  # Clipped from -5.0
    assert action[2] == 0.3  # Unchanged
    assert len(get_violation_log()) == 1


def test_raise_mode():
    """In raise mode, violations throw SafetyViolationError."""
    m = FakeModel()
    with pytest.raises(SafetyViolationError, match="action_range"):
        m.predict_strict(None, "test")


def test_velocity_clamping():
    """Velocity is clamped between consecutive calls."""
    m = FakeModel()
    # First call: no previous action, no velocity check
    a1 = m.predict_with_velocity(None, "test")
    assert a1[0] == 0.5

    # Second call: delta = 0.5 - 0.5 = 0, within limit
    a2 = m.predict_with_velocity(None, "test")
    # Delta is 0 (same output), so no clamping
    assert a2[0] == 0.5


def test_workspace_bounds():
    """Workspace bounds clip end-effector position."""
    m = FakeModel()
    action = m.predict_workspace(None, "test")
    # dim 2 (z): -0.1 clipped to 0.0 (workspace z_min)
    assert action[2] == 0.0
    # dim 0 (x): 1.0 > 0.5, clipped
    assert action[0] == 0.5
    assert len(get_violation_log()) == 1


def test_contract_metadata_attached():
    """Contract metadata is accessible on the decorated function."""
    m = FakeModel()
    contract = m.predict_bounded._safety_contract
    assert contract["action_range"] == [-1.0, 1.0]
    assert contract["on_violation"] == "clip"


def test_violation_log_records():
    """Violation log captures details."""
    m = FakeModel()
    m.predict_unsafe(None, "test")
    log = get_violation_log()
    assert len(log) == 1
    assert "action_range" in log[0]["violations"][0]
    assert log[0]["original_range"][0] < -1.0  # Original was out of bounds


def test_multiple_violations_logged():
    """Multiple calls accumulate in log."""
    m = FakeModel()
    m.predict_unsafe(None, "test")
    m.predict_unsafe(None, "test")
    assert len(get_violation_log()) == 2


def test_clear_log():
    """Log can be cleared."""
    m = FakeModel()
    m.predict_unsafe(None, "test")
    assert len(get_violation_log()) == 1
    clear_violation_log()
    assert len(get_violation_log()) == 0
