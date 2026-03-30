"""Tests for action safety validation."""

import numpy as np
import pytest

from vla_edge.validate.safety import SafetyConfig, clip_actions, validate_actions


@pytest.fixture
def basic_config():
    """Safety config with simple bounds."""
    return SafetyConfig(
        action_bounds=np.array([[-1, 1]] * 7, dtype=np.float32),
        max_velocity=np.full(7, 0.1, dtype=np.float32),
    )


def test_safe_actions_pass(basic_config):
    """Actions within bounds should pass validation."""
    actions = np.zeros((5, 7), dtype=np.float32)
    result = validate_actions(actions, basic_config)
    assert result.is_safe
    assert len(result.violations) == 0
    assert result.total_actions == 5


def test_bounds_violation_detected(basic_config):
    """Actions outside bounds should be flagged as critical."""
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[1, 0] = 2.0  # Exceeds max bound of 1.0

    result = validate_actions(actions, basic_config)
    assert not result.is_safe
    assert len(result.violations) >= 1

    bounds_violations = [v for v in result.violations if v.violation_type == "bounds"]
    assert len(bounds_violations) >= 1
    assert bounds_violations[0].joint == 0
    assert bounds_violations[0].severity == "critical"


def test_velocity_violation_detected(basic_config):
    """Large velocity changes should be flagged."""
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[1, 0] = 0.5  # Jump of 0.5 exceeds max_velocity of 0.1

    result = validate_actions(actions, basic_config)
    vel_violations = [v for v in result.violations if v.violation_type == "velocity"]
    assert len(vel_violations) >= 1


def test_workspace_violation_detected():
    """End-effector outside workspace should be flagged."""
    config = SafetyConfig(
        workspace_bounds=np.array([[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]], dtype=np.float32),
    )
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[0, 2] = -0.1  # z below workspace floor (0.0)

    result = validate_actions(actions, config)
    ws_violations = [v for v in result.violations if v.violation_type == "workspace"]
    assert len(ws_violations) >= 1


def test_clip_actions_enforces_bounds():
    """clip_actions should clip to configured bounds."""
    config = SafetyConfig(
        action_bounds=np.array([[-1, 1]] * 7, dtype=np.float32),
    )
    actions = np.array([[2.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    clipped = clip_actions(actions, config)

    assert clipped[0, 0] == 1.0
    assert clipped[0, 1] == -1.0
    assert clipped[0, 2] == 0.5  # Unchanged, within bounds


def test_single_action_input():
    """validate_actions should handle a single 1D action."""
    config = SafetyConfig(
        action_bounds=np.array([[-1, 1]] * 7, dtype=np.float32),
    )
    action = np.zeros(7, dtype=np.float32)
    result = validate_actions(action, config)
    assert result.is_safe
    assert result.total_actions == 1


def test_empty_config_passes_everything():
    """An empty config should pass any actions (no checks configured)."""
    config = SafetyConfig()
    actions = np.random.randn(10, 7).astype(np.float32) * 100
    result = validate_actions(actions, config)
    assert result.is_safe


def test_violation_rate():
    """violation_rate should be clipped_actions / total_actions."""
    config = SafetyConfig(
        action_bounds=np.array([[-1, 1]] * 7, dtype=np.float32),
    )
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[0, 0] = 5.0  # Violation
    actions[2, 0] = 5.0  # Violation

    result = validate_actions(actions, config)
    assert result.violation_rate == pytest.approx(0.5, abs=0.01)
