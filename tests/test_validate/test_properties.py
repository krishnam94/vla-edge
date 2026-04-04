"""Tests for property-based testing framework."""

import numpy as np
import pytest

from vla_edge.validate.properties import (
    PropertySuiteResult,
    check_acceleration_bound,
    check_action_bounds,
    check_gripper_stability,
    check_monotonic_approach,
    check_no_direction_reversals,
    check_terminal_stability,
    check_velocity_bound,
    run_property_suite,
)


class TestActionBounds:
    def test_within_bounds_passes(self):
        actions = np.random.uniform(-0.5, 0.5, (50, 7)).astype(np.float32)
        result = check_action_bounds(actions, -1.0, 1.0)
        assert result.passed
        assert result.category == "safety"

    def test_exceeding_bounds_fails(self):
        actions = np.array([[0.5, 0.5, 1.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        result = check_action_bounds(actions, -1.0, 1.0)
        assert not result.passed
        assert result.violation_step == 0
        assert "1.5" in result.violation_detail

    def test_empty_actions_passes(self):
        result = check_action_bounds(np.array([]), -1.0, 1.0)
        assert result.passed


class TestVelocityBound:
    def test_smooth_trajectory_passes(self):
        actions = np.linspace(0, 0.5, 50).reshape(-1, 1).astype(np.float32)
        result = check_velocity_bound(actions, v_max=0.1)
        assert result.passed

    def test_sudden_jump_fails(self):
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[5] = 1.0  # sudden jump
        result = check_velocity_bound(actions, v_max=0.1)
        assert not result.passed
        assert result.violation_step == 5

    def test_single_action_passes(self):
        result = check_velocity_bound(np.array([[0.5, 0.5]]), v_max=0.1)
        assert result.passed


class TestAccelerationBound:
    def test_constant_velocity_passes(self):
        actions = np.cumsum(np.full((50, 3), 0.01), axis=0).astype(np.float32)
        result = check_acceleration_bound(actions, acc_max=0.05)
        assert result.passed

    def test_sudden_acceleration_fails(self):
        actions = np.zeros((20, 3), dtype=np.float32)
        actions[10:] = np.cumsum(np.full((10, 3), 0.1), axis=0)
        result = check_acceleration_bound(actions, acc_max=0.05)
        assert not result.passed


class TestGripperStability:
    def test_stable_gripper_passes(self):
        actions = np.zeros((50, 7), dtype=np.float32)
        actions[:, -1] = 0.0  # gripper closed
        result = check_gripper_stability(actions)
        assert result.passed

    def test_oscillating_gripper_fails(self):
        actions = np.zeros((50, 7), dtype=np.float32)
        # Very rapid oscillation: alternating +1/-1 every step
        actions[::2, -1] = 1.0
        actions[1::2, -1] = -1.0
        result = check_gripper_stability(actions, max_oscillations=2, window=10)
        assert not result.passed


class TestDirectionReversals:
    def test_smooth_motion_passes(self):
        actions = np.cumsum(np.full((20, 3), 0.05), axis=0).astype(np.float32)
        result = check_no_direction_reversals(actions)
        assert result.passed

    def test_oscillating_motion_fails(self):
        # Alternating +1/-1 every step = maximum direction changes
        actions = np.zeros((20, 3), dtype=np.float32)
        for t in range(20):
            actions[t] = (1.0 if t % 2 == 0 else -1.0)
        result = check_no_direction_reversals(actions, window=5, max_reversals=2)
        assert not result.passed


class TestTerminalStability:
    def test_converged_passes(self):
        actions = np.zeros((50, 3), dtype=np.float32)
        actions[-10:] = 0.5  # stable at 0.5
        result = check_terminal_stability(actions, k_steps=10, epsilon=0.01)
        assert result.passed

    def test_oscillating_tail_fails(self):
        actions = np.zeros((50, 3), dtype=np.float32)
        actions[-10:, 0] = np.sin(np.linspace(0, 4 * np.pi, 10)) * 0.1
        result = check_terminal_stability(actions, k_steps=10, epsilon=0.01)
        assert not result.passed


class TestMonotonicApproach:
    def test_approaching_target_passes(self):
        target = np.array([1.0, 0.0, 0.0])
        actions = np.zeros((20, 7), dtype=np.float32)
        for t in range(20):
            actions[t, 0] = t * 0.05  # moving toward target along x only
        result = check_monotonic_approach(actions, target, tolerance=0.05)
        assert result.passed

    def test_moving_away_fails(self):
        target = np.array([1.0, 0.0, 0.0])
        actions = np.zeros((20, 7), dtype=np.float32)
        for t in range(20):
            actions[t, 0] = 0.5 - t * 0.05  # moving away from target
        result = check_monotonic_approach(actions, target, tolerance=0.02)
        assert not result.passed


class TestPropertySuite:
    def test_clean_trajectory_passes_all(self):
        rng = np.random.default_rng(42)
        actions = np.cumsum(rng.normal(0, 0.005, (50, 7)), axis=0).astype(np.float32)
        actions = np.clip(actions, -0.5, 0.5)
        result = run_property_suite(actions, bounds=(-1.0, 1.0), v_max=0.1)
        assert result.passed > 0
        assert result.total_properties >= 6

    def test_bad_trajectory_fails_multiple(self):
        actions = np.random.randn(50, 7).astype(np.float32) * 2.0
        result = run_property_suite(actions, bounds=(-1.0, 1.0), v_max=0.1)
        assert result.failed > 0
        assert result.critical_failures > 0

    def test_summary_output(self):
        actions = np.zeros((20, 7), dtype=np.float32)
        result = run_property_suite(actions)
        summary = result.summary()
        assert "Properties:" in summary
        assert "PASS" in summary
