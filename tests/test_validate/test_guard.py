"""Tests for SafetyGuard - inference-time safety enforcement."""

import numpy as np
import pytest

from vla_edge.backends.base import InferenceResult
from vla_edge.validate.guard import GuardedResult, SafetyGuard
from vla_edge.validate.safety import SafetyConfig


class FakeBackend:
    """Minimal backend for testing."""

    def infer(self, model, observation):
        actions = model(observation)
        return InferenceResult(actions=actions, latency_ms=1.0, memory_peak_mb=0.0)


class SafeModel:
    """Model that returns safe actions."""

    def predict(self, image, instruction, state=None):
        return np.zeros(7, dtype=np.float32)

    def __call__(self, obs):
        return np.zeros(7, dtype=np.float32)


class UnsafeModel:
    """Model that returns out-of-bounds actions."""

    def __call__(self, obs):
        return np.array([5.0, -5.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def basic_config():
    return SafetyConfig(
        action_bounds=np.array([[-1, 1]] * 7, dtype=np.float32),
        max_velocity=np.full(7, 0.1, dtype=np.float32),
    )


@pytest.fixture
def guard(basic_config):
    return SafetyGuard(basic_config)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def obs():
    return {"image": np.zeros((224, 224, 3), dtype=np.uint8)}


def test_safe_inference(guard, backend, obs):
    """Safe model should pass with no violations."""
    result = guard.safe_infer(backend, SafeModel(), obs)
    assert isinstance(result, GuardedResult)
    assert result.is_safe
    assert result.safety.total_actions == 1
    assert len(result.safety.violations) == 0


def test_unsafe_inference_clips(guard, backend, obs):
    """Unsafe model should be clipped and violations reported."""
    result = guard.safe_infer(backend, UnsafeModel(), obs)
    assert not result.is_safe
    assert len(result.safety.violations) > 0
    # Clipped actions should be within bounds
    assert np.all(result.clipped_actions >= -1.0)
    assert np.all(result.clipped_actions <= 1.0)
    # Original should still be out of bounds
    assert np.any(np.abs(result.original_actions) > 1.0)


def test_violation_tracking(guard, backend, obs):
    """Guard should track violation statistics."""
    assert guard.violation_rate == 0.0

    guard.safe_infer(backend, SafeModel(), obs)
    assert guard.violation_rate == 0.0

    guard.safe_infer(backend, UnsafeModel(), obs)
    assert guard.violation_rate == 0.5  # 1 of 2 calls had violations

    summary = guard.summary
    assert summary["total_calls"] == 2
    assert summary["calls_with_violations"] == 1


def test_guard_reset(guard, backend, obs):
    """Reset should clear all statistics."""
    guard.safe_infer(backend, UnsafeModel(), obs)
    assert guard._total_calls == 1

    guard.reset()
    assert guard._total_calls == 0
    assert guard.violation_rate == 0.0
    assert guard.summary["total_calls"] == 0


def test_guard_summary_structure(guard, backend, obs):
    """Summary should have all expected fields."""
    guard.safe_infer(backend, SafeModel(), obs)
    summary = guard.summary

    assert "total_calls" in summary
    assert "calls_with_violations" in summary
    assert "violation_rate" in summary
    assert "total_violations" in summary
    assert "critical_violations" in summary
    assert "max_velocity_observed" in summary
