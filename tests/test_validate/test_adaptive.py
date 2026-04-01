"""Tests for phase-adaptive safety contracts."""

import numpy as np

from vla_edge.validate.adaptive import adaptive_safety_contract


class MockModel:
    @adaptive_safety_contract(
        safe_bounds=[-1.5, 1.5],
        danger_bounds=[-0.5, 0.5],
        velocity_threshold=0.05,
    )
    def predict_adaptive(self, image, instruction, state=None):
        return np.array([0.3, -0.3, 0.1, 0.0, 0.0, 0.0, 0.0])

    @adaptive_safety_contract(
        safe_bounds=[-2.0, 2.0],
        danger_bounds=[-0.3, 0.3],
        velocity_threshold=0.1,
    )
    def predict_big_action(self, image, instruction, state=None):
        return np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0])


def test_first_call_uses_safe_bounds():
    """First call has no velocity history, assumes safe phase."""
    m = MockModel()
    action = m.predict_adaptive(None, "test")
    np.testing.assert_array_almost_equal(action, [0.3, -0.3, 0.1, 0, 0, 0, 0])


def test_danger_phase_clips_tighter():
    """During danger phase (low velocity), tighter bounds apply."""
    m = MockModel()
    m.predict_big_action(None, "test")
    action = m.predict_big_action(None, "test")
    assert np.all(action >= -0.3)
    assert np.all(action <= 0.3)


def test_phase_stats_tracking():
    """Phase stats should track calls."""
    m = MockModel()
    m.predict_adaptive(None, "test")
    m.predict_adaptive(None, "test")
    m.predict_adaptive(None, "test")

    stats = m.predict_adaptive._phase_stats
    assert stats.total_calls >= 3
    assert stats.safe_phase_calls + stats.danger_phase_calls == stats.total_calls


def test_adaptive_metadata():
    """Contract metadata should be accessible."""
    m = MockModel()
    contract = m.predict_adaptive._adaptive_contract
    assert contract["safe_bounds"][0] == -1.5
    assert contract["safe_bounds"][1] == 1.5
    assert contract["velocity_threshold"] == 0.05


def test_reclip_after_velocity():
    """Velocity clamping should not push action outside phase bounds."""

    @adaptive_safety_contract(
        safe_bounds=[-1.0, 1.0],
        danger_bounds=[-0.3, 0.3],
        velocity_threshold=0.1,
        danger_velocity_max=0.01,
    )
    def do_predict(image, instruction, state=None):
        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    do_predict(None, "test")  # First call
    action = do_predict(None, "test")  # Second call (danger phase)
    assert np.all(action >= -0.3)
    assert np.all(action <= 0.3)
