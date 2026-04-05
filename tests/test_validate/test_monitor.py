"""Tests for action health monitoring."""

import numpy as np
import pytest

from vla_edge.validate.monitor import ActionHealthMonitor, ActionHealthReport


class TestActionHealthMonitor:
    def test_clip_magnitude_zero_on_clean(self):
        """Clean actions produce zero clip magnitude."""
        monitor = ActionHealthMonitor(action_dim=7)
        raw = np.array([0.5] * 7, dtype=np.float32)
        clipped = raw.copy()  # no clipping needed
        report = monitor.update(raw, clipped, violated=False)
        assert np.allclose(report.clip_magnitude, 0.0)
        assert report.health_score < 0.1

    def test_clip_magnitude_nonzero_on_violation(self):
        """Clipped actions produce nonzero clip magnitude."""
        monitor = ActionHealthMonitor(action_dim=7)
        raw = np.array([1.5, 0.5, -2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        clipped = np.clip(raw, -1, 1)
        report = monitor.update(raw, clipped, violated=True)
        assert report.clip_magnitude[0] == pytest.approx(0.5, abs=0.01)
        assert report.clip_magnitude[2] == pytest.approx(1.0, abs=0.01)
        assert report.health_score > 0.5

    def test_ewma_trend_rises_with_violations(self):
        """EWMA trend rises when violations accumulate."""
        monitor = ActionHealthMonitor(action_dim=3, ewma_alpha=0.3)
        action = np.zeros(3, dtype=np.float32)

        # 10 clean steps
        for _ in range(10):
            monitor.update(action, action, violated=False)
        trend_clean = monitor._ewma_violation_rate

        # 10 violation steps
        for _ in range(10):
            monitor.update(action, action, violated=True)
        trend_dirty = monitor._ewma_violation_rate

        assert trend_dirty > trend_clean
        assert trend_dirty > 0.5

    def test_crest_factor_high_for_spiky(self):
        """Spiky actions produce high crest factor."""
        monitor = ActionHealthMonitor(action_dim=2, window_size=10)

        # Mostly zero with one spike
        for i in range(9):
            action = np.zeros(2, dtype=np.float32)
            monitor.update(action, action)

        spike = np.array([5.0, 0.0], dtype=np.float32)
        report = monitor.update(spike, spike)

        assert report.crest_factor is not None
        assert report.crest_factor[0] > 2.0  # high crest = spiky

    def test_crest_factor_low_for_constant(self):
        """Constant actions produce crest factor near 1."""
        monitor = ActionHealthMonitor(action_dim=2, window_size=10)

        for _ in range(10):
            action = np.array([1.0, 1.0], dtype=np.float32)
            report = monitor.update(action, action)

        assert report.crest_factor is not None
        np.testing.assert_allclose(report.crest_factor, [1.0, 1.0], atol=0.01)

    def test_z_scores_with_calibration(self):
        """Z-scores computed correctly against calibration."""
        cal_mean = np.array([0.0, 0.0], dtype=np.float32)
        cal_std = np.array([1.0, 0.5], dtype=np.float32)
        monitor = ActionHealthMonitor(
            action_dim=2, calibration_mean=cal_mean, calibration_std=cal_std
        )

        action = np.array([2.0, 1.5], dtype=np.float32)
        report = monitor.update(action, action)

        assert report.z_scores is not None
        assert report.z_scores[0] == pytest.approx(2.0, abs=0.01)  # 2 stds
        assert report.z_scores[1] == pytest.approx(3.0, abs=0.01)  # 3 stds

    def test_summary(self):
        """Summary contains expected keys."""
        monitor = ActionHealthMonitor(action_dim=3)
        action = np.ones(3, dtype=np.float32)
        for _ in range(5):
            monitor.update(action, action, violated=False)
        summary = monitor.get_summary()
        assert summary["steps"] == 5
        assert "mean_clip_magnitude_per_dim" in summary
        assert "final_violation_trend" in summary

    def test_reset(self):
        """Reset clears all state."""
        monitor = ActionHealthMonitor(action_dim=3)
        action = np.ones(3, dtype=np.float32)
        for _ in range(10):
            monitor.update(action, action, violated=True)
        assert monitor._step_count == 10

        monitor.reset()
        assert monitor._step_count == 0
        assert len(monitor._clip_magnitude_history) == 0
        assert monitor._ewma_violation_rate == 0.0
