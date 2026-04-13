"""Tests for STL runtime monitoring."""

import numpy as np

from vla_edge.validate.stl_monitor import STLActionMonitor


class TestSTLActionMonitor:
    def test_clean_actions_nonnegative_robustness(self):
        monitor = STLActionMonitor(n_dims=7, v_max=0.1, clip_mag_threshold=0.5)
        action = np.zeros(7, dtype=np.float32)
        for _ in range(20):
            report = monitor.update(action, action)  # no clipping
        assert report.min_robustness >= 0  # smoothness_trend can be 0 for constant
        assert len(report.violated_specs) == 0

    def test_violated_action_negative_robustness(self):
        monitor = STLActionMonitor(n_dims=7, v_max=0.1, clip_mag_threshold=0.5)
        raw = np.ones(7, dtype=np.float32) * 2.0
        clipped = np.ones(7, dtype=np.float32) * 1.0  # clipped by 1.0
        report = monitor.update(raw, clipped)
        assert report.robustness["bounds_health"] < 0  # 0.5 - 1.0 = -0.5

    def test_velocity_violation(self):
        monitor = STLActionMonitor(n_dims=7, v_max=0.1)
        a1 = np.zeros(7, dtype=np.float32)
        a2 = np.ones(7, dtype=np.float32) * 0.5  # big jump
        monitor.update(a1, a1)
        report = monitor.update(a2, a2)
        assert report.robustness["velocity_health"] < 0  # 0.1 - 0.5 = -0.4

    def test_recovery_spec(self):
        monitor = STLActionMonitor(n_dims=7, v_max=0.1, recovery_window=5)
        raw_bad = np.ones(7, dtype=np.float32) * 2.0
        clipped_bad = np.ones(7, dtype=np.float32)
        clean = np.zeros(7, dtype=np.float32)

        # Violation
        monitor.update(raw_bad, clipped_bad)
        # Recovery
        for _ in range(5):
            report = monitor.update(clean, clean)
        assert report.robustness["recovery"] > 0

    def test_sustained_violation_detected(self):
        monitor = STLActionMonitor(n_dims=7, sustained_window=10)
        raw_bad = np.ones(7, dtype=np.float32) * 2.0
        clipped = np.ones(7, dtype=np.float32)
        for _ in range(15):
            report = monitor.update(raw_bad, clipped)
        # 100% violation rate in window -> sustained_health = 0.5 - 1.0 = -0.5
        assert report.robustness["sustained_health"] < 0

    def test_summary(self):
        monitor = STLActionMonitor(n_dims=7)
        action = np.zeros(7, dtype=np.float32)
        for _ in range(20):
            monitor.update(action, action)
        summary = monitor.get_summary()
        assert summary["steps"] == 20
        assert "bounds_health" in summary["specs"]

    def test_reset(self):
        monitor = STLActionMonitor(n_dims=7)
        action = np.zeros(7, dtype=np.float32)
        for _ in range(10):
            monitor.update(action, action)
        monitor.reset()
        assert monitor._step == 0
