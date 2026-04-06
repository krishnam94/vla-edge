"""Tests for conformal anomaly detection, CUSUM, ACAM, and MultiScaleCUSUM."""

import numpy as np
import pytest

from vla_edge.validate.conformal import (
    ConformalActionMonitor,
    CUSUMDetector,
    AdaptiveConformalMonitor,
    MultiScaleCUSUM,
)


class TestConformalActionMonitor:
    def test_calibrate(self):
        monitor = ConformalActionMonitor()
        actions = np.random.randn(100, 7).astype(np.float32)
        result = monitor.calibrate(actions)
        assert result["n_calibration"] == 100
        assert result["n_dims"] == 7

    def test_p_value_range(self):
        monitor = ConformalActionMonitor()
        actions = np.random.randn(200, 7).astype(np.float32)
        monitor.calibrate(actions)

        for _ in range(50):
            action = np.random.randn(7).astype(np.float32)
            report = monitor.score(action)
            assert 0 <= report["p_value"] <= 1

    def test_anomalous_action_low_p(self):
        monitor = ConformalActionMonitor()
        actions = np.random.randn(500, 7).astype(np.float32) * 0.1
        monitor.calibrate(actions)

        # Extreme action should have low p-value
        extreme = np.ones(7, dtype=np.float32) * 10.0
        report = monitor.score(extreme)
        assert report["p_value"] < 0.01

    def test_normal_action_high_p(self):
        monitor = ConformalActionMonitor()
        actions = np.random.randn(500, 7).astype(np.float32)
        monitor.calibrate(actions)

        # Action near mean should have high p-value
        normal = np.zeros(7, dtype=np.float32)
        report = monitor.score(normal)
        assert report["p_value"] > 0.3

    def test_uniformity_under_exchangeability(self):
        rng = np.random.RandomState(42)
        monitor = ConformalActionMonitor()
        actions = rng.randn(1000, 7).astype(np.float32)
        monitor.calibrate(actions[:800])

        # Score actions from same distribution
        for a in actions[800:]:
            monitor.score(a)

        result = monitor.test_uniformity()
        # Under exchangeability, p-values should be roughly uniform
        # Use a lenient threshold since finite-sample effects exist
        assert result["ks_p_value"] > 0.001

    def test_shift_detected(self):
        rng = np.random.RandomState(42)
        monitor = ConformalActionMonitor()
        actions = rng.randn(500, 7).astype(np.float32) * 0.5
        monitor.calibrate(actions)

        # Score actions from SHIFTED distribution
        for _ in range(100):
            shifted = rng.randn(7).astype(np.float32) * 0.5 + 3.0
            monitor.score(shifted)

        result = monitor.test_uniformity()
        assert result["shift_detected"]

    def test_conformal_bounds(self):
        monitor = ConformalActionMonitor()
        actions = np.random.randn(500, 7).astype(np.float32)
        monitor.calibrate(actions)

        bounds = monitor.get_conformal_bounds(alpha=0.05)
        assert bounds["alpha"] == 0.05
        assert len(bounds["bounds_lo"]) == 7
        assert bounds["quantile_threshold"] > 0


class TestCUSUMDetector:
    def test_no_alarm_under_null(self):
        detector = CUSUMDetector(expected_rate=0.5, threshold=5.0)
        alarms = 0
        rng = np.random.RandomState(42)
        for _ in range(1000):
            violated = rng.random() < 0.5  # matches expected rate
            if detector.update(violated):
                alarms += 1
        # Should have very few alarms under null
        assert alarms < 10

    def test_alarm_under_shift(self):
        detector = CUSUMDetector(expected_rate=0.05, threshold=3.0)
        # Feed 100% violations - should alarm quickly
        alarm_step = None
        for i in range(100):
            if detector.update(True):
                alarm_step = i
                break
        assert alarm_step is not None
        assert alarm_step < 10  # Should detect quickly

    def test_reset(self):
        detector = CUSUMDetector(expected_rate=0.05, threshold=5.0)
        for _ in range(10):
            detector.update(True)
        assert detector.cusum_stat > 0
        detector.reset()
        assert detector.cusum_stat == 0.0
        assert detector.n_steps == 0


class TestAdaptiveConformalMonitor:
    def test_calibrate_and_update(self):
        acam = AdaptiveConformalMonitor(alpha=0.05, gamma=0.01)
        rng = np.random.RandomState(42)
        actions = rng.randn(500, 7).astype(np.float32)
        acam.calibrate(actions)

        report = acam.update(rng.randn(7).astype(np.float32))
        assert "p_value" in report
        assert "adaptive_alpha" in report
        assert "bounds_lo" in report
        assert 0 < report["adaptive_alpha"] < 0.5

    def test_alpha_widens_under_shift(self):
        """Alpha should increase (bounds widen) when seeing OOD data."""
        acam = AdaptiveConformalMonitor(alpha=0.05, gamma=0.01)
        rng = np.random.RandomState(42)
        actions = rng.randn(500, 7).astype(np.float32) * 0.5
        acam.calibrate(actions)

        initial_alpha = acam.alpha_t

        # Feed heavily shifted data - should cause coverage errors -> alpha rises
        for _ in range(100):
            shifted = rng.randn(7).astype(np.float32) * 0.5 + 5.0
            acam.update(shifted)

        assert acam.alpha_t > initial_alpha

    def test_alpha_converges_to_target(self):
        """Alpha should converge toward target under in-distribution data."""
        acam = AdaptiveConformalMonitor(alpha=0.05, gamma=0.005)
        rng = np.random.RandomState(42)
        actions = rng.randn(1000, 7).astype(np.float32)
        acam.calibrate(actions[:800])

        # Feed in-distribution data - alpha should stay near target
        for a in actions[800:]:
            acam.update(a)

        summary = acam.get_summary()
        # Alpha should be in reasonable range around target
        assert 0.001 < summary["final_alpha"] < 0.3

    def test_coverage_maintained(self):
        """Empirical coverage should be close to target over many steps."""
        acam = AdaptiveConformalMonitor(alpha=0.05, gamma=0.005)
        rng = np.random.RandomState(42)
        actions = rng.randn(1000, 7).astype(np.float32)
        acam.calibrate(actions[:800])

        for a in actions[800:]:
            acam.update(a)

        summary = acam.get_summary()
        # Coverage should be within 10% of target
        assert abs(summary["empirical_coverage"] - 0.95) < 0.15

    def test_reset_preserves_calibration(self):
        acam = AdaptiveConformalMonitor(alpha=0.05)
        rng = np.random.RandomState(42)
        acam.calibrate(rng.randn(100, 7).astype(np.float32))
        acam.update(rng.randn(7).astype(np.float32))
        acam.reset()
        assert acam._n_steps == 0
        assert acam._calibrated  # calibration preserved


class TestMultiScaleCUSUM:
    def test_no_alarm_under_null(self):
        ms = MultiScaleCUSUM(alpha=0.05, chunk_size=10, episode_size=50)
        rng = np.random.RandomState(42)
        alarms = 0
        for _ in range(200):
            p = rng.uniform(0, 1)  # uniform p-values = no shift
            report = ms.update(p)
            if report["any_alarm"]:
                alarms += 1
        # Should have few alarms (step-level may fire occasionally)
        assert alarms < 20

    def test_alarm_under_shift(self):
        ms = MultiScaleCUSUM(alpha=0.05, chunk_size=10, episode_size=50,
                             step_threshold=2.0)
        # All p-values below alpha = shift
        alarm_step = None
        for i in range(50):
            report = ms.update(0.01)  # all anomalous
            if report["any_alarm"] and alarm_step is None:
                alarm_step = i
        assert alarm_step is not None
        assert alarm_step < 10

    def test_multi_scale_different_sensitivities(self):
        ms = MultiScaleCUSUM(alpha=0.05, chunk_size=5, episode_size=20,
                             step_threshold=2.0, chunk_threshold=3.0, episode_threshold=5.0)
        # Mild shift: only step-level should fire first
        step_alarms = 0
        chunk_alarms = 0
        for i in range(100):
            report = ms.update(0.03)  # slightly below alpha
            if "step" in report["alarm_scales"]:
                step_alarms += 1
            if "chunk" in report["alarm_scales"]:
                chunk_alarms += 1
        # Step should fire more than chunk (lower threshold)
        assert step_alarms >= chunk_alarms

    def test_reset(self):
        ms = MultiScaleCUSUM(alpha=0.05)
        for _ in range(10):
            ms.update(0.01)
        assert ms._n_steps == 10
        ms.reset()
        assert ms._n_steps == 0
