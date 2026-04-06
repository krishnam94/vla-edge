"""Tests for SafetyGuard - integrated enforcement + monitoring."""

import numpy as np
import pytest

from vla_edge.validate.guard import SafetyGuard


class TestSafetyGuard:
    def test_from_demos(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32) * 0.5
        guard = SafetyGuard.from_demos(demos, alpha=0.05)
        assert guard.bounds_lo.shape == (7,)
        assert guard.bounds_hi.shape == (7,)
        assert guard.conformal is not None

    def test_clips_out_of_bounds(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32) * 0.5
        guard = SafetyGuard.from_demos(demos)

        extreme = np.ones(7, dtype=np.float32) * 10.0
        safe = guard(extreme)
        assert safe.max() <= guard.bounds_hi.max() + 1e-6

    def test_no_clip_on_safe_action(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32) * 0.5
        guard = SafetyGuard.from_demos(demos)

        safe_action = np.zeros(7, dtype=np.float32)
        result = guard(safe_action)
        np.testing.assert_allclose(result, safe_action, atol=1e-6)

    def test_velocity_clamping(self):
        demos = np.zeros((200, 2), dtype=np.float32)
        guard = SafetyGuard.from_demos(demos)

        guard(np.array([0.0, 0.0], dtype=np.float32))
        result = guard(np.array([100.0, 100.0], dtype=np.float32))
        # Should be clamped by velocity limit
        assert result.max() < 100.0

    def test_conformal_pvalue(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32) * 0.5
        guard = SafetyGuard.from_demos(demos)

        guard(rng.randn(7).astype(np.float32) * 0.5)
        report = guard.get_report()
        assert report is not None
        assert "p_value" in report
        assert 0 <= report["p_value"] <= 1

    def test_stall_detection(self):
        demos = np.zeros((200, 3), dtype=np.float32)
        guard = SafetyGuard.from_demos(demos, mode="monitor_only")

        action = np.zeros(3, dtype=np.float32)
        for _ in range(20):
            guard(action)

        report = guard.get_report()
        assert report["is_stalled"]

    def test_summary(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32)
        guard = SafetyGuard.from_demos(demos)

        for _ in range(10):
            guard(rng.randn(7).astype(np.float32) * 2)

        summary = guard.get_summary()
        assert summary["n_steps"] == 10
        assert "stall" in summary
        assert "jerk" in summary
        assert "cusum" in summary
        assert "violation_types" in summary

    def test_wrap_decorator(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32) * 0.5
        guard = SafetyGuard.from_demos(demos)

        @guard.wrap
        def predict(x):
            return x * 10.0

        result = predict(np.ones(7))
        assert result.max() <= guard.bounds_hi.max() + 1e-6

    def test_monitor_only_mode(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32) * 0.5
        guard = SafetyGuard.from_demos(demos, mode="monitor_only")

        extreme = np.ones(7, dtype=np.float32) * 10.0
        result = guard(extreme)
        # In monitor_only mode, action should NOT be clipped
        np.testing.assert_allclose(result, extreme)

    def test_reset(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32)
        guard = SafetyGuard.from_demos(demos)

        for _ in range(10):
            guard(rng.randn(7).astype(np.float32))
        assert guard._n_steps == 10

        guard.reset()
        assert guard._n_steps == 0
        assert guard.get_report() is None

    def test_shift_detection(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(500, 7).astype(np.float32) * 0.5
        guard = SafetyGuard.from_demos(demos)

        # Feed shifted actions
        for _ in range(50):
            guard(rng.randn(7).astype(np.float32) * 0.5 + 5.0)

        shift = guard.test_shift()
        assert "shift_detected" in shift or "error" in shift
