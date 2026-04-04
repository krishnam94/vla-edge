"""Tests for soft-knee compression."""

import numpy as np
import pytest

from vla_edge.validate.softknee import (
    lookahead_softknee,
    softknee_clip,
    softknee_clip_1d,
    softknee_clip_actions,
)


class TestSoftkneeClip1D:
    """Test scalar soft-knee clip."""

    def test_within_bounds_unchanged(self):
        """Values well within bounds pass through unchanged."""
        assert softknee_clip_1d(0.0, -1.0, 1.0, 0.3) == 0.0
        assert softknee_clip_1d(0.5, -1.0, 1.0, 0.3) == 0.5
        assert softknee_clip_1d(-0.5, -1.0, 1.0, 0.3) == -0.5

    def test_hard_bounds_guaranteed(self):
        """Output never exceeds hard bounds."""
        for x in np.linspace(-5, 5, 1000):
            result = softknee_clip_1d(float(x), -1.0, 1.0, 0.3)
            assert -1.0 <= result <= 1.0, f"x={x}, result={result}"

    def test_exceeding_bounds_clipped(self):
        """Values beyond bounds are at the hard limit."""
        assert softknee_clip_1d(2.0, -1.0, 1.0, 0.3) == 1.0
        assert softknee_clip_1d(-2.0, -1.0, 1.0, 0.3) == -1.0

    def test_knee_region_smooth(self):
        """Values in knee region are compressed toward the bound."""
        # Just inside the upper knee (hi - W = 0.7)
        result = softknee_clip_1d(0.8, -1.0, 1.0, 0.3)
        # Should be between input (0.8) and hard limit (1.0), compressed
        assert 0.8 < result < 1.0  # compressed toward ceiling

    def test_zero_knee_is_hard_clip(self):
        """Knee width 0 degenerates to hard clip."""
        assert softknee_clip_1d(1.5, -1.0, 1.0, 0.0) == 1.0
        assert softknee_clip_1d(0.5, -1.0, 1.0, 0.0) == 0.5

    def test_c1_continuity_at_knee_passthrough_boundary(self):
        """Derivative matches at knee-to-passthrough boundaries."""
        W = 0.3
        lo, hi = -1.0, 1.0
        eps = 1e-4

        # At upper knee-to-passthrough boundary (hi - W = 0.7)
        x = hi - W
        dy_left = (softknee_clip_1d(x, lo, hi, W) - softknee_clip_1d(x - eps, lo, hi, W)) / eps
        dy_right = (softknee_clip_1d(x + eps, lo, hi, W) - softknee_clip_1d(x, lo, hi, W)) / eps
        assert abs(dy_left - dy_right) < 0.01, f"C1 break at upper knee: {dy_left:.4f} vs {dy_right:.4f}"

        # At lower knee-to-passthrough boundary (lo + W = -0.7)
        x = lo + W
        dy_left = (softknee_clip_1d(x, lo, hi, W) - softknee_clip_1d(x - eps, lo, hi, W)) / eps
        dy_right = (softknee_clip_1d(x + eps, lo, hi, W) - softknee_clip_1d(x, lo, hi, W)) / eps
        assert abs(dy_left - dy_right) < 0.01, f"C1 break at lower knee: {dy_left:.4f} vs {dy_right:.4f}"


class TestSoftkneeClipVectorized:
    """Test vectorized soft-knee clip."""

    def test_batch_actions(self):
        """Works on (T, action_dim) arrays."""
        actions = np.array([[0.5, -0.5, 1.5], [2.0, -2.0, 0.0]], dtype=np.float32)
        result = softknee_clip(actions, -1.0, 1.0, 0.3)
        assert result.shape == (2, 3)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_per_joint_knee_width(self):
        """Different knee widths produce different outputs."""
        actions = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        W = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        result = softknee_clip(actions, -1.0, 1.0, W)
        # All within bounds
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
        # Different knee widths produce different results
        assert not np.allclose(result[0], result[2])
        # With W=0.1, x=0.9 is at passthrough boundary -> identity
        assert result[0] == pytest.approx(0.9, abs=0.01)
        # With W=0.5, x=0.9 is in knee -> curves toward 1.0
        assert result[2] > 0.9  # knee pushes toward the limit

    def test_hard_bounds_batch(self):
        """All outputs within bounds for random batch."""
        rng = np.random.default_rng(42)
        actions = rng.normal(0, 2.0, (100, 7)).astype(np.float32)
        result = softknee_clip(actions, -1.0, 1.0, 0.3)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


class TestSoftkneeClipActions:
    """Test convenience wrapper."""

    def test_with_bounds_array(self):
        """Works with SafetyConfig-style bounds array."""
        bounds = np.array([[-1, 1]] * 7, dtype=np.float32)
        actions = np.random.randn(50, 7).astype(np.float32) * 2
        result = softknee_clip_actions(actions, bounds, knee_width=0.2)
        assert result.shape == (50, 7)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


class TestLookaheadSoftknee:
    """Test lookahead-aware soft-knee clip."""

    def test_anticipatory_compression(self):
        """Lookahead begins compressing before the violation."""
        # Action chunk that ramps up to a violation at the end
        chunk = np.zeros((10, 3), dtype=np.float32)
        for t in range(10):
            chunk[t] = t * 0.15  # ramps from 0 to 1.35

        result = lookahead_softknee(chunk, -1.0, 1.0, 0.3, decay=0.5)
        # All outputs within hard bounds
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
        # Last step (1.35) should be clipped to 1.0
        assert result[9, 0] <= 1.0

    def test_no_violation_no_change(self):
        """If no violations in chunk, minimal modification."""
        chunk = np.ones((10, 3), dtype=np.float32) * 0.3  # well within bounds
        result = lookahead_softknee(chunk, -1.0, 1.0, 0.3, decay=0.5)
        np.testing.assert_allclose(result, chunk, atol=1e-5)
