"""Tests for ProbeFlow adaptive flow matching."""

import pytest

from vla_edge.optimize.probeflow import ProbeFlowStats


def test_stats_initial():
    """Fresh stats should show zero calls."""
    stats = ProbeFlowStats(min_steps=2, max_steps_used=10)
    assert stats.total_calls == 0
    assert stats.avg_steps == 0.0
    assert stats.speedup == 1.0


def test_stats_avg_steps():
    """Average steps computed correctly."""
    stats = ProbeFlowStats(min_steps=2, max_steps_used=10)
    stats.total_calls = 4
    stats.total_steps = 12  # avg = 3
    assert stats.avg_steps == 3.0
    assert stats.speedup == pytest.approx(10 / 3, rel=0.01)


def test_stats_histogram():
    """Step histogram tracks distribution."""
    stats = ProbeFlowStats(min_steps=2, max_steps_used=10)
    stats.total_calls = 3
    stats.total_steps = 8
    stats.step_histogram = {2: 2, 4: 1}
    assert stats.step_histogram[2] == 2
    assert stats.step_histogram[4] == 1


@pytest.mark.slow
def test_probeflow_apply_to_smolvla():
    """Apply ProbeFlow to SmolVLA and verify it runs with fewer steps."""
    pytest.importorskip("lerobot")
    pytest.importorskip("torch")

    import numpy as np

    from vla_edge.models.smolvla import SmolVLAAdapter
    from vla_edge.optimize.probeflow import apply_probeflow

    adapter = SmolVLAAdapter(device="cpu", dtype="float32")
    adapter._ensure_loaded()

    # Apply ProbeFlow
    stats = apply_probeflow(adapter._policy.model, epsilon=0.15, n_min=2, n_max=10)

    # Run inference
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    action = adapter.predict(image, "pick up the block", np.zeros(6, dtype=np.float32))

    assert isinstance(action, np.ndarray)
    assert stats.total_calls >= 1
    assert stats.avg_steps <= 10  # Should use fewer than max
    assert stats.avg_steps >= 2  # At least n_min

    adapter.cleanup()
