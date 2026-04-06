"""Tests for extended violation types: stall, jerk, gripper oscillation."""

import numpy as np
import pytest

from vla_edge.validate.violations import (
    StallDetector,
    JerkMonitor,
    GripperOscillationDetector,
)


class TestStallDetector:
    def test_no_stall_on_movement(self):
        detector = StallDetector(movement_threshold=0.5, stall_window=5)
        for i in range(20):
            action = np.array([float(i), 0.0], dtype=np.float32)
            report = detector.update(action)
        assert not report["is_stalled"]

    def test_stall_on_no_movement(self):
        detector = StallDetector(movement_threshold=0.5, stall_window=5)
        action = np.array([1.0, 1.0], dtype=np.float32)
        for _ in range(20):
            report = detector.update(action)  # same action every step
        assert report["is_stalled"]
        assert report["stall_duration"] >= 5

    def test_stall_recovery(self):
        detector = StallDetector(movement_threshold=0.5, stall_window=3)
        # Stall
        for _ in range(10):
            detector.update(np.zeros(2, dtype=np.float32))
        # Move
        report = detector.update(np.array([10.0, 10.0], dtype=np.float32))
        assert not report["is_stalled"]

    def test_summary(self):
        detector = StallDetector(movement_threshold=0.5, stall_window=3)
        for _ in range(10):
            detector.update(np.zeros(2, dtype=np.float32))
        s = detector.get_summary()
        assert s["total_stall_steps"] > 0
        assert s["stall_rate"] > 0


class TestJerkMonitor:
    def test_smooth_trajectory_low_jerk(self):
        monitor = JerkMonitor(jerk_limit=10.0)
        # Linear trajectory: constant velocity, zero acceleration, zero jerk
        for i in range(20):
            action = np.array([float(i) * 0.1, 0.0], dtype=np.float32)
            report = monitor.update(action)
        assert not report["jerk_violation"]
        s = monitor.get_summary()
        assert s["mean_jerk"] < 0.01

    def test_jerky_trajectory_high_jerk(self):
        monitor = JerkMonitor(jerk_limit=1.0)
        # Alternating direction: high jerk
        for i in range(20):
            sign = 1.0 if i % 2 == 0 else -1.0
            action = np.array([sign * 5.0, 0.0], dtype=np.float32)
            report = monitor.update(action)
        assert report["jerk_violation"] or monitor._violations > 0

    def test_chunk_boundary_detection(self):
        monitor = JerkMonitor(jerk_limit=100.0, chunk_size=5)
        rng = np.random.RandomState(42)
        for i in range(25):
            action = rng.randn(7).astype(np.float32) * 0.1
            report = monitor.update(action)
        s = monitor.get_summary()
        # Should have detected some chunk boundaries
        assert "chunk_boundary_mean_jerk" in s

    def test_summary(self):
        monitor = JerkMonitor(jerk_limit=10.0)
        rng = np.random.RandomState(42)
        for _ in range(20):
            monitor.update(rng.randn(7).astype(np.float32))
        s = monitor.get_summary()
        assert "jerk_rms" in s
        assert s["n_steps"] == 20


class TestGripperOscillationDetector:
    def test_no_oscillation_steady_grip(self):
        detector = GripperOscillationDetector(gripper_dim=-1, window_size=10, max_flips=4)
        for _ in range(20):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            report = detector.update(action)
        assert not report["oscillating"]
        assert report["flips"] == 0

    def test_oscillation_detected(self):
        detector = GripperOscillationDetector(gripper_dim=-1, window_size=10, max_flips=4)
        for i in range(20):
            gripper = 1.0 if i % 2 == 0 else -1.0
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float32)
            report = detector.update(action)
        assert report["oscillating"]
        assert report["flips"] >= 4

    def test_summary(self):
        detector = GripperOscillationDetector(gripper_dim=-1, window_size=5, max_flips=2)
        for i in range(15):
            gripper = 1.0 if i % 2 == 0 else -1.0
            action = np.array([0.0, gripper], dtype=np.float32)
            detector.update(action)
        s = detector.get_summary()
        assert s["oscillation_rate"] > 0
