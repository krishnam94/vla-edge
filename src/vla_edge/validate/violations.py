"""Extended violation types for VLA action monitoring.

Goes beyond bounds and velocity to catch failure modes specific to
modern VLA architectures:
- Stall/freeze detection (Diffusion policies getting stuck)
- Jerk monitoring (denoising artifacts, chunk boundary discontinuities)
- Gripper oscillation (bimodal indecision near grasp decisions)

These catch issues that bounds+velocity monitoring misses, especially
on smooth Diffusion policies that produce valid but useless actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExtendedViolationReport:
    """Extended violation metrics for a single timestep."""
    # Jerk: rate of acceleration change (3rd derivative)
    jerk: np.ndarray | None = None
    jerk_violation: bool = False

    # Stall: policy producing near-zero movement
    is_stalled: bool = False
    stall_duration: int = 0  # consecutive stall steps

    # Gripper oscillation: rapid open/close cycling
    gripper_oscillating: bool = False
    gripper_flips_in_window: int = 0

    # Chunk boundary: discontinuity at action chunk transitions
    chunk_boundary_jerk: float = 0.0
    is_chunk_boundary: bool = False


class StallDetector:
    """Detect when a VLA policy produces valid but useless micro-movements.

    The #1 failure mode for Diffusion policies: outputs stay within bounds,
    velocity is low, but the robot barely moves. Traditional monitors miss this
    because no constraint is violated - the policy is just doing nothing useful.

    Detection: if the L2 norm of action change stays below threshold for
    N consecutive steps, declare a stall.

    Usage:
        detector = StallDetector(movement_threshold=1.0, stall_window=10)
        for action in policy_outputs:
            report = detector.update(action)
            if report.is_stalled:
                print(f"Policy stalled for {report.stall_duration} steps!")
    """

    def __init__(
        self,
        movement_threshold: float = 1.0,
        stall_window: int = 10,
    ):
        self.movement_threshold = movement_threshold
        self.stall_window = stall_window

        self._prev_action: np.ndarray | None = None
        self._consecutive_low_movement: int = 0
        self._total_stalls: int = 0
        self._total_stall_steps: int = 0
        self._n_steps: int = 0

    def update(self, action: np.ndarray) -> dict:
        """Update with new action.

        Args:
            action: (D,) current action

        Returns:
            Dict with is_stalled, stall_duration, movement_norm
        """
        self._n_steps += 1

        if self._prev_action is None:
            self._prev_action = action.copy()
            return {"is_stalled": False, "stall_duration": 0, "movement_norm": 0.0}

        delta = action - self._prev_action
        movement = float(np.linalg.norm(delta))

        if movement < self.movement_threshold:
            self._consecutive_low_movement += 1
        else:
            if self._consecutive_low_movement >= self.stall_window:
                self._total_stalls += 1
            self._consecutive_low_movement = 0

        is_stalled = self._consecutive_low_movement >= self.stall_window
        if is_stalled:
            self._total_stall_steps += 1

        self._prev_action = action.copy()

        return {
            "is_stalled": is_stalled,
            "stall_duration": self._consecutive_low_movement,
            "movement_norm": movement,
        }

    def get_summary(self) -> dict:
        return {
            "n_steps": self._n_steps,
            "total_stalls": self._total_stalls,
            "total_stall_steps": self._total_stall_steps,
            "stall_rate": self._total_stall_steps / max(self._n_steps, 1),
            "current_stall_duration": self._consecutive_low_movement,
        }

    def reset(self):
        self._prev_action = None
        self._consecutive_low_movement = 0
        self._total_stalls = 0
        self._total_stall_steps = 0
        self._n_steps = 0


class JerkMonitor:
    """Monitor action jerk (3rd derivative) for denoising artifacts.

    Jerk = rate of acceleration change. High jerk means the action trajectory
    has sharp corners or sudden direction changes. This catches:
    - Denoising artifacts in Diffusion policies
    - Chunk boundary discontinuities in chunked policies (ACT, Diffusion)
    - Demonstration noise amplified by the model

    Reference: Franka Panda limits jerk to 5000 rad/s^3 per joint.
    ACG paper (arXiv:2510.22201) proposes JerkRMS as a quality metric.

    Usage:
        monitor = JerkMonitor(jerk_limit=100.0)
        for action in policy_outputs:
            report = monitor.update(action)
            if report['jerk_violation']:
                print(f"Jerk spike: {report['max_jerk']:.1f}")
    """

    def __init__(
        self,
        jerk_limit: float = 100.0,
        chunk_size: int | None = None,
    ):
        self.jerk_limit = jerk_limit
        self.chunk_size = chunk_size

        self._prev_action: np.ndarray | None = None
        self._prev_velocity: np.ndarray | None = None
        self._prev_acceleration: np.ndarray | None = None
        self._n_steps: int = 0
        self._jerk_history: list[float] = []
        self._violations: int = 0
        self._chunk_boundary_jerks: list[float] = []

    def update(self, action: np.ndarray) -> dict:
        """Update with new action, compute jerk.

        Args:
            action: (D,) current action

        Returns:
            Dict with jerk, max_jerk, jerk_violation, is_chunk_boundary
        """
        self._n_steps += 1

        # Compute velocity (1st derivative)
        velocity = None
        if self._prev_action is not None:
            velocity = action - self._prev_action

        # Compute acceleration (2nd derivative)
        acceleration = None
        if velocity is not None and self._prev_velocity is not None:
            acceleration = velocity - self._prev_velocity

        # Compute jerk (3rd derivative)
        jerk = None
        jerk_violation = False
        max_jerk = 0.0
        is_chunk_boundary = False

        if acceleration is not None and self._prev_acceleration is not None:
            jerk = acceleration - self._prev_acceleration
            max_jerk = float(np.max(np.abs(jerk)))
            self._jerk_history.append(max_jerk)

            jerk_violation = max_jerk > self.jerk_limit
            if jerk_violation:
                self._violations += 1

            # Check if this is a chunk boundary
            if self.chunk_size and self._n_steps % self.chunk_size == 0:
                is_chunk_boundary = True
                self._chunk_boundary_jerks.append(max_jerk)

        # Update state
        self._prev_acceleration = acceleration
        self._prev_velocity = velocity
        self._prev_action = action.copy()

        return {
            "jerk": jerk,
            "max_jerk": max_jerk,
            "jerk_violation": jerk_violation,
            "is_chunk_boundary": is_chunk_boundary,
        }

    def get_summary(self) -> dict:
        if not self._jerk_history:
            return {"n_steps": self._n_steps, "violations": 0}

        jerks = np.array(self._jerk_history)
        result = {
            "n_steps": self._n_steps,
            "violations": self._violations,
            "violation_rate": self._violations / len(jerks),
            "mean_jerk": float(jerks.mean()),
            "max_jerk": float(jerks.max()),
            "std_jerk": float(jerks.std()),
            "jerk_rms": float(np.sqrt(np.mean(jerks**2))),
        }

        if self._chunk_boundary_jerks:
            non_boundary = [j for i, j in enumerate(self._jerk_history)
                          if not (self.chunk_size and (i + 4) % self.chunk_size == 0)]
            result["chunk_boundary_mean_jerk"] = float(np.mean(self._chunk_boundary_jerks))
            result["non_boundary_mean_jerk"] = float(np.mean(non_boundary)) if non_boundary else 0.0
            result["boundary_vs_non_ratio"] = (
                float(np.mean(self._chunk_boundary_jerks) / (np.mean(non_boundary) + 1e-8))
                if non_boundary else 0.0
            )

        return result

    def reset(self):
        self._prev_action = None
        self._prev_velocity = None
        self._prev_acceleration = None
        self._n_steps = 0
        self._jerk_history.clear()
        self._violations = 0
        self._chunk_boundary_jerks.clear()


class GripperOscillationDetector:
    """Detect rapid gripper open/close cycling.

    When a VLA policy is uncertain about grasping, it may oscillate
    the gripper between open and closed states. Individual transitions
    pass velocity checks, but the PATTERN is pathological.

    Detection: count gripper state flips (sign changes) in a sliding window.
    More than N flips in W steps = oscillation.

    Usage:
        detector = GripperOscillationDetector(gripper_dim=-1)
        for action in policy_outputs:
            report = detector.update(action)
            if report['oscillating']:
                print(f"Gripper oscillating: {report['flips']} flips in window")
    """

    def __init__(
        self,
        gripper_dim: int = -1,
        threshold: float = 0.0,
        window_size: int = 10,
        max_flips: int = 4,
    ):
        self.gripper_dim = gripper_dim
        self.threshold = threshold
        self.window_size = window_size
        self.max_flips = max_flips

        self._gripper_history: list[float] = []
        self._n_steps: int = 0
        self._total_oscillation_steps: int = 0

    def update(self, action: np.ndarray) -> dict:
        """Update with new action.

        Args:
            action: (D,) action vector. Gripper is at gripper_dim index.

        Returns:
            Dict with oscillating, flips, gripper_state
        """
        self._n_steps += 1
        gripper_val = float(action[self.gripper_dim])
        gripper_state = 1 if gripper_val > self.threshold else 0
        self._gripper_history.append(gripper_state)

        # Count flips in window
        if len(self._gripper_history) >= self.window_size:
            window = self._gripper_history[-self.window_size:]
            flips = sum(1 for i in range(1, len(window)) if window[i] != window[i-1])
        else:
            flips = 0

        oscillating = flips >= self.max_flips
        if oscillating:
            self._total_oscillation_steps += 1

        return {
            "oscillating": oscillating,
            "flips": flips,
            "gripper_state": gripper_state,
            "gripper_value": gripper_val,
        }

    def get_summary(self) -> dict:
        return {
            "n_steps": self._n_steps,
            "total_oscillation_steps": self._total_oscillation_steps,
            "oscillation_rate": self._total_oscillation_steps / max(self._n_steps, 1),
        }

    def reset(self):
        self._gripper_history.clear()
        self._n_steps = 0
        self._total_oscillation_steps = 0
