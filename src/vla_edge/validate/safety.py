"""Action safety validation for VLA models.

Solves the problem that OpenVLA's deploy.py returns raw actions with zero clipping.
LeRobot's EEBoundsAndSafety processor is the only existing implementation - we
provide a more comprehensive, configurable safety validation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SafetyConfig:
    """Configuration for action safety validation.

    All limits are optional. Only configured checks are enforced.
    """

    # Per-joint action bounds (min, max) - shape: (action_dim, 2)
    action_bounds: np.ndarray | None = None

    # Maximum velocity (change between consecutive actions) per joint
    max_velocity: np.ndarray | None = None

    # Maximum acceleration (change of velocity) per joint
    max_acceleration: np.ndarray | None = None

    # Workspace bounds for end-effector position (x, y, z) - shape: (3, 2)
    workspace_bounds: np.ndarray | None = None

    # Maximum end-effector speed in m/s
    max_ee_speed: float | None = None


@dataclass
class SafetyResult:
    """Result of safety validation on a sequence of actions."""

    is_safe: bool
    violations: list[SafetyViolation] = field(default_factory=list)
    total_actions: int = 0
    clipped_actions: int = 0
    max_velocity_observed: float = 0.0
    max_acceleration_observed: float = 0.0

    @property
    def violation_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.clipped_actions / self.total_actions


@dataclass
class SafetyViolation:
    """A single safety violation."""

    step: int
    violation_type: str  # "bounds", "velocity", "acceleration", "workspace", "ee_speed"
    joint: int | None = None
    value: float = 0.0
    limit: float = 0.0
    severity: str = "warning"  # "warning", "critical"


def validate_actions(
    actions: np.ndarray,
    config: SafetyConfig,
) -> SafetyResult:
    """Validate a sequence of predicted actions against safety constraints.

    Args:
        actions: Array of shape (T, action_dim) - T timesteps of actions.
        config: Safety configuration with limits.

    Returns:
        SafetyResult with is_safe flag, violations, and statistics.
    """
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]

    violations: list[SafetyViolation] = []
    clipped_steps: set[int] = set()
    max_vel = 0.0
    max_acc = 0.0

    for t in range(len(actions)):
        action = actions[t]

        # Check action bounds
        if config.action_bounds is not None:
            for j in range(min(len(action), len(config.action_bounds))):
                lo, hi = config.action_bounds[j]
                if action[j] < lo or action[j] > hi:
                    violations.append(
                        SafetyViolation(
                            step=t,
                            violation_type="bounds",
                            joint=j,
                            value=float(action[j]),
                            limit=float(lo if action[j] < lo else hi),
                            severity="critical",
                        )
                    )
                    clipped_steps.add(t)

        # Check velocity (requires t > 0)
        if t > 0 and config.max_velocity is not None:
            vel = np.abs(actions[t] - actions[t - 1])
            max_vel = max(max_vel, float(np.max(vel)))
            for j in range(min(len(vel), len(config.max_velocity))):
                if vel[j] > config.max_velocity[j]:
                    violations.append(
                        SafetyViolation(
                            step=t,
                            violation_type="velocity",
                            joint=j,
                            value=float(vel[j]),
                            limit=float(config.max_velocity[j]),
                            severity="warning",
                        )
                    )

        # Check acceleration (requires t > 1)
        if t > 1 and config.max_acceleration is not None:
            vel_curr = actions[t] - actions[t - 1]
            vel_prev = actions[t - 1] - actions[t - 2]
            acc = np.abs(vel_curr - vel_prev)
            max_acc = max(max_acc, float(np.max(acc)))
            for j in range(min(len(acc), len(config.max_acceleration))):
                if acc[j] > config.max_acceleration[j]:
                    violations.append(
                        SafetyViolation(
                            step=t,
                            violation_type="acceleration",
                            joint=j,
                            value=float(acc[j]),
                            limit=float(config.max_acceleration[j]),
                            severity="warning",
                        )
                    )

    # Check workspace bounds (assumes first 3 dims are xyz position)
    if config.workspace_bounds is not None:
        for t in range(len(actions)):
            pos = actions[t, :3]
            for dim in range(3):
                lo, hi = config.workspace_bounds[dim]
                if pos[dim] < lo or pos[dim] > hi:
                    violations.append(
                        SafetyViolation(
                            step=t,
                            violation_type="workspace",
                            joint=dim,
                            value=float(pos[dim]),
                            limit=float(lo if pos[dim] < lo else hi),
                            severity="critical",
                        )
                    )

    is_safe = all(v.severity != "critical" for v in violations)

    return SafetyResult(
        is_safe=is_safe,
        violations=violations,
        total_actions=len(actions),
        clipped_actions=len(clipped_steps),
        max_velocity_observed=max_vel,
        max_acceleration_observed=max_acc,
    )


def clip_actions(actions: np.ndarray, config: SafetyConfig) -> np.ndarray:
    """Clip actions to safety bounds. Returns a new array."""
    clipped = actions.copy()

    if config.action_bounds is not None:
        for j in range(min(clipped.shape[-1], len(config.action_bounds))):
            lo, hi = config.action_bounds[j]
            clipped[..., j] = np.clip(clipped[..., j], lo, hi)

    return clipped
