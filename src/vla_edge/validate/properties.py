"""Property-Based Testing for VLA policies.

Define behavioral properties that VLA outputs must satisfy,
test them with random scenario generation and automatic shrinking
via Hypothesis. Finds minimal failing cases for debugging.

This is the testing counterpart to SafeContract's enforcement:
- SafeContract ENFORCES constraints at runtime (clip/clamp)
- PropertyGuard DISCOVERS violations via structured testing (find/shrink)

Use together: test with PropertyGuard to find what breaks,
then enforce with SafeContract to prevent it in deployment.

Usage:
    from vla_edge.validate.properties import (
        check_action_bounds,
        check_velocity_bound,
        check_no_direction_reversals,
        run_property_suite,
    )

    actions = model.predict_trajectory(obs)
    results = run_property_suite(actions, bounds=[-1, 1], v_max=0.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PropertyResult:
    """Result of checking a single property."""

    name: str
    passed: bool
    category: str  # safety, temporal, robustness
    severity: str  # critical, warning, info
    violation_step: int | None = None
    violation_detail: str = ""
    violation_value: float | None = None
    violation_limit: float | None = None


@dataclass
class PropertySuiteResult:
    """Results from running a full property suite."""

    total_properties: int = 0
    passed: int = 0
    failed: int = 0
    critical_failures: int = 0
    results: list[PropertyResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Properties: {self.passed}/{self.total_properties} passed",
            f"Critical failures: {self.critical_failures}",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            detail = f" (step {r.violation_step}: {r.violation_detail})" if not r.passed else ""
            lines.append(f"  [{status}] {r.name} [{r.category}/{r.severity}]{detail}")
        return "\n".join(lines)


# ============================================================
# Safety Properties (must ALWAYS hold)
# ============================================================


def check_action_bounds(
    actions: np.ndarray,
    lo: float = -1.0,
    hi: float = 1.0,
) -> PropertyResult:
    """P1: All action values within [lo, hi]."""
    if actions.size == 0:
        return PropertyResult("action_bounds", True, "safety", "critical")

    min_val = float(actions.min())
    max_val = float(actions.max())

    if min_val < lo or max_val > hi:
        # Find first violation
        for t in range(len(actions)):
            row = actions[t] if actions.ndim > 1 else actions
            if np.any(row < lo) or np.any(row > hi):
                worst = max(abs(min_val - lo) if min_val < lo else 0,
                           abs(max_val - hi) if max_val > hi else 0)
                return PropertyResult(
                    "action_bounds", False, "safety", "critical",
                    violation_step=t,
                    violation_detail=f"range [{min_val:.4f}, {max_val:.4f}] outside [{lo}, {hi}]",
                    violation_value=worst,
                    violation_limit=0.0,
                )

    return PropertyResult("action_bounds", True, "safety", "critical")


def check_velocity_bound(
    actions: np.ndarray,
    v_max: float = 0.1,
) -> PropertyResult:
    """P2: Velocity between consecutive steps never exceeds v_max."""
    if actions.ndim == 1 or len(actions) < 2:
        return PropertyResult("velocity_bound", True, "safety", "critical")

    velocities = np.abs(np.diff(actions, axis=0))
    max_vel = float(velocities.max())

    if max_vel > v_max:
        # Find first violation
        for t in range(len(velocities)):
            if np.any(velocities[t] > v_max):
                joint = int(np.argmax(velocities[t]))
                return PropertyResult(
                    "velocity_bound", False, "safety", "critical",
                    violation_step=t + 1,
                    violation_detail=f"joint {joint} velocity {velocities[t, joint]:.4f} > {v_max}",
                    violation_value=float(velocities[t, joint]),
                    violation_limit=v_max,
                )

    return PropertyResult("velocity_bound", True, "safety", "critical")


def check_acceleration_bound(
    actions: np.ndarray,
    acc_max: float = 0.05,
) -> PropertyResult:
    """P3: Acceleration (jerk proxy) never exceeds acc_max."""
    if actions.ndim == 1 or len(actions) < 3:
        return PropertyResult("acceleration_bound", True, "safety", "warning")

    accels = np.abs(np.diff(actions, n=2, axis=0))
    max_acc = float(accels.max())

    if max_acc > acc_max:
        for t in range(len(accels)):
            if np.any(accels[t] > acc_max):
                joint = int(np.argmax(accels[t]))
                return PropertyResult(
                    "acceleration_bound", False, "safety", "warning",
                    violation_step=t + 2,
                    violation_detail=f"joint {joint} accel {accels[t, joint]:.4f} > {acc_max}",
                    violation_value=float(accels[t, joint]),
                    violation_limit=acc_max,
                )

    return PropertyResult("acceleration_bound", True, "safety", "warning")


def check_gripper_stability(
    actions: np.ndarray,
    gripper_dim: int = -1,
    max_oscillations: int = 3,
    window: int = 10,
) -> PropertyResult:
    """P5: Gripper doesn't oscillate rapidly."""
    if actions.ndim == 1:
        return PropertyResult("gripper_stability", True, "safety", "warning")

    gripper = actions[:, gripper_dim]

    for t in range(window, len(gripper)):
        segment = gripper[t - window : t]
        sign_changes = np.sum(np.abs(np.diff(np.sign(np.diff(segment)))) > 0)
        if sign_changes > max_oscillations:
            return PropertyResult(
                "gripper_stability", False, "safety", "warning",
                violation_step=t,
                violation_detail=f"{sign_changes} oscillations in {window}-step window",
                violation_value=float(sign_changes),
                violation_limit=float(max_oscillations),
            )

    return PropertyResult("gripper_stability", True, "safety", "warning")


# ============================================================
# Temporal Properties (sequence-level)
# ============================================================


def check_no_direction_reversals(
    actions: np.ndarray,
    window: int = 5,
    max_reversals: int = 2,
) -> PropertyResult:
    """P8: No excessive direction reversals within a window."""
    if actions.ndim == 1 or len(actions) < window:
        return PropertyResult("no_direction_reversals", True, "temporal", "warning")

    for t in range(window, len(actions)):
        segment = actions[t - window : t]
        diffs = np.diff(segment, axis=0)
        for j in range(segment.shape[1]):
            signs = np.sign(diffs[:, j])
            sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
            if sign_changes > max_reversals:
                return PropertyResult(
                    "no_direction_reversals", False, "temporal", "warning",
                    violation_step=t,
                    violation_detail=f"joint {j}: {sign_changes} reversals in {window}-step window",
                    violation_value=float(sign_changes),
                    violation_limit=float(max_reversals),
                )

    return PropertyResult("no_direction_reversals", True, "temporal", "warning")


def check_terminal_stability(
    actions: np.ndarray,
    k_steps: int = 10,
    epsilon: float = 0.01,
) -> PropertyResult:
    """P10: Policy converges in last K steps (no oscillation at completion)."""
    if actions.ndim == 1 or len(actions) < k_steps:
        return PropertyResult("terminal_stability", True, "temporal", "info")

    tail = actions[-k_steps:]
    velocities = np.abs(np.diff(tail, axis=0))
    max_vel = float(velocities.max())

    if max_vel > epsilon:
        return PropertyResult(
            "terminal_stability", False, "temporal", "info",
            violation_step=len(actions) - k_steps,
            violation_detail=f"max velocity {max_vel:.4f} > {epsilon} in last {k_steps} steps",
            violation_value=max_vel,
            violation_limit=epsilon,
        )

    return PropertyResult("terminal_stability", True, "temporal", "info")


def check_monotonic_approach(
    actions: np.ndarray,
    target: np.ndarray,
    tolerance: float = 0.02,
    check_dims: int = 3,
) -> PropertyResult:
    """P6: Distance to target generally decreases (reach phase)."""
    if actions.ndim == 1 or len(actions) < 3:
        return PropertyResult("monotonic_approach", True, "temporal", "warning")

    positions = actions[:, :check_dims]
    distances = np.linalg.norm(positions - target[:check_dims], axis=1)

    for t in range(1, len(distances)):
        if distances[t] > distances[t - 1] + tolerance:
            return PropertyResult(
                "monotonic_approach", False, "temporal", "warning",
                violation_step=t,
                violation_detail=f"distance increased from {distances[t-1]:.4f} to {distances[t]:.4f}",
                violation_value=float(distances[t] - distances[t - 1]),
                violation_limit=tolerance,
            )

    return PropertyResult("monotonic_approach", True, "temporal", "warning")


# ============================================================
# Property Suite Runner
# ============================================================


def run_property_suite(
    actions: np.ndarray,
    bounds: tuple[float, float] = (-1.0, 1.0),
    v_max: float = 0.1,
    acc_max: float = 0.05,
    target: np.ndarray | None = None,
) -> PropertySuiteResult:
    """Run all applicable properties on an action sequence.

    Args:
        actions: Shape (T, action_dim) action trajectory
        bounds: Action range [lo, hi]
        v_max: Maximum velocity per joint per step
        acc_max: Maximum acceleration
        target: Target position for monotonic approach (optional)

    Returns:
        PropertySuiteResult with all property check results
    """
    results = []

    # Safety properties (always run)
    results.append(check_action_bounds(actions, bounds[0], bounds[1]))
    results.append(check_velocity_bound(actions, v_max))
    results.append(check_acceleration_bound(actions, acc_max))
    results.append(check_gripper_stability(actions))

    # Temporal properties
    results.append(check_no_direction_reversals(actions))
    results.append(check_terminal_stability(actions))

    if target is not None:
        results.append(check_monotonic_approach(actions, target))

    suite = PropertySuiteResult(
        total_properties=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        critical_failures=sum(1 for r in results if not r.passed and r.severity == "critical"),
        results=results,
    )

    return suite
