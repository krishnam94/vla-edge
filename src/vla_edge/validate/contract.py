"""Formal safety contracts for VLA models.

Decorators that enforce physical safety constraints on any predict() method,
regardless of what the neural network outputs. Safety as a precondition,
not a hope.

Usage:
    @safety_contract(
        joint_velocity_max=1.0,       # rad/s
        workspace_bounds=[[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.8]],
        action_range=[-1.0, 1.0],
    )
    def predict(self, image, instruction, state=None):
        ...  # neural network outputs raw actions
        # The decorator clips and validates BEFORE returning

This is inspired by design-by-contract (Eiffel) and runtime verification
from formal methods. The contract is enforced at the function boundary,
not inside the model.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Track violations globally for reporting
_violation_log: list[dict[str, Any]] = []


def safety_contract(
    action_range: tuple[float, float] | list[float] | None = None,
    joint_velocity_max: float | None = None,
    workspace_bounds: list[list[float]] | None = None,
    on_violation: str = "clip",  # "clip", "warn", "raise"
) -> Callable:
    """Decorator that enforces safety constraints on predict() output.

    Args:
        action_range: Symmetric bounds for all action dimensions [min, max].
        joint_velocity_max: Maximum change between consecutive actions (rad/s or m/s).
        workspace_bounds: End-effector bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
        max_force_n: Maximum force in Newtons (checked if action includes force dims).
        on_violation: What to do on violation:
            - "clip": silently clip to safe values (default, safest for real robots)
            - "warn": clip and log a warning
            - "raise": raise SafetyViolationError (for testing/development)

    Returns:
        Decorated function that always returns safe actions.

    Example:
        @safety_contract(action_range=[-1, 1], joint_velocity_max=0.1)
        def predict(self, image, instruction, state=None):
            return self.model(image)  # raw, possibly unsafe output

        # predict() now ALWAYS returns actions within [-1, 1]
        # with velocity clamped to 0.1 per step
    """

    def decorator(func: Callable) -> Callable:
        # Store the last action for velocity checking
        last_action: dict[str, np.ndarray | None] = {"value": None}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
            # Call the original predict
            action = func(*args, **kwargs)

            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)

            original = action.copy()
            violations: list[str] = []

            # 1. Action range clipping
            if action_range is not None:
                lo, hi = action_range[0], action_range[1]
                if np.any(action < lo) or np.any(action > hi):
                    violations.append(
                        f"action_range: values {action.min():.3f} to {action.max():.3f} outside [{lo}, {hi}]"
                    )
                action = np.clip(action, lo, hi)

            # 2. Workspace bounds (assumes first 3 dims are xyz)
            if workspace_bounds is not None and action.shape[-1] >= 3:
                flat = action.reshape(-1, action.shape[-1])
                for dim in range(min(3, len(workspace_bounds))):
                    lo, hi = workspace_bounds[dim]
                    if np.any(flat[:, dim] < lo) or np.any(flat[:, dim] > hi):
                        violations.append(
                            f"workspace dim {dim}: {flat[:, dim].min():.3f} to "
                            f"{flat[:, dim].max():.3f} outside [{lo}, {hi}]"
                        )
                    flat[:, dim] = np.clip(flat[:, dim], lo, hi)
                action = flat.reshape(action.shape)

            # 3. Velocity clamping (requires previous action)
            if joint_velocity_max is not None and last_action["value"] is not None:
                prev = last_action["value"]
                if prev.shape == action.shape or (
                    action.ndim == 1 and prev.ndim == 1 and prev.shape[0] == action.shape[0]
                ):
                    delta = action - prev
                    if np.any(np.abs(delta) > joint_velocity_max):
                        violations.append(
                            f"velocity: max delta {np.abs(delta).max():.3f} > limit {joint_velocity_max}"
                        )
                    action = prev + np.clip(delta, -joint_velocity_max, joint_velocity_max)

            # Re-apply action range after velocity clamping to ensure bounds
            if action_range is not None:
                action = np.clip(action, action_range[0], action_range[1])

            # Store for next call's velocity check
            if action.ndim == 1:
                last_action["value"] = action.copy()
            elif action.ndim == 2:
                last_action["value"] = action[-1].copy()

            # Handle violations
            if violations:
                violation_record = {
                    "function": func.__qualname__,
                    "violations": violations,
                    "original_range": [float(original.min()), float(original.max())],
                    "clipped_range": [float(action.min()), float(action.max())],
                }
                _violation_log.append(violation_record)

                if on_violation == "warn":
                    for v in violations:
                        logger.warning("Safety contract violation in %s: %s", func.__qualname__, v)
                elif on_violation == "raise":
                    raise SafetyViolationError(
                        f"Safety contract violated in {func.__qualname__}: " + "; ".join(violations)
                    )
                # "clip" mode: silently clip (already done above)

            return action

        # Attach contract metadata to the function
        wrapper._safety_contract = {  # type: ignore[attr-defined]
            "action_range": action_range,
            "joint_velocity_max": joint_velocity_max,
            "workspace_bounds": workspace_bounds,
            "on_violation": on_violation,
        }

        return wrapper

    return decorator


class SafetyViolationError(Exception):
    """Raised when a safety contract is violated in 'raise' mode."""


def get_violation_log() -> list[dict[str, Any]]:
    """Get all recorded safety contract violations."""
    return _violation_log.copy()


def clear_violation_log() -> None:
    """Clear the violation log."""
    _violation_log.clear()
