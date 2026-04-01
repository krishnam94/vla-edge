"""Phase-adaptive safety contracts for VLA models.

Extends @safety_contract with bounds that tighten during dangerous phases
(approaching objects, grasping) and loosen during safe transit.

Uses kinematic proxies (velocity magnitude, acceleration) to detect task phase,
inspired by DyQ-VLA's kinematic-based quantization switching.

Novel contribution: CompliantVLA-adaptor adapts impedance (soft, about interaction
quality). We adapt hard safety bounds (about never-violate guarantees). The
philosophy comes from variable impedance control (decades old) but the mechanism
is new for VLA safety contracts.

Usage:
    from vla_edge.validate.adaptive import adaptive_safety_contract

    @adaptive_safety_contract(
        safe_bounds=[-1.5, 1.5],      # Loose bounds during transit
        danger_bounds=[-0.5, 0.5],    # Tight bounds during manipulation
        velocity_threshold=0.05,       # Below this = "dangerous" (slow = precision phase)
        safe_velocity_max=0.2,         # Loose velocity limit during transit
        danger_velocity_max=0.05,      # Tight velocity limit during manipulation
    )
    def predict(self, image, instruction, state=None):
        return self.model(image)

Reference:
- DyQ-VLA (arXiv:2603.07904): kinematic-based switching for quantization
- CompliantVLA (arXiv:2601.15541): impedance adaptation by task phase
- Variable impedance control (Frontiers 2020 review)
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhaseStats:
    """Statistics about phase detection and bound adaptation."""

    total_calls: int = 0
    safe_phase_calls: int = 0
    danger_phase_calls: int = 0
    transitions: int = 0
    velocity_history: list[float] = field(default_factory=list)

    @property
    def danger_ratio(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.danger_phase_calls / self.total_calls


def adaptive_safety_contract(
    safe_bounds: tuple[float, float] | list[float] = (-1.5, 1.5),
    danger_bounds: tuple[float, float] | list[float] = (-0.5, 0.5),
    velocity_threshold: float = 0.05,
    safe_velocity_max: float = 0.2,
    danger_velocity_max: float = 0.05,
    on_violation: str = "clip",
) -> Callable:
    """Phase-adaptive safety contract decorator.

    Detects task phase from kinematic state (velocity magnitude) and
    switches between loose (transit) and tight (manipulation) bounds.

    Args:
        safe_bounds: Action range during safe/transit phase [min, max].
        danger_bounds: Action range during dangerous/manipulation phase [min, max].
        velocity_threshold: Velocity magnitude below this = danger phase.
            Low velocity means precision manipulation (approaching, grasping).
            High velocity means open transit (safe to be less restrictive).
        safe_velocity_max: Max action change per step during transit.
        danger_velocity_max: Max action change per step during manipulation.
        on_violation: "clip" (default), "warn", or "raise".
    """

    def decorator(func: Callable) -> Callable:
        last_action: dict[str, np.ndarray | None] = {"value": None}
        last_phase: dict[str, str] = {"value": "safe"}
        stats = PhaseStats()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
            action = func(*args, **kwargs)

            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)

            violations: list[str] = []

            # Phase detection from kinematic state
            if last_action["value"] is not None and last_action["value"].shape == action.shape:
                velocity = np.abs(action - last_action["value"])
                velocity_magnitude = float(np.mean(velocity))
            else:
                velocity_magnitude = 1.0  # Assume safe (high velocity) on first call

            # Determine phase
            is_danger = velocity_magnitude < velocity_threshold
            current_phase = "danger" if is_danger else "safe"

            # Track transitions
            if current_phase != last_phase["value"]:
                stats.transitions += 1
            last_phase["value"] = current_phase

            # Select bounds based on phase
            if is_danger:
                bounds = danger_bounds
                vel_max = danger_velocity_max
                stats.danger_phase_calls += 1
            else:
                bounds = safe_bounds
                vel_max = safe_velocity_max
                stats.safe_phase_calls += 1

            stats.total_calls += 1
            stats.velocity_history.append(velocity_magnitude)

            # Apply bounds
            lo, hi = bounds[0], bounds[1]
            if np.any(action < lo) or np.any(action > hi):
                violations.append(
                    f"bounds ({current_phase}): [{action.min():.3f}, {action.max():.3f}] "
                    f"outside [{lo}, {hi}]"
                )
            action = np.clip(action, lo, hi)

            # Apply velocity limit
            if last_action["value"] is not None and last_action["value"].shape == action.shape:
                delta = action - last_action["value"]
                if np.any(np.abs(delta) > vel_max):
                    violations.append(
                        f"velocity ({current_phase}): max delta {np.abs(delta).max():.3f} > {vel_max}"
                    )
                action = last_action["value"] + np.clip(delta, -vel_max, vel_max)

            # Re-clip after velocity clamping
            action = np.clip(action, lo, hi)

            # Store for next call
            last_action["value"] = action.copy()

            # Handle violations
            if violations and on_violation == "warn":
                for v in violations:
                    logger.warning("Adaptive contract (%s phase): %s", current_phase, v)
            elif violations and on_violation == "raise":
                from vla_edge.validate.contract import SafetyViolationError

                raise SafetyViolationError(
                    f"Adaptive contract violated ({current_phase} phase): " + "; ".join(violations)
                )

            return action

        # Attach metadata
        wrapper._adaptive_contract = {  # type: ignore[attr-defined]
            "safe_bounds": safe_bounds,
            "danger_bounds": danger_bounds,
            "velocity_threshold": velocity_threshold,
            "safe_velocity_max": safe_velocity_max,
            "danger_velocity_max": danger_velocity_max,
        }
        wrapper._phase_stats = stats  # type: ignore[attr-defined]

        return wrapper

    return decorator
