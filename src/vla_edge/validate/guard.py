"""SafetyGuard - wraps inference with safety enforcement.

Solves the problem that safety validation exists but isn't wired into
the inference hot path. The guard clips actions and reports violations
on every inference call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_edge.backends.base import InferenceResult
from vla_edge.validate.safety import SafetyConfig, SafetyResult, clip_actions, validate_actions

logger = logging.getLogger(__name__)


@dataclass
class GuardedResult:
    """Inference result with safety enforcement applied."""

    inference: InferenceResult
    safety: SafetyResult
    original_actions: np.ndarray
    clipped_actions: np.ndarray

    @property
    def is_safe(self) -> bool:
        return self.safety.is_safe


class SafetyGuard:
    """Wraps a backend + model with per-step safety enforcement.

    Usage:
        guard = SafetyGuard(backend, config)
        result = guard.safe_infer(model, observation)
        # result.clipped_actions are safe to execute
        # result.safety has violation details
    """

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self._violation_count = 0
        self._total_calls = 0
        self._history: list[SafetyResult] = []

    def safe_infer(
        self,
        backend: Any,
        model: Any,
        observation: dict[str, Any],
    ) -> GuardedResult:
        """Run inference with safety enforcement.

        1. Backend runs inference
        2. Validate actions against safety config
        3. Clip actions to safe bounds
        4. Return both original and clipped actions + safety report
        """
        # Run raw inference
        inference_result = backend.infer(model, observation)
        raw_actions = inference_result.actions

        if raw_actions is None:
            return GuardedResult(
                inference=inference_result,
                safety=SafetyResult(is_safe=True, total_actions=0),
                original_actions=np.array([]),
                clipped_actions=np.array([]),
            )

        # Ensure 2D for validation
        actions_2d = raw_actions if raw_actions.ndim >= 2 else raw_actions[np.newaxis, :]

        # Validate
        safety_result = validate_actions(actions_2d, self.config)

        # Clip to safe bounds
        clipped = clip_actions(actions_2d, self.config)

        # Track stats
        self._total_calls += 1
        if not safety_result.is_safe:
            self._violation_count += 1
            logger.warning(
                "Safety violation at step %d: %d violations (%d critical)",
                self._total_calls,
                len(safety_result.violations),
                sum(1 for v in safety_result.violations if v.severity == "critical"),
            )

        self._history.append(safety_result)

        # Squeeze back to original shape
        if raw_actions.ndim == 1:
            clipped = clipped.squeeze(0)

        return GuardedResult(
            inference=inference_result,
            safety=safety_result,
            original_actions=raw_actions,
            clipped_actions=clipped,
        )

    @property
    def violation_rate(self) -> float:
        """Fraction of inference calls that had violations."""
        if self._total_calls == 0:
            return 0.0
        return self._violation_count / self._total_calls

    @property
    def summary(self) -> dict[str, Any]:
        """Summary statistics across all guarded calls."""
        total_violations = sum(len(r.violations) for r in self._history)
        critical = sum(sum(1 for v in r.violations if v.severity == "critical") for r in self._history)
        return {
            "total_calls": self._total_calls,
            "calls_with_violations": self._violation_count,
            "violation_rate": round(self.violation_rate, 4),
            "total_violations": total_violations,
            "critical_violations": critical,
            "max_velocity_observed": max((r.max_velocity_observed for r in self._history), default=0.0),
        }

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._violation_count = 0
        self._total_calls = 0
        self._history.clear()
