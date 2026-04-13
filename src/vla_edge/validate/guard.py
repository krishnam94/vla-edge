"""SafetyGuard: Integrated enforcement + monitoring for VLA deployment.

This is the recommended entry point. It combines action clipping with ALL
monitoring capabilities in a single callable, calibrated from demonstration data.

Instead of:
    @safety_contract(action_range=[-1,1], joint_velocity_max=0.1)  # hand-tuned
    def predict(self, obs): ...

Use:
    guard = SafetyGuard.from_demos(demo_actions)  # data-driven

    @guard.wrap
    def predict(self, obs): ...

Every call now:
1. Clips to conformal bounds (not hand-tuned)
2. Clamps velocity to data-driven per-joint limits
3. Computes conformal p-value (anomaly score)
4. Checks for stall (policy freeze)
5. Checks for jerk spikes (denoising artifacts)
6. Updates CUSUM for distribution shift detection
7. Returns the safe action (monitoring is a side effect)
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from vla_edge.validate.conformal import ConformalActionMonitor, CUSUMDetector
from vla_edge.validate.violations import JerkMonitor, StallDetector

logger = logging.getLogger(__name__)


class SafetyGuard:
    """Integrated action-space enforcement + monitoring.

    Enforcement (what it does to actions):
        - Clip to conformal bounds (position)
        - Clamp velocity to per-joint data-driven limits
        - Re-clip after velocity clamping

    Monitoring (what it reports about actions):
        - Conformal p-value per action (anomaly score, [0,1])
        - Stall detection (consecutive low-movement steps)
        - Jerk monitoring (3rd derivative)
        - CUSUM shift detection (sequential alarm)
        - Clip magnitude (how far each joint was corrected)

    All monitoring is O(1) per step, adding ~15us total overhead.
    """

    def __init__(
        self,
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
        v_max: np.ndarray | float,
        conformal_monitor: ConformalActionMonitor | None = None,
        stall_threshold: float = 1.0,
        stall_window: int = 10,
        jerk_limit: float = 100.0,
        cusum_threshold: float = 3.0,
        alpha: float = 0.05,
        mode: str = "clip",
    ):
        self.bounds_lo = np.asarray(bounds_lo, dtype=np.float32)
        self.bounds_hi = np.asarray(bounds_hi, dtype=np.float32)
        self.v_max = np.asarray(v_max, dtype=np.float32) if not isinstance(v_max, float) else v_max
        self.mode = mode
        self.alpha = alpha

        self.conformal = conformal_monitor
        self.stall = StallDetector(movement_threshold=stall_threshold, stall_window=stall_window)
        self.jerk = JerkMonitor(jerk_limit=jerk_limit)
        self.cusum = CUSUMDetector(expected_rate=alpha, threshold=cusum_threshold)

        self._prev_action: np.ndarray | None = None
        self._n_steps: int = 0
        self._latest_report: dict | None = None
        self._violation_log: list[dict] = []

    @classmethod
    def from_demos(
        cls,
        demo_actions: np.ndarray,
        alpha: float = 0.05,
        v_max_percentile: float = 99.0,
        cal_split: float = 0.8,
        mode: str = "clip",
        **kwargs,
    ) -> SafetyGuard:
        """Create a SafetyGuard calibrated from demonstration data.

        Computes conformal position bounds + data-driven velocity limits.

        Args:
            demo_actions: (N, D) array of demonstration actions
            alpha: Significance level (0.05 = 95% coverage guarantee)
            v_max_percentile: Percentile for velocity limits
            cal_split: Calibration/holdout split ratio
            mode: "clip" (enforce), "warn" (enforce+log), "monitor_only"

        Returns:
            Calibrated SafetyGuard
        """
        n = len(demo_actions)
        n_cal = int(n * cal_split)

        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        cal = demo_actions[indices[:n_cal]]

        # Conformal bounds
        cal_mean = cal.mean(axis=0)
        cal_std = np.maximum(cal.std(axis=0), 1e-8)
        scores = np.max(np.abs(cal - cal_mean) / cal_std, axis=1)
        q_level = min((1 - alpha) * (1 + 1 / n_cal), 1.0)
        q_hat = float(np.quantile(scores, q_level))

        bounds_lo = (cal_mean - q_hat * cal_std).astype(np.float32)
        bounds_hi = (cal_mean + q_hat * cal_std).astype(np.float32)

        # Data-driven velocity limits
        deltas = np.abs(np.diff(cal, axis=0))
        v_max = np.percentile(deltas, v_max_percentile, axis=0).astype(np.float32)

        # Conformal monitor for p-values
        conformal = ConformalActionMonitor()
        conformal.calibrate(cal)

        return cls(
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            v_max=v_max,
            conformal_monitor=conformal,
            alpha=alpha,
            mode=mode,
            **kwargs,
        )

    def __call__(self, raw_action: np.ndarray) -> np.ndarray:
        """Process one action: enforce + monitor.

        Args:
            raw_action: (D,) raw action from VLA model

        Returns:
            Safe action (monitoring is a side effect)
        """
        self._n_steps += 1
        action = np.asarray(raw_action, dtype=np.float32).copy()
        original = action.copy()
        violations = []

        # === ENFORCEMENT ===
        if self.mode != "monitor_only":
            # Phase 1: Bounds clip
            clipped = np.clip(action, self.bounds_lo, self.bounds_hi)
            if not np.allclose(action, clipped, atol=1e-7):
                violations.append("bounds")
            action = clipped

            # Phase 2: Velocity clamp
            if self._prev_action is not None:
                delta = action - self._prev_action
                if np.any(np.abs(delta) > self.v_max):
                    violations.append("velocity")
                action = self._prev_action + np.clip(delta, -self.v_max, self.v_max)
                action = np.clip(action, self.bounds_lo, self.bounds_hi)

        # === MONITORING ===
        clip_magnitude = np.abs(original - action)

        # Conformal p-value
        p_value = None
        if self.conformal is not None:
            p_value = self.conformal.score(original)["p_value"]
            self.cusum.update(p_value < self.alpha)

        # Stall
        stall_report = self.stall.update(action)
        if stall_report["is_stalled"]:
            violations.append("stall")

        # Jerk
        jerk_report = self.jerk.update(action)
        if jerk_report["jerk_violation"]:
            violations.append("jerk")

        self._latest_report = {
            "step": self._n_steps,
            "violations": violations,
            "clip_magnitude": float(np.max(clip_magnitude)),
            "p_value": p_value,
            "is_stalled": stall_report["is_stalled"],
            "stall_duration": stall_report["stall_duration"],
            "max_jerk": jerk_report["max_jerk"],
            "cusum_alarm": self.cusum.n_alarms > 0,
        }

        if violations:
            self._violation_log.append(self._latest_report.copy())
            if self.mode == "warn":
                logger.warning("SafetyGuard step %d: %s", self._n_steps, "+".join(violations))

        self._prev_action = action.copy()
        return action

    def wrap(self, func: Callable) -> Callable:
        """Decorator to wrap a predict() method.

        Usage:
            guard = SafetyGuard.from_demos(demo_actions)

            @guard.wrap
            def predict(self, obs):
                return self.model(obs)
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
            raw = func(*args, **kwargs)
            if not isinstance(raw, np.ndarray):
                raw = np.array(raw, dtype=np.float32)
            return self(raw)
        return wrapper

    def get_report(self) -> dict | None:
        """Get the latest per-step monitoring report."""
        return self._latest_report

    def test_shift(self, recent_n: int | None = None) -> dict:
        """Test for distribution shift via KS test on conformal p-values."""
        if self.conformal is not None:
            return self.conformal.test_uniformity(recent_n)
        return {"error": "no conformal monitor configured"}

    def get_summary(self) -> dict:
        """Get comprehensive session summary."""
        return {
            "n_steps": self._n_steps,
            "n_violations": len(self._violation_log),
            "violation_types": self._count_violation_types(),
            "stall": self.stall.get_summary(),
            "jerk": self.jerk.get_summary(),
            "cusum": self.cusum.get_summary(),
            "shift_test": self.test_shift() if self.conformal else None,
        }

    def _count_violation_types(self) -> dict[str, int]:
        """Count violations by type."""
        counts: dict[str, int] = {}
        for record in self._violation_log:
            for v in record["violations"]:
                counts[v] = counts.get(v, 0) + 1
        return counts

    def reset(self):
        """Reset all state for a new episode."""
        self._prev_action = None
        self._n_steps = 0
        self._latest_report = None
        self._violation_log.clear()
        self.stall.reset()
        self.jerk.reset()
        self.cusum.reset()
        if self.conformal:
            self.conformal.reset()
