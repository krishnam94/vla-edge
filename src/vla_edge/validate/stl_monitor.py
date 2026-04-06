"""Signal Temporal Logic (STL) runtime monitoring for VLA actions.

Uses RTAMT to specify and monitor temporal safety properties with
quantitative robustness. Positive robustness = satisfied, negative = violated,
magnitude = how far from the boundary.

This bridges design-by-contract (per-call) and runtime verification (temporal).
Meyer's contracts are instantaneous; STL captures patterns over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import rtamt
    RTAMT_AVAILABLE = True
except ImportError:
    RTAMT_AVAILABLE = False


@dataclass
class STLReport:
    """STL monitoring report for a single timestep."""
    # Robustness scores per spec (positive = satisfied, negative = violated)
    robustness: dict[str, float]
    # Overall minimum robustness (worst spec)
    min_robustness: float
    # Which specs are currently violated
    violated_specs: list[str]
    # Timestep
    step: int


class STLActionMonitor:
    """STL runtime monitor for VLA action streams.

    Defines temporal safety properties and monitors quantitative robustness
    in real-time. Each property produces a continuous score:
    - Positive = property satisfied (by this margin)
    - Negative = property violated (by this much)
    - Zero = exactly at the boundary

    Specifications:
        1. bounds_health: "clip magnitude stays below threshold"
        2. velocity_health: "velocity stays below limit"
        3. recovery: "after a violation, system recovers within k steps"
        4. sustained_violation: "no sustained violations exceeding n steps"

    Usage:
        monitor = STLActionMonitor(n_dims=7, v_max=0.1)
        for raw_action, clipped_action in policy_outputs:
            report = monitor.update(raw_action, clipped_action)
            if report.min_robustness < 0:
                print(f"STL violation: {report.violated_specs}")
    """

    def __init__(
        self,
        n_dims: int = 7,
        bounds: tuple[float, float] = (-1.0, 1.0),
        v_max: float = 0.1,
        clip_mag_threshold: float = 0.5,
        recovery_window: int = 5,
        sustained_window: int = 10,
    ):
        self.n_dims = n_dims
        self.bounds = bounds
        self.v_max = v_max
        self.clip_mag_threshold = clip_mag_threshold
        self.recovery_window = recovery_window
        self.sustained_window = sustained_window

        self._step = 0
        self._prev_action: np.ndarray | None = None

        # Signal histories for STL evaluation
        self._clip_mag_history: list[float] = []
        self._velocity_history: list[float] = []
        self._violated_history: list[float] = []  # 1.0 = violated, 0.0 = ok

        # Robustness histories
        self._robustness_history: list[dict[str, float]] = []

    def update(
        self,
        raw_action: np.ndarray,
        clipped_action: np.ndarray,
    ) -> STLReport:
        """Update monitor with new action pair.

        Args:
            raw_action: Policy's original output
            clipped_action: Action after enforcement

        Returns:
            STLReport with robustness scores
        """
        self._step += 1

        # Compute signals
        clip_mag = float(np.max(np.abs(raw_action - clipped_action)))
        violated = 1.0 if clip_mag > 1e-7 else 0.0

        velocity = 0.0
        if self._prev_action is not None:
            velocity = float(np.max(np.abs(clipped_action - self._prev_action)))

        self._clip_mag_history.append(clip_mag)
        self._velocity_history.append(velocity)
        self._violated_history.append(violated)
        self._prev_action = clipped_action.copy()

        # Compute robustness for each specification
        robustness = {}

        # Spec 1: Bounds health - clip magnitude below threshold
        # rho = threshold - clip_mag (positive = healthy margin)
        robustness["bounds_health"] = self.clip_mag_threshold - clip_mag

        # Spec 2: Velocity health - velocity below limit
        # rho = v_max - velocity (positive = within limit)
        robustness["velocity_health"] = self.v_max - velocity

        # Spec 3: Recovery - if violated, must recover within k steps
        # Check: if there was a violation in the last k steps,
        # is the most recent step clean?
        if len(self._violated_history) >= self.recovery_window:
            window = self._violated_history[-self.recovery_window:]
            if any(v > 0.5 for v in window[:-1]):
                # There was a violation in the window
                # Robustness = whether we recovered (last step clean)
                if window[-1] < 0.5:
                    # Recovered - robustness is the margin (how clean)
                    robustness["recovery"] = self.clip_mag_threshold - self._clip_mag_history[-1]
                else:
                    # Still violating - negative robustness
                    robustness["recovery"] = -self._clip_mag_history[-1]
            else:
                # No recent violation - trivially satisfied
                robustness["recovery"] = self.clip_mag_threshold
        else:
            robustness["recovery"] = self.clip_mag_threshold

        # Spec 4: No sustained violations - violation rate in window below 50%
        if len(self._violated_history) >= self.sustained_window:
            window = self._violated_history[-self.sustained_window:]
            violation_rate = sum(window) / len(window)
            # Robustness = 0.5 - violation_rate (positive if rate < 50%)
            robustness["sustained_health"] = 0.5 - violation_rate
        else:
            robustness["sustained_health"] = 0.5

        # Spec 5: Smoothness - clip magnitude trend not increasing
        if len(self._clip_mag_history) >= 10:
            recent = self._clip_mag_history[-10:]
            first_half = np.mean(recent[:5])
            second_half = np.mean(recent[5:])
            # Robustness = first_half - second_half (positive if improving)
            robustness["smoothness_trend"] = float(first_half - second_half)
        else:
            robustness["smoothness_trend"] = 0.0

        min_rob = min(robustness.values())
        violated_specs = [k for k, v in robustness.items() if v < 0]

        self._robustness_history.append(robustness)

        return STLReport(
            robustness=robustness,
            min_robustness=min_rob,
            violated_specs=violated_specs,
            step=self._step,
        )

    def get_summary(self) -> dict:
        """Get summary of STL monitoring session."""
        if not self._robustness_history:
            return {"steps": 0}

        # Aggregate robustness per spec
        specs = list(self._robustness_history[0].keys())
        summary = {"steps": self._step, "specs": {}}

        for spec in specs:
            values = [r[spec] for r in self._robustness_history]
            n_violated = sum(1 for v in values if v < 0)
            summary["specs"][spec] = {
                "mean_robustness": round(float(np.mean(values)), 4),
                "min_robustness": round(float(np.min(values)), 4),
                "max_robustness": round(float(np.max(values)), 4),
                "violation_rate": round(n_violated / len(values), 4),
                "n_violated": n_violated,
            }

        summary["overall_min_robustness"] = round(
            float(min(min(r.values()) for r in self._robustness_history)), 4
        )
        summary["n_any_spec_violated"] = sum(
            1 for r in self._robustness_history if min(r.values()) < 0
        )

        return summary

    def reset(self):
        """Reset monitor state."""
        self._step = 0
        self._prev_action = None
        self._clip_mag_history.clear()
        self._velocity_history.clear()
        self._violated_history.clear()
        self._robustness_history.clear()
