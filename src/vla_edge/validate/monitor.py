"""Action health monitoring for VLA policies.

Goes beyond binary violation detection (np.clip) to provide continuous
behavioral health signals: clip magnitude telemetry, action crest factor,
and EWMA violation trend prediction.

These three techniques transform SafeContract from "did it violate?" to
"how wrong is it, how spiky is it, and is it getting worse?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActionHealthReport:
    """Continuous health metrics for a single action or action sequence."""

    # Clip magnitude: how far each joint was corrected (0 = no correction)
    clip_magnitude: np.ndarray | None = None

    # Anomaly z-scores per joint (how many stds from calibration mean)
    z_scores: np.ndarray | None = None

    # Crest factor per joint (peak/RMS ratio over recent window)
    crest_factor: np.ndarray | None = None

    # EWMA violation trend (0-1, rising = policy degrading)
    violation_trend: float = 0.0

    # Overall health score (0 = healthy, higher = more concerning)
    health_score: float = 0.0


class ActionHealthMonitor:
    """Continuous behavioral health monitoring for VLA action outputs.

    Tracks three complementary signals:
    1. Clip magnitude: per-joint "wrongness" (already computed by clipping, just stored)
    2. Crest factor: peak/RMS ratio detecting spikiness per joint
    3. EWMA trend: exponential moving average of violation rate for prediction

    All operations are O(1) per step with minimal state.

    Usage:
        monitor = ActionHealthMonitor(action_dim=7)
        for action in policy_outputs:
            report = monitor.update(action, clipped_action)
            if report.health_score > 3.0:
                print(f"WARNING: policy health degrading on joints {report.worst_joints()}")
    """

    def __init__(
        self,
        action_dim: int = 7,
        window_size: int = 50,
        ewma_alpha: float = 0.1,
        calibration_mean: np.ndarray | None = None,
        calibration_std: np.ndarray | None = None,
    ):
        self.action_dim = action_dim
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha

        # Calibration stats (for z-score computation)
        self.cal_mean = calibration_mean
        self.cal_std = calibration_std

        # State for crest factor (sliding window of squared actions)
        self._action_buffer: list[np.ndarray] = []

        # State for EWMA trend
        self._ewma_violation_rate: float = 0.0
        self._step_count: int = 0

        # History for analysis
        self._clip_magnitude_history: list[np.ndarray] = []
        self._violation_trend_history: list[float] = []

    def update(
        self,
        raw_action: np.ndarray,
        clipped_action: np.ndarray,
        violated: bool = False,
    ) -> ActionHealthReport:
        """Update monitor with a new action pair (raw, clipped).

        Args:
            raw_action: The policy's original output
            clipped_action: The action after SafeContract enforcement
            violated: Whether this step had any violation

        Returns:
            ActionHealthReport with current health metrics
        """
        self._step_count += 1

        # 1. Clip magnitude (FREE - already computed by clipping)
        clip_mag = np.abs(raw_action - clipped_action).astype(np.float32)
        self._clip_magnitude_history.append(clip_mag)

        # 2. Z-scores (if calibration data available)
        z_scores = None
        if self.cal_mean is not None and self.cal_std is not None:
            safe_std = np.maximum(self.cal_std, 1e-6)
            z_scores = np.abs(raw_action - self.cal_mean) / safe_std

        # 3. Crest factor (peak/RMS over sliding window)
        self._action_buffer.append(raw_action.copy())
        if len(self._action_buffer) > self.window_size:
            self._action_buffer.pop(0)

        crest_factor = None
        if len(self._action_buffer) >= 2:
            window = np.array(self._action_buffer)
            rms = np.sqrt(np.mean(window**2, axis=0))
            peak = np.max(np.abs(window), axis=0)
            safe_rms = np.maximum(rms, 1e-6)
            crest_factor = (peak / safe_rms).astype(np.float32)

        # 4. EWMA violation trend
        violation_indicator = 1.0 if violated else 0.0
        self._ewma_violation_rate = (
            self.ewma_alpha * violation_indicator
            + (1 - self.ewma_alpha) * self._ewma_violation_rate
        )
        self._violation_trend_history.append(self._ewma_violation_rate)

        # 5. Composite health score
        health_score = 0.0
        if clip_mag is not None:
            health_score += float(np.max(clip_mag))
        if z_scores is not None:
            health_score += float(np.max(z_scores)) * 0.1
        health_score += self._ewma_violation_rate * 2.0

        return ActionHealthReport(
            clip_magnitude=clip_mag,
            z_scores=z_scores,
            crest_factor=crest_factor,
            violation_trend=self._ewma_violation_rate,
            health_score=health_score,
        )

    def get_summary(self) -> dict:
        """Get summary statistics for the monitoring session."""
        if not self._clip_magnitude_history:
            return {"steps": 0}

        clip_mags = np.array(self._clip_magnitude_history)
        return {
            "steps": self._step_count,
            "mean_clip_magnitude_per_dim": clip_mags.mean(axis=0).round(4).tolist(),
            "max_clip_magnitude_per_dim": clip_mags.max(axis=0).round(4).tolist(),
            "pct_steps_with_any_clip": float(
                np.mean(np.any(clip_mags > 1e-6, axis=1))
            ),
            "final_violation_trend": round(self._ewma_violation_rate, 4),
            "trend_history": [round(t, 4) for t in self._violation_trend_history[-20:]],
        }

    def reset(self):
        """Reset monitor state for a new episode."""
        self._action_buffer.clear()
        self._ewma_violation_rate = 0.0
        self._step_count = 0
        self._clip_magnitude_history.clear()
        self._violation_trend_history.clear()
