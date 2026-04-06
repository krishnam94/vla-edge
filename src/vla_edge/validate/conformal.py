"""Conformal anomaly detection for VLA action monitoring.

Provides per-action conformal p-values with distribution-free guarantees.
Under exchangeability, p-values are Uniform[0,1]; departure from uniformity
signals distribution shift (testable via Kolmogorov-Smirnov).

Includes:
- ConformalActionMonitor: fixed-split calibration + scoring
- AdaptiveConformalMonitor (ACAM): online adaptation under shift (Gibbs & Candes 2021)
- MultiScaleCUSUM: change detection at step, chunk, and episode scales
- CUSUMDetector: single-scale optimal shift detection (Page 1954)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


class ConformalActionMonitor:
    """Conformal anomaly scoring for VLA actions.

    Calibrates from demonstration data, then scores each new action
    with a valid p-value measuring how anomalous it is relative to
    the calibration distribution.

    Usage:
        monitor = ConformalActionMonitor()
        monitor.calibrate(demo_actions)  # (N, D) array
        for action in policy_outputs:
            report = monitor.score(action)
            if report['p_value'] < 0.01:
                print(f"Highly anomalous action (p={report['p_value']:.4f})")
    """

    def __init__(self):
        self._calibrated = False
        self._cal_scores: np.ndarray | None = None
        self._cal_mean: np.ndarray | None = None
        self._cal_std: np.ndarray | None = None
        self._p_value_history: list[float] = []

    def calibrate(self, actions: np.ndarray) -> dict:
        """Calibrate from demonstration actions.

        Args:
            actions: (N, D) array of calibration actions

        Returns:
            Dict with calibration statistics
        """
        self._cal_mean = actions.mean(axis=0).astype(np.float32)
        self._cal_std = np.maximum(actions.std(axis=0), 1e-8).astype(np.float32)

        # Nonconformity scores: max normalized deviation per action
        self._cal_scores = np.sort(
            np.max(np.abs(actions - self._cal_mean) / self._cal_std, axis=1)
        )
        self._calibrated = True
        self._p_value_history.clear()

        return {
            "n_calibration": len(actions),
            "n_dims": actions.shape[1],
            "cal_mean": self._cal_mean.tolist(),
            "cal_std": self._cal_std.tolist(),
            "score_range": [float(self._cal_scores[0]), float(self._cal_scores[-1])],
        }

    def score(self, action: np.ndarray) -> dict:
        """Score a single action with conformal p-value.

        Args:
            action: (D,) action vector

        Returns:
            Dict with p_value, nonconformity_score, per_dim_scores, is_anomalous
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before score()")

        # Nonconformity score for this action
        per_dim = np.abs(action - self._cal_mean) / self._cal_std
        score = float(np.max(per_dim))

        # Conformal p-value: fraction of calibration scores >= this score
        # Valid p-value under exchangeability: P(p <= alpha) <= alpha
        n = len(self._cal_scores)
        rank = np.searchsorted(self._cal_scores, score, side="right")
        p_value = (n - rank + 1) / (n + 1)

        self._p_value_history.append(p_value)

        return {
            "p_value": p_value,
            "nonconformity_score": score,
            "per_dim_scores": per_dim.astype(np.float32),
            "is_anomalous_01": p_value < 0.01,
            "is_anomalous_05": p_value < 0.05,
        }

    def test_uniformity(self, recent_n: int | None = None) -> dict:
        """Test if recent p-values are Uniform[0,1] (KS test).

        Departure from uniformity signals distribution shift.

        Args:
            recent_n: Number of recent p-values to test (None = all)

        Returns:
            Dict with ks_statistic, ks_p_value, shift_detected
        """
        pvals = self._p_value_history[-recent_n:] if recent_n else self._p_value_history
        if len(pvals) < 10:
            return {"error": "need at least 10 p-values", "n": len(pvals)}

        ks_stat, ks_p = sp_stats.kstest(pvals, "uniform")
        return {
            "n_pvalues": len(pvals),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "shift_detected": ks_p < 0.05,
            "mean_pvalue": float(np.mean(pvals)),
            "median_pvalue": float(np.median(pvals)),
        }

    def get_conformal_bounds(self, alpha: float = 0.05) -> dict:
        """Get conformal prediction bounds at significance level alpha.

        Args:
            alpha: Significance level (0.05 = 95% coverage)

        Returns:
            Dict with bounds, quantile threshold, expected coverage
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before get_conformal_bounds()")

        n = len(self._cal_scores)
        # Finite-sample correction
        q_level = min((1 - alpha) * (1 + 1 / n), 1.0)
        q_idx = int(np.ceil(q_level * n)) - 1
        q_hat = float(self._cal_scores[min(q_idx, n - 1)])

        bounds_lo = self._cal_mean - q_hat * self._cal_std
        bounds_hi = self._cal_mean + q_hat * self._cal_std

        return {
            "alpha": alpha,
            "quantile_threshold": q_hat,
            "bounds_lo": bounds_lo.tolist(),
            "bounds_hi": bounds_hi.tolist(),
            "expected_coverage": 1 - alpha,
        }

    def reset(self):
        """Reset p-value history (keep calibration)."""
        self._p_value_history.clear()


class CUSUMDetector:
    """CUSUM sequential change-point detection on violation streams.

    Optimal (Lorden-minimax) for detecting persistent shifts in violation rate.
    Replaces ad-hoc EWMA with principled detection with ARL guarantees.

    Usage:
        detector = CUSUMDetector(expected_rate=0.05, threshold=5.0)
        for violated in violation_stream:
            alarm = detector.update(violated)
            if alarm:
                print(f"Shift detected at step {detector.n_steps}!")
    """

    def __init__(
        self,
        expected_rate: float = 0.05,
        threshold: float = 5.0,
    ):
        """
        Args:
            expected_rate: Expected violation rate under null (from calibration).
                          If using conformal bounds at alpha, set to alpha.
            threshold: Detection threshold h. Higher = fewer false alarms, slower detection.
                      ARL_0 ~ exp(h) / expected_rate for small expected_rate.
        """
        self.expected_rate = expected_rate
        self.threshold = threshold
        self.cusum_stat: float = 0.0
        self.n_steps: int = 0
        self.n_alarms: int = 0
        self._stat_history: list[float] = []

    def update(self, violated: bool) -> bool:
        """Update CUSUM with new observation.

        Args:
            violated: Whether this step had a violation

        Returns:
            True if alarm (shift detected)
        """
        self.n_steps += 1
        v = 1.0 if violated else 0.0

        # CUSUM statistic: accumulate deviations above expected rate
        self.cusum_stat = max(0.0, self.cusum_stat + v - self.expected_rate)
        self._stat_history.append(self.cusum_stat)

        alarm = self.cusum_stat > self.threshold
        if alarm:
            self.n_alarms += 1
            self.cusum_stat = 0.0  # Reset after alarm

        return alarm

    def get_summary(self) -> dict:
        """Get CUSUM state summary."""
        return {
            "n_steps": self.n_steps,
            "current_stat": self.cusum_stat,
            "n_alarms": self.n_alarms,
            "threshold": self.threshold,
            "expected_rate": self.expected_rate,
            "stat_history_last20": [round(s, 4) for s in self._stat_history[-20:]],
            "peak_stat": float(max(self._stat_history)) if self._stat_history else 0.0,
        }

    def reset(self):
        """Reset detector state."""
        self.cusum_stat = 0.0
        self.n_steps = 0
        self.n_alarms = 0
        self._stat_history.clear()


class AdaptiveConformalMonitor:
    """Adaptive Conformal Action Monitoring (ACAM).

    Online adaptation of conformal prediction under distribution shift,
    following Gibbs & Candes (NeurIPS 2021). The key insight: if the
    violation rate exceeds the target alpha, widen bounds; if below, tighten.

    This maintains long-run coverage guarantees without assuming
    exchangeability - critical for VLA deployment where the action
    distribution shifts over time (compounding errors, task changes).

    Algorithm:
        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
        where err_t = 1 if the action's score exceeds the current quantile

    Usage:
        acam = AdaptiveConformalMonitor(alpha=0.05, gamma=0.01)
        acam.calibrate(demo_actions)
        for action in policy_stream:
            report = acam.update(action)
            # report['adaptive_alpha'] changes over time
            # report['bounds'] tighten/widen to maintain coverage
    """

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.005,
    ):
        self.alpha_target = alpha
        self.gamma = gamma
        self.alpha_t = alpha  # current adaptive alpha

        self._cal_scores: np.ndarray | None = None
        self._cal_mean: np.ndarray | None = None
        self._cal_std: np.ndarray | None = None
        self._calibrated = False

        self._n_steps = 0
        self._coverage_errors: list[float] = []
        self._alpha_history: list[float] = [alpha]
        self._p_value_history: list[float] = []
        self._bounds_width_history: list[float] = []

    def calibrate(self, actions: np.ndarray) -> dict:
        """Calibrate from demonstration actions (same as ConformalActionMonitor)."""
        self._cal_mean = actions.mean(axis=0).astype(np.float32)
        self._cal_std = np.maximum(actions.std(axis=0), 1e-8).astype(np.float32)
        self._cal_scores = np.sort(
            np.max(np.abs(actions - self._cal_mean) / self._cal_std, axis=1)
        )
        self._calibrated = True
        return {
            "n_calibration": len(actions),
            "n_dims": actions.shape[1],
        }

    def _get_quantile(self, alpha: float) -> float:
        """Get the conformal quantile at current alpha level."""
        n = len(self._cal_scores)
        q_level = min((1 - alpha) * (1 + 1 / n), 1.0)
        q_idx = int(np.ceil(q_level * n)) - 1
        return float(self._cal_scores[min(q_idx, n - 1)])

    def update(self, action: np.ndarray) -> dict:
        """Score action and adapt alpha.

        Args:
            action: (D,) action vector

        Returns:
            Dict with p_value, adaptive_alpha, bounds, coverage_error
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before update()")

        self._n_steps += 1

        # Compute nonconformity score
        per_dim = np.abs(action - self._cal_mean) / self._cal_std
        score = float(np.max(per_dim))

        # P-value (same as ConformalActionMonitor)
        n = len(self._cal_scores)
        rank = np.searchsorted(self._cal_scores, score, side="right")
        p_value = (n - rank + 1) / (n + 1)
        self._p_value_history.append(p_value)

        # Current quantile threshold at adaptive alpha
        q_hat = self._get_quantile(self.alpha_t)

        # Coverage error: did this action exceed the threshold?
        err_t = 1.0 if score > q_hat else 0.0
        self._coverage_errors.append(err_t)

        # ACI update (Gibbs & Candes 2021): adapt alpha to maintain coverage
        # If undercovering (err_t > alpha_target), alpha increases -> wider bounds
        # If overcovering (err_t < alpha_target), alpha decreases -> tighter bounds
        self.alpha_t = self.alpha_t + self.gamma * (err_t - self.alpha_target)
        self.alpha_t = np.clip(self.alpha_t, 0.001, 0.5)  # keep in reasonable range
        self._alpha_history.append(float(self.alpha_t))

        # Current adaptive bounds
        bounds_lo = self._cal_mean - q_hat * self._cal_std
        bounds_hi = self._cal_mean + q_hat * self._cal_std
        bounds_width = float(np.mean(bounds_hi - bounds_lo))
        self._bounds_width_history.append(bounds_width)

        return {
            "p_value": p_value,
            "score": score,
            "adaptive_alpha": float(self.alpha_t),
            "quantile_threshold": q_hat,
            "coverage_error": err_t,
            "bounds_lo": bounds_lo,
            "bounds_hi": bounds_hi,
            "bounds_width": bounds_width,
        }

    def get_summary(self) -> dict:
        """Get ACAM monitoring summary."""
        if not self._coverage_errors:
            return {"n_steps": 0}

        empirical_coverage = 1 - np.mean(self._coverage_errors)
        return {
            "n_steps": self._n_steps,
            "target_alpha": self.alpha_target,
            "final_alpha": float(self.alpha_t),
            "empirical_coverage": float(empirical_coverage),
            "target_coverage": 1 - self.alpha_target,
            "coverage_gap": float(empirical_coverage - (1 - self.alpha_target)),
            "alpha_range": [
                float(min(self._alpha_history)),
                float(max(self._alpha_history)),
            ],
            "mean_bounds_width": float(np.mean(self._bounds_width_history)),
            "alpha_history_last20": [round(a, 4) for a in self._alpha_history[-20:]],
            "mean_p_value": float(np.mean(self._p_value_history)),
        }

    def reset(self):
        """Reset adaptive state (keep calibration)."""
        self.alpha_t = self.alpha_target
        self._n_steps = 0
        self._coverage_errors.clear()
        self._alpha_history = [self.alpha_target]
        self._p_value_history.clear()
        self._bounds_width_history.clear()


class MultiScaleCUSUM:
    """Multi-scale CUSUM for VLA action monitoring.

    Runs CUSUM at three temporal scales simultaneously:
    - Step-level: detects instantaneous spikes (window=1)
    - Chunk-level: detects short-term drift (window=action_chunk_size)
    - Episode-level: detects sustained degradation (window=episode_length)

    Each scale catches different failure modes:
    - Step: "one action was wildly wrong" (immediate danger)
    - Chunk: "the last 10 actions are trending bad" (emerging problem)
    - Episode: "this whole episode has high violations" (systematic issue)

    Usage:
        ms = MultiScaleCUSUM(alpha=0.05, chunk_size=10, episode_size=100)
        ms.calibrate(demo_p_values)
        for p_value in monitoring_stream:
            report = ms.update(p_value)
            if report['any_alarm']:
                print(f"Alarm at scale: {report['alarm_scales']}")
    """

    def __init__(
        self,
        alpha: float = 0.05,
        chunk_size: int = 10,
        episode_size: int = 100,
        step_threshold: float = 3.0,
        chunk_threshold: float = 5.0,
        episode_threshold: float = 8.0,
    ):
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.episode_size = episode_size

        # Three CUSUM detectors at different scales
        self._step_cusum = CUSUMDetector(expected_rate=alpha, threshold=step_threshold)
        self._chunk_cusum = CUSUMDetector(expected_rate=alpha, threshold=chunk_threshold)
        self._episode_cusum = CUSUMDetector(expected_rate=alpha, threshold=episode_threshold)

        self._p_buffer: list[float] = []
        self._n_steps = 0

    def update(self, p_value: float) -> dict:
        """Update all three scales with a new conformal p-value.

        Args:
            p_value: Conformal p-value from ConformalActionMonitor or ACAM

        Returns:
            Dict with alarm status at each scale
        """
        self._n_steps += 1
        self._p_buffer.append(p_value)
        violated = p_value < self.alpha

        # Step-level: every action
        step_alarm = self._step_cusum.update(violated)

        # Chunk-level: aggregate every chunk_size steps
        chunk_alarm = False
        if len(self._p_buffer) >= self.chunk_size and self._n_steps % self.chunk_size == 0:
            chunk_pvals = self._p_buffer[-self.chunk_size:]
            chunk_violation_rate = sum(1 for p in chunk_pvals if p < self.alpha) / len(chunk_pvals)
            chunk_alarm = self._chunk_cusum.update(chunk_violation_rate > self.alpha)

        # Episode-level: aggregate every episode_size steps
        episode_alarm = False
        if len(self._p_buffer) >= self.episode_size and self._n_steps % self.episode_size == 0:
            ep_pvals = self._p_buffer[-self.episode_size:]
            ep_violation_rate = sum(1 for p in ep_pvals if p < self.alpha) / len(ep_pvals)
            episode_alarm = self._episode_cusum.update(ep_violation_rate > self.alpha)

        alarm_scales = []
        if step_alarm:
            alarm_scales.append("step")
        if chunk_alarm:
            alarm_scales.append("chunk")
        if episode_alarm:
            alarm_scales.append("episode")

        return {
            "step": self._n_steps,
            "p_value": p_value,
            "any_alarm": len(alarm_scales) > 0,
            "alarm_scales": alarm_scales,
            "step_cusum": self._step_cusum.cusum_stat,
            "chunk_cusum": self._chunk_cusum.cusum_stat,
            "episode_cusum": self._episode_cusum.cusum_stat,
        }

    def get_summary(self) -> dict:
        """Get multi-scale CUSUM summary."""
        return {
            "n_steps": self._n_steps,
            "step_level": self._step_cusum.get_summary(),
            "chunk_level": self._chunk_cusum.get_summary(),
            "episode_level": self._episode_cusum.get_summary(),
            "total_alarms": (
                self._step_cusum.n_alarms
                + self._chunk_cusum.n_alarms
                + self._episode_cusum.n_alarms
            ),
        }

    def reset(self):
        """Reset all scales."""
        self._step_cusum.reset()
        self._chunk_cusum.reset()
        self._episode_cusum.reset()
        self._p_buffer.clear()
        self._n_steps = 0
