"""Rich monitoring signals for VLA action streams.

Beyond bounds and velocity, these monitors capture temporal, spectral,
structural, and infrastructure health signals. Each runs as a separate
channel alongside conformal p-values (preserving conformal guarantees).

Monitors:
    LinearPredictabilityMonitor - AR(1) prediction error (temporal coherence)
    CorrelationDivergenceMonitor - Inter-joint coordination health
    SpectralEnergyMonitor - Frequency-domain action quality
    MomentumCoherenceMonitor - Trajectory direction consistency
    LatencyMonitor - Inference timing and deadline compliance
    ActionEnergyMonitor - Total action magnitude tracking

Design note: these are NOT fed into the conformal nonconformity score
(that would break exchangeability). They run as parallel channels in
SafetyGuard, and composite AUROC is computed post-hoc.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class LinearPredictabilityMonitor:
    """AR(1) prediction error: how surprising is this action given the last one?

    Fits per-joint linear coefficients from calibration:
        a_t_predicted = alpha * a_{t-1} + beta

    The prediction error measures temporal coherence. High error means
    the action is unpredictable from recent history - either a policy
    failure or a legitimate rapid change.

    Uses per-joint AR(1) (D parameters, not D*p) with ridge regularization
    to avoid overfitting on small calibration sets.
    """

    def __init__(self, ridge_alpha: float = 1.0):
        self.ridge_alpha = ridge_alpha
        self._coeffs: np.ndarray | None = None  # (D, 2) per-joint [alpha, beta]
        self._prev_action: np.ndarray | None = None
        self._n_steps: int = 0
        self._errors: list[float] = []

    def calibrate(self, actions: np.ndarray):
        """Fit per-joint AR(1) from consecutive calibration actions."""
        n, d = actions.shape
        self._coeffs = np.zeros((d, 2), dtype=np.float32)

        for j in range(d):
            x = actions[:-1, j]  # a_{t-1}
            y = actions[1:, j]   # a_t
            # Ridge regression: [alpha, beta] = (X^T X + lambda I)^{-1} X^T y
            X = np.column_stack([x, np.ones(len(x))])
            XtX = X.T @ X + self.ridge_alpha * np.eye(2)
            Xty = X.T @ y
            self._coeffs[j] = np.linalg.solve(XtX, Xty)

    def update(self, action: np.ndarray) -> dict:
        self._n_steps += 1

        if self._prev_action is None or self._coeffs is None:
            self._prev_action = action.copy()
            return {"prediction_error": 0.0, "max_error": 0.0}

        # Predict: a_t = alpha * a_{t-1} + beta
        predicted = self._coeffs[:, 0] * self._prev_action + self._coeffs[:, 1]
        error = np.abs(action - predicted)
        max_error = float(np.max(error))
        mean_error = float(np.mean(error))

        self._errors.append(max_error)
        self._prev_action = action.copy()

        return {"prediction_error": mean_error, "max_error": max_error}

    def get_summary(self) -> dict:
        if not self._errors:
            return {"n_steps": self._n_steps}
        return {
            "n_steps": self._n_steps,
            "mean_prediction_error": float(np.mean(self._errors)),
            "max_prediction_error": float(np.max(self._errors)),
            "std_prediction_error": float(np.std(self._errors)),
        }

    def reset(self):
        self._prev_action = None
        self._n_steps = 0
        self._errors.clear()


class CorrelationDivergenceMonitor:
    """Inter-joint coordination health via correlation matrix divergence.

    Tracks the running correlation matrix between joints and compares
    to the calibration correlation structure. When joints lose coordination
    (common in OOD scenarios), the divergence spikes before any single-joint
    monitor fires.

    Uses Ledoit-Wolf shrinkage for stable covariance estimation.
    Requires D >= 4 dimensions to be meaningful (skip for PushT 2D).
    """

    def __init__(self, window_size: int = 50, min_dims: int = 4):
        self.window_size = window_size
        self.min_dims = min_dims
        self._action_buffer: deque = deque(maxlen=window_size)
        self._cal_corr: np.ndarray | None = None
        self._n_steps: int = 0
        self._divergences: list[float] = []
        self._enabled: bool = True

    def calibrate(self, actions: np.ndarray):
        """Compute calibration correlation matrix with shrinkage."""
        _, d = actions.shape
        if d < self.min_dims:
            self._enabled = False
            return

        vels = np.diff(actions, axis=0)
        # Ledoit-Wolf shrinkage for stable correlation
        self._cal_corr = self._shrunk_corr(vels)

    def _shrunk_corr(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation with Ledoit-Wolf-style shrinkage."""
        n, d = data.shape
        if n < 3:
            return np.eye(d)

        # Simple shrinkage: alpha * sample_corr + (1-alpha) * I
        sample = np.corrcoef(data.T)
        # Handle NaN from constant columns
        sample = np.nan_to_num(sample, nan=0.0)
        alpha = min(n / (n + d), 0.9)  # shrink more when samples < dims
        return alpha * sample + (1 - alpha) * np.eye(d)

    def update(self, action: np.ndarray) -> dict:
        self._n_steps += 1

        if not self._enabled:
            return {"divergence": 0.0, "enabled": False}

        self._action_buffer.append(action.copy())

        if len(self._action_buffer) < max(10, self.min_dims + 2):
            return {"divergence": 0.0, "enabled": True}

        window = np.array(self._action_buffer)
        vels = np.diff(window, axis=0)
        if vels.shape[0] < 3:
            return {"divergence": 0.0, "enabled": True}

        curr_corr = self._shrunk_corr(vels)
        divergence = float(np.linalg.norm(curr_corr - self._cal_corr, "fro"))
        d = curr_corr.shape[0]
        normalized = divergence / d  # normalize by matrix size

        self._divergences.append(normalized)

        return {"divergence": normalized, "enabled": True}

    def get_summary(self) -> dict:
        if not self._divergences:
            return {"n_steps": self._n_steps, "enabled": self._enabled}
        return {
            "n_steps": self._n_steps,
            "enabled": self._enabled,
            "mean_divergence": float(np.mean(self._divergences)),
            "max_divergence": float(np.max(self._divergences)),
        }

    def reset(self):
        self._action_buffer.clear()
        self._n_steps = 0
        self._divergences.clear()


class SpectralEnergyMonitor:
    """Frequency-domain health: ratio of low-frequency to total energy.

    Healthy policies produce smooth (low-frequency) action trajectories.
    Degrading policies inject high-frequency energy from denoising artifacts,
    chunk boundary discontinuities, or emerging instability.

    Uses Welch-style averaging for spectral stability.

    References:
        FreqPolicy (NeurIPS 2025) - validates frequency decomposition
        of robot actions captures meaningful structure.
    """

    def __init__(self, window_size: int = 32, freq_cutoff: float = 0.25):
        self.window_size = window_size
        self.freq_cutoff = freq_cutoff  # fraction of Nyquist
        self._action_buffer: deque = deque(maxlen=window_size)
        self._n_steps: int = 0
        self._ser_history: list[float] = []

    def update(self, action: np.ndarray) -> dict:
        self._n_steps += 1
        self._action_buffer.append(action.copy())

        if len(self._action_buffer) < self.window_size:
            return {"ser": 1.0, "high_freq_ratio": 0.0}

        window = np.array(self._action_buffer)
        # Apply Hann window for spectral leakage reduction
        hann = np.hanning(self.window_size).reshape(-1, 1)
        windowed = window * hann

        # FFT per joint, average energy across joints
        fft_energy = np.abs(np.fft.rfft(windowed, axis=0)) ** 2
        n_freqs = fft_energy.shape[0]
        cutoff_idx = max(1, int(n_freqs * self.freq_cutoff))

        low_energy = float(fft_energy[:cutoff_idx].sum())
        total_energy = float(fft_energy.sum()) + 1e-10

        ser = low_energy / total_energy  # 1.0 = all low freq (healthy)
        high_freq_ratio = 1.0 - ser

        self._ser_history.append(ser)

        return {"ser": ser, "high_freq_ratio": high_freq_ratio}

    def get_summary(self) -> dict:
        if not self._ser_history:
            return {"n_steps": self._n_steps}
        return {
            "n_steps": self._n_steps,
            "mean_ser": float(np.mean(self._ser_history)),
            "min_ser": float(np.min(self._ser_history)),
            "mean_high_freq": float(1.0 - np.mean(self._ser_history)),
        }

    def reset(self):
        self._action_buffer.clear()
        self._n_steps = 0
        self._ser_history.clear()


class MomentumCoherenceMonitor:
    """Trajectory direction consistency via cosine similarity.

    Compares the current velocity vector to the recent trajectory momentum
    (average velocity over a window). A sudden direction reversal (coherence
    drops to -1) indicates the policy changed its mind, which is often
    a precursor to failure.

    More informative than per-joint direction reversal counting because
    it captures multi-dimensional directional coherence.
    """

    def __init__(self, momentum_window: int = 5):
        self.momentum_window = momentum_window
        self._action_buffer: deque = deque(maxlen=momentum_window + 1)
        self._n_steps: int = 0
        self._coherence_history: list[float] = []

    def update(self, action: np.ndarray) -> dict:
        self._n_steps += 1
        self._action_buffer.append(action.copy())

        if len(self._action_buffer) < self.momentum_window + 1:
            return {"coherence": 1.0, "reversal": False}

        buf = np.array(self._action_buffer)
        # Momentum: average velocity over recent window (excluding current)
        vels = np.diff(buf[:-1], axis=0)
        momentum = vels.mean(axis=0)

        # Current velocity
        current_vel = buf[-1] - buf[-2]

        m_norm = np.linalg.norm(momentum)
        c_norm = np.linalg.norm(current_vel)

        if m_norm < 1e-8 or c_norm < 1e-8:
            coherence = 1.0  # near-zero motion, not a reversal
        else:
            coherence = float(np.dot(momentum, current_vel) / (m_norm * c_norm))

        reversal = coherence < -0.5
        self._coherence_history.append(coherence)

        return {"coherence": coherence, "reversal": reversal}

    def get_summary(self) -> dict:
        if not self._coherence_history:
            return {"n_steps": self._n_steps}
        return {
            "n_steps": self._n_steps,
            "mean_coherence": float(np.mean(self._coherence_history)),
            "min_coherence": float(np.min(self._coherence_history)),
            "reversal_rate": float(np.mean([c < -0.5 for c in self._coherence_history])),
        }

    def reset(self):
        self._action_buffer.clear()
        self._n_steps = 0
        self._coherence_history.clear()


class LatencyMonitor:
    """Inference timing and control deadline compliance.

    Tracks the wall-clock time between consecutive update() calls.
    If inference takes longer than the control deadline, the robot
    misses its next control step - a safety-critical failure that
    no action-content monitor can detect.

    The simplest and most universally useful monitor. Nobody currently
    tracks this for VLA deployment.
    """

    def __init__(self, deadline_ms: float = 100.0):
        self.deadline_ms = deadline_ms
        self._last_time: float | None = None
        self._n_steps: int = 0
        self._latencies: list[float] = []
        self._deadline_violations: int = 0

    def update(self, action: np.ndarray = None) -> dict:
        """Call this on every inference step. Action arg is ignored (timing only)."""
        self._n_steps += 1
        now = time.monotonic()

        if self._last_time is None:
            self._last_time = now
            return {"latency_ms": 0.0, "deadline_violation": False}

        latency_ms = (now - self._last_time) * 1000.0
        self._latencies.append(latency_ms)
        self._last_time = now

        violation = latency_ms > self.deadline_ms
        if violation:
            self._deadline_violations += 1

        return {
            "latency_ms": latency_ms,
            "deadline_violation": violation,
        }

    def get_summary(self) -> dict:
        if not self._latencies:
            return {"n_steps": self._n_steps}
        lats = np.array(self._latencies)
        return {
            "n_steps": self._n_steps,
            "mean_latency_ms": float(lats.mean()),
            "max_latency_ms": float(lats.max()),
            "p99_latency_ms": float(np.percentile(lats, 99)),
            "jitter_ms": float(lats.std()),
            "deadline_violations": self._deadline_violations,
            "deadline_violation_rate": self._deadline_violations / max(len(lats), 1),
        }

    def reset(self):
        self._last_time = None
        self._n_steps = 0
        self._latencies.clear()
        self._deadline_violations = 0


class ActionEnergyMonitor:
    """Track total action magnitude as a proxy for energy/effort.

    Monitors `sum(abs(action))` per step. Catches both extremes:
    - Very high energy: thrashing, oscillating, wasted motion
    - Very low energy: stalling (complementary to StallDetector)

    The calibration baseline gives the expected energy level.
    Significant deviation indicates abnormal behavior.
    """

    def __init__(self, ewma_alpha: float = 0.1):
        self.ewma_alpha = ewma_alpha
        self._cal_mean_energy: float = 0.0
        self._cal_std_energy: float = 1.0
        self._ewma_energy: float = 0.0
        self._n_steps: int = 0
        self._energies: list[float] = []

    def calibrate(self, actions: np.ndarray):
        """Compute baseline energy from calibration data."""
        energies = np.sum(np.abs(actions), axis=1)
        self._cal_mean_energy = float(energies.mean())
        self._cal_std_energy = float(max(energies.std(), 1e-8))
        self._ewma_energy = self._cal_mean_energy

    def update(self, action: np.ndarray) -> dict:
        self._n_steps += 1
        energy = float(np.sum(np.abs(action)))
        self._energies.append(energy)

        # EWMA tracking
        self._ewma_energy = (
            self.ewma_alpha * energy + (1 - self.ewma_alpha) * self._ewma_energy
        )

        # Z-score relative to calibration
        z_score = (energy - self._cal_mean_energy) / self._cal_std_energy

        return {
            "energy": energy,
            "ewma_energy": self._ewma_energy,
            "energy_zscore": z_score,
            "anomalous": abs(z_score) > 3.0,
        }

    def get_summary(self) -> dict:
        if not self._energies:
            return {"n_steps": self._n_steps}
        return {
            "n_steps": self._n_steps,
            "mean_energy": float(np.mean(self._energies)),
            "std_energy": float(np.std(self._energies)),
            "cal_mean_energy": self._cal_mean_energy,
            "energy_drift": float(np.mean(self._energies) - self._cal_mean_energy),
        }

    def reset(self):
        self._ewma_energy = self._cal_mean_energy
        self._n_steps = 0
        self._energies.clear()
