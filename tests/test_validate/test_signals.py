"""Tests for rich monitoring signals."""

import time

import numpy as np

from vla_edge.validate.signals import (
    ActionEnergyMonitor,
    CorrelationDivergenceMonitor,
    LatencyMonitor,
    LinearPredictabilityMonitor,
    MomentumCoherenceMonitor,
    SpectralEnergyMonitor,
)


class TestLinearPredictabilityMonitor:
    def test_calibrate_and_update(self):
        rng = np.random.RandomState(42)
        demos = np.cumsum(rng.randn(100, 7) * 0.1, axis=0).astype(np.float32)
        lps = LinearPredictabilityMonitor()
        lps.calibrate(demos)
        assert lps._coeffs is not None
        assert lps._coeffs.shape == (7, 2)

        for a in demos[:10]:
            r = lps.update(a)
        assert "prediction_error" in r
        assert r["prediction_error"] >= 0

    def test_smooth_trajectory_low_error(self):
        demos = np.cumsum(np.ones((100, 3)) * 0.01, axis=0).astype(np.float32)
        lps = LinearPredictabilityMonitor()
        lps.calibrate(demos)
        for a in demos[50:60]:
            r = lps.update(a)
        assert r["max_error"] < 0.1

    def test_erratic_trajectory_high_error(self):
        rng = np.random.RandomState(42)
        demos = np.cumsum(rng.randn(100, 3) * 0.01, axis=0).astype(np.float32)
        lps = LinearPredictabilityMonitor()
        lps.calibrate(demos)
        # Feed something completely different
        for _ in range(5):
            lps.update(rng.randn(3).astype(np.float32) * 10)
        s = lps.get_summary()
        assert s["mean_prediction_error"] > 1.0

    def test_reset(self):
        lps = LinearPredictabilityMonitor()
        lps._n_steps = 10
        lps.reset()
        assert lps._n_steps == 0


class TestCorrelationDivergenceMonitor:
    def test_skip_low_dims(self):
        demos = np.random.randn(100, 2).astype(np.float32)
        cd = CorrelationDivergenceMonitor(min_dims=4)
        cd.calibrate(demos)
        assert not cd._enabled
        r = cd.update(np.zeros(2, dtype=np.float32))
        assert not r["enabled"]

    def test_works_with_high_dims(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32)
        cd = CorrelationDivergenceMonitor(window_size=50, min_dims=4)
        cd.calibrate(demos)
        assert cd._enabled

        for a in demos[:60]:
            r = cd.update(a)
        assert r["divergence"] >= 0

    def test_divergence_increases_on_shift(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(200, 7).astype(np.float32)
        cd = CorrelationDivergenceMonitor(window_size=30, min_dims=4)
        cd.calibrate(demos)

        # In-distribution
        for a in demos[:40]:
            cd.update(a)
        s1 = cd.get_summary()

        # Shifted (destroy correlations)
        cd.reset()
        for _ in range(40):
            cd.update(rng.randn(7).astype(np.float32) * 5)
        s2 = cd.get_summary()

        if s1.get("mean_divergence") and s2.get("mean_divergence"):
            assert s2["mean_divergence"] > s1["mean_divergence"]


class TestSpectralEnergyMonitor:
    def test_smooth_signal_high_ser(self):
        ser_mon = SpectralEnergyMonitor(window_size=32)
        # Low-frequency sine wave
        for i in range(40):
            action = np.array([np.sin(i * 0.1), np.cos(i * 0.1)], dtype=np.float32)
            r = ser_mon.update(action)
        assert r["ser"] > 0.7  # mostly low frequency

    def test_noisy_signal_low_ser(self):
        rng = np.random.RandomState(42)
        ser_mon = SpectralEnergyMonitor(window_size=32)
        for _ in range(40):
            action = rng.randn(2).astype(np.float32) * 10  # white noise
            r = ser_mon.update(action)
        assert r["ser"] < 0.5  # high frequency content

    def test_summary(self):
        ser_mon = SpectralEnergyMonitor(window_size=16)
        rng = np.random.RandomState(42)
        for _ in range(20):
            ser_mon.update(rng.randn(3).astype(np.float32))
        s = ser_mon.get_summary()
        assert "mean_ser" in s


class TestMomentumCoherenceMonitor:
    def test_consistent_direction_high_coherence(self):
        mc = MomentumCoherenceMonitor(momentum_window=5)
        # Straight line trajectory
        for i in range(20):
            action = np.array([float(i), float(i)], dtype=np.float32)
            r = mc.update(action)
        assert r["coherence"] > 0.9
        assert not r["reversal"]

    def test_reversal_detected(self):
        mc = MomentumCoherenceMonitor(momentum_window=5)
        # Go forward for a while (build momentum)
        for i in range(15):
            mc.update(np.array([float(i) * 10, 0.0], dtype=np.float32))
        # Sudden large reverse step
        r = mc.update(np.array([-100.0, 0.0], dtype=np.float32))
        assert r["coherence"] < 0
        assert r["reversal"]

    def test_summary(self):
        mc = MomentumCoherenceMonitor()
        for i in range(20):
            mc.update(np.array([float(i), 0.0], dtype=np.float32))
        s = mc.get_summary()
        assert "mean_coherence" in s
        assert "reversal_rate" in s


class TestLatencyMonitor:
    def test_tracks_latency(self):
        lm = LatencyMonitor(deadline_ms=50.0)
        lm.update()
        time.sleep(0.01)  # 10ms
        r = lm.update()
        assert r["latency_ms"] > 5  # should be ~10ms
        assert not r["deadline_violation"]

    def test_deadline_violation(self):
        lm = LatencyMonitor(deadline_ms=1.0)  # 1ms deadline
        lm.update()
        time.sleep(0.01)  # 10ms > 1ms deadline
        r = lm.update()
        assert r["deadline_violation"]

    def test_summary(self):
        lm = LatencyMonitor()
        for _ in range(5):
            lm.update()
            time.sleep(0.001)
        s = lm.get_summary()
        assert "mean_latency_ms" in s
        assert "jitter_ms" in s

    def test_reset(self):
        lm = LatencyMonitor()
        lm.update()
        lm.update()
        lm.reset()
        assert lm._n_steps == 0


class TestActionEnergyMonitor:
    def test_calibrate_and_update(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(100, 7).astype(np.float32) * 0.5
        ae = ActionEnergyMonitor()
        ae.calibrate(demos)
        assert ae._cal_mean_energy > 0

        r = ae.update(rng.randn(7).astype(np.float32) * 0.5)
        assert "energy" in r
        assert "energy_zscore" in r

    def test_high_energy_flagged(self):
        demos = np.ones((100, 3), dtype=np.float32) * 0.1
        ae = ActionEnergyMonitor()
        ae.calibrate(demos)

        r = ae.update(np.ones(3, dtype=np.float32) * 100)
        assert r["anomalous"]
        assert r["energy_zscore"] > 3.0

    def test_normal_energy_not_flagged(self):
        rng = np.random.RandomState(42)
        demos = rng.randn(100, 3).astype(np.float32)
        ae = ActionEnergyMonitor()
        ae.calibrate(demos)

        r = ae.update(rng.randn(3).astype(np.float32))
        assert not r["anomalous"]

    def test_reset(self):
        ae = ActionEnergyMonitor()
        ae._n_steps = 10
        ae.reset()
        assert ae._n_steps == 0
