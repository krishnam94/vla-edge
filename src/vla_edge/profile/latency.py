"""Latency profiling for VLA models."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def run_profile(
    model_name: str,
    backend_name: str = "auto",
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, Any]:
    """Profile a model's inference latency, memory, and throughput.

    Args:
        model_name: Name from registry or HuggingFace model ID.
        backend_name: Backend to use (auto, cpu, cuda, jetson).
        iterations: Number of timed iterations.
        warmup: Warmup iterations before timing starts.

    Returns:
        Dict with avg_ms, p50_ms, p95_ms, p99_ms, fps, peak_memory_mb, etc.
    """
    from vla_edge.registry import get_backend

    backend = get_backend(backend_name)
    caps = backend.get_capabilities()

    # Create a dummy observation for profiling
    dummy_obs = _create_dummy_observation()

    # Load model
    load_start = time.perf_counter()
    model = backend.load_model(model_name)
    load_time_s = time.perf_counter() - load_start

    # Warmup
    for _ in range(warmup):
        backend.infer(model, dummy_obs)

    # Timed iterations
    timings: list[float] = []
    peak_memory = 0.0

    for _ in range(iterations):
        result = backend.infer(model, dummy_obs)
        timings.append(result.latency_ms)
        peak_memory = max(peak_memory, result.memory_peak_mb)

    avg = sum(timings) / len(timings)
    timings_arr = np.array(timings)

    return {
        "model": model_name,
        "backend": caps.name,
        "iterations": iterations,
        "warmup": warmup,
        "load_time_s": round(load_time_s, 2),
        "avg_ms": round(avg, 2),
        "p50_ms": round(float(np.percentile(timings_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(timings_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(timings_arr, 99)), 2),
        "min_ms": round(float(timings_arr.min()), 2),
        "max_ms": round(float(timings_arr.max()), 2),
        "fps": round(1000 / avg, 1) if avg > 0 else 0,
        "peak_memory_mb": round(peak_memory, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _create_dummy_observation(
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """Create a dummy observation for profiling."""
    return {
        "image": np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8),
        "instruction": "pick up the red block",
        "state": np.zeros(7, dtype=np.float32),
    }
