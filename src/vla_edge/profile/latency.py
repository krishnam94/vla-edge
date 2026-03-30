"""Latency profiling for VLA models.

Fixes from critic review (Phase 1):
- GC disabled during timing to prevent jitter
- stddev and coefficient of variation reported
- Seeded RNG for reproducible dummy observations
- Accepts image_size parameter (for model-specific sizes)
- Warns if CV > 15% (unstable measurement)
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_profile(
    model_name: str,
    backend_name: str = "auto",
    iterations: int = 100,
    warmup: int = 10,
    image_size: tuple[int, int] = (224, 224),
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Profile a model's inference latency, memory, and throughput.

    Args:
        model_name: Name from registry or HuggingFace model ID.
        backend_name: Backend to use (auto, cpu, cuda, jetson).
        iterations: Number of timed iterations.
        warmup: Warmup iterations before timing starts.
        image_size: Image dimensions for dummy observation (H, W).
        trust_remote_code: Pass to backend.load_model() for HuggingFace models.

    Returns:
        Dict with avg_ms, p50_ms, p95_ms, p99_ms, fps, peak_memory_mb,
        stddev_ms, cv_percent, and stability warning if CV > 15%.
    """
    from vla_edge.registry import get_backend

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    backend = get_backend(backend_name)
    caps = backend.get_capabilities()

    # Try to get model's required image size from registry
    actual_image_size = image_size
    try:
        from vla_edge.registry import get_model

        model_cls = get_model(model_name)
        # Use static model_info() if available, else fallback to info property
        if hasattr(model_cls, "model_info") and callable(model_cls.model_info):
            model_info = model_cls.model_info()
        else:
            model_info = model_cls.__new__(model_cls).info
        actual_image_size = model_info.required_image_size
        if actual_image_size != image_size and image_size == (224, 224):
            logger.info(
                "Using model's required image size %s instead of default 224x224",
                actual_image_size,
            )
    except (KeyError, AttributeError, Exception):
        pass

    # Create a seeded dummy observation for reproducibility
    dummy_obs = _create_dummy_observation(image_size=actual_image_size, seed=42)

    # Load model: prefer VLAModel adapter from registry, fall back to backend
    load_start = time.perf_counter()
    try:
        from vla_edge.registry import get_model

        model_cls = get_model(model_name)
        device = "cuda" if backend_name in ("cuda", "jetson") else "cpu"
        model = model_cls(device=device)
        logger.info("Loaded %s via model registry (VLAModel adapter)", model_name)
    except (KeyError, TypeError):
        # Not in registry or adapter failed - use backend's generic loader
        model = backend.load_model(model_name, trust_remote_code=trust_remote_code)
        logger.info("Loaded %s via backend.load_model()", model_name)
    load_time_s = time.perf_counter() - load_start

    # Warmup (GC enabled)
    for _ in range(warmup):
        backend.infer(model, dummy_obs)

    # Disable GC during timed iterations to prevent jitter
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        timings: list[float] = []
        peak_memory = 0.0

        for _ in range(iterations):
            result = backend.infer(model, dummy_obs)
            timings.append(result.latency_ms)
            peak_memory = max(peak_memory, result.memory_peak_mb)
    finally:
        if gc_was_enabled:
            gc.enable()

    timings_arr = np.array(timings)
    avg = float(timings_arr.mean())
    stddev = float(timings_arr.std())
    cv = (stddev / avg * 100) if avg > 0 else 0

    profile_result = {
        "model": model_name,
        "backend": caps.name,
        "iterations": iterations,
        "warmup": warmup,
        "image_size": list(actual_image_size),
        "load_time_s": round(load_time_s, 2),
        "avg_ms": round(avg, 2),
        "stddev_ms": round(stddev, 2),
        "cv_percent": round(cv, 1),
        "p50_ms": round(float(np.percentile(timings_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(timings_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(timings_arr, 99)), 2),
        "min_ms": round(float(timings_arr.min()), 2),
        "max_ms": round(float(timings_arr.max()), 2),
        "fps": round(1000 / avg, 1) if avg > 0 else 0,
        "peak_memory_mb": round(peak_memory, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Cleanup model resources (addresses #4)
    if hasattr(model, "cleanup"):
        model.cleanup()

    if cv > 15:
        profile_result["warning"] = (
            f"High variance (CV={cv:.1f}%). Results may be unreliable. "
            "Consider more iterations or check for thermal throttling."
        )
        logger.warning("Profiling variance is high (CV=%.1f%%)", cv)

    return profile_result


def _create_dummy_observation(
    image_size: tuple[int, int] = (224, 224),
    seed: int = 42,
) -> dict[str, Any]:
    """Create a seeded dummy observation for reproducible profiling."""
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(0, 255, (*image_size, 3), dtype=np.uint8),
        "instruction": "pick up the red block",
        "state": np.zeros(7, dtype=np.float32),
    }
