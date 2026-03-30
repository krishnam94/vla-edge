"""Shared fixtures and hardware-aware test markers."""

from __future__ import annotations

import platform
from functools import wraps
from pathlib import Path

import numpy as np
import pytest


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_jetson() -> bool:
    return platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()


def _has_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


# --- Decorators for test skipping ---


def require_cuda(func):
    """Skip test if CUDA is not available."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _has_cuda():
            pytest.skip("requires CUDA GPU")
        return func(*args, **kwargs)

    return wrapper


def require_jetson(func):
    """Skip test if not running on Jetson hardware."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _is_jetson():
            pytest.skip("requires Jetson hardware")
        return func(*args, **kwargs)

    return wrapper


def require_tensorrt(func):
    """Skip test if TensorRT is not installed."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _has_tensorrt():
            pytest.skip("requires TensorRT")
        return func(*args, **kwargs)

    return wrapper


# --- Auto-skip via markers ---


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with @pytest.mark.gpu or @pytest.mark.jetson if hardware unavailable."""
    skip_gpu = pytest.mark.skip(reason="requires CUDA GPU")
    skip_jetson = pytest.mark.skip(reason="requires Jetson hardware")
    skip_tensorrt = pytest.mark.skip(reason="requires TensorRT")

    for item in items:
        if "gpu" in item.keywords and not _has_cuda():
            item.add_marker(skip_gpu)
        if "jetson" in item.keywords and not _is_jetson():
            item.add_marker(skip_jetson)
        if "tensorrt" in item.keywords and not _has_tensorrt():
            item.add_marker(skip_tensorrt)


# --- Fixtures ---


@pytest.fixture
def dummy_image() -> np.ndarray:
    """A 224x224 RGB image for testing."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_observation(dummy_image) -> dict:
    """A minimal observation dict for testing."""
    return {
        "image": dummy_image,
        "instruction": "pick up the red block",
        "state": np.zeros(7, dtype=np.float32),
    }


@pytest.fixture
def dummy_actions() -> np.ndarray:
    """A sequence of 10 actions with 7 dims for safety testing."""
    rng = np.random.default_rng(42)
    actions = np.cumsum(rng.normal(0, 0.02, (10, 7)), axis=0)
    return actions.astype(np.float32)
