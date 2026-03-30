"""Tests for backend and model registries."""

from vla_edge.backends.base import HardwareBackend, HardwareCapabilities, InferenceResult
from vla_edge.registry import (
    _BACKENDS,
    get_backend,
    list_backends,
    register_backend,
)


def test_register_and_get_backend():
    """Test registering a custom backend and retrieving it."""

    @register_backend("test-backend")
    class TestBackend(HardwareBackend):
        def is_available(self):
            return True

        def get_capabilities(self):
            return HardwareCapabilities(name="test")

        def load_model(self, model_path, dtype="fp16"):
            return None

        def infer(self, model, observation):
            return InferenceResult(actions=None, latency_ms=0)

    assert "test-backend" in _BACKENDS
    backend = get_backend("test-backend")
    assert backend.is_available()
    assert backend.get_capabilities().name == "test"

    # Cleanup
    del _BACKENDS["test-backend"]


def test_cpu_backend_always_available():
    """CPU backend should always be available."""
    backends = list_backends()
    assert "cpu" in backends
    assert backends["cpu"] is True


def test_auto_detect_returns_something():
    """Auto-detect should return at least the CPU backend."""
    backend = get_backend("auto")
    assert backend is not None
    caps = backend.get_capabilities()
    assert caps.name != ""


def test_unknown_backend_raises():
    """Requesting an unknown backend should raise KeyError."""
    import pytest

    with pytest.raises(KeyError, match="Unknown backend"):
        get_backend("nonexistent-backend")


def test_list_backends_returns_dict():
    """list_backends should return a dict of name -> bool."""
    backends = list_backends()
    assert isinstance(backends, dict)
    assert all(isinstance(v, bool) for v in backends.values())


def test_hardware_capabilities_helpers():
    """Test HardwareCapabilities utility methods."""
    caps = HardwareCapabilities(
        name="test",
        supported_dtypes=["fp32", "fp16", "int8"],
        supported_formats=["pytorch", "onnx"],
    )
    assert caps.supports_dtype("fp16")
    assert not caps.supports_dtype("int4")
    assert caps.supports_format("pytorch")
    assert not caps.supports_format("tensorrt")
