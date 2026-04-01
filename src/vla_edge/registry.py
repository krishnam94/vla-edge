"""Backend and model registries with auto-detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vla_edge.backends.base import HardwareBackend
    from vla_edge.models.base import VLAModel

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type[HardwareBackend]] = {}
_MODELS: dict[str, type[VLAModel]] = {}
_backends_loaded = False
_models_loaded = False

# Priority order for auto-detection (highest to lowest)
_BACKEND_PRIORITY = ["jetson", "cuda", "mps", "cpu"]


def register_backend(name: str):
    """Decorator to register a hardware backend.

    Usage:
        @register_backend("jetson")
        class JetsonBackend(HardwareBackend):
            ...
    """

    def decorator(cls: type[HardwareBackend]) -> type[HardwareBackend]:
        _BACKENDS[name] = cls
        logger.debug("Registered backend: %s", name)
        return cls

    return decorator


def register_model(name: str):
    """Decorator to register a VLA model adapter.

    Usage:
        @register_model("smolvla")
        class SmolVLAModel(VLAModel):
            ...
    """

    def decorator(cls: type[VLAModel]) -> type[VLAModel]:
        _MODELS[name] = cls
        logger.debug("Registered model: %s", name)
        return cls

    return decorator


def get_backend(name: str = "auto") -> HardwareBackend:
    """Get a hardware backend by name, or auto-detect the best available.

    Auto-detection priority: jetson -> cuda -> mps -> cpu
    """
    _ensure_backends_loaded()

    if name == "auto":
        for backend_name in _BACKEND_PRIORITY:
            if backend_name in _BACKENDS:
                try:
                    backend = _BACKENDS[backend_name]()
                    if backend.is_available():
                        logger.info("Auto-detected backend: %s", backend_name)
                        return backend
                except Exception:
                    logger.debug("Backend %s failed availability check", backend_name)
                    continue
        raise RuntimeError("No available backend. Registered: " + ", ".join(_BACKENDS.keys()))

    if name not in _BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Available: {list(_BACKENDS.keys())}")
    return _BACKENDS[name]()


def get_model(name: str) -> type[VLAModel]:
    """Get a VLA model adapter class by name."""
    _ensure_models_loaded()

    if name not in _MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_MODELS.keys())}")
    return _MODELS[name]


def list_backends() -> dict[str, bool]:
    """Return all registered backends and their availability."""
    _ensure_backends_loaded()
    result = {}
    for name, cls in _BACKENDS.items():
        try:
            result[name] = cls().is_available()
        except Exception:
            result[name] = False
    return result


def list_models() -> list[str]:
    """Return all registered model names."""
    _ensure_models_loaded()
    return list(_MODELS.keys())


def _ensure_backends_loaded() -> None:
    """Import backend modules to trigger @register_backend decorators."""
    global _backends_loaded
    if _backends_loaded:
        return
    _backends_loaded = True
    import contextlib

    import vla_edge.backends.cpu

    with contextlib.suppress(ImportError):
        import vla_edge.backends.cuda

    with contextlib.suppress(ImportError):
        import vla_edge.backends.mps

    with contextlib.suppress(ImportError):
        import vla_edge.backends.jetson  # noqa: F401


def _ensure_models_loaded() -> None:
    """Import model modules to trigger @register_model decorators."""
    global _models_loaded
    if _models_loaded:
        return
    _models_loaded = True
    import contextlib

    with contextlib.suppress(ImportError):
        import vla_edge.models.smolvla

    with contextlib.suppress(ImportError):
        import vla_edge.models.openvla

    with contextlib.suppress(ImportError):
        import vla_edge.models.act_adapter

    with contextlib.suppress(ImportError):
        import vla_edge.models.diffusion_adapter  # noqa: F401
