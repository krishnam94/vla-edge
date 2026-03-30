"""Base classes for hardware backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardwareCapabilities:
    """Describes what a hardware platform can do."""

    name: str
    compute_tops: float = 0.0
    memory_mb: int = 0
    supported_dtypes: list[str] = field(default_factory=lambda: ["fp32"])
    supported_formats: list[str] = field(default_factory=lambda: ["pytorch"])
    cuda_version: str | None = None
    tensorrt_version: str | None = None
    jetpack_version: str | None = None
    compute_capability: tuple[int, int] | None = None

    def supports_dtype(self, dtype: str) -> bool:
        return dtype in self.supported_dtypes

    def supports_format(self, fmt: str) -> bool:
        return fmt in self.supported_formats


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    actions: Any  # np.ndarray, but we avoid the import at module level
    latency_ms: float
    memory_peak_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class HardwareBackend(ABC):
    """Abstract base for hardware backends.

    Adding a new backend:
      1. Create a new file in backends/ (e.g., hailo.py)
      2. Subclass HardwareBackend and implement the 4 methods
      3. Decorate with @register_backend("hailo")
      4. That's it - no core code changes needed
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can run on the current machine."""

    @abstractmethod
    def get_capabilities(self) -> HardwareCapabilities:
        """Return hardware specs and supported formats/dtypes."""

    @abstractmethod
    def load_model(self, model_path: str, dtype: str = "fp16") -> Any:
        """Load a model in this backend's preferred format.

        Args:
            model_path: HuggingFace model ID or local path.
            dtype: Data type for model weights (fp32, fp16, int8, int4).

        Returns:
            A loaded model object (type depends on backend).
        """

    @abstractmethod
    def infer(self, model: Any, observation: dict[str, Any]) -> InferenceResult:
        """Run a single inference step.

        Args:
            model: A model loaded by this backend's load_model().
            observation: Dict with keys like 'image', 'instruction', 'state'.

        Returns:
            InferenceResult with actions, latency, and memory usage.
        """
