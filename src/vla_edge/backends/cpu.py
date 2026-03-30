"""CPU backend - always available, works on Mac and Linux."""

from __future__ import annotations

import logging
import platform
import time
from typing import Any

import numpy as np
import psutil

from vla_edge.backends.base import HardwareBackend, HardwareCapabilities, InferenceResult
from vla_edge.registry import register_backend

logger = logging.getLogger(__name__)


@register_backend("cpu")
class CPUBackend(HardwareBackend):
    """CPU-only backend. Always available. Used for baseline profiling and development."""

    def is_available(self) -> bool:
        return True

    def get_capabilities(self) -> HardwareCapabilities:
        mem = psutil.virtual_memory()
        return HardwareCapabilities(
            name=f"cpu-{platform.machine()}",
            memory_mb=mem.total // (1024 * 1024),
            supported_dtypes=["fp32"],
            supported_formats=["pytorch", "onnx"],
        )

    def load_model(self, model_path: str, dtype: str = "fp32", trust_remote_code: bool = False) -> Any:
        """Load a PyTorch model on CPU."""
        try:
            import torch
            from transformers import AutoModel

            if trust_remote_code:
                logger.warning(
                    "Loading model with trust_remote_code=True - executing remote code from %s", model_path
                )

            torch_dtype = torch.float32
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            model.eval()
            return model
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for CPU backend. Install with: pip install vla-edge[pytorch]"
            ) from e

    def infer(self, model: Any, observation: dict[str, Any]) -> InferenceResult:
        """Run inference on CPU."""
        import torch

        image = observation.get("image")
        if image is not None and isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.ndim == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW

        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()

        with torch.no_grad():
            if hasattr(model, "predict_action"):
                actions = model.predict_action(image)
            elif hasattr(model, "forward"):
                actions = model(image)
            else:
                raise ValueError("Model has no predict_action() or forward() method")

        latency_ms = (time.perf_counter() - t0) * 1000
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        return InferenceResult(
            actions=actions,
            latency_ms=round(latency_ms, 2),
            memory_peak_mb=round(max(0, mem_after - mem_before), 1),
            metadata={"device": "cpu", "dtype": "fp32"},
        )
