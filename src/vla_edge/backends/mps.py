"""MPS backend - Apple Silicon GPU acceleration (Mac M-series)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vla_edge.backends.base import HardwareBackend, HardwareCapabilities, InferenceResult
from vla_edge.registry import register_backend

try:
    import torch

    _HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    _HAS_MPS = False


@register_backend("mps")
class MPSBackend(HardwareBackend):
    """Apple Silicon MPS backend. Gives 5-10x over CPU on Mac M-series."""

    def is_available(self) -> bool:
        return _HAS_MPS

    def get_capabilities(self) -> HardwareCapabilities:
        if not _HAS_MPS:
            return HardwareCapabilities(name="mps-unavailable")

        import platform
        import subprocess

        # Get Mac model and memory
        try:
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            mem_mb = mem_bytes // (1024 * 1024)
        except Exception:
            mem_mb = 0

        return HardwareCapabilities(
            name=f"apple-{platform.processor() or 'silicon'}-mps",
            memory_mb=mem_mb,
            supported_dtypes=["fp32", "fp16"],
            supported_formats=["pytorch"],
        )

    def load_model(self, model_path: str, dtype: str = "fp16", trust_remote_code: bool = False) -> Any:
        """Load a PyTorch model on MPS (Apple GPU)."""
        import logging

        from transformers import AutoModel

        if trust_remote_code:
            logging.getLogger(__name__).warning("Loading with trust_remote_code=True from %s", model_path)

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
        torch_dtype = dtype_map.get(dtype, torch.float16)

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).to("mps")
        model.eval()
        return model

    def infer(self, model: Any, observation: dict[str, Any]) -> InferenceResult:
        """Run inference on MPS. Supports VLAModel adapters and raw PyTorch models."""
        image = observation.get("image")
        if image is None:
            raise ValueError("Observation must contain 'image' key")

        t0 = time.perf_counter()

        with torch.no_grad():
            if hasattr(model, "predict") and callable(model.predict):
                instruction = observation.get("instruction", "")
                state = observation.get("state")
                actions = model.predict(image, instruction, state)
            elif hasattr(model, "predict_action"):
                image_t = self._to_mps_tensor(image)
                actions = model.predict_action(image_t)
            elif hasattr(model, "forward"):
                image_t = self._to_mps_tensor(image)
                actions = model(image_t)
            else:
                raise ValueError("Model has no predict(), predict_action(), or forward()")

        # MPS doesn't have synchronize like CUDA, but we can force sync
        if _HAS_MPS:
            torch.mps.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # MPS doesn't expose memory stats like CUDA
        return InferenceResult(
            actions=actions,
            latency_ms=round(latency_ms, 2),
            memory_peak_mb=0.0,  # MPS has no memory tracking API
            metadata={"device": "mps", "dtype": "fp16", "memory_method": "unavailable"},
        )

    @staticmethod
    def _to_mps_tensor(image: Any) -> Any:
        """Convert numpy image to MPS tensor."""
        if isinstance(image, np.ndarray):
            t = torch.from_numpy(image).float().to("mps")
            if t.ndim == 3:
                t = t.permute(2, 0, 1).unsqueeze(0)
            return t
        return image
