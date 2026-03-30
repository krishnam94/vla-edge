"""CUDA backend - desktop/server GPUs."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vla_edge.backends.base import HardwareBackend, HardwareCapabilities, InferenceResult
from vla_edge.registry import register_backend

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


@register_backend("cuda")
class CUDABackend(HardwareBackend):
    """CUDA backend for desktop/server NVIDIA GPUs (not Jetson)."""

    def is_available(self) -> bool:
        if not _HAS_CUDA:
            return False
        # Jetson is handled by the JetsonBackend
        import platform

        return platform.machine() != "aarch64"

    def get_capabilities(self) -> HardwareCapabilities:
        if not _HAS_CUDA:
            return HardwareCapabilities(name="cuda-unavailable")

        props = torch.cuda.get_device_properties(0)
        return HardwareCapabilities(
            name=torch.cuda.get_device_name(0),
            memory_mb=props.total_memory // (1024 * 1024),
            supported_dtypes=["fp32", "fp16", "bf16", "int8"],
            supported_formats=["pytorch", "onnx", "tensorrt"],
            cuda_version=torch.version.cuda,
            compute_capability=(props.major, props.minor),
        )

    def load_model(self, model_path: str, dtype: str = "fp16") -> Any:
        """Load a PyTorch model on CUDA."""
        from transformers import AutoModel

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda:0")
        model.eval()
        return model

    def infer(self, model: Any, observation: dict[str, Any]) -> InferenceResult:
        """Run inference on CUDA GPU."""
        image = observation.get("image")
        if image is not None and isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().to("cuda:0")
            if image.ndim == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if hasattr(model, "predict_action"):
                actions = model.predict_action(image)
            elif hasattr(model, "forward"):
                actions = model(image)
            else:
                raise ValueError("Model has no predict_action() or forward() method")

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        return InferenceResult(
            actions=actions,
            latency_ms=round(latency_ms, 2),
            memory_peak_mb=round(peak_mem, 1),
            metadata={
                "device": torch.cuda.get_device_name(0),
                "dtype": str(model.dtype) if hasattr(model, "dtype") else "unknown",
                "cuda_version": torch.version.cuda,
            },
        )
