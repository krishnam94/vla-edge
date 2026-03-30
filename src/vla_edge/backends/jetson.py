"""Jetson backend - NVIDIA Jetson Orin Nano, AGX Orin, Thor."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from vla_edge.backends.base import HardwareBackend, HardwareCapabilities, InferenceResult
from vla_edge.registry import register_backend

_IS_JETSON = platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


def _get_jetpack_version() -> str | None:
    """Read JetPack/L4T version from Jetson system files."""
    try:
        text = Path("/etc/nv_tegra_release").read_text().strip()
        # Format: "# R36 (release), REVISION: 4.0, ..."
        return text.split(",")[0].replace("# ", "").strip()
    except (FileNotFoundError, IndexError):
        return None


def _get_jetson_model() -> str:
    """Detect Jetson module name."""
    try:
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            return model_path.read_text().strip().rstrip("\x00")
    except Exception:
        pass
    return "jetson-unknown"


@register_backend("jetson")
class JetsonBackend(HardwareBackend):
    """Jetson backend with Jetson-specific monitoring (jtop, tegrastats, nvpmodel)."""

    def is_available(self) -> bool:
        return _IS_JETSON and _HAS_CUDA

    def get_capabilities(self) -> HardwareCapabilities:
        if not self.is_available():
            return HardwareCapabilities(name="jetson-unavailable")

        props = torch.cuda.get_device_properties(0)
        jetpack = _get_jetpack_version()
        model_name = _get_jetson_model()

        return HardwareCapabilities(
            name=model_name,
            memory_mb=props.total_memory // (1024 * 1024),
            supported_dtypes=["fp32", "fp16", "int8", "int4"],
            supported_formats=["pytorch", "gguf", "tensorrt", "onnx"],
            cuda_version=torch.version.cuda,
            jetpack_version=jetpack,
            compute_capability=(props.major, props.minor),
        )

    def load_model(self, model_path: str, dtype: str = "fp16", trust_remote_code: bool = False) -> Any:
        """Load model on Jetson GPU.

        For Jetson Orin Nano (8GB): prefer GGUF Q4 or FP16 models under 4B params.
        TensorRT-LLM is NOT supported on Orin Nano - use llama.cpp/GGUF instead.

        TODO (Phase 4): Add GGUF loading path via llama-cpp-python for LLM backbone.
        Currently uses PyTorch path - only suitable for small models (<3B FP16).
        """
        import logging

        from transformers import AutoModel

        if trust_remote_code:
            logging.getLogger(__name__).warning(
                "Loading model with trust_remote_code=True - executing remote code from %s", model_path
            )

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
        torch_dtype = dtype_map.get(dtype, torch.float16)

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).to("cuda:0")
        model.eval()
        return model

    def infer(self, model: Any, observation: dict[str, Any]) -> InferenceResult:
        """Run inference on Jetson GPU with power monitoring."""
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
                raise ValueError("Model has no predict_action() or forward()")

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        power_after = self._read_power()

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        return InferenceResult(
            actions=actions,
            latency_ms=round(latency_ms, 2),
            memory_peak_mb=round(peak_mem, 1),
            metadata={
                "device": _get_jetson_model(),
                "jetpack": _get_jetpack_version(),
                "power_mw": power_after,
            },
        )

    def _read_power(self) -> int | None:
        """Read current power consumption in milliwatts from Jetson power sensors."""
        try:
            # INA3221 power monitor (common on Orin Nano)
            power_path = Path("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon")
            if power_path.exists():
                hwmon_dirs = list(power_path.iterdir())
                if hwmon_dirs:
                    power_file = hwmon_dirs[0] / "in1_input"
                    if power_file.exists():
                        return int(power_file.read_text().strip())
        except Exception:
            pass
        return None
