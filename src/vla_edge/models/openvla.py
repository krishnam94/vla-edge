"""OpenVLA 7B adapter for vla-edge.

OpenVLA is an AUTOREGRESSIVE VLA (unlike SmolVLA which is flow matching).
It generates action tokens one at a time through a Prismatic VLM backbone.
This means different optimization strategies: speculative decoding, parallel
fixed-point iteration (PD-VLA), FAST tokenization all apply here.

7B params = 14GB bf16 = ~4GB int4. Needs quantization for edge deployment.
Requires trust_remote_code=True (custom Prismatic architecture).

Reference: https://arxiv.org/abs/2406.09246
Known issues: #145, #286, #287 (quantization crashes with bitsandbytes)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_edge.models.base import ModelInfo, VLAModel
from vla_edge.registry import register_model

logger = logging.getLogger(__name__)


@register_model("openvla")
class OpenVLAAdapter(VLAModel):
    """OpenVLA 7B autoregressive VLA adapter.

    Action generation: autoregressive token prediction (7+ decode steps).
    75% of latency is in the decode loop - optimization target for PD-VLA, speculative decoding.
    """

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b",
        device: str = "cpu",
        dtype: str = "bf16",
    ) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("OpenVLA requires PyTorch. Install with: pip install vla-edge[pytorch]") from e

        self._model_id = model_id
        self._dtype_str = dtype

        # Device selection
        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(dtype, torch.bfloat16)

        self._model: Any = None
        self._processor: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy load OpenVLA on first predict() call. Downloads ~14GB."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("Loading OpenVLA from %s (this downloads ~14GB)...", self._model_id)

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Don't call .to() on quantized models (OpenVLA #145, #286, #287)
        if self._device.type != "cpu" and not getattr(self._model, "is_quantized", False):
            self._model = self._model.to(self._device)

        self._model.eval()
        self._loaded = True
        logger.info("OpenVLA loaded on %s (%s)", self._device, self._dtype_str)

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict action via autoregressive token generation.

        Args:
            image: RGB image, shape (H, W, 3), uint8 or float32.
                   OpenVLA expects 224x224 (resized by processor).
            instruction: Natural language task instruction.
            state: Not used by OpenVLA (instruction-conditioned only).

        Returns:
            Action array, shape (7,) for 7-DoF (6 joints + gripper).
        """
        import torch
        from PIL import Image as PILImage

        self._ensure_loaded()

        # Convert numpy to PIL (processor expects PIL)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image

        # Format prompt (OpenVLA expects this specific format)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Process inputs
        inputs = self._processor(prompt, pil_image).to(self._device, dtype=self._torch_dtype)

        # Autoregressive generation (this is the bottleneck - 75% of latency)
        with torch.inference_mode():
            action = self._model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",  # Default normalization key
                do_sample=False,
            )

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        elif isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)

        return action

    @property
    def info(self) -> ModelInfo:
        return OpenVLAAdapter.model_info()

    @staticmethod
    def model_info() -> ModelInfo:
        return ModelInfo(
            name="openvla",
            param_count=7_000_000_000,
            architecture="Prismatic VLM (autoregressive action tokens)",
            action_dim=7,  # 6 joints + gripper
            required_image_size=(224, 224),
            supported_dtypes=["fp32", "fp16", "bf16", "int8", "int4"],
            source_url="https://huggingface.co/openvla/openvla-7b",
        )

    def cleanup(self) -> None:
        """Release model memory."""
        import torch

        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("OpenVLA model released")
