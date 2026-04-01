"""Diffusion Policy adapter for vla-edge.

Diffusion Policy uses DDPM denoising to generate action chunks (~30M params).
Different denoising mechanism from SmolVLA (flow matching vs DDPM).

Used in paper to show SafeContract works across denoising paradigms.

Reference: https://arxiv.org/abs/2303.04137
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_edge.models.base import ModelInfo, VLAModel
from vla_edge.registry import register_model

logger = logging.getLogger(__name__)


@register_model("diffusion")
class DiffusionPolicyAdapter(VLAModel):
    """Diffusion Policy adapter. DDPM-based, lightweight."""

    def __init__(self, model_id: str = "lerobot/diffusion_pusht", device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("Diffusion Policy requires PyTorch.") from e

        self._model_id = model_id
        self._device = torch.device(device)
        self._policy: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        logger.info("Loading Diffusion Policy from %s...", self._model_id)
        self._policy = DiffusionPolicy.from_pretrained(self._model_id)
        self._policy.to(self._device)
        self._policy.eval()
        self._loaded = True

    def predict(self, image: np.ndarray, instruction: str, state: np.ndarray | None = None) -> np.ndarray:
        import torch

        self._ensure_loaded()

        obs: dict[str, Any] = {}
        if isinstance(image, np.ndarray):
            img = (
                torch.from_numpy(image).float() / 255.0
                if image.dtype == np.uint8
                else torch.from_numpy(image).float()
            )
            if img.ndim == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)
            obs["observation.image"] = img.to(self._device)

        if state is not None:
            obs["observation.state"] = torch.from_numpy(state).float().unsqueeze(0).to(self._device)

        self._policy.reset()
        with torch.inference_mode():
            action = self._policy.select_action(obs)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action

    @property
    def info(self) -> ModelInfo:
        return DiffusionPolicyAdapter.model_info()

    @staticmethod
    def model_info() -> ModelInfo:
        return ModelInfo(
            name="diffusion",
            param_count=30_000_000,
            architecture="DDPM Diffusion (denoising, no VLM)",
            action_dim=2,  # PushT: x, y
            required_image_size=(96, 96),
            supported_dtypes=["fp32"],
            source_url="https://huggingface.co/lerobot/diffusion_pusht",
        )

    def cleanup(self) -> None:
        if self._policy is not None:
            del self._policy
            self._policy = None
            self._loaded = False
