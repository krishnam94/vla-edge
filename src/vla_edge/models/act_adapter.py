"""ACT (Action Chunking with Transformers) adapter for vla-edge.

ACT is a lightweight transformer policy (~30M params) that doesn't use a VLM.
It predicts action chunks from observations directly, using chunked
action prediction with temporal ensembling.

Used in paper to show SafeContract works across architectures (not just VLAs).

Reference: https://arxiv.org/abs/2304.13705
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_edge.models.base import ModelInfo, VLAModel
from vla_edge.registry import register_model

logger = logging.getLogger(__name__)


@register_model("act")
class ACTAdapter(VLAModel):
    """ACT policy adapter. Lightweight, no VLM."""

    def __init__(
        self, model_id: str = "lerobot/act_aloha_sim_transfer_cube_human", device: str = "cpu"
    ) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("ACT requires PyTorch.") from e

        self._model_id = model_id
        self._device = torch.device(device)
        self._policy: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        from lerobot.policies.act.modeling_act import ACTPolicy

        logger.info("Loading ACT from %s...", self._model_id)
        self._policy = ACTPolicy.from_pretrained(self._model_id)
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
            obs["observation.images.top"] = img.to(self._device)

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
        return ACTAdapter.model_info()

    @staticmethod
    def model_info() -> ModelInfo:
        return ModelInfo(
            name="act",
            param_count=30_000_000,
            architecture="Transformer (Action Chunking, no VLM)",
            action_dim=14,  # ALOHA bimanual
            required_image_size=(480, 640),
            supported_dtypes=["fp32"],
            source_url="https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human",
        )

    def cleanup(self) -> None:
        if self._policy is not None:
            del self._policy
            self._policy = None
            self._loaded = False
