"""SmolVLA 450M adapter for vla-edge.

SmolVLA is a flow-matching VLA (not autoregressive) from HuggingFace/LeRobot.
- Vision: SigLIP-B/16 (512x512 images, 64 tokens per frame)
- LLM: SmolLM2-360M (Llama arch, first 16 of 32 layers)
- Action Expert: Flow matching transformer (~100M params, 10 denoising steps)
- Chunk: 50 continuous actions per forward pass

Requires: pip install vla-edge[smolvla]
Reference: https://arxiv.org/abs/2506.01844
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_edge.models.base import ModelInfo, VLAModel
from vla_edge.registry import register_model

logger = logging.getLogger(__name__)

# Lazy imports - lerobot is optional
_LEROBOT_AVAILABLE: bool | None = None


def _check_lerobot() -> bool:
    """Check if lerobot is installed, cache result."""
    global _LEROBOT_AVAILABLE
    if _LEROBOT_AVAILABLE is None:
        try:
            import lerobot  # noqa: F401

            _LEROBOT_AVAILABLE = True
        except ImportError:
            _LEROBOT_AVAILABLE = False
    return _LEROBOT_AVAILABLE


@register_model("smolvla")
class SmolVLAAdapter(VLAModel):
    """SmolVLA 450M parameter VLA adapter.

    Uses flow matching for action generation (10 denoising steps),
    not autoregressive token generation. Different optimization
    strategies apply compared to OpenVLA/pi0.
    """

    def __init__(
        self,
        model_id: str = "lerobot/smolvla_base",
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        if not _check_lerobot():
            raise ImportError(
                "SmolVLA requires LeRobot. Install with: pip install vla-edge[smolvla]\n"
                "Or: pip install lerobot"
            )

        import torch

        self._model_id = model_id
        self._device = torch.device(device)
        self._torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        self._policy: Any = None
        self._loaded = False

        logger.info("SmolVLA adapter created (lazy loading, call predict() to load)")

    def _ensure_loaded(self) -> None:
        """Lazy load the model on first predict() call."""
        if self._loaded:
            return

        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        logger.info("Loading SmolVLAPolicy from %s...", self._model_id)
        self._policy = SmolVLAPolicy.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
        )
        self._policy.to(self._device)
        self._policy.eval()
        self._loaded = True
        logger.info("SmolVLA loaded successfully on %s", self._device)

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict action chunk using flow matching (10 denoising steps).

        Args:
            image: RGB image, shape (H, W, 3), uint8 or float32.
                   Automatically resized to 512x512.
            instruction: Natural language task instruction.
            state: Robot proprioceptive state, shape (state_dim,). Optional.

        Returns:
            Action array. Shape depends on policy config (typically (action_dim,)
            for single step or (chunk_size, action_dim) for full chunk).
        """
        import torch

        self._ensure_loaded()

        # Build observation dict matching LeRobot's expected format
        observation: dict[str, Any] = {}

        # Image: convert to tensor, add batch dim
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).float()
            # Normalize to [0, 1] only if uint8 input
            if image.dtype == np.uint8:
                img_tensor = img_tensor / 255.0
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            observation["observation.image"] = img_tensor.to(self._device)

        # State
        if state is not None:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            observation["observation.state"] = state_tensor.to(self._device)

        # Instruction
        observation["task"] = instruction

        # Run policy (handles flow matching denoising internally)
        with torch.inference_mode():
            action = self._policy.select_action(observation)

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Remove batch dim if present
        if action.ndim > 1 and action.shape[0] == 1:
            action = action.squeeze(0)

        return action

    @property
    def info(self) -> ModelInfo:
        return SmolVLAAdapter.model_info()

    @staticmethod
    def model_info() -> ModelInfo:
        """Static access to model info without needing an instance."""
        return ModelInfo(
            name="smolvla",
            param_count=450_000_000,
            architecture="VLM (SmolVLM2-500M) + Flow Matching Action Expert",
            action_dim=6,
            required_image_size=(512, 512),
            supported_dtypes=["float32", "bfloat16"],
            source_url="https://huggingface.co/lerobot/smolvla_base",
        )

    def cleanup(self) -> None:
        """Release model memory. Addresses issue #4."""
        import torch

        if self._policy is not None:
            del self._policy
            self._policy = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SmolVLA model released")
