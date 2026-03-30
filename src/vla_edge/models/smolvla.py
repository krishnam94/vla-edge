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


def _import_smolvla_policy() -> type:
    """Import SmolVLAPolicy bypassing lerobot.policies.__init__.

    lerobot 0.4.4's policies/__init__.py eagerly imports ALL policies
    including GR00T, which has a broken dataclass (non-default after default).
    We pre-create the policies module to skip that __init__.
    """
    import sys
    import types

    # Pre-create lerobot.policies without running its __init__
    if "lerobot.policies" not in sys.modules:
        import lerobot

        policies_mod = types.ModuleType("lerobot.policies")
        policies_mod.__path__ = [lerobot.__path__[0] + "/policies"]
        policies_mod.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = policies_mod

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    return SmolVLAPolicy


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
        # Auto-select device: prefer cuda, fallback to mps (Mac), then cpu
        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif device != "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        self._policy: Any = None
        self._loaded = False

        logger.info("SmolVLA adapter created (lazy loading, call predict() to load)")

    def _ensure_loaded(self) -> None:
        """Lazy load the model on first predict() call."""
        if self._loaded:
            return

        smolvla_policy_cls = _import_smolvla_policy()

        logger.info("Loading SmolVLAPolicy from %s...", self._model_id)
        self._policy = smolvla_policy_cls.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
        )
        self._policy.to(self._device)
        self._policy.eval()
        self._loaded = True
        # Load tokenizer for language processing
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._policy.config.vlm_model_name)
        self._tokenizer_max_length = self._policy.config.tokenizer_max_length

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

        # Build observation dict matching LeRobot SmolVLA's expected format.
        # SmolVLA expects keys like 'observation.images.camera1' (not 'observation.image')
        # and images as (B, C, H, W) tensors. Image size from config (typically 256x256).
        observation: dict[str, Any] = {}

        # Get expected image features from policy config
        img_features = self._policy.config.image_features
        img_key = next(iter(img_features))  # e.g., 'observation.images.camera1'
        expected_shape = img_features[img_key].shape  # e.g., (3, 256, 256)
        target_h, target_w = expected_shape[1], expected_shape[2]

        if isinstance(image, np.ndarray):
            from PIL import Image as PILImage

            # Resize to expected dimensions
            pil_img = PILImage.fromarray(image if image.dtype == np.uint8 else (image * 255).astype(np.uint8))
            pil_img = pil_img.resize((target_w, target_h))
            img_arr = np.array(pil_img, dtype=np.float32) / 255.0

            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            observation[img_key] = img_tensor.to(self._device)

            # Pad missing cameras with zeros (SmolVLA supports up to 3)
            for key, feat in img_features.items():
                if key not in observation:
                    shape = feat.shape
                    observation[key] = torch.zeros(1, *shape, device=self._device)

        # State
        if state is not None:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            observation["observation.state"] = state_tensor.to(self._device)

        # Tokenize instruction
        tokens = self._processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self._tokenizer_max_length,
            truncation=True,
        )
        observation["observation.language.tokens"] = tokens["input_ids"].to(self._device)
        observation["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(self._device)

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
            # Config uses 256x256 per camera, not 512x512 as the paper's SigLIP input.
            # The policy config is the ground truth.
            required_image_size=(256, 256),
            supported_dtypes=["fp32", "bf16"],
            source_url="https://huggingface.co/lerobot/smolvla_base",
        )

    def cleanup(self) -> None:
        """Release model memory. Addresses issue #4."""
        if self._policy is not None:
            del self._policy
            self._policy = None
        if hasattr(self, "_processor") and self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("SmolVLA model released")
