"""Base class for VLA model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ModelInfo:
    """Metadata about a VLA model."""

    name: str
    param_count: int = 0
    architecture: str = "unknown"
    action_dim: int = 7  # 6-DoF + gripper is common
    required_image_size: tuple[int, int] = (224, 224)
    supported_dtypes: list[str] | None = None
    source_url: str = ""


class VLAModel(ABC):
    """Abstract base for VLA model adapters.

    Adding a new model:
      1. Create a new file in models/ (e.g., openvla.py)
      2. Subclass VLAModel and implement predict() + info property
      3. Decorate with @register_model("openvla")
      4. That's it - no core code changes needed
    """

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict action from observation.

        Args:
            image: RGB image array, shape (H, W, 3), uint8 or float32.
            instruction: Natural language task instruction.
            state: Optional robot proprioceptive state.

        Returns:
            Action array of shape (action_dim,).
        """

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return metadata about this model."""

    def preprocess_image(self, image: np.ndarray) -> Any:
        """Preprocess image for this model. Override for custom preprocessing.

        Default: resize to required_image_size and normalize to [0, 1].
        """
        from PIL import Image

        h, w = self.info.required_image_size
        pil_img = Image.fromarray(image).resize((w, h))
        return np.array(pil_img, dtype=np.float32) / 255.0
