"""End-to-end integration tests.

These tests download real models and run full inference pipelines.
Marked @slow - skipped in normal CI. Run manually before release:
  pytest -m slow tests/test_integration.py -v
"""

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.integration
def test_smolvla_full_pipeline():
    """Full pipeline: load SmolVLA -> profile -> validate safety -> report.

    Downloads ~900MB model on first run. Takes 2-5 min on CPU.
    """
    pytest.importorskip("lerobot")
    pytest.importorskip("torch")

    from vla_edge.models.smolvla import SmolVLAAdapter
    from vla_edge.validate.safety import SafetyConfig

    # 1. Load model
    adapter = SmolVLAAdapter(device="cpu", dtype="float32")
    assert adapter.info.name == "smolvla"
    assert adapter.info.required_image_size == (256, 256)

    # 2. Run inference
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    instruction = "pick up the red block"
    state = np.zeros(6, dtype=np.float32)

    action = adapter.predict(image, instruction, state)
    assert isinstance(action, np.ndarray)
    assert action.shape[-1] >= 1  # At least 1 action dim

    # 3. Safety validation
    config = SafetyConfig(
        action_bounds=np.array([[-2, 2]] * max(action.shape[-1], 7), dtype=np.float32),
        max_velocity=np.full(max(action.shape[-1], 7), 0.5, dtype=np.float32),
    )

    from vla_edge.validate.safety import validate_actions

    actions_seq = action[np.newaxis, :] if action.ndim == 1 else action

    result = validate_actions(actions_seq, config)
    assert result.total_actions >= 1

    # 4. Cleanup
    adapter.cleanup()


@pytest.mark.slow
@pytest.mark.integration
def test_smolvla_registry_integration():
    """SmolVLA loads correctly through the registry."""
    pytest.importorskip("lerobot")

    # Force import
    import vla_edge.models.smolvla  # noqa: F401
    from vla_edge.registry import get_model, list_models

    assert "smolvla" in list_models()

    cls = get_model("smolvla")
    assert cls.__name__ == "SmolVLAAdapter"
    assert cls.model_info().required_image_size == (512, 512)
