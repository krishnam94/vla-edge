"""Tests for SmolVLA adapter."""

import numpy as np
import pytest

from vla_edge.models.base import ModelInfo


def test_smolvla_registered():
    """SmolVLA should appear in the model registry after import."""
    from vla_edge.registry import list_models

    # Force import of smolvla module
    try:
        import vla_edge.models.smolvla  # noqa: F401
    except ImportError:
        pytest.skip("lerobot not installed")

    models = list_models()
    assert "smolvla" in models


def test_smolvla_get_class():
    """Should get SmolVLAAdapter class from registry."""
    try:
        import vla_edge.models.smolvla  # noqa: F401
    except ImportError:
        pytest.skip("lerobot not installed")

    from vla_edge.registry import get_model

    cls = get_model("smolvla")
    assert cls.__name__ == "SmolVLAAdapter"


def test_smolvla_info_without_loading():
    """ModelInfo should be accessible without loading the full model."""
    try:
        from vla_edge.models.smolvla import SmolVLAAdapter
    except ImportError:
        pytest.skip("lerobot not installed")

    # Create adapter (lazy - doesn't load model yet)
    adapter = SmolVLAAdapter.__new__(SmolVLAAdapter)
    info = adapter.info

    assert isinstance(info, ModelInfo)
    assert info.name == "smolvla"
    assert info.param_count == 450_000_000
    assert info.action_dim == 6
    assert info.required_image_size == (512, 512)
    assert "Flow Matching" in info.architecture
    assert "float32" in info.supported_dtypes
    assert "bfloat16" in info.supported_dtypes


def test_smolvla_image_size_is_512():
    """SmolVLA requires 512x512, NOT 224x224. This is critical."""
    try:
        from vla_edge.models.smolvla import SmolVLAAdapter
    except ImportError:
        pytest.skip("lerobot not installed")

    adapter = SmolVLAAdapter.__new__(SmolVLAAdapter)
    assert adapter.info.required_image_size == (512, 512)


def test_smolvla_import_error_message():
    """If lerobot not installed, error message should be helpful."""
    from vla_edge.models.smolvla import _check_lerobot

    # We can't easily test the ImportError path if lerobot IS installed,
    # but we can verify the function exists and returns bool
    result = _check_lerobot()
    assert isinstance(result, bool)


@pytest.mark.slow
def test_smolvla_load_and_predict():
    """Full load + predict test. Slow - downloads ~2GB model."""
    pytest.importorskip("lerobot")
    pytest.importorskip("torch")

    from vla_edge.models.smolvla import SmolVLAAdapter

    adapter = SmolVLAAdapter(device="cpu", dtype="float32")

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    instruction = "pick up the red block"
    state = np.zeros(6, dtype=np.float32)

    action = adapter.predict(image, instruction, state)

    assert isinstance(action, np.ndarray)
    assert action.shape[-1] == 6  # action_dim for SO100

    # Cleanup
    adapter.cleanup()
