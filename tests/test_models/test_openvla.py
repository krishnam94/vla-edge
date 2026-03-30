"""Tests for OpenVLA adapter."""

import pytest

from vla_edge.models.base import ModelInfo


def test_openvla_registered():
    """OpenVLA should appear in the model registry."""
    import vla_edge.models.openvla  # noqa: F401
    from vla_edge.registry import list_models

    assert "openvla" in list_models()


def test_openvla_model_info():
    """ModelInfo should be correct without loading the model."""
    from vla_edge.models.openvla import OpenVLAAdapter

    info = OpenVLAAdapter.model_info()
    assert isinstance(info, ModelInfo)
    assert info.name == "openvla"
    assert info.param_count == 7_000_000_000
    assert info.action_dim == 7  # 6 joints + gripper
    assert info.required_image_size == (224, 224)  # NOT 256 or 512
    assert "autoregressive" in info.architecture.lower()
    assert "bf16" in info.supported_dtypes
    assert "int4" in info.supported_dtypes


def test_openvla_vs_smolvla_differences():
    """OpenVLA and SmolVLA should have different architectures and sizes."""
    from vla_edge.models.openvla import OpenVLAAdapter

    openvla_info = OpenVLAAdapter.model_info()

    try:
        from vla_edge.models.smolvla import SmolVLAAdapter

        smolvla_info = SmolVLAAdapter.model_info()

        # Different architectures
        assert "autoregressive" in openvla_info.architecture.lower()
        assert "flow matching" in smolvla_info.architecture.lower()

        # Different sizes
        assert openvla_info.param_count > smolvla_info.param_count  # 7B vs 450M

        # Different image sizes
        assert openvla_info.required_image_size != smolvla_info.required_image_size

        # Different action dims
        assert openvla_info.action_dim == 7  # 6 + gripper
        assert smolvla_info.action_dim == 6  # SO100 no gripper
    except ImportError:
        pytest.skip("lerobot not installed for SmolVLA comparison")


def test_openvla_lazy_loading():
    """OpenVLA should not download anything until predict() is called."""
    from vla_edge.models.openvla import OpenVLAAdapter

    adapter = OpenVLAAdapter(device="cpu")
    # Model not loaded yet
    assert not adapter._loaded
    assert adapter._model is None
    # Info accessible without loading
    assert adapter.info.name == "openvla"


def test_openvla_cleanup():
    """Cleanup should reset state."""
    from vla_edge.models.openvla import OpenVLAAdapter

    adapter = OpenVLAAdapter(device="cpu")
    adapter.cleanup()
    assert not adapter._loaded
    assert adapter._model is None
