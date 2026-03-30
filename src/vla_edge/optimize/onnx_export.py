"""ONNX export for VLA model components.

VLA models have 3 components that need different optimization paths:
- Vision encoder (ViT/SigLIP): TensorRT on Jetson, ONNX Runtime elsewhere
- LLM backbone: llama.cpp/GGUF on Jetson, ONNX Runtime elsewhere
- Action head: PyTorch or ONNX (lightweight, often kept in PyTorch)

This module exports each component separately to ONNX, enabling
per-component optimization on edge hardware.

Solves LeRobot #3146 (no ONNX/TRT export pipeline exists).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result from exporting a model component to ONNX."""

    component: str  # "vision", "llm", "action_head", "full"
    output_path: Path
    input_names: list[str]
    output_names: list[str]
    original_size_mb: float
    exported_size_mb: float
    success: bool
    error: str | None = None


def export_to_onnx(
    model: Any,
    output_dir: str | Path,
    component: str = "full",
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> ExportResult:
    """Export a model or model component to ONNX.

    Args:
        model: PyTorch model or VLAModel adapter.
        output_dir: Directory to save ONNX files.
        component: Which component to export - "full", "vision", "llm", "action_head".
        opset_version: ONNX opset version (17 recommended for transformer models).
        dynamic_axes: Dynamic axis specification for variable-length inputs.

    Returns:
        ExportResult with paths and metadata.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to export
    if component == "full":
        return _export_full_model(model, output_dir, opset_version, dynamic_axes)
    elif component == "vision":
        return _export_vision_encoder(model, output_dir, opset_version)
    elif component == "action_head":
        return _export_action_head(model, output_dir, opset_version)
    else:
        return ExportResult(
            component=component,
            output_path=output_dir,
            input_names=[],
            output_names=[],
            original_size_mb=0,
            exported_size_mb=0,
            success=False,
            error=f"Unknown component: {component}. Use: full, vision, action_head",
        )


def _export_full_model(
    model: Any,
    output_dir: Path,
    opset_version: int,
    dynamic_axes: dict | None,
) -> ExportResult:
    """Export full model to single ONNX file."""
    import torch

    output_path = output_dir / "model.onnx"

    # Get a dummy input matching the model's expected format
    if hasattr(model, "info"):
        h, w = model.info.required_image_size
    else:
        h, w = 256, 256

    dummy_input = torch.randn(1, 3, h, w)
    input_names = ["image"]
    output_names = ["actions"]

    if dynamic_axes is None:
        dynamic_axes = {"image": {0: "batch"}, "actions": {0: "batch"}}

    try:
        # Get the underlying torch model
        torch_model = _get_torch_model(model)
        if torch_model is None:
            return ExportResult(
                component="full",
                output_path=output_path,
                input_names=input_names,
                output_names=output_names,
                original_size_mb=0,
                exported_size_mb=0,
                success=False,
                error="Could not extract PyTorch model for ONNX export",
            )

        original_size = _model_size_mb(torch_model)

        torch.onnx.export(
            torch_model,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        exported_size = output_path.stat().st_size / (1024 * 1024)

        logger.info(
            "Exported full model to %s (%.1f MB -> %.1f MB)", output_path, original_size, exported_size
        )

        return ExportResult(
            component="full",
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            original_size_mb=round(original_size, 1),
            exported_size_mb=round(exported_size, 1),
            success=True,
        )

    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        return ExportResult(
            component="full",
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            original_size_mb=0,
            exported_size_mb=0,
            success=False,
            error=str(e),
        )


def _export_vision_encoder(model: Any, output_dir: Path, opset_version: int) -> ExportResult:
    """Export just the vision encoder to ONNX.

    This is the component that benefits most from TensorRT on Jetson.
    """
    import torch

    output_path = output_dir / "vision_encoder.onnx"

    # Try to find the vision encoder in the model
    vision_model = None
    for attr in ["vision_model", "vision_encoder", "image_encoder", "vit"]:
        if hasattr(model, attr):
            vision_model = getattr(model, attr)
            break

    # Check inside _policy for VLAModel adapters
    if vision_model is None and hasattr(model, "_policy"):
        policy = model._policy
        for attr in ["vision_model", "vision_encoder", "model"]:
            if hasattr(policy, attr):
                inner = getattr(policy, attr)
                for vattr in ["vision_model", "vision_encoder"]:
                    if hasattr(inner, vattr):
                        vision_model = getattr(inner, vattr)
                        break

    if vision_model is None:
        return ExportResult(
            component="vision",
            output_path=output_path,
            input_names=[],
            output_names=[],
            original_size_mb=0,
            exported_size_mb=0,
            success=False,
            error="Could not find vision encoder in model",
        )

    h, w = 256, 256
    if hasattr(model, "info"):
        h, w = model.info.required_image_size

    dummy_input = torch.randn(1, 3, h, w)
    original_size = _model_size_mb(vision_model)

    try:
        torch.onnx.export(
            vision_model,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
            opset_version=opset_version,
            do_constant_folding=True,
        )
        exported_size = output_path.stat().st_size / (1024 * 1024)
        logger.info("Exported vision encoder to %s (%.1f MB)", output_path, exported_size)

        return ExportResult(
            component="vision",
            output_path=output_path,
            input_names=["pixel_values"],
            output_names=["image_features"],
            original_size_mb=round(original_size, 1),
            exported_size_mb=round(exported_size, 1),
            success=True,
        )
    except Exception as e:
        return ExportResult(
            component="vision",
            output_path=output_path,
            input_names=[],
            output_names=[],
            original_size_mb=original_size,
            exported_size_mb=0,
            success=False,
            error=str(e),
        )


def _export_action_head(model: Any, output_dir: Path, opset_version: int) -> ExportResult:
    """Export the action head to ONNX. Placeholder for v0.2."""
    return ExportResult(
        component="action_head",
        output_path=output_dir / "action_head.onnx",
        input_names=[],
        output_names=[],
        original_size_mb=0,
        exported_size_mb=0,
        success=False,
        error="Action head export not yet implemented (SmolVLA uses flow matching, needs special handling)",
    )


def _get_torch_model(model: Any) -> Any:
    """Extract the underlying PyTorch model from a VLAModel adapter."""
    # SmolVLA adapter
    if hasattr(model, "_policy") and model._policy is not None:
        return model._policy
    # OpenVLA adapter
    if hasattr(model, "_model") and model._model is not None:
        return model._model
    # Raw PyTorch model
    import torch.nn as nn

    if isinstance(model, nn.Module):
        return model
    return None


def _model_size_mb(model: Any) -> float:
    """Estimate model size in MB from parameters."""
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        return 0.0
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 * 1024)
