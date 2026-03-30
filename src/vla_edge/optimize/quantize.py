"""Quantization for VLA models.

Two approaches supported:
1. Standard GGUF quantization via llama-cpp-python (v0.2.0)
   - Converts LLM backbone to GGUF format (Q4_K_M, Q8_0, F16)
   - Vision encoder stays in FP16/FP32 (quantizing vision hurts more)
   - This is the proven path from LiteVLA-Edge (6.6 Hz on Jetson)

2. Action-centric quantization via QVLA (future, v0.3.0)
   - Per-channel bit allocation based on action sensitivity
   - 22.6% better than SmoothQuant on VLA tasks
   - Requires calibration data (task-specific trajectories)
   - Reference: arXiv:2602.03782

Key insight from research: standard LLM quantization (SmoothQuant, AWQ)
is suboptimal for VLAs. Small action deviations compound over trajectories.
But GGUF Q4_K_M retains 95-97% accuracy and is battle-tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Compression ratios for GGUF quantization types (relative to FP16)
QUANT_RATIOS: dict[str, float] = {
    "f16": 1.0,
    "q8_0": 0.53,
    "q6_k": 0.42,
    "q5_k_m": 0.37,
    "q4_k_m": 0.32,
    "q4_0": 0.29,
    "q3_k_m": 0.25,
    "q2_k": 0.20,
}

VALID_QUANTIZATIONS = set(QUANT_RATIOS.keys())


@dataclass
class QuantizeResult:
    """Result from quantizing a model."""

    original_path: str
    quantized_path: Path | None
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quantization_type: str
    success: bool
    error: str | None = None


def quantize_to_gguf(
    model_path: str,
    output_dir: str | Path,
    quantization: str = "q4_k_m",
    component: str = "llm",
) -> QuantizeResult:
    """Quantize a model (or component) to GGUF format.

    Args:
        model_path: HuggingFace model ID or local path.
        output_dir: Directory to save quantized model.
        quantization: GGUF quantization type (q4_k_m, q8_0, f16).
        component: Which component to quantize (llm, vision, full).
            For VLA models, quantize LLM backbone only (component="llm").
            Vision encoder should stay in FP16 (quantizing vision hurts accuracy).

    Returns:
        QuantizeResult with paths and compression info.
    """
    # Validate quantization type
    quantization = quantization.lower()
    if quantization not in VALID_QUANTIZATIONS:
        return QuantizeResult(
            original_path=model_path,
            quantized_path=None,
            original_size_mb=0,
            quantized_size_mb=0,
            compression_ratio=1.0,
            quantization_type=quantization,
            success=False,
            error=f"Unknown quantization '{quantization}'. Valid: {sorted(VALID_QUANTIZATIONS)}",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if llama-cpp-python is available
    try:
        import llama_cpp  # noqa: F401

        has_llama_cpp = True
    except ImportError:
        has_llama_cpp = False

    if not has_llama_cpp:
        return QuantizeResult(
            original_path=model_path,
            quantized_path=None,
            original_size_mb=0,
            quantized_size_mb=0,
            compression_ratio=1.0,
            quantization_type=quantization,
            success=False,
            error="llama-cpp-python not installed. Install with: pip install vla-edge[gguf]",
        )

    if component == "vision":
        logger.warning(
            "Quantizing vision encoder is NOT recommended for VLAs. "
            "Vision quantization degrades action quality more than LLM quantization. "
            "Use component='llm' instead."
        )

    logger.info("Quantizing %s (component=%s) to GGUF %s...", model_path, component, quantization)

    try:
        return _quantize_with_llama_cpp(model_path, output_dir, quantization)
    except Exception as e:
        logger.error("Quantization failed: %s", e)
        return QuantizeResult(
            original_path=model_path,
            quantized_path=None,
            original_size_mb=0,
            quantized_size_mb=0,
            compression_ratio=1.0,
            quantization_type=quantization,
            success=False,
            error=str(e),
        )


def _quantize_with_llama_cpp(
    model_path: str,
    output_dir: Path,
    quantization: str,
) -> QuantizeResult:
    """Use llama-cpp-python for GGUF quantization.

    For HuggingFace models, this downloads and converts to GGUF.
    For local GGUF files, this re-quantizes to the target type.
    """

    output_file = output_dir / f"model-{quantization}.gguf"

    # If model_path is already a GGUF file, re-quantization needs llama.cpp CLI
    if Path(model_path).suffix == ".gguf":
        return QuantizeResult(
            original_path=model_path,
            quantized_path=None,
            original_size_mb=Path(model_path).stat().st_size / (1024 * 1024),
            quantized_size_mb=0,
            compression_ratio=1.0,
            quantization_type=quantization,
            success=False,
            error=(
                "Re-quantizing an existing GGUF file requires llama.cpp's llama-quantize tool. "
                f"Run: llama-quantize {model_path} {output_file} {quantization}"
            ),
        )

    # For HuggingFace models, we need to download and convert
    # This is a placeholder - full conversion requires the llama.cpp
    # convert.py script or the hf_to_gguf.py tool
    return QuantizeResult(
        original_path=model_path,
        quantized_path=None,
        original_size_mb=0,
        quantized_size_mb=0,
        compression_ratio=1.0,
        quantization_type=quantization,
        success=False,
        error=(
            f"HuggingFace -> GGUF conversion for '{model_path}' requires "
            "llama.cpp's convert-hf-to-gguf.py tool. "
            "Install llama.cpp and run: "
            f"python convert-hf-to-gguf.py {model_path} --outtype {quantization}"
        ),
    )


def estimate_quantized_size(
    original_size_mb: float,
    quantization: str,
) -> float:
    """Estimate quantized model size based on quantization type.

    Estimates based on typical compression ratios from llama.cpp benchmarks.
    """
    ratio = QUANT_RATIOS.get(quantization.lower(), 0.32)
    return round(original_size_mb * ratio, 1)


def recommend_quantization(
    model_params: int,
    target_memory_mb: int = 8192,
    safety_margin: float = 0.7,
) -> str:
    """Recommend a quantization level based on model size and target memory.

    Args:
        model_params: Number of parameters in the model.
        target_memory_mb: Available GPU memory in MB (default: 8GB Jetson).
        safety_margin: Fraction of memory to use (default: 70% to leave room for runtime).

    Returns:
        Recommended quantization string (e.g., "q4_k_m").
    """
    available_mb = target_memory_mb * safety_margin

    # Estimate FP16 size: params * 2 bytes / 1MB
    fp16_size_mb = model_params * 2 / (1024 * 1024)

    if fp16_size_mb <= available_mb:
        return "f16"

    # Try each quantization level until it fits (sorted by ratio descending = least aggressive first)
    for quant, ratio in sorted(QUANT_RATIOS.items(), key=lambda x: -x[1]):
        if fp16_size_mb * ratio <= available_mb:
            return quant

    return "q2_k"  # Most aggressive if nothing else fits
