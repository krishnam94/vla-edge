"""Tests for GGUF quantization module."""

from vla_edge.optimize.quantize import estimate_quantized_size, recommend_quantization


def test_estimate_q4_k_m():
    """Q4_K_M should be roughly 32% of original."""
    est = estimate_quantized_size(1000.0, "q4_k_m")
    assert 300 <= est <= 350


def test_estimate_f16():
    """F16 should be same size as original."""
    est = estimate_quantized_size(1000.0, "f16")
    assert est == 1000.0


def test_estimate_q8():
    """Q8_0 should be roughly 53% of original."""
    est = estimate_quantized_size(1000.0, "q8_0")
    assert 500 <= est <= 560


def test_estimate_unknown_defaults_q4():
    """Unknown quantization type uses Q4 ratio as default."""
    est = estimate_quantized_size(1000.0, "unknown_type")
    assert est == 320.0


def test_recommend_small_model_fits_fp16():
    """Small model (<8GB in FP16) should recommend FP16."""
    # 450M params * 2 bytes = 900MB -> fits in 8GB
    rec = recommend_quantization(450_000_000, target_memory_mb=8192)
    assert rec == "f16"


def test_recommend_7b_model_needs_quantization():
    """7B model needs quantization for 8GB Jetson."""
    rec = recommend_quantization(7_000_000_000, target_memory_mb=8192)
    # 13.4GB FP16 * 0.42 (q6_k) = 5.6GB -> fits in 8GB * 0.7 = 5.7GB
    # q6_k is the least aggressive that fits (preserves most quality)
    assert rec == "q6_k"


def test_recommend_huge_model_gets_aggressive():
    """Very large model needs aggressive quantization."""
    rec = recommend_quantization(70_000_000_000, target_memory_mb=8192)
    # 140GB FP16, nothing above q2_k fits
    assert rec == "q2_k"


def test_recommend_with_custom_memory():
    """Custom memory target changes recommendation."""
    # 7B on 24GB GPU
    rec = recommend_quantization(7_000_000_000, target_memory_mb=24576)
    assert rec == "f16"  # 14GB fits in 24GB * 0.7 = 17.2GB


def test_recommend_with_tight_margin():
    """Tight safety margin needs more aggressive quantization."""
    rec = recommend_quantization(7_000_000_000, target_memory_mb=8192, safety_margin=0.5)
    # 14GB * 0.29 (q4_0) = 4.06GB -> fits in 8GB * 0.5 = 4.1GB
    assert rec in ("q4_0", "q3_k_m")
