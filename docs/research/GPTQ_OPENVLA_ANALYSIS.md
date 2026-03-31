# GPTQ 4-bit OpenVLA on HuggingFace - Analysis

**Date**: 2026-03-31
**Model**: devanaik/openvla-7b-finetuned-libero-10-gptq-4bit
**Source**: https://huggingface.co/devanaik

## Key Findings

- 4 GPTQ variants uploaded 2026-03-28 (one per LIBERO task suite)
- No model card, no benchmarks, no documentation (~14 downloads each)
- Only LLM backbone quantized (Llama 7B). Vision encoders (SigLIP + DINOv2) kept full precision.
- GPTQ 4-bit, group_size 128, symmetric, true_sequential
- Calibration dataset NOT specified (likely c4/wikitext defaults - suboptimal for robotics)
- Total size: ~5.5 GB (from ~14 GB, ~2.5x compression)
- Split architecture: `llm_quantized/` + `non_llm_weights.pt`

## Compatibility with vla-edge

- Partially compatible. Uses `OpenVLAForActionPrediction` auto_map, not standard `AutoModelForCausalLM`
- Split structure needs custom reassembly logic
- Requires `gptqmodel` dependency (not in vla-edge currently)
- The `is_quantized` guard in openvla.py would correctly prevent `.to()` crash

## Recommendation

v0.2 nice-to-have: GPTQ model loading for desktop GPU workflows.
v0.2 priority: GGUF quantization (Jetson-critical, already planned).
The key: if someone finds a pre-quantized VLA on HF, vla-edge should load, profile, and validate it.

Pre-quantized VLA models are NOT a trend yet (4 models, 1 uploader, no documentation).
Prediction: 6-12 months before this becomes common.

## Sources
- [devanaik HuggingFace](https://huggingface.co/devanaik)
- [HuggingFace GPTQ docs](https://huggingface.co/docs/transformers/en/quantization/gptq)
- [LiteVLA-Edge arXiv:2603.03380](https://arxiv.org/abs/2603.03380)
- [GPTQModel GitHub](https://github.com/ModelCloud/GPTQModel)
