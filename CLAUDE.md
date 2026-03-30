# vla-edge - VLA Edge Deployment Readiness Toolkit

## What This Is
pip-installable toolkit for profiling, optimizing, validating, and deploying VLA (Vision-Language-Action) models to edge hardware. Primary target: Jetson Orin Nano Super.

## Architecture
- `src/vla_edge/backends/` - Hardware backends (CPU, CUDA, Jetson). Extensible via `@register_backend`.
- `src/vla_edge/models/` - VLA model adapters. Extensible via `@register_model`.
- `src/vla_edge/profile/` - Latency, memory, power profiling.
- `src/vla_edge/optimize/` - Quantization (GGUF), ONNX/TRT export.
- `src/vla_edge/validate/` - Action safety, quantization degradation.
- `src/vla_edge/cli.py` - Typer CLI entry point.
- `src/vla_edge/registry.py` - Backend + Model registries with auto-detect.
- `recipes/` - Tested deployment recipes (YAML).
- `results/` - Auto-updated benchmark results.

## Commands
- `vla-edge check` - Show hardware info and backend availability
- `vla-edge profile <model> [--hardware auto]` - Profile latency/memory/throughput
- `vla-edge models` - List available model adapters
- `vla-edge version` - Show version
- `pytest` - Run all CPU tests
- `pytest -m gpu` - Run GPU tests (needs CUDA)
- `pytest -m jetson` - Run Jetson tests (needs Jetson hardware)
- `ruff check src/ tests/` - Lint
- `ruff format src/ tests/` - Format

## Conventions
- New backend = one file in backends/ + `@register_backend("name")`. No core changes.
- New model = one file in models/ + `@register_model("name")`. No core changes.
- All backends implement HardwareBackend ABC (4 methods).
- All models implement VLAModel ABC (predict + info).
- Safety metrics are ALWAYS reported. Never optional.
- NEVER hardcode API keys or HuggingFace tokens. Use environment variables.
- NEVER use em dashes in any content. Use hyphens or rewrite.

## Testing
- pytest with hardware markers (cpu, gpu, jetson, tensorrt, slow).
- Default: runs only CPU tests (`-m 'not gpu and not jetson and not slow'`).
- GPU/Jetson tests run on self-hosted runner or manually on hardware.
- Target: 80%+ coverage on core modules (registry, safety, profiler).

## Jetson Constraints (Critical)
- TensorRT-LLM is BROKEN on Orin Nano. Do not use it. Causes kernel panics.
- Use llama.cpp + GGUF for LLM backbone inference on Jetson.
- Torch-TensorRT works for vision encoders (ViT) on Jetson.
- Max model size on Orin Nano 8GB: ~3-4B FP16, ~7B Q4.
- PyTorch must be installed from NVIDIA's JPL wheels, not pip.
- JetPack 6.2 = Ubuntu 22.04, CUDA 12.6, TensorRT 10.3, Python 3.10.
- ROS 2 Humble is the correct distro (Ubuntu 22.04). Jazzy needs 24.04.

## Key Research Insights
- Standard LLM quantization is suboptimal for VLAs. Use action-centric quantization (QVLA).
- 75% of VLA latency is in action generation, not vision encoding.
- Training-free parallel decoding (PD-VLA) gives 4x speedup on any autoregressive VLA.
- 6.6 Hz on Jetson Orin (LiteVLA-Edge) is the baseline to beat.
- Target: 10+ Hz for v0.1.

## Key Technical Decisions
- Build system: hatchling + hatch-vcs (auto-version from git tags)
- CLI: Typer + Rich
- LLM on Jetson: llama.cpp + GGUF Q4_K_M
- Vision on Jetson: Torch-TensorRT for ViT
- Python layout: src/ layout
- Tests: pytest + hardware markers
