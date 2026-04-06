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

### System Skills (available in ALL projects via `~/.claude/skills/`)
- `/investigate` - Scientific method orchestrator (plan, run, report)
- `/experiment` - Experiment management with registration + validation
- `/verify-experiments` - Methodology verification checklist
- `/novelty-check` - Prior art search (skeptical reviewer)
- `/synthesize` - Novel idea combination generator
- `/critic` - Architecture criticism (generic; overridden below for VLA)
- `/review-panel` - Multi-persona review (generic; overridden below for VLA)

### Project Commands (VLA-specific, in `.claude/commands/`)
- `/morning` - Start of session - project state + overnight insights
- `/research` - Deep research with 5 VLA-specialist agents (overrides system /research)
- `/critic` - VLA-specific architecture critic (overrides system /critic)
- `/review-panel` - VLA-specific 3-persona review (overrides system /review-panel)
- `/profile` - Profile a VLA model and get optimization suggestions
- `/add-model` - Adding a new VLA model adapter
- `/add-backend` - Adding a new hardware backend
- `/benchmark` - Full benchmark run + results update
- `/doc-hygiene` - Check all docs for staleness and inconsistencies
- `/experiment` - Project experiment registry at `experiments/icra_ws_2026/`

## Conventions
- New backend = one file in backends/ + `@register_backend("name")`. No core changes.
- New model = one file in models/ + `@register_model("name")`. No core changes.
- All backends implement HardwareBackend ABC (4 methods).
- All models implement VLAModel ABC (predict + info).
- Safety metrics are ALWAYS reported. Never optional.
- NEVER hardcode API keys or HuggingFace tokens. Use environment variables.
- NEVER use em dashes in any content. Use hyphens or rewrite.

## Process Rules (HARD - NO EXCEPTIONS)
- **Experiments**: ALWAYS register in `experiments/icra_ws_2026/EXPERIMENT_REGISTRY.md` BEFORE writing any experiment code or launching any experiment agent. Include hypothesis, parameters, design review. No "I'll register later."
- **Research/Papers**: ALWAYS compile findings to `~/Desktop/wiki/` via `/wiki compile`. Every paper read, research session, or substantial learning becomes a wiki article with Future Directions and Gaps sections.
- **Metrics**: ALWAYS verify metric definitions before accepting results. Ask: "Is this metric measuring what we CLAIM?" (e.g., velocity must be consecutive timesteps, not random samples)
- **References**: ALWAYS verify each reference against arxiv/venue before including. Check author, title, year, venue.
- **Claims**: NEVER claim "all X" without testing all X. NEVER present synthetic-data results as evidence of general model behavior without caveat.
- **Formal properties**: ALWAYS verify claims hold for the FULL system, not just a component.

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
- Standard LLM quantization is suboptimal for VLAs. Use action-centric quantization (QVLA, DyQ-VLA).
- For AUTOREGRESSIVE VLAs (OpenVLA, pi0): 75% of latency is in action token generation.
  For FLOW MATCHING VLAs (SmolVLA): bottleneck is 10 denoising steps through action expert. Different optimization strategies needed.
- SmolVLA uses SigLIP vision (512x512), SmolLM2-360M backbone, flow matching action expert (~100M params). NOT autoregressive.
- Training-free parallel decoding (PD-VLA) gives 4x speedup on autoregressive VLAs (not applicable to flow matching).
- 6.6 Hz on Jetson Orin (LiteVLA-Edge) is the baseline to beat. Target: 10+ Hz for v0.1.
- TurboQuant (Google, ICLR 2026): KV cache quantization - low relevance for VLA (short sequences). QVLA is what matters.

## Key Technical Decisions
- Build system: hatchling + hatch-vcs (auto-version from git tags)
- CLI: Typer + Rich
- LLM on Jetson: llama.cpp + GGUF Q4_K_M
- Vision on Jetson: Torch-TensorRT for ViT
- Python layout: src/ layout
- Tests: pytest + hardware markers
