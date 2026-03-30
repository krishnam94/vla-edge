# vla-edge

**Profile, optimize, validate, and deploy VLA models to edge hardware.**

VLA (Vision-Language-Action) models power the next generation of robots - but deploying them to edge devices like Jetson Orin Nano is painful. OpenVLA has [15+ open issues](https://github.com/openvla/openvla/issues) about deployment failures. LeRobot [closed Jetson support as "Not Planned"](https://github.com/huggingface/lerobot/issues/819). Nobody provides tooling for the transition from cloud training to edge deployment.

vla-edge fills that gap: one toolkit to profile latency, validate safety, and prepare VLA models for real hardware.

## Quick Start

```bash
pip install vla-edge
vla-edge check
```

```
vla-edge - Hardware Check
+---------+-----------+
| Backend | Available |
+---------+-----------+
| cpu     | YES       |
| cuda    | NO        |
| mps     | YES       |
| jetson  | NO        |
+---------+-----------+
Active backend: apple-arm-mps
  Memory: 24576 MB
  Supported dtypes: fp32, fp16

Models: smolvla, openvla
```

## First Published Mac VLA Benchmarks

SmolVLA (450M) on Mac Air M3:
- **Cold start (VLM forward)**: 52 seconds
- **Cached (action expert)**: 3.4 ms
- **Amortized (50-step chunks)**: 1.04s per step, ~1 FPS
- **99.9% of latency is in the VLM**, not the action expert (Mac-specific finding)

## What It Does

- **Profile** - Measure latency (avg/p50/p95/p99), memory, FPS with statistical rigor (GC-disabled, stddev, CV warning)
- **Validate** - Check action safety (bounds, velocity, acceleration, workspace) before deploying to a real robot
- **Safety contracts** - `@safety_contract` decorator enforces physical limits on ANY predict() method at runtime
- **Optimize** - GGUF quantization recommendations, ONNX per-component export
- **Deploy** - Tested recipes for specific model + hardware combos

## Why This Exists

Every VLA deployment framework skips safety validation. [OpenVLA's deploy.py sends raw actions to your robot with zero clipping.](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py) Here's what vla-edge adds:

| Problem | Existing tools | vla-edge |
|---------|---------------|----------|
| Quantization crashes ([#145](https://github.com/openvla/openvla/issues/145), [#286](https://github.com/openvla/openvla/issues/286)) | bitsandbytes + broken `.to()` | GGUF path via llama.cpp (coming v0.2) |
| No Jetson deployment path ([#819](https://github.com/huggingface/lerobot/issues/819)) | "Not Planned" | First-class Jetson backend |
| No ONNX/TRT export ([#3146](https://github.com/huggingface/lerobot/issues/3146)) | Zero maintainer response | `vla-edge optimize` (coming v0.2) |
| No safety validation | Raw unbounded actions | Per-joint bounds, velocity limits, workspace checks |
| No hardware benchmarks ([#3191](https://github.com/huggingface/lerobot/issues/3191)) | "Flying blind" | Reproducible profiling with statistical metrics |

## Supported Models

| Model | Params | Type | Status |
|-------|--------|------|--------|
| SmolVLA | 450M | Flow matching | Supported |
| OpenVLA | 7B | Autoregressive | Supported |
| NanoVLA | 140-520M | Dynamic routing | Planned |

## Supported Hardware

| Backend | Status | Notes |
|---------|--------|-------|
| CPU (Mac/Linux) | Supported | Development + baseline profiling |
| CUDA (desktop GPU) | Supported | Fast profiling |
| Apple MPS | Supported | Mac GPU acceleration (auto-detected) |
| Jetson Orin Nano | Supported (backend ready, hardware testing pending) | 67 TOPS, 8GB |

## CLI Commands

```bash
# Check hardware
vla-edge check

# Profile a model
vla-edge profile smolvla --hardware cpu --iterations 50

# Validate safety
vla-edge validate smolvla --action-bound 1.0 --velocity-limit 0.1

# List models
vla-edge models

# Show version
vla-edge version
```

## Architecture: Extensible by Design

Adding a new hardware backend or VLA model requires exactly **one file** and **zero changes to core code**.

```
New backend:  backends/hailo.py  + @register_backend("hailo")
New model:    models/openvla.py  + @register_model("openvla")
```

Auto-detection priority: Jetson -> CUDA -> MPS -> CPU

```python
from vla_edge.registry import get_backend

backend = get_backend("auto")  # Best available
caps = backend.get_capabilities()
print(f"{caps.name}: {caps.memory_mb}MB, dtypes={caps.supported_dtypes}")
```

## Safety Contracts (Novel)

Enforce physical safety limits on ANY VLA model's predict() method - regardless of what the neural network outputs. Inspired by design-by-contract and runtime verification from formal methods.

```python
from vla_edge.validate.contract import safety_contract

@safety_contract(
    action_range=[-1.0, 1.0],          # Per-joint bounds
    joint_velocity_max=0.1,            # Max change per step (rad/s)
    workspace_bounds=[[-0.5, 0.5],     # End-effector x
                      [-0.5, 0.5],     # End-effector y
                      [0.0, 0.8]],     # End-effector z
    on_violation="warn",               # "clip" (silent), "warn" (log), "raise" (error)
)
def predict(self, image, instruction, state=None):
    return self.model(image)  # raw output, possibly unsafe
    # The decorator ALWAYS returns safe actions
```

Nobody else has this for VLA models. Every other framework sends raw neural network output directly to the robot.

## Safety Validation

```python
from vla_edge.validate.safety import SafetyConfig, validate_actions
from vla_edge.validate.guard import SafetyGuard

# Configure safety limits
config = SafetyConfig(
    action_bounds=np.array([[-1, 1]] * 7),      # Per-joint limits
    max_velocity=np.full(7, 0.1),                # Max action change/step
    workspace_bounds=np.array([[-0.5, 0.5]] * 3) # End-effector box
)

# Validate a trajectory
result = validate_actions(actions, config)
print(f"Safe: {result.is_safe}, violations: {len(result.violations)}")

# Or enforce inline during inference
guard = SafetyGuard(config)
guarded = guard.safe_infer(backend, model, observation)
# guarded.clipped_actions are always within bounds
```

## Installation

```bash
# Base (CPU profiling + safety)
pip install vla-edge

# With SmolVLA support
pip install vla-edge[smolvla]

# With CUDA
pip install vla-edge[cuda]

# Everything
pip install vla-edge[all]
```

Requires Python >= 3.10. Python 3.12 recommended (lerobot 0.4.4 compatibility).

## Roadmap

- **v0.1.0** (current) - SmolVLA profiling + safety validation + CLI
- **v0.2.0** - GGUF quantization, ONNX/TRT export, leaderboard, OpenVLA adapter
- **v0.3.0** - Community backends (Hailo, Qualcomm), ICRA paper, entry points for plugins

## Contributing

Adding a backend or model? See:
- `src/vla_edge/backends/base.py` - HardwareBackend ABC (4 methods)
- `src/vla_edge/models/base.py` - VLAModel ABC (2 methods + 1 property)
- Each is one file with a `@register_backend` or `@register_model` decorator

## Research Foundation

Built on insights from 30+ papers. Key references:
- [QVLA (ICLR 2026)](https://arxiv.org/abs/2602.03782) - Action-centric quantization
- [LiteVLA-Edge](https://arxiv.org/abs/2603.03380) - 6.6 Hz on Jetson Orin (our baseline)
- [VLA-Perf (NVIDIA)](https://arxiv.org/abs/2602.18397) - Analytical profiling model
- [SmolVLA](https://arxiv.org/abs/2506.01844) - 450M param flow matching VLA

## License

Apache 2.0
