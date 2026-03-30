# Learning Journal: VLA Edge Deployment

A growing document that explains concepts, decisions, and discoveries as we build
vla-edge. Written to be useful for Manning VLA book chapters (especially Ch 9-11).

Each entry explains: what the concept is, why it matters for edge deployment,
and where to go deeper.

---

## Part 1: Foundations (Phase 1)

### What is a VLA model?

A Vision-Language-Action model takes three inputs - a camera image (vision),
a natural language instruction (language), and optionally robot state - and
outputs a robot action (typically 6-7 numbers: x/y/z position, rotation,
and gripper open/close).

The architecture is usually: frozen vision encoder (ViT/SigLIP) + small LLM
backbone (0.5-7B params) + action head (MLP, diffusion, or flow matching).

**Why it matters for edge**: The LLM backbone is the bottleneck. On a Jetson
with 8GB shared memory, a 7B parameter model in FP16 needs 14GB - it doesn't
fit. This is the core problem vla-edge solves.

**Key models by size**:
- SmolVLA: 450M params (fits on Jetson Orin Nano in FP16)
- NanoVLA: 140-520M params (designed for edge)
- MiniVLA: ~1B params (tight on Orin Nano)
- OpenVLA: 7B params (needs quantization or bigger hardware)
- pi0: ~3B params (JAX-based, complex deployment)

Reference: [SmolVLA paper](https://arxiv.org/abs/2506.01844) |
[OpenVLA paper](https://arxiv.org/abs/2406.09246) |
[Efficient VLA Survey](https://arxiv.org/abs/2510.24795)

---

### Why edge deployment matters for robotics

Cloud inference adds 50-200ms of network latency. A robot arm moving at 1 m/s
travels 5-20cm during that delay - enough to miss a grasp or collide with an
obstacle. Real-time robot control needs 10-30 Hz (33-100ms per cycle), which
means inference must happen on-device.

Other reasons:
- **Reliability**: Field robots (warehouses, disaster zones) can't depend on WiFi
- **Privacy**: Home robots processing camera feeds should stay local
- **Cost**: Cloud inference at scale is expensive for always-on robots
- **Physics**: Millisecond control loops can't survive network round-trips

**The gap today**: Most VLA models are developed and tested on A100/H100 GPUs.
Deploying to a $249 Jetson Orin Nano with 8GB shared memory is a completely
different engineering challenge. Nobody provides tooling for this transition.

Reference: [VLA-Perf: 15 deployment takeaways](https://arxiv.org/abs/2602.18397) |
[deepsense.ai: "Most VLAs never make it out of the lab"](https://deepsense.ai/blog/we-put-embodied-ai-on-a-100g-device-why-most-vlas-choke-on-the-edge-and-the-architecture-that-didnt/)

---

### Hardware backend abstraction - why we use it

Different hardware needs different inference strategies:
- **CPU**: PyTorch eager mode. Slow but works everywhere.
- **CUDA desktop GPU**: PyTorch + torch.compile or TensorRT. Fast, lots of memory.
- **Jetson (Ampere, 8GB)**: Split strategy - TensorRT for vision encoder, llama.cpp
  for LLM backbone, PyTorch for action head. Memory is the constraint.
- **Future: Hailo, Qualcomm**: Completely different runtimes (HailoRT, QNN SDK).

We abstracted this into a `HardwareBackend` ABC with 4 methods:
`is_available()`, `get_capabilities()`, `load_model()`, `infer()`.

This follows the pattern of:
- ONNX Runtime's Execution Providers (GPU, CPU, TensorRT, OpenVINO, etc.)
- PyTorch's device backends (cpu, cuda, mps, xla)
- LeRobot's Robot class (6 methods, one per robot type)

The key insight from ONNX Runtime: use a **priority chain with fallback**.
If TensorRT is available, use it. If not, fall back to CUDA. If not, CPU.
The user writes `--hardware auto` and gets the best available option.

Reference: [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/) |
[LeRobot hardware plugins](https://huggingface.co/docs/lerobot/integrate_hardware)

---

### Decorator registry pattern

Instead of a complex plugin system, we use a simple pattern:

```python
@register_backend("jetson")
class JetsonBackend(HardwareBackend):
    ...
```

The decorator adds the class to a dict. That's it. ~10 lines of infrastructure.

This is the same pattern used by:
- lm-evaluation-harness (`@register_model`)
- HuggingFace transformers (model auto-classes)
- pytest (marker registration)

We chose this over `entry_points` (pip plugin discovery) because we don't have
external contributors yet. When we do (v0.3+), we'll add entry_points on top
of the existing decorator system - no breaking changes needed.

Reference: [Python plugin patterns](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/) |
[lm-eval model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)

---

### Why action safety validation exists

OpenVLA's `deploy.py` returns raw predicted actions with **zero safety checking**.
If the model predicts a joint angle of 500 degrees or an end-effector velocity
of 10 m/s, the robot tries to execute it. This can damage hardware or hurt people.

LeRobot has `EEBoundsAndSafety` - a basic workspace bounds checker. It's the
ONLY existing implementation in the open-source VLA ecosystem.

We provide configurable safety with multiple dimensions:
- **Action bounds**: Per-joint min/max (prevents impossible joint angles)
- **Velocity limits**: Max change between consecutive actions (prevents jerky motion)
- **Acceleration limits**: Max rate of velocity change (prevents jerk)
- **Workspace bounds**: 3D box the end-effector must stay within
- **Severity levels**: "warning" vs "critical" (bounds violation is critical,
  velocity spike might just be a warning)

**The key insight**: In robotics, safety isn't just "did the task succeed."
A task can succeed while violating safety constraints (e.g., the robot picked
up the cup but slammed into the table on the way). vla-edge reports BOTH
task success AND safety compliance.

Reference: [OpenVLA deploy.py](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py) |
[Modular Safety Guardrails paper](https://arxiv.org/abs/2602.04056)

---

### Jetson Orin Nano Super - what you need to know

**The hardware**: NVIDIA Ampere GPU with 1024 CUDA cores, 67 TOPS INT8, 8GB
LPDDR5 unified memory (shared between CPU and GPU). Compute capability 8.7.
Costs ~$249. Runs Ubuntu 22.04 via JetPack 6.2.

**What "unified memory" means**: Unlike a desktop GPU where CPU has 32GB RAM
and GPU has 24GB VRAM (separate), on Jetson it's one pool of 8GB shared by
both. If your model uses 6GB of GPU memory, the CPU only has 2GB left for
the OS, data loading, and everything else. Budget carefully.

**What works on it**:
- SmolVLA in FP16 (~1GB) - comfortable
- OpenVLA 7B in Q4 GGUF (~4GB) - tight but possible
- TensorRT for vision encoders (ViT) - good acceleration
- llama.cpp for LLM inference - proven path (LiteVLA-Edge: 6.6 Hz)

**What does NOT work**:
- TensorRT-LLM: causes kernel panics on Orin Nano. Only works on AGX Orin+.
- Standard `pip install torch`: fails on aarch64. Must use NVIDIA's JPL wheels.
- FP16 for some models: Gemma attention layers overflow in FP16 on Jetson.

**Critical lesson**: Don't extrapolate cloud benchmarks to edge. A model that
runs at 6 Hz on an RTX 4090 might run at 0.5 Hz on Orin Nano. The performance
degradation is non-linear due to memory bandwidth bottlenecks.

Reference: [NVIDIA Jetson specs](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) |
[LiteVLA-Edge](https://arxiv.org/abs/2603.03380) |
[Cross-Platform VLA Scaling](https://arxiv.org/abs/2509.11480) |
[OpenVLA TensorRT on Jetson forum thread](https://forums.developer.nvidia.com/t/tensorrt-cross-compilation-openvla-model-pytorch-for-jetson-orin-converting-on-gpu-server-8x-l20/346524)

---

### GGUF quantization - the proven edge path

GGUF (GPT-Generated Unified Format) is a file format designed for efficient
LLM inference via llama.cpp. It supports multiple quantization levels:

| Format | Bits/weight | Model quality | Size reduction |
|--------|------------|---------------|----------------|
| F16 | 16 | Baseline | 1x |
| Q8_0 | 8 | ~99% of F16 | 2x smaller |
| Q4_K_M | 4.5 avg | ~95-97% of F16 | 3.5x smaller |
| Q4_0 | 4 | ~93-95% of F16 | 4x smaller |

**Why GGUF over bitsandbytes (the standard LLM quantization)?**
OpenVLA has THREE open issues (#145, #286, #287) about bitsandbytes quantization
crashing due to `.to()` incompatibility. The root cause: bitsandbytes wraps
tensors in a way that breaks PyTorch's device placement. No fix available.

GGUF + llama.cpp avoids this entirely - it's a completely separate inference
runtime. No PyTorch device placement issues.

**Why Q4_K_M specifically?** The "K" variants use a smarter quantization that
allocates more bits to important weights and fewer to less important ones.
Q4_K_M is the sweet spot: 4.5 bits average, retains 95-97% of F16 quality,
and fits a 7B model in ~4GB. Proven by LiteVLA-Edge at 6.6 Hz on Jetson.

**Important VLA-specific insight (from QVLA, ICLR 2026)**: Standard LLM
quantization (SmoothQuant, AWQ) is suboptimal for VLAs. Small action
deviations compound over a trajectory - a 2% error per step becomes 20%
after 10 steps. QVLA showed that action-centric, channel-wise bit allocation
outperforms SmoothQuant by 22.6%. This is a key optimization for vla-edge v0.2.

Reference: [llama.cpp GGUF format](https://github.com/ggerganov/llama.cpp) |
[QVLA paper](https://arxiv.org/abs/2602.03782) |
[OpenVLA quantization issues](https://github.com/openvla/openvla/issues/145)

---

### Two types of action generation: autoregressive vs flow matching
**Manning Chapter: 10 (Optimization)**

VLA models generate actions in two fundamentally different ways. This matters
because the optimization strategy is completely different for each.

**Autoregressive (OpenVLA, pi0, RT-2)**: The model generates action tokens
one at a time, like a text LLM generating words. For a 7-DoF robot arm,
that's 7+ sequential decode steps. Each step waits for the previous one.
Latency scales linearly with action dimensions.

**Flow matching (SmolVLA, TinyVLA)**: The model takes Gaussian noise and
iteratively refines it into actions through a fixed number of denoising steps
(SmolVLA uses 10). All action dimensions are generated simultaneously at
each step. Latency scales with denoising steps, not action dimensions.

**Why this matters for optimization**:
- For autoregressive VLAs: optimize the decode loop. PD-VLA (parallel
  fixed-point iteration) gives 4x. Speculative decoding helps. Fewer action
  tokens (FAST tokenizer: 10x compression) helps.
- For flow matching VLAs: optimize each denoising step. Fewer steps helps
  (OneDP: single-step distillation, 41x speedup). Action expert pruning helps.
  Token pruning is less relevant.

Google's profiling paper found 75% of latency is in action generation for
AUTOREGRESSIVE VLAs. For flow matching VLAs like SmolVLA, the split is
different - the action expert's 10 denoising steps dominate, but they're
each faster than autoregressive decode steps.

**Key insight for vla-edge**: We need to detect which type a model uses and
apply the right optimization strategy. This is why the HardwareBackend
abstraction is important - the backend can choose different optimization
paths based on model type.

Reference: [Characterizing VLA Bottlenecks](https://arxiv.org/abs/2603.02271) |
[SmolVLA](https://arxiv.org/abs/2506.01844) |
[PD-VLA](https://arxiv.org/abs/2503.02310) |
[OneDP](https://arxiv.org/abs/2410.21257) |
[VLASH](https://arxiv.org/abs/2512.01031) |
[EdgeVLA](https://arxiv.org/abs/2507.14049)

---

## Part 2: The Story So Far

### How we chose this project (2026-03-29)

Started with a broad question: "What open-source project should I build?"
Ran 20+ research agents across healthcare AI, robotics, LLM tooling.

**Ideas we evaluated and rejected**:
- `medeval` (medical LLM eval) - scored 90/100 but Audere conflict (Foundation Foundry)
- `SurgVLA-Bench` (surgical VLA benchmark) - scored 71/100, needs surgical collaborator
- `vla-safety` (VLA safety testing) - PKU-Alignment's SafeVLA (NeurIPS spotlight) dominates
- `vla-bench-auto` (VLA benchmarking) - Allen AI's vla-eval already exists
- `VLAs-from-scratch` (educational) - scored 95/100 raw but 3-author ownership conflict

**Why vla-edge won**: Scored 90/100. Zero collaboration dependency. Uses Krishnam's
Jetson hardware. Directly produces Manning book content (Ch 9-11). No competition
for the integrated "profile + optimize + validate + deploy" workflow.

Full research: `~/Desktop/docs/notes/PROJECT_RESEARCH_MASTER.md`

### Architecture decisions (2026-03-30)

Chose two thin ABCs + decorator registries after studying LeRobot (6-method
Robot class), vla-eval (1-method ModelServer), and lm-eval (3-method LM).
The pattern: keep the mandatory interface tiny, let subclasses add specifics.

Rejected a function-based approach (no extension contract) and a heavy plugin
system with entry_points (premature for v0.1 with 1-2 backends).

---

### Why `trust_remote_code` matters for robotics

When you call `AutoModel.from_pretrained("some-model", trust_remote_code=True)`,
HuggingFace downloads and EXECUTES arbitrary Python from that model's repo.
Most VLA models (OpenVLA, SmolVLA) require this because they have custom
architecture code that isn't in the transformers library yet.

The risk: a malicious model repo could execute code that:
- Exfiltrates your HuggingFace token or SSH keys
- Modifies your robot control pipeline
- On a robot with physical actuators, this has real-world consequences

Our approach: default to `trust_remote_code=False`. The user must explicitly
opt in via CLI flag. We log a warning when enabled. For known-safe models
(registered in our model registry), we can allowlist them in the future.

**Lesson learned**: Our first version had `trust_remote_code=True` hardcoded
in all three backends. The critic agent caught this as a critical security issue.

Reference: [HuggingFace security advisory on remote code](https://huggingface.co/docs/hub/security-code) |
[Arbitrary code execution via pickle](https://huggingface.co/docs/hub/security-pickle)

---

### Registry loading - why a boolean flag, not dict emptiness

Our first implementation checked `if _BACKENDS:` to skip re-importing backend
modules. The bug: if a test registers a fake backend, the dict becomes non-empty,
and the real backends (CPU, CUDA, Jetson) never get imported.

Fix: use a `_backends_loaded = False` boolean that's set to True after the first
import attempt, regardless of what's in the dict.

This is a common pattern in Python module initialization. Django's app registry
uses the same approach (`self.ready = False` flag).

---

### Per-timestep vs per-joint violation counting

When validating action safety, we count how many timesteps had violations.
Our first version counted per-joint - so one bad timestep with 3 joints
out of bounds counted as 3 violations. This made `violation_rate` potentially
exceed 1.0 (100%), which is semantically wrong.

Fix: track violated timestep indices in a set, count the set size.

**The broader lesson**: In robotics safety, the unit of concern is usually
the timestep (one control cycle), not individual joints. A robot that makes
one bad move affecting 7 joints made ONE bad decision, not seven.

---

### Non-linear performance on edge hardware

A model that runs at 6 Hz on an RTX 4090 won't run at 3 Hz on hardware with
"half the compute." Edge performance degrades non-linearly because of:
- Memory bandwidth bottlenecks (shared CPU/GPU memory on Jetson)
- Thermal throttling (sustained load on small form factor)
- Cache effects (smaller L2 cache on edge GPUs)
- Kernel launch overhead (bigger fraction of total time on slower hardware)

This is why vla-edge exists: you can't predict edge performance from cloud
benchmarks. You need to measure on the actual target hardware.

Reference: [Cross-Platform VLA Scaling](https://arxiv.org/abs/2509.11480) |
[VLA-Perf](https://arxiv.org/abs/2602.18397)

---

## Part 3: Process Learnings

### Critic-driven development works (2026-03-30)

After building Phase 1, we ran two parallel agents: an architecture critic
and a code reviewer. Together they found 10 issues (3 critical, 5 important,
2 minor). The critical ones (`trust_remote_code`, registry race condition,
CUDA excluding all aarch64) would have caused real problems on first use.

**The process**: propose -> build -> critique -> fix -> document. Not
propose -> critique -> build. The critic is more effective with real code
to review than hypothetical designs.

**What the critic missed**: It didn't catch that `infer()` ignores the
instruction and state fields from the observation dict. Both agents flagged
it as "important" but neither called it critical. In practice, this means
profiling would produce wrong actions (but correct latency numbers), which
is acceptable for Phase 1 profiling-only use.

---

### Why tracemalloc doesn't work for PyTorch memory measurement
**Manning Chapter: 9 (Jetson constraints)**

We initially switched from RSS (Resident Set Size) to `tracemalloc` for CPU
memory measurement, thinking it would be more accurate. The critic caught a
critical flaw: tracemalloc only tracks Python heap allocations. PyTorch allocates
tensors through its own C++ memory allocator (via libc or CUDA), bypassing the
Python heap entirely. So tracemalloc reports near-zero for any real model inference.

RSS delta is noisy (includes GC, shared libs, other processes) but at least
captures the actual memory. On GPU, `torch.cuda.max_memory_allocated()` is
accurate because PyTorch's CUDA allocator tracks everything. On CPU, there's
no equivalent - RSS is the best available option.

**For vla-edge**: CPU memory numbers are labeled "rss_delta_approx" in metadata
to set expectations. The real deployment target (Jetson GPU) uses the accurate
CUDA tracking. CPU numbers are for relative comparison only.

**Key lesson**: tracemalloc is for pure-Python programs. ML workloads with
C++ tensor allocators need different measurement strategies per platform.

---

### GC jitter in latency profiling
**Manning Chapter: 10 (Optimization)**

Python's garbage collector can add 10-50ms of jitter during benchmarking.
If GC runs during one of your 100 profiling iterations, that iteration's
latency spikes and pollutes your p95/p99 numbers.

Fix: `gc.disable()` during the timed loop, re-enable in a `finally` block.
Also report stddev and coefficient of variation (CV) so users can judge
measurement stability. CV > 15% triggers a warning suggesting more
iterations or checking for thermal throttling.

This is the same approach used by PyTorch's `torch.utils.benchmark` and
Python's `timeit` module. The key: disable GC, report variance, let the
user decide if the numbers are trustworthy.

---

### SafetyGuard - inline safety enforcement vs post-hoc validation
**Manning Chapter: 11 (Production Patterns)**

There are two ways to check safety in a robot control loop:

**Post-hoc validation** (what we had before): Run inference, collect all actions,
then validate the sequence. Good for offline analysis of recorded trajectories.
Bad for real-time control - by the time you validate, the robot already executed
the dangerous actions.

**Inline enforcement** (SafetyGuard, new): Wraps every inference call with
clip + validate. The guard sits between the backend and the caller:

```
observation -> backend.infer() -> raw actions -> SafetyGuard -> clipped actions -> robot
                                                     |
                                              violation logged
```

Design choices:
- Guard clips first, THEN validates. The caller always gets safe actions.
- Violations are logged but don't block execution (robot needs to keep moving).
- Guard tracks statistics across calls (violation rate, max velocity observed).
- Summary available via `guard.summary` for reporting.

This is the pattern from industrial robotics: safety limits are enforced in
hardware (e-stops, joint limiters) AND software (velocity clamping, workspace
bounds). The software layer is our SafetyGuard.

Reference: [Modular Safety Guardrails for FM Robots](https://arxiv.org/abs/2602.04056)

---

### Action chunking hides VLM latency (and benchmarking traps)
**Manning Chapter: 10 (Optimization)**

SmolVLA generates 50 actions per VLM forward pass (action chunking). The first
call takes ~52 seconds (full VLM + flow matching). The next 49 calls take
~3-4ms each (returning cached actions). Naive benchmarking that runs the same
input 100 times will report 4ms average - missing the 52s elephant.

Proper VLA benchmarking must report:
- **Cold start**: first frame latency (52s on Mac Air CPU)
- **Cached/amortized**: subsequent frames within a chunk (3ms)
- **Amortized per-step**: cold_start / chunk_size + cached (52s / 50 + 3ms = 1.04s)
- **Chunk exhaustion**: every Nth step re-runs the VLM

This is why our profiler uses different images per iteration and reports
both cold and amortized numbers.

**VLM vs action expert split (hardware-dependent)**:
- Mac Air CPU: 99.9% VLM, 0.1% action expert (memory-bandwidth limited)
- RTX 4090: ~73% VLM, ~27% action (per VLA-Perf, arXiv:2602.18397)
- Jetson Thor: varies by model architecture

The extreme Mac ratio is a MPS-specific finding, not universal. VLA-Perf
and AR-VLA (arXiv:2603.10126) already measured this split on NVIDIA hardware.
Our contribution: first measurements on Mac/MPS.

Reference: [VLA-Perf](https://arxiv.org/abs/2602.18397) |
[AR-VLA component breakdown](https://arxiv.org/abs/2603.10126) |
[Characterizing VLA Bottlenecks](https://arxiv.org/abs/2603.02271)

---

### Safety contracts - design by contract for robot policies
**Manning Chapter: 11 (Production Patterns)**

In software engineering, "design by contract" (from Eiffel language) means
functions declare their preconditions and postconditions, and the runtime
enforces them. We applied this to VLA model prediction:

```python
@safety_contract(action_range=[-1, 1], joint_velocity_max=0.1)
def predict(self, image, instruction, state=None):
    return self.model(image)  # decorated version always returns safe actions
```

The decorator wraps predict() and:
1. Calls the original function (gets raw neural network output)
2. Clips to action_range bounds
3. Clamps velocity (delta from previous action)
4. Clips workspace bounds (end-effector position)
5. Re-clips to action_range after velocity clamping (order matters!)
6. Logs violations for post-hoc analysis

**Why this is novel**: Every other VLA framework sends raw neural network output
directly to the robot. OpenVLA's deploy.py has zero safety checks. LeRobot's
EEBoundsAndSafety is the only existing implementation, and it's imperative
(you call it explicitly), not declarative (the contract IS the safety spec).

**Design decisions**:
- "clip" mode (default) silently enforces - the robot must keep moving
- "warn" mode logs violations - good for development, catches drifting policies
- "raise" mode throws an exception - good for testing, catches bugs early
- Velocity clamping uses the previous action from the LAST call, enabling
  state tracking across an episode without explicit state management

Reference: [Design by Contract](https://en.wikipedia.org/wiki/Design_by_contract) |
[Modular Safety Guardrails](https://arxiv.org/abs/2602.04056)

---

## Concepts Queue (to learn and document next)

- [ ] TensorRT engine building - how it works, why vision encoder but not LLM
- [ ] ONNX export for VLA models - splitting components, handling dynamic shapes
- [ ] Action chunking - predicting N future actions at once, executing at higher frequency
- [ ] Async VLA inference - the VLASH pattern of predicting while executing
- [ ] Diffusion policy heads vs autoregressive decoding - tradeoffs for edge
- [ ] ROS 2 Humble integration on Jetson - nodes, topics, action servers
- [ ] Sim-to-real transfer - domain randomization, what breaks
- [ ] Model profiling methodology - warmup, percentiles, statistical significance
