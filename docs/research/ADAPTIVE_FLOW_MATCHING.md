# Adaptive Flow Matching for VLAs - Research Analysis

Deep dive into ProbeFlow, DFM-VLA, Realtime-VLA V2, and FASTER.
Focus: what's implementable in vla-edge's SmolVLA adapter.

Research date: 2026-03-29

---

## 1. ProbeFlow (arXiv:2603.17850) - PRIMARY TARGET

**Title:** ProbeFlow: Training-Free Adaptive Flow Matching for Vision-Language-Action Models
**Authors:** Zhou Fang, Jiaqi Wang, Yi Zhou, Qiongfeng Shi (Southeast University, China)
**Date:** March 18, 2026
**Code:** NOT released (no GitHub link in paper)
**Paper:** [arxiv.org/abs/2603.17850](https://arxiv.org/abs/2603.17850)

### What It Does

ProbeFlow dynamically decides how many denoising steps to run in the flow matching
action head. Instead of always running N=10 (SmolVLA) or N=50 (their baseline),
it uses a single "probe" forward pass to estimate trajectory linearity and skips
steps when the flow is simple.

### How the Probe Works (Algorithm)

1. **Initial velocity**: Compute `v_start = v_theta(x_0, 0, c)` from pure noise
2. **Lookahead step**: Take exploratory step `x_probe = x_0 + v_start * dt_probe` (default dt_probe=0.5)
3. **Probe velocity**: Evaluate `v_probe = v_theta(x_probe, dt_probe, c)` at the probe location
4. **Cosine similarity**: Compute `S = cos(v_start, v_probe)`
   - S near 1.0 = trajectory is nearly linear, few steps needed
   - S near 0.0 = trajectory is curved, need dense integration
5. **Step allocation**: `N = clip(N_min + floor((1-S)/epsilon) * delta_N, N_min, N_max)`
   - epsilon is a sensitivity parameter (NOT a hard threshold)
   - N_min=2, N_max=10, delta_N=2 in their experiments

**Cost:** Exactly 2 extra NFEs (network forward evaluations) for the probe, but
these evaluations are reused in the actual integration when the trajectory is
linear. So in the best case, cost is zero extra.

### Results

| Benchmark | Method | Avg Steps | Solver Latency | Success Rate |
|-----------|--------|-----------|---------------|-------------|
| MetaWorld | Fixed Euler N=50 | 50.0 | 235.7ms | 82.5% |
| MetaWorld | **ProbeFlow** | **2.6** | **15.9ms** | **83.2%** |
| MetaWorld | Speedup | **14.8x** | **14.8x** | **+0.7%** |
| LIBERO | Fixed Euler N=50 | 50.0 | 278.7ms | 92.5% |
| LIBERO | **ProbeFlow** | **4.5** | **32.7ms** | **88.7%** |
| LIBERO | Speedup | **8.5x** | **8.5x** | **-3.8%** |
| Real robot | Fixed Euler N=50 | 50.0 | 270.3ms | 8/10 |
| Real robot | **ProbeFlow** | **2.1** | **12.26ms** | **7/10** |

### VLA Model Tested

ProbeFlow was tested on **Evo-1** - a 0.77B parameter VLA using:
- InternVL3-1B visual backbone (frozen)
- 8-layer Diffusion Transformer (DiT) action head, hidden dim 1024
- Flow matching with Euler integration

**NOT tested on SmolVLA, OpenVLA, or pi0.** But the approach is model-agnostic -
it only touches the flow matching ODE solver, not the model architecture.

### Real Robot Setup

- 7-DOF UFACTORY xArm7 + 6-DOF Inspire dexterous hand
- Orbbec Gemini 360 wrist camera
- Pick-and-place task

### Truly Training-Free?

**YES.** No retraining, no fine-tuning, no new parameters. It's purely an
inference-time modification to the ODE solver loop. Drop-in replacement.

### Limitations (from paper)

1. Sensitivity parameter epsilon needs calibration per domain
2. Fixed lookahead horizon dt_probe may not handle "highly volatile trajectories"
3. Not validated on "extreme non-linear dynamics in highly complex, contact-rich tasks"
4. LIBERO shows 3.8% success rate drop - the probe sometimes under-allocates steps for complex long-horizon tasks

### Applicability to SmolVLA in vla-edge

**HIGH.** Here's why:

1. SmolVLA uses 10 Euler steps in its flow matching action expert. ProbeFlow could reduce this to 2-4 steps for "easy" actions (straight-line motions, waiting, transport).

2. The modification is entirely in the denoising loop - `VLAFlowMatching.select_action()` in LeRobot. No architecture changes.

3. SmolVLA's action expert is ~100M params. Each forward pass through it is the expensive part of inference (10 passes vs potentially 2-3). On Jetson Orin Nano, reducing from 10 to 3 steps would cut action expert time by ~70%.

4. The probe costs 2 NFEs but these get reused. Net cost for simple actions: 0 extra.

**Estimated impact for SmolVLA on Jetson:**
- Baseline: 10 action expert forward passes per chunk
- With ProbeFlow: ~3-4 passes average (estimated from their MetaWorld results)
- Speedup on action decoding: ~2.5-3x
- This matters because action expert is the bottleneck for flow matching VLAs

---

## 2. FASTER (arXiv:2603.19199) - COMPLEMENTARY APPROACH

**Title:** FASTER: Rethinking Real-Time Flow VLAs
**Date:** March 19, 2026
**Code:** [innovator-zero.github.io/FASTER](https://innovator-zero.github.io/FASTER)
**Paper:** [arxiv.org/abs/2603.19199](https://arxiv.org/abs/2603.19199)

### What It Does

Instead of reducing total steps, FASTER makes the FIRST action available sooner.
It uses a "Horizon-Aware Schedule" (HAS) that prioritizes denoising near-term
actions over far-future actions.

### Key Insight

In a 50-action chunk, you need the first action RIGHT NOW but action #50 can
wait. FASTER allocates denoising budget unevenly: the first action gets completed
in a single step, while later actions get more refinement.

### Schedule Formula

- Hit times per action: `u_i = (1 - i/(H-1))^alpha * u_0`
- With default alpha=0.6 and u_0=(N-1)/N, the first action needs only 1 step
- Later actions use progressively more steps

### Results

| Model | Hardware | TTFA Baseline | TTFA FASTER | Speedup |
|-------|----------|-------------|-------------|---------|
| pi0.5 | RTX 4090 | 80.0ms | 62.1ms | 1.29x |
| X-VLA | RTX 4090 | 113.7ms | 44.8ms | 2.54x |
| X-VLA | RTX 4060 | 399.5ms | 129.2ms | 3.09x |

### Training-Free?

**NO.** Requires fine-tuning with a mixed scheduling strategy (50% HAS, 50%
constant schedule during training). This is a significant barrier for adoption.

### VLA Models Tested

pi0.5 (Physical Intelligence) and X-VLA. Not tested on SmolVLA.

### Applicability to SmolVLA in vla-edge

**MEDIUM.** The concept is interesting but:
- Requires retraining SmolVLA (not training-free)
- SmolVLA already uses n_action_steps=50 (executes all actions before re-predicting)
- FASTER's benefit is "time to first action" which matters less when you execute the full chunk
- Could combine with ProbeFlow (they're orthogonal)

---

## 3. DFM-VLA (arXiv:2603.26320)

**Title:** DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching
**Authors:** Jiayi Chen, Wenxuan Song, Shuai Chen, Jingbo Wang, Zhijun Li, Haoang Li
**Date:** March 27, 2026
**Project page:** [chris1220313648.github.io/DFM-VLA](https://chris1220313648.github.io/DFM-VLA/)
**Paper:** [arxiv.org/abs/2603.26320](https://arxiv.org/abs/2603.26320)

### What It Does

DFM-VLA uses DISCRETE flow matching (not continuous like SmolVLA/pi0). Actions
are tokenized into discrete tokens, then iteratively refined using a probability
velocity field over the token vocabulary. Unlike autoregressive decoding (which
commits to each token sequentially), DFM-VLA can revise ALL tokens simultaneously
across refinement iterations.

### Discrete vs Continuous Flow Matching

| | SmolVLA (Continuous FM) | DFM-VLA (Discrete FM) |
|---|---|---|
| Action space | Continuous (float vectors) | Discrete (token indices) |
| Flow matching target | Velocity field in R^n | Probability velocity over vocabulary |
| Denoising | Euler integration in continuous space | Token probability updates |
| Error correction | Limited (each step is independent) | Can revise earlier tokens |
| Chunk output | (50, action_dim) continuous | (seq_len,) discrete tokens |

### Architecture

- Base model: **UniVLA** (initializes from robotic video pretraining)
- Tokenizer: Emu3 for text, VQ tokenizers for images/actions
- Two decoding strategies: auxiliary velocity-head design vs action-embedding-guided
- Two-stage decoding: 14 refinement steps + 2 validation steps = 16 total

### Results

| Benchmark | DFM-VLA | OpenVLA | pi0-FAST | Dream-VLA |
|-----------|---------|---------|----------|-----------|
| CALVIN ABCD-D (avg length) | **4.44** | - | - | - |
| LIBERO (avg success) | **95.7%** | - | - | 92.6% |
| Real-world (avg success) | **70.8%** | - | 42.5% | 54.2% |

Speed: 121 tokens/sec with adaptive KV caching (2.4x faster than autoregressive).

### Training-Free?

**NO.** Requires full training (20-32k steps on CALVIN/LIBERO). Different paradigm
entirely from SmolVLA's continuous flow matching.

### Applicability to SmolVLA in vla-edge

**LOW for direct use.** DFM-VLA is a fundamentally different architecture:
- SmolVLA uses continuous actions (no tokenization)
- DFM-VLA requires its own trained model (UniVLA-based)
- Would need to replace SmolVLA entirely, not augment it

**Interesting for future consideration:** The iterative refinement concept could
inspire a "refine-if-uncertain" scheme where easy actions use fewer steps and
uncertain actions get extra refinement passes. This overlaps with ProbeFlow's
adaptive step allocation.

---

## 4. Realtime-VLA V2 (arXiv:2603.26360)

**Title:** Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate
**Authors:** Chen Yang, Yucheng Hu, Yunchao Ma, Yunhuan Yang, Jing Tan, Haoqiang Fan
**Date:** March 27, 2026
**Project page:** [dexmal.github.io/realtime-vla-v2](https://dexmal.github.io/realtime-vla-v2/)
**Paper:** [arxiv.org/abs/2603.26360](https://arxiv.org/abs/2603.26360)

### What It Does

Realtime-VLA V2 focuses on end-to-end system optimization for running VLAs on
real robots. Not about the model itself, but about the entire deployment pipeline:
calibration, planning, control, and learned speed adaptation.

### Key Techniques

1. **Temporal optimization**: Quadratic programming to redistribute acceleration peaks
2. **Spatial optimization**: MPC (model predictive control) using acados for real-time tracking
3. **Speed adaptation**: Learns when to accelerate/decelerate based on task phase
4. **Calibration**: Compensates for system delays (camera: 33ms, exposure: 55ms, proprioception: 50ms, motion tracking: 150ms)
5. **Human-in-the-loop**: Operators provide "throttle" during rollouts to supervise speed

### Hardware

RealSense D435 camera + Airbot Play arm (DOS W1 system)

### Results

Achieves "near-human operating speed approaching hardware limits." Demonstrated on
shirt folding, PCB placement, pick-and-latch. No specific Hz numbers published.

### Applicability to SmolVLA in vla-edge

**LOW for direct code reuse, HIGH for calibration insights.** The system delay
measurements (camera latency, proprioception lag, motion tracking delay) are
directly relevant for our Jetson deployment:
- Camera readout: 33ms (we should measure this on Jetson)
- These delays stack and can exceed the action expert compute time
- Their MPC-based trajectory smoothing could complement our action safety validation

---

## 5. Integration Plan for vla-edge

### Priority 1: ProbeFlow Integration (training-free, high impact)

ProbeFlow is the clear winner for vla-edge. Here's the implementation plan:

**Where to modify:** The flow matching denoising loop in SmolVLA's action expert.

In LeRobot's code, the denoising loop lives in:
```
lerobot/policies/smolvla/modeling_smolvla.py
  -> VLAFlowMatching.select_action()
    -> flow matching loop: 10 Euler steps
```

In vla-edge, we'd add this as an optimization option in the SmolVLA adapter:

```python
# Pseudocode for ProbeFlow integration
def probeflow_denoise(model, x_noise, cached_prefix, config):
    """Adaptive flow matching with linearity probe."""
    # Step 1: Initial velocity
    v_start = model.denoise_step(x_noise, t=1.0, prefix=cached_prefix)

    # Step 2: Lookahead probe
    dt_probe = config.get("probeflow_dt_probe", 0.5)
    x_probe = x_noise + v_start * dt_probe
    v_probe = model.denoise_step(x_probe, dt_probe, prefix=cached_prefix)

    # Step 3: Cosine similarity
    cos_sim = F.cosine_similarity(
        v_start.flatten(), v_probe.flatten(), dim=0
    )

    # Step 4: Adaptive step count
    epsilon = config.get("probeflow_epsilon", 0.1)
    n_min, n_max, delta_n = 2, 10, 2
    n_steps = min(n_max, max(n_min,
        n_min + int((1 - cos_sim.item()) / epsilon) * delta_n
    ))

    # Step 5: Euler integration with allocated steps
    x_t = x_noise
    dt = -1.0 / n_steps
    for step in range(n_steps):
        t = 1.0 - step / n_steps
        v_t = model.denoise_step(x_t, t, cached_prefix)
        x_t = x_t + dt * v_t

    return x_t
```

**Config addition for recipes:**
```yaml
# recipes/smolvla-jetson-orin-nano.yaml
optimize:
  probeflow:
    enabled: true
    dt_probe: 0.5
    epsilon: 0.1       # Lower = more aggressive (fewer steps)
    n_min: 2
    n_max: 10
    delta_n: 2
```

### Priority 2: Measure System Delays (from Realtime-VLA V2)

Before optimizing the model, measure the pipeline delays on Jetson:
- Camera capture latency
- Image preprocessing time
- VLM prefix computation time (one-time per chunk)
- Action expert per-step time (this is what ProbeFlow reduces)
- Action post-processing time
- Total end-to-end latency

The profiler module already handles some of this. Extend it.

### Priority 3: Monitor (from FASTER)

FASTER's insight about "time to first action" mattering is valid even without
retraining. We could implement a simple version:
- Start executing the first few actions from a partially-denoised chunk
- Continue denoising the remaining actions while the robot moves
- This is complementary to ProbeFlow (fewer total steps) + async execution

SmolVLA already hints at this with its "async inference" design (execute current
chunk while computing next). We could take it further.

---

## 6. Comparison Matrix

| Paper | Training-Free | VLA Tested | Speedup | Accuracy Cost | Code | vla-edge Priority |
|-------|:---:|---|---|---|:---:|:---:|
| ProbeFlow | YES | Evo-1 | 8.5-14.8x action head | 0 to -3.8% | No | HIGH |
| FASTER | No (fine-tune) | pi0.5, X-VLA | 1.3-3.1x TTFA | Minimal | Yes | LOW |
| DFM-VLA | No (full train) | UniVLA-based | 2.4x vs AR | +3-16% vs baselines | Page only | LOW |
| Realtime-VLA V2 | Partially | Not specified | Near-human speed | N/A | Page only | MEDIUM (insights) |

---

## 7. Open Questions

1. **ProbeFlow + SmolVLA calibration**: What epsilon value works best for SmolVLA's 10-step baseline? Their paper used N_max=10 already, so for SmolVLA the ceiling is the same but the floor (N_min=2) gives up to 5x speedup.

2. **Probe overhead on Jetson**: The probe costs 2 NFEs. On Jetson with the action expert in ONNX, how much does this cost vs the savings?

3. **Safety interaction**: When ProbeFlow reduces steps, does action smoothness degrade? Need to measure with our safety metrics (jerk, workspace bounds).

4. **Chunk-level vs step-level**: ProbeFlow probes once per chunk. But within a 50-action chunk, some actions might be easy and others hard. Could we do a hierarchical probe?

5. **Combining ProbeFlow + caching**: SmolVLA already caches VLM prefix. ProbeFlow only adds 2 action expert forward passes. The two optimizations are orthogonal and should stack.

---

## Sources

- [ProbeFlow paper](https://arxiv.org/abs/2603.17850)
- [ProbeFlow HTML](https://arxiv.org/html/2603.17850v1)
- [FASTER paper](https://arxiv.org/abs/2603.19199)
- [FASTER project page](https://innovator-zero.github.io/FASTER)
- [DFM-VLA paper](https://arxiv.org/abs/2603.26320)
- [DFM-VLA project page](https://chris1220313648.github.io/DFM-VLA/)
- [Realtime-VLA V2 paper](https://arxiv.org/abs/2603.26360)
- [Realtime-VLA V2 project page](https://dexmal.github.io/realtime-vla-v2/)
- [Evo-1 paper](https://arxiv.org/abs/2511.04555)
- [Evo-1 GitHub](https://github.com/MINT-SJTU/Evo-1)
- [State of VLA at ICLR 2026](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- [SmolVLA paper](https://arxiv.org/abs/2506.01844)
