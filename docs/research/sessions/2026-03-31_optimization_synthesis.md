# Research Synthesis: Top 3 Paper Ideas

**Date**: 2026-03-31
**Input**: 3 specialist agents (Efficiency, Systems, Safety+Eval) + biorxiv paper analysis
**Method**: Multi-agent debate with novel thinking methods

---

## TOP 3 IDEAS (Arbiter Ranked)

### #1: SAAD - Safety-Aware Adaptive Denoising (STRONGEST)

**Source agents**: Safety + Efficiency + Systems (all three contributed)
**Novel thinking**: TRIZ contradiction resolution (speed vs safety)

**The idea**: Use safety state signals (distance to objects, EE velocity, workspace
bounds proximity, gripper activity) to modulate ProbeFlow's denoising step allocation.
When robot is in a dangerous phase (near objects, high velocity, grasping) -> allocate
MORE steps. When in safe transit -> allocate FEWER steps. Connect optimization to safety.

**Why novel**: Verified by Safety agent - nobody has done this. ProbeFlow uses geometry
(cosine similarity). CoDiG uses constraints in the gradient. SafeVLA uses constrained RL.
NOBODY uses safety state to control step count. The gap is confirmed.

**Algorithm**:
```
safety_score = weighted(d_bounds, 1-velocity, d_objects, 1-gripper_active)  # [0,1]
linearity = cosine_similarity(v_start, v_probe)  # [0,1] from ProbeFlow
difficulty = beta * (1-linearity) + (1-beta) * (1-safety_score)
n_steps = clip(n_min + difficulty * (n_max - n_min), n_min, n_max)
```

**Cost**: Zero extra NFEs beyond ProbeFlow (safety signals come from observation).

**Expected result**: Match ProbeFlow speedup in safe regions, allocate MORE compute
in dangerous phases, reduce safety violations. Shift the Pareto frontier.

**Measurement**: LIBERO 4 suites x 50 episodes x 5 methods. Metrics: success rate,
avg steps, safety violations, action divergence, steps-in-danger-zone correlation.

**Venue**: CoRL 2026 or NeurIPS Robot Learning Workshop

**Implementation**: ~2 days on top of existing ProbeFlow + SafetyGuard code.

---

### #2: SplitPipe - Heterogeneous Component Runtime (HIGHEST SPEEDUP)

**Source agent**: Systems
**Novel thinking**: Bisociation (mobile ML delegate pattern -> VLA components)

**The idea**: Split SmolVLA's VLM into 3 components, each in its optimal runtime:
- SigLIP vision encoder -> ONNX Runtime (2-4x over PyTorch eager for ViTs)
- SmolLM2-360M backbone -> llama.cpp GGUF Q4_K_M (10x over PyTorch FP32 eager)
- Perceiver Resampler -> PyTorch (too custom for other runtimes)

**Why novel**: Nobody has done heterogeneous per-component runtime selection for VLAs.
The TFLite delegate pattern from mobile ML applied to robot policy inference.

**Expected result**: VLM forward 28s -> 2-3.5s (8-12x). Stacks with ProbeFlow.

**Full stack**: SplitPipe + DeltaVLM (cache reuse) + AsyncPrefetch (hide latency)
+ ProbeFlow (fewer action steps) = near-zero stall between chunks.

**Risk**: llama.cpp embedding injection API is the hard part (1 week implementation).

**Measurement**: Same LIBERO protocol. Latency breakdown per component.

**Venue**: ICRA 2027 systems paper

**Implementation**: 1-2 weeks.

---

### #3: QuantProbe - Action-Centric Quantization + Adaptive Steps (MULTIPLICATIVE)

**Source agent**: Efficiency
**Novel thinking**: First principles (minimum compute for action prediction)

**The idea**: Apply QVLA's per-channel action-sensitive quantization to SmolVLA's
action expert, THEN use ProbeFlow to reduce steps. The speedups multiply:
QVLA (1.5x per step) x ProbeFlow (3x fewer steps) = 4.5x total on action expert.

**Why novel**: QVLA tested only on autoregressive VLAs. ProbeFlow tested on Evo-1.
Nobody combined action-centric quantization with adaptive flow matching stepping.

**Risk**: Quantization noise may corrupt ProbeFlow's cosine similarity probe.

**Measurement**: LIBERO with success rate + divergence from FP32/10-step baseline.

**Implementation**: ~1 week (QVLA code is open source, 500 lines core).

---

## CRITICAL FINDING FROM ALL AGENTS

**The 28s VLM forward is NOT the true bottleneck - it's PyTorch's overhead.**

Systems agent analysis: SmolVLA's VLM should take ~4-5s on CPU with proper threading
and runtime selection. The 28s measurement reflects FP32 eager mode with poor thread
config. SplitPipe alone (no novel research) would fix 80% of the latency.

**Implication for paper**: Don't claim ProbeFlow gives 2.4x end-to-end. Claim it gives
5x on the action expert denoising. The VLM bottleneck is an engineering problem
(SplitPipe), not a research contribution.

**Paper framing**: SAAD (safety-aware scheduling) is the RESEARCH contribution.
SplitPipe is the ENGINEERING contribution. Together they form a complete system paper.

---

## EVALUATION PLAN (from Evaluation Agent)

| Metric | What | Why |
|--------|------|-----|
| Task success rate | Standard LIBERO metric | Required |
| Average NFEs per episode | Compute efficiency | Core claim |
| Safety violation rate | From SafetyGuard | Novel metric |
| Steps-in-danger correlation | Safety signal -> step allocation | Proves SAAD works |
| Action smoothness (jerk) | Trajectory quality | Real-world deployment |
| Wall-clock latency (mean + p95) | Actual speed | Edge deployment |

**Baselines**: SmolVLA N=10, N=4, N=2 (fixed), ProbeFlow (linearity only), SAAD (proposed)
**Episodes**: 50 per task x 40 tasks = 2000 per method
**Statistical test**: Paired bootstrap, 95% CI

---

## VIRTUAL BIOTECH INSIGHTS (from paper analysis)

The multi-agent orchestration pattern validates our approach:
- **CSO orchestrator** -> SAAD's fused decision function (routes compute where needed)
- **Specialist agents** -> Per-component optimization (SplitPipe's heterogeneous runtimes)
- **Failure analysis** -> Safety-aware scheduling (analyze WHY actions fail, allocate compute there)
- **Structured schemas** -> Our SafetyConfig + SafetyResult + ProbeFlowStats

---

## RECOMMENDED PAPER STRUCTURE

**Title**: "Safety-Aware Adaptive Denoising for Edge-Deployed Flow Matching VLA Models"

1. Introduction: VLA edge deployment gap, safety-speed tradeoff
2. Background: Flow matching VLAs, ProbeFlow, safety validation
3. Method: SAAD algorithm (safety signals + linearity probe -> fused step allocation)
4. System: SplitPipe for VLM optimization (engineering contribution)
5. Experiments: LIBERO 4 suites, 5 methods, 8 metrics
6. Results: Pareto improvement on safety-speed frontier
7. Ablations: beta sweep, signal ablation, n_min/n_max sensitivity
8. Discussion: Limitations (LIBERO sim only), future (Jetson deployment)

**Sources for this synthesis**:
- ProbeFlow: [arXiv:2603.17850](https://arxiv.org/abs/2603.17850)
- QVLA: [arXiv:2602.03782](https://arxiv.org/abs/2602.03782)
- DyQ-VLA: [arXiv:2603.07904](https://arxiv.org/abs/2603.07904)
- SafeVLA: [arXiv:2503.03480](https://arxiv.org/abs/2503.03480)
- CoDiG: [arXiv:2505.13131](https://arxiv.org/abs/2505.13131)
- A2A Flow Matching: [arXiv:2602.07322](https://arxiv.org/abs/2602.07322)
- ActionFlow: [arXiv:2512.20276](https://arxiv.org/abs/2512.20276)
- VLASH: [arXiv:2512.01031](https://arxiv.org/abs/2512.01031)
- Virtual Biotech: [bioRxiv:2026.02.23.707551](https://www.biorxiv.org/content/10.64898/2026.02.23.707551v1)
- LIBERO: [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO)
- SmolVLA: [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- VLA-Perf: [arXiv:2602.18397](https://arxiv.org/abs/2602.18397)
