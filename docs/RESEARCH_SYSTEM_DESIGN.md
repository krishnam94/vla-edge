# Research Agent System Design

**Goal**: Multi-agent system that researches VLA edge optimization, argues
about approaches, and proposes novel experiments for a publishable paper.

---

## Architecture: 4 Specialist Agents + 1 Arbiter

### Agent 1: Efficiency Researcher
**Focus**: Model compression, quantization, pruning, distillation
**Knowledge base**: QVLA, DyQ-VLA, SQAP-VLA, BitVLA, NanoVLA, ProbeFlow
**Question**: "How to make the model smaller/faster without losing action quality?"

### Agent 2: Systems Researcher
**Focus**: Runtime optimization, pipelining, caching, hardware-aware scheduling
**Knowledge base**: ActionFlow, VLASH, DuoCore-FS, VLA-Perf, AsyncVLA
**Question**: "How to make the inference pipeline faster regardless of model size?"

### Agent 3: Safety Researcher
**Focus**: Robustness, adversarial testing, formal verification, runtime monitoring
**Knowledge base**: SafeVLA, RobustVLA, VLATest, SAFE, ASIMOV, our @safety_contract
**Question**: "Does the optimization break safety? What new failure modes does it create?"

### Agent 4: Evaluation Researcher
**Focus**: Benchmarks, metrics, datasets, reproducibility
**Knowledge base**: vla-eval, LIBERO, CALVIN, MetaWorld, MedAgentBench methodology
**Question**: "How do we measure this properly? What datasets to use? What baselines?"

### Arbiter: Synthesis Agent
**Reads all 4 agents' outputs. Resolves disagreements. Ranks ideas by:**
1. Novelty (has anyone done this?)
2. Feasibility (can we implement this in vla-edge?)
3. Measurability (can we show results on public datasets?)
4. Paper potential (would a venue accept this?)

---

## Discussion Protocol

```
Round 1: Each agent proposes 2-3 optimization ideas from their specialty
Round 2: Agents critique each other's proposals (cross-examination)
Round 3: Agents defend or revise based on critiques
Arbiter: Synthesizes top 3 ideas with experimental plan
```

---

## Public Datasets for Measurement

| Dataset | Tasks | Size | Access |
|---------|-------|------|--------|
| LIBERO | 130 sim tasks (4 suites) | ~2K episodes | HuggingFace |
| CALVIN | Language-conditioned manipulation | 24 tasks | GitHub |
| MetaWorld | 50 manipulation tasks | Continuous | pip install |
| SimplerEnv | Google Robot, WidowX | 4 envs | GitHub |
| Bridge V2 | Real robot, diverse tasks | 60K demos | HuggingFace |

LIBERO is the standard. If we beat baselines on LIBERO with ProbeFlow,
that's a paper.

---

## Paper Angle

**Title candidate**: "Adaptive Flow Matching Scheduling for Edge-Deployed VLA
Models: Training-Free Speedup with Safety-Aware Step Allocation"

**Novel contribution**: ProbeFlow + safety-aware step allocation. When safety
metrics detect the robot is in a precision-critical phase (close to objects,
high velocity), allocate MORE steps. When in transit (open space, low velocity),
allocate FEWER. This connects optimization to safety - nobody has done this.

**Venue targets**: CoRL 2026, ICRA 2027, NeurIPS Robot Learning Workshop

---

## Experimental Plan

1. Baseline: SmolVLA 10-step on LIBERO (success rate, latency)
2. ProbeFlow: SmolVLA with adaptive steps (success rate, latency, step distribution)
3. Safety-aware ProbeFlow: add safety metrics to step allocation
4. Ablation: epsilon sensitivity, n_min/n_max, dt_probe
5. Hardware: Mac CPU, CUDA GPU, Jetson Orin Nano (when available)
