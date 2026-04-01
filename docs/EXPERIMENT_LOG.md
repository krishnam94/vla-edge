# Experiment Log

Track what we ran, what worked, what failed. Single source of truth for results.

---

## Completed Experiments

### EXP-001: SmolVLA Baseline Benchmark (2026-03-31)
- **Model**: SmolVLA 450M, FP32
- **Hardware**: Mac Air M3 CPU
- **Result**: Cold start 52s, cached 3ms, amortized 0.96 FPS
- **Finding**: 99.9% of latency in VLM forward (Mac-specific, ~73% on RTX 4090 per VLA-Perf)
- **Data**: `results/smolvla_mac-air-m3_cpu.json`

### EXP-002: ProbeFlow on SmolVLA (2026-03-31)
- **Model**: SmolVLA 450M + ProbeFlow (epsilon=0.15, n_min=2, n_max=10)
- **Hardware**: Mac Air M3 CPU
- **Result**: Cold 12s (2.4x speedup), 2 steps allocated (minimum)
- **Finding**: High cosine similarity -> trajectory nearly linear for random obs
- **Action divergence**: L1=0.15 (needs task-level validation)
- **Data**: `results/smolvla_mac-air-m3_probeflow.json`

### EXP-003: ICRA Workshop Experiments (2026-03-31)
- **What**: 4 synthetic experiments (no real model)
- **Results**:
  - Safety contract overhead: 27 us (negligible, 1M x less than VLA inference)
  - Violation detection: scales linearly with OOB rate
  - ProbeFlow sensitivity: 36 config points mapped
  - Composition interference: loose=0.6 to very_tight=460 violations/trajectory
- **Data**: `experiments/icra_ws_2026/results/exp*.json`

## Planned Experiments

### EXP-004: SmolVLA on LIBERO (CORRECTED, queue flushed)
- **CRITICAL**: Previous 3.42x was artifact. Real speedup: **1.28x overall, 4.1x cold only**
- **Fix**: Flush action queue before each call (policy.reset())
- **Corrected results** (10 samples, CPU):
  - Baseline: 17.9s avg, 51s cold. ProbeFlow: 14.0s avg, 12s cold.
  - **Actions exceed [-1,1] on ALL observations (max 2.34)** - safety contracts needed
  - Violations: 13 vs 10 (comparable)
- **Data**: `experiments/icra_ws_2026/results/exp4_corrected.json`
- **Lesson 007**: Always flush model caches before benchmarking

### EXP-A: Baseline Violation Profiling (2026-04-01)
- **Model**: SmolVLA on 100 LIBERO observations, queue flushed
- **Finding**: **100% OOB**, range [-2.16, 3.70], 544 violations (121 bounds + 423 velocity)
- **Data**: `experiments/icra_ws_2026/results/exp_a_baseline.json`

### EXP-B: Method Comparison (2026-04-01)
- SafeContract vs AEGIS-lite vs naive clip vs no safety (200 actions)
- Naive clip misses 199 velocity violations. SafeContract = AEGIS output, **17.1x faster**
- **Data**: `experiments/icra_ws_2026/results/exp_b_comparison.json`

### EXP-C: Composition Verification (2026-04-01)
- Composed catches union of individual violations (1,482 = 490 + 675 + 317)
- **Data**: `experiments/icra_ws_2026/results/exp_c_composition.json`

### EXP-D: Pareto Sweep (2026-04-01)
- 5 strictness levels. 948 to 2,561 violations, 0.026 to 0.908 clip magnitude
- **Data**: `experiments/icra_ws_2026/results/exp_d_pareto.json`

### EXP-E: Overhead Microbenchmark (2026-04-01)
- SafeContract clip 0.9us, adaptive 14.6us, AEGIS 13.5us. Contract = 0.000081% of inference
- **Data**: `experiments/icra_ws_2026/results/exp_e_overhead.json`

### EXP-F: Component Ablation (2026-04-01)
- Each catches unique violations. Full = 2,467 (more than any individual)
- **Data**: `experiments/icra_ws_2026/results/exp_f_ablation.json`

### Paper Figures (2026-04-01)
- 4 figures generated: action ranges, Pareto, ablation, overhead
- Location: `paper/icra_ws_2026/figures/`

### EXP-005: Cross-Architecture (partial, 2026-04-01)
- SmolVLA complete (n=5). ACT/Diffusion failed on LeRobot video format.
- **Data**: `experiments/icra_ws_2026/results/exp5_cross_architecture.json`

### EXP-006: SmolVLA on Jetson Orin Nano
- **Status**: BLOCKED - Jetson hardware not set up
- **Guide**: docs/JETSON_SETUP_GUIDE.md

---

## Experiment Naming Convention
- EXP-NNN: sequential, never reused
- Each has: model, hardware, result, finding, data path
- Failed experiments documented with root cause
