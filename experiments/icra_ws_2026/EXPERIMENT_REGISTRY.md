# Experiment Registry - SafeContract ICRA WS Paper

Track all experiments, validation status, and paper linkage.
Updated by `/experiment` skill. Audited before submission.

---

### EXP-A: Baseline Violation Profiling
- **Hypothesis**: SmolVLA produces OOB actions on LIBERO
- **Script**: Interactive session (no standalone script)
- **Results**: `results/exp_a_baseline.json`
- **Key metrics**: 100% OOB rate, 544 violations, range [-2.16, 3.70]
- **Paper section**: Section 4.1 (EXP-A)
- **Status**: VALIDATED (2026-04-04)
- **Issues**: No standalone generation script; queue_flushed claimed but unverifiable from code

### EXP-B: Method Comparison
- **Hypothesis**: SafeContract catches violations naive clip misses
- **Script**: `run_paper_experiments.py::exp_b_method_comparison()`
- **Results**: `results/exp_b_comparison.json`
- **Key metrics**: 2019 total (680 bounds + 1339 velocity), naive misses 199 velocity
- **Paper section**: Section 4.2 (EXP-B), Table I
- **Status**: VALIDATED (2026-04-04)

### EXP-D: Pareto Strictness Sweep
- **Hypothesis**: Tighter contracts catch more violations but clip more
- **Script**: `run_paper_experiments.py::exp_d_pareto_sweep()`
- **Results**: `results/exp_d_pareto.json`
- **Key metrics**: 948 (loose) to 2561 (tight) violations
- **Paper section**: Section 4.3 (EXP-D), Table IV
- **Status**: VALIDATED (2026-04-04)

### EXP-E: Overhead Microbenchmark
- **Hypothesis**: SafeContract overhead is negligible vs VLA inference
- **Script**: `run_paper_experiments.py::exp_e_overhead()`
- **Results**: `results/exp_e_overhead.json`
- **Key metrics**: 15.6 +/- 3.8 us (10 trials), 0.000075% of inference
- **Paper section**: Section 4.4 (EXP-E), Table II
- **Status**: VALIDATED (2026-04-04)

### EXP-G: Controlled Ablation
- **Hypothesis**: Zero false positives on clean policies, 100% catch on corrupted
- **Script**: `exp_controlled_ablation.py`
- **Results**: `results/exp_g_controlled_ablation.json`
- **Key metrics**: Clean=0/0%, Drifting=115/42%, Jerky=75/12%
- **Paper section**: Section 4.5 (EXP-G), Table III
- **Status**: VALIDATED (2026-04-04)
- **Issues**: 100-step sequences (paper correctly states 100)

### EXP-H: Violation Fingerprinting + Shift Detection (Synthetic)
- **Hypothesis**: Different tasks produce distinct violation patterns
- **Script**: `exp_violation_fingerprint.py`
- **Results**: `results/exp_h_fingerprint_shift.json`
- **Key metrics**: 4 task fingerprints, p<0.05 all 6 pairs, CV<0.38
- **Paper section**: Section 4.6 (EXP-H), Fig 2
- **Status**: VALIDATED (2026-04-04)
- **Issues**: Synthetic distributions, not real VLA. Caveat in paper.

### EXP-H-REAL: Real SmolVLA Task Fingerprints (4 suites)
- **Hypothesis**: Real SmolVLA produces distinct patterns on different LIBERO tasks
- **Script**: Inline (from v14 session)
- **Results**: `results/exp_real_task_fingerprints.json`
- **Key metrics**: spatial=60% x OOB, goal=90% yaw, object=40% yaw
- **Paper section**: Section 4.6 (validation paragraph)
- **Status**: VALIDATED (2026-04-04)
- **Issues**: Only 10 steps per task, task 0 only per suite

### EXP-COMPREHENSIVE: All 40 LIBERO Tasks
- **Hypothesis**: Fingerprints remain distinct across all tasks within suites
- **Script**: `exp_comprehensive_fingerprints.py`
- **Results**: `results/exp_comprehensive_fingerprints.json` (pending)
- **Status**: RUNNING

### EXP-NOISE: Noise Injection Ablation
- **Hypothesis**: Violation rate scales with noise level, zero on clean data
- **Script**: `exp_noise_ablation.py`
- **Results**: `results/exp_noise_ablation.json` (pending)
- **Status**: RUNNING

### EXP-TEMPORAL: Temporal Violation Evolution
- **Hypothesis**: Violation rates vary within episodes by phase
- **Script**: `exp_temporal_evolution.py`
- **Results**: `results/exp_temporal_evolution.json` (pending)
- **Status**: RUNNING

### EXP-LIBERO90: Full libero_90 Fingerprints
- **Hypothesis**: 90-task suite shows diverse violation patterns
- **Script**: `exp_libero90_fingerprints.py`
- **Results**: `results/exp_libero90_fingerprints.json` (pending)
- **Status**: RUNNING
