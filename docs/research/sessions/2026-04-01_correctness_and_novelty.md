# Correctness Review + Novel Directions Synthesis

**Date**: 2026-04-01

## Retracted Claims
1. ~~3.42x speedup~~ -> Real: 1.28x (action queue artifact, Lesson 007)
2. ~~ProbeFlow reduces violations~~ -> Noise (n=20, KV cache reuse)
3. ~~Over-denoising hurts safety~~ -> Artifact (but literature supports the hypothesis)

## Confirmed Claims
1. SmolVLA outputs exceed [-1,1] on ALL LIBERO observations (max 2.34)
2. Safety contract overhead 27us (negligible)
3. Composition theorems correct (sequential clipping = intersection, deadlock unreachable)

## Novel Directions for CoRL Paper (all HIGH novelty)
1. **Adaptive contracts**: Phase-aware bounds (tight during grasp, loose during transit)
   - Uses DyQ-VLA kinematic proxies (velocity, acceleration)
   - Nobody has done this for VLA safety contracts
2. **Cross-model transfer**: Learn contracts from SmolVLA, apply to OpenVLA unchanged
   - Tests: are safety bounds task-specific or model-specific?
   - Nobody has studied this
3. **Conformal calibration**: CP-calibrated contract bounds with coverage guarantees
   - Replaces hand-tuned percentiles with "99% coverage" formal guarantee
   - SAFE (NeurIPS 2025) does CP for failure prediction, not contract calibration

## Papers to Read
- Two-Steps Diffusion Policy (NeurIPS 2025): arXiv:2510.21991
- One-Step Flow Policy (Mar 2026): arXiv:2603.12480
- Mean-Flow One-Step VLA: arXiv:2603.01469
- SAFE conformal prediction for VLA: vla-safe.github.io
- ICRL Survey (TMLR 2025): arXiv:2409.07569
