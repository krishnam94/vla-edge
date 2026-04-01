# Adaptive Safety Bounds Survey

**Date**: 2026-04-01
**Verdict**: MODERATE-TO-STRONG NOVELTY for phase-adaptive safety contracts

## Closest Competitors
- **CompliantVLA-adaptor** (arXiv:2601.15541): adapts IMPEDANCE by task phase, not hard safety bounds
- **DyQ-VLA** (arXiv:2603.07904): adapts QUANTIZATION by kinematic state - same pattern, different target
- **Adaptive CBFs**: adapt to uncertainty/feasibility, not task semantics
- **CORE/Semantic Safety**: adapt WHICH constraints, not HOW STRICT

## The Gap
Nobody adapts hard safety contract BOUNDS by task phase for VLA policies.
Impedance = soft, about interaction quality. Our bounds = hard, about never-violate.
The philosophy exists (variable impedance, decades old) but the mechanism doesn't.

## Key Sources
- [CompliantVLA-adaptor](https://arxiv.org/abs/2601.15541)
- [DyQ-VLA](https://arxiv.org/abs/2603.07904)
- [Risk-Aware Adaptive Safety Margins](https://www.mdpi.com/2076-0825/15/2/116)
- [Variable Impedance Review](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.590681/full)
