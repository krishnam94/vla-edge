# Final Paper Synthesis for ICRA VLA Pipelines Workshop

**Date**: 2026-04-01
**Based on**: 5 deep research agents + correctness critic + 3 novelty agents

---

## Paper Title (revised)
"SafeContract: Composable, Phase-Adaptive Safety Guarantees for VLA Policies"

## Core Contributions (4, all verified novel)

1. **SafeContract formalization** - design-by-contract decorator for VLA
   - Formally verified: composition = intersection, deadlock unreachable
   - Zero overhead (27us vs 14,000ms inference)
   - Novel: nobody has decorator-pattern safety contracts for VLA

2. **Phase-adaptive bounds** - tighten during dangerous phases, loosen during transit
   - Uses kinematic proxies (velocity, acceleration) from DyQ-VLA
   - Novel: CompliantVLA-adaptor adapts impedance (soft), we adapt bounds (hard)
   - Gap confirmed by adaptive bounds survey

3. **Empirical proof VLAs need safety** - SmolVLA outputs exceed [-1,1] on ALL LIBERO data
   - Ground truth: always within bounds. Model: max 2.34.
   - Nobody has reported this systematically

4. **Comparison vs AEGIS** - our approach is:
   - 27us vs AEGIS QP solver overhead
   - Formally verified vs empirically tested
   - Composable decorators vs monolithic layer
   - No VLM needed (AEGIS uses VLM for obstacle detection)

## Positioning vs All Competitors

| Method | Category | Our Advantage |
|--------|----------|---------------|
| AEGIS/VLSA | Post-hoc CBF+QP | Lighter (27us vs QP), formally verified, composable |
| SafeVLA | Training-time CMDP | No retraining needed, wraps ANY policy |
| SafeDiffuser | Denoising-time CBF | Works for any VLA, not just diffusion |
| CoDiG | Gradient guidance | Training-free, no gradient access needed |
| CompliantVLA | Impedance adaptation | Hard bounds vs soft compliance |

## Experiment Plan (500 forward passes, 2 hours)

| Exp | Passes | What | Key Figure |
|-----|--------|------|-----------|
| A | 100 | Baseline violation profiling | "VLAs output unsafe actions" bar chart |
| B | 100 | SafeContract vs AEGIS-lite vs naive clip | Comparison table |
| C | 100 | Composition verification (stack 2-3 contracts) | Theorem validation |
| D | 100 | Adaptive bounds (strictness sweep by phase) | Pareto curve |
| F | 100 | Ablation (each contract component) | Contribution of each |
| E | 0 | Overhead microbenchmark | 27us number |

## What We Need to Build (3-4 days)

1. **Adaptive contract** - extend @safety_contract with `phase_detector` parameter
2. **AEGIS-lite baseline** - simple CBF-QP reimplementation for comparison
3. **Phase detector** - kinematic-based (velocity magnitude + acceleration)
4. **Corrected experiment runner** - with queue flush, 100+ samples each

## What We Already Have

- @safety_contract decorator (working, tested)
- Composition theorems (proofs written)
- LIBERO data access (13K frames loaded)
- SmolVLA adapter (end-to-end working)
- LaTeX template (draft exists)
- 68 tests passing

## Timeline (13 days to Apr 14)

| Day | What |
|-----|------|
| Apr 1 (today) | Research synthesis complete. Start implementing. |
| Apr 2-3 | Implement adaptive contracts + AEGIS-lite baseline |
| Apr 4-5 | Run all 6 experiments (2 hours compute) |
| Apr 6-7 | Generate figures, tables |
| Apr 8-10 | Write paper (revise LaTeX draft) |
| Apr 11-12 | /review-panel + /novelty-check |
| Apr 13 | Final revisions |
| Apr 14 | Submit to OpenReview |
