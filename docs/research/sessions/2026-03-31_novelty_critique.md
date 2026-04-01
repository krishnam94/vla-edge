# Novelty Critique: Honest Assessment of Our 3 Paper Ideas

**Date**: 2026-03-31
**Method**: Skeptical reviewer simulation with exhaustive prior work search

---

## Verdict: All 3 ideas have thin novelty as-is

| Idea | Novelty | Key Prior Work That Hurts Us |
|------|---------|------------------------------|
| SAAD | THIN | D3P (arXiv:2508.06804) already does adaptive step allocation for robot diffusion policies |
| SplitPipe | NONE | HeteroLLM, TFLite delegates, NVIDIA TensorRT Edge-LLM |
| QuantProbe | NONE | TMPQ-DM (arXiv:2404.09532) did joint quantization + step reduction 2 years ago |

## What's ACTUALLY Novel (from the critique)

1. **Prove D3P's RL-trained adaptor is unnecessary** - Show training-free safety heuristic matches D3P. The contribution: "physics-based safety signals are sufficient, no RL needed."

2. **Formal Safety Contracts for VLA** - Our @safety_contract decorator IS genuinely novel. Nobody has compile-time/decorator-time formal contracts with runtime verification for VLA policies.

3. **Chaos Engineering for VLA** - Systematic fault injection to characterize VLA robustness. Nobody has done this.

## Key Prior Work We Missed
- D3P (Dynamic Denoising Diffusion Policy) - arXiv:2508.06804
- TMPQ-DM - arXiv:2404.09532
- HeteroLLM - arXiv:2501.14794
- SafeDiffuser - ICLR 2025
- CoDiG - CoRL 2025

## Conference Deadlines
- CoRL 2026: May 28 (tight)
- NeurIPS 2026: May 6
- NeurIPS Workshops: Sep 2026
- ICRA 2027: Sep 2026

## Source
Full analysis with all citations in novelty_critique agent output.
