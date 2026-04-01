# Training-Free vs D3P: Can Heuristics Match RL?

**Date**: 2026-03-31
**Verdict**: YES - strong theoretical + empirical arguments

## Key Finding

D3P's RL adaptor learns to allocate: 8 steps for grasping, 6 for aligning, 3 for transit.
This EXACTLY correlates with safety signals (object proximity, velocity, gripper state).
The RL spends GPU hours of PPO to discover what physics provides for free.

## Theoretical Arguments (3 convergent)
1. **Gigerenzer's less-is-more**: Low-dim decision space + strong regularities -> heuristics match RL
2. **Near-linear mapping**: D3P's Figure 6 shows monotonic danger->steps relationship
3. **Feature equivalence**: D3P learns to use task phase, which IS defined by safety signals

## D3P Details
- Code NOT available. Must reproduce or use published numbers.
- Architecture: lightweight MLP adaptor (~1/15 of base policy params)
- Training: 3-stage PPO + DPPO on Robomimic
- Result: 2.2x speedup, minimal success drop

## Paper Title
"Is Learned Step Scheduling Necessary? Training-Free Safety Heuristics Match RL for Adaptive Denoising in Robot Policies"

## Sources
- [D3P](https://arxiv.org/abs/2508.06804)
- [Gigerenzer - Homo Heuristicus](https://onlinelibrary.wiley.com/doi/10.1111/j.1756-8765.2008.01006.x)
- [Robomimic](https://robomimic.github.io/)
- [DPPO](https://diffusion-ppo.github.io/)
