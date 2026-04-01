# Formal Safety Contracts for VLA - Research Deep Dive

**Date**: 2026-03-31
**Agent**: Formal Methods Researcher
**Verdict**: GENUINELY NOVEL. Design-by-contract for VLA is an unexplored gap.

---

## The Gap (confirmed by exhaustive search)

| What Exists | What's Missing |
|------------|----------------|
| AEGIS (CBF post-hoc optimization) | **Runtime contract enforcement with formal guarantees** |
| SafeVLA (training-time CMDP) | **Design-by-contract decorator pattern** |
| Safety Chip (LTL temporal logic) | **Action-space contract composition theory** |
| VerSAILLE (dL for NN controllers) | **Contract parameter learning from demos** |
| Constraint inference from demos | **Pareto analysis of contract strictness vs task success** |

## Strongest Paper Angle

**"SafeContract: Formally Verified Design-by-Contract Safety for VLA Policies"**

Four contributions:
1. **C1 - Formalization**: Define contracts with assume-guarantee semantics. Prove clipping correctness.
2. **C2 - Composition Theory**: When do stacked contracts (workspace + velocity + force) interfere? Derive interference-free conditions.
3. **C3 - Contract Learning**: Learn parameters from expert demos (DROID, Bridge V2). 99th percentile with confidence bounds.
4. **C4 - Pareto Analysis**: Sweep contract strictness on LIBERO. Plot safety violations vs task success.

## Why This Survives Review

- Not "just clipping" - composition theory (C2) is non-trivial
- Not "just experiments" - formal proofs + learned parameters + Pareto analysis
- Practical: pip-installable, 3 lines of code, zero runtime overhead
- Positioned as: "verify the contract layer (trivial), not the neural network (intractable)"

## Key Prior Work (not competing, complementary)
- AEGIS: [arXiv:2512.11891](https://arxiv.org/abs/2512.11891) - CBF-QP, harder problem, more overhead
- SafeVLA: [arXiv:2503.03480](https://arxiv.org/abs/2503.03480) - Training-time, we wrap any policy
- Safety Chip: [arXiv:2309.09919](https://arxiv.org/abs/2309.09919) - LTL temporal, we do continuous
- VerSAILLE: [arXiv:2402.10998](https://arxiv.org/abs/2402.10998) - Verifies NN, we sidestep it
- Boolean CBF Composition: [IEEE 2018](https://ieeexplore.ieee.org/document/8511471/) - Continuous-time, we're discrete
- Formal Methods Survey: [arXiv:2602.06971](https://arxiv.org/abs/2602.06971)

## Target Venues
- CoRL 2026 (May 28 deadline - tight)
- NeurIPS 2026 SafeGenAI Workshop (Sep 2026)
- RSS 2026 Safe Robot Learning Workshop
- ICRA 2027 (Sep 2026 deadline)

## Next Steps
1. Write C2 (composition theory) - most publishable standalone
2. Implement learned contract params from DROID dataset
3. Run LIBERO Pareto sweep
4. 4-page workshop paper first, then expand
