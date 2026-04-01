# VLA Safety Competitor Deep Dive

**Date**: 2026-03-29
**Focus**: AEGIS/VLSA, SafeVLA, SafeDiffuser, CoDiG, ATACOM
**Purpose**: Know exactly what competitors do so we can position SafeContract clearly.

---

## 1. AEGIS/VLSA (Closest Competitor)

**Paper**: [arXiv:2512.11891](https://arxiv.org/abs/2512.11891) - "VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer"
**Code**: [github.com/THU-RCSCT/vlsa-aegis](https://github.com/THU-RCSCT/vlsa-aegis)
**Project Page**: [vlsa-aegis.github.io](https://vlsa-aegis.github.io/)
**Benchmark**: [SafeLIBERO on HuggingFace](https://huggingface.co/datasets/THURCSCT/SafeLIBERO)
**Venue**: arXiv Dec 2025 (appears under review)

### What EXACTLY Is Their Method?

AEGIS is a **plug-and-play safety constraint layer** that sits after a VLA model's action output. Two modules:

1. **Vision-Language Safety Assessment Module**: Uses a VLM (ZhipuAI) + GroundingDINO (open-set detection) + depth information to identify obstacles in the scene. Translates semantic risks ("avoid the moka pot") into physical avoidance constraints.

2. **Action-Driven Safety-Guaranteed Control Module**: CBF-based QP solver that minimally adjusts VLA actions to avoid collisions.

### Safety Constraints Enforced

- **Collision avoidance only**. The entire method is about avoiding obstacles.
- Models both end-effector and obstacles as **ellipsoids** via Minimum Volume Enclosing Ellipsoid (MVEE) fitting.
- Barrier function h(x) is derived from **ellipsoid-to-ellipsoid signed distance**.
- CBF constraint: dh/dt >= -alpha(h(x)), with alpha(h) = 10h (extended class-K-infinity function).

### How (CBF? QP? Clipping?)

- **CBF + QP**. The QP minimizes ||u - u_vla||^2 subject to the CBF constraint.
- Single linear constraint per obstacle -> convex QP.
- Ellipsoids fitted via MVEE optimization: minimize -log det(Q) with SDP constraints.
- **NOT clipping**. This is proper optimization-based safety.

### Runtime Overhead

- **0.356 ms average** for the safety constraint layer.
- ~1.86% of total inference latency.
- ~1/47th of VLA inference time.
- System maintains **20 Hz** control frequency.
- Verdict: Very fast. The QP is small (single constraint) and solves in sub-millisecond time.

### Benchmarks and Metrics

**SafeLIBERO benchmark** (their contribution):
- 4 suites (Spatial, Goal, Object, Long) x 4 tasks each = 16 tasks
- 2 safety levels per task = 32 scenarios
- 50 episodes per scenario = 1,600 total episodes
- Level I: obstacle near target object
- Level II: obstacle obstructs movement path

**Metrics**:
- Collision Avoidance Rate (CAR) - % collision-free trajectories
- Task Success Rate (TSR) - % successful task completions
- Execution Time Steps (ETS) - efficiency

**Results**:
| Model | CAR | TSR |
|-------|-----|-----|
| pi0.5 (no safety) | 18.69% | 50.88% |
| OpenVLA-OFT (no safety) | 15.13% | 22.81% |
| pi0.5 + AEGIS | **77.85%** | **68.13%** |

AEGIS improves CAR by ~4x and TSR by ~17%.

### VLA Models Tested
- **pi0.5** (primary, flow-matching)
- **OpenVLA-OFT** (transformer + online fine-tuning, baseline comparison)

### Limitations (Stated)
1. VLM misidentification of critical obstacles
2. Inaccurate localization by GroundingDINO
3. Aggressive point cloud filtering causes geometric under-estimation
4. Collisions with unmodeled kinematic components (penultimate arm link)
5. Safety-induced **distribution shift** - robot reaches OOD states (high altitude, extreme offsets) unfamiliar to training data, causing erratic behavior
6. Only translational motion - orientation locked. No 6-DoF.
7. No formal ablation studies

### Limitations (Unstated)
1. **Requires VLM inference per step** (or at some frequency) for obstacle detection - adds to total system latency beyond the 0.356ms QP solve
2. **Only handles collision avoidance** - no joint limits, velocity limits, acceleration limits, workspace bounds, force limits
3. **Requires depth sensor** - can't work with RGB-only setups
4. **Requires GroundingDINO** - adds significant model complexity (GroundingDINO is ~600M params)
5. **No formal composition theory** - what happens when you add multiple safety constraints?
6. **Two separate Python environments** (3.8 + 3.11) - painful deployment
7. **ZhipuAI API dependency** - cloud API call in the safety loop (latency variable, availability risk)
8. **Only 2 VLA models tested** - no evidence of generality beyond pi0.5 and OpenVLA-OFT

### Code Complexity
- **High**. Requires conda, two Python environments, CUDA 11.3, manual checkpoint downloads.
- Depends on ZhipuAI cloud API, GroundingDINO, depth processing pipeline.
- 111 commits, mix of Python (65.7%) and Jupyter notebooks (33.5%).
- Estimated setup time: 30+ minutes for experienced users.
- MIT license.

---

## 2. SafeVLA (Training-Time Safety)

**Paper**: [arXiv:2503.03480](https://arxiv.org/abs/2503.03480)
**Code**: [github.com/PKU-Alignment/SafeVLA](https://github.com/PKU-Alignment/SafeVLA)
**Venue**: NeurIPS 2025 Spotlight
**Benchmark**: Safety-CHORES (their contribution)

### Method
- **CMDP formulation**: Maximize reward subject to safety cost constraints.
- **Lagrangian relaxation**: min_theta max_lambda [-J_r(theta) + sum(lambda_i * J_ci(theta))]
- **ISA (Integrated Safety Approach)**: Model safety requirements -> elicit unsafe behaviors -> constrain via SafeRL -> evaluate.
- Five safety predicates with binary costs (1 if violated, 0 otherwise).
- Iterative policy + Lagrange multiplier updates.

### Key Distinction
- **Training-time** safety alignment. Requires retraining the VLA model.
- Not plug-and-play. You need to fine-tune with their safety objective.

### Results
- 83.58% reduction in cumulative safety cost vs. strongest RL baseline (FLaRe).
- +3.85% task success rate improvement.
- Extreme failure safety: cost of 2.20 vs FLaRe's 71.68.

### Models Tested
- SPOC (DINOv2 and SigLip variants)
- Compared against EmbCLIP, Embodied-Codebook, Poliformer

### Runtime Overhead
- Training: 15-25M steps on 8x H100 GPUs.
- Inference: No explicit overhead stated (safety is baked into weights).

### Limitations
- **Requires retraining** for each new safety specification.
- **Training cost**: 8x H100 GPUs is expensive.
- **Binary cost scheme** - severity-weighted costs deferred to future work.
- **Sim-to-real gap** acknowledged.
- Does not compare to AEGIS/VLSA.

---

## 3. SafeDiffuser (ICLR 2025)

**Paper**: [arXiv:2306.00148](https://arxiv.org/abs/2306.00148)
**Code**: [github.com/Weixy21/SafeDiffuser](https://github.com/Weixy21/SafeDiffuser)
**Venue**: ICLR 2025

### Method
- Embeds **finite-time diffusion invariance** into the denoising procedure.
- Three variants: RoS (convex unsafe sets), TVS (nonlinear class K functions), ReS (otherwise).
- Maintains generative performance while providing safety guarantees.
- Solves QP at each denoising step.

### Runtime Overhead
- QP complexity: O(q^3) where q = decision variable dimension.
- Per-step cost: 0.106-0.107s (maze), 0.170-0.183s (locomotion).
- vs baseline Diffuser: 0.007-0.038s per step.
- **3-15x slower** than unconstrained diffusion.
- Optimization: apply CBF to limited diffusion steps, batch QP solving.

### Benchmarks
- Maze path generation
- Legged robot locomotion
- 3D space manipulation

### Key Distinction
- **Diffusion-specific**. Modifies the denoising process itself.
- Not applicable to autoregressive VLAs (OpenVLA).
- Heavy computational overhead.
- Not designed for VLA specifically.

### Limitations
- Significant runtime cost (3-15x overhead).
- Only works with diffusion-based policies.
- Requires unsafe set geometry to be known.
- Not tested on VLA models.

---

## 4. CoDiG (CoRL 2025)

**Paper**: [arXiv:2505.13131](https://arxiv.org/abs/2505.13131)
**Venue**: CoRL 2025

### Method
- Integrates barrier function gradients into the reverse SDE during denoising.
- Time-varying weight gamma_t (starts at zero, increases) for constraint influence.
- No projections, auxiliary models, or simulators needed.
- Warm-start acceleration: reuses previous trajectories with minimal noise.

### Runtime
- **2.5 Hz** on RTX 4090 with warm-start.
- Standard diffusion: ~0.25 Hz (1000 steps).
- ~10x speedup from warm-start strategy.

### Benchmarks
- Miniature autonomous racing (real-world).
- 5 trials, 15 laps each, 10 obstacle configs.
- 100% obstacle avoidance success.

### Key Distinction
- General-purpose diffusion guidance, not VLA-specific.
- Real-world validation (racing, not manipulation).
- Data-efficient (80 training trajectories after augmentation).

### Limitations
- Only tested on autonomous racing.
- Warm-started trajectories have "coarser structure" and more conservative behavior.
- 2.5 Hz is too slow for most manipulation tasks.
- Not tested on VLA models.

---

## 5. ATACOM (Safe Robot Foundation Models)

**Paper**: [arXiv:2505.10219](https://arxiv.org/abs/2505.10219) - "Towards Safe Robot Foundation Models Using Inductive Biases"
**Project Page**: sites.google.com/view/safe-robot-foundation-models
**Venue**: Appears under review (May 2025)

### Method
- Modular safety layer placed **after** a foundation model policy.
- Constrains actions to the **tangent space** of the constraint manifold.
- Safe action = drift compensation + error correction + basis transformation of the model's action.
- "Morphs" the action to avoid constraint violations while preserving semantics.

### Constraints Enforced
- Joint limits
- Workspace constraints
- Self-collision (geometric volume approximations)
- Collision avoidance (signed distance fields)
- Visual constraints (from segmentation + depth)

### Runtime
- Foundation model runs at ~15 Hz.
- Safety layer runs at ~60 Hz (higher frequency, decoupled).
- Execution time: ~2 seconds additional overhead per task.

### Models Tested
- **pi0** (pick-and-place tasks)
- **Octo** (air hockey)

### Results
- 100% safety rate with ATACOM vs frequent violations without.
- Task success comparable to vanilla policies (slight decrease in harder tasks).
- Real-world deployment validated.

### Key Distinction
- **Most similar to our approach** in philosophy (post-hoc, model-agnostic).
- But uses differential geometry (tangent space), not design-by-contract.
- More sophisticated constraint handling (signed distance fields).
- Real robot validation.

### Limitations
- More complex implementation than simple clipping.
- Requires robot kinematics model.
- May restrict policy expressiveness on hard tasks.
- No formal composition theory.

---

## Comparison Table

| Feature | **SafeContract (Ours)** | **AEGIS/VLSA** | **SafeVLA** | **SafeDiffuser** | **CoDiG** | **ATACOM** |
|---------|------------------------|----------------|-------------|-----------------|-----------|-----------|
| **Paper** | In progress | [2512.11891](https://arxiv.org/abs/2512.11891) | [2503.03480](https://arxiv.org/abs/2503.03480) | [2306.00148](https://arxiv.org/abs/2306.00148) | [2505.13131](https://arxiv.org/abs/2505.13131) | [2505.10219](https://arxiv.org/abs/2505.10219) |
| **Venue** | CoRL 2026 target | Under review | NeurIPS 2025 Spotlight | ICLR 2025 | CoRL 2025 | Under review |
| **Code** | [vla-edge](https://github.com/) | [vlsa-aegis](https://github.com/THU-RCSCT/vlsa-aegis) | [SafeVLA](https://github.com/PKU-Alignment/SafeVLA) | [SafeDiffuser](https://github.com/Weixy21/SafeDiffuser) | Not found | Not found |
| **Approach** | Design-by-contract decorator | CBF-QP + VLM obstacle detection | CMDP training-time alignment | CBF in diffusion denoising | Barrier gradients in denoising | Tangent space projection |
| **When Applied** | Inference-time (post-hoc) | Inference-time (post-hoc) | Training-time | Inference-time (in-diffusion) | Inference-time (in-diffusion) | Inference-time (post-hoc) |
| **Requires Retraining** | No | No | **Yes** | No | No | No |
| **Model-Agnostic** | Yes (any VLA) | Yes (any VLA) | No (must retrain) | No (diffusion only) | No (diffusion only) | Yes (any policy) |
| **Runtime Overhead** | **<50 us** | 0.356 ms | None (baked in) | 100-180 ms/step | 400 ms/step | ~2s/task |
| **Constraints** | Bounds, velocity, acceleration, workspace, EE speed | Collision avoidance only | Learned from CMDP | Obstacle avoidance | Obstacle avoidance | Joints, workspace, self-collision, obstacles |
| **Formal Guarantees** | Assume-guarantee contracts, composition proofs | CBF forward invariance | CMDP convergence | Diffusion invariance | Barrier function | Tangent space feasibility |
| **Composition Theory** | Yes (Theorems 2-3) | No | N/A | No | No | No |
| **Requires Depth Sensor** | No | **Yes** | No | No | No | Depends |
| **Requires VLM** | No | **Yes** (ZhipuAI API) | No | No | No | No |
| **Requires Kinematics** | No | Partial (ellipsoid fitting) | No | No | No | **Yes** |
| **VLA Models Tested** | OpenVLA, SmolVLA (planned: pi0) | pi0.5, OpenVLA-OFT | SPOC variants | None (not VLA) | None (not VLA) | pi0, Octo |
| **Benchmark** | LIBERO-Long (planned) | SafeLIBERO (32 scenarios) | Safety-CHORES | Maze, locomotion | Racing | Pick-place, air hockey |
| **Setup Complexity** | pip install | 2 conda envs, API keys, checkpoints | 8x H100, retraining | Moderate | Moderate | Requires URDF |
| **License** | Apache 2.0 | MIT | Apache 2.0 | MIT | Unknown | Unknown |

---

## What WE Do That THEY Don't

1. **Contract composition theory with formal proofs.** Nobody else has Theorems 2-3 (composition preserves guarantees IFF order-independent, velocity + bounds requires re-application). AEGIS has no composition theory at all. SafeVLA doesn't need it (single training objective). ATACOM has no formal composition story.

2. **Contract parameter learning from demonstrations.** Learn tight-but-safe bounds from DROID/Bridge V2 via percentile estimation. AEGIS uses manual obstacle detection. SafeVLA learns from reward/cost. Nobody learns contract parameters from expert data.

3. **Pareto analysis of safety vs. performance.** Explicit sweep of contract strictness. Nobody else characterizes the tradeoff curve.

4. **Zero sensor requirements.** No depth sensor, no VLM API, no kinematics model. Pure action-space constraints. AEGIS needs depth + VLM. ATACOM needs URDF.

5. **Truly zero overhead.** <50 us vs AEGIS's 0.356 ms (7x faster) vs SafeDiffuser's 100+ ms (2000x faster). Simple numpy clipping, no QP solver.

6. **One-line integration.** `@safety_contract(action_range=[-1,1])` - decorator on any predict(). AEGIS needs two conda environments and a cloud API key.

7. **Velocity and acceleration limits.** AEGIS only does collision avoidance. We enforce joint velocity, acceleration, workspace, and EE speed constraints.

8. **pip-installable.** `pip install vla-edge`. No CUDA required for safety layer. Runs on CPU, Mac, Jetson.

---

## What THEY Do That WE Don't

1. **AEGIS: Scene-aware obstacle detection.** AEGIS identifies NEW obstacles at runtime using VLM + GroundingDINO. Our contracts are pre-specified, not adaptive to novel obstacles. This is a fundamental capability gap.

2. **AEGIS: CBF with formal forward invariance.** CBF provides stronger guarantees than box-constraint clipping for collision avoidance. Our Theorem 1 only covers box constraints.

3. **SafeVLA: End-to-end safety alignment.** The model itself learns to be safe, not just clipped. Better task completion under constraints because the policy adapts, rather than being clipped post-hoc.

4. **SafeVLA: Long-tail risk mitigation.** Explicitly handles extreme failure scenarios (cost 2.20 vs 71.68).

5. **SafeDiffuser: Safety during generation.** Modifies the generative process itself, not just the output. Theoretically cleaner for diffusion-based policies.

6. **ATACOM: Tangent space projection.** More principled than clipping - preserves action semantics better. Handles signed distance fields natively.

7. **ATACOM: Real robot validation.** Both pi0 and Octo on physical hardware with 100% safety rate.

8. **Nobody: We don't have real robot experiments yet.** This is our biggest weakness.

---

## "Why Not Just Use AEGIS?" - The Reviewer Answer

**Short answer**: AEGIS solves a different (harder, narrower) problem at higher cost. SafeContract solves a simpler (broader) problem at near-zero cost.

**Full argument**:

1. **Different problem scope.** AEGIS is specifically about collision avoidance with detected obstacles. SafeContract enforces a broader class of constraints: joint limits, velocity bounds, acceleration limits, workspace bounds, EE speed. These are the constraints that matter most for preventing hardware damage on real robots. A robot can damage itself by exceeding joint velocity limits even in an empty workspace - AEGIS won't catch that.

2. **System complexity.** AEGIS requires a VLM (cloud API), GroundingDINO (600M params), depth sensor, point cloud processing, MVEE fitting, and a QP solver. SafeContract requires numpy. For edge deployment on Jetson Orin Nano (8GB RAM), AEGIS is infeasible - the overhead models alone exceed the memory budget. SafeContract runs in <50 microseconds on CPU.

3. **Complementary, not competing.** AEGIS handles dynamic obstacle avoidance. SafeContract handles static physical constraints. A production system would use both: SafeContract as the inner safety layer (always-on, near-zero cost), AEGIS-style obstacle avoidance as the outer layer (when perception is available). Our composition theory (Theorem 2) provides the formal framework for stacking these layers correctly.

4. **Formal composition.** AEGIS provides no theory for composing multiple safety constraints. What happens when you add joint limits AND obstacle avoidance AND velocity bounds? Our Theorems 2-3 prove when composition is safe and when naive stacking silently breaks guarantees. This matters for production deployment.

5. **Deployment practicality.** SafeContract is pip-installable, works on any hardware (CPU/GPU/Mac/Jetson), requires no sensors, no API keys, no model downloads. AEGIS requires CUDA 11.3, two conda environments, GroundingDINO checkpoint, and a ZhipuAI API key. For the researcher who wants to add safety to their VLA in 5 minutes, SafeContract is the answer.

6. **AEGIS's own failure modes validate our approach.** AEGIS reports safety-induced distribution shift - the robot reaches OOD states after avoidance. Our Pareto analysis explicitly characterizes this tradeoff. AEGIS reports collisions with unmodeled kinematic components (penultimate arm link). Our workspace bounds would catch this. AEGIS reports failures from VLM misidentification. Our constraints don't depend on perception.

**One-line positioning**: "AEGIS is a perception-driven collision avoidance system; SafeContract is a formally verified action-space constraint layer. Use both."

---

## Other Papers We Should Know About

| Paper | arXiv | Key Idea | Relevance |
|-------|-------|----------|-----------|
| **ATACOM** (Safe Robot Foundation Models) | [2505.10219](https://arxiv.org/abs/2505.10219) | Tangent space projection after foundation model | HIGH - same philosophy, more sophisticated math, real robot results. Must cite and differentiate. |
| **Towards Safe Robot Foundation Models** | [2503.07404](https://arxiv.org/abs/2503.07404) | Null-space control for safe action space | MEDIUM - similar idea, different mechanism |
| **Safety Chip** | [2309.09919](https://arxiv.org/abs/2309.09919) | LTL temporal logic for safety | MEDIUM - cite in related work |
| **VerSAILLE** | [2402.10998](https://arxiv.org/abs/2402.10998) | Formal verification of NN controllers | MEDIUM - cite as "verify the network" approach |
| **Formal Methods Survey** | [2602.06971](https://arxiv.org/abs/2602.06971) | Survey of formal methods for safe learning | LOW - cite for context |

---

## Strategic Implications for SafeContract Paper

### Strengths to Emphasize
1. Composition theory (unique contribution - nobody else has this)
2. Contract parameter learning from data (DROID/Bridge V2)
3. Pareto analysis (quantifies the safety-performance tradeoff)
4. Zero overhead, zero requirements, pip-installable
5. Broader constraint types than AEGIS

### Weaknesses to Address
1. **No real robot experiments.** Mitigate with thorough simulation (LIBERO-Long, 7500 episodes).
2. **"Just clipping" criticism.** Counter with composition theory (Theorem 3 counterexample), parameter learning, Pareto analysis.
3. **No dynamic obstacle avoidance.** Acknowledge explicitly. Position as complementary to AEGIS.
4. **ATACOM comparison.** ATACOM is the most philosophically similar. Differentiate on: (a) simpler implementation, (b) formal composition, (c) parameter learning, (d) broader VLA evaluation.

### Must-Cite Papers
- AEGIS/VLSA [2512.11891] - primary comparison
- SafeVLA [2503.03480] - training-time baseline
- SafeDiffuser [2306.00148] - diffusion-specific baseline
- ATACOM [2505.10219] - most similar philosophy
- CBF Composition (Glotfelter 2017) - theoretical inspiration
- VerSAILLE [2402.10998] - formal verification alternative

### Key Risk
ATACOM is very close to our positioning (post-hoc, model-agnostic safety layer). If ATACOM publishes before us at a top venue, our novelty claim weakens. Differentiation must be crystal clear: we offer **formal composition theory + contract learning + VLA-specific evaluation**, while ATACOM offers **differential geometry + kinematics-aware + real robot demos**.
