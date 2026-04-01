# VLA Safety and Robustness - Comprehensive Literature Survey

**Created**: 2026-03-29
**Purpose**: Position SafeContract paper against ALL relevant prior work
**Scope**: 2023-2026, VLA/embodied AI safety, robustness, adversarial attacks, formal methods

---

## Master Table

### Category A: VLA Safety Alignment (Training-Time)

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| A1 | **SafeVLA: Towards Safety Alignment of VLA Model via Constrained Learning** | NeurIPS 2025 Spotlight | Mar 2025 | [2503.03480](https://arxiv.org/abs/2503.03480) | CMDP-based safety alignment that elicits unsafe behaviors and constrains VLA via safe RL; reduces violation cost by 83.58% | Training-time (CMDP + safe RL) | Minimizes cumulative safety cost; reduces long-tail risk to 1/35 of baseline | Yes - [GitHub](https://github.com/PKU-Alignment/SafeVLA) |
| A2 | **RobustVLA: Robustness-Aware RL Post-Training for VLA Models** | ICLR 2026 | Oct 2025 | [2510.00037](https://arxiv.org/abs/2510.00037) | Multi-modal robustness via Jacobian + smoothness regularization; 12.6% gain on pi0, 10.4% on OpenVLA across 17 perturbations | Training-time (post-training RL) | Robustness to multi-modal perturbations (action, vision, language, environment) | Likely (ICLR) |
| A3 | **Narrow Fine-Tuning Erodes Safety Alignment in Vision-Language Agents** | Preprint | Feb 2026 | [2602.16931](https://arxiv.org/abs/2602.16931) | Shows LoRA fine-tuning on narrow domains causes 70.71% misalignment at r=128; even 10% harmful data degrades alignment | Analysis / Red-teaming | Identifies safety erosion during fine-tuning; no enforcement | No |

### Category B: VLA Safety Enforcement (Inference-Time / Post-Hoc)

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| B1 | **VLSA/AEGIS: VLA Models with Plug-and-Play Safety Constraint Layer** | Preprint | Dec 2025 | [2512.11891](https://arxiv.org/abs/2512.11891) | CBF-based QP safety layer using VLM scene reasoning + depth for obstacle avoidance; 59.16% improvement in collision avoidance | Inference-time (CBF-QP) | Obstacle avoidance with theoretical CBF guarantees | Yes - [Website](https://vlsa-aegis.github.io/) |
| B2 | **Safety Chip: Enforcing Constraints for LLM-driven Robot Agents** | ICRA 2024 | Sep 2023 | [2309.09919](https://arxiv.org/abs/2309.09919) | LTL-based safety constraint module that translates NL to temporal logic, monitors agent decisions, and prunes unsafe actions; 100% safety rate with expert-verified LTL | Inference-time (LTL monitoring + re-planning) | Temporal safety constraints (sequencing, forbidden actions) | Yes - [Website](https://yzylmc.github.io/safety-chip/) |
| B3 | **SELP: Safe and Efficient Task Plans for Robot Agents with LLMs** | ICRA 2025 (Best Paper Finalist) | Sep 2024 | [2409.19471](https://arxiv.org/abs/2409.19471) | Equivalence voting + LTL constrained decoding + domain fine-tuning for safe LLM planning; 20.4% safety improvement in manipulation | Inference-time (constrained decoding via LTL) | Plan-level safety (conformance to NL constraints) | Yes - [GitHub](https://github.com/lt-asset/selp) |
| B4 | **RoboGuard: Safety Guardrails for LLM-Enabled Robots** | Preprint | Mar 2025 | [2503.07885](https://arxiv.org/abs/2503.07885) | Two-stage guardrail: root-of-trust LLM grounds safety rules via CoT, then temporal logic synthesis resolves conflicts; reduces unsafe plans from 92% to <2.5% | Inference-time (LLM grounding + temporal logic synthesis) | Decision-level safety (rejects jailbreaks and unsafe plans) | Yes - [GitHub](https://github.com/KumarRobotics/RoboGuard) |
| B5 | **SafeEmbodAI: Safety Framework for Mobile Robots in Embodied AI** | Preprint | Sep 2024 | [2409.01630](https://arxiv.org/abs/2409.01630) | Secure prompting + state management + safety validation for LLM-driven mobile robots; 267% improvement in attack scenarios | Inference-time (prompt filtering + state validation) | Navigation safety, malicious command injection defense | No |
| B6 | **SafeMindAgent: Modular Planner-Executor with Cascaded Safety Modules** | Preprint | Sep 2025 | [2509.25885](https://arxiv.org/abs/2509.25885) | Three cascaded safety modules (factual, causal, temporal) integrated into planner-executor; improves safety rate while maintaining task completion | Inference-time (cascaded safety reasoning) | Multi-stage safety (task understanding, perception, planning, execution) | Partial |

### Category C: Diffusion / Flow Matching Policy Safety

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| C1 | **SafeDiffuser: Safe Planning with Diffusion Probabilistic Models** | ICLR 2025 | Jun 2023 | [2306.00148](https://arxiv.org/abs/2306.00148) | Embeds finite-time diffusion invariance + CBF into denoising; safe trajectories for maze, locomotion, manipulation | Training-time (modified denoising process) | CBF-based safety during trajectory generation | Yes - [GitHub](https://github.com/Weixy21/SafeDiffuser) |
| C2 | **CoDiG: Constraint-Aware Diffusion Guidance for Robotics** | CoRL 2025 | May 2025 | [2505.13131](https://arxiv.org/abs/2505.13131) | Integrates barrier functions into reverse diffusion + warm-start strategy; real-time constraint-satisfying outputs in milliseconds | Inference-time (guided diffusion via barrier functions) | Obstacle avoidance, dynamic feasibility | Yes |
| C3 | **PACS: Path-Consistent Safety Filtering for Diffusion Policies** | Preprint | Nov 2025 | [2511.06385](https://arxiv.org/abs/2511.06385) | Performs path-consistent braking on diffusion policy trajectories + set-based reachability verification; outperforms CBFs by 68% in task success | Inference-time (reachability-based safety filter) | Formal safety via set-based reachability analysis | No |
| C4 | **SafeFlow: Safe Robot Motion Planning with Flow Matching via CBFs** | Preprint | Apr 2025 | [2504.08661](https://arxiv.org/abs/2504.08661) | Flow Matching Barrier Functions (FMBF) for training-free safety enforcement on flow matching policies; works on 7-DoF manipulation | Inference-time (flow matching + CBF) | Trajectory-level safety across planning horizon | No |
| C5 | **SafeFlowMatcher: Safe and Fast Planning with Flow Matching + CBFs** | Preprint | Sep 2025 | [2509.24243](https://arxiv.org/abs/2509.24243) | Two-phase prediction-correction: generate candidate path, then refine with CBF-QP; fast and certified safe | Inference-time (prediction-correction with CBF-QP) | Certified safety via CBF | No |
| C6 | **UniConFlow: Unified Constrained Flow-Matching for Certified Motion Planning** | Preprint | Jun 2025 | [2506.02955](https://arxiv.org/abs/2506.02955) | Prescribed-time zeroing function for training-free constraint satisfaction in flow matching; handles equality + inequality constraints | Inference-time (constrained flow matching) | Certified safety, kinodynamic consistency, action feasibility | No |

### Category D: VLA Robustness Testing and Benchmarks

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| D1 | **LIBERO-Plus: In-depth Robustness Analysis of VLA Models** | Preprint | Oct 2025 | [2510.13626](https://arxiv.org/abs/2510.13626) | 7 robustness dimensions, 21 sub-dimensions, 5 difficulty levels; performance drops from 95% to <30% under modest perturbations | Testing (benchmark) | Exposes brittleness to camera, initial state, object layout changes | Yes - [GitHub](https://github.com/sylvestf/LIBERO-plus) |
| D2 | **LIBERO-PRO: Robust and Fair Evaluation of VLA Models Beyond Memorization** | Preprint | Oct 2025 | [2510.03827](https://arxiv.org/abs/2510.03827) | Shows VLA performance collapses from >90% to 0.0% under perturbations; exposes pure memorization | Testing (benchmark) | Reveals memorization vs. generalization gap | Yes - [GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO) |
| D3 | **LIBERO-X: Robustness Litmus for VLA Models** | Preprint | Feb 2026 | [2602.06556](https://arxiv.org/abs/2602.06556) | Hierarchical evaluation protocol with progressive difficulty; spatial, object, semantic perturbations | Testing (benchmark) | Multi-dimensional robustness evaluation | Likely |
| D4 | **Eva-VLA: Evaluating VLA Robustness Under Real-World Physical Variations** | Preprint | Sep 2025 | [2509.18953](https://arxiv.org/abs/2509.18953) | Continuous optimization-based worst-case discovery; >60% failure rate across all variation categories, up to 97.8% in long-horizon | Testing (adversarial evaluation framework) | Physical robustness (3D transforms, illumination, patches) | No |
| D5 | **VLA-Arena: Open-Source Framework for Benchmarking VLA Models** | Preprint | Dec 2025 | [2512.22539](https://arxiv.org/abs/2512.22539) | 170 tasks, 11 suites, hierarchical difficulty; includes safety, distractor, extrapolation dimensions | Testing (benchmark) | Safety suite + robustness evaluation | Yes - [GitHub](https://github.com/PKU-Alignment/VLA-Arena) |
| D6 | **SafeLIBERO: Safety-Critical Benchmark** (part of VLSA/AEGIS) | (with VLSA) | Dec 2025 | [Website](https://vlsa-aegis.github.io/benchmark.html) | 32 scenarios, 1600 episodes, two safety levels per task for obstacle avoidance evaluation | Testing (benchmark) | Collision avoidance in manipulation | Yes |

### Category E: VLA Adversarial Attacks

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| E1 | **Exploring Adversarial Vulnerabilities of VLA Models in Robotics** | Preprint | Nov 2024 | [2411.13587](https://arxiv.org/abs/2411.13587) | Adversarial patch attacks cause up to 100% task failure in simulation; works in physical environments too | Attack (adversarial patches) | Exposes visual vulnerability | Yes - [Website](https://vlaattacker.github.io/) |
| E2 | **Model-agnostic Adversarial Attack and Defense for VLA Models** | Preprint | Oct 2025 | [2510.13237](https://arxiv.org/abs/2510.13237) | EDPA: model-agnostic patch attack + adversarial fine-tuning defense for visual encoder | Attack + Defense | Visual robustness via adversarial training | No |
| E3 | **Adversarial Attacks on Robotic VLA Models** (Jailbreaking) | Preprint | Jun 2025 | [2506.03350](https://arxiv.org/abs/2506.03350) | Adapts LLM jailbreaking to VLAs; textual attacks enable full action space reachability | Attack (textual jailbreaking) | Shows language-channel vulnerability | No |
| E4 | **AttackVLA: Benchmarking Adversarial and Backdoor Attacks on VLA Models** | Preprint | Nov 2025 | [2511.12149](https://arxiv.org/abs/2511.12149) | BackdoorVLA achieves 58.4% targeted attack success (100% on select tasks); unified attack framework across VLA lifecycle | Attack (adversarial + backdoor) | Comprehensive attack taxonomy | No |
| E5 | **VLA-Fool: When Alignment Fails - Multimodal Adversarial Attacks on VLA** | Preprint | Dec 2025 | [2511.16203](https://arxiv.org/abs/2511.16203) | Unifies text, visual, and cross-modal misalignment attacks; minor perturbations cause significant behavioral deviations on LIBERO | Attack (multimodal) | Cross-modal alignment vulnerability | No |
| E6 | **ANNIE: Be Careful of Your Robots** | Preprint | Sep 2025 | [2509.03383](https://arxiv.org/abs/2509.03383) | ISO-grounded safety violation taxonomy (critical/dangerous/risky); ANNIEBench with 2400 sequences; >50% attack success across categories | Attack (task-aware, ISO-grounded) | ISO human-robot interaction safety standards | No |
| E7 | **AdvEDM: Fine-grained Adversarial Attack against VLM-based Embodied Agents** | NeurIPS 2025 | Sep 2025 | [2509.16645](https://arxiv.org/abs/2509.16645) | Fine-grained object-level semantic manipulation; 70%+ attack success in driving, 64% in manipulation | Attack (fine-grained visual) | Perception integrity | Yes - [Website](https://advedm.github.io/demo/) |
| E8 | **Sensor Attacks on VLA Robustness** | LAMPS 2025 (ACM Workshop) | 2025 | [ACM](https://dl.acm.org/doi/10.1145/3733800.3763262) | Physical sensor attacks (sound, light, EM) against VLA models; end-to-end evaluation | Attack (physical sensor) | Physical-world sensor integrity | No |

### Category F: Embodied Agent Safety Benchmarks

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| F1 | **SafeAgentBench: Safe Task Planning of Embodied LLM Agents** | Preprint | Dec 2024 | [2412.13178](https://arxiv.org/abs/2412.13178) | 750 tasks, 10 hazard types, 3 task types; best baseline achieves only 10% hazard rejection rate | Testing (benchmark) | Task-level safety awareness | Yes - [GitHub](https://github.com/shengyin1224/SafeAgentBench) |
| F2 | **AGENTSAFE: Benchmarking Safety of Embodied Agents on Hazardous Instructions** | ICML 2025 | Jun 2025 | [2506.14697](https://arxiv.org/abs/2506.14697) | Asimov's Laws-inspired benchmark; 45 scenarios, 1350 hazardous tasks; systematic failures in all 9 VLMs tested | Testing (benchmark) | Perception-planning-execution safety | Yes |
| F3 | **BeSafe-Bench: Behavioral Safety Risks of Situated Agents** | Preprint | Jan 2026 | [2603.25747](https://arxiv.org/abs/2603.25747) | Covers Web, Mobile, Embodied VLM, and VLA domains; 9 risk categories; best agent <40% safe task completion | Testing (benchmark) | Behavioral safety across digital + physical domains | Yes |
| F4 | **IS-Bench: Interactive Safety of VLM-Driven Embodied Agents** | AAAI 2026 | Jun 2025 | [2506.16402](https://arxiv.org/abs/2506.16402) | 161 scenarios, 388 safety risks; process-oriented evaluation of risk mitigation ordering; <40% safe task completion | Testing (benchmark) | Interactive safety (risk perception + ordered mitigation) | Yes - [GitHub](https://github.com/AI45Lab/IS-Bench) |
| F5 | **HomeSafeBench: Free-Exploration Home Safety Inspection** | Preprint | Sep 2025 | [2509.23690](https://arxiv.org/abs/2509.23690) | 12,900 data points, 5 hazard types; best model achieves only 10.23% F1 on safety inspection | Testing (benchmark) | Environmental hazard detection | Upcoming |
| F6 | **SafeMind: Benchmarking Safety Risks in Embodied LLM Agents** | Preprint | Sep 2025 | [2509.25885](https://arxiv.org/abs/2509.25885) | 5,558 samples, 4 task categories; identifies failures across task understanding, perception, planning, execution | Testing + Defense (benchmark + SafeMindAgent) | Factual, causal, temporal safety constraints | Partial |

### Category G: Formal Methods and Runtime Verification

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| G1 | **VerSAILLE: Provably Safe NN Controllers via Differential Dynamic Logic** | NeurIPS 2024 | Feb 2024 | [2402.10998](https://arxiv.org/abs/2402.10998) | Combines NN verification tools with differential dynamic logic (dL) for infinite-time safety proofs | Post-hoc verification | Infinite-time safety for NN controllers | Yes - [GitHub](https://github.com/samysweb/VerSAILLE) |
| G2 | **Verified Safe RL for Neural Network Dynamic Models** | NeurIPS 2024 | May 2024 | [2405.15994](https://arxiv.org/abs/2405.15994) | Curriculum learning + incremental verification for finite-horizon safety proofs of NN controllers | Training-time (verified training) | Finite-horizon reachability safety | No |
| G3 | **Dynamic Model Predictive Shielding (DMPS)** | NeurIPS 2024 | May 2024 | [2405.13863](https://arxiv.org/abs/2405.13863) | Dynamic safe recovery actions via local planner + NN policy for long-term reward; provably safe with exponentially decreasing regret | Inference-time (shielding) | Provable safety via backup planning | No |
| G4 | **Formal Methods in Robot Policy Learning and Verification: A Survey** | TMLR 2025 | Jan 2026 | [2602.06971](https://arxiv.org/abs/2602.06971) | Comprehensive survey of FM-informed policy learning + verification of learned policies; first integrated perspective | Survey | Covers specification, synthesis, verification | N/A |
| G5 | **Runtime Safety Verification of NN Controlled System** (Case Study) | RV 2024 | Aug 2024 | [2408.08592](https://arxiv.org/abs/2408.08592) | POLAR-Express reachability analysis for Turtlebot with safe controller switching | Inference-time (runtime verification + switching) | Online reachability-based safety | No |

### Category H: Architectural Safety Frameworks and Position Papers

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| H1 | **Modular Safety Guardrails for FM-Enabled Robots in the Real World** | Preprint | Feb 2026 | [2602.04056](https://arxiv.org/abs/2602.04056) | Two-layer architecture: monitoring/evaluation + intervention; characterizes action, decision, and human-centered safety | Position paper / Framework | Composable, non-bypassable safety across autonomy stack | No |
| H2 | **Towards Robust and Secure Embodied AI: Survey on Vulnerabilities and Attacks** | Preprint | Feb 2025 | [2502.13175](https://arxiv.org/abs/2502.13175) | Categorizes exogenous + endogenous vulnerabilities; analyzes attacks on perception, decision-making, interaction | Survey | Comprehensive vulnerability taxonomy | N/A |
| H3 | **Safety Case Patterns for VLA-based Driving: RAISE** | Preprint | Mar 2026 | [2603.16013](https://arxiv.org/abs/2603.16013) | RAISE framework for constructing safety cases for VLA driving systems; demonstrates on SimLingo | Framework (safety assurance) | Instruction rejection + acceptance safety | No |

### Category I: Adjacent / Complementary Work

| # | Paper | Venue | Date | arXiv/URL | One-Line Summary | Method | Safety Property | Open Source |
|---|-------|-------|------|-----------|-----------------|--------|----------------|-------------|
| I1 | **Boolean CBF Composition** (Glotfelter et al.) | IEEE CDC 2017 | 2017 | [IEEE](https://ieeexplore.ieee.org/document/8511471/) | Boolean composition of CBFs in continuous time for multi-agent systems | Theory (continuous-time CBF) | Safety under CBF composition | N/A |
| I2 | **Design by Contract** (Meyer, 1992) | IEEE Computer | 1992 | [DOI](https://doi.org/10.1109/2.161279) | Original design-by-contract paradigm for software correctness | Theory (software engineering) | Preconditions, postconditions, invariants | N/A |

---

## Summary Statistics

| Category | Count | Key Takeaway |
|----------|-------|-------------|
| **A. Training-time safety** | 3 | SafeVLA (CMDP) is the flagship. RobustVLA addresses multi-modal robustness. Fine-tuning erosion is a new concern. |
| **B. Inference-time enforcement** | 6 | AEGIS (CBF-QP), Safety Chip (LTL), SELP (constrained decoding), RoboGuard (LLM+temporal logic). All higher overhead than SafeContract. |
| **C. Diffusion/flow safety** | 6 | Active area. SafeDiffuser, CoDiG, PACS, SafeFlow, SafeFlowMatcher, UniConFlow. All modify the generative process or add CBF post-hoc. |
| **D. Robustness benchmarks** | 6 | LIBERO-Plus, LIBERO-PRO, LIBERO-X, Eva-VLA, VLA-Arena, SafeLIBERO. All expose severe brittleness. |
| **E. Adversarial attacks** | 8 | Visual patches, textual jailbreaks, backdoors, cross-modal, sensor attacks, ISO-grounded. VLAs are highly vulnerable. |
| **F. Embodied safety benchmarks** | 6 | SafeAgentBench, AGENTSAFE, BeSafe-Bench, IS-Bench, HomeSafeBench, SafeMind. All show <40% safe completion. |
| **G. Formal methods** | 5 | VerSAILLE, verified safe RL, DMPS, runtime verification. None applied to VLA specifically. |
| **H. Frameworks/surveys** | 3 | Modular guardrails, vulnerability survey, safety cases. All argue for composable safety. |
| **TOTAL** | **43 papers** | |

---

## SafeContract Positioning Against Each Paper

### Direct Comparisons (must cite and differentiate)

| Paper | How SafeContract Differs |
|-------|------------------------|
| **SafeVLA (A1)** | SafeVLA requires retraining via CMDP. SafeContract wraps any pretrained VLA with zero model modification. Orthogonal - you could use both. |
| **AEGIS/VLSA (B1)** | AEGIS solves CBF-QP at runtime (1-10ms). SafeContract uses box constraint clipping (<50us). Different capability/overhead tradeoff. AEGIS handles semantic obstacles; SafeContract handles action-space bounds. Complementary. |
| **Safety Chip (B2)** | Safety Chip operates at plan level (LTL over discrete actions). SafeContract operates at action level (continuous constraints). Different layers of the safety stack. |
| **SELP (B3)** | SELP constrains plan generation via LTL decoding. SafeContract constrains action execution. Could stack SELP + SafeContract for plan + action safety. |
| **RoboGuard (B4)** | RoboGuard is decision-level (rejects unsafe plans). SafeContract is action-level (clips unsafe actions). Complementary layers. |
| **SafeDiffuser (C1)** | SafeDiffuser modifies the diffusion process. SafeContract is architecture-agnostic - works on autoregressive, flow matching, and diffusion VLAs equally. |
| **CoDiG (C2)** | CoDiG embeds barriers into diffusion. SafeContract wraps any policy output. CoDiG is more elegant for diffusion; SafeContract is more general. |
| **PACS (C3)** | PACS uses reachability analysis - stronger guarantees but only for diffusion policies. SafeContract works on any policy with simpler (box) guarantees. |
| **SafeFlow/SafeFlowMatcher (C4-C5)** | These are flow-matching-specific CBF methods. SafeContract is architecture-agnostic. |
| **VerSAILLE (G1)** | VerSAILLE verifies the NN controller itself (intractable for VLAs). SafeContract sidesteps NN verification by verifying the contract layer (trivial). |
| **Modular Guardrails (H1)** | SafeContract is a concrete instantiation of their intervention layer at the action level. Validates their architectural argument with formal theory. |

### Papers to Cite but Not Compare Against

| Paper | Why Cite |
|-------|---------|
| **RobustVLA (A2)** | Shows multi-modal perturbation vulnerability - motivates why post-hoc safety is needed. |
| **Fine-tuning erosion (A3)** | Motivates why training-time safety is fragile - SafeContract is robust to fine-tuning. |
| **All attack papers (E1-E8)** | Motivate the threat model. VLAs produce dangerous actions under both natural and adversarial conditions. |
| **All robustness benchmarks (D1-D6)** | Establish that VLAs fail frequently. SafeContract catches these failures at the action boundary. |
| **All embodied benchmarks (F1-F6)** | Background on broader embodied safety. Less directly relevant (plan-level, not action-level). |
| **Formal methods survey (G4)** | Positions SafeContract within the formal methods landscape for robot policies. |
| **DMPS (G3)** | Related shielding approach, but for RL policies, not VLAs. Different compute regime. |

---

## Key Gaps SafeContract Fills

1. **No design-by-contract for VLA**: Nobody has applied assume-guarantee contracts to VLA policies. All prior work uses either training-time constraints (SafeVLA), CBF optimization (AEGIS), or temporal logic (Safety Chip). The design-by-contract framing is novel.

2. **No composition theory**: AEGIS handles single CBFs. Safety Chip handles single LTL formulas. Nobody has formalized when stacking multiple safety constraints (workspace + velocity + force) preserves individual guarantees. Theorem 2-3 are novel.

3. **No contract parameter learning from demos**: AEGIS requires manual CBF specification. SafeVLA requires reward engineering. SafeContract learns parameters from DROID/Bridge V2 demonstrations via confidence-bounded percentile estimation.

4. **No Pareto analysis of safety vs. task success**: Prior work reports one operating point. SafeContract maps the full tradeoff curve, letting practitioners choose their risk tolerance.

5. **No architecture-agnostic solution**: SafeDiffuser is diffusion-only. SafeFlow is flow-matching-only. SafeContract wraps autoregressive (OpenVLA), flow matching (SmolVLA, pi0), and any future architecture.

6. **No sub-millisecond enforcement**: AEGIS QP is 1-10ms. PACS reachability is slower. SafeContract clipping is <50us. Critical for 10+ Hz real-time control.

---

## Related Work Section Outline (for paper)

### Paragraph 1: VLA Safety Alignment
SafeVLA [A1], RobustVLA [A2]. Training-time approaches require retraining for each safety spec. SafeContract is zero-shot.

### Paragraph 2: Inference-Time Safety Enforcement
AEGIS [B1] (CBF-QP), Safety Chip [B2] (LTL), SELP [B3] (constrained decoding), RoboGuard [B4] (LLM+logic). All operate at different levels. SafeContract is the first at the continuous action level with formal composition theory.

### Paragraph 3: Safe Diffusion and Flow Matching
SafeDiffuser [C1], CoDiG [C2], PACS [C3], SafeFlow [C4], SafeFlowMatcher [C5], UniConFlow [C6]. Architecture-specific methods. SafeContract is architecture-agnostic.

### Paragraph 4: Control Barrier Functions and Composition
Boolean CBF composition [I1]. Continuous-time theory. SafeContract provides discrete-time box constraint composition - simpler but directly applicable to VLA inference loops.

### Paragraph 5: Formal Methods for Robot Policies
VerSAILLE [G1], verified safe RL [G2], DMPS [G3], FM survey [G4]. Verify the NN or the dynamics. SafeContract: verify the contract layer, not the neural network.

### Paragraph 6: Design by Contract
Meyer's DbC [I2]. Applied extensively in software (Eiffel, Ada SPARK). Never applied to robot learning policies. SafeContract bridges software engineering DbC with VLA safety.

---

## Papers to Watch (not yet published but anticipated)

- SafeVLA v2 / follow-ups from PKU Alignment team
- AEGIS journal extension with more benchmarks
- CoRL 2026 VLA safety papers (deadline May 29)
- NeurIPS 2026 SafeGenAI Workshop papers
- RSS 2026 Safe Robot Learning Workshop

---

## Last Updated: 2026-03-29
