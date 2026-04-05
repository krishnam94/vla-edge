# Literature Sweep: VLA Safety & Deployment (March-April 2026)

**Date**: 2026-04-01
**Purpose**: Pre-submission sweep for SafeContract ICRA WS paper (deadline: ~April 12)
**Queries run**: 15+ across arxiv, semanticscholar

---

## CRITICAL FINDINGS (Must Address in Paper)

### 1. Agent Behavioral Contracts (ABC) - CLOSEST COMPETITOR
- **Title**: Agent Behavioral Contracts: Formal Specification and Runtime Enforcement for Reliable Autonomous AI Agents
- **Date**: February 25, 2026
- **Link**: [arXiv:2602.22302](https://arxiv.org/abs/2602.22302)
- **What it does**: Brings Design-by-Contract to autonomous AI agents. Defines (p, delta, k)-satisfaction for probabilistic contract compliance. Proves safe contract composition in multi-agent chains. Implemented as AgentAssert runtime library. Benchmark: 200 scenarios, 7 models, 1980 sessions.
- **Impact on SafeContract**: **MUST CITE AND DIFFERENTIATE.** ABC targets LLM agents (software), we target VLA physical robot actions (continuous action spaces, clipping). ABC's composition theory is for sequential agent chains; ours is for stacked spatial/velocity/force contracts on a single policy. Key differentiator: we operate on continuous action spaces with geometric proofs, they operate on discrete agent decisions with probabilistic bounds. **Helps** - validates the DBC-for-AI framing, but different domain.

### 2. Modular Safety Guardrails for FM-Robots
- **Title**: Modular Safety Guardrails Are Necessary for Foundation-Model-Enabled Robots in the Real World
- **Date**: February 3, 2026
- **Link**: [arXiv:2602.04056](https://arxiv.org/abs/2602.04056)
- **What it does**: Proposes two-layer modular guardrail architecture: Monitoring+Evaluation Layer and Intervention Layer. Characterizes FM-robot safety along 3 axes: action safety, decision safety, human-centered safety.
- **Impact on SafeContract**: **HELPS strongly.** Their taxonomy validates our approach - SafeContract is precisely an "action-level filtering" intervention. Cite this as the motivating position paper. Our work provides the formal contract theory they call for but don't implement.

### 3. SafeDec: Constrained Decoding for Robotics FMs
- **Title**: Constrained Decoding for Robotics Foundation Models
- **Date**: September 2025 (updated v3 in 2026)
- **Link**: [arXiv:2509.01728](https://arxiv.org/abs/2509.01728)
- **What it does**: STL-based constrained decoding at inference time for autoregressive robot FMs. Enforces temporal logic safety specs on candidate action trajectories without retraining.
- **Impact on SafeContract**: **MUST DIFFERENTIATE.** SafeDec operates at decoding time (modifies token sampling), we operate post-hoc on output actions (zero model coupling). SafeDec requires STL formulas; we use simpler assume-guarantee contracts. SafeDec is heavier (modifies inference); we're lighter (pure clipping). Different tradeoffs - complement each other.

---

## IMPORTANT PAPERS (Cite in Related Work)

### 4. RoboSafe: Executable Safety Logic for Embodied Agents
- **Date**: December 24, 2025
- **Link**: [arXiv:2512.21220](https://arxiv.org/abs/2512.21220)
- **What it does**: Hybrid reasoning runtime guardrail with backward reflective reasoning + forward predictive reasoning. Reduces hazardous actions by 36.8%. Tested on physical robotic arm.
- **Impact**: **Neutral-to-helps.** Different approach (LLM-based reasoning vs formal contracts). We can cite as "soft safety" vs our "hard safety guarantees."

### 5. AgentSpec: DSL for Runtime Constraints (ICSE 2026)
- **Date**: March 2025 (accepted ICSE 2026)
- **Link**: [arXiv:2503.18666](https://arxiv.org/abs/2503.18666)
- **What it does**: Lightweight DSL for specifying runtime constraints on LLM agents. Covers code execution, embodied agents, autonomous driving. 90%+ unsafe execution prevention. Millisecond overhead.
- **Impact**: **Neutral.** Targets LLM agents broadly, not continuous VLA action spaces. Different abstraction level. Can cite as related runtime enforcement.

### 6. ShieldAgent: Verifiable Safety Policy Reasoning
- **Date**: March 2025 (revised November 2025)
- **Link**: [arXiv:2503.22738](https://arxiv.org/abs/2503.22738)
- **What it does**: Guardrail agent using probabilistic rule circuits extracted from policy documents.
- **Impact**: **Neutral.** Software agent safety, not physical robot safety.

### 7. CompliantVLA-adaptor: Variable Impedance for Safe Contact
- **Date**: January 21, 2026
- **Link**: [arXiv:2601.15541](https://arxiv.org/abs/2601.15541)
- **What it does**: VLM-guided variable impedance control for safe contact-rich manipulation. Regulates stiffness/damping using real-time force feedback. Reduced force violations.
- **Impact**: **Complementary.** They address force safety via impedance control (physics-based). We address via force contracts (formal guarantees). Could cite as domain motivation - force safety in VLA is clearly needed.

### 8. Safe LLM-Controlled Robots via Reachability
- **Date**: March 5, 2025
- **Link**: [arXiv:2503.03911](https://arxiv.org/abs/2503.03911)
- **What it does**: Reachability analysis for LLM-robot systems. Constructs reachable sets from historical data for formal safety guarantees.
- **Impact**: **Neutral-helps.** Heavier formal methods approach. We can position SafeContract as lightweight alternative.

### 9. Conformal STL Shield (ICASSP 2026)
- **Date**: February 15, 2026
- **Link**: [arXiv:2602.14322](https://arxiv.org/abs/2602.14322)
- **What it does**: Conformal prediction + STL monitoring shield for RL policies. F-16 flight control case study.
- **Impact**: **Neutral.** Different domain (flight control), different mechanism (conformal + STL). Shows conformal prediction for action safety is active area.

---

## VLA EDGE DEPLOYMENT PAPERS (Context for Motivation)

### 10. Characterizing VLA Bottleneck for Edge AI
- **Date**: March 1, 2026
- **Link**: [arXiv:2603.02271](https://arxiv.org/abs/2603.02271)
- **What it does**: Google/Purdue. MolmoAct-7B on Jetson Orin/Thor. 75% latency in action generation phase (memory-bound).
- **Impact**: **Helps motivation.** Safety overhead must be minimal on edge. Our zero-overhead claim matters.

### 11. Embodied Foundation Models at Edge: Deployment Survey
- **Date**: March 16, 2026
- **Link**: [arXiv:2603.16952](https://arxiv.org/abs/2603.16952)
- **What it does**: Survey of 8 coupled barriers for edge deployment. Autoregressive VLAs limited by memory bandwidth; diffusion by compute latency.
- **Impact**: **Helps motivation.** Further validates that safety layers must be lightweight for edge.

### 12. RoboECC: Edge-Cloud Collaborative VLA Deployment
- **Date**: March 2026
- **Link**: [arXiv:2603.20711](https://arxiv.org/abs/2603.20711)
- **What it does**: Edge-cloud split for VLA models. Up to 3.28x speedup.
- **Impact**: **Neutral.** Deployment framework, not safety-focused.

### 13. AsyncVLA: Asynchronous Navigation on Edge
- **Date**: February 13, 2026
- **Link**: [arXiv:2602.13476](https://arxiv.org/abs/2602.13476)
- **What it does**: Decouples semantic reasoning (cloud) from reactive execution (edge). Handles 6s communication delays.
- **Impact**: **Neutral.** Architecture paper, not safety-focused.

---

## SAFETY BENCHMARKS (Useful for Evaluation Context)

### 14. BeSafe-Bench: Behavioral Safety for Situated Agents
- **Date**: January 30, 2026
- **Link**: [arXiv:2603.25747](https://arxiv.org/abs/2603.25747)
- **What it does**: Safety benchmark across Web, Mobile, Embodied VLM, Embodied VLA domains. 9 categories of safety-critical risks. Even best agents fail >60% on safety.
- **Impact**: **Helps motivation.** Empirical evidence that VLA safety is unsolved. Cite as motivation.

### 15. Hazard-Informed Data Pipeline for Robotics Safety
- **Date**: March 6, 2026
- **Link**: [arXiv:2603.06130](https://arxiv.org/abs/2603.06130)
- **What it does**: Bridges classical risk engineering with ML pipelines. Hazard ontology for safety envelope learning.
- **Impact**: **Neutral.** Different approach (data pipeline vs runtime contracts).

---

## EXISTING WORK - NO NEW VERSIONS FOUND

| Paper | Status | Notes |
|-------|--------|-------|
| **SafeVLA** (2503.03480) | No follow-up found | NeurIPS 2025 Spotlight. Still training-time CMDP. No v2. |
| **AEGIS/VLSA** (2512.11891) | No follow-up found | CBF-QP layer. Code released. No new version. |
| **Safety Chip** (2309.09919) | No follow-up found | LTL temporal logic. Still 2023 version. |
| **VerSAILLE** (2402.10998) | No follow-up found | dL for NN verification. Still 2024 version. |

---

## DESIGN-BY-CONTRACT FOR ROBOTS - GAP CONFIRMED

No paper applies DBC with assume-guarantee semantics to continuous VLA action spaces. The closest work is:

1. **ABC (2602.22302)** - DBC for LLM agents (discrete decisions, not continuous actions)
2. **Correct-by-Construction Missions (2306.08144)** - Contracts for mission planning (task-level, not action-level)
3. **Agent Contracts (2601.08815)** - Resource-bounded AI systems (theoretical framework, not VLA-specific)

**The gap remains: nobody has formal DBC contracts for continuous robot action spaces with composition theory and learned parameters.**

---

## PROPERTY-BASED TESTING FOR ROBOTS

- Only one older paper found: PBT for tabletop manipulation simulation (2108.08726)
- One 2026 survey mentions PBT briefly: "Before Autonomy Takes Control" (2602.02293)
- **No one is publishing PBT specifically for VLA policies.** This could be a separate contribution.

---

## CONFORMAL PREDICTION FOR ROBOT SAFETY (2026)

- Active area but focused on navigation/planning, not VLA action safety
- Conformal STL (2602.14322) - flight control, accepted ICASSP 2026
- Safe POMDP with adaptive CP (2404.15557)
- CP + semantic maps for safe planning (2509.25124)
- **No CP applied to VLA action contract parameter learning.** Our C3 (learning contract params from demos with confidence bounds) could connect here.

---

## ICRA 2026 WORKSHOPS (Submission Targets)

1. **"From Data to Decisions: VLA Pipelines for Real Robots"**
   - Paper deadline: May 1, 2026
   - Focus: VLA pipelines, safety metrics, benchmarks
   - 36 teams in competition track
   - [Website](https://icra2026vlapipeline.github.io/)

2. **"Semantics for Reliable Robot Autonomy"**
   - Paper deadline: April 1, 2026 (TODAY!)
   - Focus: semantic understanding, safe interaction
   - [Website](https://www.dynsyslab.org/icra2026-workshop-on-semantics-for-reliable-robot-autonomy/)

---

## RECOMMENDED ACTIONS FOR SAFECONTRACT PAPER

### Must Do (Before Submission)
1. **Cite ABC (2602.22302)** and differentiate clearly: they do discrete agent contracts, we do continuous action-space contracts with geometric proofs
2. **Cite Modular Guardrails (2602.04056)** as motivating position paper that validates our approach
3. **Cite SafeDec (2509.01728)** and position as complementary: they modify decoding, we filter output
4. **Cite BeSafe-Bench (2603.25747)** for motivation: VLA safety is empirically unsolved
5. **Cite VLA Edge Bottleneck (2603.02271)** for why zero-overhead safety matters

### Should Do
6. Add RoboSafe, AgentSpec to related work as "soft/heuristic safety" contrast
7. Add CompliantVLA-adaptor as evidence force safety in VLA is an active need
8. Note that SafeVLA, AEGIS have no follow-ups - our framing as complementary still holds

### Paper Positioning Update
- Before: "DBC for VLA is unexplored"
- After: "DBC for continuous robot action spaces is unexplored. ABC (Feb 2026) applies DBC to discrete LLM agent decisions, but continuous action spaces with composition theory and learned parameters remain open."
