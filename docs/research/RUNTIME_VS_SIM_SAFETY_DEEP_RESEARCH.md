# Deep Research: Runtime Action-Space Monitoring vs Sim-Based Approaches for VLA Robot Safety

**Date**: 2026-04-08
**Question**: Is runtime action-space monitoring (conformal bounds, jerk detection, stall detection) genuinely valuable compared to simulation-based approaches?
**Method**: Web search across industry, academia, standards bodies, open-source repos

---

## EXECUTIVE SUMMARY

**Runtime monitoring is not just valuable - it is becoming recognized as essential, and the field is converging on this view.** The evidence from industry, academia, and standards bodies all point the same direction: simulation alone cannot guarantee safety at deployment time. Every company deploying real robots uses runtime monitoring in some form. The research gap for lightweight, black-box, action-space monitoring that works across VLA architectures is real but narrowing fast (SAFE, FIPER, FAIL-Detect, SV-VLA all appeared in 2025-2026).

**The honest assessment**: Runtime monitoring is a necessary layer, not a sufficient one. The strongest safety posture combines sim-based training/testing with runtime monitoring at deployment. The question is not "runtime OR sim" but "how much of each, and what does each layer catch?"

---

## 1. WHAT COMPANIES ACTUALLY USE FOR SAFETY

### Google DeepMind (AutoRT / Gemini Robotics)

**Approach: Layered sim + runtime + human oversight**

- **AutoRT** (2024): Deployed up to 20 robots simultaneously over 7 months, 77K trials. Safety architecture includes:
  - LLM-based "Robot Constitution" (Asimov-inspired rules) for task selection
  - Classical runtime monitors: force/torque thresholds, joint stress limits, automatic stop on overload
  - Human supervisor always in line-of-sight with physical kill switch
  - Source: https://auto-rt.github.io/

- **Gemini Robotics** (2025-2026): Three-layer "Swiss cheese" safety model:
  - Layer 1 - Semantic safety: Model reasons about harmful actions before executing
  - Layer 2 - Physical safety: VLA composed with "low-level safety-critical controllers" for collision avoidance, force limiting
  - Layer 3 - Operational safety: Continuous vulnerability assessment
  - Critical caveat from Google: "This feature is not a guaranteed safety-rated system"
  - Source: https://deepmind.google/models/gemini-robotics/responsibly-advancing-ai-and-robotics/

**Key insight**: Google uses BOTH sim and runtime. Sim for training and task selection. Runtime monitors (force limits, joint stress, collision detection) are always active. Human oversight as final layer.

### Physical Intelligence (pi0)

**Approach: Runtime action clamping (basic) + human oversight**

- pi0/pi0-FAST deployed with basic safety controllers that clip velocity actions
- Users report "robot drives randomly with no clear task intent" and safety controller regularly clips velocity (GitHub openpi #414)
- No published formal safety framework beyond action clamping
- Key gap: No transformation utilities between platforms - policy commands valid for one robot exceed joint limits on another (openpi #802)
- Source: https://github.com/Physical-Intelligence/openpi

**Key insight**: Even the most capable VLA lab relies on primitive runtime monitoring (velocity clipping). No sim-based safety verification published.

### Tesla Optimus

**Approach: Hardware safety design + runtime sensing + sim training**

- "Human-safe design with limited force output and smooth motion profiles"
- Runtime: Autopilot-grade cameras for visual perception, force/torque sensing across all joints and feet
- Balance Control System patent: real-time posture monitoring with sensor array for position data, immediate posture correction
- Sim: Trained using motion capture + simulation-based ML (same FSD architecture)
- Source: https://en.wikipedia.org/wiki/Optimus_(robot), https://botinfo.ai/articles/tesla-optimus

**Key insight**: Tesla uses sim for training but runtime sensing/monitoring for safety. Force limiting and collision detection are runtime features.

### Figure AI

**Approach: Sim training + (reportedly) limited runtime safety**

- Figure 02 trained with "motion-capture data and simulation-based machine learning"
- Former head of product safety sued the company (Nov 2025) for being fired after raising concern that robots were "strong enough to fracture a human skull"
- Contributing to ISO humanoid safety standards development
- No published runtime safety framework
- Source: https://en.wikipedia.org/wiki/Figure_AI

**Key insight**: Figure's safety controversy suggests runtime safety monitoring may be insufficient at the company. Standards work is ongoing but years from finalization.

### 1X Technologies (NEO)

**Approach: Hardware safety-by-design + teleoperation fallback**

- Physical: Soft 3D lattice polymer exterior, rounded edges, 66 lb lightweight frame, pinch-proof joints, tendon-driven (inherently compliant)
- Runtime: Reinforcement learning with real-time sensor fusion, "no go" zones
- Teleoperation fallback for complex tasks
- "Supervised operation around young children and pets during early adoption"
- Source: https://www.1x.tech/neo

**Key insight**: 1X invests heavily in hardware-level safety AND runtime monitoring. They explicitly recommend human supervision, acknowledging runtime monitoring alone is insufficient.

### Boston Dynamics (Atlas)

**Approach: AV-inspired runtime safety + standards leadership**

- Onboard safety system "leveraging best practices from the autonomous vehicle industry"
- Runtime: Detects people and vehicles in workplaces, pauses when human enters radius
- Fenceless guarding based on runtime proximity detection
- Co-leading ISO 25785-1 development for dynamically stable robots
- Source: https://bostondynamics.com/products/atlas/

### Agility Robotics (Digit)

**Approach: Pursuing first humanoid ISO functional safety certification**

- Currently operates in segregated zones (no human co-workers present)
- Passed OSHA-recognized NRTL field inspection (2025)
- Working toward ISO functional safety certification - would be first humanoid cleared for fenceless collaboration
- Actively defining requirements for force/speed limiting and proximity detection/auto-stop
- Deployed 7+ units at Toyota Manufacturing Canada (Feb 2026)
- Target: ISO certification mid-to-late 2026
- Source: https://www.automationworld.com/factory/robotics/article/55303585/

**Key insight**: Agility is the most standards-focused humanoid company. Their approach is runtime monitoring (force limiting, proximity detection, auto-stop) validated through ISO certification.

### NVIDIA (Halos + Isaac)

**Approach: Full-stack safety platform (sim + runtime + certification)**

- NVIDIA Halos: "Full-stack safety system designed to quickly develop and deploy physical AI"
  - Design-time, deployment-time, and validation-time guardrails
  - Expanding from autonomous vehicles to robotics (2025-2026)
  - Source: https://www.nvidia.com/en-us/ai-trust-center/physical-ai/safety-certification/
- Isaac Lab: Action clipping recently added but has fundamental bugs (applying joint-space clips to task-space actions - GitHub #1548)
- Source: https://developer.nvidia.com/isaac

**Key insight**: NVIDIA's Halos platform explicitly includes runtime ("deployment-time") guardrails alongside sim-based testing.

### Summary Table: Industry Safety Approaches

| Company | Sim-Based Safety | Runtime Monitoring | Human Oversight | ISO Standards |
|---------|-----------------|-------------------|----------------|---------------|
| Google DeepMind | Training + task selection | Force/torque limits, joint stress, collision | Always in line-of-sight | No |
| Physical Intelligence | Not published | Velocity clamping (basic) | Required | No |
| Tesla | FSD training pipeline | Force limiting, sensor fusion, posture monitoring | Planned phase-out | Contributing |
| Figure AI | Sim-based ML training | Reportedly limited | Required | Contributing |
| 1X | RL with sensor fusion | Hardware compliance, no-go zones, sensor fusion | Recommended | No |
| Boston Dynamics | Sim training | AV-style proximity detection, fenceless guarding | Reduces need but present | Co-leading |
| Agility | Sim training | Force/speed limiting, proximity detection | Currently required | Pursuing certification |
| NVIDIA | Isaac Sim full-stack | Halos deployment-time guardrails | Platform-dependent | Halos certification path |

**Every single company uses runtime monitoring. None relies on sim alone.**

---

## 2. THE SIM-TO-REAL SAFETY GAP - WHY SIM ALONE IS INSUFFICIENT

### Known Failure Modes Simulation Cannot Catch

1. **Contact dynamics**: Real-world contact is inherently complex and nonlinear. Materials deform under pressure, friction varies with relative velocity, and contact states alternate between sticking, slipping, and separation. Simulators rely on simplified point contacts, linearized friction cones, or compliant spring-damper systems. Source: https://arxiv.org/html/2510.20808v1

2. **Deformable objects**: Currently difficult to do effortless sim-to-real transfer and simulate deformable objects or liquids. Modeling errors in contact dynamics between robotic arms and deformable objects lead to policy failure. Source: https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515

3. **Sensor noise and degradation**: Real sensors degrade over time (dirty cameras, drifting IMUs, loose cables). Sim cannot model progressive hardware degradation.

4. **Environmental non-stationarity**: Conditions change rapidly and unpredictably in real environments. A robot navigating a simulated environment encounters uniform friction and predictable obstacles. In reality, surfaces vary in texture and obstacles move.

5. **Human behavior**: Humans are unpredictable. Sim-based human models are oversimplified. Runtime proximity detection catches actual humans.

6. **Policy distribution shift**: "Once a robotic model is trained in simulation, it may not be able to adapt to new or changing conditions in the real world without significant retraining." Source: https://arxiv.org/html/2510.20808v1

7. **Multi-robot interaction**: Emergent failure modes from fleet deployments are hard to predict in sim.

### Formal Argument: Safety Properties Do Not Transfer

Key paper - **Sim-to-Lab-to-Real** (Hsu et al., Artificial Intelligence Journal 2023):
- "Previous Sim-to-Real techniques do not explicitly address safety of the robots"
- "Safety violations are inconsequential in simulation, robots trained without safety considerations will tend to exhibit similar unsafe behavior once deployed"
- "Previous techniques do not provide any guarantees on robots' performance or safety when deployed in different real environments"
- Proposes dual policy (performance + safety backup via HJ reachability) with PAC-Bayes generalization bounds
- Even with their formal approach, they acknowledge a "certificate" of generalization performance and safety is necessary BEFORE deployment
- Source: https://arxiv.org/abs/2201.08355

### Digital Twin Approaches (Real-is-Sim) - The "Middle Ground"

**Real-is-Sim** (2025): Dynamic digital twin synchronized at 60Hz with real world. Policy always acts on simulated robot; real robot follows. This provides "consistent policy behavior between simulation and deployment, scalable and safe offline evaluation."
- Source: https://arxiv.org/abs/2504.03597

However, digital twin approaches still have limitations:
- The twin must be continuously calibrated against reality
- Physics fidelity gaps remain (deformable objects, liquids, friction)
- Adds latency (simulation must run in parallel)
- Still needs runtime monitoring for twin-reality divergence

---

## 3. RUNTIME MONITORING FOR LEARNED POLICIES - THE ACADEMIC LANDSCAPE

### Tier 1: VLA-Specific Runtime Monitors (2025-2026)

| Paper | Venue | Method | Uses CP? | VLA-Specific? | Black-Box? | Source |
|-------|-------|--------|----------|---------------|------------|--------|
| **SAFE** | NeurIPS 2025 | Learned failure score from VLA internal features + conformal calibration | Yes | Yes (OpenVLA, pi0, pi0-FAST) | No (needs model internals) | https://arxiv.org/abs/2506.09937 |
| **FIPER** | NeurIPS 2025 | RND for OOD + action chunk entropy, CP calibration | Yes | Yes (diffusion, flow matching) | No (trains RND network) | https://arxiv.org/abs/2510.09459 |
| **FAIL-Detect** | RSS 2025 | Sequential OOD via flow-based density estimator, CP thresholds | Yes | Imitation learning policies | No (trains density model) | https://arxiv.org/abs/2503.08558 |
| **CoVer-VLA** | arXiv Feb 2026 | Contrastive verifier for instruction-action alignment, test-time verification | No | Yes (VLA-specific) | No (trains verifier) | https://arxiv.org/abs/2602.12281 |
| **SV-VLA** | arXiv Apr 2026 | Lightweight verifier for speculative execution, triggers replanning | No | Yes (any VLA with action chunks) | Semi (lightweight verifier) | https://arxiv.org/abs/2604.02965 |
| **Modular Guardrails** | arXiv Feb 2026 | Position paper: monitoring + intervention layers | N/A | Any FM robot | Architecture | https://arxiv.org/abs/2602.04056 |

**Critical observation**: SAFE, FIPER, and FAIL-Detect all require training auxiliary models on VLA features or demonstrations. None is truly black-box or training-free at the action level. CoVer-VLA and SV-VLA require training contrastive/lightweight verifiers.

### Tier 2: General Runtime Monitoring for Learned Policies

| Paper | Venue | Method | Source |
|-------|-------|--------|--------|
| **Black-Box Simplex Architecture** | ISSE 2024 | Runtime checks replace static verification; switches from unverified NN controller to safe baseline | https://arxiv.org/abs/2102.12981 |
| **WATCH** | ICML 2025 | Weighted conformal test martingales for AI deployment monitoring | https://arxiv.org/abs/2505.04608 |
| **VerSAILLE** | NeurIPS 2024 | Provably safe NN controllers via differential dynamic logic (offline, not runtime) | https://arxiv.org/abs/2402.10998 |
| **DMPS** | NeurIPS 2024 | Dynamic safe recovery actions via local planner + NN policy shielding | https://arxiv.org/abs/2405.13863 |
| **Deployment-Time Reliability** | Stanford PhD Thesis 2026 | Detect failures via closed-loop behavior inconsistencies, no failure data needed | https://arxiv.org/abs/2603.11400 |
| **LLM Runtime Anomaly Detection** | RSS 2024 (Best Paper) | LLM embedding space for fast anomaly detection + MPC fallback | https://arxiv.org/abs/2407.08735 |
| **RoboSafe** | arXiv Dec 2025 | Executable safety logic with backward-reflective + forward-predictive reasoning | https://arxiv.org/abs/2512.21220 |
| **SPROUT** | Scientific Reports 2026 | Safety wrapper for black-box classifiers using ensemble uncertainty | https://www.nature.com/articles/s41598-026-45091-2 |
| **Conformal STL Shield** | ICASSP 2026 | CP + Signal Temporal Logic monitoring shield for RL | https://arxiv.org/abs/2602.14322 |

### Tier 3: CBF-Based Safety Filters (Architecture-Specific)

| Paper | Venue | Target Architecture | Source |
|-------|-------|-------------------|--------|
| **AEGIS/VLSA** | arXiv Dec 2025 | VLA + CBF-QP for obstacle avoidance | https://arxiv.org/abs/2512.11891 |
| **SafeDiffuser** | ICLR 2025 | Diffusion policies only | https://arxiv.org/abs/2306.00148 |
| **CoDiG** | CoRL 2025 | Diffusion policies only | https://arxiv.org/abs/2505.13131 |
| **PACS** | arXiv Nov 2025 | Diffusion policies only | https://arxiv.org/abs/2511.06385 |
| **SafeFlow/SafeFlowMatcher** | arXiv 2025 | Flow matching only | https://arxiv.org/abs/2504.08661 |
| **Neural CBFs (PNCBF)** | CDC 2024 | Quadcopters | https://arxiv.org/abs/2310.15478 |

**Pattern**: CBF methods are powerful but architecture-specific. Each generative model type (diffusion, flow matching, autoregressive) requires a different CBF formulation. No single CBF approach works across all VLA architectures.

---

## 4. INDUSTRY STANDARDS AND RUNTIME MONITORING

### ISO 10218-1:2025 (Industrial Robots) - Recently Updated

- Now includes 30+ safety functions (up from 2-3 in 2012 version)
- Collaborative robot requirements integrated from ISO/TS 15066
- Four methods of safe collaboration all require runtime monitoring:
  1. **Safety-rated monitored stop** - runtime detection of human presence
  2. **Hand guiding** - runtime force/torque sensing
  3. **Speed and separation monitoring** - runtime sensing (laser scanners) + real-time speed adjustment
  4. **Power and force limiting** - runtime force limiting on every joint
- Source: https://blog.ansi.org/ansi/iso-10218-1-2025-robots-and-robotic-devices-safety/

### ISO 25785-1 (Humanoid Robots) - Under Development

- First standard for dynamically stable robots (requires active balance control)
- Led by Agility, Boston Dynamics, A3
- Will require runtime monitoring for proximity detection and automatic stop
- Expected 2026-2027
- Source: https://www.iso.org/standard/91469.html

### ANSI/A3 R15.06-2025 (US Industrial Robot Safety) - Published Oct 2025

- 403-page three-part framework
- Includes cybersecurity considerations (new)
- Functional safety requirements (new emphasis)
- Source: https://www.automate.org/robotics/news/new-ansi-a3-r15-06-2025-american-national-standard-for-industrial-robot-safety-now-available-for-purchase

### Key Finding from Standards

**Every safety standard for collaborative or autonomous robots requires runtime monitoring.** There is no standard that accepts simulation-only safety validation. ISO 10218 explicitly requires runtime safety-rated sensing, force limiting, and speed monitoring for any robot operating near humans. This is not optional - it is a regulatory requirement.

---

## 5. CONFORMAL PREDICTION IN ROBOT SAFETY - SPECIFIC PAPERS

### Foundational
- **Sample-Efficient Safety Assurances** (Luo et al., 2021/2024): Framework combining CP with simulator. Needs only O(1/epsilon) samples for false negative rate guarantee. Applied to driver warning and grasping. Source: https://arxiv.org/abs/2109.14082
- **Conformal Decision Theory** (Lekeufack et al., 2023): Calibrate decisions (not predictions) with distribution-free guarantees. Applied to robot navigation. Source: https://arxiv.org/abs/2310.05921
- **Conformal CUSUM** (Vovk et al., ALT 2025): Theoretical foundation for conformal CUSUM change detection. Validity and efficiency proofs. Source: https://arxiv.org/abs/2412.03464

### Applied to Robot Safety
- **Safe Planning with CP** (Lindemann et al., RA-L 2023): MPC with conformal prediction regions in dynamic environments. Source: https://www.georgejpappas.org/wp-content/uploads/2023/08/Safe_Planning_in_Dynamic_Environments_Using_Conformal_Prediction.pdf
- **CP for Robot Safety from Sparse Human Feedback** (Jan 2025): CP identifies region guaranteed to contain user-specified fraction of future policy errors. Tested on 30 quadcopter flights. Source: https://arxiv.org/abs/2501.04823
- **Safe Multi-Robot Planning with CP** (Feb 2024): Task planning for language-instructed robot teams. Source: https://arxiv.org/abs/2402.15368
- **Conformal Safety Monitoring for Flight** (ICRA 2025 WS): CP-calibrated nearest-neighbor safety classification for flight abort criteria. Source: https://arxiv.org/abs/2511.20811
- **CPED-NCBFs** (2025): CP verifies learned CBFs. Probabilistic safety certificates from demonstrations. Source: https://arxiv.org/abs/2507.15022

### VLA-Specific CP Papers
- **SAFE** (NeurIPS 2025): Functional CP for VLA failure detection thresholds. Source: https://arxiv.org/abs/2506.09937
- **FIPER** (NeurIPS 2025): Split conformal for threshold calibration on action chunk entropy. Source: https://arxiv.org/abs/2510.09459
- **FAIL-Detect** (RSS 2025): CP as framework for OOD threshold detection. Source: https://arxiv.org/abs/2503.08558

---

## 6. THE GAP QUESTION: IS THERE A GENUINE RESEARCH GAP?

### What Exists (as of April 2026)

**Runtime monitors requiring trained auxiliary models:**
- SAFE (NeurIPS 2025): trains failure detector on VLA features
- FIPER (NeurIPS 2025): trains RND network
- FAIL-Detect (RSS 2025): trains flow-based density estimator
- CoVer-VLA (2026): trains contrastive verifier
- SV-VLA (April 2026): trains lightweight verifier

**Architecture-specific safety filters:**
- SafeDiffuser, CoDiG, PACS, SafeFlow: diffusion only
- SafeFlowMatcher, UniConFlow: flow matching only
- AEGIS: requires scene-specific CBF design
- All CBF methods require knowing the dynamics model or learning it

**What does NOT exist (the gap):**

1. **Truly black-box, training-free action-space monitoring for VLAs**: All existing VLA monitors (SAFE, FIPER, FAIL-Detect) require training auxiliary networks. No published work does pure action-space monitoring (e.g., statistical tests on the action stream itself) without any model training. This is a real gap.

2. **Architecture-agnostic safety layer that works across autoregressive, diffusion, AND flow matching VLAs**: SafeDiffuser is diffusion-only. SafeFlow is flow-matching-only. SAFE/FIPER test on multiple architectures but still train architecture-specific detectors. A single plug-and-play module that wraps any VLA policy with statistical monitoring on the output actions does not exist in published form.

3. **Multi-scale temporal anomaly detection for VLA actions**: FIPER aggregates over "short time windows" but doesn't do hierarchical detection (step/chunk/episode). CUSUM-style sequential change detection on VLA action streams is novel (confirmed by existing conformal prediction deep dive research).

4. **Sub-millisecond overhead action monitoring**: AEGIS is 1-10ms. PACS is slower. Existing runtime monitors don't report latency benchmarks suitable for 10+ Hz control loops on edge hardware.

5. **Standards-aligned VLA safety monitoring**: No published work maps runtime VLA monitoring to ISO 10218/15066/25785 requirements. The "Modular Guardrails" position paper (Feb 2026) argues this is needed but provides no implementation.

### Honest Assessment: How Big Is This Gap?

**The gap is real but narrowing.** In 2024, there was almost nothing. In 2025, SAFE, FIPER, and FAIL-Detect appeared. In early 2026, CoVer-VLA and SV-VLA push further. The trend is clear: runtime monitoring for VLA is becoming a hot topic.

**The specific gap for lightweight, black-box, action-space monitoring remains unfilled.** All existing approaches require:
- Training auxiliary models (SAFE, FIPER, FAIL-Detect)
- Access to model internals (SAFE uses VLA features)
- Architecture-specific designs (CBF methods)
- Significant compute overhead (LLM-based anomaly detection)

A truly lightweight approach that monitors ONLY the action output stream, uses conformal prediction for calibrated thresholds, requires NO training, and works across any VLA architecture - this specific combination does not exist in published form.

**But the window is closing.** Given the pace of work (3 major papers in 2025, 2+ in early 2026), someone will likely fill this gap within 6-12 months if you don't.

---

## 7. THE HONEST ANSWER: RUNTIME VS SIM

### Runtime monitoring IS genuinely valuable because:

1. **Sim cannot model everything**: Contact dynamics, sensor degradation, human behavior, deformable objects, environmental non-stationarity
2. **Standards require it**: ISO 10218, ISO/TS 15066, the upcoming ISO 25785 all mandate runtime monitoring
3. **Industry uses it universally**: Every company deploying robots uses runtime monitoring (Section 1)
4. **It's the last line of defense**: When the policy fails (and VLAs DO fail - LIBERO benchmarks show drops from 95% to <30% under perturbation), runtime monitoring catches it before the robot damages itself or its environment
5. **It's architecture-agnostic**: One runtime monitor works regardless of whether the policy is autoregressive, diffusion, or flow matching

### But runtime monitoring alone is NOT sufficient because:

1. **It's reactive, not proactive**: By the time you detect a bad action, the robot may have already committed to a trajectory
2. **Simple bounds miss semantic safety**: Velocity clamping doesn't know whether the robot is about to knock over a glass vs. performing a fast-reach task
3. **Threshold tuning is hard**: Tight bounds kill task performance; loose bounds miss violations (the Pareto tradeoff)
4. **No substitute for good training**: Runtime monitoring paper-clips a bad policy; it doesn't fix the root cause

### The strongest position is both:

**Simulation for training, testing, and pre-deployment validation + Runtime monitoring for deployment-time assurance.**

This is exactly what Google DeepMind, NVIDIA Halos, and the ISO standards ecosystem converge on. The "Modular Safety Guardrails" position paper (Feb 2026) from CMU/NVIDIA formalizes this as the monitoring + intervention architecture.

### Where runtime action-space monitoring specifically fits:

It is the **lowest layer** of the runtime monitoring stack:
1. **Semantic safety** (LLM/VLM reasoning about task appropriateness) - highest level
2. **Decision-level safety** (plan verification, RoboGuard, Safety Chip) - plan level
3. **Trajectory-level safety** (CoVer-VLA, SV-VLA, SAFE, FIPER) - rollout level
4. **Action-level safety** (conformal bounds, jerk detection, stall detection) - **lowest level, fastest, simplest, last line of defense**

This lowest layer is valuable precisely BECAUSE it is simple, fast, and architecture-agnostic. It catches the failure modes that higher layers miss (e.g., a semantically reasonable plan that produces physically dangerous actions due to distribution shift).

---

## 8. KEY PAPERS TO CITE (ORGANIZED BY RELEVANCE)

### Must-Cite (Directly Competing or Complementary)
1. SAFE: Multitask Failure Detection for VLAs - https://arxiv.org/abs/2506.09937
2. FIPER: Failure Prediction at Runtime - https://arxiv.org/abs/2510.09459
3. FAIL-Detect: Uncertainty-Aware Runtime Failure Detection - https://arxiv.org/abs/2503.08558
4. CoVer-VLA: Scaling Verification > Scaling Policy - https://arxiv.org/abs/2602.12281
5. SV-VLA: Speculative Verification for VLA - https://arxiv.org/abs/2604.02965
6. Modular Safety Guardrails for FM Robots - https://arxiv.org/abs/2602.04056

### Must-Cite (Theoretical Foundations)
7. Conformal CUSUM (Vovk et al., ALT 2025) - https://arxiv.org/abs/2412.03464
8. WATCH: Weighted-Conformal Martingales (ICML 2025) - https://arxiv.org/abs/2505.04608
9. Sample-Efficient Safety Assurances with CP - https://arxiv.org/abs/2109.14082
10. Black-Box Simplex Architecture - https://arxiv.org/abs/2102.12981

### Should-Cite (Broader Context)
11. Formal Methods in Robot Policy Verification Survey (TMLR 2025) - https://arxiv.org/abs/2602.06971
12. Sim-to-Lab-to-Real (Artificial Intelligence 2023) - https://arxiv.org/abs/2201.08355
13. LLM Runtime Anomaly Detection (RSS 2024 Best Paper) - https://arxiv.org/abs/2407.08735
14. Swiss Cheese Model for AI Safety - https://arxiv.org/abs/2408.02205
15. RoboSafe: Executable Safety Logic - https://arxiv.org/abs/2512.21220

### Industry References
16. Google DeepMind AutoRT - https://auto-rt.github.io/
17. Google Gemini Robotics Safety - https://deepmind.google/models/gemini-robotics/responsibly-advancing-ai-and-robotics/
18. NVIDIA Halos Safety Platform - https://www.nvidia.com/en-us/ai-trust-center/physical-ai/safety-certification/
19. ISO 10218-1:2025 - https://www.iso.org/standard/73933.html
20. ISO 25785-1 (under development) - https://www.iso.org/standard/91469.html

---

## 9. STRATEGIC IMPLICATIONS FOR SAFECONTRACT / VLA-EDGE

### The positioning that works:

**"SafeContract is the action-level safety layer in a multi-layer runtime monitoring stack. It is the ONLY approach that is simultaneously (a) black-box (no model internals), (b) training-free (no auxiliary networks), (c) architecture-agnostic (works on autoregressive, diffusion, flow matching), (d) sub-millisecond, and (e) formally composed via conformal calibration."**

This is defensible because:
- SAFE requires training a failure detector on VLA features (not black-box)
- FIPER requires training an RND network (not training-free)
- FAIL-Detect requires training a density estimator (not training-free)
- CoVer-VLA requires training a contrastive verifier (not training-free)
- CBF methods are architecture-specific (not agnostic)
- LLM anomaly detection is heavy (not sub-millisecond)

### The positioning to AVOID:

- Do NOT claim runtime monitoring replaces sim-based safety. It doesn't.
- Do NOT claim action-space monitoring is sufficient for safety. It's one layer.
- Do NOT overclaim novelty of individual components (conformal prediction, CUSUM). They are established tools.
- DO claim the specific combination and application domain is novel.
- DO claim the practical properties (lightweight, black-box, agnostic) fill a real deployment gap.
- DO acknowledge complementarity with SAFE, FIPER, etc. - they work at trajectory level, you work at action level.
