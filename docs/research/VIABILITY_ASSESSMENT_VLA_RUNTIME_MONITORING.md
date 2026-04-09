# Long-Term Viability Assessment: VLA Runtime Monitoring as a Research Direction

**Date**: 2026-04-08
**Scope**: SafeContract and broader VLA runtime monitoring
**Method**: Web search across academic papers, industry announcements, funding programs, conference workshops, regulatory docs, hiring signals

---

## Executive Summary

**Verdict: Multi-year research program, not a 1-paper idea. But the strongest angle is NOT "runtime monitoring for VLAs" - it is "deployment-time reliability infrastructure for learned robot policies."**

The VLA deployment wave is real and accelerating. 164 VLA papers at ICLR 2026 (18x increase from 9 at ICLR 2025). Physical Intelligence, Google DeepMind, and dozens of startups are deploying VLA-based robots commercially. But safety infrastructure is 2-3 years behind model capabilities. This gap is widening, not closing. The research community knows it - DARPA SAFRON, two CoRL SAFE-ROL workshops, EU AI Act requirements, Google's ASIMOV benchmark - but nobody has built the standard toolkit yet. SafeContract occupies a specific niche (action-space guardrails) within this larger opportunity.

---

## 1. Market Trajectory: Is VLA Deployment Actually Happening?

### YES - and accelerating faster than expected

**Evidence of real deployment (not just papers):**

- **Physical Intelligence pi0**: Trained on 8 embodiments, 68 tasks. Open-sourced Feb 2025. pi0.5 (Sep 2025) and pi0.6 shipped. Commercial partnerships announced Feb 2026. Not just research - production trajectory.
  - Source: https://www.pi.website/blog/openpi

- **Google Gemini Robotics**: Full VLA (Gemini Robotics) + reasoning model (Gemini Robotics-ER) + edge model (Gemini Robotics On-Device, Jun 2025). Gemini Robotics 1.5 (Sep 2025) with ASIMOV safety benchmark. On-Device version specifically optimized for "low-latency and high reliability" local execution.
  - Source: https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/

- **ICLR 2026**: 164 VLA submissions (vs 9 at ICLR 2025) - an 18x increase in one year. The field has exploded.
  - Source: https://mbreuss.github.io/blog_post_iclr_26_vla.html

- **Tesla Optimus, Figure 02, and others**: Now rely on VLA models for household tasks - laundry, cooking, tidying.
  - Source: https://www.onoff.gr/blog/en/robot/foundation-models-ai-robotics-2026/

- **GR00T N1.7**: NVIDIA's model now available in early access with *commercial licensing* for production deployments.
  - Source: https://blogs.nvidia.com/blog/european-robot-makers-isaac-omniverse-halos-safe-physical-ai/

**The critical observation from ICLR 2026 analysis**: Open-weight VLAs achieve strong simulation scores but lag significantly in free-form, zero-shot real-world tasks vs closed-weight frontier models. LIBERO and CALVIN are "basically solved" as benchmarks - masking real deployment gaps. The gap between "works in sim" and "works in kitchens" is exactly where runtime monitoring lives.

**Assessment**: VLA deployment is not hypothetical. It's happening in 2026 with real robots, real commercial licenses, and real customers. The market for safety infrastructure will grow proportionally.

---

## 2. Competing Approaches

### 2a. Control Barrier Functions (CBFs)

**Status**: Mature theory, active research, dominant in safe control literature.

- **VLSA/AEGIS** (Xiao et al., Dec 2025): CBF-based QP safety layer for VLAs specifically. 59.16% improvement in obstacle avoidance, 17.25% improvement in task success. Created SafeLIBERO benchmark. This is the closest direct competitor to SafeContract.
  - Source: https://arxiv.org/abs/2512.11891

- **Learned CBFs (BarrierNet)**: End-to-end differentiable CBFs integrated into neural network training. The trend is toward learning certificates alongside policies.
  - Source: https://ieeexplore.ieee.org/document/10077790/

- **Uncertainty-Aware CBFs**: UA-PCBFs fuse probabilistic motion forecasting with formal safety guarantees, adjusting margins based on human motion uncertainty.
  - Source: https://www.sciencedirect.com/science/article/pii/S2949855425000589

- **2025 survey of advances in CBF theory**: Addresses practical challenges in safe control synthesis for autonomous and robotic systems.
  - Source: https://www.sciencedirect.com/science/article/abs/pii/S1367578824000142

**Honest take on CBFs vs SafeContract**: CBFs solve a harder problem (continuous safety constraint optimization) at higher cost (~1-10ms per action). SafeContract's box constraints are simpler but faster (<50us). CBFs handle obstacle avoidance, workspace geometry, contact forces. SafeContract handles bounds, velocity, workspace boxes. Different tradeoff points. They are complementary. But if someone asks "which do I use?", CBFs are the more general answer for obstacle-rich environments.

### 2b. Hamilton-Jacobi Reachability / Shielding

- **Latent Safety Filters** (2025): Generalize HJ reachability to operate on raw RGB observations via latent space. Can shield imitation-learned policies without modifying them.
  - Source: https://arxiv.org/html/2502.00935

- **Verifiable Safety Q-Filters** (2025): Model-free safety filters based on HJ reachability analysis with multiplicative Q-network structure.
  - Source: https://arxiv.org/abs/2506.15693

- **Sim-to-Lab-to-Real**: Uses HJ reachability for shielding + PAC-Bayes guarantees for lab-to-real transfer.
  - Source: https://www.sciencedirect.com/science/article/abs/pii/S0004370222001515

**Honest take**: HJ reachability is computationally expensive (offline computation, exponential in state dimension) but provides the strongest guarantees. Latent-space approaches are making it more practical. This is the "gold standard" that SafeContract should position itself *below* on the formality spectrum - simpler, cheaper, faster, but weaker guarantees.

### 2c. Safe RL / Constrained RL (Training-Time Safety)

- **SafeVLA** (PKU, NeurIPS 2025 Spotlight): CMDP-based safety alignment for VLAs. 83.58% improvement in safety metrics, reduces unsafe behaviors to 1/35 of baseline. Requires retraining.
  - Source: https://arxiv.org/abs/2503.03480

- **Survey of Safe RL** (2025): Comprehensive taxonomy - CPO, Lagrangian methods, Lyapunov-based, barrier function integration.
  - Source: https://arxiv.org/html/2505.17342v1

**Does training-time safety make runtime monitoring redundant?** NO, and here's why:

The safe RL literature itself acknowledges this. From the survey: "Current reactive approaches rely on anomaly detection, runtime monitoring, or fallback controllers that intervene after an unsafe action has been executed, but they cannot provide formal guarantees." But conversely, training-time safety cannot guarantee safety under distribution shift, hardware degradation, or novel adversarial inputs. The consensus is BOTH are needed: safe training reduces the frequency of unsafe actions, runtime monitoring catches the ones that slip through. Defense in depth.

### 2d. NVIDIA Isaac / Halos (Commercial)

- **NVIDIA Halos**: Full-stack safety system spanning hardware, AI models, software, tools and services. Now expanding from automotive to robotics. ANAB-accredited inspection lab for functional safety.
  - Source: https://blogs.nvidia.com/blog/european-robot-makers-isaac-omniverse-halos-safe-physical-ai/

- **Isaac Manipulator**: cuMotion for trajectory planning, nvblox for real-time 3D mapping and obstacle avoidance. GPU-accelerated.
  - Source: https://developer.nvidia.com/blog/making-industrial-robots-more-nimble-with-nvidia-isaac-manipulator-and-vention-machinemotion-ai/

**Honest take**: NVIDIA is building the *commercial infrastructure* for robot safety, focused on perception-level safety (obstacle avoidance via nvblox) and platform-level safety (Halos certification). They are NOT building action-space runtime monitors for learned policies. This is a gap SafeContract could fill - or that NVIDIA could acquire/integrate.

---

## 3. Where Runtime Monitoring MUST Exist (Regardless of Training Quality)

### 3a. Distribution Shift in Deployment

The single strongest argument for runtime monitoring. From the Deployment-Time Reliability dissertation (Mar 2026):

> "Reliability in the long tail of real-world deployment is not a property of any single model class nor a simple consequence of increased data scale, but rather emerges from the careful orchestration of reliability-promoting mechanisms spanning the autonomy stack."

Specific evidence:
- Even 95% per-step reliability yields only 60% success on a 10-step chain, 54% on a 12-step chain
- VLA policies remain "highly sensitive to variations in object geometry, pose, and appearance, to changes in lighting and scene context, to instruction phrasing, and to spurious environmental features"
- Source: https://arxiv.org/abs/2603.11400

**RAPT** (Feb 2026): Lightweight, self-supervised OOD detection for humanoid sim-to-real deployment. Uses calibrated reconstruction likelihoods. Directly addresses "silent policy failure."
- Source: https://arxiv.org/html/2602.01515

**Task-Driven OOD Detection with Statistical Guarantees** (IEEE TRO 2025): PAC-Bayes bounds on OOD detection performance. Guaranteed confidence bounds on when the robot is in-distribution.
- Source: https://ieeexplore.ieee.org/iel8/8860/4359257/10815081.pdf

### 3b. Hardware Degradation

- A worn hydraulic lift cylinder on an AMR in a Midwest distribution center caused a $1.8M incident (Jan 2025)
- Physics-informed neural networks for robot reducer degradation assessment
- Predictive maintenance reduces costs 25%, unplanned downtime 50%
- Sources: https://oxmaint.com, https://www.sciencedirect.com/science/article/abs/pii/S0888327025004947

Runtime monitoring must detect when the physical plant has drifted from what the policy was trained on. No amount of training-time safety handles a degraded actuator.

### 3c. Adversarial / Novel Environments

- **VLA-RISK Benchmark** (2025-2026): 296 scenarios, 3784 episodes testing VLA robustness. Current SOTA VLAs "face substantial challenges."
  - Source: https://openreview.net/forum?id=31EjDFwFEe

- **AttackVLA** (2025): Unified adversarial + backdoor attack framework for VLAs covering data construction, training, and inference.
  - Source: https://arxiv.org/html/2511.12149

- **Sensor attacks**: Physical-world interference (sound, light, EM) on VLA sensors is underexplored but demonstrated.
  - Source: https://dl.acm.org/doi/10.1145/3733800.3763262

### 3d. Regulatory Compliance (THE GROWING FORCE)

**EU AI Act (effective Aug 2024, enforcement Aug 2026):**
- High-risk AI systems MUST "automatically record events relevant for identifying national level risks and substantial modifications throughout the system's lifecycle"
- Runtime monitoring is not optional - it is LEGALLY REQUIRED for high-risk systems
- First harmonized standard for robotics under the AI Act targeting late 2026
- Sources: https://artificialintelligenceact.eu/, https://interoperable-europe.ec.europa.eu/collection/rolling-plan-ict-standardisation/robotics-and-autonomous-systems-rp-2026

**ISO Standards:**
- ISO 10218-1:2025 and ISO 10218-2:2025 - updated industrial robot safety with functional safety requirements
- ISO/TS 15066 integrated into ISO 10218-2:2025 - collaborative robot safety monitoring modes include "safety-rated monitored stop" and "speed and separation monitoring"
- Sources: https://www.iso.org/standard/73933.html, https://www.testriq.com/blog/post/robotic-safety-testing-meeting-iso-10218-13482-and-beyond

**NVIDIA Halos**: ANAB-accredited for functional safety inspections across robotics.

**Assessment**: Regulatory compliance is a guaranteed demand driver. Every robot with a learned policy in the EU will need runtime monitoring by Aug 2026. This is not speculative - it's law.

### 3e. Multi-Agent / Fleet Scenarios

- Boston Dynamics Orbit: Fleet management with telemetry and anomaly detection
- Modern CMMS platforms integrate IoT telemetry with ML anomaly detection
- Formation-aware conformal prediction for multi-robot safety
- Sources: https://bostondynamics.com/products/orbit/, https://arxiv.org/html/2603.08958

---

## 4. Research Community Interest

### Workshops (Strong Signal)

- **SAFE-ROL at CoRL 2025 (Seoul)**: 2nd edition. Organizers from FieldAI, CMU, Stanford, Mercedes-Benz, NVIDIA. Covers data curation, generalization, validation & verification.
  - Source: https://sites.google.com/view/corl-2025-safe-rol-workshop

- **SAFE-ROL 2nd edition at CoRL 2026**: Confirmed. This is becoming a standing workshop series.
  - Source: https://www.corl.org/program/workshops

- **ICRA 2025 - Robot Safety Under Uncertainty**: Workshop on safe control synthesis with intangible specifications.
  - Source: https://iscicra25.github.io/

- **ICRA 2025 - Public Trust in Autonomous Systems**: Safety in HRI.
  - Source: https://saferobotics.princeton.edu/ptas-icra25

### Funding (DARPA + NSF)

- **DARPA SAFRON**: "Safe and Assured Foundation Robots for Open Environments." Directly targets foundation model safety in robotics - hallucination, false confidence, jailbreaking. Solicitation closed Jan 2025. Active program.
  - Source: https://www.darpa.mil/research/programs/safron

- **NSF Safe Learning-Enabled Systems**: Partnership with Open Philanthropy and Good Ventures. Funds "technology that can continuously monitor the actions of autonomous robotic systems and intervene as needed to ensure safety." This is EXACTLY runtime monitoring.
  - Source: https://www.nsf.gov/funding/opportunities/dcl-funding-opportunities-engineering-research-artificial/nsf24-039

- **DoD FY2026**: $13.4B dedicated to autonomy and AI systems - first standalone budget line.

- **NSF budget risk**: Proposed 57% cut in FY2026 (not enacted, but signals headwinds for civilian research).

### Key Papers Forming the Field

| Paper | Venue | Contribution | Year |
|-------|-------|-------------|------|
| SafeVLA | NeurIPS 2025 Spotlight | CMDP training-time safety | 2025 |
| VLSA/AEGIS | arXiv | CBF-QP runtime safety for VLAs | 2025 |
| Deployment-Time Reliability | PhD dissertation | Monitoring taxonomy (Sentinel) | 2026 |
| RAPT | arXiv | OOD detection for humanoid sim-to-real | 2026 |
| VLA-RISK | OpenReview | Safety benchmark (296 scenarios) | 2025 |
| ASIMOV | Google DeepMind | Semantic safety benchmark | 2025 |
| Conformal CBF | arXiv | Adaptive conformal + CBF | 2025 |
| Learnable CP for Robotics | arXiv | Context-aware conformal prediction | 2025 |
| Latent Safety Filters | arXiv | HJ reachability in latent space | 2025 |
| SafeDiffuser | ICLR 2025 | CBF-guided diffusion | 2025 |
| Safety Filters (Annual Reviews) | Annual Reviews | Unified view of safety-critical control | 2025 |

### Companies Active in This Space

- **Google DeepMind**: ASIMOV benchmark, Gemini Robotics safety layers, Robot Constitutions
- **Physical Intelligence**: Open-source pi0 but safety is user's problem
- **NVIDIA**: Halos (platform safety), nvblox (obstacle avoidance), not action-space monitoring
- **Boston Dynamics**: Orbit fleet management, telemetry, predictive maintenance
- **1X, Figure, Tesla**: Deploying humanoid VLAs but safety approaches undisclosed
- **Standard Bots**: Documentation/education on robot safety standards

---

## 5. Honest Assessment: Limitations and Critiques

### Known Limitations of Runtime Monitoring

1. **False positives are a real problem**: "OOD detection methods often generate excessive false alarms since not all novel situations lead to actual failures." A monitor that cries wolf constantly will be disabled by operators.
   - Source: https://arxiv.org/html/2505.00779v1

2. **Post-hoc detection may be too late**: "External monitoring systems using VLMs typically detect errors after they manifest, providing insufficient time for intervention." If you detect the failure after the robot drops a knife, you haven't prevented anything.
   - Source: https://www.alphaxiv.org/overview/2510.09459v1

3. **Clipping degrades policy quality silently**: SafeContract's core mechanism (action clipping) can prevent task completion without any obvious signal. The Pareto curve can look bad - tight constraints kill success rate.

4. **Computational overhead**: Safety filter simulations have "a realtime factor of 1.8 without optimization" - meaning 1.8x slower than real-time. For <50us SafeContract this is fine, but more sophisticated approaches struggle at 10+ Hz control loops.
   - Source: https://arxiv.org/html/2509.12674

5. **Threshold tuning is unsolved**: "A key challenge is setting appropriate monitoring thresholds to balance sensitivity and avoiding false positives." This is where conformal prediction helps, but only partially.

6. **Scalability**: "In real applications, one would likely need to monitor more critical state-transitions than what was tested in controlled examples." Contract stacks grow complex.

### Specific Critiques of SafeContract's Approach

- **"Just clipping with a fancy name"**: The composition theory + conformal calibration are the differentiators, but reviewers at top venues may still find it incremental
- **Box constraints miss complex hazards**: CBFs handle arbitrary nonlinear constraints. SafeContract handles boxes. Real-world safety often involves obstacle geometry, not just joint limits.
- **No perception-level safety**: SafeContract operates on actions only. If the vision encoder hallucinates, SafeContract doesn't catch it.
- **LIBERO is "basically solved"**: If the benchmark is saturated, showing Pareto curves on it may not impress reviewers.

---

## 6. Growth Vectors: 3-Year Roadmap If You Built This Toolkit

### Year 1: Foundation (SafeContract + Extensions)
- Action-space contracts (what you have)
- Conformal calibration for automatic threshold tuning
- Composition theory and interference detection
- Support 3+ VLA architectures (OpenVLA, pi0, SmolVLA)
- pip-installable package with decorator API
- Pareto analysis framework for safety-performance tradeoffs

### Year 2: Beyond Action Space
- **Behavioral telemetry / anomaly detection**: CUSUM, STAC-style action distribution monitoring
- **OOD detection**: Conformal-calibrated OOD detectors using policy activations
- **Multi-level monitoring**: perception anomalies + action anomalies + task progress
- **Fleet monitoring**: Aggregated safety metrics across robot deployments
- **Predictive maintenance integration**: Hardware degradation signals feeding into safety margins
- **Sim-to-real validation**: Quantify and monitor sim-to-real gap at deployment time

### Year 3: Deployment-Time Reliability Platform
- **Online adaptation**: Adjust contracts based on deployment conditions
- **Policy selection/routing**: Choose between multiple policies based on safety profile
- **Regulatory compliance toolkit**: EU AI Act logging, ISO 10218 compliance reports
- **Integration with major frameworks**: LeRobot, Isaac Sim, ROS 2
- **Certification support**: Generate evidence for functional safety certification

### Adjacent Problems This Solves

| Problem | How runtime monitoring connects |
|---------|-------------------------------|
| Sim-to-real validation | Quantify action distribution shift between sim and real |
| Policy selection | Route tasks to the safest capable policy |
| Online adaptation | Tighten/loosen contracts based on observed performance |
| Fleet monitoring | Aggregate safety metrics, detect fleet-wide degradation |
| Dataset quality | Contract violations during replay indicate bad demonstrations |
| Regulatory compliance | Runtime logging satisfies EU AI Act requirements |
| Predictive maintenance | Safety margin adjustments based on hardware health |

---

## 7. Strongest Long-Term Angle

### Not "runtime monitoring" - instead: "Deployment-Time Reliability Infrastructure for Learned Robot Policies"

The framing matters enormously. "Runtime monitoring" sounds reactive and narrow. "Deployment-time reliability" positions the work as:

1. **A systems problem, not a control theory problem** - differentiated from CBF/HJ people
2. **Infrastructure, not a technique** - like how Prometheus is monitoring infrastructure vs a specific anomaly detector
3. **Complementary to everything** - works with SafeVLA (training), AEGIS (CBFs), HJ reachability (shielding), Halos (platform)
4. **Aligned with the money** - DARPA SAFRON, NSF Safe Learning, EU AI Act all fund reliability infrastructure
5. **Aligned with the gap** - 164 VLA papers at ICLR 2026, exactly 0 on deployment safety infrastructure

### The Killer Insight

From the ICLR 2026 VLA analysis: "LIBERO and CALVIN are basically solved" as benchmarks, yet real-world deployment remains fragile. The gap between benchmark performance and deployment reliability is the fundamental unsolved problem of the VLA era. SafeContract is one layer of the solution - the action-space layer. The full stack includes perception monitoring, behavioral anomaly detection, task progress tracking, hardware health, and fleet-level analytics.

### Concrete Paper Pipeline (if pursuing multi-year)

| Paper | Target | Angle |
|-------|--------|-------|
| SafeContract (formal contracts) | CoRL 2026 or ICRA 2027 | Composable action-space safety |
| Conformal action monitors | NeurIPS WS 2026 | CP-calibrated anomaly detection |
| ChaosVLA (fault injection) | NeurIPS WS 2026 | Chaos engineering for VLA reliability |
| Deployment-time reliability toolkit | RSS or CoRL 2027 | Full-stack monitoring integration |
| Fleet safety analytics | IROS 2027 | Multi-robot safety aggregation |
| Regulatory compliance framework | IEEE RA-L | EU AI Act / ISO 10218 compliance |

---

## 8. Final Verdict

### Is this a 1-paper idea or a multi-year program?

**Multi-year program.** But only IF you expand the scope beyond action clipping.

SafeContract as "composable box contracts with conformal calibration" is a solid 1-2 paper contribution. The formal composition theory is novel. The conformal calibration is timely. The Pareto analysis is practical. This gets you a workshop paper and possibly a top-venue paper if experiments are strong.

But the REAL opportunity is the deployment-time reliability platform. Nobody owns this space. Google has ASIMOV (benchmark, not toolkit). DARPA is funding SAFRON (the need). NVIDIA has Halos (platform-level, not policy-level). Physical Intelligence open-sourced pi0 with zero safety tooling. The community is screaming for this - 164 VLA papers at ICLR and the safety infrastructure is a handful of benchmark papers.

### Strongest angle for YOU specifically:

1. SafeContract is your wedge paper - publishable, practical, citable
2. Manning book chapters 9-11 are the educational platform
3. vla-edge toolkit is the code contribution
4. The 3-year program builds on all three

### What would kill this direction:

- A major lab (Google, NVIDIA, Meta) open-sources a comprehensive VLA safety toolkit (currently nobody has)
- CBF methods become cheap enough that box contracts are seen as redundant (currently CBFs cost 100-200x more compute)
- VLA deployment stalls (currently accelerating - 18x ICLR growth)
- Regulatory requirements don't materialize (EU AI Act already law)

### Probability assessment:

| Outcome | Probability |
|---------|------------|
| SafeContract accepted at workshop | 80% |
| SafeContract accepted at top venue | 40% |
| VLA safety becomes major research area by 2028 | 90% |
| Someone else builds the "standard" toolkit before you | 30% |
| Regulatory compliance drives commercial demand | 85% |
| This direction produces 3+ papers over 3 years | 70% |

---

## Sources

### VLA Deployment & Market
- [ICLR 2026 VLA Analysis - Moritz Reuss](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- [Physical Intelligence pi0](https://www.pi.website/blog/openpi)
- [Gemini Robotics 1.5 - Google DeepMind](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/)
- [Gemini Robotics On-Device](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)
- [Foundation Models for Robotics 2026](https://www.onoff.gr/blog/en/robot/foundation-models-ai-robotics-2026/)
- [VLA Survey - Real-World Applications](https://vla-survey.github.io/)

### Safety Approaches (CBF, HJ, Safe RL)
- [VLSA/AEGIS - CBF Safety Layer](https://arxiv.org/abs/2512.11891)
- [SafeVLA - NeurIPS 2025 Spotlight](https://arxiv.org/abs/2503.03480)
- [CBF Survey - Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-control-071723-102940)
- [Advances in CBF Theory](https://www.sciencedirect.com/science/article/abs/pii/S1367578824000142)
- [Latent Safety Filters - HJ Reachability](https://arxiv.org/html/2502.00935)
- [Verifiable Safety Q-Filters](https://arxiv.org/abs/2506.15693)
- [Safe RL Survey](https://arxiv.org/html/2505.17342v1)
- [SafeDiffuser](https://arxiv.org/html/2509.12674)

### Runtime Monitoring & OOD Detection
- [Deployment-Time Reliability Dissertation](https://arxiv.org/abs/2603.11400)
- [RAPT - OOD for Humanoids](https://arxiv.org/html/2602.01515)
- [Task-Driven OOD Detection - IEEE TRO](https://ieeexplore.ieee.org/iel8/8860/4359257/10815081.pdf)
- [Failure Prediction for Generative Policies](https://www.alphaxiv.org/overview/2510.09459v1)
- [Uncertainty-Aware Latent Safety Filters](https://arxiv.org/html/2505.00779v1)
- [Runtime Verification for Robotic Security](https://www.mdpi.com/2075-1702/11/2/166)

### Conformal Prediction for Robot Safety
- [Adaptive Conformal + CBF](https://arxiv.org/html/2407.03569)
- [Learnable CP for Robotics](https://arxiv.org/abs/2509.21955)
- [Formation-Aware Conformal Prediction](https://arxiv.org/html/2603.08958)
- [Conformal Prediction Survey - Foundation Models](https://link.springer.com/chapter/10.1007/978-3-032-15120-9_13)
- [Formal Verification with Conformal Prediction - IEEE](https://ieeexplore.ieee.org/iel8/5488303/11274416/11274485.pdf)

### Benchmarks & Evaluation
- [VLA-RISK Benchmark](https://openreview.net/forum?id=31EjDFwFEe)
- [ASIMOV Robot Safety Benchmark](https://arxiv.org/html/2503.08663v1)
- [AttackVLA](https://arxiv.org/html/2511.12149)

### Funding & Policy
- [DARPA SAFRON](https://www.darpa.mil/research/programs/safron)
- [NSF Safe Learning-Enabled Systems](https://www.nsf.gov/funding/opportunities/dcl-funding-opportunities-engineering-research-artificial/nsf24-039)
- [EU AI Act 2026 Compliance](https://artificialintelligenceact.eu/)
- [EU Robotics Standards Rolling Plan 2026](https://interoperable-europe.ec.europa.eu/collection/rolling-plan-ict-standardisation/robotics-and-autonomous-systems-rp-2026)
- [ISO 10218-1:2025](https://www.iso.org/standard/73933.html)

### Workshops & Community
- [SAFE-ROL CoRL 2025](https://sites.google.com/view/corl-2025-safe-rol-workshop)
- [ICRA 2025 Robot Safety Workshop](https://iscicra25.github.io/)
- [CoRL 2026 Workshops](https://www.corl.org/program/workshops)

### Industry & Commercial
- [NVIDIA Halos + Isaac](https://blogs.nvidia.com/blog/european-robot-makers-isaac-omniverse-halos-safe-physical-ai/)
- [Boston Dynamics Orbit](https://bostondynamics.com/products/orbit/)
- [Fleet Management 2026](https://cobotfinder.com/guides/robot-fleet-management)

### VLA Online Adaptation & Recovery
- [VLA-in-the-Loop](https://openreview.net/forum?id=aT4LG8c6DE)
- [Confidence-Aware Failure Recovery](https://arxiv.org/html/2602.10289)
