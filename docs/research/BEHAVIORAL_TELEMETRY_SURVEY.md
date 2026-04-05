# Behavioral Telemetry for VLA Systems - Deep Research Survey

**Date**: 2026-04-01
**Purpose**: Assess whether "behavioral telemetry" is a recognized concept, map the landscape of action monitoring for deployed robots, and determine if SafeContract can claim first-mover framing as a behavioral telemetry system for VLAs.
**Status**: Research complete

---

## Executive Summary

**"Behavioral telemetry" is NOT an established term in robotics or VLA research.** It exists in two adjacent fields - cybersecurity (UEBA/behavioral analytics) and AI agent observability (LLM agent monitoring) - but nobody has applied it to robot action monitoring. This is a naming opportunity.

The closest work monitors observation/state patterns (sensor drift, OOD detection on inputs), NOT action patterns. SafeContract's insight - that constraint boundary interactions are information-rich diagnostic signals - has no direct precedent. The framing "behavioral telemetry for VLAs" is defensible and novel.

**Recommendation**: Coin "behavioral telemetry" for VLA systems. Reference the cybersecurity UEBA lineage (Gartner 2015) and the a16z "physical AI deployment gap" (2026) as motivation. Position SafeContract as the first system that treats safety contract violations as telemetry, not just enforcement.

---

## Finding 1: "Behavioral Telemetry" in Adjacent Fields

### Cybersecurity - UEBA (User and Entity Behavior Analytics)
- **Source**: Gartner coined UEBA in 2015, evolving from UBA
- **Key insight**: Establish behavioral baselines, detect anomalies from deviations
- **How it works**: Collects raw activity data from auth logs, endpoint telemetry, SaaS events, network flows. ML/statistical modeling establishes "normal" and flags deviations.
- **Relevance to SafeContract**: Direct conceptual parallel. UEBA monitors user actions against behavioral baselines. SafeContract monitors robot actions against safety contracts. The contract boundary IS the behavioral baseline.
- **URLs**: [CrowdStrike overview](https://www.crowdstrike.com/en-us/cybersecurity-101/exposure-management/behavioral-analytics/), [Vectra AI](https://www.vectra.ai/topics/behavioral-analytics)

### AI Agent Observability (LLM Agents)
- **Source**: Multiple platforms (Arize, Groundcover, OpenTelemetry)
- **Key insight**: "When AI systems are embedded into workflows, telemetry becomes behavioral monitoring" - monitoring tool calls, decision paths, execution traces, not just latency/throughput.
- **How it works**: Logs every task, tool call, and decision step. Tracks usage, performance, behavior, quality, and governance signals.
- **Relevance to SafeContract**: This is the software-agent analog. But no one has applied it to physical robot actions. The gap between "agent observability" and "robot observability" is exactly where behavioral telemetry sits.
- **URLs**: [Groundcover guide](https://www.groundcover.com/learn/observability/ai-agent-observability), [Arize LLM observability](https://arize.com/blog/llm-observability-for-ai-agents-and-applications/), [NexaStack OpenTelemetry](https://www.nexastack.ai/blog/open-telemetry-ai-agents)

### Assessment
**Nobody uses "behavioral telemetry" for robots.** The term is ours to define.

---

## Finding 2: The Physical AI Deployment Gap (a16z, 2026)

### Source
Oliver Hsu, a16z, 2026 - "The Physical AI Deployment Gap"

### Key Insight
"Observability tooling for deployed systems - logging, metrics, tracing, and alerting - is needed. The robotics equivalent of DevOps practices doesn't exist yet, and building it would dramatically reduce the operational burden of deployment."

### Why This Matters
a16z explicitly calls out robot observability as a missing layer. They frame it as infrastructure, not research. SafeContract's telemetry features are exactly what this gap demands - but nobody is building it from the safety contract angle.

### Framing for Paper
Cite this directly. "Hsu (2026) identifies robot observability as the critical missing infrastructure. We show that safety contracts naturally produce behavioral telemetry as a byproduct of enforcement."

### URL
[a16z - The Physical AI Deployment Gap](https://www.a16z.news/p/the-physical-ai-deployment-gap)

---

## Finding 3: Existing Robot Observability Platforms

### Foxglove (Series B, $40M raised)
- **What**: Visualization and observability platform for robotics
- **Focus**: Multimodal data ingestion, replay, debugging. Fleet monitoring with remote visualization/teleoperation over lossy networks.
- **Customers**: Amazon, Anduril, Chef Robotics, NVIDIA, Shield AI
- **What they monitor**: Sensor data, ROS topics, video feeds, point clouds. DevOps-style dashboards.
- **What they DON'T monitor**: Action patterns, safety constraint violations, policy behavior. They're infrastructure, not safety.
- **Relevance**: Foxglove is the "Datadog for robots" but focuses on sensor/state data, not action telemetry. SafeContract operates at a different level - the action boundary.
- **URLs**: [Foxglove](https://foxglove.dev/), [Foxglove remote viz](https://foxglove.dev/blog/announcing-remote-visualization-teleoperation-private-beta)

### Sift Stack (SpaceX alumni)
- **What**: Unified observability for mission-critical hardware
- **Focus**: High-frequency telemetry for rockets, aircraft, autonomous vehicles, robotics
- **What they monitor**: Torque, encoder, vision, force data. Joint motion profiles. Sensor drift, overheating.
- **Customers**: K2 Space, Impulse, Astrolab, Astranis
- **What they DON'T monitor**: Policy-level action patterns, safety contract violations. They monitor hardware signals, not policy behavior.
- **Relevance**: Sift is closest to "action telemetry" but still operates at the hardware/sensor level. They track joint torques but not "how close did the policy come to violating workspace bounds."
- **URLs**: [Sift Stack](https://www.siftstack.com/), [Sift robotics](https://www.siftstack.com/industry/robotics)

### Ubuntu Core 24 - Fleet Telemetry
- **What**: OS-level robotics fleet telemetry
- **Focus**: System health, device management
- **Relevance**: Infrastructure layer, not policy monitoring.
- **URL**: [Ubuntu Core 24](https://ubuntu.com/blog/ubuntu-core-24-robotics-telemetry)

### Assessment
**All existing platforms monitor sensor/hardware data. None monitor action patterns from learned policies.** The policy-action boundary is unmonitored in production.

---

## Finding 4: Runtime Verification for VLAs (Closest Competitors)

### "Do What You Say" - Runtime Reasoning-Action Alignment (NVIDIA, ICRA 2026)
- **Authors**: Yilin Wu et al. (NVLabs)
- **Paper**: arXiv 2510.16281
- **Key idea**: Sample multiple candidate action sequences, simulate outcomes, use VLM to select the one that best aligns with the VLA's textual plan.
- **What they monitor**: Reasoning-action alignment at runtime. Whether the VLA does what it says.
- **Difference from SafeContract**: They verify semantic alignment (is the action correct?). We verify physical safety (is the action safe?). Orthogonal concerns. Could compose.
- **URLs**: [arXiv](https://arxiv.org/abs/2510.16281), [Project page](https://yilin-wu98.github.io/steering-reasoning-vla/)

### CoVer-VLA - Contrastive Verification (Feb 2026)
- **Paper**: arXiv 2602.12281
- **Key idea**: Train a contrastive verifier on large-scale robotics datasets. At deployment, score instruction-action alignment. Select best action chunks.
- **What they monitor**: Vision-language-action alignment quality
- **Difference from SafeContract**: They verify task correctness. We verify physical safety. They need a trained verifier. We need zero training.
- **Performance**: 22% in-distribution gains, 45% improvement in real-world
- **URLs**: [arXiv](https://arxiv.org/abs/2602.12281), [GitHub](https://github.com/cover-vla/cover-vla)

### Code-as-Monitor (CVPR 2025)
- **Authors**: Zhou, Enshen et al.
- **Key idea**: Use VLM-generated code to formalize spatio-temporal constraints as monitoring functions. Track geometric elements (points, lines, surfaces) for constraint satisfaction.
- **What they monitor**: Spatial constraint violations during task execution. Reactive AND proactive failure detection.
- **Difference from SafeContract**: They monitor observation-space constraints with VLM code. We monitor action-space constraints with formal contracts. They're heavier (VLM in the loop). We're sub-50-microsecond.
- **Performance**: 28.7% higher success rate under disturbances
- **Relevance**: HIGHEST. This is the closest work to "constraint violations as diagnostic signals." But they operate in observation space, not action space.
- **URLs**: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Code-as-Monitor_Constraint-aware_Visual_Programming_for_Reactive_and_Proactive_Robotic_Failure_CVPR_2025_paper.html), [arXiv](https://arxiv.org/abs/2412.04455)

### Assessment
**No existing VLA verification system treats constraint violations as telemetry data.** They all use verification to improve policy output. None log, aggregate, or analyze violation patterns as a diagnostic signal.

---

## Finding 5: Safety Filters and Boundary Monitoring

### The Safety Filter - Unified View (Annual Reviews, 2024)
- **Source**: Annual Reviews of Control, Robotics, and Autonomous Systems
- **Key idea**: Three components - safety monitor (checks compliance), intervention (modifies unsafe inputs), fallback policy. Framework unifies CBF, Hamilton-Jacobi, MPC-based filters.
- **What they DON'T do**: Track violation patterns over time. The filter enforces, it doesn't observe.
- **URL**: [arXiv 2309.05837](https://arxiv.org/abs/2309.05837)

### Data-Driven Safety Filters (Berkeley Hybrid Robotics)
- **Source**: UC Berkeley
- **Key idea**: Learn safety filters from data when dynamics models are unavailable. Actuation-projected state predictors generate anomaly scores.
- **Relevance**: Anomaly scoring from action data is close to our violation telemetry. But it's real-time intervention, not pattern analysis.
- **URL**: [Berkeley paper](https://hybrid-robotics.berkeley.edu/publications/CSM2023_Safety_Filters.pdf)

### Gameplay Safety Filters (2024)
- **Key idea**: Simulate adversarial futures. If the robot would lose the "safety game," intervene. Continuously monitors policy safety.
- **Relevance**: Novel monitoring paradigm but focused on intervention, not telemetry.
- **URL**: [arXiv](https://arxiv.org/html/2405.00846v1)

### Assessment
**Safety filters enforce but don't observe.** They modify actions when boundaries are hit. They don't log violation rates, track which constraints fire most, or analyze temporal patterns. SafeContract's telemetry is the missing observability layer on top of safety enforcement.

---

## Finding 6: SPC (Statistical Process Control) - The Manufacturing Analog

### Core Concept
SPC uses control charts (X-bar, R-charts, p-charts) to track process performance over time. Upper and lower control limits define acceptable variation. Points outside limits signal process drift.

### Key Insight for SafeContract
**SPC is exactly what behavioral telemetry should look like for VLAs.** Replace "manufacturing process" with "VLA policy" and "product dimensions" with "action dimensions." A control chart of joint velocity values over episodes, with safety contract bounds as control limits, is SPC for robots.

### Modern Evolution
Industry 4.0 has broadened SPC to cyber-physical systems. AI-driven SPC integrates ML for real-time quality monitoring. But nobody has applied SPC concepts to learned robot policies.

### Opportunity
Frame SafeContract's telemetry as "SPC for VLA policies." Track violation rates per contract, per dimension, over time. Use control chart logic to detect policy degradation.

### URLs
[ASQ SPC overview](https://asq.org/quality-resources/statistical-process-control), [SPC + OOD detection paper](https://arxiv.org/abs/2402.08088)

---

## Finding 7: Digital Twin Runtime Verification

### Paper: Digital Twin Enabled Runtime Verification for AMRs (arXiv 2412.09913)
- **Key idea**: Use executable digital twins with TeSSLa-specified safety monitors. Continuous real-time validation of robot behavior.
- **What they monitor**: Safety and performance properties under uncertainty (sensor noise, environment variations)
- **Relevance**: Runtime monitors that check property satisfaction continuously - similar concept to contract monitoring. But focused on mobile robot navigation, not VLA action safety.
- **URL**: [arXiv](https://arxiv.org/abs/2412.09913)

### Infineon + NVIDIA (2025-2026)
- Digital twin architectures for humanoid robotics safety validation
- Simulation-driven development using digital twins of actuators and sensors
- **Relevance**: Validation in sim, not runtime monitoring of deployed policies

---

## Finding 8: Action Distribution Drift Monitoring

### OOD Detection via SPC (arXiv 2402.08088)
- **Key idea**: ML-enabled SPC framework for out-of-distribution detection and drift monitoring. Uses control charts to visually and statistically highlight deviations.
- **Relevance**: VERY HIGH. This paper bridges SPC and ML deployment monitoring. Apply the same concept to action distributions from VLA policies.
- **URL**: [arXiv](https://arxiv.org/abs/2402.08088)

### Drift Detection in Visual RL (PMC 2025)
- **Key idea**: Monitor deployed RL models using feature attribution drift. Reward signals are sparse, action distributions are high-dimensional.
- **Methods**: KS test, MMD for distribution shifts
- **Relevance**: They monitor input drift. We could monitor ACTION drift via contract violation rates.
- **URL**: [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12859422/)

### Real-Time Robot Anomaly Detection (2025)
- **Key idea**: Sparse Masked Autoregressive Flow model for real-time anomaly detection in robotic pick-and-place. Inferences within 1ms.
- **Relevance**: Action-level anomaly detection exists but uses learned models. SafeContract uses formal contracts - lighter, interpretable, composable.
- **URL**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197625013120)

---

## Finding 9: Autonomous Vehicle Telemetry - The Mature Analog

### What AVs Monitor
- Ego-centric operational performance: throttle, brake, steering, accelerations
- Leading metrics as indicators of future collision risk
- Safety assessment of tactical and operational driving performance
- Tesla publishes FSD Safety Reports with aggregate telemetry

### Key Insight
AV telemetry monitors action-level data (steering, throttle, brake) as safety signals. This is the closest existing paradigm to what we propose. But it's custom per AV company, not a reusable framework.

### Opportunity
SafeContract generalizes what AV companies do with custom telemetry into a reusable, pip-installable framework for any VLA policy.

### URLs
[Tesla FSD Safety Report](https://www.tesla.com/fsd/safety), [RAND AV Safety Framework](https://www.rand.org/content/dam/rand/pubs/research_reports/RR2600/RR2662/RAND_RR2662.pdf), [TRL AV Monitoring Framework](https://www.trl.co.uk/Uploads/TRL/Documents/PPR2018--In-use-monitoring-of-AVs---Safety-Monitoring-Framework.pdf)

---

## Core Insight: The Boundary as Information

**Nobody treats safety constraint violations as a rich diagnostic signal.** Everyone treats them as binary (violated or not) and responds (clip, reject, halt). SafeContract's key insight:

1. **Violation rate per contract** tells you which physical constraints the policy struggles with
2. **Violation rate over time** tells you if the policy is degrading (drift detection)
3. **Violation rate per task** tells you which tasks push the policy to its limits
4. **Clipping magnitude distribution** tells you HOW FAR the policy is from safety bounds
5. **Correlation between contracts** tells you if the policy has systematic blind spots
6. **Violation rate vs. strictness** gives you the Pareto frontier for deployment decisions

This is SPC for robot policies. The contract boundary IS the control chart limit.

---

## Competitive Positioning Map

| System | Monitors What | Action-Level? | Telemetry? | Runtime Cost |
|--------|--------------|--------------|-----------|-------------|
| Foxglove | Sensor/ROS data | No | Yes (infra) | N/A |
| Sift Stack | Hardware signals | Partial (torque) | Yes (infra) | N/A |
| Code-as-Monitor | Observation constraints | No (obs-space) | No (reactive) | Heavy (VLM) |
| CoVer-VLA | Task alignment | Indirect | No (scoring) | Moderate |
| Do What You Say | Reasoning alignment | Indirect | No (selection) | Heavy (sim) |
| Safety Filters (CBF) | State safety | No (state-space) | No (enforce only) | Low-Moderate |
| AV Telemetry | Driving actions | Yes | Yes (custom) | Custom |
| **SafeContract** | **Action constraints** | **Yes (action-space)** | **Yes (built-in)** | **<50us** |

**SafeContract is the only system that combines action-level monitoring, telemetry output, and sub-50-microsecond enforcement.**

---

## Recommended Framing for Paper

### Title Option
"SafeContract: Composable Safety Contracts with Behavioral Telemetry for Vision-Language-Action Policies"

### Key Claims
1. **First behavioral telemetry system for VLAs** - treats constraint boundary interactions as diagnostic signals, not just enforcement events
2. **Bridges the physical AI deployment gap** (a16z 2026) - provides the action-level observability that robot DevOps is missing
3. **SPC for robot policies** - violation rate tracking as control charts for deployed VLA health
4. **Zero-cost telemetry** - safety enforcement and behavioral telemetry are the same operation

### Related Work Positioning
- Reference UEBA (Gartner 2015) as the conceptual ancestor from cybersecurity
- Reference a16z deployment gap (2026) as the infrastructure motivation
- Contrast with Code-as-Monitor (CVPR 2025) - they monitor observation constraints, we monitor action constraints
- Contrast with CoVer-VLA (2026) and NVIDIA's steering (ICRA 2026) - they verify correctness, we verify safety
- Reference SPC (manufacturing quality) as the statistical framework we inherit
- Reference AV telemetry as the mature domain analog

### Paragraph Draft
"We introduce the concept of behavioral telemetry for VLA policies - continuous monitoring of action-level constraint interactions during deployment. Unlike sensor-level observability platforms (Foxglove, Sift) that monitor hardware signals, and unlike runtime verification systems (CoVer-VLA, Code-as-Monitor) that assess task correctness, behavioral telemetry monitors the physical safety envelope of the policy's outputs. Each contract enforcement event generates telemetry: which constraints fired, how far the raw action exceeded bounds, and how the clipped action differs from the policy's intent. Aggregated over episodes, this telemetry reveals policy health - degradation trends, task-specific weaknesses, and the Pareto frontier between safety and performance. This parallels Statistical Process Control in manufacturing, where control chart limits serve the same role as our safety contract bounds."

---

## References for Paper

1. Hsu, O. (2026). "The Physical AI Deployment Gap." a16z.
2. Zhou et al. (2025). "Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection." CVPR 2025.
3. Wu et al. (2025). "Do What You Say: Steering VLAs via Runtime Reasoning-Action Alignment Verification." ICRA 2026.
4. CoVer-VLA (2026). "Scaling Verification Can Be More Effective than Scaling Policy Learning." arXiv 2602.12281.
5. Wabersich et al. (2024). "The Safety Filter: A Unified View of Safety-Critical Control." Annual Reviews.
6. Cai et al. (2024). "Digital Twin Enabled Runtime Verification for AMRs under Uncertainty." arXiv 2412.09913.
7. Almeida et al. (2024). "OOD Detection and Data Drift Monitoring using SPC." arXiv 2402.08088.
8. Gartner (2015). User and Entity Behavior Analytics (UEBA) - market category definition.
9. RAND Corporation. "Measuring Automated Vehicle Safety: Forging a Framework."
10. TRL (2025). "Automated Vehicle Safety Assurance - In-use Monitoring Framework."
