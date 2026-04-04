# Fresh Research Directions for vla-edge - April 2026

Research-only exploration of 7 underresearched areas beyond safety, quantization, and testing.
Generated 2026-04-01. Each topic rated on Novelty (1-5) and Relevance to vla-edge (1-5).

---

## 1. VLA + Human-Robot Interaction Safety

**State of the art:**
- ISO 10218:2025 was overhauled (Oct 2025) - now incorporates ISO/TS 15066 collaborative requirements directly. Defines safeguarded spaces, force/speed limits, and cybersecurity requirements for the first time.
- Trust research (Frontiers in Organizational Psychology, 2025) shows trust loss is driven by robot failures - specifically design failures, expectation failures, and system failures. Subjective trust shapes interaction more than objective safety metrics.
- Kite Compliance (2026) proposes "Trust Architectures" for humanoid robots - moving beyond safety checklists to systems that build trust over time.
- Nobody has studied how VLA-specific behaviors (action chunking jitter, denoising artifacts, safety clipping corrections) affect human trust perception.

**The gap:**
VLA policies produce qualitatively different motion than traditional controllers. Flow matching VLAs (SmolVLA) produce smooth but sometimes "drifty" trajectories. Autoregressive VLAs (OpenVLA) can produce jerky, token-quantized motions. Safety clipping adds another motion artifact. Zero research exists on how these VLA-specific motion characteristics map to human trust. Does a SafeContract clip (sudden velocity correction) erode trust more than a gradual slow-down?

**Could we fill it?**
Yes, but it requires a user study - expensive and outside our core competency. Better angle: quantify the motion smoothness impact of SafeContract clipping and frame it as a "trust-relevant metric." Publish the metric, let HRI researchers do the user study.

**Novelty: 4/5** - VLA motion artifacts + trust is untouched territory
**Relevance: 2/5** - Requires user studies we can't run; metric contribution is indirect

---

## 2. VLA Failure Mode Analysis at Scale

**State of the art:**
- LIBERO-Plus (arXiv:2510.13626) is THE paper here. 10,030 tasks, 7 perturbation dimensions, 21 sub-dimensions, difficulty levels L1-L5. Key finding: 95% to below 30% under camera viewpoint and robot initial state perturbations. Models are INSENSITIVE to language variations - they literally ignore instructions.
- World Action Models vs VLAs comparison (arXiv:2603.22078) shows world models generalize better than VLAs under distribution shift.
- VLSA/AEGIS (vlsa-aegis.github.io) is closest competitor to SafeContract. Uses Control Barrier Functions (CBFs) + QP solver for plug-and-play safety. Achieves 77.85% collision avoidance rate vs pi0.5's 18.69%. Comes with SafeLIBERO benchmark.

**The gap:**
LIBERO-Plus identifies THAT failures happen under perturbation, but doesn't classify WHY at the action level. Is the model failing at reaching? Grasping? Sequencing? A systematic failure taxonomy at the action phase level (approach, contact, manipulation, release) doesn't exist. We know the perturbation triggers, but not the failure mechanisms.

**Could we fill it?**
Very well. SafeContract already instruments action streams. We could classify violations by task phase and build a "failure fingerprint" - which safety contracts are violated during which task phases under which perturbations. This produces a failure taxonomy FOR FREE from our existing safety infrastructure.

**Novelty: 5/5** - Phase-level failure taxonomy from safety violations is completely new
**Relevance: 5/5** - Directly uses SafeContract, strengthens the paper, differentiates from AEGIS

---

## 3. Multi-Task Safety Contract Transfer

**State of the art:**
- ISO 10218:2025 defines task-level risk assessment but at the industrial cell level, not the learned policy level.
- MPC-based safety (collision avoidance via CBFs, artificial potential fields) is inherently task-agnostic - it operates on geometric constraints.
- AEGIS's CBF approach is also task-agnostic by design (it constrains workspace, not task semantics).
- No research exists on whether safety contracts LEARNED or CALIBRATED on one task transfer to another.

**The gap:**
SafeContract's parameters (velocity limits, workspace bounds, acceleration limits) are currently set per-task. The question nobody has asked: which safety parameters are universal (e.g., "never exceed 1.5 m/s end-effector speed") vs task-specific (e.g., "approach speed during pour must be under 0.3 m/s")? If we calibrate SafeContract on LIBERO "pick" tasks, does it transfer to "stack" tasks? To "drawer" tasks?

**Could we fill it?**
Yes. Run SafeContract with the same parameters across all 4 LIBERO task suites. Measure: (a) violation rates per suite, (b) false positive rates (safe actions clipped unnecessarily), (c) which contract types (velocity vs bounds vs acceleration) transfer and which don't. This gives a "transferability matrix."

**Novelty: 4/5** - Safety contract transfer analysis is new; the concept of universal vs task-specific safety is novel
**Relevance: 5/5** - Direct SafeContract experiment, answers a practical question for users

---

## 4. VLA + Sim-to-Real Gap for Safety

**State of the art:**
- TRANSIC (arXiv:2405.10315) uses human-in-the-loop correction for sim-to-real transfer. Emergent behaviors include error recovery, safety-aware actions, and failure prevention.
- Conformal prediction for robot safety (arXiv:2501.04823) calibrates safety classifiers from sparse human feedback in Gaussian Splat simulators, then transfers to real quadcopters.
- Adaptive conformal prediction + probabilistic CBFs (IEEE 2024) provide high-probability safety guarantees under unknown noise distributions.
- We already researched conformal prediction for SafeContract (docs/research/CONFORMAL_PREDICTION_VLA_SAFETY.md).

**The gap:**
Nobody has measured the "safety sim-to-real gap" specifically. The sim-to-real literature focuses on task success transfer. But safety constraints calibrated in sim (where physics is approximate) may be too tight or too loose in reality. Specifically: if you calibrate SafeContract velocity limits in IsaacSim, are they correct on real hardware? What's the safety calibration error?

**Could we fill it?**
Partially. Without real hardware (Jetson + robot arm), we can't do the full study. But we CAN: (a) calibrate SafeContract in LIBERO sim, (b) add physics randomization and measure how safety parameters need to change, (c) propose a "safety calibration protocol" for sim-to-real. This is a theoretical + sim contribution.

**Novelty: 4/5** - "Safety sim-to-real gap" as a named concept doesn't exist
**Relevance: 3/5** - We lack real hardware for full validation; sim-only analysis is partial

---

## 5. Compositional VLA Systems

**State of the art:**
- Dual Process VLA (DP-VLA, arXiv:2410.15549) - System 1 (fast, small model for motor control) + System 2 (slow, large VLM for reasoning). Figure AI's Helix and NVIDIA Groot N1 both use this pattern in production.
- LeVERB - hierarchical VLA with latent action vocabularies (VLM for planning) + RL control layer (for dynamics). 150+ tasks with sim-to-real transfer.
- RISE (arXiv:2602.11075) - compositional world model with progress value model for self-improvement.
- 164 VLA submissions at ICLR 2026 - hierarchical/compositional is a major trend.

**The gap:**
When you stack two VLA systems (planner + controller), where does safety belong? Options: (a) only at the low-level controller output, (b) only at the high-level planner output, (c) at both levels with different contracts. Nobody has analyzed this. If the planner says "move left" but the controller's safety contract clips the action, does the planner get confused next step? Safety contract interference in hierarchical systems is completely unexplored.

**Could we fill it?**
This is a strong theoretical contribution. We could formalize: "Given a hierarchical VLA with planner P and controller C, and safety contracts S_P and S_C, under what conditions does S_C's clipping invalidate P's assumptions?" This extends our composition theorem (Theorem 2 in the SafeContract paper) to hierarchical architectures.

**Novelty: 5/5** - Safety composition in hierarchical VLA is virgin territory
**Relevance: 4/5** - Extends SafeContract theorems naturally; the field is moving toward hierarchical VLAs fast

---

## 6. Energy Efficiency + Safety Tradeoffs on Edge

**State of the art:**
- Smart actuators with load-adaptive power (2025-2026) adjust power based on sensed load.
- Dual-battery hot-swap for continuous operation is production-ready (EVE robotics, 2025).
- AI-driven battery management predicts failures and adapts charging rates.
- NO research connects safety monitoring overhead to energy consumption on edge devices.

**The gap:**
On a Jetson Orin Nano (15W TDP), SafeContract's 27us overhead per action seems negligible. But what about continuous monitoring at 10Hz with full workspace bounds checking, collision distance computation, and safety logging? Over 8 hours of battery operation, does safety monitoring consume 1% of energy? 5%? Nobody has measured. And the deeper question: can we do adaptive safety monitoring (skip checks when confident) to save energy without compromising safety guarantees?

**Could we fill it?**
Perfectly. We have profiling infrastructure, we have SafeContract, and Jetson power measurement is trivial (tegrastats). Measure: (a) SafeContract energy overhead at various check frequencies, (b) propose "Safety SLOs with Error Budgets" - allow N% of cycles to skip safety checks based on recent violation history, (c) prove the safety bound still holds under adaptive monitoring. This connects to our existing "Safety SLOs with Error Budgets" idea.

**Novelty: 4/5** - Nobody has measured safety monitoring energy cost on edge robots
**Relevance: 5/5** - Directly uses our profiler + SafeContract + Jetson target hardware

---

## 7. VLA Interpretability Through Safety Contracts

**State of the art:**
- XAI for robotics (IJSRA, 2026) identifies 5 paradigms: interpretable-by-design, surrogate models, interpretable monitoring, auxiliary explanations, interpretable safety validation.
- Explainable Monitoring Systems (EMS) validate robot behavior via trajectory deviations and rule violations.
- VLA adversarial vulnerabilities (arXiv:2411.13587) show up to 100% task success reduction from adversarial attacks, but no interpretability of WHY.
- VLA interpretability is called out as an "unresolved challenge" in multiple 2025-2026 surveys.

**The gap:**
Safety violation patterns ARE interpretability signals, but nobody has framed them this way. If a VLA consistently violates joint 3 velocity limits during grasps but not during reaches, that tells you something about the model's internal representations. Violation fingerprints could serve as a FREE interpretability tool - no saliency maps, no attention visualization, no auxiliary networks. Just observe which contracts break, when, and for which inputs.

**Could we fill it?**
Exceptionally well. SafeContract already produces structured violation logs (step, joint, violation type, magnitude). We just need to aggregate and analyze: (a) cluster violation patterns by task, (b) correlate with task phase, (c) show that violation patterns predict task failure before it happens. If violation pattern X at step 10 predicts failure at step 30, that's both interpretability AND early warning.

**Novelty: 5/5** - "Safety violations as interpretability" is a completely novel framing
**Relevance: 5/5** - Zero new infrastructure needed; pure analysis of SafeContract output

---

## Summary Rankings

| # | Topic | Novelty | Relevance | Combined | Effort |
|---|-------|---------|-----------|----------|--------|
| 7 | Interpretability via safety violations | 5 | 5 | 10 | Low |
| 2 | Failure mode taxonomy | 5 | 5 | 10 | Medium |
| 6 | Energy + safety tradeoffs | 4 | 5 | 9 | Medium (needs Jetson) |
| 3 | Multi-task contract transfer | 4 | 5 | 9 | Medium |
| 5 | Compositional VLA safety | 5 | 4 | 9 | Low (theoretical) |
| 4 | Safety sim-to-real gap | 4 | 3 | 7 | High (needs hardware) |
| 1 | HRI trust + VLA motion | 4 | 2 | 6 | High (needs user study) |

---

## Recommended Next Steps

**Immediate (this week):** Topics 7 and 2 are the highest-value, lowest-effort additions to the SafeContract paper. "Safety violations as interpretability" and "phase-level failure taxonomy" can both be demonstrated from the same LIBERO experiment data we're already collecting.

**CoRL 2026 framing:** The SafeContract paper could include a "Safety-Driven Interpretability" section showing violation fingerprints. This differentiates sharply from AEGIS (which focuses on collision avoidance rate, not interpretability).

**Post-CoRL pipeline:**
- Topic 5 (compositional safety) is a standalone theory paper for a workshop
- Topic 6 (energy tradeoffs) becomes the "edge deployment" paper once we have Jetson benchmarks
- Topic 3 (transfer) is a straightforward empirical study across LIBERO suites

**Skip for now:** Topics 1 and 4 require resources we don't have (user studies, real robot hardware).

---

## Key Competitor to Track

**VLSA/AEGIS** (vlsa-aegis.github.io) is the closest competitor to SafeContract. Key differences:
- AEGIS uses CBFs + QP solver (continuous optimization). SafeContract uses contract-based clipping (discrete, faster).
- AEGIS measures collision avoidance rate. We should measure violation patterns, interpretability, transferability.
- AEGIS has SafeLIBERO benchmark. We should use it for direct comparison.
- AEGIS overhead is QP solve time (heavier than our 27us). This is an edge deployment advantage.

---

## Sources

- [LIBERO-Plus: In-depth Robustness Analysis of VLA Models](https://arxiv.org/abs/2510.13626)
- [VLSA: VLA Models with Plug-and-Play Safety Constraint Layer (AEGIS)](https://vlsa-aegis.github.io/)
- [State of VLA Research at ICLR 2026 - Moritz Reuss](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- [Dual Process VLA (DP-VLA)](https://arxiv.org/abs/2410.15549)
- [RISE: Self-Improving Robot Policy with Compositional World Model](https://arxiv.org/abs/2602.11075)
- [Do World Action Models Generalize Better than VLAs?](https://arxiv.org/html/2603.22078v1)
- [Learning Robot Safety from Sparse Human Feedback using Conformal Prediction](https://arxiv.org/html/2501.04823)
- [Conformal Prediction in the Loop: Risk-Aware Control](https://www.bhoxha.com/papers/letters-zhang-2025.pdf)
- [TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction](https://transic-robot.github.io/)
- [Safe and Explainable AI for Safety-Critical Robotic Systems (2026)](https://ijsra.net/content/safe-and-explainable-artificial-intelligence-safety-critical-robotic-systems)
- [Exploring Adversarial Vulnerabilities of VLA Models in Robotics](https://arxiv.org/html/2411.13587v2)
- [Human Trust and Safety Perception in HRI (Frontiers, 2025)](https://www.frontiersin.org/journals/organizational-psychology/articles/10.3389/forgp.2025.1669782/full)
- [ISO 10218-1:2025 - Robotics Safety Requirements](https://www.iso.org/standard/73933.html)
- [How Brittle Are VLA Models in Robotics](https://medium.com/@yananchen1116/how-brittle-are-the-vla-models-in-robotics-66ab85286ecf)
- [Energy-Efficient Robotics: Designing Greener Automation Systems](https://roboticsandautomationnews.com/2026/03/30/energy-efficient-robotics-designing-greener-automation-systems-for-a-power-constrained-future/100240/)
- [Empirical Analysis of Dual-System VLA Models](https://openreview.net/pdf/eb616a4ff2ceea71cc780bbcdc89b3939abe5944.pdf)
