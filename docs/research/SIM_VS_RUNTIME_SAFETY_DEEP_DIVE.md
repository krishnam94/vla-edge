# Can Simulation Alone Guarantee VLA Safety? A Deep Research Report

**Date**: 2026-04-08
**Question**: Is "just do more sim" sufficient, or is runtime monitoring a fundamental necessity for real-world VLA deployment?
**Verdict**: Runtime monitoring is a fundamental necessity. Simulation is necessary but provably insufficient.

---

## Executive Summary

Six lines of evidence converge on a single conclusion: you cannot fully validate a VLA policy's safety in simulation alone. The sim-to-real gap is not a temporary engineering limitation that will eventually be solved - it reflects fundamental epistemological limits about what can be known before deployment. Every mature safety-critical industry (aviation, nuclear, medical devices, automotive) has independently arrived at the same architecture: pre-deployment testing PLUS runtime monitoring. This is not coincidence. It is necessity.

---

## 1. The Sim-to-Real Gap is Structural, Not Just Engineering

### Physics Engine Limitations

Modern simulators (MuJoCo, Isaac Sim, PyBullet) systematically fail to capture:

- **Contact dynamics**: Friction, contact forces, and material properties vary in reality in ways simulators approximate poorly. Simplified geometric approximations (bounding volumes, convex decompositions) replace accurate collision representations.
- **Deformable objects**: The simulation stack is fragmented - separate engines handle rigid bodies, finite element methods for deformables, and smoothed particle hydrodynamics for fluids. Cross-material interaction is poorly modeled and breaks end-to-end differentiability.
- **Manufacturing tolerances and wear**: Real-world factors like backlash, mechanical wear, and actuator aging are rarely modeled, creating reality gaps that cause self-collisions, unstable motions, or failed execution.
- **Sensor noise and latency**: Realistic sensor delays, CAN packet timing, IMU drift, and thread scheduling misses are essentially impossible to simulate faithfully.

**Key finding**: Policies that are safe in sim fail in real life not because friction was 3% off, but because "a CAN packet arrived late, a thread missed its deadline, or an IMU integration drifted just enough to destabilize the controller." Real-world performance drops of 24-30% are documented for policies transferred directly from high-fidelity simulators.

### What Domain Randomization Cannot Fix

Domain randomization (DR) improves robustness by exposing policies to randomized dynamics during training. But DR only covers the distribution of variations the engineer thought to randomize. It cannot cover:

- Unknown unknowns (failure modes never conceived)
- Emergent multi-body interactions
- Human behavioral patterns in shared workspaces
- Environmental conditions outside the randomization envelope

**Source**: [NVIDIA Isaac Lab Blog](https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/), [Reality Gap in Robotics (UZH)](https://rpg.ifi.uzh.ch/docs/AR25_Aljalbout.pdf), [Sim-to-Real Gap TIAGo](https://arxiv.org/html/2403.07091v2)

---

## 2. Formal Verification of Neural Policies is Intractable at Scale

### The Scalability Wall

Formal verification of neural network controllers faces fundamental computational barriers:

- **Discrete abstractions** suffer from combinatorial explosions in complexity
- **Hamilton-Jacobi reachability** costs scale exponentially with state space dimensionality
- **SMT-based methods** (Satisfiability Modulo Theories) are "either not scalable to deep neural networks or are highly conservative"
- **Set-propagation methods** are "fundamentally limited by their reliance on specific formats of highly accurate and deterministic system models" and demonstrate success only in "small, simulated robotic domains"

VLA models (like OpenVLA, RT-2, pi0) have millions to billions of parameters, process high-dimensional visual inputs, and produce continuous action outputs. They are many orders of magnitude beyond what formal verification can currently handle.

### Verification Gaps Even When Attempted

Even over sampled datasets, strict satisfaction guarantees are not achieved. Positive loss terms can be canceled by larger negative terms, offering "neither empirical nor formal guarantees on validity." Methods using reward functions or exploration constraints can minimize undesirable behaviors but cannot provide formal guarantees.

### Conformal Prediction as a Bridge

Conformal prediction has emerged as a practical middle ground - providing PAC-like (Probably Approximately Correct) guarantees that are distribution-free. But crucially, conformal prediction provides **runtime** guarantees by monitoring deployment-time behavior, not pre-deployment certification. It is itself a form of runtime monitoring.

**Source**: [Formal Methods in Robot Policy Learning Survey](https://arxiv.org/html/2602.06971), [Formal Verification of NN Controlled Autonomous Systems](https://arxiv.org/abs/1810.13072), [Conformal Safety Monitoring](https://arxiv.org/html/2501.04823)

---

## 3. Distribution Shift at Deployment is Inevitable

### The Core Argument

Amodei et al. (2016) identified distributional shift as one of five concrete problems in AI safety: systems make "bad decisions (particularly silent and unpredictable bad decisions) when given inputs that are potentially very different than what was seen during training."

For VLA models specifically, deployment-time reliability research establishes that "regardless of training data scale or model sophistication, deployed systems will eventually encounter situations that lead to erroneous or undesirable behavior." This is not a solvable problem - it is a property of open-world deployment.

### What Shifts and Why

Specific VLA vulnerabilities documented in recent research:

- **Object geometry**: Novel shapes not in training data
- **Lighting variations**: Shadows, reflections, time-of-day changes
- **Scene context**: Background clutter, workspace rearrangement
- **Instruction phrasing**: Natural language variation
- **Spurious features**: The policy latches onto irrelevant visual correlations
- **Human behavior**: Unpredictable proximity, movement patterns, interventions
- **Compounding errors**: Small deviations accumulate over long-horizon tasks

### Two Failure Modes at Deployment

Research identifies two complementary failure categories for generative robot policies:

1. **Erratic failures**: Temporal inconsistency - the policy produces conflicting action modes across timesteps, leading to collisions or chaotic behavior
2. **Task progression failures**: The policy remains temporally consistent but fails to advance toward goals - stalling, drifting, or producing outputs disconnected from task objectives

Both are invisible to pre-deployment testing because they emerge from the specific distribution encountered at runtime.

**Source**: [Deployment-Time Reliability of Learned Robot Policies](https://arxiv.org/html/2603.11400), [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565), [Sentinel - Failure Modes of Generative Policies](https://proceedings.mlr.press/v270/agia25a.html)

---

## 4. The Statistical Impossibility Argument

### The RAND Corporation Analysis

The landmark RAND study by Nidhi Kalra and Susan Paddock demonstrated that proving autonomous vehicle safety through testing alone requires astronomical numbers:

- To demonstrate a 20% improvement over human fatality rates with 95% confidence: **~8 billion kilometers (5 billion miles)** of driving
- Under aggressive testing assumptions, existing fleets would need "tens and even hundreds of years" to log sufficient miles

This analysis applies directly to robot manipulation. If a VLA policy fails at a rate of 1 in 10,000 episodes, you need on the order of 30,000 episodes just to observe 3 failures (barely enough for any statistical analysis). For safety-critical failure modes that occur at 1 in 1,000,000 - the "black swan" events - pre-deployment testing is mathematically inadequate.

### Black Swan Events

Neural networks are "extremely surprised by rare events" because they "tend to produce similar predictions for points outside the training distribution." The rarity of catastrophic failures means there is little data for predictive models, and these events "often appear to arise from an unknown set of physical conditions, making predicting them analogous to finding a needle in an infinite-dimensional haystack."

**Source**: [RAND Driving to Safety Report](https://www.rand.org/pubs/research_reports/RR1478.html), [Black-Box Rare-Event Simulation](https://link.springer.com/article/10.1007/s40305-025-00585-0)

---

## 5. Real-World Failure Case Studies

### Uber ATG Fatal Crash (2018)

**What happened**: On March 18, 2018, Elaine Herzberg was struck and killed by a prototype Uber self-driving Volvo XC90 in Tempe, Arizona.

**What testing missed**:
- The perception system classified Herzberg as "unknown object," then "vehicle," then "bicycle" over 6 seconds - each with a different predicted path
- The system could not classify pedestrians unless they were near a crosswalk
- Emergency braking was disabled during autonomous operation "to reduce potential for erratic vehicle behavior"
- Uber had reduced safety staffing from two operators to one

**The monitoring lesson**: A runtime safety monitor that could have triggered emergency braking regardless of classification state would have prevented the fatality. The system had 1.3 seconds of warning before impact but could not act because the safety layer was disabled.

**Source**: [Wikipedia - Death of Elaine Herzberg](https://en.wikipedia.org/wiki/Death_of_Elaine_Herzberg), [IEEE Spectrum NTSB Investigation](https://spectrum.ieee.org/ntsb-investigation-into-deadly-uber-selfdriving-car-crash-reveals-lax-attitude-toward-safety)

### Cruise Robotaxi Pedestrian Dragging (2023)

**What happened**: On October 2, 2023, a Cruise autonomous vehicle in San Francisco ran over a pedestrian who had been knocked into its path by another vehicle, then dragged her approximately 20 feet while attempting to pull over.

**What testing missed**:
- The vehicle stopped after initial contact but "didn't detect her under the car"
- The pullover maneuver then dragged the victim
- Cruise initially hid this portion of the video from the DMV

**The monitoring lesson**: A runtime monitor detecting anomalous forces/drag during pullover maneuvers, or a sensor confirming clearance before vehicle movement, would have prevented the secondary injury.

**Source**: [DMV Suspension Report](https://sfstandard.com/2023/10/24/cruise-robotaxi-suspended-dmv-san-francisco/), [Carscoops](https://www.carscoops.com/2023/10/california-dmv-suspends-cruise-permits-after-robotaxi-ran-over-and-dragged-pedestrian/)

### Tesla Autopilot Phantom Braking

**What happened**: Persistent phantom braking events across the Tesla fleet, where vehicles brake suddenly with no obstacle present.

**What testing missed**:
- Occurs during "normal use" of Autopilot, "even when the system is functioning as intended"
- Camera-only system fails on unusual shadows, road markings, overhead signs
- In Mark Rober's test, Tesla drove into a fake road wall while lidar-equipped vehicles avoided it; also failed to stop for a child mannequin in fog and heavy rain
- German court ruled Autopilot "defective" and "not suitable for normal use"

**The monitoring lesson**: Long-tail visual edge cases are essentially infinite. No amount of pre-deployment testing can cover all combinations of lighting, weather, road conditions, and visual artifacts.

**Source**: [Electrek - Tesla Phantom Braking](https://electrek.co/2021/11/15/tesla-serious-phantom-braking-problem-autopilot/), [Carscoops German Court Ruling](https://www.carscoops.com/2025/02/german-court-finds-teslas-autopilot-defective-after-lawsuit/)

### Amazon Warehouse Robot Injuries

**What happened**: Amazon facilities with robots have higher injury rates than those without. OSHA investigations found "willful" violations for ergonomic hazards.

**The monitoring lesson**: Robots that are "safe" in isolation create emergent safety hazards when combined with human workers under production pressure. The interaction dynamics are not captured in any simulation.

**Source**: [Amazon Robot Safety - OnLabor](https://onlabor.org/amazons-approach-to-robotics-is-seriously-injuring-warehouse-workers/), [OSHA Federal Safety Inspections](https://www.osha.gov/news/newsreleases/national/01182023)

### Waymo/Cruise Fleet-Wide Software Bugs

Systemic software errors can propagate across fleets - Waymo experienced barrier collisions and Cruise had hard braking across multiple vehicles simultaneously. These are correlated failures that no amount of single-vehicle simulation can predict.

**Source**: [Waymo Safety Case](https://assets.ctfassets.net/e6t5diu0txbw/66jOjPtNIjzawaK0ZjpU3q/7f081b392cf29a3355c97d0d758fe6cf/Waymo_Safety_Case_Approach.pdf)

---

## 6. The Defense-in-Depth Argument from Mature Industries

### Every Safety-Critical Industry Uses Runtime Monitoring

No mature safety-critical industry relies solely on pre-deployment testing. The pattern is universal:

| Industry | Pre-Deployment | Runtime Monitoring | Standard |
|----------|---------------|-------------------|----------|
| Nuclear | Design basis analysis, testing | 5-layer defense in depth, continuous monitoring | IAEA INSAG-10 |
| Aviation | Certification, flight testing | TCAS, stall warning, GPWS, flight recorders | DO-178C, ASTM F3269 |
| Medical devices | 510(k)/PMA premarket review | Post-market surveillance, adverse event reporting | FDA 21 CFR 803 |
| Automotive | NCAP testing, type approval | ABS, ESC, airbag monitors, OBD diagnostics | IEC 61508, ISO 26262 |
| Self-driving | Simulation, road testing | ODD monitors, fallback systems, remote operators | UL 4600 |

### The Nuclear Model: Defense in Depth

The IAEA defense-in-depth framework defines five independent layers:

1. **Prevention**: Design to prevent deviations from normal operation
2. **Detection and control**: Detect deviations early and correct them
3. **Safety systems**: Activate specific safety systems if detection/control fails
4. **Accident management**: Limit accident progression if safety systems fail
5. **Consequence mitigation**: Mitigate radiological consequences if containment is breached

Layers 2-5 are all forms of runtime monitoring. The key principle: "no single layer is exclusively relied upon." A single barrier is inherently fragile - one component failure, human error, or design flaw can cascade into catastrophe.

**Source**: [IAEA Defense in Depth](https://www-pub.iaea.org/MTCD/publications/PDF/Pub1013e_web.pdf), [NRC Defense in Depth](https://www.nrc.gov/reading-rm/basic-ref/glossary/defense-in-depth), [Defense in Depth Principles](https://risk-engineering.org/concept/defence-in-depth)

### Aviation: TCAS as the Runtime Monitoring Exemplar

TCAS (Traffic Collision Avoidance System) operates independently of air traffic control, monitoring airspace around the aircraft in real-time. It is the canonical example of an independent safety monitor:

- Operates even when primary navigation systems fail
- Can override pilot/ATC instructions when collision is imminent
- Has a priority hierarchy with other safety monitors (stall warning, GPWS take precedence)

This is exactly the Simplex Architecture applied to aviation decades before the term existed.

**Source**: [TCAS Wikipedia](https://en.wikipedia.org/wiki/Traffic_collision_avoidance_system), [ASTM F3269 Standard](https://store.astm.org/f3269-21.html)

### The Simplex Architecture for Robot Safety

ASTM F3269 codifies the Runtime Assurance (RTA) concept: use a high-assurance control subsystem (Recovery Function) to protect a high-performance control subsystem (Complex Function). This is "using simplicity to control complexity" (Sha 2001).

The architecture has three components:
1. **Advanced Controller**: The unverified VLA policy (high performance, not safety-guaranteed)
2. **Baseline Controller**: A verified safe fallback (lower performance, provably safe)
3. **Decision Module**: A safety monitor that switches from advanced to baseline when safety violations are imminent

This is directly applicable to VLA deployment: the VLA policy runs as the advanced controller, with a simple verified controller (e.g., stop + retract) as baseline, and a lightweight monitor deciding when to switch.

**Source**: [Black-Box Simplex Architecture](https://par.nsf.gov/servlets/purl/10327769), [ASTM F3269 RTA Standard](https://arc.aiaa.org/doi/abs/10.2514/6.2021-0525)

### Medical Devices: Post-Market Surveillance

FDA data shows that premarket issues account for about 45% of 510(k) device recalls - meaning 55% of problems emerge only after deployment. The FDA mandates post-market surveillance precisely because "premarket testing alone is insufficient to identify all safety concerns."

### IEC 61508: Functional Safety Requires Runtime Diagnostics

IEC 61508 - the foundational functional safety standard - explicitly requires runtime diagnostics including watchdogs, hardware diagnostics, and fault injection testing. Safety integrity levels (SIL 1-4) specify required diagnostic coverage at runtime. The standard covers an end-to-end lifecycle from concept through decommissioning, with operation-phase monitoring as a mandatory component.

**Source**: [IEC 61508 Wikipedia](https://en.wikipedia.org/wiki/IEC_61508), [FDA Post-Market Surveillance](https://www.ncbi.nlm.nih.gov/books/NBK209652/)

---

## 7. Academic Arguments for Runtime Monitoring Necessity

### "You Cannot Test/Inspect Quality Into a Product"

This principle, attributed to W. Edwards Deming, is foundational in quality engineering. Testing reveals defects but cannot create quality. Applied to safety: testing reveals unsafe behaviors but cannot create safety. Safety must be designed in AND monitored continuously.

**Source**: [You Can't Inspect Quality Into a Product](https://developsense.com/blog/2025/04/you-cant-inspect-quality-into-a-product)

### Runtime Verification: The Theoretical Foundation

Runtime verification occupies a strategic middle ground between testing and formal verification:
- More principled than testing (uses formal specifications, not ad-hoc test cases)
- More scalable than formal verification (analyzes actual execution traces, not all possible traces)
- Addresses a gap neither can fill: "Not all errors can be detected during development"
- Enables recovery through "reconfiguration, micro-resets, or finer intervention mechanisms"

**Source**: [Runtime Verification (Wikipedia)](https://en.wikipedia.org/wiki/Runtime_verification)

### VLA-Specific Safety Frameworks

Two recent frameworks explicitly address VLA safety through runtime mechanisms:

1. **SafeVLA** (2025): Uses constrained Markov decision processes with safe RL. Achieved 83.58% reduction in safety violations while maintaining task performance. Identifies "harm to the environment, the robot itself, and humans" as three threat categories.

2. **VLSA/AEGIS** (2025): Plug-and-play safety constraint layer using control barrier functions. Integrates directly with existing VLA models. Activates "only when potential safety violations are detected" - a runtime mechanism by definition.

Both frameworks are runtime monitoring systems, confirming that even VLA-specific safety research concludes pre-deployment training alone is insufficient.

**Source**: [SafeVLA](https://arxiv.org/abs/2503.03480), [VLSA](https://arxiv.org/pdf/2512.11891)

### Sentinel: Runtime Failure Detection for Generative Policies

The Sentinel framework (Agia et al., 2025) combines statistical temporal action consistency (STAC) monitoring with VLM-based task progress evaluation. Key finding: combining complementary runtime detectors caught "18% more failures than using either detector alone." The paper's core argument is direct: "Robot behavior policies trained via imitation learning are prone to failure under conditions that deviate from their training data" - making deployment-time monitoring essential.

**Source**: [Sentinel](https://proceedings.mlr.press/v270/agia25a.html)

### Conformal Prediction for Runtime Safety Guarantees

Multiple recent papers use conformal prediction for deployment-time safety:

- **FAIL-Detect**: Uses CP for uncertainty quantification with statistical guarantees, detecting failures without requiring failure data
- **FIPER**: Predicts impending failures by combining OOD observation detection with action uncertainty, calibrated via conformal prediction
- **Adaptive Conformal Prediction**: Provides probabilistic prediction guarantees even on OOD data, with incremental learning preventing overly conservative predictions

All of these are runtime monitoring techniques. Conformal prediction's value is precisely that it provides statistical guarantees during deployment, not before it.

**Source**: [FAIL-Detect (RSS 2025)](https://roboticsconference.org/program/papers/73/), [FIPER](https://arxiv.org/abs/2510.09459), [Safety Monitoring for CPS in OOD](https://arxiv.org/html/2504.13478)

---

## 8. The Synthesis: Why Runtime Monitoring is Fundamental

### Seven Independent Arguments

1. **Physics gap**: Simulators cannot faithfully reproduce all physical phenomena (contact, deformables, wear, timing). Policies that pass sim testing fail in reality.

2. **Formal verification intractability**: Neural network verification does not scale to VLA-class models. You cannot prove safety mathematically for these systems.

3. **Statistical impossibility**: The number of test episodes needed for statistical confidence in rare failure rates exceeds practical limits by orders of magnitude (RAND analysis).

4. **Distribution shift inevitability**: Open-world deployment guarantees encountering situations outside any training/testing distribution. This is a property of the world, not a limitation of the engineer.

5. **Black swan events**: Catastrophic failures arise from unknown conditions - "finding a needle in an infinite-dimensional haystack." No finite amount of simulation covers infinite failure modes.

6. **Industry consensus**: Every mature safety-critical industry independently arrived at defense-in-depth with mandatory runtime monitoring. This convergent evolution reflects fundamental necessity, not convention.

7. **Empirical evidence**: Real-world incidents (Uber ATG, Cruise, Tesla, Amazon) demonstrate that pre-deployment testing missed failure modes that runtime monitoring could have caught or mitigated.

### The Correct Architecture for VLA Deployment

```
Pre-deployment (Necessary but Insufficient):
  - Simulation testing with domain randomization
  - Formal verification of safety-critical subsystems (where tractable)
  - Conformal calibration on held-out distributions
  - Hardware-in-the-loop testing

Runtime Monitoring (The Missing Piece):
  - Action consistency monitoring (STAC-style)
  - Task progress monitoring (VLM-based)
  - OOD detection (embedding space, conformal prediction)
  - Physical constraint enforcement (force limits, workspace bounds)
  - Safety constraint layers (control barrier functions)
  - Simplex switching to verified safe fallback controllers
  - Human oversight integration for high-risk scenarios
```

### The Bottom Line

"Just do more sim" is the robotics equivalent of "just do more testing" in software - a strategy that every mature engineering discipline has learned the hard way is necessary but not sufficient. The question is not whether you need runtime monitoring, but how to implement it efficiently without destroying the performance benefits that VLA models provide.

---

## Key References

### Sim-to-Real Gap
- [NVIDIA Isaac Lab Blog](https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/)
- [Reality Gap in Robotics (UZH)](https://rpg.ifi.uzh.ch/docs/AR25_Aljalbout.pdf)
- [What Went Wrong? Closing the Sim-to-Real Gap](https://proceedings.mlr.press/v229/huang23c/huang23c.pdf)

### Formal Verification
- [Formal Methods in Robot Policy Learning Survey (2025)](https://arxiv.org/html/2602.06971)
- [Formal Verification of NN Controlled Autonomous Systems](https://arxiv.org/abs/1810.13072)
- [Formal Verification for Safety-Critical Tasks](https://proceedings.mlr.press/v161/corsi21a/corsi21a.pdf)

### Distribution Shift and Deployment Reliability
- [Deployment-Time Reliability of Learned Robot Policies (2026)](https://arxiv.org/html/2603.11400)
- [Concrete Problems in AI Safety (Amodei et al. 2016)](https://arxiv.org/abs/1606.06565)
- [Sentinel - Failure Modes of Generative Policies](https://proceedings.mlr.press/v270/agia25a.html)

### Statistical Arguments
- [RAND: Driving to Safety](https://www.rand.org/pubs/research_reports/RR1478.html)
- [Black-Box Rare-Event Simulation](https://link.springer.com/article/10.1007/s40305-025-00585-0)

### Runtime Assurance Standards
- [ASTM F3269 - RTA for Aircraft Systems](https://arc.aiaa.org/doi/abs/10.2514/6.2021-0525)
- [Black-Box Simplex Architecture](https://par.nsf.gov/servlets/purl/10327769)
- [IEC 61508 Functional Safety](https://en.wikipedia.org/wiki/IEC_61508)

### VLA Safety
- [SafeVLA (2025)](https://arxiv.org/abs/2503.03480)
- [VLSA Safety Constraint Layer (2025)](https://arxiv.org/pdf/2512.11891)
- [VLA Models: Concepts, Progress, Challenges](https://arxiv.org/html/2505.04769v2)

### Conformal Prediction for Runtime Safety
- [FAIL-Detect (RSS 2025)](https://roboticsconference.org/program/papers/73/)
- [FIPER - Failure Prediction at Runtime](https://arxiv.org/abs/2510.09459)
- [Safety Monitoring for CPS in OOD Scenarios](https://arxiv.org/html/2504.13478)
- [Conformal Safety Monitoring for Flight Testing](https://arxiv.org/html/2511.20811)

### Real-World Incidents
- [Death of Elaine Herzberg (Wikipedia)](https://en.wikipedia.org/wiki/Death_of_Elaine_Herzberg)
- [Cruise DMV Suspension](https://sfstandard.com/2023/10/24/cruise-robotaxi-suspended-dmv-san-francisco/)
- [Tesla Phantom Braking](https://electrek.co/2021/11/15/tesla-serious-phantom-braking-problem-autopilot/)
- [Amazon Warehouse Robot Injuries](https://onlabor.org/amazons-approach-to-robotics-is-seriously-injuring-warehouse-workers/)

### Defense in Depth
- [IAEA Defense in Depth (INSAG-10)](https://www-pub.iaea.org/MTCD/publications/PDF/Pub1013e_web.pdf)
- [Defense in Depth Principles](https://risk-engineering.org/concept/defence-in-depth)
- [TCAS (Wikipedia)](https://en.wikipedia.org/wiki/Traffic_collision_avoidance_system)
- [Runtime Verification (Wikipedia)](https://en.wikipedia.org/wiki/Runtime_verification)
