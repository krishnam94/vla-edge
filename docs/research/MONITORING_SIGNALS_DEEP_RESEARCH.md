# Comprehensive Monitoring Signals for VLA/Robot Policy Deployment

*Deep research - April 2026*

## Current Baseline (vla-edge SafeContract)

Our 5 existing monitors are **all action-space, black-box**:

1. **ActionHealthMonitor** - clip magnitude, z-scores, crest factor, EWMA violation trend
2. **StallDetector** - consecutive low-movement detection (Diffusion freeze)
3. **JerkMonitor** - 3rd derivative, chunk boundary discontinuities
4. **GripperOscillationDetector** - rapid open/close cycling
5. **ConformalActionMonitor / ACAM** - p-values, KS test, CUSUM change detection

Plus STL temporal logic specs (bounds_health, velocity_health, recovery, sustained_health, smoothness_trend).

**Gap**: Everything above watches action outputs only. No observation quality, no timing, no hardware health, no internal model signals, no environment state.

---

## TIER 1: HIGH NOVELTY x HIGH VALUE x IMPLEMENTABLE NOW

### 1. Inference Latency & Control Loop Timing

**What**: Monitor wall-clock time of each policy inference call. Flag when inference exceeds the control deadline (e.g., >100ms for 10Hz control). Track jitter (variance in inference time), not just mean.

**Access**: Black-box (just wraps the inference call with timers).

**Prior art**: Extensive in industrial robotics (ISO 15066 SSM requires real-time response). AsyncVLA (arxiv 2602.13476), VLASH (arxiv 2512.01031), VLA-Perf (arxiv 2602.18397) all identify latency as critical. Edge AI deployments show 15-45ms response times prevent 70-85% of safety incidents that cloud-based (200-600ms) systems miss. Manufacturing control systems treat burst tasks that cause deadline misses as safety hazards.

**Applied to VLA?**: VLA-Perf benchmarks inference speed but doesn't **monitor** it at runtime. TIDAL (arxiv 2601.14945) proposes temporally interleaved inference but doesn't alarm on deadline misses. **Nobody monitors this as a safety signal.**

**Complexity**: Trivial. `time.perf_counter()` around inference. 5 lines of code.

**Catches what we miss**: A VLA that suddenly takes 3x longer (GPU thermal throttling, memory pressure, GC pause) will produce stale actions. Our action monitors see nothing wrong with the actions themselves.

**Rank**: 10/10. Free to implement, universal value, nobody does it for VLAs.

### 2. Action-Observation Consistency (Forward Model Check)

**What**: After executing an action, predict what the next observation *should* look like, then compare to the actual observation. Large prediction error = something unexpected happened (collision, object slipped, environment changed).

**Access**: Needs environment access (next observation after action execution).

**Prior art**: Sentinel (arxiv 2410.04640, RSS 2024) monitors "task progress" via VLM video QA. FIPER (arxiv 2510.09459, NeurIPS 2025) uses OOD detection in the policy's embedding space. World model-based approaches (arxiv 2506.00613, 2602.16182) use prediction error as anomaly signal. F1 model (arxiv 2509.06951) bridges flow matching with explicit world modeling.

**Applied to VLA?**: Sentinel does this indirectly via VLM "is the task progressing?" queries. Not directly measured as prediction error. VLA-in-the-loop (OpenReview) uses a world model to detect unviable actions but doesn't monitor observation consistency post-execution.

**Complexity**: Medium-high. Needs a lightweight world model or observation predictor (could be as simple as optical flow magnitude comparison).

**Catches what we miss**: Robot pushes an object but it doesn't move (stuck). Actions look fine, observations reveal failure.

**Rank**: 9/10. Critical for closed-loop, but needs observation access.

### 3. Observation Quality / Sensor Health

**What**: Monitor the quality of input observations before they reach the policy. Track image sharpness (gradient magnitude), brightness, noise level, entropy, and frame-to-frame consistency. Detect camera occlusion, defocus, overexposure, or signal freezing.

**Access**: Needs observation access (raw images/sensor data).

**Prior art**: Obsurver.com - commercial sensor degradation monitoring for autonomous systems. Camera exposure control for robot vision (arxiv 1907.12646) measures gradient, entropy, noise. Nuclear robot vision research tracks gamma-induced image degradation. Statistical-Entropy framework for embedded robotic sensor fault detection (ResearchGate 398528252) uses Shannon entropy for drift/spike/freeze detection. NIST robot calibration research tracks accuracy degradation via fiducial markers.

**Applied to VLA?**: VLA robustness against sensor attacks (ACM 2025) tests adversarial perturbations but doesn't propose runtime monitoring. Nobody monitors observation quality as part of VLA deployment.

**Complexity**: Low-medium. Image quality metrics (Laplacian variance for blur, histogram stats for exposure) are well-established. Frozen frame detection = trivial.

**Catches what we miss**: A dirty camera lens, USB bandwidth issue, or lighting change will cause the VLA to receive garbage input. Our action monitors only see the symptoms (bad actions) after the damage is done.

**Rank**: 9/10. Cheap, high impact, completely absent from VLA tooling.

### 4. Token-Level Entropy / Per-Step Uncertainty (Autoregressive VLAs)

**What**: For autoregressive VLAs (OpenVLA, pi0-FAST), extract per-token entropy and perplexity during action token generation. Use max-pooled sliding windows rather than mean aggregation (mean dilutes safety-critical spikes).

**Access**: Needs model internals (logits/probabilities during decoding).

**Prior art**: CriticalUQ (arxiv 2603.18342) proposes sliding window + action transfer reweighting + DoF-adaptive calibration. Achieves 0.936 AUROC on LIBERO vs 0.845 baseline. Token-Level Uncertainty (OpenReview NX0euXAv98) uses entropy + perplexity on pi0-FAST. UPS (arxiv 2602.22474) maps VLM verifier uncertainty into execute/clarify/retrain decisions, calibrated via conformal prediction.

**Applied to VLA?**: YES - active research area (2025-2026). CriticalUQ, FIPER, UPS all do this. But none are in deployment toolkits. Key insight: mean entropy is useless (0.47-0.51 AUROC, essentially random). You MUST use temporal windowed max-pooling.

**Complexity**: Medium. Requires access to model logits during autoregressive decoding. Not applicable to flow-matching VLAs (SmolVLA, pi0).

**Catches what we miss**: Policy internally uncertain but outputs plausible-looking actions that happen to be wrong. Our monitors see the action as "within bounds" but the model was guessing.

**Rank**: 8/10. Powerful but architecture-specific (autoregressive only).

### 5. Action Chunk Staleness / Temporal Misalignment

**What**: When using asynchronous inference with action chunks, monitor how "stale" the currently executing action chunk is relative to the latest observation. Track the temporal gap between observation capture and action execution.

**Access**: Black-box (timestamps only).

**Prior art**: AsyncVLA (arxiv 2602.13476) decouples semantic reasoning from reactive execution. VLASH (arxiv 2512.01031) rolls robot state forward to compensate for staleness. RTC (arxiv 2506.07339) generates next chunk while executing current. "Leave No Observation Behind" (arxiv 2509.23224) corrects action chunks in real-time. All identify staleness as degrading action quality.

**Applied to VLA?**: These papers propose solutions to staleness but don't propose monitoring staleness as a deployment health signal with alarms.

**Complexity**: Trivial. Track observation timestamp, inference start/end, execution timestamp. Compute delta.

**Catches what we miss**: Action chunk is 500ms stale because inference took too long. Actions are internally consistent but temporally wrong.

**Rank**: 8/10. Especially critical for edge deployment where inference is slow.

---

## TIER 2: HIGH VALUE, MODERATE NOVELTY

### 6. Attention Pattern Degradation (VLA Internals)

**What**: Monitor cross-attention patterns in the VLA. Track whether action tokens attend to relevant image regions. Detect attention entropy collapse (model stops looking at scene) or attention dispersion (model can't decide what to look at).

**Access**: Needs model internals (attention weights).

**Prior art**: "VLA Knows Its Limits" (arxiv 2602.21445) discovers that intra-chunk actions attend invariantly to vision-language tokens - the model doesn't adapt attention within a chunk. AutoHorizon uses attention weight patterns to determine how many actions in a chunk are reliable. Uses entropy-filtered attention rows and "radial action sinks" as confidence indicators.

**Applied to VLA?**: YES - AutoHorizon directly monitors attention for adaptive execution horizon. But framed as "how many steps to execute" rather than "is the policy degrading."

**Complexity**: Medium. Extracting attention maps requires model hooks. Entropy computation is cheap.

**Catches what we miss**: Model is attending to irrelevant background (e.g., TV screen reflection) while producing actions that pass bounds checks.

**Rank**: 7/10. Requires model access but provides deep interpretability.

### 7. Embedding Space OOD Detection

**What**: Track the policy's internal embedding of current observations. Compare to embeddings from calibration/training data using distance metrics (Mahalanobis, cosine similarity) or learned detectors (Random Network Distillation).

**Access**: Needs model internals (intermediate embeddings).

**Prior art**: FIPER (NeurIPS 2025) uses Random Network Distillation in the policy's embedding space for OOD detection. DriftLens uses distribution distances in embeddings for unsupervised drift detection. Evidently AI documents 5 methods for embedding drift detection. Key insight from FIPER: OOD alone has false positives (benign OOD exists). Must combine with action uncertainty.

**Applied to VLA?**: YES - FIPER directly applies this. Calibrated via conformal prediction with only successful rollouts. NeurIPS 2025 publication.

**Complexity**: Medium. Requires storing calibration embeddings and computing distances. RND adds a small network overhead.

**Catches what we miss**: Scene looks completely different from training (new room, new lighting, new objects) but actions happen to stay within bounds.

**Rank**: 7/10. Well-validated but needs model access.

### 8. Action-Chunk Entropy (Diffusion/Flow VLAs)

**What**: For diffusion/flow-matching VLAs that output action chunks, measure the entropy or diversity of actions within a chunk. High intra-chunk variance = model is uncertain about the trajectory.

**Access**: Needs model access (sample multiple action chunks from the same observation).

**Prior art**: FIPER introduces "action-chunk entropy score" - sample multiple action chunks from the same observation and measure their disagreement. CriticalUQ uses per-DoF entropy with DoF-adaptive weights. UPS samples multiple actions from a policy and uses a VLM verifier to score them.

**Applied to VLA?**: YES - FIPER (NeurIPS 2025). But requires multiple forward passes per step (expensive on edge).

**Complexity**: High for edge deployment. Need 5-10 forward passes to estimate chunk entropy. Could be approximated with MC dropout or reduced sampling.

**Catches what we miss**: Policy produces one action chunk that looks fine, but would produce wildly different chunks if sampled again. Our monitors see one sample and can't tell it was unreliable.

**Rank**: 6/10. Powerful signal but expensive to compute on edge.

### 9. Task Progress Estimation

**What**: Periodically assess whether the robot is making progress toward the task goal. Can use a VLM (Sentinel approach) or simpler heuristics (distance to goal position, gripper-to-object distance).

**Access**: Needs environment access + task specification.

**Prior art**: Sentinel (arxiv 2410.04640) uses VLM video QA for task progress monitoring. SOLE-R1 uses video-language reasoning for per-timestep progress estimation. VLM-TAMP decomposes long-horizon tasks into subgoals for progress tracking. SC-VLA detects and recovers from failure conditions.

**Applied to VLA?**: YES - Sentinel directly. But requires cloud-based VLM inference (slow, expensive).

**Complexity**: High if using VLM. Low if using simple heuristics (requires task-specific implementation).

**Catches what we miss**: Robot is executing smooth, valid actions but going in circles / not making task progress. Classic "Diffusion stall" that's different from micro-movement stall.

**Rank**: 7/10. Critical for long-horizon tasks but implementation varies widely.

---

## TIER 3: CROSS-DOMAIN INSPIRATION (NOVEL APPLICATIONS TO VLA)

### 10. Motor Current / Torque Anomaly Detection

**What**: Monitor joint motor currents and compare to expected torque profiles. Unexpected current spikes = collision. Gradual current increase = mechanical degradation. Current-position mismatch = payload changed.

**Access**: Needs hardware access (motor current sensors).

**Prior art**: Universal in industrial robotics. FANUC ZDT monitors motors, reducers, mechanical parts via cloud analytics. ABB integrates AI vibration analysis for movement deviation prediction. KUKA Connect tracks motor temperatures and CPU load. Research on virtual torque sensing from motor current (no additional sensors needed). Current-based collision detection achieves 0.05N sensitivity.

**Applied to VLA?**: NO. VLA research treats the robot as a black box below the policy. Nobody monitors motor health as part of policy deployment.

**Complexity**: Low if robot API exposes current readings (UR robots, Franka do). Medium otherwise.

**Catches what we miss**: Robot arm is degrading mechanically (worn gearbox). Policy outputs correct actions but physical execution is drifting. Our monitors can't see this at all.

**Rank**: 7/10. Huge practical value but requires hardware integration.

### 11. Vibration / Acoustic Signature Monitoring

**What**: Track vibration frequencies from robot joints. FFT analysis reveals bearing wear (characteristic defect frequencies: BPFO, BPFI), gear mesh issues, and resonance problems. Acoustic monitoring catches unusual sounds.

**Access**: Needs hardware access (accelerometers or microphone).

**Prior art**: Standard in industrial predictive maintenance. ABB reduced unplanned downtime 30% with AI vibration analysis. Three harmonics of bearing defect frequency = replacement needed. IEEE DataPort has "Comprehensive Dynamic Stability Dataset" for ABB, FANUC, KUKA, UR5.

**Applied to VLA?**: NO. Academic VLA research ignores mechanical health entirely.

**Complexity**: Medium-high. Requires accelerometers and FFT pipeline. Some robots (Franka) have built-in vibration sensing.

**Catches what we miss**: Bearing in joint 3 is wearing out. Actions are correct but physical precision is degrading over weeks.

**Rank**: 5/10. High value for production but significant hardware requirements.

### 12. Power / Energy Consumption Monitoring

**What**: Track power draw of the robot and compute device. Anomalous energy signatures indicate mechanical faults, payload changes, or compute throttling. Monitor battery voltage on mobile robots to detect brownout risk.

**Access**: Needs hardware access (power sensors / BMS).

**Prior art**: IEEE paper on robot fault detection via power consumption modeling. Research on deep autoencoders for energy anomaly detection. Motor Current Signature Analysis (MCSA) detects bearing wear, rotor bar issues, short circuits from current alone. Jetson Orin has power monitoring via tegrastats.

**Applied to VLA?**: Partially - `tegrastats` monitors Jetson power but nobody uses it as a policy health signal. Power spikes during inference could indicate thermal throttling.

**Complexity**: Low for compute power (tegrastats). Medium for robot power.

**Catches what we miss**: Jetson is thermally throttling, inference slows down, but we only see the latency symptom, not the power/thermal root cause.

**Rank**: 6/10. Easy for edge compute monitoring, harder for robot side.

### 13. Thermal Monitoring

**What**: Track temperatures of motors, compute devices, and environment. Motor overheating degrades torque control. GPU/SoC overheating causes throttling. Novel: motor temperature as a leading indicator of mechanical failure.

**Access**: Needs hardware access.

**Prior art**: Smart actuators with PCB/winding temperature monitoring use "soft-thermal-limiting" to reduce torque before shutdown. Research shows motors without thermal feedback exceed 75C and degrade; with feedback they hold 60C setpoint. Jetson thermal zones accessible via sysfs.

**Applied to VLA?**: NO. VLA deployment papers don't mention thermal management.

**Complexity**: Low for Jetson (sysfs temperature readings). Medium for robot motors.

**Catches what we miss**: Motor in a heavily-used joint is approaching thermal limits. Policy actions are fine but the next heavy movement could trigger thermal shutdown.

**Rank**: 6/10. Cheap for compute side, valuable for continuous operation.

### 14. Denoising Trajectory Quality (Diffusion/Flow VLAs)

**What**: Monitor intermediate states during the denoising process. Track whether the denoising trajectory converges smoothly or oscillates. Dynamic denoising (D3P, arxiv 2508.06804) shows that crucial vs routine actions need different numbers of denoising steps.

**Access**: Needs model internals (intermediate denoising states).

**Prior art**: D3P dynamically adjusts denoising steps (3 for simple motions, 8 for grasping). Two-Steps Diffusion (arxiv 2510.21991) uses genetic denoising. Research shows intermediate states can be out-of-distribution due to clipping. Fast Policy Synthesis (arxiv 2406.04806) uses variable noise schedules.

**Applied to VLA?**: D3P monitors action criticality to adjust steps, but not for health monitoring. Nobody tracks denoising convergence as a runtime health signal.

**Complexity**: Medium. Requires hooks into the denoising loop.

**Catches what we miss**: Denoising process is not converging cleanly (oscillating between modes). Final action looks OK but is unstable - small perturbation would produce very different result.

**Rank**: 5/10. Novel and interesting but niche to diffusion architectures.

### 15. Scene Change Detection

**What**: Track whether the environment has changed unexpectedly. Detect new objects, removed objects, or moved objects that weren't part of the task. Critical for long-running deployments.

**Access**: Needs environment access (camera observations over time).

**Prior art**: SceneDiff (arxiv 2512.16908) - training-free 3D scene change detection. Research on scene change for robotic patrol (IEEE 2024). Graph neural networks for object correspondence tracking. Zero-shot methods (arxiv 2406.11210).

**Applied to VLA?**: NO. VLA policies assume a relatively static environment during execution.

**Complexity**: Medium-high. Needs object detection + tracking + change classification.

**Catches what we miss**: A human placed a new obstacle in the workspace between episodes. Policy doesn't know about it because it's not in the task description.

**Rank**: 5/10. Important for long-running autonomous operation, overkill for short tasks.

### 16. End-Effector Pose Tracking Accuracy

**What**: Compare commanded end-effector pose (from forward kinematics + action) to actual measured pose (from external tracking or secondary sensors). Drift indicates mechanical wear, collision damage, or calibration loss.

**Access**: Needs external measurement system or secondary encoders.

**Prior art**: Motion capture-based calibration. Secondary encoders improve TCP accuracy 70-80%. Online photogrammetric correction systems. NIST robot performance testing standards.

**Applied to VLA?**: NO. VLA research assumes perfect actuation.

**Complexity**: High. Requires external tracking infrastructure.

**Catches what we miss**: Robot's TCP has drifted 5mm from collision damage. Policy commands the right position but robot goes to the wrong position. Our action monitors see correct actions.

**Rank**: 4/10. Valuable but heavy infrastructure requirement.

### 17. Cross-Modal Alignment Score

**What**: Monitor whether the VLA's internal vision and language representations remain well-aligned during execution. Misalignment = model is confused about what it's seeing vs what it's been told.

**Access**: Needs model internals (vision and language embeddings).

**Prior art**: VLA-FEB introduces Cross-Modal Alignment Score (CMAS) and Fusion Energy Index (FEI). HMVLA uses hyperbolic space for better vision-language alignment. Research on cross-modality alignment perception for humanoid robots.

**Applied to VLA?**: As a benchmark metric, yes. As a runtime monitor, no.

**Complexity**: Medium. Requires extracting and comparing vision/language embeddings.

**Catches what we miss**: Language instruction says "pick up the red cup" but the vision encoder is attending to a blue object. Actions may still be within bounds.

**Rank**: 5/10. Interesting but requires model hooks and unclear runtime overhead.

### 18. Human Proximity / Workspace Occupancy

**What**: Monitor the presence and position of humans in the robot's workspace. Adjust behavior based on proximity zones (ISO 15066 SSM). Track whether the workspace is clear for autonomous operation.

**Access**: Needs environment sensors (lidar, cameras, safety scanners).

**Prior art**: ISO 15066 / ISO 10218-2:2025 define safety zones and SSM requirements. Adaptive SSM with dynamically switched zones. Safety barrier functions with multi-camera tracking. 3D occupancy maps from RGBD cameras.

**Applied to VLA?**: NO. VLA policies have no awareness of human presence. This is handled by external safety systems if at all.

**Complexity**: Medium-high. Requires safety-rated sensing infrastructure.

**Catches what we miss**: A human walks into the workspace. Policy continues executing because it has no human detection. Our action monitors see valid actions.

**Rank**: 6/10. Critical for real deployment but orthogonal to policy monitoring.

### 19. Reward/Value Function Runtime Monitoring

**What**: If a value function or reward model was trained alongside the policy, evaluate it at runtime. Declining value estimates = policy expects poor outcomes ahead.

**Access**: Needs trained value function + model access.

**Prior art**: LIRF (CoRL 2022 Best Paper) trains interactive reward functions for both training and runtime verification. TD3 value function predictions can be monitored for consistency. Formal methods survey (arxiv 2602.06971) covers value function verification.

**Applied to VLA?**: Not directly. VLA models are typically trained via imitation learning (no value function). But could train a separate value estimator from demonstrations.

**Complexity**: High. Requires a separate model.

**Catches what we miss**: Policy is heading toward a state from which recovery is impossible, but current actions look fine.

**Rank**: 4/10. Conceptually powerful but practically heavy for edge deployment.

---

## TIER 4: APM/NETWORKING/MEDICAL INSPIRATION

### 20. "Golden Signals" from APM (Adapted for Robotics)

Software observability uses four golden signals: **latency, traffic, errors, saturation**. Adapted:

| APM Signal | Robot Policy Analog |
|---|---|
| Latency | Inference time (Signal #1) |
| Traffic | Control frequency / throughput (steps/sec) |
| Errors | Violation rate, OOD rate, stall rate |
| Saturation | GPU utilization, memory pressure, queue depth |

**Distributed tracing**: Trace an observation from capture through preprocessing, inference, postprocessing, to actuation. Identify bottlenecks.

**SLOs/SLIs**: Define "99% of inferences complete in <100ms" as a service level objective. Alert when SLO is breached.

### 21. SPC Control Charts for Action Distributions

From manufacturing quality control: Plot action statistics (mean, variance, range per joint) on control charts with Upper/Lower Control Limits (UCL/LCL). Apply Western Electric rules for out-of-control signals:

- 1 point beyond 3-sigma = immediate alarm
- 2 of 3 points beyond 2-sigma = warning
- 4 of 5 points beyond 1-sigma = trending
- 8 consecutive points on same side of mean = shift

**Advantage over CUSUM**: More interpretable, established methodology, catches patterns CUSUM misses.

**Complexity**: Low. We already compute per-joint statistics. Just need control chart logic.

**Rank**: 7/10. Novel application to VLA, very implementable.

### 22. Patient Monitoring Analog: Multi-Signal Composite Scores

Medical ICU monitoring combines HR, BP, SpO2, temperature, respiratory rate into composite early warning scores (NEWS, MEWS). Alert thresholds are per-signal AND composite.

**Robot analog**: Combine inference latency + action violation rate + observation quality + embedding drift into a single "Robot Vital Signs" score with per-component and composite thresholds.

**Key insight from medical**: Gradual decline across multiple signals is more dangerous than a spike in one signal. Track trend across all signals simultaneously.

**Rank**: 6/10. Great framing, needs the individual signals first.

---

## SUMMARY: TOP 10 SIGNALS TO IMPLEMENT

Ranked by (novelty x practical value x implementability):

| Rank | Signal | Access Type | Complexity | Catches What We Miss |
|---|---|---|---|---|
| 1 | **Inference Latency & Timing** | Black-box | Trivial | GPU throttling, stale actions, deadline misses |
| 2 | **Observation Quality** | Observation | Low | Camera degradation, occlusion, frozen frames |
| 3 | **Action-Observation Consistency** | Observation | Medium | Environment changed but actions look fine |
| 4 | **SPC Control Charts** | Black-box | Low | Systematic drift patterns CUSUM misses |
| 5 | **Action Chunk Staleness** | Black-box | Trivial | Async inference temporal misalignment |
| 6 | **Token-Level Entropy** | Model internals | Medium | Model internally uncertain, actions look valid |
| 7 | **Embedding Space OOD** | Model internals | Medium | Completely novel scene, actions happen to be within bounds |
| 8 | **Motor Current/Torque** | Hardware | Low-Medium | Mechanical degradation, unexpected contact |
| 9 | **Compute Thermal/Power** | Hardware | Low | Thermal throttling before it causes latency |
| 10 | **Task Progress Estimation** | Observation + task | Medium-High | Robot doing valid actions but not making progress |

---

## IMPLEMENTATION ROADMAP FOR VLA-EDGE

### Phase 1: Black-box signals (no model/env access needed)
- Inference latency monitor (wrap policy.predict with timing)
- Control frequency monitor (steps/sec tracker)
- SPC control charts on action distributions
- Action chunk staleness tracker

### Phase 2: Observation-access signals
- Image quality monitor (blur, brightness, noise, frozen frame)
- Scene change detector (lightweight: frame-to-frame optical flow magnitude)
- Action-observation consistency (simple: did the observation change proportionally to the action magnitude?)

### Phase 3: Model-internal signals
- Token entropy (autoregressive VLAs only)
- Embedding drift (Mahalanobis distance from calibration)
- Attention pattern monitor (entropy of attention maps)
- Denoising convergence (diffusion/flow VLAs only)

### Phase 4: Hardware integration
- Jetson power/thermal (tegrastats integration)
- Motor current monitoring (robot-specific, start with UR/Franka API)
- Human proximity (external safety system integration)

---

## KEY PAPERS DISCOVERED

1. **FIPER** (NeurIPS 2025) - Failure Prediction at Runtime via OOD + action-chunk entropy. Conformal calibration from success-only data. https://arxiv.org/abs/2510.09459
2. **Sentinel** (RSS 2024 area) - Temporal Action Consistency (STAC) + VLM task progress. Splits failures into erratic vs progression types. https://arxiv.org/abs/2410.04640
3. **CriticalUQ** (2026) - Sliding window max-pooling + action transfer reweighting + DoF-adaptive calibration for VLA uncertainty. 0.936 AUROC. https://arxiv.org/abs/2603.18342
4. **VLA Knows Its Limits** (2026) - Cross-attention invariance + radial action sinks as implicit confidence. AutoHorizon for adaptive chunk execution. https://arxiv.org/abs/2602.21445
5. **UPS** (2026) - Uncertainty-Aware Policy Steering: execute/clarify/retrain decisions from VLM verifier uncertainty + conformal prediction. https://arxiv.org/abs/2602.22474
6. **VLA Quality Metrics** (2025) - 8 uncertainty + 5 quality metrics for VLA evaluation. 908 executions, human-labeled. https://arxiv.org/abs/2507.17049
7. **RTC** (NeurIPS 2025) - Real-Time Chunking: overlap RMSE as inter-chunk consistency metric. https://arxiv.org/abs/2506.07339
8. **AsyncVLA** (2026) - Asynchronous inference framework, staleness vs throughput tradeoffs. https://arxiv.org/abs/2602.13476
9. **D3P** (2025) - Dynamic Denoising Diffusion Policy, monitors action criticality to adjust denoising steps. https://arxiv.org/abs/2508.06804
10. **World Model Anomaly** (2026) - Conformal prediction on world model outputs for distribution-free anomaly detection. Deployed on Boston Dynamics Spot. https://arxiv.org/abs/2602.16182

---

## COMPETITIVE POSITIONING

**What exists**: FIPER and Sentinel monitor VLA failures at runtime. CriticalUQ quantifies VLA uncertainty. All are research prototypes focused on single failure modes.

**What doesn't exist**: A comprehensive monitoring toolkit that combines black-box action monitoring (our strength) + observation quality + timing signals + model uncertainty into a unified deployment health dashboard.

**Our opportunity**: vla-edge already has the strongest action-space monitoring (5 monitors + STL + conformal). Adding even just Tier 1 signals (#1-5) would make it the most comprehensive VLA deployment monitoring system available. The "Robot Vital Signs" composite score (medical analog) is completely novel framing.

Sources consulted: 30+ papers, ISO 15066/10218, FANUC ZDT docs, KUKA Connect docs, ABB predictive maintenance, Waymo safety framework, APM/observability literature (Datadog, New Relic, AWS CloudWatch), SPC/Six Sigma literature, medical monitoring literature.
