# Deep Research: Additional Violation Types for VLA Action Monitoring

**Date**: 2026-04-05
**Purpose**: Identify new violation types beyond bounds/velocity/acceleration that are practical, meaningful, and would catch issues even on smooth Diffusion/flow-matching policies.
**Status**: Research complete

---

## 1. Ranked Violation Types Worth Adding

Ranked by: (real-world importance) x (implementability from actions alone) x (novelty for VLA monitoring)

### Tier 1: High Priority - Build These

| Rank | Violation Type | Importance | Implementability | Novelty | Score |
|------|---------------|------------|-----------------|---------|-------|
| 1 | **Jerk (3rd derivative)** | 5/5 | 5/5 | 3/5 | 75 |
| 2 | **Stall/Freeze Detection** | 5/5 | 5/5 | 4/5 | 100 |
| 3 | **Chunk Boundary Discontinuity** | 4/5 | 4/5 | 5/5 | 80 |
| 4 | **Gripper Oscillation** | 4/5 | 5/5 | 4/5 | 80 |
| 5 | **Direction Reversal Rate** | 3/5 | 5/5 | 4/5 | 60 |

### Tier 2: Medium Priority - Nice to Have

| Rank | Violation Type | Importance | Implementability | Novelty | Score |
|------|---------------|------------|-----------------|---------|-------|
| 6 | **Action Total Variation (ATV)** | 3/5 | 5/5 | 3/5 | 45 |
| 7 | **Repetition/Cycle Detection** | 3/5 | 4/5 | 4/5 | 48 |
| 8 | **Workspace Utilization Drift** | 2/5 | 3/5 | 4/5 | 24 |

### Tier 3: Research Only - Needs Model Internals

| Rank | Violation Type | Importance | Implementability | Novelty | Score |
|------|---------------|------------|-----------------|---------|-------|
| 9 | **Action Entropy (distribution)** | 4/5 | 1/5 | 4/5 | 16 |
| 10 | **Denoising Loss OOD Score** | 5/5 | 1/5 | 3/5 | 15 |

---

## 2. Detailed Analysis of Each Violation Type

### 2.1 Jerk Monitoring (Rank 1)

**What it is**: The third derivative of position (or first derivative of acceleration). Measures how "smooth" the acceleration changes are.

**Why it matters for real robots**:
- High jerk excites structural resonances and increases actuator wear. Source: [Zaber S-Curve guide](https://www.zaber.com/articles/jerk-control)
- Non-smooth trajectories cause mechanical vibration, over-current of drives, and degradation of control performance. Source: [IJFIS jerk-limited reference generation](https://www.ijfis.org/journal/view.html?uid=801&vmd=Full)
- Vibrations from high jerk damage robot actuators and introduce large tracking errors. Source: [MDPI - Should We Forget Jerk?](https://www.mdpi.com/2571-631X/9/1/1)
- Human movements naturally minimize RMS jerk - it is a fundamental smoothness metric. Source: [Min-jerk trajectory generation, arXiv:2102.07459](https://arxiv.org/pdf/2102.07459)

**Real robot jerk limits**:
- **Franka Panda**: Joint jerk limit = 5000 rad/s^3 per joint, Cartesian translation jerk = 4500 m/s^3, rotation jerk = 8500 rad/s^3. Control loop at 1000 Hz. Source: [franky library docs](https://timschneider42.github.io/franky/index.html), [franka_ros joint_limits.yaml](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml)
- **UR5**: Max joint velocity 180 deg/s, joint acceleration 2292 deg/s^2. Jerk limits not published but the controller enforces them internally. Source: [UR5 datasheet](https://www.universal-robots.com/media/50588/ur5_en.pdf)
- **SO-100/101**: Uses STS3215 bus servos (19.5 kg-cm torque). No published jerk limits - these are hobby servos without jerk control. Source: [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- **ISO 15066**: Does not specify jerk limits directly. Focuses on force/pressure limits by body region (Annex A). Speed limits depend on effective mass and contact geometry. Practical PFL cobot speeds: 250-1000 mm/s. Source: [Robotiq ISO 15066 explanation](https://www.automate.org/robotics/tech-papers/iso-ts-15066-explained)

**Why it catches Diffusion Policy issues**: The ACG paper (arXiv:2510.22201) showed that flow-matching VLAs memorize noise from human demonstrations - "pauses, jerks, and jitter." They introduced JerkRMS as a primary metric. Even "smooth" diffusion policies have jerk spikes at denoising artifacts and chunk boundaries.

**Formula**:
```
jerk[t] = acc[t] - acc[t-1] = (a[t] - 2*a[t-1] + a[t-2]) - (a[t-1] - 2*a[t-2] + a[t-3])
        = a[t] - 3*a[t-1] + 3*a[t-2] - a[t-3]

jerk_rms = sqrt(mean(jerk^2))  # per joint, over a window
```

**Metric from ACG paper**: Action Total Variation (ATV) and JerkRMS are the two metrics they use. ATV = sum of absolute differences between consecutive actions. JerkRMS = root mean square of the third derivative.

---

### 2.2 Stall/Freeze Detection (Rank 2)

**What it is**: Detect when the policy gets "stuck" producing near-zero or repetitive actions for too long.

**Why it matters**:
- One of the clearest failure modes in VLA deployment: "the model gets stuck against a wall or in a corner and doesn't know how to recover, because the dataset didn't contain recovery behavior." Source: [VLA article, learnopencv.com](https://learnopencv.com/vision-language-action-models-lerobot-policy/)
- Diffusion policies under OOD inputs literally replay training data: "Even when conditioned on wildly OOD images of cats and dogs, the Diffusion Policy still outputs an action sequence from the training data." Source: [Demystifying Diffusion Policies, arXiv:2505.05787](https://arxiv.org/html/2505.05787)
- FPC-VLA identifies stuck states as a primary failure requiring supervisor intervention. Source: [FPC-VLA, arXiv:2509.04018](https://arxiv.org/html/2509.04018)
- SAFE (Multitask Failure Detection for VLAs) specifically targets detecting these failure modes at runtime. Source: [SAFE, OpenReview](https://openreview.net/forum?id=XPyAukgsFf)

**Why it catches Diffusion Policy issues**: This is THE failure mode for diffusion policies. They are smooth, bounded, and continuous - but they can get stuck in local minima, replaying demonstration segments indefinitely. No existing bounds/velocity check will catch this because the robot IS moving, just not making progress.

**Formula**:
```
# Displacement over window of W steps
displacement = norm(a[t] - a[t-W])
# Or: mean speed over window
mean_speed = mean(norm(a[t] - a[t-1]) for t in window)

# Stall if displacement < threshold for > max_stall_steps
is_stalled = mean_speed < stall_threshold for stall_duration > max_stall_steps
```

---

### 2.3 Chunk Boundary Discontinuity (Rank 3)

**What it is**: Detect discontinuities at the point where one action chunk ends and the next begins.

**Why it matters**:
- "Adjacent chunks may jump between different modes (or 'strategies') from the learned action distribution" creating "very high, out-of-distribution acceleration." Source: [RTC, arXiv:2506.07339](https://arxiv.org/html/2506.07339v1)
- "The transition from chunk N to chunk N+1 is not going to be smooth." Source: [Alexander Soare blog on smooth policies](https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html)
- REMAC identifies two failure types: inter-chunk discontinuity (mode jumping) and intra-chunk inconsistency (stale observation conditioning). Source: [REMAC, arXiv:2601.20130](https://arxiv.org/html/2601.20130)
- BID (Bidirectional Decoding) explicitly optimizes for "backward coherence" at chunk boundaries. Source: [BID, arXiv:2408.17355](https://arxiv.org/html/2408.17355v1)

**Why it catches Diffusion Policy issues**: Chunk boundaries are the Achilles heel of diffusion/flow-matching policies. Within a chunk, the denoising process ensures smoothness. But when a new chunk starts from a fresh noise sample, it may select a different mode entirely. The RTC paper shows this causes "very high, out-of-distribution acceleration" at boundaries.

**Requirements**: Needs knowledge of chunk boundaries (when the model generated a new chunk). Can be computed as: check if the step index is a multiple of chunk_size, then measure the L2 discontinuity between the last action of old chunk and first of new chunk.

**Formula**:
```
# At chunk boundary (step t = k * chunk_size):
boundary_discontinuity = norm(a[t] - a[t-1])  # simple
boundary_acceleration = norm((a[t] - a[t-1]) - (a[t-1] - a[t-2]))  # acceleration spike

# Compare to within-chunk statistics
within_chunk_mean_vel = mean(norm(a[i] - a[i-1]) for i in chunk)
boundary_ratio = boundary_discontinuity / (within_chunk_mean_vel + eps)
# Violation if boundary_ratio > threshold (e.g., 3.0)
```

---

### 2.4 Gripper Oscillation (Rank 4)

**What it is**: Detect rapid open-close-open-close cycling of the gripper dimension.

**Why it matters**:
- Gripper oscillation is a clear indicator of policy confusion/indecision. The policy cannot decide whether to grasp or release.
- In RoboFail taxonomy, "Manipulation & Control Failures" include failed grasps and unintended drops - gripper oscillation precedes both. Source: [RoboFail, arXiv:2412.02818](https://arxiv.org/html/2412.02818v1)
- Derivative control damps oscillations for smoother stopping, and high-frequency position commands make grippers appear shaky. Source: [Hitbot gripper control](https://www.hitbotrobot.com/control-principle-of-electric-gripper/)
- In real deployments, gripper oscillation causes mechanical wear and drops objects.

**Why it catches Diffusion Policy issues**: Gripper actions in diffusion policies are typically the last dimension and often exhibit bimodal behavior (open vs closed). The denoising process can oscillate between these modes, especially near decision boundaries. This is invisible to velocity/acceleration checks because individual transitions may be within limits - it is the PATTERN of reversals that is pathological.

**Formula**:
```
# Assuming gripper is last dimension of action
gripper = actions[:, -1]
# Count sign changes in gripper velocity
gripper_vel = diff(gripper)
sign_changes = sum(gripper_vel[t] * gripper_vel[t-1] < 0 for t in range(1, len(gripper_vel)))
oscillation_rate = sign_changes / len(gripper_vel)
# Violation if oscillation_rate > threshold (e.g., 0.3 means 30% of steps are reversals)
```

---

### 2.5 Direction Reversal Rate (Rank 5)

**What it is**: Generalization of gripper oscillation to all joints. Measures how often each joint reverses direction.

**Why it matters**:
- High reversal rates indicate indecision, mode-switching, or denoising artifacts.
- "Reduced action coherence causes instability and trajectory drift during deployment, which are catastrophic failures in fine-grained manipulation tasks." Source: [ACG, arXiv:2510.22201](https://arxiv.org/pdf/2510.22201)
- Frequent reversals cause excess mechanical wear and are a sign the policy has no coherent plan.

**Formula**:
```
# Per joint j:
vel = diff(actions[:, j])
reversals = sum(vel[t] * vel[t-1] < 0 for t in range(1, len(vel)))
reversal_rate_j = reversals / (len(vel) - 1)
# Aggregate: mean reversal rate across joints
# Violation if any joint exceeds reversal_threshold (e.g., 0.4)
```

---

### 2.6 Action Total Variation (Rank 6)

**What it is**: Sum of absolute differences between consecutive actions. A direct smoothness metric used in the ACG paper.

**Formula**:
```
ATV = sum(norm(a[t] - a[t-1]) for t in range(1, T))
# Or per-joint: ATV_j = sum(abs(a[t,j] - a[t-1,j]) for t in range(1, T))
```

**Why it is ranked lower**: ATV is a summary statistic, not a per-step violation. It is useful as a trajectory-level health metric but does not trigger per-step interventions. Better as a dashboard metric than a guard check.

---

### 2.7 Repetition/Cycle Detection (Rank 7)

**What it is**: Detect when the policy enters a repetitive loop, executing the same sequence of actions cyclically.

**Why it matters**: Different from stalling - the robot IS moving but repeating the same trajectory segment. Common failure mode when the policy has learned a periodic sub-behavior (e.g., stirring) and cannot exit it.

**Formula**:
```
# Autocorrelation of action sequence at lag L
autocorr(L) = correlation(actions[:-L], actions[L:])
# High autocorrelation at some lag L indicates repetitive behavior
# Violation if autocorr(L) > threshold for any L in [min_cycle, max_cycle]
```

---

### 2.8 Workspace Utilization Drift (Rank 8)

**What it is**: Track the running mean/variance of action positions. If the distribution narrows significantly (policy always goes to the same spot) or drifts (gradual shift away from expected workspace), flag it.

**Lower priority because**: Requires baseline statistics from training data or calibration runs. Not self-contained from actions alone.

---

## 3. What Would Catch Issues on Diffusion Policies Specifically?

Diffusion/flow-matching policies (SmolVLA, pi0) are smooth, bounded, and continuous by design. Standard bounds/velocity/acceleration checks will rarely fire. The failures that actually occur:

### Failure Mode 1: Stalling (HIGHEST PRIORITY)
- **Mechanism**: Policy gets stuck in a local energy minimum. Produces valid but useless actions - gentle rocking, micro-movements, or holding position.
- **Detection**: Stall detector (Rank 2). Simple mean-speed-over-window check.
- **Evidence**: Documented in multiple papers: FPC-VLA, SAFE, LearnOpenCV VLA review.

### Failure Mode 2: Chunk Boundary Mode Switching
- **Mechanism**: New action chunk starts from fresh noise, selects different strategy than previous chunk. Causes discontinuous acceleration at boundary.
- **Detection**: Chunk boundary discontinuity (Rank 3). Requires chunk_size parameter.
- **Evidence**: RTC (RSS 2025), REMAC (2026), BID (2024), Alexander Soare blog.

### Failure Mode 3: Denoising Artifacts in Gripper
- **Mechanism**: Gripper dimension is bimodal (open/closed). Denoising can oscillate between modes across steps.
- **Detection**: Gripper oscillation (Rank 4).
- **Evidence**: ACG paper notes "jitter" in demonstrations propagates through VLAs.

### Failure Mode 4: Temporal Drift / Coherence Loss
- **Mechanism**: Over long horizons, small errors compound. Policy drifts away from intended trajectory. Actions individually look fine but the integrated trajectory diverges.
- **Detection**: Direction reversal rate (Rank 5) catches early signs. Jerk monitoring (Rank 1) catches the acceleration profile deviating from smooth.
- **Evidence**: ACG paper, FreqPolicy (NeurIPS 2025), Dual-Memory VLA (arXiv:2602.20200).

### What WON'T catch Diffusion Policy failures:
- **Bounds checking**: Diffusion policies naturally stay in-distribution. Rarely violate.
- **Velocity checking alone**: Smooth policies have smooth velocities. The problem is not speed but COHERENCE.
- **Acceleration checking alone**: Within-chunk acceleration is controlled by denoising. Between-chunk is the issue.

---

## 4. Implementation Sketches for Top 3

### 4.1 Jerk Monitor - Add to SafetyConfig + validate_actions

```python
# In SafetyConfig, add:
max_jerk: np.ndarray | None = None  # per-joint jerk limit (rad/s^3 or units/step^3)

# In SafetyResult, add:
max_jerk_observed: float = 0.0

# In validate_actions, add after acceleration check (~15 lines):
# Check jerk (requires t > 2)
if t > 2 and config.max_jerk is not None:
    acc_curr = (actions[t] - actions[t-1]) - (actions[t-1] - actions[t-2])
    acc_prev = (actions[t-1] - actions[t-2]) - (actions[t-2] - actions[t-3])
    jerk = np.abs(acc_curr - acc_prev)
    max_jerk_val = max(max_jerk_val, float(np.max(jerk)))
    for j in range(min(len(jerk), len(config.max_jerk))):
        if jerk[j] > config.max_jerk[j]:
            violations.append(SafetyViolation(
                step=t, violation_type="jerk", joint=j,
                value=float(jerk[j]), limit=float(config.max_jerk[j]),
                severity="warning",
            ))
```

Estimated: ~20 lines in safety.py, ~15 lines test. Mirrors existing acceleration check pattern.

### 4.2 Stall Detector - New function in safety.py

```python
@dataclass
class StallConfig:
    """Configuration for stall detection."""
    speed_threshold: float = 0.01   # minimum mean speed (units/step)
    window_size: int = 10           # how many steps to average over
    max_stall_steps: int = 20       # how long before it is a violation

def detect_stall(
    actions: np.ndarray,
    config: StallConfig,
) -> list[SafetyViolation]:
    """Detect stall/freeze periods in action sequence."""
    violations = []
    if len(actions) < config.window_size:
        return violations

    stall_count = 0
    for t in range(1, len(actions)):
        speed = np.linalg.norm(actions[t] - actions[t-1])
        if speed < config.speed_threshold:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= config.max_stall_steps:
            violations.append(SafetyViolation(
                step=t, violation_type="stall", joint=None,
                value=float(speed), limit=config.speed_threshold,
                severity="warning",
            ))
            stall_count = 0  # reset to avoid duplicate reports

    return violations
```

Estimated: ~30 lines in safety.py, ~25 lines test.

### 4.3 Gripper Oscillation Detector

```python
@dataclass
class GripperOscillationConfig:
    """Configuration for gripper oscillation detection."""
    gripper_dim: int = -1           # which dimension is the gripper (default: last)
    window_size: int = 10           # window to count reversals
    max_oscillation_rate: float = 0.3  # max fraction of steps that are reversals

def detect_gripper_oscillation(
    actions: np.ndarray,
    config: GripperOscillationConfig,
) -> list[SafetyViolation]:
    """Detect rapid gripper open/close oscillation."""
    violations = []
    gripper = actions[:, config.gripper_dim]
    if len(gripper) < 3:
        return violations

    vel = np.diff(gripper)
    # Sign changes in velocity = direction reversals
    sign_changes = (vel[:-1] * vel[1:]) < 0

    # Sliding window oscillation rate
    for t in range(config.window_size, len(sign_changes)):
        window = sign_changes[t - config.window_size : t]
        rate = np.mean(window)
        if rate > config.max_oscillation_rate:
            violations.append(SafetyViolation(
                step=t, violation_type="gripper_oscillation", joint=config.gripper_dim,
                value=float(rate), limit=config.max_oscillation_rate,
                severity="warning",
            ))

    return violations
```

Estimated: ~30 lines in safety.py, ~25 lines test.

---

## 5. Experiments to Run on RunPod

### Experiment 1: Violation Type Sensitivity Analysis
**Hypothesis**: New violation types detect failure modes that existing bounds/velocity checks miss.
**Design**:
- Run SmolVLA on LIBERO tasks (pick, stack, drawer, pour)
- Record full action trajectories
- Compute all violation types (existing + new) per trajectory
- Correlate each violation type with task success/failure
- Key metric: precision/recall of each violation type as a failure predictor

### Experiment 2: Diffusion Policy Chunk Boundary Analysis
**Hypothesis**: Chunk boundaries have higher jerk and discontinuity than within-chunk steps.
**Design**:
- Run SmolVLA (chunk_size=10, 16, 32) on LIBERO
- Measure: jerk at boundary steps vs non-boundary steps
- Measure: velocity discontinuity at boundaries
- Plot: distribution of jerk values at boundary vs non-boundary

### Experiment 3: Stall Prevalence Study
**Hypothesis**: Stalling is a primary failure mode for diffusion policies on OOD inputs.
**Design**:
- Run SmolVLA on LIBERO with increasing perturbation levels (camera, lighting, position)
- Count stall events (mean speed < threshold for > N steps)
- Correlate stall frequency with perturbation severity
- Key metric: what fraction of failures are preceded by stall events?

### Experiment 4: Gripper Oscillation as Failure Predictor
**Hypothesis**: Gripper oscillation rate predicts grasp failures.
**Design**:
- Run SmolVLA on LIBERO pick tasks
- Compute gripper oscillation rate for successful vs failed episodes
- Determine threshold that maximizes F1 for failure prediction

### Estimated RunPod cost: ~$10-15 (A100 for ~3-4 hours across all experiments)

---

## 6. Papers to Cite

### Core (already cited or should be):
1. **ACG**: "Action Coherence Guidance for Flow-based VLA Models" (arXiv:2510.22201) - Defines ATV and JerkRMS metrics. Shows flow-matching VLAs memorize demonstration noise.
2. **RTC**: "Real-Time Execution of Action Chunking Flow Policies" (arXiv:2506.07339, NeurIPS 2025) - Documents chunk boundary discontinuity problem.
3. **REMAC**: "Real-Time Robot Execution with Masked Action Chunking" (arXiv:2601.20130) - Inter-chunk and intra-chunk inconsistency.
4. **BID**: "Bidirectional Decoding: Improving Action Chunking via Closed-Loop Resampling" (arXiv:2408.17355) - Backward coherence at boundaries.
5. **FreqPolicy**: "Efficient Flow-based Visuomotor Policy via Frequency Consistency" (arXiv:2506.08822, NeurIPS 2025) - Frequency domain temporal consistency.

### Failure Detection:
6. **RoboFail**: "Analyzing Failures in Robot Learning Policies" (arXiv:2412.02818) - Failure mode taxonomy and probability estimation.
7. **SAFE**: "Multitask Failure Detection for VLA Models" (OpenReview) - Runtime failure detection for generalist policies.
8. **FPC-VLA**: "A Vision-Language-Action Framework with a Supervisor for Failure Prediction and Correction" (arXiv:2509.04018) - VLM supervisor for failure prediction.
9. **Diff-DAgger**: "Uncertainty Estimation with Diffusion Policy for Robotic Manipulation" (arXiv:2410.14868, ICRA 2025) - Denoising loss as OOD score.

### Anomaly Detection:
10. **AESOP**: "Real-Time Anomaly Detection and Reactive Planning with LLMs" (RSS 2024 Best Paper, arXiv:2407.08735) - Embedding-space anomaly detection for robot safety.

### Jerk and Smoothness:
11. "Should We Forget the Jerk in Trajectory Generation?" (Robotics, 2020) - Comprehensive analysis of why jerk matters.
12. "Gradient-based Regularization for Action Smoothness in Robotic Control" (arXiv:2407.04315) - CAPS temporal smoothness regularization.

### Safety Standards:
13. **ISO/TS 15066:2016** - Collaborative robot safety. Force/pressure limits by body region, PFL mode requirements.
14. **ISO 10218:2025** - Updated robot safety standard with integrated collaborative requirements.

---

## 7. Summary: What to Build

**Immediate (Phase 2 scope)**:
1. **Jerk monitoring** - Add `max_jerk` to SafetyConfig, ~20 lines. Mirrors acceleration check. Catches denoising artifacts.
2. **Stall detection** - New `StallConfig` + `detect_stall()`, ~30 lines. Catches THE primary diffusion policy failure mode.
3. **Gripper oscillation** - New `GripperOscillationConfig` + `detect_gripper_oscillation()`, ~30 lines. Catches grasp indecision.

**Later (CoRL scope)**:
4. **Chunk boundary discontinuity** - Requires chunk_size metadata. Add to SafetyGuard, not safety.py.
5. **Direction reversal rate** - Windowed reversal counting per joint. Dashboard metric.
6. **Action Total Variation** - Trajectory-level summary. Dashboard metric.

**Key insight**: The first three are pure action-stream computations. No model internals, no observation data, no chunk metadata needed. They slot directly into the existing `validate_actions()` pattern and the `@safety_contract` decorator.
