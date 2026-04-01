# Conformal Prediction for VLA Safety Bounds

**Research Date**: 2026-03-29
**Question**: Can split conformal prediction give us provably safe action bounds for VLA models that are tighter and more principled than hand-tuned percentiles?

**TL;DR**: Yes. Split CP provides finite-sample, distribution-free guarantees of the form P(action in safe set) >= 1-alpha using only ~1/alpha calibration samples. This is strictly better than percentile-based bounds because: (1) it comes with a formal coverage guarantee, (2) it adapts to the score distribution shape, and (3) it requires O(1/epsilon) samples vs O(1/epsilon^2) for PAC-style approaches. The literature is rich and directly applicable to VLA action bounds.

---

## 1. Key Papers

### 1.1 SAFE (TRI, NeurIPS 2025)

**Paper**: "SAFE: Multitask Failure Detection for Vision-Language-Action Models" - Gu, Ju et al.
**URL**: https://arxiv.org/abs/2506.09937 | https://vla-safe.github.io/

**What they do**:
- Train a lightweight failure detector on top of VLA internal features (not raw actions)
- The detector outputs a scalar failure score s_t at each timestep
- They use **functional conformal prediction** (one-sided time-varying CP band) to set detection thresholds
- Calibrated on successful rollouts only from D_eval-seen

**CP formulation**:
- CP band: C_alpha = {[lower_t, upper_t] : t=1,...,T} where lower_t = -inf, upper_t = mu_t + h_t
- Guarantee: For any new successful rollout, s_t < mu_t + h_t for all t with probability >= 1-alpha
- When failure score exceeds threshold, raise alarm
- Alpha controls the accuracy/detection-time tradeoff

**How this differs from what we'd do**:
- SAFE detects failures (binary: will this trajectory fail?) using VLA feature space
- We want to bound actions directly (continuous: is this action within safe limits?)
- SAFE uses CP for threshold calibration on a learned score
- We'd use CP to construct action-space prediction sets directly
- SAFE is complementary - we could use SAFE-style failure detection AND action bounds together

**Tested on**: OpenVLA, pi_0, pi_0-FAST (simulated + real)

### 1.2 FIPER (TUM, NeurIPS 2025)

**Paper**: "Failure Prediction at Runtime for Generative Robot Policies"
**URL**: https://arxiv.org/abs/2510.09459 | https://github.com/utiasDSL/fiper

**What they do**:
- Two failure prediction scores: (1) Random Network Distillation in observation embedding space (OOD detection), (2) Action Chunk Entropy score (uncertainty in generated actions)
- Both scores calibrated via CP on a small set of successful rollouts
- Failure alarm when both indicators exceed CP-calibrated thresholds
- Does NOT require failure data

**Relevance**: Shows CP calibration on successful rollouts is practical with small datasets. Their action entropy score is related to what we'd compute for action bounds.

### 1.3 Sample-Efficient Safety Assurances (Luo et al., WAFR 2022)

**Paper**: "Sample-Efficient Safety Assurances using Conformal Prediction"
**URL**: https://arxiv.org/abs/2109.14082

**This is the closest to what we want to do.** Key results:

- **Nonconformity score**: psi(X,Z) = |h(X) - Z| where h is a prediction function
- **Safety definition**: A warning system w is epsilon-safe if P[w(Y)=1 | f(Z) < f_0] >= 1-epsilon
- **Sample complexity**: Need |A| > 1/epsilon - 1 unsafe examples. ~30 samples for 5% FNR
- **Guarantee**: Algorithm achieves epsilon + 1/(1+|A|)-safety under exchangeability
- **Key advantage over PAC**: CP needs Theta(1/epsilon) samples vs Theta(1/epsilon^2) for PAC learning
- **Applied to**: Driver warning (nuScenes, Lyft) and robotic grasping

### 1.4 CPED-NCBFs (2025)

**Paper**: "Conformal Prediction for Expert Demonstration-based Neural Control Barrier Functions"
**URL**: https://arxiv.org/abs/2507.15022

**What they do**:
- Learn Neural Control Barrier Functions (NCBFs) from expert demonstrations
- Use split-conformal prediction to verify the learned NCBF
- CP provides probabilistic safety certificates that are less conservative than SMT solvers or interval bound propagation
- Tradeoff: resilience to safety violations vs probabilistic strength of guarantee

**Relevance**: Directly applicable. Expert demonstrations = our calibration data. CBF verification via CP = our action bound verification.

### 1.5 Adaptive CP for Safety-Critical Control (2024)

**Paper**: "Safety-Critical Control with Uncertainty Quantification using Adaptive Conformal Prediction"
**URL**: https://arxiv.org/abs/2407.03569

- Combines probabilistic CBFs with adaptive CP for stochastic systems
- Quantifies future state uncertainty distribution-free
- Adapts online to actual environmental conditions (not pre-computed)
- Tested on multi-robot systems

### 1.6 Learnable Conformal Prediction (2025)

**Paper**: "Learnable Conformal Prediction with Context-Aware Nonconformity Functions"
**URL**: https://arxiv.org/abs/2509.21955

- Trains a lightweight neural net for the nonconformity function itself
- Maintains CP guarantees while reducing prediction set sizes by 18%
- Path planning: safety improves from 72% to 91.5% with only 4.5% path inflation (vs 12.2% for standard CP)
- Shows that smarter nonconformity scores = tighter bounds without losing coverage

### 1.7 Ensembles of Safety Filters + CP (2025)

**Paper**: "Statistically Assuring Safety of Control Systems using Ensembles of Safety Filters and Conformal Prediction"
**URL**: https://arxiv.org/abs/2511.07899

- Uses CP to bound uncertainty in learned Hamilton-Jacobi value functions
- Two-stage CP framework: calibrates switching between nominal and safe controllers
- Ensemble of independently trained value functions improves robustness

---

## 2. Mathematical Formulation for VLA Action Bounds

### 2.1 Problem Setup

Given:
- A VLA policy pi that maps observation o to action a in R^d (e.g., d=7 for 6-DOF + gripper)
- A calibration set D_cal = {(o_i, a_i^expert, safe_i)}_{i=1}^n of expert demonstrations with safety labels
- Desired coverage level 1-alpha (e.g., 0.95 or 0.99)

Goal: Construct a prediction region C(o) such that:

    P(a^expert in C(o)) >= 1 - alpha

for any new observation o drawn exchangeably with the calibration data.

### 2.2 Split Conformal Prediction (the method that fits)

**Step 1: Split data**
- D_train: used to train the VLA policy pi (already done)
- D_cal: held-out calibration set, NOT used for training

**Step 2: Define nonconformity score**

For VLA action bounds, the natural choices are:

**(a) Residual score (simplest)**:
```
s_i = ||a_i^expert - pi(o_i)||_2
```
This measures how far the expert action is from the VLA's predicted action.

**(b) Normalized residual (heteroscedastic)**:
```
s_i = ||a_i^expert - pi(o_i)||_2 / sigma(o_i)
```
where sigma(o_i) is a learned uncertainty estimate (e.g., from ensemble variance or a separate head). This gives tighter bounds in low-uncertainty regions.

**(c) Mahalanobis distance (correlated actions)**:
```
s_i = sqrt((a_i - mu(o_i))^T Sigma(o_i)^{-1} (a_i - mu(o_i)))
```
where mu(o_i), Sigma(o_i) are predicted mean and covariance. This handles correlations between action dimensions (e.g., joint angles that co-vary).

**(d) Per-dimension scores (what we likely want)**:
```
s_i^j = |a_i^{expert,j} - pi(o_i)^j| / sigma^j(o_i)    for each dimension j
```
Apply CP independently per dimension. Bonferroni correction: use alpha/d per dimension for joint guarantee of 1-alpha.

**Step 3: Compute conformal quantile**

Sort calibration scores: s_(1) <= s_(2) <= ... <= s_(n)

The conformal quantile is:
```
q_hat = s_(ceil((n+1)(1-alpha)/1))
```

That is, take the ceil((n+1)(1-alpha))-th smallest score.

**Step 4: Construct prediction set**

For a new observation o_new:
```
C(o_new) = {a : ||a - pi(o_new)||_2 <= q_hat}                    (L2 ball)
C(o_new) = {a : |a^j - pi(o_new)^j| <= q_hat * sigma^j(o_new)}   (per-dim, normalized)
```

**Coverage guarantee (exact)**:
```
P(a^{n+1} in C(o_{n+1})) >= 1 - alpha
```

This holds for ANY distribution, with NO assumptions beyond exchangeability of calibration and test data.

The guarantee is also bounded above:
```
P(a^{n+1} in C(o_{n+1})) <= 1 - alpha + 1/(n+1)
```

So with n=999 calibration points, you get coverage in [1-alpha, 1-alpha+0.001].

### 2.3 Which CP Variant?

| Method | Pros | Cons | Fits our case? |
|--------|------|------|----------------|
| **Split CP** | Simple, one-pass, no retraining | Wastes some data for calibration | **Yes - primary choice** |
| Full CP | Uses all data, tighter | Retrains model n times (infeasible for VLAs) | No |
| **Adaptive CP** | Adjusts online to distribution shift | More complex, weaker guarantees | Yes - for deployment |
| Conformalized Quantile Regression | Tighter intervals, heteroscedastic | Needs quantile regression model | Maybe - if we train uncertainty head |

**Recommendation**: Start with split CP (simplest, strongest guarantees). Graduate to adaptive CP for online deployment where the distribution shifts over time.

---

## 3. Calibration Set Size Analysis

### 3.1 Minimum Samples

The coverage guarantee holds for ANY n >= 1, but the bound tightness depends on n:

- **Hard minimum**: n >= ceil(1/alpha) - 1 to get a non-trivial quantile
  - alpha=0.05: need n >= 19
  - alpha=0.01: need n >= 99
  - alpha=0.001: need n >= 999

- **Practical minimum** (from Sample-Efficient Safety paper): ~1.5/epsilon - 1
  - For 5% error rate: ~29 samples
  - For 1% error rate: ~149 samples

- **For tight bounds**: n >= 500 gives coverage in [1-alpha, 1-alpha+0.002]

### 3.2 Comparison to Percentile Approach

Using the 99th percentile of expert actions (n=100, p=0.99):

- **No formal guarantee** - the 99th percentile of a finite sample is NOT guaranteed to cover 99% of the population
- **PAC bound**: To guarantee P(coverage >= 1-epsilon) >= 1-delta, you need n >= log(1/delta) / log(1/(1-epsilon)) samples
  - For epsilon=0.01, delta=0.05: n >= 299
  - This is Theta(1/epsilon^2) in general

- **CP advantage**: Needs only Theta(1/epsilon) samples for the same guarantee
- **CP also adapts**: The conformal quantile adjusts to the actual score distribution shape, while a fixed percentile is distribution-agnostic

### 3.3 Is CP Actually Tighter Than 99th Percentile?

**Yes, in two ways**:

1. **Statistically**: CP gives a guarantee at level 1-alpha. The 99th percentile of n samples overshoots (is conservative) by an amount that depends on the tail shape. CP calibrates exactly to achieve the target coverage.

2. **With normalized scores**: If you use s_i = |a_i - pi(o_i)| / sigma(o_i), the CP bound adapts per-observation. In low-uncertainty regions, bounds are tight. In high-uncertainty regions, bounds are wide. A fixed percentile gives the same width everywhere.

3. **Formally**: The CP prediction interval width is at most the width you'd get from the (1-alpha+1/(n+1)) quantile of the population score distribution. The percentile approach overshoots this because it doesn't account for the test point.

**When CP is NOT tighter**: If you have a massive dataset (n >> 1/alpha) and the score distribution is well-behaved, the 99th percentile and CP quantile converge to the same value. CP's advantage is strongest in the small-to-medium data regime (n=50-500).

---

## 4. Defining Safety for VLA Actions

### 4.1 Direct Action Safety

The simplest formulation: an action is "safe" if it stays within bounds derived from expert behavior:

```
safe(a | o) = 1  iff  a in C_alpha(o)
```

where C_alpha(o) is the conformal prediction set calibrated at level alpha.

**Guarantee**: With probability >= 1-alpha, an expert-like action falls within C_alpha(o).

**Interpretation**: This is NOT "any action in the set is safe" - it's "expert actions fall in this set with high probability." If expert behavior IS safe behavior, then the set characterizes safe actions.

### 4.2 Safety via Control Barrier Functions (stronger)

Following CPED-NCBFs:
1. Learn a CBF h(x) from expert demonstrations where h(x) >= 0 means safe
2. Use CP to verify: P(h(x) >= 0 for all x in trajectory) >= 1-alpha
3. The CP-verified safe set is provably larger (less conservative) than SMT-verified sets

### 4.3 What We Should Do (Practical Proposal)

For the VLA-edge safety contract system:

```python
# During calibration (offline, on held-out expert data)
scores = []
for (obs, expert_action) in D_cal:
    predicted_action = vla_policy(obs)
    score = compute_nonconformity(expert_action, predicted_action, obs)
    scores.append(score)

# Compute conformal quantile
scores.sort()
q_hat = scores[math.ceil((len(scores) + 1) * (1 - alpha)) - 1]

# During deployment (online)
def is_action_safe(obs, proposed_action, alpha=0.05):
    predicted = vla_policy(obs)
    score = compute_nonconformity(proposed_action, predicted, obs)
    return score <= q_hat  # True = within conformal bounds
```

The nonconformity function choices, in order of sophistication:
1. L2 distance (isotropic ball)
2. Per-dimension absolute difference (hyperrectangle)
3. Mahalanobis distance (ellipsoid, handles correlations)
4. Learned nonconformity function (Learnable CP paper - tightest)

---

## 5. Gap Analysis - What SAFE/FIPER Do vs What We Need

| Aspect | SAFE/FIPER | Our Need |
|--------|-----------|----------|
| **Output** | Binary failure flag | Continuous action bounds |
| **Score space** | Learned failure score | Action residuals |
| **CP type** | Functional CP (time-varying bands) | Split CP on action space |
| **Calibration data** | Successful rollouts only | Expert demos with safety labels |
| **Guarantee** | "Successful rollouts stay below threshold" | "Expert-like actions fall within bounds" |
| **Use case** | Detect and stop failures | Constrain actions to safe region |

**Key insight**: SAFE answers "is this trajectory going to fail?" We want to answer "is this specific action within safe limits?" These are complementary. You could use both:
- CP action bounds to constrain the VLA output
- SAFE-style failure detection as a trajectory-level backup

---

## 6. Recommended Implementation Plan

### Phase 1: Baseline Split CP (1-2 days)
1. Collect calibration set: run VLA on N episodes, record (obs, predicted_action, expert_action)
2. Compute L2 residual scores
3. Compute conformal quantile at alpha=0.05
4. At inference: clip actions to conformal ball, measure task success

### Phase 2: Normalized Scores (1 day)
1. Train simple uncertainty estimator (ensemble of 3-5 action heads, or MC dropout)
2. Use normalized residuals as scores
3. Compare bound tightness to Phase 1

### Phase 3: Per-Dimension Bounds (1 day)
1. Apply CP independently per action dimension with Bonferroni correction (alpha/d)
2. This gives axis-aligned box constraints instead of L2 balls
3. More interpretable, easier to integrate with joint limits

### Phase 4: Adaptive CP for Deployment (2-3 days)
1. Implement online score tracking
2. Adapt quantile threshold as new observations come in
3. Handle distribution shift between calibration and deployment environments

---

## 7. Key Takeaways

1. **Split CP is the right starting point.** Simple, strong guarantees, well-understood.

2. **You need ~100 calibration episodes for alpha=0.01.** This is very reasonable. Much less than the thousands needed for percentile-based PAC guarantees.

3. **CP IS better than hand-tuned percentiles** because it provides finite-sample, distribution-free coverage guarantees. The 99th percentile of 100 samples has no such guarantee.

4. **The nonconformity score design matters more than the CP method.** Normalized scores (by learned uncertainty) give dramatically tighter bounds than raw residuals.

5. **SAFE and FIPER are complementary, not competing.** They detect trajectory-level failures. We bound action-level safety. Use both.

6. **The literature is mature.** CP for robot safety is well-established (WAFR 2022, NeurIPS 2024-2025, ICLR 2025). We're not inventing something new - we're applying a proven technique to VLA action spaces specifically.

---

## Sources

- [SAFE: Multitask Failure Detection for VLAs](https://arxiv.org/abs/2506.09937) - TRI, NeurIPS 2025
- [SAFE Project Page](https://vla-safe.github.io/)
- [FIPER: Failure Prediction at Runtime](https://arxiv.org/abs/2510.09459) - TUM, NeurIPS 2025
- [Sample-Efficient Safety Assurances using CP](https://arxiv.org/abs/2109.14082) - Luo et al., WAFR 2022
- [CPED-NCBFs: CP for Neural Control Barrier Functions](https://arxiv.org/abs/2507.15022) - 2025
- [CP-NCBF: CP-based Verified Neural CBFs](https://arxiv.org/abs/2503.17395) - 2025
- [Adaptive CP for Safety-Critical Control](https://arxiv.org/abs/2407.03569) - 2024
- [Learnable CP for Robotic Planning](https://arxiv.org/abs/2509.21955) - 2025
- [Ensembles of Safety Filters + CP](https://arxiv.org/abs/2511.07899) - 2025
- [Safe Task Planning with CP for Multi-Robot](https://arxiv.org/abs/2402.15368) - 2024
- [Conformal Prediction Lecture Notes - Ryan Tibshirani](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)
- [Distribution-Free Predictive Inference for Regression](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)
- [Conformalized Quantile Regression - Romano et al.](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf)
- [Learning Robot Safety from Sparse Human Feedback using CP](https://arxiv.org/abs/2501.04823) - 2025
- [Introspective Planning - NeurIPS 2024](https://github.com/kevinliang888/IntroPlan)
- [Formal Verification and Control with CP - IEEE](https://ieeexplore.ieee.org/iel8/5488303/11274416/11274485.pdf)
- [S2-NNDS: Safe and Stable Neural Dynamical Systems](https://arxiv.org/abs/2511.20593)
