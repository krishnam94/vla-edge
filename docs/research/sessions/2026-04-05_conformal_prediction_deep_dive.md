# Deep Research: Conformal Prediction in Robotics and VLA Systems

**Date**: 2026-04-05
**Purpose**: Comprehensive survey of CP + robotics/VLA literature to assess novelty of "conformal p-values + CUSUM for VLA action monitoring"
**Queries run**: 15+ across Google Scholar, arXiv, ICLR 2026, ICML 2025

---

## EXECUTIVE SUMMARY

**Key finding: "Conformal p-values + CUSUM for VLA action monitoring" is genuinely novel.** No paper combines all three elements: (1) conformal p-values on VLA action residuals, (2) CUSUM-based sequential change detection, (3) applied to VLA policy monitoring at deployment time. The closest competitors are WATCH (conformal martingales for general AI monitoring, ICML 2025), Vovk et al. (conformal CUSUM theory, no robotics), and SAFE/FIPER (CP for VLA failure detection, but not p-value + CUSUM). Our specific combination is novel.

**However, the components are well-established individually.** A reviewer could argue we're "just combining known tools." The novelty argument must emphasize: (a) the multi-scale temporal structure (step/chunk/episode CUSUM), (b) application to VLA action spaces specifically, and (c) the integration with safety contracts.

---

## 1. PAPERS ANALYZED (Organized by Relevance)

### TIER 1: CLOSEST COMPETITORS (Must cite and differentiate)

#### 1.1 WATCH: Weighted-Conformal Martingales (ICML 2025) -- CLOSEST OVERALL
- **Title**: WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales
- **Authors**: Drew Prinster, Xing Han, Anqi Liu, Suchi Saria
- **Venue**: ICML 2025
- **Link**: [arXiv:2505.04608](https://arxiv.org/abs/2505.04608)
- **How they use CP**: Propose weighted conformal test martingales (WCTMs) for online monitoring. Extend conformal test martingales with weighting to adapt to mild covariate shifts.
- **Change detection**: Martingale-based sequential testing (not CUSUM). Anytime-valid inference.
- **Domain**: General AI/ML deployment monitoring, not robotics-specific.
- **Overlap with SafeContract**: Both do online monitoring with CP-based statistics. Both detect distribution shift.
- **Key difference**: WATCH is model-agnostic (any ML system). We are VLA-specific (action residuals, multi-scale temporal). WATCH uses martingales; we use CUSUM. WATCH cannot diagnose which action dimensions are drifting; our per-dimension scores can.
- **Does this hurt novelty?** MODERATELY. Shows the general idea of "CP for deployment monitoring" is well-established. We must position as VLA-specific instantiation with action-space structure.

#### 1.2 SAFE: Multitask Failure Detection for VLAs (NeurIPS 2025) -- CLOSEST IN DOMAIN
- **Title**: SAFE: Multitask Failure Detection for Vision-Language-Action Models
- **Authors**: Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, Florian Shkurti
- **Venue**: NeurIPS 2025
- **Link**: [arXiv:2506.09937](https://arxiv.org/abs/2506.09937)
- **How they use CP**: Functional conformal prediction (one-sided time-varying CP band) to set failure detection thresholds. CP calibrates a learned failure score, not raw action residuals.
- **Nonconformity score**: Learned failure score from VLA internal features (not action residuals).
- **VLA models tested**: OpenVLA, pi0, pi0-FAST
- **Overlap**: Both monitor VLA policies at runtime. Both use CP for calibration.
- **Key difference**: SAFE detects trajectory-level failure (binary: will this fail?). We detect action-level anomaly (continuous: how anomalous is this action?). SAFE requires training a failure detector on VLA features. We are training-free (just action residuals). SAFE is complementary - you could use both.
- **Does this hurt novelty?** LOW. Different mechanism (learned score vs action residuals), different granularity (trajectory vs action), different requirement (training vs training-free).

#### 1.3 FIPER: Failure Prediction at Runtime (NeurIPS 2025)
- **Title**: Failure Prediction at Runtime for Generative Robot Policies
- **Authors**: Ralf Romer, Adrian Kobras, Luca Worbis, Angela P. Schoellig
- **Venue**: NeurIPS 2025
- **Link**: [arXiv:2510.09459](https://arxiv.org/abs/2510.09459)
- **How they use CP**: Calibrate failure prediction scores (OOD via random network distillation + action chunk entropy) using CP on successful rollouts. Two scores aggregated over short time windows.
- **CP variant**: Split conformal for threshold calibration.
- **Overlap**: Both use CP calibration on demonstration data. Both detect anomalies at runtime. FIPER's "action chunk entropy" is related to our action residual scores.
- **Key difference**: FIPER trains an RND network (requires training). We use raw action residuals (training-free). FIPER uses binary thresholding; we use p-values + CUSUM for sequential detection. FIPER does not do change-point detection.
- **Does this hurt novelty?** LOW. Different technical approach. Our multi-scale CUSUM is more principled for sequential detection than their fixed threshold.

#### 1.4 Conformal CUSUM (Vovk et al., 2025) -- CLOSEST METHODOLOGICALLY
- **Title**: Validity and Efficiency of the Conformal CUSUM Procedure
- **Authors**: Vladimir Vovk, Ilia Nouretdinov, Alex Gammerman
- **Venue**: PMLR vol 266 (ALT 2025)
- **Link**: [arXiv:2412.03464](https://arxiv.org/abs/2412.03464)
- **How they use CP+CUSUM**: Define a conformal version of CUSUM for change detection. Test the IID null hypothesis repeatedly. Establish validity (controlled false alarm) and efficiency (bounded detection delay) properties.
- **Domain**: Pure theory / change detection. NOT applied to robotics.
- **Overlap**: We literally implement conformal + CUSUM. This paper provides the theoretical foundation.
- **Key difference**: They provide the theory; we provide the robotics application. They test IID assumption; we test "VLA policy is behaving normally." They have no multi-scale structure.
- **Does this hurt novelty?** HELPS. We should cite this as theoretical foundation. Shows our approach has rigorous underpinning. The APPLICATION to VLA is novel, not the method itself.

### TIER 2: IMPORTANT RELATED WORK (Should cite)

#### 2.1 Conformal Decision Theory (Angelopoulos et al., 2023)
- **Title**: Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions
- **Authors**: Jordan Lekeufack, Anastasios N. Angelopoulos, Andrea Bajcsy, Michael I. Jordan, Jitendra Malik
- **Venue**: arXiv:2310.05921 (major venues pending)
- **Link**: [arXiv:2310.05921](https://arxiv.org/abs/2310.05921)
- **How applicable**: Highly. CDT calibrates decisions (not predictions) with distribution-free guarantees. Applied to robot navigation around humans. Could calibrate VLA action decisions.
- **Key insight**: "Calibrate decisions directly, without requiring prediction sets."
- **Difference from our approach**: CDT modifies the decision/action. We monitor it. CDT is offline calibration; our CUSUM is online detection. Complementary.

#### 2.2 Conformal Policy Learning (Huang et al., 2023)
- **Title**: Conformal Policy Learning for Sensorimotor Control Under Distribution Shifts
- **Authors**: Huang Huang, Satvik Sharma, Antonio Loquercio, Anastasios Angelopoulos, Ken Goldberg, Jitendra Malik
- **Venue**: arXiv:2311.01457
- **Link**: [arXiv:2311.01457](https://arxiv.org/abs/2311.01457)
- **How they use CP**: Conformal quantiles as input to switching policies. Robot detects distribution shift and switches between base policies.
- **Difference**: They switch policies. We monitor a single policy. They use conformal quantiles, not p-values. No CUSUM-style sequential detection.

#### 2.3 SoNIC: Safe Social Navigation with ACI (TASL Lab, 2024-2025)
- **Title**: SoNIC: Safe Social Navigation with Adaptive Conformal Inference and Constrained Reinforcement Learning
- **Authors**: Jaeuk Shin et al.
- **Venue**: arXiv:2407.17460, accepted at major venue
- **Link**: [arXiv:2407.17460](https://arxiv.org/abs/2407.17460)
- **How they use ACI**: Quantify pedestrian trajectory prediction uncertainty. ACI provides area with pre-defined probability that humans will appear. Fed into constrained RL.
- **Domain**: Social navigation only. Not manipulation or VLA.
- **Difference**: ACI adapts uncertainty bounds for planning. We use ACI (Gibbs & Candes 2021) for monitoring. Different use case. Navigation vs manipulation.

#### 2.4 Conformal Safety Monitoring for Flight Testing (Stanford ASL, ICRA 2025 WS)
- **Title**: Conformal Safety Monitoring for Flight Testing: A Case Study in Data-Driven Safety Learning
- **Authors**: Aaron O. Feldman, D. Isaiah Harp, Joseph Duncan, Mac Schwager
- **Venue**: ICRA 2025 Workshop on Robot Safety Under Uncertainty
- **Link**: [arXiv:2511.20811](https://arxiv.org/abs/2511.20811)
- **How they use CP**: Nearest-neighbor classification of predicted state safety, calibrated via CP. Learn abort criteria for flight maneuvers.
- **Difference**: Flight domain, state-based (not action-based), no sequential change detection.

#### 2.5 Conformal Changepoint Localization (Ramdas group, 2025-2026)
- **Title**: Offline Changepoint Localization Using a Matrix of Conformal P-values
- **Authors**: Sanjit Dandapanthula, Aaditya Ramdas
- **Venue**: arXiv:2505.00292 (revised Feb 2026)
- **Link**: [arXiv:2505.00292](https://arxiv.org/abs/2505.00292)
- **How they use CP**: Matrix of conformal p-values for offline changepoint localization. Prove a conformal Neyman-Pearson lemma.
- **Difference**: OFFLINE changepoint localization, not online detection. Theory paper. Not robotics.

#### 2.6 Conditional Conformal Test Martingales (Romano group, 2026)
- **Title**: Testing For Distribution Shifts with Conditional Conformal Test Martingales
- **Authors**: Shalev Shaer, Yarin Bar, Drew Prinster, Yaniv Romano
- **Venue**: arXiv:2602.13848
- **Link**: [arXiv:2602.13848](https://arxiv.org/abs/2602.13848)
- **How they use CP**: Sequential distribution-shift detection. Compare new samples to fixed reference dataset. Anytime-valid type-I error control.
- **Difference**: General methodology, not robotics-specific. Fixed reference set (our calibration set). Complements our approach theoretically.

### TIER 3: BROADER CONTEXT (Cite as needed)

#### 3.1 Learnable Conformal Prediction for Robotic Planning (2025)
- **Link**: [arXiv:2509.21955](https://arxiv.org/abs/2509.21955)
- **Key insight**: Neural nonconformity functions give 18% tighter sets, improve path planning safety 72% to 91.5%.
- **Relevance**: Could improve our nonconformity score design. Currently we use simple L-inf normalized residuals.

#### 3.2 Sample-Efficient Safety Assurances (Stanford ASL, WAFR 2022)
- **Link**: [arXiv:2109.14082](https://arxiv.org/abs/2109.14082)
- **Key insight**: CP needs O(1/epsilon) samples vs O(1/epsilon^2) for PAC. Applied to driver warning and grasping.
- **Relevance**: Foundational paper for CP in robot safety. We already cite it.

#### 3.3 Safe Planning with Robust CP and Generative Priors (2026)
- **Link**: [arXiv:2602.12616](https://arxiv.org/abs/2602.12616)
- **Key insight**: Conditional diffusion model + robust CP handles environment shifts. Plans with probabilistic safety.
- **Relevance**: Shows CP adapting to distribution shift is active area. Navigation domain.

#### 3.4 CPED-NCBFs: CP for Neural Control Barrier Functions (2025)
- **Link**: [arXiv:2507.15022](https://arxiv.org/abs/2507.15022)
- **Key insight**: CP verifies learned CBFs. Probabilistic safety certificates from demonstrations.
- **Relevance**: Different mechanism (CBF verification vs action monitoring).

#### 3.5 Egocentric CP for Safe Navigation (2025)
- **Link**: [arXiv:2504.00447](https://arxiv.org/abs/2504.00447)
- **Key insight**: Egocentric score functions that measure how much closer obstacles are than anticipated. Integrated into MPC.
- **Relevance**: Smart nonconformity score design for navigation. Inspiration for action-space scores.

#### 3.6 Adaptive CP for Safety-Critical Control (IEEE, 2024-2025)
- **Link**: [arXiv:2407.03569](https://arxiv.org/abs/2407.03569)
- **Key insight**: Probabilistic CBFs + adaptive CP for multi-robot systems. Online adaptation.
- **Relevance**: Adaptive CP in robot safety is established.

#### 3.7 Path-Consistent Safety Filtering (ICRA 2026)
- **Title**: From Demonstrations to Safe Deployment: Path-Consistent Safety Filtering for Diffusion Policies
- **Authors**: Romer, Balletshofer, Thumm, Pavone, Schoellig, Althoff
- **Venue**: ICRA 2026
- **Link**: [arXiv:2511.06385](https://arxiv.org/abs/2511.06385)
- **Key insight**: Reachability-based safety filtering outperforms CBFs by 68% in task success for diffusion policies. Does NOT use CP.
- **Relevance**: Shows the space of "safety for diffusion policies" is active but reachability-based, not CP-based.

#### 3.8 Conformal STL Shield (ICASSP 2026)
- **Link**: [arXiv:2602.14322](https://arxiv.org/abs/2602.14322)
- **Key insight**: CP + STL monitoring shield for RL. Robust CP calibrates short-horizon predictor.
- **Relevance**: CP for runtime safety shielding in RL. F-16 domain, not manipulation.

#### 3.9 Calibrated Safety Prediction for Image-Controlled Autonomy (2025)
- **Link**: [arXiv:2508.09346](https://arxiv.org/abs/2508.09346)
- **Key insight**: VAE + conformal calibration for safety chance prediction from images. Domain adaptation for shift.
- **Relevance**: Vision-based safety prediction with CP, but prediction, not monitoring.

#### 3.10 Adaptive Conformal Anomaly Detection (ICLR 2026)
- **Venue**: ICLR 2026 poster
- **Link**: [OpenReview](https://openreview.net/forum?id=7uFbs68MSI)
- **Key insight**: Post-hoc adaptive CP anomaly detection for time series using foundation models. Weighted quantile CP bounds. Model-agnostic.
- **Relevance**: General time-series anomaly detection with CP. Could be applied to action time series. But they don't do it.

---

## 2. NOVELTY ASSESSMENT

### What has been done (independently):
1. **Conformal prediction for robot safety** - Well-established (ASL 2022, multiple 2024-2025 papers)
2. **CP for VLA failure detection** - SAFE, FIPER (NeurIPS 2025)
3. **Conformal CUSUM for change detection** - Vovk et al. 2025 (theory)
4. **Conformal martingales for AI monitoring** - WATCH (ICML 2025)
5. **Adaptive conformal inference** - Gibbs & Candes 2021, many follow-ups
6. **CP for robot planning/navigation** - SoNIC, Egocentric CP, S-ATLAS, etc.

### What has NOT been done (our gap):
1. **Conformal p-values on VLA action residuals** - Nobody computes conformal p-values directly on (action - prediction) residuals for VLA models
2. **CUSUM on conformal p-values in robotics** - Conformal CUSUM exists in theory (Vovk), but nobody has applied it to robot policy monitoring
3. **Multi-scale temporal CUSUM for VLA** - Our step/chunk/episode architecture is novel
4. **Training-free VLA anomaly detection via conformal p-values** - SAFE/FIPER require training auxiliary networks; we don't
5. **Integration of CP monitoring with safety contracts** - Nobody connects conformal bounds to DBC-style safety contracts

### Novelty rating: MODERATE-TO-STRONG

The specific combination is novel. The individual components are not. This is typical of applied CP papers - the method is well-understood; the application is new.

---

## 3. THREAT ANALYSIS: Could Someone Scoop Us?

### High risk of overlap:
- **SAFE team** (TRI/U of T) could easily add CUSUM to their failure detector. They have the VLA infrastructure.
- **WATCH team** (Johns Hopkins) could instantiate WCTMs for VLA monitoring. They have the monitoring theory.
- **Vovk's group** could apply conformal CUSUM to any domain.

### Why we're still safe (for now):
- SAFE/FIPER focus on failure prediction (binary), not anomaly scoring (continuous). Different problem framing.
- WATCH is domain-agnostic and unlikely to specialize to VLAs.
- Vovk's group is pure theory; they don't do robotics.
- The VLA-specific multi-scale structure requires domain knowledge they'd need to develop.

### Time pressure: MODERATE
The space is active. ICML 2025 + NeurIPS 2025 papers show 6-month iteration cycles. We should submit to CoRL 2026 (deadline May 29) or risk being scooped by SAFE v2 or WATCH v2.

---

## 4. WHAT WE CAN LEARN FROM THESE PAPERS

### For our nonconformity score design:
- **Learnable CP** (arXiv:2509.21955): Train a lightweight neural nonconformity function. Could give tighter bounds than our current L-inf normalized residuals. 18% tighter sets.
- **FIPER**: Action chunk entropy as a score. We could add entropy alongside residuals.
- **Egocentric CP**: Task-relevant scores (only measure safety-critical deviations, not all deviations).

### For our CUSUM design:
- **Vovk et al.**: Cite for theoretical validity of conformal CUSUM.
- **WATCH**: Cite for martingale alternative. Our CUSUM is simpler; their martingales are more flexible. Position as complementary.
- **Conformal changepoint localization** (Ramdas): Offline version. Could be used for post-hoc analysis of episodes.

### For our adaptive monitoring:
- **ACI (Gibbs & Candes 2021)**: Already implemented in our AdaptiveConformalMonitor. Cite.
- **SoNIC**: Shows ACI works in robotics (navigation). We extend to manipulation/VLA.
- **Conditional CTMs** (Romano 2026): Their fixed-reference approach avoids contamination. Could improve our calibration robustness.

### For the paper positioning:
- **Conformal Decision Theory**: Position as "CDT calibrates decisions, we monitor them." Complementary framing.
- **PACS (ICRA 2026)**: They filter actions via reachability. We monitor actions via CP. Different mechanism, same goal.
- **Modular Guardrails (arXiv:2602.04056)**: Our CP monitor is precisely the "monitoring layer" they call for.

---

## 5. COMPANY/INDUSTRY USE OF CP FOR ROBOT SAFETY

### Toyota Research Institute (TRI):
- Published SAFE (NeurIPS 2025). CP for VLA failure detection. Most advanced industrial application.
- Working with OpenVLA, pi0, pi0-FAST. Real robot experiments.

### Stanford ASL (Pavone lab, now partly at NVIDIA):
- Founded the field with "Sample-Efficient Safety Assurances" (2022).
- New in 2026: Path-Consistent Safety Filtering (PACS) for diffusion policies (ICRA 2026). Reachability-based, not CP.
- Not pushing CP further themselves; their students are.

### Google DeepMind:
- No public CP work for robot safety found. Gemini Robotics focuses on foundation model capabilities, not formal safety.

### Other companies:
- No evidence of Agility, Figure, Boston Dynamics, 1X using CP for robot safety.
- Industrial CP use is growing in manufacturing QC (conformal segmentation for defect detection).

### Assessment: CP for robot safety is academic-led, not industry-led (except TRI). This means the space is still open for methodological contributions.

---

## 6. ANSWERS TO KEY QUESTIONS

### Q1: Is "conformal p-values + CUSUM for VLA action monitoring" genuinely novel?
**YES**, in combination. The individual pieces exist:
- Conformal p-values: standard (Vovk, Shafer, Gammerman)
- CUSUM: classical (Page 1954)
- Conformal CUSUM: Vovk et al. 2025 (theory)
- VLA monitoring: SAFE, FIPER (2025)

But nobody has put conformal p-values on VLA action residuals into a CUSUM detector with multi-scale temporal structure. The specific application + the multi-scale design are novel.

### Q2: What is the closest competitor after WATCH?
1. **WATCH** (ICML 2025) - general AI monitoring with conformal martingales
2. **Vovk et al.** (ALT 2025) - conformal CUSUM theory
3. **SAFE** (NeurIPS 2025) - CP for VLA failure detection (different mechanism)
4. **FIPER** (NeurIPS 2025) - CP for generative policy failure prediction

### Q3: Does this make our work MORE or LESS novel?
**More novel by differentiation, less novel in isolation.** The fact that CP for robot safety is well-studied (Tier 3 papers) means reviewers will expect us to know the literature. But the specific VLA + CUSUM + multi-scale combination doesn't exist. We must frame as "principled instantiation for VLA monitoring" not "invention of CP for robotics."

### Q4: What should we cite?
Must cite: WATCH, SAFE, FIPER, Vovk conformal CUSUM, Gibbs & Candes ACI, Sample-Efficient Safety (ASL 2022)
Should cite: SoNIC, Conformal Decision Theory, Conformal STL Shield, Modular Guardrails
Could cite: Learnable CP, CPED-NCBFs, Egocentric CP, PACS

---

## 7. RECOMMENDATIONS FOR SAFECONTRACT/CORL PAPER

### Positioning statement (draft):
"Conformal prediction has been widely applied to robot safety in navigation (SoNIC, Egocentric CP), planning (S-ATLAS, CDT), and failure detection (SAFE, FIPER). Conformal CUSUM procedures have been studied theoretically (Vovk et al. 2025), and conformal martingales have been proposed for general AI monitoring (WATCH, ICML 2025). However, no prior work combines conformal p-values on VLA action residuals with multi-scale CUSUM detection for runtime VLA monitoring. Our contribution instantiates this combination, providing training-free, calibration-efficient monitoring with formal false alarm control."

### Key differentiation table for related work:

| Method | CP Type | Detection | Domain | Training |
|--------|---------|-----------|--------|----------|
| SAFE | Functional CP | Threshold on learned score | VLA | Requires failure detector training |
| FIPER | Split CP | Threshold on RND + entropy | Diffusion policy | Requires RND training |
| WATCH | Weighted CTM | Martingale | Any AI | Model-agnostic |
| Vovk CUSUM | Conformal CUSUM | CUSUM | Theory | N/A |
| SoNIC | ACI | None (feeds into CRL) | Navigation | Policy training |
| **Ours** | **Split CP + ACI** | **Multi-scale CUSUM** | **VLA actions** | **Training-free** |

### Strongest novelty claims:
1. First training-free conformal anomaly detection for VLA action spaces
2. First multi-scale CUSUM (step/chunk/episode) for robot policy monitoring
3. First integration of conformal monitoring with design-by-contract safety systems
4. Formal false alarm control (from conformal CUSUM theory) applied to robot deployment

### Weakest novelty claims (avoid):
1. "First use of CP in robotics" (false - ASL 2022)
2. "First conformal CUSUM" (false - Vovk 2025)
3. "First CP for VLA" (false - SAFE, FIPER 2025)

---

## Sources

### Tier 1 (Must Cite)
- [WATCH: Weighted-Conformal Martingales](https://arxiv.org/abs/2505.04608) - Prinster et al., ICML 2025
- [SAFE: Multitask Failure Detection for VLAs](https://arxiv.org/abs/2506.09937) - Gu et al., NeurIPS 2025
- [FIPER: Failure Prediction at Runtime](https://arxiv.org/abs/2510.09459) - Romer et al., NeurIPS 2025
- [Conformal CUSUM Validity and Efficiency](https://arxiv.org/abs/2412.03464) - Vovk et al., ALT 2025

### Tier 2 (Should Cite)
- [Conformal Decision Theory](https://arxiv.org/abs/2310.05921) - Lekeufack et al., 2023
- [Conformal Policy Learning](https://arxiv.org/abs/2311.01457) - Huang et al., 2023
- [SoNIC: Adaptive Conformal for Social Navigation](https://arxiv.org/abs/2407.17460) - Shin et al., 2024-2025
- [Conformal Safety Monitoring for Flight](https://arxiv.org/abs/2511.20811) - Feldman et al., ICRA 2025 WS
- [Conformal Changepoint Localization](https://arxiv.org/abs/2505.00292) - Dandapanthula & Ramdas, 2025-2026
- [Conditional Conformal Test Martingales](https://arxiv.org/abs/2602.13848) - Shaer et al., 2026
- [Sample-Efficient Safety Assurances](https://arxiv.org/abs/2109.14082) - Luo et al., WAFR 2022 / IJRR 2024
- [ACI: Adaptive Conformal Inference](https://arxiv.org/abs/2106.00170) - Gibbs & Candes, NeurIPS 2021

### Tier 3 (Could Cite)
- [Learnable CP for Robotic Planning](https://arxiv.org/abs/2509.21955) - 2025
- [CPED-NCBFs](https://arxiv.org/abs/2507.15022) - 2025
- [Egocentric CP for Navigation](https://arxiv.org/abs/2504.00447) - 2025
- [Safe Planning with Robust CP](https://arxiv.org/abs/2602.12616) - 2026
- [PACS: Safety Filtering for Diffusion Policies](https://arxiv.org/abs/2511.06385) - ICRA 2026
- [Conformal STL Shield](https://arxiv.org/abs/2602.14322) - ICASSP 2026
- [Calibrated Safety for Image-Controlled Autonomy](https://arxiv.org/abs/2508.09346) - 2025
- [Adaptive CP Anomaly Detection](https://openreview.net/forum?id=7uFbs68MSI) - ICLR 2026
- [Adaptive CP for Safety-Critical Control](https://arxiv.org/abs/2407.03569) - 2024
- [Ensembles of Safety Filters + CP](https://arxiv.org/abs/2511.07899) - 2025
- [Safe Planning Interactive Environments](https://arxiv.org/abs/2511.10586) - 2025
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814) - ICLR 2024
