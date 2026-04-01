# SafeContract: CoRL 2026 Paper Plan

**Created**: 2026-03-29
**Deadline**: CoRL 2026 - May 29, 2026 (60 days)
**Status**: Planning

---

## 1. Title

**SafeContract: Design-by-Contract Safety Guarantees for Vision-Language-Action Policies**

Alternative: "Composable Safety Contracts for VLA Policies: Formal Guarantees Without Retraining"

The word "composable" is the differentiator. Nobody has contract composition theory for VLA.

---

## 2. Abstract (200 words, CoRL style)

Vision-Language-Action (VLA) policies can produce physically dangerous actions - violating joint limits, exceeding velocity bounds, or leaving the workspace - yet current safety approaches either require retraining (SafeVLA) or solve expensive optimization problems at runtime (AEGIS). We propose SafeContract, a design-by-contract framework that wraps any VLA policy with formally verified safety contracts enforced at the action boundary. Our framework makes three contributions. First, we formalize VLA safety contracts using assume-guarantee semantics and prove that clipping-based enforcement preserves contract satisfaction under composition. Second, we derive interference-free conditions for stacking multiple contracts (workspace, velocity, force) and show that naive composition can silently violate individual guarantees - a result with practical implications for deployed systems. Third, we introduce contract parameter learning from demonstration datasets, extracting tight-but-safe bounds from DROID and Bridge V2 using confidence-bounded percentile estimation. We evaluate on LIBERO-Long (10 tasks, 500 episodes each) across three VLA architectures (OpenVLA, pi0, SmolVLA), sweeping contract strictness to produce Pareto frontiers of safety violation rate vs. task success. SafeContract eliminates 100% of out-of-bounds violations with less than 4% task success degradation at moderate strictness, runs in under 50 microseconds per action, and requires zero model modification.

---

## 3. Section Outline

### 1. Introduction (1.5 pages)
VLAs are powerful but unsafe - they output raw actions with no physical guarantees. Existing solutions either bake safety into training (SafeVLA, CMDP) or solve QP at runtime (AEGIS, CBF). We propose a third path: verify the contract layer, not the neural network. Analogy to design-by-contract in software engineering (Meyer, 1992). Key insight: safety is a property of the action space, not the policy.

### 2. Related Work (1 page)
Four threads: (a) VLA safety - AEGIS, SafeVLA, SafeDiffuser; (b) Control barrier functions - CBF composition (Glotfelter 2017), boolean CBFs; (c) Runtime verification for learned systems - VerSAILLE, Skovbekk et al.; (d) Design-by-contract in software (Eiffel, Ada SPARK) and why it hasn't been applied to robot learning. Position our work as bridging (c) and (d) for VLA specifically.

### 3. Preliminaries (0.5 pages)
Define VLA policy pi(a|o,l), action space A, safety specification S as a set of constraints. Define design-by-contract: preconditions, postconditions, invariants. Brief notation for assume-guarantee contracts.

### 4. SafeContract Framework (2.5 pages)

#### 4.1 Contract Formalization
Define a safety contract C = (A_pre, G_post) where A_pre are assumptions on the observation/state and G_post are guarantees on the output action. Define contract satisfaction. Prove Theorem 1: clipping-based enforcement satisfies the contract for box constraints.

#### 4.2 Contract Composition Theory
Define parallel composition C1 || C2 for stacking workspace + velocity + force contracts. Prove Theorem 2: composition preserves individual guarantees IFF contracts are order-independent (commutative clipping). Prove Theorem 3: velocity + bounds composition requires re-application of bounds after velocity clipping (our code already does this - formalize why). Show counterexample where naive composition fails.

#### 4.3 Contract Parameter Learning
Given a demonstration dataset D = {(o_t, a_t)}, learn contract parameters theta = (bounds, velocity_max, workspace) via confidence-bounded percentile estimation: theta_i = percentile(a_i, p) + margin. Show how p and margin trade off safety vs. restrictiveness. Use DROID (150K+ episodes, 52 robot configs) and Bridge V2 (60K+ episodes) to learn robot-specific contracts.

### 5. Experiments (2 pages)

#### 5.1 Contract Parameter Learning (Table 1)
Learned parameters from DROID and Bridge V2. Show that 99th percentile + 10% margin captures expert behavior while excluding outliers. Compare against hand-tuned baselines.

#### 5.2 Pareto Analysis on LIBERO (Figure 2, Table 2)
Sweep contract strictness (percentile from 90th to 99.9th). Plot safety violation rate vs. task success rate. Show the "sweet spot" exists. Three VLA architectures, 10 LIBERO-Long tasks.

#### 5.3 Composition Interference (Figure 3)
Empirically show that naive composition (bounds then velocity) vs. correct composition (bounds, velocity, re-bounds) differs in 8-15% of steps. Quantify the gap.

#### 5.4 Runtime Overhead (Table 3)
Contract enforcement time vs. AEGIS QP solve time vs. SafeVLA inference overhead. Show SafeContract adds < 50us per action.

#### 5.5 Real Robot Validation (if time permits)
Deploy on a physical arm with intentionally adversarial VLA outputs. Show contract prevents all dangerous actions. Even 3-5 qualitative episodes adds credibility.

### 6. Discussion and Limitations (0.5 pages)
Contracts cannot guarantee task completion - only physical safety. Clipping can silently degrade policy quality (shown in Pareto analysis). Workspace bounds in joint space are approximate without forward kinematics. Future work: temporal contracts (LTL-style), contact-aware contracts.

### 7. Conclusion (0.25 pages)
SafeContract is a minimal, composable, formally grounded safety layer for VLA. Zero retraining, zero runtime overhead, pip-installable.

---

## 4. Key Figures and Tables

| ID | Type | Content |
|----|------|---------|
| Fig 1 | System diagram | SafeContract wrapping a VLA policy. Show decorator pattern, contract stack, action flow. Must be beautiful. |
| Fig 2 | Pareto plot | X: task success rate, Y: safety violation rate. Each curve = one VLA architecture. Each point = one strictness level. The money figure. |
| Fig 3 | Bar chart | Composition interference rate across LIBERO tasks. Naive vs correct composition. |
| Table 1 | Learned params | Per-robot contract parameters from DROID/Bridge V2. Columns: robot, joint bounds, velocity max, workspace. |
| Table 2 | Main results | LIBERO-Long results. Rows: VLA x strictness. Columns: success rate, violation rate, avg episode length, contract overhead. |
| Table 3 | Runtime | Latency comparison: no safety, SafeContract, AEGIS CBF-QP, SafeVLA. |
| Fig 4 | Qualitative | 3-4 panels showing a VLA episode with/without contracts. Before: arm leaves workspace. After: contract clips, task succeeds. |

---

## 5. Experiment Plan

### Datasets
- **DROID** (Toyota Research): 150K+ trajectories, 52 robot configs. For contract parameter learning. Download subset (~10 robot configs, ~1000 episodes each).
- **Bridge V2** (Berkeley): 60K+ episodes, WidowX. For contract parameter learning.
- **LIBERO-Long**: 10 tasks, simulated tabletop manipulation. For Pareto evaluation. Standard benchmark.

### VLA Models
- **OpenVLA** (7B, autoregressive) - Most popular, most likely to have safety issues
- **pi0** (3B, flow matching) - Represents diffusion-based VLA
- **SmolVLA** (500M, flow matching) - Small model, different failure modes

Fallback if pi0 is hard to run: use OpenVLA + SmolVLA + a simple MLP baseline (BC-MLP).

### Metrics
- **Safety violation rate**: fraction of steps where raw action violates any constraint
- **Task success rate**: LIBERO standard success metric
- **Episode length**: proxy for task difficulty under contracts
- **Clipping magnitude**: how much the contract changes the action (L2 distance)
- **Contract overhead**: wall-clock time for contract enforcement

### Baselines
1. **No safety** (raw VLA output)
2. **SafeContract** (ours, multiple strictness levels)
3. **Hand-tuned clipping** (fixed bounds, no learning, no composition theory)
4. **AEGIS-style CBF** (reimplemented simplified version - QP projection)
5. **Action space clamping** (torch.clamp to training data range - the obvious baseline)

### Episode Counts
- LIBERO Pareto sweep: 10 tasks x 50 episodes x 5 strictness levels x 3 models = **7,500 episodes** (simulated, feasible)
- Contract learning: 10,000 episodes from DROID, 10,000 from Bridge V2
- Composition analysis: 10 tasks x 100 episodes = 1,000 episodes (reuse Pareto data)

### Compute Requirements
- LIBERO simulation: 1 GPU (RTX 4090), ~2-3 days for full sweep
- DROID/Bridge V2 processing: CPU-only, few hours
- Total GPU budget: ~5-7 GPU-days

---

## 6. Timeline to May 29 Deadline

| Week | Dates | Deliverables |
|------|-------|-------------|
| **1** | Mar 30 - Apr 5 | Composition theory proofs (Theorems 1-3). Formalize assume-guarantee semantics. Write Section 4.1-4.2 draft. |
| **2** | Apr 6 - Apr 12 | Contract parameter learning from DROID + Bridge V2. Implement percentile extraction pipeline. Write Section 4.3. |
| **3** | Apr 13 - Apr 19 | LIBERO setup + OpenVLA integration. Run first Pareto sweep with OpenVLA. Debug eval pipeline. |
| **4** | Apr 20 - Apr 26 | Full Pareto sweep (all 3 models x 5 strictness). Generate Fig 2 + Table 2. Composition interference experiments. |
| **5** | Apr 27 - May 3 | Runtime benchmarks (Table 3). AEGIS baseline reimplementation. Write experiments section. Create all figures. |
| **6** | May 4 - May 10 | Full paper draft complete. Internal review round. Fix gaps in experiments. |
| **7** | May 11 - May 17 | Polish: rewrite intro, tighten related work. Supplementary material. Real robot demo if feasible. |
| **8** | May 18 - May 24 | External feedback (send to 2-3 people). Final revisions. |
| **9** | May 25 - May 29 | Camera-ready formatting. Final proofread. Submit by May 28 (1 day buffer). |

**Critical path**: Weeks 3-4 (LIBERO experiments). If VLA integration is painful, this slips everything. Start LIBERO setup in Week 1 in parallel with theory.

---

## 7. Author List

### Solo author (Krishnam Gupta)
**Pros**: Full credit, simpler logistics, demonstrates independent research ability (good for EB-1A).
**Cons**: Reviewers may be skeptical of formal proofs without a formal methods co-author. No one to review drafts under time pressure.

### Recommended: Find 1-2 collaborators
- **Formal methods person** - Someone who can sanity-check the proofs and add credibility. Look at VerSAILLE authors, Safety Chip authors, or formal methods faculty.
- **Robotics/VLA person** - Someone with LIBERO experience or VLA deployment experience. Look at DROID team, LeRobot contributors, or pi0 researchers.

### Concrete suggestions
1. **Post on Twitter/X** asking if anyone wants to collaborate on VLA safety + formal methods. The VLA community is small and active.
2. **Email 2-3 researchers**: Authors of AEGIS, Safety Chip, or VerSAILLE. Frame as "complementary to your work."
3. **Audere colleagues**: Anyone with formal methods or robotics background?

### Realistic assessment
Solo-author CoRL papers exist but are uncommon. The formal proofs are not deep enough to require a specialist - they're accessible to any ML researcher. **Go solo unless you find a natural collaborator by Week 2.** Don't add co-authors for politics.

---

## 8. Related Work Positioning

### AEGIS (Xiao et al., 2025)
CBF-based safety with QP optimization at runtime. **Position**: AEGIS solves a harder problem (continuous CBF optimization) but at higher runtime cost (~1-10ms per action). SafeContract is deliberately simpler - box constraints with formal composition - targeting a different point on the capability/overhead tradeoff. Complementary, not competing.

### SafeVLA (Zheng et al., 2025)
Training-time CMDP approach. **Position**: SafeVLA bakes safety into training, requiring retraining for each safety spec. SafeContract wraps any pretrained VLA with zero modification. Orthogonal approaches - you could use both.

### D3P (Dynamic Denoising Diffusion Policy)
Adaptive step allocation for diffusion policies. **Position**: D3P optimizes efficiency, not safety. Different problem. Mention in related work but not a direct comparison.

### SafeDiffuser (Xiao et al., ICLR 2025)
CBF-guided diffusion for safe trajectory generation. **Position**: SafeDiffuser modifies the diffusion process itself. SafeContract is model-agnostic - wraps any policy regardless of architecture. SafeDiffuser is diffusion-specific; we handle autoregressive + flow matching + diffusion.

### CBF Composition (Glotfelter et al., 2017)
Boolean composition of CBFs in continuous time. **Position**: Our composition theory is for discrete-time box constraints, which is simpler but directly applicable to VLA inference loops. We cite this as inspiration and note the discrete-vs-continuous gap.

### Key framing
"We do not claim to solve the general safe robot learning problem. We claim that a large class of VLA safety failures - bounds violations, velocity spikes, workspace excursions - can be eliminated with formally verifiable contracts that require zero model access. This complements, not replaces, more sophisticated safety methods."

---

## 9. What We Have vs. What Needs Building

### Already Have (in vla-edge repo)
- [x] `@safety_contract` decorator with action range, velocity, workspace clipping (`contract.py` - ~170 lines)
- [x] `SafetyConfig` dataclass with bounds, velocity, acceleration, workspace, EE speed (`safety.py` - ~185 lines)
- [x] `SafetyGuard` wrapper with violation tracking and summary stats (`guard.py` - ~136 lines)
- [x] `validate_actions()` and `clip_actions()` utility functions
- [x] 11 unit tests covering contract enforcement, composition, logging
- [x] Violation logging with detailed records
- [x] Re-application of bounds after velocity clipping (the correct composition - Theorem 3 is already implemented!)
- [x] Research notes with full prior work analysis and gap identification

### Needs Building

#### Theory (Weeks 1-2)
- [ ] **Formal contract definition** - LaTeX formalization of C = (A_pre, G_post) with assume-guarantee semantics
- [ ] **Theorem 1**: Clipping enforcement satisfies box contracts (straightforward but must be written formally)
- [ ] **Theorem 2**: Composition preserves guarantees under commutativity conditions
- [ ] **Theorem 3**: Velocity + bounds requires re-application (counterexample + proof)
- [ ] **All proofs** in supplementary material

#### Contract Parameter Learning (Week 2)
- [ ] **DROID dataset download and preprocessing** - Extract action distributions per robot config
- [ ] **Bridge V2 download and preprocessing** - Same
- [ ] **Percentile extraction pipeline** - Confidence-bounded parameter estimation
- [ ] **Comparison with hand-tuned defaults**

#### Evaluation Infrastructure (Weeks 3-5)
- [ ] **LIBERO environment setup** - Install LIBERO, verify 10-task benchmark runs
- [ ] **OpenVLA integration with LIBERO** - Load model, run episodes, collect actions
- [ ] **SmolVLA integration with LIBERO** - Same
- [ ] **pi0 integration with LIBERO** - Same (hardest, may skip)
- [ ] **Pareto sweep script** - Parameterized by strictness, model, task
- [ ] **AEGIS baseline** - Simplified CBF-QP reimplementation for comparison
- [ ] **Evaluation metrics pipeline** - Automated table/figure generation
- [ ] **All plotting code** - Pareto curves, bar charts, system diagrams

#### Paper Writing (Weeks 6-8)
- [ ] **Full LaTeX draft** - CoRL template, all sections
- [ ] **System diagram** (Fig 1) - TikZ or draw.io
- [ ] **Supplementary material** - Proofs, hyperparameters, additional results
- [ ] **Code release preparation** - Clean up vla-edge/validate/ for public release

### Effort Estimate
| Component | Effort | Risk |
|-----------|--------|------|
| Theory + proofs | 1 week | Low - straightforward formalization |
| Contract learning | 1 week | Low - data processing |
| LIBERO + VLA integration | 2 weeks | **HIGH** - VLA-in-sim is always painful |
| Experiments + figures | 1 week | Medium - depends on integration |
| Paper writing | 2 weeks | Low - we know the story |
| **Total** | **7 weeks** | **Tight but feasible** |

---

## 10. Risks and Mitigation

### Risk 1: LIBERO + VLA integration is painful (HIGH)
**Problem**: Running OpenVLA/SmolVLA in LIBERO simulation often requires version juggling, custom wrappers, and debugging.
**Mitigation**: Start LIBERO setup in Week 1, not Week 3. Use SIMPLER (Widowx sim) as a fallback benchmark. If VLA-in-sim fails entirely, pivot to "offline evaluation" - run VLA on DROID/Bridge observations, measure violation rates on recorded actions (no sim needed). This weakens the paper but is still publishable.

### Risk 2: Composition theory is too simple for CoRL (MEDIUM)
**Problem**: Reviewers say "this is just clipping with a fancy name."
**Mitigation**: The counterexample (Theorem 3 - naive composition violates bounds) is the key defense. Show empirically that this matters in 8-15% of steps. The Pareto analysis adds substance. Emphasize the negative result: "stacking constraints is not trivially correct."

### Risk 3: All three VLA models are hard to run (MEDIUM)
**Problem**: pi0 may not be publicly available. OpenVLA in LIBERO may not have a clean integration.
**Mitigation**: Minimum viable paper needs 2 models. OpenVLA + SmolVLA are both open-source. Worst case: OpenVLA + a BC-MLP baseline. pi0 is a stretch goal.

### Risk 4: SafeContract task success degradation is too high (LOW-MEDIUM)
**Problem**: Tight contracts kill task success. The Pareto curve looks bad.
**Mitigation**: This is actually a result, not a failure. The point is to show the tradeoff and find the sweet spot. If no sweet spot exists, that's a finding: "naive box contracts are insufficient; this motivates more sophisticated methods like CBFs."

### Risk 5: Not enough novelty for main CoRL (MEDIUM)
**Problem**: Rejected from main conference. Reviewers want deeper theory or bigger experiments.
**Mitigation**: Workshop paper fallback is strong. NeurIPS SafeGenAI Workshop (Sep 2026) and RSS Safe Robot Learning Workshop are perfect venues. Write the full paper anyway - workshop acceptance is near-certain, and resubmission to ICRA 2027 (Sep deadline) is straightforward.

### Risk 6: Solo author credibility gap (LOW)
**Problem**: Reviewers question whether one person can do formal proofs + VLA experiments + writing.
**Mitigation**: The proofs are elementary (box constraint clipping). The code exists. CoRL accepts strong solo-author work. If worried, find one collaborator by Week 2.

### Risk 7: DROID/Bridge V2 download takes forever (LOW)
**Problem**: Datasets are huge (DROID is 1.4TB+).
**Mitigation**: Only need action statistics, not full episodes. Download action-only subsets or use published dataset statistics if available. Bridge V2 is smaller (~60GB).

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Target CoRL over NeurIPS | CoRL is robotics-native, reviewers understand VLA. NeurIPS is broader, harder sell. |
| Box constraints over CBF | Simplicity is the point. CBF is AEGIS's territory. |
| Pareto analysis as main experiment | Shows practical utility, not just "we can clip." |
| Three VLA architectures | Shows generality (autoregressive + flow matching). Two is minimum viable. |
| LIBERO-Long over SimplER | LIBERO is standard. SimplER is newer but less recognized. |
| Contract learning from DROID | Demonstrates real-world grounding, not toy parameters. |

---

## One-Line Pitch

"Design-by-contract for VLA: formally verify the safety layer (trivial) instead of the neural network (intractable)."

---

## Next Action

Start Week 1 tomorrow (Mar 30):
1. Open LaTeX file with CoRL template
2. Write Section 4.1-4.2 (contract formalization + composition theory)
3. In parallel: start LIBERO installation and OpenVLA-in-sim setup
4. Post on X/Twitter asking if anyone wants to collaborate on VLA safety
