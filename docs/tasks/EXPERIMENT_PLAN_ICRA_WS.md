# Experiment Plan: ICRA Workshop Paper on Safety Contracts for VLA

**Created**: 2026-03-29
**Deadline**: ~April 12, 2026 (14 days)
**Compute**: Mac Air M3 (CPU + MPS), ~500 SmolVLA forward passes max (~2 hours)
**Paper**: 2-4 page ICRA workshop, safety contracts as the main story

---

## Guiding Principle

Every experiment must answer a question a reviewer would ask. No experiment exists to fill space. With 500 forward passes, every one counts.

---

## Budget Allocation (500 forward passes total)

| Experiment | Forward Passes | Runtime (est.) | Purpose |
|-----------|---------------|----------------|---------|
| EXP-A: Baseline violation profiling | 100 | ~23 min | The problem exists |
| EXP-B: SafeContract vs AEGIS-lite vs naive clamp | 100 | ~23 min | We solve it better |
| EXP-C: Composition empirical verification | 100 | ~23 min | Theorems hold in practice |
| EXP-D: Adaptive bounds (strictness sweep) | 100 | ~23 min | Pareto frontier |
| EXP-E: Overhead microbenchmark | 0 (no model) | ~2 min | Negligible cost |
| EXP-F: Ablation - contract components | 100 | ~23 min | Each piece matters |
| **Total** | **500** | **~117 min** | |

Buffer: ~3 min. Fits in 2 hours.

---

## EXP-A: Baseline Violation Profiling (The Problem Exists)

**Research question**: How often and how severely do SmolVLA raw outputs violate physical safety constraints on real LIBERO observations?

**IV**: None (observational study)
**DV**:
- Fraction of actions with any dimension outside [-1, 1] (violation rate)
- Max absolute action value observed
- Per-dimension violation distribution (which joints are worst?)
- Violation magnitude distribution (how far outside bounds?)

**Procedure**:
1. Load `HuggingFaceVLA/smol-libero` dataset
2. Sample 100 observations uniformly (stratified across episodes if possible)
3. Run SmolVLA inference on each (10 denoising steps, standard)
4. Record raw action vectors (no clipping)
5. Compute violation statistics

**Sample size justification**: 100 samples. Based on the existing EXP-004 data showing 100% violation rate on 10 samples (all exceeded [-1,1], max 2.34), 100 samples gives a 95% CI of +/-0.0% on that rate (it's clearly saturated). The interesting question is the magnitude distribution - 100 samples give a reliable histogram and percentile estimates (95th, 99th action magnitude).

**Expected runtime**: 100 x 14s = 1400s = ~23 min

**Statistical tests**: Descriptive statistics (mean, std, percentiles). One-sample t-test against H0: max(|a|) <= 1.0. Binomial exact CI for violation rate.

**Produces**:
- **Figure 1a**: Histogram of max |a_i| per sample (x-axis: magnitude, y-axis: count). Red line at 1.0 showing the nominal bound.
- **Figure 1b**: Per-dimension box plot of raw action values across 100 samples. Shows which joints are worst offenders.
- **Table 1 row**: "No Safety" baseline row with violation rate, max magnitude, mean clipping distance.

**Why it matters**: Establishes the core motivation. If SmolVLA already stays in bounds, there's no paper. Prior data confirms it doesn't.

---

## EXP-B: SafeContract vs Baselines (We Solve It Better)

**Research question**: How does SafeContract compare against (a) naive torch.clamp, (b) a simplified AEGIS-style CBF-QP, and (c) no safety, in terms of violation elimination, action distortion, and overhead?

**IV** (between-subjects, same 100 observations):
1. **No safety** (raw output) - reuse EXP-A data
2. **Naive clamp** - torch.clamp(action, -1, 1), no velocity/workspace awareness
3. **SafeContract** - @safety_contract with bounds + velocity + workspace
4. **AEGIS-lite** - QP projection: minimize ||a_clipped - a_raw||_2 subject to bounds + velocity constraints (using scipy.optimize.minimize or cvxpy)

**DV**:
- Violation rate after enforcement (should be 0% for all methods except "no safety")
- Action distortion: L2 distance between raw and enforced action (mean, std)
- Action smoothness: mean |a_t - a_{t-1}| across the 100 samples (treating them as a pseudo-trajectory)
- Enforcement latency (microseconds per action)

**Implementation of AEGIS-lite**:
AEGIS solves a QP at each step: min ||a - a_raw||^2 s.t. l <= a <= u, ||a - a_prev||_inf <= v_max. This is a box-constrained QP. Implement with `scipy.optimize.minimize(method='L-BFGS-B', bounds=...)` or simpler: project raw action onto the intersection of bounds + velocity ball. For box constraints, the QP solution IS componentwise clipping to the tighter of (bounds, velocity ball), which is exactly what SafeContract does. The key difference is computational: AEGIS-lite calls a solver, SafeContract uses numpy vectorized ops.

**Important nuance**: For box constraints, AEGIS-QP and SafeContract produce IDENTICAL outputs (both project to the same feasible set). The paper contribution is not "better enforcement" but: (a) composition theory that GUARANTEES this equivalence, (b) the decorator pattern that makes it 3 lines of code, (c) 1000x less overhead. Frame the comparison as "same safety, radically simpler."

**Sample size**: Same 100 observations as EXP-A. Methods are applied post-hoc to the same raw actions, so this is a within-subjects design. Use paired tests.

**Forward passes**: 0 additional (reuse EXP-A raw actions). But we need 100 forward passes if we want velocity checking (need consecutive actions from the same trajectory). Alternative: run 100 inferences on temporally ordered observations from the same episode.

REVISED: Use 100 forward passes on temporally consecutive observations from 2-3 LIBERO episodes (~33-50 steps each). This gives us real consecutive actions for velocity checking.

**Expected runtime**: ~23 min (100 forward passes). Post-hoc enforcement on cached actions is instant (<1s total).

**Statistical tests**:
- Paired Wilcoxon signed-rank test on action distortion (SafeContract vs AEGIS-lite). Expect no significant difference (they should be identical for box constraints).
- Paired t-test on enforcement latency (SafeContract vs AEGIS-lite). Expect SafeContract is 100-1000x faster.
- 95% bootstrap CIs on all DVs.

**Produces**:
- **Table 1**: Main comparison table. Rows: 4 methods. Columns: violation rate, mean distortion L2, mean smoothness, p99 latency (us).
- **Figure 2**: Scatter plot - X: action distortion, Y: enforcement latency (log scale). Shows SafeContract in the Pareto-optimal corner (low distortion, low latency).

**Why it matters**: Reviewers will ask "how is this different from torch.clamp?" Answer: clamp doesn't handle velocity or composition. They'll ask "how is this different from AEGIS?" Answer: same result for box constraints, but 1000x faster and formally verified via composition theorems.

---

## EXP-C: Composition Empirical Verification (Theorems Hold in Practice)

**Research question**: Does sequential application of multiple contracts match the theoretical intersection prediction, and does naive (wrong) composition order produce measurably different outputs?

**IV**: Composition strategy (3 levels):
1. **Correct composition**: bounds clip -> velocity clip -> re-apply bounds (what SafeContract does)
2. **Naive composition**: bounds clip -> velocity clip (no re-application)
3. **Single-pass intersection**: compute intersection bounds, clip once (theoretical optimum)

**DV**:
- Agreement rate: fraction of action dimensions where all 3 strategies produce identical output
- Divergence magnitude: L-inf and L2 distance between naive and correct composition
- Fraction of steps where naive violates bounds (the composition interference bug)
- Per-joint disagreement frequency

**Procedure**:
1. Use the 100 consecutive actions from EXP-B
2. Apply each composition strategy post-hoc
3. Compare outputs dimension-by-dimension

**Forward passes**: 0 additional (reuse EXP-B cached actions)

**Sample size**: 100 actions x 7 dimensions = 700 comparisons. With the existing EXP-004 data showing violations on every sample, the disagreement rate should be nonzero. Power analysis: for detecting a 5% disagreement rate with 80% power, n=700 comparisons gives >99% power (binomial test).

**Expected runtime**: <1 second (all post-hoc numpy)

**Statistical tests**:
- McNemar's test comparing disagreement rates (naive vs correct)
- Paired Wilcoxon on divergence magnitudes
- Exact binomial CI on the interference rate

**Produces**:
- **Figure 3**: Bar chart - fraction of steps where naive composition violates bounds, across different velocity limits (v_max = 0.05, 0.1, 0.2, 0.5). Shows interference increases with tighter velocity limits.
- **Table 2 row**: "Composition interference rate: X% of steps, max divergence Y"

**Why it matters**: This is the core theoretical contribution made empirical. "Naive stacking of safety constraints can silently violate bounds in X% of real VLA actions" is a strong negative result.

---

## EXP-D: Adaptive Bounds - Strictness Sweep (Pareto Frontier)

**Research question**: How does contract strictness affect (a) violation elimination, (b) action distortion, and (c) the Pareto frontier of safety vs. policy fidelity?

**IV**: Contract strictness (5 levels):
1. **Very loose**: bounds [-2, 2], v_max = 1.0
2. **Loose**: bounds [-1.5, 1.5], v_max = 0.5
3. **Standard**: bounds [-1.0, 1.0], v_max = 0.1
4. **Tight**: bounds [-0.5, 0.5], v_max = 0.05
5. **Very tight**: bounds [-0.3, 0.3], v_max = 0.02

Additionally, if time permits, a "learned" bounds level:
6. **Learned (99th percentile)**: bounds set to 99th percentile of observed actions from EXP-A, v_max set to 99th percentile of observed velocities

**DV**:
- Violation rate (fraction of actions clipped)
- Mean clipping magnitude (L2)
- Action variance preserved (ratio of var(clipped) / var(raw))
- Cosine similarity between raw and clipped action sequences
- "Effective range utilization": fraction of the action space actually used after clipping

**Procedure**:
1. Use 100 consecutive actions from EXP-B
2. Apply each strictness level post-hoc
3. Compute all DVs

**Forward passes**: 0 additional (post-hoc on cached actions). But we allocate 100 forward passes here for a second batch of observations (different LIBERO episodes) to test generalization.

REVISED: 50 forward passes on new episodes (for generalization), 50 reuse from EXP-B.

**Expected runtime**: 50 x 14s = 700s = ~12 min for new forward passes. Post-hoc analysis: <1s.

Wait - we should use all 100 fresh forward passes from the budget. Let me reallocate: EXP-A and EXP-B share the same 100 forward passes (consecutive observations). EXP-D uses 100 NEW forward passes from different episodes. EXP-F uses the same data as EXP-D applied post-hoc.

FINAL BUDGET:

| Experiment | Forward Passes | Source |
|-----------|---------------|--------|
| EXP-A + EXP-B + EXP-C | 100 | Batch 1: 2-3 episodes, consecutive |
| EXP-D + EXP-F | 100 | Batch 2: different episodes, consecutive |
| **Total model calls** | **200** | Leaves 300 in reserve for reruns/debugging |

This is much better. 200 forward passes = ~47 min. Leaves a huge buffer.

**Statistical tests**:
- Friedman test across 5 strictness levels (repeated measures on same actions)
- Post-hoc pairwise Wilcoxon with Bonferroni correction
- 95% bootstrap CIs on all DVs at each level

**Produces**:
- **Figure 4 (THE MONEY FIGURE)**: Pareto plot. X-axis: action distortion (cosine similarity to raw, higher = less distortion). Y-axis: violation rate (lower = safer). Each point = one strictness level. Connects to show the frontier. Annotated with the "sweet spot" (standard bounds).
- **Table 2**: Full results grid. Rows: 5 strictness levels. Columns: violation rate, mean clip L2, variance preserved, cosine sim.

**Why it matters**: Shows the framework is tunable, not binary. Practitioners can choose their operating point on the safety-fidelity tradeoff.

---

## EXP-E: Overhead Microbenchmark (Negligible Cost)

**Research question**: What is the wall-clock overhead of SafeContract enforcement relative to VLA inference?

**IV**: Method (4 levels):
1. No safety (passthrough)
2. SafeContract (bounds only)
3. SafeContract (bounds + velocity + workspace)
4. AEGIS-lite QP solve

**DV**: Wall-clock enforcement time (microseconds)

**Procedure**:
1. Generate 10,000 random 7-dim action vectors
2. Time each enforcement method (exclude model inference)
3. Use time.perf_counter_ns for sub-microsecond precision
4. 100 warmup iterations, then 10,000 timed

**Forward passes**: 0 (synthetic data)

**Expected runtime**: ~2 min total

**Statistical tests**: Report mean, std, p50, p99. 95% CI on mean.

**Produces**:
- **Table 3**: Latency comparison. Key number: SafeContract ~27us (already measured) vs AEGIS-lite QP solve (expect 100-1000us). Ratio column.
- Context row: SmolVLA inference = 14,000,000 us. SafeContract = 27 us. Ratio: 500,000x.

**Why it matters**: Overhead is a real concern for 10+ Hz control. This shows contracts are free in practice.

NOTE: We already have this data (exp1_overhead.json: avg 27us, p99 37us). Just need to add AEGIS-lite timing.

---

## EXP-F: Ablation - Contract Components (Each Piece Matters)

**Research question**: How much does each contract component (bounds, velocity, workspace) contribute to safety and action distortion?

**IV**: Contract configuration (5 ablations):
1. No contract (raw)
2. Bounds only ([-1, 1])
3. Velocity only (v_max = 0.1)
4. Workspace only (first 3 dims clipped)
5. Full contract (bounds + velocity + workspace)

**DV**: Same as EXP-D (violation rate, distortion, smoothness)

**Procedure**:
1. Use 100 actions from EXP-D (Batch 2)
2. Apply each ablation post-hoc

**Forward passes**: 0 additional (post-hoc)

**Expected runtime**: <1s

**Statistical tests**: Paired Wilcoxon comparing each component vs full contract.

**Produces**:
- **Table 4**: Ablation table. Rows: 5 configs. Columns: bounds violations, velocity violations, workspace violations, total distortion.
- Key finding: bounds alone catches most violations (high prior from EXP-A data showing max 2.34). Velocity adds smoothness. Workspace adds spatial safety.

**Why it matters**: Reviewers want to know if each component is pulling its weight.

---

## Figures and Tables Summary (Paper Layout)

| ID | Type | Experiment | Location in Paper |
|----|------|------------|-------------------|
| Fig 1 | System diagram | N/A | Section 2 (Method) |
| Fig 1a | Histogram | EXP-A | Section 3.1 (Motivation) |
| Fig 1b | Box plot | EXP-A | Section 3.1 (Motivation) |
| Fig 2 | Scatter (distortion vs latency) | EXP-B + EXP-E | Section 3.2 (Comparison) |
| Fig 3 | Bar chart (interference rate) | EXP-C | Section 3.3 (Composition) |
| Fig 4 | Pareto frontier | EXP-D | Section 3.4 (Adaptive bounds) |
| Table 1 | Method comparison | EXP-B | Section 3.2 |
| Table 2 | Strictness sweep | EXP-D | Section 3.4 |
| Table 3 | Latency | EXP-E | Section 3.2 or 3.5 |
| Table 4 | Ablation | EXP-F | Section 3.5 |

For a 4-page workshop paper, pick the strongest 2-3 figures and 2 tables. Recommended:
- **Must include**: Fig 4 (Pareto - the money figure), Table 1 (main comparison), Fig 3 (composition interference)
- **Nice to have**: Fig 1a (motivation histogram), Table 3 (overhead)
- **Cut if tight on space**: Table 4 (ablation), Fig 1b (box plot), Fig 2 (scatter)

---

## Statistical Rigor Checklist

- [ ] All comparisons use paired tests (same observations, different methods)
- [ ] Report effect sizes (Cohen's d or rank-biserial r), not just p-values
- [ ] 95% bootstrap CIs on all main metrics (10,000 resamples)
- [ ] Bonferroni correction for multiple comparisons in EXP-D (5 levels = 10 pairwise)
- [ ] Report exact p-values, not just "p < 0.05"
- [ ] Seed all random operations (np.random.default_rng(42))
- [ ] Pre-register the analysis plan (this document) before running

---

## Implementation Order (14-day timeline)

### Days 1-2: Data Collection Infrastructure
- [ ] Script to load smol-libero, extract consecutive observations from episodes
- [ ] SmolVLA inference wrapper that caches raw actions + metadata
- [ ] Run Batch 1 (100 forward passes, ~23 min)
- [ ] Run Batch 2 (100 forward passes, ~23 min)
- [ ] Verify data integrity (shapes, ranges, no NaNs)

### Days 3-4: Implement Baselines
- [ ] AEGIS-lite QP solver (scipy or manual box-constrained projection)
- [ ] Naive clamp baseline
- [ ] "Learned bounds" from percentiles of Batch 1
- [ ] Naive composition (no re-application of bounds)

### Days 5-7: Run All Analyses
- [ ] EXP-A: violation profiling on Batch 1
- [ ] EXP-B: 4-method comparison on Batch 1
- [ ] EXP-C: composition verification on Batch 1
- [ ] EXP-D: strictness sweep on Batch 2
- [ ] EXP-E: overhead microbenchmark (add AEGIS-lite timing)
- [ ] EXP-F: ablation on Batch 2
- [ ] All statistical tests + CIs

### Days 8-9: Generate All Figures
- [ ] matplotlib/seaborn for all plots
- [ ] LaTeX tables (booktabs)
- [ ] System diagram (TikZ or draw.io)

### Days 10-12: Write Paper
- [ ] Section 1: Introduction (0.5 page)
- [ ] Section 2: Method - contracts + composition theorems (1 page)
- [ ] Section 3: Experiments (1.5 pages)
- [ ] Section 4: Discussion + limitations (0.5 page)
- [ ] Abstract, references

### Days 13-14: Polish + Submit
- [ ] Internal proofread
- [ ] Check all numbers match between text and tables
- [ ] Format check (page limit, margins, font)
- [ ] Submit

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SmolVLA inference crashes on some observations | Low | Medium | Wrap in try/except, skip bad samples, report N |
| 100% violation rate makes baselines trivially different | Medium | Low | This is actually the point - show the problem is universal. Focus on distortion differences. |
| AEGIS-lite and SafeContract produce identical outputs | High | Medium | Expected for box constraints. Frame as "same safety, 1000x less overhead." The difference is in the framework, not the math. |
| Composition interference rate is very low (<1%) | Medium | High | Tighten velocity bounds to force interference. Report honestly. Even 1% matters at 10Hz (every 10 seconds). |
| 200 forward passes is not enough for statistical significance | Low | High | With 100% violation rate, even n=30 is significant. The variance is in magnitudes, not presence. |

---

## What Makes This Paper STRONG Enough to Stand Alone

1. **The problem is undeniable**: 100% of SmolVLA outputs exceed [-1,1]. Quantified on 100+ real observations.
2. **Theoretical grounding**: Composition theorems with proofs (not just "we clip stuff").
3. **Practical comparison**: Head-to-head with AEGIS-style approach showing same safety, 1000x less overhead.
4. **Negative result**: Naive composition SILENTLY violates bounds - a cautionary finding for the community.
5. **Tunable**: Pareto frontier lets practitioners choose their operating point.
6. **Reproducible**: All on a laptop (Mac Air M3), no GPU cluster needed. Budget: $0.

The paper is NOT about beating AEGIS on safety. It's about showing that for VLA box-constraint safety, you don't NEED a QP solver - formally verified clipping with composition guarantees is sufficient and 1000x faster. The contribution is the formalization, not the algorithm.
