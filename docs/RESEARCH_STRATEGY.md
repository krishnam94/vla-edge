# Research Strategy for vla-edge

Single source of truth for research direction, paper plans, and process.

**Navigation**: See [docs/INDEX.md](INDEX.md) for full project doc index.
**Experiments**: See [docs/EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for all results.
**Process**: See [docs/META_ENGINEERING.md](META_ENGINEERING.md) for dev workflow.
**Milestones**: GitHub milestones (v0.1.0 done, v0.2.0 in progress, v0.3.0 planned).

---

## Primary Target: CoRL 2026 (May 29, archival) - EB-1A value

**Title**: "SafeContract: Formally Verified, Phase-Adaptive Safety Contracts for VLA Policies"
**Format**: 8 pages, archival, peer-reviewed
**EB-1A value**: HIGH (archival at top robotics venue, Criterion 5+6)

## Draft Milestone: ICRA VLA Pipelines Workshop (Apr 15, non-archival)

**Title**: "SafeContract: Composable Safety Guarantees for VLA Action Spaces"
**Format**: 2-4 pages, non-archival (stepping stone to CoRL)
**Approach**: Mathematical theorems + offline LIBERO analysis (no sim loop needed)

### Why this paper, why this venue
- VLA Pipelines workshop is about deploying VLAs to real robots - safety is first-class
- Nobody has formalized VLA action safety with provable guarantees
- 2-4 pages non-archival = clean idea + supporting evidence, not full benchmark
- Mathematical rigor differentiates from teams submitting LIBERO success rates

### Three Theorems
1. **Individual guarantee**: clipping preserves Lipschitz continuity within bounds
2. **Composition safety**: when workspace + velocity contracts compose safely (non-empty feasible set)
3. **Interference conditions**: when contracts CONFLICT (deadlock where no action is feasible)

### Experiments (offline, no sim loop)
- Download LIBERO trajectories as HDF5 (observations only)
- Run SmolVLA batch inference on 500 observations (overnight on MPS)
- Apply SafeContract post-hoc: measure violations, clipping magnitude, smoothness
- ProbeFlow interaction: does fewer steps increase violations?
- Plot: Pareto curve (contract strictness vs violation rate)

### 14-Day Timeline (revised 2026-04-01)
| Day | Task | Status |
|-----|------|--------|
| 1 (Apr 1) | Deep research: competitors, CP, adaptive bounds, survey | IN PROGRESS |
| 2 | Synthesize, finalize contributions, run /novelty-check | |
| 3-4 | Implement adaptive contracts + CP calibration | |
| 5-6 | Run corrected experiments (queue-flushed, 100+ samples) | |
| 7-8 | Generate figures, comparison tables vs AEGIS | |
| 9-11 | Write paper (LaTeX) | Draft exists |
| 12-13 | /review-panel + final /novelty-check | |
| 14 (Apr 14) | Submit | |

### Corrected Claims (2026-04-01, post correctness-critic)
- ~~3.42x speedup~~ -> 1.28x overall, 4.1x cold start only (queue artifact)
- ~~ProbeFlow reduces violations~~ -> Noise (n=20, KV cache reuse)
- Actions exceed [-1,1]: CONFIRMED (max 2.34) - strongest empirical finding
- 27us overhead: CONFIRMED
- Lesson 007: ALWAYS flush model caches before benchmarking

---

## Pipeline Papers (in priority order)

| # | Paper | Status | Target | Key Result Needed |
|---|-------|--------|--------|-------------------|
| 1 | **SafeContract** (formal safety) | ACTIVE - targeting Apr 15 | ICRA WS | 3 theorems + LIBERO offline |
| 2 | **SafeContract** (full version) | PLANNED | CoRL 2026 (May 29) | + learned params + full Pareto |
| 3 | **ChaosVLA** (chaos engineering) | PLANNED | NeurIPS WS (Sep) | 12 faults x 2 models x SIMPLER |
| 4 | **"Is RL Necessary?"** (D3P comparison) | PLANNED | NeurIPS WS (Sep) | Pareto frontier comparison |

---

## Research Process

### How ideas are generated
1. **Novel thinking agents** using 5 methods: first principles, bisociation, TRIZ, adjacent possible, Hamming
2. **Cross-domain inspiration** from audio DSP, game engines, AV safety, cybersecurity (UEBA), HFT, SPC manufacturing, compilers
3. **Wiki gap mining** - `/wiki gaps` cross-references "Gaps We Spotted" sections across all wiki articles. Gaps that appear in multiple articles independently are high-signal.
4. **Future directions harvesting** - every wiki article has "Future Directions (from authors)" sections extracted from papers. These are raw material for novel combinations.
5. **`/skill-sync audit`** flags patterns from recent sessions that should become instincts or skills.

### How ideas are validated
1. **Novelty critic** - skeptical reviewer searches for damaging prior work
2. **Must survive**: "Is this just gluing two papers?" test
3. **Must have**: counterintuitive finding or formal contribution, not just combination

### How experiments are managed
1. **`/experiment register`** after creating any new experiment (adds to EXPERIMENT_REGISTRY.md)
2. **`/experiment validate`** runs after EVERY experiment (mandatory, extends /verify-experiments)
3. Checks: caching artifacts, sample size, normalization (Lesson 010), all-X claims (Lesson: verify-all-claims)
4. **`/experiment link`** connects results to specific paper claims
5. **`/experiment audit`** before submission - cross-references ALL paper numbers against JSONs
6. Any SUSPICIOUS result must be investigated before citing in paper
7. Better to have no number than a wrong number

### How ideas become papers
1. Generate (specialist agents) -> Kill (novelty critic) -> Frame (paper planner) -> Execute (experiments)
2. **After experiments**: mandatory /verify-experiments
3. **Before submission**: mandatory /review-panel + /novelty-check
4. Workshop papers first (lower bar, faster feedback), then expand to full papers

### Key lesson (Lesson 006)
Never propose a paper without running skeptical novelty reviewer. Initial agents produced
obvious combinations (SAAD, SplitPipe, QuantProbe) that were killed by critic.
The pivoted directions (formal contracts, chaos engineering, cross-domain) survived.

### Key lesson (Lesson 008)
Soft-knee compression seemed novel (no direct competitor) but experiments showed negligible
benefit on realistic data. Motor controllers absorb clipping discontinuities. Always validate
with experiments before committing to a paper direction. "Novel" is not enough - it must also
matter in practice.

### Key lesson (Lesson 013)
Process rules exist for a reason. The register-before-run instinct was violated THREE
times despite being documented. Each time it was "we're moving fast" or "I'll register
later." The user should NEVER have to remind about process compliance. If an instinct
exists, it must be followed WITHOUT exception. Speed does not justify skipping process.

### Key lesson (Lesson 012)
Question every metric definition BEFORE celebrating results. EXP-NORM reported "98%
velocity violations persist after normalization" but was computing velocity between
non-consecutive samples from different episodes. The real rate on consecutive steps
is 22% - still significant but 4.5x lower. The error: optimizing for speed over
correctness. Fix: mandatory design review before running, asking "is this metric
measuring what we claim?" Added question-metric-definition instinct.

### Key lesson (Lesson 010)
ALWAYS check action normalization when connecting a model to an environment.
SmolVLA outputs MEAN_STD normalized actions. We sent them raw to LIBERO sim,
causing 0% task success (robot barely moved). The fix was unnormalize with
dataset stats. Created INTEGRATION_CHECKLIST.md to prevent this class of bug.

### Key lesson (Lesson 009)
Wiki "Future Directions" and "Gaps We Spotted" sections from paper analysis are a goldmine
for idea generation. Cross-referencing gaps across articles surfaces research opportunities
that individual paper reading misses. Formalized as `/wiki gaps` command.
Of 50+ ideas generated via 5 thinking methods, only 3 survived deep novelty validation.
Kill rate: 86%. The novelty critic is essential.

---

## Research Sessions Index

| Date | File | Topic |
|------|------|-------|
| 2026-03-31 | optimization_synthesis.md | Top 3 ideas (pre-critique) |
| 2026-03-31 | novelty_critique.md | Critique killed all 3, identified novel directions |
| 2026-03-31 | formal_safety_contracts.md | SafeContract deep dive - CONFIRMED NOVEL |
| 2026-03-31 | d3p_comparison.md | Training-free vs D3P - Gigerenzer argument |
| 2026-03-29 | cross_domain_techniques.md | 6 techniques from adjacent fields |

## Paper Analysis Index

| Paper | File | Relevance |
|-------|------|-----------|
| QVLA | docs/research/QVLA_ANALYSIS.md | Action-centric quantization |
| ProbeFlow | docs/research/ADAPTIVE_FLOW_MATCHING.md | Adaptive denoising |
| SmolVLA | docs/SMOLVLA_ANALYSIS.md | Our primary model |
| GPTQ OpenVLA | docs/research/GPTQ_OPENVLA_ANALYSIS.md | Pre-quantized models |

## Conference Deadlines

See `docs/CONFERENCE_TRACKER.md` for full table. Key:
- **Apr 15**: ICRA VLA Pipelines Workshop (2-4 pg, non-archival) - ACTIVE TARGET
- **May 29**: CoRL 2026 (full paper) - stretch goal
- **Sep 2026**: NeurIPS workshops - ChaosVLA + "Is RL Necessary?"


### Key lesson (Lesson 014)
Don't reimplement preprocessing pipelines. Use the framework's own pipeline.
SmolVLA on LIBERO got 0% because we reimplemented observation preprocessing
incorrectly: wrong state (joint_pos vs eef_pos), missing normalization, missing
image flip. Three lines of code. Use lerobot's LiberoProcessorStep instead.
