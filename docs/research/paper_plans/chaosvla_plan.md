# ChaosVLA: Chaos Engineering for VLA Deployment Pipelines

**Status**: Backup paper (if SafeContract misses CoRL)
**Target**: NeurIPS 2026 Workshop (Sep deadline) or ICRA 2027 (Sep deadline)
**Created**: 2026-03-29

---

## 1. Title + Abstract

**Title**: ChaosVLA: Systematic Fault Injection for Characterizing Vision-Language-Action Model Robustness

**Abstract** (148 words):
Vision-Language-Action (VLA) models are evaluated almost exclusively under clean conditions - perfect lighting, zero latency, uncorrupted observations. Real deployment on edge hardware introduces faults that no benchmark captures: dropped camera frames, thermal throttling, quantization drift, network partitions, and corrupted action tokens. We introduce ChaosVLA, a systematic fault injection framework that applies chaos engineering principles - borrowed from distributed systems reliability - to VLA inference pipelines. We define 12 fault types spanning sensor, compute, model, and communication layers. We inject these faults into OpenVLA-7B, SmolVLA-2B, and pi0 during simulated manipulation tasks and measure safety degradation (collision rate, workspace violations, action discontinuity). Our key finding: VLA models exhibit cliff-edge failure modes where small perturbations cause catastrophic safety violations, and these cliffs differ dramatically across architectures. ChaosVLA produces per-model "resilience scorecards" that predict deployment failure rates before hardware-in-the-loop testing.

---

## 2. Five-Section Outline

| # | Section | Content | Pages |
|---|---------|---------|-------|
| 1 | Introduction | Clean-condition evaluation gap. Chaos engineering in software (Netflix) vs robotics (nobody). Contribution: first systematic fault taxonomy + injection framework for VLAs. | 1 |
| 2 | Fault Taxonomy | 12 fault types across 4 layers. Severity levels. Injection protocol. | 1.5 |
| 3 | Framework Design | ChaosVLA architecture: fault injector, safety monitor, resilience scorer. Integration with vla-edge toolkit. | 1 |
| 4 | Experiments | 3 models x 12 faults x 3 severity levels. Metrics: collision rate, action MSE, task completion, safety score. Cliff-edge analysis. | 2.5 |
| 5 | Discussion + Conclusion | Resilience scorecards. Deployment recommendations per architecture. Limitations (sim only). Future: hardware-in-the-loop. | 1 |

---

## 3. The 12 Fault Types

**Sensor Layer**
1. **Frame drop** - Skip 1-5 consecutive camera frames (simulates USB disconnect, thermal throttle)
2. **Image corruption** - Gaussian noise, JPEG artifacts, partial occlusion
3. **Lighting shift** - Sudden brightness/contrast change mid-episode
4. **Resolution degradation** - Downsample input below training resolution

**Compute Layer**
5. **Latency spike** - Inject 50-500ms delay in inference (simulates thermal throttle, GC pause)
6. **Memory pressure** - Reduce available VRAM, force swap/recompute
7. **Quantization drift** - Switch precision mid-episode (FP16 to INT8 to INT4)

**Model Layer**
8. **Action token corruption** - Bit-flip or noise injection in output action tokens
9. **KV cache corruption** - Corrupt cached key-value pairs mid-sequence
10. **Denoising step reduction** - Cut flow matching steps from 10 to 3 mid-episode (SmolVLA specific)

**Communication Layer**
11. **Control signal delay** - Stale actions replayed due to network lag
12. **Partial observation** - Language instruction truncated or garbled

---

## 4. Models + Datasets

**Models** (3, covering both VLA architectures):
- **OpenVLA-7B** - Autoregressive, Llama backbone (represents large AR VLAs)
- **SmolVLA-2B** - Flow matching, SmolLM2 backbone (represents efficient FM VLAs)
- **pi0** - Flow matching, 3B params (represents frontier FM VLAs)

**Evaluation environments**:
- **SIMPLER** (Google) - Bridge V2 tasks, standardized sim benchmark
- **LIBERO** - 130 manipulation tasks, 5 suites
- Fallback: **RLBench** if SIMPLER/LIBERO don't support all 3 models

---

## 5. Key Hypothesis

**H1**: VLA models exhibit architecture-dependent cliff-edge failures - small fault increases that cause disproportionate safety degradation. Autoregressive VLAs (OpenVLA) are more fragile to action token corruption (error compounds across tokens). Flow matching VLAs (SmolVLA, pi0) are more fragile to denoising step reduction and latency spikes.

**H2**: A composite "resilience score" computed from the 12 fault responses predicts real-world deployment failure rate better than clean-condition accuracy.

---

## 6. Minimum Viable Submission

| Format | Pages | What's needed | Feasibility |
|--------|-------|---------------|-------------|
| **Workshop (4-page)** | 4 | 2 models, 6 faults, SIMPLER only. Resilience scorecard concept. | HIGH - 4-6 weeks work |
| **Main conf (8-page)** | 8 | 3 models, 12 faults, 2 benchmarks. Full statistical analysis. Hardware validation. | MEDIUM - 8-10 weeks |

**Recommendation**: Target 4-page NeurIPS workshop. Expand to ICRA 2027 main if results are strong.

---

## 7. Timeline (NeurIPS Workshop - Sep 2026 deadline)

| Week | Dates | Milestone |
|------|-------|-----------|
| 1-2 | Jun 1-14 | Framework: fault injector + safety monitor integrated with vla-edge |
| 3-4 | Jun 15-28 | Run OpenVLA + SmolVLA on SIMPLER with 6 core faults |
| 5 | Jun 29-Jul 5 | Analyze cliff-edge patterns, generate resilience scorecards |
| 6 | Jul 6-12 | Add pi0, expand to all 12 faults if time allows |
| 7-8 | Jul 13-26 | Write 4-page paper, figures, tables |
| 9 | Jul 27-Aug 2 | Internal review, revisions |
| 10 | Aug 3-9 | Buffer week |
| -- | ~Sep 2026 | Submit to NeurIPS workshop (exact deadline TBD) |

**Go/no-go checkpoint**: Jul 5. If cliff-edge hypothesis confirmed in 2+ faults, proceed. Otherwise pivot.
