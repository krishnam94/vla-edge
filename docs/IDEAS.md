# Idea Log - vla-edge

Log ideas and move on. Don't evaluate immediately. Review monthly.

---

## 2026-03-29: ProbeFlow Adaptive Denoising for SmolVLA
**Source**: ProbeFlow paper (arXiv:2603.17850) + SmolVLA's 10-step flow matching bottleneck
**Idea**: Integrate ProbeFlow's linearity probe into SmolVLA's action expert denoising loop. Use cosine similarity between initial and lookahead velocity vectors to skip unnecessary steps. Training-free, drop-in modification.
**Why novel**: Nobody has applied adaptive flow matching step scheduling to SmolVLA specifically. ProbeFlow was tested on Evo-1, not SmolVLA. Combining it with Jetson edge constraints and our safety validation is new territory.
**Estimated impact**: Reduce action expert forward passes from 10 to ~3-4 average. ~2.5-3x speedup on the action decoding bottleneck.
**Risk**: LIBERO showed 3.8% accuracy drop. Need to validate safety metrics don't degrade.
**Status**: RESEARCHED - see `docs/research/ADAPTIVE_FLOW_MATCHING.md`
**Adjacent to**: profiling, action expert optimization, safety validation

## 2026-03-29: Horizon-Aware Action Execution (inspired by FASTER)
**Source**: FASTER paper (arXiv:2603.19199) - horizon-aware schedule for flow VLAs
**Idea**: Even without retraining, start executing early actions from a partially-denoised chunk while continuing to refine later actions. Combines with ProbeFlow (fewer total steps) and SmolVLA's async inference design.
**Why novel**: FASTER requires fine-tuning. A training-free approximation using SmolVLA's existing async execution + partial chunk emission is unexplored.
**Status**: UNEXPLORED
**Adjacent to**: ProbeFlow, async inference, latency optimization

---

## 2026-03-30: Chaos Engineering for Robot Policies
**Source**: Netflix chaos monkey + VLA safety validation (collision matrix)
**Idea**: `vla-edge chaos run --model smolvla --scenario action-corruption` - systematically inject faults (corrupt action tokens, drop frames, add noise) and measure safety degradation. Produces a "resilience report."
**Why novel**: Nobody has applied chaos engineering to VLA inference. Everyone validates in clean conditions.
**Status**: UNEXPLORED
**Adjacent to**: safety validation, profiling

## 2026-03-30: Canary Deployment for Robot Policies
**Source**: Web DevOps canary deploys + robot policy switching
**Idea**: Run two policies simultaneously (10/90 split), compare safety metrics, auto-rollback on regression. Like Netflix deploys code, applied to robot brains.
**Why novel**: Robot policy deployment is currently "test in sim, deploy, pray."
**Status**: UNEXPLORED
**Adjacent to**: A/B testing, safety, recipes

## 2026-03-30: Formal Safety Contracts (Decorators)
**Source**: Formal methods runtime verification + Python decorators
**Idea**: `@safety_contract(joint_velocity_max=1.0, workspace_bounds=...)` that enforces bounds regardless of neural network output. Safety as a precondition, not a hope.
**Why novel**: Current VLA safety is post-hoc. This makes it compile-time / decorator-time.
**Status**: UNEXPLORED - could prototype in Phase 3 safety work
**Adjacent to**: safety validation, Python decorators, design by contract

## 2026-03-30: Policy LOD (Level of Detail)
**Source**: Game engine LOD systems (UE5, Unity)
**Idea**: Dynamically choose model complexity based on task demands. Full VLA for precision grasping, distilled model for transport, scripted fallback for e-stop.
**Why novel**: Everyone optimizes one model. Nobody dynamically routes to different complexity levels.
**Status**: UNEXPLORED
**Adjacent to**: model registry, SP-VLA's deliberative/intuitive routing

## 2026-03-30: A/B Testing for Action Quality
**Source**: Web analytics + VLA evaluation
**Idea**: `vla-edge ab-test --model-a smolvla-fp16 --model-b smolvla-q4` with statistical significance testing on action-level differences. Confidence intervals, not just averages.
**Why novel**: Current benchmarks measure task success rate. Nobody measures statistical significance of action-level differences between model variants.
**Status**: UNEXPLORED
**Adjacent to**: profiling, validation, degradation measurement

---

## 2026-03-29: Variable Bitrate Denoising (from Audio Codecs)
**Source**: Opus codec SILK/CELT/Hybrid per-frame mode switching + SmolVLA flow matching
**Idea**: Classify action DIFFICULTY before denoising (proactive) rather than probing during denoising (reactive like ProbeFlow). Simple linear motions get 2-3 steps ("SILK mode"), precision grasping gets 8-10 steps ("CELT mode"). Plus: per-dimension denoising precision allocation - joints near target get fewer steps, joints making large moves get more.
**Why novel**: Nobody has framed adaptive denoising as a signal-classification problem. ProbeFlow is reactive. Per-dimension allocation appears completely unexplored.
**Status**: RESEARCHED - see `docs/research/sessions/2026-03-29_cross_domain_techniques.md`
**Adjacent to**: ProbeFlow, action expert optimization, safety validation

## 2026-03-29: Inference Frame Budget / DRS for VLA (from Game Engines)
**Source**: UE5/Unity Dynamic Resolution Scaling + VLA inference pipeline
**Idea**: When inference is about to miss the control deadline (e.g. 100ms for 10Hz), automatically degrade quality knobs in priority order: reduce denoising steps, reduce image resolution, reuse cached vision embedding, emit last-known-safe action. Key game engine insight: scale DOWN fast (aggressive), scale UP slow (conservative), with hysteresis to prevent oscillation. Decompose the inference budget into per-subsystem budgets with independent quality controls.
**Why novel**: DRS has never been applied to robot inference. Async inference exists but doesn't adaptively adjust quality knobs. The asymmetric scaling + hysteresis pattern is well-proven in games but absent from robotics.
**Status**: RESEARCHED - see `docs/research/sessions/2026-03-29_cross_domain_techniques.md`
**Adjacent to**: profiling, safety validation, thermal management

## 2026-03-29: SOTIF-VLA Safety Framework (from AV Safety Certification)
**Source**: ISO 21448 SOTIF + Waymo's Demonstrably Safe AI + VLA safety validation
**Idea**: Adapt SOTIF's systematic framework to VLA manipulation. Define "VLA triggering conditions" (unusual lighting, ambiguous instructions, transparent objects, thermal throttle). Define "VLA functional insufficiencies" (quantization drift, under-denoising, vision resolution loss). Use SOTIF's 4-quadrant model (Known Safe / Known Unsafe / Unknown Safe / Unknown Unsafe) to systematically reduce residual risk. Add Waymo-inspired pre-execution action critic.
**Why novel**: SOTIF is strictly automotive. Nobody has adapted it to robot manipulation or VLAs. The concept of "VLA triggering conditions" as a formal safety category doesn't exist.
**Status**: RESEARCHED - see `docs/research/sessions/2026-03-29_cross_domain_techniques.md`
**Adjacent to**: safety validation, chaos engineering, formal contracts

## 2026-03-29: Prediction-Error Gated Model Switching (from Neuroscience)
**Source**: Cerebellar forward models in motor control + VLA model routing
**Idea**: Train a tiny forward dynamics model (~5M params) that predicts the next observation given current observation + action. If prediction error is low (world matches expectations), use fast/distilled policy. If prediction error is high (unexpected happened), escalate to full VLA. Different from DeeR-VLA (which checks action consistency across exits - self-referential) because this checks if the WORLD matches expectations (grounded in reality).
**Why novel**: SP-VLA has dual paths but not prediction-error gated. DeeR-VLA and NanoVLA route by task complexity, not by world-model prediction error. The neuroscience literature is clear that prediction error is the biological gating signal, but nobody has implemented it in VLA.
**Status**: RESEARCHED - see `docs/research/sessions/2026-03-29_cross_domain_techniques.md`
**Adjacent to**: model registry, DeeR-VLA, NanoVLA, safety

## 2026-03-29: Speculative Action Execution (from HFT)
**Source**: HFT FPGA speculative execution + flow matching denoising
**Idea**: After 2 denoising steps (out of 10), emit a "speculative action" to the robot. Continue denoising. After step 10, compare final vs speculative. If they agree, the robot is already executing correctly (saved 80% wait time). If they disagree, emit a smooth correction. Core HFT insight: cost of slightly-wrong-but-fast is usually less than perfectly-right-but-late.
**Why novel**: FASTER addresses TTFA but requires retraining. Async inference decouples execution from prediction but doesn't emit speculative early actions. The speculative-execute-then-correct pattern from HFT is new for VLA denoising.
**Status**: RESEARCHED - see `docs/research/sessions/2026-03-29_cross_domain_techniques.md`
**Adjacent to**: ProbeFlow, FASTER, async inference, latency optimization
