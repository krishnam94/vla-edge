# Idea Log - vla-edge

Log ideas and move on. Don't evaluate immediately. Review monthly.

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
