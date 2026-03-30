# Meta-Engineering: How We Build vla-edge

This document describes the engineering process, tools, and self-improvement
practices used to develop vla-edge. It evolves as we learn.

---

## Development Philosophy

1. **Ship small, learn fast.** v0.1 does one thing well (profile SmolVLA on Jetson).
2. **Extensibility through simplicity.** Two ABCs, two registries. New backend = one file.
3. **Safety is not optional.** Every report includes safety metrics.
4. **Real hardware, real numbers.** Jetson self-hosted runner for honest benchmarks.
5. **Automate the boring parts.** CI, linting, model scanning, paper tracking are automated.

## Process: Critic-Driven Development

Before major architectural decisions, run the `/critic` skill:

```
/critic registry pattern
```

The critic agent argues against the current approach, grounded in real-world
precedents from Physical Intelligence, LeRobot, NVIDIA, etc. The process:

1. **Propose** an architecture or design
2. **Critique** via `/critic` - find weaknesses
3. **Decide** - accept criticism that makes sense, reject what doesn't
4. **Document** the decision and rationale in this file (below)

## Architecture Decisions Record

### ADR-001: Two ABCs + Decorator Registry (2026-03-30)
**Context**: Need to support multiple hardware backends and VLA models.
**Decision**: HardwareBackend ABC (4 methods) + VLAModel ABC (2 methods + 1 property)
with `@register_backend`/`@register_model` decorator registries.
**Rationale**: Follows LeRobot (Robot: 6 methods), vla-eval (ModelServer: 1 method),
lm-eval (LM: 3 methods) patterns. Proven at scale.
**Trade-off**: Could be simpler (just functions) for v0.1 with only 1-2 backends.
The ABCs add ~50 lines of code but make the extension contract explicit.

### ADR-002: GGUF/llama.cpp over TensorRT-LLM for Jetson (2026-03-30)
**Context**: Need LLM backbone inference on Jetson Orin Nano 8GB.
**Decision**: Use llama.cpp + GGUF Q4_K_M. NOT TensorRT-LLM.
**Rationale**: TensorRT-LLM causes kernel panics on Orin Nano (confirmed NVIDIA forums).
LiteVLA-Edge (arXiv:2603.03380) proves GGUF path at 6.6 Hz. Proven, stable, open.
**Trade-off**: llama.cpp may be slower than a working TRT-LLM. If NVIDIA fixes
TRT-LLM for Orin Nano, revisit.

### ADR-003: Typer over Click for CLI (2026-03-30)
**Context**: Need a CLI framework for `vla-edge` commands.
**Decision**: Typer + Rich (same as vla-eval).
**Rationale**: Typer gives type-hinted CLI args, auto-generated help, Rich integration.
Click is more mature but Typer's DX is better for this use case.

### ADR-004: Action-Centric Safety over Generic Bounds (2026-03-30)
**Context**: Need to validate VLA actions before execution on real robots.
**Decision**: SafetyConfig with per-joint bounds, velocity limits, acceleration limits,
workspace bounds. NOT just generic clipping.
**Rationale**: OpenVLA deploy.py has zero clipping. LeRobot's EEBoundsAndSafety is the
only existing implementation. We provide configurable, per-joint safety with
severity levels (warning vs critical).
**Trade-off**: More complex than simple `np.clip()`. Worth it for real robot safety.

---

## Self-Improvement Practices

### After Every Session
- Run tests, check coverage
- Update CHANGELOG.md with what changed
- If any lesson learned, add to this file

### After Corrections
- Update `lessons` section below
- Create or update relevant instinct in `.claude/homunculus/instincts/`

### Weekly
- Run `/critic` on any new architecture
- Review open issues and PRs
- Update research tracker with new papers

### Monthly
- Full architecture review
- Update competition landscape
- Review and prune instincts that are no longer relevant

---

## Lessons Learned

(Add entries here as we learn from mistakes)

### Template
**Date**: YYYY-MM-DD
**What happened**: ...
**Root cause**: ...
**Fix applied**: ...
**Prevention rule**: ... (add to instincts if recurring)

---

## Tools & Automation

| Tool | Purpose | Config |
|------|---------|--------|
| ruff | Lint + format | pyproject.toml |
| mypy | Type checking | pyproject.toml |
| pytest | Testing with hardware markers | pyproject.toml |
| pre-commit | Pre-push quality gate | .pre-commit-config.yaml |
| GitHub Actions | CPU CI on every PR | .github/workflows/tests.yml |
| Jetson runner | GPU/hardware CI on main | .github/workflows/test-jetson.yml |
| Claude Code | PR review, paper scanning | Scheduled tasks (add after Phase 1) |
| `/critic` | Architecture criticism | .claude/commands/critic.md |
| `/profile` | Run profiling | .claude/commands/profile.md |
| `/add-model` | Add new VLA model | .claude/commands/add-model.md |
| `/add-backend` | Add new hardware backend | .claude/commands/add-backend.md |

---

## Research Tracking

Key papers we track (citations, new methods):
- QVLA (arXiv:2602.03782) - action-centric quantization
- PD-VLA (arXiv:2503.02310) - parallel decoding 4x speedup
- VLASH (arXiv:2512.01031) - async inference
- LiteVLA-Edge (arXiv:2603.03380) - our baseline (6.6 Hz on Jetson)
- VLA-Perf (arXiv:2602.18397) - analytical profiling model
- ActionFlow (arXiv:2512.20276) - pipelined inference
- NanoVLA (arXiv:2510.25122) - 52x faster, dynamic routing

Full paper database: `~/Desktop/docs/notes/research_vla_edge_papers_2025_2026.md`
