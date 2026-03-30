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

## Development Workflow: When to Use What

### Building a New Feature
Use `/feature-dev <description>` for any non-trivial feature. It runs:
1. Discovery - clarify what to build
2. Codebase exploration via `code-explorer` agents
3. Architecture design via `code-architect` agent
4. Implementation with TodoWrite tracking
5. Tests and verification

For simple additions (new model adapter, new backend), use the repo-specific
`/add-model` or `/add-backend` skills instead - they're faster and focused.

### Before Major Architecture Decisions
Run `/critic <area>` to stress-test the design. The critic argues against
the current approach grounded in real-world precedents (Physical Intelligence,
LeRobot, NVIDIA). The process:
1. **Propose** an architecture or design
2. **Critique** via `/critic` - find weaknesses
3. **Decide** - accept criticism that makes sense, reject what doesn't
4. **Document** the decision as an ADR in this file (below)

### Committing Code
Use `/commit` for smart commits with context-aware messages.
Use `/commit-push-pr` when ready to push a feature branch and create a PR.

### Code Review
Use `/code-review` on every PR before merging. It runs 5 parallel agents:
1. CLAUDE.md compliance check
2. Shallow bug scan on changed lines
3. Git history context review
4. Previous PR comment review
5. Code comment compliance

Only flags issues with >80% confidence. Filters out false positives.

### Periodic Maintenance
- Run `code-simplifier` agent monthly to find code that can be simplified
- Run `/revise-claude-md` after major changes to keep CLAUDE.md current
- Run `/critic` quarterly on the full architecture

### Research
- Use `/project-research` for evaluating new project ideas
- Use subagents for arXiv scanning, competitive analysis, technical deep dives
- Always save research with source URLs (instinct: sources-in-research)

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

### Code Quality (automated, every commit)
| Tool | Purpose | Config |
|------|---------|--------|
| ruff | Lint + format (pre-commit + CI) | pyproject.toml |
| mypy | Type checking (pre-commit + CI) | pyproject.toml |
| pytest | Testing with hardware markers | pyproject.toml |
| pre-commit | Pre-push quality gate | .pre-commit-config.yaml |

### CI/CD
| Tool | Purpose | Config |
|------|---------|--------|
| GitHub Actions | CPU CI on every PR | .github/workflows/tests.yml |
| Jetson runner | GPU/hardware CI on main | .github/workflows/test-jetson.yml |
| PyPI publish | Auto-release on git tag | .github/workflows/release.yml |

### Claude Code - Repo Skills (custom for vla-edge)
| Skill | When to use |
|-------|------------|
| `/critic` | Before major architecture decisions |
| `/profile` | Profile a model and get optimization suggestions |
| `/add-model` | Adding a new VLA model adapter |
| `/add-backend` | Adding a new hardware backend |
| `/benchmark` | Full benchmark run + results update |

### Claude Code - Installed Plugins (global, use as-is)
| Plugin | When to use |
|--------|------------|
| `/code-review` | On every PR before merging (5 parallel review agents) |
| `/commit` | For clean, context-aware commit messages |
| `/commit-push-pr` | When shipping a feature branch to remote + PR |
| `/feature-dev` | For non-trivial new features (explore -> design -> build) |
| `/revise-claude-md` | After major changes to keep CLAUDE.md current |
| `code-simplifier` | Monthly cleanup pass to find simplifiable code |

### Claude Code - Scheduled Tasks (set up on claude.ai/code/scheduled)
| Schedule | Task |
|----------|------|
| Daily 8 AM | HuggingFace VLA model scan |
| Monday 9 AM | arXiv + Semantic Scholar paper scan |
| Friday 5 PM | Project status review |

### Instincts (auto-enforced behavior)
| Instinct | What it enforces |
|----------|-----------------|
| jetson-constraints | TRT-LLM broken, use llama.cpp, 8GB memory budget |
| safety-always | Never skip safety metrics in reports |
| extensible-arch | New backend/model = one file, no core changes |
| no-hardcoded-keys | Env vars only for tokens/keys |
| test-before-ship | Tests required before marking features done |
| sources-in-research | Source URLs in all research docs |

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
