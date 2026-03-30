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
Two levels of review depending on the scope:

**`/critic <area>`** - Single persona (production robotics engineer). Use for:
- Targeted code-level questions ("is this ABC right?")
- Quick sanity checks on a specific module
- Post-implementation review of a feature

**`/review-panel <decision>`** - 3 personas in parallel. Use for:
- Major architecture changes (new ABC, new abstraction layer)
- Roadmap pivots (changing project direction, adding major features)
- Pre-launch review (before v0.1 ship, before HN post)
- The 3 personas and what they catch:
  1. **Production Robotics Engineer** - will it work on real hardware?
  2. **Startup ML Engineer (The User)** - can someone actually use this?
  3. **Open Source Strategist** - will this get adopted?

The process:
1. **Propose** an architecture or roadmap change
2. **Review** via `/critic` (small) or `/review-panel` (big)
3. **Decide** - unanimous findings = must fix, majority = should fix, single = discuss
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

### Learning Journal
After implementing any concept, document it in `docs/LEARNING.md`:
- What the concept is (explain like teaching someone)
- Why it matters for edge deployment specifically
- Reference links (papers, docs, code)
- How it connects to the Manning VLA book chapters

The learning journal grows with the project and feeds directly into book writing.
It also documents the story of the project - pivots, rejected ideas, and why.

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

### ADR-005: Phase 2 Order - Profiler First, Then Model Adapter (2026-03-30)
**Context**: Review panel (3 personas) ran at end of Phase 1.
**Decision**: Fix profiler methodology BEFORE building SmolVLA adapter.
**Rationale**: All 3 personas agreed SmolVLA adapter is #1 priority, but the
Production Engineer argued: fix measurement first, so you don't re-run everything.
Profiler fixes (GC disable, stddev, cuda.Event) take 2 hours. SmolVLA adapter
takes 4-6 hours. Do the 2-hour fix first.
**Phase 2 order**: (1) profiler methodology, (2) SmolVLA adapter with correct
AutoClass, (3) wire safety into profile output, (4) end-to-end integration test,
(5) README rewrite with real numbers, (6) remaining issues #4-#10.

### ADR-006: Leaderboard as Growth Strategy (2026-03-30)
**Context**: Open Source Strategist persona called this "the one thing that would
10x adoption." Auto-updated model x hardware x FPS leaderboard.
**Decision**: Seed leaderboard in v0.1 with 3-5 entries. Build submission flow in v0.2.
**Rationale**: Open LLM Leaderboard turned HuggingFace into the default comparison
tool. Same opportunity for VLA edge deployment. But need real numbers first.

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

### Daily Innovation Rhythm (15 min total)
- **Morning (5 min)**: Scan daily digest output. Read titles only, star 1-2 for later.
- **Question capture (2 min)**: Write 1 question in `docs/QUESTIONS.md` before starting work.
- **End-of-day (5 min)**: What surprised you? Write one sentence in LEARNING.md.
- **Cross-domain (3 min)**: Skim one non-robotics source (HN, game dev, audio DSP, mobile ML).

### Weekly Innovation (1 hour)
- Deep paper read (30 min): 3-pass method, extract transferable insight
- Collision matrix (15 min): Pick one cell from NOVEL_THINKING.md matrix, explore
- Question review (10 min): Answer 1, add 3, cross out dead ends
- Idea log review (5 min): Any ideas worth prototyping?

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

### Lesson 004: Critic must run BEFORE push, not after (2026-03-31)
**What happened**: Pushed MPS backend, ONNX export, and OpenVLA adapter across 3
commits without running critic. Batch critic afterwards found 7 issues including
2 HIGH (MPS hasattr guard, dtype inconsistency).
**Root cause**: Lesson 001 said "run critic before push" but the instinct only
reminded AFTER push. No enforcement mechanism for "before."
**Fix applied**: Renamed instinct to `pre-push-and-post-push`. Now explicitly says
"Run /critic BEFORE pushing. Do NOT skip." Batching rule added for multiple commits.
**Prevention rule**: Critic is mandatory before `git push` for any src/ changes.

### Lesson 003: /feature-dev plugin skill not loading (2026-03-31)
**What happened**: Tried `Skill(feature-dev)`, `Skill(commit-commands:feature-dev)`,
`Skill(feature-dev:feature-dev)` - all returned "Unknown skill."
**Root cause**: Plugin is installed at correct path but Skill tool resolution doesn't
find it. Possibly a plugin manifest or loading order issue.
**Fix applied**: Ran the feature-dev exploration manually as an agent - same process,
different invocation. Got full implementation plan.
**Prevention rule**: When a tool fails, investigate root cause (check paths, config,
manifests) before retrying with different syntax. Added `investigate-errors` instinct.
**TODO**: Debug why Skill tool can't find installed plugins and fix.

### Lesson 002: Cloud scheduled tasks can't run gh CLI (2026-03-31)
**What happened**: Updated scheduled tasks to create GitHub issues via `gh issue create`.
Cloud environment doesn't have gh CLI authenticated - issues never appeared.
**Root cause**: Cloud tasks run on Anthropic's infra, not your machine. No gh/git auth.
**Fix applied**: Cloud tasks produce output only (viewable at claude.ai). The /status
skill creates GitHub issues locally using your authenticated gh CLI.
**Prevention rule**: Cloud scheduled tasks can only use WebSearch/WebFetch. Any GitHub
write operations must happen from the local CLI session via /status or manually.

### Lesson 001: Run critic + code review BEFORE pushing, not after (2026-03-30)
**What happened**: Phase 1 shipped with 3 critical issues (trust_remote_code
everywhere, registry race condition, CUDA excluding all aarch64).
**Root cause**: Committed and nearly pushed without running review agents.
**Fix applied**: Ran /critic + code review agent in parallel. Fixed 7 issues.
**Prevention rule**: Always run critic + code review before `git push`. Added
to META_ENGINEERING.md workflow. The /code-review plugin runs on PRs, but
for local development we also run the code review agent before pushing.

### Template
**Date**: YYYY-MM-DD
**What happened**: ...
**Root cause**: ...
**Fix applied**: ...
**Prevention rule**: ... (add to instincts if recurring)

---

## Tools & Automation

### Hooks (enforced automatically via .claude/settings.json)
| Hook | Trigger | What it does |
|------|---------|-------------|
| Ruff format | After .py edit/write | Auto-formats Python files |
| Block credentials | Before writing .env/.key/.token | Hard block (exit 2) |
| Post-push reminder | After `git push` | Reminds: check CI, update LEARNING.md |
| Post-commit reminder | After `git commit` | Reminds: close issues (closes #N) |
| trust_remote_code guard | Before writing backend .py | Warns if hardcoded True |

### Code Quality (automated, every commit)
| Tool | Purpose | Config |
|------|---------|--------|
| ruff | Lint + format (pre-commit + CI) | pyproject.toml |
| mypy | Type checking (pre-commit + CI) | pyproject.toml |
| pytest | Testing with hardware markers | pyproject.toml |
| pre-commit | Pre-push quality gate | .pre-commit-config.yaml |

### Milestones (GitHub)
| Milestone | Target | What's in it |
|-----------|--------|-------------|
| v0.1.0 - First Working Demo | 2026-05-01 | SmolVLA profiling, safety, recipes, README. HN-launchable. |
| v0.2.0 - Optimization Pipeline | 2026-06-15 | GGUF quantization, ONNX/TRT, leaderboard, multiple models. |
| v0.3.0 - Community & Scale | 2026-08-01 | Entry points, ICRA paper, 20+ leaderboard entries. |

### CI/CD
| Tool | Purpose | Config |
|------|---------|--------|
| GitHub Actions | CPU CI on every PR | .github/workflows/tests.yml |
| Jetson runner | GPU/hardware CI on main | .github/workflows/test-jetson.yml |
| PyPI publish | Auto-release on git tag | .github/workflows/release.yml |

### Claude Code - Repo Skills (custom for vla-edge)
| Skill | When to use |
|-------|------------|
| `/status` | Start of session - show project state and suggest next steps |
| `/critic` | Before major architecture decisions (single persona) |
| `/review-panel` | Major arch/roadmap decisions (3 personas in parallel) |
| `/profile` | Profile a model and get optimization suggestions |
| `/add-model` | Adding a new VLA model adapter |
| `/add-backend` | Adding a new hardware backend |
| `/benchmark` | Full benchmark run + results update |
| `/doc-hygiene` | Check all docs for staleness and inconsistencies |
| `/synthesize` | Generate novel idea combinations from questions + papers + code |

### Claude Code - Installed Plugins (global, use as-is)
| Plugin | When to use |
|--------|------------|
| `/code-review` | On every PR before merging (5 parallel review agents) |
| `/commit` | For clean, context-aware commit messages |
| `/commit-push-pr` | When shipping a feature branch to remote + PR |
| `/feature-dev` | For non-trivial new features (explore -> design -> build) |
| `/revise-claude-md` | After major changes to keep CLAUDE.md current |
| `code-simplifier` | Monthly cleanup pass to find simplifiable code |

### Claude Code - Scheduled Tasks (live on claude.ai/code/scheduled, 3 AM PT)

All tasks write results as GitHub issues (readable by /status skill).

| Schedule | Task | Output Label | Novel Thinking |
|----------|------|-------------|----------------|
| Daily 3:07 AM | Digest + VLA model scan | `daily-digest` | Cross-domain insight |
| Mon+Fri 3:23 AM | Paper scan + citations + competitors | `paper-scan` | 3 AI-generated questions |
| Friday 3:41 AM | Project review + doc hygiene + planning | `weekly-status` | Collision of the Week |

**How it works**: Cloud tasks run at 3 AM and produce output (viewable at
claude.ai/code/scheduled). The `/status` skill then either reads existing
digest issues from GitHub, or runs a quick local research scan and creates
the digest issue from your authenticated machine. This closes the feedback loop:

```
3 AM: Cloud task runs research (viewable at claude.ai)
Morning: You run /status in CLI
/status: Checks for existing digest issues
  -> If found: reads and summarizes
  -> If not: runs quick local research + creates GitHub issue
  -> Shows priorities for today based on insights + open issues
```

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

### Paper Status Tracking

| Paper | Status | Target Code | Notes |
|-------|--------|-------------|-------|
| QVLA (2602.03782) | ANALYZED | optimize/quantize.py | Action-centric quant, Phase 4. Code: github.com/AutoLab-SAI-SJTU/QVLA |
| SmolVLA (2506.01844) | ANALYZED | models/smolvla.py | Flow matching, 512x512, Phase 2. Analysis: docs/SMOLVLA_ANALYSIS.md |
| DyQ-VLA (2603.07904) | TRACKED | optimize/ (future) | Dynamic bit-width by kinematic state |
| PD-VLA (2503.02310) | TRACKED | N/A | Training-free parallel decoding for autoregressive VLAs |
| VLASH (2512.01031) | TRACKED | N/A | Async inference pattern (MIT Han Lab) |
| LiteVLA-Edge (2603.03380) | BASELINE | - | 6.6 Hz reference on Jetson Orin |
| VLA-Perf (2602.18397) | TRACKED | - | Analytical profiling model |
| NanoVLA (2510.25122) | TRACKED | - | Vision-language decoupling + dynamic routing |
| TurboQuant (2504.19874) | TRACKED | N/A | KV cache quant - low relevance for VLA (short sequences) |
| EaqVLA (2505.21567) | TRACKED | optimize/ (future) | Encoding-aligned VLA quantization |

Statuses: TRACKED -> ANALYZED -> IMPLEMENTING -> IMPLEMENTED -> ARCHIVED

---

## Doc Hygiene Process

### Freshness Windows

| Doc | Window | Staleness Signal |
|-----|--------|------------------|
| CLAUDE.md | Must match code always | Any claim contradicted by src/ |
| LEARNING.md | Append-only journal | Missing entry for a shipped feature |
| QUESTIONS.md | Weekly review | >7 days since last edit |
| IDEAS.md | Monthly review | >30 days since last edit |
| META_ENGINEERING.md | Update after each phase | New phase started without update |
| SCHEDULED_TASKS.md | Must match live triggers | Trigger IDs or schedules change |
| docs/research/*.md | Valid until implemented or superseded | Paper's insights fully in src/ |

### Decision Tree: Update vs Archive vs Delete

- **Update** if: doc covers an active concept and info is just outdated
- **Archive** to `docs/archive/` if: analysis informed a decision already made, insights fully implemented
- **Delete** if: duplicate or draft superseded by final version. Prefer archive.

### Automated Detection

The Friday scheduled task checks: CLAUDE.md vs code drift, QUESTIONS.md freshness,
LEARNING.md activity, paper implementation status. `/doc-hygiene` skill runs on demand.

---

## Review Log

Track /critic and /review-panel effectiveness over time.

| Date | Tool | Findings | Critical | Instincts Created |
|------|------|----------|----------|-------------------|
| 2026-03-30 | /critic | 10 | 3 | (fixed inline: trust_remote_code, registry, aarch64) |
| 2026-03-30 | /code-review | 16 | 3 | (overlapping with /critic) |
| 2026-03-30 | /review-panel | 12 | 4 (unanimous) | ADR-005 (profiler first), ADR-006 (leaderboard) |

---

## Self-Improvement Rules

### Proactive Process Improvement
After every /critic, /review-panel, or Phase completion, ask:
- "Is there a **hook** that could have prevented this automatically?"
- "Is there a **skill** that would make this workflow reusable?"
- "Is there an **instinct** that captures a new design rule?"
- "Should an existing hook/skill/instinct be **updated** based on what we learned?"

Bring up candidates for discussion. Don't auto-create - discuss first.

### Instinct Creation
- Critical finding NOT caught by existing instinct -> candidate for new instinct
- Pattern appearing in 2+ reviews -> MUST become instinct
- Do NOT inflate beyond current count unless a clear pattern emerges

### Skill Evolution
- After each use of /critic or /review-panel, check: did the prompt miss something?
- Update the skill's file list or evaluation criteria if needed
- /critic auto-discovers modules (reads CLAUDE.md + lists src/) rather than hardcoded files

### Persona Calibration (/review-panel)
- Production Engineer: strong on memory/latency, missed trust_remote_code in Phase 1
- Startup ML Engineer: strong on DX/onboarding, no false positives yet
- Open Source Strategist: called the leaderboard (ADR-006), weakest on technical correctness
- Update after each panel session

---

## Knowledge Location Matrix

Where to put things so they don't get duplicated:

| Knowledge type | Where it lives | Why |
|---------------|----------------|-----|
| Hard constraint (will crash if violated) | Instinct + CLAUDE.md | Auto-enforced |
| Architecture decision + reasoning | ADR in this file | Needs reasoning trail |
| Concept explanation (for learning/book) | LEARNING.md | Append-only, feeds Manning book |
| Paper deep dive | docs/research/PAPER_ANALYSIS.md | One file per paper |
| Open question (generative) | QUESTIONS.md | Reviewed weekly |
| Concrete idea (log and move on) | IDEAS.md | Reviewed monthly |
| Implementation detail | Code comments in src/ | Close to the code |
| API contract / usage | CLAUDE.md | First thing every session reads |
| Process / workflow | This file (META_ENGINEERING.md) | How we work |

**Rule**: Factual claims have ONE canonical location (CLAUDE.md Research Insights).
Other docs reference it, don't repeat it.
