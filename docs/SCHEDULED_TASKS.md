# Scheduled Tasks for vla-edge

Set up at: https://claude.ai/code/scheduled
Repo: krishnam94/vla-edge
Model: Opus 4.6 (1M context)

---

## Task 1: Daily VLA Model Scan

**Name**: vla-edge: Daily Model Scan
**Schedule**: Daily, 8:00 AM PT
**Repo**: krishnam94/vla-edge

```
Search HuggingFace API for new robotics models (pipeline_tag=robotics)
published in the last 24 hours. Check for models matching VLA patterns
(smolvla, openvla, pi0, octo, minivla, litevla, groot, nanovla, edgevla,
rt-2, cogact in tags or name).

For each new model found:
1. Check if a GitHub issue already exists for it in krishnam94/vla-edge
2. If not, create a GitHub issue with label 'new-model-candidate' including:
   - Model name and HuggingFace link
   - Estimated param count (from model card or config)
   - Whether it's likely edge-deployable (<3B params)
   - Architecture type if identifiable

If no new models found, do nothing (no noise).
```

---

## Task 2: Weekly Paper + Competitor Scan (Monday)

**Name**: vla-edge: Monday Research Scan
**Schedule**: Weekly, Monday 9:00 AM PT
**Repo**: krishnam94/vla-edge

```
Do three scans and combine into one GitHub issue:

SCAN 1 - arXiv Papers:
Search arXiv (cs.RO, cs.AI, cs.CV, cs.LG) for papers from the past 7 days:
- "VLA" AND ("edge" OR "deployment" OR "quantization" OR "optimization" OR "efficient")
- "robot policy" AND ("Jetson" OR "embedded" OR "real-time" OR "inference")
- "action tokenization" OR "speculative decoding" AND "robot"
- "CUDA kernel" AND ("transformer" OR "attention" OR "inference")
- "vision language action" AND ("pruning" OR "distillation" OR "compression")

SCAN 2 - Citations:
Check Semantic Scholar for new citations to these papers since last Monday:
- QVLA (2602.03782), PD-VLA (2503.02310), VLASH (2512.01031)
- LiteVLA-Edge (2603.03380), VLA-Perf (2602.18397), NanoVLA (2510.25122)

SCAN 3 - Competitor Activity:
Check these repos for new releases, major commits, or announcements:
- NVlabs/vla-perf (NVIDIA's profiler)
- allenai/vla-evaluation-harness (Allen AI's eval)
- huggingface/lerobot (LeRobot updates)
- PKU-Alignment/SafeVLA (safety competitor)

Combine into one GitHub issue with label 'paper-scan':
- Section 1: New Papers (title, arXiv link, one-line summary, relevance)
- Section 2: New Citations (who cited what, paper link)
- Section 3: Competitor Updates (what changed, link)
- Flag any paper with open-source code prominently
- Flag anything that directly competes with or validates vla-edge

If nothing notable found, don't create an issue.
```

---

## Task 3: Friday Project Health Check

**Name**: vla-edge: Friday Review
**Schedule**: Weekly, Friday 5:00 PM PT
**Repo**: krishnam94/vla-edge

```
Review krishnam94/vla-edge and create a status summary as a GitHub issue
with label 'weekly-status':

1. PROGRESS: List commits this week with one-line summaries
2. TESTS: Are CI tests passing? Any new test failures?
3. ISSUES: Count open issues by label. Any stale (>14 days no activity)?
4. COVERAGE: If coverage data available, report current %
5. BACKLOG: List the top 3 highest-priority open issues for next week
6. LEARNING: Check if docs/LEARNING.md was updated this week. If not, flag it.
7. COMPETITORS: Quick check - did VLA-Perf, vla-eval, or LeRobot release
   anything this week that affects our roadmap?

Keep it concise - this is a 2-minute read, not a report.
```

---

## Task 4: Monthly Architecture Review

**Name**: vla-edge: Monthly Architecture Review
**Schedule**: Monthly, 1st of month, 10:00 AM PT
**Repo**: krishnam94/vla-edge

```
Do a thorough monthly review of krishnam94/vla-edge:

1. ARCHITECTURE: Read CLAUDE.md and all files in src/vla_edge/. Are the
   abstractions still right? Has any module grown beyond its intended scope?
   Flag any code smells or architectural drift.

2. DEPENDENCIES: Check pyproject.toml. Are any deps outdated? Any new
   security advisories? Run a conceptual dependency audit.

3. COMPETITION: Deep check on VLA-Perf, vla-eval, LeRobot, and any new
   tools. Has anyone shipped something that overlaps with vla-edge?
   Should we adjust our roadmap?

4. PAPERS: Review docs/notes/research_vla_edge_papers_2025_2026.md.
   Are there papers from the past month we should incorporate? Any new
   optimization techniques we're missing?

5. COMMUNITY: Check GitHub stars, forks, issues from external users.
   Any patterns in what people are asking for?

6. RECOMMENDATIONS: Based on the above, suggest 1-3 specific actions
   for the coming month.

Create a GitHub issue with label 'monthly-review' with the full report.
```

---

## Why NOT More Tasks

We considered and rejected:
- **Daily code review**: Overkill without daily commits. Weekly is enough.
- **Daily competitor scan**: Nothing changes daily. Weekly Monday scan covers this.
- **Auto-implementation of issues**: Claude can't autonomously build features
  well enough without human judgment. Dev is interactive, not scheduled.
- **Hourly anything**: Noise. Even daily is borderline for model scanning.

---

## Labels Needed for Scheduled Tasks

Already created:
- `new-model-candidate` (green)
- `paper-scan` (purple)

Need to create:
- `weekly-status` (gray)
- `monthly-review` (gold)
