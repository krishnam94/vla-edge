# Scheduled Tasks for vla-edge

Live on claude.ai/code/scheduled. All run at ~3 AM PT (10 AM UTC).

## Architecture

Cloud tasks run on Anthropic's servers. They CAN use WebSearch and WebFetch.
They CANNOT use gh CLI, git, or any local tools (Lesson 002).

Output is viewable at claude.ai/code/scheduled. The `/morning` skill bridges
the gap: it checks for digest issues on GitHub, and if none exist, runs a
quick local research scan and creates the issue using your authenticated gh CLI.

```
Cloud (3 AM)           Local (/morning, morning)
──────────             ────────────────────────
WebSearch for papers   Read overnight output from claude.ai
WebSearch for models   OR run quick local research
Generate insights      Create GitHub issue (gh issue create)
Output to session      Show priorities for today
```

---

## Task 1: Daily Digest + VLA Model Scan
**ID**: trig_011zbXF2sXPktw41MCjSgxLD
**Schedule**: Daily 3:07 AM PT (10:07 UTC)
**Tools**: WebSearch, WebFetch

Scans: VLA models on HuggingFace, reviewer calls, healthcare AI papers,
robotics/edge AI papers, EB-1A citations, Claude Code updates, one cross-domain insight.

**Output**: Structured report in session (viewable at claude.ai).

---

## Task 2: Research Scan (Mon + Fri)
**ID**: trig_01AjkQfFmEQW6s3Zb7ZrHgux
**Schedule**: Mon+Fri 3:23 AM PT (10:23 UTC)
**Tools**: WebSearch, WebFetch

Scans: arXiv papers (VLA edge, quantization, optimization), Semantic Scholar
citations to 7 key papers, competitor repos (VLA-Perf, vla-eval, LeRobot, SafeVLA),
3 AI-generated novel questions.

**Output**: Structured report in session (viewable at claude.ai).

---

## Task 3: Friday Review + Planning
**ID**: trig_01TUcuKYujeL18fJypMVyZjo
**Schedule**: Friday 3:41 AM PT (10:41 UTC)
**Tools**: WebSearch, WebFetch

Scans: GitHub commits/issues this week, competitor activity, LEARNING.md/QUESTIONS.md
freshness, collision of the week exploration, top 3 priorities, strategic flags,
monthly Hamming audit (first Friday).

**Output**: Structured report in session (viewable at claude.ai).

---

## GitHub Issue Labels for Digest

Created and ready (issues created by /morning locally):
- `daily-digest` - Morning briefing
- `paper-scan` - Research scan results
- `weekly-status` - Friday review

---

## Lesson Learned

Cloud scheduled tasks cannot create GitHub issues directly (no gh CLI auth).
The /morning skill handles issue creation from the local machine. See Lesson 002
in META_ENGINEERING.md.
