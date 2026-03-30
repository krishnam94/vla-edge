# Scheduled Tasks for vla-edge

Live on claude.ai/code/scheduled. All run at ~3 AM PT to avoid session limits.

---

## Task 1: Daily Digest + VLA Model Scan
**ID**: trig_011zbXF2sXPktw41MCjSgxLD
**Schedule**: Every night 3:07 AM PT
**Model**: Sonnet 4.6

Combines the personal daily research digest with vla-edge model scanning:
1. **VLA Models**: New HuggingFace robotics models matching VLA patterns
2. **Review Opportunities**: Conference/journal reviewer calls, CFPs
3. **Healthcare AI**: New papers/tools from arxiv and journals
4. **Robotics/Edge AI**: VLA deployment, edge inference, Jetson papers
5. **EB-1A**: Citation tracking for MergeNet and HealthPulse
6. **Claude Code**: Changelog and feature updates

**What to do with output**: Scan in 2 min each morning. Act on tagged items.

---

## Task 2: Monday Research Scan
**ID**: trig_01AjkQfFmEQW6s3Zb7ZrHgux
**Schedule**: Mondays 3:23 AM PT
**Model**: Sonnet 4.6

Deep research scan for vla-edge:
1. **arXiv papers**: VLA edge deployment, quantization, optimization, CUDA kernels
2. **Citation tracking**: Who cited QVLA, PD-VLA, VLASH, LiteVLA-Edge, VLA-Perf, NanoVLA
3. **Competitor watch**: VLA-Perf, vla-eval, LeRobot, SafeVLA new releases

**What to do with output**: Read Monday morning. If strategic question found,
run `/review-panel <topic>` before starting the week's work.

---

## Task 3: Friday Review + Next Week Planning
**ID**: trig_01TUcuKYujeL18fJypMVyZjo
**Schedule**: Fridays 3:41 AM PT
**Model**: Sonnet 4.6

Weekly project health check:
1. **Progress**: Commits this week
2. **Issues**: Open count by label, stale flags
3. **Tests**: CI status
4. **Competitors**: Anything shipped this week that affects us?
5. **Process**: Was LEARNING.md updated?
6. **Next week**: Top 3 priorities recommended
7. **Strategic flags**: Explicitly calls out "REVIEW-PANEL NEEDED: [topic]" if
   any competitor move or paper finding needs a strategic discussion

**What to do with output**: Read Friday morning. If REVIEW-PANEL flagged, run it.
Use the top 3 priorities to plan Monday's /feature-dev sessions.

---

## The Weekly Flow

```
Mon 3:23 AM  [auto] Monday Research Scan runs
Mon morning  [you]  Read scan -> /review-panel if strategic question
Mon-Thu      [you]  /feature-dev on issues, /critic on new code
Fri 3:41 AM  [auto] Friday Review runs
Fri morning  [you]  Read review -> update LEARNING.md -> plan next week
Daily 3:07AM [auto] Digest runs
Daily AM     [you]  2-min scan of digest
```

---

## Disabled Tasks
- **vla-edge: Daily Model Scan** (trig_019kPJJjqxbuZT5YYy8Wfabg) - merged into Task 1

## Why Only 3 Tasks
Plan limit is 3 cloud scheduled tasks. We merged the daily model scan into the
daily digest to free a slot for the Friday review. Monthly architecture review
is done manually (run `/review-panel` on the 1st of each month).
