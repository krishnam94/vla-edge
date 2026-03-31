---
description: vla-edge morning briefing - overnight research, project status, suggest next steps
allowed-tools: Bash(gh issue *), Bash(git *), Bash(pytest *), Read, Glob, Grep, WebSearch, WebFetch
---

# vla-edge Morning Briefing

Complete morning workflow: check overnight insights, run fresh research, create
a traceable GitHub issue, and suggest what to work on today.

## Step 1: Quick Project Status (30 seconds)

Run these in parallel:
- `git log --oneline -5` and `git status`
- `pytest -x -q --tb=short -m "not gpu and not jetson and not slow" 2>&1 | tail -3`
- `gh issue list --repo krishnam94/vla-edge --state open --limit 10`

## Step 2: Read Overnight Scheduled Task Outputs (if any exist as issues)

Check for issues created by overnight scheduled tasks:
- `gh issue list --repo krishnam94/vla-edge --label daily-digest --limit 1 --json number,title,body,createdAt`
- `gh issue list --repo krishnam94/vla-edge --label paper-scan --limit 1 --json number,title,body,createdAt`
- `gh issue list --repo krishnam94/vla-edge --label weekly-status --limit 1 --json number,title,body,createdAt`

If recent digest exists (< 24 hours old), summarize it. If not, run Step 3.

## Step 3: Quick Research Scan (only if no overnight digest exists)

Launch a quick research agent (WebSearch) checking:
1. HuggingFace for new VLA/robotics models in last 48 hours
2. arXiv cs.RO for VLA edge/optimization papers in last 48 hours
3. One cross-domain insight (game engines, audio DSP, mobile ML, DevOps, formal methods)

Keep it fast - 3 queries max, 2 minutes total.

## Step 4: Create Morning Digest Issue

Create a GitHub issue with the combined status + research:

```bash
gh issue create --repo krishnam94/vla-edge \
  --label 'daily-digest' \
  --title 'Daily Digest: [DATE]' \
  --body '[FORMATTED REPORT]'
```

## Step 5: Read Current Phase and Suggest Next Steps

Read CLAUDE.md and docs/META_ENGINEERING.md for current phase.
Based on: overnight insights + open issues + current phase, recommend 2-3 tasks.

## Output Format

```
## vla-edge Morning Briefing [DATE]

**Branch**: main | **Tests**: X passing | **Phase**: N
**Open issues**: X total (Y phase-2, Z phase-3)

### Overnight Insights
- New models: [any] or none
- New papers: [any] or none
- Cross-domain: [insight]
- Generated questions: [from research scan]

### Today's Priorities
1. [highest impact task with issue #]
2. [next task]
3. [next task]

### Start with: [specific task and first command to run]
```

## After Reading
- Interesting cross-domain insight -> add to docs/IDEAS.md
- Good generated question -> add to docs/QUESTIONS.md
- REVIEW-PANEL NEEDED flag -> run /review-panel before coding
- Close the daily-digest issue after reading (or leave open for reference)
