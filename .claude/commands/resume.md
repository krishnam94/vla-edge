---
description: Resume work on vla-edge - show status and suggest next steps
---

# Resume vla-edge Development

Show current project status and suggest what to work on next.

## Steps

1. **Git status**: Show recent commits, any uncommitted changes, current branch
   - Run: `git log --oneline -10`
   - Run: `git status`

2. **Test status**: Run the test suite quickly
   - Run: `pytest -x -q --tb=short -m "not gpu and not jetson and not slow"`

3. **Open issues**: Check GitHub for open issues and priorities
   - Run: `gh issue list --repo krishnam94/vla-edge --state open --limit 15` (if gh available)
   - Or: read from docs/META_ENGINEERING.md for tracked work

4. **Phase check**: Read CLAUDE.md and docs/META_ENGINEERING.md to determine current phase

5. **Suggest next steps**: Based on the above, recommend 2-3 concrete tasks to work on

## Output Format

```
## vla-edge Status

**Branch**: main
**Last commit**: <hash> <message> (<when>)
**Tests**: X passing, Y failing
**Current phase**: Phase N - <name>

## Open Work
- [ ] <highest priority item>
- [ ] <next priority item>
- [ ] <next priority item>

## Suggested: Start with <specific task>
```
