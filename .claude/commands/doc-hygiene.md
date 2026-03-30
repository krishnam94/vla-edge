---
description: Check all docs for staleness, inconsistencies, and archive candidates
---

# Doc Hygiene Check

Scan all project documentation for staleness, factual drift, and inconsistencies.

## Steps

1. **CLAUDE.md vs code drift**: Read CLAUDE.md architecture section. List all modules
   mentioned. Then list actual modules in src/vla_edge/. Flag any mismatch.

2. **Freshness check**: For each doc, check last modified date against its freshness window:
   - CLAUDE.md: must match current code (always)
   - QUESTIONS.md: must be modified within 7 days
   - IDEAS.md: must be modified within 30 days
   - LEARNING.md: should have entry for each shipped feature
   - META_ENGINEERING.md: should reflect current phase
   - SCHEDULED_TASKS.md: must match live scheduled task IDs

3. **Cross-doc consistency**: Check these known duplication points:
   - SmolVLA architecture claims (CLAUDE.md vs SMOLVLA_ANALYSIS.md vs LEARNING.md)
   - Latency bottleneck claims (autoregressive vs flow matching distinction)
   - Paper status (META_ENGINEERING.md tracking table vs actual code in src/)
   - Instinct list (META_ENGINEERING.md vs actual .yml files in .claude/homunculus/instincts/)

4. **Archive candidates**: Flag any doc in docs/ or docs/research/ where:
   - The paper's insights are fully implemented in src/
   - The analysis has been superseded by a newer paper
   - The content is >3 months old with no references from other docs

5. **Instinct relevance**: For each instinct .yml file, check if the trigger
   condition still applies to current code patterns.

## Output Format

```
## Doc Hygiene Report

### Stale Docs
- [file] - last modified [date], freshness window: [X days], STATUS: [stale/fresh]

### Inconsistencies Found
- [claim in file A] contradicts [claim in file B]

### Archive Candidates
- [file] - reason: [superseded/implemented/orphaned]

### Instinct Check
- [instinct] - still relevant: yes/no, reason: [...]

### Recommendations
1. Update [file] because [reason]
2. Archive [file] to docs/archive/
3. Remove instinct [name] because [reason]
```
