---
description: Manage, validate, and audit experiments. Track linkage between experiment results and paper claims.
argument-hint: "register <name> | validate <name> | status | link <exp> <section> | audit"
---

# /experiment - Experiment Management

Structured tracking of all experiments, their validation status, and linkage to paper claims. Prevents the class of bugs we found in this paper (wrong step counts, untested claims, inconsistent numbers).

## Sub-commands

### `register <name>`
Register a new experiment. This MUST happen BEFORE running.

1. Ask for: hypothesis, script path, results path, key metrics, paper section it supports
2. **MANDATORY: Design review checklist** (answer before running):
   - What EXACTLY does each metric measure? (e.g., "velocity" = diff between WHAT?)
   - Are samples consecutive/independent/stratified? Does the metric require a specific sampling?
   - Does the expected result make physical sense? (e.g., 98% violation rate on expert data = red flag)
   - Could the metric be inflated by the sampling strategy? (non-sequential diff, overlapping windows)
   - What is the null hypothesis? What would a meaningless result look like?
   - Are the baseline/test conditions truly comparable? (same data, same model, same seeds)
3. Add entry to `experiments/icra_ws_2026/EXPERIMENT_REGISTRY.md` with design review answers
4. Mark status as REGISTERED (not yet run)
5. ONLY THEN proceed to implementation and execution

### `validate <name>`
Run the `/verify-experiments` checklist on a specific experiment:
1. Read the experiment script AND its results JSON
2. Check: do paper numbers match JSON? Methodology described accurately? Sample size sufficient?
3. Check: random seeds set? Queue flushed (Lesson 007)? Normalization correct (Lesson 010)?
4. Check: all claimed comparisons actually run? (Lesson: "never claim all X without testing all X")
5. **Data leakage check**: Are calibration parameters (bounds, thresholds) computed from separate data than test data? Are both conditions using the same episodes/seeds?
6. **Circular baseline check**: Is any result true by construction? (e.g., "clean policy has 0 violations" when clean = within bounds by design)
7. **Confound check**: Could the result be explained by something other than what's claimed? (normalization mismatch, environment differences, model warmup)
8. Mark status as VALIDATED or FAILED with issues list
9. Update EXPERIMENT_REGISTRY.md

### `status`
Show all experiments with:
- Name, script, results file, validation status
- Which paper section each supports
- Last validated date
- Any UNVALIDATED or FAILED experiments (blocking submission)

### `link <exp> <paper-section>`
Connect an experiment to a paper claim:
- Record: "EXP-H supports Section 4.5 claim about fingerprint distinctness"
- Enable `/experiment audit` to cross-check

### `audit`
Cross-reference ALL paper claims against experiment results:
1. Read the paper (main.tex)
2. For each number/claim, find the supporting experiment
3. Verify the number matches the JSON
4. Flag: claims with no experiment support, experiments with no paper citation
5. Output: PASS/FAIL table (like the experiment validator agent produces)

## Integration

- Run `/experiment validate` after EVERY experiment completes (mandatory, like /verify-experiments)
- Run `/experiment audit` before paper submission
- `/experiment status` in Friday review and daily standup
- `/skill-sync` checks that all experiments are validated before submission

## Experiment Registry Format

Each entry in EXPERIMENT_REGISTRY.md:
```
### EXP-<ID>: <name>
- **Hypothesis**: What we're testing
- **Script**: `experiments/icra_ws_2026/<script>.py`
- **Results**: `experiments/icra_ws_2026/results/<results>.json`
- **Key metrics**: violation rate, accuracy, overhead, etc.
- **Paper section**: Section X.Y, claim "..."
- **Status**: VALIDATED / UNVALIDATED / FAILED
- **Validated**: YYYY-MM-DD
- **Issues**: (if any)
```

## Instinct Integration

These instincts should fire during experiment work:
- `verify-all-claims`: Never claim "all X" without testing all X
- `check-normalization`: Verify action normalization when connecting model to env
- `synthetic-data-honesty`: Add caveat when presenting synthetic results as general claims
- `verify-references`: Cross-check experiment numbers against paper claims
