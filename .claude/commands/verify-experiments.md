---
description: Critically verify experiment settings, methodology, and results for correctness
argument-hint: Path to results JSON or experiment script
---

# Experiment Results Verifier

You are a skeptical reviewer checking if experimental results are CORRECT,
not just novel. Your job: find bugs, artifacts, and misleading numbers.

Target: $ARGUMENTS (or all experiments in experiments/ if not specified)

## Verification Checklist

For each experiment result file:

### 1. Methodology Check
- Is the measurement measuring what it claims? (e.g., "speedup" actually timing the right thing)
- Are caches flushed between runs? (Lesson 007: SmolVLA action queue caching)
- Is the sample size sufficient for the claim? (n<30 = suspicious for statistical claims)
- Are baseline and treatment seeing the SAME inputs?
- Are random seeds set for reproducibility?

### 2. Number Sanity
- Do the numbers make physical sense? (e.g., 3ms for a 450M model = suspicious)
- Do computed values match raw data? (e.g., violation_rate = violations / total)
- Are min/max/avg consistent? (avg should be between min and max)
- Do percentages sum to <= 100%?
- Are latency numbers consistent with known hardware capabilities?

### 3. Comparison Fairness
- Are methods compared on the SAME data?
- Is the baseline fair? (not artificially weak)
- Are overhead measurements isolated from model compute?
- Would a different random seed change the conclusion?

### 4. Statistical Validity
- Are confidence intervals reported for variable measurements?
- Is the variance reported alongside means?
- For A/B comparisons: is the difference statistically significant?
- Could the result be explained by noise alone?

### 5. Artifact Detection
- Is the result too good to be true? (16.9x speedup needs justification)
- Does the result contradict known properties? (e.g., more steps = worse = suspicious)
- Are there hidden confounds? (cache state, warmup, GC, thermal throttling)

### 6. Data Leakage and Methodology Integrity
- **Calibration leakage**: Are safety bounds/thresholds computed from the SAME data they're tested on? (e.g., bounds = mean +/- 4*std from test actions = leakage). Calibration and test data must be separate.
- **Condition independence**: Are both conditions (with/without SafeContract) run on the SAME episodes with the SAME seeds? Or could ordering effects bias results?
- **Circular baselines**: Is the "clean" policy designed to pass by construction? If so, the zero-false-positive claim is tautological, not empirical.
- **Designed distinctness**: Were synthetic distributions designed to be different, then shown to be different? This is circular. Real model data on genuinely different tasks is needed.
- **Environment confounds**: Could fingerprint differences come from different scene layouts (environments) rather than different task behaviors?
- **Future information**: Does any enforcement step use future actions (lookahead) that wouldn't be available at deployment time?
- **Normalization direction**: Is noise added in raw space or normalized space? Are violations checked on the right representation?
- **Model warmup**: Could the first condition benefit from/suffer from model warmup effects that the second condition doesn't experience?
- **Episode independence**: Are episodes truly independent? (different seeds, different init states) Or are they correlated?

## Output Format

For each result:
### [Experiment Name]
**Verdict**: CORRECT / LIKELY CORRECT / SUSPICIOUS / LIKELY WRONG

**Issues found**:
1. [issue] - [severity: critical/important/minor]

**What to fix before publishing**:
- [specific action]

## Rules
- Read the experiment SCRIPT (not just results) to understand methodology
- Check for the caching bug from Lesson 007 (SmolVLA policy.reset())
- If you find a critical issue, say STOP - do not publish this number
- Better to have no number than a wrong number in a paper
