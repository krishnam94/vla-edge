#!/usr/bin/env python3
"""EXP-VIOLATION-PRED: Do violation rates predict episode failure?

Analyzes existing closed-loop data to test whether episodes with
higher violation rates are more likely to fail. No new inference needed.

Tests: Mann-Whitney U, point-biserial correlation, logistic regression, AUROC.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
EXP_RESULTS_DIR = Path(__file__).parent / "results"


def analyze_dataset(name: str, filepath: Path) -> dict:
    """Analyze violation-failure correlation for one dataset."""
    with open(filepath) as f:
        data = json.load(f)

    episodes = data["episodes"]["with_contract"]
    if not episodes:
        return {"name": name, "error": "no with_contract episodes"}

    successes = []
    violation_rates = []
    violations_raw = []

    for ep in episodes:
        success = 1 if ep["success"] else 0
        viol_rate = ep["violations"] / max(ep["steps"], 1)
        successes.append(success)
        violation_rates.append(viol_rate)
        violations_raw.append(ep["violations"])

    successes = np.array(successes)
    violation_rates = np.array(violation_rates)
    violations_raw = np.array(violations_raw)

    n_success = int(successes.sum())
    n_fail = len(successes) - n_success

    if n_success == 0 or n_fail == 0:
        return {
            "name": name,
            "n_episodes": len(episodes),
            "n_success": n_success,
            "n_fail": n_fail,
            "error": "all same outcome, cannot compute correlation",
        }

    # Split violation rates by outcome
    vr_success = violation_rates[successes == 1]
    vr_fail = violation_rates[successes == 0]

    # Mann-Whitney U test
    u_stat, mw_p = stats.mannwhitneyu(vr_fail, vr_success, alternative="greater")

    # Point-biserial correlation (failure=1 vs violation_rate)
    # Flip success to failure for positive correlation
    failures = 1 - successes
    pb_r, pb_p = stats.pointbiserialr(failures, violation_rates)

    # Logistic regression (simple, via statsmodels-free approach)
    # Use sklearn-free logistic: just compute AUROC directly
    # AUROC: probability that a random failure has higher violation_rate than a random success
    auroc = u_stat / (n_fail * n_success)

    # Threshold analysis: find optimal threshold
    thresholds = np.linspace(violation_rates.min(), violation_rates.max(), 100)
    best_f1 = 0
    best_thresh = 0
    best_prec = 0
    best_rec = 0

    for thresh in thresholds:
        pred_fail = violation_rates > thresh
        tp = int(np.sum(pred_fail & (failures == 1)))
        fp = int(np.sum(pred_fail & (failures == 0)))
        fn = int(np.sum(~pred_fail & (failures == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_prec = prec
            best_rec = rec

    result = {
        "name": name,
        "n_episodes": len(episodes),
        "n_success": n_success,
        "n_fail": n_fail,
        "success_violation_rate": {
            "mean": float(vr_success.mean()),
            "std": float(vr_success.std()),
            "median": float(np.median(vr_success)),
        },
        "failure_violation_rate": {
            "mean": float(vr_fail.mean()),
            "std": float(vr_fail.std()),
            "median": float(np.median(vr_fail)),
        },
        "mann_whitney": {
            "U": float(u_stat),
            "p_value": float(mw_p),
            "significant": mw_p < 0.05,
        },
        "point_biserial": {
            "r": float(pb_r),
            "p_value": float(pb_p),
            "significant": pb_p < 0.05,
        },
        "auroc": float(auroc),
        "best_threshold": {
            "violation_rate": best_thresh,
            "precision": best_prec,
            "recall": best_rec,
            "f1": best_f1,
        },
    }

    return result


def main():
    print("EXP-VIOLATION-PRED: Do violation rates predict episode failure?")
    print("=" * 60)

    results = []

    # ACT holdout (primary)
    holdout_file = RESULTS_DIR / "closed_loop_eval_holdout.json"
    if holdout_file.exists():
        r = analyze_dataset("ACT-holdout (n=50)", holdout_file)
        results.append(r)
        print(f"\n{r['name']}:")
        print(f"  Success viol rate: {r['success_violation_rate']['mean']:.4f} +/- {r['success_violation_rate']['std']:.4f}")
        print(f"  Failure viol rate: {r['failure_violation_rate']['mean']:.4f} +/- {r['failure_violation_rate']['std']:.4f}")
        print(f"  Mann-Whitney p = {r['mann_whitney']['p_value']:.4f} {'*' if r['mann_whitney']['significant'] else 'ns'}")
        print(f"  Point-biserial r = {r['point_biserial']['r']:.4f}, p = {r['point_biserial']['p_value']:.4f}")
        print(f"  AUROC = {r['auroc']:.4f}")
        print(f"  Best threshold: vr > {r['best_threshold']['violation_rate']:.4f} -> P={r['best_threshold']['precision']:.2f}, R={r['best_threshold']['recall']:.2f}, F1={r['best_threshold']['f1']:.2f}")

    # ACT original (secondary)
    orig_file = RESULTS_DIR / "closed_loop_eval_50ep.json"
    if orig_file.exists():
        r = analyze_dataset("ACT-original (n=50)", orig_file)
        results.append(r)
        print(f"\n{r['name']}:")
        if "error" not in r:
            print(f"  Success viol rate: {r['success_violation_rate']['mean']:.4f} +/- {r['success_violation_rate']['std']:.4f}")
            print(f"  Failure viol rate: {r['failure_violation_rate']['mean']:.4f} +/- {r['failure_violation_rate']['std']:.4f}")
            print(f"  Mann-Whitney p = {r['mann_whitney']['p_value']:.4f} {'*' if r['mann_whitney']['significant'] else 'ns'}")
            print(f"  AUROC = {r['auroc']:.4f}")
        else:
            print(f"  {r.get('error', 'unknown error')}")

    # PushT (if enough failures)
    pusht_file = RESULTS_DIR / "closed_loop_diffusion_pusht_10ep.json"
    if pusht_file.exists():
        r = analyze_dataset("PushT-Diffusion (n=10)", pusht_file)
        results.append(r)
        print(f"\n{r['name']}:")
        if "error" not in r:
            print(f"  AUROC = {r['auroc']:.4f}")
            print(f"  Mann-Whitney p = {r['mann_whitney']['p_value']:.4f}")
        else:
            print(f"  {r.get('error', 'unknown error')}")

    output = {
        "experiment": "EXP-VIOLATION-PRED: Violation Rate as Failure Predictor",
        "description": "Tests whether per-episode violation rate predicts task failure",
        "datasets": results,
    }

    out_path = EXP_RESULTS_DIR / "exp_violation_predictiveness.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
