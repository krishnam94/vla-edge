#!/usr/bin/env python3
"""Cross-architecture conformal p-value analysis on PushT results.

Post-processing: loads per-episode results from VQ-BeT and Diffusion on PushT,
computes conformal p-values, and compares distributions (KS test).
Runs AFTER the closed-loop experiments.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from datasets import load_dataset


def compute_conformal_pvalues(actions, cal_mean, cal_std, cal_scores):
    """Compute conformal p-values for a batch of actions."""
    pvalues = []
    n = len(cal_scores)
    for action in actions:
        per_dim = np.abs(action - cal_mean) / (cal_std + 1e-8)
        score = float(np.max(per_dim))
        rank = np.searchsorted(cal_scores, score, side="right")
        p = (n - rank + 1) / (n + 1)
        pvalues.append(p)
    return pvalues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="/workspace/results/")
    parser.add_argument("--output", type=str, default="/workspace/results/crossarch_conformal.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=== Cross-Architecture Conformal P-Value Analysis ===")

    # Load PushT demo data for calibration
    print("Loading PushT demo data...")
    ds = load_dataset("lerobot/pusht", split="train")
    demo_actions = np.array([row["action"] for row in ds], dtype=np.float32)
    print(f"  Demo actions: {demo_actions.shape}")

    # Split 80/20
    rng = np.random.RandomState(42)
    n_cal = int(len(demo_actions) * 0.8)
    indices = rng.permutation(len(demo_actions))
    cal_actions = demo_actions[indices[:n_cal]]
    holdout_actions = demo_actions[indices[n_cal:]]

    cal_mean = cal_actions.mean(axis=0)
    cal_std = np.maximum(cal_actions.std(axis=0), 1e-8)
    cal_scores = np.sort(np.max(np.abs(cal_actions - cal_mean) / cal_std, axis=1))

    print(f"  Calibration: {cal_actions.shape}, Holdout: {holdout_actions.shape}")

    # Compute p-values on holdout (should be ~uniform)
    holdout_pvals = compute_conformal_pvalues(holdout_actions[:500], cal_mean, cal_std, cal_scores)
    ks_holdout = sp_stats.kstest(holdout_pvals, "uniform")
    print(f"  Holdout p-values: mean={np.mean(holdout_pvals):.3f}, KS p={ks_holdout.pvalue:.4f}")

    # Load experiment results and extract violation info per architecture
    arch_results = {}

    for name, filename in [("Diffusion", "diffusion_pusht_200ep.json"),
                           ("VQ-BeT", "vqbet_pusht_200ep.json")]:
        filepath = results_dir / filename
        if not filepath.exists():
            print(f"  {name}: file not found at {filepath}, skipping")
            continue

        with open(filepath) as f:
            data = json.load(f)

        summary = data["summary"]
        no_sr = summary["no_contract"]["success_rate"]
        with_sr = summary["with_contract"]["success_rate"]
        no_viol = summary["no_contract"].get("total_violations", 0)
        with_viol = summary["with_contract"]["total_violations"]

        arch_results[name] = {
            "success_no_contract": no_sr,
            "success_with_contract": with_sr,
            "violations_no_contract": no_viol,
            "violations_with_contract": with_viol,
            "fisher_p": summary["fisher_p_value"],
            "n_episodes": summary["no_contract"]["total_episodes"],
        }
        print(f"  {name}: {no_sr:.0%}/{with_sr:.0%}, p={summary['fisher_p_value']}, viol={with_viol}")

    # Compare violation profiles
    print("\n=== Architecture Comparison on Same Dataset (PushT) ===")
    if len(arch_results) >= 2:
        for name, data in arch_results.items():
            print(f"  {name}: SR {data['success_no_contract']:.0%}->{data['success_with_contract']:.0%}, "
                  f"violations={data['violations_with_contract']}")

        # Statistical comparison of violation rates
        archs = list(arch_results.keys())
        if len(archs) >= 2:
            a1, a2 = archs[0], archs[1]
            v1 = arch_results[a1]["violations_with_contract"] / (arch_results[a1]["n_episodes"] * 300)
            v2 = arch_results[a2]["violations_with_contract"] / (arch_results[a2]["n_episodes"] * 300)
            print(f"\n  Violation rate per step: {a1}={v1:.4f}, {a2}={v2:.4f}")
            print(f"  Ratio: {v1/v2:.2f}x" if v2 > 0 else "  Cannot compute ratio")

    output = {
        "experiment": "Cross-architecture conformal analysis on PushT",
        "calibration": {
            "n_cal": int(n_cal),
            "holdout_mean_pvalue": float(np.mean(holdout_pvals)),
            "holdout_ks_pvalue": float(ks_holdout.pvalue),
        },
        "architectures": arch_results,
        "same_dataset": "PushT",
    }

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
