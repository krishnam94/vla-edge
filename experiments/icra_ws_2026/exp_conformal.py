#!/usr/bin/env python3
"""EXP-CONFORMAL: Conformal prediction bounds on ALOHA demo data.

Computes distribution-free safety bounds using split conformal prediction
and compares them to the heuristic 4-sigma bounds used in EXP-ACT-CL-HOLDOUT.

Key idea: conformal prediction gives guaranteed coverage (>= 1-alpha) on
held-out data without distributional assumptions. The 4-sigma heuristic
assumes Gaussianity and targets ~99.994% coverage, which is unnecessarily
conservative if 95% suffices.

This experiment also computes a data-driven v_max at the 99th percentile
of consecutive action deltas from calibration data, replacing the hand-tuned
v_max=0.05 or v_max=0.1 used in other experiments.

Data: lerobot/aloha_sim_transfer_cube_human (50 episodes, 20k frames, 14 dims)
Split: 80/20 episodes (seed=42, shuffled)
Nonconformity score: max_j |a_ij - cal_mean_j| / cal_std_j
Conformal level: alpha=0.05 (95% coverage guarantee)
Comparison: conformal bounds vs 4-sigma bounds

Usage:
  python experiments/icra_ws_2026/exp_conformal.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"

JOINT_NAMES = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]

# Parameters
CALIBRATION_SPLIT = 0.8
SEED = 42
ALPHA = 0.05          # conformal level (95% coverage)
N_SIGMA = 4.0         # heuristic bound width
V_MAX_PERCENTILE = 99  # for data-driven velocity limit


def split_episodes(ds, cal_frac: float = 0.8, seed: int = 42) -> tuple:
    """Split dataset into calibration and holdout by episode.

    Returns (cal_actions, holdout_actions, cal_episodes, holdout_episodes)
    where actions are (n_frames, n_dims) arrays.
    """
    episodes = sorted(set(ds["episode_index"]))
    n_episodes = len(episodes)

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(episodes).tolist()
    n_cal = int(n_episodes * cal_frac)
    cal_ep_set = set(shuffled[:n_cal])
    holdout_ep_set = set(shuffled[n_cal:])

    cal_actions = []
    holdout_actions = []
    # Also track episode boundaries for velocity computation
    cal_ep_actions = {}  # ep_idx -> list of actions (ordered by frame)
    holdout_ep_actions = {}

    for row in ds:
        ep = row["episode_index"]
        action = np.array(row["action"], dtype=np.float64)
        if ep in cal_ep_set:
            cal_actions.append(action)
            cal_ep_actions.setdefault(ep, []).append(action)
        elif ep in holdout_ep_set:
            holdout_actions.append(action)
            holdout_ep_actions.setdefault(ep, []).append(action)

    cal_actions = np.array(cal_actions)
    holdout_actions = np.array(holdout_actions)

    # Sort episode actions by frame order (dataset is already ordered, but be safe)
    for ep in cal_ep_actions:
        cal_ep_actions[ep] = np.array(cal_ep_actions[ep])
    for ep in holdout_ep_actions:
        holdout_ep_actions[ep] = np.array(holdout_ep_actions[ep])

    return (
        cal_actions,
        holdout_actions,
        sorted(cal_ep_set),
        sorted(holdout_ep_set),
        cal_ep_actions,
        holdout_ep_actions,
    )


def compute_nonconformity_scores(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Compute nonconformity score for each action.

    score_i = max_j |a_ij - mean_j| / std_j

    This is the maximum normalized absolute deviation across all dimensions.
    Using the max makes the resulting prediction region a hyperrectangle
    (box constraint), which is exactly what SafeContract enforces.
    """
    # Avoid division by zero for near-constant dimensions
    safe_std = np.where(std < 1e-8, 1.0, std)
    normalized_devs = np.abs(actions - mean) / safe_std  # (n, d)
    scores = np.max(normalized_devs, axis=1)  # (n,)
    return scores


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal prediction quantile.

    q = ceil((1-alpha)(1 + 1/n)) quantile of the scores.
    This guarantees P(Y_{n+1} in C) >= 1-alpha.
    """
    n = len(scores)
    level = (1 - alpha) * (1 + 1 / n)
    # Clip to 1.0 in case (1-alpha)(1+1/n) > 1
    level = min(level, 1.0)
    # np.quantile with interpolation='higher' gives the ceil behavior
    q = float(np.quantile(scores, level, method="higher"))
    return q


def compute_coverage(
    actions: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> dict:
    """Compute coverage statistics for bounds on a set of actions."""
    in_bounds = (actions >= lo) & (actions <= hi)
    all_in = np.all(in_bounds, axis=1)

    per_dim_coverage = in_bounds.mean(axis=0)  # (d,)
    overall_coverage = float(all_in.mean())

    # How many dims are violated per action (for violating actions)?
    n_violated_dims = (~in_bounds).sum(axis=1)  # (n,)
    violating_mask = ~all_in
    mean_violated_dims = float(n_violated_dims[violating_mask].mean()) if violating_mask.any() else 0.0

    return {
        "overall_coverage": overall_coverage,
        "per_dim_coverage": per_dim_coverage.tolist(),
        "n_violating_actions": int((~all_in).sum()),
        "n_total_actions": len(actions),
        "mean_violated_dims_when_violating": mean_violated_dims,
    }


def compute_velocity_stats(ep_actions: dict) -> tuple:
    """Compute consecutive action deltas within episodes.

    Returns (all_deltas, per_dim_stats) where all_deltas is (n, d).
    """
    all_deltas = []
    for ep_idx in sorted(ep_actions.keys()):
        actions = ep_actions[ep_idx]
        if len(actions) < 2:
            continue
        deltas = np.abs(np.diff(actions, axis=0))  # (T-1, d)
        all_deltas.append(deltas)

    all_deltas = np.concatenate(all_deltas, axis=0)  # (total, d)
    return all_deltas


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP-CONFORMAL: Conformal Prediction Bounds on ALOHA Demo Data")
    print("=" * 70)

    # --- Load and split ---
    print(f"\nLoading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  {len(ds)} frames, {len(set(ds['episode_index']))} episodes, {len(ds[0]['action'])} dims")

    print(f"\nSplitting episodes {int(CALIBRATION_SPLIT*100)}/{int((1-CALIBRATION_SPLIT)*100)} (seed={SEED})...")
    cal_actions, holdout_actions, cal_eps, holdout_eps, cal_ep_actions, holdout_ep_actions = split_episodes(
        ds, CALIBRATION_SPLIT, SEED
    )
    print(f"  Calibration: {len(cal_eps)} episodes, {cal_actions.shape[0]} actions")
    print(f"  Holdout: {len(holdout_eps)} episodes, {holdout_actions.shape[0]} actions")

    n_dims = cal_actions.shape[1]

    # --- Calibration statistics ---
    cal_mean = cal_actions.mean(axis=0)
    cal_std = cal_actions.std(axis=0)

    print(f"\nCalibration statistics ({n_dims} dims):")
    for j in range(n_dims):
        print(f"  {JOINT_NAMES[j]:25s}: mean={cal_mean[j]:+.4f}, std={cal_std[j]:.4f}")

    # --- Nonconformity scores on calibration set ---
    print(f"\nComputing nonconformity scores on calibration set...")
    cal_scores = compute_nonconformity_scores(cal_actions, cal_mean, cal_std)
    print(f"  Score range: [{cal_scores.min():.4f}, {cal_scores.max():.4f}]")
    print(f"  Score mean: {cal_scores.mean():.4f}, median: {np.median(cal_scores):.4f}")

    # --- Conformal threshold ---
    q_hat = conformal_quantile(cal_scores, ALPHA)
    print(f"\nConformal threshold (alpha={ALPHA}): q_hat = {q_hat:.4f}")
    print(f"  Level: (1-{ALPHA})*(1 + 1/{len(cal_scores)}) = {(1-ALPHA)*(1+1/len(cal_scores)):.6f}")

    # --- Conformal bounds ---
    safe_std = np.where(cal_std < 1e-8, 1.0, cal_std)
    conformal_lo = cal_mean - q_hat * safe_std
    conformal_hi = cal_mean + q_hat * safe_std

    print(f"\nConformal bounds (per dimension):")
    for j in range(n_dims):
        width = conformal_hi[j] - conformal_lo[j]
        print(f"  {JOINT_NAMES[j]:25s}: [{conformal_lo[j]:+.4f}, {conformal_hi[j]:+.4f}] (width={width:.4f})")

    # --- 4-sigma heuristic bounds ---
    heuristic_lo = cal_mean - N_SIGMA * cal_std
    heuristic_hi = cal_mean + N_SIGMA * cal_std

    print(f"\n4-sigma heuristic bounds (per dimension):")
    for j in range(n_dims):
        width = heuristic_hi[j] - heuristic_lo[j]
        print(f"  {JOINT_NAMES[j]:25s}: [{heuristic_lo[j]:+.4f}, {heuristic_hi[j]:+.4f}] (width={width:.4f})")

    # --- Width comparison ---
    conformal_widths = conformal_hi - conformal_lo
    heuristic_widths = heuristic_hi - heuristic_lo
    width_ratio = conformal_widths / np.where(heuristic_widths < 1e-8, 1.0, heuristic_widths)

    print(f"\nWidth comparison (conformal / heuristic):")
    for j in range(n_dims):
        print(f"  {JOINT_NAMES[j]:25s}: {width_ratio[j]:.4f}x ({conformal_widths[j]:.4f} vs {heuristic_widths[j]:.4f})")
    print(f"  Mean ratio: {width_ratio.mean():.4f}x")
    print(f"  Conformal is {'tighter' if width_ratio.mean() < 1.0 else 'wider'} on average")

    # --- Coverage on holdout set ---
    print(f"\n--- Coverage on holdout set ({holdout_actions.shape[0]} actions) ---")

    conformal_cov = compute_coverage(holdout_actions, conformal_lo, conformal_hi)
    heuristic_cov = compute_coverage(holdout_actions, heuristic_lo, heuristic_hi)

    print(f"\nConformal bounds (alpha={ALPHA}, target >= {1-ALPHA:.0%}):")
    print(f"  Overall coverage: {conformal_cov['overall_coverage']:.4f} ({conformal_cov['overall_coverage']*100:.2f}%)")
    print(f"  Violating actions: {conformal_cov['n_violating_actions']} / {conformal_cov['n_total_actions']}")
    print(f"  Mean violated dims (when violating): {conformal_cov['mean_violated_dims_when_violating']:.1f}")
    guarantee_met = conformal_cov["overall_coverage"] >= (1 - ALPHA)
    print(f"  Coverage guarantee met: {'YES' if guarantee_met else 'NO'}")

    print(f"\n4-sigma heuristic bounds:")
    print(f"  Overall coverage: {heuristic_cov['overall_coverage']:.4f} ({heuristic_cov['overall_coverage']*100:.2f}%)")
    print(f"  Violating actions: {heuristic_cov['n_violating_actions']} / {heuristic_cov['n_total_actions']}")

    # Per-dim coverage comparison
    print(f"\nPer-dim coverage comparison:")
    print(f"  {'Joint':25s}  {'Conformal':>10s}  {'4-sigma':>10s}")
    for j in range(n_dims):
        cc = conformal_cov["per_dim_coverage"][j]
        hc = heuristic_cov["per_dim_coverage"][j]
        print(f"  {JOINT_NAMES[j]:25s}  {cc:10.4f}  {hc:10.4f}")

    # --- Data-driven v_max (99th percentile of consecutive deltas) ---
    print(f"\n--- Data-driven v_max ({V_MAX_PERCENTILE}th percentile of consecutive deltas) ---")

    cal_deltas = compute_velocity_stats(cal_ep_actions)
    print(f"  Total consecutive deltas: {cal_deltas.shape[0]}")

    v_max_data = np.percentile(cal_deltas, V_MAX_PERCENTILE, axis=0)
    v_max_50 = np.percentile(cal_deltas, 50, axis=0)
    v_max_95 = np.percentile(cal_deltas, 95, axis=0)
    v_max_max = cal_deltas.max(axis=0)

    print(f"\n  {'Joint':25s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}  {'max':>8s}")
    for j in range(n_dims):
        print(f"  {JOINT_NAMES[j]:25s}  {v_max_50[j]:.5f}  {v_max_95[j]:.5f}  {v_max_data[j]:.5f}  {v_max_max[j]:.5f}")

    # Compare to hand-tuned values
    print(f"\n  Comparison to hand-tuned v_max:")
    for hand_v, label in [(0.05, "EXP-ACT-CL (0.05)"), (0.1, "standard (0.1)")]:
        pct_above = (cal_deltas > hand_v).mean(axis=0)
        print(f"  {label}:")
        for j in range(n_dims):
            print(f"    {JOINT_NAMES[j]:25s}: {pct_above[j]*100:.2f}% of deltas exceed {hand_v}")

    # Verify holdout velocity coverage
    holdout_deltas = compute_velocity_stats(holdout_ep_actions)
    holdout_vel_coverage = (holdout_deltas <= v_max_data).all(axis=1).mean()
    holdout_vel_per_dim = (holdout_deltas <= v_max_data).mean(axis=0)

    print(f"\n  Holdout velocity coverage with data-driven v_max (p{V_MAX_PERCENTILE}):")
    print(f"    Overall: {holdout_vel_coverage:.4f} ({holdout_vel_coverage*100:.2f}%)")
    for j in range(n_dims):
        print(f"    {JOINT_NAMES[j]:25s}: {holdout_vel_per_dim[j]:.4f}")

    elapsed = time.time() - t0

    # --- Build results ---
    results = {
        "experiment": "EXP-CONFORMAL",
        "description": "Conformal prediction bounds vs 4-sigma heuristic on ALOHA demo data",
        "dataset": DATASET_ID,
        "n_frames": len(ds),
        "n_episodes": len(set(ds["episode_index"])),
        "n_dims": n_dims,
        "joint_names": JOINT_NAMES,
        "parameters": {
            "calibration_split": CALIBRATION_SPLIT,
            "seed": SEED,
            "alpha": ALPHA,
            "n_sigma_heuristic": N_SIGMA,
            "v_max_percentile": V_MAX_PERCENTILE,
        },
        "split": {
            "n_cal_episodes": len(cal_eps),
            "n_holdout_episodes": len(holdout_eps),
            "cal_episodes": cal_eps,
            "holdout_episodes": holdout_eps,
            "n_cal_actions": int(cal_actions.shape[0]),
            "n_holdout_actions": int(holdout_actions.shape[0]),
        },
        "calibration_stats": {
            "mean": cal_mean.tolist(),
            "std": cal_std.tolist(),
        },
        "conformal": {
            "alpha": ALPHA,
            "q_hat": q_hat,
            "conformal_level": float((1 - ALPHA) * (1 + 1 / len(cal_scores))),
            "score_range": [float(cal_scores.min()), float(cal_scores.max())],
            "score_mean": float(cal_scores.mean()),
            "score_median": float(np.median(cal_scores)),
            "bounds_lo": conformal_lo.tolist(),
            "bounds_hi": conformal_hi.tolist(),
            "bounds_width": conformal_widths.tolist(),
            "holdout_coverage": conformal_cov,
            "coverage_guarantee_met": guarantee_met,
        },
        "heuristic_4sigma": {
            "n_sigma": N_SIGMA,
            "bounds_lo": heuristic_lo.tolist(),
            "bounds_hi": heuristic_hi.tolist(),
            "bounds_width": heuristic_widths.tolist(),
            "holdout_coverage": heuristic_cov,
        },
        "width_comparison": {
            "ratio_conformal_over_heuristic": width_ratio.tolist(),
            "mean_ratio": float(width_ratio.mean()),
            "conformal_tighter": bool(width_ratio.mean() < 1.0),
        },
        "data_driven_vmax": {
            "percentile": V_MAX_PERCENTILE,
            "v_max_per_dim": v_max_data.tolist(),
            "v_max_p50_per_dim": v_max_50.tolist(),
            "v_max_p95_per_dim": v_max_95.tolist(),
            "v_max_max_per_dim": v_max_max.tolist(),
            "n_cal_deltas": int(cal_deltas.shape[0]),
            "holdout_velocity_coverage": float(holdout_vel_coverage),
            "holdout_velocity_per_dim_coverage": holdout_vel_per_dim.tolist(),
        },
        "elapsed_seconds": round(elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp_conformal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Conformal (alpha={ALPHA}):")
    print(f"  Holdout coverage: {conformal_cov['overall_coverage']*100:.2f}% (target >= {(1-ALPHA)*100:.0f}%)")
    print(f"  Guarantee met: {'YES' if guarantee_met else 'NO'}")
    print(f"4-sigma heuristic:")
    print(f"  Holdout coverage: {heuristic_cov['overall_coverage']*100:.2f}%")
    print(f"Width: conformal is {width_ratio.mean():.2f}x the 4-sigma width on average")
    tightening = (1 - width_ratio.mean()) * 100
    if tightening > 0:
        print(f"  -> {tightening:.1f}% tighter bounds with statistical guarantee")
    else:
        print(f"  -> {-tightening:.1f}% wider (conformal needs wider for non-Gaussian tails)")
    print(f"Data-driven v_max (p{V_MAX_PERCENTILE}): {v_max_data.mean():.5f} mean across dims")
    print(f"  Holdout velocity coverage: {holdout_vel_coverage*100:.2f}%")


if __name__ == "__main__":
    main()
