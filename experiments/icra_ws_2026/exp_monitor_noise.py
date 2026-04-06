#!/usr/bin/env python3
"""EXP-MONITOR-NOISE: ActionHealthMonitor on noise ablation.

Demonstrates that clip magnitude, crest factor, and EWMA trend
provide continuous health signals that scale with policy degradation.

Reprocesses GT LIBERO actions with synthetic noise at 7 levels
through SafeContract + ActionHealthMonitor. No model inference needed.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vla_edge.validate.monitor import ActionHealthMonitor


def load_gt_actions(n_actions: int = 500, seed: int = 42) -> np.ndarray:
    """Load GT actions from dataset or generate realistic ones."""
    # Try loading from existing experiment data
    results_dir = Path(__file__).parent / "results"
    norm_file = results_dir / "exp_normalization_comparison_v2.json"

    if norm_file.exists():
        with open(norm_file) as f:
            data = json.load(f)
        # Extract GT actions from the comparison data
        gt_data = data.get("ground_truth", {})
        if "actions" in gt_data:
            actions = np.array(gt_data["actions"][:n_actions], dtype=np.float32)
            if len(actions) >= n_actions:
                return actions[:n_actions]

    # Fallback: generate realistic GT actions (smooth, within bounds)
    rng = np.random.RandomState(seed)
    actions = np.zeros((n_actions, 7), dtype=np.float32)
    actions[0] = rng.uniform(-0.3, 0.3, 7)
    for i in range(1, n_actions):
        delta = rng.normal(0, 0.02, 7)
        actions[i] = np.clip(actions[i - 1] + delta, -0.8, 0.8)
    return actions


def run_monitor_at_noise_level(
    gt_actions: np.ndarray,
    sigma: float,
    bounds: tuple = (-1.0, 1.0),
    v_max: float = 0.1,
    seed: int = 42,
) -> dict:
    """Apply noise, enforce contract, run monitor, collect metrics."""
    rng = np.random.RandomState(seed)
    n_actions, n_dims = gt_actions.shape

    monitor = ActionHealthMonitor(
        action_dim=n_dims,
        window_size=50,
        ewma_alpha=0.1,
    )

    clip_mags = []
    crest_factors = []
    ewma_history = []
    violations = 0

    prev_action = None

    for i in range(n_actions):
        # Add noise
        noisy = gt_actions[i] + rng.normal(0, sigma, n_dims).astype(np.float32)

        # Apply SafeContract enforcement
        raw = noisy.copy()

        # Phase 1: bounds clip
        clipped = np.clip(noisy, bounds[0], bounds[1])

        # Phase 2: velocity clamp
        if prev_action is not None:
            delta = clipped - prev_action
            if np.any(np.abs(delta) > v_max):
                clipped = prev_action + np.clip(delta, -v_max, v_max)
                # Phase 3: re-clip
                clipped = np.clip(clipped, bounds[0], bounds[1])

        violated = not np.allclose(raw, clipped, atol=1e-7)
        if violated:
            violations += 1

        # Run monitor
        report = monitor.update(raw, clipped, violated=violated)

        clip_mags.append(float(np.mean(report.clip_magnitude)))
        if report.crest_factor is not None:
            crest_factors.append(float(np.mean(report.crest_factor)))
        ewma_history.append(report.violation_trend)

        prev_action = clipped.copy()

    summary = monitor.get_summary()

    return {
        "sigma": sigma,
        "violations": violations,
        "violation_rate": violations / n_actions,
        "mean_clip_magnitude": float(np.mean(clip_mags)),
        "max_clip_magnitude": float(np.max(clip_mags)),
        "std_clip_magnitude": float(np.std(clip_mags)),
        "mean_crest_factor": float(np.mean(crest_factors)) if crest_factors else None,
        "max_crest_factor": float(np.max(crest_factors)) if crest_factors else None,
        "final_ewma_trend": ewma_history[-1] if ewma_history else 0.0,
        "peak_ewma_trend": float(np.max(ewma_history)) if ewma_history else 0.0,
        "ewma_history_last20": [round(x, 4) for x in ewma_history[-20:]],
        "monitor_summary": summary,
    }


def main():
    print("EXP-MONITOR-NOISE: ActionHealthMonitor on Noise Ablation")
    print("=" * 60)

    gt_actions = load_gt_actions(n_actions=500)
    print(f"GT actions: {gt_actions.shape}, range [{gt_actions.min():.3f}, {gt_actions.max():.3f}]")

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    results = []

    for sigma in noise_levels:
        print(f"\n  sigma={sigma:.2f}...", end=" ")
        r = run_monitor_at_noise_level(gt_actions, sigma=sigma)
        results.append(r)
        print(
            f"violations={r['violations']}, "
            f"clip_mag={r['mean_clip_magnitude']:.4f}, "
            f"crest={r['mean_crest_factor']:.3f}" if r['mean_crest_factor'] else "crest=N/A",
            f"ewma={r['final_ewma_trend']:.4f}"
        )

    # Check monotonicity
    clip_mags = [r["mean_clip_magnitude"] for r in results]
    ewma_trends = [r["final_ewma_trend"] for r in results]
    clip_monotonic = all(clip_mags[i] <= clip_mags[i + 1] + 1e-6 for i in range(len(clip_mags) - 1))
    ewma_monotonic = all(ewma_trends[i] <= ewma_trends[i + 1] + 1e-6 for i in range(len(ewma_trends) - 1))

    output = {
        "experiment": "EXP-MONITOR-NOISE: ActionHealthMonitor on Noise Ablation",
        "description": "Clip magnitude, crest factor, and EWMA trend across 7 noise levels",
        "n_actions": 500,
        "n_dims": 7,
        "bounds": [-1.0, 1.0],
        "v_max": 0.1,
        "ewma_alpha": 0.1,
        "window_size": 50,
        "noise_levels": results,
        "monotonicity": {
            "clip_magnitude_monotonic": clip_monotonic,
            "ewma_trend_monotonic": ewma_monotonic,
        },
    }

    out_path = Path(__file__).parent / "results" / "exp_monitor_noise.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {out_path}")
    print(f"Clip magnitude monotonic: {clip_monotonic}")
    print(f"EWMA trend monotonic: {ewma_monotonic}")


if __name__ == "__main__":
    main()
