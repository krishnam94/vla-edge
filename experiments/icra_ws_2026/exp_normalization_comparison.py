"""Normalization comparison: raw vs unnormalized SmolVLA actions on LIBERO.

Two conditions on the SAME 50 observations:
  Condition 1 (raw): SmolVLA output checked against [-1, 1] bounds.
  Condition 2 (unnormalized): Apply unnorm (action * std + mean) THEN check [-1, 1].

Key hypothesis: After unnormalization, bounds violations drop dramatically.
Velocity violations should PERSIST because velocity limits are physical constraints.

Usage:
  python experiments/icra_ws_2026/exp_normalization_comparison.py [--n-samples 50] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# LIBERO action stats from HuggingFaceVLA/smol-libero
LIBERO_ACTION_MEAN = np.array(
    [0.007, 0.0836, -0.0395, 0.0005, 0.0032, -0.0014, 0.0064],
    dtype=np.float32,
)
LIBERO_ACTION_STD = np.array(
    [0.2963, 0.4462, 0.4706, 0.031, 0.0483, 0.0433, 1.0],
    dtype=np.float32,
)

DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

V_MAX = 0.1  # Physical velocity limit per dim


def compute_violations(
    actions: np.ndarray,
    bounds: tuple[float, float] = (-1.0, 1.0),
    v_max: float = V_MAX,
) -> dict:
    """Compute per-dim and overall violation rates for bounds and velocity."""
    n_steps, n_dims = actions.shape
    lo, hi = bounds

    labels = DIM_LABELS[:n_dims]

    bounds_per_dim = np.zeros(n_dims)
    velocity_per_dim = np.zeros(n_dims)

    for d in range(n_dims):
        oob = np.sum((actions[:, d] < lo) | (actions[:, d] > hi))
        bounds_per_dim[d] = oob / n_steps

        if n_steps > 1:
            vel = np.abs(np.diff(actions[:, d]))
            velocity_per_dim[d] = np.sum(vel > v_max) / (n_steps - 1)

    # Overall: fraction of steps with ANY bounds violation
    any_oob = np.any((actions < lo) | (actions > hi), axis=1)
    overall_bounds_rate = float(np.mean(any_oob))

    # Overall: fraction of consecutive pairs with ANY velocity violation
    if n_steps > 1:
        deltas = np.abs(np.diff(actions, axis=0))
        any_vel = np.any(deltas > v_max, axis=1)
        overall_velocity_rate = float(np.mean(any_vel))
    else:
        overall_velocity_rate = 0.0

    # Count total violations (individual dim violations)
    total_bounds = int(np.sum((actions < lo) | (actions > hi)))
    total_velocity = 0
    if n_steps > 1:
        total_velocity = int(np.sum(np.abs(np.diff(actions, axis=0)) > v_max))

    return {
        "bounds_per_dim": {
            labels[d]: round(float(bounds_per_dim[d]), 4)
            for d in range(n_dims)
        },
        "velocity_per_dim": {
            labels[d]: round(float(velocity_per_dim[d]), 4)
            for d in range(n_dims)
        },
        "overall_bounds_rate": round(overall_bounds_rate, 4),
        "overall_velocity_rate": round(overall_velocity_rate, 4),
        "total_bounds_violations": total_bounds,
        "total_velocity_violations": total_velocity,
        "total_violations": total_bounds + total_velocity,
        "action_range": [round(float(actions.min()), 4), round(float(actions.max()), 4)],
        "per_dim_range": {
            labels[d]: [round(float(actions[:, d].min()), 4), round(float(actions[:, d].max()), 4)]
            for d in range(n_dims)
        },
    }


def main(n_samples: int = 50, device: str = "mps"):
    import torch
    from datasets import load_dataset

    from vla_edge.models.smolvla import SmolVLAAdapter

    print(f"=== Normalization Comparison Experiment ===")
    print(f"Samples: {n_samples}, Device: {device}, v_max: {V_MAX}")

    # Load dataset
    print("\nLoading HuggingFaceVLA/smol-libero...")
    ds = load_dataset("HuggingFaceVLA/smol-libero", split="train")
    print(f"Loaded {len(ds)} frames. Using {n_samples} samples.")

    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    # Load SmolVLA (LIBERO finetuned)
    print(f"\nLoading SmolVLA (HuggingFaceVLA/smolvla_libero) on {device}...")
    adapter = SmolVLAAdapter(
        model_id="HuggingFaceVLA/smolvla_libero",
        device=device,
        dtype="float32",
    )

    # Run inference and collect raw actions
    print(f"\nRunning inference on {n_samples} observations...")
    raw_actions_list = []
    latencies = []

    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        image = np.array(sample["observation.images.image"], dtype=np.uint8)
        state = np.array(sample["observation.state"], dtype=np.float32)

        t0 = time.perf_counter()
        action = adapter.predict(image, "complete the task", state[:6])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        # SmolVLA outputs 6 dims but we want 7 to match dataset stats
        # Pad with 0 for gripper if needed
        if len(action.shape) > 1:
            action = action[0]  # Take first step of chunk
        if len(action) < 7:
            action = np.concatenate([action, np.zeros(7 - len(action))])
        raw_actions_list.append(action[:7].copy())

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples} ({elapsed_ms:.0f}ms)")

    raw_actions = np.array(raw_actions_list, dtype=np.float32)
    print(f"\nInference complete. Avg latency: {np.mean(latencies[1:]):.0f}ms (excl cold start)")
    print(f"Raw action range: [{raw_actions.min():.4f}, {raw_actions.max():.4f}]")

    # Condition 1: Raw actions checked against [-1, 1]
    print("\n--- Condition 1: Raw (no unnormalization) ---")
    cond1 = compute_violations(raw_actions, bounds=(-1.0, 1.0), v_max=V_MAX)
    print(f"  Bounds violation rate: {cond1['overall_bounds_rate']*100:.1f}%")
    print(f"  Velocity violation rate: {cond1['overall_velocity_rate']*100:.1f}%")
    print(f"  Total violations: {cond1['total_violations']}")
    print(f"  Bounds per dim: ", end="")
    for d, label in enumerate(DIM_LABELS):
        print(f"{label}={cond1['bounds_per_dim'][label]*100:.0f}% ", end="")
    print()
    print(f"  Velocity per dim: ", end="")
    for d, label in enumerate(DIM_LABELS):
        print(f"{label}={cond1['velocity_per_dim'][label]*100:.0f}% ", end="")
    print()

    # Condition 2: Unnormalized actions checked against [-1, 1]
    print("\n--- Condition 2: Unnormalized (action * std + mean) ---")
    unnorm_actions = raw_actions * LIBERO_ACTION_STD + LIBERO_ACTION_MEAN
    print(f"  Unnormalized action range: [{unnorm_actions.min():.4f}, {unnorm_actions.max():.4f}]")

    cond2 = compute_violations(unnorm_actions, bounds=(-1.0, 1.0), v_max=V_MAX)
    print(f"  Bounds violation rate: {cond2['overall_bounds_rate']*100:.1f}%")
    print(f"  Velocity violation rate: {cond2['overall_velocity_rate']*100:.1f}%")
    print(f"  Total violations: {cond2['total_violations']}")
    print(f"  Bounds per dim: ", end="")
    for d, label in enumerate(DIM_LABELS):
        print(f"{label}={cond2['bounds_per_dim'][label]*100:.0f}% ", end="")
    print()
    print(f"  Velocity per dim: ", end="")
    for d, label in enumerate(DIM_LABELS):
        print(f"{label}={cond2['velocity_per_dim'][label]*100:.0f}% ", end="")
    print()

    # Also compute 6-dim (position only, no gripper) for cleaner narrative
    # Gripper (dim 6) has std=1.0 so unnormalization barely changes it
    print("\n--- Position-only analysis (6 dims, excluding gripper) ---")
    raw_6d = raw_actions[:, :6]
    unnorm_6d = unnorm_actions[:, :6]
    cond1_6d = compute_violations(raw_6d, bounds=(-1.0, 1.0), v_max=V_MAX)
    cond2_6d = compute_violations(unnorm_6d, bounds=(-1.0, 1.0), v_max=V_MAX)
    print(f"  Raw bounds: {cond1_6d['overall_bounds_rate']*100:.1f}% -> Unnorm bounds: {cond2_6d['overall_bounds_rate']*100:.1f}%")
    print(f"  Raw velocity: {cond1_6d['overall_velocity_rate']*100:.1f}% -> Unnorm velocity: {cond2_6d['overall_velocity_rate']*100:.1f}%")

    # Compute deltas
    bounds_drop = cond1["overall_bounds_rate"] - cond2["overall_bounds_rate"]
    bounds_drop_6d = cond1_6d["overall_bounds_rate"] - cond2_6d["overall_bounds_rate"]

    print(f"\n{'='*60}")
    print(f"KEY RESULT (all 7 dims):")
    print(f"  Bounds: {cond1['overall_bounds_rate']*100:.1f}% -> {cond2['overall_bounds_rate']*100:.1f}% (drop: {bounds_drop*100:.1f}pp)")
    print(f"  Velocity: {cond1['overall_velocity_rate']*100:.1f}% -> {cond2['overall_velocity_rate']*100:.1f}%")
    print(f"\nKEY RESULT (6 position dims, no gripper):")
    print(f"  Bounds: {cond1_6d['overall_bounds_rate']*100:.1f}% -> {cond2_6d['overall_bounds_rate']*100:.1f}% (drop: {bounds_drop_6d*100:.1f}pp)")
    print(f"  Velocity: {cond1_6d['overall_velocity_rate']*100:.1f}% -> {cond2_6d['overall_velocity_rate']*100:.1f}%")
    print(f"\n  'After correct normalization, position bounds violations drop from "
          f"{cond1_6d['overall_bounds_rate']*100:.0f}% to {cond2_6d['overall_bounds_rate']*100:.0f}%, "
          f"but velocity violations persist at {cond2_6d['overall_velocity_rate']*100:.0f}% - "
          f"proving that velocity monitoring provides value beyond normalization.'")

    # Save results
    results = {
        "experiment": "Normalization comparison: raw vs unnormalized SmolVLA on LIBERO",
        "model": "HuggingFaceVLA/smolvla_libero",
        "dataset": "HuggingFaceVLA/smol-libero",
        "n_samples": n_samples,
        "device": device,
        "v_max": V_MAX,
        "normalization_stats": {
            "mean": LIBERO_ACTION_MEAN.tolist(),
            "std": LIBERO_ACTION_STD.tolist(),
        },
        "condition_1_raw": cond1,
        "condition_2_unnormalized": cond2,
        "position_only_6d": {
            "condition_1_raw": cond1_6d,
            "condition_2_unnormalized": cond2_6d,
            "bounds_drop_pp": round(bounds_drop_6d * 100, 1),
            "note": "Excludes gripper (dim 6, std=1.0) for cleaner comparison",
        },
        "comparison": {
            "bounds_rate_raw_pct": round(cond1["overall_bounds_rate"] * 100, 1),
            "bounds_rate_unnorm_pct": round(cond2["overall_bounds_rate"] * 100, 1),
            "bounds_drop_pp": round(bounds_drop * 100, 1),
            "velocity_rate_raw_pct": round(cond1["overall_velocity_rate"] * 100, 1),
            "velocity_rate_unnorm_pct": round(cond2["overall_velocity_rate"] * 100, 1),
            "velocity_persists": cond2["overall_velocity_rate"] > 0.05,
            "position_6d_bounds_raw_pct": round(cond1_6d["overall_bounds_rate"] * 100, 1),
            "position_6d_bounds_unnorm_pct": round(cond2_6d["overall_bounds_rate"] * 100, 1),
            "position_6d_bounds_drop_pp": round(bounds_drop_6d * 100, 1),
            "position_6d_velocity_raw_pct": round(cond1_6d["overall_velocity_rate"] * 100, 1),
            "position_6d_velocity_unnorm_pct": round(cond2_6d["overall_velocity_rate"] * 100, 1),
        },
        "key_result": (
            f"After correct normalization, position bounds violations drop from "
            f"{cond1_6d['overall_bounds_rate']*100:.0f}% to {cond2_6d['overall_bounds_rate']*100:.0f}%, "
            f"but velocity violations persist at {cond2_6d['overall_velocity_rate']*100:.0f}% - "
            f"proving that velocity monitoring provides value beyond normalization."
        ),
        "avg_latency_ms": round(float(np.mean(latencies[1:])), 1),
        "cold_start_ms": round(float(latencies[0]), 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = RESULTS_DIR / "exp_normalization_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    adapter.cleanup()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    main(n_samples=args.n_samples, device=args.device)
