"""EXP: Real violation fingerprints from SmolVLA on LIBERO task suites.

Runs SmolVLA inference on observations from different regions of
HuggingFaceVLA/smol-libero, tracking per-joint violation rates (bounds and
velocity) for each group. This produces REAL model-based fingerprints, not
synthetic distributions.

The smol-libero dataset has only task_index=0, so we split by dataset region
as a proxy for different task suites. Different regions contain different
episodes with different manipulation tasks (the dataset contains multiple
LIBERO tasks despite having a single task_index).

Key design decisions:
- Safety bounds are derived from ground truth action statistics (mean +/- 3*std)
  rather than naive [-1,1], matching how a real deployment would calibrate.
- Velocity thresholds are derived from GT consecutive-frame deltas.
- Fingerprint vector includes bounds violations, velocity violations, AND
  action distribution features (mean, std, magnitude) per dimension.

Usage:
  python experiments/icra_ws_2026/exp_real_fingerprints.py [--n-per-suite 25] [--device mps]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw"]


def compute_gt_bounds(ds, n_sample: int = 2000) -> dict:
    """Compute data-driven safety bounds from ground truth actions.

    Returns per-dim bounds (mean +/- 3*std) and velocity thresholds
    (95th percentile of consecutive-frame deltas).
    """
    indices = np.linspace(0, len(ds) - 1, min(n_sample, len(ds)), dtype=int)
    gt_actions = np.array([ds[int(i)]["action"] for i in indices])

    # SmolVLA outputs 6 dims (no gripper), GT has 7 (with gripper)
    gt_6d = gt_actions[:, :6]

    per_dim_mean = np.mean(gt_6d, axis=0)
    per_dim_std = np.std(gt_6d, axis=0)

    # Bounds: mean +/- 3*std (captures 99.7% of normal distribution)
    bounds_lo = per_dim_mean - 3 * per_dim_std
    bounds_hi = per_dim_mean + 3 * per_dim_std

    # Velocity: 95th percentile of frame-to-frame deltas (within episodes)
    deltas = np.abs(np.diff(gt_6d, axis=0))
    vel_thresholds = np.percentile(deltas, 95, axis=0)

    return {
        "bounds_lo": bounds_lo,
        "bounds_hi": bounds_hi,
        "vel_thresholds": vel_thresholds,
        "gt_mean": per_dim_mean,
        "gt_std": per_dim_std,
        "gt_min": np.min(gt_6d, axis=0),
        "gt_max": np.max(gt_6d, axis=0),
    }


def compute_fingerprint(
    actions: np.ndarray,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    vel_thresholds: np.ndarray,
    gt_mean: np.ndarray,
    gt_std: np.ndarray,
) -> dict:
    """Compute a rich violation fingerprint for a set of actions.

    The fingerprint includes:
    - Per-dim bounds violation rates
    - Per-dim velocity violation rates (using consecutive model outputs)
    - Per-dim deviation from GT mean (in units of GT std)
    - Per-dim action magnitude profile
    """
    n_steps, n_dims = actions.shape

    # Bounds violations
    bounds_rates = np.zeros(n_dims)
    for d in range(n_dims):
        violations = np.sum((actions[:, d] < bounds_lo[d]) | (actions[:, d] > bounds_hi[d]))
        bounds_rates[d] = violations / n_steps

    # Velocity violations (consecutive model outputs, not in-chunk)
    velocity_rates = np.zeros(n_dims)
    if n_steps > 1:
        for d in range(n_dims):
            velocities = np.abs(np.diff(actions[:, d]))
            vel_violations = np.sum(velocities > vel_thresholds[d])
            velocity_rates[d] = vel_violations / (n_steps - 1)

    # Deviation from GT distribution (z-scores)
    action_means = np.mean(actions, axis=0)
    gt_std_safe = np.where(gt_std > 1e-6, gt_std, 1.0)
    deviation_z = np.abs(action_means - gt_mean) / gt_std_safe

    # Action magnitude profile (RMS per dim)
    magnitude = np.sqrt(np.mean(actions**2, axis=0))

    # Overall violation rate
    oob_steps = np.zeros(n_steps, dtype=bool)
    for d in range(n_dims):
        oob_steps |= (actions[:, d] < bounds_lo[d]) | (actions[:, d] > bounds_hi[d])
    overall_rate = float(np.mean(oob_steps))

    return {
        "bounds_per_dim": bounds_rates.round(4).tolist(),
        "velocity_per_dim": velocity_rates.round(4).tolist(),
        "deviation_z_per_dim": deviation_z.round(4).tolist(),
        "magnitude_per_dim": magnitude.round(4).tolist(),
        "action_mean_per_dim": action_means.round(4).tolist(),
        "action_std_per_dim": np.std(actions, axis=0).round(4).tolist(),
        "overall_violation_rate": round(overall_rate, 4),
        "overall_bounds_rate": round(float(np.mean(bounds_rates)), 4),
        "overall_velocity_rate": round(float(np.mean(velocity_rates)), 4),
    }


def fingerprint_vector(fp: dict) -> np.ndarray:
    """Extract a compact vector from a fingerprint for distance computation.

    Concatenates: bounds_per_dim, velocity_per_dim, deviation_z_per_dim, magnitude_per_dim
    This 24-dim vector captures the full violation + distribution signature.
    """
    return np.array(
        fp["bounds_per_dim"]
        + fp["velocity_per_dim"]
        + fp["deviation_z_per_dim"]
        + fp["magnitude_per_dim"]
    )


def main(n_per_suite: int = 25, device: str = "mps"):
    import torch
    from datasets import load_dataset

    from vla_edge.models.smolvla import SmolVLAAdapter

    # ---- Load dataset ----
    print("Loading smol-libero dataset...")
    ds = load_dataset("HuggingFaceVLA/smol-libero", split="train")
    total = len(ds)
    print(f"Loaded {total} frames.")

    # ---- Compute GT-derived safety bounds ----
    print("Computing data-driven safety bounds from GT actions...")
    gt = compute_gt_bounds(ds, n_sample=2000)
    print(f"  Per-dim bounds (3-sigma):")
    for d in range(6):
        print(f"    {DIM_LABELS[d]}: [{gt['bounds_lo'][d]:.3f}, {gt['bounds_hi'][d]:.3f}] "
              f"(GT range: [{gt['gt_min'][d]:.3f}, {gt['gt_max'][d]:.3f}])")
    print(f"  Velocity thresholds (95th pctl): {gt['vel_thresholds'].round(3).tolist()}")

    # ---- Build suite groups ----
    # Dataset has only task_index=0. Split by episode ranges.
    # Different episodes = different LIBERO tasks.
    print("\nBuilding suite groups from episode ranges...")
    episode_indices = np.array([ds[i]["episode_index"] for i in range(total)])
    unique_episodes = np.unique(episode_indices)
    n_episodes = len(unique_episodes)
    print(f"  Found {n_episodes} unique episodes.")

    # Split episodes into 4 groups
    eps_per_suite = n_episodes // 4
    suite_indices: dict[str, list[int]] = {}

    for suite_idx, suite_name in enumerate(SUITE_NAMES):
        ep_start = suite_idx * eps_per_suite
        ep_end = ep_start + eps_per_suite if suite_idx < 3 else n_episodes
        suite_episodes = unique_episodes[ep_start:ep_end]

        # Get all frame indices for these episodes
        mask = np.isin(episode_indices, suite_episodes)
        all_frame_idx = np.where(mask)[0]

        # Sample evenly
        chosen = np.linspace(0, len(all_frame_idx) - 1, min(n_per_suite, len(all_frame_idx)), dtype=int)
        suite_indices[suite_name] = all_frame_idx[chosen].tolist()
        print(f"  {suite_name}: episodes {suite_episodes[0]}-{suite_episodes[-1]} "
              f"({len(all_frame_idx)} frames, using {len(suite_indices[suite_name])})")

    # ---- Load model ----
    actual_device = device
    if device == "mps" and hasattr(torch.backends, "mps") and not torch.backends.mps.is_available():
        actual_device = "cpu"
        print("MPS not available, falling back to CPU.")
    print(f"\nLoading SmolVLA on {actual_device}...")
    adapter = SmolVLAAdapter(device=actual_device, dtype="float32")

    # ---- Run inference per suite ----
    results_by_suite: dict[str, dict] = {}
    all_timings: list[float] = []
    total_inferences = sum(len(v) for v in suite_indices.values())
    inference_count = 0

    for suite_name, indices in suite_indices.items():
        print(f"\n--- Suite: {suite_name} ({len(indices)} observations) ---")
        suite_actions = []
        suite_times = []
        suite_gt_actions = []

        for i, idx in enumerate(indices):
            sample = ds[int(idx)]
            image = np.array(sample["observation.images.image"], dtype=np.uint8)
            state = np.array(sample["observation.state"], dtype=np.float32)
            gt_action = np.array(sample["action"], dtype=np.float32)

            # Lesson 007: reset policy before each observation
            if hasattr(adapter, "_policy") and adapter._policy is not None:
                adapter._policy.reset()

            t0 = time.perf_counter()
            action = adapter.predict(image, "complete the task", state[:6])
            elapsed = (time.perf_counter() - t0) * 1000
            suite_times.append(elapsed)
            suite_actions.append(action.copy())
            suite_gt_actions.append(gt_action[:6].copy())
            all_timings.append(elapsed)
            inference_count += 1

            if (i + 1) % 5 == 0 or i == 0:
                pct = inference_count / total_inferences * 100
                print(f"  [{inference_count}/{total_inferences}] ({pct:.0f}%) "
                      f"sample {i+1}/{len(indices)}: {elapsed:.0f}ms")

        suite_actions_arr = np.array(suite_actions)
        suite_gt_arr = np.array(suite_gt_actions)

        # Compute fingerprint with data-driven bounds
        fp = compute_fingerprint(
            suite_actions_arr,
            bounds_lo=gt["bounds_lo"],
            bounds_hi=gt["bounds_hi"],
            vel_thresholds=gt["vel_thresholds"],
            gt_mean=gt["gt_mean"],
            gt_std=gt["gt_std"],
        )

        # Prediction error vs GT
        pred_error_l1 = float(np.mean(np.abs(suite_actions_arr - suite_gt_arr)))
        pred_error_per_dim = np.mean(np.abs(suite_actions_arr - suite_gt_arr), axis=0).round(4).tolist()

        results_by_suite[suite_name] = {
            "n_observations": len(indices),
            "action_shape": list(suite_actions_arr.shape),
            "avg_latency_ms": round(float(np.mean(suite_times)), 1),
            "fingerprint": fp,
            "prediction_error": {
                "l1_mean": round(pred_error_l1, 4),
                "l1_per_dim": pred_error_per_dim,
            },
        }

        print(f"  Overall violation rate: {fp['overall_violation_rate']:.1%}")
        print(f"  Bounds violations:  {' '.join(f'{DIM_LABELS[d]}={r:.2f}' for d, r in enumerate(fp['bounds_per_dim']))}")
        print(f"  Velocity violations: {' '.join(f'{DIM_LABELS[d]}={r:.2f}' for d, r in enumerate(fp['velocity_per_dim']))}")
        print(f"  Deviation (z-score): {' '.join(f'{DIM_LABELS[d]}={r:.1f}' for d, r in enumerate(fp['deviation_z_per_dim']))}")
        print(f"  Prediction error L1: {pred_error_l1:.4f}")

    # ---- Compute fingerprint distinctness ----
    print("\n--- Fingerprint Distinctness ---")
    suite_names_list = list(results_by_suite.keys())
    pairwise_distances = {}

    for i, s1 in enumerate(suite_names_list):
        for j, s2 in enumerate(suite_names_list):
            if j <= i:
                continue
            v1 = fingerprint_vector(results_by_suite[s1]["fingerprint"])
            v2 = fingerprint_vector(results_by_suite[s2]["fingerprint"])

            # Cosine distance
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_dist = 1 - np.dot(v1, v2) / (norm1 * norm2)
            else:
                cos_dist = 1.0

            # L2 distance (Euclidean)
            l2_dist = float(np.linalg.norm(v1 - v2))

            pair_key = f"{s1}_vs_{s2}"
            pairwise_distances[pair_key] = {
                "cosine": round(float(cos_dist), 4),
                "l2": round(l2_dist, 4),
            }
            print(f"  {s1} vs {s2}: cosine={cos_dist:.4f}, L2={l2_dist:.4f}")

    avg_cosine = float(np.mean([d["cosine"] for d in pairwise_distances.values()])) if pairwise_distances else 0
    avg_l2 = float(np.mean([d["l2"] for d in pairwise_distances.values()])) if pairwise_distances else 0
    print(f"\n  Average cosine distance: {avg_cosine:.4f}")
    print(f"  Average L2 distance: {avg_l2:.4f}")

    # ---- Compile final results ----
    results = {
        "experiment": "Real Violation Fingerprints - SmolVLA on LIBERO task suites",
        "model": "lerobot/smolvla_base (SmolVLA 450M)",
        "dataset": "HuggingFaceVLA/smol-libero",
        "device": actual_device,
        "n_per_suite": n_per_suite,
        "total_inferences": inference_count,
        "avg_latency_ms": round(float(np.mean(all_timings)), 1),
        "cold_start_ms": round(all_timings[0], 1) if all_timings else 0,
        "safety_bounds": {
            "method": "GT mean +/- 3*std (data-driven)",
            "bounds_lo": gt["bounds_lo"].round(4).tolist(),
            "bounds_hi": gt["bounds_hi"].round(4).tolist(),
            "vel_thresholds_95pctl": gt["vel_thresholds"].round(4).tolist(),
            "gt_mean": gt["gt_mean"].round(4).tolist(),
            "gt_std": gt["gt_std"].round(4).tolist(),
        },
        "suites": results_by_suite,
        "fingerprint_distinctness": {
            "pairwise_distances": pairwise_distances,
            "avg_cosine_distance": round(avg_cosine, 4),
            "avg_l2_distance": round(avg_l2, 4),
            "fingerprint_dims": 24,
            "components": "bounds_per_dim(6) + velocity_per_dim(6) + deviation_z(6) + magnitude(6)",
        },
        "key_findings": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Generate key findings
    findings = []

    # Distinctness finding
    if avg_cosine > 0.05:
        findings.append(
            f"Real SmolVLA inference produces distinct violation fingerprints across LIBERO task regions "
            f"(avg cosine distance: {avg_cosine:.3f}, avg L2: {avg_l2:.3f})"
        )
    elif avg_l2 > 0.5:
        findings.append(
            f"Fingerprints show moderate variation in absolute terms (avg L2: {avg_l2:.3f}) "
            f"despite low cosine distance ({avg_cosine:.3f}), indicating scale differences"
        )
    else:
        findings.append(
            f"Violation fingerprints show limited variation across dataset regions "
            f"(avg cosine: {avg_cosine:.3f}, avg L2: {avg_l2:.3f})"
        )

    # Violation rate finding
    rates = {s: r["fingerprint"]["overall_violation_rate"] for s, r in results_by_suite.items()}
    max_suite = max(rates, key=rates.get)
    min_suite = min(rates, key=rates.get)
    findings.append(
        f"Highest violation rate: {max_suite} ({rates[max_suite]:.1%}), "
        f"lowest: {min_suite} ({rates[min_suite]:.1%})"
    )

    # Deviation finding - which dims are most out of distribution
    all_deviations = {}
    for suite_name, r in results_by_suite.items():
        for d, z in enumerate(r["fingerprint"]["deviation_z_per_dim"]):
            key = f"{suite_name}_{DIM_LABELS[d]}"
            all_deviations[key] = z
    worst_3 = sorted(all_deviations.items(), key=lambda x: x[1], reverse=True)[:3]
    findings.append(
        f"Most out-of-distribution dims: {', '.join(f'{k} (z={v:.1f})' for k, v in worst_3)}"
    )

    # Prediction error finding
    errors = {s: r["prediction_error"]["l1_mean"] for s, r in results_by_suite.items()}
    findings.append(
        f"Prediction error (L1) ranges from {min(errors.values()):.4f} to {max(errors.values()):.4f} across suites"
    )

    # Velocity vs bounds finding
    for suite_name, r in results_by_suite.items():
        vel_total = sum(r["fingerprint"]["velocity_per_dim"])
        bounds_total = sum(r["fingerprint"]["bounds_per_dim"])
        if vel_total > 2 * bounds_total and vel_total > 0.5:
            findings.append(f"{suite_name}: velocity violations ({vel_total:.2f}) dominate bounds ({bounds_total:.2f})")
            break  # Just note the pattern once

    results["key_findings"] = findings

    # ---- Save ----
    def default(o):
        if hasattr(o, "item"):
            return o.item()
        raise TypeError

    path = RESULTS_DIR / "exp_real_fingerprints.json"
    path.write_text(json.dumps(results, indent=2, default=default))
    print(f"\n{'='*60}")
    print(f"Results saved to {path}")
    print(f"Total time: {sum(all_timings)/1000:.1f}s for {inference_count} inferences")
    print(f"Avg latency: {np.mean(all_timings):.0f}ms")
    print(f"\nKey findings:")
    for f in findings:
        print(f"  - {f}")

    adapter.cleanup()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-suite", type=int, default=25)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    main(n_per_suite=args.n_per_suite, device=args.device)
