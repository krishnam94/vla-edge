"""EXP-004: SmolVLA inference on LIBERO observations (offline).

No sim loop needed - just batch inference on saved observations.
Measures: violation rates, clipping magnitude, action smoothness, ProbeFlow step distribution.

Usage:
  python experiments/icra_ws_2026/exp4_libero_offline.py [--n-samples 100] [--device cpu]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main(n_samples: int = 100, device: str = "cpu"):
    import torch
    from datasets import load_dataset

    from vla_edge.models.smolvla import SmolVLAAdapter
    from vla_edge.optimize.probeflow import apply_probeflow
    from vla_edge.validate.contract import clear_violation_log, get_violation_log, safety_contract
    from vla_edge.validate.safety import SafetyConfig, validate_actions

    print(f"Loading LIBERO smol dataset...")
    ds = load_dataset("HuggingFaceVLA/smol-libero", split="train")
    print(f"Loaded {len(ds)} frames. Using {n_samples} samples.")

    # Sample evenly across the dataset
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    print(f"Loading SmolVLA on {device}...")
    adapter = SmolVLAAdapter(device=device, dtype="float32")

    # Collect raw actions (baseline, 10 steps)
    print("Running baseline inference (10 steps)...")
    baseline_actions = []
    baseline_times = []

    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        image = np.array(sample["observation.images.image"], dtype=np.uint8)
        state = np.array(sample["observation.state"], dtype=np.float32)

        t0 = time.perf_counter()
        action = adapter.predict(image, "complete the task", state[:6])
        elapsed = (time.perf_counter() - t0) * 1000
        baseline_times.append(elapsed)
        baseline_actions.append(action.copy())

        if (i + 1) % 10 == 0:
            print(f"  Baseline: {i+1}/{n_samples} ({elapsed:.0f}ms)")

    baseline_actions = np.array(baseline_actions)

    # Apply ProbeFlow and collect actions
    print("\nApplying ProbeFlow (epsilon=0.15)...")
    adapter.cleanup()
    adapter = SmolVLAAdapter(device=device, dtype="float32")
    adapter._ensure_loaded()
    stats = apply_probeflow(adapter._policy.model, epsilon=0.15, n_min=2, n_max=10)

    probeflow_actions = []
    probeflow_times = []

    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        image = np.array(sample["observation.images.image"], dtype=np.uint8)
        state = np.array(sample["observation.state"], dtype=np.float32)

        t0 = time.perf_counter()
        action = adapter.predict(image, "complete the task", state[:6])
        elapsed = (time.perf_counter() - t0) * 1000
        probeflow_times.append(elapsed)
        probeflow_actions.append(action.copy())

        if (i + 1) % 10 == 0:
            print(f"  ProbeFlow: {i+1}/{n_samples} ({elapsed:.0f}ms, steps={stats.avg_steps:.1f})")

    probeflow_actions = np.array(probeflow_actions)

    # Safety analysis
    print("\nRunning safety analysis...")
    action_dim = min(baseline_actions.shape[-1], 7)
    config = SafetyConfig(
        action_bounds=np.array([[-1, 1]] * action_dim, dtype=np.float32),
        max_velocity=np.full(action_dim, 0.1, dtype=np.float32),
    )

    baseline_safety = validate_actions(baseline_actions[:, :action_dim], config)
    probeflow_safety = validate_actions(probeflow_actions[:, :action_dim], config)

    # Action divergence
    min_len = min(len(baseline_actions), len(probeflow_actions))
    min_dim = min(baseline_actions.shape[-1], probeflow_actions.shape[-1])
    divergence_l1 = float(np.mean(np.abs(
        baseline_actions[:min_len, :min_dim] - probeflow_actions[:min_len, :min_dim]
    )))

    # Compile results
    results = {
        "experiment": "EXP-004: SmolVLA on LIBERO (offline)",
        "n_samples": n_samples,
        "device": device,
        "dataset": "HuggingFaceVLA/smol-libero",
        "baseline": {
            "avg_latency_ms": round(np.mean(baseline_times), 1),
            "cold_start_ms": round(baseline_times[0], 1),
            "violations": len(baseline_safety.violations),
            "violation_rate": round(baseline_safety.clipped_actions / max(baseline_safety.total_actions, 1), 4),
            "action_range": [round(float(baseline_actions.min()), 4), round(float(baseline_actions.max()), 4)],
        },
        "probeflow": {
            "avg_latency_ms": round(np.mean(probeflow_times), 1),
            "cold_start_ms": round(probeflow_times[0], 1),
            "avg_steps": round(stats.avg_steps, 1),
            "step_histogram": dict(sorted(stats.step_histogram.items())),
            "violations": len(probeflow_safety.violations),
            "violation_rate": round(probeflow_safety.clipped_actions / max(probeflow_safety.total_actions, 1), 4),
            "action_range": [round(float(probeflow_actions.min()), 4), round(float(probeflow_actions.max()), 4)],
        },
        "comparison": {
            "action_divergence_l1": round(divergence_l1, 4),
            "speedup": round(np.mean(baseline_times) / max(np.mean(probeflow_times), 0.1), 2),
            "baseline_violations_more": len(baseline_safety.violations) - len(probeflow_safety.violations),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    path = RESULTS_DIR / "exp4_libero_offline.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"Results saved to {path}")
    print(f"Baseline: {results['baseline']['avg_latency_ms']}ms avg, {results['baseline']['violations']} violations")
    print(f"ProbeFlow: {results['probeflow']['avg_latency_ms']}ms avg, {results['probeflow']['avg_steps']} avg steps")
    print(f"Divergence: L1={results['comparison']['action_divergence_l1']}")
    print(f"Speedup: {results['comparison']['speedup']}x")

    adapter.cleanup()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(n_samples=args.n_samples, device=args.device)
