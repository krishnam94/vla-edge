"""Experiments for ICRA 2026 VLA Pipelines Workshop Paper.

"Safety Contracts and Adaptive Denoising for Edge-Ready VLA Inference"

Run all experiments with: python experiments/icra_ws_2026/run_experiments.py
Results saved to: experiments/icra_ws_2026/results/
"""

import json
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def experiment_1_safety_contract_overhead():
    """Measure @safety_contract decorator overhead.

    Key claim: safety contracts add negligible overhead (<0.1ms per action).
    """
    from vla_edge.validate.contract import safety_contract

    @safety_contract(
        action_range=[-1.0, 1.0],
        joint_velocity_max=0.1,
        workspace_bounds=[[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.8]],
        on_violation="clip",
    )
    def mock_predict(self, image, instruction, state=None):
        return np.random.randn(7).astype(np.float32) * 0.5

    class MockModel:
        predict = mock_predict

    model = MockModel()
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Warmup
    for _ in range(100):
        model.predict(image, "test")

    # Timed
    times = []
    for _ in range(10000):
        t0 = time.perf_counter()
        model.predict(image, "test")
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    results = {
        "experiment": "safety_contract_overhead",
        "iterations": 10000,
        "avg_us": round(np.mean(times), 2),
        "p50_us": round(np.percentile(times, 50), 2),
        "p99_us": round(np.percentile(times, 99), 2),
        "stddev_us": round(np.std(times), 2),
    }

    path = RESULTS_DIR / "exp1_overhead.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Exp 1: Safety contract overhead = {results['avg_us']:.1f} us avg ({results['p99_us']:.1f} us p99)")
    return results


def experiment_2_violation_detection():
    """Measure violation detection rate across varying OOB percentages.

    Sweep: 0%, 5%, 10%, 20%, 50%, 100% of actions out of bounds.
    For each: measure violations detected, clipping magnitude, action smoothness.
    """
    from vla_edge.validate.contract import clear_violation_log, get_violation_log, safety_contract

    results = []

    for oob_pct in [0, 5, 10, 20, 50, 100]:

        @safety_contract(action_range=[-1.0, 1.0], joint_velocity_max=0.1, on_violation="clip")
        def predict_with_oob(self, image, instruction, state=None):
            action = np.random.randn(7).astype(np.float32) * 0.3
            # Inject OOB actions
            if np.random.random() * 100 < oob_pct:
                action = action * 5.0  # Push out of bounds
            return action

        class Model:
            predict = predict_with_oob

        model = Model()
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        clear_violation_log()

        actions_before = []
        actions_after = []

        for _ in range(1000):
            raw = np.random.randn(7).astype(np.float32) * 0.3
            if np.random.random() * 100 < oob_pct:
                raw = raw * 5.0
            actions_before.append(raw.copy())

            clipped = model.predict(image, "test")
            actions_after.append(clipped.copy())

        violations = get_violation_log()
        actions_before = np.array(actions_before)
        actions_after = np.array(actions_after)

        # Compute smoothness (jerk = diff of diff of diff)
        if len(actions_after) > 3:
            vel = np.diff(actions_after, axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            smoothness = float(np.mean(np.abs(jerk)))
        else:
            smoothness = 0.0

        clip_magnitude = float(np.mean(np.abs(actions_before - actions_after)))

        results.append({
            "oob_percent": oob_pct,
            "violations_detected": len(violations),
            "violation_rate": len(violations) / 1000,
            "avg_clip_magnitude": round(clip_magnitude, 4),
            "action_smoothness_jerk": round(smoothness, 4),
        })

    path = RESULTS_DIR / "exp2_violations.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Exp 2: Violation detection across {len(results)} OOB levels")
    for r in results:
        print(f"  {r['oob_percent']}% OOB: {r['violations_detected']} violations, clip={r['avg_clip_magnitude']:.4f}")
    return results


def experiment_3_probeflow_epsilon_sensitivity():
    """Measure ProbeFlow step allocation across epsilon values.

    This uses synthetic actions (no real model) to show the allocation behavior.
    Real SmolVLA experiment runs separately (slow).
    """
    results = []

    for epsilon in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        # Simulate ProbeFlow allocation with varying cosine similarities
        for similarity in [0.2, 0.5, 0.7, 0.9, 0.95, 0.99]:
            n_min, n_max, delta_n = 2, 10, 2
            n_steps = int(min(n_max, max(n_min, n_min + ((1.0 - similarity) / epsilon) * delta_n)))

            results.append({
                "epsilon": epsilon,
                "cosine_similarity": similarity,
                "allocated_steps": n_steps,
            })

    path = RESULTS_DIR / "exp3_probeflow_sensitivity.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Exp 3: ProbeFlow sensitivity - {len(results)} configurations")
    return results


def experiment_4_composition_interference():
    """Demonstrate contract composition interference.

    Shows that workspace + velocity contracts can create infeasible regions,
    and measure how often this happens with random action sequences.
    """
    from vla_edge.validate.safety import SafetyConfig, validate_actions

    configs = [
        ("loose", SafetyConfig(
            action_bounds=np.array([[-2, 2]] * 7, dtype=np.float32),
            max_velocity=np.full(7, 0.5, dtype=np.float32),
        )),
        ("medium", SafetyConfig(
            action_bounds=np.array([[-1, 1]] * 7, dtype=np.float32),
            max_velocity=np.full(7, 0.1, dtype=np.float32),
        )),
        ("tight", SafetyConfig(
            action_bounds=np.array([[-0.5, 0.5]] * 7, dtype=np.float32),
            max_velocity=np.full(7, 0.05, dtype=np.float32),
        )),
        ("very_tight", SafetyConfig(
            action_bounds=np.array([[-0.3, 0.3]] * 7, dtype=np.float32),
            max_velocity=np.full(7, 0.02, dtype=np.float32),
        )),
    ]

    results = []
    rng = np.random.default_rng(42)

    for name, config in configs:
        # Generate 100 random trajectories of 50 steps
        violations_total = 0
        for _ in range(100):
            actions = np.cumsum(rng.normal(0, 0.1, (50, 7)), axis=0).astype(np.float32)
            result = validate_actions(actions, config)
            violations_total += len(result.violations)

        results.append({
            "config": name,
            "total_violations": violations_total,
            "avg_violations_per_trajectory": violations_total / 100,
            "bounds": float(config.action_bounds[0][1]) if config.action_bounds is not None else None,
            "velocity_limit": float(config.max_velocity[0]) if config.max_velocity is not None else None,
        })

    path = RESULTS_DIR / "exp4_composition.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Exp 4: Composition interference across {len(configs)} configs")
    for r in results:
        print(f"  {r['config']}: {r['avg_violations_per_trajectory']:.1f} violations/trajectory")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("ICRA VLA Pipelines Workshop - Experiments")
    print("=" * 60)
    print()

    experiment_1_safety_contract_overhead()
    print()
    experiment_2_violation_detection()
    print()
    experiment_3_probeflow_epsilon_sensitivity()
    print()
    experiment_4_composition_interference()

    print()
    print(f"All results saved to {RESULTS_DIR}")
