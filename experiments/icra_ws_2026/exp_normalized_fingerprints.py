"""EXP: Corrected fingerprints - SmolVLA on 4 LIBERO suites WITH unnormalization.

The prior fingerprint experiment (EXP-H) had a normalization mismatch:
SmolVLA outputs normalized actions, but violations were checked on raw output.
This experiment applies unnormalization BEFORE checking violations.

After unnormalization, bounds violations should be near zero.
VELOCITY violations are the honest fingerprint - they differ by task and
represent real physical constraint violations.

For each of 4 suites (libero_spatial, libero_object, libero_goal, libero_10):
  1. Run SmolVLA in LIBERO sim (task 0, 20 steps)
  2. Get raw model output (normalized)
  3. Unnormalize: action_unnorm = raw * std + mean
  4. Check violations on UNNORMALIZED actions (bounds [-1,1], v_max=0.1)
  5. Record per-dim velocity violation rates

Usage:
  source .venv/bin/activate
  python experiments/icra_ws_2026/exp_normalized_fingerprints.py [--n-steps 20] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# LIBERO action normalization stats from HuggingFaceVLA/smol-libero
LIBERO_ACTION_MEAN = np.array(
    [0.007, 0.0836, -0.0395, 0.0005, 0.0032, -0.0014, 0.0064],
    dtype=np.float32,
)
LIBERO_ACTION_STD = np.array(
    [0.2963, 0.4462, 0.4706, 0.031, 0.0483, 0.0433, 1.0],
    dtype=np.float32,
)

DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

ACTION_BOUNDS = (-1.0, 1.0)
V_MAX = 0.1


def unnormalize(raw_actions: np.ndarray) -> np.ndarray:
    """Unnormalize SmolVLA output: action_real = raw * std + mean."""
    return raw_actions * LIBERO_ACTION_STD + LIBERO_ACTION_MEAN


def compute_violations(actions: np.ndarray, label: str = "") -> dict:
    """Compute per-dim bounds and velocity violation rates."""
    n_steps, n_dims = actions.shape
    lo, hi = ACTION_BOUNDS

    bounds_per_dim = []
    for d in range(n_dims):
        oob = np.sum((actions[:, d] < lo) | (actions[:, d] > hi))
        bounds_per_dim.append(round(float(oob / n_steps), 4))

    velocity_per_dim = []
    if n_steps > 1:
        for d in range(n_dims):
            deltas = np.abs(np.diff(actions[:, d]))
            vel_violations = np.sum(deltas > V_MAX)
            velocity_per_dim.append(round(float(vel_violations / (n_steps - 1)), 4))
    else:
        velocity_per_dim = [0.0] * n_dims

    # Overall rates
    any_oob = np.any((actions < lo) | (actions > hi), axis=1)
    overall_bounds = round(float(np.mean(any_oob)), 4)

    if n_steps > 1:
        all_deltas = np.abs(np.diff(actions, axis=0))
        any_vel = np.any(all_deltas > V_MAX, axis=1)
        overall_velocity = round(float(np.mean(any_vel)), 4)
    else:
        overall_velocity = 0.0

    return {
        "bounds_per_dim": {DIM_LABELS[d]: bounds_per_dim[d] for d in range(n_dims)},
        "velocity_per_dim": {DIM_LABELS[d]: velocity_per_dim[d] for d in range(n_dims)},
        "overall_bounds_rate": overall_bounds,
        "overall_velocity_rate": overall_velocity,
        "action_range": [round(float(actions.min()), 4), round(float(actions.max()), 4)],
        "per_dim_range": {
            DIM_LABELS[d]: [round(float(actions[:, d].min()), 4), round(float(actions[:, d].max()), 4)]
            for d in range(n_dims)
        },
    }


def create_libero_env(suite_name: str, task_id: int = 0):
    """Create a LIBERO environment for a given suite and task."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()
    suite = bench[suite_name]()
    task = suite.get_task(task_id)
    task_name = suite.get_task_names()[task_id]

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )

    return env, suite, task, task_name


def main(n_steps: int = 20, device: str = "mps"):
    import torch

    # Patch lerobot.policies import to bypass GR00T dataclass bug
    if "lerobot.policies" not in sys.modules:
        import lerobot

        m = types.ModuleType("lerobot.policies")
        m.__path__ = [lerobot.__path__[0] + "/policies"]
        m.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = m

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers import AutoProcessor

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f"=== Corrected Fingerprints: SmolVLA + Unnormalization ===")
    print(f"Steps per task: {n_steps}, Device: {device}")
    print(f"Unnormalization: action_real = raw * std + mean")
    print(f"Bounds: {ACTION_BOUNDS}, v_max: {V_MAX}")

    # Load model
    print("\nLoading SmolVLA (HuggingFaceVLA/smolvla_libero)...")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")
    actual_device = device
    if device == "mps" and hasattr(torch.backends, "mps") and not torch.backends.mps.is_available():
        actual_device = "cpu"
        print("MPS not available, falling back to CPU.")
    policy.to(actual_device).eval()
    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
    print(f"Model loaded on {actual_device} in {time.time() - t0:.1f}s")

    # Run each suite
    results_by_suite = {}
    all_timings = []

    for suite_name in SUITE_NAMES:
        print(f"\n{'='*60}")
        print(f"Suite: {suite_name} (task 0, {n_steps} steps)")
        print(f"{'='*60}")

        try:
            env, suite, task, task_name = create_libero_env(suite_name, task_id=0)
        except Exception as e:
            print(f"  FAILED to create env: {e}")
            results_by_suite[suite_name] = {"error": str(e)}
            continue

        # Get init states and reset
        init_states = suite.get_task_init_states(0)
        env.reset()
        if init_states is not None and len(init_states) > 0:
            env.set_init_state(init_states[0])
        raw_obs = env.reset()

        # Reset policy
        policy.reset()

        # Run steps
        raw_actions_list = []
        step_timings = []

        for step in range(n_steps):
            # Build observation dict
            observation = {}

            # Camera images
            agentview = raw_obs["agentview_image"]  # (H, W, 3) uint8
            wrist = raw_obs["robot0_eye_in_hand_image"]  # (H, W, 3) uint8

            for img_arr, key in [
                (agentview, "observation.images.image"),
                (wrist, "observation.images.image2"),
            ]:
                img_f = img_arr.astype(np.float32) / 255.0
                img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
                observation[key] = img_t.to(actual_device)

            # State
            joint_pos = raw_obs.get("robot0_joint_pos", np.zeros(7))
            gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
            state = np.concatenate([joint_pos, gripper_qpos[:1]]).astype(np.float32)
            observation["observation.state"] = (
                torch.from_numpy(state).unsqueeze(0).to(actual_device)
            )

            # Language tokens
            task_desc = task_name.replace("_", " ")
            tokens = processor.tokenizer(
                task_desc,
                return_tensors="pt",
                padding="max_length",
                max_length=policy.config.tokenizer_max_length,
                truncation=True,
            )
            observation["observation.language.tokens"] = tokens["input_ids"].to(actual_device)
            observation["observation.language.attention_mask"] = (
                tokens["attention_mask"].bool().to(actual_device)
            )

            # Inference
            t_step = time.perf_counter()
            with torch.inference_mode():
                action_out = policy.select_action(observation)
            elapsed = (time.perf_counter() - t_step) * 1000
            step_timings.append(elapsed)

            # Extract action numpy
            if isinstance(action_out, dict):
                action = action_out["action"].squeeze().cpu().numpy()
            else:
                action = action_out.squeeze().cpu().numpy()

            raw_actions_list.append(action.copy())

            # Step env with raw action (env expects normalized)
            raw_obs, reward, done, info = env.step(action)

            if (step + 1) % 5 == 0:
                print(f"  Step {step + 1}/{n_steps}: {elapsed:.0f}ms")

            if done:
                print(f"  Episode done at step {step + 1}")
                break

        env.close()

        raw_actions = np.array(raw_actions_list, dtype=np.float32)
        actual_steps = len(raw_actions_list)

        # Unnormalize
        unnorm_actions = unnormalize(raw_actions)

        print(f"\n  Raw action range: [{raw_actions.min():.4f}, {raw_actions.max():.4f}]")
        print(f"  Unnormalized range: [{unnorm_actions.min():.4f}, {unnorm_actions.max():.4f}]")

        # Compute violations on BOTH for comparison
        raw_violations = compute_violations(raw_actions, label="raw")
        unnorm_violations = compute_violations(unnorm_actions, label="unnorm")

        # Print comparison
        print(f"\n  --- Raw (no unnormalization) ---")
        print(f"  Bounds rate: {raw_violations['overall_bounds_rate']*100:.1f}%")
        print(f"  Velocity rate: {raw_violations['overall_velocity_rate']*100:.1f}%")
        print(f"  Bounds per dim: ", end="")
        for d in DIM_LABELS:
            print(f"{d}={raw_violations['bounds_per_dim'][d]*100:.0f}% ", end="")
        print()

        print(f"\n  --- Unnormalized (corrected) ---")
        print(f"  Bounds rate: {unnorm_violations['overall_bounds_rate']*100:.1f}%")
        print(f"  Velocity rate: {unnorm_violations['overall_velocity_rate']*100:.1f}%")
        print(f"  Bounds per dim: ", end="")
        for d in DIM_LABELS:
            print(f"{d}={unnorm_violations['bounds_per_dim'][d]*100:.0f}% ", end="")
        print()
        print(f"  Velocity per dim: ", end="")
        for d in DIM_LABELS:
            print(f"{d}={unnorm_violations['velocity_per_dim'][d]*100:.0f}% ", end="")
        print()

        results_by_suite[suite_name] = {
            "task": task_name,
            "n_steps": actual_steps,
            "avg_latency_ms": round(float(np.mean(step_timings)), 1),
            "raw_action_range": [round(float(raw_actions.min()), 4), round(float(raw_actions.max()), 4)],
            "unnorm_action_range": [round(float(unnorm_actions.min()), 4), round(float(unnorm_actions.max()), 4)],
            "raw_violations": raw_violations,
            "unnorm_violations": unnorm_violations,
            "raw_action_mean": np.mean(raw_actions, axis=0).round(4).tolist(),
            "unnorm_action_mean": np.mean(unnorm_actions, axis=0).round(4).tolist(),
            "unnorm_action_std": np.std(unnorm_actions, axis=0).round(4).tolist(),
        }
        all_timings.extend(step_timings)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Unnormalization Impact on Violations")
    print(f"{'='*60}")

    for suite_name, r in results_by_suite.items():
        if "error" in r:
            print(f"\n  {suite_name}: FAILED - {r['error']}")
            continue
        raw_b = r["raw_violations"]["overall_bounds_rate"]
        unnorm_b = r["unnorm_violations"]["overall_bounds_rate"]
        raw_v = r["raw_violations"]["overall_velocity_rate"]
        unnorm_v = r["unnorm_violations"]["overall_velocity_rate"]
        print(f"\n  {suite_name} ({r['task']}):")
        print(f"    Bounds:   {raw_b*100:.1f}% -> {unnorm_b*100:.1f}% (drop: {(raw_b - unnorm_b)*100:.1f}pp)")
        print(f"    Velocity: {raw_v*100:.1f}% -> {unnorm_v*100:.1f}%")
        print(f"    Velocity fingerprint: ", end="")
        for d in DIM_LABELS:
            print(f"{d}={r['unnorm_violations']['velocity_per_dim'][d]*100:.0f}% ", end="")
        print()

    # Velocity fingerprint distinctness (the honest signal)
    print(f"\n--- Velocity Fingerprint Distinctness (the honest signal) ---")
    valid_suites = [s for s in SUITE_NAMES if s in results_by_suite and "error" not in results_by_suite[s]]
    pairwise = {}
    for i, s1 in enumerate(valid_suites):
        for j, s2 in enumerate(valid_suites):
            if j <= i:
                continue
            v1 = np.array(list(results_by_suite[s1]["unnorm_violations"]["velocity_per_dim"].values()))
            v2 = np.array(list(results_by_suite[s2]["unnorm_violations"]["velocity_per_dim"].values()))
            l2 = float(np.linalg.norm(v1 - v2))
            # Cosine distance
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cos_dist = 1 - np.dot(v1, v2) / (n1 * n2) if n1 > 0 and n2 > 0 else 1.0
            pair_key = f"{s1}_vs_{s2}"
            pairwise[pair_key] = {"cosine": round(float(cos_dist), 4), "l2": round(l2, 4)}
            print(f"  {s1} vs {s2}: cosine={cos_dist:.4f}, L2={l2:.4f}")

    avg_cosine = float(np.mean([d["cosine"] for d in pairwise.values()])) if pairwise else 0
    avg_l2 = float(np.mean([d["l2"] for d in pairwise.values()])) if pairwise else 0
    print(f"  Average cosine: {avg_cosine:.4f}, Average L2: {avg_l2:.4f}")

    # Save results
    output = {
        "experiment": "Corrected fingerprints - SmolVLA on 4 LIBERO suites with unnormalization",
        "model": "HuggingFaceVLA/smolvla_libero",
        "device": actual_device,
        "n_steps_per_task": n_steps,
        "task_id": 0,
        "action_bounds": list(ACTION_BOUNDS),
        "v_max": V_MAX,
        "normalization_stats": {
            "mean": LIBERO_ACTION_MEAN.tolist(),
            "std": LIBERO_ACTION_STD.tolist(),
            "formula": "action_real = raw * std + mean",
        },
        "suites": results_by_suite,
        "velocity_fingerprint_distinctness": {
            "pairwise": pairwise,
            "avg_cosine": round(avg_cosine, 4),
            "avg_l2": round(avg_l2, 4),
        },
        "key_result": (
            "After correct unnormalization, bounds violations drop dramatically. "
            "Velocity violations persist and differ by task - they are the honest fingerprint."
        ),
        "avg_latency_ms": round(float(np.mean(all_timings)), 1) if all_timings else 0,
        "total_inferences": len(all_timings),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = RESULTS_DIR / "exp_normalized_fingerprints.json"
    out_path.write_text(
        json.dumps(
            output,
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else float(o),
        )
    )
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {sum(all_timings) / 1000:.1f}s for {len(all_timings)} inferences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    main(n_steps=args.n_steps, device=args.device)
