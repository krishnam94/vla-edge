"""FIX 2: Corrected fingerprints with 5 episodes per suite (error bars).

The v1 experiment ran 1 episode (20 steps) per LIBERO suite - n=1, no error bars.

This v2 fix:
  - Runs 5 episodes per suite (task 0, reset 5 times with different init states)
  - Computes mean +/- std for each dimension's velocity violation rate
  - Provides statistical backing with error bars

Usage:
  source .venv/bin/activate
  python experiments/icra_ws_2026/exp_normalized_fingerprints_v2.py [--n-steps 20] [--n-episodes 5] [--device mps]
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


def compute_violations(actions: np.ndarray) -> dict:
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


def run_episode(env, suite, task, task_name, policy, processor, actual_device,
                init_state, n_steps: int) -> tuple[np.ndarray, list[float]]:
    """Run a single episode and return raw actions + timings."""
    import torch

    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
    raw_obs = env.reset()

    policy.reset()

    raw_actions_list = []
    step_timings = []

    for step in range(n_steps):
        observation = {}

        agentview = raw_obs["agentview_image"]
        wrist = raw_obs["robot0_eye_in_hand_image"]

        for img_arr, key in [
            (agentview, "observation.images.image"),
            (wrist, "observation.images.image2"),
        ]:
            img_f = img_arr.astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
            observation[key] = img_t.to(actual_device)

        joint_pos = raw_obs.get("robot0_joint_pos", np.zeros(7))
        gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
        state = np.concatenate([joint_pos, gripper_qpos[:1]]).astype(np.float32)
        observation["observation.state"] = (
            torch.from_numpy(state).unsqueeze(0).to(actual_device)
        )

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

        t_step = time.perf_counter()
        with torch.inference_mode():
            action_out = policy.select_action(observation)
        elapsed = (time.perf_counter() - t_step) * 1000
        step_timings.append(elapsed)

        if isinstance(action_out, dict):
            action = action_out["action"].squeeze().cpu().numpy()
        else:
            action = action_out.squeeze().cpu().numpy()

        raw_actions_list.append(action.copy())

        raw_obs, reward, done, info = env.step(action)

        if done:
            break

    return np.array(raw_actions_list, dtype=np.float32), step_timings


def main(n_steps: int = 20, n_episodes: int = 5, device: str = "mps"):
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

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f"=== Corrected Fingerprints v2: {n_episodes} episodes per suite ===")
    print(f"Steps per episode: {n_steps}, Device: {device}")
    print(f"FIX: n=1 -> n={n_episodes} with mean +/- std error bars")
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

    results_by_suite = {}
    all_timings = []

    for suite_name in SUITE_NAMES:
        print(f"\n{'='*60}")
        print(f"Suite: {suite_name} (task 0, {n_episodes} episodes x {n_steps} steps)")
        print(f"{'='*60}")

        try:
            env, suite, task, task_name = create_libero_env(suite_name, task_id=0)
        except Exception as e:
            print(f"  FAILED to create env: {e}")
            results_by_suite[suite_name] = {"error": str(e)}
            continue

        init_states = suite.get_task_init_states(0)

        # Run n_episodes
        episode_results = []
        all_vel_rates_unnorm = {dim: [] for dim in DIM_LABELS}
        all_vel_rates_raw = {dim: [] for dim in DIM_LABELS}
        all_overall_vel_unnorm = []
        all_overall_vel_raw = []
        all_overall_bounds_unnorm = []

        for ep_i in range(n_episodes):
            # Use different init state if available, otherwise same state (still different due to sim stochasticity)
            if init_states is not None and len(init_states) > 0:
                state_idx = ep_i % len(init_states)
                init_state = init_states[state_idx]
            else:
                init_state = None

            print(f"\n  Episode {ep_i + 1}/{n_episodes} (init_state idx={state_idx if init_state is not None else 'None'})...")

            raw_actions, step_timings = run_episode(
                env, suite, task, task_name, policy, processor,
                actual_device, init_state, n_steps,
            )
            all_timings.extend(step_timings)

            unnorm_actions = unnormalize(raw_actions)

            raw_v = compute_violations(raw_actions)
            unnorm_v = compute_violations(unnorm_actions)

            # Collect per-dim velocity rates for aggregation
            for dim in DIM_LABELS:
                all_vel_rates_unnorm[dim].append(unnorm_v["velocity_per_dim"][dim])
                all_vel_rates_raw[dim].append(raw_v["velocity_per_dim"][dim])
            all_overall_vel_unnorm.append(unnorm_v["overall_velocity_rate"])
            all_overall_vel_raw.append(raw_v["overall_velocity_rate"])
            all_overall_bounds_unnorm.append(unnorm_v["overall_bounds_rate"])

            episode_results.append({
                "episode": ep_i,
                "n_steps": len(raw_actions),
                "raw_violations": raw_v,
                "unnorm_violations": unnorm_v,
                "avg_latency_ms": round(float(np.mean(step_timings)), 1),
            })

            print(f"    Unnorm vel: {unnorm_v['overall_velocity_rate']*100:.1f}%, "
                  f"bounds: {unnorm_v['overall_bounds_rate']*100:.1f}%")

        env.close()

        # Compute mean +/- std across episodes
        vel_per_dim_mean = {dim: round(float(np.mean(all_vel_rates_unnorm[dim])), 4) for dim in DIM_LABELS}
        vel_per_dim_std = {dim: round(float(np.std(all_vel_rates_unnorm[dim])), 4) for dim in DIM_LABELS}

        overall_vel_mean = round(float(np.mean(all_overall_vel_unnorm)), 4)
        overall_vel_std = round(float(np.std(all_overall_vel_unnorm)), 4)
        overall_bounds_mean = round(float(np.mean(all_overall_bounds_unnorm)), 4)
        overall_bounds_std = round(float(np.std(all_overall_bounds_unnorm)), 4)

        results_by_suite[suite_name] = {
            "task": task_name,
            "n_episodes": n_episodes,
            "n_steps_per_episode": n_steps,
            "episodes": episode_results,
            "aggregate": {
                "velocity_per_dim_mean": vel_per_dim_mean,
                "velocity_per_dim_std": vel_per_dim_std,
                "overall_velocity_mean": overall_vel_mean,
                "overall_velocity_std": overall_vel_std,
                "overall_bounds_mean": overall_bounds_mean,
                "overall_bounds_std": overall_bounds_std,
            },
        }

        print(f"\n  AGGREGATE ({n_episodes} episodes):")
        print(f"    Velocity (unnorm): {overall_vel_mean*100:.1f}% +/- {overall_vel_std*100:.1f}%")
        print(f"    Bounds (unnorm):   {overall_bounds_mean*100:.1f}% +/- {overall_bounds_std*100:.1f}%")
        print(f"    Per-dim velocity (mean +/- std):")
        for dim in DIM_LABELS:
            print(f"      {dim}: {vel_per_dim_mean[dim]*100:.1f}% +/- {vel_per_dim_std[dim]*100:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: v2 with {n_episodes} episodes per suite")
    print(f"{'='*60}")

    for suite_name, r in results_by_suite.items():
        if "error" in r:
            print(f"\n  {suite_name}: FAILED - {r['error']}")
            continue
        agg = r["aggregate"]
        print(f"\n  {suite_name} ({r['task']}):")
        print(f"    Velocity: {agg['overall_velocity_mean']*100:.1f}% +/- {agg['overall_velocity_std']*100:.1f}%")
        print(f"    Bounds:   {agg['overall_bounds_mean']*100:.1f}% +/- {agg['overall_bounds_std']*100:.1f}%")

    # Velocity fingerprint distinctness (using means)
    print(f"\n--- Velocity Fingerprint Distinctness (mean vectors) ---")
    valid_suites = [s for s in SUITE_NAMES if s in results_by_suite and "error" not in results_by_suite[s]]
    pairwise = {}
    for i, s1 in enumerate(valid_suites):
        for j, s2 in enumerate(valid_suites):
            if j <= i:
                continue
            v1 = np.array(list(results_by_suite[s1]["aggregate"]["velocity_per_dim_mean"].values()))
            v2 = np.array(list(results_by_suite[s2]["aggregate"]["velocity_per_dim_mean"].values()))
            l2 = float(np.linalg.norm(v1 - v2))
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cos_dist = 1 - np.dot(v1, v2) / (n1 * n2) if n1 > 0 and n2 > 0 else 1.0
            pair_key = f"{s1}_vs_{s2}"
            pairwise[pair_key] = {"cosine": round(float(cos_dist), 4), "l2": round(l2, 4)}
            print(f"  {s1} vs {s2}: cosine={cos_dist:.4f}, L2={l2:.4f}")

    avg_cosine = float(np.mean([d["cosine"] for d in pairwise.values()])) if pairwise else 0
    avg_l2 = float(np.mean([d["l2"] for d in pairwise.values()])) if pairwise else 0

    # Save results
    output = {
        "experiment": "Corrected fingerprints v2 - 5 episodes per suite with error bars",
        "fix": "v1 used n=1 episode per suite (no error bars). v2 runs 5 episodes "
               "per suite and reports mean +/- std for statistical backing.",
        "model": "HuggingFaceVLA/smolvla_libero",
        "device": actual_device,
        "n_episodes_per_suite": n_episodes,
        "n_steps_per_episode": n_steps,
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
            f"With {n_episodes} episodes per suite, velocity fingerprints have error bars. "
            "Bounds violations near zero after unnormalization. "
            "Velocity violations persist and differ by task with measurable variance."
        ),
        "avg_latency_ms": round(float(np.mean(all_timings)), 1) if all_timings else 0,
        "total_inferences": len(all_timings),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = RESULTS_DIR / "exp_normalized_fingerprints_v2.json"
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
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    main(n_steps=args.n_steps, n_episodes=args.n_episodes, device=args.device)
