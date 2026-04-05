"""EXP: SmolVLA fingerprints on ALL 90 tasks in libero_90 suite.

Runs SmolVLA inference (10 steps per task) on each of the 90 LIBERO tasks,
collecting per-dim action bounds and velocity violation rates.

Follows the exact pattern from exp_real_task_fingerprints (inline script).

Usage:
  source .venv/bin/activate
  python experiments/icra_ws_2026/exp_libero90_fingerprints.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import traceback
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_STEPS = 10
ACTION_BOUNDS = (-1.0, 1.0)
VELOCITY_THRESHOLD = 0.15  # per-step delta threshold


def compute_task_fingerprint(actions: np.ndarray) -> dict:
    """Compute per-dim bounds and velocity violation rates."""
    n_steps, n_dims = actions.shape

    # Bounds violations per dim
    bounds_per_dim = []
    for d in range(n_dims):
        violations = np.sum(
            (actions[:, d] < ACTION_BOUNDS[0]) | (actions[:, d] > ACTION_BOUNDS[1])
        )
        bounds_per_dim.append(round(violations / n_steps, 4))

    # Velocity violations per dim
    velocity_per_dim = []
    if n_steps > 1:
        for d in range(n_dims):
            deltas = np.abs(np.diff(actions[:, d]))
            vel_violations = np.sum(deltas > VELOCITY_THRESHOLD)
            velocity_per_dim.append(round(vel_violations / (n_steps - 1), 4))
    else:
        velocity_per_dim = [0.0] * n_dims

    action_mean = np.mean(actions, axis=0).tolist()
    action_std = np.std(actions, axis=0).tolist()
    action_range = [float(actions.min()), float(actions.max())]

    return {
        "n_steps": n_steps,
        "action_range": action_range,
        "action_mean": action_mean,
        "action_std": action_std,
        "bounds_per_dim": bounds_per_dim,
        "velocity_per_dim": velocity_per_dim,
    }


def main():
    import torch

    # ---- Patch lerobot.policies import ----
    if "lerobot.policies" not in sys.modules:
        import lerobot
        m = types.ModuleType("lerobot.policies")
        m.__path__ = [lerobot.__path__[0] + "/policies"]
        m.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = m

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from transformers import AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # ---- Load model ----
    print("Loading SmolVLA policy...")
    t0 = time.time()
    policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    policy.to(device).eval()
    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
    print(f"Model loaded on {device} in {time.time() - t0:.1f}s")

    # ---- Get all 90 tasks ----
    print("Loading libero_90 benchmark...")
    bench_dict = benchmark.get_benchmark_dict()
    Libero90 = bench_dict["libero_90"]
    bm = Libero90(task_order_index=0)
    n_tasks = bm.get_num_tasks()
    task_names = bm.get_task_names()
    print(f"Found {n_tasks} tasks")

    # ---- Run per-task fingerprinting ----
    results_per_task = {}
    skipped_tasks = []
    all_timings = []

    for task_id in range(n_tasks):
        task_name = task_names[task_id]
        print(f"\n[{task_id + 1}/{n_tasks}] {task_name}")

        try:
            # Create environment
            task = bm.get_task(task_id)
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )

            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=256,
                camera_widths=256,
            )

            # Get init states and reset
            init_states = bm.get_task_init_states(task_id)
            env.reset()
            if init_states is not None and len(init_states) > 0:
                env.set_init_state(init_states[0])
            raw_obs = env.reset()

            # Reset policy
            policy.reset()

            # Run N_STEPS
            actions_collected = []
            step_timings = []

            for step in range(N_STEPS):
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
                    observation[key] = img_t.to(device)

                # State
                joint_pos = raw_obs.get("robot0_joint_pos", np.zeros(7))
                gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
                state = np.concatenate([joint_pos, gripper_qpos[:1]]).astype(np.float32)
                observation["observation.state"] = (
                    torch.from_numpy(state).unsqueeze(0).to(device)
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
                observation["observation.language.tokens"] = (
                    tokens["input_ids"].to(device)
                )
                observation["observation.language.attention_mask"] = (
                    tokens["attention_mask"].bool().to(device)
                )

                # Inference
                t_step = time.perf_counter()
                with torch.inference_mode():
                    action_out = policy.select_action(observation)
                elapsed = (time.perf_counter() - t_step) * 1000
                step_timings.append(elapsed)

                # select_action returns a tensor (1, 7) not a dict
                if isinstance(action_out, dict):
                    action = action_out["action"].squeeze().cpu().numpy()
                else:
                    action = action_out.squeeze().cpu().numpy()
                actions_collected.append(action.copy())

                # Step env
                raw_obs, reward, done, info = env.step(action)
                if done:
                    break

            env.close()

            # Compute fingerprint
            actions_arr = np.array(actions_collected)
            fp = compute_task_fingerprint(actions_arr)
            fp["avg_latency_ms"] = round(float(np.mean(step_timings)), 1)

            results_per_task[f"task_{task_id:02d}"] = {
                "task_name": task_name,
                **fp,
            }
            all_timings.extend(step_timings)

            print(
                f"  OK: {len(actions_collected)} steps, "
                f"range=[{actions_arr.min():.2f}, {actions_arr.max():.2f}], "
                f"latency={np.mean(step_timings):.0f}ms"
            )

        except Exception as e:
            skipped_tasks.append({"task_id": task_id, "task_name": task_name, "error": str(e)})
            print(f"  SKIPPED: {e}")
            traceback.print_exc()
            # Try to close env if it was created
            try:
                env.close()
            except Exception:
                pass
            continue

    # ---- Aggregate stats ----
    print(f"\n{'='*60}")
    print(f"Completed: {len(results_per_task)}/{n_tasks} tasks")
    print(f"Skipped:   {len(skipped_tasks)}")

    # Per-dim aggregate
    all_bounds = []
    all_velocity = []
    for r in results_per_task.values():
        all_bounds.append(r["bounds_per_dim"])
        all_velocity.append(r["velocity_per_dim"])

    if all_bounds:
        avg_bounds = np.mean(all_bounds, axis=0).round(4).tolist()
        avg_velocity = np.mean(all_velocity, axis=0).round(4).tolist()
        std_bounds = np.std(all_bounds, axis=0).round(4).tolist()
        std_velocity = np.std(all_velocity, axis=0).round(4).tolist()
    else:
        avg_bounds = avg_velocity = std_bounds = std_velocity = []

    # Overall violation rates
    overall_bounds_rates = [
        np.mean(r["bounds_per_dim"]) for r in results_per_task.values()
    ]
    overall_vel_rates = [
        np.mean(r["velocity_per_dim"]) for r in results_per_task.values()
    ]

    aggregate = {
        "n_tasks_completed": len(results_per_task),
        "n_tasks_skipped": len(skipped_tasks),
        "n_steps_per_task": N_STEPS,
        "total_inferences": len(all_timings),
        "avg_latency_ms": round(float(np.mean(all_timings)), 1) if all_timings else 0,
        "bounds_violation_rate": {
            "mean_per_dim": avg_bounds,
            "std_per_dim": std_bounds,
            "overall_mean": round(float(np.mean(overall_bounds_rates)), 4) if overall_bounds_rates else 0,
            "overall_std": round(float(np.std(overall_bounds_rates)), 4) if overall_bounds_rates else 0,
        },
        "velocity_violation_rate": {
            "mean_per_dim": avg_velocity,
            "std_per_dim": std_velocity,
            "overall_mean": round(float(np.mean(overall_vel_rates)), 4) if overall_vel_rates else 0,
            "overall_std": round(float(np.std(overall_vel_rates)), 4) if overall_vel_rates else 0,
        },
    }

    # ---- Save ----
    output = {
        "experiment": "SmolVLA fingerprints on all 90 LIBERO-90 tasks",
        "model": "HuggingFaceVLA/smolvla_libero",
        "device": device,
        "n_steps_per_task": N_STEPS,
        "action_bounds": list(ACTION_BOUNDS),
        "velocity_threshold": VELOCITY_THRESHOLD,
        "aggregate": aggregate,
        "tasks": results_per_task,
        "skipped_tasks": skipped_tasks,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    path = RESULTS_DIR / "exp_libero90_fingerprints.json"
    path.write_text(
        json.dumps(
            output,
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else float(o),
        )
    )

    print(f"\nResults saved to {path}")
    print(f"Total time: {sum(all_timings) / 1000:.1f}s for {len(all_timings)} inferences")
    if all_timings:
        print(f"Avg latency: {np.mean(all_timings):.0f}ms")
    print(f"\nAggregate bounds violation (mean across tasks): {aggregate['bounds_violation_rate']['overall_mean']:.4f}")
    print(f"Aggregate velocity violation (mean across tasks): {aggregate['velocity_violation_rate']['overall_mean']:.4f}")

    if skipped_tasks:
        print(f"\nSkipped tasks:")
        for s in skipped_tasks:
            print(f"  task {s['task_id']}: {s['task_name']} - {s['error']}")


if __name__ == "__main__":
    main()
