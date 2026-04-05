"""EXP: Comprehensive LIBERO fingerprints - SmolVLA on ALL 4 suites x 10 tasks x 20 steps.

Runs SmolVLA closed-loop in LIBERO simulation environments across all 4 benchmark
suites (spatial, object, goal, 10). For each suite, runs all 10 tasks for 20 steps,
collecting raw (normalized) actions and computing per-dim violation fingerprints.

Produces:
  - Per-task fingerprints (bounds + velocity violation rates)
  - Per-suite aggregated fingerprints (mean/std across 10 tasks)
  - Cosine distance matrix between the 4 suites

Usage:
  cd ~/Desktop/projects/vla-edge
  source .venv/bin/activate
  python experiments/icra_ws_2026/exp_comprehensive_fingerprints.py
"""

import gc
import json
import os
import sys
import time
import traceback
import types
from collections import deque
from pathlib import Path

import numpy as np

# ---- Bypass GR00T policy registration issue ----
if "lerobot.policies" not in sys.modules:
    import lerobot

    policies_mod = types.ModuleType("lerobot.policies")
    policies_mod.__path__ = [lerobot.__path__[0] + "/policies"]
    policies_mod.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = policies_mod

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "exp_comprehensive_fingerprints.json"

SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
N_STEPS = 20
BOUNDS_LO, BOUNDS_HI = -1.0, 1.0
V_MAX = 0.1

# Dataset stats from HuggingFaceVLA/smol-libero (for unnormalization before env.step)
LIBERO_ACTION_MEAN = np.array(
    [0.007, 0.0836, -0.0395, 0.0005, 0.0032, -0.0014, 0.0064], dtype=np.float32
)
LIBERO_ACTION_STD = np.array(
    [0.2963, 0.4462, 0.4706, 0.031, 0.0483, 0.0433, 1.0], dtype=np.float32
)


# ---------------------------------------------------------------------------
# Observation helper
# ---------------------------------------------------------------------------

def build_observation(raw_obs: dict, task_description: str, policy, processor, device) -> dict:
    """Convert LIBERO raw observation to SmolVLA input format."""
    observation = {}

    # Camera images: LIBERO gives (H, W, 3) uint8
    # SmolVLA expects (B, C, H, W) float [0, 1]
    agentview = raw_obs["agentview_image"]
    wrist = raw_obs["robot0_eye_in_hand_image"]

    for img_arr, key in [
        (agentview, "observation.images.image"),
        (wrist, "observation.images.image2"),
    ]:
        img_f = img_arr.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        observation[key] = img_t.to(device)

    # State: joint_pos(7) + gripper_qpos(1) = 8 dims
    joint_pos = raw_obs.get("robot0_joint_pos", np.zeros(7))
    gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([joint_pos, gripper_qpos[:1]]).astype(np.float32)
    observation["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    # Language tokens
    tokens = processor.tokenizer(
        task_description,
        return_tensors="pt",
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )
    observation["observation.language.tokens"] = tokens["input_ids"].to(device)
    observation["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(device)

    return observation


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_task_fingerprint(actions: np.ndarray) -> dict:
    """Compute violation fingerprint for one task's action trajectory."""
    n_steps, n_dims = actions.shape

    # Bounds violations
    bounds_violations = np.zeros(n_dims)
    for d in range(n_dims):
        viols = np.sum((actions[:, d] < BOUNDS_LO) | (actions[:, d] > BOUNDS_HI))
        bounds_violations[d] = viols / n_steps

    # Velocity violations
    velocity_violations = np.zeros(n_dims)
    if n_steps > 1:
        for d in range(n_dims):
            velocities = np.abs(np.diff(actions[:, d]))
            vel_viols = np.sum(velocities > V_MAX)
            velocity_violations[d] = vel_viols / (n_steps - 1)

    action_mean = np.mean(actions, axis=0)
    action_std = np.std(actions, axis=0)
    action_min = np.min(actions, axis=0)
    action_max = np.max(actions, axis=0)

    oob_mask = (actions < BOUNDS_LO) | (actions > BOUNDS_HI)
    overall_bounds_rate = float(np.any(oob_mask, axis=1).mean())

    return {
        "bounds_per_dim": bounds_violations.tolist(),
        "velocity_per_dim": velocity_violations.tolist(),
        "action_mean_per_dim": action_mean.tolist(),
        "action_std_per_dim": action_std.tolist(),
        "action_min_per_dim": action_min.tolist(),
        "action_max_per_dim": action_max.tolist(),
        "overall_bounds_rate": overall_bounds_rate,
        "overall_velocity_rate": float(np.mean(velocity_violations)),
    }


def aggregate_fingerprints(task_fps: list[dict]) -> dict:
    """Mean and std of fingerprints across tasks in a suite."""
    bounds_all = np.array([fp["bounds_per_dim"] for fp in task_fps])
    velocity_all = np.array([fp["velocity_per_dim"] for fp in task_fps])

    return {
        "bounds_mean": np.mean(bounds_all, axis=0).tolist(),
        "bounds_std": np.std(bounds_all, axis=0).tolist(),
        "velocity_mean": np.mean(velocity_all, axis=0).tolist(),
        "velocity_std": np.std(velocity_all, axis=0).tolist(),
        "overall_bounds_mean": float(np.mean([fp["overall_bounds_rate"] for fp in task_fps])),
        "overall_bounds_std": float(np.std([fp["overall_bounds_rate"] for fp in task_fps])),
        "overall_velocity_mean": float(np.mean([fp["overall_velocity_rate"] for fp in task_fps])),
        "overall_velocity_std": float(np.std([fp["overall_velocity_rate"] for fp in task_fps])),
    }


def fingerprint_vector(agg: dict) -> np.ndarray:
    """Compact vector for distance computation: bounds_mean + velocity_mean."""
    return np.array(agg["bounds_mean"] + agg["velocity_mean"])


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 > 0 and n2 > 0:
        return float(1 - np.dot(v1, v2) / (n1 * n2))
    return 1.0


def save_results(all_results, device, load_time, total_tasks, total_steps, elapsed_total):
    """Save current results (supports partial saves on crash)."""
    # Cosine distance matrix (only for completed suites)
    completed = [s for s in SUITE_NAMES if s in all_results]
    distance_matrix = {}
    vectors = {s: fingerprint_vector(all_results[s]["aggregated"]) for s in completed}

    for i, s1 in enumerate(completed):
        for j, s2 in enumerate(completed):
            if j <= i:
                continue
            d = cosine_distance(vectors[s1], vectors[s2])
            distance_matrix[f"{s1}_vs_{s2}"] = round(d, 6)

    avg_cosine = float(np.mean(list(distance_matrix.values()))) if distance_matrix else 0.0

    output = {
        "experiment": "Comprehensive LIBERO Fingerprints",
        "description": "SmolVLA on ALL 4 LIBERO suites x 10 tasks x 20 steps",
        "model": "HuggingFaceVLA/smolvla_libero",
        "device": device,
        "n_steps_per_task": N_STEPS,
        "bounds": [BOUNDS_LO, BOUNDS_HI],
        "v_max": V_MAX,
        "total_tasks": total_tasks,
        "total_steps": total_steps,
        "total_time_s": round(elapsed_total, 1),
        "model_load_time_s": round(load_time, 1),
        "suites_completed": len(completed),
        "suites": all_results,
        "cosine_distance_matrix": distance_matrix,
        "avg_cosine_distance": round(avg_cosine, 6),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    OUT_PATH.write_text(json.dumps(
        output, indent=2,
        default=lambda o: o.item() if hasattr(o, "item") else float(o),
    ))
    return output, distance_matrix, avg_cosine


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

def run_task(task_id, suite, policy, processor, torch_device):
    """Run one task for N_STEPS, return actions array and latencies."""
    task = suite.get_task(task_id)
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )

    try:
        raw_obs = env.reset()

        # Reset policy action queues
        policy.reset()
        if hasattr(policy, "_queues"):
            for k in policy._queues:
                policy._queues[k] = deque(maxlen=policy._queues[k].maxlen)

        actions_list = []
        latencies = []

        for step in range(N_STEPS):
            obs = build_observation(raw_obs, task_description, policy, processor, torch_device)

            t_start = time.perf_counter()
            with torch.inference_mode():
                action_tensor = policy.select_action(obs)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            latencies.append(elapsed_ms)

            action = action_tensor.detach().cpu().numpy().squeeze()
            actions_list.append(action.copy())

            # Unnormalize for env stepping
            action_unnorm = action * LIBERO_ACTION_STD[:len(action)] + LIBERO_ACTION_MEAN[:len(action)]
            env_action = np.zeros(7)
            env_action[:min(len(action_unnorm), 7)] = action_unnorm[:7]
            raw_obs, reward, done, info = env.step(env_action)

            if done:
                break

        return task_description, np.array(actions_list), latencies

    finally:
        env.close()
        # Force cleanup to prevent MuJoCo memory leaks
        gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    print(f"Device: {device}", flush=True)

    # ---- Load model ----
    print("Loading SmolVLA policy...", flush=True)
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")
    torch_device = torch.device(device)
    policy.to(torch_device)
    policy.config.device = torch_device
    policy.eval()
    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    # ---- Run all suites ----
    all_results = {}
    total_tasks = 0
    total_steps = 0
    global_t0 = time.perf_counter()

    for suite_name in SUITE_NAMES:
        print(f"\n{'='*60}", flush=True)
        print(f"Suite: {suite_name}", flush=True)
        print(f"{'='*60}", flush=True)

        bench = benchmark.get_benchmark_dict()
        suite = bench[suite_name]()
        n_tasks = suite.n_tasks
        print(f"  {n_tasks} tasks", flush=True)

        suite_task_fps = []
        suite_task_details = []

        for task_id in range(n_tasks):
            try:
                task_desc, actions_arr, latencies = run_task(
                    task_id, suite, policy, processor, torch_device
                )

                fp = compute_task_fingerprint(actions_arr)
                suite_task_fps.append(fp)
                suite_task_details.append({
                    "task_index": task_id,
                    "task_name": task_desc,
                    "n_steps_actual": actions_arr.shape[0],
                    "action_shape": list(actions_arr.shape),
                    "avg_latency_ms": round(float(np.mean(latencies)), 1),
                    "fingerprint": fp,
                })

                total_tasks += 1
                total_steps += actions_arr.shape[0]

                print(f"\n  Task {task_id}/{n_tasks}: {task_desc}", flush=True)
                print(f"    Steps: {actions_arr.shape[0]}, Avg latency: {np.mean(latencies):.0f}ms", flush=True)
                print(f"    Bounds viol: {fp['overall_bounds_rate']:.1%}, "
                      f"Velocity viol: {fp['overall_velocity_rate']:.1%}", flush=True)
                bounds_str = " ".join(
                    f"{DIM_LABELS[d]}={fp['bounds_per_dim'][d]:.2f}"
                    for d in range(len(fp["bounds_per_dim"]))
                )
                print(f"    Per-dim bounds: {bounds_str}", flush=True)

            except Exception as e:
                print(f"\n  Task {task_id}/{n_tasks}: FAILED - {e}", flush=True)
                traceback.print_exc()
                gc.collect()
                continue

        if not suite_task_fps:
            print(f"\n  WARNING: No tasks completed for {suite_name}, skipping.", flush=True)
            continue

        # Aggregate suite
        agg = aggregate_fingerprints(suite_task_fps)
        all_results[suite_name] = {
            "n_tasks": n_tasks,
            "n_tasks_completed": len(suite_task_fps),
            "tasks": suite_task_details,
            "aggregated": agg,
        }

        print(f"\n  Suite aggregate ({suite_name}):", flush=True)
        print(f"    Bounds mean:    {[f'{v:.3f}' for v in agg['bounds_mean']]}", flush=True)
        print(f"    Bounds std:     {[f'{v:.3f}' for v in agg['bounds_std']]}", flush=True)
        print(f"    Velocity mean:  {[f'{v:.3f}' for v in agg['velocity_mean']]}", flush=True)
        print(f"    Velocity std:   {[f'{v:.3f}' for v in agg['velocity_std']]}", flush=True)
        print(f"    Overall bounds: {agg['overall_bounds_mean']:.1%} +/- {agg['overall_bounds_std']:.1%}", flush=True)
        print(f"    Overall velocity: {agg['overall_velocity_mean']:.1%} +/- {agg['overall_velocity_std']:.1%}", flush=True)

        # Save after each suite completes (crash recovery)
        elapsed_so_far = time.perf_counter() - global_t0
        save_results(all_results, device, load_time, total_tasks, total_steps, elapsed_so_far)
        print(f"\n  [checkpoint saved to {OUT_PATH}]", flush=True)

    # ---- Final save with cosine distance matrix ----
    elapsed_total = time.perf_counter() - global_t0
    output, distance_matrix, avg_cosine = save_results(
        all_results, device, load_time, total_tasks, total_steps, elapsed_total
    )

    print(f"\n{'='*60}", flush=True)
    print("Cosine distance matrix between suites", flush=True)
    print(f"{'='*60}", flush=True)
    for pair, dist in distance_matrix.items():
        print(f"  {pair}: {dist:.6f}", flush=True)
    print(f"\n  Average cosine distance: {avg_cosine:.6f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Results saved to {OUT_PATH}", flush=True)
    print(f"Total: {total_tasks} tasks, {total_steps} steps in {elapsed_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
