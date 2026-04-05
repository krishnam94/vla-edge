"""Temporal violation evolution: how violation rates change WITHIN an episode.

Runs SmolVLA on two LIBERO tasks for 50-step episodes and tracks per-step
violation status (bounds, velocity, which dims). Computes sliding-window
violation rate to reveal phase-dependent safety patterns (approach vs contact
vs manipulation).

Usage:
  python experiments/icra_ws_2026/exp_temporal_evolution.py [--device cpu]

Requirements:
  pip install robosuite==1.4.1 mujoco libero future easydict
  pip install vla-edge[smolvla]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ACTION_DIM = 7
EPISODE_STEPS = 50
SLIDING_WINDOW = 10

# Dataset stats from HuggingFaceVLA/smol-libero
LIBERO_ACTION_MEAN = np.array([0.007, 0.0836, -0.0395, 0.0005, 0.0032, -0.0014, 0.0064], dtype=np.float32)
LIBERO_ACTION_STD = np.array([0.2963, 0.4462, 0.4706, 0.031, 0.0483, 0.0433, 1.0], dtype=np.float32)

TASKS = [
    {"suite": "libero_object", "task_id": 0, "label": "pick_up_alphabet_soup"},
    {"suite": "libero_goal", "task_id": 0, "label": "open_middle_drawer"},
]


def create_libero_env(suite_name: str, task_id: int):
    """Create a single LIBERO environment."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()
    suite = bench[suite_name]()
    task = suite.get_task(task_id)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    return env, suite, task


def load_smolvla_policy(model_id: str, device: str):
    """Load SmolVLA policy directly (bypass GR00T conflict)."""
    import torch
    import types

    if "lerobot.policies" not in sys.modules:
        import lerobot
        policies_mod = types.ModuleType("lerobot.policies")
        policies_mod.__path__ = [lerobot.__path__[0] + "/policies"]
        policies_mod.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = policies_mod

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    logger.info("Loading SmolVLA from %s on %s...", model_id, device)
    policy = SmolVLAPolicy.from_pretrained(model_id)

    torch_device = torch.device(device)
    policy.to(torch_device)
    policy.config.device = torch_device
    policy.eval()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)

    logger.info("SmolVLA loaded. Action dim: %d, chunk size: %d",
                policy.config.output_features["action"].shape[0],
                policy.config.chunk_size)
    return policy, processor, torch_device


def build_observation(raw_obs, task_description, policy, processor, device):
    """Convert LIBERO raw observation to SmolVLA input format."""
    import torch

    observation = {}

    agentview = raw_obs["agentview_image"]
    wrist = raw_obs["robot0_eye_in_hand_image"]

    for img_arr, key in [
        (agentview, "observation.images.image"),
        (wrist, "observation.images.image2"),
    ]:
        img_f = img_arr.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        observation[key] = img_t.to(device)

    joint_pos = raw_obs.get("robot0_joint_pos", np.zeros(7))
    gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
    state = np.concatenate([joint_pos, gripper_qpos[:1]]).astype(np.float32)
    observation["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

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


def check_step_violations(
    action: np.ndarray,
    last_action: np.ndarray | None,
    action_range: tuple[float, float] = (-1.0, 1.0),
    velocity_max: float = 0.15,
) -> dict:
    """Check a single action for violations. Returns detailed per-dim info."""
    result = {
        "bounds_violated": False,
        "velocity_violated": False,
        "bounds_dims": [],
        "velocity_dims": [],
        "action_min": float(action.min()),
        "action_max": float(action.max()),
        "action_norm": float(np.linalg.norm(action)),
        "max_velocity": 0.0,
    }

    # Bounds check per dim
    for d in range(len(action)):
        if action[d] < action_range[0] or action[d] > action_range[1]:
            result["bounds_violated"] = True
            result["bounds_dims"].append(d)

    # Velocity check per dim
    if last_action is not None:
        delta = np.abs(action - last_action)
        result["max_velocity"] = float(delta.max())
        for d in range(len(action)):
            if delta[d] > velocity_max:
                result["velocity_violated"] = True
                result["velocity_dims"].append(d)

    return result


def compute_sliding_window_rate(violations_bool: list[bool], window: int) -> list[float | None]:
    """Compute sliding window violation rate. Returns None for steps before window is full."""
    rates = []
    for i in range(len(violations_bool)):
        if i < window - 1:
            rates.append(None)
        else:
            window_slice = violations_bool[i - window + 1 : i + 1]
            rates.append(sum(window_slice) / window)
    return rates


def run_temporal_episode(
    env,
    task_description: str,
    policy,
    processor,
    device,
    max_steps: int,
    init_states=None,
    init_state_id: int = 0,
) -> dict:
    """Run one episode and return per-step violation data."""
    import torch
    from collections import deque

    raw_obs = env.reset()

    if init_states is not None:
        raw_obs = env.set_init_state(init_states[init_state_id % len(init_states)])
        for _ in range(10):
            raw_obs, _, _, _ = env.step(np.zeros(ACTION_DIM))

    # Reset policy action queue
    if hasattr(policy, "_queues"):
        for k in policy._queues:
            policy._queues[k] = deque(maxlen=policy._queues[k].maxlen)

    per_step = []
    last_action = None

    t_start = time.perf_counter()
    for step in range(max_steps):
        obs = build_observation(raw_obs, task_description, policy, processor, device)

        with torch.inference_mode():
            action_tensor = policy.select_action(obs)
        action = action_tensor.detach().cpu().numpy().squeeze()

        # Check violations on raw normalized output
        step_info = check_step_violations(action, last_action)
        step_info["step"] = step
        step_info["any_violation"] = step_info["bounds_violated"] or step_info["velocity_violated"]

        per_step.append(step_info)
        last_action = action.copy()

        # Step env with zeros (we care about model output patterns, not task success)
        raw_obs, _, _, _ = env.step(np.zeros(ACTION_DIM))

        if (step + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            n_viol = sum(1 for s in per_step if s["any_violation"])
            logger.info("    Step %d/%d (%.1fs, violations so far: %d)",
                        step + 1, max_steps, elapsed, n_viol)

    elapsed = time.perf_counter() - t_start
    logger.info("    Episode done in %.1fs", elapsed)

    # Compute sliding window rates
    any_violations = [s["any_violation"] for s in per_step]
    bounds_violations = [s["bounds_violated"] for s in per_step]
    velocity_violations = [s["velocity_violated"] for s in per_step]

    sliding_any = compute_sliding_window_rate(any_violations, SLIDING_WINDOW)
    sliding_bounds = compute_sliding_window_rate(bounds_violations, SLIDING_WINDOW)
    sliding_velocity = compute_sliding_window_rate(velocity_violations, SLIDING_WINDOW)

    return {
        "per_step": per_step,
        "sliding_window": {
            "window_size": SLIDING_WINDOW,
            "any_violation_rate": sliding_any,
            "bounds_violation_rate": sliding_bounds,
            "velocity_violation_rate": sliding_velocity,
        },
        "summary": {
            "total_steps": max_steps,
            "total_violations": sum(any_violations),
            "total_bounds_violations": sum(bounds_violations),
            "total_velocity_violations": sum(velocity_violations),
            "violation_rate": sum(any_violations) / max_steps,
            "elapsed_seconds": round(elapsed, 1),
        },
    }


def main(device: str = "cpu", model_id: str = "HuggingFaceVLA/smolvla_libero"):
    import torch

    logger.info("=== Temporal Violation Evolution Experiment ===")
    logger.info("Device: %s, Steps per episode: %d, Sliding window: %d",
                device, EPISODE_STEPS, SLIDING_WINDOW)

    # Load policy once
    policy, processor, torch_device = load_smolvla_policy(model_id, device)

    results = {
        "experiment": "Temporal Violation Evolution",
        "description": (
            "Tracks per-step violation status across episode phases to show "
            "how violation patterns evolve over time within an episode."
        ),
        "config": {
            "model_id": model_id,
            "device": device,
            "episode_steps": EPISODE_STEPS,
            "sliding_window": SLIDING_WINDOW,
            "action_range": [-1.0, 1.0],
            "velocity_max": 0.15,
        },
        "tasks": {},
    }

    for task_cfg in TASKS:
        suite_name = task_cfg["suite"]
        task_id = task_cfg["task_id"]
        label = task_cfg["label"]

        logger.info("\n--- Task: %s (suite=%s, id=%d) ---", label, suite_name, task_id)

        env, suite, task = create_libero_env(suite_name, task_id)
        task_description = task.language
        logger.info("Task description: '%s'", task_description)

        # Load init states
        from libero.libero import get_libero_path
        init_states_path = (
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        init_states = torch.load(init_states_path, weights_only=False)
        logger.info("Loaded %d init states", len(init_states))

        # Run single episode per task (50 steps each)
        episode_data = run_temporal_episode(
            env=env,
            task_description=task_description,
            policy=policy,
            processor=processor,
            device=torch_device,
            max_steps=EPISODE_STEPS,
            init_states=init_states,
            init_state_id=0,
        )

        results["tasks"][label] = {
            "suite": suite_name,
            "task_id": task_id,
            "task_description": task_description,
            **episode_data,
        }

        # Print phase summary
        per_step = episode_data["per_step"]
        n = len(per_step)
        thirds = [per_step[:n // 3], per_step[n // 3:2 * n // 3], per_step[2 * n // 3:]]
        phase_names = ["early (approach)", "mid (contact)", "late (manipulation)"]

        logger.info("  Phase breakdown for %s:", label)
        for phase_name, phase_steps in zip(phase_names, thirds):
            n_any = sum(1 for s in phase_steps if s["any_violation"])
            n_bounds = sum(1 for s in phase_steps if s["bounds_violated"])
            n_vel = sum(1 for s in phase_steps if s["velocity_violated"])
            logger.info("    %s: %d/%d violations (bounds=%d, velocity=%d)",
                        phase_name, n_any, len(phase_steps), n_bounds, n_vel)

        env.close()

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Print summary table
    print(f"\n{'='*70}")
    print("TEMPORAL VIOLATION EVOLUTION RESULTS")
    print(f"{'='*70}")
    for label, data in results["tasks"].items():
        s = data["summary"]
        print(f"\nTask: {data['task_description']} ({label})")
        print(f"  Total steps: {s['total_steps']}")
        print(f"  Total violations: {s['total_violations']} ({s['violation_rate']:.1%})")
        print(f"  Bounds violations: {s['total_bounds_violations']}")
        print(f"  Velocity violations: {s['total_velocity_violations']}")

        # Phase breakdown
        per_step = data["per_step"]
        n = len(per_step)
        phases = {
            "early (0-{})".format(n // 3 - 1): per_step[:n // 3],
            "mid ({}-{})".format(n // 3, 2 * n // 3 - 1): per_step[n // 3:2 * n // 3],
            "late ({}-{})".format(2 * n // 3, n - 1): per_step[2 * n // 3:],
        }
        print(f"  {'Phase':<22} {'Violations':>11} {'Bounds':>8} {'Velocity':>10} {'Rate':>6}")
        print(f"  {'-'*57}")
        for phase_name, phase_steps in phases.items():
            n_any = sum(1 for s in phase_steps if s["any_violation"])
            n_bounds = sum(1 for s in phase_steps if s["bounds_violated"])
            n_vel = sum(1 for s in phase_steps if s["velocity_violated"])
            rate = n_any / len(phase_steps) if phase_steps else 0
            print(f"  {phase_name:<22} {n_any:>11} {n_bounds:>8} {n_vel:>10} {rate:>5.0%}")

    # Save
    out_path = RESULTS_DIR / "exp_temporal_evolution.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Temporal violation evolution experiment")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-id", type=str, default="HuggingFaceVLA/smolvla_libero")
    args = parser.parse_args()
    main(device=args.device, model_id=args.model_id)
