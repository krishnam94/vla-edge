"""Closed-loop LIBERO evaluation: SmolVLA with and without SafeContract.

Loads the HuggingFaceVLA/smolvla_libero checkpoint (finetuned for LIBERO),
runs episodes in the LIBERO simulator, and compares:
  - Raw SmolVLA (no safety wrapper)
  - SafeContract-wrapped SmolVLA (action_range + velocity clamping)

Metrics: task success rate, violation count, mean trajectory length, mean reward.

Usage:
  python experiments/icra_ws_2026/exp_closed_loop.py [--n-episodes 5] [--device cpu] [--suite libero_object] [--task-id 0]

Requirements:
  pip install robosuite==1.4.1 mujoco libero future easydict
  pip install vla-edge[smolvla]

Known issues:
  - MuJoCo rendering on headless needs MUJOCO_GL=egl or osmesa
  - numpy <2.0 may be pulled by robosuite; ensure lerobot compat
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

# Force unbuffered output for background/piped execution
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# LIBERO constants (from lerobot/envs/libero.py)
ACTION_DIM = 7
MAX_EPISODE_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    success: bool
    steps: int
    total_reward: float
    violations: int
    violation_details: list[str] = field(default_factory=list)
    avg_action_magnitude: float = 0.0
    max_action_magnitude: float = 0.0


@dataclass
class ConditionResult:
    """Aggregated results for one condition (raw vs safe)."""
    condition: str
    episodes: list[EpisodeResult]
    success_rate: float = 0.0
    mean_steps: float = 0.0
    mean_reward: float = 0.0
    total_violations: int = 0
    mean_action_magnitude: float = 0.0

    def compute(self) -> None:
        n = len(self.episodes)
        if n == 0:
            return
        self.success_rate = sum(e.success for e in self.episodes) / n
        self.mean_steps = sum(e.steps for e in self.episodes) / n
        self.mean_reward = sum(e.total_reward for e in self.episodes) / n
        self.total_violations = sum(e.violations for e in self.episodes)
        self.mean_action_magnitude = sum(e.avg_action_magnitude for e in self.episodes) / n


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
    """Load SmolVLA policy directly (not through adapter, for dual-camera support)."""
    import torch

    # Import SmolVLA bypassing lerobot.policies.__init__
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

    # Force to the requested device (from_pretrained may auto-detect MPS on Mac)
    torch_device = torch.device(device)
    policy.to(torch_device)
    # Also override config.device so internal methods use the correct device
    policy.config.device = torch_device
    policy.eval()

    # Load tokenizer
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)

    logger.info("SmolVLA loaded on %s. Action dim: %d, chunk size: %d",
                torch_device,
                policy.config.output_features["action"].shape[0],
                policy.config.chunk_size)
    return policy, processor, torch_device


def build_observation(
    raw_obs: dict,
    task_description: str,
    policy,
    processor,
    device,
) -> dict:
    """Convert LIBERO raw observation to SmolVLA input format."""
    import torch

    observation = {}

    # Camera images: LIBERO gives (H, W, 3) uint8
    # SmolVLA expects (B, C, H, W) float [0, 1]
    agentview = raw_obs["agentview_image"]  # (256, 256, 3)
    wrist = raw_obs["robot0_eye_in_hand_image"]  # (256, 256, 3)

    for img_arr, key in [
        (agentview, "observation.images.image"),
        (wrist, "observation.images.image2"),
    ]:
        img_f = img_arr.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
        observation[key] = img_t.to(device)

    # State: LIBERO has joint_pos(7) + gripper_qpos(2) = 9 dims
    # The checkpoint expects 8 dims. Use joint_pos(7) + gripper(1).
    joint_pos = raw_obs.get("robot0_joint_pos", np.zeros(7))
    gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
    # Use first gripper value as single gripper state
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


class ViolationTracker:
    """Track safety violations during an episode."""

    def __init__(
        self,
        action_range: tuple[float, float] = (-1.0, 1.0),
        velocity_max: float = 0.15,
    ):
        self.action_range = action_range
        self.velocity_max = velocity_max
        self.violations: list[str] = []
        self.last_action: np.ndarray | None = None

    def check(self, action: np.ndarray) -> list[str]:
        """Check action for violations. Returns list of violation descriptions."""
        step_violations = []

        # Range check
        if np.any(action < self.action_range[0]) or np.any(action > self.action_range[1]):
            step_violations.append(
                f"range: [{action.min():.3f}, {action.max():.3f}] outside [{self.action_range[0]}, {self.action_range[1]}]"
            )

        # Velocity check
        if self.last_action is not None:
            delta = np.abs(action - self.last_action)
            if np.any(delta > self.velocity_max):
                step_violations.append(
                    f"velocity: max_delta={delta.max():.3f} > {self.velocity_max}"
                )

        self.last_action = action.copy()
        self.violations.extend(step_violations)
        return step_violations

    def reset(self):
        self.violations.clear()
        self.last_action = None


def unnormalize_action(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Unnormalize MEAN_STD normalized actions: action * std + mean."""
    return action * std + mean


# Dataset stats from HuggingFaceVLA/smol-libero (5000 samples)
LIBERO_ACTION_MEAN = np.array([0.007, 0.0836, -0.0395, 0.0005, 0.0032, -0.0014, 0.0064], dtype=np.float32)
LIBERO_ACTION_STD = np.array([0.2963, 0.4462, 0.4706, 0.031, 0.0483, 0.0433, 1.0], dtype=np.float32)


def apply_safety_contract(action: np.ndarray, last_action: np.ndarray | None,
                          action_range: tuple[float, float] = (-1.0, 1.0),
                          velocity_max: float = 0.15) -> np.ndarray:
    """Apply SafeContract-style clipping to an action."""
    # 1. Range clip
    action = np.clip(action, action_range[0], action_range[1])

    # 2. Velocity clip
    if last_action is not None:
        delta = action - last_action
        delta = np.clip(delta, -velocity_max, velocity_max)
        action = last_action + delta
        # Re-clip after velocity
        action = np.clip(action, action_range[0], action_range[1])

    return action


def run_episode(
    env,
    suite,
    task_id: int,
    task_description: str,
    policy,
    processor,
    device,
    use_safety: bool,
    max_steps: int,
    action_range: tuple[float, float] = (-1.0, 1.0),
    velocity_max: float = 0.15,
    init_states=None,
    init_state_id: int = 0,
) -> EpisodeResult:
    """Run a single closed-loop episode."""
    import torch

    raw_obs = env.reset()

    # Set deterministic init state
    if init_states is not None:
        raw_obs = env.set_init_state(init_states[init_state_id % len(init_states)])
        # Let objects settle
        for _ in range(10):
            raw_obs, _, _, _ = env.step(np.zeros(ACTION_DIM))

    tracker = ViolationTracker(action_range=action_range, velocity_max=velocity_max)
    last_safe_action = None
    total_reward = 0.0
    action_magnitudes = []

    # Reset policy action queue
    if hasattr(policy, "_queues"):
        from collections import deque
        for k in policy._queues:
            policy._queues[k] = deque(maxlen=policy._queues[k].maxlen)

    t_start = time.perf_counter()
    for step in range(max_steps):
        try:
            # Build observation
            obs = build_observation(raw_obs, task_description, policy, processor, device)

            # Get action from policy
            with torch.inference_mode():
                action_tensor = policy.select_action(obs)
            action = action_tensor.detach().cpu().numpy().squeeze()

            # Track violations on raw (normalized) model output
            tracker.check(action)

            # Unnormalize: model outputs MEAN_STD normalized actions
            # Must convert back to LIBERO's action space before env.step()
            action_unnorm = unnormalize_action(action, LIBERO_ACTION_MEAN, LIBERO_ACTION_STD)

            if step == 0:
                logger.info("    First action (normalized): shape=%s, range=[%.3f, %.3f]",
                            action.shape, action.min(), action.max())
                logger.info("    First action (unnorm): range=[%.3f, %.3f]",
                            action_unnorm.min(), action_unnorm.max())

            # Apply safety contract on unnormalized actions if enabled
            if use_safety:
                action_unnorm = apply_safety_contract(
                    action_unnorm, last_safe_action, action_range, velocity_max
                )
                last_safe_action = action_unnorm.copy()

            action_magnitudes.append(float(np.linalg.norm(action_unnorm)))

            # Step environment with unnormalized actions
            raw_obs, reward, done, info = env.step(action_unnorm)
            total_reward += reward

            # Progress logging every 50 steps
            if (step + 1) % 50 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info("    Step %d/%d (%.1fs elapsed, %.1f steps/s, violations=%d)",
                            step + 1, max_steps, elapsed, (step + 1) / elapsed, len(tracker.violations))

            # Check success
            is_success = env.check_success()
            if done or is_success:
                elapsed = time.perf_counter() - t_start
                logger.info("    Episode done at step %d (%.1fs, success=%s)", step + 1, elapsed, is_success)
                return EpisodeResult(
                    success=is_success,
                    steps=step + 1,
                    total_reward=total_reward,
                    violations=len(tracker.violations),
                    violation_details=tracker.violations[-10:],
                    avg_action_magnitude=float(np.mean(action_magnitudes)),
                    max_action_magnitude=float(np.max(action_magnitudes)),
                )
        except Exception as e:
            logger.error("    Error at step %d: %s", step, e)
            import traceback
            traceback.print_exc()
            return EpisodeResult(
                success=False,
                steps=step,
                total_reward=total_reward,
                violations=len(tracker.violations),
                violation_details=tracker.violations[-10:] + [f"ERROR: {e}"],
                avg_action_magnitude=float(np.mean(action_magnitudes)) if action_magnitudes else 0.0,
                max_action_magnitude=float(np.max(action_magnitudes)) if action_magnitudes else 0.0,
            )

    # Timed out
    elapsed = time.perf_counter() - t_start
    logger.info("    Episode timed out at %d steps (%.1fs)", max_steps, elapsed)
    return EpisodeResult(
        success=False,
        steps=max_steps,
        total_reward=total_reward,
        violations=len(tracker.violations),
        violation_details=tracker.violations[-10:],
        avg_action_magnitude=float(np.mean(action_magnitudes)),
        max_action_magnitude=float(np.max(action_magnitudes)),
    )


def main(
    n_episodes: int = 5,
    device: str = "cpu",
    suite_name: str = "libero_object",
    task_id: int = 0,
    model_id: str = "HuggingFaceVLA/smolvla_libero",
    velocity_max: float = 0.15,
):
    import torch

    logger.info("=== Closed-Loop LIBERO Evaluation ===")
    logger.info("Suite: %s, Task: %d, Episodes: %d, Device: %s", suite_name, task_id, n_episodes, device)

    # Create environment
    env, suite, task = create_libero_env(suite_name, task_id)
    task_description = task.language
    max_steps = MAX_EPISODE_STEPS.get(suite_name, 300)
    logger.info("Task: '%s', Max steps: %d", task_description, max_steps)

    # Load init states for deterministic resets
    from libero.libero import get_libero_path

    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)
    logger.info("Loaded %d init states", len(init_states))

    # Load policy
    policy, processor, torch_device = load_smolvla_policy(model_id, device)

    # Run episodes: raw (no safety)
    logger.info("\n--- Condition: RAW (no safety contract) ---")
    raw_results = ConditionResult(condition="raw", episodes=[])
    for ep in range(n_episodes):
        logger.info("  Episode %d/%d...", ep + 1, n_episodes)
        result = run_episode(
            env, suite, task_id, task_description,
            policy, processor, torch_device,
            use_safety=False,
            max_steps=max_steps,
            velocity_max=velocity_max,
            init_states=init_states,
            init_state_id=ep,
        )
        raw_results.episodes.append(result)
        logger.info("    success=%s, steps=%d, violations=%d, reward=%.2f",
                     result.success, result.steps, result.violations, result.total_reward)
    raw_results.compute()

    # Run episodes: SafeContract
    logger.info("\n--- Condition: SAFE (SafeContract wrapped) ---")
    safe_results = ConditionResult(condition="safe", episodes=[])
    for ep in range(n_episodes):
        logger.info("  Episode %d/%d...", ep + 1, n_episodes)
        result = run_episode(
            env, suite, task_id, task_description,
            policy, processor, torch_device,
            use_safety=True,
            max_steps=max_steps,
            velocity_max=velocity_max,
            init_states=init_states,
            init_state_id=ep,
        )
        safe_results.episodes.append(result)
        logger.info("    success=%s, steps=%d, violations=%d, reward=%.2f",
                     result.success, result.steps, result.violations, result.total_reward)
    safe_results.compute()

    # Compile results
    results = {
        "experiment": "Closed-Loop LIBERO: SmolVLA with SafeContract",
        "config": {
            "suite": suite_name,
            "task_id": task_id,
            "task_description": task_description,
            "model_id": model_id,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "device": device,
            "velocity_max": velocity_max,
            "action_range": [-1.0, 1.0],
        },
        "raw": {
            "success_rate": round(raw_results.success_rate, 3),
            "mean_steps": round(raw_results.mean_steps, 1),
            "mean_reward": round(raw_results.mean_reward, 3),
            "total_violations": raw_results.total_violations,
            "mean_action_magnitude": round(raw_results.mean_action_magnitude, 4),
            "episodes": [asdict(e) for e in raw_results.episodes],
        },
        "safe": {
            "success_rate": round(safe_results.success_rate, 3),
            "mean_steps": round(safe_results.mean_steps, 1),
            "mean_reward": round(safe_results.mean_reward, 3),
            "total_violations": safe_results.total_violations,
            "mean_action_magnitude": round(safe_results.mean_action_magnitude, 4),
            "episodes": [asdict(e) for e in safe_results.episodes],
        },
        "comparison": {
            "success_rate_delta": round(safe_results.success_rate - raw_results.success_rate, 3),
            "violation_reduction": raw_results.total_violations - safe_results.total_violations,
            "mean_steps_delta": round(safe_results.mean_steps - raw_results.mean_steps, 1),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save
    path = RESULTS_DIR / "exp_closed_loop.json"
    path.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*60}")
    print(f"CLOSED-LOOP LIBERO RESULTS")
    print(f"{'='*60}")
    print(f"Task: {task_description}")
    print(f"Episodes: {n_episodes}")
    print(f"")
    print(f"{'Condition':<12} {'Success':>8} {'Avg Steps':>10} {'Violations':>11} {'Avg Reward':>11}")
    print(f"{'-'*52}")
    print(f"{'Raw':<12} {raw_results.success_rate:>7.1%} {raw_results.mean_steps:>10.1f} {raw_results.total_violations:>11d} {raw_results.mean_reward:>11.3f}")
    print(f"{'SafeContract':<12} {safe_results.success_rate:>7.1%} {safe_results.mean_steps:>10.1f} {safe_results.total_violations:>11d} {safe_results.mean_reward:>11.3f}")
    print(f"")
    print(f"Success rate delta: {results['comparison']['success_rate_delta']:+.1%}")
    print(f"Violation reduction: {results['comparison']['violation_reduction']}")
    print(f"")
    print(f"Results saved to {path}")

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop LIBERO evaluation")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--model-id", type=str, default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--velocity-max", type=float, default=0.15)
    args = parser.parse_args()

    main(
        n_episodes=args.n_episodes,
        device=args.device,
        suite_name=args.suite,
        task_id=args.task_id,
        model_id=args.model_id,
        velocity_max=args.velocity_max,
    )
