"""Closed-loop LIBERO evaluation V2: SmolVLA with fixed preprocessing.

Fixes 3 preprocessing bugs from V1 (EXP-SMOLVLA-CL-V2):
  Bug 1 - State: Use eef_pos(3) + quat_to_axis_angle(eef_quat)(3) + gripper_qpos(2) = 8 dims
           (V1 used joint_pos(7) + gripper(1) which doesn't match training data)
  Bug 2 - State normalization: Apply MEAN_STD from dataset stats before feeding to model
           (V1 had no state normalization)
  Bug 3 - Image flip: Rotate images 180 degrees (flip H and W) to match LIBERO camera convention
           (V1 fed raw images without flipping)

All three fixes match the official lerobot LiberoProcessorStep + NormalizerProcessorStep pipeline.

Usage:
  python experiments/icra_ws_2026/exp_closed_loop_v2.py [--n-episodes 10] [--device mps] [--suite libero_object] [--task-id 0]

Requirements:
  pip install robosuite==1.4.1 mujoco libero future easydict
  pip install vla-edge[smolvla]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

# Force unbuffered output for background/piped execution
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# LIBERO constants
ACTION_DIM = 7
MAX_EPISODE_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# ------------------------------------------------------------------
# Dataset stats from HuggingFaceVLA/smol-libero (full 13021 samples)
# observation.state = [eef_pos(3), axis_angle(3), gripper_qpos(2)]
# ------------------------------------------------------------------
STATE_MEAN = np.array([
    0.05755217, 0.04148761, 0.59735191,   # eef_pos
    3.13184023, -0.10081545, -0.09122046,  # axis_angle
    0.02927421, -0.02970476,               # gripper_qpos
], dtype=np.float32)

STATE_STD = np.array([
    0.05567992, 0.16069064, 0.08197999,   # eef_pos
    0.06108680, 0.20972775, 0.18352441,    # axis_angle
    0.00926126, 0.00916287,                # gripper_qpos
], dtype=np.float32)

# action = [delta_eef(3), delta_rotation(3), gripper_action(1)]
ACTION_MEAN = np.array([
    0.00818362, 0.08518664, -0.04004815,
    -0.00016292, 0.00302347, -0.00026142,
    -0.01451501,
], dtype=np.float32)

ACTION_STD = np.array([
    0.29792726, 0.44671473, 0.47144267,
    0.03022212, 0.05201731, 0.04574548,
    0.99989468,
], dtype=np.float32)

NORM_EPS = 1e-8


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
    """Load SmolVLA policy."""
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

    logger.info("SmolVLA loaded on %s. Action dim: %d, chunk size: %d, n_action_steps: %d",
                torch_device,
                policy.config.output_features["action"].shape[0],
                policy.config.chunk_size,
                policy.config.n_action_steps)
    return policy, processor, torch_device


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle representation.

    Matches lerobot's LiberoProcessorStep._quat2axisangle exactly.
    """
    # quat is (4,) in [x, y, z, w] format (LIBERO/scipy convention)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))

    if den > 1e-10:
        angle = 2.0 * np.arccos(w)
        axis = quat[:3] / den
        return (axis * angle).astype(np.float32)
    else:
        return np.zeros(3, dtype=np.float32)


def build_observation(
    raw_obs: dict,
    task_description: str,
    policy,
    processor,
    device,
) -> dict:
    """Convert LIBERO raw observation to SmolVLA input format.

    Implements the same pipeline as lerobot's:
      1. preprocess_observation() - convert images to (B, C, H, W) float [0, 1]
      2. LiberoProcessorStep - flip images 180, build state from eef_pos + axis_angle + gripper
      3. NormalizerProcessorStep - apply MEAN_STD normalization to state
    """
    import torch

    observation = {}

    # ---- Images ----
    # LIBERO gives (H, W, 3) uint8
    # SmolVLA expects (B, C, H, W) float [0, 1]
    agentview = raw_obs["agentview_image"]            # (256, 256, 3)
    wrist = raw_obs["robot0_eye_in_hand_image"]       # (256, 256, 3)

    for img_arr, key in [
        (agentview, "observation.images.image"),
        (wrist, "observation.images.image2"),
    ]:
        img_f = img_arr.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)

        # BUG FIX 3: Flip both H and W (180 degree rotation)
        # Matches LiberoProcessorStep._process_observation: torch.flip(img, dims=[2, 3])
        img_t = torch.flip(img_t, dims=[2, 3])

        observation[key] = img_t.to(device)

    # ---- State (BUG FIX 1) ----
    # Use eef_pos(3) + axis_angle(3) + gripper_qpos(2) = 8 dims
    # NOT joint_pos(7) + gripper(1) like V1
    eef_pos = raw_obs["robot0_eef_pos"]          # (3,) end-effector position
    eef_quat = raw_obs["robot0_eef_quat"]        # (4,) quaternion [x, y, z, w]
    gripper_qpos = raw_obs["robot0_gripper_qpos"] # (2,)

    axis_angle = quat_to_axis_angle(eef_quat)    # (3,)
    state = np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32)  # (8,)

    # BUG FIX 2: Normalize state with MEAN_STD
    # Matches NormalizerProcessorStep: normalized = (state - mean) / (std + eps)
    state_normalized = (state - STATE_MEAN) / (STATE_STD + NORM_EPS)

    observation["observation.state"] = torch.from_numpy(state_normalized).unsqueeze(0).to(device)

    # ---- Language tokens ----
    # SmolVLA expects task description with trailing newline
    task_with_newline = task_description if task_description.endswith("\n") else task_description + "\n"
    tokens = processor.tokenizer(
        task_with_newline,
        return_tensors="pt",
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
    )
    observation["observation.language.tokens"] = tokens["input_ids"].to(device)
    observation["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(device)

    return observation


def unnormalize_action(action: np.ndarray) -> np.ndarray:
    """Unnormalize MEAN_STD action: action * std + mean.

    Matches UnnormalizerProcessorStep._apply_transform with inverse=True.
    """
    return action * ACTION_STD + ACTION_MEAN


class ViolationTracker:
    """Track safety violations during an episode."""

    def __init__(
        self,
        action_range: tuple[float, float] = (-1.0, 1.0),
        velocity_max: float = 0.1,
    ):
        self.action_range = action_range
        self.velocity_max = velocity_max
        self.violations: list[str] = []
        self.last_action: np.ndarray | None = None

    def check(self, action: np.ndarray) -> list[str]:
        step_violations = []

        if np.any(action < self.action_range[0]) or np.any(action > self.action_range[1]):
            step_violations.append(
                f"range: [{action.min():.3f}, {action.max():.3f}] outside [{self.action_range[0]}, {self.action_range[1]}]"
            )

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


def apply_safety_contract(action: np.ndarray, last_action: np.ndarray | None,
                          action_range: tuple[float, float] = (-1.0, 1.0),
                          velocity_max: float = 0.1) -> np.ndarray:
    """Apply SafeContract-style clipping to an action."""
    action = np.clip(action, action_range[0], action_range[1])

    if last_action is not None:
        delta = action - last_action
        delta = np.clip(delta, -velocity_max, velocity_max)
        action = last_action + delta
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
    velocity_max: float = 0.1,
    init_states=None,
    init_state_id: int = 0,
) -> EpisodeResult:
    """Run a single closed-loop episode."""
    import torch

    raw_obs = env.reset()

    if init_states is not None:
        raw_obs = env.set_init_state(init_states[init_state_id % len(init_states)])
        # Let objects settle (10 no-op steps with gripper open, matching lerobot convention)
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
        for _ in range(10):
            raw_obs, _, _, _ = env.step(dummy_action)

    tracker = ViolationTracker(action_range=action_range, velocity_max=velocity_max)
    last_safe_action = None
    total_reward = 0.0
    action_magnitudes = []

    # Reset policy action queue
    if hasattr(policy, "_queues"):
        for k in policy._queues:
            policy._queues[k] = deque(maxlen=policy._queues[k].maxlen)

    t_start = time.perf_counter()
    for step in range(max_steps):
        try:
            obs = build_observation(raw_obs, task_description, policy, processor, device)

            with torch.inference_mode():
                action_tensor = policy.select_action(obs)
            action = action_tensor.detach().cpu().numpy().squeeze()

            # Unnormalize: model outputs MEAN_STD normalized actions
            action_unnorm = unnormalize_action(action)

            # Track violations on unnormalized (env-space) actions
            tracker.check(action_unnorm)

            if step == 0:
                logger.info("    First action (normalized): shape=%s, range=[%.3f, %.3f]",
                            action.shape, action.min(), action.max())
                logger.info("    First action (unnorm): range=[%.3f, %.3f]",
                            action_unnorm.min(), action_unnorm.max())
                # Log state for debugging
                eef_pos = raw_obs["robot0_eef_pos"]
                logger.info("    Initial eef_pos: [%.4f, %.4f, %.4f]", *eef_pos)

            if use_safety:
                action_unnorm = apply_safety_contract(
                    action_unnorm, last_safe_action, action_range, velocity_max
                )
                last_safe_action = action_unnorm.copy()

            action_magnitudes.append(float(np.linalg.norm(action_unnorm)))

            raw_obs, reward, done, info = env.step(action_unnorm)
            total_reward += reward

            if (step + 1) % 50 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info("    Step %d/%d (%.1fs elapsed, %.1f steps/s, violations=%d)",
                            step + 1, max_steps, elapsed, (step + 1) / elapsed, len(tracker.violations))

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
    n_episodes: int = 10,
    device: str = "mps",
    suite_name: str = "libero_object",
    task_id: int = 0,
    model_id: str = "HuggingFaceVLA/smolvla_libero",
    velocity_max: float = 0.1,
    sanity_check: bool = False,
):
    import torch

    if sanity_check:
        n_episodes = 1
        logger.info("=== SANITY CHECK MODE: 1 episode only ===")

    logger.info("=== Closed-Loop LIBERO Evaluation V2 (Fixed Preprocessing) ===")
    logger.info("Suite: %s, Task: %d, Episodes: %d, Device: %s", suite_name, task_id, n_episodes, device)
    logger.info("Fixes applied: state_repr(eef+axisangle+gripper), state_normalization(MEAN_STD), image_flip(180)")

    # Create environment
    env, suite, task = create_libero_env(suite_name, task_id)
    task_description = task.language
    max_steps = MAX_EPISODE_STEPS.get(suite_name, 300)
    logger.info("Task: '%s', Max steps: %d", task_description, max_steps)

    # Load init states
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

    if sanity_check:
        # Print quick summary and exit
        print(f"\n{'='*60}")
        print(f"SANITY CHECK RESULT")
        print(f"{'='*60}")
        print(f"Task: {task_description}")
        print(f"Success: {raw_results.episodes[0].success}")
        print(f"Steps: {raw_results.episodes[0].steps}")
        print(f"Violations: {raw_results.episodes[0].violations}")
        print(f"Reward: {raw_results.episodes[0].total_reward:.3f}")
        print(f"Avg action magnitude: {raw_results.episodes[0].avg_action_magnitude:.4f}")
        env.close()
        return {"sanity_check": True, "success": raw_results.episodes[0].success}

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
        "experiment": "Closed-Loop LIBERO V2: SmolVLA with Fixed Preprocessing + SafeContract",
        "experiment_id": "EXP-SMOLVLA-CL-V2",
        "fixes": [
            "state_representation: eef_pos(3) + axis_angle(3) + gripper_qpos(2) instead of joint_pos(7) + gripper(1)",
            "state_normalization: MEAN_STD from HuggingFaceVLA/smol-libero dataset",
            "image_flip: 180-degree rotation (flip H and W) to match training convention",
        ],
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
        "normalization_stats": {
            "state_mean": STATE_MEAN.tolist(),
            "state_std": STATE_STD.tolist(),
            "action_mean": ACTION_MEAN.tolist(),
            "action_std": ACTION_STD.tolist(),
            "source": "HuggingFaceVLA/smol-libero (full 13021 samples)",
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
        "v1_comparison": {
            "v1_raw_success_rate": 0.0,
            "v1_raw_violations_per_episode": 525,
            "note": "V1 had 0% success and ~525 violations/episode due to preprocessing bugs",
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save
    path = RESULTS_DIR / "exp_closed_loop_v2.json"
    path.write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*60}")
    print(f"CLOSED-LOOP LIBERO V2 RESULTS (Fixed Preprocessing)")
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
    print(f"V1 comparison: was 0% success, ~525 violations/ep")
    print(f"Results saved to {path}")

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop LIBERO evaluation V2 (fixed preprocessing)")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--model-id", type=str, default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--velocity-max", type=float, default=0.1)
    parser.add_argument("--sanity-check", action="store_true", help="Run 1 episode only (quick test)")
    args = parser.parse_args()

    main(
        n_episodes=args.n_episodes,
        device=args.device,
        suite_name=args.suite,
        task_id=args.task_id,
        model_id=args.model_id,
        velocity_max=args.velocity_max,
        sanity_check=args.sanity_check,
    )
