"""Oracle closed-loop experiment: GT action replay with SafeContract.

Answers: "Does SafeContract clipping degrade task success?"

Instead of relying on a broken SmolVLA checkpoint, we use ground-truth
actions from the smol-libero dataset as a scripted oracle policy. These
GT actions achieved success when originally collected, so replaying them
in LIBERO sim should yield high success rates at noise=0.

Protocol:
  1. Load GT actions from HuggingFaceVLA/smol-libero, grouped by episode
  2. For each noise level sigma in {0, 0.05, 0.1, 0.2, 0.5}:
     a. Add Gaussian noise: noisy = gt + N(0, sigma) in normalized space
     b. Run WITH SafeContract (bounds [-1,1], v_max=0.1)
     c. Run WITHOUT SafeContract (raw noisy actions)
     d. Replay in LIBERO sim, measure success + reward + violations
  3. Compare metrics to quantify SafeContract's impact

Key detail: smol-libero actions are MEAN_STD normalized in [-1,1].
Before env.step(), we unnormalize: action_unnorm = action * std + mean.
SafeContract operates in normalized space (bounds [-1,1], v_max=0.1),
matching all other paper experiments.

Known limitation: LIBERO uses sparse reward (1.0 only on task completion).
Open-loop replay of GT actions may not achieve success because:
  - smol-libero episodes may span multiple LIBERO tasks
  - Open-loop replay has no feedback correction (compounding errors)
The key metric is the COMPARATIVE delta between safety=on and safety=off
at each noise level, not absolute success rate.

Usage:
  # Quick sanity check (1 episode, sigma=0)
  python experiments/icra_ws_2026/exp_oracle_closed_loop.py --sanity

  # Full experiment
  python experiments/icra_ws_2026/exp_oracle_closed_loop.py --n-episodes 8

  # Custom noise levels
  python experiments/icra_ws_2026/exp_oracle_closed_loop.py --noise-levels 0.0 0.1 0.5

Requirements:
  pip install robosuite==1.4.1 mujoco libero future easydict datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

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

# Dataset stats from HuggingFaceVLA/smol-libero (used for unnormalization)
LIBERO_ACTION_MEAN = np.array(
    [0.007, 0.0836, -0.0395, 0.0005, 0.0032, -0.0014, 0.0064],
    dtype=np.float32,
)
LIBERO_ACTION_STD = np.array(
    [0.2963, 0.4462, 0.4706, 0.031, 0.0483, 0.0433, 1.0],
    dtype=np.float32,
)

# SafeContract parameters (consistent with EXP-B, EXP-G, EXP-NOISE)
BOUNDS = (-1.0, 1.0)
V_MAX = 0.1
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.5]
SEED = 42


@dataclass
class EpisodeResult:
    """Results from a single episode replay."""

    success: bool
    steps: int
    total_reward: float
    violations_detected: int
    bounds_violations: int
    velocity_violations: int
    actions_modified: int
    total_actions: int
    mean_clip_magnitude: float = 0.0
    max_action_magnitude: float = 0.0


@dataclass
class ConditionResult:
    """Aggregated results for one condition."""

    noise_level: float
    safety_on: bool
    episodes: list[EpisodeResult] = field(default_factory=list)
    success_rate: float = 0.0
    mean_steps: float = 0.0
    mean_reward: float = 0.0
    total_violations: int = 0
    total_bounds_violations: int = 0
    total_velocity_violations: int = 0
    total_actions_modified: int = 0
    total_actions: int = 0
    pct_modified: float = 0.0

    def compute(self) -> None:
        n = len(self.episodes)
        if n == 0:
            return
        self.success_rate = sum(e.success for e in self.episodes) / n
        self.mean_steps = sum(e.steps for e in self.episodes) / n
        self.mean_reward = sum(e.total_reward for e in self.episodes) / n
        self.total_violations = sum(e.violations_detected for e in self.episodes)
        self.total_bounds_violations = sum(e.bounds_violations for e in self.episodes)
        self.total_velocity_violations = sum(e.velocity_violations for e in self.episodes)
        self.total_actions_modified = sum(e.actions_modified for e in self.episodes)
        self.total_actions = sum(e.total_actions for e in self.episodes)
        self.pct_modified = (
            self.total_actions_modified / self.total_actions * 100 if self.total_actions > 0 else 0.0
        )


def unnormalize_action(action: np.ndarray) -> np.ndarray:
    """Unnormalize MEAN_STD normalized action for env.step()."""
    return action * LIBERO_ACTION_STD + LIBERO_ACTION_MEAN


def apply_safecontract(
    action: np.ndarray,
    last_action: np.ndarray | None,
    lo: float = -1.0,
    hi: float = 1.0,
    v_max: float = 0.1,
) -> np.ndarray:
    """Apply SafeContract: clip bounds + velocity clamp + re-clip."""
    clipped = np.clip(action, lo, hi)
    if last_action is not None:
        delta = clipped - last_action
        delta = np.clip(delta, -v_max, v_max)
        clipped = last_action + delta
        clipped = np.clip(clipped, lo, hi)
    return clipped


def count_step_violations(
    action: np.ndarray,
    last_action: np.ndarray | None,
    lo: float = -1.0,
    hi: float = 1.0,
    v_max: float = 0.1,
) -> tuple[int, int]:
    """Count bounds and velocity violations for a single step.

    Returns (bounds_violations, velocity_violations) as element counts.
    """
    bounds_v = int(np.sum((action < lo) | (action > hi)))
    vel_v = 0
    if last_action is not None:
        delta = np.abs(action - last_action)
        vel_v = int(np.sum(delta > v_max + 1e-6))
    return bounds_v, vel_v


def create_libero_env(suite_name: str, task_id: int):
    """Create a single LIBERO environment."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()
    suite = bench[suite_name]()
    task = suite.get_task(task_id)

    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    return env, suite, task


def load_gt_episodes(n_episodes: int) -> list[np.ndarray]:
    """Load ground-truth episodes from smol-libero.

    Returns list of (T, 7) arrays in normalized space [-1, 1].
    """
    from datasets import load_dataset

    logger.info("Loading HuggingFaceVLA/smol-libero dataset...")
    ds = load_dataset("HuggingFaceVLA/smol-libero", split="train")

    ep_indices = np.array(ds["episode_index"])
    actions_all = np.array(ds["action"])
    unique_eps = np.unique(ep_indices)

    episodes = []
    for ep_id in unique_eps[:n_episodes]:
        mask = ep_indices == ep_id
        ep_actions = actions_all[mask]
        episodes.append(ep_actions)

    logger.info(
        "Loaded %d episodes, lengths: %s",
        len(episodes),
        [len(ep) for ep in episodes],
    )
    for i, ep in enumerate(episodes):
        logger.info(
            "  Episode %d: shape=%s, range=[%.4f, %.4f]",
            i,
            ep.shape,
            ep.min(),
            ep.max(),
        )

    return episodes


def run_oracle_episode(
    env,
    gt_actions_norm: np.ndarray,
    noise_sigma: float,
    use_safety: bool,
    rng: np.random.Generator,
    init_states=None,
    init_state_id: int = 0,
) -> EpisodeResult:
    """Run one oracle episode: replay GT actions with optional noise and safety.

    Args:
        env: LIBERO environment.
        gt_actions_norm: (T, 7) normalized GT actions.
        noise_sigma: Gaussian noise std to add in normalized space.
        use_safety: Whether to apply SafeContract before env.step().
        rng: Random number generator for noise.
        init_states: Deterministic init states for env reset.
        init_state_id: Which init state to use.

    Returns:
        EpisodeResult with success, violations, modifications, etc.
    """
    env.reset()

    # Set deterministic init state
    if init_states is not None:
        env.set_init_state(init_states[init_state_id % len(init_states)])
        # Let objects settle
        for _ in range(5):
            env.step(np.zeros(ACTION_DIM))

    total_reward = 0.0
    total_bounds_v = 0
    total_vel_v = 0
    actions_modified = 0
    clip_magnitudes = []
    last_safe_action = None
    last_noisy_action = None

    n_steps = len(gt_actions_norm)

    for t in range(n_steps):
        gt_action = gt_actions_norm[t]

        # Add noise in normalized space
        if noise_sigma > 0:
            noise = rng.normal(0, noise_sigma, gt_action.shape).astype(gt_action.dtype)
            noisy_action = gt_action + noise
        else:
            noisy_action = gt_action.copy()

        # Count violations on noisy action (pre-safety)
        bv, vv = count_step_violations(noisy_action, last_noisy_action, BOUNDS[0], BOUNDS[1], V_MAX)
        total_bounds_v += bv
        total_vel_v += vv
        last_noisy_action = noisy_action.copy()

        # Apply SafeContract if enabled
        if use_safety:
            safe_action = apply_safecontract(noisy_action, last_safe_action, BOUNDS[0], BOUNDS[1], V_MAX)
            clip_mag = np.abs(noisy_action - safe_action)
            if np.any(clip_mag > 1e-6):
                actions_modified += 1
                clip_magnitudes.extend(clip_mag[clip_mag > 1e-6].tolist())
            action_to_step = safe_action
            last_safe_action = safe_action.copy()
        else:
            action_to_step = noisy_action

        # Unnormalize for env.step()
        action_unnorm = unnormalize_action(action_to_step)

        # Step environment
        _obs, reward, done, _info = env.step(action_unnorm)
        total_reward += reward

        # Check success
        is_success = env.check_success()
        if done or is_success:
            return EpisodeResult(
                success=is_success,
                steps=t + 1,
                total_reward=total_reward,
                violations_detected=total_bounds_v + total_vel_v,
                bounds_violations=total_bounds_v,
                velocity_violations=total_vel_v,
                actions_modified=actions_modified,
                total_actions=t + 1,
                mean_clip_magnitude=(float(np.mean(clip_magnitudes)) if clip_magnitudes else 0.0),
                max_action_magnitude=float(np.max(np.abs(action_unnorm))),
            )

    # Replay finished without early termination
    return EpisodeResult(
        success=env.check_success(),
        steps=n_steps,
        total_reward=total_reward,
        violations_detected=total_bounds_v + total_vel_v,
        bounds_violations=total_bounds_v,
        velocity_violations=total_vel_v,
        actions_modified=actions_modified,
        total_actions=n_steps,
        mean_clip_magnitude=(float(np.mean(clip_magnitudes)) if clip_magnitudes else 0.0),
        max_action_magnitude=0.0,
    )


def main(
    n_episodes: int = 8,
    noise_levels: list[float] | None = None,
    suite_name: str = "libero_object",
    task_id: int = 0,
    sanity: bool = False,
):
    """Run the oracle closed-loop experiment.

    Args:
        n_episodes: Number of GT episodes to replay per condition.
        noise_levels: List of noise sigma values. Defaults to NOISE_LEVELS.
        suite_name: LIBERO suite name.
        task_id: LIBERO task ID.
        sanity: If True, run 1 episode at sigma=0 only (quick check).
    """
    import torch

    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    if sanity:
        noise_levels = [0.0]
        n_episodes = 1

    logger.info("=" * 70)
    logger.info("Oracle Closed-Loop Experiment: GT Replay + SafeContract")
    logger.info("=" * 70)
    logger.info("Suite: %s, Task: %d", suite_name, task_id)
    logger.info("Episodes: %d, Noise levels: %s", n_episodes, noise_levels)
    logger.info("SafeContract: bounds=%s, v_max=%.2f", BOUNDS, V_MAX)
    logger.info("Seed: %d", SEED)

    # Load GT episodes
    gt_episodes = load_gt_episodes(n_episodes)

    # Create environment
    logger.info("Creating LIBERO environment...")
    env, _suite, task = create_libero_env(suite_name, task_id)
    task_description = task.language
    logger.info("Task: '%s'", task_description)

    # Load init states for deterministic resets
    from libero.libero import get_libero_path

    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_states_path, weights_only=False)
    logger.info("Loaded %d init states", len(init_states))

    # Run all conditions
    all_conditions = {}
    t_start = time.perf_counter()

    for sigma in noise_levels:
        for safety_on in [False, True]:
            label = f"sigma={sigma}_safety={'on' if safety_on else 'off'}"
            logger.info("\n--- Condition: %s ---", label)

            condition = ConditionResult(
                noise_level=sigma,
                safety_on=safety_on,
            )

            for ep_idx in range(min(n_episodes, len(gt_episodes))):
                gt_actions = gt_episodes[ep_idx]
                logger.info(
                    "  Episode %d/%d (%d steps)...",
                    ep_idx + 1,
                    min(n_episodes, len(gt_episodes)),
                    len(gt_actions),
                )

                # Use a fresh RNG fork per episode so noise is identical
                # between safety=on and safety=off for the same sigma+episode
                ep_rng = np.random.default_rng(SEED + ep_idx + int(sigma * 10000))

                result = run_oracle_episode(
                    env=env,
                    gt_actions_norm=gt_actions,
                    noise_sigma=sigma,
                    use_safety=safety_on,
                    rng=ep_rng,
                    init_states=init_states,
                    init_state_id=ep_idx,
                )

                condition.episodes.append(result)
                logger.info(
                    "    success=%s, steps=%d, reward=%.2f, "
                    "violations=%d (bounds=%d, vel=%d), modified=%d/%d",
                    result.success,
                    result.steps,
                    result.total_reward,
                    result.violations_detected,
                    result.bounds_violations,
                    result.velocity_violations,
                    result.actions_modified,
                    result.total_actions,
                )

            condition.compute()
            all_conditions[label] = condition
            logger.info(
                "  => Success rate: %.1f%%, Violations: %d, Modified: %.1f%%",
                condition.success_rate * 100,
                condition.total_violations,
                condition.pct_modified,
            )

    elapsed = time.perf_counter() - t_start
    logger.info("\nTotal experiment time: %.1fs", elapsed)

    # Compile results
    results = {
        "experiment": "Oracle Closed-Loop: GT Replay with SafeContract",
        "description": (
            "Replays ground-truth actions from smol-libero in LIBERO sim "
            "with controlled Gaussian noise. Compares task success with and "
            "without SafeContract to answer: does safety clipping degrade "
            "task performance?"
        ),
        "config": {
            "suite": suite_name,
            "task_id": task_id,
            "task_description": task_description,
            "n_episodes": min(n_episodes, len(gt_episodes)),
            "noise_levels": noise_levels,
            "bounds": list(BOUNDS),
            "v_max": V_MAX,
            "seed": SEED,
            "dataset": "HuggingFaceVLA/smol-libero",
            "episode_lengths": [len(ep) for ep in gt_episodes],
        },
        "conditions": {},
        "comparison": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
    }

    # Serialize each condition
    for label, cond in all_conditions.items():
        results["conditions"][label] = {
            "noise_level": cond.noise_level,
            "safety_on": cond.safety_on,
            "success_rate": round(cond.success_rate, 3),
            "mean_steps": round(cond.mean_steps, 1),
            "mean_reward": round(cond.mean_reward, 3),
            "total_violations": cond.total_violations,
            "bounds_violations": cond.total_bounds_violations,
            "velocity_violations": cond.total_velocity_violations,
            "pct_actions_modified": round(cond.pct_modified, 2),
            "total_actions": cond.total_actions,
            "episodes": [asdict(e) for e in cond.episodes],
        }

    # Build comparison table: for each noise level, compare safe vs unsafe
    for sigma in noise_levels:
        off_key = f"sigma={sigma}_safety=off"
        on_key = f"sigma={sigma}_safety=on"
        if off_key in all_conditions and on_key in all_conditions:
            off = all_conditions[off_key]
            on = all_conditions[on_key]
            results["comparison"][f"sigma={sigma}"] = {
                "success_rate_off": round(off.success_rate, 3),
                "success_rate_on": round(on.success_rate, 3),
                "success_delta": round(on.success_rate - off.success_rate, 3),
                "reward_off": round(off.mean_reward, 3),
                "reward_on": round(on.mean_reward, 3),
                "reward_delta": round(on.mean_reward - off.mean_reward, 3),
                "violations_off": off.total_violations,
                "violations_on": on.total_violations,
                "pct_modified": round(on.pct_modified, 2),
            }

    # Key findings
    sigma_0_off = all_conditions.get("sigma=0_safety=off")
    sigma_0_on = all_conditions.get("sigma=0_safety=on")

    findings = {}
    if sigma_0_off:
        findings["gt_replay_success_rate"] = round(sigma_0_off.success_rate, 3)
        findings["gt_replay_mean_reward"] = round(sigma_0_off.mean_reward, 3)
        findings["gt_replay_viable"] = sigma_0_off.success_rate > 0
    if sigma_0_on and sigma_0_off:
        findings["safety_degrades_clean_gt"] = sigma_0_on.success_rate < sigma_0_off.success_rate
        findings["clean_success_delta"] = round(sigma_0_on.success_rate - sigma_0_off.success_rate, 3)
        findings["clean_reward_delta"] = round(sigma_0_on.mean_reward - sigma_0_off.mean_reward, 3)

    # Check if SafeContract preserves success across noise levels
    preserves_success = []
    preserves_reward = []
    for sigma in noise_levels:
        off_key = f"sigma={sigma}_safety=off"
        on_key = f"sigma={sigma}_safety=on"
        if off_key in all_conditions and on_key in all_conditions:
            preserves_success.append(
                all_conditions[on_key].success_rate >= all_conditions[off_key].success_rate
            )
            preserves_reward.append(
                all_conditions[on_key].mean_reward
                >= all_conditions[off_key].mean_reward - 0.01  # small tolerance
            )
    findings["safety_preserves_or_improves_success_all_levels"] = all(preserves_success)
    findings["safety_preserves_reward_all_levels"] = all(preserves_reward)

    # Note about open-loop replay limitations
    findings["note"] = (
        "Open-loop GT replay may not achieve high success because: "
        "(1) smol-libero episodes may span multiple tasks, not just the "
        "configured one, and (2) open-loop replay has no feedback correction. "
        "The key metric is the COMPARATIVE delta between safety=on and "
        "safety=off at each noise level - not absolute success rate."
    )

    results["key_findings"] = findings

    # Save results
    out_path = RESULTS_DIR / "exp_oracle_closed_loop.json"

    def default_serializer(o):
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    out_path.write_text(json.dumps(results, indent=2, default=default_serializer))

    # Print summary table
    print(f"\n{'=' * 80}")
    print("ORACLE CLOSED-LOOP RESULTS: GT Replay + SafeContract")
    print(f"{'=' * 80}")
    print(f"Task: {task_description}")
    print(f"Episodes: {min(n_episodes, len(gt_episodes))}, Seed: {SEED}")
    print(f"SafeContract: bounds={BOUNDS}, v_max={V_MAX}")
    print()

    header = (
        f"{'Sigma':>6} | {'Safety':>6} | {'Success':>8} | "
        f"{'AvgSteps':>9} | {'AvgReward':>10} | "
        f"{'BndViol':>8} | {'VelViol':>8} | {'Modified':>9}"
    )
    print(header)
    print("-" * len(header))

    for sigma in noise_levels:
        for safety_on in [False, True]:
            label = f"sigma={sigma}_safety={'on' if safety_on else 'off'}"
            cond = all_conditions.get(label)
            if cond is None:
                continue
            print(
                f"{sigma:>6.2f} | "
                f"{'ON' if safety_on else 'OFF':>6} | "
                f"{cond.success_rate:>7.1%} | "
                f"{cond.mean_steps:>9.1f} | "
                f"{cond.mean_reward:>10.3f} | "
                f"{cond.total_bounds_violations:>8d} | "
                f"{cond.total_velocity_violations:>8d} | "
                f"{cond.pct_modified:>8.1f}%"
            )
        if sigma != noise_levels[-1]:
            print()

    # Print comparison summary
    print(f"\n{'=' * 80}")
    print("COMPARISON: Success Rate Delta (SafeContract ON - OFF)")
    print(f"{'=' * 80}")
    for sigma in noise_levels:
        comp = results["comparison"].get(f"sigma={sigma}", {})
        delta = comp.get("success_delta", 0)
        symbol = "+" if delta > 0 else ("=" if delta == 0 else "")
        print(
            f"  sigma={sigma:.2f}: "
            f"{comp.get('success_rate_off', 0):.1%} -> {comp.get('success_rate_on', 0):.1%} "
            f"({symbol}{delta:+.1%}), "
            f"modified={comp.get('pct_modified', 0):.1f}%"
        )

    print(f"\nKey findings: {json.dumps(findings, indent=2)}")
    print(f"\nResults saved to {out_path}")

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle closed-loop: GT action replay with SafeContract")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=8,
        help="Number of GT episodes to replay per condition",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help="Noise sigma values (default: 0.0 0.05 0.1 0.2 0.5)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_object",
        help="LIBERO suite name",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="LIBERO task ID",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Quick sanity check: 1 episode, sigma=0 only",
    )
    args = parser.parse_args()

    main(
        n_episodes=args.n_episodes,
        noise_levels=args.noise_levels,
        suite_name=args.suite,
        task_id=args.task_id,
        sanity=args.sanity,
    )
