#!/usr/bin/env python3
"""Closed-loop evaluation: ACT on ALOHA sim with and without SafetyContract.

Demonstrates that SafetyContract does not degrade task success rate
when applied to a real pretrained policy in a physics simulation.

Key implementation details:
- lerobot 0.5.0 moved normalization from model submodules to a processor pipeline.
  Pretrained 0.4.x checkpoints lose their normalization buffers on load.
  We manually extract them from the raw safetensors file.
- ALOHA actions are absolute joint positions (not in [-1, 1]).
  The gym action_space says [-1, 1] but actual joint angles range wider.
  We pass full unnormalized actions to the sim - MuJoCo handles limits.
- SafetyContract bounds are set to mean +/- 4*std of training distribution
  with velocity clamping at 0.05 rad/step.

Results (50 episodes, seed 0-49):
  Without SafetyContract: 58% success (29/50)
  With SafetyContract:    60% success (30/50)
  Fisher's exact p=1.0 (no significant difference)
  3,949 violations caught, 3,891 actions modified
"""

from __future__ import annotations

import sys
import types
import json
import time
from math import sqrt
from pathlib import Path

import numpy as np

# Block groot import to avoid lerobot 0.5.0 dataclass bug (GR00T N1.5)
_gm = types.ModuleType("lerobot.policies.groot")
_gm.__path__ = []
sys.modules["lerobot.policies.groot"] = _gm
_cm = types.ModuleType("lerobot.policies.groot.configuration_groot")
_cm.GrootConfig = type("GC", (), {})
sys.modules["lerobot.policies.groot.configuration_groot"] = _cm
_mm = types.ModuleType("lerobot.policies.groot.modeling_groot")
_mm.GrootPolicy = type("GP", (), {})
sys.modules["lerobot.policies.groot.modeling_groot"] = _mm
_gm.GrootConfig = _cm.GrootConfig
_gm.GrootPolicy = _mm.GrootPolicy

import torch
import gym_aloha  # noqa: F401 - triggers env registration
import gymnasium
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import fisher_exact
from lerobot.policies.act.modeling_act import ACTPolicy


MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"


class NormalizedACT:
    """ACT policy with manual normalization for lerobot 0.5.0 compatibility.

    lerobot 0.5.0 moved normalization from model submodules to a processor
    pipeline. Pretrained 0.4.x checkpoints lose their normalization buffers
    on load. This class extracts them from the raw safetensors file and
    applies them manually around the model's forward pass.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(MODEL_ID)
        self.policy.to(self.device)
        self.policy.eval()

        # Load normalization stats from raw checkpoint
        ckpt_path = hf_hub_download(MODEL_ID, "model.safetensors")
        state = load_file(ckpt_path)

        self.img_mean = state["normalize_inputs.buffer_observation_images_top.mean"].to(self.device)
        self.img_std = state["normalize_inputs.buffer_observation_images_top.std"].to(self.device)
        self.state_mean = state["normalize_inputs.buffer_observation_state.mean"].to(self.device)
        self.state_std = state["normalize_inputs.buffer_observation_state.std"].to(self.device)
        self.action_mean = state["unnormalize_outputs.buffer_action.mean"].to(self.device)
        self.action_std = state["unnormalize_outputs.buffer_action.std"].to(self.device)

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Loaded ACT on {device}. Params: {n_params:,}")

    def reset(self):
        self.policy.reset()

    def predict(self, obs_image: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Predict action from observation.

        Args:
            obs_image: (H, W, 3) uint8 image from top camera
            qpos: (14,) joint positions

        Returns:
            (14,) action in joint space (absolute joint positions)
        """
        # Normalize image: uint8 -> [0,1] -> mean/std normalization
        img = torch.from_numpy(obs_image).float().to(self.device) / 255.0
        img = img.permute(2, 0, 1)  # (3, H, W)
        img = (img - self.img_mean) / (self.img_std + 1e-8)
        img = img.unsqueeze(0)  # (1, 3, H, W)

        # Normalize state
        state = torch.from_numpy(qpos).float().to(self.device)
        state = (state - self.state_mean) / (self.state_std + 1e-8)
        state = state.unsqueeze(0)  # (1, 14)

        obs_dict = {
            "observation.images.top": img,
            "observation.state": state,
        }

        with torch.inference_mode():
            action_norm = self.policy.select_action(obs_dict)

        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.detach().cpu()

        # Unnormalize: model outputs normalized actions, convert to joint positions
        action = action_norm * self.action_std.cpu() + self.action_mean.cpu()
        action = action.numpy()

        if action.ndim > 1:
            action = action[0]
        return action


def run_episode(
    env,
    model: NormalizedACT,
    safety_contract: dict | None = None,
    max_steps: int = 300,
) -> dict:
    """Run one episode with optional safety contract.

    Args:
        env: ALOHA gymnasium environment
        model: NormalizedACT policy
        safety_contract: if provided, dict with action_lo, action_hi, v_max
        max_steps: maximum steps per episode

    Returns:
        Dict with success, reward, violations, etc.
    """
    obs, info = env.reset()
    model.reset()

    total_reward = 0.0
    max_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None

    for step in range(max_steps):
        # Get proper 14-dim qpos (with gripper normalization)
        qpos = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
        action = model.predict(obs["top"], qpos)
        original_action = action.copy()

        # Apply safety contract
        if safety_contract is not None:
            lo = safety_contract["action_lo"]
            hi = safety_contract["action_hi"]
            v_max = safety_contract["v_max"]

            # 1. Per-joint bounds clipping
            clipped = np.clip(action, lo, hi)
            if not np.allclose(action, clipped, atol=1e-7):
                violations += 1
            action = clipped

            # 2. Velocity clamping (smooth action transitions)
            if prev_action is not None:
                delta = action - prev_action
                if np.any(np.abs(delta) > v_max):
                    violations += 1
                action = prev_action + np.clip(delta, -v_max, v_max)
                action = np.clip(action, lo, hi)

            if not np.allclose(original_action, action, atol=1e-6):
                actions_modified += 1

        prev_action = action.copy()

        # Pass absolute joint positions to sim (no [-1,1] clipping)
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        total_reward += reward
        max_reward = max(max_reward, reward)

        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    return {
        "success": success,
        "total_reward": total_reward,
        "max_reward": max_reward,
        "steps": step + 1,
        "violations": violations,
        "actions_modified": actions_modified,
    }


def evaluate(
    n_episodes: int = 50,
    device: str = "cpu",
    seed_base: int = 0,
) -> dict:
    """Run evaluation with and without safety contract.

    Args:
        n_episodes: Number of episodes per condition
        device: torch device
        seed_base: Starting seed for reproducibility

    Returns:
        Full results dict (also saved to results/closed_loop_eval_50ep.json)
    """
    print(f"Loading ACT policy from {MODEL_ID}...")
    model = NormalizedACT(device=device)

    # Safety contract: bounds from training distribution (mean +/- 4*std)
    action_mean = model.action_mean.cpu().numpy()
    action_std = model.action_std.cpu().numpy()
    action_lo = action_mean - 4 * action_std
    action_hi = action_mean + 4 * action_std
    # Constrain gripper dims to valid range
    for gi in [6, 13]:
        action_lo[gi] = max(action_lo[gi], -0.1)
        action_hi[gi] = min(action_hi[gi], 1.1)

    contract = {"action_lo": action_lo, "action_hi": action_hi, "v_max": 0.05}

    print(f"Safety contract: v_max={contract['v_max']}, bounds=mean+/-4*std")

    results = {"no_contract": [], "with_contract": []}

    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*60}")
        print(f"Running {n_episodes} episodes: {condition}")
        print(f"{'='*60}")

        sc = contract if condition == "with_contract" else None

        for ep in range(n_episodes):
            env = gymnasium.make("gym_aloha/AlohaTransferCube-v0")
            env.reset(seed=seed_base + ep)

            t0 = time.time()
            metrics = run_episode(env, model, safety_contract=sc)
            elapsed = time.time() - t0

            metrics["episode"] = ep
            metrics["seed"] = seed_base + ep
            metrics["elapsed_s"] = elapsed
            results[condition].append(metrics)

            status = "OK" if metrics["success"] else "--"
            print(
                f"  Ep {ep:2d}: {status} | max_r={metrics['max_reward']:.0f} | "
                f"steps={metrics['steps']:3d} | viol={metrics['violations']:3d} | "
                f"mod={metrics['actions_modified']:3d} | {elapsed:.1f}s"
            )
            env.close()

    # Summarize
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = {}
    for condition in ["no_contract", "with_contract"]:
        eps = results[condition]
        successes = sum(1 for e in eps if e["success"])
        sr = successes / len(eps)
        n = len(eps)

        # Wilson 95% CI
        z = 1.96
        ci_lo = (sr + z * z / (2 * n) - z * sqrt(sr * (1 - sr) / n + z * z / (4 * n * n))) / (1 + z * z / n)
        ci_hi = (sr + z * z / (2 * n) + z * sqrt(sr * (1 - sr) / n + z * z / (4 * n * n))) / (1 + z * z / n)

        summary[condition] = {
            "success_rate": sr,
            "successes": successes,
            "total_episodes": n,
            "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
            "avg_reward": float(np.mean([e["total_reward"] for e in eps])),
            "avg_max_reward": float(np.mean([e["max_reward"] for e in eps])),
            "total_violations": sum(e["violations"] for e in eps),
            "total_actions_modified": sum(e["actions_modified"] for e in eps),
            "avg_episode_time_s": float(np.mean([e["elapsed_s"] for e in eps])),
        }

        print(f"\n{condition}:")
        print(f"  Success rate: {successes}/{n} ({sr*100:.0f}%) [95% CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%]")
        print(f"  Avg reward:   {summary[condition]['avg_reward']:.1f}")
        if condition == "with_contract":
            print(f"  Violations:   {summary[condition]['total_violations']}")
            print(f"  Actions modified: {summary[condition]['total_actions_modified']}")
        print(f"  Avg time/ep:  {summary[condition]['avg_episode_time_s']:.1f}s")

    # Statistical test
    a = summary["with_contract"]["successes"]
    b = summary["with_contract"]["total_episodes"] - a
    c = summary["no_contract"]["successes"]
    d = summary["no_contract"]["total_episodes"] - c
    _, p_value = fisher_exact([[a, b], [c, d]])

    sr_no = summary["no_contract"]["success_rate"]
    sr_with = summary["with_contract"]["success_rate"]
    delta = sr_with - sr_no

    print(f"\nKEY RESULT: SafetyContract {'DOES NOT degrade' if sr_with >= sr_no else 'DEGRADES'} performance")
    print(f"  Without: {sr_no*100:.0f}%  |  With: {sr_with*100:.0f}%  |  Delta: {delta*100:+.0f}%")
    print(f"  Fisher's exact p={p_value:.3f} ({'significant' if p_value < 0.05 else 'not significant'})")

    summary["fisher_p_value"] = float(p_value)

    # Save
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "closed_loop_eval_50ep.json"

    output = {
        "summary": summary,
        "episodes": results,
        "config": {
            "model": MODEL_ID,
            "n_episodes": n_episodes,
            "seed_base": seed_base,
            "device": device,
            "safety_contract": {
                "action_lo": contract["action_lo"].tolist(),
                "action_hi": contract["action_hi"].tolist(),
                "v_max": contract["v_max"],
            },
        },
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    return output


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    evaluate(n_episodes=50, device=device, seed_base=0)
