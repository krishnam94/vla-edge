#!/usr/bin/env python3
"""Closed-loop evaluation: ACT on ALOHA sim with and without SafetyContract.

Demonstrates that SafetyContract does not degrade task success rate.

The key challenge: lerobot 0.5.0 changed normalization API, so pretrained
ACT checkpoints (trained on 0.4.x) don't load normalization stats correctly.
We manually extract and apply normalization from the checkpoint.
"""

from __future__ import annotations

import sys
import types
import json
import time
from pathlib import Path

import numpy as np

# Block groot import to avoid lerobot 0.5.0 dataclass bug
_groot_mod = types.ModuleType("lerobot.policies.groot")
_groot_mod.__path__ = []
sys.modules["lerobot.policies.groot"] = _groot_mod
_cfg_mod = types.ModuleType("lerobot.policies.groot.configuration_groot")
_cfg_mod.GrootConfig = type("GrootConfig", (), {})
sys.modules["lerobot.policies.groot.configuration_groot"] = _cfg_mod
_mdl_mod = types.ModuleType("lerobot.policies.groot.modeling_groot")
_mdl_mod.GrootPolicy = type("GrootPolicy", (), {})
sys.modules["lerobot.policies.groot.modeling_groot"] = _mdl_mod
_groot_mod.GrootConfig = _cfg_mod.GrootConfig
_groot_mod.GrootPolicy = _mdl_mod.GrootPolicy

import torch
import gym_aloha  # noqa: F401 - triggers registration
import gymnasium
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from lerobot.policies.act.modeling_act import ACTPolicy


MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"


class NormalizedACT:
    """ACT policy with manual normalization handling.

    lerobot 0.5.0 changed normalization internals, so pretrained 0.4.x checkpoints
    lose their normalization buffers on load. This class extracts them from the
    safetensors checkpoint and applies them manually.
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
        print(f"  State mean (first 3): {self.state_mean[:3].tolist()}")
        print(f"  Action mean (first 3): {self.action_mean[:3].tolist()}")

    def reset(self):
        self.policy.reset()

    def predict(self, obs_image: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Predict action from observation.

        Args:
            obs_image: (H, W, 3) uint8 image from top camera
            qpos: (14,) joint positions

        Returns:
            (14,) raw action in joint space (unnormalized)
        """
        # Normalize image: uint8 -> [0,1] -> ImageNet-style normalization
        img = torch.from_numpy(obs_image).float().to(self.device) / 255.0
        img = img.permute(2, 0, 1)  # (3, H, W)
        # img_mean/std are (3, 1, 1), broadcast over spatial dims
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
            action_normalized = self.policy.select_action(obs_dict)

        if isinstance(action_normalized, torch.Tensor):
            action_normalized = action_normalized.detach().cpu()

        # Unnormalize action
        action = action_normalized * self.action_std.cpu() + self.action_mean.cpu()
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
    """Run one episode. Returns metrics dict.

    safety_contract: if provided, dict with:
        - action_range: (lo, hi) per-dim or scalar
        - v_max: float (velocity clamp per step)
    """
    obs, info = env.reset()
    model.reset()

    total_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None
    all_actions = []

    for step in range(max_steps):
        # Get joint positions from mujoco sim
        try:
            qpos = env.unwrapped._env.physics.data.qpos[:14].copy().astype(np.float32)
        except Exception:
            qpos = np.zeros(14, dtype=np.float32)

        action = model.predict(obs["top"], qpos)
        original_action = action.copy()

        # Apply safety contract if enabled
        if safety_contract is not None:
            lo, hi = safety_contract["action_range"]
            v_max = safety_contract["v_max"]

            # 1. Action range clipping
            clipped = np.clip(action, lo, hi)
            if not np.allclose(action, clipped, atol=1e-7):
                violations += 1
            action = clipped

            # 2. Velocity clamping
            if prev_action is not None:
                delta = action - prev_action
                max_delta = np.max(np.abs(delta))
                if max_delta > v_max:
                    violations += 1
                action = prev_action + np.clip(delta, -v_max, v_max)
                # Re-clip after velocity clamping
                action = np.clip(action, lo, hi)

            if not np.allclose(original_action, action, atol=1e-6):
                actions_modified += 1

        prev_action = action.copy()
        all_actions.append(action.copy())

        # Clip to env action space bounds
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get("is_success", False):
            success = True
            break

        if terminated or truncated:
            break

    all_actions_arr = np.array(all_actions)
    return {
        "success": success,
        "total_reward": total_reward,
        "steps": step + 1,
        "violations": violations,
        "actions_modified": actions_modified,
        "action_range": [float(all_actions_arr.min()), float(all_actions_arr.max())],
        "action_mean_abs": float(np.mean(np.abs(all_actions_arr))),
    }


def evaluate(
    n_episodes: int = 10,
    device: str = "cpu",
    seed_base: int = 42,
) -> dict:
    """Run evaluation with and without safety contract."""

    print(f"Loading ACT policy from {MODEL_ID}...")
    model = NormalizedACT(device=device)

    # Safety contract: action range for ALOHA and velocity limit
    # ALOHA actions are in joint-space, range roughly [-1, 1] for normalized
    # but in raw space they're around the mean of the training data
    contract = {
        "action_range": (-1.0, 1.0),
        "v_max": 0.05,  # moderate velocity limit per step
    }

    results = {"no_contract": [], "with_contract": []}

    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*60}")
        print(f"Running {n_episodes} episodes: {condition}")
        print(f"{'='*60}")

        sc = contract if condition == "with_contract" else None

        for ep in range(n_episodes):
            env = gymnasium.make("gym_aloha/AlohaTransferCube-v0")
            obs, _ = env.reset(seed=seed_base + ep)

            t0 = time.time()
            metrics = run_episode(env, model, safety_contract=sc)
            elapsed = time.time() - t0

            metrics["episode"] = ep
            metrics["elapsed_s"] = elapsed
            results[condition].append(metrics)

            status = "SUCCESS" if metrics["success"] else "fail"
            print(
                f"  Ep {ep:2d}: {status} | reward={metrics['total_reward']:.2f} | "
                f"steps={metrics['steps']:3d} | violations={metrics['violations']:3d} | "
                f"modified={metrics['actions_modified']:3d} | "
                f"act_range=[{metrics['action_range'][0]:.3f},{metrics['action_range'][1]:.3f}] | "
                f"time={elapsed:.1f}s"
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
        avg_reward = np.mean([e["total_reward"] for e in eps])
        avg_steps = np.mean([e["steps"] for e in eps])
        total_violations = sum(e["violations"] for e in eps)
        total_modified = sum(e["actions_modified"] for e in eps)
        avg_time = np.mean([e["elapsed_s"] for e in eps])

        summary[condition] = {
            "success_rate": successes / len(eps),
            "successes": successes,
            "total_episodes": len(eps),
            "avg_reward": float(avg_reward),
            "avg_steps": float(avg_steps),
            "total_violations": total_violations,
            "total_actions_modified": total_modified,
            "avg_episode_time_s": float(avg_time),
        }

        print(f"\n{condition}:")
        print(f"  Success rate: {successes}/{len(eps)} ({100*successes/len(eps):.0f}%)")
        print(f"  Avg reward:   {avg_reward:.2f}")
        print(f"  Avg steps:    {avg_steps:.0f}")
        if condition == "with_contract":
            print(f"  Violations:   {total_violations}")
            print(f"  Actions modified: {total_modified}")
        print(f"  Avg time/ep:  {avg_time:.1f}s")

    # Key result
    sr_no = summary["no_contract"]["success_rate"]
    sr_with = summary["with_contract"]["success_rate"]
    delta = sr_with - sr_no
    print(f"\nKEY RESULT: SafetyContract {'DOES NOT degrade' if sr_with >= sr_no else 'DEGRADES'} performance")
    print(f"  Without: {sr_no*100:.0f}%  |  With: {sr_with*100:.0f}%  |  Delta: {delta*100:+.0f}%")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "closed_loop_eval.json"

    output = {"summary": summary, "episodes": results}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    return output


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    evaluate(n_episodes=10, device=device)
