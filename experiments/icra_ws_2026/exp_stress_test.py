#!/usr/bin/env python3
"""EXP-STRESS: Stress test - task success vs enforcement strictness.

Runs ACT closed-loop at 5 velocity limits to find the operating envelope:
at what v_max does SafeContract start degrading task success?

Uses held-out calibration bounds (same as EXP-ACT-CL-HOLDOUT).
"""

from __future__ import annotations

import sys
import types
import json
import time
import argparse
from math import sqrt
from pathlib import Path

import numpy as np

# Block groot import
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
import gym_aloha  # noqa: F401
import gymnasium
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import fisher_exact
from lerobot.policies.act.modeling_act import ACTPolicy

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"


def compute_holdout_bounds(calibration_split=0.8, n_sigma=4.0, seed=42):
    """Compute bounds from held-out calibration (same as EXP-ACT-CL-HOLDOUT)."""
    ds = load_dataset(DATASET_ID, split="train")
    episodes = sorted(set(ds["episode_index"]))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(episodes).tolist()
    n_cal = int(len(episodes) * calibration_split)
    cal_episodes = set(shuffled[:n_cal])

    cal_actions = []
    for row in ds:
        if row["episode_index"] in cal_episodes:
            cal_actions.append(np.array(row["action"], dtype=np.float32))
    cal_actions = np.array(cal_actions)

    cal_mean = cal_actions.mean(axis=0)
    cal_std = cal_actions.std(axis=0)
    action_lo = cal_mean - n_sigma * cal_std
    action_hi = cal_mean + n_sigma * cal_std
    for gi in [6, 13]:
        action_lo[gi] = max(action_lo[gi], -0.1)
        action_hi[gi] = min(action_hi[gi], 1.1)

    return action_lo, action_hi


class NormalizedACT:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(MODEL_ID)
        self.policy.to(self.device)
        self.policy.eval()
        ckpt_path = hf_hub_download(MODEL_ID, "model.safetensors")
        state = load_file(ckpt_path)
        self.img_mean = state["normalize_inputs.buffer_observation_images_top.mean"].to(self.device)
        self.img_std = state["normalize_inputs.buffer_observation_images_top.std"].to(self.device)
        self.state_mean = state["normalize_inputs.buffer_observation_state.mean"].to(self.device)
        self.state_std = state["normalize_inputs.buffer_observation_state.std"].to(self.device)
        self.action_mean = state["unnormalize_outputs.buffer_action.mean"].to(self.device)
        self.action_std = state["unnormalize_outputs.buffer_action.std"].to(self.device)
        print(f"  ACT loaded on {device}")

    def reset(self):
        self.policy.reset()

    def predict(self, obs_image, qpos):
        img = torch.from_numpy(obs_image).float().to(self.device) / 255.0
        img = img.permute(2, 0, 1)
        img = (img - self.img_mean) / (self.img_std + 1e-8)
        img = img.unsqueeze(0)
        state = torch.from_numpy(qpos).float().to(self.device)
        state = (state - self.state_mean) / (self.state_std + 1e-8)
        state = state.unsqueeze(0)
        obs_dict = {"observation.images.top": img, "observation.state": state}
        with torch.inference_mode():
            action_norm = self.policy.select_action(obs_dict)
        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.detach().cpu()
        action = action_norm * self.action_std.cpu() + self.action_mean.cpu()
        action = action.numpy()
        if action.ndim > 1:
            action = action[0]
        return action


def run_episode(env, model, lo, hi, v_max, max_steps=300):
    obs, info = env.reset()
    model.reset()
    total_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None

    for step in range(max_steps):
        qpos = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
        action = model.predict(obs["top"], qpos)
        original = action.copy()

        clipped = np.clip(action, lo, hi)
        if not np.allclose(action, clipped, atol=1e-7):
            violations += 1
        action = clipped

        if prev_action is not None:
            delta = action - prev_action
            if np.any(np.abs(delta) > v_max):
                violations += 1
            action = prev_action + np.clip(delta, -v_max, v_max)
            action = np.clip(action, lo, hi)

        if not np.allclose(original, action, atol=1e-6):
            actions_modified += 1

        prev_action = action.copy()
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        total_reward += reward

        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    return {"success": success, "violations": violations, "actions_modified": actions_modified, "steps": step + 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("EXP-STRESS: Task Success vs Enforcement Strictness")
    print("=" * 60)

    print("Computing held-out calibration bounds...")
    action_lo, action_hi = compute_holdout_bounds()

    print(f"Loading ACT policy...")
    model = NormalizedACT(device=args.device)

    v_max_levels = [0.01, 0.02, 0.05, 0.1, 0.2]
    # Also include baseline (no contract = very large v_max)
    all_levels = [None] + v_max_levels  # None = no contract

    results = []

    for v_max in all_levels:
        label = f"v_max={v_max}" if v_max is not None else "no_contract"
        print(f"\n{'='*40}")
        print(f"  {label} ({args.n_episodes} episodes)")
        print(f"{'='*40}")

        successes = 0
        total_violations = 0
        total_modified = 0

        for ep in range(args.n_episodes):
            env = gymnasium.make("gym_aloha/AlohaTransferCube-v0")
            env.reset(seed=ep)

            if v_max is not None:
                metrics = run_episode(env, model, action_lo, action_hi, v_max)
            else:
                # No contract baseline
                obs, info = env.reset()
                model.reset()
                success = False
                for step in range(300):
                    qpos = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
                    action = model.predict(obs["top"], qpos)
                    obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
                    if info.get("is_success", False):
                        success = True
                        break
                    if terminated or truncated:
                        break
                metrics = {"success": success, "violations": 0, "actions_modified": 0, "steps": step + 1}

            env.close()
            if metrics["success"]:
                successes += 1
            total_violations += metrics["violations"]
            total_modified += metrics["actions_modified"]

            status = "OK" if metrics["success"] else "--"
            print(f"    Ep {ep:2d}: {status} | viol={metrics['violations']:4d} | mod={metrics['actions_modified']:4d}")

        sr = successes / args.n_episodes
        z = 1.96
        n = args.n_episodes
        ci_lo = (sr + z*z/(2*n) - z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        ci_hi = (sr + z*z/(2*n) + z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)

        level_result = {
            "v_max": v_max,
            "label": label,
            "success_rate": sr,
            "successes": successes,
            "total_episodes": args.n_episodes,
            "ci_95": [round(max(0, ci_lo), 3), round(min(1, ci_hi), 3)],
            "total_violations": total_violations,
            "total_actions_modified": total_modified,
            "mean_violations_per_ep": total_violations / args.n_episodes,
        }
        results.append(level_result)
        print(f"  Result: {sr:.0%} ({successes}/{args.n_episodes}), {total_violations} violations")

    # Summary
    print(f"\n{'='*60}")
    print("STRESS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'v_max':>12} {'Success':>10} {'CI':>20} {'Violations':>12}")
    for r in results:
        print(f"{r['label']:>12} {r['success_rate']:>9.0%} {str(r['ci_95']):>20} {r['total_violations']:>12}")

    output = {
        "experiment": "EXP-STRESS: Task Success vs Enforcement Strictness",
        "model": MODEL_ID,
        "n_episodes_per_level": args.n_episodes,
        "calibration": "holdout_80_20",
        "levels": results,
    }

    out_path = Path("experiments/icra_ws_2026/results/exp_stress_test.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
