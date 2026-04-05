#!/usr/bin/env python3
"""Closed-loop evaluation with 50 episodes for statistical significance.

Uses same script as closed_loop_eval.py but with more episodes.
"""

from __future__ import annotations

import sys
import types
import json
import time
from pathlib import Path

import numpy as np

# Block groot import
_gm = types.ModuleType("lerobot.policies.groot"); _gm.__path__ = []; sys.modules["lerobot.policies.groot"] = _gm
_cm = types.ModuleType("lerobot.policies.groot.configuration_groot"); _cm.GrootConfig = type("GC", (), {}); sys.modules["lerobot.policies.groot.configuration_groot"] = _cm
_mm = types.ModuleType("lerobot.policies.groot.modeling_groot"); _mm.GrootPolicy = type("GP", (), {}); sys.modules["lerobot.policies.groot.modeling_groot"] = _mm
_gm.GrootConfig = _cm.GrootConfig; _gm.GrootPolicy = _mm.GrootPolicy

import torch
import gym_aloha  # noqa: F401
import gymnasium
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from lerobot.policies.act.modeling_act import ACTPolicy

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"


class NormalizedACT:
    def __init__(self, device: str = "cpu"):
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

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Loaded ACT on {device}. Params: {n_params:,}")

    def reset(self):
        self.policy.reset()

    def predict(self, obs_image: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        img = torch.from_numpy(obs_image).float().to(self.device) / 255.0
        img = img.permute(2, 0, 1)
        img = (img - self.img_mean) / (self.img_std + 1e-8)
        img = img.unsqueeze(0)

        state = torch.from_numpy(qpos).float().to(self.device)
        state = (state - self.state_mean) / (self.state_std + 1e-8)
        state = state.unsqueeze(0)

        obs_dict = {
            "observation.images.top": img,
            "observation.state": state,
        }

        with torch.inference_mode():
            action_norm = self.policy.select_action(obs_dict)

        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.detach().cpu()

        action = action_norm * self.action_std.cpu() + self.action_mean.cpu()
        action = action.numpy()

        if action.ndim > 1:
            action = action[0]
        return action


def run_episode(env, model, safety_contract=None, max_steps=300):
    obs, info = env.reset()
    model.reset()

    total_reward = 0.0
    max_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None

    for step in range(max_steps):
        qpos = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
        action = model.predict(obs["top"], qpos)
        original_action = action.copy()

        if safety_contract is not None:
            lo = safety_contract["action_lo"]
            hi = safety_contract["action_hi"]
            v_max = safety_contract["v_max"]

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

            if not np.allclose(original_action, action, atol=1e-6):
                actions_modified += 1

        prev_action = action.copy()
        action = action.astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
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


def evaluate(n_episodes=50, device="cpu", seed_base=0):
    print(f"Loading ACT policy from {MODEL_ID}...")
    model = NormalizedACT(device=device)

    action_mean = model.action_mean.cpu().numpy()
    action_std = model.action_std.cpu().numpy()
    action_lo = action_mean - 4 * action_std
    action_hi = action_mean + 4 * action_std
    for gi in [6, 13]:
        action_lo[gi] = max(action_lo[gi], -0.1)
        action_hi[gi] = min(action_hi[gi], 1.1)

    contract = {"action_lo": action_lo, "action_hi": action_hi, "v_max": 0.05}

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
            metrics["elapsed_s"] = elapsed
            metrics["seed"] = seed_base + ep
            results[condition].append(metrics)

            status = "OK" if metrics["success"] else "--"
            print(
                f"  Ep {ep:2d}: {status} | max_r={metrics['max_reward']:.0f} | "
                f"steps={metrics['steps']:3d} | viol={metrics['violations']:3d} | "
                f"mod={metrics['actions_modified']:3d} | {elapsed:.1f}s"
            )
            env.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = {}
    for condition in ["no_contract", "with_contract"]:
        eps = results[condition]
        successes = sum(1 for e in eps if e["success"])
        sr = successes / len(eps)
        avg_reward = np.mean([e["total_reward"] for e in eps])
        avg_max_r = np.mean([e["max_reward"] for e in eps])
        total_viol = sum(e["violations"] for e in eps)
        total_mod = sum(e["actions_modified"] for e in eps)
        avg_time = np.mean([e["elapsed_s"] for e in eps])

        # 95% CI for success rate (Wilson)
        from math import sqrt
        n = len(eps)
        z = 1.96
        p = sr
        ci_lo = (p + z*z/(2*n) - z*sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
        ci_hi = (p + z*z/(2*n) + z*sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)

        summary[condition] = {
            "success_rate": sr,
            "successes": successes,
            "total_episodes": n,
            "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
            "avg_reward": float(avg_reward),
            "avg_max_reward": float(avg_max_r),
            "total_violations": total_viol,
            "total_actions_modified": total_mod,
            "avg_episode_time_s": float(avg_time),
        }

        print(f"\n{condition}:")
        print(f"  Success rate: {successes}/{n} ({sr*100:.0f}%) [95% CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%]")
        print(f"  Avg reward:   {avg_reward:.1f}")
        print(f"  Avg max reward: {avg_max_r:.1f}")
        if condition == "with_contract":
            print(f"  Violations:   {total_viol}")
            print(f"  Actions modified: {total_mod}")
        print(f"  Avg time/ep:  {avg_time:.1f}s")

    sr_no = summary["no_contract"]["success_rate"]
    sr_with = summary["with_contract"]["success_rate"]
    delta = sr_with - sr_no

    # Fisher's exact test for significance
    from scipy.stats import fisher_exact
    a = summary["with_contract"]["successes"]
    b = summary["with_contract"]["total_episodes"] - a
    c = summary["no_contract"]["successes"]
    d = summary["no_contract"]["total_episodes"] - c
    _, p_value = fisher_exact([[a, b], [c, d]])

    print(f"\nKEY RESULT: SafetyContract {'DOES NOT degrade' if sr_with >= sr_no else 'DEGRADES'} performance")
    print(f"  Without: {sr_no*100:.0f}%  |  With: {sr_with*100:.0f}%  |  Delta: {delta*100:+.0f}%")
    print(f"  Fisher's exact p-value: {p_value:.3f} ({'significant' if p_value < 0.05 else 'not significant'})")

    summary["fisher_p_value"] = float(p_value)

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
