#!/usr/bin/env python3
"""Closed-loop evaluation: ACT on ALOHA sim with HOLD-OUT calibration.

Addresses calibration leakage from EXP-ACT-CL: instead of computing bounds
from the full training distribution, we split demo episodes 80/20, calibrate
on 80%, and verify coverage on the held-out 20%.

This is methodologically rigorous: calibration and test data are separate.
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

# Block groot import to avoid lerobot 0.5.0 dataclass bug
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


def compute_holdout_bounds(
    calibration_split: float = 0.8,
    n_sigma: float = 4.0,
    seed: int = 42,
) -> dict:
    """Compute safety bounds from a held-out subset of demo episodes.

    Args:
        calibration_split: Fraction of episodes for calibration (rest is holdout)
        n_sigma: Number of std deviations for bounds (4 = generous)
        seed: Random seed for episode shuffling

    Returns:
        Dict with action_lo, action_hi, holdout_coverage, etc.
    """
    print(f"Loading dataset {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")

    # Get unique episode indices
    episodes = sorted(set(ds["episode_index"]))
    n_episodes = len(episodes)
    print(f"  {len(ds)} frames, {n_episodes} episodes")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(episodes).tolist()
    n_cal = int(n_episodes * calibration_split)
    cal_episodes = set(shuffled[:n_cal])
    holdout_episodes = set(shuffled[n_cal:])

    print(f"  Calibration: {len(cal_episodes)} episodes, Holdout: {len(holdout_episodes)} episodes")

    # Extract actions per split
    cal_actions = []
    holdout_actions = []
    for row in ds:
        ep = row["episode_index"]
        action = np.array(row["action"], dtype=np.float32)
        if ep in cal_episodes:
            cal_actions.append(action)
        elif ep in holdout_episodes:
            holdout_actions.append(action)

    cal_actions = np.array(cal_actions)
    holdout_actions = np.array(holdout_actions)
    print(f"  Calibration: {cal_actions.shape[0]} actions, Holdout: {holdout_actions.shape[0]} actions")

    # Compute bounds from calibration set only
    cal_mean = cal_actions.mean(axis=0)
    cal_std = cal_actions.std(axis=0)
    action_lo = cal_mean - n_sigma * cal_std
    action_hi = cal_mean + n_sigma * cal_std

    # Constrain gripper dims to valid range
    for gi in [6, 13]:
        action_lo[gi] = max(action_lo[gi], -0.1)
        action_hi[gi] = min(action_hi[gi], 1.1)

    # Verify coverage on holdout set
    holdout_in_bounds = np.all(
        (holdout_actions >= action_lo) & (holdout_actions <= action_hi),
        axis=1,
    )
    holdout_coverage = float(holdout_in_bounds.mean())

    # Per-dim coverage on holdout
    per_dim_coverage = float(
        np.mean((holdout_actions >= action_lo) & (holdout_actions <= action_hi))
    )

    # Compare to full-data bounds for reference
    full_mean = np.concatenate([cal_actions, holdout_actions]).mean(axis=0)
    full_std = np.concatenate([cal_actions, holdout_actions]).std(axis=0)
    full_lo = full_mean - n_sigma * full_std
    full_hi = full_mean + n_sigma * full_std
    for gi in [6, 13]:
        full_lo[gi] = max(full_lo[gi], -0.1)
        full_hi[gi] = min(full_hi[gi], 1.1)

    max_bound_diff = float(np.max(np.abs(action_lo - full_lo) + np.abs(action_hi - full_hi)))

    print(f"  Holdout coverage (all dims in bounds): {holdout_coverage:.4f}")
    print(f"  Holdout per-dim coverage: {per_dim_coverage:.4f}")
    print(f"  Max bound difference from full-data: {max_bound_diff:.6f}")

    return {
        "action_lo": action_lo,
        "action_hi": action_hi,
        "calibration": {
            "n_cal_episodes": len(cal_episodes),
            "n_holdout_episodes": len(holdout_episodes),
            "n_cal_actions": int(cal_actions.shape[0]),
            "n_holdout_actions": int(holdout_actions.shape[0]),
            "n_sigma": n_sigma,
            "split_seed": seed,
            "holdout_coverage_all_dims": holdout_coverage,
            "holdout_coverage_per_dim": per_dim_coverage,
            "max_bound_diff_from_full": max_bound_diff,
            "cal_mean": cal_mean.tolist(),
            "cal_std": cal_std.tolist(),
        },
    }


class NormalizedACT:
    """ACT policy with manual normalization for lerobot 0.5.0 compat."""

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


def evaluate(n_episodes=50, device="cpu", seed_base=0):
    # Step 1: Compute bounds from held-out calibration
    bounds_info = compute_holdout_bounds(calibration_split=0.8, n_sigma=4.0, seed=42)

    # Step 2: Load model
    print(f"\nLoading ACT policy from {MODEL_ID}...")
    model = NormalizedACT(device=device)

    contract = {
        "action_lo": bounds_info["action_lo"],
        "action_hi": bounds_info["action_hi"],
        "v_max": 0.05,
    }
    print(f"Safety contract: v_max={contract['v_max']}, bounds from 80% held-out calibration")

    # Step 3: Run episodes
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
    summary = {}
    for condition in ["no_contract", "with_contract"]:
        eps = results[condition]
        successes = sum(1 for e in eps if e["success"])
        sr = successes / len(eps)
        n = len(eps)
        z = 1.96
        ci_lo = (sr + z*z/(2*n) - z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        ci_hi = (sr + z*z/(2*n) + z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)

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

    # Fisher's exact test
    a = summary["no_contract"]["successes"]
    b = summary["no_contract"]["total_episodes"] - a
    c = summary["with_contract"]["successes"]
    d = summary["with_contract"]["total_episodes"] - c
    _, p_value = fisher_exact([[a, b], [c, d]])
    summary["fisher_p_value"] = round(p_value, 4)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cond in ["no_contract", "with_contract"]:
        s = summary[cond]
        print(f"  {cond}: {s['success_rate']:.0%} ({s['successes']}/{s['total_episodes']})")
        print(f"    CI: [{s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f}]")
        print(f"    Violations: {s['total_violations']}, Modified: {s['total_actions_modified']}")
    print(f"  Fisher p = {summary['fisher_p_value']}")
    print(f"  Holdout coverage: {bounds_info['calibration']['holdout_coverage_all_dims']:.4f}")

    # Save
    output = {
        "experiment": "EXP-ACT-CL-HOLDOUT: ACT Closed-Loop with Hold-Out Calibration",
        "summary": summary,
        "calibration": bounds_info["calibration"],
        "episodes": results,
        "config": {
            "model": MODEL_ID,
            "n_episodes": n_episodes,
            "seed_base": seed_base,
            "device": device,
            "calibration_method": "holdout_80_20_split",
            "safety_contract": {
                "action_lo": bounds_info["action_lo"].tolist(),
                "action_hi": bounds_info["action_hi"].tolist(),
                "v_max": 0.05,
            },
        },
    }

    out_path = Path("results/closed_loop_eval_holdout.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()

    evaluate(
        n_episodes=args.n_episodes,
        device=args.device,
        seed_base=args.seed_base,
    )
