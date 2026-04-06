#!/usr/bin/env python3
"""Closed-loop evaluation: VQ-BeT on PushT with and without SafeContract.

Same-dataset comparison with Diffusion Policy on PushT.
Both architectures on the same environment, same seeds, same contract.
"""

from __future__ import annotations

import json
import sys
import time
import types
import argparse
from dataclasses import fields as dc_fields, MISSING
from math import sqrt
from pathlib import Path

import numpy as np
import torch

# Block groot
_gm = types.ModuleType("lerobot.policies.groot"); _gm.__path__ = []
sys.modules["lerobot.policies.groot"] = _gm
_cm = types.ModuleType("lerobot.policies.groot.configuration_groot")
_cm.GrootConfig = type("GC", (), {})
sys.modules["lerobot.policies.groot.configuration_groot"] = _cm
_mm = types.ModuleType("lerobot.policies.groot.modeling_groot")
_mm.GrootPolicy = type("GP", (), {})
sys.modules["lerobot.policies.groot.modeling_groot"] = _mm
_gm.GrootConfig = _cm.GrootConfig
_gm.GrootPolicy = _mm.GrootPolicy

import gym_pusht  # noqa: F401
import gymnasium
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.stats import fisher_exact

from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.configs.types import FeatureType, PolicyFeature

MODEL_ID = "lerobot/vqbet_pusht"


def load_vqbet(device="mps"):
    """Load VQ-BeT with config compatibility fix for lerobot 0.5.0."""
    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    for field in ["mlp_hidden_dim", "type"]:
        raw.pop(field, None)

    def convert_features(feat_dict):
        result = {}
        for key, val in feat_dict.items():
            ft = FeatureType[val["type"]] if isinstance(val.get("type"), str) else FeatureType.STATE
            result[key] = PolicyFeature(type=ft, shape=val["shape"])
        return result

    if "input_features" in raw and isinstance(raw["input_features"], dict):
        raw["input_features"] = convert_features(raw["input_features"])
    if "output_features" in raw and isinstance(raw["output_features"], dict):
        raw["output_features"] = convert_features(raw["output_features"])

    valid = {f.name for f in dc_fields(VQBeTConfig)}
    clean = {k: v for k, v in raw.items() if k in valid}

    config = VQBeTConfig.__new__(VQBeTConfig)
    for f in dc_fields(VQBeTConfig):
        if f.name in clean:
            setattr(config, f.name, clean[f.name])
        elif f.default is not MISSING:
            setattr(config, f.name, f.default)
        elif f.default_factory is not MISSING:
            setattr(config, f.name, f.default_factory())

    policy = VQBeTPolicy(config)
    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    state = load_file(weights_path)
    policy.load_state_dict(state, strict=False)
    policy.to(device)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  VQ-BeT loaded: {n_params:,} params on {device}")
    return policy


def get_norm_stats():
    """Get normalization stats from the model's checkpoint."""
    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    state = load_file(weights_path)

    stats = {}
    for key in state:
        if "normalize" in key or "buffer" in key:
            stats[key] = state[key]
    return stats


class NormalizedVQBeT:
    """VQ-BeT with manual MIN_MAX normalization for lerobot 0.5.0 compat."""

    def __init__(self, policy, device="mps"):
        self.policy = policy
        self.device = device

        # Load normalization stats
        weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
        state = load_file(weights_path)

        self.state_min = state["normalize_inputs.buffer_observation_state.min"].to(device)
        self.state_max = state["normalize_inputs.buffer_observation_state.max"].to(device)
        self.action_min = state["unnormalize_outputs.buffer_action.min"].to(device)
        self.action_max = state["unnormalize_outputs.buffer_action.max"].to(device)

    def reset(self):
        self.policy.reset()

    def predict(self, obs_pixels, agent_pos):
        img = torch.from_numpy(obs_pixels).float().to(self.device) / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        # MIN_MAX normalize state: (x - min) / (max - min) -> [0, 1]
        state = torch.from_numpy(agent_pos).float().to(self.device)
        state = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        state = state.unsqueeze(0)

        obs_dict = {
            "observation.image": img,
            "observation.state": state,
        }

        with torch.inference_mode():
            action_norm = self.policy.select_action(obs_dict)

        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.detach().cpu()

        # MIN_MAX unnormalize action: x * (max - min) + min
        action = action_norm * (self.action_max.cpu() - self.action_min.cpu()) + self.action_min.cpu()
        action = action.numpy()
        if action.ndim > 1:
            action = action[0]
        return action


def run_episode(env, model, device, safety_contract=None, max_steps=300):
    obs, info = env.reset()
    model.reset()

    total_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None
    max_coverage = 0.0

    for step in range(max_steps):
        action = model.predict(obs["pixels"], obs["agent_pos"])
        original_action = action.copy()

        # Apply safety contract
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

        coverage = info.get("coverage", 0.0)
        max_coverage = max(max_coverage, coverage)

        if coverage >= 0.95:
            success = True
            break
        if terminated or truncated:
            break

    return {
        "success": success,
        "total_reward": total_reward,
        "max_coverage": max_coverage,
        "final_coverage": coverage,
        "steps": step + 1,
        "violations": violations,
        "actions_modified": actions_modified,
    }


def evaluate(n_episodes=20, device="mps", seed_base=0):
    print(f"Loading VQ-BeT from {MODEL_ID}...")
    raw_policy = load_vqbet(device)
    model = NormalizedVQBeT(raw_policy, device)

    # Same contract as Diffusion PushT experiment
    contract = {
        "action_lo": np.array([0.0, 0.0]),
        "action_hi": np.array([512.0, 512.0]),
        "v_max": 30.0,
    }
    print(f"Safety contract: bounds [0, 512], v_max={contract['v_max']} px/step")

    results = {"no_contract": [], "with_contract": []}

    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*50}")
        print(f"  {condition} ({n_episodes} episodes)")
        print(f"{'='*50}")

        sc = contract if condition == "with_contract" else None

        for ep in range(n_episodes):
            env = gymnasium.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
            )
            env.reset(seed=seed_base + ep)

            t0 = time.time()
            metrics = run_episode(env, model, device, safety_contract=sc)
            elapsed = time.time() - t0

            metrics["episode"] = ep
            metrics["seed"] = seed_base + ep
            metrics["elapsed_s"] = elapsed
            results[condition].append(metrics)

            status = "OK" if metrics["success"] else "--"
            print(
                f"  Ep {ep:2d}: {status} | cov={metrics['max_coverage']:.3f} | "
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
            "ci_95": [round(max(0, ci_lo), 3), round(min(1, ci_hi), 3)],
            "avg_reward": float(np.mean([e["total_reward"] for e in eps])),
            "avg_max_coverage": float(np.mean([e["max_coverage"] for e in eps])),
            "avg_final_coverage": float(np.mean([e["final_coverage"] for e in eps])),
            "total_violations": sum(e["violations"] for e in eps),
            "total_actions_modified": sum(e["actions_modified"] for e in eps),
            "avg_episode_time_s": float(np.mean([e["elapsed_s"] for e in eps])),
        }

    a = summary["no_contract"]["successes"]
    b = summary["no_contract"]["total_episodes"] - a
    c = summary["with_contract"]["successes"]
    d = summary["with_contract"]["total_episodes"] - c
    _, p_value = fisher_exact([[a, b], [c, d]])
    summary["fisher_p_value"] = round(p_value, 4)

    print(f"\n{'='*50}")
    print("SUMMARY")
    for cond in ["no_contract", "with_contract"]:
        s = summary[cond]
        print(f"  {cond}: {s['success_rate']:.0%} ({s['successes']}/{s['total_episodes']})")
        print(f"    CI: {s['ci_95']}, Violations: {s['total_violations']}")
    print(f"  Fisher p = {summary['fisher_p_value']}")

    output = {
        "experiment": "EXP-VQBET-CL: VQ-BeT Closed-Loop on PushT (same-dataset as Diffusion)",
        "summary": summary,
        "episodes": results,
        "config": {
            "model": MODEL_ID,
            "model_params": 37515530,
            "architecture": "VQ-VAE + Behavior Transformer (VQ-BeT)",
            "env": "gym_pusht/PushT-v0",
            "action_dim": 2,
            "action_space": "pixel [0, 512]",
            "n_episodes": n_episodes,
            "seed_base": seed_base,
            "max_steps": 300,
            "device": device,
            "safety_contract": {
                "action_lo": [0.0, 0.0],
                "action_hi": [512.0, 512.0],
                "v_max": 30.0,
            },
        },
    }

    out_path = Path("results/closed_loop_vqbet_pusht.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.n_episodes, args.device, args.seed_base)
