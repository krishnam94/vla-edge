#!/usr/bin/env python3
"""Diffusion Policy on PushT - GPU version for lerobot 0.4.x. n=100."""

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import gym_pusht  # noqa: F401
import gymnasium
from scipy.stats import fisher_exact

MODEL_ID = "lerobot/diffusion_pusht"


def load_policy(device="cuda"):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    print(f"Loading Diffusion Policy from {MODEL_ID}...")
    policy = DiffusionPolicy.from_pretrained(MODEL_ID)
    policy.to(device)
    policy.eval()
    n = sum(p.numel() for p in policy.parameters())
    print(f"  Loaded: {n:,} params on {device}")

    # Sanity check
    env = gymnasium.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    obs, _ = env.reset(seed=9999)
    policy.reset()
    img = torch.from_numpy(obs["pixels"]).float().to(device) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    state = torch.from_numpy(obs["agent_pos"]).float().to(device).unsqueeze(0)
    with torch.inference_mode():
        action = policy.select_action({"observation.image": img, "observation.state": state})
    action_np = action.detach().cpu().numpy()
    if action_np.ndim > 1: action_np = action_np[0]
    print(f"  Sanity: action={action_np.round(2)}, range=[{action_np.min():.1f}, {action_np.max():.1f}]")
    is_normalized = action_np.max() < 2.0 and action_np.min() > -2.0
    print(f"  {'NORMALIZED' if is_normalized else 'RAW'}")
    env.close()
    return policy, is_normalized


def get_unnorm_stats():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    state = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))
    action_min = state.get("unnormalize_outputs.buffer_action.min",
                          state.get("normalize_targets.buffer_action.min"))
    action_max = state.get("unnormalize_outputs.buffer_action.max",
                          state.get("normalize_targets.buffer_action.max"))
    if action_min is not None:
        print(f"  Action unnorm: min={action_min.numpy()}, max={action_max.numpy()}")
    return action_min, action_max


def run_episode(env, policy, device, is_normalized, action_min, action_max,
                safety_contract=None, max_steps=300):
    obs, info = env.reset()
    policy.reset()
    total_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None
    max_coverage = 0.0

    for step in range(max_steps):
        img = torch.from_numpy(obs["pixels"]).float().to(device) / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        state = torch.from_numpy(obs["agent_pos"]).float().to(device).unsqueeze(0)
        with torch.inference_mode():
            action = policy.select_action({"observation.image": img, "observation.state": state})
        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1: action_np = action_np[0]

        if is_normalized and action_min is not None:
            a_min, a_max = action_min.numpy(), action_max.numpy()
            # MIN_MAX unnormalize: [-1,1] -> [min,max]
            action_np = (action_np + 1) / 2 * (a_max - a_min) + a_min

        original = action_np.copy()

        if safety_contract is not None:
            lo, hi = safety_contract["action_lo"], safety_contract["action_hi"]
            v_max = safety_contract["v_max"]
            clipped = np.clip(action_np, lo, hi)
            if not np.allclose(action_np, clipped, atol=1e-7): violations += 1
            action_np = clipped
            if prev_action is not None:
                delta = action_np - prev_action
                if np.any(np.abs(delta) > v_max): violations += 1
                action_np = prev_action + np.clip(delta, -v_max, v_max)
                action_np = np.clip(action_np, lo, hi)
            if not np.allclose(original, action_np, atol=1e-6): actions_modified += 1

        prev_action = action_np.copy()
        obs, reward, terminated, truncated, info = env.step(action_np.astype(np.float32))
        total_reward += reward
        coverage = info.get("coverage", 0.0)
        max_coverage = max(max_coverage, coverage)
        if coverage >= 0.95: success = True; break
        if terminated or truncated: break

    return {"success": success, "total_reward": total_reward, "max_coverage": max_coverage,
            "final_coverage": coverage, "steps": step + 1, "violations": violations,
            "actions_modified": actions_modified}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=str, default="/workspace/results/diffusion_pusht_100ep.json")
    args = parser.parse_args()

    policy, is_normalized = load_policy(args.device)
    action_min, action_max = get_unnorm_stats()
    contract = {"action_lo": np.array([0.0, 0.0]), "action_hi": np.array([512.0, 512.0]), "v_max": 30.0}

    results = {"no_contract": [], "with_contract": []}
    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*50}\n  {condition} ({args.n_episodes} episodes)\n{'='*50}")
        sc = contract if condition == "with_contract" else None
        for ep in range(args.n_episodes):
            env = gymnasium.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
            env.reset(seed=args.seed_base + ep)
            t0 = time.time()
            m = run_episode(env, policy, args.device, is_normalized, action_min, action_max, safety_contract=sc)
            m["episode"] = ep; m["seed"] = args.seed_base + ep; m["elapsed_s"] = time.time() - t0
            results[condition].append(m)
            st = "OK" if m["success"] else "--"
            print(f"  Ep {ep:2d}: {st} | cov={m['max_coverage']:.3f} | viol={m['violations']:3d} | {m['elapsed_s']:.1f}s")
            env.close()

    summary = {}
    for cond in ["no_contract", "with_contract"]:
        eps = results[cond]
        successes = sum(1 for e in eps if e["success"])
        sr = successes / len(eps); n = len(eps); z = 1.96
        ci_lo = (sr + z*z/(2*n) - z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        ci_hi = (sr + z*z/(2*n) + z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        summary[cond] = {"success_rate": sr, "successes": successes, "total_episodes": n,
                         "ci_95": [round(max(0, ci_lo), 3), round(min(1, ci_hi), 3)],
                         "total_violations": sum(e["violations"] for e in eps),
                         "total_actions_modified": sum(e["actions_modified"] for e in eps)}

    a, b = summary["no_contract"]["successes"], summary["no_contract"]["total_episodes"] - summary["no_contract"]["successes"]
    c, d = summary["with_contract"]["successes"], summary["with_contract"]["total_episodes"] - summary["with_contract"]["successes"]
    _, p = fisher_exact([[a, b], [c, d]]); summary["fisher_p_value"] = round(p, 4)

    print(f"\nSUMMARY")
    for cond in ["no_contract", "with_contract"]:
        s = summary[cond]; print(f"  {cond}: {s['success_rate']:.0%} ({s['successes']}/{s['total_episodes']}), CI={s['ci_95']}")
    print(f"  Fisher p = {summary['fisher_p_value']}")

    output = {"experiment": "Diffusion PushT n=100", "summary": summary, "episodes": results,
              "config": {"model": MODEL_ID, "n_episodes": args.n_episodes, "device": args.device,
                         "safety_contract": {"action_lo": [0, 0], "action_hi": [512, 512], "v_max": 30.0}}}
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
