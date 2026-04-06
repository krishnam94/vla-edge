#!/usr/bin/env python3
"""VQ-BeT on PushT - GPU version for lerobot 0.4.x.

Same-dataset comparison with Diffusion Policy.
Same seeds, same contract, same environment.
"""

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

MODEL_ID = "lerobot/vqbet_pusht"


def load_policy(device="cuda"):
    """Load VQ-BeT with manual config patch (from_pretrained has mlp_hidden_dim bug)."""
    import json
    from dataclasses import fields as dc_fields, MISSING
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print(f"Loading VQ-BeT from {MODEL_ID} (manual config patch)...")

    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    for field in ["mlp_hidden_dim", "type"]:
        raw.pop(field, None)

    def convert_features(d):
        return {k: PolicyFeature(type=FeatureType[v["type"]], shape=v["shape"]) for k, v in d.items()}
    raw["input_features"] = convert_features(raw["input_features"])
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
    n = sum(p.numel() for p in policy.parameters())
    print(f"  Loaded: {n:,} params on {device}")

    # Sanity check: run one inference to see action range
    env = gymnasium.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    obs, _ = env.reset(seed=9999)
    policy.reset()

    img = torch.from_numpy(obs["pixels"]).float().to(device) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    state = torch.from_numpy(obs["agent_pos"]).float().to(device).unsqueeze(0)

    with torch.inference_mode():
        action = policy.select_action({"observation.image": img, "observation.state": state})

    action_np = action.detach().cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np[0]

    print(f"  Sanity check action: {action_np.round(2)}")
    print(f"  Action range: [{action_np.min():.1f}, {action_np.max():.1f}]")

    # Detect if actions are normalized ([-1,1]) or raw ([0,512])
    is_normalized = action_np.max() < 2.0 and action_np.min() > -2.0
    print(f"  Actions appear {'NORMALIZED [-1,1]' if is_normalized else 'RAW [0,512]'}")

    env.close()
    return policy, is_normalized


def get_unnorm_stats(device="cuda"):
    """Get unnormalization stats from model checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(MODEL_ID, "model.safetensors")
    state = load_file(path)

    stats = {}
    for key in state:
        if "unnormalize" in key or "normalize" in key:
            stats[key] = state[key]

    action_min = stats.get("unnormalize_outputs.buffer_action.min",
                          stats.get("normalize_targets.buffer_action.min"))
    action_max = stats.get("unnormalize_outputs.buffer_action.max",
                          stats.get("normalize_targets.buffer_action.max"))

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
    per_step_actions = []

    for step in range(max_steps):
        img = torch.from_numpy(obs["pixels"]).float().to(device) / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        state = torch.from_numpy(obs["agent_pos"]).float().to(device).unsqueeze(0)

        with torch.inference_mode():
            action = policy.select_action({"observation.image": img, "observation.state": state})

        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np[0]

        # Unnormalize if needed
        if is_normalized and action_min is not None:
            a_min = action_min.numpy()
            a_max = action_max.numpy()
            # MIN_MAX unnormalize: [-1,1] -> [min,max]
            action_np = (action_np + 1) / 2 * (a_max - a_min) + a_min

        per_step_actions.append(action_np.tolist())
        original = action_np.copy()

        # Apply safety contract
        if safety_contract is not None:
            lo = safety_contract["action_lo"]
            hi = safety_contract["action_hi"]
            v_max = safety_contract["v_max"]

            clipped = np.clip(action_np, lo, hi)
            if not np.allclose(action_np, clipped, atol=1e-7):
                violations += 1
            action_np = clipped

            if prev_action is not None:
                delta = action_np - prev_action
                if np.any(np.abs(delta) > v_max):
                    violations += 1
                action_np = prev_action + np.clip(delta, -v_max, v_max)
                action_np = np.clip(action_np, lo, hi)

            if not np.allclose(original, action_np, atol=1e-6):
                actions_modified += 1

        prev_action = action_np.copy()
        obs, reward, terminated, truncated, info = env.step(action_np.astype(np.float32))
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
        "per_step_actions": per_step_actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=str, default="/workspace/results/vqbet_pusht_50ep.json")
    args = parser.parse_args()

    policy, is_normalized = load_policy(args.device)
    action_min, action_max = get_unnorm_stats(args.device)

    # Same contract as Diffusion PushT experiment
    contract = {
        "action_lo": np.array([0.0, 0.0]),
        "action_hi": np.array([512.0, 512.0]),
        "v_max": 30.0,
    }

    results = {"no_contract": [], "with_contract": []}

    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*50}")
        print(f"  {condition} ({args.n_episodes} episodes)")
        print(f"{'='*50}")

        sc = contract if condition == "with_contract" else None

        for ep in range(args.n_episodes):
            env = gymnasium.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
            env.reset(seed=args.seed_base + ep)

            t0 = time.time()
            metrics = run_episode(env, policy, args.device, is_normalized, action_min, action_max,
                                  safety_contract=sc)
            elapsed = time.time() - t0

            # Don't store per-step actions in episode results (too large)
            per_step = metrics.pop("per_step_actions")
            metrics["episode"] = ep
            metrics["seed"] = args.seed_base + ep
            metrics["elapsed_s"] = elapsed
            results[condition].append(metrics)

            status = "OK" if metrics["success"] else "--"
            print(f"  Ep {ep:2d}: {status} | cov={metrics['max_coverage']:.3f} | "
                  f"steps={metrics['steps']:3d} | viol={metrics['violations']:3d} | {elapsed:.1f}s")
            env.close()

    # Summary
    summary = {}
    for cond in ["no_contract", "with_contract"]:
        eps = results[cond]
        successes = sum(1 for e in eps if e["success"])
        sr = successes / len(eps)
        n = len(eps)
        z = 1.96
        ci_lo = (sr + z*z/(2*n) - z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        ci_hi = (sr + z*z/(2*n) + z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)

        summary[cond] = {
            "success_rate": sr,
            "successes": successes,
            "total_episodes": n,
            "ci_95": [round(max(0, ci_lo), 3), round(min(1, ci_hi), 3)],
            "total_violations": sum(e["violations"] for e in eps),
            "total_actions_modified": sum(e["actions_modified"] for e in eps),
            "avg_max_coverage": float(np.mean([e["max_coverage"] for e in eps])),
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
        "experiment": "VQ-BeT Closed-Loop on PushT (same-dataset as Diffusion)",
        "summary": summary,
        "episodes": results,
        "config": {
            "model": MODEL_ID,
            "model_params": 37515530,
            "architecture": "VQ-VAE + Behavior Transformer",
            "env": "gym_pusht/PushT-v0",
            "action_dim": 2,
            "is_normalized": is_normalized,
            "n_episodes": args.n_episodes,
            "seed_base": args.seed_base,
            "device": args.device,
            "safety_contract": {"action_lo": [0, 0], "action_hi": [512, 512], "v_max": 30.0},
        },
    }

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
