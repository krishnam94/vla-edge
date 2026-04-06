#!/usr/bin/env python3
"""ACT on ALOHA - GPU version for lerobot 0.4.x. n=100, conformal calibration."""

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import gym_aloha  # noqa: F401
import gymnasium
from datasets import load_dataset
from scipy.stats import fisher_exact

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"


def compute_conformal_calibration(seed=42):
    """Compute conformal bounds + data-driven v_max from demo data."""
    print("Computing conformal calibration from demos...")
    ds = load_dataset(DATASET_ID, split="train")
    episodes = sorted(set(ds["episode_index"]))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(episodes).tolist()
    n_cal = int(len(episodes) * 0.8)
    cal_episodes = set(shuffled[:n_cal])

    cal_actions_by_ep = {}
    for row in ds:
        ep = row["episode_index"]
        if ep in cal_episodes:
            if ep not in cal_actions_by_ep: cal_actions_by_ep[ep] = []
            cal_actions_by_ep[ep].append(np.array(row["action"], dtype=np.float32))

    cal_actions = np.concatenate([np.array(v) for v in cal_actions_by_ep.values()])
    cal_mean = cal_actions.mean(axis=0)
    cal_std = cal_actions.std(axis=0)

    # Conformal bounds (alpha=0.05)
    scores = np.max(np.abs(cal_actions - cal_mean) / (cal_std + 1e-8), axis=1)
    q_hat = np.quantile(scores, 0.95 * (1 + 1/len(cal_actions)))
    action_lo = cal_mean - q_hat * cal_std
    action_hi = cal_mean + q_hat * cal_std
    for gi in [6, 13]:
        action_lo[gi] = max(action_lo[gi], -0.1)
        action_hi[gi] = min(action_hi[gi], 1.1)

    # Data-driven v_max (99th percentile per joint)
    all_deltas = []
    for ep in sorted(cal_actions_by_ep.keys()):
        acts = np.array(cal_actions_by_ep[ep])
        if len(acts) > 1:
            all_deltas.append(np.abs(np.diff(acts, axis=0)))
    all_deltas = np.concatenate(all_deltas)
    v_max_per_joint = np.percentile(all_deltas, 99, axis=0)

    print(f"  Cal actions: {cal_actions.shape}")
    print(f"  Conformal q_hat: {q_hat:.3f}")
    print(f"  v_max per joint (p99): mean={v_max_per_joint.mean():.4f}")
    return action_lo, action_hi, v_max_per_joint


def load_policy(device="cuda"):
    from lerobot.policies.act.modeling_act import ACTPolicy
    print(f"Loading ACT from {MODEL_ID}...")
    policy = ACTPolicy.from_pretrained(MODEL_ID)
    policy.to(device)
    policy.eval()
    n = sum(p.numel() for p in policy.parameters())
    print(f"  Loaded: {n:,} params on {device}")

    # Check normalization
    env = gymnasium.make("gym_aloha/AlohaTransferCube-v0")
    obs, _ = env.reset(seed=9999)
    policy.reset()

    img = torch.from_numpy(obs["top"]).float().to(device) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    qpos = torch.from_numpy(
        env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
    ).to(device).unsqueeze(0)

    obs_dict = {"observation.images.top": img, "observation.state": qpos}
    with torch.inference_mode():
        action = policy.select_action(obs_dict)

    action_np = action.detach().cpu().numpy()
    if action_np.ndim > 1: action_np = action_np[0]
    print(f"  Sanity: action range=[{action_np.min():.3f}, {action_np.max():.3f}]")

    is_normalized = action_np.max() < 2.0 and action_np.min() > -2.0
    print(f"  {'NORMALIZED' if is_normalized else 'RAW'}")
    env.close()
    return policy, is_normalized


def get_unnorm_stats():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    state = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))
    action_mean = state.get("unnormalize_outputs.buffer_action.mean")
    action_std = state.get("unnormalize_outputs.buffer_action.std")
    if action_mean is not None:
        print(f"  Action unnorm: mean_range=[{action_mean.min():.3f}, {action_mean.max():.3f}]")
    return action_mean, action_std


def run_episode(env, policy, device, is_normalized, action_mean, action_std,
                safety_contract=None, max_steps=300):
    obs, info = env.reset()
    policy.reset()
    total_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None

    for step in range(max_steps):
        img = torch.from_numpy(obs["top"]).float().to(device) / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        qpos = torch.from_numpy(
            env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
        ).to(device).unsqueeze(0)

        with torch.inference_mode():
            action = policy.select_action({"observation.images.top": img, "observation.state": qpos})

        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1: action_np = action_np[0]

        if is_normalized and action_mean is not None:
            action_np = action_np * action_std.numpy() + action_mean.numpy()

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
        if info.get("is_success", False): success = True; break
        if terminated or truncated: break

    return {"success": success, "total_reward": total_reward, "steps": step + 1,
            "violations": violations, "actions_modified": actions_modified}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=str, default="/workspace/results/act_aloha_100ep.json")
    args = parser.parse_args()

    action_lo, action_hi, v_max_per_joint = compute_conformal_calibration()
    policy, is_normalized = load_policy(args.device)
    action_mean, action_std = get_unnorm_stats()

    contract = {"action_lo": action_lo, "action_hi": action_hi, "v_max": v_max_per_joint}

    results = {"no_contract": [], "with_contract": []}
    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*50}\n  {condition} ({args.n_episodes} episodes)\n{'='*50}")
        sc = contract if condition == "with_contract" else None
        for ep in range(args.n_episodes):
            env = gymnasium.make("gym_aloha/AlohaTransferCube-v0")
            env.reset(seed=args.seed_base + ep)
            t0 = time.time()
            m = run_episode(env, policy, args.device, is_normalized, action_mean, action_std, safety_contract=sc)
            m["episode"] = ep; m["seed"] = args.seed_base + ep; m["elapsed_s"] = time.time() - t0
            results[condition].append(m)
            st = "OK" if m["success"] else "--"
            print(f"  Ep {ep:2d}: {st} | viol={m['violations']:4d} | {m['elapsed_s']:.1f}s")
            env.close()

    summary = {}
    for cond in ["no_contract", "with_contract"]:
        eps = results[cond]; successes = sum(1 for e in eps if e["success"])
        sr = successes / len(eps); n = len(eps); z = 1.96
        ci_lo = (sr + z*z/(2*n) - z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        ci_hi = (sr + z*z/(2*n) + z*sqrt(sr*(1-sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        summary[cond] = {"success_rate": sr, "successes": successes, "total_episodes": n,
                         "ci_95": [round(max(0, ci_lo), 3), round(min(1, ci_hi), 3)],
                         "total_violations": sum(e["violations"] for e in eps),
                         "total_actions_modified": sum(e["actions_modified"] for e in eps)}

    a = summary["no_contract"]["successes"]; b = summary["no_contract"]["total_episodes"] - a
    c = summary["with_contract"]["successes"]; d = summary["with_contract"]["total_episodes"] - c
    _, p = fisher_exact([[a, b], [c, d]]); summary["fisher_p_value"] = round(p, 4)

    print(f"\nSUMMARY")
    for cond in summary:
        if cond == "fisher_p_value": continue
        s = summary[cond]; print(f"  {cond}: {s['success_rate']:.0%} ({s['successes']}/{s['total_episodes']})")
    print(f"  Fisher p = {summary['fisher_p_value']}")

    output = {"experiment": "ACT ALOHA n=100 conformal calibration", "summary": summary,
              "episodes": results, "calibration": {"method": "conformal_95pct + p99_vmax"},
              "config": {"model": MODEL_ID, "n_episodes": args.n_episodes, "device": args.device}}
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
