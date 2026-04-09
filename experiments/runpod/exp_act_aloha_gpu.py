#!/usr/bin/env python3
"""ACT on ALOHA - GPU version for lerobot 0.4.x. n=100, conformal calibration.

Key fix (2026-04-08): lerobot 0.5.x moved normalization from model submodules
to a processor pipeline. Pretrained 0.4.x checkpoints lose their normalization
buffers on load. We must manually normalize inputs (images, qpos) and
unnormalize outputs (actions) using stats from the safetensors file.
"""

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

TASKS = {
    "transfer": {
        "model": "lerobot/act_aloha_sim_transfer_cube_human",
        "dataset": "lerobot/aloha_sim_transfer_cube_human",
        "env": "gym_aloha/AlohaTransferCube-v0",
    },
    "insertion": {
        "model": "lerobot/act_aloha_sim_insertion_human",
        "dataset": "lerobot/aloha_sim_insertion_human",
        "env": "gym_aloha/AlohaInsertion-v0",
    },
}
MODEL_ID = None  # set in main()
DATASET_ID = None
ENV_ID = None


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
    """Load ACT policy and all normalization stats from safetensors.

    lerobot 0.5.x drops normalization buffers on load, so we extract them
    manually from the raw checkpoint. Returns the policy plus all six
    normalization tensors (img mean/std, state mean/std, action mean/std).
    """
    from lerobot.policies.act.modeling_act import ACTPolicy
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print(f"Loading ACT from {MODEL_ID}...")
    policy = ACTPolicy.from_pretrained(MODEL_ID)
    policy.to(device)
    policy.eval()
    n = sum(p.numel() for p in policy.parameters())
    print(f"  Loaded: {n:,} params on {device}")

    # Extract all normalization stats from raw safetensors
    ckpt_path = hf_hub_download(MODEL_ID, "model.safetensors")
    state = load_file(ckpt_path)

    norm_stats = {}
    norm_stats["img_mean"] = state["normalize_inputs.buffer_observation_images_top.mean"].to(device)
    norm_stats["img_std"] = state["normalize_inputs.buffer_observation_images_top.std"].to(device)
    norm_stats["state_mean"] = state["normalize_inputs.buffer_observation_state.mean"].to(device)
    norm_stats["state_std"] = state["normalize_inputs.buffer_observation_state.std"].to(device)
    norm_stats["action_mean"] = state["unnormalize_outputs.buffer_action.mean"].to(device)
    norm_stats["action_std"] = state["unnormalize_outputs.buffer_action.std"].to(device)

    print(f"  Normalization stats loaded from safetensors")
    print(f"  Image mean shape: {norm_stats['img_mean'].shape}")
    print(f"  State mean shape: {norm_stats['state_mean'].shape}")
    print(f"  Action mean range: [{norm_stats['action_mean'].min():.3f}, {norm_stats['action_mean'].max():.3f}]")

    # Sanity check: run one forward pass with proper normalization
    env = gymnasium.make(ENV_ID)
    obs, _ = env.reset(seed=9999)
    policy.reset()

    img = torch.from_numpy(obs["top"]).float().to(device) / 255.0
    img = img.permute(2, 0, 1)
    img = (img - norm_stats["img_mean"]) / (norm_stats["img_std"] + 1e-8)
    img = img.unsqueeze(0)

    qpos_raw = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
    qpos = torch.from_numpy(qpos_raw).float().to(device)
    qpos = (qpos - norm_stats["state_mean"]) / (norm_stats["state_std"] + 1e-8)
    qpos = qpos.unsqueeze(0)

    obs_dict = {"observation.images.top": img, "observation.state": qpos}
    with torch.inference_mode():
        action = policy.select_action(obs_dict)

    action_np = action.detach().cpu().numpy()
    if action_np.ndim > 1: action_np = action_np[0]
    # Unnormalize action
    action_unnorm = action_np * norm_stats["action_std"].cpu().numpy() + norm_stats["action_mean"].cpu().numpy()
    print(f"  Sanity: normalized action range=[{action_np.min():.3f}, {action_np.max():.3f}]")
    print(f"  Sanity: unnormalized action range=[{action_unnorm.min():.3f}, {action_unnorm.max():.3f}]")

    env.close()
    return policy, norm_stats


def run_episode(env, policy, device, norm_stats,
                safety_contract=None, max_steps=400):
    """Run one episode with proper input normalization and output unnormalization.

    Args:
        env: ALOHA gymnasium environment (already reset with seed)
        policy: ACTPolicy instance
        device: torch device string
        norm_stats: dict with img_mean/std, state_mean/std, action_mean/std
        safety_contract: optional dict with action_lo, action_hi, v_max
        max_steps: maximum steps per episode (400 matches gym-aloha default)
    """
    obs, info = env.reset()
    policy.reset()
    total_reward = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None

    for step in range(max_steps):
        # Normalize image: uint8 -> [0,1] -> dataset mean/std normalization
        img = torch.from_numpy(obs["top"]).float().to(device) / 255.0
        img = img.permute(2, 0, 1)  # (3, H, W)
        img = (img - norm_stats["img_mean"]) / (norm_stats["img_std"] + 1e-8)
        img = img.unsqueeze(0)  # (1, 3, H, W)

        # Normalize qpos state
        qpos_raw = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
        qpos = torch.from_numpy(qpos_raw).float().to(device)
        qpos = (qpos - norm_stats["state_mean"]) / (norm_stats["state_std"] + 1e-8)
        qpos = qpos.unsqueeze(0)  # (1, 14)

        with torch.inference_mode():
            action = policy.select_action({"observation.images.top": img, "observation.state": qpos})

        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1: action_np = action_np[0]

        # Unnormalize action: model outputs normalized, convert to joint positions
        action_np = action_np * norm_stats["action_std"].cpu().numpy() + norm_stats["action_mean"].cpu().numpy()

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
    parser.add_argument("--task", type=str, default="transfer", choices=["transfer", "insertion"])
    parser.add_argument("--output", type=str, default="/workspace/results/act_aloha_100ep.json")
    args = parser.parse_args()

    global MODEL_ID, DATASET_ID, ENV_ID
    task_cfg = TASKS[args.task]
    MODEL_ID = task_cfg["model"]
    DATASET_ID = task_cfg["dataset"]
    ENV_ID = task_cfg["env"]
    print(f"Task: {args.task} | Model: {MODEL_ID} | Env: {ENV_ID}")

    action_lo, action_hi, v_max_per_joint = compute_conformal_calibration()
    policy, norm_stats = load_policy(args.device)

    contract = {"action_lo": action_lo, "action_hi": action_hi, "v_max": v_max_per_joint}

    results = {"no_contract": [], "with_contract": []}
    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*50}\n  {condition} ({args.n_episodes} episodes)\n{'='*50}")
        sc = contract if condition == "with_contract" else None
        for ep in range(args.n_episodes):
            env = gymnasium.make(ENV_ID)
            env.reset(seed=args.seed_base + ep)
            t0 = time.time()
            m = run_episode(env, policy, args.device, norm_stats, safety_contract=sc)
            m["episode"] = ep; m["seed"] = args.seed_base + ep; m["elapsed_s"] = time.time() - t0
            results[condition].append(m)
            st = "OK" if m["success"] else "--"
            print(f"  Ep {ep:2d}: {st} | viol={m['violations']:4d} | {m['elapsed_s']:.1f}s")
            env.close()

        # Save after each condition completes
        partial = {"experiment": f"ACT ALOHA {args.task} (partial)", "episodes": results,
                   "config": {"model": MODEL_ID, "task": args.task, "condition_done": condition}}
        Path(args.output + ".partial").parent.mkdir(exist_ok=True, parents=True)
        with open(args.output + ".partial", "w") as f:
            json.dump(partial, f, indent=2, default=str)
        print(f"  Saved partial results ({condition} done)")

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
