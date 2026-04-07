#!/usr/bin/env python3
"""ALOHA Monitor Analysis: Run ALL monitors on ACT closed-loop episodes.

The generalization test: do monitors that predict failure on PushT (2D)
also work on ALOHA (14-DOF bimanual manipulation)?

Key question: does reversal rate AUROC > 0.70 on ALOHA?
"""

import argparse
import json
import os
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import gym_aloha  # noqa
import gymnasium
from datasets import load_dataset
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vla_edge.validate.violations import StallDetector, JerkMonitor

try:
    from vla_edge.validate.signals import (
        SpectralEnergyMonitor, LinearPredictabilityMonitor, MomentumCoherenceMonitor
    )
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False
    print("WARNING: signals module not found")

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
ENV_ID = "gym_aloha/AlohaTransferCube-v0"


def compute_conformal_calibration(seed=42):
    """Compute conformal bounds + data-driven v_max."""
    print("Computing conformal calibration...")
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
            if ep not in cal_actions_by_ep:
                cal_actions_by_ep[ep] = []
            cal_actions_by_ep[ep].append(np.array(row["action"], dtype=np.float32))

    cal_actions = np.concatenate([np.array(v) for v in cal_actions_by_ep.values()])
    cal_mean = cal_actions.mean(axis=0)
    cal_std = cal_actions.std(axis=0)

    scores = np.max(np.abs(cal_actions - cal_mean) / (cal_std + 1e-8), axis=1)
    q_hat = np.quantile(scores, 0.95 * (1 + 1 / len(cal_actions)))
    action_lo = cal_mean - q_hat * cal_std
    action_hi = cal_mean + q_hat * cal_std
    for gi in [6, 13]:
        action_lo[gi] = max(action_lo[gi], -0.1)
        action_hi[gi] = min(action_hi[gi], 1.1)

    all_deltas = []
    for ep in sorted(cal_actions_by_ep.keys()):
        acts = np.array(cal_actions_by_ep[ep])
        if len(acts) > 1:
            all_deltas.append(np.abs(np.diff(acts, axis=0)))
    all_deltas = np.concatenate(all_deltas)
    v_max_per_joint = np.percentile(all_deltas, 99, axis=0)

    print(f"  Cal: {cal_actions.shape}, v_max mean: {v_max_per_joint.mean():.4f}")
    return action_lo, action_hi, v_max_per_joint


def load_policy(device="cuda"):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print(f"Loading ACT from {MODEL_ID}...")
    policy = ACTPolicy.from_pretrained(MODEL_ID)
    policy.to(device)
    policy.eval()

    state = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))
    action_mean = state.get("unnormalize_outputs.buffer_action.mean")
    action_std = state.get("unnormalize_outputs.buffer_action.std")

    n = sum(p.numel() for p in policy.parameters())
    print(f"  Loaded: {n:,} params")
    return policy, action_mean, action_std


def compute_episode_atv(per_step):
    if len(per_step) < 2:
        return 0.0
    actions = np.array([s["action"] for s in per_step])
    return float(np.sum(np.abs(np.diff(actions, axis=0))))


def compute_episode_reversals(per_step):
    if len(per_step) < 3:
        return 0.0
    actions = np.array([s["action"] for s in per_step])
    velocities = np.diff(actions, axis=0)
    sign_changes = np.sum(np.abs(np.diff(np.sign(velocities), axis=0)) > 0)
    return float(sign_changes / max(len(velocities) - 1, 1))


def run_monitored_episode(env, policy, device, action_mean, action_std,
                          action_lo, action_hi, v_max, max_steps=300):
    obs, info = env.reset()
    policy.reset()

    stall = StallDetector(movement_threshold=0.01, stall_window=10)
    jerk = JerkMonitor(jerk_limit=0.5)
    spectral = SpectralEnergyMonitor(window_size=32) if HAS_SIGNALS else None
    lps = LinearPredictabilityMonitor() if HAS_SIGNALS else None
    momentum = MomentumCoherenceMonitor(momentum_window=5) if HAS_SIGNALS else None

    per_step = []
    prev_action = None
    success = False
    total_bounds_viol = 0
    total_vel_viol = 0

    for step in range(max_steps):
        img = torch.from_numpy(obs["top"]).float().to(device) / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        qpos = torch.from_numpy(
            env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics).astype(np.float32)
        ).to(device).unsqueeze(0)

        with torch.inference_mode():
            action = policy.select_action({"observation.images.top": img, "observation.state": qpos})

        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np[0]

        # Unnormalize (MEAN_STD for ACT)
        if action_mean is not None and action_np.max() < 2.0:
            action_np = action_np * action_std.numpy() + action_mean.numpy()

        # Bounds check
        bounds_viol = not np.all((action_np >= action_lo) & (action_np <= action_hi))
        if bounds_viol:
            total_bounds_viol += 1

        # Velocity check
        vel_viol = False
        if prev_action is not None:
            delta = np.abs(action_np - prev_action)
            vel_viol = bool(np.any(delta > v_max))
            if vel_viol:
                total_vel_viol += 1

        # Monitors
        stall_report = stall.update(action_np)
        jerk_report = jerk.update(action_np)
        ser_val = spectral.update(action_np)["ser"] if spectral else 1.0
        lps_err = lps.update(action_np)["prediction_error"] if lps else 0.0
        coherence = momentum.update(action_np)["coherence"] if momentum else 1.0

        per_step.append({
            "action": action_np.tolist(),
            "bounds_viol": bounds_viol,
            "vel_viol": vel_viol,
            "stalled": stall_report["is_stalled"],
            "movement": stall_report["movement_norm"],
            "max_jerk": jerk_report["max_jerk"],
            "jerk_viol": jerk_report["jerk_violation"],
            "ser": ser_val,
            "prediction_error": lps_err,
            "coherence": coherence,
        })

        prev_action = action_np.copy()

        obs, reward, terminated, truncated, info = env.step(action_np.astype(np.float32))
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    return {
        "success": success,
        "steps": step + 1,
        "bounds_violations": total_bounds_viol,
        "velocity_violations": total_vel_viol,
        "stall_summary": stall.get_summary(),
        "jerk_summary": jerk.get_summary(),
        "spectral_summary": spectral.get_summary() if spectral else {},
        "lps_summary": lps.get_summary() if lps else {},
        "momentum_summary": momentum.get_summary() if momentum else {},
        "per_step": per_step,
    }


def compute_aurocs(episodes):
    from scipy.stats import mannwhitneyu
    results = {}
    successes = [e for e in episodes if e["success"]]
    failures = [e for e in episodes if not e["success"]]

    if len(successes) < 5 or len(failures) < 5:
        return {"error": f"Too few success={len(successes)} or failure={len(failures)}"}

    metrics = {
        "velocity_violations": lambda e: e["velocity_violations"],
        "stall_rate": lambda e: e["stall_summary"]["stall_rate"],
        "mean_jerk": lambda e: e["jerk_summary"].get("mean_jerk", 0),
        "jerk_rms": lambda e: e["jerk_summary"].get("jerk_rms", 0),
        "atv": lambda e: e.get("atv", 0),
        "reversal_rate": lambda e: e.get("reversal_rate", 0),
        "mean_ser": lambda e: 1.0 - e.get("spectral_summary", {}).get("mean_ser", 1.0),
        "mean_prediction_error": lambda e: e.get("lps_summary", {}).get("mean_prediction_error", 0),
        "min_coherence": lambda e: -e.get("momentum_summary", {}).get("min_coherence", 1.0),
        "reversal_rate_tmc": lambda e: e.get("momentum_summary", {}).get("reversal_rate", 0),
    }

    for name, fn in metrics.items():
        try:
            s_vals = [fn(e) for e in successes]
            f_vals = [fn(e) for e in failures]
            u, p = mannwhitneyu(f_vals, s_vals, alternative="greater")
            auroc = u / (len(f_vals) * len(s_vals))
            results[name] = {"auroc": round(auroc, 4), "p_value": round(p, 4)}
        except Exception as ex:
            results[name] = {"error": str(ex)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="/workspace/results/aloha_monitor_analysis.json")
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "osmesa"

    action_lo, action_hi, v_max = compute_conformal_calibration()
    policy, action_mean, action_std = load_policy(args.device)

    # Calibrate LPS from demo data if available
    if HAS_SIGNALS:
        ds = load_dataset(DATASET_ID, split="train")
        cal_actions = np.array([row["action"] for row in ds][:2000], dtype=np.float32)
        lps_cal = LinearPredictabilityMonitor()
        lps_cal.calibrate(cal_actions)

    episodes = []
    print(f"\n{'='*50}")
    print(f"  ALOHA ACT Monitor Analysis ({args.n_episodes} episodes)")
    print(f"{'='*50}")

    for ep in range(args.n_episodes):
        env = gymnasium.make(ENV_ID)
        env.reset(seed=ep)

        t0 = time.time()
        m = run_monitored_episode(env, policy, args.device, action_mean, action_std,
                                  action_lo, action_hi, v_max)
        elapsed = time.time() - t0

        per_step = m.pop("per_step")
        m["atv"] = compute_episode_atv(per_step)
        m["reversal_rate"] = compute_episode_reversals(per_step)
        m["episode"] = ep
        m["elapsed_s"] = elapsed
        episodes.append(m)

        st = "OK" if m["success"] else "--"
        stall_str = f"{m['stall_summary']['stall_rate']:.2f}"
        print(f"  Ep {ep:2d}: {st} | vel={m['velocity_violations']:4d} | "
              f"stall={stall_str} | jerk={m['jerk_summary'].get('mean_jerk', 0):.2f} | {elapsed:.1f}s")
        env.close()

    # Compute AUROCs
    aurocs = compute_aurocs(episodes)
    sr = sum(1 for e in episodes if e["success"]) / len(episodes)

    print(f"\n  Summary: SR={sr:.0%}, vel={sum(e['velocity_violations'] for e in episodes)}, "
          f"stall={sum(e['stall_summary']['total_stall_steps'] for e in episodes)}")
    print(f"  AUROCs:")
    for name, vals in sorted(aurocs.items(), key=lambda x: -x[1].get('auroc', 0) if isinstance(x[1], dict) else 0):
        if isinstance(vals, dict) and 'auroc' in vals:
            flag = ' ***' if vals['auroc'] > 0.65 else ''
            print(f"    {name:30s} {vals['auroc']:.4f} (p={vals.get('p_value', '?')}){flag}")

    output = {
        "experiment": "ALOHA ACT Monitor Analysis (14-DOF generalization test)",
        "model": MODEL_ID,
        "env": ENV_ID,
        "n_episodes": len(episodes),
        "success_rate": sr,
        "aurocs": aurocs,
        "episodes": episodes,
    }

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
