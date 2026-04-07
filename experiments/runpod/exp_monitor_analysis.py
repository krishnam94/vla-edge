#!/usr/bin/env python3
"""Run all 5 violation monitors on VQ-BeT and Diffusion PushT episodes.

Collects per-step actions and feeds them through:
1. Bounds checking
2. Velocity checking
3. StallDetector
4. JerkMonitor
5. GripperOscillation (N/A for PushT - no gripper)

Plus conformal p-values for each step.

This is the KEY experiment: does the composite monitor predict failure
better than any single monitor alone?
"""

import argparse
import json
import sys
import time
from dataclasses import fields as dc_fields, MISSING
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import gym_pusht  # noqa: F401
import gymnasium
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vla_edge.validate.violations import StallDetector, JerkMonitor
from vla_edge.validate.conformal import ConformalActionMonitor
try:
    from vla_edge.validate.signals import (
        SpectralEnergyMonitor, LinearPredictabilityMonitor, MomentumCoherenceMonitor
    )
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False
    print("WARNING: signals module not found, running without SER/LPS/TMC")

# PushT contract
CONTRACT = {
    "action_lo": np.array([0.0, 0.0]),
    "action_hi": np.array([512.0, 512.0]),
    "v_max": 30.0,
}


def load_vqbet(device="cuda"):
    """Load VQ-BeT with manual config patch."""
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = hf_hub_download("lerobot/vqbet_pusht", "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    for field in ["mlp_hidden_dim", "type"]:
        raw.pop(field, None)
    def cf(d):
        return {k: PolicyFeature(type=FeatureType[v["type"]], shape=v["shape"]) for k, v in d.items()}
    raw["input_features"] = cf(raw["input_features"])
    raw["output_features"] = cf(raw["output_features"])
    valid = {f.name for f in dc_fields(VQBeTConfig)}
    clean = {k: v for k, v in raw.items() if k in valid}
    config = VQBeTConfig.__new__(VQBeTConfig)
    for f in dc_fields(VQBeTConfig):
        if f.name in clean: setattr(config, f.name, clean[f.name])
        elif f.default is not MISSING: setattr(config, f.name, f.default)
        elif f.default_factory is not MISSING: setattr(config, f.name, f.default_factory())

    policy = VQBeTPolicy(config)
    state = load_file(hf_hub_download("lerobot/vqbet_pusht", "model.safetensors"))
    policy.load_state_dict(state, strict=False)
    policy.to(device); policy.eval()

    stats = {
        "state_min": state.get("normalize_inputs.buffer_observation_state.min"),
        "state_max": state.get("normalize_inputs.buffer_observation_state.max"),
        "action_min": state.get("unnormalize_outputs.buffer_action.min",
                                state.get("normalize_targets.buffer_action.min")),
        "action_max": state.get("unnormalize_outputs.buffer_action.max",
                                state.get("normalize_targets.buffer_action.max")),
    }
    return policy, stats


def load_diffusion(device="cuda"):
    """Load Diffusion Policy."""
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
    policy.to(device); policy.eval()

    state = load_file(hf_hub_download("lerobot/diffusion_pusht", "model.safetensors"))
    stats = {
        "state_min": state.get("normalize_inputs.buffer_observation_state.min"),
        "state_max": state.get("normalize_inputs.buffer_observation_state.max"),
        "action_min": state.get("unnormalize_outputs.buffer_action.min",
                                state.get("normalize_targets.buffer_action.min")),
        "action_max": state.get("unnormalize_outputs.buffer_action.max",
                                state.get("normalize_targets.buffer_action.max")),
    }
    return policy, stats


IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def run_monitored_episode(env, policy, device, stats, use_imagenet_norm=False, max_steps=300):
    """Run one episode, collecting per-step monitor data."""
    obs, info = env.reset()
    policy.reset()

    s_min = stats["state_min"].numpy() if stats["state_min"] is not None else None
    s_max = stats["state_max"].numpy() if stats["state_max"] is not None else None
    a_min = stats["action_min"].numpy() if stats["action_min"] is not None else None
    a_max = stats["action_max"].numpy() if stats["action_max"] is not None else None

    # Initialize monitors
    stall = StallDetector(movement_threshold=2.0, stall_window=10)
    jerk = JerkMonitor(jerk_limit=50.0)
    spectral = SpectralEnergyMonitor(window_size=32) if HAS_SIGNALS else None
    lps = LinearPredictabilityMonitor() if HAS_SIGNALS else None
    momentum = MomentumCoherenceMonitor(momentum_window=5) if HAS_SIGNALS else None

    per_step = []
    prev_action = None
    success = False
    max_cov = 0.0
    total_bounds_viol = 0
    total_vel_viol = 0

    for step in range(max_steps):
        # Prepare observation
        if use_imagenet_norm:
            img_np = obs["pixels"].astype(np.float32) / 255.0
            img_np = (img_np - IMG_MEAN) / IMG_STD
            img = torch.from_numpy(img_np.astype(np.float32)).to(device).permute(2, 0, 1).unsqueeze(0)
        else:
            img = torch.from_numpy(obs["pixels"]).float().to(device) / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)

        agent_pos = obs["agent_pos"].astype(np.float32)
        if s_min is not None:
            agent_pos_norm = (agent_pos - s_min) / (s_max - s_min + 1e-8) * 2 - 1
        else:
            agent_pos_norm = agent_pos
        state_t = torch.from_numpy(agent_pos_norm).float().to(device).unsqueeze(0)

        with torch.inference_mode():
            action = policy.select_action({"observation.image": img, "observation.state": state_t})

        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np[0]

        # Unnormalize
        if a_min is not None and action_np.max() < 2.0:
            action_np = (action_np + 1) / 2 * (a_max - a_min) + a_min

        # Monitor: bounds
        bounds_viol = not np.all((action_np >= CONTRACT["action_lo"]) & (action_np <= CONTRACT["action_hi"]))
        if bounds_viol:
            total_bounds_viol += 1

        # Monitor: velocity
        vel_viol = False
        if prev_action is not None:
            delta = np.abs(action_np - prev_action)
            vel_viol = bool(np.any(delta > CONTRACT["v_max"]))
            if vel_viol:
                total_vel_viol += 1

        # Monitor: stall
        stall_report = stall.update(action_np)

        # Monitor: jerk
        jerk_report = jerk.update(action_np)

        # New monitors (if available)
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

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_np.astype(np.float32))
        cov = info.get("coverage", 0.0)
        max_cov = max(max_cov, cov)
        if cov >= 0.95:
            success = True
            break
        if terminated or truncated:
            break

    return {
        "success": success,
        "max_coverage": max_cov,
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


def compute_episode_atv(per_step):
    """Action Total Variation: sum of absolute deltas (ACG paper, arXiv:2510.22201)."""
    if len(per_step) < 2:
        return 0.0
    actions = np.array([s["action"] for s in per_step])
    return float(np.sum(np.abs(np.diff(actions, axis=0))))


def compute_episode_reversals(per_step):
    """Direction reversal rate: fraction of steps where velocity sign flips."""
    if len(per_step) < 3:
        return 0.0
    actions = np.array([s["action"] for s in per_step])
    velocities = np.diff(actions, axis=0)
    sign_changes = np.sum(np.abs(np.diff(np.sign(velocities), axis=0)) > 0)
    return float(sign_changes / max(len(velocities) - 1, 1))


def compute_aurocs(episodes):
    """Compute AUROC for each monitor as failure predictor."""
    from scipy.stats import mannwhitneyu

    results = {}
    successes = [e for e in episodes if e["success"]]
    failures = [e for e in episodes if not e["success"]]

    if len(successes) < 5 or len(failures) < 5:
        return {"error": f"Too few success={len(successes)} or failure={len(failures)}"}

    metrics = {
        "velocity_violations": lambda e: e["velocity_violations"],
        "stall_rate": lambda e: e["stall_summary"]["stall_rate"],
        "total_stall_steps": lambda e: e["stall_summary"]["total_stall_steps"],
        "mean_jerk": lambda e: e["jerk_summary"].get("mean_jerk", 0),
        "jerk_rms": lambda e: e["jerk_summary"].get("jerk_rms", 0),
        "jerk_violations": lambda e: e["jerk_summary"].get("violations", 0),
        "atv": lambda e: e.get("atv", 0),
        "reversal_rate": lambda e: e.get("reversal_rate", 0),
        # New signal monitors
        "mean_ser": lambda e: 1.0 - e.get("spectral_summary", {}).get("mean_ser", 1.0),  # invert: higher = worse
        "min_ser": lambda e: 1.0 - e.get("spectral_summary", {}).get("min_ser", 1.0),
        "mean_prediction_error": lambda e: e.get("lps_summary", {}).get("mean_prediction_error", 0),
        "min_coherence": lambda e: -e.get("momentum_summary", {}).get("min_coherence", 1.0),  # invert: lower coherence = worse
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

    # Composite AUROC: max of normalized metrics across all monitors
    try:
        all_metric_fns = {k: v for k, v in metrics.items() if k in results and "auroc" in results.get(k, {})}
        if len(all_metric_fns) >= 3:
            # Normalize each metric to [0,1] using min-max across all episodes
            all_eps = successes + failures
            composite_scores = []
            for ep in all_eps:
                normalized = []
                for name, fn in all_metric_fns.items():
                    vals = [fn(e) for e in all_eps]
                    v = fn(ep)
                    v_min, v_max = min(vals), max(vals)
                    norm = (v - v_min) / (v_max - v_min + 1e-10)
                    normalized.append(norm)
                composite_scores.append(max(normalized))  # max-aggregation

            s_composite = composite_scores[:len(successes)]
            f_composite = composite_scores[len(successes):]
            u, p = mannwhitneyu(f_composite, s_composite, alternative="greater")
            composite_auroc = u / (len(f_composite) * len(s_composite))
            results["composite_max"] = {"auroc": round(composite_auroc, 4), "p_value": round(p, 4)}
    except Exception as ex:
        results["composite_max"] = {"error": str(ex)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="/workspace/results/monitor_analysis.json")
    args = parser.parse_args()

    results = {}

    for model_name, load_fn, use_imagenet in [
        ("vqbet", load_vqbet, False),
        ("diffusion", load_diffusion, True),
    ]:
        print(f"\n{'='*50}")
        print(f"  {model_name} - {args.n_episodes} monitored episodes")
        print(f"{'='*50}")

        policy, stats = load_fn(args.device)
        episodes = []

        for ep in range(args.n_episodes):
            env = gymnasium.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
            env.reset(seed=ep)
            t0 = time.time()
            m = run_monitored_episode(env, policy, args.device, stats, use_imagenet_norm=use_imagenet)
            elapsed = time.time() - t0

            # Compute trajectory-level metrics from per-step data
            per_step = m.pop("per_step")
            m["atv"] = compute_episode_atv(per_step)
            m["reversal_rate"] = compute_episode_reversals(per_step)
            m["episode"] = ep
            m["elapsed_s"] = elapsed
            episodes.append(m)

            st = "OK" if m["success"] else "--"
            stall_str = f"stall={m['stall_summary']['stall_rate']:.2f}" if m['stall_summary']['stall_rate'] > 0 else "stall=0"
            print(f"  Ep {ep:2d}: {st} | cov={m['max_coverage']:.3f} | "
                  f"vel={m['velocity_violations']:3d} | {stall_str} | "
                  f"jerk={m['jerk_summary'].get('mean_jerk', 0):.1f} | {elapsed:.1f}s")
            env.close()

        # Compute AUROCs
        aurocs = compute_aurocs(episodes)

        sr = sum(1 for e in episodes if e["success"]) / len(episodes)
        results[model_name] = {
            "success_rate": sr,
            "n_episodes": len(episodes),
            "total_velocity_violations": sum(e["velocity_violations"] for e in episodes),
            "total_stall_steps": sum(e["stall_summary"]["total_stall_steps"] for e in episodes),
            "mean_jerk_rms": float(np.mean([e["jerk_summary"].get("jerk_rms", 0) for e in episodes])),
            "aurocs": aurocs,
            "episodes": episodes,
        }

        print(f"\n  Summary: SR={sr:.0%}, vel_viol={results[model_name]['total_velocity_violations']}, "
              f"stall_steps={results[model_name]['total_stall_steps']}, "
              f"jerk_rms={results[model_name]['mean_jerk_rms']:.2f}")
        print(f"  AUROCs: {json.dumps(aurocs, indent=2)}")

    # Cross-architecture comparison
    print(f"\n{'='*50}")
    print("CROSS-ARCHITECTURE MONITOR COMPARISON")
    print(f"{'='*50}")
    for name in results:
        r = results[name]
        print(f"  {name}: SR={r['success_rate']:.0%}, vel={r['total_velocity_violations']}, "
              f"stall={r['total_stall_steps']}, jerk_rms={r['mean_jerk_rms']:.2f}")

    output = {
        "experiment": "Monitor Analysis: All 5 violation types on PushT (VQ-BeT + Diffusion)",
        "monitors": ["bounds", "velocity", "stall", "jerk"],
        "contract": {"bounds": [0, 512], "v_max": 30.0},
        "stall_config": {"movement_threshold": 2.0, "stall_window": 10},
        "jerk_config": {"jerk_limit": 50.0},
        "results": results,
    }

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
