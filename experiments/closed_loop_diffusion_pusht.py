#!/usr/bin/env python3
"""Closed-loop evaluation: Diffusion Policy on PushT with and without SafetyContract.

Second closed-loop result (first was ACT/ALOHA). PushT is 2D, simple, fast.
Demonstrates SafeContract works across denoising paradigms (DDPM vs flow matching).

Key implementation details:
- lerobot 0.5.0 moved normalization from model submodules to a processor pipeline.
  Pretrained 0.4.x checkpoints lose their normalization buffers on load.
  We manually extract them from the raw safetensors file.
- PushT actions are 2D pixel coordinates (x, y) in [0, 512] workspace.
  Model outputs normalized actions in [-1, 1] (DDPM clip_sample_range=1.0).
  We unnormalize to pixel space before passing to the environment.
- SafetyContract bounds are [0, 512] (workspace limits) with v_max=30 px/step.
- DiffusionPolicy uses action chunking (8 actions per inference, n_obs_steps=2).
  The policy.select_action() method handles the internal queue.
- Success: coverage >= 0.95 (T-block sufficiently overlaps goal).

Usage:
  source .venv/bin/activate
  python experiments/closed_loop_diffusion_pusht.py [--n-episodes 20] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from math import sqrt
from pathlib import Path

import numpy as np

# Block groot import to avoid lerobot 0.5.0 dataclass bug (GR00T N1.5)
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
import gym_pusht  # noqa: F401 - triggers env registration
import gymnasium
from scipy.stats import fisher_exact
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


MODEL_ID = "lerobot/diffusion_pusht"

# Normalization stats from checkpoint
STATE_MIN = np.array([13.456424, 32.938293], dtype=np.float32)
STATE_MAX = np.array([496.14618, 510.9579], dtype=np.float32)
ACTION_MIN = np.array([12.0, 25.0], dtype=np.float32)
ACTION_MAX = np.array([511.0, 511.0], dtype=np.float32)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class NormalizedDiffusion:
    """Diffusion Policy with manual normalization for lerobot 0.5.0 compatibility.

    lerobot 0.5.0 moved normalization from model submodules to a processor
    pipeline. Pretrained 0.4.x checkpoints lose their normalization buffers
    on load. This class applies them manually.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # Load policy - lerobot auto-selects device in config
        self.policy = DiffusionPolicy.from_pretrained(MODEL_ID)
        self.policy.to(self.device)
        self.policy.eval()

        # Verify normalization stats from checkpoint
        ckpt_path = hf_hub_download(MODEL_ID, "model.safetensors")
        state = load_file(ckpt_path)

        # Validate our hardcoded stats match the checkpoint
        assert np.allclose(
            state["normalize_inputs.buffer_observation_state.min"].numpy(),
            STATE_MIN, atol=1e-3
        ), "State MIN mismatch with checkpoint"
        assert np.allclose(
            state["unnormalize_outputs.buffer_action.max"].numpy(),
            ACTION_MAX, atol=1e-3
        ), "Action MAX mismatch with checkpoint"

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Loaded Diffusion Policy on {device}. Params: {n_params:,}")
        print(f"  n_obs_steps={self.policy.config.n_obs_steps}, "
              f"n_action_steps={self.policy.config.n_action_steps}, "
              f"horizon={self.policy.config.horizon}")

    def reset(self):
        self.policy.reset()

    def predict(self, obs_image: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
        """Predict action from observation.

        Args:
            obs_image: (96, 96, 3) uint8 image from PushT render
            agent_pos: (2,) agent x, y position in pixel space

        Returns:
            (2,) action in pixel space [0, 512]
        """
        # Normalize image: uint8 -> [0,1] -> ImageNet mean/std
        img = obs_image.astype(np.float32) / 255.0
        img = (img - IMAGE_MEAN) / IMAGE_STD
        img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Normalize state: pixel -> [-1, 1] via MIN_MAX
        state_norm = (agent_pos.astype(np.float32) - STATE_MIN) / (STATE_MAX - STATE_MIN) * 2 - 1
        state_t = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)

        obs = {
            "observation.image": img_t,
            "observation.state": state_t,
        }

        with torch.inference_mode():
            action_norm = self.policy.select_action(obs)

        if isinstance(action_norm, torch.Tensor):
            action_norm = action_norm.detach().cpu().numpy()

        # Flatten in case of extra dims
        action_norm = action_norm.flatten()[:2]

        # Unnormalize: [-1, 1] -> pixel space
        action_pixel = (action_norm + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
        return action_pixel


def run_episode(
    env,
    model: NormalizedDiffusion,
    safety_contract: dict | None = None,
    max_steps: int = 300,
) -> dict:
    """Run one episode with optional safety contract.

    Args:
        env: PushT gymnasium environment (obs_type=pixels_agent_pos)
        model: NormalizedDiffusion policy
        safety_contract: if provided, dict with action_lo, action_hi, v_max
        max_steps: maximum steps per episode

    Returns:
        Dict with success, coverage, violations, etc.
    """
    obs, info = env.reset()
    model.reset()

    total_reward = 0.0
    max_coverage = 0.0
    success = False
    violations = 0
    actions_modified = 0
    prev_action = None

    for step in range(max_steps):
        image = obs["pixels"]      # (96, 96, 3) uint8
        agent_pos = obs["agent_pos"]  # (2,) float64

        action = model.predict(image, agent_pos)
        original_action = action.copy()

        # Apply safety contract
        if safety_contract is not None:
            lo = safety_contract["action_lo"]
            hi = safety_contract["action_hi"]
            v_max = safety_contract["v_max"]

            # 1. Per-dim bounds clipping (workspace limits)
            clipped = np.clip(action, lo, hi)
            if not np.allclose(action, clipped, atol=1e-7):
                violations += 1
            action = clipped

            # 2. Velocity clamping (smooth action transitions)
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

        if info.get("is_success", False):
            success = True

        if terminated or truncated:
            break

    return {
        "success": success,
        "total_reward": float(total_reward),
        "max_coverage": float(max_coverage),
        "final_coverage": float(coverage),
        "steps": step + 1,
        "violations": violations,
        "actions_modified": actions_modified,
    }


def evaluate(
    n_episodes: int = 20,
    device: str = "mps",
    seed_base: int = 0,
    v_max: float = 30.0,
) -> dict:
    """Run evaluation with and without safety contract.

    Args:
        n_episodes: Number of episodes per condition
        device: torch device
        seed_base: Starting seed for reproducibility
        v_max: velocity limit in pixels/step

    Returns:
        Full results dict (also saved to results/)
    """
    print(f"Loading Diffusion Policy from {MODEL_ID}...")
    model = NormalizedDiffusion(device=device)

    # Safety contract: workspace bounds [0, 512] + velocity limit
    action_lo = np.array([0.0, 0.0], dtype=np.float32)
    action_hi = np.array([512.0, 512.0], dtype=np.float32)

    contract = {
        "action_lo": action_lo,
        "action_hi": action_hi,
        "v_max": v_max,
    }

    print(f"Safety contract: bounds=[0, 512], v_max={v_max} px/step")

    results = {"no_contract": [], "with_contract": []}

    for condition in ["no_contract", "with_contract"]:
        print(f"\n{'='*60}")
        print(f"Running {n_episodes} episodes: {condition}")
        print(f"{'='*60}")

        sc = contract if condition == "with_contract" else None

        for ep in range(n_episodes):
            env = gymnasium.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
            )
            env.reset(seed=seed_base + ep)

            t0 = time.time()
            metrics = run_episode(env, model, safety_contract=sc)
            elapsed = time.time() - t0

            metrics["episode"] = ep
            metrics["seed"] = seed_base + ep
            metrics["elapsed_s"] = elapsed
            results[condition].append(metrics)

            status = "OK" if metrics["success"] else "--"
            cov = metrics["max_coverage"]
            print(
                f"  Ep {ep:2d}: {status} | cov={cov:.3f} | "
                f"steps={metrics['steps']:3d} | viol={metrics['violations']:3d} | "
                f"mod={metrics['actions_modified']:3d} | {elapsed:.1f}s"
            )
            env.close()

    # Summarize
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = {}
    for condition in ["no_contract", "with_contract"]:
        eps = results[condition]
        successes = sum(1 for e in eps if e["success"])
        sr = successes / len(eps) if eps else 0.0
        n = len(eps)

        # Wilson 95% CI
        z = 1.96
        if n > 0 and sr > 0 and sr < 1:
            ci_lo = (sr + z * z / (2 * n) - z * sqrt(sr * (1 - sr) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            ci_hi = (sr + z * z / (2 * n) + z * sqrt(sr * (1 - sr) / n + z * z / (4 * n * n))) / (1 + z * z / n)
        elif sr == 0:
            ci_lo = 0.0
            ci_hi = z * z / (n + z * z) if n > 0 else 0.0
        else:  # sr == 1
            ci_lo = 1.0 - z * z / (n + z * z) if n > 0 else 1.0
            ci_hi = 1.0

        summary[condition] = {
            "success_rate": sr,
            "successes": successes,
            "total_episodes": n,
            "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
            "avg_reward": float(np.mean([e["total_reward"] for e in eps])),
            "avg_max_coverage": float(np.mean([e["max_coverage"] for e in eps])),
            "avg_final_coverage": float(np.mean([e["final_coverage"] for e in eps])),
            "total_violations": sum(e["violations"] for e in eps),
            "total_actions_modified": sum(e["actions_modified"] for e in eps),
            "avg_episode_time_s": float(np.mean([e["elapsed_s"] for e in eps])),
        }

        print(f"\n{condition}:")
        print(f"  Success rate: {successes}/{n} ({sr*100:.0f}%) [95% CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%]")
        print(f"  Avg max coverage: {summary[condition]['avg_max_coverage']:.3f}")
        print(f"  Avg final coverage: {summary[condition]['avg_final_coverage']:.3f}")
        print(f"  Avg reward:   {summary[condition]['avg_reward']:.1f}")
        if condition == "with_contract":
            print(f"  Violations:   {summary[condition]['total_violations']}")
            print(f"  Actions modified: {summary[condition]['total_actions_modified']}")
        print(f"  Avg time/ep:  {summary[condition]['avg_episode_time_s']:.1f}s")

    # Statistical test
    a = summary["with_contract"]["successes"]
    b = summary["with_contract"]["total_episodes"] - a
    c = summary["no_contract"]["successes"]
    d = summary["no_contract"]["total_episodes"] - c
    _, p_value = fisher_exact([[a, b], [c, d]])

    sr_no = summary["no_contract"]["success_rate"]
    sr_with = summary["with_contract"]["success_rate"]
    delta = sr_with - sr_no

    print(f"\nKEY RESULT: SafetyContract {'DOES NOT degrade' if sr_with >= sr_no - 0.05 else 'may DEGRADE'} performance")
    print(f"  Without: {sr_no*100:.0f}%  |  With: {sr_with*100:.0f}%  |  Delta: {delta*100:+.0f}%")
    print(f"  Fisher's exact p={p_value:.3f} ({'significant' if p_value < 0.05 else 'not significant'})")

    summary["fisher_p_value"] = float(p_value)

    # Save
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"closed_loop_diffusion_pusht_{n_episodes}ep.json"

    output = {
        "experiment": "EXP-PUSHT-CL: Diffusion Policy Closed-Loop on PushT",
        "summary": summary,
        "episodes": results,
        "config": {
            "model": MODEL_ID,
            "model_params": 262_709_026,
            "architecture": "DDPM Diffusion (100 denoising steps, ResNet18 vision, 1D U-Net)",
            "env": "gym_pusht/PushT-v0",
            "obs_type": "pixels_agent_pos",
            "action_dim": 2,
            "action_space": "pixel [0, 512]",
            "normalization": "MIN_MAX (state/action), MEAN_STD (image)",
            "n_episodes": n_episodes,
            "seed_base": seed_base,
            "max_steps": 300,
            "device": device,
            "safety_contract": {
                "action_lo": action_lo.tolist(),
                "action_hi": action_hi.tolist(),
                "v_max": v_max,
                "description": "Workspace bounds [0, 512] + velocity clamp",
            },
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXP-PUSHT-CL: Diffusion Policy closed-loop on PushT"
    )
    parser.add_argument("--n-episodes", type=int, default=20, help="Episodes per condition")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, mps")
    parser.add_argument("--seed-base", type=int, default=0, help="Starting seed")
    parser.add_argument("--v-max", type=float, default=30.0, help="Velocity limit (px/step)")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    print(f"Using device: {args.device}")
    evaluate(
        n_episodes=args.n_episodes,
        device=args.device,
        seed_base=args.seed_base,
        v_max=args.v_max,
    )
