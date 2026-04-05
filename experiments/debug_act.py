#!/usr/bin/env python3
"""Debug ACT policy actions in ALOHA sim."""

from __future__ import annotations

import sys
import types

import numpy as np

# Block groot
gm = types.ModuleType("lerobot.policies.groot"); gm.__path__ = []; sys.modules["lerobot.policies.groot"] = gm
cm = types.ModuleType("lerobot.policies.groot.configuration_groot"); cm.GrootConfig = type("GC", (), {}); sys.modules["lerobot.policies.groot.configuration_groot"] = cm
mm = types.ModuleType("lerobot.policies.groot.modeling_groot"); mm.GrootPolicy = type("GP", (), {}); sys.modules["lerobot.policies.groot.modeling_groot"] = mm
gm.GrootConfig = cm.GrootConfig; gm.GrootPolicy = mm.GrootPolicy

import torch
import gym_aloha  # noqa: F401
import gymnasium
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from lerobot.policies.act.modeling_act import ACTPolicy

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load policy and normalization stats
policy = ACTPolicy.from_pretrained(MODEL_ID)
policy.to(device)
policy.eval()

ckpt_path = hf_hub_download(MODEL_ID, "model.safetensors")
state = load_file(ckpt_path)

img_mean = state["normalize_inputs.buffer_observation_images_top.mean"]
img_std = state["normalize_inputs.buffer_observation_images_top.std"]
state_mean = state["normalize_inputs.buffer_observation_state.mean"]
state_std = state["normalize_inputs.buffer_observation_state.std"]
action_mean = state["unnormalize_outputs.buffer_action.mean"]
action_std = state["unnormalize_outputs.buffer_action.std"]

print("State mean:", state_mean.numpy())
print("State std:", state_std.numpy())
print("Action mean:", action_mean.numpy())
print("Action std:", action_std.numpy())

env = gymnasium.make("gym_aloha/AlohaTransferCube-v0")
obs, _ = env.reset(seed=42)

# Get qpos
qpos = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics)
print("\nInitial qpos (14):", qpos)
print("qpos range:", qpos.min(), qpos.max())

# Try approach 1: with normalization
img = torch.from_numpy(obs["top"]).float() / 255.0
img = img.permute(2, 0, 1)
img_norm = (img - img_mean) / (img_std + 1e-8)
state_norm = (torch.from_numpy(qpos).float() - state_mean) / (state_std + 1e-8)

obs_dict = {
    "observation.images.top": img_norm.unsqueeze(0).to(device),
    "observation.state": state_norm.unsqueeze(0).to(device),
}

policy.reset()
with torch.inference_mode():
    action_raw = policy.select_action(obs_dict)

action_raw = action_raw.numpy() if isinstance(action_raw, torch.Tensor) else action_raw
print("\nApproach 1 (with normalization):")
print("  Raw model output:", action_raw.flatten()[:7])
print("  Output range:", action_raw.min(), action_raw.max())

# Unnormalize
action_unnorm = action_raw.flatten() * action_std.numpy() + action_mean.numpy()
print("  Unnormalized:", action_unnorm[:7])
print("  Unnorm range:", action_unnorm.min(), action_unnorm.max())

# Try approach 2: WITHOUT normalization (raw image [0,1] + raw qpos)
obs_dict2 = {
    "observation.images.top": img.unsqueeze(0).to(device),
    "observation.state": torch.from_numpy(qpos).float().unsqueeze(0).to(device),
}

policy.reset()
with torch.inference_mode():
    action_raw2 = policy.select_action(obs_dict2)

action_raw2 = action_raw2.numpy() if isinstance(action_raw2, torch.Tensor) else action_raw2
print("\nApproach 2 (WITHOUT normalization - raw input):")
print("  Raw model output:", action_raw2.flatten()[:7])
print("  Output range:", action_raw2.min(), action_raw2.max())

# Compare with qpos (if model is doing identity-like prediction, they should be close)
print("\nExpected action close to qpos:", qpos[:7])
print("Diff approach 1 unnorm vs qpos:", np.abs(action_unnorm[:7] - qpos[:7]))
print("Diff approach 2 raw vs qpos:", np.abs(action_raw2.flatten()[:7] - qpos[:7]))

# Try running 5 steps with approach 2 to see if reward changes
print("\nRunning 5 steps with approach 2 (no norm)...")
policy.reset()
env.reset(seed=42)
for i in range(5):
    qpos = env.unwrapped._env.task.get_qpos(env.unwrapped._env.physics)
    obs_dict = {
        "observation.images.top": (torch.from_numpy(obs["top"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0).to(device),
        "observation.state": torch.from_numpy(qpos).float().unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        action = policy.select_action(obs_dict)
    action = action.numpy().flatten() if isinstance(action, torch.Tensor) else action.flatten()
    action_clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
    obs, reward, term, trunc, info = env.step(action_clipped)
    print(f"  Step {i}: action_range=[{action.min():.3f},{action.max():.3f}] reward={reward} qpos_first3={qpos[:3].round(3)}")

env.close()
