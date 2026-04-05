#!/usr/bin/env python3
"""Quick test: Load Diffusion Policy and run one inference on PushT."""

import sys
import types
import numpy as np
import torch

# Block groot import (same pattern as closed_loop_eval.py)
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

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_ID = "lerobot/diffusion_pusht"

print("Loading Diffusion Policy...")
policy = DiffusionPolicy.from_pretrained(MODEL_ID)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
policy.to(device)
policy.eval()

n_params = sum(p.numel() for p in policy.parameters())
print(f"Loaded on {device}. Params: {n_params:,}")

# Check config
print(f"\nPolicy config (all attrs):")
cfg = policy.config
for attr in sorted(dir(cfg)):
    if not attr.startswith("_"):
        try:
            val = getattr(cfg, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except Exception:
            pass

# Check normalization stats
ckpt_path = hf_hub_download(MODEL_ID, "model.safetensors")
state = load_file(ckpt_path)
norm_keys = [k for k in state.keys() if "normalize" in k or "unnormalize" in k]
print(f"\nNormalization keys:")
for k in norm_keys:
    t = state[k]
    print(f"  {k}: shape={t.shape}, values={t.tolist()}")

# Test inference with proper obs for PushT
policy.reset()
print("\nTest inference...")

# PushT expects: observation.image (1, 3, 96, 96) and observation.state (1, 2)
# Need to check n_obs_steps - Diffusion uses observation history
n_obs = getattr(cfg, "n_obs_steps", 2)
print(f"  n_obs_steps: {n_obs}")

# Build dummy observation
obs = {
    "observation.image": torch.randn(1, 3, 96, 96).to(device),
    "observation.state": torch.randn(1, 2).to(device),
}
print(f"  Obs keys: {list(obs.keys())}")

with torch.inference_mode():
    action = policy.select_action(obs)

if isinstance(action, torch.Tensor):
    action_np = action.detach().cpu().numpy()
else:
    action_np = np.array(action)

print(f"  Output action: shape={action_np.shape}, min={action_np.min():.4f}, max={action_np.max():.4f}")
print(f"\nSUCCESS")
