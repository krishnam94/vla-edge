#!/usr/bin/env python3
"""Quick test: PushT gym environment interface."""

import gym_pusht  # noqa: F401 - triggers env registration
import gymnasium
import numpy as np

# Test 1: state obs type
print("=== Test state obs ===")
env = gymnasium.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
obs, info = env.reset(seed=42)
print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
print(f"Obs values: {obs}")
print(f"Action space: {env.action_space}")
a = env.action_space.sample()
print(f"Sample action: {a}")
o2, r, t, tr, i2 = env.step(a)
print(f"Reward: {r}, Info: {i2}")
print(f"MaxSteps: {env.spec.max_episode_steps}")
env.close()

# Test 2: pixels_agent_pos (needs render)
print("\n=== Test pixels_agent_pos obs ===")
env2 = gymnasium.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
obs2, info2 = env2.reset(seed=42)
print(f"Obs type: {type(obs2)}")
for k, v in obs2.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.2f}, max={v.max():.2f}")
env2.close()

print("\nSUCCESS - PushT gym is working")
