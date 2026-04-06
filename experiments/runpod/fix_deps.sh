#!/bin/bash
# Fix RunPod dependencies after setup.sh
# Run this before run_all.sh

set -e
echo "=== Fixing Dependencies ==="

# Fix 1: PushT env needs pymunk 6.x (not 7.x)
echo "Fixing pymunk version..."
pip install -q pymunk==6.10.0

# Fix 2: Verify PushT env works
echo "Testing PushT env..."
python -c "
import gym_pusht, gymnasium
env = gymnasium.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos', render_mode='rgb_array')
obs, _ = env.reset(seed=42)
print(f'  PushT OK: obs keys = {list(obs.keys())}')
env.close()
"

# Fix 3: Verify VQ-BeT loads with manual config patch
echo "Testing VQ-BeT (manual config)..."
python -c "
import json, torch
from dataclasses import fields as dc_fields, MISSING
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

config_path = hf_hub_download('lerobot/vqbet_pusht', 'config.json')
with open(config_path) as f: raw = json.load(f)
for field in ['mlp_hidden_dim', 'type']: raw.pop(field, None)

def convert_features(d):
    return {k: PolicyFeature(type=FeatureType[v['type']], shape=v['shape']) for k, v in d.items()}
raw['input_features'] = convert_features(raw['input_features'])
raw['output_features'] = convert_features(raw['output_features'])

valid = {f.name for f in dc_fields(VQBeTConfig)}
clean = {k: v for k, v in raw.items() if k in valid}
config = VQBeTConfig.__new__(VQBeTConfig)
for f in dc_fields(VQBeTConfig):
    if f.name in clean: setattr(config, f.name, clean[f.name])
    elif f.default is not MISSING: setattr(config, f.name, f.default)
    elif f.default_factory is not MISSING: setattr(config, f.name, f.default_factory())

policy = VQBeTPolicy(config)
state = load_file(hf_hub_download('lerobot/vqbet_pusht', 'model.safetensors'))
missing, unexpected = policy.load_state_dict(state, strict=False)
policy.to('cuda'); policy.eval()
n = sum(p.numel() for p in policy.parameters())
print(f'  VQ-BeT OK: {n:,} params on CUDA ({len(missing)} missing, {len(unexpected)} unexpected keys)')

# Quick inference test
import gymnasium, gym_pusht, numpy as np
env = gymnasium.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos', render_mode='rgb_array')
obs, _ = env.reset(seed=42)
policy.reset()

# Get unnorm stats
action_min = state.get('unnormalize_outputs.buffer_action.min', state.get('normalize_targets.buffer_action.min'))
action_max = state.get('unnormalize_outputs.buffer_action.max', state.get('normalize_targets.buffer_action.max'))

img = torch.from_numpy(obs['pixels']).float().to('cuda') / 255.0
img = img.permute(2, 0, 1).unsqueeze(0)
st = torch.from_numpy(obs['agent_pos']).float().to('cuda').unsqueeze(0)

with torch.inference_mode():
    action = policy.select_action({'observation.image': img, 'observation.state': st})

a = action.detach().cpu().numpy()
if a.ndim > 1: a = a[0]

# Check if normalized
is_norm = a.max() < 2.0 and a.min() > -2.0
if is_norm and action_min is not None:
    a_unnorm = a * (action_max.numpy() - action_min.numpy()) + action_min.numpy()
    print(f'  Action (normalized): {a.round(3)}, unnormalized: {a_unnorm.round(1)}')
else:
    print(f'  Action (raw): {a.round(1)}')

env.close()
print('  VQ-BeT inference: PASSED')
"

# Fix 4: Verify ALOHA still works
echo "Testing ALOHA env..."
python -c "
import gym_aloha, gymnasium
env = gymnasium.make('gym_aloha/AlohaTransferCube-v0')
obs, _ = env.reset(seed=42)
print(f'  ALOHA OK')
env.close()
"

echo ""
echo "=== All Fixes Applied ==="
echo "Run: bash experiments/runpod/run_all.sh"
