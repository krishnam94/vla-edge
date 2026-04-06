#!/bin/bash
# Test if ACT on PushT works (community checkpoint)
# Run this while Diffusion is still going

set -e
export MUJOCO_GL=osmesa

echo "=== Testing ACT on PushT ==="

# Test 1: Can we load the model?
echo "Loading Menoking/act-pusht..."
python -c "
from lerobot.policies.act.modeling_act import ACTPolicy
p = ACTPolicy.from_pretrained('Menoking/act-pusht')
n = sum(x.numel() for x in p.parameters())
print(f'  ACT PushT loaded: {n:,} params')
p.to('cuda'); p.eval()
print('  CUDA: OK')
p.reset()
print('  Reset: OK')
"

# Test 2: Can we run 3 episodes?
echo ""
echo "Running 3 test episodes..."
python -c "
import torch, numpy as np, gymnasium, gym_pusht
from lerobot.policies.act.modeling_act import ACTPolicy
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL = 'Menoking/act-pusht'
policy = ACTPolicy.from_pretrained(MODEL)
policy.to('cuda'); policy.eval()

# Check normalization buffers
state = load_file(hf_hub_download(MODEL, 'model.safetensors'))
norm_keys = [k for k in state if 'normalize' in k.lower() or 'buffer' in k.lower()]
print(f'  Norm keys: {len(norm_keys)}')
for k in sorted(norm_keys):
    print(f'    {k}: {state[k].shape}')

# Get state normalization
state_min = state.get('normalize_inputs.buffer_observation_state.min')
state_max = state.get('normalize_inputs.buffer_observation_state.max')
action_min = state.get('unnormalize_outputs.buffer_action.min')
action_max = state.get('unnormalize_outputs.buffer_action.max')

if state_min is not None:
    print(f'  State norm: min={state_min.numpy().round(1)}, max={state_max.numpy().round(1)}')
if action_min is not None:
    print(f'  Action unnorm: min={action_min.numpy().round(1)}, max={action_max.numpy().round(1)}')

# Check what observation keys ACT expects
print(f'  Config input features:')
for k in dir(policy.config):
    if 'feature' in k.lower() or 'input' in k.lower():
        val = getattr(policy.config, k, None)
        if val and not callable(val):
            print(f'    {k}: {val}')

# Try running episodes
for seed in [42, 100, 200]:
    env = gymnasium.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos', render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    policy.reset()

    success = False
    max_cov = 0
    for step in range(300):
        img = torch.from_numpy(obs['pixels']).float().to('cuda') / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        agent_pos = obs['agent_pos'].astype(np.float32)
        if state_min is not None:
            s_min, s_max = state_min.numpy(), state_max.numpy()
            agent_pos = (agent_pos - s_min) / (s_max - s_min + 1e-8) * 2 - 1
        state_t = torch.from_numpy(agent_pos).float().to('cuda').unsqueeze(0)

        with torch.inference_mode():
            action = policy.select_action({'observation.image': img, 'observation.state': state_t})

        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1: action_np = action_np[0]

        # Check if normalized
        if action_np.max() < 2.0 and action_min is not None:
            a_min, a_max = action_min.numpy(), action_max.numpy()
            action_np = (action_np + 1) / 2 * (a_max - a_min) + a_min

        obs, reward, terminated, truncated, info = env.step(action_np.astype(np.float32))
        cov = info.get('coverage', 0)
        max_cov = max(max_cov, cov)
        if cov >= 0.95:
            success = True
            break
        if terminated or truncated:
            break

    env.close()
    status = 'OK' if success else '--'
    print(f'  Seed {seed}: {status} | max_cov={max_cov:.3f} | steps={step+1}')

print()
print('If success > 0%, ACT on PushT works!')
print('Run full eval with: python -u experiments/runpod/exp_act_pusht_gpu.py --n-episodes 200')
"
