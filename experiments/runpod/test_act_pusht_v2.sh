#!/bin/bash
# Test ACT on PushT - try multiple community checkpoints
set -e
export MUJOCO_GL=osmesa

echo "=== Testing ACT on PushT (multiple checkpoints) ==="

python -c "
import torch, numpy as np, gymnasium, gym_pusht
from lerobot.policies.act.modeling_act import ACTPolicy
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

CANDIDATES = [
    'aadarshram/act_pusht',
    'pepijn223/act-pusht',
    'lepao/act_pusht_best',
    'wlinhugging/lerobot-pusht-act-policy',
    'thehui/act_pusht_model',
    'qualia-robotics/act-pusht-575fb811',
]

for MODEL in CANDIDATES:
    print(f'\n--- Testing {MODEL} ---')
    try:
        policy = ACTPolicy.from_pretrained(MODEL)
        n = sum(x.numel() for x in policy.parameters())
        policy.to('cuda'); policy.eval()
        print(f'  Loaded: {n:,} params')

        # Get norm stats
        state = load_file(hf_hub_download(MODEL, 'model.safetensors'))
        state_min = state.get('normalize_inputs.buffer_observation_state.min')
        state_max = state.get('normalize_inputs.buffer_observation_state.max')
        action_min = state.get('unnormalize_outputs.buffer_action.min')
        action_max = state.get('unnormalize_outputs.buffer_action.max')
        action_mean = state.get('unnormalize_outputs.buffer_action.mean')
        action_std = state.get('unnormalize_outputs.buffer_action.std')

        norm_type = 'MEAN_STD' if action_mean is not None else 'MIN_MAX' if action_min is not None else 'NONE'
        print(f'  Action norm: {norm_type}')
        if state_min is not None:
            print(f'  State: min={state_min.numpy().round(1)}, max={state_max.numpy().round(1)}')

        # Quick 3-episode test
        successes = 0
        for seed in [42, 100, 200]:
            env = gymnasium.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos', render_mode='rgb_array')
            obs, _ = env.reset(seed=seed)
            policy.reset()
            max_cov = 0

            for step in range(300):
                img = torch.from_numpy(obs['pixels']).float().to('cuda') / 255.0
                # Try ImageNet normalization (needed for ResNet backbone)
                IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
                IMG_STD = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
                img_np = obs['pixels'].astype(np.float32) / 255.0
                img_np = (img_np - IMG_MEAN) / IMG_STD
                img = torch.from_numpy(img_np.astype(np.float32)).to('cuda')
                img = img.permute(2, 0, 1).unsqueeze(0)

                agent_pos = obs['agent_pos'].astype(np.float32)
                if state_min is not None:
                    s_min, s_max = state_min.numpy(), state_max.numpy()
                    agent_pos = (agent_pos - s_min) / (s_max - s_min + 1e-8) * 2 - 1
                st = torch.from_numpy(agent_pos).float().to('cuda').unsqueeze(0)

                with torch.inference_mode():
                    action = policy.select_action({'observation.image': img, 'observation.state': st})

                a = action.detach().cpu().numpy()
                if a.ndim > 1: a = a[0]

                # Unnormalize
                if norm_type == 'MEAN_STD':
                    a = a * action_std.numpy() + action_mean.numpy()
                elif norm_type == 'MIN_MAX':
                    a = (a + 1) / 2 * (action_max.numpy() - action_min.numpy()) + action_min.numpy()

                obs, reward, term, trunc, info = env.step(a.astype(np.float32))
                cov = info.get('coverage', 0)
                max_cov = max(max_cov, cov)
                if cov >= 0.95:
                    successes += 1
                    break
                if term or trunc: break

            env.close()
            status = 'OK' if max_cov >= 0.95 else '--'
            print(f'  Seed {seed}: {status} | cov={max_cov:.3f}')

        print(f'  Result: {successes}/3 successes')
        if successes > 0:
            print(f'  >>> WINNER: {MODEL} works! Use this for full eval.')
            break

    except Exception as e:
        print(f'  FAILED: {type(e).__name__}: {str(e)[:100]}')
        continue

print()
print('Done testing ACT checkpoints.')
"
