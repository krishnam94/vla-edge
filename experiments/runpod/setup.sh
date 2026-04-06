#!/bin/bash
# RunPod A100 Setup Script for SafeContract GPU Experiments
# Run this first after launching the instance
# Estimated time: ~15 min

set -e
echo "=== SafeContract RunPod Setup ==="
echo "Time: $(date)"

# 1. System dependencies for headless rendering
echo "Installing system deps..."
apt-get update -qq && apt-get install -y -qq libosmesa6-dev libgl1-mesa-glx libglfw3 > /dev/null 2>&1
export MUJOCO_GL=osmesa
echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc

# 2. Install lerobot 0.4.4 (the version checkpoints were trained with)
echo "Installing lerobot 0.4.4..."
pip install -q "lerobot==0.4.4" 2>/dev/null || \
  pip install -q "lerobot @ git+https://github.com/huggingface/lerobot.git@v0.4.4" 2>/dev/null || \
  echo "WARNING: lerobot 0.4.4 failed, trying latest..." && pip install -q lerobot

# 3. Install environment packages
echo "Installing gym envs..."
pip install -q gym-pusht gym-aloha scipy datasets huggingface-hub safetensors

# 4. Clone vla-edge repo
echo "Cloning vla-edge..."
if [ ! -d /workspace/vla-edge ]; then
    git clone https://github.com/krishnam94/vla-edge.git /workspace/vla-edge
fi
cd /workspace/vla-edge

# 4b. Install vla-edge package (for violations, conformal, monitor modules)
echo "Installing vla-edge package..."
pip install -q -e . 2>/dev/null || echo "WARNING: vla-edge install failed, using sys.path fallback"

# 5. Create results directory
mkdir -p /workspace/results

# 6. Pre-download models
echo "Pre-downloading models..."
python -c "
from huggingface_hub import snapshot_download
for m in ['lerobot/vqbet_pusht', 'lerobot/diffusion_pusht', 'lerobot/act_aloha_sim_transfer_cube_human', 'lerobot/act_aloha_sim_insertion_human']:
    print(f'  Downloading {m}...')
    snapshot_download(m)
print('All models cached.')
"

# 7. Sanity checks
echo ""
echo "=== Sanity Checks ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')

import lerobot
print(f'lerobot: {lerobot.__version__}')

# Test VQ-BeT load
try:
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
    p = VQBeTPolicy.from_pretrained('lerobot/vqbet_pusht')
    p.to('cuda'); p.eval()
    print(f'VQ-BeT: OK ({sum(x.numel() for x in p.parameters()):,} params)')
except Exception as e:
    print(f'VQ-BeT: FAILED ({e})')

# Test Diffusion load
try:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    p = DiffusionPolicy.from_pretrained('lerobot/diffusion_pusht')
    p.to('cuda'); p.eval()
    print(f'Diffusion: OK ({sum(x.numel() for x in p.parameters()):,} params)')
except Exception as e:
    print(f'Diffusion: FAILED ({e})')

# Test ACT load
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    p = ACTPolicy.from_pretrained('lerobot/act_aloha_sim_transfer_cube_human')
    p.to('cuda'); p.eval()
    print(f'ACT: OK ({sum(x.numel() for x in p.parameters()):,} params)')
except Exception as e:
    print(f'ACT: FAILED ({e})')

# Test PushT env
try:
    import gym_pusht
    import gymnasium
    env = gymnasium.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos', render_mode='rgb_array')
    obs, _ = env.reset(seed=42)
    print(f'PushT env: OK (obs keys: {list(obs.keys())})')
    env.close()
except Exception as e:
    print(f'PushT env: FAILED ({e})')

# Test ALOHA env
try:
    import gym_aloha
    import gymnasium
    env = gymnasium.make('gym_aloha/AlohaTransferCube-v0')
    obs, _ = env.reset(seed=42)
    print(f'ALOHA env: OK')
    env.close()
except Exception as e:
    print(f'ALOHA env: FAILED ({e})')
"

echo ""
echo "=== Setup Complete ==="
echo "Run: bash /workspace/vla-edge/experiments/runpod/run_all.sh"
