#!/bin/bash
# ONE script to set up RunPod for ALOHA experiments
# Handles all the dependency issues from previous attempts
set -e

echo "=== ALOHA Setup (fixed for PyTorch 2.1 template) ==="
echo "Start: $(date)"

# 0. Move all caches to workspace (unlimited storage)
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
export PIP_CACHE_DIR=/workspace/.cache/pip
export MUJOCO_GL=osmesa
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export TORCH_HOME=/workspace/.cache/torch' >> ~/.bashrc
echo 'export XDG_CACHE_HOME=/workspace/.cache' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=/workspace/.cache/pip' >> ~/.bashrc
echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
mkdir -p /workspace/.cache/pip

# 1. Clone repo
cd /workspace
if [ ! -d vla-edge ]; then
    git clone https://github.com/krishnam94/vla-edge.git
fi
cd vla-edge
git pull

# 2. Install deps WITHOUT upgrading PyTorch (the key fix)
echo "Installing dependencies (keeping existing PyTorch)..."
pip install --cache-dir=/workspace/.cache/pip -q \
    "numpy<2" scipy pymunk==6.10.0 PyOpenGL datasets \
    gym-aloha gym-pusht 2>/dev/null

# 3. Install lerobot WITHOUT its PyTorch dependency
echo "Installing lerobot 0.4.4 (no-deps to avoid PyTorch conflict)..."
pip install --cache-dir=/workspace/.cache/pip -q \
    lerobot==0.4.4 --no-deps 2>/dev/null

# 4. Install lerobot's other deps (not PyTorch/torchvision)
pip install --cache-dir=/workspace/.cache/pip -q \
    draccus safetensors diffusers accelerate einops \
    huggingface-hub pyyaml-include 2>/dev/null

# 5. Install vla-edge package
pip install --cache-dir=/workspace/.cache/pip -q -e . 2>/dev/null

# 6. System deps for rendering
apt-get update -qq && apt-get install -y -qq libosmesa6-dev 2>/dev/null

# 7. Pre-download models
echo "Pre-downloading models..."
python -c "
from huggingface_hub import snapshot_download
for m in ['lerobot/act_aloha_sim_transfer_cube_human', 'lerobot/act_aloha_sim_insertion_human']:
    print(f'  {m}...')
    snapshot_download(m)
print('Done.')
"

# 8. Sanity checks
echo ""
echo "=== Sanity Checks ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import numpy as np
print(f'NumPy: {np.__version__}')
import lerobot
print(f'lerobot: {lerobot.__version__}')

# Test ACT load
from lerobot.policies.act.modeling_act import ACTPolicy
p = ACTPolicy.from_pretrained('lerobot/act_aloha_sim_transfer_cube_human')
p.to('cuda'); p.eval()
print(f'ACT: OK ({sum(x.numel() for x in p.parameters()):,} params)')

# Test ALOHA env
import os; os.environ['MUJOCO_GL'] = 'osmesa'
import gym_aloha, gymnasium
env = gymnasium.make('gym_aloha/AlohaTransferCube-v0')
obs, _ = env.reset(seed=42)
print(f'ALOHA env: OK')
env.close()

# Test vla-edge monitors
from vla_edge.validate import SafetyGuard, StallDetector
print(f'vla-edge monitors: OK')

print()
print('ALL CHECKS PASSED - ready to run experiments')
"

echo ""
echo "=== Setup Complete ==="
echo "Disk usage:"
df -h / | tail -1
echo ""
echo "Run: bash experiments/runpod/run_aloha_monitors.sh"
