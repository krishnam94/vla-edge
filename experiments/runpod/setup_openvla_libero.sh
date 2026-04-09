#!/bin/bash
# RunPod A100 Setup for OpenVLA + LIBERO Experiments
# Installs LIBERO env, OpenVLA deps, downloads checkpoint, runs sanity checks.
#
# OpenVLA is a 7B VLA that uses 256-bin action tokenization.
# LIBERO is a robosuite-based benchmark with 4 task suites.
# We use openvla-7b-finetuned-libero-10 (LIBERO-Long, 10 tasks).
#
# Expected time: ~20 min (mostly model download)
# Expected disk: ~25 GB (model ~15GB, LIBERO ~2GB, deps ~8GB)

set -e

echo "=== OpenVLA + LIBERO Setup ==="
echo "Start: $(date)"

# ---------- 0. Move all caches to workspace (unlimited storage) ----------
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
export PIP_CACHE_DIR=/workspace/.cache/pip
export MUJOCO_GL=osmesa
cat >> ~/.bashrc << 'ENV'
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
export PIP_CACHE_DIR=/workspace/.cache/pip
export MUJOCO_GL=osmesa
ENV
mkdir -p /workspace/.cache/pip

# ---------- 1. System deps for headless rendering ----------
echo "Installing system deps..."
apt-get update -qq && apt-get install -y -qq \
    libosmesa6-dev libgl1-mesa-glx libglfw3 git-lfs > /dev/null 2>&1

# ---------- 2. Clone vla-edge repo ----------
cd /workspace
if [ ! -d vla-edge ]; then
    git clone https://github.com/krishnam94/vla-edge.git
fi
cd vla-edge
git pull

# ---------- 3. Install LIBERO and robosuite ----------
# LIBERO depends on robosuite (MuJoCo-based sim).
# We install from source to get the BDDL task files.
echo "Installing robosuite..."
pip install --cache-dir=/workspace/.cache/pip -q \
    "robosuite>=1.4" mujoco 2>/dev/null

echo "Cloning and installing LIBERO..."
cd /workspace
if [ ! -d LIBERO ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd LIBERO
pip install --cache-dir=/workspace/.cache/pip -q -e . 2>/dev/null || {
    echo "LIBERO pip install failed, installing deps manually..."
    pip install --cache-dir=/workspace/.cache/pip -q \
        h5py pyyaml egl-probe bddl 2>/dev/null
    pip install --cache-dir=/workspace/.cache/pip -q --no-deps -e . 2>/dev/null
}
cd /workspace/vla-edge

# ---------- 4. Install OpenVLA inference deps ----------
# OpenVLA uses HuggingFace transformers (AutoModelForVision2Seq).
# The model weights include custom modeling code (trust_remote_code=True).
# flash-attn is optional but recommended for speed.
echo "Installing OpenVLA inference deps..."
pip install --cache-dir=/workspace/.cache/pip -q \
    "transformers>=4.40" accelerate safetensors \
    "huggingface-hub>=0.25" Pillow scipy datasets \
    "numpy<2" 2>/dev/null

# Try flash-attn (may fail on older CUDA, that's OK)
echo "Attempting flash-attn install (optional)..."
pip install --cache-dir=/workspace/.cache/pip -q flash-attn --no-build-isolation 2>/dev/null || \
    echo "  flash-attn not available, will use eager attention (slower but works)"

# ---------- 5. Install vla-edge package ----------
echo "Installing vla-edge..."
cd /workspace/vla-edge
pip install --cache-dir=/workspace/.cache/pip -q -e . 2>/dev/null

# Force numpy<2 LAST (transitive deps may upgrade it)
pip install --cache-dir=/workspace/.cache/pip -q "numpy<2" 2>/dev/null

# ---------- 6. Download LIBERO benchmark data ----------
# LIBERO needs its BDDL task definition files and initial states.
# The demo datasets are downloaded by the experiment script on-demand.
echo "Downloading LIBERO benchmark data..."
cd /workspace/LIBERO
python -c "
from libero.libero import benchmark
# This triggers downloading the benchmark data (BDDL files + init states)
for suite_name in ['libero_10']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    print(f'  {suite_name}: {suite.n_tasks} tasks loaded')
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        print(f'    Task {i}: {task.language}')
" || echo "WARNING: LIBERO benchmark data download failed - will retry in experiment"

# ---------- 7. Pre-download OpenVLA model ----------
echo "Pre-downloading OpenVLA model (openvla-7b-finetuned-libero-10)..."
echo "  This is ~15 GB and may take 5-10 minutes..."
python -c "
from huggingface_hub import snapshot_download
model_id = 'openvla/openvla-7b-finetuned-libero-10'
print(f'  Downloading {model_id}...')
path = snapshot_download(model_id)
print(f'  Cached at: {path}')
print('Done.')
"

# ---------- 8. Create results directory ----------
mkdir -p /workspace/results

# ---------- 9. Sanity checks ----------
echo ""
echo "=== Sanity Checks ==="

# Check PyTorch + CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Check transformers version
python -c "
import transformers
print(f'Transformers: {transformers.__version__}')
"

# Check LIBERO environment
MUJOCO_GL=osmesa python -c "
import robosuite
print(f'Robosuite: {robosuite.__version__}')

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import os
os.environ['MUJOCO_GL'] = 'osmesa'

suite = benchmark.get_benchmark_dict()['libero_10']()
task = suite.get_task(0)
print(f'LIBERO-10: {suite.n_tasks} tasks')
print(f'  Task 0: {task.language}')

# Create env and check obs/action space
bddl_file = os.path.join(
    benchmark.get_libero_path('bddl_files'),
    task.problem_folder, task.bddl_file
)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
print(f'  Obs keys: {list(obs.keys())}')
print(f'  agentview_image shape: {obs[\"agentview_image\"].shape}')
print(f'  Action dim: {env.action_spec[0].shape}')

# Take a dummy step
action = [0, 0, 0, 0, 0, 0, -1]  # 7-DOF: delta EEF xyz + rpy + gripper
obs, reward, done, info = env.step(action)
print(f'  Dummy step: reward={reward}, done={done}')
env.close()
print('LIBERO env: OK')
"

# Check OpenVLA model loads
python -c "
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

model_id = 'openvla/openvla-7b-finetuned-libero-10'
print(f'Loading OpenVLA processor from {model_id}...')
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
print('  Processor: OK')

print(f'Loading OpenVLA model (bf16)...')
try:
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to('cuda')
except Exception:
    print('  flash_attention_2 failed, falling back to eager...')
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to('cuda')
vla.eval()
n = sum(p.numel() for p in vla.parameters())
print(f'  OpenVLA: OK ({n:,} params on cuda)')

# Check norm_stats exist
if hasattr(vla, 'norm_stats') and vla.norm_stats:
    keys = list(vla.norm_stats.keys())
    print(f'  norm_stats keys: {keys}')
    for k in keys:
        stats = vla.norm_stats[k]['action']
        print(f'    {k} q01: {[round(x,3) for x in stats[\"q01\"]]}')
        print(f'    {k} q99: {[round(x,3) for x in stats[\"q99\"]]}')
else:
    print('  WARNING: no norm_stats found - will need dataset_statistics.json')

# Quick inference test with dummy image
from PIL import Image
import numpy as np
dummy_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
prompt = 'In: What action should the robot take to pick up the object?\\nOut:'
inputs = processor(prompt, dummy_img).to('cuda', dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key='libero_10', do_sample=False)
print(f'  Test action: {[round(float(x), 4) for x in action]}')
print(f'  Action shape: {action.shape}, range: [{action.min():.4f}, {action.max():.4f}]')
print('OpenVLA inference: OK')
"

# Check vla-edge monitors
python -c "
from vla_edge.validate.violations import StallDetector, JerkMonitor
from vla_edge.validate.signals import (
    SpectralEnergyMonitor, LinearPredictabilityMonitor, MomentumCoherenceMonitor
)
print('vla-edge monitors: OK')
"

echo ""
echo "=== Setup Complete ==="
echo "Disk usage:"
df -h / | tail -1
du -sh /workspace/.cache/huggingface/ 2>/dev/null || true
echo ""
echo "Run: python /workspace/vla-edge/experiments/runpod/exp_openvla_libero_gpu.py"
echo "  Options: --n-episodes 20 --task-suite libero_10 --output /workspace/results/openvla_libero.json"
