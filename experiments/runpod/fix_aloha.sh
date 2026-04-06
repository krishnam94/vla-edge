#!/bin/bash
# Fix ALOHA OpenGL rendering on headless RunPod
set -e

echo "=== Fixing ALOHA OpenGL ==="

# Try EGL first (preferred for headless GPU)
pip install -q PyOpenGL PyOpenGL-accelerate 2>/dev/null

# Try EGL
echo "Trying EGL..."
export MUJOCO_GL=egl
if python -c "
import os; os.environ['MUJOCO_GL'] = 'egl'
import gym_aloha, gymnasium
env = gymnasium.make('gym_aloha/AlohaTransferCube-v0')
obs, _ = env.reset(seed=42)
print('  ALOHA OK with EGL')
env.close()
" 2>/dev/null; then
    echo "export MUJOCO_GL=egl" >> ~/.bashrc
    echo "EGL works. Added to bashrc."
    exit 0
fi

# Try OSMesa fallback
echo "EGL failed. Trying OSMesa..."
apt-get install -y -qq libosmesa6-dev 2>/dev/null
export MUJOCO_GL=osmesa
if python -c "
import os; os.environ['MUJOCO_GL'] = 'osmesa'
import gym_aloha, gymnasium
env = gymnasium.make('gym_aloha/AlohaTransferCube-v0')
obs, _ = env.reset(seed=42)
print('  ALOHA OK with OSMesa')
env.close()
" 2>/dev/null; then
    echo "export MUJOCO_GL=osmesa" >> ~/.bashrc
    echo "OSMesa works. Added to bashrc."
    exit 0
fi

echo ""
echo "WARNING: ALOHA rendering failed with both EGL and OSMesa."
echo "PushT experiments will still work (no MuJoCo needed)."
echo "Skipping ACT ALOHA experiment. Running PushT only."
