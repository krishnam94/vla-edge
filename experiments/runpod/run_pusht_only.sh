#!/bin/bash
# Run just the PushT experiments (works even if ALOHA is broken)
# These are the #1 priority: same-dataset cross-architecture comparison
set -e

export MUJOCO_GL=egl
cd /workspace/vla-edge
mkdir -p /workspace/results

echo "=== SafeContract PushT Experiments ==="
echo "Start: $(date)"

echo ""
echo "=========================================="
echo "1/2: VQ-BeT on PushT (n=50)"
echo "  THE critical same-dataset experiment"
echo "=========================================="
python -u experiments/runpod/exp_vqbet_pusht_gpu.py \
    --n-episodes 50 --device cuda --seed-base 0 \
    --output /workspace/results/vqbet_pusht_50ep.json \
    2>&1 | tee /workspace/results/vqbet_pusht.log

echo ""
echo "=========================================="
echo "2/2: Diffusion on PushT (n=100)"
echo "  Upgrade from n=50 for tighter CIs"
echo "=========================================="
python -u experiments/runpod/exp_diffusion_pusht_gpu.py \
    --n-episodes 100 --device cuda --seed-base 0 \
    --output /workspace/results/diffusion_pusht_100ep.json \
    2>&1 | tee /workspace/results/diffusion_pusht.log

echo ""
echo "=========================================="
echo "DONE. Results:"
echo "=========================================="
ls -la /workspace/results/*.json
echo ""
echo "End: $(date)"
echo ""
echo "Download with:"
echo "  scp runpod:/workspace/results/*.json ~/Desktop/projects/vla-edge/results/runpod/"
