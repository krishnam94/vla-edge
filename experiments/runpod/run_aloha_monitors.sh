#!/bin/bash
# Run ALOHA ACT monitor analysis (14-DOF, the generalization test)
# ~5 hours on A40. Cost: ~$2
set -e
export MUJOCO_GL=osmesa
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
cd /workspace/vla-edge
git pull
pip install -q -e . 2>/dev/null || true

echo "=== ALOHA Monitor Analysis ==="
echo "Start: $(date)"

# Fix pymunk if needed
pip install -q pymunk==6.10.0 2>/dev/null

# Step 1: ACT transfer-cube closed-loop (n=50 x 2 conditions)
echo ""
echo "=========================================="
echo "1/3: ACT ALOHA transfer-cube (n=50)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 50 --device cuda --seed-base 0 \
    --output /workspace/results/act_aloha_transfer_50ep.json \
    2>&1 | tee /workspace/results/act_aloha_transfer.log

# Step 2: ACT insertion (n=50 x 2 conditions)
echo ""
echo "=========================================="
echo "2/3: ACT ALOHA insertion (n=50)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 50 --device cuda --seed-base 0 \
    --task insertion \
    --output /workspace/results/act_aloha_insertion_50ep.json \
    2>&1 | tee /workspace/results/act_aloha_insertion.log

# Step 3: Monitor analysis on ALOHA (the KEY experiment)
# This needs a custom script since exp_monitor_analysis.py is PushT-specific
echo ""
echo "=========================================="
echo "3/3: ALOHA Monitor Analysis (ALL monitors)"
echo "=========================================="
python -u experiments/runpod/exp_aloha_monitor_analysis.py \
    --n-episodes 50 --device cuda \
    --output /workspace/results/aloha_monitor_analysis.json \
    2>&1 | tee /workspace/results/aloha_monitor_analysis.log

echo ""
echo "=========================================="
echo "DONE"
echo "End: $(date)"
echo "=========================================="
ls -la /workspace/results/*.json

# Show results summary
bash experiments/runpod/show_results.sh
