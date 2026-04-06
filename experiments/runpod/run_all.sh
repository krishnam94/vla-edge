#!/bin/bash
# RunPod A100 - Run ALL SafeContract experiments
# Run after setup.sh completes successfully
# Estimated total: ~3-4 hours

set -e
export MUJOCO_GL=osmesa
cd /workspace/vla-edge

echo "=== SafeContract GPU Experiments ==="
echo "Start: $(date)"

# ============================================
# TIER 1: MUST RUN (addresses reviewer critique)
# ============================================

echo ""
echo "=========================================="
echo "TIER 1a: VQ-BeT on PushT (n=50)"
echo "  THE critical experiment for same-dataset comparison"
echo "=========================================="
python -u experiments/runpod/exp_vqbet_pusht_gpu.py \
    --n-episodes 50 --device cuda --seed-base 0 \
    --output /workspace/results/vqbet_pusht_50ep.json \
    2>&1 | tee /workspace/results/vqbet_pusht_50ep.log

echo ""
echo "=========================================="
echo "TIER 1b: Diffusion on PushT (n=100)"
echo "  Upgrade from n=50 for tighter CIs"
echo "=========================================="
python -u experiments/runpod/exp_diffusion_pusht_gpu.py \
    --n-episodes 100 --device cuda --seed-base 0 \
    --output /workspace/results/diffusion_pusht_100ep.json \
    2>&1 | tee /workspace/results/diffusion_pusht_100ep.log

# ============================================
# TIER 2: HIGH VALUE
# ============================================

echo ""
echo "=========================================="
echo "TIER 2a: ACT on ALOHA (n=100, conformal cal)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 100 --device cuda --seed-base 0 \
    --output /workspace/results/act_aloha_100ep.json \
    2>&1 | tee /workspace/results/act_aloha_100ep.log

echo ""
echo "=========================================="
echo "TIER 2b: Cross-architecture conformal p-values"
echo "=========================================="
python -u experiments/runpod/exp_crossarch_conformal.py \
    --results-dir /workspace/results/ \
    --output /workspace/results/crossarch_conformal.json \
    2>&1 | tee /workspace/results/crossarch_conformal.log

# ============================================
# SUMMARY
# ============================================

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "End: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -la /workspace/results/*.json
echo ""
echo "Download with: scp runpod:/workspace/results/*.json ./results/"
