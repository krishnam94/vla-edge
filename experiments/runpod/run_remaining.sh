#!/bin/bash
# Run REMAINING experiments (VQ-BeT already done)
# Use inside screen: screen -S exp
set -e

export MUJOCO_GL=osmesa
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
cd /workspace/vla-edge
mkdir -p /workspace/results

echo "=== Remaining Experiments ==="
echo "Start: $(date)"
echo "VQ-BeT: ALREADY DONE (check /workspace/results/vqbet_pusht_200ep.json)"

# Skip VQ-BeT, start from Diffusion
echo ""
echo "=========================================="
echo "1/5: Diffusion on PushT (n=200) - FULL RUN"
echo "=========================================="
python -u experiments/runpod/exp_diffusion_pusht_gpu.py \
    --n-episodes 200 --device cuda --seed-base 0 \
    --output /workspace/results/diffusion_pusht_200ep.json \
    2>&1 | tee /workspace/results/diffusion_pusht.log

echo ""
echo "=========================================="
echo "2/5: ACT on ALOHA transfer-cube (n=200)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 200 --device cuda --seed-base 0 \
    --output /workspace/results/act_aloha_transfer_200ep.json \
    2>&1 | tee /workspace/results/act_aloha_transfer.log

echo ""
echo "=========================================="
echo "3/5: ACT on ALOHA insertion (n=200)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 200 --device cuda --seed-base 0 \
    --task insertion \
    --output /workspace/results/act_aloha_insertion_200ep.json \
    2>&1 | tee /workspace/results/act_aloha_insertion.log

echo ""
echo "=========================================="
echo "4/5: Monitor analysis (stall + jerk on PushT)"
echo "=========================================="
python -u experiments/runpod/exp_monitor_analysis.py \
    --n-episodes 100 --device cuda \
    --output /workspace/results/monitor_analysis.json \
    2>&1 | tee /workspace/results/monitor_analysis.log

echo ""
echo "=========================================="
echo "5/5: Cross-arch conformal analysis"
echo "=========================================="
python -u experiments/runpod/exp_crossarch_conformal.py \
    --results-dir /workspace/results/ \
    --output /workspace/results/crossarch_conformal.json \
    2>&1 | tee /workspace/results/crossarch_conformal.log

echo ""
echo "=========================================="
echo "ALL DONE"
echo "End: $(date)"
echo "=========================================="
ls -la /workspace/results/*.json

python -c "
import json, glob
for f in sorted(glob.glob('/workspace/results/*.json')):
    try:
        with open(f) as fh:
            d = json.load(fh)
        s = d.get('summary', {})
        nc = s.get('no_contract', {})
        wc = s.get('with_contract', {})
        if nc:
            print(f'{f.split(\"/\")[-1]:40s} {nc.get(\"success_rate\",0):.0%} -> {wc.get(\"success_rate\",0):.0%}  p={s.get(\"fisher_p_value\",\"?\")}  viol={wc.get(\"total_violations\",0)}')
    except: pass
"
