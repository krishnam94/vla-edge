#!/bin/bash
# Run ALL SafeContract experiments on RunPod A40
# 6 experiments: PushT (2 arch) + ALOHA (2 tasks) + Monitor Analysis + Conformal
set -e

export MUJOCO_GL=osmesa
cd /workspace/vla-edge
mkdir -p /workspace/results

echo "=== SafeContract FULL GPU Experiments ==="
echo "Start: $(date)"

# ============================================
# PUSHT: Same-dataset cross-architecture (n=200)
# ============================================

echo ""
echo "=========================================="
echo "1/6: VQ-BeT on PushT (n=200)"
echo "=========================================="
python -u experiments/runpod/exp_vqbet_pusht_gpu.py \
    --n-episodes 200 --device cuda --seed-base 0 \
    --output /workspace/results/vqbet_pusht_200ep.json \
    2>&1 | tee /workspace/results/vqbet_pusht.log

echo ""
echo "=========================================="
echo "2/6: Diffusion on PushT (n=200)"
echo "=========================================="
python -u experiments/runpod/exp_diffusion_pusht_gpu.py \
    --n-episodes 200 --device cuda --seed-base 0 \
    --output /workspace/results/diffusion_pusht_200ep.json \
    2>&1 | tee /workspace/results/diffusion_pusht.log

# ============================================
# ALOHA: Transfer cube + Insertion (n=200)
# ============================================

echo ""
echo "=========================================="
echo "3/6: ACT on ALOHA transfer-cube (n=200)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 200 --device cuda --seed-base 0 \
    --output /workspace/results/act_aloha_transfer_200ep.json \
    2>&1 | tee /workspace/results/act_aloha_transfer.log

echo ""
echo "=========================================="
echo "4/6: ACT on ALOHA insertion (n=200)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 200 --device cuda --seed-base 0 \
    --task insertion \
    --output /workspace/results/act_aloha_insertion_200ep.json \
    2>&1 | tee /workspace/results/act_aloha_insertion.log

# ============================================
# FULL MONITOR ANALYSIS (stall + jerk + velocity)
# The KEY experiment: does composite monitor beat AUROC 0.54?
# ============================================

echo ""
echo "=========================================="
echo "5/6: Full monitor analysis (all violation types on PushT)"
echo "  Runs StallDetector + JerkMonitor on both architectures"
echo "  Computes AUROC for each monitor as failure predictor"
echo "=========================================="
python -u experiments/runpod/exp_monitor_analysis.py \
    --n-episodes 100 --device cuda \
    --output /workspace/results/monitor_analysis.json \
    2>&1 | tee /workspace/results/monitor_analysis.log

# ============================================
# CROSS-ARCHITECTURE CONFORMAL ANALYSIS
# ============================================

echo ""
echo "=========================================="
echo "6/6: Cross-architecture conformal analysis"
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
python -c "
import json, glob
for f in sorted(glob.glob('/workspace/results/*.json')):
    try:
        with open(f) as fh:
            d = json.load(fh)
        s = d.get('summary', {})
        nc = s.get('no_contract', {})
        wc = s.get('with_contract', {})
        if nc and wc:
            print(f'{f.split(\"/\")[-1]:40s} {nc.get(\"success_rate\",0):.0%} -> {wc.get(\"success_rate\",0):.0%}  p={s.get(\"fisher_p_value\",\"?\")}  viol={wc.get(\"total_violations\",0)}')
        elif 'results' in d:
            for name, r in d['results'].items():
                if isinstance(r, dict) and 'success_rate' in r:
                    print(f'{f.split(\"/\")[-1]:40s} {name}: SR={r[\"success_rate\"]:.0%}  stall={r.get(\"total_stall_steps\",\"?\")}  jerk={r.get(\"mean_jerk_rms\",\"?\"):.2f}')
    except: pass
"
