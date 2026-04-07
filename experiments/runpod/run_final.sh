#!/bin/bash
# Final RunPod run: monitor analysis (KEY) then ALOHA n=50
set -e
export MUJOCO_GL=osmesa
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
cd /workspace/vla-edge
git pull
mkdir -p /workspace/results

echo "=== Final Experiments ==="
echo "Start: $(date)"
echo "VQ-BeT PushT: DONE"
echo "Diffusion PushT: DONE"

echo ""
echo "=========================================="
echo "1/3: Monitor Analysis (ALL 10 monitors, n=100)"
echo "  THE key experiment: composite AUROC"
echo "=========================================="
python -u experiments/runpod/exp_monitor_analysis.py \
    --n-episodes 100 --device cuda \
    --output /workspace/results/monitor_analysis.json \
    2>&1 | tee /workspace/results/monitor_analysis.log

echo ""
echo "=========================================="
echo "2/3: ACT ALOHA transfer (n=50)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 50 --device cuda --seed-base 0 \
    --output /workspace/results/act_aloha_transfer_50ep.json \
    2>&1 | tee /workspace/results/act_aloha_transfer.log

echo ""
echo "=========================================="
echo "3/3: ACT ALOHA insertion (n=50)"
echo "=========================================="
python -u experiments/runpod/exp_act_aloha_gpu.py \
    --n-episodes 50 --device cuda --seed-base 0 \
    --task insertion \
    --output /workspace/results/act_aloha_insertion_50ep.json \
    2>&1 | tee /workspace/results/act_aloha_insertion.log

echo ""
echo "=========================================="
echo "ALL DONE"
echo "End: $(date)"
echo "=========================================="
ls -la /workspace/results/*.json

# Print all results
python -c "
import json, glob
print()
for f in sorted(glob.glob('/workspace/results/*.json')):
    try:
        with open(f) as fh:
            d = json.load(fh)
        name = f.split('/')[-1]
        s = d.get('summary', {})
        nc = s.get('no_contract', {})
        wc = s.get('with_contract', {})
        if nc and wc:
            print(f'{name:40s} {nc.get(\"success_rate\",0):.0%} -> {wc.get(\"success_rate\",0):.0%}  p={s.get(\"fisher_p_value\",\"?\")}  viol={wc.get(\"total_violations\",0)}')
        # Monitor analysis
        if 'results' in d:
            for arch, r in d['results'].items():
                if isinstance(r, dict) and 'aurocs' in r:
                    aurocs = r['aurocs']
                    print(f'{name:40s} {arch}: SR={r.get(\"success_rate\",0):.0%}')
                    for metric, vals in aurocs.items():
                        if isinstance(vals, dict) and 'auroc' in vals:
                            flag = ' ***' if vals['auroc'] > 0.65 else ''
                            print(f'  {metric:30s} AUROC={vals[\"auroc\"]:.4f} p={vals.get(\"p_value\",\"?\")}{flag}')
    except: pass
"
