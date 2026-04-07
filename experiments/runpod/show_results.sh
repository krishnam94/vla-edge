#!/bin/bash
# Show all RunPod experiment results
cd /workspace/vla-edge 2>/dev/null || cd /workspace

echo "=== ALL RUNPOD RESULTS ==="
echo ""

python -c "
import json, glob

for f in sorted(glob.glob('/workspace/results/*.json')):
    try:
        with open(f) as fh:
            d = json.load(fh)
        name = f.split('/')[-1]
        print(f'--- {name} ---')

        # Closed-loop experiments
        s = d.get('summary', {})
        nc = s.get('no_contract', {})
        wc = s.get('with_contract', {})
        if nc and wc:
            print(f'  no_contract:   {nc.get(\"success_rate\",0):.1%} ({nc.get(\"successes\",\"?\")}/{nc.get(\"total_episodes\",\"?\")})')
            print(f'  with_contract: {wc.get(\"success_rate\",0):.1%} ({wc.get(\"successes\",\"?\")}/{wc.get(\"total_episodes\",\"?\")})')
            print(f'  Fisher p = {s.get(\"fisher_p_value\",\"?\")}')
            print(f'  Violations: {wc.get(\"total_violations\",0)}')

        # Monitor analysis
        if 'results' in d and isinstance(d['results'], dict):
            for arch, r in d['results'].items():
                if isinstance(r, dict) and 'aurocs' in r:
                    print(f'  {arch}: SR={r.get(\"success_rate\",0):.0%}, vel={r.get(\"total_velocity_violations\",0)}, stall={r.get(\"total_stall_steps\",0)}')
                    aurocs = r['aurocs']
                    if isinstance(aurocs, dict) and 'error' not in aurocs:
                        print(f'  AUROCs:')
                        for metric, vals in sorted(aurocs.items(), key=lambda x: -x[1].get('auroc',0) if isinstance(x[1],dict) else 0):
                            if isinstance(vals, dict) and 'auroc' in vals:
                                flag = ' ***' if vals['auroc'] > 0.70 else ''
                                print(f'    {metric:30s} {vals[\"auroc\"]:.4f} (p={vals.get(\"p_value\",\"?\")}){flag}')
                    elif isinstance(aurocs, dict):
                        print(f'  AUROCs: {aurocs.get(\"error\",\"unknown error\")}')

        print()
    except Exception as e:
        print(f'  ERROR: {e}')
        print()
"
