# Experiment Registry - vla-edge

All experiments MUST be registered here BEFORE running. No exceptions.

## Completed Experiments

### EXP-A: SmolVLA Open-Loop LIBERO
- **Date**: 2026-03-28
- **Hypothesis**: SmolVLA produces out-of-bounds actions on LIBERO observations
- **Architecture**: SmolVLA (flow matching, 450M params)
- **Task/Env**: LIBERO (100 observations, open-loop)
- **Result**: 100% OOB (pre-unnorm), 0% bounds + 98% velocity violations (post-unnorm)
- **Paper section**: EXP-A (Sec 4.1)
- **Data**: Not saved to JSON (stdout only)

### EXP-CL-VQ: VQ-BeT Closed-Loop PushT
- **Date**: 2026-04-03
- **Hypothesis**: VQ-BeT produces more violations than Diffusion due to discrete tokenization
- **Architecture**: VQ-BeT (37.5M params, VQ-VAE + Behavior Transformer)
- **Task/Env**: PushT (n=200 per condition, seeds 0-199, coverage >= 0.95)
- **Contract**: bounds [0, 512], v_max=30 px/step
- **Result**: 56.5%/54.5% (p=0.76), 1847 violations
- **Paper section**: EXP-CL (Sec 4.2)
- **Data**: `results/runpod/vqbet_pusht_200ep.json`

### EXP-CL-DP: Diffusion Policy Closed-Loop PushT
- **Date**: 2026-04-03
- **Hypothesis**: Diffusion produces fewer violations but may stall
- **Architecture**: Diffusion Policy (262M params, DDPM)
- **Task/Env**: PushT (n=200 per condition, seeds 0-199, coverage >= 0.95)
- **Contract**: bounds [0, 512], v_max=30 px/step
- **Result**: 58%/57% (p=0.92), 772 violations
- **Paper section**: EXP-CL (Sec 4.2)
- **Data**: `results/runpod/diffusion_pusht_200ep.json`

### EXP-MON: PushT Monitor Analysis (AUROC)
- **Date**: 2026-04-04
- **Hypothesis**: Different monitors predict failure differently per architecture
- **Architecture**: VQ-BeT + Diffusion (same as EXP-CL)
- **Task/Env**: PushT (n=100 per architecture, no-contract condition)
- **Result**: Reversal AUROC 0.93/0.79, Jerk 0.88/0.43, Velocity 0.69/0.41
- **Paper section**: Failure Prediction (Sec 4.2)
- **Data**: `results/runpod/monitor_analysis.json`

### EXP-G: Controlled Ablation (Synthetic)
- **Date**: 2026-03-30
- **Hypothesis**: SafeContract catches corrupted policies with zero false positives
- **Architecture**: Synthetic (clean, drifting, jerky policies)
- **Task/Env**: Synthetic 7-DOF, 100 steps
- **Result**: Clean=0 violations, Drifting=115, Jerky=75
- **Paper section**: Cut from final version (was Sec 4.3)
- **Data**: Not saved (inline in paper)

### EXP-CONF: Conformal Calibration
- **Date**: 2026-04-01
- **Hypothesis**: Conformal bounds tighter than heuristic with coverage guarantee
- **Architecture**: ACT on ALOHA demos
- **Result**: 97.9% coverage, 25% tighter than 4-sigma
- **Data**: `results/runpod/exp_conformal.json` (if exists)

## Running Experiments

### EXP-ALOHA: ACT ALOHA Closed-Loop + Monitor Analysis
- **Date**: 2026-04-09
- **Hypothesis**: ACT on 14-DOF ALOHA validates the continuous-family monitoring pattern
- **Architecture**: ACT (51.6M params, CVAE + Transformer)
- **Task/Env**: ALOHA transfer-cube + insertion (n=50 per condition, gym_aloha)
- **Contract**: Conformal bounds (alpha=0.05), v_max per-joint 99th percentile
- **Expected SR**: ~83% transfer, ~20% insertion
- **Key question**: Does reversal rate predict ACT failure? Does jerk remain non-predictive (continuous family)?
- **Normalization**: Image mean/std + qpos mean/std from safetensors (CRITICAL - fixed Apr 8)
- **RunPod**: RTX 3090, log at `/workspace/results/aloha_run2.log`
- **Steps**:
  1. ACT transfer-cube n=50 (with/without contract)
  2. ACT insertion n=50 (with/without contract)
  3. Full monitor analysis n=50 (ALL monitors + AUROC)
- **Status**: RUNNING (episodes succeeding after normalization fix)

## Planned Experiments

### EXP-OPENVLA: OpenVLA on LIBERO (Autoregressive)
- **Date**: TBD (post-submission, ~Apr 16-20)
- **Hypothesis**: Autoregressive VLAs show discrete-family failure signatures (high jerk, token repetition) distinct from both VQ-BeT and Diffusion
- **Architecture**: OpenVLA (7B params, autoregressive 256-bin tokenization)
- **Task/Env**: LIBERO-Long (10 tasks, n=20 per task = 200 total)
- **Checkpoint**: `openvla/openvla-7b-finetuned-libero-10` (HuggingFace)
- **Expected SR**: ~54% (published baseline)
- **Contract**: Conformal bounds from LIBERO demo data, v_max 99th percentile
- **Key questions**:
  1. Does jerk AUROC match VQ-BeT (discrete family) or Diffusion (continuous)?
  2. Does token repetition appear as a new failure mode (stall-like but discrete)?
  3. Does the two-family law hold for a 3rd discrete architecture?
- **Requirements**:
  - GPU: A40 or better (7B model needs 16GB+ VRAM)
  - LIBERO environment (Linux only, MuJoCo)
  - ~4-6 hours runtime
  - Estimated cost: ~$3-5 RunPod
- **Design**: Same SafeContract monitoring as PushT/ALOHA. Conformal calibration from LIBERO demo episodes.
- **Script**: `experiments/runpod/exp_openvla_libero_gpu.py` (TO BE WRITTEN)
- **Normalization**: OpenVLA uses 256-bin discretization, unnormalize via bin centers. Check carefully!
- **Risk**: LIBERO setup may be complex. Fallback: SimplerEnv with Google Robot.
