#!/usr/bin/env python3
"""OpenVLA on LIBERO-Long (10 tasks) with SafeContract monitoring.

Evaluates openvla-7b-finetuned-libero-10 on all 10 LIBERO-Long tasks,
running SafeContract monitors (stall, jerk, spectral, momentum, LPS, reversal)
and computing per-monitor AUROC for failure prediction.

Key architecture notes (OpenVLA is DIFFERENT from ACT/Diffusion):
- OpenVLA uses 256-bin action TOKENIZATION, not continuous regression
- Actions are decoded from token IDs to bin centers in [-1, 1]
- Unnormalization uses q01/q99 quantile scaling (NOT mean/std)
- Formula: action = 0.5 * (normalized + 1) * (q99 - q01) + q01
- Gripper dim (index 6) is masked from unnormalization (binary 0/1)
- The model's predict_action() handles all of this internally
- 7-DOF delta actions: EEF delta xyz (3) + rpy (3) + gripper (1)
- Image input: 224x224 (processor handles resize from 256x256 LIBERO obs)
- Prompt: "In: What action should the robot take to {task}?\\nOut:"

Expected results:
- Published success rate: 53.7% +/- 1.3% on LIBERO-Long (3 seeds x 500 rollouts)
- We run n=20 per task x 10 tasks = 200 total rollouts
- Per-task step limits from OpenVLA eval: 520 for libero_10

Normalization WARNING (lesson from ALOHA experiment):
- OpenVLA's predict_action() returns UNNORMALIZED continuous actions
- We do NOT need to manually unnormalize (unlike ACT/lerobot models)
- But we DO need to verify action ranges are sane after predict_action()
- The 256-bin quantization means actions snap to discrete grid points
"""

import argparse
import json
import os
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu

# Add vla-edge to path for monitors
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vla_edge.validate.violations import StallDetector, JerkMonitor

try:
    from vla_edge.validate.signals import (
        SpectralEnergyMonitor,
        LinearPredictabilityMonitor,
        MomentumCoherenceMonitor,
    )
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False
    print("WARNING: signals module not found, running without spectral/LPS/momentum")

# -------------------------------------------------------------------------
# LIBERO task suite configuration
# -------------------------------------------------------------------------
# Step limits per suite (from OpenVLA eval: run_libero_eval.py)
STEP_LIMITS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Model checkpoints per suite
MODELS = {
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
}

# Number of initial wait steps (objects settling) - from OpenVLA eval
NUM_STEPS_WAIT = 10


def load_openvla(model_id: str, device: str = "cuda"):
    """Load OpenVLA model and processor from HuggingFace.

    Returns (model, processor). The model's predict_action() handles:
    1. Token generation (autoregressive, 7 tokens for 7-DOF)
    2. Token-to-bin-center decoding (256 bins in [-1, 1])
    3. Unnormalization via q01/q99 quantile scaling

    We do NOT need separate normalization stats - it's all in the model.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading OpenVLA from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    try:
        vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        print("  Using flash_attention_2")
    except Exception as e:
        print(f"  flash_attention_2 failed ({e}), falling back to eager...")
        vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
    vla.eval()

    n_params = sum(p.numel() for p in vla.parameters())
    print(f"  Loaded: {n_params:,} params on {device}")

    # Load dataset_statistics.json if missing from model
    # Some finetuned checkpoints have it embedded, some need it loaded manually
    if not hasattr(vla, "norm_stats") or not vla.norm_stats:
        from huggingface_hub import hf_hub_download
        try:
            stats_path = hf_hub_download(model_id, "dataset_statistics.json")
            with open(stats_path) as f:
                vla.norm_stats = json.load(f)
            print("  Loaded norm_stats from dataset_statistics.json")
        except Exception:
            print("  WARNING: no dataset_statistics.json found")

    # Print norm_stats for verification
    if hasattr(vla, "norm_stats") and vla.norm_stats:
        for key in vla.norm_stats:
            stats = vla.norm_stats[key]["action"]
            print(f"  norm_stats[{key}]:")
            print(f"    q01: {[round(x, 3) for x in stats['q01']]}")
            print(f"    q99: {[round(x, 3) for x in stats['q99']]}")
            if "mask" in stats:
                print(f"    mask: {stats['mask']}")

    return vla, processor


def get_action(vla, processor, image, task_description, unnorm_key, device="cuda"):
    """Get a single 7-DOF action from OpenVLA.

    Args:
        vla: OpenVLA model (AutoModelForVision2Seq)
        processor: OpenVLA processor (handles image resize + tokenization)
        image: PIL Image (will be resized to 224x224 by processor)
        task_description: natural language task instruction
        unnorm_key: dataset key for action unnormalization (e.g. "libero_10")
        device: torch device

    Returns:
        np.ndarray of shape (7,) - unnormalized delta actions
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    return action  # np.ndarray, shape (7,), already unnormalized


def load_libero_task_suite(suite_name: str):
    """Load LIBERO task suite and return tasks with metadata.

    Returns:
        suite: LIBERO benchmark suite object
        tasks: list of task objects
    """
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    tasks = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        tasks.append(task)
        print(f"  Task {i}: {task.language}")
    return suite, tasks


def create_libero_env(task, resolution=256):
    """Create a LIBERO environment for a given task.

    LIBERO uses robosuite's OffScreenRenderEnv with BDDL task files.
    Camera resolution is 256x256 (OpenVLA processor resizes to 224x224).
    """
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = os.path.join(
        benchmark.get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    return env


def get_libero_image(obs):
    """Extract and preprocess image from LIBERO observation.

    LIBERO's agentview_image is upside-down (180-degree rotated) compared
    to the training data. The OpenVLA LIBERO eval script rotates it back.
    Returns a PIL Image ready for the OpenVLA processor.
    """
    from PIL import Image

    img = obs["agentview_image"]
    # Rotate 180 degrees (flip both axes) - critical for matching training data
    img = img[::-1, ::-1].copy()
    return Image.fromarray(img)


def compute_conformal_calibration_from_demos(task, suite, suite_name, seed=42):
    """Compute conformal action bounds from LIBERO demo data for a single task.

    Unlike ACT/lerobot where demos are in HuggingFace datasets, LIBERO demos
    are stored as HDF5 files. We load them and compute per-task calibration.

    Returns:
        action_lo, action_hi: conformal bounds per action dimension
        v_max: per-dimension velocity limit (99th percentile)
    """
    import h5py
    from libero.libero import benchmark

    # Try to load demo data
    demo_path = os.path.join(
        benchmark.get_libero_path("datasets"),
        f"{task.problem_folder}",
        f"{task.bddl_file.replace('.bddl', '_demo.hdf5')}",
    )

    if not os.path.exists(demo_path):
        # Fall back: try the standard LIBERO demo path
        demo_dir = os.path.join(benchmark.get_libero_path("datasets"), suite_name)
        if os.path.isdir(demo_dir):
            hdf5_files = [f for f in os.listdir(demo_dir) if f.endswith(".hdf5")]
            if hdf5_files:
                demo_path = os.path.join(demo_dir, hdf5_files[0])

    if not os.path.exists(demo_path):
        print(f"    WARNING: No demo data found at {demo_path}")
        print(f"    Using fallback conservative bounds for 7-DOF delta actions")
        # Conservative fallback bounds for LIBERO 7-DOF delta actions
        # Based on published q01/q99 from openvla config
        action_lo = np.array([-1.0, -1.0, -1.0, -0.3, -0.4, -0.4, -1.0])
        action_hi = np.array([1.0, 1.0, 1.0, 0.3, 0.4, 0.4, 1.0])
        v_max = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.2, 2.0])
        return action_lo, action_hi, v_max

    print(f"    Loading demos from {demo_path}")
    try:
        with h5py.File(demo_path, "r") as f:
            data = f["data"]
            all_actions = []
            for demo_key in sorted(data.keys()):
                actions = data[demo_key]["actions"][:]
                all_actions.append(actions)

        cal_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    except Exception as e:
        print(f"    WARNING: Failed to load demos: {e}")
        action_lo = np.array([-1.0, -1.0, -1.0, -0.3, -0.4, -0.4, -1.0])
        action_hi = np.array([1.0, 1.0, 1.0, 0.3, 0.4, 0.4, 1.0])
        v_max = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.2, 2.0])
        return action_lo, action_hi, v_max

    rng = np.random.RandomState(seed)
    n_cal = int(len(all_actions) * 0.8)
    shuffled_idx = rng.permutation(len(all_actions))
    cal_ep_idx = shuffled_idx[:n_cal]
    cal_actions_flat = np.concatenate(
        [all_actions[i] for i in cal_ep_idx], axis=0
    ).astype(np.float32)

    # Conformal bounds (alpha=0.05)
    cal_mean = cal_actions_flat.mean(axis=0)
    cal_std = cal_actions_flat.std(axis=0) + 1e-8
    scores = np.max(np.abs(cal_actions_flat - cal_mean) / cal_std, axis=1)
    q_hat = np.quantile(scores, 0.95 * (1 + 1 / len(cal_actions_flat)))
    action_lo = cal_mean - q_hat * cal_std
    action_hi = cal_mean + q_hat * cal_std

    # Gripper dim: clamp to valid range
    action_lo[6] = max(action_lo[6], -1.1)
    action_hi[6] = min(action_hi[6], 1.1)

    # Velocity limits (99th percentile of consecutive action deltas)
    all_deltas = []
    for ep_actions in all_actions:
        if len(ep_actions) > 1:
            all_deltas.append(np.abs(np.diff(ep_actions, axis=0)))
    if all_deltas:
        all_deltas = np.concatenate(all_deltas, axis=0)
        v_max = np.percentile(all_deltas, 99, axis=0)
    else:
        v_max = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.2, 2.0])

    print(f"    Cal actions: {cal_actions_flat.shape}")
    print(f"    Conformal q_hat: {q_hat:.3f}")
    print(f"    Bounds: lo={action_lo.round(3)}, hi={action_hi.round(3)}")
    print(f"    v_max: {v_max.round(4)}")
    return action_lo, action_hi, v_max


def compute_episode_reversals(per_step_actions):
    """Compute reversal rate from a list of action arrays."""
    if len(per_step_actions) < 3:
        return 0.0
    actions = np.array(per_step_actions)
    velocities = np.diff(actions, axis=0)
    sign_changes = np.sum(np.abs(np.diff(np.sign(velocities), axis=0)) > 0)
    return float(sign_changes / max(len(velocities) - 1, 1))


def compute_episode_atv(per_step_actions):
    """Compute Absolute Total Variation from action trajectory."""
    if len(per_step_actions) < 2:
        return 0.0
    actions = np.array(per_step_actions)
    return float(np.sum(np.abs(np.diff(actions, axis=0))))


def run_monitored_episode(
    env, vla, processor, task_description, unnorm_key,
    init_states, trial_idx,
    action_lo, action_hi, v_max,
    device="cuda", max_steps=520,
):
    """Run one LIBERO episode with full monitor suite.

    Follows the OpenVLA eval protocol:
    1. Reset env with specific initial state
    2. Wait NUM_STEPS_WAIT steps for objects to settle (dummy actions)
    3. Run policy for max_steps
    4. Track all monitors per-step

    Returns dict with success, monitor summaries, per-step data.
    """
    # Reset with specific initial state (LIBERO protocol)
    obs = env.reset()
    if init_states is not None and trial_idx < len(init_states):
        obs = env.set_init_state(init_states[trial_idx])

    # Initialize monitors
    stall = StallDetector(movement_threshold=0.005, stall_window=15)
    jerk = JerkMonitor(jerk_limit=0.3)  # tighter for delta actions
    spectral = SpectralEnergyMonitor(window_size=32) if HAS_SIGNALS else None
    lps = LinearPredictabilityMonitor() if HAS_SIGNALS else None
    momentum = MomentumCoherenceMonitor(momentum_window=5) if HAS_SIGNALS else None

    per_step_data = []
    per_step_actions = []
    prev_action = None
    success = False
    total_bounds_viol = 0
    total_vel_viol = 0

    # Wait steps: send dummy actions while objects settle
    dummy_action = [0, 0, 0, 0, 0, 0, -1]  # no movement, gripper open
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, info = env.step(dummy_action)
        if done:
            break

    for step in range(max_steps):
        # Get image from observation
        image = get_libero_image(obs)

        # Get action from OpenVLA
        t_infer = time.time()
        action_np = get_action(vla, processor, image, task_description,
                               unnorm_key, device)
        infer_time = time.time() - t_infer

        # SANITY CHECK: verify action ranges
        # OpenVLA delta actions should be small (typically < 1.0 for xyz, < 0.5 for rpy)
        if step == 0:
            print(f"      Step 0 action: {[round(float(x), 4) for x in action_np]}")
            if np.any(np.abs(action_np[:3]) > 2.0):
                print(f"      WARNING: XYZ deltas unusually large - check unnormalization!")
            if np.any(np.abs(action_np[3:6]) > 1.0):
                print(f"      WARNING: RPY deltas unusually large - check unnormalization!")

        # Gripper action mapping (matching OpenVLA eval protocol exactly):
        # Step 1: normalize_gripper_action(binarize=True): [0,1] -> [-1,+1] via 2*x-1, then sign()
        # Step 2: invert_gripper_action(): flip sign (negate)
        # Net effect for RLDS convention (0=close, 1=open):
        #   openvla=0 (close) -> normalize: -1 -> invert: +1 (LIBERO close)
        #   openvla=1 (open)  -> normalize: +1 -> invert: -1 (LIBERO open)
        # Formula: libero_gripper = -(sign(2*x - 1)) = sign(1 - 2*x)
        gripper_raw = action_np[6]
        action_for_env = action_np.copy()
        gripper_normalized = 2.0 * np.clip(float(gripper_raw), 0.0, 1.0) - 1.0
        gripper_binarized = 1.0 if gripper_normalized >= 0 else -1.0  # binarize (tie-break: close)
        action_for_env[6] = -gripper_binarized  # invert: now -1=open, +1=close for LIBERO

        # Monitor on the 7-DOF action (before gripper remap, for consistency)
        action_monitor = action_np.copy()

        # Bounds check (on raw unnormalized action)
        bounds_viol = not np.all(
            (action_monitor >= action_lo) & (action_monitor <= action_hi)
        )
        if bounds_viol:
            total_bounds_viol += 1

        # Velocity check
        vel_viol = False
        if prev_action is not None:
            delta = np.abs(action_monitor - prev_action)
            vel_viol = bool(np.any(delta > v_max))
            if vel_viol:
                total_vel_viol += 1

        # Run all monitors
        stall_report = stall.update(action_monitor)
        jerk_report = jerk.update(action_monitor)
        ser_val = spectral.update(action_monitor)["ser"] if spectral else 1.0
        lps_err = lps.update(action_monitor)["prediction_error"] if lps else 0.0
        coherence = momentum.update(action_monitor)["coherence"] if momentum else 1.0

        per_step_data.append({
            "bounds_viol": bounds_viol,
            "vel_viol": vel_viol,
            "stalled": stall_report["is_stalled"],
            "movement": stall_report["movement_norm"],
            "max_jerk": jerk_report["max_jerk"],
            "jerk_viol": jerk_report["jerk_violation"],
            "ser": ser_val,
            "prediction_error": lps_err,
            "coherence": coherence,
            "infer_time": infer_time,
        })
        per_step_actions.append(action_monitor.tolist())
        prev_action = action_monitor.copy()

        # Step environment (robosuite expects list, not ndarray)
        obs, reward, done, info = env.step(action_for_env.tolist())
        if done:
            success = True
            break

    # Compute trajectory-level metrics
    reversal_rate = compute_episode_reversals(per_step_actions)
    atv = compute_episode_atv(per_step_actions)
    avg_infer_time = np.mean([s["infer_time"] for s in per_step_data]) if per_step_data else 0

    return {
        "success": success,
        "steps": step + 1 if per_step_data else 0,
        "bounds_violations": total_bounds_viol,
        "velocity_violations": total_vel_viol,
        "reversal_rate": reversal_rate,
        "atv": atv,
        "avg_infer_time_s": float(avg_infer_time),
        "stall_summary": stall.get_summary(),
        "jerk_summary": jerk.get_summary(),
        "spectral_summary": spectral.get_summary() if spectral else {},
        "lps_summary": lps.get_summary() if lps else {},
        "momentum_summary": momentum.get_summary() if momentum else {},
    }


def compute_aurocs(episodes):
    """Compute AUROC for each monitor metric (failure vs success discrimination)."""
    results = {}
    successes = [e for e in episodes if e["success"]]
    failures = [e for e in episodes if not e["success"]]

    if len(successes) < 3 or len(failures) < 3:
        return {"error": f"Too few: success={len(successes)}, failure={len(failures)}"}

    # Each metric: higher value should predict failure
    metrics = {
        "velocity_violations": lambda e: e["velocity_violations"],
        "bounds_violations": lambda e: e["bounds_violations"],
        "stall_rate": lambda e: e["stall_summary"].get("stall_rate", 0),
        "mean_jerk": lambda e: e["jerk_summary"].get("mean_jerk", 0),
        "jerk_rms": lambda e: e["jerk_summary"].get("jerk_rms", 0),
        "atv": lambda e: e.get("atv", 0),
        "reversal_rate": lambda e: e.get("reversal_rate", 0),
        # Inverted metrics (low = bad)
        "inv_mean_ser": lambda e: 1.0 - e.get("spectral_summary", {}).get("mean_ser", 1.0),
        "mean_prediction_error": lambda e: e.get("lps_summary", {}).get("mean_prediction_error", 0),
        "inv_min_coherence": lambda e: -e.get("momentum_summary", {}).get("min_coherence", 1.0),
    }

    for name, fn in metrics.items():
        try:
            s_vals = [fn(e) for e in successes]
            f_vals = [fn(e) for e in failures]
            # Mann-Whitney U: failures should have higher values
            u, p = mannwhitneyu(f_vals, s_vals, alternative="greater")
            auroc = u / (len(f_vals) * len(s_vals))
            results[name] = {"auroc": round(auroc, 4), "p_value": round(p, 4)}
        except Exception as ex:
            results[name] = {"error": str(ex)}

    return results


def save_results(output_path, data):
    """Save results as JSON with numpy type handling."""
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))


def main():
    parser = argparse.ArgumentParser(description="OpenVLA LIBERO-Long Evaluation with SafeContract")
    parser.add_argument("--task-suite", type=str, default="libero_10",
                        choices=list(MODELS.keys()))
    parser.add_argument("--n-episodes", type=int, default=20,
                        help="Episodes per task (default: 20, published uses 50)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str,
                        default="/workspace/results/openvla_libero_monitors.json")
    parser.add_argument("--task-ids", type=str, default=None,
                        help="Comma-separated task IDs to run (e.g. '0,1,2'). Default: all")
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "osmesa"

    suite_name = args.task_suite
    model_id = MODELS[suite_name]
    max_steps = STEP_LIMITS[suite_name]
    unnorm_key = suite_name

    print(f"{'='*60}")
    print(f"  OpenVLA LIBERO Evaluation with SafeContract Monitors")
    print(f"  Suite: {suite_name} | Model: {model_id}")
    print(f"  Episodes/task: {args.n_episodes} | Max steps: {max_steps}")
    print(f"{'='*60}")

    # Load model
    vla, processor = load_openvla(model_id, args.device)

    # Load task suite
    print(f"\nLoading {suite_name} tasks...")
    suite, tasks = load_libero_task_suite(suite_name)

    # Select tasks
    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = list(range(len(tasks)))

    print(f"\nRunning {len(task_ids)} tasks x {args.n_episodes} episodes = "
          f"{len(task_ids) * args.n_episodes} total rollouts")

    # Results storage
    all_episodes = []
    per_task_results = {}

    for task_idx in task_ids:
        task = tasks[task_idx]
        task_name = task.language
        print(f"\n{'='*60}")
        print(f"  Task {task_idx}: {task_name}")
        print(f"{'='*60}")

        # Compute conformal calibration for this task
        print("  Computing conformal calibration...")
        action_lo, action_hi, v_max = compute_conformal_calibration_from_demos(
            task, suite, suite_name
        )

        # Get initial states for this task
        init_states = suite.get_task_init_states(task_idx)

        task_episodes = []
        task_successes = 0

        for ep in range(args.n_episodes):
            # Create fresh env per episode (LIBERO protocol)
            env = create_libero_env(task)
            env.seed(ep)  # Seed affects object positions even with fixed init states

            t0 = time.time()
            try:
                metrics = run_monitored_episode(
                    env, vla, processor, task_name, unnorm_key,
                    init_states, ep,
                    action_lo, action_hi, v_max,
                    device=args.device, max_steps=max_steps,
                )
            except Exception as ex:
                print(f"    Ep {ep}: CRASHED - {ex}")
                metrics = {
                    "success": False, "steps": 0, "error": str(ex),
                    "bounds_violations": 0, "velocity_violations": 0,
                    "reversal_rate": 0, "atv": 0, "avg_infer_time_s": 0,
                    "stall_summary": {}, "jerk_summary": {},
                    "spectral_summary": {}, "lps_summary": {}, "momentum_summary": {},
                }
            elapsed = time.time() - t0

            metrics["episode"] = ep
            metrics["task_idx"] = task_idx
            metrics["task_name"] = task_name
            metrics["elapsed_s"] = elapsed
            task_episodes.append(metrics)
            all_episodes.append(metrics)

            if metrics["success"]:
                task_successes += 1

            st = "OK" if metrics["success"] else "--"
            viol = metrics.get("velocity_violations", 0)
            steps = metrics.get("steps", 0)
            infer_t = metrics.get("avg_infer_time_s", 0)
            print(f"    Ep {ep:2d}: {st} | steps={steps:3d} | "
                  f"vel_viol={viol:3d} | infer={infer_t:.2f}s/step | {elapsed:.1f}s total")

            env.close()

        # Per-task summary
        task_sr = task_successes / args.n_episodes
        n = args.n_episodes
        z = 1.96
        if n > 0 and 0 < task_sr < 1:
            ci_lo = (task_sr + z*z/(2*n) - z*sqrt(task_sr*(1-task_sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
            ci_hi = (task_sr + z*z/(2*n) + z*sqrt(task_sr*(1-task_sr)/n + z*z/(4*n*n))) / (1 + z*z/n)
        else:
            ci_lo, ci_hi = task_sr, task_sr

        task_aurocs = compute_aurocs(task_episodes)

        per_task_results[f"task_{task_idx}"] = {
            "task_name": task_name,
            "success_rate": task_sr,
            "successes": task_successes,
            "total_episodes": n,
            "ci_95": [round(max(0, ci_lo), 3), round(min(1, ci_hi), 3)],
            "aurocs": task_aurocs,
        }

        print(f"  Task {task_idx} SR: {task_sr:.0%} ({task_successes}/{n})")
        if isinstance(task_aurocs, dict) and "error" not in task_aurocs:
            top_aurocs = sorted(
                [(k, v["auroc"]) for k, v in task_aurocs.items() if isinstance(v, dict) and "auroc" in v],
                key=lambda x: -x[1]
            )[:3]
            for name, auroc in top_aurocs:
                print(f"    Top AUROC: {name} = {auroc:.4f}")

        # Incremental save after each task
        partial = {
            "experiment": f"OpenVLA LIBERO {suite_name} (partial, {task_idx + 1}/{len(task_ids)} tasks done)",
            "model": model_id,
            "suite": suite_name,
            "per_task": per_task_results,
            "episodes": all_episodes,
            "config": {
                "n_episodes_per_task": args.n_episodes,
                "max_steps": max_steps,
                "device": args.device,
                "tasks_done": task_idx + 1,
                "tasks_total": len(task_ids),
            },
        }
        save_results(args.output + ".partial", partial)
        print(f"  Saved partial results to {args.output}.partial")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    total_success = sum(1 for e in all_episodes if e["success"])
    total_n = len(all_episodes)
    overall_sr = total_success / total_n if total_n > 0 else 0

    # Wilson CI for overall
    z = 1.96
    if total_n > 0 and 0 < overall_sr < 1:
        ci_lo = (overall_sr + z*z/(2*total_n) - z*sqrt(overall_sr*(1-overall_sr)/total_n + z*z/(4*total_n*total_n))) / (1 + z*z/total_n)
        ci_hi = (overall_sr + z*z/(2*total_n) + z*sqrt(overall_sr*(1-overall_sr)/total_n + z*z/(4*total_n*total_n))) / (1 + z*z/total_n)
    else:
        ci_lo, ci_hi = overall_sr, overall_sr

    print(f"  Overall SR: {overall_sr:.1%} ({total_success}/{total_n})")
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Published: 53.7% +/- 1.3% (3 seeds x 500 rollouts)")

    # Per-task breakdown
    print(f"\n  Per-task success rates:")
    for task_key, task_data in per_task_results.items():
        sr = task_data["success_rate"]
        name = task_data["task_name"][:50]
        print(f"    {task_key}: {sr:.0%} ({task_data['successes']}/{task_data['total_episodes']}) - {name}")

    # Overall AUROCs across all episodes
    overall_aurocs = compute_aurocs(all_episodes)
    print(f"\n  Overall AUROCs (failure prediction):")
    if isinstance(overall_aurocs, dict) and "error" not in overall_aurocs:
        for name, vals in sorted(
            overall_aurocs.items(),
            key=lambda x: -x[1].get("auroc", 0) if isinstance(x[1], dict) else 0
        ):
            if isinstance(vals, dict) and "auroc" in vals:
                flag = " ***" if vals["auroc"] > 0.65 else ""
                print(f"    {name:30s} {vals['auroc']:.4f} (p={vals.get('p_value', '?')}){flag}")
    else:
        print(f"    {overall_aurocs}")

    # Average inference time
    avg_infer = np.mean([e.get("avg_infer_time_s", 0) for e in all_episodes if e.get("avg_infer_time_s", 0) > 0])
    print(f"\n  Avg inference time: {avg_infer:.3f}s/step ({1/avg_infer:.1f} Hz)" if avg_infer > 0 else "")

    # Save final results
    output = {
        "experiment": f"OpenVLA LIBERO {suite_name} with SafeContract Monitors",
        "model": model_id,
        "suite": suite_name,
        "overall": {
            "success_rate": overall_sr,
            "successes": total_success,
            "total_episodes": total_n,
            "ci_95": [round(max(0, ci_lo), 3), round(min(1, ci_hi), 3)],
            "published_sr": "53.7% +/- 1.3%",
            "avg_inference_time_s": float(avg_infer) if avg_infer > 0 else None,
        },
        "overall_aurocs": overall_aurocs,
        "per_task": per_task_results,
        "episodes": all_episodes,
        "config": {
            "model": model_id,
            "suite": suite_name,
            "n_episodes_per_task": args.n_episodes,
            "max_steps": max_steps,
            "num_steps_wait": NUM_STEPS_WAIT,
            "unnorm_key": unnorm_key,
            "device": args.device,
            "action_dim": 7,
            "action_format": "delta EEF xyz(3) + rpy(3) + gripper(1)",
            "tokenization": "256-bin, decoded via q01/q99 quantile scaling",
            "image_resolution": "256x256 (resized to 224x224 by processor)",
            "monitors": [
                "bounds_violations", "velocity_violations",
                "stall_rate", "mean_jerk", "reversal_rate", "atv",
                "spectral_energy", "linear_predictability", "momentum_coherence",
            ],
        },
    }

    save_results(args.output, output)
    print(f"\nSaved final results to {args.output}")

    # Clean up partial file
    partial_path = args.output + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)


if __name__ == "__main__":
    main()
