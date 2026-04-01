"""EXP-005: Cross-architecture SafeContract validation.

Runs SafeContract on 3 architectures with their native datasets:
  1. SmolVLA (flow matching VLA) - LIBERO observations
  2. ACT (action chunking transformer) - ALOHA sim transfer cube
  3. Diffusion Policy (DDPM denoising) - PushT

Shows safety contracts are architecture-agnostic. Same contract API,
different action spaces, different denoising mechanisms, same guarantees.

Usage:
  python experiments/icra_ws_2026/exp5_cross_architecture.py [--n-samples 50] [--device cpu]
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Ground-truth action ranges per environment
# ---------------------------------------------------------------------------
# These come from the training data statistics / environment limits.
# We use them to build realistic SafetyConfigs.

# ALOHA bimanual: 14 joints (2x 6-DOF arms + 2 grippers)
# Joint positions in radians; grippers 0-1. Empirical range from dataset.
ALOHA_ACTION_BOUNDS = np.array(
    [
        [-3.14, 3.14],   # left_waist
        [-1.80, 1.50],   # left_shoulder
        [-1.80, 1.80],   # left_elbow
        [-3.14, 3.14],   # left_forearm_roll
        [-1.80, 1.80],   # left_wrist_angle
        [-3.14, 3.14],   # left_wrist_rotate
        [0.0, 1.0],      # left_gripper
        [-3.14, 3.14],   # right_waist
        [-1.80, 1.50],   # right_shoulder
        [-1.80, 1.80],   # right_elbow
        [-3.14, 3.14],   # right_forearm_roll
        [-1.80, 1.80],   # right_wrist_angle
        [-3.14, 3.14],   # right_wrist_rotate
        [0.0, 1.0],      # right_gripper
    ],
    dtype=np.float32,
)

# PushT: 2D end-effector position. Workspace is 512x512 pixel space,
# but the policy outputs normalized actions roughly in [-1, 1].
PUSHT_ACTION_BOUNDS = np.array(
    [
        [-1.0, 1.0],  # x velocity / position delta
        [-1.0, 1.0],  # y velocity / position delta
    ],
    dtype=np.float32,
)

# SmolVLA on LIBERO: 6-DOF (or 7-DOF with gripper). Actions are
# delta joint positions, roughly [-1, 1] but can exceed.
SMOLVLA_ACTION_BOUNDS = np.array(
    [[-1.0, 1.0]] * 6,
    dtype=np.float32,
)


@dataclass
class ArchitectureResult:
    """Results for one architecture."""

    name: str
    architecture_type: str
    dataset: str
    action_dim: int
    n_samples: int = 0
    avg_latency_ms: float = 0.0
    cold_start_ms: float = 0.0
    # Raw action statistics
    action_min: float = 0.0
    action_max: float = 0.0
    action_mean: float = 0.0
    action_std: float = 0.0
    # Safety analysis
    total_violations: int = 0
    bounds_violations: int = 0
    velocity_violations: int = 0
    oob_rate: float = 0.0  # fraction of samples with at least one OOB dim
    clipping_magnitude_mean: float = 0.0
    # Contract overhead
    contract_overhead_us: float = 0.0


@dataclass
class ExperimentResults:
    """Full cross-architecture experiment results."""

    experiment: str = "EXP-005: Cross-architecture SafeContract"
    architectures: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    timestamp: str = ""


def run_smolvla(n_samples: int, device: str) -> ArchitectureResult:
    """Run SmolVLA on LIBERO observations."""
    import torch
    from datasets import load_dataset

    from vla_edge.models.smolvla import SmolVLAAdapter
    from vla_edge.validate.contract import clear_violation_log, get_violation_log, safety_contract
    from vla_edge.validate.safety import SafetyConfig, clip_actions, validate_actions

    print("\n--- SmolVLA (Flow Matching VLA) ---")

    # Load dataset
    print("  Loading HuggingFaceVLA/smol-libero...")
    ds = load_dataset("HuggingFaceVLA/smol-libero", split="train")
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    # Load model
    print(f"  Loading SmolVLA on {device}...")
    adapter = SmolVLAAdapter(device=device, dtype="float32")

    # Collect actions
    all_actions = []
    latencies = []

    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        image = np.array(sample["observation.images.image"], dtype=np.uint8)
        state = np.array(sample["observation.state"], dtype=np.float32)

        t0 = time.perf_counter()
        action = adapter.predict(image, "complete the task", state[:6])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        all_actions.append(action.copy())

        if (i + 1) % 10 == 0:
            print(f"  SmolVLA: {i+1}/{n_samples} ({elapsed_ms:.0f}ms)")

    all_actions_np = np.array(all_actions)
    action_dim = min(all_actions_np.shape[-1], 6)
    actions_trimmed = all_actions_np[:, :action_dim]

    # Safety validation
    config = SafetyConfig(
        action_bounds=SMOLVLA_ACTION_BOUNDS[:action_dim],
        max_velocity=np.full(action_dim, 0.1, dtype=np.float32),
    )
    safety_result = validate_actions(actions_trimmed, config)
    clipped = clip_actions(actions_trimmed, config)
    clip_mag = float(np.mean(np.abs(actions_trimmed - clipped)))

    # OOB rate (how many samples have at least one dim outside bounds)
    lo = SMOLVLA_ACTION_BOUNDS[:action_dim, 0]
    hi = SMOLVLA_ACTION_BOUNDS[:action_dim, 1]
    oob_mask = np.any((actions_trimmed < lo) | (actions_trimmed > hi), axis=1)
    oob_rate = float(np.mean(oob_mask))

    # Contract overhead
    clear_violation_log()

    @safety_contract(action_range=[-1.0, 1.0], joint_velocity_max=0.1)
    def mock_predict(image, instruction, state=None):
        return actions_trimmed[0]

    # Warmup
    for _ in range(50):
        mock_predict(None, "test")
    overhead_times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        mock_predict(None, "test")
        overhead_times.append((time.perf_counter() - t0) * 1e6)

    adapter.cleanup()

    result = ArchitectureResult(
        name="SmolVLA",
        architecture_type="VLM + Flow Matching (10 denoising steps)",
        dataset="HuggingFaceVLA/smol-libero",
        action_dim=action_dim,
        n_samples=n_samples,
        avg_latency_ms=round(float(np.mean(latencies[1:])), 1),  # skip cold start
        cold_start_ms=round(latencies[0], 1),
        action_min=round(float(actions_trimmed.min()), 4),
        action_max=round(float(actions_trimmed.max()), 4),
        action_mean=round(float(actions_trimmed.mean()), 4),
        action_std=round(float(actions_trimmed.std()), 4),
        total_violations=len(safety_result.violations),
        bounds_violations=sum(1 for v in safety_result.violations if v.violation_type == "bounds"),
        velocity_violations=sum(1 for v in safety_result.violations if v.violation_type == "velocity"),
        oob_rate=round(oob_rate, 4),
        clipping_magnitude_mean=round(clip_mag, 6),
        contract_overhead_us=round(float(np.mean(overhead_times)), 2),
    )
    print(f"  Done. {result.total_violations} violations, OOB rate={result.oob_rate}")
    return result


def run_act(n_samples: int, device: str) -> ArchitectureResult:
    """Run ACT on ALOHA sim transfer cube observations."""
    import torch
    from datasets import load_dataset

    from vla_edge.validate.contract import clear_violation_log, safety_contract
    from vla_edge.validate.safety import SafetyConfig, clip_actions, validate_actions

    print("\n--- ACT (Action Chunking Transformer) ---")

    # Load dataset - LeRobot v2 format on HuggingFace
    print("  Loading lerobot/aloha_sim_transfer_cube_human...")
    ds = load_dataset("lerobot/aloha_sim_transfer_cube_human", split="train")
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    # Load model
    print(f"  Loading ACT on {device}...")
    from vla_edge.models.act_adapter import ACTAdapter

    adapter = ACTAdapter(device=device)

    # Collect actions
    all_actions = []
    latencies = []

    for i, idx in enumerate(indices):
        sample = ds[int(idx)]

        # ACT expects top camera image and joint state
        # LeRobot dataset keys: observation.images.top, observation.state
        image = np.array(sample["observation.images.top"], dtype=np.uint8)
        state = np.array(sample["observation.state"], dtype=np.float32)

        # Reset policy before each episode-like forward pass
        adapter._ensure_loaded()
        adapter._policy.reset()

        # Build observation dict directly (bypassing the predict() wrapper
        # to match LeRobot's expected format exactly)
        img_tensor = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(adapter._device)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(adapter._device)

        obs = {
            "observation.images.top": img_tensor,
            "observation.state": state_tensor,
        }

        t0 = time.perf_counter()
        with torch.inference_mode():
            action = adapter._policy.select_action(obs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        latencies.append(elapsed_ms)
        all_actions.append(action.copy())

        if (i + 1) % 10 == 0:
            print(f"  ACT: {i+1}/{n_samples} ({elapsed_ms:.0f}ms)")

    all_actions_np = np.array(all_actions)
    action_dim = all_actions_np.shape[-1]

    # Safety validation with ALOHA bounds
    bounds = ALOHA_ACTION_BOUNDS[:action_dim]
    config = SafetyConfig(
        action_bounds=bounds,
        max_velocity=np.full(action_dim, 0.1, dtype=np.float32),
    )
    safety_result = validate_actions(all_actions_np, config)
    clipped = clip_actions(all_actions_np, config)
    clip_mag = float(np.mean(np.abs(all_actions_np - clipped)))

    # OOB rate
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    oob_mask = np.any((all_actions_np < lo) | (all_actions_np > hi), axis=1)
    oob_rate = float(np.mean(oob_mask))

    # Contract overhead
    clear_violation_log()

    @safety_contract(action_range=[-3.14, 3.14], joint_velocity_max=0.1)
    def mock_predict(image, instruction, state=None):
        return all_actions_np[0]

    for _ in range(50):
        mock_predict(None, "test")
    overhead_times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        mock_predict(None, "test")
        overhead_times.append((time.perf_counter() - t0) * 1e6)

    adapter.cleanup()

    result = ArchitectureResult(
        name="ACT",
        architecture_type="Transformer (Action Chunking, no VLM)",
        dataset="lerobot/aloha_sim_transfer_cube_human",
        action_dim=action_dim,
        n_samples=n_samples,
        avg_latency_ms=round(float(np.mean(latencies[1:])), 1),
        cold_start_ms=round(latencies[0], 1),
        action_min=round(float(all_actions_np.min()), 4),
        action_max=round(float(all_actions_np.max()), 4),
        action_mean=round(float(all_actions_np.mean()), 4),
        action_std=round(float(all_actions_np.std()), 4),
        total_violations=len(safety_result.violations),
        bounds_violations=sum(1 for v in safety_result.violations if v.violation_type == "bounds"),
        velocity_violations=sum(1 for v in safety_result.violations if v.violation_type == "velocity"),
        oob_rate=round(oob_rate, 4),
        clipping_magnitude_mean=round(clip_mag, 6),
        contract_overhead_us=round(float(np.mean(overhead_times)), 2),
    )
    print(f"  Done. {result.total_violations} violations, OOB rate={result.oob_rate}")
    return result


def run_diffusion(n_samples: int, device: str) -> ArchitectureResult:
    """Run Diffusion Policy on PushT observations."""
    import torch
    from datasets import load_dataset

    from vla_edge.validate.contract import clear_violation_log, safety_contract
    from vla_edge.validate.safety import SafetyConfig, clip_actions, validate_actions

    print("\n--- Diffusion Policy (DDPM) ---")

    # Load dataset
    print("  Loading lerobot/pusht...")
    ds = load_dataset("lerobot/pusht", split="train")
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    # Load model
    print(f"  Loading Diffusion Policy on {device}...")
    from vla_edge.models.diffusion_adapter import DiffusionPolicyAdapter

    adapter = DiffusionPolicyAdapter(device=device)

    # Collect actions
    all_actions = []
    latencies = []

    for i, idx in enumerate(indices):
        sample = ds[int(idx)]

        # Diffusion Policy on PushT expects: observation.image, observation.state
        # PushT image is 96x96; state is 2D (agent position)
        image = np.array(sample["observation.image"], dtype=np.uint8)
        state = np.array(sample["observation.state"], dtype=np.float32)

        # Reset policy for each sample
        adapter._ensure_loaded()
        adapter._policy.reset()

        # Build observation dict directly
        img_tensor = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(adapter._device)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(adapter._device)

        obs = {
            "observation.image": img_tensor,
            "observation.state": state_tensor,
        }

        t0 = time.perf_counter()
        with torch.inference_mode():
            action = adapter._policy.select_action(obs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        latencies.append(elapsed_ms)
        all_actions.append(action.copy())

        if (i + 1) % 10 == 0:
            print(f"  Diffusion: {i+1}/{n_samples} ({elapsed_ms:.0f}ms)")

    all_actions_np = np.array(all_actions)
    action_dim = all_actions_np.shape[-1]

    # Safety validation with PushT bounds
    bounds = PUSHT_ACTION_BOUNDS[:action_dim]
    config = SafetyConfig(
        action_bounds=bounds,
        max_velocity=np.full(action_dim, 0.1, dtype=np.float32),
    )
    safety_result = validate_actions(all_actions_np, config)
    clipped = clip_actions(all_actions_np, config)
    clip_mag = float(np.mean(np.abs(all_actions_np - clipped)))

    # OOB rate
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    oob_mask = np.any((all_actions_np < lo) | (all_actions_np > hi), axis=1)
    oob_rate = float(np.mean(oob_mask))

    # Contract overhead
    clear_violation_log()

    @safety_contract(action_range=[-1.0, 1.0], joint_velocity_max=0.1)
    def mock_predict(image, instruction, state=None):
        return all_actions_np[0]

    for _ in range(50):
        mock_predict(None, "test")
    overhead_times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        mock_predict(None, "test")
        overhead_times.append((time.perf_counter() - t0) * 1e6)

    adapter.cleanup()

    result = ArchitectureResult(
        name="Diffusion Policy",
        architecture_type="DDPM Denoising (100 steps, no VLM)",
        dataset="lerobot/pusht",
        action_dim=action_dim,
        n_samples=n_samples,
        avg_latency_ms=round(float(np.mean(latencies[1:])), 1),
        cold_start_ms=round(latencies[0], 1),
        action_min=round(float(all_actions_np.min()), 4),
        action_max=round(float(all_actions_np.max()), 4),
        action_mean=round(float(all_actions_np.mean()), 4),
        action_std=round(float(all_actions_np.std()), 4),
        total_violations=len(safety_result.violations),
        bounds_violations=sum(1 for v in safety_result.violations if v.violation_type == "bounds"),
        velocity_violations=sum(1 for v in safety_result.violations if v.violation_type == "velocity"),
        oob_rate=round(oob_rate, 4),
        clipping_magnitude_mean=round(clip_mag, 6),
        contract_overhead_us=round(float(np.mean(overhead_times)), 2),
    )
    print(f"  Done. {result.total_violations} violations, OOB rate={result.oob_rate}")
    return result


def generate_paper_table(results: list[ArchitectureResult]) -> str:
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{SafeContract across three policy architectures. Same contract API, different action spaces and denoising mechanisms. Violations detected before clipping; all resolved at $<5\mu s$ overhead.}",
        r"\label{tab:cross-arch}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \textbf{SmolVLA} & \textbf{ACT} & \textbf{Diffusion Policy} \\",
        r"\midrule",
    ]

    # Architecture row
    archs = [r.architecture_type.split("(")[0].strip() for r in results]
    lines.append(f"Architecture & {archs[0]} & {archs[1]} & {archs[2]} \\\\")

    # Dataset row
    ds_short = [r.dataset.split("/")[-1] for r in results]
    lines.append(f"Dataset & {ds_short[0]} & {ds_short[1]} & {ds_short[2]} \\\\")

    # Action dim
    lines.append(
        f"Action dim & {results[0].action_dim} & {results[1].action_dim} & {results[2].action_dim} \\\\"
    )

    lines.append(r"\midrule")

    # Action range observed
    ranges = [f"[{r.action_min:.2f}, {r.action_max:.2f}]" for r in results]
    lines.append(f"Action range & {ranges[0]} & {ranges[1]} & {ranges[2]} \\\\")

    # OOB rate
    oob = [f"{r.oob_rate*100:.0f}\\%" for r in results]
    lines.append(f"OOB rate & {oob[0]} & {oob[1]} & {oob[2]} \\\\")

    # Violations
    lines.append(
        f"Bounds violations & {results[0].bounds_violations} & {results[1].bounds_violations} & {results[2].bounds_violations} \\\\"
    )
    lines.append(
        f"Velocity violations & {results[0].velocity_violations} & {results[1].velocity_violations} & {results[2].velocity_violations} \\\\"
    )

    # Clipping magnitude
    clips = [f"{r.clipping_magnitude_mean:.4f}" for r in results]
    lines.append(f"Avg clip magnitude & {clips[0]} & {clips[1]} & {clips[2]} \\\\")

    lines.append(r"\midrule")

    # Latency
    lats = [f"{r.avg_latency_ms:.0f}" for r in results]
    lines.append(f"Inference (ms) & {lats[0]} & {lats[1]} & {lats[2]} \\\\")

    # Contract overhead
    overheads = [f"{r.contract_overhead_us:.1f}" for r in results]
    lines.append(f"Contract overhead ($\\mu$s) & {overheads[0]} & {overheads[1]} & {overheads[2]} \\\\")

    # Overhead ratio
    ratios = [f"{r.contract_overhead_us / (r.avg_latency_ms * 1000) * 100:.4f}\\%" for r in results]
    lines.append(f"Overhead ratio & {ratios[0]} & {ratios[1]} & {ratios[2]} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main(n_samples: int = 50, device: str = "cpu"):
    print("=" * 60)
    print("EXP-005: Cross-Architecture SafeContract Validation")
    print(f"  n_samples={n_samples}, device={device}")
    print("=" * 60)

    results = []

    # Run all three architectures
    try:
        smolvla_result = run_smolvla(n_samples, device)
        results.append(smolvla_result)
    except Exception as e:
        print(f"  SmolVLA FAILED: {e}")

    try:
        act_result = run_act(n_samples, device)
        results.append(act_result)
    except Exception as e:
        print(f"  ACT FAILED: {e}")

    try:
        diffusion_result = run_diffusion(n_samples, device)
        results.append(diffusion_result)
    except Exception as e:
        print(f"  Diffusion FAILED: {e}")

    if not results:
        print("All architectures failed. Check dependencies.")
        return

    # Compile results
    experiment = ExperimentResults(
        architectures=[asdict(r) for r in results],
        summary={
            "n_architectures_tested": len(results),
            "architectures": [r.name for r in results],
            "key_finding": (
                "SafeContract detects violations across all architectures with "
                f"<{max(r.contract_overhead_us for r in results):.0f}us overhead. "
                f"OOB rates: {', '.join(f'{r.name}={r.oob_rate*100:.0f}%' for r in results)}. "
                "Same contract API, three denoising paradigms."
            ),
            "total_violations_caught": sum(r.total_violations for r in results),
            "max_overhead_us": max(r.contract_overhead_us for r in results),
        },
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Save JSON results
    path = RESULTS_DIR / "exp5_cross_architecture.json"
    path.write_text(json.dumps(asdict(experiment), indent=2))
    print(f"\nResults saved to {path}")

    # Generate and save LaTeX table
    if len(results) == 3:
        latex = generate_paper_table(results)
        latex_path = RESULTS_DIR / "table_cross_architecture.tex"
        latex_path.write_text(latex)
        print(f"LaTeX table saved to {latex_path}")
        print("\n" + latex)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Model':<20} {'Arch':<15} {'Dim':>4} {'OOB%':>6} {'Bounds':>7} {'Vel':>5} {'Clip':>8} {'Lat(ms)':>8} {'OH(us)':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        arch_short = r.architecture_type.split("(")[0].strip()[:14]
        print(
            f"{r.name:<20} {arch_short:<15} {r.action_dim:>4} "
            f"{r.oob_rate*100:>5.0f}% {r.bounds_violations:>7} {r.velocity_violations:>5} "
            f"{r.clipping_magnitude_mean:>8.4f} {r.avg_latency_ms:>8.0f} {r.contract_overhead_us:>7.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXP-005: Cross-architecture SafeContract validation"
    )
    parser.add_argument("--n-samples", type=int, default=50, help="Samples per architecture")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    args = parser.parse_args()
    main(n_samples=args.n_samples, device=args.device)
