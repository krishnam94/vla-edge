"""Generate all paper figures from experiment results.

Produces 4 figures for the ICRA VLA Pipelines Workshop paper:
  1. fig1_action_ranges.pdf  - Per-dimension action ranges showing OOB (EXP-A)
  2. fig2_pareto.pdf          - Strictness vs violation/clipping tradeoff (EXP-D)
  3. fig3_ablation.pdf        - Component ablation bar chart (EXP-F)
  4. fig4_overhead.pdf        - Overhead comparison bar chart (EXP-B + EXP-E)

Usage:
  python experiments/icra_ws_2026/generate_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent.parent / "paper" / "icra_ws_2026" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (3.5, 2.8),  # IEEE column width
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "safe": "#2ecc71",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "dark": "#2c3e50",
}


def fig1_action_ranges():
    """Fig 1: Per-dimension action ranges from SmolVLA on LIBERO."""
    with open(RESULTS_DIR / "exp_a_baseline.json") as f:
        data = json.load(f)

    dims = list(range(len(data["per_dim_max"])))
    maxes = data["per_dim_max"]
    mins = data["per_dim_min"]
    labels = [f"dim {i}" for i in dims]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(dims))
    width = 0.35

    bars_max = ax.bar(x - width/2, maxes, width, label="Max", color=COLORS["danger"], alpha=0.85)
    bars_min = ax.bar(x + width/2, mins, width, label="Min", color=COLORS["primary"], alpha=0.85)

    # Draw safe bounds
    ax.axhline(y=1.0, color=COLORS["safe"], linestyle="--", linewidth=1.5, label="Safe bound (+1)")
    ax.axhline(y=-1.0, color=COLORS["safe"], linestyle="--", linewidth=1.5, label="Safe bound (-1)")

    # Shade OOB regions
    ax.axhspan(1.0, max(maxes) + 0.5, alpha=0.08, color=COLORS["danger"])
    ax.axhspan(min(mins) - 0.5, -1.0, alpha=0.08, color=COLORS["danger"])

    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Action Value")
    ax.set_title("SmolVLA outputs exceed safe bounds on LIBERO")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(min(mins) - 0.5, max(maxes) + 0.5)

    fig.savefig(FIGURES_DIR / "fig1_action_ranges.pdf")
    fig.savefig(FIGURES_DIR / "fig1_action_ranges.png")
    plt.close(fig)
    print(f"  Fig 1: {FIGURES_DIR / 'fig1_action_ranges.pdf'}")


def fig2_pareto():
    """Fig 2: Pareto frontier - strictness vs violation rate and clipping."""
    with open(RESULTS_DIR / "exp_d_pareto.json") as f:
        data = json.load(f)

    levels = data["levels"]
    names = [l["strictness"].replace("_", "\n") for l in levels]
    violations = [l["violations"] for l in levels]
    clip_mag = [l["clip_magnitude"] for l in levels]
    smoothness = [l["action_smoothness"] for l in levels]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(levels))

    # Left axis: violations
    color1 = COLORS["danger"]
    ax1.bar(x - 0.15, violations, 0.3, color=color1, alpha=0.8, label="Violations")
    ax1.set_xlabel("Contract Strictness")
    ax1.set_ylabel("Total Violations", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7)

    # Right axis: clipping magnitude
    ax2 = ax1.twinx()
    color2 = COLORS["primary"]
    ax2.plot(x, clip_mag, "o-", color=color2, linewidth=2, markersize=5, label="Clip magnitude")
    ax2.set_ylabel("Mean Clip Magnitude", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    ax1.set_title("Safety-Fidelity Tradeoff (Pareto Frontier)")

    fig.savefig(FIGURES_DIR / "fig2_pareto.pdf")
    fig.savefig(FIGURES_DIR / "fig2_pareto.png")
    plt.close(fig)
    print(f"  Fig 2: {FIGURES_DIR / 'fig2_pareto.pdf'}")


def fig3_ablation():
    """Fig 3: Component ablation showing each catches unique violations."""
    with open(RESULTS_DIR / "exp_f_ablation.json") as f:
        data = json.load(f)

    components = data["components"]
    # Skip "none" (0 violations)
    components = [c for c in components if c["component"] != "none"]

    names = [c["component"].replace("_", "\n").replace("+", "+\n") for c in components]
    violations = [c["violations"] for c in components]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    colors = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["warning"],
        COLORS["danger"],
        COLORS["dark"],
    ]

    bars = ax.bar(range(len(names)), violations, color=colors[:len(names)], alpha=0.85)

    # Add value labels on bars
    for bar, v in zip(bars, violations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(v), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Contract Components")
    ax.set_ylabel("Violations Detected")
    ax.set_title("Component Ablation: Each Catches Unique Violations")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)

    fig.savefig(FIGURES_DIR / "fig3_ablation.pdf")
    fig.savefig(FIGURES_DIR / "fig3_ablation.png")
    plt.close(fig)
    print(f"  Fig 3: {FIGURES_DIR / 'fig3_ablation.pdf'}")


def fig4_overhead():
    """Fig 4: Overhead comparison - SafeContract vs AEGIS vs inference."""
    with open(RESULTS_DIR / "exp_e_overhead.json") as f:
        data = json.load(f)

    methods = ["SafeContract\n(clip)", "SafeContract\n(adaptive)", "AEGIS-lite\n(QP)"]
    times_us = [
        data["safecontract_clip_us"],
        data["adaptive_contract_us"],
        data["aegis_lite_us"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.5), gridspec_kw={"width_ratios": [3, 1]})

    # Left: method comparison
    colors = [COLORS["safe"], COLORS["primary"], COLORS["danger"]]
    bars = ax1.bar(range(len(methods)), times_us, color=colors, alpha=0.85)

    for bar, t in zip(bars, times_us):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{t:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_ylabel("Overhead ($\\mu$s)")
    ax1.set_title("Contract Enforcement Overhead")
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=7)

    # Right: proportion of inference time
    inference_ms = data["vla_inference_us"] / 1000  # to ms
    contract_ms = data["safecontract_clip_us"] / 1000  # to ms
    sizes = [contract_ms, inference_ms - contract_ms]
    ax2.pie(sizes, labels=None, colors=[COLORS["safe"], COLORS["dark"]],
            startangle=90, autopct=None)
    ax2.set_title(f"Contract:\n{data['contract_overhead_ratio']*100:.5f}%\nof inference", fontsize=7)

    fig.savefig(FIGURES_DIR / "fig4_overhead.pdf")
    fig.savefig(FIGURES_DIR / "fig4_overhead.png")
    plt.close(fig)
    print(f"  Fig 4: {FIGURES_DIR / 'fig4_overhead.pdf'}")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_action_ranges()
    fig2_pareto()
    fig3_ablation()
    fig4_overhead()
    print(f"\nAll figures saved to {FIGURES_DIR}")
