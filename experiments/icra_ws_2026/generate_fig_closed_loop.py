"""Generate closed-loop experiment figure (EXP-CL).

Bar chart: Without SafeContract (58%) vs With SafeContract (60%)
with Wilson 95% CI error bars and statistical annotations.

Usage:
  python experiments/icra_ws_2026/generate_fig_closed_loop.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent.parent / "paper" / "icra_ws_2026" / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Match existing figure style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (3.5, 2.8),
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


def wilson_ci(successes, total, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    p = successes / total
    n = total
    z = 1.96  # 95% CI
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


def main():
    with open(RESULTS_DIR / "closed_loop_eval_50ep.json") as f:
        data = json.load(f)

    s = data["summary"]
    no_sr = s["no_contract"]["success_rate"]
    with_sr = s["with_contract"]["success_rate"]
    no_n = s["no_contract"]["total_episodes"]
    with_n = s["with_contract"]["total_episodes"]
    no_succ = s["no_contract"]["successes"]
    with_succ = s["with_contract"]["successes"]
    violations = s["with_contract"]["total_violations"]
    fisher_p = s["fisher_p_value"]

    # Wilson CIs
    no_lo, no_hi = wilson_ci(no_succ, no_n)
    with_lo, with_hi = wilson_ci(with_succ, with_n)

    # Error bar arrays (distance from center)
    no_err = [[no_sr - no_lo], [no_hi - no_sr]]
    with_err = [[with_sr - with_lo], [with_hi - with_sr]]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = [0, 1]
    heights = [no_sr * 100, with_sr * 100]
    errs_lo = [no_err[0][0] * 100, with_err[0][0] * 100]
    errs_hi = [no_err[1][0] * 100, with_err[1][0] * 100]
    colors = [COLORS["danger"], COLORS["safe"]]
    labels = ["Without\nSafeContract", "With\nSafeContract"]

    bars = ax.bar(
        x, heights, width=0.5,
        color=colors, alpha=0.85,
        yerr=[errs_lo, errs_hi],
        capsize=6, ecolor=COLORS["dark"], error_kw={"linewidth": 1.5},
    )

    # Value labels on bars
    for i, (bar, h) in enumerate(zip(bars, heights)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + errs_hi[i] + 2,
            f"{h:.0f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Annotation box
    ax.annotate(
        f"p = {fisher_p:.1f} (Fisher's exact)\n"
        f"{violations:,} violations caught\n"
        f"0 task degradation",
        xy=(0.98, 0.55), xycoords="axes fraction",
        ha="right", va="center",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title(f"Closed-Loop: ACT on ALOHA Transfer Cube (n={no_n})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGURES_DIR / "fig_closed_loop.pdf")
    fig.savefig(FIGURES_DIR / "fig_closed_loop.png")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'fig_closed_loop.pdf'}")
    print(f"Saved: {FIGURES_DIR / 'fig_closed_loop.png'}")
    print(f"\nWilson 95% CIs:")
    print(f"  No contract:   {no_lo:.3f} - {no_hi:.3f}")
    print(f"  With contract: {with_lo:.3f} - {with_hi:.3f}")


if __name__ == "__main__":
    main()
