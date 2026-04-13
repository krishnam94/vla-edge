#!/usr/bin/env python3
"""Replay PushT experiment data and show monitor values.

Uses the saved experiment JSONs to demonstrate how different
monitors predict failure for different VLA architectures.

Requires: results/runpod/monitor_analysis.json (from RunPod experiments)
"""

import json
import sys
from pathlib import Path


def main():
    # Find the monitor analysis JSON
    results_dir = Path(__file__).parent.parent / "results" / "runpod"
    monitor_file = results_dir / "monitor_analysis.json"

    if not monitor_file.exists():
        print(f"Monitor analysis data not found at {monitor_file}")
        print("Run the PushT experiments first (see experiments/runpod/)")
        sys.exit(1)

    with open(monitor_file) as f:
        data = json.load(f)

    print("=== PushT Monitor Analysis ===\n")

    # Show AUROCs for each architecture
    for arch_name in ["vqbet", "diffusion"]:
        if arch_name not in data:
            continue
        arch = data[arch_name]
        aurocs = arch.get("aurocs", {})

        print(f"--- {arch_name.upper()} ---")
        print(f"  Success rate: {arch.get('success_rate', 'N/A')}")
        print(f"  Total violations: {arch.get('total_violations', 'N/A')}")
        print(f"\n  Failure Prediction AUROCs (higher = better):")

        for monitor, vals in sorted(
            aurocs.items(),
            key=lambda x: -x[1].get("auroc", 0) if isinstance(x[1], dict) else 0,
        ):
            if isinstance(vals, dict) and "auroc" in vals:
                auroc = vals["auroc"]
                flag = " ***" if auroc > 0.75 else ""
                print(f"    {monitor:30s} {auroc:.4f}{flag}")
        print()

    # Key finding
    print("=== Key Finding ===")
    print("Reversal rate predicts failure across BOTH architectures.")
    print("Jerk is predictive ONLY for discrete-token VQ-BeT (0.88 vs 0.41).")
    print("Velocity violations are NON-PREDICTIVE for both.")
    print("\nSee the paper: 'How VLAs Fail Differently'")


if __name__ == "__main__":
    main()
