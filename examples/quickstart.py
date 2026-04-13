#!/usr/bin/env python3
"""SafetyGuard quickstart - 3 minutes to safe VLA actions.

No hardware needed. Uses synthetic data to demonstrate:
1. Calibrating bounds from demonstration data
2. Wrapping a policy with SafetyGuard
3. Monitoring violations in real-time
"""

import numpy as np
from vla_edge import SafetyGuard


def main():
    # 1. Generate synthetic "demonstration" data (replace with your demos)
    rng = np.random.RandomState(42)
    demo_actions = rng.randn(500, 7).astype(np.float32) * 0.3  # 500 steps, 7-DOF

    # 2. Create SafetyGuard calibrated from demos
    guard = SafetyGuard.from_demos(demo_actions, alpha=0.05)
    print(f"SafetyGuard created: {guard}")
    print(f"  Conformal coverage: {1 - 0.05:.0%}")
    print(f"  Action dimensions: {demo_actions.shape[1]}")

    # 3. Simulate a policy that sometimes produces unsafe actions
    print("\n--- Running 100 steps ---")
    for step in range(100):
        # Normal action (safe)
        if step < 70:
            action = rng.randn(7).astype(np.float32) * 0.3
        # Drifting action (increasingly unsafe)
        else:
            drift = (step - 70) * 0.1
            action = rng.randn(7).astype(np.float32) * 0.3 + drift

        safe_action = guard(action)

    # 4. Check the summary
    summary = guard.get_summary()
    print(f"\n--- Summary ---")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Bounds violations: {summary.get('bounds_violations', 0)}")
    print(f"  Velocity violations: {summary.get('velocity_violations', 0)}")
    print(f"  Stall detected: {summary.get('stall_detected', False)}")
    print(f"  CUSUM alarm: {summary.get('cusum_alarm', False)}")

    if summary.get('bounds_violations', 0) > 0 or summary.get('velocity_violations', 0) > 0:
        print("\n  SafetyGuard caught unsafe actions and clipped them.")
        print("  All output actions are within calibrated bounds.")
    else:
        print("\n  All actions were within calibrated bounds.")


if __name__ == "__main__":
    main()
