#!/usr/bin/env python3
"""Template: integrate SafetyGuard with your own robot/VLA.

Replace the placeholder policy and demo loading with your actual code.
This shows the minimal integration pattern.
"""

import numpy as np
from vla_edge import SafetyGuard


# ============================================================
# STEP 1: Load your demonstration data
# Replace this with your actual demo loading code
# ============================================================
def load_demos():
    """Load demonstration actions for calibration.

    Returns array of shape (n_steps, action_dim).
    These should be UNNORMALIZED actions in the robot's joint space.
    """
    # Example: load from a LeRobot dataset
    # from datasets import load_dataset
    # ds = load_dataset("your/dataset", split="train")
    # actions = np.array([row["action"] for row in ds])

    # Placeholder: 7-DOF arm, 1000 demo steps
    rng = np.random.RandomState(42)
    return rng.randn(1000, 7).astype(np.float32) * 0.5


# ============================================================
# STEP 2: Create SafetyGuard from demos
# ============================================================
demo_actions = load_demos()
guard = SafetyGuard.from_demos(demo_actions, alpha=0.05)
print(f"SafetyGuard calibrated from {len(demo_actions)} demo steps")


# ============================================================
# STEP 3: Wrap your predict function
# ============================================================
@guard.wrap
def predict(obs):
    """Your VLA policy. Replace with actual model inference.

    Args:
        obs: observation dict (images, state, etc.)

    Returns:
        action: numpy array of shape (action_dim,)
    """
    # Example: SmolVLA inference
    # action = model.predict(obs)

    # Placeholder
    rng = np.random.RandomState(hash(str(obs)) % 2**31)
    return rng.randn(7).astype(np.float32) * 0.5


# ============================================================
# STEP 4: Run your control loop
# ============================================================
def main():
    print("\n--- Control Loop (50 steps) ---")
    for step in range(50):
        obs = {"step": step}  # Replace with actual observation
        action = predict(obs)  # SafetyGuard enforces automatically
        # env.step(action)     # Send to robot

    # Check health
    summary = guard.get_summary()
    print(f"\nViolations: {summary.get('bounds_violations', 0)} bounds, "
          f"{summary.get('velocity_violations', 0)} velocity")
    print(f"Stalled: {summary.get('stall_detected', False)}")
    print(f"CUSUM alarm: {summary.get('cusum_alarm', False)}")

    # Architecture-matched monitor recommendation
    print("\n--- Monitor Recommendations ---")
    print("For discrete-token VLAs (VQ-BeT, OpenVLA):")
    print("  Primary: reversal rate + jerk")
    print("For continuous VLAs (Diffusion, pi0, ACT):")
    print("  Primary: reversal rate + momentum coherence")
    print("For ALL architectures:")
    print("  Avoid: velocity violation counting (non-predictive)")


if __name__ == "__main__":
    main()
