"""Quick dataset inspection script - temporary."""
from datasets import load_dataset
import numpy as np

ds = load_dataset("HuggingFaceVLA/smol-libero", split="train")

# Full dataset stats
states = np.array(ds["observation.state"])
actions = np.array(ds["action"])
print(f"Full dataset ({len(ds)} samples):")
print(f"state mean: {repr(states.mean(axis=0).astype(np.float32).tolist())}")
print(f"state std:  {repr(states.std(axis=0).astype(np.float32).tolist())}")
print(f"action mean: {repr(actions.mean(axis=0).astype(np.float32).tolist())}")
print(f"action std:  {repr(actions.std(axis=0).astype(np.float32).tolist())}")
