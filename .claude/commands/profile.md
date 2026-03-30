---
description: Profile a VLA model and show optimization suggestions
argument-hint: Model name (e.g., smolvla)
---

# Profile VLA Model

Profile a model and suggest optimizations.

Model: $ARGUMENTS

## Steps

1. Run `vla-edge check` to show current hardware
2. Run `vla-edge profile <model> --hardware auto --iterations 100`
3. Display results (latency, memory, FPS)
4. Based on results, suggest optimizations:
   - If FPS < 10: suggest quantization (GGUF Q4)
   - If memory > 6GB on Jetson: suggest smaller model or quantization
   - If latency p99 > 200ms: suggest TensorRT for vision encoder
5. Check if a recipe exists for this model + hardware combo
6. If no recipe: offer to create one from the profile results
