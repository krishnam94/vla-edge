---
description: Run full benchmark suite and update results
argument-hint: Optional model name or "all"
---

# Run Benchmarks

Run the full benchmark suite on available hardware.

Target: $ARGUMENTS (default: all registered models on auto-detected hardware)

## Steps

1. Detect available hardware via `vla-edge check`
2. List registered models via `vla-edge models`
3. For each model x hardware combination:
   - Run `vla-edge profile <model> --hardware <backend> --iterations 100`
   - Save results to `results/<model>_<hardware>.json`
4. Run safety validation on each model
5. Generate comparison table
6. Ask user if they want to commit updated results
