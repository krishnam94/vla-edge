# Open Questions - vla-edge

More valuable than an idea backlog. Questions are generative.
Add 3/week. Answer 1/week. Cross out dead ends.

---

## Fundamental

- Why do VLAs decode actions autoregressively? Is there a parallel alternative that preserves quality?
- What is the theoretical minimum latency for a 450M VLA on 8GB shared VRAM?
- Can you formally verify a neural policy's safety bounds at compile time, not runtime?
- If 75% of latency is action generation, why does everyone optimize the vision encoder?
- What's the minimum number of action tokens needed for a 7-DoF robot arm? (FAST uses 30-60, but why?)

## Practical

- Does quantizing the vision encoder hurt more or less than quantizing the action decoder?
- What happens to safety metrics when you drop frames during high GPU load on Jetson?
- Can you hot-swap policies on a running robot without stopping? (Like blue-green deployment)
- How does unified memory on Jetson actually behave under concurrent CPU+GPU load?
- What's the real-world failure mode when a VLA model OOMs mid-inference on Jetson?

## Cross-Domain

- How does Spotify handle ML model A/B testing at the edge? Any lessons for robot policies?
- What can VLA deployment learn from real-time audio (fixed latency budgets, buffer underruns)?
- Game engines use LOD (level of detail) - is there an equivalent for model complexity based on task difficulty?
- Erlang's "let it crash" philosophy - what does fault-tolerant VLA inference look like?
- Mobile ML (TFLite) delegates subgraphs to different accelerators per-op - can we do this per-layer for VLAs?

## Contrarian

- What if bigger models are actually EASIER to deploy on edge? (more pruning headroom, better sparsity)
- What if safety validation is the wrong abstraction? (should it be continuous monitoring, not pre/post check?)
- What if we don't need VLAs at all for most manipulation tasks? (ACT/diffusion policies are smaller and faster)
- What if the leaderboard is more valuable than the toolkit itself?

## Cross-Domain (added 2026-03-29)

- Opus allocates bits per-band based on perceptual masking. Can we allocate denoising precision per-action-dimension based on "action masking" (some joints matter more than others)?
- Game engine DRS uses asymmetric scaling (degrade fast, recover slow). What's the right asymmetry ratio for VLA inference quality adjustment?
- SOTIF defines "triggering conditions" for AV failures. What are the triggering conditions for VLA manipulation failures? Has anyone systematically cataloged them?
- V8 resets its tier-up counter when type feedback changes. What's the VLA equivalent of "type feedback changed" - scene change? New object? Different instruction structure?
- The cerebellum gates control using prediction error, not task difficulty. Why does every VLA routing paper (DeeR-VLA, NanoVLA) route by difficulty rather than by prediction error?
- HFT systems act on partial information because being late is worse than being wrong. At what control frequency does this become true for robot manipulation? 5Hz? 10Hz? 20Hz?
- Waymo distills large teacher models into small student models for vehicle deployment. Could vla-edge's optimization pipeline include a distillation step, not just quantization?
