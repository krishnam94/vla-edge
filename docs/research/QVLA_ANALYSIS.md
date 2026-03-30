# QVLA Deep Technical Analysis

**Paper**: "Not All Channels Are Equal in Vision-Language-Action Model's Quantization"
**Authors**: Yuhao Xu, Yantai Yang, Zhenyang Fan, Yufan Liu, Yuming Li, Bing Li, Zhipeng Zhang (SJTU AutoLab)
**Venue**: ICLR 2026
**arXiv**: [2602.03782](https://arxiv.org/abs/2602.03782)
**Code**: [github.com/AutoLab-SAI-SJTU/QVLA](https://github.com/AutoLab-SAI-SJTU/QVLA)
**Date of analysis**: 2026-03-29

---

## TL;DR for vla-edge Integration

QVLA is the first quantization framework designed specifically for VLA models.
Instead of applying uniform INT4 or INT8 to all channels (like SmoothQuant/AWQ),
it measures how much each channel matters for **action prediction accuracy** and
assigns bit-widths per-channel: 16-bit for critical channels, 4-bit or 2-bit for
unimportant ones, and 0-bit (pruned) for useless ones. On OpenVLA-OFT, this
gives 98.9% of original performance at 29.2% VRAM, beating SmoothQuant by 22.6%.

---

## 1. The Core Problem: Why LLM Quantization Fails for VLAs

Standard LLM quantization methods (SmoothQuant, AWQ, GPTQ) optimize for
**text generation quality** - they minimize reconstruction error on weight
matrices or perplexity on text. This is wrong for VLAs because:

1. **VLAs output actions, not text.** A 2% error in token probabilities barely
   affects text quality. A 2% error in a 7-DoF action compound across 50
   timesteps of a manipulation task.

2. **Sensitivity is heterogeneous across channels.** Within a single layer, some
   channels are critical for action prediction (high sensitivity) while others
   barely affect the output. Uniform quantization wastes bits on unimportant
   channels and starves important ones.

3. **Error accumulation is catastrophic in robotics.** An autoregressive policy
   generates actions conditioned on previous actions. Small errors compound
   into trajectory divergence. The paper shows this with cumulative L2 error
   plots - QVLA maintains tighter error bounds over long horizons.

**Key observation (Figure 1)**: The projector and action head are the most
sensitive modules. The vision encoder is the most robust. Within any single
layer, channel sensitivity varies by orders of magnitude.

---

## 2. Action-Centric Channel Sensitivity - The Core Innovation

### What it is

A metric that quantifies how much quantizing a specific channel at a specific
bit-width affects the model's final action output. Not intermediate features,
not text perplexity - the actual robot action.

### The math

**Single-step action sensitivity** for layer l, channel c, at bit-width b:

```
s^(b)_{l,c} = E_{x~D} [||A_tilde^(b)_{l,c}(V,l) - A*(V,l)||^2_2]
```

Where:
- `A*` = full-precision reference action
- `A_tilde` = action produced when channel (l,c) is quantized to b bits
- `D` = calibration dataset (512 trajectories from LIBERO training demos)

**Cumulative sensitivity** (over a full episode of T steps):

```
S^(b)_{l,c} = E[sum_{t=1}^{T} ||A_tilde^(b)_{l,c}(V_t,l) - A*(V_t,l)||_2]
```

This captures long-horizon error accumulation - the signature problem of VLA
quantization.

### Why this matters

SmoothQuant and AWQ measure weight reconstruction error or activation outliers.
QVLA measures what actually matters: does the robot do the right thing? Two
channels with identical weight magnitudes can have very different action
sensitivities because of how they contribute to the action head's computation.

---

## 3. Proxy Sensitivity Estimation (Making It Practical)

Computing exact sensitivity requires forward passes for every channel at every
bit-width. For a 7B model with thousands of channels, this is prohibitive.
QVLA uses a two-stage proxy approach.

### Stage 1: Hessian-based approximation

Uses first-order Taylor expansion to approximate the action sensitivity:

```
S^(b)_{l,c} ~ (sigma^(b)_{l,c})^2 * ||J_{A,X_{l,c}}||^2_F
```

Where:
- `sigma^(b)_{l,c}` = quantization noise (depends on bit-width)
- `J_{A,X_{l,c}}` = Jacobian of action w.r.t. channel output
- `||.||_F` = Frobenius norm

**Implementation** (from `sensitivity_hessian_proxy.py`):

1. Register forward hooks on all `nn.Linear` and `nn.Conv2d` layers in the
   language model and vision backbone
2. Run calibration batches through the model, accumulating input covariance
   matrices (the Hessian proxy H)
3. For each layer, compute the diagonal of the inverse Hessian via Cholesky
   decomposition with damping
4. For each bit-width, quantize each output channel and compute
   Hessian-weighted reconstruction error:
   `loss = sum((W - Q)^2 / diag_hinv^2)`

The output is a dict mapping `layer_name -> {proxy_0: tensor, proxy_2: tensor,
proxy_4: tensor, proxy_8: tensor}` where each tensor has per-channel loss
values.

**Quantization scheme used**: Symmetric per-channel with dynamic range:
```python
qmax = (1 << (bits-1)) - 1
scale = max_abs / qmax
quantized = round(input/scale) * scale
```

### Stage 2: Selective validation

Top-ranked channels (most sensitive) get validated with full forward passes
for precision. This is a refinement step, not the primary computation.

### Calibration data requirements

512 trajectories from LIBERO training demonstrations. The authors tested
128/256/512/1024 trajectories and found minimal difference (96.4% to 97.0%
success rate), suggesting the method is robust to calibration set size.

---

## 4. Channel-Wise Bit Allocation Algorithm

### The optimization problem

```
min_{b_{l,c}} sum_{l,c} s^(b_{l,c})_{l,c}

subject to: (1/N) sum_{l,c} b_{l,c} <= B_bar
            b_{l,c} in {0, 2, 4, 8, 16}
```

Where N = total channels and B_bar = average bit budget (e.g., 4 or 8).

### Greedy demotion algorithm

This is the core algorithm. Instead of searching the exponentially large space
of all possible bit assignments, QVLA uses a greedy approach:

1. **Initialize**: All channels at 16-bit
2. **Compute cost-effectiveness ratio** for each channel's next demotion:
   ```
   rho_{l,c} = (s^(b_lo) - s^(b_hi)) / (b_hi - b_lo)
   ```
   This is "how much action error per bit saved" - lower is better.
3. **Build min-heap** keyed by rho
4. **Greedily demote**: Pop the channel with the lowest rho (least
   action-sensitive per bit saved), demote it to the next lower bit-width
5. **Requeue**: After demotion, compute the cost of the NEXT demotion for
   that channel and push it back onto the heap
6. **Stop** when the average bit budget is met

**Implementation** (from `assign_gates_from_sensitivity.py`):

```python
heap = []  # min-heap of (unit_cost, step_id, layer, idx, next_bits)
while heap and avg_bit > target_avg:
    unit_cost, _, layer, idx, nb = heapq.heappop(heap)
    cur_bit = layer_bits[layer][idx]
    layer_bits[layer][idx] = nb
    # Push next demotion candidate
    next_nb = next_lower_bit(nb, bit_list_desc)
    if next_nb is not None:
        push_candidate(layer, idx, nb)
```

**Complexity**: O(C log C) where C = total channel count. Fast enough for a
7B model.

**Demotion stages**: 16 -> 8 -> 4 -> 2 -> 0

The final 2 -> 0 stage uses additional constraints (dual-threshold and
L0-style regularization) to prevent over-pruning.

### Output format

A JSON file mapping each layer to a list of per-channel bit-widths:
```json
{
  "assign": {
    "language_model.layers.0.self_attn.q_proj": [8, 4, 8, 16, 4, 2, ...],
    "language_model.layers.0.self_attn.k_proj": [4, 4, 8, 4, 0, 2, ...],
    ...
  },
  "stats": {
    "target_avg_bits": 4.0,
    "final_avg_bits": 3.95,
    "bit_hist": {0: 50, 2: 100, 4: 800, 8: 50}
  }
}
```

---

## 5. Unifying Quantization and Pruning (0-bit)

This is an elegant contribution. By including 0 in the bit-width options
{0, 2, 4, 8, 16}, pruning falls out naturally from the same optimization:

- A channel assigned 0 bits is set to all zeros (`.zero_()`)
- The greedy algorithm treats pruning as just another demotion step (2 -> 0)
- No separate pruning framework needed

**Implementation** (from `inject_fake_w.py`):
```python
for i in range(out_channels):
    bw = int(g[i].item())
    if bw >= 16:
        continue  # keep full precision
    if bw <= 0:
        w[i, :].zero_()  # prune channel entirely
        continue
    w[i, :] = _fake_quantize_tensor_sym(row, bw)
```

**Results show this matters**: Table 4 in the paper:
- Without 0-bit (channel-wise {2,4,8,16}): 76.7%, 7.5GB
- With 0-bit (channel-wise {0,2,4,8,16}): 76.8%, 7.0GB
- Gains 0.5GB memory with no accuracy loss by pruning ~1% of channels

This is clever because it means a single optimization handles both compression
techniques. AutoPrune (NeurIPS 2025, same lab) does pruning-only; QVLA
subsumes it.

---

## 6. How QVLA Differs from SmoothQuant/AWQ/GPTQ

| Aspect | SmoothQuant | AWQ | GPTQ | QVLA |
|--------|-------------|-----|------|------|
| Granularity | Per-tensor/per-token | Per-channel (weights) | Per-column (weights) | Per-channel (weights) |
| Bit allocation | Uniform (e.g., all W8A8) | Uniform with salient weight protection | Uniform with compensation | Mixed {0,2,4,8,16} per channel |
| Sensitivity metric | Activation outlier magnitude | Weight importance * activation magnitude | Weight reconstruction error | Action-space L2 deviation |
| Optimization target | Feature reconstruction | Weight reconstruction | Weight reconstruction | Robot action accuracy |
| Supports pruning | No | No | No | Yes (0-bit) |
| VLA-aware | No | No | No | Yes |
| Calibration | Text tokens | Text tokens | Text tokens | Robot trajectories |
| Latency overhead | Low (uniform compute) | Low | None (weight-only) | Low (uses fake quant with uniform kernels) |

**The fundamental difference**: SmoothQuant/AWQ/GPTQ try to preserve the model's
internal representations. QVLA tries to preserve the model's external behavior
(action accuracy). For text generation, internal representations correlate well
with output quality. For robotics, this correlation breaks down because the
action head creates a nonlinear bottleneck.

---

## 7. Experimental Results on LIBERO

### Models tested
- **OpenVLA** (7B params) - the standard VLA baseline
- **OpenVLA-OFT** - OpenVLA with Orthogonal Finetuning (stronger baseline)
- **UniVLA-7B** - another VLA architecture (Appendix D)

### LIBERO benchmark
Four task suites with 10 tasks each, 50 trials per task:
- **LIBERO-Spatial**: Spatial reasoning tasks
- **LIBERO-Object**: Object manipulation tasks
- **LIBERO-Goal**: Goal-conditioned tasks
- **LIBERO-Long**: Long-horizon multi-step tasks

### Weight-Activation Quantization (W4A4)

| Model | Method | Spatial | Object | Goal | Long | Avg | VRAM | Speedup |
|-------|--------|---------|--------|------|------|-----|------|---------|
| OpenVLA | FP16 baseline | 84.8 | 88.4 | 79.8 | 52.2 | 76.5 | 15.2GB | 1.00x |
| OpenVLA | SmoothQuant | 72.8 | 72.0 | 60.6 | 47.6 | 63.2 | 4.3GB | 1.41x |
| OpenVLA | OmniQuant | 82.0 | 84.8 | 76.6 | 49.6 | 73.3 | 4.3GB | 1.40x |
| OpenVLA | **QVLA** | **84.4** | **87.6** | **78.8** | **53.0** | **76.0** | 4.3GB | 1.47x |
| OFT | FP16 baseline | 96.8 | 98.0 | 96.8 | 96.0 | 97.1 | 15.4GB | 1.00x |
| OFT | SmoothQuant | 79.4 | 82.4 | 71.0 | 60.8 | 73.4 | 4.5GB | 1.42x |
| OFT | OmniQuant | 94.6 | 95.0 | 94.4 | 91.4 | 93.9 | 4.5GB | 1.41x |
| OFT | **QVLA** | **96.2** | **97.6** | **96.4** | **93.8** | **96.0** | 4.5GB | 1.49x |

**Key takeaway**: At W4A4, QVLA retains 99.3% of OpenVLA's performance while
SmoothQuant drops to 82.6%. The gap is even larger on OFT (98.9% vs 75.6%).

### Weight-Only Quantization (W4A16)

| Model | Method | Avg | VRAM |
|-------|--------|-----|------|
| OpenVLA | AWQ | 70.8 | 4.3GB |
| OpenVLA | **QVLA** | **76.5** | 4.3GB |
| OFT | AWQ | 92.5 | 4.5GB |
| OFT | **QVLA** | **96.7** | 4.5GB |

QVLA beats AWQ by 5.7% on OpenVLA and 4.2% on OFT in weight-only mode.

### W8A8 Results (closer to baseline)

| Model | Method | Avg | VRAM | Speedup |
|-------|--------|-----|------|---------|
| OpenVLA | SmoothQuant | 75.8 | 7.1GB | 1.38x |
| OpenVLA | **QVLA** | **76.3** | 7.1GB | 1.42x |
| OFT | SmoothQuant | 96.0 | 7.2GB | 1.32x |
| OFT | **QVLA** | **96.4** | 7.2GB | 1.36x |

At INT8, the gap is smaller (0.5% for OpenVLA, 0.4% for OFT). This makes
sense - with more bits, uniform allocation wastes fewer bits.

### Real-world results (Table 5)

Tested on bimanual robot (IMETA-Y1 arms + Orbbec DaBai cameras) with pi0:
- Pick white pen: 8/10 (both FP and quantized)
- Pick potato chips: 7/10 vs 6/10
- Fold towels: 4/10 vs 5/10 (quantized actually better on one task)
- Average: 63.3%, Speedup: 1.28x

---

## 8. Ablation Studies

### Channel-wise vs Layer-wise (Table 3)

| Granularity | INT4 Avg | INT8 Avg |
|-------------|----------|----------|
| Layer-wise | 74.8% | 74.9% |
| Channel-wise | **76.5%** | **76.8%** |

Channel-level allocation gains ~2% over layer-level at both bit budgets.

### Impact of bit-width options (Table 4)

| Config | Bit options | Avg | VRAM |
|--------|-----------|-----|------|
| FP baseline | {16} | 76.5% | 15.2GB |
| Uniform INT8 | {8} | 74.6% | 7.6GB |
| Mixed no-prune | {2,4,8,16} | 76.7% | 7.5GB |
| **Mixed + prune** | **{0,2,4,8,16}** | **76.8%** | **7.0GB** |

### Calibration set size (Table 9)

| Trajectories | Success Rate |
|-------------|-------------|
| 128 | 96.4% |
| 256 | 96.8% |
| 512 | 97.0% |
| 1024 | 96.7% |

Robust to calibration size. 512 is the sweet spot.

### Bit distribution at INT4 budget

Typical allocation: 1% at 0-bit, 5% at 2-bit, 22% at 4-bit, 56% at 8-bit,
16% at 16-bit. The "average 4-bit" is achieved by mixing, not by making
everything 4-bit.

---

## 9. Limitations

The paper doesn't have an explicit limitations section, but from the method
and experiments:

1. **Requires calibration data from the target domain.** Uses 512 LIBERO
   trajectories. Unclear how well sensitivity transfers across tasks/domains.
   If you're deploying on a different robot with different tasks, you may
   need new calibration data.

2. **Tested on limited model diversity.** Only OpenVLA, OpenVLA-OFT, UniVLA,
   and pi0. Not tested on diffusion-based action heads (pi0's flow matching
   head is only tested in real-world, not the full ablation). QuantVLA
   specifically targets diffusion transformers, which QVLA doesn't address.

3. **Projector and action head kept at full precision.** The paper excludes
   `projector.*`, `action_head`, and `language_model.lm_head` from
   quantization. This means the method only compresses the backbone, not the
   most sensitive modules. For small models, these excluded parameters may be
   a significant fraction.

4. **Adjacent-only demotion may miss global optimum.** The greedy algorithm
   only considers the next lower bit-width (16->8->4->2->0). It can't skip
   levels (e.g., 16->4 directly). This is a local optimization, not global.

5. **Fake quantization only.** The released code uses fake quantization
   (simulate quantized values in FP32). Actual INT4/INT8 kernels for
   mixed-precision per-channel execution are not provided. Real speedups
   depend on kernel availability.

6. **LIBERO-centric evaluation.** LIBERO is simulation-only with relatively
   simple manipulation tasks. Real-world results are limited (3 tasks, 10
   trials each). The CALVIN benchmark is only in the appendix.

7. **No latency breakdown.** The 1.49x speedup number doesn't distinguish
   between memory savings (enabling larger batch sizes) and actual
   computational speedup from lower-precision ops. On edge hardware with
   unified memory, the memory savings alone could be transformative.

8. **Static allocation.** The bit assignments are fixed at calibration time.
   DyQ-VLA addresses this by dynamically adjusting precision based on
   real-time kinematic state (see Section 11).

---

## 10. Comparison: QVLA vs DyQ-VLA

DyQ-VLA (arXiv: [2603.07904](https://arxiv.org/abs/2603.07904)) is the main
follow-up that addresses QVLA's static allocation limitation.

| Aspect | QVLA | DyQ-VLA |
|--------|------|---------|
| Allocation | Static per-channel | Dynamic per-timestep |
| Trigger | Calibration-time sensitivity | Real-time kinematic state |
| Weight quant | Mixed {0,2,4,8,16} | Fixed W4 |
| Activation quant | Uniform per bit budget | Dynamic {2,4,8} per timestep |
| When to use high precision | Sensitive channels always | Fine manipulation phases |
| When to use low precision | Insensitive channels always | Coarse movement phases |
| LIBERO avg | 76.0% (W4A4) | 76.1% (W4AX) |
| Memory | 29.2% of original | 30.9% of original |
| Speedup | 1.49x | 1.49x |

**DyQ-VLA's key metrics:**
- Motion Fineness: `M_t = 1 - ||a_t^xyz||_2 / mu_max` (higher = finer motion)
- Angular Jerk: `J_t = ||a_t^rot - a_{t-1}^rot||_2 / nu_max`
- Fused: `S_t = max(0, lambda * M_tilde + (1-lambda) * J_tilde)`

When the robot is doing precise manipulation (high S_t), it uses 8-bit
activations. During coarse movement (low S_t), it drops to 2-bit. This
switching is constant-time via a precomputed lookup table.

**For vla-edge**: These approaches are complementary. QVLA for weight
quantization, DyQ-VLA for dynamic activation precision. We could implement
both.

---

## 11. The Broader VLA Quantization Landscape (as of March 2026)

| Method | Venue | Key Idea | Target |
|--------|-------|----------|--------|
| **QVLA** | ICLR 2026 | Action-centric channel sensitivity | Weights (mixed precision) |
| **DyQ-VLA** | arXiv Mar 2026 | Kinematic-state-driven dynamic quant | Activations (dynamic) |
| **QuantVLA** | arXiv Feb 2026 | Scale-calibrated PTQ for diffusion heads | DiT action heads |
| **EaqVLA** | arXiv May 2025 | Encoding-aligned modular mixed precision | Cross-module alignment |
| **HBVLA** | arXiv Feb 2026 | 1-bit post-training quantization | Extreme compression |

**QuantVLA** is notable: it's the first to successfully quantize diffusion
transformer (DiT) action heads using Attention Temperature Matching and
Output Head Balancing. Tested on pi0.5 and GR00T N1.5 (97.6% and 88.0%
respectively, 70% memory savings). QVLA explicitly avoids quantizing the
action head.

**EaqVLA** discovered that GPTQ completely breaks the projector module
because its weight compensation disrupts modality mapping. This aligns with
QVLA's decision to keep the projector at full precision.

---

## 12. Integration Plan for vla-edge

### What to implement

The QVLA algorithm has three components that map cleanly to our optimize module:

1. **Sensitivity profiler** (`optimize/sensitivity.py`)
   - Register hooks on Linear/Conv2d layers
   - Accumulate Hessian proxy from calibration trajectories
   - Output per-channel sensitivity at each bit-width
   - This is the most complex piece - ~300 lines of code

2. **Bit allocator** (`optimize/bit_allocator.py`)
   - Greedy demotion algorithm with min-heap
   - Input: sensitivity profile + bit budget
   - Output: per-channel bit assignment JSON
   - This is straightforward - ~150 lines

3. **Quantizer** (`optimize/quantize.py`)
   - Apply fake quantization using bit assignment
   - Symmetric per-channel quantization
   - 0-bit channels zeroed out
   - Export quantized checkpoint
   - ~200 lines

### Module exclusion rules

Following QVLA, we should NEVER quantize:
- `projector.*` (modality bridge - extremely sensitive)
- `action_head` (final action prediction)
- `language_model.lm_head` (token prediction head)

Only target `language_model.*` and `vision_backbone.*` Linear and Conv2d layers.

### Calibration data strategy

QVLA uses LIBERO training demos. For vla-edge, we need:
- A `CalibrationDataset` abstraction that works with any task
- Default: 512 trajectories (QVLA shows this is sufficient)
- Support LIBERO, CALVIN, and custom datasets
- Store as JSONL (image path + instruction + action sequence)

### What we can simplify for v0.2

- Skip Stage 2 (selective validation) - the Hessian proxy is good enough
- Skip 0-bit pruning initially (just use {2, 4, 8, 16})
- Use fake quantization for profiling, defer real INT4 kernels to v0.3
- Hardcode the module exclusion list rather than making it configurable

### What we should NOT simplify

- Per-channel granularity is essential (layer-wise loses 2%)
- The action-centric sensitivity metric is the whole point
- The greedy algorithm - it's already efficient enough (O(C log C))

### Compatibility with GGUF path

QVLA operates at the PyTorch level (fake quantization on nn.Module). Our
current Jetson path uses llama.cpp + GGUF. These are complementary:

- QVLA for **determining which channels need which precision**
- GGUF for **runtime execution** on Jetson

The bit allocation JSON from QVLA could inform a custom GGUF quantization
that varies bit-width per channel. llama.cpp supports group-wise quantization
(Q4_K_M already does this at the group level). Extending to channel-level
mixed precision in GGUF is future work but architecturally feasible.

### Dependency on upstream code

The QVLA repo depends on the full OpenVLA codebase. For vla-edge, we should
reimplement the core algorithms (sensitivity estimation + greedy allocation)
without that dependency. The algorithms are clean and self-contained:
- Hessian proxy: ~100 lines of PyTorch
- Greedy allocation: ~80 lines of pure Python (heapq)
- Fake quantization: ~50 lines of PyTorch

---

## 13. Key Takeaways for Manning Book (Ch 9-11)

1. **Standard LLM quantization is not enough for robots.** SmoothQuant drops
   OpenVLA-OFT from 97.1% to 73.4% at W4A4. QVLA retains 96.0%. The
   difference is action-awareness.

2. **Channel sensitivity varies by orders of magnitude.** This is the key
   insight worth a figure in the book.

3. **Quantization and pruning are the same optimization.** 0-bit = pruned.
   Elegant framing worth teaching.

4. **512 calibration trajectories is enough.** Good news for practitioners -
   you don't need massive datasets for calibration.

5. **The projector is sacred.** Both QVLA and EaqVLA independently found that
   the vision-to-language projector must stay at full precision. This is a
   strong architectural insight.

---

## Sources

- [QVLA Paper (arXiv)](https://arxiv.org/abs/2602.03782)
- [QVLA Paper (HTML)](https://arxiv.org/html/2602.03782)
- [QVLA Code (GitHub)](https://github.com/AutoLab-SAI-SJTU/QVLA)
- [QVLA OpenReview](https://openreview.net/forum?id=TpL2nXanru)
- [DyQ-VLA Paper](https://arxiv.org/html/2603.07904)
- [QuantVLA Paper](https://arxiv.org/abs/2602.20309)
- [EaqVLA Paper](https://arxiv.org/abs/2505.21567)
- [HBVLA Paper](https://arxiv.org/abs/2602.13710)
- [AutoLab-SAI-SJTU GitHub](https://github.com/AutoLab-SAI-SJTU)
- [ICLR 2026 VLA Research Overview (Moritz Reuss)](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- [Awesome Model Quantization](https://github.com/Kai-Liu001/Awesome-Model-Quantization)
