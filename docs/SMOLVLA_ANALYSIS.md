# SmolVLA Deep Technical Analysis

Research for building `src/vla_edge/models/smolvla.py` - the first real model adapter.

Paper: [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
Model: [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
Blog: [huggingface.co/blog/smolvla](https://huggingface.co/blog/smolvla)

---

## 1. Architecture Overview

SmolVLA is a 450M parameter VLA with two components:

```
[RGB Images] --> SigLIP Vision Encoder (93M) --> Pixel Shuffle --> 64 tokens per frame
[Language]   --> SmolLM2 Tokenizer --> Token Embeddings
[State]      --> Linear Projection --> 1 token
                        |
                        v
              SmolVLM2-500M (first 16 of 32 layers) --> VLM features (prefix)
                        |
                        v
              Action Expert (~100M) - Flow Matching Transformer
                        |
                        v
              [Action Chunk: 50 continuous actions]
```

### Component Breakdown

| Component | Model | Params | Details |
|-----------|-------|--------|---------|
| Vision Encoder | SigLIP-B/16 | ~93M | Patch size 16, image 512x512, hidden 768 |
| Connector | Perceiver Resampler | small | 6 layers, 16 heads, 64 latents, head_dim 96 |
| Language Model | SmolLM2-360M (VLlama3) | ~360M | hidden 960, 32 layers (only 16 used), 15 attn heads, 5 KV heads, head_dim 64 |
| Action Expert | Custom Transformer | ~100M | hidden 720 (0.75 x 960), interleaved cross/self attention |
| **Total** | | **~450M** | |

### VLM Backbone: SmolVLM2-500M-Video-Instruct

Full config from HuggingFace:

```
Vision Config:
  hidden_size: 768
  image_size: 512
  patch_size: 16
  num_attention_heads: 12

Text Config (SmolLM2-360M, Llama architecture):
  hidden_size: 960
  num_hidden_layers: 32 (SmolVLA uses only first 16)
  num_attention_heads: 15
  num_key_value_heads: 5
  head_dim: 64
  intermediate_size: 2560
  vocab_size: 49280
  rope_theta: 100000
  pixel_shuffle_factor: 4

Perceiver Connector:
  resampler_depth: 6
  resampler_n_heads: 16
  resampler_n_latents: 64
  resampler_head_dim: 96

AutoClass: SmolVLMForConditionalGeneration
Loaded via: AutoModelForImageTextToText.from_pretrained()
```

---

## 2. The "Fast-Slow" Design

SmolVLA does NOT have a hierarchical fast-slow policy in the traditional sense (like pi0's slow VLM + fast action head). Instead, it achieves a similar effect through:

1. **Slow path (VLM prefix)**: Images + language + state are processed through 16 VLM layers. This is the expensive part but gets **cached** after one forward pass.

2. **Fast path (Action Expert)**: The action expert runs iteratively (10 flow matching steps) using cross-attention to the cached VLM features. Each denoising step is cheap because:
   - Expert hidden dim is 0.75x the VLM (720 vs 960)
   - Cross-attention reuses cached KV from prefix
   - Only action tokens (50) need processing per step

3. **Async inference**: Action execution is decoupled from prediction. While the robot executes the current chunk, the next chunk is computed in parallel.

### Attention Pattern

The action expert alternates between:
- **Self-attention** (every 2nd layer): Action tokens attend to each other with causal masking
- **Cross-attention** (other layers): Action tokens attend to cached VLM prefix features

Config: `self_attn_every_n_layers: 2`, `attention_mode: "cross_attn"`

---

## 3. Flow Matching (Action Generation)

SmolVLA uses conditional flow matching (NOT diffusion, NOT autoregressive tokenization).

### Training
```python
# Sample noise and timestep
noise = torch.randn_like(actions)  # shape: (B, chunk_size, action_dim)
t = Beta(1.5, 1.0).sample()       # scaled to [0.001, 1.0]

# Create noisy trajectory
x_t = t * noise + (1 - t) * actions

# Target velocity field
u_t = noise - actions

# Model predicts velocity, loss = MSE(predicted_v, u_t)
```

### Inference (10 steps by default)
```python
x_t = torch.randn(B, chunk_size, action_dim)  # start from noise
dt = -1.0 / num_steps  # = -0.1 for 10 steps

for step in range(num_steps):
    t = 1.0 - step / num_steps
    v_t = model.denoise_step(x_t, t, cached_prefix)
    x_t = x_t + dt * v_t  # Euler integration

return x_t  # denoised action chunk
```

Key detail: `num_steps: 10` (not 50 as initially described - the paper ablates both).

---

## 4. Action Space

- **Type**: Continuous (NOT tokenized)
- **Chunk size**: 50 actions per forward pass
- **n_action_steps**: 50 (execute all actions in chunk before re-predicting)
- **Action dim**: Configurable. Default config shows `shape: [6]` for SO100/SO101 arms
- **State dim**: Configurable. Default: `shape: [6]`
- **Padded to**: max_action_dim=32, max_state_dim=32 internally
- **Normalization**: MEAN_STD for state and action, IDENTITY for images

For SO100/SO101: 6-DOF (6 joint angles, no separate gripper)
Pi-ALOHA mode: 14-DOF (2x 7-DOF arms) with special gripper transforms

---

## 5. Image Preprocessing

```
Input: RGB images, 640x480 typical from cameras
  1. Resize to 512x512 with aspect ratio preservation (pad with zeros)
  2. Normalize: [0,1] -> [-1,1] for SigLIP
  3. Pixel shuffle compression: 512x512 -> 64 visual tokens per frame
  4. Supports up to 3 camera views (camera1, camera2, camera3)
  5. Missing cameras filled with -1 padding + zero attention masks

Config: resize_imgs_with_padding: [512, 512]
```

---

## 6. Loading in LeRobot

### AutoClass
SmolVLA does NOT use a standard HuggingFace AutoClass directly. It uses LeRobot's policy system:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Load policy (includes VLM + action expert)
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# The VLM backbone is loaded internally via:
# AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
# Wrapped in SmolVLMWithExpertModel class
```

### Internal Loading Chain
```
SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
  -> VLAFlowMatching.__init__()
    -> SmolVLMWithExpertModel.__init__()
      -> AutoModelForImageTextToText.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
      -> Truncate to 16 VLM layers
      -> Create lm_expert from config (AutoModel.from_config)
      -> AutoProcessor.from_pretrained() for tokenizer
```

### Pre/Post Processing (New Pipeline API)
```python
from lerobot.policies.factory import make_pre_post_processors

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    "lerobot/smolvla_base",
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Inference loop
batch = preprocess(observation_frame)
with torch.inference_mode():
    action = policy.select_action(batch)
    action = postprocess(action)
```

---

## 7. Key Source Files in LeRobot

```
src/lerobot/policies/smolvla/
  configuration_smolvla.py   - SmolVLAConfig (all hyperparameters)
  modeling_smolvla.py         - SmolVLAPolicy + VLAFlowMatching
  smolvlm_with_expert.py     - SmolVLMWithExpertModel (VLM + action expert)
  processor_smolvla.py        - Pre/post processing pipeline
```

---

## 8. CPU Inference

SmolVLA can run on CPU. No specific latency numbers are published. Expected behavior:
- 450M params in FP32 = ~1.8 GB RAM
- 450M params in BF16 = ~900 MB RAM
- Flow matching requires 10 forward passes through action expert per chunk
- VLM prefix is cached, so only computed once per chunk
- Estimated: 2-10 seconds per chunk on modern CPU (no published benchmarks)
- The model was designed to be trainable on a MacBook, so CPU inference is feasible

---

## 9. GGUF / llama.cpp Quantization

### LLM Backbone (SmolLM2-360M)
- SmolLM2 IS compatible with llama.cpp (Llama architecture)
- GGUF quantized versions exist: Q4_K_M (~200MB), Q8_0 (~360MB)
- The text_config uses `model_type: "llama"` - direct llama.cpp support

### Challenges for Full SmolVLA Quantization
- The **vision encoder (SigLIP)** is NOT supported by llama.cpp natively
- The **action expert** is a custom transformer - not llama.cpp compatible
- The **flow matching loop** requires iterative inference - not standard LLM generation

### Recommended Strategy for Edge (from CLAUDE.md)
```
Vision Encoder (SigLIP):  -> Torch-TensorRT on Jetson, ONNX on CPU
LLM Backbone (SmolLM2):   -> GGUF Q4_K_M via llama.cpp
Action Expert:             -> ONNX or keep in PyTorch (small enough)
```

This matches the vla-edge CLAUDE.md guidance: "llama.cpp + GGUF for LLM backbone, Torch-TensorRT for vision encoders (ViT)"

---

## 10. Known Issues (GitHub)

### Issue #1234: Training bugs on 3090
- Training gets stuck on RTX 3090

### Issue #1239: Model not working properly
- 30 episodes insufficient for SmolVLA (ACT works with 30, SmolVLA needs 50+)
- Need structured data: 50 episodes across 5 positions, 10 per position

### Issue #1370: Poor performance on custom tasks
- SmolVLA pretrained on SO-100 arm data only
- Fine-tuning on different embodiments (forklift) shows poor transfer
- Domain mismatch between pretrained weights and target task

### Issue #1702: Vision embeddings on wrong device
- SmolVLMVisionEmbeddings device mismatch during inference

### Issue #2210: Inference with SO-101 failed
- Compatibility issues between SO-100 pretrained model and SO-101

### Issue #2374: Pretrained smolvla_base problems
- Action scale mismatch: model outputs [-5, 5], robot expects [-100, 100]
- Post-processing pipeline naming mismatch with action keys
- Unresolved as of Feb 2026

---

## 11. Standalone Inference (Without Full LeRobot)

There is NO official standalone inference path. SmolVLA is tightly coupled to LeRobot.

For vla-edge, we need to extract the core inference logic:

```python
# Minimal inference pseudocode (what smolvla.py adapter needs to do)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# 1. Load VLM backbone
vlm = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

# 2. Truncate to 16 layers
vlm.model.text_model.layers = vlm.model.text_model.layers[:16]

# 3. Load action expert weights from lerobot/smolvla_base safetensors
# Action expert config: hidden=720, same num_layers=16, interleaved cross/self attn

# 4. Load normalization stats from preprocessor_config.json / postprocessor_config.json

# 5. For inference:
#    a. Preprocess image to 512x512, normalize to [-1,1]
#    b. Get 64 visual tokens from SigLIP + connector
#    c. Tokenize instruction
#    d. Project state to single token
#    e. Forward through 16 VLM layers (cache KV)
#    f. Run 10 flow matching steps with action expert
#    g. Denormalize output actions
```

---

## 12. Adapter Design for vla-edge

Based on this analysis, the `smolvla.py` adapter should:

```python
@register_model("smolvla")
class SmolVLAAdapter(VLAModel):
    """SmolVLA 450M adapter for vla-edge."""

    def __init__(self, device="cpu", dtype=torch.float32):
        # Option A: Use LeRobot's SmolVLAPolicy (requires lerobot dependency)
        # Option B: Extract minimal inference code (no lerobot dependency)
        #
        # Recommendation: Option A for v0.1, Option B for edge deployment
        pass

    def predict(self, image, instruction, state=None):
        # Returns: np.ndarray of shape (action_dim,) or (chunk_size, action_dim)
        pass

    @property
    def info(self):
        return ModelInfo(
            name="smolvla",
            param_count=450_000_000,
            architecture="VLM (SmolVLM2-500M) + Flow Matching Action Expert",
            action_dim=6,  # SO100 default, configurable
            required_image_size=(512, 512),
            supported_dtypes=["float32", "bfloat16"],
            source_url="https://huggingface.co/lerobot/smolvla_base",
        )
```

### Key Design Decisions
1. **Dependency**: Accept lerobot as optional dependency for v0.1. Mark with `pip install vla-edge[smolvla]`
2. **Chunk handling**: `predict()` returns full chunk (50 actions). Caller manages execution.
3. **Image size**: 512x512 (NOT 224x224 like many ViTs)
4. **State input**: Required. 6-DOF for SO100, configurable.
5. **Quantization path**: Split model - GGUF for LLM, ONNX for vision + expert

---

## 13. VLAb (Pretraining Framework)

HuggingFace released [VLAb](https://github.com/huggingface/VLAb) - the pretraining library for SmolVLA. It contains SmolVLA2 architecture and community dataset configs. For inference, they explicitly recommend LeRobot, not VLAb.

---

## Summary Table

| Property | Value |
|----------|-------|
| Total params | 450M |
| Vision encoder | SigLIP-B/16 (93M) |
| LLM backbone | SmolLM2-360M (Llama arch) |
| Action expert | ~100M flow matching transformer |
| Image resolution | 512x512 |
| Visual tokens | 64 per frame |
| VLM layers used | 16 (of 32) |
| Action type | Continuous (flow matching) |
| Chunk size | 50 |
| Flow matching steps | 10 |
| Action dim | 6 (SO100), configurable |
| State dim | 6 (SO100), configurable |
| Max cameras | 3 |
| AutoClass | AutoModelForImageTextToText (VLM), custom (full policy) |
| HF model class | SmolVLMForConditionalGeneration |
| CPU capable | Yes |
| FP32 size | ~1.8 GB |
| BF16 size | ~900 MB |
| LLM GGUF compatible | Yes (Llama arch) |
| Full GGUF | No (vision + expert need separate handling) |
