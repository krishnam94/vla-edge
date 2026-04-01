"""ProbeFlow: Training-free adaptive flow matching for VLA models.

Reduces the number of denoising steps in flow matching VLAs by probing
trajectory linearity. If the flow is nearly linear (simple action),
skip most steps. If curved (complex action), use more steps.

Based on: ProbeFlow (arXiv:2603.17850)
Applied to: SmolVLA's 10-step Euler denoising loop

Expected speedup: 10 steps -> ~3-4 average = 2.5-3x on action expert.
Training-free: YES. No retraining, no new parameters.

Usage:
    from vla_edge.optimize.probeflow import apply_probeflow

    adapter = SmolVLAAdapter(device="cpu")
    adapter._ensure_loaded()
    stats = apply_probeflow(adapter._policy.model, epsilon=0.15)
    # Now inference uses adaptive steps instead of fixed 10
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProbeFlowStats:
    """Statistics from ProbeFlow adaptive step allocation."""

    total_calls: int = 0
    total_steps: int = 0
    min_steps: int = 0
    max_steps_used: int = 0
    step_histogram: dict[int, int] = field(default_factory=dict)

    @property
    def avg_steps(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_steps / self.total_calls

    @property
    def speedup(self) -> float:
        """Speedup over fixed N steps."""
        if self.avg_steps == 0:
            return 1.0
        return self.max_steps_used / self.avg_steps


def apply_probeflow(
    flow_model: Any,
    epsilon: float = 0.15,
    n_min: int = 2,
    n_max: int | None = None,
    delta_n: int = 2,
    dt_probe: float = 0.5,
) -> ProbeFlowStats:
    """Monkey-patch a VLAFlowMatching model to use ProbeFlow adaptive stepping.

    Args:
        flow_model: A VLAFlowMatching model instance (from SmolVLA's policy.model).
        epsilon: Sensitivity parameter. Lower = more steps (safer). Higher = fewer steps (faster).
            Recommended: 0.10-0.20. Default 0.15 balances speed and accuracy.
        n_min: Minimum denoising steps (always run at least this many).
        n_max: Maximum steps (defaults to model's config.num_steps).
        delta_n: Step increment when linearity is low.
        dt_probe: Lookahead timestep for the probe (fraction of total trajectory).

    Returns:
        ProbeFlowStats object that accumulates statistics across calls.
        Access stats.avg_steps, stats.speedup after running inference.
    """
    import torch

    if n_max is None:
        n_max = getattr(flow_model.config, "num_steps", 10)

    stats = ProbeFlowStats(min_steps=n_min, max_steps_used=n_max)

    # Save original for remove_probeflow()
    flow_model._original_sample_actions = flow_model.sample_actions

    def probeflow_sample_actions(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs,
    ):
        """ProbeFlow-enhanced sample_actions with adaptive step count."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, flow_model.config.chunk_size, flow_model.config.max_action_dim)
            noise = flow_model.sample_noise(actions_shape, device)

        # Compute prefix embeddings and KV cache (same as original)
        prefix_embs, prefix_pad_masks, prefix_att_masks = flow_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = flow_model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=flow_model.config.use_cache,
            fill_kv_cache=True,
        )

        # === PROBEFLOW: Probe trajectory linearity ===
        x_0 = noise
        time_start = torch.tensor(1.0, dtype=torch.float32, device=device).expand(bsize)

        # Step 1: Initial velocity
        v_start = flow_model.denoise_step(
            x_t=x_0,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=time_start,
        )

        # Step 2: Probe step
        x_probe = x_0 + v_start * (-dt_probe)  # negative because time goes 1.0 -> 0.0
        time_probe = torch.tensor(1.0 - dt_probe, dtype=torch.float32, device=device).expand(bsize)

        v_probe = flow_model.denoise_step(
            x_t=x_probe,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=time_probe,
        )

        # Step 3: Cosine similarity between initial and probe velocities
        v_start_flat = v_start.reshape(bsize, -1)
        v_probe_flat = v_probe.reshape(bsize, -1)
        cos_sim = torch.nn.functional.cosine_similarity(v_start_flat, v_probe_flat, dim=1)
        avg_similarity = cos_sim.mean().item()

        # Step 4: Adaptive step allocation
        n_steps = int(min(n_max, max(n_min, n_min + ((1.0 - avg_similarity) / epsilon) * delta_n)))

        # Track stats
        stats.total_calls += 1
        stats.total_steps += n_steps
        stats.step_histogram[n_steps] = stats.step_histogram.get(n_steps, 0) + 1

        logger.debug(
            "ProbeFlow: similarity=%.3f, steps=%d/%d",
            avg_similarity,
            n_steps,
            n_max,
        )

        # === Run denoising with adaptive step count ===
        # Reuse v_start from probe (step 0 at t=1.0 is identical)
        dt = -1.0 / n_steps
        x_t = x_0 + dt * v_start  # First step reuses probe's v_start

        for step in range(1, n_steps):  # Start from 1, step 0 already done
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            v_t = flow_model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    # Monkey-patch
    flow_model.sample_actions = probeflow_sample_actions
    logger.info(
        "ProbeFlow applied: epsilon=%.2f, n_min=%d, n_max=%d, dt_probe=%.2f",
        epsilon,
        n_min,
        n_max,
        dt_probe,
    )

    return stats


def remove_probeflow(flow_model: Any) -> None:
    """Remove ProbeFlow patching and restore original behavior."""
    if hasattr(flow_model, "_original_sample_actions"):
        flow_model.sample_actions = flow_model._original_sample_actions
        del flow_model._original_sample_actions
        logger.info("ProbeFlow removed, original sample_actions restored")
