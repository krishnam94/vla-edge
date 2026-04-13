"""Soft-knee compression for VLA action safety.

Applies audio DSP-inspired soft-knee limiting to robot actions.
Unlike hard clipping (np.clip), soft-knee provides C1 continuous
enforcement - actions smoothly approach bounds instead of hitting
a discontinuous wall. Hard bounds are still guaranteed.

Uses cubic Hermite interpolation in the knee region to smoothly
transition from identity (slope 1) to the hard limit (slope 0).
Knee width W controls the transition zone. W=0 degenerates to hard clip.

Reference: Giannoulis, Massberg, Reiss - "Digital Dynamic Range
Compressor Design" (JAES 2012) - adapted from dB domain to linear
action space with cubic Hermite knees for C1 continuity.
"""

from __future__ import annotations

import numpy as np


def _cubic_hermite(t: float, p0: float, m0: float, p1: float, m1: float) -> float:
    """Cubic Hermite interpolation. t in [0,1]."""
    t2 = t * t
    t3 = t2 * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1


def softknee_clip_1d(x: float, lo: float, hi: float, knee_width: float) -> float:
    """Soft-knee clip for a single scalar value.

    C1 continuous, guaranteed within [lo, hi]. Passthrough in the middle
    is identity (zero distortion for in-range actions).

    Knee regions use cubic Hermite interpolation:
    - Lower knee [lo, lo+W]: from (lo, slope=0) to (lo+W, slope=1)
    - Upper knee [hi-W, hi]: from (hi-W, slope=1) to (hi, slope=0)

    Args:
        x: Input value
        lo: Lower hard bound
        hi: Upper hard bound
        knee_width: Width of transition zone (0 = hard clip)

    Returns:
        Clipped value in [lo, hi]
    """
    w = knee_width
    if w <= 0 or (hi - lo) / 2 < w:
        return float(np.clip(x, lo, hi))

    if x <= lo:
        return lo
    elif x < lo + w:
        t = (x - lo) / w
        # Hermite: p0=lo, m0=0 (slope 0), p1=lo+w, m1=w (slope 1 * interval w)
        return _cubic_hermite(t, lo, 0.0, lo + w, w)
    elif x <= hi - w:
        return x
    elif x < hi:
        t = (x - (hi - w)) / w
        # Hermite: p0=hi-w, m0=w (slope 1 * interval w), p1=hi, m1=0 (slope 0)
        return _cubic_hermite(t, hi - w, w, hi, 0.0)
    else:
        return hi


def softknee_clip(
    actions: np.ndarray,
    lo: np.ndarray | float,
    hi: np.ndarray | float,
    knee_width: np.ndarray | float,
) -> np.ndarray:
    """Vectorized soft-knee clip for action arrays.

    C1 continuous, guaranteed within [lo, hi] for all elements.

    Args:
        actions: Shape (action_dim,) or (T, action_dim)
        lo: Lower bounds. Scalar or shape (action_dim,)
        hi: Upper bounds. Scalar or shape (action_dim,)
        knee_width: Knee width per joint. Scalar or shape (action_dim,).
            Controls how gradually actions approach the bound.
            W=0 is hard clip. W=0.3 is moderate. W>=(hi-lo)/2 is invalid.

    Returns:
        Soft-clipped actions, same shape as input. Guaranteed in [lo, hi].
    """
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float32), actions.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float32), actions.shape)
    w = np.broadcast_to(np.asarray(knee_width, dtype=np.float32), actions.shape)

    result = actions.copy().astype(np.float32)

    # Hard floor
    mask_floor = actions <= lo
    result[mask_floor] = lo[mask_floor]

    # Lower knee: cubic Hermite from (lo, slope=0) to (lo+w, slope=1)
    mask_lower = (actions > lo) & (actions < lo + w)
    t = (actions[mask_lower] - lo[mask_lower]) / w[mask_lower]
    t2 = t * t
    t3 = t2 * t
    p0 = lo[mask_lower]
    p1 = lo[mask_lower] + w[mask_lower]
    m1 = w[mask_lower]  # slope 1 * interval width
    result[mask_lower] = (2 * t3 - 3 * t2 + 1) * p0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1

    # Passthrough: identity (no distortion)
    # Already correct in result

    # Upper knee: cubic Hermite from (hi-w, slope=1) to (hi, slope=0)
    mask_upper = (actions > hi - w) & (actions < hi)
    t = (actions[mask_upper] - (hi[mask_upper] - w[mask_upper])) / w[mask_upper]
    t2 = t * t
    t3 = t2 * t
    p0 = hi[mask_upper] - w[mask_upper]
    p1 = hi[mask_upper]
    m0 = w[mask_upper]  # slope 1 * interval width
    result[mask_upper] = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1

    # Hard ceiling
    mask_ceil = actions >= hi
    result[mask_ceil] = hi[mask_ceil]

    return result


def softknee_clip_actions(
    actions: np.ndarray,
    bounds: np.ndarray,
    knee_width: np.ndarray | float = 0.2,
) -> np.ndarray:
    """Convenience wrapper matching clip_actions() signature.

    Args:
        actions: Shape (T, action_dim) or (action_dim,)
        bounds: Shape (action_dim, 2) with [lo, hi] per joint
        knee_width: Scalar or per-joint array

    Returns:
        Soft-clipped actions
    """
    action_dim = min(actions.shape[-1], len(bounds))
    result = actions.copy().astype(np.float32)

    lo = bounds[:action_dim, 0]
    hi = bounds[:action_dim, 1]

    if actions.ndim == 1:
        result[:action_dim] = softknee_clip(actions[:action_dim], lo, hi, knee_width)
    else:
        result[:, :action_dim] = softknee_clip(actions[:, :action_dim], lo, hi, knee_width)

    return result


def lookahead_softknee(
    action_chunk: np.ndarray,
    lo: np.ndarray | float,
    hi: np.ndarray | float,
    knee_width: np.ndarray | float,
    decay: float = 0.5,
) -> np.ndarray:
    """Lookahead-aware soft-knee clip for action chunks.

    For VLAs that predict action chunks (e.g., SmolVLA outputs 10-50 steps),
    lookahead begins smoothing before a violation occurs by propagating
    future overshoot backward through the chunk.

    Args:
        action_chunk: Shape (T, action_dim) - future action predictions
        lo, hi: Bounds per joint
        knee_width: Knee width per joint
        decay: How quickly future violations influence past steps (0-1)

    Returns:
        Smoothed action chunk with anticipatory compression.
    """
    lo_arr = np.broadcast_to(np.asarray(lo, dtype=np.float32), action_chunk.shape)
    hi_arr = np.broadcast_to(np.asarray(hi, dtype=np.float32), action_chunk.shape)

    n_steps = action_chunk.shape[0]

    # Step 1: Compute per-step overshoot magnitude
    overshoot = np.maximum(0, action_chunk - hi_arr) + np.maximum(0, lo_arr - action_chunk)

    # Step 2: Propagate future violations backward with exponential decay
    for t in range(n_steps - 2, -1, -1):
        overshoot[t] = np.maximum(overshoot[t], decay * overshoot[t + 1])

    # Step 3: Blend actions toward center proportional to anticipated overshoot
    center = (lo_arr + hi_arr) / 2.0
    span = hi_arr - lo_arr
    blend = np.clip(overshoot / np.maximum(span, 1e-6), 0, 0.5)

    result = action_chunk * (1 - blend) + center * blend

    # Step 4: Final soft-knee clip to guarantee hard bounds
    result = softknee_clip(result, lo_arr[0], hi_arr[0], knee_width)

    return result
