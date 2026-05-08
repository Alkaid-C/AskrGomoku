"""
Entropy-targeted temperature rescaling for prior softening.

The naive `p**alpha / Z` form gives an entropy that depends on the input's
shape. `rescale_to_entropy_np` instead solves for the per-row softmax
temperature that yields a requested target entropy via bisection in log-tau
space (24 iters, ~1.4e-6 nat precision). Degenerate near-onehot rows fall back
to plain softmax.

Used only by `mcts.py::_evaluate_with_cache` for prior softening:
target_H = H(softmax(logits)) * entropy_multiplier.
"""

import math

import numpy as np


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax on numpy."""
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _logsumexp_np(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def rescale_to_entropy_np(
    log_input: np.ndarray,
    target_H: np.ndarray,
    n_iter: int = 24,
) -> np.ndarray:
    """Solve per-row softmax temperature so the rescaled distribution has
    entropy `target_H`, via batched bisection in log-tau space.

    Args:
        log_input: [B, K] logits (float32). Illegal slots may be set to a very
            negative value (e.g. -1e9); they remain ~0 after softmax under any
            positive tau.
        target_H: [B] desired per-row entropy in nats (float32). Clamped to
            [0, log K].
        n_iter: bisection iterations. Initial log-tau window width = 24, so
            after 24 steps entropy precision ~24/2**24 ≈ 1.4e-6 nat.

    Returns:
        [B, K] rescaled probability distribution (rows sum to 1) in float32.
    """
    log_input = log_input.astype(np.float32, copy=False)
    K = log_input.shape[-1]
    H_max = math.log(K)
    target_H = np.clip(target_H.astype(np.float32, copy=False), 0.0, H_max)

    log_tau_lo = np.full_like(target_H, -12.0)
    log_tau_hi = np.full_like(target_H, 12.0)

    pp = np.empty_like(log_input)
    for _ in range(n_iter):
        log_tau_mid = 0.5 * (log_tau_lo + log_tau_hi)
        tau = np.exp(log_tau_mid)[:, None]  # [B, 1]
        scaled = log_input / tau
        log_pp = scaled - _logsumexp_np(scaled, axis=-1, keepdims=True)
        pp = np.exp(log_pp)
        H = -(pp * log_pp).sum(axis=-1)  # [B]
        # Higher tau ⇒ flatter ⇒ higher H. If H > target, sharpen: lower hi.
        too_flat = target_H < H
        log_tau_hi = np.where(too_flat, log_tau_mid, log_tau_hi)
        log_tau_lo = np.where(too_flat, log_tau_lo, log_tau_mid)

    # Degenerate rows (input is essentially onehot ⇒ H_in ≈ 0): fall back to
    # plain softmax, which preserves the onehot.
    natural = softmax_np(log_input, axis=-1)
    log_natural = log_input - _logsumexp_np(log_input, axis=-1, keepdims=True)
    H_in = -(natural * log_natural).sum(axis=-1)
    degenerate = H_in < 1e-6
    if degenerate.any():
        pp = np.where(degenerate[:, None], natural, pp)

    return pp.astype(np.float32, copy=False)
