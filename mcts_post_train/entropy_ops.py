"""
Entropy-targeted temperature rescaling.

Both prior softening (pre-search) and visit-distribution sharpening (post-search)
are the same operation: softmax temperature scaling. The naive `^alpha / Z` form
gives an exponent that does not predictably control output entropy — the
entropy of the rescaled distribution depends on the input shape.

`rescale_to_entropy` instead solves directly for the per-row temperature that
yields a requested target entropy.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

LOG_EPS = 1e-30
MAX_ENTROPY = math.log(225)  # max entropy for K=225


def rescale_to_entropy(
    log_input: Tensor,
    target_H: Tensor,
    n_iter: int = 24,
) -> Tensor:
    """
    Rescale `softmax(log_input / tau)` per row so its entropy equals `target_H`.

    Works in both directions: `target_H` above the input's natural entropy
    flattens (tau > 1); below sharpens (tau < 1). Solves for tau via batched
    bisection in log-space — `softmax`'s entropy is monotone in `tau`.

    Args:
        log_input: [B, K] logits or log-probabilities. Only relative values
            within a row matter, so either is fine. Illegal-move slots may be
            set to a very negative value (e.g. -1e9) — they contribute zero
            mass after softmax under any tau.
        target_H: [B] desired per-row entropy in nats. Clamped to [0, log K].
        n_iter: bisection iterations. Each step halves the log-tau search
            window (initial width = 24); after 24 steps the entropy precision
            is ~24/2**24 ≈ 1.4e-6 nat — far below any downstream noise floor
            (test atol 1e-3, EMA step ~1e-3). 14–17 would suffice; 24 leaves
            safety margin at negligible extra cost (one softmax per iter).

    Returns:
        [B, K] rescaled probability distribution (rows sum to 1).
    """
    K = log_input.shape[-1]
    log_input = log_input.float()

    # Clamp target into the achievable range. ln(K) is the uniform-distribution
    # ceiling; 0 is the onehot floor. Outside this, no tau exists.
    H_max = math.log(K)
    target_H = target_H.clamp(min=0.0, max=H_max)

    # Bisect in log-tau space. tau ∈ [exp(-12), exp(12)] ≈ [6e-6, 1.6e5] is
    # wide enough to cover anything realistic for K=225.
    log_tau_lo = torch.full_like(target_H, -12.0)
    log_tau_hi = torch.full_like(target_H, 12.0)

    pp = torch.empty_like(log_input)
    for _ in range(n_iter):
        log_tau_mid = 0.5 * (log_tau_lo + log_tau_hi)
        tau = log_tau_mid.exp().unsqueeze(-1)  # [B, 1]
        scaled = log_input / tau
        log_pp = scaled - scaled.logsumexp(dim=-1, keepdim=True)
        pp = log_pp.exp()
        H = -(pp * log_pp).sum(dim=-1)  # [B]
        # Higher tau ⇒ flatter ⇒ higher H. If H > target, need to sharpen
        # (lower tau): move the upper bound down.
        too_flat = target_H < H
        log_tau_hi = torch.where(too_flat, log_tau_mid, log_tau_hi)
        log_tau_lo = torch.where(too_flat, log_tau_lo, log_tau_mid)

    # Degenerate rows (input is essentially onehot ⇒ H_in = 0): bisection
    # cannot raise entropy because there is no probability mass to spread.
    # Fall back to plain softmax of the input (preserves the onehot).
    H_in = -(F.softmax(log_input, dim=-1)
             * F.log_softmax(log_input, dim=-1)).sum(dim=-1)
    degenerate = H_in < 1e-6
    if degenerate.any():
        natural = F.softmax(log_input, dim=-1)
        pp = torch.where(degenerate.unsqueeze(-1), natural, pp)

    return pp
