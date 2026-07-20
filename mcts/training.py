"""
MCTS Training: Supervised Distillation

Trains the policy/value network to match MCTS search results:
- Policy: cross-entropy vs raw MCTS visit distribution
- Value: MSE vs MCTS root Q-value
- 8-fold dihedral augmentation (with distribution permutation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gomoku import LOGIT_MASK_VALUE

# ============================================================================
# Training Constants
# ============================================================================

TRAIN_BATCH_SIZE = 512
GRAD_CLIP_NORM = 16.0

# ============================================================================
# 8-Fold Distribution Permutation Tables
# ============================================================================

# Coordinate transforms matching enhancement.py's augment_batch_8fold:
# new_rows = [r, c, 14-r, 14-c, r, 14-r, c, 14-c]
# new_cols = [c, 14-r, 14-c, r, 14-c, c, r, 14-r]
#
# For each symmetry s, the forward mapping is:
#   old_flat = r*15 + c  ->  new_flat = new_rows[s]*15 + new_cols[s]
#
# For distributions, we need:
#   new_dist[new_flat] = old_dist[old_flat]
# Which is equivalent to:
#   new_dist = old_dist[inv_perm]  where inv_perm[new_flat] = old_flat


def _build_permutation_tables() -> np.ndarray:
    """Build 8 inverse permutation tables for distribution augmentation.

    Returns:
        [8, 225] int array where table[s, new_idx] = old_idx
    """
    coords = np.arange(225)
    r = coords // 15
    c = coords % 15

    new_rows = np.stack([r, c, 14 - r, 14 - c, r, 14 - r, c, 14 - c])         # [8, 225]
    new_cols = np.stack([c, 14 - r, 14 - c, r, 14 - c, c, r, 14 - r])          # [8, 225]
    forward = new_rows * 15 + new_cols  # [8, 225]: forward[s, old] = new

    # Build inverse: inv[s, new] = old
    inv = np.zeros_like(forward)
    for s in range(8):
        inv[s, forward[s]] = coords

    return inv


DIST_PERM_TABLES = _build_permutation_tables()  # [8, 225]
_PERM_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_perm_tables(device: torch.device) -> torch.Tensor:
    if device not in _PERM_CACHE:
        _PERM_CACHE[device] = torch.from_numpy(DIST_PERM_TABLES).long().to(device)
    return _PERM_CACHE[device]


# ============================================================================
# 8-Fold Augmentation for MCTS Data
# ============================================================================


def augment_mcts_batch_8fold(
    obs: torch.Tensor,
    visit_dists: torch.Tensor,
    masks: torch.Tensor,
    values: torch.Tensor,
    policy_weights: torch.Tensor,
    value_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply 8-fold dihedral augmentation to MCTS training data.

    Args:
        obs: [B, 3, 15, 15] observations
        visit_dists: [B, 225] normalized visit distributions
        masks: [B, 15, 15] legal move masks
        values: [B] target values
        policy_weights: [B] per-sample policy-loss weights
        value_weights: [B] per-sample value-loss weights

    Returns:
        Augmented (obs, visit_dists, masks, values, policy_weights,
        value_weights) with batch size 8*B
    """
    # Spatial transforms for obs and masks (same as enhancement.py)
    obs_t = obs.transpose(-2, -1)
    obs_r180 = obs.flip(-2, -1)
    masks_t = masks.transpose(-2, -1)
    masks_r180 = masks.flip(-2, -1)

    all_obs = torch.cat([
        obs, obs_t.flip(-1), obs_r180, obs_t.flip(-2),
        obs.flip(-1), obs.flip(-2), obs_t, obs_r180.transpose(-2, -1)
    ])
    all_masks = torch.cat([
        masks, masks_t.flip(-1), masks_r180, masks_t.flip(-2),
        masks.flip(-1), masks.flip(-2), masks_t, masks_r180.transpose(-2, -1)
    ])

    # Permute distributions using precomputed tables
    # DIST_PERM_TABLES[s] is inverse perm: new_dist = old_dist[inv_perm]
    perm_tables = _get_perm_tables(visit_dists.device)  # [8, 225]

    aug_dists = []
    for s in range(8):
        # For each symmetry: new_dist[i] = old_dist[inv[i]]
        aug_dists.append(visit_dists[:, perm_tables[s]])  # [B, 225]
    all_dists = torch.cat(aug_dists, dim=0)  # [8B, 225]

    # Values and per-sample weights are scalars per sample: just repeat.
    all_values = values.repeat(8)
    all_policy_weights = policy_weights.repeat(8)
    all_value_weights = value_weights.repeat(8)

    return all_obs, all_dists, all_masks, all_values, all_policy_weights, all_value_weights


# ============================================================================
# Training Function
# ============================================================================


def train_on_mcts_batch(
    model: nn.Module,
    samples: list[tuple],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_loss_coeff: float,
) -> dict:
    """
    Train model on MCTS self-play data via supervised distillation.

    Args:
        model: Policy/value network
        samples: Per-ply training tuples
            (obs uint8[3,15,15], dist f32[225], value f32,
             policy_weight f32, value_weight f32).
            Played roots use (1.0, 1.0); harvested internal nodes carry
            their own weights (0 policy weight = value-only).
        optimizer: Optimizer
        device: Torch device
        value_loss_coeff: Weight on the MSE value loss in the combined loss.

    Returns:
        Dict with training metrics (`policy_loss`, `value_loss`), reported as
        unweighted means over played-root (weight-1) samples only.
    """
    all_obs = []
    all_dists = []
    all_values = []
    all_masks = []
    all_policy_weights = []
    all_value_weights = []

    for obs, dist, val, policy_w, value_w in samples:
        all_obs.append(obs)
        all_dists.append(dist)
        all_values.append(val)
        # Derive legal mask from observation (empty = legal)
        occupied = obs[0] | obs[1]
        all_masks.append((1 - occupied).astype(np.uint8))
        all_policy_weights.append(policy_w)
        all_value_weights.append(value_w)

    # Convert to tensors
    obs_tensor = torch.from_numpy(np.stack(all_obs)).float().to(device)
    dist_tensor = torch.from_numpy(np.stack(all_dists)).float().to(device)
    mask_tensor = torch.from_numpy(np.stack(all_masks)).bool().to(device)
    value_tensor = torch.tensor(all_values, dtype=torch.float32, device=device)
    policy_w_tensor = torch.tensor(all_policy_weights, dtype=torch.float32, device=device)
    value_w_tensor = torch.tensor(all_value_weights, dtype=torch.float32, device=device)

    # Apply 8-fold augmentation
    obs_tensor, dist_tensor, mask_tensor, value_tensor, policy_w_tensor, value_w_tensor = (
        augment_mcts_batch_8fold(
            obs_tensor, dist_tensor, mask_tensor, value_tensor, policy_w_tensor, value_w_tensor
        )
    )
    n_augmented = obs_tensor.shape[0]

    # Gradient accumulation across micro-batches.
    #
    # Each loss term is a per-sample weighted mean: the weighted sum divided by
    # that term's own weight total. Rationale for the weighting scheme is in
    # mcts/CLAUDE.md, "Subtree harvesting" — not restated here.
    #
    # The denominators are computed once, globally over the augmented batch, so
    # they are constant across micro-batches and the accumulated gradient is
    # exact. Reported policy_loss / value_loss are unweighted means over the
    # played-root samples only, identified by value_weight == 1.0.
    policy_w_total = policy_w_tensor.sum().clamp_min(1.0)
    value_w_total = value_w_tensor.sum().clamp_min(1.0)
    optimizer.zero_grad()
    n_played_aug = 0
    total_policy_loss = 0.0
    total_value_loss = 0.0

    for start in range(0, n_augmented, TRAIN_BATCH_SIZE):
        end = min(start + TRAIN_BATCH_SIZE, n_augmented)
        mb_obs = obs_tensor[start:end]
        mb_dist = dist_tensor[start:end]
        mb_mask = mask_tensor[start:end]
        mb_values = value_tensor[start:end]
        mb_policy_w = policy_w_tensor[start:end]
        mb_value_w = value_w_tensor[start:end]
        mb_size = end - start

        target = mb_dist

        # Forward pass
        logits, pred_values = model(mb_obs)
        logits = logits.squeeze(1).view(mb_size, 225)
        pred_values = pred_values.squeeze(-1)

        # Mask illegal moves
        logits = logits.masked_fill(~mb_mask.view(mb_size, 225), LOGIT_MASK_VALUE)

        # Per-sample policy cross-entropy and value squared error
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(target * log_probs).sum(dim=-1)          # [mb_size]
        se = (pred_values - mb_values) ** 2             # [mb_size]

        # Weighted means: each term's weighted sum over its own global weight total.
        policy_term = (mb_policy_w * ce).sum()
        value_term = (mb_value_w * se).sum()
        loss = policy_term / policy_w_total + value_loss_coeff * value_term / value_w_total
        loss.backward()

        # Diagnostics over played-root samples only (value_weight == 1.0).
        with torch.no_grad():
            played = mb_value_w == 1.0
            n_played_mb = int(played.sum().item())
            if n_played_mb > 0:
                total_policy_loss += ce[played].sum().item()
                total_value_loss += se[played].sum().item()
                n_played_aug += n_played_mb

    # Clip gradients and step
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    denom = max(n_played_aug, 1)
    return {
        'policy_loss': total_policy_loss / denom,
        'value_loss': total_value_loss / denom,
    }
