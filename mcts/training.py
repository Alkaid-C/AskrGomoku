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
from self_play import MCTSGameRecord

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply 8-fold dihedral augmentation to MCTS training data.

    Args:
        obs: [B, 3, 15, 15] observations
        visit_dists: [B, 225] normalized visit distributions
        masks: [B, 15, 15] legal move masks
        values: [B] target values

    Returns:
        Augmented (obs, visit_dists, masks, values) with batch size 8*B
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

    # Values just repeat
    all_values = values.repeat(8)

    return all_obs, all_dists, all_masks, all_values


# ============================================================================
# Training Function
# ============================================================================


def train_on_mcts_batch(
    model: nn.Module,
    game_records: list[MCTSGameRecord],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_loss_coeff: float,
) -> dict:
    """
    Train model on MCTS self-play data via supervised distillation.

    Args:
        model: Policy/value network
        game_records: MCTS game records with visit distributions
        optimizer: Optimizer
        device: Torch device
        value_loss_coeff: Weight on the MSE value loss in the combined loss.

    Returns:
        Dict with training metrics, including `kl_target_student` =
        CE(target, student) - H(target).
    """
    # Collect all training samples from game records
    all_obs = []
    all_dists = []
    all_values = []
    all_masks = []

    for record in game_records:
        for obs, dist, val in zip(record.observations, record.visit_distributions, record.root_values):
            all_obs.append(obs)
            all_dists.append(dist)
            all_values.append(val)
            # Derive legal mask from observation (empty = legal)
            occupied = obs[0] | obs[1]
            legal_mask = (1 - occupied).astype(np.uint8)
            all_masks.append(legal_mask)

    # Convert to tensors
    obs_tensor = torch.from_numpy(np.stack(all_obs)).float().to(device)
    dist_tensor = torch.from_numpy(np.stack(all_dists)).float().to(device)
    mask_tensor = torch.from_numpy(np.stack(all_masks)).bool().to(device)
    value_tensor = torch.tensor(all_values, dtype=torch.float32, device=device)

    # Apply 8-fold augmentation
    obs_tensor, dist_tensor, mask_tensor, value_tensor = augment_mcts_batch_8fold(
        obs_tensor, dist_tensor, mask_tensor, value_tensor
    )
    n_augmented = obs_tensor.shape[0]

    # Gradient accumulation across micro-batches
    optimizer.zero_grad()
    n_micro_batches = 0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_kl = 0.0

    for start in range(0, n_augmented, TRAIN_BATCH_SIZE):
        end = min(start + TRAIN_BATCH_SIZE, n_augmented)
        mb_obs = obs_tensor[start:end]
        mb_dist = dist_tensor[start:end]
        mb_mask = mask_tensor[start:end]
        mb_values = value_tensor[start:end]
        mb_size = end - start

        target = mb_dist

        # Forward pass
        logits, pred_values = model(mb_obs)
        logits = logits.squeeze(1).view(mb_size, 225)
        pred_values = pred_values.squeeze(-1)

        # Mask illegal moves
        logits = logits.masked_fill(~mb_mask.view(mb_size, 225), LOGIT_MASK_VALUE)

        # Policy loss: cross-entropy vs target distribution
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(target * log_probs).sum(dim=-1).mean()

        # Value loss: MSE vs MCTS root Q
        value_loss = F.mse_loss(pred_values, mb_values)

        # Combined loss (scaled for gradient accumulation)
        loss = (policy_loss + value_loss_coeff * value_loss) * (mb_size / n_augmented)
        loss.backward()

        # KL(target || student) = CE(target, student) - H(target)
        with torch.no_grad():
            kl = (policy_loss - (-(target * (target + 1e-10).log()).sum(dim=-1).mean())).item()

        total_policy_loss += policy_loss.item() * mb_size
        total_value_loss += value_loss.item() * mb_size
        total_kl += kl * mb_size
        n_micro_batches += mb_size

    # Clip gradients and step
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return {
        'policy_loss': total_policy_loss / n_micro_batches,
        'value_loss': total_value_loss / n_micro_batches,
        'kl_target_student': total_kl / n_micro_batches,
    }
