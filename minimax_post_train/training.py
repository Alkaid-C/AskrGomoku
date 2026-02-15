"""
Training Module

Contains the core training logic:
- GAE (Generalized Advantage Estimation) computation
- Loss computation (policy, value, entropy)
- Training batch processing with gradient accumulation
- Gradient conflict probing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional, Dict

from model import N_BLOCKS, N_SHARED_BLOCKS
from gomoku import (
    Trajectory,
    obs_batch_to_tensor, mask_batch_to_tensor,
    compute_returns, TEMPERATURE_TRAIN, LOG_PROB_MIN, LOGIT_MASK_VALUE
)
from enhancement import (
    augment_batch_8fold,
    TacticalStats,
)

# Sample weight exponent for per-episode weighting (0=per-step, 1=per-episode equal)
EPISODE_WEIGHT_ALPHA = 0.25


# ============================================================================
# Training Constants
# ============================================================================

# --- Training Duration ---
TOTAL_UPDATES = 65536

# --- Optimizer & Learning Rate ---
LEARNING_RATE = 1.0/8192
MIN_LR = 0.125/8192
LR_DECAY_MIDPOINT_PERCENTAGE = 0.75  # Decay midpoint at 75% of training
LR_DECAY_STEEPNESS = 0.5             # Transition spread over 50% of total training
WEIGHT_DECAY = 1e-8
GRAD_CLIP_NORM = 16.0

# --- Batching & Memory ---
EPISODES_PER_UPDATE = 64       # Episodes to collect before each training update
EPISODES_CHUNK_SIZE = 32       # Chunk size for gradient accumulation (saves VRAM)
TRAIN_BATCH_SIZE = 256 * 2     # Micro-batch size for training

# --- EMA Smoothing ---
EMA_WINDOW = 64                # Effective window for per-update EMA tracking (alpha = 1/window)
EVAL_WIN_RATE_EMA_WINDOW = 2   # Effective window for evaluation win rate EMA (evals happen less frequently)

# --- Entropy Bonus ---
ENTROPY_TARGET_START = 1.0      # Entropy bonus numerator at start (nats)
ENTROPY_TARGET_END = 0.1875      # Entropy bonus numerator at end (nats)
ENTROPY_BONUS_COEFF = 1/128.0   # Coefficient for entropy bonus
ENTROPY_DECAY_MIDPOINT_PERCENTAGE = 0.625  # Sigmoid midpoint as fraction of total training
ENTROPY_DECAY_STEEPNESS = 0.625  # Sigmoid width as fraction of total training

# --- Value Head & Advantage Estimation ---
VALUE_LOSS_COEFF = 1.0         # Weight for value head loss
GAE_LAMBDA = 0.95              # GAE lambda (0=TD(0), 1=MC)
VALUE_BASELINE_START = 512     # Update at which to start using value baseline

# --- Logging ---
PRINT_INTERVAL = 1             # Print stats every N updates
PROBE_INTERVAL = 64            # Probe gradient conflict every N updates (0 = disable)

# --- Search-Based Training (Post-Train) ---
M_RANK = 0.15        # Ranking margin (max) for inside loss
M_SEP = 0.15         # Separation margin for outside loss
ALPHA_SEP = 1.0      # Separation loss weight
LAMBDA_V = 1.0       # Value loss weight

# --- Progressive Unfreezing Schedule ---
HEADS_ONLY_UPDATES = 2048    # N: updates with only heads trainable
BLOCK_UNFREEZE_INTERVAL = 128  # M: interval between block unfreezes


# ============================================================================
# Entropy Target Helper
# ============================================================================

def compute_entropy_schedule(update: int) -> float:
    """
    Compute entropy schedule value with sigmoid decay.

    Used by both training loop (for entropy bonus scaling) and enhancement module
    (for off-policy rollout entropy threshold).

    Args:
        update: Current training update number

    Returns:
        Scheduled entropy value (in nats)
    """
    midpoint = TOTAL_UPDATES * ENTROPY_DECAY_MIDPOINT_PERCENTAGE
    steepness_k = 3.0 / (TOTAL_UPDATES * ENTROPY_DECAY_STEEPNESS)
    sigmoid_factor = (1.0 - np.tanh(steepness_k * (update - midpoint))) / 2.0
    entropy_schedule = ENTROPY_TARGET_END + (ENTROPY_TARGET_START - ENTROPY_TARGET_END) * sigmoid_factor
    return float(entropy_schedule)


# ============================================================================
# Gradient Conflict Probing
# ============================================================================

def probe_gradient_conflict_chunked(
    model: nn.Module,
    obs_chunks: List[torch.Tensor],
    next_obs_chunks: List[torch.Tensor],
    actions_chunks: List[torch.Tensor],
    masks_chunks: List[torch.Tensor],
    returns_chunks: List[torch.Tensor],
    value_targets_chunks: List[torch.Tensor],
    is_terminal_chunks: List[torch.Tensor],
    is_synthetic_chunks: List[torch.Tensor],
    weights_chunks: List[torch.Tensor],
    gae_advantages_chunks: List[torch.Tensor],
    next_values_chunks: List[Optional[torch.Tensor]],
    global_normalizers: Tuple[float, float],
    use_value_baseline: bool
) -> Dict[str, float]:
    """
    Probe gradient conflict using chunked gradient accumulation.

    Computes gradients for policy and value losses separately across chunks,
    accumulating them without retaining the full computational graph.
    This is memory-efficient for large batches.
    """
    # Save current gradients
    saved_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            saved_grads[name] = param.grad.clone()

    # Initialize accumulated gradients
    policy_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    value_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    global_policy_entropy_normalizer, global_value_normalizer = global_normalizers

    # Accumulate policy gradients across chunks
    model.zero_grad()
    for i in range(len(obs_chunks)):
        obs = obs_chunks[i]
        next_obs = next_obs_chunks[i]
        actions = actions_chunks[i]
        masks = masks_chunks[i]
        returns = returns_chunks[i]
        value_targets = value_targets_chunks[i]
        is_terminal = is_terminal_chunks[i]
        is_synthetic = is_synthetic_chunks[i]
        weights = weights_chunks[i]
        gae_advantages = gae_advantages_chunks[i]
        next_values = next_values_chunks[i]

        batch_size = obs.size(0)

        logits_grid, values = model(obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~masks, LOGIT_MASK_VALUE)
        logits_flat = logits.view(batch_size, 225)

        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled, validate_args=False)
        batch_log_probs = dist.log_prob(actions)
        batch_log_probs = torch.clamp(batch_log_probs, min=LOG_PROB_MIN)

        advantages = gae_advantages if use_value_baseline else returns
        policy_loss = -(weights * advantages * batch_log_probs).sum() / max(global_policy_entropy_normalizer, 1.0)

        policy_loss.backward()

        # Accumulate policy gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                policy_grads[name] += param.grad.detach()

        model.zero_grad()

    # Accumulate value gradients across chunks
    for i in range(len(obs_chunks)):
        obs = obs_chunks[i]
        next_obs = next_obs_chunks[i]
        value_targets = value_targets_chunks[i]
        is_terminal = is_terminal_chunks[i]
        is_synthetic = is_synthetic_chunks[i]
        weights = weights_chunks[i]
        next_values = next_values_chunks[i]

        values = model.forward_value_only(obs)
        values = values.squeeze(1)

        if next_values is None:
            with torch.no_grad():
                next_values_computed = model.forward_value_only(next_obs)
            next_values = next_values_computed.squeeze(1)

        effective_value_targets = torch.where(
            is_terminal,
            value_targets,
            -next_values.detach()
        )

        value_mse = F.mse_loss(values, effective_value_targets, reduction='none')
        value_loss_mask = ~is_synthetic
        value_loss = (weights * value_loss_mask.float() * value_mse).sum() / max(global_value_normalizer, 1.0)

        value_loss.backward()

        # Accumulate value gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                value_grads[name] += param.grad.detach()

        model.zero_grad()

    # Compute cosine similarity helper
    def cosine_sim(grad_dict_1: dict, grad_dict_2: dict, param_names: List[str]) -> Tuple[float, float, float]:
        grads_1 = []
        grads_2 = []
        for name in param_names:
            if name in grad_dict_1 and name in grad_dict_2:
                grads_1.append(grad_dict_1[name].flatten())
                grads_2.append(grad_dict_2[name].flatten())

        if not grads_1:
            return 0.0, 0.0, 0.0

        vec1 = torch.cat(grads_1)
        vec2 = torch.cat(grads_2)

        norm1 = vec1.norm().item()
        norm2 = vec2.norm().item()

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0, norm1, norm2

        cos = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        return cos, norm1, norm2

    # Categorize parameters
    stem_params = []
    trunk_params = {
        'shared_0-2': [], 'shared_3-5': [], 'shared_6-8': [], 'shared_9-11': [],
        'dual_se_0-2': [], 'dual_se_3-5': [],
    }
    all_trunk_stem_params = []

    for name in policy_grads.keys():
        if any(x in name for x in ['conv_3x3', 'conv_directional5', 'conv_full5',
                                     'conv_directional7', 'conv_full7', 'conv_1x1',
                                     'stem_norm']):
            stem_params.append(name)
            all_trunk_stem_params.append(name)
        elif 'shared_blocks.' in name:
            block_idx = int(name.split('shared_blocks.')[1].split('.')[0])
            layer_start = (block_idx // 3) * 3
            layer_key = f'shared_{layer_start}-{layer_start+2}'
            trunk_params[layer_key].append(name)
            all_trunk_stem_params.append(name)
        elif 'dual_se_blocks.' in name:
            # Skip head-specific SE params: se_policy only gets policy grads,
            # se_value only gets value grads, so they'd bias cosine sim toward 0
            if '.se_policy.' in name or '.se_value.' in name:
                continue
            block_idx = int(name.split('dual_se_blocks.')[1].split('.')[0])
            layer_start = (block_idx // 3) * 3
            layer_key = f'dual_se_{layer_start}-{layer_start+2}'
            trunk_params[layer_key].append(name)
            all_trunk_stem_params.append(name)

    # Compute cosine similarities
    overall_cos, overall_norm_p, overall_norm_v = cosine_sim(policy_grads, value_grads, all_trunk_stem_params)
    stem_cos, stem_norm_p, stem_norm_v = cosine_sim(policy_grads, value_grads, stem_params)

    trunk_metrics = {}
    for layer_key in sorted(trunk_params.keys()):
        layer_cos, layer_norm_p, layer_norm_v = cosine_sim(policy_grads, value_grads, trunk_params[layer_key])
        trunk_metrics[layer_key] = (layer_cos, layer_norm_p, layer_norm_v)

    # Restore original gradients
    model.zero_grad()
    for name, param in model.named_parameters():
        if name in saved_grads:
            param.grad = saved_grads[name]

    # Build metrics dictionary
    metrics = {
        'overall_cos_sim': overall_cos,
        'overall_policy_norm': overall_norm_p,
        'overall_value_norm': overall_norm_v,
        'stem_cos_sim': stem_cos,
        'stem_policy_norm': stem_norm_p,
        'stem_value_norm': stem_norm_v,
    }

    for layer_key in ['shared_0-2', 'shared_3-5', 'shared_6-8', 'shared_9-11', 'dual_se_0-2', 'dual_se_3-5']:
        cos, norm_p, norm_v = trunk_metrics.get(layer_key, (0.0, 0.0, 0.0))
        prefix = layer_key.replace('-', '_')
        metrics[f'{prefix}_cos_sim'] = cos
        metrics[f'{prefix}_policy_norm'] = norm_p
        metrics[f'{prefix}_value_norm'] = norm_v

    return metrics


# ============================================================================
# Training Functions
# ============================================================================

def _train_on_batch_internal(model: nn.Module, trajectories: List[Trajectory],
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             num_accumulation_steps: int,
                             update: int,
                             ema_entropy: float) -> Tuple[float, float, float, float, float, TacticalStats, Optional[dict]]:
    """
    Internal training function - processes a batch of trajectories.

    Returns:
        Tuple of (loss, mean_return, entropy, value_loss, raw_value_mse,
                  tactical_stats, probe_data)
        probe_data is a dict with chunked tensors for gradient probing, or None
    """
    use_value_baseline = (update >= VALUE_BASELINE_START)

    # Deferred collection: advantages are computed AFTER augmentation (no placeholder pattern)
    all_obs = []
    all_next_obs = []
    all_actions = []
    all_masks = []
    all_returns = []
    all_value_targets = []
    all_is_synthetic = []
    all_is_terminal = []
    all_weights = []
    all_returns_for_logging = []

    # Track trajectory structure for GAE computation
    sample_to_traj = []  # Maps sample index to (traj_idx, step_idx)

    for traj_idx, traj in enumerate(trajectories):
        returns = compute_returns(traj)

        current_steps = sum(1 for is_current in traj.is_current_policy if is_current)

        if current_steps == 0:
            continue

        current_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0

        for step_idx, (obs, action, legal_mask, z_t, is_current) in enumerate(zip(
            traj.observations, traj.actions, traj.legal_masks, returns, traj.is_current_policy
        )):
            if is_current:
                all_obs.append(obs)
                all_actions.append(action)
                all_masks.append(legal_mask)
                all_returns.append(z_t)
                all_value_targets.append(z_t)
                all_is_synthetic.append(False)
                all_weights.append(current_weight)
                all_returns_for_logging.append(z_t)
                sample_to_traj.append((traj_idx, step_idx))

                if step_idx + 1 < len(traj.observations):
                    all_next_obs.append(traj.observations[step_idx + 1])
                    all_is_terminal.append(False)
                else:
                    all_next_obs.append(np.zeros_like(obs))
                    all_is_terminal.append(True)

    # Empty tactical stats (no longer modifying training)
    tactical_stats = TacticalStats()

    # Convert to GPU tensors
    obs_tensor = obs_batch_to_tensor(all_obs, device)
    next_obs_tensor = obs_batch_to_tensor(all_next_obs, device)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=device)
    masks_tensor = mask_batch_to_tensor(all_masks, device)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
    value_targets_tensor = torch.tensor(all_value_targets, dtype=torch.float32, device=device)
    is_synthetic_tensor = torch.tensor(all_is_synthetic, dtype=torch.bool, device=device)
    is_terminal_tensor = torch.tensor(all_is_terminal, dtype=torch.bool, device=device)
    weights_tensor = torch.tensor(all_weights, dtype=torch.float32, device=device)

    # Apply all 8 symmetries
    aug_obs, aug_actions, aug_masks = augment_batch_8fold(obs_tensor, actions_tensor, masks_tensor)
    dummy_masks = torch.ones_like(masks_tensor)
    aug_next_obs, _, _ = augment_batch_8fold(next_obs_tensor, actions_tensor, dummy_masks)

    aug_returns = returns_tensor.repeat(8)
    aug_value_targets = value_targets_tensor.repeat(8)
    aug_is_synthetic = is_synthetic_tensor.repeat(8)
    aug_is_terminal = is_terminal_tensor.repeat(8)
    aug_weights = weights_tensor.repeat(8)

    # Compute advantages after augmentation (deferred collection - no placeholder pattern)
    B = len(all_obs)  # Original batch size before augmentation

    if use_value_baseline and GAE_LAMBDA < 1.0:
        # Compute base GAE advantages using 8-fold averaged values
        with torch.no_grad():
            aug_values_all = model.forward_value_only(aug_obs)  # [B*8, 1]

        aug_values_all = aug_values_all.squeeze(1)  # [B*8]

        # Reshape to separate augmentations: [8, B]
        values_per_aug = aug_values_all.view(8, B)

        # Average across augmentations (dim=0) to get ensemble estimate
        avg_values = values_per_aug.mean(dim=0).cpu().numpy()  # [B] numpy array

        # Build per-trajectory value lists
        traj_value_lists = [[] for _ in trajectories]
        traj_sample_indices = [[] for _ in trajectories]

        for sample_idx in range(B):
            traj_idx, step_idx = sample_to_traj[sample_idx]
            if traj_idx >= 0:  # Not synthetic
                traj_value_lists[traj_idx].append(avg_values[sample_idx])
                traj_sample_indices[traj_idx].append(sample_idx)

        # Compute base GAE per trajectory
        gae_advantages = np.zeros(B, dtype=np.float32)

        for traj_idx, traj in enumerate(trajectories):
            if not traj_value_lists[traj_idx]:
                continue

            values = np.array(traj_value_lists[traj_idx])
            sample_indices = traj_sample_indices[traj_idx]

            # GAE backward recursion
            gae = 0.0
            for i in reversed(range(len(values))):
                sample_idx = sample_indices[i]
                z_t = all_returns[sample_idx]

                # Compute delta
                if i == len(values) - 1:
                    # Terminal step
                    delta = z_t - values[i]
                else:
                    # Non-terminal: use next value
                    delta = -values[i + 1] - values[i]

                gae = delta - GAE_LAMBDA * gae
                gae_advantages[sample_idx] = max(0.0, gae)

        advantages_tensor = torch.tensor(gae_advantages, dtype=torch.float32, device=device)
        aug_gae_advantages = advantages_tensor.repeat(8)

        # Also average next_values for consistent TD targets
        with torch.no_grad():
            aug_next_values_all = model.forward_value_only(aug_next_obs).squeeze(1)  # [B*8]
        next_values_per_aug = aug_next_values_all.view(8, B)
        avg_next_values = next_values_per_aug.mean(dim=0)  # [B]
        aug_next_values_averaged = avg_next_values.repeat(8)
    else:
        # No GAE: use raw returns as advantages (clamped to non-negative)
        advantages_tensor = torch.clamp(returns_tensor, min=0.0)
        aug_gae_advantages = advantages_tensor.repeat(8)
        aug_next_values_averaged = None

    global_policy_entropy_normalizer = aug_weights.sum().item()
    value_loss_mask_global = ~aug_is_synthetic
    global_value_normalizer = (aug_weights * value_loss_mask_global.float()).sum().item()

    # Scheduled entropy value with sigmoid decay
    entropy_schedule = compute_entropy_schedule(update)

    accumulated_loss = 0.0
    accumulated_value_loss = 0.0
    accumulated_value_mse_sum = 0.0
    accumulated_value_mse_count = 0
    accumulated_weighted_entropy_sum = 0.0
    accumulated_weight_sum = 0.0

    # Collect data for chunked gradient probing (if needed)
    should_probe = PROBE_INTERVAL > 0 and (update + 1) % PROBE_INTERVAL == 0
    probe_obs_chunks = [] if should_probe else None
    probe_next_obs_chunks = [] if should_probe else None
    probe_actions_chunks = [] if should_probe else None
    probe_masks_chunks = [] if should_probe else None
    probe_returns_chunks = [] if should_probe else None
    probe_value_targets_chunks = [] if should_probe else None
    probe_is_terminal_chunks = [] if should_probe else None
    probe_is_synthetic_chunks = [] if should_probe else None
    probe_weights_chunks = [] if should_probe else None
    probe_gae_advantages_chunks = [] if should_probe else None
    probe_next_values_chunks = [] if should_probe else None

    # Process in micro-batches
    for batch_start in range(0, len(aug_obs), TRAIN_BATCH_SIZE):
        batch_end = min(batch_start + TRAIN_BATCH_SIZE, len(aug_obs))

        batch_obs = aug_obs[batch_start:batch_end]
        batch_next_obs = aug_next_obs[batch_start:batch_end]
        batch_actions = aug_actions[batch_start:batch_end]
        batch_masks = aug_masks[batch_start:batch_end]
        batch_returns = aug_returns[batch_start:batch_end]
        batch_value_targets = aug_value_targets[batch_start:batch_end]
        batch_gae_advantages = aug_gae_advantages[batch_start:batch_end]
        batch_is_synthetic = aug_is_synthetic[batch_start:batch_end]
        batch_is_terminal = aug_is_terminal[batch_start:batch_end]
        batch_weights = aug_weights[batch_start:batch_end]
        batch_size = batch_end - batch_start

        # Collect chunks for gradient probing (detached copies to avoid graph retention)
        if should_probe:
            probe_obs_chunks.append(batch_obs.detach())
            probe_next_obs_chunks.append(batch_next_obs.detach())
            probe_actions_chunks.append(batch_actions.detach())
            probe_masks_chunks.append(batch_masks.detach())
            probe_returns_chunks.append(batch_returns.detach())
            probe_value_targets_chunks.append(batch_value_targets.detach())
            probe_is_terminal_chunks.append(batch_is_terminal.detach())
            probe_is_synthetic_chunks.append(batch_is_synthetic.detach())
            probe_weights_chunks.append(batch_weights.detach())
            probe_gae_advantages_chunks.append(batch_gae_advantages.detach())

        logits_grid, values = model(batch_obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~batch_masks, LOGIT_MASK_VALUE)
        logits_flat = logits.view(batch_size, 225)

        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled, validate_args=False)
        entropies = dist.entropy()
        batch_log_probs = dist.log_prob(batch_actions)
        batch_log_probs = torch.clamp(batch_log_probs, min=LOG_PROB_MIN)

        values = values.squeeze(1)

        # Use 8-fold averaged next_values for consistent TD targets if available
        if aug_next_values_averaged is not None:
            next_values = aug_next_values_averaged[batch_start:batch_end]
            if should_probe:
                probe_next_values_chunks.append(next_values.detach())
        else:
            with torch.no_grad():
                next_values = model.forward_value_only(batch_next_obs)
            next_values = next_values.squeeze(1)
            if should_probe:
                probe_next_values_chunks.append(None)  # Will be computed in probe

        if not use_value_baseline:
            advantages = batch_returns
        else:
            advantages = batch_gae_advantages

        effective_value_targets = torch.where(
            batch_is_terminal,
            batch_value_targets,
            -next_values.detach()
        )

        value_mse = F.mse_loss(values, effective_value_targets, reduction='none')
        batch_value_loss_mask = ~batch_is_synthetic
        value_loss_mb = (batch_weights * batch_value_loss_mask.float() * value_mse).sum() / max(global_value_normalizer, 1.0)

        policy_loss_mb = -(batch_weights * advantages * batch_log_probs).sum() / max(global_policy_entropy_normalizer, 1.0)

        # Adaptive entropy bonus (ratio-based: schedule / current)
        entropy_bonus_scale = entropy_schedule / max(ema_entropy, 1e-8) if ema_entropy is not None else 1.0
        entropy_loss_mb = -(batch_weights * entropies).sum() / max(global_policy_entropy_normalizer, 1.0)

        loss_mb = (policy_loss_mb + VALUE_LOSS_COEFF * value_loss_mb + ENTROPY_BONUS_COEFF * entropy_bonus_scale * entropy_loss_mb) / num_accumulation_steps
        loss_mb.backward()

        accumulated_loss += loss_mb.item()
        accumulated_value_loss += value_loss_mb.item()
        accumulated_weighted_entropy_sum += (batch_weights * entropies).sum().item()
        accumulated_weight_sum += batch_weights.sum().item()
        accumulated_value_mse_sum += (value_mse * batch_value_loss_mask.float()).sum().item()
        accumulated_value_mse_count += batch_value_loss_mask.sum().item()

    total_loss_scalar = accumulated_loss
    weight_denom = max(accumulated_weight_sum, 1.0)
    total_entropy_scalar = accumulated_weighted_entropy_sum / weight_denom
    total_value_loss_scalar = accumulated_value_loss
    raw_value_mse = accumulated_value_mse_sum / max(accumulated_value_mse_count, 1)

    mean_return = np.mean(all_returns_for_logging) if all_returns_for_logging else 0.0

    # Package probe data if collected
    probe_data = None
    if should_probe:
        probe_data = {
            'obs_chunks': probe_obs_chunks,
            'next_obs_chunks': probe_next_obs_chunks,
            'actions_chunks': probe_actions_chunks,
            'masks_chunks': probe_masks_chunks,
            'returns_chunks': probe_returns_chunks,
            'value_targets_chunks': probe_value_targets_chunks,
            'is_terminal_chunks': probe_is_terminal_chunks,
            'is_synthetic_chunks': probe_is_synthetic_chunks,
            'weights_chunks': probe_weights_chunks,
            'gae_advantages_chunks': probe_gae_advantages_chunks,
            'next_values_chunks': probe_next_values_chunks,
            'global_normalizers': (global_policy_entropy_normalizer, global_value_normalizer),
            'use_value_baseline': use_value_baseline
        }

    return (total_loss_scalar, mean_return, total_entropy_scalar, total_value_loss_scalar,
            raw_value_mse, tactical_stats, probe_data)


def train_on_batch(model: nn.Module, trajectories: List[Trajectory],
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   chunk_size: int,
                   update: int,
                   ema_entropy: float) -> dict:
    """
    Train on a batch of trajectories with gradient accumulation.

    Returns:
        Dictionary with training statistics
    """
    num_chunks = (len(trajectories) + chunk_size - 1) // chunk_size
    chunks = [trajectories[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    optimizer.zero_grad()

    total_loss = 0.0
    total_returns = []
    total_entropy = 0.0
    total_value_loss = 0.0
    total_raw_value_mse = 0.0
    total_tactical_stats = TacticalStats()
    num_chunks_processed = 0
    collected_probe_metrics = None

    # Collect probe data from all chunks
    should_probe = PROBE_INTERVAL > 0 and (update + 1) % PROBE_INTERVAL == 0
    all_probe_data = [] if should_probe else None

    for i, chunk in enumerate(chunks):
        (loss, mean_return, mean_entropy, value_loss, raw_value_mse,
         tactical_stats, probe_data) = _train_on_batch_internal(
            model, chunk, optimizer, device,
            num_accumulation_steps=num_chunks,
            update=update,
            ema_entropy=ema_entropy
        )

        # Collect probe data from each chunk
        if should_probe and probe_data is not None:
            all_probe_data.append(probe_data)

        total_loss += loss * num_chunks
        total_entropy += mean_entropy
        total_value_loss += value_loss
        total_raw_value_mse += raw_value_mse

        # Aggregate tactical stats (now empty, kept for compatibility)
        total_tactical_stats.win_opportunities += tactical_stats.win_opportunities
        total_tactical_stats.win_hits += tactical_stats.win_hits
        total_tactical_stats.win_misses += tactical_stats.win_misses
        total_tactical_stats.block_opportunities += tactical_stats.block_opportunities
        total_tactical_stats.block_hits += tactical_stats.block_hits
        total_tactical_stats.block_misses += tactical_stats.block_misses

        num_chunks_processed += 1

        for traj in chunk:
            returns = compute_returns(traj)
            for z_t, is_current in zip(returns, traj.is_current_policy):
                if is_current:
                    total_returns.append(z_t)

    # Run gradient probe on full update (after all chunks processed, before optimizer step)
    if should_probe and all_probe_data:
        # Merge probe data from all chunks
        merged_obs_chunks = []
        merged_next_obs_chunks = []
        merged_actions_chunks = []
        merged_masks_chunks = []
        merged_returns_chunks = []
        merged_value_targets_chunks = []
        merged_is_terminal_chunks = []
        merged_is_synthetic_chunks = []
        merged_weights_chunks = []
        merged_gae_advantages_chunks = []
        merged_next_values_chunks = []

        # Concatenate chunks from all episode chunks
        for probe_data in all_probe_data:
            merged_obs_chunks.extend(probe_data['obs_chunks'])
            merged_next_obs_chunks.extend(probe_data['next_obs_chunks'])
            merged_actions_chunks.extend(probe_data['actions_chunks'])
            merged_masks_chunks.extend(probe_data['masks_chunks'])
            merged_returns_chunks.extend(probe_data['returns_chunks'])
            merged_value_targets_chunks.extend(probe_data['value_targets_chunks'])
            merged_is_terminal_chunks.extend(probe_data['is_terminal_chunks'])
            merged_is_synthetic_chunks.extend(probe_data['is_synthetic_chunks'])
            merged_weights_chunks.extend(probe_data['weights_chunks'])
            merged_gae_advantages_chunks.extend(probe_data['gae_advantages_chunks'])
            merged_next_values_chunks.extend(probe_data['next_values_chunks'])

        # Use global normalizers from first chunk (they should be the same across chunks)
        global_normalizers = all_probe_data[0]['global_normalizers']
        use_value_baseline = all_probe_data[0]['use_value_baseline']

        collected_probe_metrics = probe_gradient_conflict_chunked(
            model,
            merged_obs_chunks,
            merged_next_obs_chunks,
            merged_actions_chunks,
            merged_masks_chunks,
            merged_returns_chunks,
            merged_value_targets_chunks,
            merged_is_terminal_chunks,
            merged_is_synthetic_chunks,
            merged_weights_chunks,
            merged_gae_advantages_chunks,
            merged_next_values_chunks,
            global_normalizers,
            use_value_baseline
        )

    # Optimizer step (after all gradients accumulated)
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    avg_loss = total_loss / num_chunks_processed
    avg_entropy = total_entropy / num_chunks_processed
    avg_value_loss = total_value_loss / num_chunks_processed
    avg_raw_value_mse = total_raw_value_mse / num_chunks_processed
    mean_return = np.mean(total_returns) if total_returns else 0.0

    return {
        'loss': avg_loss,
        'mean_return': mean_return,
        'entropy': avg_entropy,
        'value_loss': avg_value_loss,
        'raw_value_mse': avg_raw_value_mse,
        'tactical_stats': total_tactical_stats,
        'probe_metrics': collected_probe_metrics
    }


# ============================================================================
# Progressive Unfreezing Logic
# ============================================================================

def get_unfrozen_param_groups(model: nn.Module, update: int) -> Tuple[List[str], int]:
    """
    Determine which parameters should be unfrozen based on update count.

    Unfreezing schedule:
    - [0, N): Only policy_* and value_* trainable
    - Then unfreeze trunk from the end:
      dual_se_blocks[5] -> ... -> dual_se_blocks[0] -> shared_blocks[11] -> ... -> shared_blocks[0]
    - After all blocks unfrozen: + stem

    Args:
        model: The neural network model
        update: Current training update number

    Returns:
        Tuple of (list of param name prefixes to unfreeze, number of unfrozen blocks)
    """
    # Always unfreeze heads
    unfrozen_prefixes = ['policy_', 'value_']
    unfrozen_blocks = 0

    if update < HEADS_ONLY_UPDATES:
        # Phase 1: heads only
        return unfrozen_prefixes, unfrozen_blocks

    # How many blocks to unfreeze
    updates_after_heads = update - HEADS_ONLY_UPDATES
    blocks_to_unfreeze = min(N_BLOCKS, 1 + updates_after_heads // BLOCK_UNFREEZE_INTERVAL)

    # Unfreeze blocks from the end (dual_se last block first, then backwards).
    for i in range(blocks_to_unfreeze):
        block_idx = N_BLOCKS - 1 - i
        if block_idx >= N_SHARED_BLOCKS:
            dual_idx = block_idx - N_SHARED_BLOCKS
            unfrozen_prefixes.append(f'dual_se_blocks.{dual_idx}.')
        else:
            unfrozen_prefixes.append(f'shared_blocks.{block_idx}.')
    unfrozen_blocks = blocks_to_unfreeze

    # After all blocks unfrozen, unfreeze stem
    updates_for_all_blocks = HEADS_ONLY_UPDATES + (N_BLOCKS - 1) * BLOCK_UNFREEZE_INTERVAL
    if update >= updates_for_all_blocks + BLOCK_UNFREEZE_INTERVAL:
        # Stem parameters
        stem_names = ['conv_3x3', 'conv_directional5', 'conv_full5',
                      'conv_directional7', 'conv_full7', 'conv_1x1', 'stem_norm']
        unfrozen_prefixes.extend(stem_names)

    return unfrozen_prefixes, unfrozen_blocks


def apply_freeze_schedule(model: nn.Module, update: int) -> int:
    """
    Apply freeze/unfreeze schedule to model parameters.

    Modifies requires_grad for each parameter based on the unfreezing schedule.

    Args:
        model: The neural network model
        update: Current training update number

    Returns:
        Number of blocks currently unfrozen
    """
    unfrozen_prefixes, unfrozen_blocks = get_unfrozen_param_groups(model, update)

    for name, param in model.named_parameters():
        # Check if this parameter should be unfrozen
        should_unfreeze = any(name.startswith(prefix) or prefix in name
                             for prefix in unfrozen_prefixes)
        param.requires_grad = should_unfreeze

    return unfrozen_blocks


def create_optimizer_for_unfrozen(model: nn.Module, update: int,
                                  lr: float = LEARNING_RATE,
                                  weight_decay: float = WEIGHT_DECAY) -> torch.optim.AdamW:
    """
    Create optimizer with only unfrozen parameters.

    Args:
        model: The neural network model
        update: Current training update number
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        AdamW optimizer for unfrozen parameters
    """
    # First apply freeze schedule
    apply_freeze_schedule(model, update)

    # Collect unfrozen parameters
    unfrozen_params = [p for p in model.parameters() if p.requires_grad]

    if not unfrozen_params:
        raise ValueError("No parameters to optimize!")

    return torch.optim.AdamW(unfrozen_params, lr=lr, weight_decay=weight_decay)


def maybe_update_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer,
                           update: int, prev_unfrozen_blocks: int,
                           lr: float = LEARNING_RATE,
                           weight_decay: float = WEIGHT_DECAY) -> Tuple[torch.optim.Optimizer, int]:
    """
    Check if optimizer needs to be recreated due to unfreezing schedule change.

    Args:
        model: The neural network model
        optimizer: Current optimizer
        update: Current training update number
        prev_unfrozen_blocks: Number of blocks unfrozen in previous update
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        Tuple of (optimizer, current_unfrozen_blocks)
        If unfreezing changed, returns new optimizer; otherwise returns same optimizer
    """
    _, current_unfrozen_blocks = get_unfrozen_param_groups(model, update)

    # Check if we're transitioning to unfreeze stem
    updates_for_all_blocks = HEADS_ONLY_UPDATES + (N_BLOCKS - 1) * BLOCK_UNFREEZE_INTERVAL
    stem_unfreeze_update = updates_for_all_blocks + BLOCK_UNFREEZE_INTERVAL

    # Check if unfreezing schedule changed
    need_new_optimizer = False

    if current_unfrozen_blocks != prev_unfrozen_blocks:
        need_new_optimizer = True
    elif update == stem_unfreeze_update:
        need_new_optimizer = True
    elif update == HEADS_ONLY_UPDATES:
        need_new_optimizer = True

    if need_new_optimizer:
        # Apply new freeze schedule and create new optimizer
        apply_freeze_schedule(model, update)
        unfrozen_params = [p for p in model.parameters() if p.requires_grad]
        new_optimizer = torch.optim.AdamW(unfrozen_params, lr=lr, weight_decay=weight_decay)
        return new_optimizer, current_unfrozen_blocks

    return optimizer, prev_unfrozen_blocks


# ============================================================================
# Search-Based Training Loss Functions
# ============================================================================

def compute_ranking_inside_loss(logits_flat: torch.Tensor,
                                sorted_candidates: torch.Tensor,
                                Q_norms: torch.Tensor) -> torch.Tensor:
    """
    Margin-based ranking loss for sorted candidates.

    Uses dynamic margin: margin(i,j) = min(m_rank, Q_norm[ci] - Q_norm[cj])

    Args:
        logits_flat: [B, 225] policy logits
        sorted_candidates: [B, TOP_K_SAMPLE] candidates sorted by Q descending
        Q_norms: [B, TOP_K_SAMPLE] normalized Q values (in [0, 1])

    Returns:
        Scalar ranking inside loss
    """
    B = logits_flat.size(0)
    device = logits_flat.device

    # Gather logits for c1, c2, c3, c4, c5
    L = logits_flat.gather(1, sorted_candidates)  # [B, 5]

    total_loss = torch.zeros(1, device=device)

    # Pairs: (c1,c2), (c2,c3), (c3,c4), (c4,c5)
    for i in range(4):  # i = 0, 1, 2, 3 -> pairs (1,2), (2,3), (3,4), (4,5)
        L_lower = L[:, i + 1]  # L(c_{i+2})
        L_higher = L[:, i]     # L(c_{i+1})
        Q_diff = Q_norms[:, i] - Q_norms[:, i + 1]
        margin = torch.clamp(Q_diff, max=M_RANK)
        loss_term = F.relu(L_lower - L_higher + margin)
        total_loss = total_loss + loss_term.mean()

    return total_loss


def compute_separation_outside_loss(logits_flat: torch.Tensor,
                                     all_candidates: torch.Tensor,
                                     c4_indices: torch.Tensor,
                                     legal_masks_flat: torch.Tensor) -> torch.Tensor:
    """
    Push non-candidates below c4.

    For each non-candidate action n: loss += ReLU(L(n) - L(c4) + m_sep)

    Args:
        logits_flat: [B, 225] policy logits
        all_candidates: [B, 6] all candidate indices
        c4_indices: [B] index of c4 (4th best candidate)
        legal_masks_flat: [B, 225] legal mask

    Returns:
        Scalar separation outside loss (per-sample mean, then averaged across samples)
    """
    B = logits_flat.size(0)
    device = logits_flat.device

    # Get L(c4) for each batch element
    L_c4 = logits_flat.gather(1, c4_indices.unsqueeze(1)).squeeze(1)  # [B]

    # Create candidate mask
    candidate_mask = torch.zeros(B, 225, dtype=torch.bool, device=device)
    for i in range(6):
        idx = all_candidates[:, i:i+1]  # [B, 1]
        candidate_mask.scatter_(1, idx, True)

    # Non-candidate legal actions
    non_candidate_mask = legal_masks_flat & ~candidate_mask

    # Compute loss for all positions, then mask
    L_diff = logits_flat - L_c4.unsqueeze(1) + M_SEP  # [B, 225]
    loss_per_pos = F.relu(L_diff)  # [B, 225]

    # Apply mask
    masked_loss = loss_per_pos * non_candidate_mask.float()

    # Per-sample mean, then average across samples
    num_non_candidates_per_sample = non_candidate_mask.sum(dim=1).float()  # [B]
    # Avoid division by zero (shouldn't happen in practice)
    num_non_candidates_per_sample = torch.clamp(num_non_candidates_per_sample, min=1.0)
    per_sample_loss = masked_loss.sum(dim=1) / num_non_candidates_per_sample  # [B]

    return per_sample_loss.mean()


def compute_search_value_loss(V_pred: torch.Tensor, V_target: torch.Tensor) -> torch.Tensor:
    """
    MSE between predicted and search backup value.

    Args:
        V_pred: [B] predicted values
        V_target: [B] target values from search

    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(V_pred, V_target)


# ============================================================================
# Search-Based Training Function
# ============================================================================

def train_on_search_samples(model: nn.Module,
                            samples: List,
                            optimizer: torch.optim.Optimizer,
                            device: torch.device,
                            batch_size: int = TRAIN_BATCH_SIZE) -> dict:
    """
    Train on search-generated samples.

    1. Apply 8-fold augmentation (transform candidates too)
    2. Compute policy ranking loss (inside + outside)
    3. Compute value MSE loss
    4. Total = PolicyLoss + λ_v * ValueLoss

    Args:
        model: Policy/value network
        samples: List of SearchSample objects
        optimizer: Optimizer
        device: torch device
        batch_size: Micro-batch size

    Returns:
        Dictionary with training metrics
    """
    from gomoku import SearchSample
    from enhancement import augment_batch_8fold, augment_candidates_8fold

    if not samples:
        return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'ranking_inside_loss': 0.0,
            'separation_outside_loss': 0.0,
            'value_loss': 0.0,
            'top1_acc': 0.0,
            'top3_acc': 0.0,
            'value_mse': 0.0
        }

    # Flatten samples from all games
    all_samples = []
    for game_samples in samples:
        if isinstance(game_samples, list):
            all_samples.extend(game_samples)
        else:
            all_samples.append(game_samples)

    if not all_samples:
        return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'ranking_inside_loss': 0.0,
            'separation_outside_loss': 0.0,
            'value_loss': 0.0,
            'top1_acc': 0.0,
            'top3_acc': 0.0,
            'value_mse': 0.0
        }

    # Convert to tensors
    obs_list = [s.obs for s in all_samples]
    sorted_cands_list = [s.sorted_candidates for s in all_samples]  # top 5
    all_cands_list = [s.all_candidates for s in all_samples]  # all 6
    Q_values_list = [s.Q_values for s in all_samples]  # Q values for top 5
    mask_list = [s.legal_mask for s in all_samples]
    V_targets = [s.V_target for s in all_samples]

    obs_tensor = torch.from_numpy(np.stack(obs_list)).float().to(device)
    masks_tensor = torch.from_numpy(np.stack(mask_list)).bool().to(device)
    actions_dummy = torch.zeros(len(all_samples), dtype=torch.long, device=device)  # For augment_batch_8fold

    V_targets_tensor = torch.tensor(V_targets, dtype=torch.float32, device=device)

    # Apply 8-fold augmentation to observations and masks
    aug_obs, _, aug_masks = augment_batch_8fold(obs_tensor, actions_dummy, masks_tensor)
    aug_V_targets = V_targets_tensor.repeat(8)

    B = len(all_samples)

    # Augment candidates for each symmetry
    aug_sorted_cands = []
    aug_all_cands = []

    for sym_id in range(8):
        for sample_idx in range(B):
            # Transform sorted candidates (top 5)
            sorted_cands = sorted_cands_list[sample_idx]
            aug_sorted = augment_candidates_8fold(sorted_cands, sym_id)
            aug_sorted_cands.append(aug_sorted)

            # Transform all candidates (6)
            all_cands = all_cands_list[sample_idx]
            aug_all = augment_candidates_8fold(all_cands, sym_id)
            aug_all_cands.append(aug_all)

    aug_sorted_cands_tensor = torch.tensor(aug_sorted_cands, dtype=torch.long, device=device)  # [B*8, 5]
    aug_all_cands_tensor = torch.tensor(aug_all_cands, dtype=torch.long, device=device)  # [B*8, 6]

    # Compute Q_norm for each sample using actual Q values
    from gomoku import Q_NORM_EPSILON
    Q_norms_list = []
    for Q_vals in Q_values_list:
        Q_min = min(Q_vals)
        Q_max = max(Q_vals)
        Q_range = Q_max - Q_min + Q_NORM_EPSILON
        Q_norm = [(q - Q_min) / Q_range for q in Q_vals]
        Q_norms_list.append(Q_norm)

    Q_norms_tensor = torch.tensor(Q_norms_list, dtype=torch.float32, device=device)
    aug_Q_norms = Q_norms_tensor.repeat(8, 1)  # [B*8, 5]

    # Training loop
    optimizer.zero_grad()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_ranking_inside = 0.0
    total_separation_outside = 0.0
    total_value_loss = 0.0
    total_top1_correct = 0
    total_top3_correct = 0
    total_value_mse_sum = 0.0
    total_samples = 0

    num_aug_samples = len(aug_obs)
    num_batches = (num_aug_samples + batch_size - 1) // batch_size

    for batch_start in range(0, num_aug_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_aug_samples)

        batch_obs = aug_obs[batch_start:batch_end]
        batch_masks = aug_masks[batch_start:batch_end]
        batch_sorted_cands = aug_sorted_cands_tensor[batch_start:batch_end]
        batch_all_cands = aug_all_cands_tensor[batch_start:batch_end]
        batch_Q_norms = aug_Q_norms[batch_start:batch_end]
        batch_V_targets = aug_V_targets[batch_start:batch_end]
        batch_size_actual = batch_end - batch_start

        # Forward pass
        logits_grid, values = model(batch_obs)
        logits = logits_grid.squeeze(1)  # [B, 15, 15]
        logits_flat = logits.view(batch_size_actual, 225)
        values = values.squeeze(1)  # [B]

        # Apply legal mask
        masks_flat = batch_masks.view(batch_size_actual, 225)
        logits_masked = logits_flat.masked_fill(~masks_flat, LOGIT_MASK_VALUE)

        # c4 indices (4th candidate, index 3 in sorted_candidates)
        c4_indices = batch_sorted_cands[:, 3]

        # Compute losses
        ranking_inside = compute_ranking_inside_loss(logits_masked, batch_sorted_cands, batch_Q_norms)
        separation_outside = compute_separation_outside_loss(logits_masked, batch_all_cands, c4_indices, masks_flat)
        policy_loss = ranking_inside + ALPHA_SEP * separation_outside
        value_loss = compute_search_value_loss(values, batch_V_targets)

        total_batch_loss = (policy_loss + LAMBDA_V * value_loss) / num_batches
        total_batch_loss.backward()

        # Accumulate metrics
        total_loss += total_batch_loss.item() * num_batches
        total_policy_loss += policy_loss.item()
        total_ranking_inside += ranking_inside.item()
        total_separation_outside += separation_outside.item()
        total_value_loss += value_loss.item()

        # Compute accuracy metrics
        with torch.no_grad():
            # Top-1 accuracy: c1 has highest logit
            top1_pred = logits_masked.argmax(dim=1)
            c1_indices = batch_sorted_cands[:, 0]
            top1_correct = (top1_pred == c1_indices).sum().item()
            total_top1_correct += top1_correct

            # Top-3 accuracy: c1, c2, c3 are in top-3 by logit
            _, top3_pred = torch.topk(logits_masked, 3, dim=1)
            c123 = batch_sorted_cands[:, :3]  # [B, 3]
            top3_correct = 0
            for i in range(batch_size_actual):
                pred_set = set(top3_pred[i].cpu().tolist())
                true_set = set(c123[i].cpu().tolist())
                if pred_set == true_set:
                    top3_correct += 1
            total_top3_correct += top3_correct

            # Value MSE
            total_value_mse_sum += F.mse_loss(values, batch_V_targets, reduction='sum').item()

        total_samples += batch_size_actual

    # Optimizer step
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0.0,
        'ranking_inside_loss': total_ranking_inside / num_batches if num_batches > 0 else 0.0,
        'separation_outside_loss': total_separation_outside / num_batches if num_batches > 0 else 0.0,
        'value_loss': total_value_loss / num_batches if num_batches > 0 else 0.0,
        'top1_acc': total_top1_correct / total_samples if total_samples > 0 else 0.0,
        'top3_acc': total_top3_correct / total_samples if total_samples > 0 else 0.0,
        'value_mse': total_value_mse_sum / total_samples if total_samples > 0 else 0.0
    }
