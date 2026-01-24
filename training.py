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
from dataclasses import dataclass

from model import DEVICE, N_BLOCKS
from gomoku import (
    Trajectory, GameState, Player,
    obs_batch_to_tensor, mask_batch_to_tensor,
    compute_returns, TEMPERATURE_TRAIN, LOG_PROB_MIN, LOGIT_MASK_VALUE
)
from enhancement import (
    apply_tactical_enhancements, augment_batch_8fold,
    TacticalStats, TacticalBoostInfo, EPISODE_WEIGHT_ALPHA,
    IMITATION_MAX_WEIGHT, IMITATION_MIN_WEIGHT, IMITATION_START_UPDATE
)


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
TRAIN_BATCH_SIZE = 256 * 3     # Micro-batch size for training

# --- EMA Smoothing ---
EMA_WINDOW = 128               # Effective window for per-update EMA tracking (alpha = 1/window)
EVAL_WIN_RATE_EMA_WINDOW = 3   # Effective window for evaluation win rate EMA (evals happen less frequently)

# --- Entropy Bonus ---
ENTROPY_TARGET_START = 1.0     # Entropy bonus numerator at start (nats)
ENTROPY_TARGET_END = 0.25      # Entropy bonus numerator at end (nats)
ENTROPY_BONUS_COEFF = 1/64.0   # Coefficient for entropy bonus
ENTROPY_DECAY_MIDPOINT_PERCENTAGE = 0.625  # Transition occurs at 75% of training
ENTROPY_DECAY_STEEPNESS = 0.5  # Transition spread over 50% of total training duration

# --- Value Head & Advantage Estimation ---
VALUE_LOSS_COEFF = 1.0         # Weight for value head loss
GAE_LAMBDA = 0.95              # GAE lambda (0=TD(0), 1=MC)
VALUE_BASELINE_START = 128 * 6 # Update at which to start using value baseline

# --- Logging ---
PRINT_INTERVAL = 8             # Print stats every N updates
PROBE_INTERVAL = 64            # Probe gradient conflict every N updates (0 = disable)


# ============================================================================
# GAE Computation
# ============================================================================

def compute_gae_for_trajectories(model: nn.Module, trajectories: List[Trajectory],
                                  device: torch.device, gae_lambda: float) -> List[np.ndarray]:
    """
    Compute GAE advantages for all trajectories.

    For two-player games with canonical representation:
        delta_n = -V(S_{n+1}) - V(S_n) for non-terminal
        delta_n = z - V(S_n) for terminal
        A_n = delta_n - lambda * A_{n+1}

    Args:
        model: Neural network model with value head
        trajectories: List of trajectory objects
        device: torch device
        gae_lambda: GAE lambda parameter
    """
    all_gae = []

    for traj in trajectories:
        T = len(traj.observations)
        if T == 0:
            all_gae.append(np.array([]))
            continue

        obs_tensor = obs_batch_to_tensor(traj.observations, device)
        with torch.no_grad():
            _, values = model(obs_tensor)
        values = values.squeeze(1).cpu().numpy()

        returns = compute_returns(traj)

        gae_advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                delta = returns[t] - values[t]
            else:
                delta = -values[t + 1] - values[t]

            gae = delta - gae_lambda * gae
            gae_advantages[t] = gae

        all_gae.append(gae_advantages)

    return all_gae


# ============================================================================
# Gradient Conflict Probing
# ============================================================================

def probe_gradient_conflict(model: nn.Module, policy_loss: torch.Tensor,
                             value_loss: torch.Tensor, update: int) -> Dict[str, float]:
    """
    Probe gradient conflict between policy and value losses.

    Computes cosine similarity between policy gradients and value gradients
    for different parts of the network.
    """
    # Temporarily save current gradients (if any) and zero them
    saved_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            saved_grads[name] = param.grad.clone()
            param.grad = None

    # Compute policy gradients
    policy_loss.backward(retain_graph=True)
    policy_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            policy_grads[name] = param.grad.clone()

    # Zero gradients and compute value gradients
    model.zero_grad()
    value_loss.backward(retain_graph=True)
    value_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            value_grads[name] = param.grad.clone()

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
    trunk_params = {f'blocks_{i}-{i+3}': [] for i in range(0, N_BLOCKS, 4)}
    all_trunk_stem_params = []

    for name in policy_grads.keys():
        if any(x in name for x in ['conv_3x3', 'conv_sparse5', 'conv_dense_5x5',
                                     'conv_sparse7', 'conv_dense_7x7', 'conv_1x1',
                                     'stem_norm']):
            stem_params.append(name)
            all_trunk_stem_params.append(name)
        elif 'blocks.' in name:
            block_idx = int(name.split('blocks.')[1].split('.')[0])
            layer_start = (block_idx // 4) * 4
            layer_key = f'blocks_{layer_start}-{layer_start+3}'
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

    for layer_key in ['blocks_0-3', 'blocks_4-7', 'blocks_8-11', 'blocks_12-15']:
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
                             num_accumulation_steps: int = 1,
                             update: int = 0,
                             win_boost: float = 0.0,
                             block_boost: float = 0.0,
                             cler_samples: List[dict] = None,
                             ema_entropy: float = None,
                             win_rate: float = 0.5) -> Tuple[float, float, float, float, float, TacticalStats, int, int, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Internal training function - processes a batch of trajectories.

    Returns:
        Tuple of (loss, mean_return, entropy, value_loss, raw_value_mse,
                  tactical_stats, num_imitation_black, num_imitation_white,
                  num_cler_samples, probe_policy_loss, probe_value_loss)
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

    num_trajectories = 0
    num_imitation_black = 0
    num_imitation_white = 0

    for traj_idx, traj in enumerate(trajectories):
        returns = compute_returns(traj)

        current_steps = sum(1 for is_current in traj.is_current_policy if is_current)

        if current_steps == 0:
            continue

        num_trajectories += 1

        imitation_enabled = IMITATION_MAX_WEIGHT > 0 and update >= IMITATION_START_UPDATE

        current_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0
        # Dynamic imitation weight based on win rate: (1 - win_rate) * (max - min) + min
        base_imitation_weight = (1.0 - win_rate) * (IMITATION_MAX_WEIGHT - IMITATION_MIN_WEIGHT) + IMITATION_MIN_WEIGHT
        imitation_weight = base_imitation_weight * current_weight if imitation_enabled else 0

        for step_idx, (obs, action, legal_mask, log_prob, z_t, is_current) in enumerate(zip(
            traj.observations, traj.actions, traj.legal_masks, traj.log_probs, returns, traj.is_current_policy
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

            elif imitation_enabled and z_t > 0:
                all_obs.append(obs)
                all_actions.append(action)
                all_masks.append(legal_mask)
                all_returns.append(z_t)
                all_value_targets.append(z_t)
                all_is_synthetic.append(False)
                all_weights.append(imitation_weight)
                sample_to_traj.append((traj_idx, step_idx))
                pieces_self = np.sum(obs[0])
                pieces_opponent = np.sum(obs[1])
                if pieces_self == pieces_opponent:
                    num_imitation_black += 1
                else:
                    num_imitation_white += 1

                if step_idx + 1 < len(traj.observations):
                    all_next_obs.append(traj.observations[step_idx + 1])
                    all_is_terminal.append(False)
                else:
                    all_next_obs.append(np.zeros_like(obs))
                    all_is_terminal.append(True)

    # Track original length before tactical enhancements (which adds synthetic samples)
    num_samples_before_tactical = len(all_obs)

    # Apply tactical enhancements (returns boost info for applying after GAE)
    tactical_stats, tactical_boost_info = apply_tactical_enhancements(
        all_obs, all_actions, all_masks, all_returns,
        all_weights, all_value_targets, all_is_synthetic, all_next_obs, all_is_terminal,
        win_boost, block_boost
    )

    # Mark tactical synthetic samples in sample_to_traj
    num_tactical_synthetic = len(all_obs) - num_samples_before_tactical
    for _ in range(num_tactical_synthetic):
        sample_to_traj.append((-1, -1))  # Mark as synthetic

    # Track CLER sample advantages
    cler_advantages = []

    # Add CLER samples
    num_cler_samples = 0
    if cler_samples:
        for sample in cler_samples:
            all_obs.append(sample['obs'])
            all_actions.append(sample['action'])
            all_masks.append(sample['mask'])
            all_returns.append(sample['strength'])
            all_value_targets.append(0.0)
            all_is_synthetic.append(True)
            cler_advantages.append(sample['strength'])  # Track for later
            all_weights.append(sample['weight'])
            all_next_obs.append(np.zeros_like(sample['obs']))
            all_is_terminal.append(True)
            sample_to_traj.append((-1, -1))  # Mark as synthetic
            num_cler_samples += 1

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
            _, aug_values_all = model(aug_obs)  # [B*8, 1]

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
        base_advantages = np.zeros(B, dtype=np.float32)

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
                base_advantages[sample_idx] = gae

        # Apply tactical boosts to base advantages
        final_advantages = np.zeros(B, dtype=np.float32)

        for sample_idx in range(num_samples_before_tactical):
            # Original trajectory samples: base GAE + tactical boost
            final_advantages[sample_idx] = max(0.0, base_advantages[sample_idx]) + tactical_boost_info.sample_boosts[sample_idx]

        # Tactical synthetic samples: use their advantage values directly
        for i, advantage_value in enumerate(tactical_boost_info.synthetic_advantages):
            final_advantages[num_samples_before_tactical + i] = advantage_value

        # CLER synthetic samples: use their strength values
        cler_start_idx = num_samples_before_tactical + len(tactical_boost_info.synthetic_advantages)
        for i, advantage_value in enumerate(cler_advantages):
            final_advantages[cler_start_idx + i] = advantage_value

        advantages_tensor = torch.tensor(final_advantages, dtype=torch.float32, device=device)
        aug_gae_advantages = advantages_tensor.repeat(8)

        # Also average next_values for consistent TD targets
        with torch.no_grad():
            aug_next_values_all = model(aug_next_obs)[1].squeeze(1)  # [B*8]
        next_values_per_aug = aug_next_values_all.view(8, B)
        avg_next_values = next_values_per_aug.mean(dim=0)  # [B]
        aug_next_values_averaged = avg_next_values.repeat(8)
    else:
        # No GAE: use raw returns as base advantages
        base_advantages = returns_tensor.cpu().numpy()

        # Apply tactical boosts
        final_advantages = np.zeros(B, dtype=np.float32)

        for sample_idx in range(num_samples_before_tactical):
            # Original trajectory samples: return + tactical boost
            final_advantages[sample_idx] = max(0.0, base_advantages[sample_idx]) + tactical_boost_info.sample_boosts[sample_idx]

        # Tactical synthetic samples
        for i, advantage_value in enumerate(tactical_boost_info.synthetic_advantages):
            final_advantages[num_samples_before_tactical + i] = advantage_value

        # CLER synthetic samples
        cler_start_idx = num_samples_before_tactical + len(tactical_boost_info.synthetic_advantages)
        for i, advantage_value in enumerate(cler_advantages):
            final_advantages[cler_start_idx + i] = advantage_value

        advantages_tensor = torch.tensor(final_advantages, dtype=torch.float32, device=device)
        aug_gae_advantages = advantages_tensor.repeat(8)
        aug_next_values_averaged = None

    global_policy_entropy_normalizer = aug_weights.sum().item()
    value_loss_mask_global = ~aug_is_synthetic
    global_value_normalizer = (aug_weights * value_loss_mask_global.float()).sum().item()

    # Target entropy with sigmoid decay
    midpoint = TOTAL_UPDATES * ENTROPY_DECAY_MIDPOINT_PERCENTAGE
    steepness_k = 3.0 / (TOTAL_UPDATES * ENTROPY_DECAY_STEEPNESS)
    sigmoid_factor = (1.0 - torch.tanh(torch.tensor(steepness_k * (update - midpoint)))) / 2.0
    target_entropy = ENTROPY_TARGET_END + (ENTROPY_TARGET_START - ENTROPY_TARGET_END) * sigmoid_factor.item()

    accumulated_loss = 0.0
    accumulated_value_loss = 0.0
    accumulated_value_mse_sum = 0.0
    accumulated_value_mse_count = 0
    accumulated_weighted_policy_loss_sum = 0.0
    accumulated_weighted_entropy_sum = 0.0
    accumulated_weight_sum = 0.0

    # Accumulate losses for full-update probing
    should_probe = PROBE_INTERVAL > 0 and (update + 1) % PROBE_INTERVAL == 0
    probe_policy_loss_accum = None
    probe_value_loss_accum = None

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

        logits_grid, values = model(batch_obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~batch_masks, LOGIT_MASK_VALUE)
        logits_flat = logits.view(batch_size, 225)

        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled)
        entropies = dist.entropy()
        batch_log_probs = dist.log_prob(batch_actions)
        batch_log_probs = torch.clamp(batch_log_probs, min=LOG_PROB_MIN)

        values = values.squeeze(1)

        # Use 8-fold averaged next_values for consistent TD targets if available
        if aug_next_values_averaged is not None:
            next_values = aug_next_values_averaged[batch_start:batch_end]
        else:
            with torch.no_grad():
                _, next_values = model(batch_next_obs)
            next_values = next_values.squeeze(1)

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

        # Adaptive entropy bonus (ratio-based: target / current)
        entropy_bonus_scale = target_entropy / max(ema_entropy, 1e-8) if ema_entropy is not None else 1.0
        entropy_loss_mb = -(batch_weights * entropies).sum() / max(global_policy_entropy_normalizer, 1.0)

        # Accumulate losses for full-update gradient probing
        if should_probe:
            if probe_policy_loss_accum is None:
                probe_policy_loss_accum = policy_loss_mb
                probe_value_loss_accum = value_loss_mb
            else:
                probe_policy_loss_accum = probe_policy_loss_accum + policy_loss_mb
                probe_value_loss_accum = probe_value_loss_accum + value_loss_mb

        loss_mb = (policy_loss_mb + VALUE_LOSS_COEFF * value_loss_mb + ENTROPY_BONUS_COEFF * entropy_bonus_scale * entropy_loss_mb) / num_accumulation_steps
        loss_mb.backward()

        accumulated_loss += loss_mb.item()
        accumulated_value_loss += value_loss_mb.item()
        accumulated_weighted_policy_loss_sum += (batch_weights * advantages * batch_log_probs).sum().item()
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

    return (total_loss_scalar, mean_return, total_entropy_scalar, total_value_loss_scalar,
            raw_value_mse, tactical_stats, num_imitation_black, num_imitation_white,
            num_cler_samples, probe_policy_loss_accum, probe_value_loss_accum)


def train_on_batch(model: nn.Module, trajectories: List[Trajectory],
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   chunk_size: int = EPISODES_CHUNK_SIZE,
                   update: int = 0,
                   win_boost: float = 0.0,
                   block_boost: float = 0.0,
                   cler_samples: List[dict] = None,
                   ema_entropy: float = None,
                   win_rate: float = 0.5) -> dict:
    """
    Train on a batch of trajectories with gradient accumulation.

    Returns:
        Dictionary with training statistics
    """
    if len(trajectories) == 0:
        return {
            'loss': 0.0, 'mean_return': 0.0, 'entropy': 0.0, 'value_loss': 0.0,
            'raw_value_mse': 0.0, 'tactical_stats': TacticalStats(),
            'imitation_black': 0, 'imitation_white': 0, 'cler_samples': 0,
            'probe_metrics': None
        }

    num_chunks = (len(trajectories) + chunk_size - 1) // chunk_size
    chunks = [trajectories[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    optimizer.zero_grad()

    total_loss = 0.0
    total_returns = []
    total_entropy = 0.0
    total_value_loss = 0.0
    total_raw_value_mse = 0.0
    total_tactical_stats = TacticalStats()
    total_imitation_black = 0
    total_imitation_white = 0
    total_cler_samples = 0
    num_chunks_processed = 0
    collected_probe_metrics = None

    # Accumulate losses for full-update gradient probing
    should_probe = PROBE_INTERVAL > 0 and (update + 1) % PROBE_INTERVAL == 0
    total_probe_policy_loss = None
    total_probe_value_loss = None

    for i, chunk in enumerate(chunks):

        # Pass CLER samples only to first chunk to avoid duplicating them
        chunk_cler_samples = cler_samples if i == 0 else None

        (loss, mean_return, mean_entropy, value_loss, raw_value_mse,
         tactical_stats, num_imitation_black, num_imitation_white,
         num_cler, probe_policy_loss, probe_value_loss) = _train_on_batch_internal(
            model, chunk, optimizer, device,
            num_accumulation_steps=num_chunks,
            update=update,
            win_boost=win_boost,
            block_boost=block_boost,
            cler_samples=chunk_cler_samples,
            ema_entropy=ema_entropy,
            win_rate=win_rate
        )

        # Accumulate probe losses across chunks for full-update probing
        if should_probe and probe_policy_loss is not None:
            if total_probe_policy_loss is None:
                total_probe_policy_loss = probe_policy_loss
                total_probe_value_loss = probe_value_loss
            else:
                total_probe_policy_loss = total_probe_policy_loss + probe_policy_loss
                total_probe_value_loss = total_probe_value_loss + probe_value_loss

        total_loss += loss * num_chunks
        total_entropy += mean_entropy
        total_value_loss += value_loss
        total_raw_value_mse += raw_value_mse

        # Aggregate tactical stats
        total_tactical_stats.wins_found += tactical_stats.wins_found
        total_tactical_stats.blocks_found += tactical_stats.blocks_found
        total_tactical_stats.synthetic_wins_eq += tactical_stats.synthetic_wins_eq
        total_tactical_stats.synthetic_wins_missed += tactical_stats.synthetic_wins_missed
        total_tactical_stats.synthetic_blocks += tactical_stats.synthetic_blocks
        total_tactical_stats.win_opportunities += tactical_stats.win_opportunities
        total_tactical_stats.win_misses += tactical_stats.win_misses
        total_tactical_stats.block_opportunities += tactical_stats.block_opportunities
        total_tactical_stats.block_misses += tactical_stats.block_misses

        total_imitation_black += num_imitation_black
        total_imitation_white += num_imitation_white
        total_cler_samples += num_cler
        num_chunks_processed += 1

        for traj in chunk:
            returns = compute_returns(traj)
            for z_t, is_current in zip(returns, traj.is_current_policy):
                if is_current:
                    total_returns.append(z_t)

    # Run gradient probe on full update (after all chunks processed, before optimizer step)
    if should_probe and total_probe_policy_loss is not None:
        collected_probe_metrics = probe_gradient_conflict(
            model, total_probe_policy_loss, total_probe_value_loss, update
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
        'imitation_black': total_imitation_black,
        'imitation_white': total_imitation_white,
        'cler_samples': total_cler_samples,
        'probe_metrics': collected_probe_metrics
    }
