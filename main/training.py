"""
Training Module

Contains the core training logic:
- GAE (Generalized Advantage Estimation) computation
- Loss computation (policy, value, entropy)
- Training batch processing with gradient accumulation
- Gradient conflict probing
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enhancement import EPISODE_WEIGHT_ALPHA, IMITATION_MAX_WEIGHT, IMITATION_MIN_WEIGHT, IMITATION_START_UPDATE, TacticalStats, apply_tactical_enhancements, augment_batch_8fold
from gomoku import LOG_PROB_MIN, LOGIT_MASK_VALUE, TEMPERATURE_TRAIN, Trajectory, compute_returns, mask_batch_to_tensor, obs_batch_to_tensor
from torch.distributions import Categorical

# ============================================================================
# Training Constants
# ============================================================================

# --- Training Duration ---
TOTAL_UPDATES = 65536

# --- Optimizer & Learning Rate ---
LEARNING_RATE = 1.0/8192
MIN_LR = 0.125/8192
LR_DECAY_MIDPOINT_PERCENTAGE = 0.75  # Decay midpoint at 75% of training
LR_DECAY_STEEPNESS = 0.5  # Transition spread over 50% of total training
WEIGHT_DECAY = 1.0/ 2 ** 24
GRAD_CLIP_NORM = 16.0

# --- Batching & Memory ---
EPISODES_PER_UPDATE = 96  # Episodes to collect before each training update
TRAIN_BATCH_SIZE = 256 * 2  # Micro-batch size for training

# --- EMA Smoothing ---
EMA_WINDOW = 64  # Effective window for per-update EMA tracking (alpha = 1/window)
EVAL_WIN_RATE_EMA_WINDOW = 2  # Effective window for evaluation win rate EMA (evals happen less frequently)

# --- Entropy Bonus ---
ENTROPY_TARGET_START = 1.0  # Entropy bonus numerator at start (nats)
ENTROPY_TARGET_END = 0.1875  # Entropy bonus numerator at end (nats)
ENTROPY_BONUS_COEFF = 1/128.0  # Coefficient for entropy bonus
ENTROPY_DECAY_MIDPOINT_PERCENTAGE = 0.625  # Sigmoid midpoint as fraction of total training
ENTROPY_DECAY_STEEPNESS = 0.625  # Sigmoid width as fraction of total training

# --- Value Head & Advantage Estimation ---
VALUE_LOSS_COEFF_START = 1.0  # Value loss coefficient at alpha=0 (start of training)
VALUE_LOSS_COEFF_END = 0.25  # Value loss coefficient at alpha=1 (after ramp)
GAE_LAMBDA = 0.95  # GAE lambda (0=TD(0), 1=MC)
BASELINE_RAMP_END = 1024  # Cosine ramp from raw returns to GAE over [0, BASELINE_RAMP_END]
NEGATIVE_ADVANTAGE_SLOPE = 0.25  # Leaky ReLU slope for negative advantages (0=max(0,x), 1=identity)

# --- Logging ---
PRINT_INTERVAL = 1             # Print stats every N updates
PROBE_INTERVAL = 64            # Probe gradient conflict every N updates (0 = disable)


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


def compute_baseline_alpha(update: int) -> float:
    """Cosine ramp from 0 to 1 over [0, BASELINE_RAMP_END]."""
    if update >= BASELINE_RAMP_END:
        return 1.0
    progress = update / BASELINE_RAMP_END
    return 0.5 * (1 - np.cos(np.pi * progress))


# ============================================================================
# Gradient Conflict Probing
# ============================================================================

def probe_gradient_conflict_chunked(
    model: nn.Module,
    obs_chunks: List[torch.Tensor],
    next_obs_chunks: List[torch.Tensor],
    actions_chunks: List[torch.Tensor],
    masks_chunks: List[torch.Tensor],
    value_targets_chunks: List[torch.Tensor],
    is_terminal_chunks: List[torch.Tensor],
    is_synthetic_chunks: List[torch.Tensor],
    weights_chunks: List[torch.Tensor],
    gae_advantages_chunks: List[torch.Tensor],
    next_values_chunks: List[Optional[torch.Tensor]],
    global_normalizers: Tuple[float, float],
    entropy_bonus_scale: float = 1.0,
    update: int = 0,
    output_dir: str = ''
) -> None:
    """
    Probe gradient conflict using chunked gradient accumulation.

    Computes 5 gradient vectors (policy_real, policy_synthetic, value_real,
    entropy_real, entropy_synthetic) separately across chunks and saves full
    vectors to .npz for post-hoc analysis.
    """
    # Save current gradients
    saved_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            saved_grads[name] = param.grad.clone()

    # Initialize 5 gradient accumulators
    grad_keys = ['policy_real', 'policy_synthetic', 'value_real', 'entropy_real', 'entropy_synthetic']
    grad_accum = {
        key: {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        for key in grad_keys
    }

    global_policy_entropy_normalizer, global_value_normalizer = global_normalizers

    # --- Loop 1: policy_real ---
    model.zero_grad()
    for i in range(len(obs_chunks)):
        obs = obs_chunks[i]
        actions = actions_chunks[i]
        masks = masks_chunks[i]
        is_synthetic = is_synthetic_chunks[i]
        weights = weights_chunks[i]
        gae_advantages = gae_advantages_chunks[i]

        real_mask = ~is_synthetic
        if not real_mask.any():
            continue

        batch_size = obs.size(0)
        logits_grid, values = model(obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~masks, LOGIT_MASK_VALUE)
        logits_flat = logits.view(batch_size, 225)

        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled, validate_args=False)
        batch_log_probs = dist.log_prob(actions)
        batch_log_probs = torch.clamp(batch_log_probs, min=LOG_PROB_MIN)

        policy_loss = -(weights * real_mask.float() * gae_advantages * batch_log_probs).sum() / max(global_policy_entropy_normalizer, 1.0)
        policy_loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum['policy_real'][name] += param.grad.detach()
        model.zero_grad()

    # --- Loop 2: policy_synthetic ---
    for i in range(len(obs_chunks)):
        obs = obs_chunks[i]
        actions = actions_chunks[i]
        masks = masks_chunks[i]
        is_synthetic = is_synthetic_chunks[i]
        weights = weights_chunks[i]
        gae_advantages = gae_advantages_chunks[i]

        synth_mask = is_synthetic
        if not synth_mask.any():
            continue

        batch_size = obs.size(0)
        logits_grid, values = model(obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~masks, LOGIT_MASK_VALUE)
        logits_flat = logits.view(batch_size, 225)

        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled, validate_args=False)
        batch_log_probs = dist.log_prob(actions)
        batch_log_probs = torch.clamp(batch_log_probs, min=LOG_PROB_MIN)

        policy_loss = -(weights * synth_mask.float() * gae_advantages * batch_log_probs).sum() / max(global_policy_entropy_normalizer, 1.0)
        policy_loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum['policy_synthetic'][name] += param.grad.detach()
        model.zero_grad()

    # --- Loop 3: value_real (already masks synthetic internally) ---
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

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum['value_real'][name] += param.grad.detach()
        model.zero_grad()

    # --- Loop 4: entropy_real ---
    for i in range(len(obs_chunks)):
        obs = obs_chunks[i]
        masks = masks_chunks[i]
        is_synthetic = is_synthetic_chunks[i]
        weights = weights_chunks[i]

        real_mask = ~is_synthetic
        if not real_mask.any():
            continue

        batch_size = obs.size(0)
        logits_grid, values = model(obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~masks, LOGIT_MASK_VALUE)
        logits_flat = logits.view(batch_size, 225)

        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled, validate_args=False)
        entropies = dist.entropy()

        entropy_loss = entropy_bonus_scale * (-(weights * real_mask.float() * entropies).sum() / max(global_policy_entropy_normalizer, 1.0))
        entropy_loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum['entropy_real'][name] += param.grad.detach()
        model.zero_grad()

    # --- Loop 5: entropy_synthetic ---
    for i in range(len(obs_chunks)):
        obs = obs_chunks[i]
        masks = masks_chunks[i]
        is_synthetic = is_synthetic_chunks[i]
        weights = weights_chunks[i]

        synth_mask = is_synthetic
        if not synth_mask.any():
            continue

        batch_size = obs.size(0)
        logits_grid, values = model(obs)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~masks, LOGIT_MASK_VALUE)
        logits_flat = logits.view(batch_size, 225)

        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled, validate_args=False)
        entropies = dist.entropy()

        entropy_loss = entropy_bonus_scale * (-(weights * synth_mask.float() * entropies).sum() / max(global_policy_entropy_normalizer, 1.0))
        entropy_loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_accum['entropy_synthetic'][name] += param.grad.detach()
        model.zero_grad()

    # --- Flatten vectors & save .npz ---
    param_names_ordered = [name for name, _ in model.named_parameters()]
    flat_vectors = {}
    param_offsets = []
    offset = 0
    for name in param_names_ordered:
        numel = grad_accum['policy_real'][name].numel()
        param_offsets.append(offset)
        offset += numel

    for key in grad_keys:
        parts = [grad_accum[key][name].flatten().cpu().float().numpy() for name in param_names_ordered]
        flat_vectors[key] = np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    if output_dir:
        npz_path = os.path.join(output_dir, f'gradient_probe_{update+1:06d}.npz')
        np.savez_compressed(
            npz_path,
            policy_real=flat_vectors['policy_real'],
            policy_synthetic=flat_vectors['policy_synthetic'],
            value_real=flat_vectors['value_real'],
            entropy_real=flat_vectors['entropy_real'],
            entropy_synthetic=flat_vectors['entropy_synthetic'],
            param_names=np.array(param_names_ordered, dtype=object),
            param_offsets=np.array(param_offsets, dtype=np.int64),
            entropy_bonus_scale=np.float32(entropy_bonus_scale),
            update=np.int32(update + 1)
        )

    # Restore original gradients
    model.zero_grad()
    for name, param in model.named_parameters():
        if name in saved_grads:
            param.grad = saved_grads[name]


# ============================================================================
# Training Functions
# ============================================================================

def _train_on_batch_internal(model: nn.Module, trajectories: List[Trajectory],
                             device: torch.device,
                             update: int,
                             win_boost: float,
                             block_boost: float,
                             opr_samples: List[dict],
                             ema_entropy: float,
                             win_rate: float) -> Tuple[float, float, float, float, float, TacticalStats, int, int, int, Optional[dict]]:
    """
    Internal training function - processes a batch of trajectories.

    Returns:
        Tuple of (loss, mean_return, entropy, value_loss, raw_value_mse,
                  tactical_stats, num_imitation_black, num_imitation_white,
                  num_opr_samples, probe_data)
        probe_data is a dict with chunked tensors for gradient probing, or None
    """
    alpha = compute_baseline_alpha(update)

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

    num_imitation_black = 0
    num_imitation_white = 0

    for traj_idx, traj in enumerate(trajectories):
        returns = compute_returns(traj)

        current_steps = sum(1 for is_current in traj.is_current_policy if is_current)

        if current_steps == 0:
            continue

        imitation_enabled = IMITATION_MAX_WEIGHT > 0 and update >= IMITATION_START_UPDATE

        current_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0
        # Dynamic imitation weight based on win rate: (1 - win_rate) * (max - min) + min
        base_imitation_weight = (1.0 - win_rate) * (IMITATION_MAX_WEIGHT - IMITATION_MIN_WEIGHT) + IMITATION_MIN_WEIGHT
        imitation_weight = base_imitation_weight * current_weight if imitation_enabled else 0

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

    # Track off-policy rollout sample advantages
    opr_advantages = []

    # Add off-policy rollout samples
    num_opr_samples = 0
    if opr_samples:
        for sample in opr_samples:
            all_obs.append(sample['obs'])
            all_actions.append(sample['action'])
            all_masks.append(sample['mask'])
            all_returns.append(sample['strength'])
            all_value_targets.append(0.0)
            all_is_synthetic.append(True)
            opr_advantages.append(sample['strength'])  # Track for later
            all_weights.append(sample['weight'])
            all_next_obs.append(np.zeros_like(sample['obs']))
            all_is_terminal.append(True)
            sample_to_traj.append((-1, -1))  # Mark as synthetic
            num_opr_samples += 1

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

    aug_value_targets = value_targets_tensor.repeat(8)
    aug_is_synthetic = is_synthetic_tensor.repeat(8)
    aug_is_terminal = is_terminal_tensor.repeat(8)
    aug_weights = weights_tensor.repeat(8)

    # Compute advantages after augmentation (deferred collection - no placeholder pattern)
    B = len(all_obs)  # Original batch size before augmentation

    # Always compute 8-fold averaged values and GAE, then blend with raw returns via alpha
    # Process in chunks to avoid OOM on the full B*8 tensor
    aug_values_list = []
    with torch.no_grad():
        for chunk_start in range(0, len(aug_obs), TRAIN_BATCH_SIZE):
            chunk = aug_obs[chunk_start:chunk_start + TRAIN_BATCH_SIZE]
            aug_values_list.append(model.forward_value_only(chunk).squeeze(1))
    aug_values_all = torch.cat(aug_values_list)  # [B*8]

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

    for traj_idx in range(len(trajectories)):
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
            gae_advantages[sample_idx] = gae

    # Blend advantages with leaky ReLU: attenuate negative advantages by NEGATIVE_ADVANTAGE_SLOPE
    raw_returns = returns_tensor.cpu().numpy()
    final_advantages = np.zeros(B, dtype=np.float32)
    s = NEGATIVE_ADVANTAGE_SLOPE * alpha

    for sample_idx in range(num_samples_before_tactical):
        blended = (1 - alpha) * raw_returns[sample_idx] + alpha * gae_advantages[sample_idx]
        final_advantages[sample_idx] = blended * (s + (1 - s) * (blended > 0)) + tactical_boost_info.sample_boosts[sample_idx]

    # Tactical synthetic samples: use their advantage values directly
    for i, advantage_value in enumerate(tactical_boost_info.synthetic_advantages):
        final_advantages[num_samples_before_tactical + i] = advantage_value

    # Off-policy rollout synthetic samples: use their strength values
    opr_start_idx = num_samples_before_tactical + len(tactical_boost_info.synthetic_advantages)
    for i, advantage_value in enumerate(opr_advantages):
        final_advantages[opr_start_idx + i] = advantage_value

    advantages_tensor = torch.tensor(final_advantages, dtype=torch.float32, device=device)
    aug_gae_advantages = advantages_tensor.repeat(8)

    # Also average next_values for consistent TD targets (chunked to match VRAM)
    aug_next_values_list = []
    with torch.no_grad():
        for chunk_start in range(0, len(aug_next_obs), TRAIN_BATCH_SIZE):
            chunk = aug_next_obs[chunk_start:chunk_start + TRAIN_BATCH_SIZE]
            aug_next_values_list.append(model.forward_value_only(chunk).squeeze(1))
    aug_next_values_all = torch.cat(aug_next_values_list)  # [B*8]
    next_values_per_aug = aug_next_values_all.view(8, B)
    avg_next_values = next_values_per_aug.mean(dim=0)  # [B]
    aug_next_values_averaged = avg_next_values.repeat(8)

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
    entropy_bonus_scale = 0.0
    accumulated_weight_sum = 0.0

    # Collect data for chunked gradient probing (if needed)
    should_probe = PROBE_INTERVAL > 0 and (update + 1) % PROBE_INTERVAL == 0
    probe_obs_chunks = [] if should_probe else None
    probe_next_obs_chunks = [] if should_probe else None
    probe_actions_chunks = [] if should_probe else None
    probe_masks_chunks = [] if should_probe else None
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

        # Use 8-fold averaged next_values for consistent TD targets
        next_values = aug_next_values_averaged[batch_start:batch_end]
        if should_probe:
            probe_next_values_chunks.append(next_values.detach())

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
        entropy_bonus_scale = entropy_schedule / max(ema_entropy, 1e-8)
        entropy_loss_mb = -(batch_weights * entropies).sum() / max(global_policy_entropy_normalizer, 1.0)

        value_loss_coeff = VALUE_LOSS_COEFF_START + (VALUE_LOSS_COEFF_END - VALUE_LOSS_COEFF_START) * alpha
        loss_mb = policy_loss_mb + value_loss_coeff * value_loss_mb + ENTROPY_BONUS_COEFF * entropy_bonus_scale * entropy_loss_mb
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
            'value_targets_chunks': probe_value_targets_chunks,
            'is_terminal_chunks': probe_is_terminal_chunks,
            'is_synthetic_chunks': probe_is_synthetic_chunks,
            'weights_chunks': probe_weights_chunks,
            'gae_advantages_chunks': probe_gae_advantages_chunks,
            'next_values_chunks': probe_next_values_chunks,
            'global_normalizers': (global_policy_entropy_normalizer, global_value_normalizer),
            'entropy_bonus_scale': ENTROPY_BONUS_COEFF * entropy_bonus_scale
        }

    return (total_loss_scalar, mean_return, total_entropy_scalar, total_value_loss_scalar,
            raw_value_mse, tactical_stats, num_imitation_black, num_imitation_white,
            num_opr_samples, probe_data)


def train_on_batch(model: nn.Module, trajectories: List[Trajectory],
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   update: int,
                   win_boost: float,
                   block_boost: float,
                   opr_samples: List[dict],
                   ema_entropy: float,
                   win_rate: float,
                   output_dir: str = '') -> dict:
    """
    Train on a batch of trajectories.

    Returns:
        Dictionary with training statistics
    """
    optimizer.zero_grad()

    (loss, mean_return, entropy, value_loss, raw_value_mse,
     tactical_stats, num_imitation_black, num_imitation_white,
     num_opr, probe_data) = _train_on_batch_internal(
        model, trajectories, device,
        update=update,
        win_boost=win_boost,
        block_boost=block_boost,
        opr_samples=opr_samples,
        ema_entropy=ema_entropy,
        win_rate=win_rate
    )

    # Run gradient probe (before optimizer step)
    if probe_data is not None:
        probe_gradient_conflict_chunked(
            model,
            probe_data['obs_chunks'],
            probe_data['next_obs_chunks'],
            probe_data['actions_chunks'],
            probe_data['masks_chunks'],
            probe_data['value_targets_chunks'],
            probe_data['is_terminal_chunks'],
            probe_data['is_synthetic_chunks'],
            probe_data['weights_chunks'],
            probe_data['gae_advantages_chunks'],
            probe_data['next_values_chunks'],
            probe_data['global_normalizers'],
            entropy_bonus_scale=probe_data['entropy_bonus_scale'],
            update=update,
            output_dir=output_dir
        )

    # Optimizer step
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return {
        'loss': loss,
        'mean_return': mean_return,
        'entropy': entropy,
        'value_loss': value_loss,
        'raw_value_mse': raw_value_mse,
        'tactical_stats': tactical_stats,
        'imitation_black': num_imitation_black,
        'imitation_white': num_imitation_white,
        'opr_samples': num_opr,
        'probe_ran': probe_data is not None
    }
