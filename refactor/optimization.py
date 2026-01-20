"""
Optimization and Training Loop.

Handles the gradient descent process:
- Batch preparation (flattening trajectories, tactic augmentation)
- Gradient calculation and updates
- Gradient conflict probing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional, Dict
import copy

from model import N_BLOCKS, GomokuPolicyNet, zero_center_taps
from gomoku_rules import find_all_win_in_1, find_blocking_moves
from gameplay_loop import Trajectory, obs_batch_to_tensor, mask_batch_to_tensor
from data_augmentation import (
    compute_returns, compute_gae_for_trajectories,
    GAE_LAMBDA, IMITATION_WEIGHT, IMITATION_START_UPDATE, EPISODE_WEIGHT_ALPHA,
    SYNTHETIC_WIN_BOOST, SYNTHETIC_BLOCKING_BOOST, MAX_SYNTHETIC_WINS, MAX_SYNTHETIC_BLOCKS
)

# ============================================================================
# Configuration
# ============================================================================

TOTAL_UPDATES = 65536

# Optimizer
LEARNING_RATE = 5e-4
MIN_LR = 1e-4
LR_DECAY = (MIN_LR / LEARNING_RATE) ** (1.0 / TOTAL_UPDATES)
WEIGHT_DECAY = 1e-8
GRAD_CLIP_NORM = 16.0

# Batching
EPISODES_PER_UPDATE = 64
EPISODES_CHUNK_SIZE = 32
TRAIN_BATCH_SIZE = 1024

# Entropy & Loss
TEMPERATURE_TRAIN = 1.25
ENTROPY_TARGET_START = 1.25
ENTROPY_TARGET_END = 0.5
ENTROPY_BONUS_COEFF = 1/128.0
ENTROPY_DECAY_MIDPOINT_PERCENTAGE = 0.75
ENTROPY_DECAY_STEEPNESS = 0.5
ENTROPY_EMA_LAMBDA = 1/16.0
VALUE_LOSS_COEFF = 0.5
VALUE_BASELINE_START = 512

# Tactics
MISS_RATE_EMA_WINDOW = 128
WIN_MIN_BOOST = 0.0
WIN_MAX_BOOST = 1.0
BLOCK_MIN_BOOST = 0.0
BLOCK_MAX_BOOST = 0.75

# Probing
PROBE_INTERVAL = 64


# ============================================================================
# GPU Data Augmentation
# ============================================================================

def augment_batch_gpu(obs_batch: torch.Tensor, actions: torch.Tensor,
                      masks_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply 8-fold symmetry to batch on GPU."""
    B = obs_batch.size(0)
    device = obs_batch.device
    action_rows = actions // 15
    action_cols = actions % 15

    all_obs = torch.empty(B * 8, 3, 15, 15, dtype=obs_batch.dtype, device=device)
    all_actions = torch.empty(B * 8, dtype=torch.long, device=device)
    all_masks = torch.empty(B * 8, 15, 15, dtype=torch.bool, device=device)

    for sym_id in range(8):
        start_idx = sym_id * B
        end_idx = (sym_id + 1) * B

        if sym_id == 0:  # Identity
            all_obs[start_idx:end_idx] = obs_batch
            all_masks[start_idx:end_idx] = masks_batch
            new_rows, new_cols = action_rows, action_cols
        elif sym_id == 1:  # Rotate 90 CW
            all_obs[start_idx:end_idx] = obs_batch.transpose(-2, -1).flip(-1)
            all_masks[start_idx:end_idx] = masks_batch.transpose(-2, -1).flip(-1)
            new_rows, new_cols = action_cols, 14 - action_rows
        elif sym_id == 2:  # Rotate 180
            all_obs[start_idx:end_idx] = obs_batch.flip(-2).flip(-1)
            all_masks[start_idx:end_idx] = masks_batch.flip(-2).flip(-1)
            new_rows, new_cols = 14 - action_rows, 14 - action_cols
        elif sym_id == 3:  # Rotate 270 CW
            all_obs[start_idx:end_idx] = obs_batch.transpose(-2, -1).flip(-2)
            all_masks[start_idx:end_idx] = masks_batch.transpose(-2, -1).flip(-2)
            new_rows, new_cols = 14 - action_cols, action_rows
        elif sym_id == 4:  # Flip horizontal
            all_obs[start_idx:end_idx] = obs_batch.flip(-1)
            all_masks[start_idx:end_idx] = masks_batch.flip(-1)
            new_rows, new_cols = action_rows, 14 - action_cols
        elif sym_id == 5:  # Flip vertical
            all_obs[start_idx:end_idx] = obs_batch.flip(-2)
            all_masks[start_idx:end_idx] = masks_batch.flip(-2)
            new_rows, new_cols = 14 - action_rows, action_cols
        elif sym_id == 6:  # Transpose
            all_obs[start_idx:end_idx] = obs_batch.transpose(-2, -1)
            all_masks[start_idx:end_idx] = masks_batch.transpose(-2, -1)
            new_rows, new_cols = action_cols, action_rows
        elif sym_id == 7:  # Anti-transpose
            all_obs[start_idx:end_idx] = obs_batch.flip(-2).flip(-1).transpose(-2, -1)
            all_masks[start_idx:end_idx] = masks_batch.flip(-2).flip(-1).transpose(-2, -1)
            new_rows, new_cols = 14 - action_cols, 14 - action_rows

        all_actions[start_idx:end_idx] = new_rows * 15 + new_cols

    return all_obs, all_actions, all_masks


# ============================================================================
# Gradient Probing
# ============================================================================

def probe_gradient_conflict(model: nn.Module, policy_loss: torch.Tensor,
                             value_loss: torch.Tensor) -> Dict[str, float]:
    """Probe gradient conflict between policy and value losses."""
    saved_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            saved_grads[name] = param.grad.clone()
            param.grad = None

    policy_loss.backward(retain_graph=True)
    policy_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            policy_grads[name] = param.grad.clone()

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
        # Stem parameters
        if any(x in name for x in ['conv_3x3', 'conv_sparse5', 'conv_dense_5x5',
                                     'conv_sparse7', 'conv_dense_7x7', 'conv_1x1',
                                     'stem_norm']):
            stem_params.append(name)
            all_trunk_stem_params.append(name)

        # Trunk blocks
        elif 'blocks.' in name:
            # Extract block index
            block_idx = int(name.split('blocks.')[1].split('.')[0])
            # Determine which layer (every 4 blocks)
            layer_start = (block_idx // 4) * 4
            layer_key = f'blocks_{layer_start}-{layer_start+3}'
            trunk_params[layer_key].append(name)
            all_trunk_stem_params.append(name)

    # Compute cosine similarities
    # Overall
    overall_cos, overall_norm_p, overall_norm_v = cosine_sim(policy_grads, value_grads, all_trunk_stem_params)

    # Stem
    stem_cos, stem_norm_p, stem_norm_v = cosine_sim(policy_grads, value_grads, stem_params)

    # Trunk layers (every 4 blocks)
    trunk_metrics = {}
    for layer_key in sorted(trunk_params.keys()):
        layer_cos, layer_norm_p, layer_norm_v = cosine_sim(policy_grads, value_grads, trunk_params[layer_key])
        trunk_metrics[layer_key] = (layer_cos, layer_norm_p, layer_norm_v)

    # Build metrics dictionary
    metrics = {
        'overall_cos_sim': overall_cos,
        'overall_policy_norm': overall_norm_p,
        'overall_value_norm': overall_norm_v,
        'stem_cos_sim': stem_cos,
        'stem_policy_norm': stem_norm_p,
        'stem_value_norm': stem_norm_v,
    }

    # Add trunk block metrics
    for layer_key in ['blocks_0-3', 'blocks_4-7', 'blocks_8-11', 'blocks_12-15']:
        cos, norm_p, norm_v = trunk_metrics.get(layer_key, (0.0, 0.0, 0.0))
        prefix = layer_key.replace('-', '_')
        metrics[f'{prefix}_cos_sim'] = cos
        metrics[f'{prefix}_policy_norm'] = norm_p
        metrics[f'{prefix}_value_norm'] = norm_v
    
    model.zero_grad()
    for name, param in model.named_parameters():
        if name in saved_grads:
            param.grad = saved_grads[name]
    return metrics


# ============================================================================
# Training Logic
# ============================================================================

def prepare_batch_data(model: nn.Module, trajectories: List[Trajectory],
                       device: torch.device, update: int,
                       win_boost: float, block_boost: float,
                       cler_samples: List[dict] = None) -> Tuple[torch.Tensor, ...]:
    """Prepare tensor batches from trajectories, applying tactics and CLER."""
    use_value_baseline = (update >= VALUE_BASELINE_START)
    if use_value_baseline and GAE_LAMBDA < 1.0:
        all_traj_gae = compute_gae_for_trajectories(model, trajectories, device)
    else:
        all_traj_gae = None

    all_obs, all_next_obs, all_actions, all_masks = [], [], [], []
    all_returns, all_value_targets, all_gae_advantages = [], [], []
    all_is_synthetic, all_is_terminal, all_weights = [], [], []
    
    # Pre-logging lists
    all_returns_for_logging = []

    imitation_enabled = IMITATION_WEIGHT > 0 and update >= IMITATION_START_UPDATE
    
    # Stats tracking
    stats = {
        'num_wins': 0, 'num_blocks': 0, 
        'num_synthetic_wins_eq': 0, 'num_synthetic_wins_missed': 0, 'num_synthetic_blocks': 0,
        'win_opp': 0, 'win_miss': 0, 'block_opp': 0, 'block_miss': 0,
        'num_imitation_black': 0, 'num_imitation_white': 0
    }

    for traj_idx, traj in enumerate(trajectories):
        returns = compute_returns(traj)
        current_steps = sum(1 for is_current in traj.is_current_policy if is_current)
        if current_steps == 0: continue

        current_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0
        imitation_weight = IMITATION_WEIGHT * current_weight if imitation_enabled else 0

        for step_idx, (obs, action, legal_mask, z_t, is_current) in enumerate(zip(
            traj.observations, traj.actions, traj.legal_masks, returns, traj.is_current_policy
        )):
            gae_adv = all_traj_gae[traj_idx][step_idx] if all_traj_gae is not None else None
            
            # Determine if we keep this sample
            keep = False
            weight = 0.0
            
            if is_current:
                keep = True
                weight = current_weight
                all_returns_for_logging.append(z_t)
            elif imitation_enabled and z_t > 0:
                keep = True
                weight = imitation_weight
                pieces = np.sum(obs[0]) + np.sum(obs[1])
                if pieces % 2 == 0: stats['num_imitation_black'] += 1
                else: stats['num_imitation_white'] += 1

            if keep:
                all_obs.append(obs)
                all_actions.append(action)
                all_masks.append(legal_mask)
                all_returns.append(z_t)
                all_value_targets.append(z_t)
                all_gae_advantages.append(gae_adv if gae_adv is not None else z_t)
                all_is_synthetic.append(False)
                all_weights.append(weight)
                
                if step_idx + 1 < len(traj.observations):
                    all_next_obs.append(traj.observations[step_idx + 1])
                    all_is_terminal.append(False)
                else:
                    all_next_obs.append(np.zeros_like(obs))
                    all_is_terminal.append(True)

    # Tactical Augmentation
    original_length = len(all_obs)
    for i in range(original_length):
        winning_moves = find_all_win_in_1(all_obs[i], all_masks[i])
        if winning_moves:
            stats['win_opp'] += 1
            if all_actions[i] in winning_moves:
                all_returns[i] = max(0.0, all_returns[i]) + win_boost
                all_gae_advantages[i] = max(0.0, all_gae_advantages[i]) + win_boost
                stats['num_wins'] += 1
                # Add equivalents
                for other_win in winning_moves:
                    if other_win != all_actions[i] and (stats['num_synthetic_wins_eq'] + stats['num_synthetic_wins_missed']) < MAX_SYNTHETIC_WINS:
                        all_obs.append(all_obs[i])
                        all_actions.append(other_win)
                        all_masks.append(all_masks[i])
                        all_returns.append(win_boost)
                        all_value_targets.append(0.0)
                        all_gae_advantages.append(win_boost)
                        all_is_synthetic.append(True)
                        all_weights.append(all_weights[i])
                        all_next_obs.append(np.zeros_like(all_obs[i]))
                        all_is_terminal.append(True)
                        stats['num_synthetic_wins_eq'] += 1
            else:
                stats['win_miss'] += 1
                for winning_move in winning_moves:
                    if (stats['num_synthetic_wins_eq'] + stats['num_synthetic_wins_missed']) < MAX_SYNTHETIC_WINS:
                        all_obs.append(all_obs[i])
                        all_actions.append(winning_move)
                        all_masks.append(all_masks[i])
                        all_returns.append(SYNTHETIC_WIN_BOOST)
                        all_value_targets.append(0.0)
                        all_gae_advantages.append(SYNTHETIC_WIN_BOOST)
                        all_is_synthetic.append(True)
                        all_weights.append(all_weights[i])
                        all_next_obs.append(np.zeros_like(all_obs[i]))
                        all_is_terminal.append(True)
                        stats['num_synthetic_wins_missed'] += 1
        else:
            blocking_moves = find_blocking_moves(all_obs[i], all_masks[i])
            if blocking_moves is not None:
                stats['block_opp'] += 1
                if all_actions[i] in blocking_moves:
                    all_returns[i] = max(0.0, all_returns[i]) + block_boost
                    all_gae_advantages[i] = max(0.0, all_gae_advantages[i]) + block_boost
                    stats['num_blocks'] += 1
                else:
                    stats['block_miss'] += 1
                    if stats['num_synthetic_blocks'] < MAX_SYNTHETIC_BLOCKS:
                        all_obs.append(all_obs[i])
                        all_actions.append(blocking_moves[0])
                        all_masks.append(all_masks[i])
                        all_returns.append(SYNTHETIC_BLOCKING_BOOST)
                        all_value_targets.append(0.0)
                        all_gae_advantages.append(SYNTHETIC_BLOCKING_BOOST)
                        all_is_synthetic.append(True)
                        all_weights.append(all_weights[i])
                        all_next_obs.append(np.zeros_like(all_obs[i]))
                        all_is_terminal.append(False)
                        stats['num_synthetic_blocks'] += 1

    # CLER Samples
    if cler_samples:
        for sample in cler_samples:
            all_obs.append(sample['obs'])
            all_actions.append(sample['action'])
            all_masks.append(sample['mask'])
            all_returns.append(sample['strength'])
            all_value_targets.append(0.0)
            all_gae_advantages.append(sample['strength'])
            all_is_synthetic.append(True)
            all_weights.append(sample['weight'])
            all_next_obs.append(np.zeros_like(sample['obs']))
            all_is_terminal.append(True)

    # To Tensors
    tensors = (
        obs_batch_to_tensor(all_obs, device),
        obs_batch_to_tensor(all_next_obs, device),
        torch.tensor(all_actions, dtype=torch.long, device=device),
        mask_batch_to_tensor(all_masks, device),
        torch.tensor(all_returns, dtype=torch.float32, device=device),
        torch.tensor(all_value_targets, dtype=torch.float32, device=device),
        torch.tensor(all_gae_advantages, dtype=torch.float32, device=device),
        torch.tensor(all_is_synthetic, dtype=torch.bool, device=device),
        torch.tensor(all_is_terminal, dtype=torch.bool, device=device),
        torch.tensor(all_weights, dtype=torch.float32, device=device)
    )
    return tensors, stats, all_returns_for_logging


def _train_on_batch_internal(model: nn.Module, tensors: Tuple[torch.Tensor, ...],
                             optimizer: torch.optim.Optimizer,
                             update: int, ema_entropy: float,
                             num_accumulation_steps: int) -> Tuple[float, ...]:
    """
    Internal training step: processes one chunk of data.
    """
    (obs, next_obs, actions, masks, returns, val_targets, advantages, is_synth, is_term, weights) = tensors
    
    # 8-fold Augmentation
    aug_obs, aug_actions, aug_masks = augment_batch_gpu(obs, actions, masks)
    dummy_masks = torch.ones_like(masks)
    aug_next_obs, _, _ = augment_batch_gpu(next_obs, actions, dummy_masks)
    
    aug_returns = returns.repeat(8)
    aug_val_targets = val_targets.repeat(8)
    aug_advantages = advantages.repeat(8)
    aug_is_synth = is_synth.repeat(8)
    aug_is_term = is_term.repeat(8)
    aug_weights = weights.repeat(8)

    # Norms
    global_policy_norm = aug_weights.sum().item()
    value_loss_mask = ~aug_is_synth
    global_value_norm = (aug_weights * value_loss_mask.float()).sum().item()

    # Target Entropy
    midpoint = TOTAL_UPDATES * ENTROPY_DECAY_MIDPOINT_PERCENTAGE
    steepness_k = 3.0 / (TOTAL_UPDATES * ENTROPY_DECAY_STEEPNESS)
    sigmoid_factor = (1.0 - torch.tanh(torch.tensor(steepness_k * (update - midpoint)))) / 2.0
    target_entropy = ENTROPY_TARGET_END + (ENTROPY_TARGET_START - ENTROPY_TARGET_END) * sigmoid_factor.item()
    entropy_bonus_scale = max(0.0, target_entropy - ema_entropy) if ema_entropy is not None else 1.0

    accum_loss = 0.0
    accum_val_loss = 0.0
    accum_entropy = 0.0
    accum_mse = 0.0
    accum_mse_count = 0
    accum_weight = 0.0
    
    probe_metrics = None
    use_value_baseline = (update >= VALUE_BASELINE_START)

    # Micro-batches
    num_samples = len(aug_obs)
    for start in range(0, num_samples, TRAIN_BATCH_SIZE):
        end = min(start + TRAIN_BATCH_SIZE, num_samples)
        
        b_obs = aug_obs[start:end]
        b_next_obs = aug_next_obs[start:end]
        b_actions = aug_actions[start:end]
        b_masks = aug_masks[start:end]
        b_returns = aug_returns[start:end]
        b_val_targets = aug_val_targets[start:end]
        b_advantages = aug_advantages[start:end]
        b_is_synth = aug_is_synth[start:end]
        b_is_term = aug_is_term[start:end]
        b_weights = aug_weights[start:end]
        
        logits_grid, values = model(b_obs)
        logits = logits_grid.squeeze(1).masked_fill(~b_masks, -1e9)
        logits_flat = logits.view(end-start, 225)
        
        logits_scaled = logits_flat / TEMPERATURE_TRAIN if TEMPERATURE_TRAIN > 0 else logits_flat
        dist = Categorical(logits=logits_scaled)
        entropies = dist.entropy()
        log_probs = dist.log_prob(b_actions)
        log_probs = torch.clamp(log_probs, min=-10.0)
        
        with torch.no_grad():
            _, next_values = model(b_next_obs)
        values = values.squeeze(1)
        next_values = next_values.squeeze(1)
        
        eff_advantages = b_advantages if use_value_baseline else b_returns
        
        # Value Loss
        eff_val_targets = torch.where(b_is_term, b_val_targets, -next_values.detach())
        val_mse = F.mse_loss(values, eff_val_targets, reduction='none')
        b_val_mask = ~b_is_synth
        val_loss = (b_weights * b_val_mask.float() * val_mse).sum() / max(global_value_norm, 1.0)
        
        # Policy Loss
        pol_loss = -(b_weights * eff_advantages * log_probs).sum() / max(global_policy_norm, 1.0)
        
        # Entropy Loss
        ent_loss = -(b_weights * entropies).sum() / max(global_policy_norm, 1.0)
        
        if probe_metrics is None and PROBE_INTERVAL > 0 and (update + 1) % PROBE_INTERVAL == 0 and start == 0:
            probe_metrics = probe_gradient_conflict(model, pol_loss, val_loss)

        total_loss = pol_loss + VALUE_LOSS_COEFF * val_loss + ENTROPY_BONUS_COEFF * entropy_bonus_scale * ent_loss
        
        # Accumulate: scale by accumulation steps (chunks)
        loss_scaled = total_loss / num_accumulation_steps
        loss_scaled.backward()
        
        accum_loss += total_loss.item()
        accum_val_loss += val_loss.item()
        accum_entropy += (b_weights * entropies).sum().item()
        accum_mse += (val_mse * b_val_mask.float()).sum().item()
        accum_mse_count += b_val_mask.sum().item()
        accum_weight += b_weights.sum().item()

    return (
        accum_loss, # Not averaged over batch, will be averaged by num_chunks in outer loop
        accum_entropy / max(accum_weight, 1.0),
        accum_val_loss,
        accum_mse / max(accum_mse_count, 1),
        probe_metrics
    )


def train_on_batch(model: nn.Module, trajectories: List[Trajectory],
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   update: int,
                   win_boost: float, block_boost: float,
                   cler_samples: List[dict],
                   ema_entropy: float) -> Tuple[float, ...]:
    """Train on a batch of trajectories with chunking."""
    if len(trajectories) == 0:
        return (0.0,) * 5 + (0,) * 12 + (None,)

    # Process chunks
    num_chunks = (len(trajectories) + EPISODES_CHUNK_SIZE - 1) // EPISODES_CHUNK_SIZE
    chunks = [trajectories[i * EPISODES_CHUNK_SIZE:(i + 1) * EPISODES_CHUNK_SIZE] for i in range(num_chunks)]
    
    optimizer.zero_grad()
    
    total_loss = 0.0
    total_ent = 0.0
    total_val_loss = 0.0
    total_mse = 0.0
    
    # Accumulate stats
    total_wins = 0; total_blocks = 0
    total_sw_eq = 0; total_sw_miss = 0; total_sb = 0
    total_imi_b = 0; total_imi_w = 0
    total_w_opp = 0; total_w_miss = 0; total_b_opp = 0; total_b_miss = 0
    total_returns = []
    
    collected_probe = None

    for i, chunk in enumerate(chunks):
        chunk_cler = cler_samples if i == 0 else None # Give CLER to first chunk
        
        tensors, stats, chunk_returns = prepare_batch_data(model, chunk, device, update, win_boost, block_boost, chunk_cler)
        
        loss, ent, val_loss, mse, probe = _train_on_batch_internal(
            model, tensors, optimizer, update, ema_entropy, num_accumulation_steps=num_chunks
        )
        
        if i == 0 and probe: collected_probe = probe
        
        total_loss += loss
        total_ent += ent
        total_val_loss += val_loss
        total_mse += mse
        
        total_wins += stats['num_wins']
        total_blocks += stats['num_blocks']
        total_sw_eq += stats['num_synthetic_wins_eq']
        total_sw_miss += stats['num_synthetic_wins_missed']
        total_sb += stats['num_synthetic_blocks']
        total_imi_b += stats['num_imitation_black']
        total_imi_w += stats['num_imitation_white']
        total_w_opp += stats['win_opp']
        total_w_miss += stats['win_miss']
        total_b_opp += stats['block_opp']
        total_b_miss += stats['block_miss']
        total_returns.extend(chunk_returns)

    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad()

    return (
        total_loss / num_chunks,
        np.mean(total_returns) if total_returns else 0.0,
        total_ent / num_chunks,
        total_val_loss / num_chunks,
        total_mse / num_chunks,
        total_wins, total_blocks,
        total_sw_eq, total_sw_miss, total_sb,
        total_imi_b, total_imi_w,
        total_w_opp, total_w_miss, total_b_opp, total_b_miss,
        len(cler_samples) if cler_samples else 0,
        collected_probe
    )