"""
Self-Play REINFORCE Training for Gomoku Policy Network - BATCHED VERSION

Implements batched inference during self-play for much higher GPU utilization.
Processes multiple game positions simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import copy
import numpy as np
from typing import List, Tuple, Optional, Dict
import time
import json
import os
import glob
import re
import argparse

from csv_logger import CSVLogger

from gomoku_selfplay import (
    Player, GameState, Trajectory,
    play_episodes_batched, play_eval_games, augment_batch_gpu,
    find_all_win_in_1, find_blocking_moves,
    compute_returns, compute_gae_for_trajectories, compute_outcome_stats,
    board_from_observation, get_local_candidate_moves, idx_to_pos, encode_observation,
    RENJU_OPENING_SEQUENCES
)

from model import (
    # Configuration
    TOTAL_UPDATES, N_BLOCKS, WIDTH,
    STEM_3X3_CHANNELS, STEM_SPARSE_5X5_CHANNELS, STEM_DENSE_5X5_CHANNELS,
    STEM_SPARSE_7X7_CHANNELS, STEM_DENSE_7X7_CHANNELS, GROUPNORM_GROUPS,
    TRUNK_DILATION2_SCHEDULE, SE_SCHEDULE,
    POLICY_HEAD_D, VALUE_HEAD_CHANNELS, VALUE_HEAD_HIDDEN,
    LEARNING_RATE, MIN_LR, LR_DECAY, WEIGHT_DECAY, GRAD_CLIP_NORM,
    EPISODES_PER_UPDATE, EPISODES_CHUNK_SIZE, BATCH_INFERENCE_SIZE, TRAIN_BATCH_SIZE,
    TEMPERATURE_TRAIN, ENTROPY_COEFF_START, ENTROPY_COEFF_END,
    ENTROPY_DECAY_MIDPOINT_PERCENTAGE, ENTROPY_DECAY_STEEPNESS,
    VALUE_LOSS_COEFF, GAE_LAMBDA, VALUE_BASELINE_START,
    MISS_RATE_EMA_WINDOW, WIN_MIN_BOOST, WIN_MAX_BOOST, BLOCK_MIN_BOOST, BLOCK_MAX_BOOST,
    SYNTHETIC_WIN_BOOST, SYNTHETIC_BLOCKING_BOOST, MAX_SYNTHETIC_WINS, MAX_SYNTHETIC_BLOCKS,
    EPISODE_WEIGHT_ALPHA, IMITATION_WEIGHT, IMITATION_START_UPDATE,
    CF_START_UPDATE, CF_TRIGGER_PROB, CF_ADVANTAGE, CF_MIN_STEPS_TO_END, CF_ENTROPY_TH,
    CF_RADIUS, CF_NUM_ACTIONS, CF_NUM_ROLLOUTS, CF_MIN_ADD_WINRATE,
    CF_MAX_SAMPLES_PER_UPDATE, CF_ROLLOUT_TEMP,
    SEED_PROBABILITY,
    OPPONENT_POOL_SIZE, EVAL_ROUNDS, EVAL_TEMP,
    EVAL_INTERVAL_EARLY, EVAL_INTERVAL_MID, EVAL_INTERVAL_LATE, WIN_RATE_THRESHOLD,
    SCAN_START_UPDATE, SCAN_PERIOD, NUM_SCAN_BUCKETS,
    QUICK_SCREEN_ROUNDS, TOP_K_QUICK_SCREEN, FINAL_SCREEN_ROUNDS, MAX_MINED_OPPONENTS_PER_EVENT,
    UNIFORM_SAMPLING_FRACTION, DEFAULT_WIN_RATE,
    LOG_PROB_MIN, LOGIT_MASK_VALUE,
    PRINT_INTERVAL, TRAINING_STATE_FILE, PROBE_INTERVAL, DEVICE,
    # Classes and functions
    GomokuPolicyNet, zero_center_taps,
    obs_batch_to_tensor, mask_batch_to_tensor, select_action_batch, select_action_batch_eval
)

# ============================================================================
# Counterfactual Low-Entropy Rescue (CLER)
# ============================================================================

def run_cler_rollout(model: nn.Module, obs: np.ndarray, next_player: Player,
                     first_action: int, temperature: float, device: torch.device) -> bool:
    """
    Run a single rollout from a position with a forced first move.

    Args:
        model: Policy network
        obs: Observation at the decision point [3, 15, 15]
        next_player: Player to move at decision point
        first_action: Forced first action (flat index)
        temperature: Rollout temperature
        device: Torch device

    Returns:
        True if the player who made first_action wins, False otherwise
    """
    # Reconstruct board and apply first move
    board = board_from_observation(obs, next_player)
    row, col = idx_to_pos(first_action)
    outcome = board.Move((row, col))

    if outcome != GameState.CONTINUE:
        # Game ended immediately
        if outcome == GameState.DRAW:
            return False  # Draw counts as loss for CLER purposes
        return (outcome == GameState.BLACK_WIN and next_player == Player.BLACK) or \
               (outcome == GameState.WHITE_WIN and next_player == Player.WHITE)

    # Play out the rest of the game
    while True:
        legal_mask, current_player = board.GetLegalMoves()
        c0, c1, _ = board.GetBoardState()
        current_obs = encode_observation(c0, c1)

        # Get action from model
        actions = select_action_batch_eval(model, [current_obs], [legal_mask], temperature, device)
        action = actions[0]

        row, col = idx_to_pos(action)
        outcome = board.Move((row, col))

        if outcome != GameState.CONTINUE:
            if outcome == GameState.DRAW:
                return False
            return (outcome == GameState.BLACK_WIN and next_player == Player.BLACK) or \
                   (outcome == GameState.WHITE_WIN and next_player == Player.WHITE)


def generate_cler_samples(trajectories: List[Trajectory],
                          current_is_black: List[bool],
                          model: nn.Module,
                          device: torch.device,
                          update: int = 0) -> Tuple[List[dict], dict]:
    """
    Generate Counterfactual Low-Entropy Rescue samples from lost games.

    For lost games, identifies steps where the policy was overconfident (low entropy)
    but far from terminal. Tries alternative moves via rollouts to find better options.

    Args:
        trajectories: List of game trajectories
        current_is_black: List indicating if current policy played as black
        model: Policy network for rollouts
        device: Torch device
        update: Current update number

    Returns:
        Tuple of (cler_samples, metrics) where:
        - cler_samples: List of dicts with keys: obs, action, mask, strength, weight
        - metrics: Dict with logging metrics
    """
    cler_samples = []
    metrics = {
        'cf_attempted_episodes': 0,
        'cf_steps_candidates_total': 0,
        'cf_added_samples': 0,
        'cf_best_winrate_sum': 0.0,
        'cf_entropy_selected_sum': 0.0,
    }

    # Skip if before start update
    if update < CF_START_UPDATE:
        return cler_samples, metrics

    model.eval()

    for traj, is_black in zip(trajectories, current_is_black):
        # Only process losses for current policy
        outcome = traj.outcome
        is_loss = (outcome == GameState.BLACK_WIN and not is_black) or \
                  (outcome == GameState.WHITE_WIN and is_black)

        if not is_loss:
            continue

        # Trigger with probability
        if random.random() > CF_TRIGGER_PROB:
            continue

        metrics['cf_attempted_episodes'] += 1

        # Find candidate steps: low entropy, far from terminal, current policy's turn
        T = len(traj.observations)
        candidates = []

        for t in range(T):
            # Check if current policy's turn
            if not traj.is_current_policy[t]:
                continue

            # Check steps to end
            steps_to_end = T - t - 1
            if steps_to_end < CF_MIN_STEPS_TO_END:
                continue

            # Check entropy threshold
            entropy = traj.entropies[t]
            if entropy >= CF_ENTROPY_TH:
                continue

            # Skip if tactical position (already handled by tactical bonuses)
            obs = traj.observations[t]
            mask = traj.legal_masks[t]
            if find_all_win_in_1(obs, mask):
                continue
            if find_blocking_moves(obs, mask) is not None:
                continue

            # Weight: lower entropy = higher weight
            weight = CF_ENTROPY_TH - entropy
            candidates.append((t, weight, entropy))

        metrics['cf_steps_candidates_total'] += len(candidates)

        if not candidates:
            continue

        # Weighted sample one step
        weights = [c[1] for c in candidates]
        total_weight = sum(weights)
        r = random.random() * total_weight
        cumsum = 0.0
        selected_idx = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                selected_idx = i
                break

        t_star, _, selected_entropy = candidates[selected_idx]
        metrics['cf_entropy_selected_sum'] += selected_entropy

        # Get state at t_star
        obs = traj.observations[t_star]
        mask = traj.legal_masks[t_star]
        player = traj.players[t_star]
        original_action = traj.actions[t_star]

        # Get local candidate moves
        local_candidates = get_local_candidate_moves(obs, mask, radius=CF_RADIUS)

        # Remove original action from candidates
        local_candidates = [a for a in local_candidates if a != original_action]

        if not local_candidates:
            continue

        # Sample up to CF_NUM_ACTIONS alternatives
        if len(local_candidates) > CF_NUM_ACTIONS:
            alt_actions = random.sample(local_candidates, CF_NUM_ACTIONS)
        else:
            alt_actions = local_candidates

        # Run rollouts for each alternative
        best_action = None
        best_winrate = 0.0

        for action in alt_actions:
            wins = 0
            for _ in range(CF_NUM_ROLLOUTS):
                if run_cler_rollout(model, obs, player, action, CF_ROLLOUT_TEMP, device):
                    wins += 1

            winrate = wins / CF_NUM_ROLLOUTS
            if winrate > best_winrate:
                best_winrate = winrate
                best_action = action

        # Check if best alternative passes threshold
        if best_winrate >= CF_MIN_ADD_WINRATE and best_action is not None:
            # Compute weight (same as step weight from original trajectory)
            current_steps = sum(1 for is_current in traj.is_current_policy if is_current)
            step_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0

            cler_samples.append({
                'obs': obs,
                'action': best_action,
                'mask': mask,
                'strength': best_winrate * CF_ADVANTAGE,
                'weight': step_weight,
            })

            metrics['cf_added_samples'] += 1
            metrics['cf_best_winrate_sum'] += best_winrate

            # Limit samples per update
            if len(cler_samples) >= CF_MAX_SAMPLES_PER_UPDATE:
                break

        if len(cler_samples) >= CF_MAX_SAMPLES_PER_UPDATE:
            break

    model.train()

    return cler_samples, metrics


# ============================================================================
# Gradient Conflict Probing
# ============================================================================

def probe_gradient_conflict(model: nn.Module, policy_loss: torch.Tensor,
                             value_loss: torch.Tensor, update: int) -> Dict[str, float]:
    """
    Probe gradient conflict between policy and value losses.

    Computes cosine similarity between policy gradients and value gradients
    for different parts of the network:
    - Stem
    - Trunk blocks 0-3, 4-7, 8-11, 12-15
    - Overall (all trunk + stem)

    Args:
        model: The neural network model
        policy_loss: Policy loss tensor (single scalar)
        value_loss: Value loss tensor (single scalar)
        update: Current update number (for logging)

    Returns:
        Dictionary containing gradient conflict metrics
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

    # Helper function to compute cosine similarity
    def cosine_sim(grad_dict_1: dict, grad_dict_2: dict, param_names: List[str]) -> Tuple[float, float, float]:
        """
        Compute cosine similarity between two gradient dictionaries for specified parameters.

        Returns:
            Tuple of (cosine_similarity, norm1, norm2)
        """
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

    # Add trunk block metrics
    for layer_key in ['blocks_0-3', 'blocks_4-7', 'blocks_8-11', 'blocks_12-15']:
        cos, norm_p, norm_v = trunk_metrics.get(layer_key, (0.0, 0.0, 0.0))
        prefix = layer_key.replace('-', '_')
        metrics[f'{prefix}_cos_sim'] = cos
        metrics[f'{prefix}_policy_norm'] = norm_p
        metrics[f'{prefix}_value_norm'] = norm_v

    return metrics

# ============================================================================
# Training
# ============================================================================

def _train_on_batch_internal(model: nn.Module, trajectories: List[Trajectory],
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             num_accumulation_steps: int = 1,
                             do_optimizer_step: bool = True,
                             update: int = 0,
                             win_boost: float = 0.0,
                             block_boost: float = 0.0,
                             cler_samples: List[dict] = None) -> Tuple[float, float, float, float, float, int, int, int, int, int, int, int, int, int, int, int, int, Optional[Dict[str, float]]]:
    """
    Internal training function - processes a batch of trajectories.

    Phased Training:
    - Phase 1 (before VALUE_BASELINE_START): Use boosted returns (no value baseline)
    - Phase 2 (VALUE_BASELINE_START onwards): Use boosted returns with value baseline

    Tactical bonuses (win-in-1, blocking) are ALWAYS active to overcome terminal state problem.
    Value head is trained on UNBOOSTED actual game outcomes.
    """
    use_value_baseline = (update >= VALUE_BASELINE_START)

    if use_value_baseline and GAE_LAMBDA < 1.0:
        all_traj_gae = compute_gae_for_trajectories(model, trajectories, device, GAE_LAMBDA, obs_batch_to_tensor)
    else:
        all_traj_gae = None

    all_obs = []
    all_next_obs = []
    all_actions = []
    all_masks = []
    all_returns = []
    all_value_targets = []
    all_is_synthetic = []
    all_is_terminal = []
    all_gae_advantages = []
    all_weights = []
    all_returns_for_logging = []

    num_trajectories = 0
    num_imitation_black = 0
    num_imitation_white = 0

    for traj_idx, traj in enumerate(trajectories):
        returns = compute_returns(traj)

        current_steps = sum(1 for is_current in traj.is_current_policy if is_current)

        if current_steps == 0:
            continue

        num_trajectories += 1

        imitation_enabled = IMITATION_WEIGHT > 0 and update >= IMITATION_START_UPDATE

        current_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0
        imitation_weight = IMITATION_WEIGHT * current_weight if imitation_enabled else 0

        for step_idx, (obs, action, legal_mask, log_prob, z_t, is_current) in enumerate(zip(
            traj.observations, traj.actions, traj.legal_masks, traj.log_probs, returns, traj.is_current_policy
        )):
            gae_adv = all_traj_gae[traj_idx][step_idx] if all_traj_gae is not None else None

            if is_current:
                all_obs.append(obs)
                all_actions.append(action)
                all_masks.append(legal_mask)
                all_returns.append(z_t)
                all_value_targets.append(z_t)
                all_is_synthetic.append(False)
                all_gae_advantages.append(gae_adv if gae_adv is not None else z_t)
                all_weights.append(current_weight)
                all_returns_for_logging.append(z_t)

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
                all_gae_advantages.append(gae_adv if gae_adv is not None else z_t)
                all_weights.append(imitation_weight)
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

    # Tactical enhancement
    num_wins = 0
    num_blocks = 0
    num_synthetic_wins_eq = 0
    num_synthetic_wins_missed = 0
    num_synthetic_blocks = 0

    win_opp = 0
    win_miss = 0
    block_opp = 0
    block_miss = 0

    original_length = len(all_obs)

    for i in range(original_length):
        winning_moves = find_all_win_in_1(all_obs[i], all_masks[i])
        if winning_moves:
            win_opp += 1

            if all_actions[i] in winning_moves:
                all_returns[i] = max(0.0, all_returns[i]) + win_boost
                all_gae_advantages[i] = max(0.0, all_gae_advantages[i]) + win_boost
                num_wins += 1

                for other_win in winning_moves:
                    if other_win != all_actions[i] and (num_synthetic_wins_eq + num_synthetic_wins_missed) < MAX_SYNTHETIC_WINS:
                        all_obs.append(all_obs[i])
                        all_actions.append(other_win)
                        all_masks.append(all_masks[i])
                        all_returns.append(win_boost)
                        all_value_targets.append(0.0)
                        all_is_synthetic.append(True)
                        all_gae_advantages.append(win_boost)
                        all_weights.append(all_weights[i])
                        all_next_obs.append(np.zeros_like(all_obs[i]))
                        all_is_terminal.append(True)
                        num_synthetic_wins_eq += 1
            else:
                win_miss += 1
                for winning_move in winning_moves:
                    if (num_synthetic_wins_eq + num_synthetic_wins_missed) < MAX_SYNTHETIC_WINS:
                        all_obs.append(all_obs[i])
                        all_actions.append(winning_move)
                        all_masks.append(all_masks[i])
                        all_returns.append(SYNTHETIC_WIN_BOOST)
                        all_value_targets.append(0.0)
                        all_is_synthetic.append(True)
                        all_gae_advantages.append(SYNTHETIC_WIN_BOOST)
                        all_weights.append(all_weights[i])
                        all_next_obs.append(np.zeros_like(all_obs[i]))
                        all_is_terminal.append(True)
                        num_synthetic_wins_missed += 1
        else:
            blocking_moves = find_blocking_moves(all_obs[i], all_masks[i])
            if blocking_moves is not None:
                block_opp += 1

                if all_actions[i] in blocking_moves:
                    all_returns[i] = max(0.0, all_returns[i]) + block_boost
                    all_gae_advantages[i] = max(0.0, all_gae_advantages[i]) + block_boost
                    num_blocks += 1
                else:
                    block_miss += 1
                    if num_synthetic_blocks < MAX_SYNTHETIC_BLOCKS:
                        all_obs.append(all_obs[i])
                        all_actions.append(blocking_moves[0])
                        all_masks.append(all_masks[i])
                        all_returns.append(SYNTHETIC_BLOCKING_BOOST)
                        all_value_targets.append(0.0)
                        all_is_synthetic.append(True)
                        all_gae_advantages.append(SYNTHETIC_BLOCKING_BOOST)
                        all_weights.append(all_weights[i])
                        all_next_obs.append(np.zeros_like(all_obs[i]))
                        all_is_terminal.append(False)
                        num_synthetic_blocks += 1

    # CLER (Counterfactual Low-Entropy Rescue) samples
    num_cler_samples = 0
    if cler_samples:
        for sample in cler_samples:
            all_obs.append(sample['obs'])
            all_actions.append(sample['action'])
            all_masks.append(sample['mask'])
            all_returns.append(sample['strength'])
            all_value_targets.append(0.0)
            all_is_synthetic.append(True)
            all_gae_advantages.append(sample['strength'])
            all_weights.append(sample['weight'])
            all_next_obs.append(np.zeros_like(sample['obs']))
            all_is_terminal.append(True)
            num_cler_samples += 1

    # Convert to GPU tensors
    obs_tensor = obs_batch_to_tensor(all_obs, device)
    next_obs_tensor = obs_batch_to_tensor(all_next_obs, device)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=device)
    masks_tensor = mask_batch_to_tensor(all_masks, device)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=device)
    value_targets_tensor = torch.tensor(all_value_targets, dtype=torch.float32, device=device)
    gae_advantages_tensor = torch.tensor(all_gae_advantages, dtype=torch.float32, device=device)
    is_synthetic_tensor = torch.tensor(all_is_synthetic, dtype=torch.bool, device=device)
    is_terminal_tensor = torch.tensor(all_is_terminal, dtype=torch.bool, device=device)
    weights_tensor = torch.tensor(all_weights, dtype=torch.float32, device=device)

    # Apply all 8 symmetries
    aug_obs, aug_actions, aug_masks = augment_batch_gpu(obs_tensor, actions_tensor, masks_tensor)
    dummy_masks = torch.ones_like(masks_tensor)
    aug_next_obs, _, _ = augment_batch_gpu(next_obs_tensor, actions_tensor, dummy_masks)

    aug_returns = returns_tensor.repeat(8)
    aug_value_targets = value_targets_tensor.repeat(8)
    aug_gae_advantages = gae_advantages_tensor.repeat(8)
    aug_is_synthetic = is_synthetic_tensor.repeat(8)
    aug_is_terminal = is_terminal_tensor.repeat(8)
    aug_weights = weights_tensor.repeat(8)

    global_policy_entropy_normalizer = aug_weights.sum().item()
    value_loss_mask_global = ~aug_is_synthetic
    global_value_normalizer = (aug_weights * value_loss_mask_global.float()).sum().item()

    # Entropy coefficient with sigmoid decay
    midpoint = TOTAL_UPDATES * ENTROPY_DECAY_MIDPOINT_PERCENTAGE
    steepness_k = 3.0 / (TOTAL_UPDATES * ENTROPY_DECAY_STEEPNESS)
    sigmoid_factor = (1.0 - torch.tanh(torch.tensor(steepness_k * (update - midpoint)))) / 2.0
    current_entropy_coeff = ENTROPY_COEFF_END + (ENTROPY_COEFF_START - ENTROPY_COEFF_END) * sigmoid_factor.item()

    accumulated_loss = 0.0
    accumulated_value_loss = 0.0
    accumulated_value_mse_sum = 0.0
    accumulated_value_mse_count = 0
    accumulated_weighted_policy_loss_sum = 0.0
    accumulated_weighted_entropy_sum = 0.0
    accumulated_weight_sum = 0.0

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

        with torch.no_grad():
            _, next_values = model(batch_next_obs)

        values = values.squeeze(1)
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
        entropy_loss_mb = -(batch_weights * entropies).sum() / max(global_policy_entropy_normalizer, 1.0)

        # Gradient conflict probing (only on first micro-batch)
        probe_metrics = None
        if PROBE_INTERVAL > 0 and update % PROBE_INTERVAL == 0 and batch_start == 0:
            probe_metrics = probe_gradient_conflict(model, policy_loss_mb, value_loss_mb, update)

        loss_mb = (policy_loss_mb + VALUE_LOSS_COEFF * value_loss_mb + current_entropy_coeff * entropy_loss_mb) / num_accumulation_steps
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

    if do_optimizer_step:
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

    mean_return = np.mean(all_returns_for_logging)

    return total_loss_scalar, mean_return, total_entropy_scalar, total_value_loss_scalar, raw_value_mse, num_wins, num_blocks, num_synthetic_wins_eq, num_synthetic_wins_missed, num_synthetic_blocks, num_imitation_black, num_imitation_white, win_opp, win_miss, block_opp, block_miss, num_cler_samples, probe_metrics


def train_on_batch(model: nn.Module, trajectories: List[Trajectory],
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   chunk_size: int = EPISODES_CHUNK_SIZE,
                   update: int = 0,
                   win_boost: float = 0.0,
                   block_boost: float = 0.0,
                   cler_samples: List[dict] = None) -> Tuple[float, float, float, float, float, int, int, int, int, int, int, int, int, int, int, int, int, Optional[Dict[str, float]]]:
    """Train on a batch of trajectories with gradient accumulation."""
    if len(trajectories) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None

    num_chunks = (len(trajectories) + chunk_size - 1) // chunk_size
    chunks = [trajectories[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    optimizer.zero_grad()

    total_loss = 0.0
    total_returns = []
    total_entropy = 0.0
    total_value_loss = 0.0
    total_raw_value_mse = 0.0
    total_wins = 0
    total_blocks = 0
    total_synthetic_wins_eq = 0
    total_synthetic_wins_missed = 0
    total_synthetic_blocks = 0
    total_imitation_black = 0
    total_imitation_white = 0
    total_win_opp = 0
    total_win_miss = 0
    total_block_opp = 0
    total_block_miss = 0
    total_cler_samples = 0
    num_chunks_processed = 0
    collected_probe_metrics = None

    for i, chunk in enumerate(chunks):
        is_last_chunk = (i == len(chunks) - 1)

        # Pass CLER samples only to first chunk to avoid duplicating them
        chunk_cler_samples = cler_samples if i == 0 else None

        loss, mean_return, mean_entropy, value_loss, raw_value_mse, num_wins, num_blocks, num_synthetic_wins_eq, num_synthetic_wins_missed, num_synthetic_blocks, num_imitation_black, num_imitation_white, chunk_win_opp, chunk_win_miss, chunk_block_opp, chunk_block_miss, num_cler, probe_metrics = _train_on_batch_internal(
            model, chunk, optimizer, device,
            num_accumulation_steps=num_chunks,
            do_optimizer_step=is_last_chunk,
            update=update,
            win_boost=win_boost,
            block_boost=block_boost,
            cler_samples=chunk_cler_samples
        )

        # Collect probe metrics from first chunk only
        if i == 0 and probe_metrics is not None:
            collected_probe_metrics = probe_metrics

        total_loss += loss * num_chunks
        total_entropy += mean_entropy
        total_value_loss += value_loss
        total_raw_value_mse += raw_value_mse
        total_wins += num_wins
        total_blocks += num_blocks
        total_synthetic_wins_eq += num_synthetic_wins_eq
        total_synthetic_wins_missed += num_synthetic_wins_missed
        total_synthetic_blocks += num_synthetic_blocks
        total_imitation_black += num_imitation_black
        total_imitation_white += num_imitation_white
        total_win_opp += chunk_win_opp
        total_win_miss += chunk_win_miss
        total_block_opp += chunk_block_opp
        total_block_miss += chunk_block_miss
        total_cler_samples += num_cler
        num_chunks_processed += 1

        for traj in chunk:
            returns = compute_returns(traj)
            for z_t, is_current in zip(returns, traj.is_current_policy):
                if is_current:
                    total_returns.append(z_t)

    avg_loss = total_loss / num_chunks_processed
    avg_entropy = total_entropy / num_chunks_processed
    avg_value_loss = total_value_loss / num_chunks_processed
    avg_raw_value_mse = total_raw_value_mse / num_chunks_processed
    mean_return = np.mean(total_returns) if total_returns else 0.0

    return avg_loss, mean_return, avg_entropy, avg_value_loss, avg_raw_value_mse, total_wins, total_blocks, total_synthetic_wins_eq, total_synthetic_wins_missed, total_synthetic_blocks, total_imitation_black, total_imitation_white, total_win_opp, total_win_miss, total_block_opp, total_block_miss, total_cler_samples, collected_probe_metrics


# ============================================================================
# Evaluation Helpers
# ============================================================================

def get_eval_interval(update: int) -> int:
    """Get adaptive evaluation interval based on training progress."""
    if update < 512:
        return EVAL_INTERVAL_EARLY
    elif update < 8192:
        return EVAL_INTERVAL_MID
    else:
        return EVAL_INTERVAL_LATE


# ============================================================================
# Opponent Pool Management
# ============================================================================

def create_random_policy(device: torch.device) -> nn.Module:
    """Create a policy network with random weights."""
    model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    return model


def copy_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Create a deep copy of a model."""
    model_copy = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    model_copy.eval()
    return model_copy


def evaluate_policy(current_model: nn.Module, opponent_pool: deque,
                    device: torch.device,
                    opponent_pool_updates: List[int] = None,
                    num_rounds: int = None) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Evaluate current policy against opponents from the pool.

    Returns:
        Tuple of (overall_win_rate, per_opponent_stats) where per_opponent_stats maps
        opponent update number (as string) to {'wins': int, 'draws': int, 'games': int, 'win_rate': float}
    """
    current_model.eval()

    if num_rounds is None:
        num_rounds = EVAL_ROUNDS

    total_wins = 0
    total_draws = 0
    total_games = 0

    # Track per-opponent statistics
    num_opponents = len(opponent_pool)
    per_opponent_wins = [0] * num_opponents
    per_opponent_draws = [0] * num_opponents
    per_opponent_games = [0] * num_opponents

    for round_idx in range(num_rounds):
        pairs = []
        current_is_black_list = []
        opponent_indices = []  # Track which opponent each game is against

        for opp_idx, opponent in enumerate(opponent_pool):
            # Current plays as black
            pairs.append((current_model, opponent))
            current_is_black_list.append(True)
            opponent_indices.append(opp_idx)
            # Current plays as white
            pairs.append((opponent, current_model))
            current_is_black_list.append(False)
            opponent_indices.append(opp_idx)

        results = play_eval_games(
            pairs, current_is_black_list, EVAL_TEMP, device,
            batch_size=len(pairs),
            select_action_fn=select_action_batch_eval
        )

        for (outcome, current_is_black), opp_idx in zip(results, opponent_indices):
            total_games += 1
            per_opponent_games[opp_idx] += 1

            if outcome == GameState.DRAW:
                total_draws += 1
                per_opponent_draws[opp_idx] += 1
            elif (outcome == GameState.BLACK_WIN and current_is_black) or \
                 (outcome == GameState.WHITE_WIN and not current_is_black):
                total_wins += 1
                per_opponent_wins[opp_idx] += 1

    current_model.train()

    overall_win_rate = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0

    # Build per-opponent stats dictionary
    per_opponent_stats = {}
    for opp_idx in range(num_opponents):
        # Use update number as key if available, otherwise use index
        if opponent_pool_updates is not None and opp_idx < len(opponent_pool_updates):
            key = str(opponent_pool_updates[opp_idx])
        else:
            key = str(opp_idx)

        games = per_opponent_games[opp_idx]
        wins = per_opponent_wins[opp_idx]
        draws = per_opponent_draws[opp_idx]
        win_rate = (wins + 0.5 * draws) / games if games > 0 else DEFAULT_WIN_RATE

        per_opponent_stats[key] = {
            'wins': wins,
            'draws': draws,
            'games': games,
            'win_rate': win_rate
        }

    return overall_win_rate, per_opponent_stats


def find_easiest_opponent_index(opponent_pool_updates: List[int],
                                 per_opponent_win_rates: Dict[str, float]) -> int:
    """
    Find the index of the easiest opponent in the pool (highest win rate for current).

    Args:
        opponent_pool_updates: List of update numbers for each opponent in pool
        per_opponent_win_rates: Dict mapping update number (str) to win rate

    Returns:
        Index of the easiest opponent to evict
    """
    easiest_idx = 0
    easiest_win_rate = -1.0

    for idx, update_num in enumerate(opponent_pool_updates):
        key = str(update_num)
        win_rate = per_opponent_win_rates.get(key, DEFAULT_WIN_RATE)
        if win_rate > easiest_win_rate:
            easiest_win_rate = win_rate
            easiest_idx = idx

    return easiest_idx


def evict_easiest_opponent(opponent_pool: deque, opponent_pool_updates: List[int],
                           per_opponent_win_rates: Dict[str, float]) -> int:
    """
    Remove the easiest opponent from the pool.

    Args:
        opponent_pool: Deque of opponent models
        opponent_pool_updates: List of update numbers for each opponent
        per_opponent_win_rates: Dict mapping update number (str) to win rate

    Returns:
        The update number of the evicted opponent
    """
    evict_idx = find_easiest_opponent_index(opponent_pool_updates, per_opponent_win_rates)

    # Convert deque to list for indexed removal
    pool_list = list(opponent_pool)
    evicted_update = opponent_pool_updates[evict_idx]

    del pool_list[evict_idx]
    del opponent_pool_updates[evict_idx]

    # Clear and repopulate deque
    opponent_pool.clear()
    opponent_pool.extend(pool_list)

    return evicted_update


def add_opponent_to_pool(opponent_pool: deque, opponent_pool_updates: List[int],
                         new_model: nn.Module, new_update: int,
                         per_opponent_win_rates: Dict[str, float],
                         device: torch.device) -> Optional[int]:
    """
    Add a new opponent to the pool, evicting the easiest if pool is full.

    Args:
        opponent_pool: Deque of opponent models
        opponent_pool_updates: List of update numbers for each opponent
        new_model: Model to add (will be copied)
        new_update: Update number of the new model
        per_opponent_win_rates: Dict mapping update number (str) to win rate
        device: Torch device

    Returns:
        Update number of evicted opponent if eviction occurred, None otherwise
    """
    snapshot = copy_model(new_model, device)
    evicted_update = None

    if len(opponent_pool) >= OPPONENT_POOL_SIZE:
        evicted_update = evict_easiest_opponent(opponent_pool, opponent_pool_updates,
                                                 per_opponent_win_rates)

    opponent_pool.append(snapshot)
    opponent_pool_updates.append(new_update)

    return evicted_update


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> Optional[nn.Module]:
    """Load a model from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def discover_historical_checkpoints(output_dir: str, min_update: int = None) -> List[int]:
    """
    Discover all checkpoint files and return their update numbers.

    Args:
        output_dir: Directory containing checkpoints
        min_update: If specified, only return checkpoints >= this update number

    Returns:
        Sorted list of update numbers for available checkpoints
    """
    if min_update is None:
        min_update = SCAN_START_UPDATE

    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint_update_*.pt"))
    update_numbers = []

    pattern = re.compile(r'checkpoint_update_(\d+)\.pt')
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            update_num = int(match.group(1))
            if update_num >= min_update:
                update_numbers.append(update_num)

    return sorted(update_numbers)


def get_bucket_candidates(scan_event_num: int, all_checkpoints: List[int]) -> List[int]:
    """
    Get checkpoint candidates for a given scan event using round-robin bucketing.

    Args:
        scan_event_num: The scan event number (0-indexed)
        all_checkpoints: List of all available checkpoint update numbers

    Returns:
        List of checkpoint update numbers in the current bucket
    """
    target_bucket = scan_event_num % NUM_SCAN_BUCKETS
    candidates = []

    for update_num in all_checkpoints:
        # Bucket assignment based on checkpoint index in evaluation cadence
        checkpoint_bucket = (update_num // EVAL_INTERVAL_LATE) % NUM_SCAN_BUCKETS
        if checkpoint_bucket == target_bucket:
            candidates.append(update_num)

    return candidates


def evaluate_single_opponent(current_model: nn.Module, opponent_model: nn.Module,
                             device: torch.device, num_rounds: int) -> float:
    """
    Evaluate current model against a single opponent.

    Returns:
        Win rate of current model against the opponent
    """
    current_model.eval()

    total_wins = 0
    total_draws = 0
    total_games = 0

    for _ in range(num_rounds):
        pairs = [
            (current_model, opponent_model),  # Current as black
            (opponent_model, current_model),  # Current as white
        ]
        current_is_black_list = [True, False]

        results = play_eval_games(
            pairs, current_is_black_list, EVAL_TEMP, device,
            batch_size=2,
            select_action_fn=select_action_batch_eval
        )

        for outcome, current_is_black in results:
            total_games += 1
            if outcome == GameState.DRAW:
                total_draws += 1
            elif (outcome == GameState.BLACK_WIN and current_is_black) or \
                 (outcome == GameState.WHITE_WIN and not current_is_black):
                total_wins += 1

    current_model.train()

    return (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else DEFAULT_WIN_RATE


def scan_historical_exploiters(output_dir: str, current_model: nn.Module, opponent_pool_updates: List[int],
                                scan_event_num: int, device: torch.device) -> Tuple[List[Tuple[int, float]], int, int]:
    """
    Scan historical checkpoints to find exploiters (hard opponents for current policy).

    Args:
        output_dir: Directory containing checkpoints
        current_model: The current policy model
        opponent_pool_updates: List of update numbers already in the pool (to skip)
        scan_event_num: The scan event number for bucket selection
        device: Torch device

    Returns:
        Tuple of (mined_exploiters, total_candidates, candidates_after_filter) where:
        - mined_exploiters: List of (update_number, win_rate) tuples, sorted by difficulty
        - total_candidates: Total number of candidates in this bucket
        - candidates_after_filter: Number of candidates after filtering pool duplicates
    """
    print(f"  Scanning historical checkpoints (scan event {scan_event_num}, bucket {scan_event_num % NUM_SCAN_BUCKETS})...")

    # Discover all late-phase checkpoints
    all_checkpoints = discover_historical_checkpoints(output_dir, min_update=SCAN_START_UPDATE)
    if not all_checkpoints:
        print(f"  No historical checkpoints found >= update {SCAN_START_UPDATE}")
        return [], 0, 0

    # Get candidates for this bucket
    candidates = get_bucket_candidates(scan_event_num, all_checkpoints)
    total_candidates = len(candidates)
    print(f"  Found {total_candidates} checkpoints in bucket {scan_event_num % NUM_SCAN_BUCKETS}")

    # Filter out checkpoints already in the pool
    pool_set = set(opponent_pool_updates)
    candidates = [c for c in candidates if c not in pool_set]
    candidates_after_filter = len(candidates)
    print(f"  {candidates_after_filter} candidates after filtering pool duplicates")

    if not candidates:
        return [], total_candidates, candidates_after_filter

    # Quick screen: evaluate each candidate with few rounds
    print(f"  Quick screen ({QUICK_SCREEN_ROUNDS} rounds per candidate)...")
    quick_results = []

    for update_num in candidates:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{update_num}.pt")
        opponent = load_checkpoint_model(checkpoint_path, device)
        if opponent is None:
            continue

        win_rate = evaluate_single_opponent(current_model, opponent, device, QUICK_SCREEN_ROUNDS)
        quick_results.append((update_num, win_rate))

        # Clean up
        del opponent
        torch.cuda.empty_cache()

    if not quick_results:
        return [], total_candidates, candidates_after_filter

    # Sort by win rate ascending (lowest = hardest opponents)
    quick_results.sort(key=lambda x: x[1])

    # Keep top K hardest for final screen
    hardest_candidates = quick_results[:TOP_K_QUICK_SCREEN]
    print(f"  Top {len(hardest_candidates)} hardest: {[(u, f'{wr:.2%}') for u, wr in hardest_candidates]}")

    # Final screen: more thorough evaluation
    print(f"  Final screen ({FINAL_SCREEN_ROUNDS} rounds per candidate)...")
    final_results = []

    for update_num, _ in hardest_candidates:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{update_num}.pt")
        opponent = load_checkpoint_model(checkpoint_path, device)
        if opponent is None:
            continue

        win_rate = evaluate_single_opponent(current_model, opponent, device, FINAL_SCREEN_ROUNDS)
        final_results.append((update_num, win_rate))

        del opponent
        torch.cuda.empty_cache()

    # Sort by win rate ascending and return top MAX_MINED_OPPONENTS_PER_EVENT
    final_results.sort(key=lambda x: x[1])
    mined = final_results[:MAX_MINED_OPPONENTS_PER_EVENT]

    if mined:
        print(f"  Mined exploiters: {[(u, f'{wr:.2%}') for u, wr in mined]}")

    return mined, total_candidates, candidates_after_filter


def sample_opponent_weighted(opponent_pool: deque, opponent_pool_updates: List[int],
                              per_opponent_win_rates: Dict[str, float]) -> nn.Module:
    """
    Sample an opponent using difficulty-weighted distribution.

    With probability UNIFORM_SAMPLING_FRACTION: sample uniformly.
    Otherwise: sample proportional to (1 - win_rate) to favor harder opponents.

    Args:
        opponent_pool: Deque of opponent models
        opponent_pool_updates: List of update numbers for each opponent
        per_opponent_win_rates: Dict mapping update number (str) to win rate

    Returns:
        Sampled opponent model
    """
    pool_list = list(opponent_pool)

    if random.random() < UNIFORM_SAMPLING_FRACTION:
        # Uniform sampling
        return random.choice(pool_list)

    # Difficulty-weighted sampling: weight = 1 - win_rate
    weights = []
    for update_num in opponent_pool_updates:
        key = str(update_num)
        win_rate = per_opponent_win_rates.get(key, DEFAULT_WIN_RATE)
        # Higher weight for lower win rate (harder opponents)
        weight = 1.0 - win_rate
        # Ensure minimum weight to avoid zero probability
        weight = max(weight, 0.01)
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        return random.choice(pool_list)

    probs = [w / total_weight for w in weights]

    # Sample according to weights
    idx = random.choices(range(len(pool_list)), weights=probs, k=1)[0]
    return pool_list[idx]


# ============================================================================
# Training State Management
# ============================================================================

def save_training_state(output_dir: str, update: int, opponent_pool_updates: List[int],
                        win_miss_ema: float = 1.0,
                        block_miss_ema: float = 1.0,
                        per_opponent_win_rates: Dict[str, float] = None,
                        scan_event_counter: int = 0,
                        evals_since_last_scan: int = 0) -> None:
    """Save training state to JSON for resume capability."""
    state = {
        'current_update': update,
        'opponent_pool_updates': opponent_pool_updates,
        'total_updates': TOTAL_UPDATES,
        'win_miss_ema': win_miss_ema,
        'block_miss_ema': block_miss_ema,
        'per_opponent_win_rates': per_opponent_win_rates if per_opponent_win_rates is not None else {},
        'scan_event_counter': scan_event_counter,
        'evals_since_last_scan': evals_since_last_scan
    }

    training_state_file = os.path.join(output_dir, TRAINING_STATE_FILE)
    temp_file = training_state_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(temp_file, training_state_file)


def load_training_state(output_dir: str, device: torch.device) -> Optional[Tuple[nn.Module, torch.optim.Optimizer,
                                                                  torch.optim.lr_scheduler.LambdaLR,
                                                                  deque, int, int, float, float,
                                                                  Dict[str, float], int, int]]:
    """
    Load training state from checkpoint to resume training.

    Returns:
        Tuple of (model, optimizer, scheduler, opponent_pool, start_update, next_eval_update,
                  win_miss_ema, block_miss_ema, per_opponent_win_rates, scan_event_counter,
                  evals_since_last_scan) or None if loading fails
    """
    training_state_file = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(training_state_file):
        print(f"No training state file found ({training_state_file})")
        return None

    print(f"Found training state file: {training_state_file}")

    try:
        with open(training_state_file, 'r') as f:
            state = json.load(f)
    except Exception as e:
        print(f"Error loading training state JSON: {e}")
        return None

    current_update = state['current_update']
    opponent_pool_updates = state['opponent_pool_updates']

    win_miss_ema = state.get('win_miss_ema', 1.0)
    block_miss_ema = state.get('block_miss_ema', 1.0)
    per_opponent_win_rates = state.get('per_opponent_win_rates', {})
    scan_event_counter = state.get('scan_event_counter', 0)
    evals_since_last_scan = state.get('evals_since_last_scan', 0)

    print(f"Resuming from update {current_update}")
    print(f"Opponent pool has {len(opponent_pool_updates)} models: {opponent_pool_updates}")
    print(f"Miss rate EMAs: win={win_miss_ema:.3f}, block={block_miss_ema:.3f}")
    print(f"Scan state: event_counter={scan_event_counter}, evals_since_last_scan={evals_since_last_scan}")
    if per_opponent_win_rates:
        print(f"Per-opponent win rates: {len(per_opponent_win_rates)} entries")

    checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{current_update}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return None

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    zero_center_taps(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def lr_lambda(epoch):
        decayed_lr = LEARNING_RATE * (LR_DECAY ** epoch)
        return max(decayed_lr, MIN_LR) / LEARNING_RATE

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    opponent_pool = deque()
    for pool_update in opponent_pool_updates:
        pool_checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{pool_update}.pt")
        if not os.path.exists(pool_checkpoint_path):
            print(f"Warning: Opponent pool checkpoint not found: {pool_checkpoint_path}")
            print(f"Skipping this opponent")
            continue

        try:
            pool_checkpoint = torch.load(pool_checkpoint_path, map_location=device, weights_only=False)
            pool_model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
            pool_model.load_state_dict(pool_checkpoint['model_state_dict'])
            pool_model.eval()
            opponent_pool.append(pool_model)
            print(f"Loaded opponent from update {pool_update}")
        except Exception as e:
            print(f"Error loading opponent pool checkpoint {pool_checkpoint_path}: {e}")
            continue

    if len(opponent_pool) == 0:
        print("Error: No opponent models loaded from pool")
        return None

    print(f"Successfully loaded {len(opponent_pool)} opponents")

    next_eval_update = current_update + get_eval_interval(current_update)

    print(f"Next evaluation scheduled at update {next_eval_update}")
    print(f"Resume training starting from update {current_update + 1}")
    print()

    return (model, optimizer, scheduler, opponent_pool, current_update - 1, next_eval_update,
            win_miss_ema, block_miss_ema, per_opponent_win_rates, scan_event_counter, evals_since_last_scan)


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training loop."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Gomoku policy network with self-play', add_help=False)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CSV logger
    csv_logger = CSVLogger(output_dir)
    print(f"Output directory: {output_dir}")
    print()

    effective_chunk_size = min(EPISODES_CHUNK_SIZE, EPISODES_PER_UPDATE)
    if EPISODES_PER_UPDATE < EPISODES_CHUNK_SIZE:
        print(f"NOTE: EPISODES_PER_UPDATE ({EPISODES_PER_UPDATE}) < EPISODES_CHUNK_SIZE ({EPISODES_CHUNK_SIZE})")
        print(f"  Using chunk size = {effective_chunk_size} (no gradient accumulation)")
        print()

    print(f"Using device: {DEVICE}")
    print(f"Model architecture:")
    print(f"  Stem (dilated design):")
    print(f"    - 3x3: {STEM_3X3_CHANNELS}ch")
    print(f"    - 5x5 sparse (d1+d2): {STEM_SPARSE_5X5_CHANNELS}ch, 5x5 dense: {STEM_DENSE_5X5_CHANNELS}ch")
    print(f"    - 7x7 sparse (d1+d2+d3): {STEM_SPARSE_7X7_CHANNELS}ch, 7x7 dense: {STEM_DENSE_7X7_CHANNELS}ch")
    print(f"    - Total: {WIDTH} channels (center taps zeroed for d>1)")
    print(f"  Residual blocks: {N_BLOCKS} x {WIDTH} channels")
    print(f"    - Dilation schedule (conv2): {TRUNK_DILATION2_SCHEDULE}")
    print(f"    - SE schedule: {SE_SCHEDULE} ({sum(SE_SCHEDULE)} blocks with SE)")
    print(f"  Policy head: {WIDTH} -> {POLICY_HEAD_D} (+SiLU) -> 225")
    print(f"  Value head: {WIDTH} -> {VALUE_HEAD_CHANNELS} (+SiLU) -> fc{VALUE_HEAD_HIDDEN} -> 1")
    num_accumulation_steps = (EPISODES_PER_UPDATE + effective_chunk_size - 1) // effective_chunk_size

    print(f"Training configuration:")
    print(f"  Learning rate: {LEARNING_RATE} (decay: {LR_DECAY}, min: {MIN_LR})")
    print(f"  Exploration (hybrid): Temperature={TEMPERATURE_TRAIN} (behavior) + Entropy bonus (gradient)")
    print(f"    Entropy coefficient: {ENTROPY_COEFF_START} -> {ENTROPY_COEFF_END} (sigmoid: mid={ENTROPY_DECAY_MIDPOINT_PERCENTAGE:.0%}, steep={ENTROPY_DECAY_STEEPNESS:.0%})")
    print(f"    Effective strength: {ENTROPY_COEFF_START/TEMPERATURE_TRAIN:.2e} -> {ENTROPY_COEFF_END/TEMPERATURE_TRAIN:.2e} (scaled by 1/T)")
    print(f"  Episodes per update: {EPISODES_PER_UPDATE} (chunks: {effective_chunk_size} x {num_accumulation_steps} accumulation steps)")
    print(f"  Batch inference size (self-play): {BATCH_INFERENCE_SIZE}")
    print(f"  Batch size (training): {TRAIN_BATCH_SIZE}")
    print(f"  Data augmentation: 8-fold symmetry (rot + flip)")
    print(f"  Value head: ENABLED (weight: {VALUE_LOSS_COEFF}, targets: TD(0))")
    print(f"  GAE: lambda={GAE_LAMBDA} ({'TD(0)' if GAE_LAMBDA == 0 else 'MC' if GAE_LAMBDA == 1 else 'blend'})")
    print(f"  Phased training:")
    print(f"    - Phase 1 (0-{VALUE_BASELINE_START-1}): Raw returns + tactical bonuses")
    print(f"    - Phase 2 ({VALUE_BASELINE_START}+): Value baseline + tactical bonuses (always)")
    print(f"    - Tactical bonuses prevent terminal state learning collapse")
    print(f"  Imitation learning: {IMITATION_WEIGHT} (learn from opponent's winning moves, starts at update {IMITATION_START_UPDATE})")
    print(f"  Opening seeding: {SEED_PROBABILITY:.0%} of games start from Renju opening ({len(RENJU_OPENING_SEQUENCES)} patterns)")
    print(f"  Opponent pool size: {OPPONENT_POOL_SIZE}")
    print(f"  Eval interval: {EVAL_INTERVAL_EARLY} (0-512) -> {EVAL_INTERVAL_MID} (512-8192) -> {EVAL_INTERVAL_LATE} (8192+)")
    print(f"  Pool eviction: evict-easiest (by current win rate)")
    print(f"  Opponent sampling: {UNIFORM_SAMPLING_FRACTION:.0%} uniform + {1-UNIFORM_SAMPLING_FRACTION:.0%} difficulty-weighted")
    print(f"  Historical exploiter scanning:")
    print(f"    - Starts at update {SCAN_START_UPDATE}, every {SCAN_PERIOD} evals")
    print(f"    - {NUM_SCAN_BUCKETS} buckets for round-robin coverage")
    print(f"    - Quick screen: {QUICK_SCREEN_ROUNDS} rounds, keep top {TOP_K_QUICK_SCREEN}")
    print(f"    - Final screen: {FINAL_SCREEN_ROUNDS} rounds, add up to {MAX_MINED_OPPONENTS_PER_EVENT} exploiters")
    print(f"  Total updates: {TOTAL_UPDATES}")
    print()

    # Try to resume from existing training state
    print("=" * 60)
    print("Checking for existing training state...")
    print("=" * 60)
    resume_result = load_training_state(output_dir, DEVICE)

    if resume_result is not None:
        (current_policy, optimizer, scheduler, opponent_pool, start_update, next_eval_update,
         win_miss_ema, block_miss_ema, per_opponent_win_rates, scan_event_counter,
         evals_since_last_scan) = resume_result
        opponent_pool_updates = []
        print("=" * 60)
        print("Successfully resumed training!")
        print("=" * 60)
        print()
    else:
        print("Starting fresh training (no existing state found)")
        print("=" * 60)
        print()

        start_update = -1

        current_policy = GomokuPolicyNet(n_blocks=N_BLOCKS).to(DEVICE)
        current_policy.train()
        zero_center_taps(current_policy)

        optimizer = torch.optim.AdamW(
            current_policy.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            fused=True
        )

        def lr_lambda(epoch):
            decayed_lr = LEARNING_RATE * (LR_DECAY ** epoch)
            return max(decayed_lr, MIN_LR) / LEARNING_RATE

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print("Initializing opponent pool with random policies...")
        opponent_pool = deque()
        opponent_pool_updates = []
        for i in range(OPPONENT_POOL_SIZE):
            opponent_pool.append(create_random_policy(DEVICE))
            opponent_pool[-1].eval()
            opponent_pool_updates.append(0)
        print(f"Opponent pool initialized with {len(opponent_pool)} models")
        print()

        win_miss_ema = 1.0
        block_miss_ema = 1.0
        next_eval_update = get_eval_interval(0)

        # Initialize new state variables for historical exploiter scanning
        per_opponent_win_rates = {}
        scan_event_counter = 0
        evals_since_last_scan = 0

    # Metrics tracking
    metric_buffer = {
        'loss': [], 'win_rate': [], 'win_rate_as_black': [], 'win_rate_as_white': [],
        'wins': [], 'losses': [], 'draws': [], 'entropy': [], 'value_loss': [],
        'raw_value_mse': [], 'avg_length': [], 'time': [], 'selfplay_time': [],
        'train_time': [], 'tactics_wins': [], 'tactics_blocks': [],
        'tactics_synthetic_wins_eq': [], 'tactics_synthetic_wins_missed': [],
        'tactics_synthetic_blocks': [], 'imitation_black': [], 'imitation_white': [],
        'win_miss_ema': [], 'block_miss_ema': [], 'win_boost': [], 'block_boost': [],
        'cler_attempted': [], 'cler_candidates': [], 'cler_added': [],
        'cler_winrate_sum': [], 'cler_entropy_sum': []
    }

    training_start_time = time.time()

    if resume_result is not None:
        try:
            with open(TRAINING_STATE_FILE, 'r') as f:
                state = json.load(f)
            opponent_pool_updates = state['opponent_pool_updates']
        except:
            opponent_pool_updates = [start_update] * len(opponent_pool)

    # Training loop
    for update in range(start_update + 1, TOTAL_UPDATES):
        t_start = time.time()

        pairs = []
        current_is_black = []
        opening_ids = []
        num_openings = len(RENJU_OPENING_SEQUENCES)
        for _ in range(EPISODES_PER_UPDATE):
            # Use difficulty-weighted sampling instead of uniform
            opponent = sample_opponent_weighted(opponent_pool, opponent_pool_updates, per_opponent_win_rates)
            if random.random() < 0.5:
                pairs.append((current_policy, opponent))
                current_is_black.append(True)
            else:
                pairs.append((opponent, current_policy))
                current_is_black.append(False)

            # Opening seeding: with probability SEED_PROBABILITY, start from a Renju opening
            if random.random() < SEED_PROBABILITY:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        t0 = time.time()
        trajectories = play_episodes_batched(
            pairs, current_is_black, TEMPERATURE_TRAIN, DEVICE,
            batch_size=BATCH_INFERENCE_SIZE,
            select_action_batch_fn=select_action_batch,
            opening_ids=opening_ids
        )
        t_selfplay = time.time() - t0

        stats = compute_outcome_stats(trajectories, current_is_black)

        # Nonlinear boost decay: 1 - hit_rate^2 (slower decay than linear)
        win_hit_rate_ema = 1.0 - win_miss_ema
        win_boost = WIN_MIN_BOOST + (1.0 - win_hit_rate_ema ** 2) * (WIN_MAX_BOOST - WIN_MIN_BOOST)

        block_hit_rate_ema = 1.0 - block_miss_ema
        block_boost = BLOCK_MIN_BOOST + (1.0 - block_hit_rate_ema ** 2) * (BLOCK_MAX_BOOST - BLOCK_MIN_BOOST)

        # Generate CLER samples from lost games
        cler_samples, cler_metrics = generate_cler_samples(trajectories, current_is_black, current_policy, DEVICE, update)

        t0 = time.time()
        loss, mean_return, mean_entropy, value_loss, raw_value_mse, num_wins, num_blocks, num_synthetic_wins_eq, num_synthetic_wins_missed, num_synthetic_blocks, num_imitation_black, num_imitation_white, win_opp, win_miss, block_opp, block_miss, num_cler, probe_metrics = train_on_batch(
            current_policy, trajectories, optimizer, DEVICE, chunk_size=effective_chunk_size, update=update,
            win_boost=win_boost, block_boost=block_boost, cler_samples=cler_samples
        )
        t_train = time.time() - t0

        # Log gradient probe if metrics collected
        if probe_metrics is not None:
            csv_logger.log_gradient_probe(update + 1, probe_metrics)
            print(f"  [Probe] Overall cos_sim={probe_metrics['overall_cos_sim']:+.3f} | "
                  f"Stem={probe_metrics['stem_cos_sim']:+.3f}")

        this_win_miss_rate = win_miss / win_opp if win_opp > 0 else 0.0
        this_block_miss_rate = block_miss / block_opp if block_opp > 0 else 0.0
        ema_alpha = 1.0 / MISS_RATE_EMA_WINDOW
        win_miss_ema = ema_alpha * this_win_miss_rate + (1.0 - ema_alpha) * win_miss_ema
        block_miss_ema = ema_alpha * this_block_miss_rate + (1.0 - ema_alpha) * block_miss_ema

        t_total = time.time() - t_start

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        metric_buffer['loss'].append(loss)
        metric_buffer['win_rate'].append(stats['win_rate'])
        metric_buffer['win_rate_as_black'].append(stats['win_rate_as_black'])
        metric_buffer['win_rate_as_white'].append(stats['win_rate_as_white'])
        metric_buffer['wins'].append(stats['wins'])
        metric_buffer['losses'].append(stats['losses'])
        metric_buffer['draws'].append(stats['draws'])
        metric_buffer['entropy'].append(mean_entropy)
        metric_buffer['value_loss'].append(value_loss)
        metric_buffer['raw_value_mse'].append(raw_value_mse)
        metric_buffer['avg_length'].append(stats['avg_length'])
        metric_buffer['time'].append(t_total)
        metric_buffer['selfplay_time'].append(t_selfplay)
        metric_buffer['train_time'].append(t_train)
        metric_buffer['tactics_wins'].append(num_wins)
        metric_buffer['tactics_blocks'].append(num_blocks)
        metric_buffer['tactics_synthetic_wins_eq'].append(num_synthetic_wins_eq)
        metric_buffer['tactics_synthetic_wins_missed'].append(num_synthetic_wins_missed)
        metric_buffer['tactics_synthetic_blocks'].append(num_synthetic_blocks)
        metric_buffer['imitation_black'].append(num_imitation_black)
        metric_buffer['imitation_white'].append(num_imitation_white)
        metric_buffer['win_miss_ema'].append(win_miss_ema)
        metric_buffer['block_miss_ema'].append(block_miss_ema)
        metric_buffer['win_boost'].append(win_boost)
        metric_buffer['block_boost'].append(block_boost)
        metric_buffer['cler_attempted'].append(cler_metrics['cf_attempted_episodes'])
        metric_buffer['cler_candidates'].append(cler_metrics['cf_steps_candidates_total'])
        metric_buffer['cler_added'].append(cler_metrics['cf_added_samples'])
        metric_buffer['cler_winrate_sum'].append(cler_metrics['cf_best_winrate_sum'])
        metric_buffer['cler_entropy_sum'].append(cler_metrics['cf_entropy_selected_sum'])

        if (update + 1) % PRINT_INTERVAL == 0:
            avg_loss = np.mean(metric_buffer['loss'])
            avg_win_rate = np.mean(metric_buffer['win_rate'])
            avg_win_rate_black = np.mean(metric_buffer['win_rate_as_black'])
            avg_win_rate_white = np.mean(metric_buffer['win_rate_as_white'])
            total_wins = sum(metric_buffer['wins'])
            total_losses = sum(metric_buffer['losses'])
            total_draws = sum(metric_buffer['draws'])
            avg_entropy = np.mean(metric_buffer['entropy'])
            avg_value_loss = np.mean(metric_buffer['value_loss'])
            avg_raw_value_mse = np.mean(metric_buffer['raw_value_mse'])
            avg_length = np.mean(metric_buffer['avg_length'])
            avg_time = np.mean(metric_buffer['time'])
            avg_selfplay_time = np.mean(metric_buffer['selfplay_time'])
            avg_train_time = np.mean(metric_buffer['train_time'])
            total_wins_found = sum(metric_buffer['tactics_wins'])
            total_blocks_found = sum(metric_buffer['tactics_blocks'])
            total_synthetic_wins_eq = sum(metric_buffer['tactics_synthetic_wins_eq'])
            total_synthetic_wins_missed = sum(metric_buffer['tactics_synthetic_wins_missed'])
            total_synthetic_blocks = sum(metric_buffer['tactics_synthetic_blocks'])
            total_imitation_black = sum(metric_buffer['imitation_black'])
            total_imitation_white = sum(metric_buffer['imitation_white'])
            total_imitation = total_imitation_black + total_imitation_white

            latest_win_miss_ema = metric_buffer['win_miss_ema'][-1] if metric_buffer['win_miss_ema'] else 0.0
            latest_block_miss_ema = metric_buffer['block_miss_ema'][-1] if metric_buffer['block_miss_ema'] else 0.0
            latest_win_boost = metric_buffer['win_boost'][-1] if metric_buffer['win_boost'] else 0.0
            latest_block_boost = metric_buffer['block_boost'][-1] if metric_buffer['block_boost'] else 0.0

            # CLER metrics aggregation
            total_cler_attempted = sum(metric_buffer['cler_attempted'])
            total_cler_candidates = sum(metric_buffer['cler_candidates'])
            total_cler_added = sum(metric_buffer['cler_added'])
            total_cler_winrate_sum = sum(metric_buffer['cler_winrate_sum'])
            total_cler_entropy_sum = sum(metric_buffer['cler_entropy_sum'])
            avg_cler_candidates = total_cler_candidates / max(total_cler_attempted, 1)
            avg_cler_winrate = total_cler_winrate_sum / max(total_cler_added, 1)
            avg_cler_entropy = total_cler_entropy_sum / max(total_cler_added, 1)

            elapsed_time = time.time() - training_start_time
            updates_done = update + 1
            updates_remaining = TOTAL_UPDATES - updates_done
            time_per_update = elapsed_time / updates_done
            eta_seconds = updates_remaining * time_per_update

            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}h{minutes:02d}m"
                elif minutes > 0:
                    return f"{minutes}m{secs:02d}s"
                else:
                    return f"{secs}s"

            elapsed_str = format_time(elapsed_time)
            eta_str = format_time(eta_seconds)

            # Simplified print output
            print(f"Update {update + 1:5d}/{TOTAL_UPDATES} | "
                  f"WinRate: {avg_win_rate:.0%}(W{avg_win_rate_white:.0%}-B{avg_win_rate_black:.0%}) | "
                  f"AvgLen: {avg_length:.1f} | "
                  f"Entropy: {avg_entropy:.3f} | "
                  f"MissEMA: W={latest_win_miss_ema:.0%} B={latest_block_miss_ema:.0%} | "
                  f"{avg_time:.2f}s/upd | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")

            # Log to CSV
            csv_logger.log_training_update(update + 1, {
                'loss': avg_loss,
                'win_rate': avg_win_rate,
                'win_rate_black': avg_win_rate_black,
                'win_rate_white': avg_win_rate_white,
                'avg_game_length': avg_length,
                'entropy': avg_entropy,
                'value_loss': avg_value_loss,
                'raw_value_mse': avg_raw_value_mse,
                'tactics_wins': total_wins_found,
                'tactics_blocks': total_blocks_found,
                'tactics_synth_wins_eq': total_synthetic_wins_eq,
                'tactics_synth_wins_missed': total_synthetic_wins_missed,
                'tactics_synth_blocks': total_synthetic_blocks,
                'imitation_black': total_imitation_black,
                'imitation_white': total_imitation_white,
                'win_miss_ema': latest_win_miss_ema,
                'block_miss_ema': latest_block_miss_ema,
                'win_boost': latest_win_boost,
                'block_boost': latest_block_boost,
                'cf_attempted': total_cler_attempted,
                'cf_candidates_avg': avg_cler_candidates,
                'cf_added': total_cler_added,
                'cf_winrate_avg': avg_cler_winrate,
                'cf_entropy_avg': avg_cler_entropy,
                'time_total': avg_time,
                'time_selfplay': avg_selfplay_time,
                'time_train': avg_train_time,
                'learning_rate': current_lr
            })

            for key in metric_buffer:
                metric_buffer[key] = []

        if update + 1 >= next_eval_update:
            print(f"\n--- Eval @ {update + 1} ---")

            # Step 1: Save current checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{update + 1}.pt")
            torch.save({
                'update': update + 1,
                'model_state_dict': current_policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"  Saved: {os.path.basename(checkpoint_path)}")

            # Step 2: Run evaluation with per-opponent stats
            eval_start_time = time.time()
            win_rate, per_opp_stats = evaluate_policy(
                current_policy, opponent_pool, DEVICE,
                opponent_pool_updates=opponent_pool_updates
            )
            eval_time = time.time() - eval_start_time

            # Update per_opponent_win_rates with latest evaluation data
            for opp_key, stats in per_opp_stats.items():
                per_opponent_win_rates[opp_key] = stats['win_rate']

            # Show per-opponent breakdown (hardest and easiest)
            sorted_opps = sorted(per_opp_stats.items(), key=lambda x: x[1]['win_rate'])
            hardest_opp_id = int(sorted_opps[0][0]) if sorted_opps else -1
            hardest_win_rate = sorted_opps[0][1]['win_rate'] if sorted_opps else 0.0
            easiest_opp_id = int(sorted_opps[-1][0]) if sorted_opps else -1
            easiest_win_rate = sorted_opps[-1][1]['win_rate'] if sorted_opps else 0.0

            # Simplified print
            total_games = EVAL_ROUNDS * 2 * len(opponent_pool)
            print(f"  WinRate: {win_rate:.1%} ({total_games} games, {eval_time:.1f}s) | "
                  f"Hard: {hardest_opp_id}:{hardest_win_rate:.0%} Easy: {easiest_opp_id}:{easiest_win_rate:.0%}")

            # Step 3: Conditionally add current snapshot to pool (using evict-easiest)
            checkpoint_added = False
            evicted_opponent_id = -1
            if win_rate >= WIN_RATE_THRESHOLD:
                checkpoint_added = True
                evicted = add_opponent_to_pool(
                    opponent_pool, opponent_pool_updates, current_policy, update + 1,
                    per_opponent_win_rates, DEVICE
                )
                if evicted is not None:
                    evicted_opponent_id = evicted
                    print(f"  Pool: +current -{evicted}")
                    # Remove evicted opponent from win rate tracking
                    per_opponent_win_rates.pop(str(evicted), None)
                else:
                    print(f"  Pool: +current")
            else:
                print(f"  Pool: no change (WR {win_rate:.1%} < {WIN_RATE_THRESHOLD:.1%})")

            # Log eval summary to CSV
            csv_logger.log_eval_summary(update + 1, {
                'overall_win_rate': win_rate,
                'total_games': total_games,
                'eval_time': eval_time,
                'hardest_opponent_id': hardest_opp_id,
                'hardest_win_rate': hardest_win_rate,
                'easiest_opponent_id': easiest_opp_id,
                'easiest_win_rate': easiest_win_rate,
                'pool_size': len(opponent_pool),
                'checkpoint_added': checkpoint_added,
                'evicted_opponent_id': evicted_opponent_id
            })

            # Log per-opponent details to CSV
            for opp_key, stats in per_opp_stats.items():
                opp_id = int(opp_key)
                losses = stats['games'] - stats['wins'] - stats['draws']
                csv_logger.log_eval_opponent_details(update + 1, opp_id, {
                    'wins': stats['wins'],
                    'losses': losses,
                    'draws': stats['draws'],
                    'games': stats['games'],
                    'win_rate': stats['win_rate']
                })

            # Step 4: Check scan trigger and run historical exploiter scan
            evals_since_last_scan += 1

            # Scan triggers: after SCAN_START_UPDATE, every SCAN_PERIOD evaluations
            should_scan = (
                (update + 1) >= SCAN_START_UPDATE and
                evals_since_last_scan >= SCAN_PERIOD
            )

            if should_scan:
                print(f"\n--- Mining @ {update + 1} | Bucket {scan_event_counter % NUM_SCAN_BUCKETS} ---")
                scan_start_time = time.time()

                mined_exploiters, total_candidates, candidates_after_filter = scan_historical_exploiters(
                    output_dir, current_policy, opponent_pool_updates, scan_event_counter, DEVICE
                )

                # Add mined exploiters to pool and log to CSV
                bucket_id = scan_event_counter % NUM_SCAN_BUCKETS
                for rank, (mined_update, mined_win_rate) in enumerate(mined_exploiters, start=1):
                    mined_checkpoint = os.path.join(output_dir, f"checkpoint_update_{mined_update}.pt")
                    mined_model = load_checkpoint_model(mined_checkpoint, DEVICE)
                    if mined_model is not None:
                        evicted = add_opponent_to_pool(
                            opponent_pool, opponent_pool_updates, mined_model, mined_update,
                            per_opponent_win_rates, DEVICE
                        )
                        # Set the mined opponent's win rate based on scanning result
                        per_opponent_win_rates[str(mined_update)] = mined_win_rate

                        evicted_id = evicted if evicted is not None else -1
                        print(f"  Found: {mined_update}:{mined_win_rate:.0%} | +{mined_update} -{evicted_id if evicted_id != -1 else 'none'}")

                        # Log to CSV
                        csv_logger.log_mining_event(
                            scan_update=update + 1,
                            scan_event_num=scan_event_counter,
                            bucket_id=bucket_id,
                            total_candidates=total_candidates,
                            candidates_after_filter=candidates_after_filter,
                            mined_opponent_id=mined_update,
                            mined_win_rate=mined_win_rate,
                            mined_rank=rank,
                            added_to_pool=True,
                            evicted_opponent_id=evicted_id,
                            scan_time=time.time() - scan_start_time
                        )

                        if evicted is not None:
                            per_opponent_win_rates.pop(str(evicted), None)
                        del mined_model

                scan_time = time.time() - scan_start_time
                if not mined_exploiters:
                    print(f"  No exploiters found ({scan_time:.1f}s)")

                # Update scan counters
                scan_event_counter += 1
                evals_since_last_scan = 0

            # Step 5: Save training state with all new fields
            save_training_state(
                output_dir, update + 1, opponent_pool_updates, win_miss_ema, block_miss_ema,
                per_opponent_win_rates, scan_event_counter, evals_since_last_scan
            )

            eval_interval = get_eval_interval(update + 1)
            next_eval_update = (update + 1) + eval_interval
            print(f"  Next eval: {next_eval_update}")

            torch.cuda.empty_cache()
            print()

    final_path = os.path.join(output_dir, "final_policy.pt")
    torch.save(current_policy.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    save_training_state(
        output_dir, TOTAL_UPDATES, opponent_pool_updates, win_miss_ema, block_miss_ema,
        per_opponent_win_rates, scan_event_counter, evals_since_last_scan
    )
    print(f"Final training state saved")


if __name__ == "__main__":
    main()
