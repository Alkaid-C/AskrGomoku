"""
Data Augmentation and Trajectory Analysis.

Logic for transforming raw gameplay trajectories into training targets:
- GAE Computation
- CLER (Counterfactual Low-Entropy Rescue) sample generation
- Tactical enhancements (synthetic win-in-1/blocking)
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple, Dict
from gomoku_rules import (
    GameState, Player, GomokuBoard,
    find_all_win_in_1, find_blocking_moves, get_local_candidate_moves
)
from gameplay_loop import (
    Trajectory, play_cler_rollouts_batched,
    select_action_batch_eval, obs_batch_to_tensor, BATCH_INFERENCE_SIZE
)

# ============================================================================ 
# Configuration
# ============================================================================ 

# Advantage Estimation
GAE_LAMBDA = 0.95
EPISODE_WEIGHT_ALPHA = 0.5  # 0 => per-step weighting, 1 => per-episode equal mass

# Imitation Learning
IMITATION_WEIGHT = 0.6
IMITATION_START_UPDATE = 128 * 6

# CLER
CF_START_UPDATE = 128 * 6
CF_TRIGGER_PROB = 0.25
CF_ADVANTAGE = 1.0
CF_MIN_STEPS_TO_END = 5
CF_ENTROPY_TH = 0.5
CF_RADIUS = 2
CF_NUM_ACTIONS = 8
CF_NUM_ROLLOUTS = 8
CF_WIN_MARGIN = 0.375
CF_MAX_SAMPLES_PER_UPDATE = 999
CF_ROLLOUT_TEMP = 1.0

# Synthetic Tactics
SYNTHETIC_WIN_BOOST = 2.0
SYNTHETIC_BLOCKING_BOOST = 1.5
MAX_SYNTHETIC_WINS = 256
MAX_SYNTHETIC_BLOCKS = 256


# ============================================================================ 
# Return & GAE Computation
# ============================================================================ 

def compute_returns(traj: Trajectory) -> List[float]:
    """Compute per-step returns (z_t) from trajectory."""
    returns = []
    outcome = traj.outcome

    if outcome == GameState.DRAW:
        winner = None
    elif outcome == GameState.BLACK_WIN:
        winner = Player.BLACK
    elif outcome == GameState.WHITE_WIN:
        winner = Player.WHITE
    else:
        raise ValueError(f"Invalid outcome: {outcome}")

    for player_t in traj.players:
        if winner is None:
            z_t = 0.0
        elif winner == player_t:
            z_t = 1.0
        else:
            z_t = -1.0
        returns.append(z_t)
    return returns


def compute_gae_for_trajectories(model: nn.Module, trajectories: List[Trajectory],
                                  device: torch.device) -> List[np.ndarray]:
    """Compute GAE advantages for all trajectories."""
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
            gae = delta - GAE_LAMBDA * gae
            gae_advantages[t] = gae

        all_gae.append(gae_advantages)

    return all_gae


def compute_outcome_stats(trajectories: List[Trajectory], current_is_black: List[bool]) -> dict:
    """Compute basic stats about game outcomes."""
    current_wins = 0
    current_losses = 0
    draws = 0
    total_steps = []
    wins_as_black = 0
    wins_as_white = 0
    games_as_black = 0
    games_as_white = 0

    for traj, is_black in zip(trajectories, current_is_black):
        total_steps.append(len(traj.actions))
        if is_black:
            games_as_black += 1
        else:
            games_as_white += 1

        if traj.outcome == GameState.DRAW:
            draws += 1
        elif traj.outcome == GameState.BLACK_WIN:
            if is_black:
                current_wins += 1
                wins_as_black += 1
            else:
                current_losses += 1
        elif traj.outcome == GameState.WHITE_WIN:
            if not is_black:
                current_wins += 1
                wins_as_white += 1
            else:
                current_losses += 1

    total_games = len(trajectories)
    return {
        'wins': current_wins,
        'losses': current_losses,
        'draws': draws,
        'win_rate': current_wins / total_games if total_games > 0 else 0,
        'win_rate_as_black': wins_as_black / games_as_black if games_as_black > 0 else 0,
        'win_rate_as_white': wins_as_white / games_as_white if games_as_white > 0 else 0,
        'avg_length': np.mean(total_steps) if total_steps else 0,
    }


# ============================================================================ 
# CLER (Counterfactual Low-Entropy Rescue)
# ============================================================================ 

def generate_cler_samples(trajectories: List[Trajectory],
                          current_is_black: List[bool],
                          opponents: List[nn.Module],
                          current_policy: nn.Module,
                          device: torch.device,
                          update: int) -> Tuple[List[dict], dict]:
    """
    Generate CLER samples from lost games.
    """
    cler_samples = []
    metrics = {
        'cf_attempted_episodes': 0,
        'cf_steps_candidates_total': 0,
        'cf_added_samples': 0,
        'cf_best_winrate_sum': 0.0,
        'cf_orig_winrate_sum': 0.0,
        'cf_entropy_selected_sum': 0.0,
    }

    if update < CF_START_UPDATE:
        return cler_samples, metrics

    current_policy.eval()

    # Phase 1: Collect candidates
    candidate_info = []
    for traj_idx, (traj, is_black, opponent) in enumerate(zip(trajectories, current_is_black, opponents)):
        outcome = traj.outcome
        is_loss = (outcome == GameState.BLACK_WIN and not is_black) or \
                  (outcome == GameState.WHITE_WIN and is_black)
        if not is_loss:
            continue
        if random.random() > CF_TRIGGER_PROB:
            continue

        metrics['cf_attempted_episodes'] += 1
        if is_black:
            black_model, white_model = current_policy, opponent
        else:
            black_model, white_model = opponent, current_policy

        T = len(traj.observations)
        candidates = []
        for t in range(T):
            if not traj.is_current_policy[t]:
                continue
            steps_to_end = T - t - 1
            if steps_to_end < CF_MIN_STEPS_TO_END:
                continue
            entropy = traj.entropies[t]
            if entropy >= CF_ENTROPY_TH:
                continue

            # Skip tactical
            obs = traj.observations[t]
            mask = traj.legal_masks[t]
            if find_all_win_in_1(obs, mask) or find_blocking_moves(obs, mask) is not None:
                continue

            weight = CF_ENTROPY_TH - entropy
            candidates.append((t, weight, entropy))

        metrics['cf_steps_candidates_total'] += len(candidates)
        if not candidates:
            continue

        # Sample one step
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

        obs = traj.observations[t_star]
        mask = traj.legal_masks[t_star]
        player = traj.players[t_star]
        original_action = traj.actions[t_star]

        local_candidates = get_local_candidate_moves(obs, mask, radius=CF_RADIUS)
        alt_candidates = [a for a in local_candidates if a != original_action]
        if not alt_candidates:
            continue

        if len(alt_candidates) > CF_NUM_ACTIONS:
            alt_actions = random.sample(alt_candidates, CF_NUM_ACTIONS)
        else:
            alt_actions = alt_candidates

        candidate_info.append((
            traj_idx, t_star, obs, mask, player, original_action, alt_actions,
            black_model, white_model, selected_entropy
        ))
        if len(candidate_info) >= CF_MAX_SAMPLES_PER_UPDATE:
            break

    if not candidate_info:
        current_policy.train()
        return cler_samples, metrics

    # Phase 2: Rollouts
    rollout_configs = []
    rollout_metadata = []

    for cand_idx, (_, _, obs, _, player, original_action, alt_actions, black_model, white_model, _) in enumerate(candidate_info):
        for _ in range(CF_NUM_ROLLOUTS):
            rollout_configs.append((obs, player, original_action, black_model, white_model))
            rollout_metadata.append((cand_idx, True, original_action))
        for alt_action in alt_actions:
            for _ in range(CF_NUM_ROLLOUTS):
                rollout_configs.append((obs, player, alt_action, black_model, white_model))
                rollout_metadata.append((cand_idx, False, alt_action))

    rollout_results = play_cler_rollouts_batched(
        rollout_configs, CF_ROLLOUT_TEMP, device, batch_size=BATCH_INFERENCE_SIZE,
        select_action_batch_fn=select_action_batch_eval
    )

    # Phase 3: Aggregation
    candidate_results = {}
    for (cand_idx, is_original, action), won in zip(rollout_metadata, rollout_results):
        if cand_idx not in candidate_results:
            candidate_results[cand_idx] = {'original_wins': 0, 'original_count': 0, 'alts': {}}
        if is_original:
            candidate_results[cand_idx]['original_wins'] += int(won)
            candidate_results[cand_idx]['original_count'] += 1
        else:
            if action not in candidate_results[cand_idx]['alts']:
                candidate_results[cand_idx]['alts'][action] = {'wins': 0, 'count': 0}
            candidate_results[cand_idx]['alts'][action]['wins'] += int(won)
            candidate_results[cand_idx]['alts'][action]['count'] += 1

    # Phase 4: Selection
    for cand_idx, (traj_idx, t_star, obs, mask, player, original_action, alt_actions, _, _, selected_entropy) in enumerate(candidate_info):
        if cand_idx not in candidate_results:
            continue

        results = candidate_results[cand_idx]
        original_winrate = results['original_wins'] / max(results['original_count'], 1)
        best_action = None
        best_winrate = 0.0

        for action, stats in results['alts'].items():
            winrate = stats['wins'] / max(stats['count'], 1)
            if winrate > best_winrate:
                best_winrate = winrate
                best_action = action

        margin = best_winrate - original_winrate
        if margin >= CF_WIN_MARGIN and best_action is not None:
            traj = trajectories[traj_idx]
            current_steps = sum(1 for is_current in traj.is_current_policy if is_current)
            step_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0

            cler_samples.append({
                'obs': obs,
                'action': best_action,
                'mask': mask,
                'strength': margin * CF_ADVANTAGE,
                'weight': step_weight,
            })
            metrics['cf_added_samples'] += 1
            metrics['cf_best_winrate_sum'] += best_winrate
            metrics['cf_orig_winrate_sum'] += original_winrate
            metrics['cf_entropy_selected_sum'] += selected_entropy

    current_policy.train()
    return cler_samples, metrics
