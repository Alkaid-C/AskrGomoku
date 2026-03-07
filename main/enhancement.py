"""
Training Enhancements Module

Contains sample enhancement logic that modifies or generates training samples:
- Tactical search (win-in-1, blocking detection)
- 8-fold symmetry data augmentation (GPU accelerated)
- Off-policy rollout sample generation
- Imitation learning sample extraction

All enhancements are "sample generators/modifiers" with clean interfaces.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gomoku import GameState, Trajectory, get_local_candidate_moves, play_offpolicy_rollouts_batched, select_action_batch_eval

# ============================================================================
# Enhancement Constants
# ============================================================================

# --- Tactical Enhancements ---
WIN_MIN_BOOST = 0.0            # Minimum boost for win-in-1 (when miss rate is 0)
WIN_MAX_BOOST = 1.0            # Maximum boost for win-in-1 (when miss rate is 1)
BLOCK_MIN_BOOST = 0.0          # Minimum boost for blocking (when miss rate is 0)
BLOCK_MAX_BOOST = 0.75         # Maximum boost for blocking (when miss rate is 1)

SYNTHETIC_WIN_BOOST = 2.0      # Signal for missed win-in-1 (synthetic examples)
SYNTHETIC_BLOCKING_BOOST = 1.5 # Signal for missed blocks (synthetic examples)

# --- Imitation Learning ---
IMITATION_MAX_WEIGHT = 1.5      # Maximum weight for imitation learning (at 0% win rate)
IMITATION_MIN_WEIGHT = 0.0      # Minimum weight for imitation learning (at 100% win rate)
IMITATION_START_UPDATE = 128    # Update at which to enable imitation learning

# --- Off-Policy Rollout ---
OPR_START_UPDATE = 1024         # Update at which to enable off-policy rollout
OPR_TRIGGER_PROB = 0.25         # Probability of triggering off-policy rollout on a lost game
OPR_ADVANTAGE = 1.25            # Strength multiplier for off-policy rollout samples
OPR_MIN_STEPS_TO_END = 6        # Minimum steps from terminal to consider
OPR_ENTROPY_TH_MULTIPLIER = 0.5 # Entropy threshold multiplier (actual threshold = entropy_schedule * multiplier)
OPR_RADIUS = 1                  # Chebyshev distance for local candidate moves
OPR_NUM_ACTIONS = 8             # Number of alternative actions to evaluate
OPR_NUM_ROLLOUTS = 4            # Rollouts per alternative action
OPR_WIN_MARGIN = 0.5            # Minimum margin over original action's winrate
OPR_ROLLOUT_TEMP = 1.0          # Temperature for off-policy rollouts

# --- Sample Weighting ---
EPISODE_WEIGHT_ALPHA = 0.25     # 0 => per-step weighting, 1 => per-episode equal mass


# ============================================================================
# Stats Dataclasses
# ============================================================================

@dataclass
class TacticalStats:
    """Statistics from tactical enhancement."""
    wins_found: int = 0
    blocks_found: int = 0
    synthetic_wins_eq: int = 0
    synthetic_wins_missed: int = 0
    synthetic_blocks: int = 0
    win_opportunities: int = 0
    win_misses: int = 0
    block_opportunities: int = 0
    block_misses: int = 0


@dataclass
class TacticalBoostInfo:
    """Tactical boost information for applying to advantages after GAE computation."""
    sample_boosts: List[float]  # Boost to add to each sample's advantage (0.0 if no boost)
    synthetic_advantages: List[float]  # Advantages for synthetic samples added by tactical enhancement


@dataclass
class OffPolicyRolloutStats:
    """Statistics from off-policy rollout generation."""
    attempted_episodes: int = 0
    candidates_total: int = 0
    samples_added: int = 0
    best_winrate_sum: float = 0.0
    orig_winrate_sum: float = 0.0
    entropy_selected_sum: float = 0.0


# ============================================================================
# Tactical Search (Win-in-1 and Blocking Detection)
# ============================================================================

def get_local_candidate_positions(obs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """
    Get legal positions within Chebyshev distance 1 of any occupied cell.

    For Gomoku, a win-in-1 position must be adjacent (including diagonally) to
    at least one existing stone. Chebyshev distance handles all 8 directions:
    horizontal, vertical, diagonal, and anti-diagonal.

    Args:
        obs: Observation [3, 15, 15] where [0] is current player, [1] is opponent
        legal_mask: Legal moves mask [15, 15] numpy array

    Returns:
        Array of (row, col) positions that are legal AND adjacent to occupied cells
    """
    # Get all occupied positions (either player)
    occupied = (obs[0] == 1) | (obs[1] == 1)

    # Create neighbor mask using dilation (expand occupied region by 1 in all directions)
    neighbor_mask = np.zeros((15, 15), dtype=bool)

    # For each occupied cell, mark all 8 neighbors
    occupied_positions = np.argwhere(occupied)
    for r, c in occupied_positions:
        # Chebyshev distance 1: mark all positions where max(|dr|, |dc|) <= 1
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the occupied cell itself
                nr, nc = r + dr, c + dc
                if 0 <= nr < 15 and 0 <= nc < 15:
                    neighbor_mask[nr, nc] = True

    # Intersection: positions that are BOTH legal AND adjacent to occupied cells
    candidate_mask = neighbor_mask & (legal_mask == 1)
    candidate_positions = np.argwhere(candidate_mask)

    return candidate_positions


def is_winning_move(board_c0: np.ndarray, row: int, col: int) -> bool:
    """
    Check if placing current player's piece at (row, col) creates 5-in-a-row.

    Args:
        board_c0: Current player's pieces [15, 15] (0 or 1)
        row: Row index
        col: Column index

    Returns:
        True if the move wins, False otherwise
    """
    # Four directions: horizontal, vertical, diagonal, anti-diagonal
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        count = 1  # Count the placed stone itself

        # Count contiguous stones in forward direction
        r, c = row + dr, col + dc
        while 0 <= r < 15 and 0 <= c < 15 and board_c0[r, c] == 1:
            count += 1
            r += dr
            c += dc

        # Count contiguous stones in backward direction
        r, c = row - dr, col - dc
        while 0 <= r < 15 and 0 <= c < 15 and board_c0[r, c] == 1:
            count += 1
            r -= dr
            c -= dc

        if count >= 5:
            return True

    return False


def find_all_win_in_1(obs: np.ndarray, legal_mask: np.ndarray) -> List[int]:
    """
    Search for all win-in-1 moves for the current player.

    Args:
        obs: Observation [3, 15, 15] where [0] is current player, [1] is opponent, [2] is board mask
        legal_mask: Legal moves mask [15, 15] numpy array

    Returns:
        List of flat indices of all winning moves (empty list if none found)
    """
    board_c0 = obs[0]  # Current player's pieces
    winning_moves = []

    # OPTIMIZATION: Only check positions adjacent to existing stones (Chebyshev distance 1)
    # A win-in-1 position must be adjacent to at least one existing stone to complete 5-in-a-row
    candidate_positions = get_local_candidate_positions(obs, legal_mask)

    for pos in candidate_positions:
        row, col = pos[0], pos[1]
        if is_winning_move(board_c0, row, col):
            winning_moves.append(row * 15 + col)

    return winning_moves


def find_blocking_moves(obs: np.ndarray, legal_mask: np.ndarray) -> Optional[List[int]]:
    """
    Find moves that block opponent's win-in-1 threats.

    IMPORTANT: Returns None if there are multiple independent threats (dual threat / "dual of 4"),
    since only one threat can be blocked per move, making the position unwinnable.

    Args:
        obs: Observation [3, 15, 15] where [0] is current player, [1] is opponent, [2] is board mask
        legal_mask: Legal moves mask [15, 15] numpy array

    Returns:
        List of blocking move indices if opponent has exactly one win-in-1 threat, None otherwise
        (returns None if no threats, or if multiple unblockable threats exist)
    """
    board_opponent = obs[1]  # Opponent's pieces

    blocking_positions = []

    # OPTIMIZATION: Only check positions adjacent to existing stones (Chebyshev distance 1)
    # An opponent win-in-1 position must be adjacent to at least one existing stone
    candidate_positions = get_local_candidate_positions(obs, legal_mask)

    for pos in candidate_positions:
        row, col = pos[0], pos[1]
        # Would opponent win by playing here?
        if is_winning_move(board_opponent, row, col):
            blocking_positions.append(row * 15 + col)
            # Early exit if we find 2+ threats (dual threat = unblockable)
            if len(blocking_positions) >= 2:
                return None

    return blocking_positions if blocking_positions else None


# ============================================================================
# Data Augmentation (8-fold symmetry) - GPU Accelerated
# ============================================================================

def augment_batch_8fold(obs_batch: torch.Tensor, actions: torch.Tensor,
                        masks_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply all 8 symmetries to a batch on GPU simultaneously.

    Args:
        obs_batch: [B, 3, 15, 15] tensor on GPU
        actions: [B] flat action indices
        masks_batch: [B, 15, 15] bool tensor on GPU

    Returns:
        Tuple of (aug_obs [B*8, 3, 15, 15], aug_actions [B*8], aug_masks [B*8, 15, 15])
    """
    r, c = actions // 15, actions % 15

    # Reusable transforms
    obs_t = obs_batch.transpose(-2, -1)
    obs_r180 = obs_batch.flip(-2, -1)
    masks_t = masks_batch.transpose(-2, -1)
    masks_r180 = masks_batch.flip(-2, -1)

    # All 8 dihedral symmetries at once
    # Order: identity, rot90, rot180, rot270, flip-H, flip-V, transpose, anti-transpose
    all_obs = torch.cat([obs_batch, obs_t.flip(-1), obs_r180, obs_t.flip(-2),
                         obs_batch.flip(-1), obs_batch.flip(-2), obs_t, obs_r180.transpose(-2, -1)])
    all_masks = torch.cat([masks_batch, masks_t.flip(-1), masks_r180, masks_t.flip(-2),
                           masks_batch.flip(-1), masks_batch.flip(-2), masks_t, masks_r180.transpose(-2, -1)])

    # Action coordinate transforms: [8, B] → flatten to [8*B]
    new_rows = torch.stack([r, c, 14 - r, 14 - c, r, 14 - r, c, 14 - c])
    new_cols = torch.stack([c, 14 - r, 14 - c, r, 14 - c, c, r, 14 - r])
    all_actions = (new_rows * 15 + new_cols).reshape(-1)

    return all_obs, all_actions, all_masks


# ============================================================================
# Off-Policy Rollout
# ============================================================================

def generate_offpolicy_rollout_samples(trajectories: List[Trajectory],
                                       current_is_black: List[bool],
                                       opponents: List[nn.Module],
                                       current_policy: nn.Module,
                                       device: torch.device,
                                       update: int,
                                       entropy_schedule: float) -> Tuple[List[dict], OffPolicyRolloutStats]:
    """
    Generate off-policy rollout samples from lost games.

    For lost games, identifies steps where the policy was overconfident (low entropy)
    but far from terminal. Tries alternative moves via batched rollouts to find better options.
    Both original action and alternatives are evaluated with rollouts for fair comparison.

    Args:
        trajectories: List of game trajectories
        current_is_black: List indicating if current policy played as black
        opponents: List of opponent models (one per trajectory)
        current_policy: Current policy network
        device: Torch device
        update: Current update number
        entropy_schedule: Scheduled entropy value used to compute OPR threshold

    Returns:
        Tuple of (opr_samples, stats) where:
        - opr_samples: List of dicts with keys: obs, action, mask, strength, weight
        - stats: OffPolicyRolloutStats with logging metrics
    """
    opr_samples = []
    stats = OffPolicyRolloutStats()

    # Skip if before start update
    if update < OPR_START_UPDATE:
        return opr_samples, stats

    current_policy.eval()

    # Phase 1: Collect all candidate (trajectory, step) pairs from lost games
    candidate_info = []

    for traj_idx, (traj, is_black, opponent) in enumerate(zip(trajectories, current_is_black, opponents)):
        # Only process losses for current policy
        outcome = traj.outcome
        is_loss = (outcome == GameState.BLACK_WIN and not is_black) or \
                  (outcome == GameState.WHITE_WIN and is_black)

        if not is_loss:
            continue

        # Trigger with probability
        if random.random() > OPR_TRIGGER_PROB:
            continue

        stats.attempted_episodes += 1

        # Determine black/white model assignment
        if is_black:
            black_model, white_model = current_policy, opponent
        else:
            black_model, white_model = opponent, current_policy

        # Find candidate steps: low entropy, far from terminal, current policy's turn
        T = len(traj.observations)
        candidates = []

        # Compute adaptive entropy threshold
        entropy_threshold = entropy_schedule * OPR_ENTROPY_TH_MULTIPLIER

        for t in range(T):
            # Check if current policy's turn
            if not traj.is_current_policy[t]:
                continue

            # Check steps to end
            steps_to_end = T - t - 1
            if steps_to_end < OPR_MIN_STEPS_TO_END:
                continue

            # Check entropy threshold (adaptive based on target entropy)
            entropy = traj.entropies[t]
            if entropy >= entropy_threshold:
                continue

            # Skip if tactical position (already handled by tactical bonuses)
            obs = traj.observations[t]
            mask = traj.legal_masks[t]
            if find_all_win_in_1(obs, mask):
                continue
            if find_blocking_moves(obs, mask) is not None:
                continue

            # Weight: lower entropy = higher weight
            weight = entropy_threshold - entropy
            candidates.append((t, weight, entropy))

        stats.candidates_total += len(candidates)

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

        # Get state at t_star
        obs = traj.observations[t_star]
        mask = traj.legal_masks[t_star]
        player = traj.players[t_star]
        original_action = traj.actions[t_star]

        # Get local candidate moves
        local_candidates = get_local_candidate_moves(obs, mask, radius=OPR_RADIUS)

        # Remove original action from local candidates for alternatives
        alt_candidates = [a for a in local_candidates if a != original_action]

        if not alt_candidates:
            continue

        # Sample up to OPR_NUM_ACTIONS alternatives
        if len(alt_candidates) > OPR_NUM_ACTIONS:
            alt_actions = random.sample(alt_candidates, OPR_NUM_ACTIONS)
        else:
            alt_actions = alt_candidates

        candidate_info.append((
            traj_idx, t_star, obs, mask, player, original_action, alt_actions,
            black_model, white_model, selected_entropy
        ))

    if not candidate_info:
        current_policy.train()
        return opr_samples, stats

    # Phase 2: Build all rollout configurations
    rollout_configs = []
    rollout_metadata = []

    for cand_idx, (_, _, obs, _, player, original_action, alt_actions, black_model, white_model, _) in enumerate(candidate_info):
        # Add rollouts for original action
        for _ in range(OPR_NUM_ROLLOUTS):
            rollout_configs.append((obs, player, original_action, black_model, white_model))
            rollout_metadata.append((cand_idx, True, original_action))

        # Add rollouts for each alternative action
        for alt_action in alt_actions:
            for _ in range(OPR_NUM_ROLLOUTS):
                rollout_configs.append((obs, player, alt_action, black_model, white_model))
                rollout_metadata.append((cand_idx, False, alt_action))

    # Phase 3: Execute all rollouts in batch
    rollout_results = play_offpolicy_rollouts_batched(
        rollout_configs,
        OPR_ROLLOUT_TEMP,
        device,
        select_action_fn=select_action_batch_eval
    )

    # Phase 4: Aggregate results and select best alternatives
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

    # Phase 5: Create off-policy rollout samples for candidates that pass the threshold
    for cand_idx, (traj_idx, _, obs, mask, _, _, _, _, _, selected_entropy) in enumerate(candidate_info):
        if cand_idx not in candidate_results:
            continue

        results = candidate_results[cand_idx]
        original_winrate = results['original_wins'] / max(results['original_count'], 1)

        # Find best alternative
        best_action = None
        best_winrate = 0.0
        for action, action_stats in results['alts'].items():
            winrate = action_stats['wins'] / max(action_stats['count'], 1)
            if winrate > best_winrate:
                best_winrate = winrate
                best_action = action

        # Check if best alternative exceeds original by margin
        margin = best_winrate - original_winrate
        if margin >= OPR_WIN_MARGIN and best_action is not None:
            # Compute weight
            traj = trajectories[traj_idx]
            current_steps = sum(1 for is_current in traj.is_current_policy if is_current)
            step_weight = (1.0 / (current_steps ** EPISODE_WEIGHT_ALPHA)) / 8.0

            opr_samples.append({
                'obs': obs,
                'action': best_action,
                'mask': mask,
                'strength': margin * OPR_ADVANTAGE,
                'weight': step_weight,
            })

            stats.samples_added += 1
            stats.best_winrate_sum += best_winrate
            stats.orig_winrate_sum += original_winrate
            stats.entropy_selected_sum += selected_entropy

    current_policy.train()

    return opr_samples, stats


# ============================================================================
# Tactical Enhancement Application
# ============================================================================

def apply_tactical_enhancements(
    all_obs: List[np.ndarray],
    all_actions: List[int],
    all_masks: List[np.ndarray],
    all_returns: List[float],
    all_weights: List[float],
    all_is_synthetic: List[bool],
    all_is_terminal: List[bool],
    win_boost: float,
    block_boost: float
) -> Tuple[TacticalStats, TacticalBoostInfo]:
    """
    Apply tactical enhancements (win-in-1, blocking) to training samples.

    Modifies input lists in-place for existing samples (appends synthetic samples).
    Returns boost information to be applied to advantages after GAE computation.

    Args:
        all_obs: List of observations (extended with synthetic)
        all_actions: List of actions (extended with synthetic)
        all_masks: List of masks (extended with synthetic)
        all_returns: List of returns (modified in place, extended with synthetic)
        all_weights: List of weights (extended with synthetic)
        all_is_synthetic: List of synthetic flags (extended)
        all_is_terminal: List of terminal flags (extended)
        win_boost: Boost for win-in-1 situations
        block_boost: Boost for blocking situations

    Returns:
        Tuple of (TacticalStats, TacticalBoostInfo)
    """
    stats = TacticalStats()
    original_length = len(all_obs)

    # Track boosts to apply to existing samples
    sample_boosts = [0.0] * original_length

    # Track advantages for synthetic samples we'll add
    synthetic_advantages = []

    for i in range(original_length):
        winning_moves = find_all_win_in_1(all_obs[i], all_masks[i])
        if winning_moves:
            stats.win_opportunities += 1

            if all_actions[i] in winning_moves:
                # Don't boost returns - only boost advantages (applied later in training.py)
                sample_boosts[i] = win_boost
                stats.wins_found += 1

                # Add synthetic samples for other winning moves
                for other_win in winning_moves:
                    if other_win != all_actions[i]:
                        all_obs.append(all_obs[i])
                        all_actions.append(other_win)
                        all_masks.append(all_masks[i])
                        all_returns.append(all_returns[i])  # Use original return, not boosted
                        all_is_synthetic.append(True)
                        synthetic_advantages.append(win_boost)
                        all_weights.append(all_weights[i])
                        all_is_terminal.append(True)
                        stats.synthetic_wins_eq += 1
            else:
                stats.win_misses += 1
                # Add synthetic samples for all winning moves
                for winning_move in winning_moves:
                    all_obs.append(all_obs[i])
                    all_actions.append(winning_move)
                    all_masks.append(all_masks[i])
                    all_returns.append(all_returns[i])  # Use original return, not boosted
                    all_is_synthetic.append(True)
                    synthetic_advantages.append(SYNTHETIC_WIN_BOOST)
                    all_weights.append(all_weights[i])
                    all_is_terminal.append(True)
                    stats.synthetic_wins_missed += 1
        else:
            blocking_moves = find_blocking_moves(all_obs[i], all_masks[i])
            if blocking_moves is not None:
                stats.block_opportunities += 1

                if all_actions[i] in blocking_moves:
                    # Don't boost returns - only boost advantages (applied later in training.py)
                    sample_boosts[i] = block_boost
                    stats.blocks_found += 1
                else:
                    stats.block_misses += 1
                    all_obs.append(all_obs[i])
                    all_actions.append(blocking_moves[0])
                    all_masks.append(all_masks[i])
                    all_returns.append(all_returns[i])  # Use original return, not boosted
                    all_is_synthetic.append(True)
                    synthetic_advantages.append(SYNTHETIC_BLOCKING_BOOST)
                    all_weights.append(all_weights[i])
                    all_is_terminal.append(False)
                    stats.synthetic_blocks += 1

    boost_info = TacticalBoostInfo(
        sample_boosts=sample_boosts,
        synthetic_advantages=synthetic_advantages
    )

    return stats, boost_info


def compute_adaptive_boosts(win_miss_ema: float, block_miss_ema: float) -> Tuple[float, float]:
    """
    Compute adaptive tactical boosts based on miss rate EMAs.

    Uses nonlinear decay: 1 - hit_rate^2 (slower decay than linear)

    Args:
        win_miss_ema: EMA of win-in-1 miss rate
        block_miss_ema: EMA of blocking miss rate

    Returns:
        Tuple of (win_boost, block_boost)
    """
    win_hit_rate_ema = 1.0 - win_miss_ema
    win_boost = WIN_MIN_BOOST + (1.0 - win_hit_rate_ema ** 2) * (WIN_MAX_BOOST - WIN_MIN_BOOST)

    block_hit_rate_ema = 1.0 - block_miss_ema
    block_boost = BLOCK_MIN_BOOST + (1.0 - block_hit_rate_ema ** 2) * (BLOCK_MAX_BOOST - BLOCK_MIN_BOOST)

    return win_boost, block_boost


def update_miss_rate_ema(this_miss_rate: float, ema: float, ema_window: int) -> float:
    """Update EMA for miss rate tracking."""
    alpha = 1.0 / ema_window
    return alpha * this_miss_rate + (1.0 - alpha) * ema
