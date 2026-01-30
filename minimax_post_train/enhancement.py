"""
Training Enhancements Module

Contains:
- Tactical search (win-in-1, blocking detection) for probing accuracy
- 8-fold symmetry data augmentation (GPU accelerated)
- Local candidate position utilities for negamax search
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from gomoku import Trajectory, GameState


# ============================================================================
# Stats Dataclasses
# ============================================================================

@dataclass
class TacticalStats:
    """Statistics from tactical probe."""
    win_opportunities: int = 0
    win_hits: int = 0
    win_misses: int = 0
    block_opportunities: int = 0
    block_hits: int = 0
    block_misses: int = 0

    @property
    def win_accuracy(self) -> float:
        """Win-in-1 detection accuracy."""
        if self.win_opportunities == 0:
            return 1.0
        return self.win_hits / self.win_opportunities

    @property
    def block_accuracy(self) -> float:
        """Blocking detection accuracy."""
        if self.block_opportunities == 0:
            return 1.0
        return self.block_hits / self.block_opportunities


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
# Tactical Probe (Accuracy Metrics Only)
# ============================================================================

def probe_tactical_accuracy(trajectories: List[Trajectory],
                            current_is_black: List[bool]) -> TacticalStats:
    """
    Probe tactical accuracy of the current policy without modifying training.

    Scans all positions in trajectories where current policy moved, checking:
    - Win-in-1 opportunities and whether the policy found them
    - Blocking opportunities and whether the policy blocked

    Args:
        trajectories: List of game trajectories
        current_is_black: List indicating if current policy played as black

    Returns:
        TacticalStats with accuracy metrics
    """
    stats = TacticalStats()

    for traj, is_black in zip(trajectories, current_is_black):
        for t in range(len(traj.observations)):
            # Only check current policy's moves
            if not traj.is_current_policy[t]:
                continue

            obs = traj.observations[t]
            mask = traj.legal_masks[t]
            action = traj.actions[t]

            # Check win-in-1
            winning_moves = find_all_win_in_1(obs, mask)
            if winning_moves:
                stats.win_opportunities += 1
                if action in winning_moves:
                    stats.win_hits += 1
                else:
                    stats.win_misses += 1
            else:
                # Only check blocking if no win opportunity
                blocking_moves = find_blocking_moves(obs, mask)
                if blocking_moves is not None:
                    stats.block_opportunities += 1
                    if action in blocking_moves:
                        stats.block_hits += 1
                    else:
                        stats.block_misses += 1

    return stats


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
    B = obs_batch.size(0)
    device = obs_batch.device

    # Convert actions to 2D coordinates
    action_rows = actions // 15
    action_cols = actions % 15

    # Pre-allocate output tensors
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
        elif sym_id == 1:  # Rotate 90° CW
            all_obs[start_idx:end_idx] = obs_batch.transpose(-2, -1).flip(-1)
            all_masks[start_idx:end_idx] = masks_batch.transpose(-2, -1).flip(-1)
            new_rows, new_cols = action_cols, 14 - action_rows
        elif sym_id == 2:  # Rotate 180°
            all_obs[start_idx:end_idx] = obs_batch.flip(-2).flip(-1)
            all_masks[start_idx:end_idx] = masks_batch.flip(-2).flip(-1)
            new_rows, new_cols = 14 - action_rows, 14 - action_cols
        elif sym_id == 3:  # Rotate 270° CW
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


def augment_candidates_8fold(candidates: List[int], sym_id: int) -> List[int]:
    """
    Transform candidate action indices using the same symmetry transform.

    Args:
        candidates: List of flat action indices
        sym_id: Symmetry ID (0-7)

    Returns:
        List of transformed action indices
    """
    result = []
    for action in candidates:
        row, col = action // 15, action % 15

        if sym_id == 0:  # Identity
            new_row, new_col = row, col
        elif sym_id == 1:  # Rotate 90° CW
            new_row, new_col = col, 14 - row
        elif sym_id == 2:  # Rotate 180°
            new_row, new_col = 14 - row, 14 - col
        elif sym_id == 3:  # Rotate 270° CW
            new_row, new_col = 14 - col, row
        elif sym_id == 4:  # Flip horizontal
            new_row, new_col = row, 14 - col
        elif sym_id == 5:  # Flip vertical
            new_row, new_col = 14 - row, col
        elif sym_id == 6:  # Transpose
            new_row, new_col = col, row
        elif sym_id == 7:  # Anti-transpose
            new_row, new_col = 14 - col, 14 - row

        result.append(new_row * 15 + new_col)

    return result
