"""
Gomoku Game Engine and Self-Play Logic

This module provides the stable game engine and self-play infrastructure:
- Gomoku board implementation with canonical representation
- Batched self-play with multiple games running in parallel
- Data augmentation via 8-fold symmetry transformations
- Tactical search for win-in-1 and blocking detection

This module rarely needs changes unless game rules are modified.
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Optional
from enum import Enum


# ============================================================================
# Renju Opening Sequences (for Opening Seeding)
# ============================================================================

def _generate_renju_openings() -> List[Tuple[Tuple[int, int], ...]]:
    """
    Generate all 184 Renju opening sequences.

    Renju opening rules (relative to first move at origin):
    - Move 1: (0, 0) - center
    - Move 2: Any of 8 positions in [-1,1] x [-1,1] excluding (0,0)
    - Move 3: Any of 23 positions in [-2,2] x [-2,2] excluding moves 1 and 2

    Total: 8 * 23 = 184 unique openings
    """
    openings = []
    # Second move: [-1,1] x [-1,1] excluding (0,0)
    for r2 in range(-1, 2):
        for c2 in range(-1, 2):
            if (r2, c2) == (0, 0):
                continue
            # Third move: [-2,2] x [-2,2] excluding first and second moves
            for r3 in range(-2, 3):
                for c3 in range(-2, 3):
                    if (r3, c3) == (0, 0) or (r3, c3) == (r2, c2):
                        continue
                    openings.append(((0, 0), (r2, c2), (r3, c3)))
    return openings


RENJU_OPENING_SEQUENCES = _generate_renju_openings()


# ============================================================================
# Gomoku Board Engine
# ============================================================================

class Player(Enum):
    """Enum representing the two players."""
    BLACK = 1
    WHITE = 2


class GameState(Enum):
    """Enum representing the game state after a move."""
    CONTINUE = 0
    BLACK_WIN = 1
    WHITE_WIN = 2
    DRAW = 3


class GomokuBoard:
    """
    Gomoku board engine with canonical output for neural networks.

    Internal state maintains absolute colors (black_pieces, white_pieces) as numpy arrays.
    Public API exposes board in canonical form (current player, opponent).
    """

    def __init__(self, opening_id: int = -1):
        """
        Initialize a 15x15 Gomoku board.

        Args:
            opening_id: If -1, start with empty board. If >= 0, start with the
                        specified Renju opening (index into RENJU_OPENING_SEQUENCES)
                        with a random offset applied.
        """
        self.black_pieces = np.zeros((15, 15), dtype=np.uint8)
        self.white_pieces = np.zeros((15, 15), dtype=np.uint8)
        self.who_to_play = Player.BLACK
        self.occupied_count = 0  # Track number of stones on board

        if opening_id >= 0:
            # Apply Renju opening with random offset
            # Offset: first move can be anywhere in center ±3 (rows/cols 4-10)
            offset_r = random.randint(-3, 3)
            offset_c = random.randint(-3, 3)
            base_r, base_c = 7 + offset_r, 7 + offset_c

            for rel_r, rel_c in RENJU_OPENING_SEQUENCES[opening_id]:
                self.Move((base_r + rel_r, base_c + rel_c))

    def GetLegalMoves(self) -> Tuple[np.ndarray, Player]:
        """
        Returns legal moves mask and current player to move.

        A move is legal if the cell is empty (neither black nor white stone).

        Returns:
            Tuple of (legal_mask, next_player) where:
            - legal_mask: 15x15 numpy array with 1 for legal moves, 0 for illegal
            - next_player: absolute color (BLACK or WHITE) to move
        """
        # Vectorized: legal where neither black nor white has a piece
        legal_mask = ((self.black_pieces == 0) & (self.white_pieces == 0)).astype(np.uint8)
        return (legal_mask, self.who_to_play)

    def GetBoardState(self) -> Tuple[np.ndarray, np.ndarray, Player]:
        """
        Returns canonical board state for neural network input.

        The canonical view always presents the board from the perspective of
        the player to move:
        - c0: current player's pieces
        - c1: opponent's pieces

        WARNING: Returns references to internal state, not copies.
        Do not modify the returned grids. They are immediately converted
        to tensors by the caller, so copying is unnecessary.

        Returns:
            Tuple of (c0, c1, next_player) where:
            - c0: next player's pieces (15x15 numpy array, REFERENCE)
            - c1: opponent's pieces (15x15 numpy array, REFERENCE)
            - next_player: absolute color (BLACK or WHITE)
        """
        if self.who_to_play == Player.BLACK:
            return (self.black_pieces, self.white_pieces, self.who_to_play)
        else:  # WHITE
            return (self.white_pieces, self.black_pieces, self.who_to_play)

    def Move(self, position: Tuple[int, int]) -> GameState:
        """
        Place a stone at the given position for the current player.

        Assumes the position is legal (caller guarantees). Does NOT check
        bounds or occupancy.

        After placing the stone:
        1. Checks if current player wins (5+ contiguous stones)
        2. If not, checks if board is full (draw)
        3. Toggles to opposite player (regardless of outcome)

        Args:
            position: Tuple (row, col) where 0 <= row, col < 15

        Returns:
            GameState: BLACK_WIN, WHITE_WIN, DRAW, or CONTINUE
        """
        row, col = position
        current_player = self.who_to_play

        # Place the stone for current player
        if current_player == Player.BLACK:
            self.black_pieces[row, col] = 1
            current_pieces = self.black_pieces
        else:
            self.white_pieces[row, col] = 1
            current_pieces = self.white_pieces

        # Update occupied count
        self.occupied_count += 1

        # Evaluate terminal state
        state = GameState.CONTINUE
        if self._check_win(row, col, current_pieces):
            state = GameState.BLACK_WIN if current_player == Player.BLACK else GameState.WHITE_WIN
        elif self._is_board_full():
            state = GameState.DRAW

        # Toggle player (regardless of outcome)
        self.who_to_play = Player.WHITE if current_player == Player.BLACK else Player.BLACK

        return state

    def Render(self) -> str:
        """
        Render the board as an ASCII string.

        Returns:
            String representation where:
            - '.' represents empty cell
            - 'x' represents black stone
            - 'o' represents white stone
        """
        chars = np.full((15, 15), '.', dtype='U1')
        chars[self.black_pieces == 1] = 'x'
        chars[self.white_pieces == 1] = 'o'
        return '\n'.join(''.join(row) for row in chars)

    def _check_win(self, row: int, col: int, pieces: np.ndarray) -> bool:
        """
        Check if placing a stone at (row, col) results in a win.

        Only checks lines passing through the given position (efficient).
        Checks four direction pairs: horizontal, vertical, diagonal, anti-diagonal.

        Args:
            row: Row of the placed stone
            col: Column of the placed stone
            pieces: The piece grid to check (black_pieces or white_pieces)

        Returns:
            True if the move creates 5+ contiguous stones, False otherwise
        """
        # Four directions: horizontal, vertical, diagonal, anti-diagonal
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))

        for dr, dc in directions:
            count = 1  # Count the placed stone itself

            # Count contiguous stones in forward direction
            r, c = row + dr, col + dc
            while 0 <= r < 15 and 0 <= c < 15 and pieces[r, c] == 1:
                count += 1
                r += dr
                c += dc

            # Count contiguous stones in backward direction
            r, c = row - dr, col - dc
            while 0 <= r < 15 and 0 <= c < 15 and pieces[r, c] == 1:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False

    def _is_board_full(self) -> bool:
        """
        Check if the board is completely full.

        Returns:
            True if no empty cells remain, False otherwise
        """
        return self.occupied_count == 225


# ============================================================================
# Helper Functions
# ============================================================================

def encode_observation(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """
    Encode board state into a numpy array.

    Args:
        c0: Current player's pieces [15, 15] numpy array
        c1: Opponent's pieces [15, 15] numpy array

    Returns:
        obs: [3, 15, 15] numpy array (uint8) where:
            - Channel 0: Current player's pieces
            - Channel 1: Opponent's pieces
            - Channel 2: Board mask (all 1s, provides explicit boundary info to convolutions)
    """
    # Channel 2 is a board mask: all 1s within the valid board region.
    # This allows convolutions to distinguish "empty cell inside board" (0,0,1)
    # from "padding region outside board" (0,0,0).
    board_mask = np.ones((15, 15), dtype=np.uint8)
    return np.stack([c0, c1, board_mask], axis=0)


def idx_to_pos(idx: int) -> Tuple[int, int]:
    """Convert flat index to (row, col)."""
    return (idx // 15, idx % 15)


def pos_to_idx(row: int, col: int) -> int:
    """Convert (row, col) to flat index."""
    return row * 15 + col


# ============================================================================
# Self-Play with Batched Inference
# ============================================================================

class Trajectory:
    """Stores a single episode trajectory."""

    def __init__(self):
        self.observations = []  # List of [3, 15, 15] numpy arrays
        self.actions = []       # List of action indices
        self.players = []       # List of Player enum values (absolute)
        self.legal_masks = []   # List of [15, 15] legal masks
        self.is_current_policy = []  # List of bools: True if current_policy moved
        self.log_probs = []     # List of log probabilities (from temperature-scaled distribution)
        self.outcome = None     # GameState enum


class GameState_InProgress:
    """Tracks state of a game in progress."""

    def __init__(self, game_id: int, black_model, white_model,
                 current_is_black: bool, opening_id: int = -1):
        self.game_id = game_id
        self.board = GomokuBoard(opening_id=opening_id)
        self.black_model = black_model
        self.white_model = white_model
        self.current_is_black = current_is_black  # True if black_model is current_policy
        self.traj = Trajectory()
        self.done = False


def play_episodes_batched(black_white_pairs: List[Tuple],
                          current_is_black: List[bool],
                          temperature: float, device: torch.device,
                          batch_size: int,
                          select_action_batch_fn,
                          deterministic: bool = False,
                          opening_ids: List[int] = None) -> List[Trajectory]:
    """
    Play multiple episodes with batched inference.

    Args:
        black_white_pairs: List of (black_model, white_model) tuples
        current_is_black: List indicating if current_policy plays as BLACK
        temperature: Sampling temperature
        device: torch device
        batch_size: Maximum batch size for inference
        select_action_batch_fn: Function to select actions for a batch
        deterministic: If True, use argmax
        opening_ids: List of opening IDs for each game (-1 for empty board,
                     >= 0 for Renju opening). If None, all games start empty.

    Returns:
        List of trajectories
    """
    # Initialize all games
    if opening_ids is None:
        opening_ids = [-1] * len(black_white_pairs)

    games = [GameState_InProgress(i, black, white, is_black, opening_id)
             for i, ((black, white), is_black, opening_id) in enumerate(
                 zip(black_white_pairs, current_is_black, opening_ids))]

    while True:
        # Get active games
        active_games = [g for g in games if not g.done]
        if len(active_games) == 0:
            break

        # Process in batches
        for batch_start in range(0, len(active_games), batch_size):
            batch_games = active_games[batch_start:batch_start + batch_size]

            # Collect observations and masks
            obs_list = []
            mask_list = []
            models_list = []

            for game in batch_games:
                legal_mask, next_player = game.board.GetLegalMoves()
                c0, c1, _ = game.board.GetBoardState()
                obs = encode_observation(c0, c1)

                # Store for trajectory
                game.traj.observations.append(obs)
                game.traj.legal_masks.append(legal_mask)
                game.traj.players.append(next_player)

                # Track if current_policy is moving
                is_current_moving = (
                    (next_player == Player.BLACK and game.current_is_black) or
                    (next_player == Player.WHITE and not game.current_is_black)
                )
                game.traj.is_current_policy.append(is_current_moving)

                # Collect for batched inference
                obs_list.append(obs)
                mask_list.append(legal_mask)

                # Select appropriate model
                model = game.black_model if next_player == Player.BLACK else game.white_model
                models_list.append(model)

            # Group by model to enable batching
            model_groups = {}
            for i, model in enumerate(models_list):
                model_id = id(model)
                if model_id not in model_groups:
                    model_groups[model_id] = {'model': model, 'indices': [], 'obs': [], 'masks': []}
                model_groups[model_id]['indices'].append(i)
                model_groups[model_id]['obs'].append(obs_list[i])
                model_groups[model_id]['masks'].append(mask_list[i])

            # Run batched inference for each unique model
            all_actions = [None] * len(batch_games)
            all_log_probs = [None] * len(batch_games)
            for model_id, group in model_groups.items():
                actions, log_probs = select_action_batch_fn(
                    group['model'], group['obs'], group['masks'],
                    temperature, device, deterministic
                )
                for idx, action, log_prob in zip(group['indices'], actions, log_probs):
                    all_actions[idx] = action
                    all_log_probs[idx] = log_prob

            # Execute moves
            for game, action, log_prob in zip(batch_games, all_actions, all_log_probs):
                game.traj.actions.append(action)
                game.traj.log_probs.append(log_prob)
                row, col = idx_to_pos(action)
                outcome = game.board.Move((row, col))

                if outcome != GameState.CONTINUE:
                    game.traj.outcome = outcome
                    game.done = True

    return [g.traj for g in games]


# ============================================================================
# Evaluation Play (Lightweight)
# ============================================================================

class EvalGameState:
    """Minimal state for evaluation games - no trajectory storage."""
    __slots__ = ['board', 'black_model', 'white_model', 'current_is_black', 'done', 'outcome']

    def __init__(self, black_model, white_model, current_is_black: bool):
        self.board = GomokuBoard()
        self.black_model = black_model
        self.white_model = white_model
        self.current_is_black = current_is_black
        self.done = False
        self.outcome = None


def play_eval_games(black_white_pairs: List[Tuple],
                    current_is_black: List[bool],
                    temperature: float, device: torch.device,
                    batch_size: int,
                    select_action_fn) -> List[Tuple[GameState, bool]]:
    """
    Play evaluation games - returns only (outcome, current_is_black) pairs.

    This is a lightweight version of play_episodes_batched that skips
    trajectory storage and log_prob computation for faster evaluation.

    Args:
        black_white_pairs: List of (black_model, white_model) tuples
        current_is_black: List indicating if current_policy plays as BLACK
        temperature: Sampling temperature
        device: torch device
        batch_size: Maximum batch size for inference
        select_action_fn: Function to select actions (should return List[int])

    Returns:
        List of (outcome, current_is_black) tuples
    """
    games = [EvalGameState(black, white, is_black)
             for (black, white), is_black in zip(black_white_pairs, current_is_black)]

    while True:
        active_games = [g for g in games if not g.done]
        if not active_games:
            break

        for batch_start in range(0, len(active_games), batch_size):
            batch_games = active_games[batch_start:batch_start + batch_size]

            obs_list = []
            mask_list = []
            models_list = []

            for game in batch_games:
                legal_mask, next_player = game.board.GetLegalMoves()
                c0, c1, _ = game.board.GetBoardState()
                obs = encode_observation(c0, c1)

                obs_list.append(obs)
                mask_list.append(legal_mask)
                model = game.black_model if next_player == Player.BLACK else game.white_model
                models_list.append(model)

            # Group by model for batching
            model_groups = {}
            for i, model in enumerate(models_list):
                model_id = id(model)
                if model_id not in model_groups:
                    model_groups[model_id] = {'model': model, 'indices': [], 'obs': [], 'masks': []}
                model_groups[model_id]['indices'].append(i)
                model_groups[model_id]['obs'].append(obs_list[i])
                model_groups[model_id]['masks'].append(mask_list[i])

            # Batched inference - no log_probs needed
            all_actions = [None] * len(batch_games)
            for group in model_groups.values():
                actions = select_action_fn(
                    group['model'], group['obs'], group['masks'],
                    temperature, device
                )
                for idx, action in zip(group['indices'], actions):
                    all_actions[idx] = action

            # Execute moves
            for game, action in zip(batch_games, all_actions):
                row, col = idx_to_pos(action)
                outcome = game.board.Move((row, col))
                if outcome != GameState.CONTINUE:
                    game.outcome = outcome
                    game.done = True

    return [(g.outcome, g.current_is_black) for g in games]


# ============================================================================
# Data Augmentation (8-fold symmetry) - GPU Accelerated
# ============================================================================

def augment_batch_gpu(obs_batch: torch.Tensor, actions: torch.Tensor,
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


# ============================================================================
# Tactical Search (Win-in-1 and Blocking Detection)
# ============================================================================

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

    # Get legal positions using numpy (avoids iterating all 225 cells)
    legal_positions = np.argwhere(legal_mask == 1)

    for pos in legal_positions:
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

    # Get legal positions using numpy (avoids iterating all 225 cells)
    legal_positions = np.argwhere(legal_mask == 1)

    for pos in legal_positions:
        row, col = pos[0], pos[1]
        # Would opponent win by playing here?
        if is_winning_move(board_opponent, row, col):
            blocking_positions.append(row * 15 + col)
            # Early exit if we find 2+ threats (dual threat = unblockable)
            if len(blocking_positions) >= 2:
                return None

    return blocking_positions if blocking_positions else None
