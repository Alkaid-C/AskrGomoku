"""
Gomoku Game Engine and Self-Play Logic

This module provides the stable game engine and self-play infrastructure:
- Gomoku board implementation with canonical representation
- Batched self-play with multiple games running in parallel
- Action selection helpers for batched inference
- Trajectory processing utilities

This module rarely needs changes unless game rules are modified.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from typing import List, Tuple, Optional
from enum import Enum



# ============================================================================
# Inference Constants
# ============================================================================

BATCH_INFERENCE_SIZE = 1024      # Positions processed simultaneously during self-play
TEMPERATURE_TRAIN = 1.0        # Softmax temperature for training (>1 flattens, <1 sharpens)
SEED_PROBABILITY = 0.25        # Probability of starting from a Renju opening
LOG_PROB_MIN = -10.0           # Minimum log probability
LOGIT_MASK_VALUE = -1e9


# ============================================================================
# Negamax Search Constants
# ============================================================================

SEARCH_DEPTH = 3                 # Default search depth

# Root node candidate generation
ROOT_TOP_K = 5                   # Top-k from policy at root
ROOT_RANDOM_K = 1                # Random neighbors at root
# Total root candidates = 6

# Internal node candidate generation
INTERNAL_TOP_K = 5               # Top-k from policy at internal nodes
INTERNAL_RANDOM_K = 1            # Random neighbors at internal nodes
# Total internal candidates = 6

# Sampling parameters
TOP_K_SAMPLE = 5                 # Top candidates for sampling (excludes worst random)
SAMPLING_TAU = 0.5               # Temperature for Q-based sampling
Q_NORM_EPSILON = 1e-6            # Normalization epsilon for Q values

# Search tree size at depth=3: 6 × 6 × 6 leaf nodes


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


def board_from_observation(obs: np.ndarray, next_player: Player) -> GomokuBoard:
    """
    Reconstruct a GomokuBoard from an observation and player to move.

    Args:
        obs: Observation [3, 15, 15] where [0] is current player, [1] is opponent
        next_player: The player who is to move next (absolute color)

    Returns:
        GomokuBoard with the reconstructed state
    """
    board = GomokuBoard()
    current_pieces = obs[0]
    opponent_pieces = obs[1]

    if next_player == Player.BLACK:
        # Current player is black, opponent is white
        board.black_pieces = current_pieces.copy()
        board.white_pieces = opponent_pieces.copy()
    else:
        # Current player is white, opponent is black
        board.white_pieces = current_pieces.copy()
        board.black_pieces = opponent_pieces.copy()

    board.who_to_play = next_player
    board.occupied_count = int(np.sum(current_pieces) + np.sum(opponent_pieces))

    return board


def get_local_candidate_moves(obs: np.ndarray, legal_mask: np.ndarray, radius: int) -> List[int]:
    """
    Get legal moves within Chebyshev distance of existing stones.

    Args:
        obs: Observation [3, 15, 15]
        legal_mask: Legal moves mask [15, 15]
        radius: Chebyshev distance radius

    Returns:
        List of flat action indices for local candidate moves
    """
    occupied = ((obs[0] == 1) | (obs[1] == 1))
    candidates = []

    # Get all legal positions
    legal_positions = np.argwhere(legal_mask == 1)

    for pos in legal_positions:
        r, c = pos[0], pos[1]
        # Check if any stone within Manhattan distance
        found_neighbor = False
        for dr in range(-radius, radius + 1):
            if found_neighbor:
                break
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) > radius:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < 15 and 0 <= nc < 15 and occupied[nr, nc]:
                    candidates.append(r * 15 + c)
                    found_neighbor = True
                    break

    return candidates


# ============================================================================
# Tensor Conversion Helpers
# ============================================================================

def obs_batch_to_tensor(obs_list: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert list of observations to batched tensor."""
    return torch.from_numpy(np.stack(obs_list)).float().to(device)


def mask_batch_to_tensor(mask_list: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert list of legal masks to batched tensor."""
    return torch.from_numpy(np.stack(mask_list)).bool().to(device)


# ============================================================================
# Action Selection (Batched Inference)
# ============================================================================

def select_action_batch(model: torch.nn.Module, obs_list: List[np.ndarray],
                        mask_list: List[np.ndarray],
                        temperature: float, device: torch.device,
                        deterministic: bool) -> Tuple[List[int], List[float]]:
    """
    Select actions for a batch of positions using the policy network.

    Returns:
        Tuple of (actions, entropies)
    """
    with torch.no_grad():
        obs_tensor = obs_batch_to_tensor(obs_list, device)
        mask_tensor = mask_batch_to_tensor(mask_list, device)

        logits_grid, _ = model(obs_tensor)
        logits = logits_grid.squeeze(1)

        logits = logits.masked_fill(~mask_tensor, LOGIT_MASK_VALUE)

        if temperature > 0 and not deterministic:
            logits = logits / temperature

        logits_flat = logits.view(len(obs_list), 225)

        if deterministic or temperature == 0:
            actions_tensor = logits_flat.argmax(dim=1)
            dist = Categorical(logits=logits_flat, validate_args=False)
        else:
            dist = Categorical(logits=logits_flat, validate_args=False)
            actions_tensor = dist.sample()

        # Separate transfers (stacking adds overhead)
        actions = actions_tensor.cpu().numpy().tolist()
        entropies = dist.entropy().cpu().numpy().tolist()

    return actions, entropies


def select_action_batch_eval(model: torch.nn.Module, obs_list: List[np.ndarray],
                              mask_list: List[np.ndarray],
                              temperature: float, device: torch.device,
                              deterministic: bool = False) -> List[int]:
    """
    Select actions for evaluation - no log_prob computation.

    Returns:
        List of action indices
    """
    with torch.no_grad():
        obs_tensor = obs_batch_to_tensor(obs_list, device)
        mask_tensor = mask_batch_to_tensor(mask_list, device)

        logits_grid, _ = model(obs_tensor)
        logits = logits_grid.squeeze(1)
        logits = logits.masked_fill(~mask_tensor, LOGIT_MASK_VALUE)

        if temperature > 0 and not deterministic:
            logits = logits / temperature

        logits_flat = logits.view(len(obs_list), 225)

        if deterministic or temperature == 0:
            actions = logits_flat.argmax(dim=1).cpu().tolist()
        else:
            probs = F.softmax(logits_flat, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().tolist()

    return actions


# ============================================================================
# Trajectory and Self-Play
# ============================================================================

class Trajectory:
    """Stores a single episode trajectory."""

    def __init__(self):
        self.observations = []  # List of [3, 15, 15] numpy arrays
        self.actions = []       # List of action indices
        self.players = []       # List of Player enum values (absolute)
        self.legal_masks = []   # List of [15, 15] legal masks
        self.is_current_policy = []  # List of bools: True if current_policy moved
        self.entropies = []     # List of policy entropies (nats) for off-policy rollout
        self.outcome = None     # GameState enum


class GameState_InProgress:
    """Tracks state of a game in progress."""

    def __init__(self, game_id: int, black_model, white_model,
                 current_is_black: bool, opening_id: int):
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
                          opening_ids: Optional[List[int]],
                          deterministic: bool = False) -> List[Trajectory]:
    """
    Play multiple episodes with batched inference.

    Args:
        black_white_pairs: List of (black_model, white_model) tuples
        current_is_black: List indicating if current_policy plays as BLACK
        temperature: Sampling temperature
        device: torch device
        batch_size: Maximum batch size for inference
        select_action_batch_fn: Function to select actions for a batch
        opening_ids: List of opening IDs for each game (-1 for empty board,
                     >= 0 for Renju opening). If None, all games start empty.
        deterministic: If True, use argmax

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
            all_entropies = [None] * len(batch_games)
            for model_id, group in model_groups.items():
                actions, entropies = select_action_batch_fn(
                    group['model'], group['obs'], group['masks'],
                    temperature, device, deterministic
                )
                for idx, action, entropy in zip(group['indices'], actions, entropies):
                    all_actions[idx] = action
                    all_entropies[idx] = entropy

            # Execute moves
            for game, action, entropy in zip(batch_games, all_actions, all_entropies):
                game.traj.actions.append(action)
                game.traj.entropies.append(entropy)
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
# Off-Policy Rollout Batched Rollouts
# ============================================================================

class OffPolicyRolloutState:
    """Minimal state for off-policy rollout games."""
    __slots__ = ['board', 'black_model', 'white_model', 'first_player', 'done', 'won']

    def __init__(self, obs: np.ndarray, next_player: Player, first_action: int,
                 black_model, white_model):
        """
        Initialize an off-policy rollout game with a forced first move.

        Args:
            obs: Observation [3, 15, 15] at the decision point
            next_player: Player to move at decision point (who plays first_action)
            first_action: Forced first action (flat index)
            black_model: Model playing as black
            white_model: Model playing as white
        """
        self.board = board_from_observation(obs, next_player)
        self.black_model = black_model
        self.white_model = white_model
        self.first_player = next_player
        self.done = False
        self.won = False

        # Apply first move
        row, col = idx_to_pos(first_action)
        outcome = self.board.Move((row, col))

        if outcome != GameState.CONTINUE:
            self.done = True
            if outcome == GameState.DRAW:
                self.won = False
            else:
                self.won = (outcome == GameState.BLACK_WIN and self.first_player == Player.BLACK) or \
                           (outcome == GameState.WHITE_WIN and self.first_player == Player.WHITE)


def play_offpolicy_rollouts_batched(rollout_configs: List[Tuple[np.ndarray, Player, int, object, object]],
                                    temperature: float, device, batch_size: int,
                                    select_action_fn) -> List[bool]:
    """
    Play off-policy rollout games in batches.

    Each rollout starts from a given position with a forced first move,
    then plays out using the specified models until game end.

    Args:
        rollout_configs: List of (obs, next_player, first_action, black_model, white_model) tuples
        temperature: Rollout temperature
        device: torch device
        batch_size: Batch size for inference
        select_action_fn: Function to select actions (should return List[int])

    Returns:
        List of bool indicating if first_player won for each rollout
    """
    # Initialize all rollout games
    games = [OffPolicyRolloutState(obs, player, action, black_model, white_model)
             for obs, player, action, black_model, white_model in rollout_configs]

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

            # Batched inference
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
                    game.done = True
                    if outcome == GameState.DRAW:
                        game.won = False
                    else:
                        game.won = (outcome == GameState.BLACK_WIN and game.first_player == Player.BLACK) or \
                                   (outcome == GameState.WHITE_WIN and game.first_player == Player.WHITE)

    return [g.won for g in games]


# ============================================================================
# Trajectory Processing Utilities
# ============================================================================

def compute_returns(traj: Trajectory) -> List[float]:
    """
    Compute per-step returns from trajectory.

    Returns z_t for each step t: +1 if player_t won, -1 if lost, 0 if draw.
    """
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


def compute_outcome_stats(trajectories: List[Trajectory], current_is_black: List[bool]) -> dict:
    """Compute statistics about game outcomes from current policy's perspective."""
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
    win_rate = current_wins / total_games if total_games > 0 else 0
    win_rate_as_black = wins_as_black / games_as_black if games_as_black > 0 else 0
    win_rate_as_white = wins_as_white / games_as_white if games_as_white > 0 else 0

    return {
        'wins': current_wins,
        'losses': current_losses,
        'draws': draws,
        'win_rate': win_rate,
        'win_rate_as_black': win_rate_as_black,
        'win_rate_as_white': win_rate_as_white,
        'avg_length': np.mean(total_steps) if total_steps else 0,
        'draw_rate': draws / total_games if total_games else 0
    }


# ============================================================================
# Search Sample Dataclass
# ============================================================================

from dataclasses import dataclass

@dataclass
class SearchSample:
    """Training sample generated from negamax search."""
    obs: np.ndarray               # [3, 15, 15] canonical observation
    sorted_candidates: List[int]  # Top candidates sorted by Q descending (length: TOP_K_SAMPLE)
    all_candidates: List[int]     # All candidates (length: ROOT_TOP_K + ROOT_RANDOM_K)
    Q_values: List[float]         # Q values for sorted_candidates [Q(c1), ..., Q(c5)]
    legal_mask: np.ndarray        # [15, 15]
    V_target: float               # max Q_search value (search backup target)


# ============================================================================
# Candidate Generation
# ============================================================================

def generate_candidates(obs: np.ndarray, legal_mask: np.ndarray,
                        model: torch.nn.Module, device: torch.device,
                        is_root: bool = True) -> List[int]:
    """
    Generate candidates: top-k from policy + random neighbors.

    Args:
        obs: Observation [3, 15, 15]
        legal_mask: Legal moves mask [15, 15]
        model: Policy model
        device: torch device
        is_root: If True, use ROOT_TOP_K + ROOT_RANDOM_K
                 If False, use INTERNAL_TOP_K + INTERNAL_RANDOM_K

    Returns:
        List of action indices (6 for root, 5 for internal)
    """
    top_k = ROOT_TOP_K if is_root else INTERNAL_TOP_K
    random_k = ROOT_RANDOM_K if is_root else INTERNAL_RANDOM_K

    # Get policy logits
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        logits_grid, _ = model(obs_tensor)
        logits = logits_grid.squeeze(0).squeeze(0)  # [15, 15]

    # Mask illegal moves
    mask_tensor = torch.from_numpy(legal_mask).bool().to(device)
    logits = logits.masked_fill(~mask_tensor, LOGIT_MASK_VALUE)

    # Get top-k legal actions by logit
    logits_flat = logits.view(225)
    _, top_indices = torch.topk(logits_flat, min(top_k, int(mask_tensor.sum().item())))
    top_actions = top_indices.cpu().tolist()

    # Get neighbor positions (Chebyshev distance <= 1 from any stone) using vectorized ops
    occupied = (obs[0] == 1) | (obs[1] == 1)

    # Compute neighbor mask via padded shifted views (binary dilation)
    padded = np.pad(occupied, 1, mode='constant', constant_values=False)
    is_neighbor = (
        padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
        padded[1:-1, :-2] |                    padded[1:-1, 2:] |
        padded[2:, :-2]  | padded[2:, 1:-1]  | padded[2:, 2:]
    )

    # Valid neighbors: adjacent to stone, not occupied, legal, not in top-k
    top_actions_set = set(top_actions)
    neighbor_mask = is_neighbor & ~occupied & (legal_mask == 1)
    neighbor_candidates = [a for a in np.where(neighbor_mask.flatten())[0] if a not in top_actions_set]

    # Select random neighbors
    if len(neighbor_candidates) >= random_k:
        random_actions = random.sample(neighbor_candidates, random_k)
    else:
        # Fallback: use any remaining legal moves not in top-k
        all_legal = [i for i in range(225) if legal_mask[i // 15, i % 15] == 1]
        remaining = [a for a in all_legal if a not in top_actions]
        if len(remaining) >= random_k:
            random_actions = random.sample(remaining, random_k)
        else:
            random_actions = remaining  # Use all remaining

    return top_actions + random_actions


def generate_candidates_batched(obs_batch: List[np.ndarray],
                                mask_batch: List[np.ndarray],
                                model: torch.nn.Module,
                                device: torch.device,
                                is_root: bool = True) -> List[List[int]]:
    """
    Generate candidates for a batch of positions.

    Args:
        obs_batch: List of observations [3, 15, 15]
        mask_batch: List of legal masks [15, 15]
        model: Policy model
        device: torch device
        is_root: If True, use ROOT_TOP_K + ROOT_RANDOM_K

    Returns:
        List of candidate lists (one per position)
    """
    top_k = ROOT_TOP_K if is_root else INTERNAL_TOP_K
    random_k = ROOT_RANDOM_K if is_root else INTERNAL_RANDOM_K
    batch_size = len(obs_batch)

    # Batch forward pass for logits
    with torch.no_grad():
        obs_tensor = torch.from_numpy(np.stack(obs_batch)).float().to(device)
        logits_grid, _ = model(obs_tensor)
        logits = logits_grid.squeeze(1)  # [B, 15, 15]

    mask_tensor = torch.from_numpy(np.stack(mask_batch)).bool().to(device)
    logits = logits.masked_fill(~mask_tensor, LOGIT_MASK_VALUE)
    logits_flat = logits.view(batch_size, 225)

    # Get top-k for each position
    _, top_indices = torch.topk(logits_flat, top_k, dim=1)
    top_indices_list = top_indices.cpu().tolist()

    all_candidates = []
    for i in range(batch_size):
        obs = obs_batch[i]
        legal_mask = mask_batch[i]
        top_actions = top_indices_list[i]
        top_actions_set = set(top_actions)

        # Get neighbor candidates using vectorized ops (binary dilation via shifted views)
        occupied = (obs[0] == 1) | (obs[1] == 1)
        padded = np.pad(occupied, 1, mode='constant', constant_values=False)
        is_neighbor = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
            padded[1:-1, :-2] |                    padded[1:-1, 2:] |
            padded[2:, :-2]  | padded[2:, 1:-1]  | padded[2:, 2:]
        )

        # Valid neighbors: adjacent to stone, not occupied, legal, not in top-k
        neighbor_mask = is_neighbor & ~occupied & (legal_mask == 1)
        neighbor_candidates = [a for a in np.where(neighbor_mask.flatten())[0] if a not in top_actions_set]

        # Select random neighbors
        if len(neighbor_candidates) >= random_k:
            random_actions = random.sample(neighbor_candidates, random_k)
        else:
            all_legal = [j for j in range(225) if legal_mask[j // 15, j % 15] == 1]
            remaining = [a for a in all_legal if a not in top_actions]
            if len(remaining) >= random_k:
                random_actions = random.sample(remaining, random_k)
            else:
                random_actions = remaining

        all_candidates.append(top_actions + random_actions)

    return all_candidates


# ============================================================================
# Negamax Search
# ============================================================================

def negamax_batched(root_obs: np.ndarray, root_mask: np.ndarray,
                    root_player: Player,
                    current_model: torch.nn.Module,
                    opponent_model: torch.nn.Module,
                    device: torch.device,
                    depth: int = SEARCH_DEPTH) -> Tuple[List[int], dict]:
    """
    Batch-efficient negamax search.

    Strategy: Generate all leaf positions upfront, batch evaluate,
    then backpropagate values using negamax rule.

    Tree structure:
    - Root: ROOT_TOP_K + ROOT_RANDOM_K candidates
    - Internal nodes: INTERNAL_TOP_K + INTERNAL_RANDOM_K candidates each
    - Leaf count depends on depth and candidate counts

    Args:
        root_obs: Root observation [3, 15, 15]
        root_mask: Root legal mask [15, 15]
        root_player: Player to move at root (absolute color)
        current_model: Model being trained
        opponent_model: Opponent model for search
        device: torch device
        depth: Search depth

    Returns:
        - candidates: List of 6 root candidates
        - Q_search: dict mapping action -> Q value for all 6
    """
    # Phase 1: Generate search tree and collect all leaf positions
    # We store (obs, mask, player, path_from_root) for each node

    # Root candidates
    root_candidates = generate_candidates(root_obs, root_mask, current_model, device, is_root=True)

    if depth == 1:
        # Special case: leaf nodes are immediate children of root
        leaf_obs_list = []
        leaf_paths = []  # Track which root candidate each leaf came from

        for i, action in enumerate(root_candidates):
            # Apply action to get child state
            child_board = board_from_observation(root_obs, root_player)
            row, col = idx_to_pos(action)
            outcome = child_board.Move((row, col))

            if outcome != GameState.CONTINUE:
                # Terminal: use actual outcome
                if outcome == GameState.DRAW:
                    value = 0.0
                elif (outcome == GameState.BLACK_WIN and root_player == Player.BLACK) or \
                     (outcome == GameState.WHITE_WIN and root_player == Player.WHITE):
                    value = 1.0
                else:
                    value = -1.0
                leaf_paths.append((i, True, value))  # (root_idx, is_terminal, value)
            else:
                c0, c1, _ = child_board.GetBoardState()
                child_obs = encode_observation(c0, c1)
                leaf_obs_list.append(child_obs)
                leaf_paths.append((i, False, len(leaf_obs_list) - 1))  # (root_idx, is_terminal, leaf_idx)

        # Batch evaluate non-terminal leaves
        if leaf_obs_list:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.stack(leaf_obs_list)).float().to(device)
                _, values = current_model(obs_tensor)
                leaf_values = values.squeeze(-1).cpu().numpy()
        else:
            leaf_values = np.array([])

        # Compute Q for each root candidate
        Q_search = {}
        for root_idx, is_terminal, val_or_idx in leaf_paths:
            action = root_candidates[root_idx]
            if is_terminal:
                Q_search[action] = val_or_idx  # Direct terminal value
            else:
                # Negate: leaf value is from opponent's perspective
                Q_search[action] = -leaf_values[val_or_idx]

        return root_candidates, Q_search

    # For depth >= 2, we need to expand further
    # Build tree level by level

    # Tree structure for depth=3:
    # - Level 0: root (not stored)
    # - Level 1: children of root (6 nodes) - after root's move
    # - Level 2: children of level 1 (6 * 5 = 30 nodes) - after opponent's move
    # - Level 3: leaves (6 * 5 * 5 = 150 nodes) - evaluated by value network
    # Total: 3 plies of search

    # Tree node: (obs, mask, player, parent_idx, action_from_parent, terminal_value or None)
    levels = []

    # Level 0 (root) - not stored, we process directly to level 1
    level_1_nodes = []

    for i, action in enumerate(root_candidates):
        child_board = board_from_observation(root_obs, root_player)
        row, col = idx_to_pos(action)
        outcome = child_board.Move((row, col))

        if outcome != GameState.CONTINUE:
            if outcome == GameState.DRAW:
                value = 0.0
            elif (outcome == GameState.BLACK_WIN and root_player == Player.BLACK) or \
                 (outcome == GameState.WHITE_WIN and root_player == Player.WHITE):
                value = 1.0
            else:
                value = -1.0
            # Store from opponent's perspective (level 1 is one ply from root)
            # This will be negated during backup to get root's perspective
            value = -value
            level_1_nodes.append({
                'obs': None, 'mask': None, 'player': None,
                'parent_idx': -1, 'action': action, 'terminal_value': value, 'root_action': action
            })
        else:
            c0, c1, _ = child_board.GetBoardState()
            child_obs = encode_observation(c0, c1)
            child_mask, child_player = child_board.GetLegalMoves()
            level_1_nodes.append({
                'obs': child_obs, 'mask': child_mask, 'player': child_player,
                'parent_idx': -1, 'action': action, 'terminal_value': None, 'root_action': action
            })

    levels.append(level_1_nodes)

    # Expand deeper levels
    # d represents current level being expanded (1 = expand level 1 to level 2, etc.)
    # Loop until d = depth - 1 to get 'depth' total plies
    for d in range(1, depth):
        current_level = levels[-1]
        next_level = []

        # Collect non-terminal nodes that need expansion
        expand_indices = []
        expand_obs = []
        expand_masks = []

        for idx, node in enumerate(current_level):
            if node['terminal_value'] is None:
                expand_indices.append(idx)
                expand_obs.append(node['obs'])
                expand_masks.append(node['mask'])

        if not expand_obs:
            # All terminal
            levels.append([])
            continue

        # Determine which model to use for candidate generation
        # At level d, if d is odd, it's opponent's turn; if even, it's current's turn
        # (root is current, level 1 is opponent, level 2 is current, ...)
        if d % 2 == 1:
            expand_model = opponent_model
        else:
            expand_model = current_model

        # Generate candidates for all nodes at this level
        candidates_list = generate_candidates_batched(expand_obs, expand_masks, expand_model, device, is_root=False)

        # Expand each node
        for i, idx in enumerate(expand_indices):
            parent_node = current_level[idx]
            obs = expand_obs[i]
            mask = expand_masks[i]
            player = parent_node['player']
            candidates = candidates_list[i]

            for action in candidates:
                child_board = board_from_observation(obs, player)
                row, col = idx_to_pos(action)
                outcome = child_board.Move((row, col))

                if outcome != GameState.CONTINUE:
                    # Terminal from root player's perspective
                    if outcome == GameState.DRAW:
                        value = 0.0
                    elif (outcome == GameState.BLACK_WIN and root_player == Player.BLACK) or \
                         (outcome == GameState.WHITE_WIN and root_player == Player.WHITE):
                        value = 1.0
                    else:
                        value = -1.0
                    # Adjust sign based on depth (how many negations back to root)
                    # At level d+1, we need (d+1) negations to get back to root
                    if (d + 1) % 2 == 1:
                        value = -value
                    next_level.append({
                        'obs': None, 'mask': None, 'player': None,
                        'parent_idx': idx, 'action': action, 'terminal_value': value,
                        'root_action': parent_node['root_action']
                    })
                else:
                    c0, c1, _ = child_board.GetBoardState()
                    child_obs = encode_observation(c0, c1)
                    child_mask, child_player = child_board.GetLegalMoves()
                    next_level.append({
                        'obs': child_obs, 'mask': child_mask, 'player': child_player,
                        'parent_idx': idx, 'action': action, 'terminal_value': None,
                        'root_action': parent_node['root_action']
                    })

        levels.append(next_level)

    # Phase 2: Batch evaluate all leaf nodes
    leaf_level = levels[-1] if levels else level_1_nodes
    leaf_obs_list = []
    leaf_indices = []

    for idx, node in enumerate(leaf_level):
        if node['terminal_value'] is None:
            leaf_obs_list.append(node['obs'])
            leaf_indices.append(idx)

    if leaf_obs_list:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.stack(leaf_obs_list)).float().to(device)
            _, values = current_model(obs_tensor)
            leaf_values = values.squeeze(-1).cpu().numpy()

        # Assign values back
        # Value is from leaf player's perspective (canonical observation)
        # Negamax backup (max of -children) handles perspective alternation automatically
        for i, idx in enumerate(leaf_indices):
            leaf_level[idx]['value'] = leaf_values[i]
    else:
        pass  # All leaves are terminal

    # Set terminal values
    for node in leaf_level:
        if node['terminal_value'] is not None:
            node['value'] = node['terminal_value']

    # Phase 3: Negamax backup
    # Work backwards through levels
    for d in range(len(levels) - 1, 0, -1):
        current_lvl = levels[d]
        parent_lvl = levels[d - 1]

        # Group children by parent
        parent_children = {}
        for node in current_lvl:
            pidx = node['parent_idx']
            if pidx not in parent_children:
                parent_children[pidx] = []
            parent_children[pidx].append(node['value'])

        # Backup: parent value = max of negated children
        for pidx, child_vals in parent_children.items():
            parent_lvl[pidx]['value'] = max(-v for v in child_vals)

        # Handle parents with no children (were terminal)
        for node in parent_lvl:
            if 'value' not in node:
                node['value'] = node['terminal_value']

    # Final backup to root
    Q_search = {}
    for node in levels[0]:
        action = node['action']
        if 'value' in node:
            Q_search[action] = -node['value']  # Negate for root's perspective
        else:
            Q_search[action] = node['terminal_value']

    return root_candidates, Q_search


# ============================================================================
# Q-Based Move Sampling
# ============================================================================

def sample_move_from_q(candidates: List[int], Q_search: dict,
                       tau: float = SAMPLING_TAU) -> int:
    """
    Sample from top candidates using normalized Q softmax.

    Args:
        candidates: All candidates (sorted by Q descending)
        Q_search: dict mapping action -> Q value
        tau: Temperature for softmax

    Returns:
        Sampled action
    """
    # Sort candidates by Q value descending
    sorted_candidates = sorted(candidates, key=lambda a: Q_search.get(a, -float('inf')), reverse=True)

    # Use top TOP_K_SAMPLE candidates (exclude worst random)
    top5 = sorted_candidates[:TOP_K_SAMPLE]
    top5_q = [Q_search[a] for a in top5]

    # Scale normalization
    q_min = min(top5_q)
    q_max = max(top5_q)
    q_range = q_max - q_min + Q_NORM_EPSILON

    q_norm = [(q - q_min) / q_range for q in top5_q]

    # Softmax with temperature
    exp_q = [np.exp(q / tau) for q in q_norm]
    sum_exp = sum(exp_q)
    probs = [e / sum_exp for e in exp_q]

    # Sample
    action = random.choices(top5, weights=probs, k=1)[0]
    return action


# ============================================================================
# Search-Based Self-Play
# ============================================================================

class SearchGameState:
    """Tracks state of a game using search-based move selection."""
    __slots__ = ['board', 'black_model', 'white_model', 'current_is_black',
                 'samples', 'done', 'outcome']

    def __init__(self, black_model, white_model, current_is_black: bool, opening_id: int = -1):
        self.board = GomokuBoard(opening_id=opening_id)
        self.black_model = black_model
        self.white_model = white_model
        self.current_is_black = current_is_black
        self.samples = []  # List of SearchSample
        self.done = False
        self.outcome = None


def play_episodes_with_search(num_episodes: int,
                              current_policy: torch.nn.Module,
                              opponents: List[torch.nn.Module],
                              opponent_indices: List[int],
                              current_is_black: List[bool],
                              device: torch.device,
                              depth: int = SEARCH_DEPTH,
                              opening_ids: Optional[List[int]] = None,
                              tau: float = SAMPLING_TAU) -> Tuple[List[List[SearchSample]], List[GameState]]:
    """
    Play games using negamax search for move selection.

    For each move by current_policy:
    1. Generate candidates (ROOT_TOP_K + ROOT_RANDOM_K)
    2. Run negamax search
    3. Record SearchSample
    4. Sample move from top candidates using Q softmax

    Args:
        num_episodes: Number of games to play
        current_policy: Model being trained
        opponents: List of opponent models
        opponent_indices: Index into opponents for each game
        current_is_black: Whether current_policy plays as black for each game
        device: torch device
        depth: Search depth
        opening_ids: Optional list of opening IDs
        tau: Sampling temperature

    Returns:
        - List of SearchSample lists (one list per game)
        - List of GameState outcomes
    """
    if opening_ids is None:
        opening_ids = [-1] * num_episodes

    # Initialize games
    games = []
    for i in range(num_episodes):
        is_black = current_is_black[i]
        opponent = opponents[opponent_indices[i]]

        if is_black:
            black_model = current_policy
            white_model = opponent
        else:
            black_model = opponent
            white_model = current_policy

        game = SearchGameState(black_model, white_model, is_black, opening_ids[i])
        games.append(game)

    # Play games (not batched across games due to search complexity)
    while True:
        active_games = [g for g in games if not g.done]
        if not active_games:
            break

        for game in active_games:
            legal_mask, next_player = game.board.GetLegalMoves()
            c0, c1, _ = game.board.GetBoardState()
            obs = encode_observation(c0, c1)

            # Determine if it's current policy's turn
            is_current_turn = (
                (next_player == Player.BLACK and game.current_is_black) or
                (next_player == Player.WHITE and not game.current_is_black)
            )

            if is_current_turn:
                # Use search
                opponent_model = game.white_model if game.current_is_black else game.black_model
                candidates, Q_search = negamax_batched(
                    obs, legal_mask, next_player,
                    current_policy, opponent_model, device, depth
                )

                # Sort candidates by Q
                sorted_cands = sorted(candidates, key=lambda a: Q_search.get(a, -float('inf')), reverse=True)

                # Record sample (top 5 for ranking loss)
                V_target = max(Q_search.values())
                Q_values_top5 = [Q_search.get(a, 0.0) for a in sorted_cands[:5]]
                sample = SearchSample(
                    obs=obs.copy(),
                    sorted_candidates=sorted_cands[:5],
                    all_candidates=candidates,
                    Q_values=Q_values_top5,
                    legal_mask=legal_mask.copy(),
                    V_target=V_target
                )
                game.samples.append(sample)

                # Sample move
                action = sample_move_from_q(candidates, Q_search, tau)
            else:
                # Opponent's turn - also use search for equally smart moves
                opp_model = game.black_model if next_player == Player.BLACK else game.white_model
                # From opponent's perspective: opp_model is "current", current_policy is "opponent"
                opp_candidates, opp_Q_search = negamax_batched(
                    obs, legal_mask, next_player,
                    opp_model, current_policy, device, depth
                )
                # Sample move using same mechanism
                action = sample_move_from_q(opp_candidates, opp_Q_search, tau)

            # Apply move
            row, col = idx_to_pos(action)
            outcome = game.board.Move((row, col))

            if outcome != GameState.CONTINUE:
                game.outcome = outcome
                game.done = True

    return [g.samples for g in games], [g.outcome for g in games]
