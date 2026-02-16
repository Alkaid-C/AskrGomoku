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

TEMPERATURE_TRAIN = 1.0        # Softmax temperature for training (>1 flattens, <1 sharpens)
SEED_PROBABILITY = 0.25        # Probability of starting from a Renju opening
LOG_PROB_MIN = -10.0           # Minimum log probability
LOGIT_MASK_VALUE = -1e9


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

_BOARD_MASK = np.ones((15, 15), dtype=np.uint8)
_BOARD_MASK.flags.writeable = False


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
    return np.stack([c0, c1, _BOARD_MASK], axis=0)


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
        # Check if any stone within Chebyshev distance
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

    def __init__(self, black_model, white_model,
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

    games = [GameState_InProgress(black, white, is_black, opening_id)
             for (black, white), is_black, opening_id in
                 zip(black_white_pairs, current_is_black, opening_ids)]

    n_games = len(games)

    # Precompute per-model game indices. Model assignments (black_model,
    # white_model) are fixed per game, so we build the mapping once and
    # filter by active + turn color each step, avoiding per-step dict rebuilds.
    # model_id -> (model, black_game_indices, white_game_indices)
    model_game_map = {}
    for i, game in enumerate(games):
        bid = id(game.black_model)
        if bid not in model_game_map:
            model_game_map[bid] = (game.black_model, [], [])
        model_game_map[bid][1].append(i)

        wid = id(game.white_model)
        if wid not in model_game_map:
            model_game_map[wid] = (game.white_model, [], [])
        model_game_map[wid][2].append(i)

    active_mask = [True] * n_games
    n_active = n_games

    while n_active > 0:
        all_actions = [None] * n_games
        all_entropies = [None] * n_games

        for model, black_indices, white_indices in model_game_map.values():
            # Collect active games where this model is to move
            batch_indices = [i for i in black_indices
                            if active_mask[i] and games[i].board.who_to_play == Player.BLACK]
            batch_indices.extend(i for i in white_indices
                                if active_mask[i] and games[i].board.who_to_play == Player.WHITE)
            if not batch_indices:
                continue

            # Collect observations and masks
            obs_list = []
            mask_list = []
            for i in batch_indices:
                game = games[i]
                legal_mask, next_player = game.board.GetLegalMoves()
                c0, c1, _ = game.board.GetBoardState()
                obs = encode_observation(c0, c1)

                game.traj.observations.append(obs)
                game.traj.legal_masks.append(legal_mask)
                game.traj.players.append(next_player)

                is_current_moving = (
                    (next_player == Player.BLACK and game.current_is_black) or
                    (next_player == Player.WHITE and not game.current_is_black)
                )
                game.traj.is_current_policy.append(is_current_moving)

                obs_list.append(obs)
                mask_list.append(legal_mask)

            # Batched inference for this model group
            actions, entropies = select_action_batch_fn(
                model, obs_list, mask_list,
                temperature, device, deterministic
            )

            for i, action, entropy in zip(batch_indices, actions, entropies):
                all_actions[i] = action
                all_entropies[i] = entropy

        # Execute moves for all active games
        for i in range(n_games):
            if not active_mask[i]:
                continue
            game = games[i]
            action = all_actions[i]
            entropy = all_entropies[i]

            game.traj.actions.append(action)
            game.traj.entropies.append(entropy)
            row, col = idx_to_pos(action)
            outcome = game.board.Move((row, col))

            if outcome != GameState.CONTINUE:
                game.traj.outcome = outcome
                game.done = True
                active_mask[i] = False
                n_active -= 1

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
        select_action_fn: Function to select actions (should return List[int])

    Returns:
        List of (outcome, current_is_black) tuples
    """
    games = [EvalGameState(black, white, is_black)
             for (black, white), is_black in zip(black_white_pairs, current_is_black)]

    n_games = len(games)

    # Precompute per-model game indices (same pattern as play_episodes_batched)
    model_game_map = {}
    for i, game in enumerate(games):
        bid = id(game.black_model)
        if bid not in model_game_map:
            model_game_map[bid] = (game.black_model, [], [])
        model_game_map[bid][1].append(i)

        wid = id(game.white_model)
        if wid not in model_game_map:
            model_game_map[wid] = (game.white_model, [], [])
        model_game_map[wid][2].append(i)

    active_mask = [True] * n_games
    n_active = n_games

    while n_active > 0:
        all_actions = [None] * n_games

        for model, black_indices, white_indices in model_game_map.values():
            batch_indices = [i for i in black_indices
                            if active_mask[i] and games[i].board.who_to_play == Player.BLACK]
            batch_indices.extend(i for i in white_indices
                                if active_mask[i] and games[i].board.who_to_play == Player.WHITE)
            if not batch_indices:
                continue

            obs_list = []
            mask_list = []
            for i in batch_indices:
                game = games[i]
                legal_mask, _ = game.board.GetLegalMoves()
                c0, c1, _ = game.board.GetBoardState()
                obs_list.append(encode_observation(c0, c1))
                mask_list.append(legal_mask)

            actions = select_action_fn(
                model, obs_list, mask_list,
                temperature, device
            )
            for i, action in zip(batch_indices, actions):
                all_actions[i] = action

        for i in range(n_games):
            if not active_mask[i]:
                continue
            game = games[i]
            row, col = idx_to_pos(all_actions[i])
            outcome = game.board.Move((row, col))
            if outcome != GameState.CONTINUE:
                game.outcome = outcome
                game.done = True
                active_mask[i] = False
                n_active -= 1

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
                                    temperature: float, device,
                                    select_action_fn) -> List[bool]:
    """
    Play off-policy rollout games in batches.

    Each rollout starts from a given position with a forced first move,
    then plays out using the specified models until game end.

    Args:
        rollout_configs: List of (obs, next_player, first_action, black_model, white_model) tuples
        temperature: Rollout temperature
        device: torch device
        select_action_fn: Function to select actions (should return List[int])

    Returns:
        List of bool indicating if first_player won for each rollout
    """
    # Initialize all rollout games
    games = [OffPolicyRolloutState(obs, player, action, black_model, white_model)
             for obs, player, action, black_model, white_model in rollout_configs]

    n_games = len(games)

    # Precompute per-model game indices (same pattern as play_episodes_batched)
    model_game_map = {}
    for i, game in enumerate(games):
        bid = id(game.black_model)
        if bid not in model_game_map:
            model_game_map[bid] = (game.black_model, [], [])
        model_game_map[bid][1].append(i)

        wid = id(game.white_model)
        if wid not in model_game_map:
            model_game_map[wid] = (game.white_model, [], [])
        model_game_map[wid][2].append(i)

    active_mask = [True] * n_games
    n_active = n_games

    while n_active > 0:
        all_actions = [None] * n_games

        for model, black_indices, white_indices in model_game_map.values():
            batch_indices = [i for i in black_indices
                            if active_mask[i] and games[i].board.who_to_play == Player.BLACK]
            batch_indices.extend(i for i in white_indices
                                if active_mask[i] and games[i].board.who_to_play == Player.WHITE)
            if not batch_indices:
                continue

            obs_list = []
            mask_list = []
            for i in batch_indices:
                game = games[i]
                legal_mask, _ = game.board.GetLegalMoves()
                c0, c1, _ = game.board.GetBoardState()
                obs_list.append(encode_observation(c0, c1))
                mask_list.append(legal_mask)

            actions = select_action_fn(
                model, obs_list, mask_list,
                temperature, device
            )
            for i, action in zip(batch_indices, actions):
                all_actions[i] = action

        for i in range(n_games):
            if not active_mask[i]:
                continue
            game = games[i]
            row, col = idx_to_pos(all_actions[i])
            outcome = game.board.Move((row, col))
            if outcome != GameState.CONTINUE:
                game.done = True
                active_mask[i] = False
                n_active -= 1
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
