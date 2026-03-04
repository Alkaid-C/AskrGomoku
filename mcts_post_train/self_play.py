"""
MCTS Self-Play Game Generation

Generates training data by playing games where both sides use MCTS search.
Only records positions from the current model's turns for training.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from gomoku import (
    GameState,
    GomokuBoard,
    Player,
    encode_observation,
    idx_to_pos,
)
from mcts import mcts_search_batched

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class MCTSGameRecord:
    """Training data from a single MCTS self-play game."""
    observations: list[np.ndarray] = field(default_factory=list)        # [3, 15, 15] per move
    visit_distributions: list[np.ndarray] = field(default_factory=list) # [225] normalized
    root_values: list[float] = field(default_factory=list)              # MCTS root Q
    actions: list[int] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)
    outcome: GameState = GameState.CONTINUE
    current_is_black: bool = True


# ============================================================================
# MCTS Game Generation
# ============================================================================


def play_mcts_games(
    current_model: nn.Module,
    opponent_models: list[nn.Module],
    current_is_black: list[bool],
    num_simulations: int,
    c_puct: float,
    prior_temperature: float,
    device: torch.device,
    opening_ids: list[int],
    dirichlet_alpha: float = 0.15,
    dirichlet_epsilon: float = 0.25,
    action_temperature: float = 1.0,
) -> list[MCTSGameRecord]:
    """
    Play MCTS self-play games with batched search.

    Both current model and opponents use MCTS. Training data is only
    collected from the current model's moves.

    Args:
        current_model: Model being trained
        opponent_models: One opponent model per game
        current_is_black: Per-game flag for current model's color
        num_simulations: MCTS simulations per move
        c_puct: PUCT exploration constant
        prior_temperature: Temperature for softening prior in MCTS
        device: Torch device
        opening_ids: Per-game opening ID (-1 for empty board)
        dirichlet_alpha: Dirichlet noise alpha
        dirichlet_epsilon: Dirichlet noise weight
        action_temperature: Temperature for sampling actions from visit counts

    Returns:
        List of MCTSGameRecord with training data
    """
    n_games = len(opponent_models)
    assert len(current_is_black) == n_games
    assert len(opening_ids) == n_games

    # Initialize boards and records
    boards: list[GomokuBoard] = []
    records: list[MCTSGameRecord] = []

    for i in range(n_games):
        boards.append(GomokuBoard(opening_id=opening_ids[i]))
        records.append(MCTSGameRecord(current_is_black=current_is_black[i]))

    active_mask = [True] * n_games
    n_active = n_games

    # Precompute model assignments per game
    # game_index -> (black_model, white_model)
    game_models: list[tuple[nn.Module, nn.Module]] = []
    for i in range(n_games):
        if current_is_black[i]:
            game_models.append((current_model, opponent_models[i]))
        else:
            game_models.append((opponent_models[i], current_model))

    while n_active > 0:
        # Partition active games by which model moves next
        # model_id -> (model, game_indices, is_current_list)
        model_groups: dict[int, tuple[nn.Module, list[int], list[bool]]] = {}

        for i in range(n_games):
            if not active_mask[i]:
                continue

            who = boards[i].who_to_play
            if who == Player.BLACK:
                model = game_models[i][0]
                is_current = current_is_black[i]
            else:
                model = game_models[i][1]
                is_current = not current_is_black[i]

            mid = id(model)
            if mid not in model_groups:
                model_groups[mid] = (model, [], [])
            model_groups[mid][1].append(i)
            model_groups[mid][2].append(is_current)

        # Run MCTS for each model group
        actions_for_game: dict[int, int] = {}

        for model, game_indices, is_current_flags in model_groups.values():
            group_boards = [boards[i] for i in game_indices]

            visit_dists, root_values = mcts_search_batched(
                model, group_boards, num_simulations, c_puct,
                prior_temperature, device,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
            )

            for j, (i, is_current) in enumerate(zip(game_indices, is_current_flags)):
                # Record training data for current model's moves
                if is_current:
                    c0, c1, _ = boards[i].GetBoardState()
                    obs = encode_observation(c0, c1)
                    records[i].observations.append(obs)
                    records[i].visit_distributions.append(visit_dists[j])
                    records[i].root_values.append(float(root_values[j]))

                # Sample action from visit distribution
                dist = visit_dists[j]
                if action_temperature != 1.0:
                    # Sharpen/flatten visit distribution for action selection
                    dist = dist ** (1.0 / action_temperature)
                    dist = dist / dist.sum()
                action = np.random.choice(225, p=dist)

                records[i].actions.append(action)
                records[i].players.append(boards[i].who_to_play)
                actions_for_game[i] = action

        # Apply actions
        for i in range(n_games):
            if not active_mask[i]:
                continue

            action = actions_for_game[i]
            row, col = idx_to_pos(action)
            outcome = boards[i].Move((row, col))

            if outcome != GameState.CONTINUE:
                records[i].outcome = outcome
                active_mask[i] = False
                n_active -= 1

    return records
