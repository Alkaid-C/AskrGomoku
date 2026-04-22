"""
MCTS Self-Play Game Generation

Pure self-play: the current model plays both sides. Every position becomes
a training sample.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from gomoku import (
    GameState,
    GomokuBoard,
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
    observations: list[np.ndarray] = field(default_factory=list)        # [3, 15, 15] per move, side-to-move perspective
    visit_distributions: list[np.ndarray] = field(default_factory=list) # [225] normalized
    root_values: list[float] = field(default_factory=list)              # MCTS root Q, side-to-move perspective
    actions: list[int] = field(default_factory=list)
    outcome: GameState = GameState.CONTINUE


# ============================================================================
# MCTS Game Generation
# ============================================================================


def play_mcts_games(
    model: nn.Module,
    num_games: int,
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
    Play pure-self-play MCTS games with batched search.

    The same model plays both sides in every game. All active games are
    batched into a single MCTS call per ply, and every position is recorded
    as a training sample.

    Args:
        model: Policy/value network being trained
        num_games: Number of concurrent games to play
        num_simulations: MCTS simulations per move
        c_puct: PUCT exploration constant
        prior_temperature: Temperature for softening prior in MCTS
        device: Torch device
        opening_ids: Per-game opening ID (-1 for empty board)
        dirichlet_alpha: Dirichlet noise alpha (root only)
        dirichlet_epsilon: Dirichlet noise weight (root only)
        action_temperature: Temperature for sampling actions from visit counts

    Returns:
        List of MCTSGameRecord with training data from every ply.
    """
    assert len(opening_ids) == num_games

    boards: list[GomokuBoard] = [GomokuBoard(opening_id=opening_ids[i]) for i in range(num_games)]
    records: list[MCTSGameRecord] = [MCTSGameRecord() for _ in range(num_games)]

    active_indices = list(range(num_games))

    while active_indices:
        active_boards = [boards[i] for i in active_indices]

        visit_dists, root_values = mcts_search_batched(
            model, active_boards, num_simulations, c_puct,
            prior_temperature, device,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        still_active: list[int] = []
        for j, i in enumerate(active_indices):
            c0, c1, _ = boards[i].GetBoardState()
            obs = encode_observation(c0, c1)
            records[i].observations.append(obs)
            records[i].visit_distributions.append(visit_dists[j])
            records[i].root_values.append(float(root_values[j]))

            dist = visit_dists[j]
            if action_temperature != 1.0:
                dist = dist ** (1.0 / action_temperature)
                dist = dist / dist.sum()
            action = int(np.random.choice(225, p=dist))

            records[i].actions.append(action)

            row, col = idx_to_pos(action)
            outcome = boards[i].Move((row, col))
            if outcome != GameState.CONTINUE:
                records[i].outcome = outcome
            else:
                still_active.append(i)

        active_indices = still_active

    return records
