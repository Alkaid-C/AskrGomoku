"""
MCTS Self-Play Game Generation

Pure self-play: the current model plays both sides. Every position becomes
a training sample.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from enhancement import find_all_win_in_1, find_blocking_moves
from gomoku import (
    LOGIT_MASK_VALUE,
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
    raw_entropy: list[float] = field(default_factory=list)              # entropy of model's masked-softmax prior at root, pre-Dirichlet (diagnostic)
    outcome: Optional[GameState] = None


# ============================================================================
# MCTS Game Generation
# ============================================================================


def play_mcts_games(
    model: nn.Module,
    num_games: int,
    num_simulations: int,
    c_puct: float,
    entropy_multiplier: Optional[float],
    device: torch.device,
    opening_ids: list[int],
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    gamma: float,
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
        entropy_multiplier: When set, per-position prior is rescaled to entropy
            H(softmax(logits)) * entropy_multiplier. When None, priors are the
            raw masked softmax (vanilla AlphaZero).
        device: Torch device
        opening_ids: Per-game opening ID (-1 for empty board)
        dirichlet_alpha: Dirichlet noise alpha (root only)
        dirichlet_epsilon: Dirichlet noise weight (root only)
        gamma: Per-ply MCTS backup discount (see mcts.py::backup).
        action_temperature: Temperature applied to the visit distribution
            *only* at action sampling time. The supervision target recorded
            into the MCTSGameRecord is the original visit distribution; this
            broadens trajectory coverage without altering targets.

    Returns:
        List of MCTSGameRecord with training data from every ply.
    """
    assert len(opening_ids) == num_games

    boards: list[GomokuBoard] = [GomokuBoard(opening_id=opening_ids[i]) for i in range(num_games)]
    records: list[MCTSGameRecord] = [MCTSGameRecord() for _ in range(num_games)]

    active_indices = list(range(num_games))

    while active_indices:
        active_boards = [boards[i] for i in active_indices]

        visit_dists, root_values, raw_entropies = mcts_search_batched(
            model, active_boards, num_simulations, c_puct,
            entropy_multiplier, device,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            gamma=gamma,
        )

        still_active: list[int] = []
        for j, i in enumerate(active_indices):
            c0, c1, _ = boards[i].GetBoardState()
            obs = encode_observation(c0, c1)
            records[i].observations.append(obs)
            records[i].visit_distributions.append(visit_dists[j])
            records[i].root_values.append(float(root_values[j]))
            records[i].raw_entropy.append(float(raw_entropies[j]))

            if action_temperature == 1.0:
                sample_dist = visit_dists[j]
            else:
                sample_dist = np.maximum(visit_dists[j], 1e-30) ** (1.0 / action_temperature)
                sample_dist = sample_dist / sample_dist.sum()
            action = int(np.random.choice(225, p=sample_dist))

            row, col = idx_to_pos(action)
            outcome = boards[i].Move((row, col))
            if outcome != GameState.CONTINUE:
                records[i].outcome = outcome
            else:
                still_active.append(i)

        active_indices = still_active

    return records


# ============================================================================
# Block-Win-in-1 Hit Rate Diagnostic
# ============================================================================


def compute_block_rates(
    records: list[MCTSGameRecord],
    model: nn.Module,
    device: torch.device,
) -> dict:
    """
    Blocking-move mass diagnostic, split by side-to-move.

    For positions where the opponent has a single blockable win-in-1 threat
    AND the current player has no winning move of their own, measure the
    probability the player's action distribution assigns to any blocking
    move, averaged per-position. The win-in-1 filter matches the RL pipeline's
    tactical_boost precedence (win > block) — otherwise dual-threat positions
    where the player (correctly) plays the winning move would be mis-counted
    as block misses.

    - MCTS rate: mass under the visit distribution the game samples from.
    - Raw rate:  mass under softmax of raw policy logits with illegal moves
                 masked out (what a stochastic deploy would sample).

    Positions with dual unblockable threats are also skipped.
    """
    obs_list, bmasks, mcts_mass, is_white = [], [], [], []
    for record in records:
        for obs, vd in zip(record.observations, record.visit_distributions):
            legal = ((obs[0] == 0) & (obs[1] == 0)).astype(np.int32)
            if find_all_win_in_1(obs, legal):
                continue
            blocking = find_blocking_moves(obs, legal)
            if blocking is None:
                continue
            bmask = np.zeros(225, dtype=np.float32)
            bmask[blocking] = 1.0
            obs_list.append(obs)
            bmasks.append(bmask)
            mcts_mass.append(float(vd[blocking].sum()))
            is_white.append(int(obs[0].sum() + obs[1].sum()) % 2 == 1)

    results: dict = {f"{s}_block_{k}": v for s in ("black", "white")
                     for k, v in (("opps", 0), ("mcts_rate", float("nan")), ("raw_rate", float("nan")))}
    if not obs_list:
        return results

    obs_t = torch.from_numpy(np.stack(obs_list)).float().to(device)
    bmask_t = torch.from_numpy(np.stack(bmasks)).to(device)
    with torch.no_grad():
        legal_t = (obs_t[:, 0] + obs_t[:, 1] == 0).view(-1, 225)
        logits = model.forward_policy_only(obs_t).view(-1, 225).masked_fill(~legal_t, LOGIT_MASK_VALUE)
        raw_mass = (torch.softmax(logits, dim=-1) * bmask_t).sum(dim=-1).cpu().numpy()

    mcts_arr = np.asarray(mcts_mass)
    white_sel = np.asarray(is_white)
    for side, sel in (("black", ~white_sel), ("white", white_sel)):
        results[f"{side}_block_opps"] = int(sel.sum())
        if sel.any():
            results[f"{side}_block_mcts_rate"] = float(mcts_arr[sel].mean())
            results[f"{side}_block_raw_rate"] = float(raw_mass[sel].mean())
    return results
