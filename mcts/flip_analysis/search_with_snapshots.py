"""
MCTS search with per-simulation significant-preference-flip tracking.

A self-contained copy of ``mcts.mcts_search_batched``'s simulation loop,
specialized for the top-1-flip study: no Dirichlet noise, ``entropy_multiplier``
is always None (raw masked-softmax priors), no subtree harvesting. After every
simulation it reads each root's ``child_n`` and updates a committed winner
using a full-distribution snapshot as the current preference baseline.

Flip definition:
  * The initial committed winner is alpha = argmax(raw prior), and the initial
    baseline is the complete raw-prior distribution.
  * At sim t, candidates are all moves tied for the largest visit count. For
    candidate c and committed winner w, the preference displacement since the
    last confirmed flip is

        delta(c, w) = (v_t(c) - v_t(w)) - (baseline(c) - baseline(w)).

  * The best top candidate replaces w iff delta >= margin. Exact visit ties are
    eligible: catching up from a sufficiently large baseline deficit is itself
    a significant preference change.
  * On every confirmed flip the complete visit distribution becomes the new
    baseline. Thus every later flip must accumulate a fresh margin of relative
    movement; hovering around an old threshold cannot produce repeated flips.
  * Flip tracking starts at the first simulation t for which 1/t < margin.
    Before then a single root visit is itself at least one margin wide, so
    visit-count quantization would turn ordinary early alternation into flips.
    The search still runs normally during this burn-in; the committed winner
    and raw-prior baseline simply remain unchanged.

Probabilities are compared at temperature 1: the raw prior is the masked softmax
of the logits, the search distribution is the normalized visit fraction N_i / sum(N).

Reuses ``_evaluate_with_cache``, ``copy_board`` and ``MCTSNode`` from ``mcts`` so
the leaf evaluation, expansion and backup match production exactly; ``mcts.py``
itself is left untouched.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from gomoku import GameState, GomokuBoard, encode_observation, idx_to_pos
from mcts_ext import MCTSNode

from mcts import _evaluate_with_cache, copy_board

_FLOAT_COMPARE_ATOL = 8 * np.finfo(np.float32).eps


@dataclass
class FlipResult:
    """Per-game output of one snapshotting search."""
    alpha: int                       # raw-prior argmax (network top-1)
    beta_final: int                  # instantaneous visit argmax at total_sims
    winner_final: int                # committed snapshot-reset winner at total_sims
    flipped: bool                    # committed winner at total_sims != alpha
    flip_tracking_start_sim: int     # first sim at which flips can be confirmed
    flip_sims: list[int] = field(default_factory=list)  # sim index of every flip event
    flip_events: list[dict[str, int | float]] = field(default_factory=list)
    p_raw_alpha: float = 0.0         # raw prob of alpha
    p_raw_beta: float = 0.0          # raw prob of instantaneous beta_final
    p_search_alpha: float = 0.0      # search prob of alpha at total_sims
    p_search_beta: float = 0.0       # search prob of instantaneous beta_final
    p_raw_winner: float = 0.0        # raw prob of committed winner_final
    p_search_winner: float = 0.0     # final search prob of committed winner_final
    raw_entropy: float = 0.0         # entropy of the raw prior
    raw_value: float = 0.0           # model value at root (side-to-move perspective)
    mcts_value: float = 0.0          # MCTS root Q at total_sims
    raw_mcts_kl_action: float = 0.0  # KL(visits_action_sim || raw prior)
    raw_mcts_kl_final: float = 0.0   # KL(visits_total_sim  || raw prior)
    dist_action: np.ndarray = field(default_factory=lambda: np.zeros(225, np.float32))  # visit dist at action_sim


def _visit_dist_full(child_actions: np.ndarray, ns: np.ndarray, total: int) -> np.ndarray:
    dist = np.zeros(225, dtype=np.float32)
    dist[child_actions] = ns.astype(np.float32) / total
    return dist


def _kl_to_raw(dist: np.ndarray, prior: np.ndarray) -> float:
    return float((dist * (np.log(dist + 1e-30) - np.log(prior + 1e-30))).sum())


def _flip_tracking_start_sim(margin: float) -> int:
    """First integer t whose visit-probability resolution is below margin."""
    if margin <= 0:
        raise ValueError(f"margin must be positive, got {margin}")
    return int(np.floor(1.0 / margin)) + 1


def _significant_top_candidate(
    ns: np.ndarray,
    total: int,
    baseline: np.ndarray,
    winner_pos: int,
    margin: float,
) -> tuple[int, float] | None:
    """Return a newly qualified top candidate and its preference displacement.

    Arrays use the root's compact ``child_actions`` ordering. All candidates
    tied for the current largest visit count are considered. If several clear
    the threshold, choose the one whose pairwise preference relative to the
    committed winner improved the most since the last baseline snapshot.
    """
    max_n = int(ns.max())
    top_positions = np.flatnonzero(ns == max_n)
    candidate_positions = top_positions[top_positions != winner_pos]
    if candidate_positions.size == 0:
        return None

    current_relative = (
        ns[candidate_positions].astype(np.float64) - float(ns[winner_pos])
    ) / total
    baseline_relative = (
        baseline[candidate_positions] - float(baseline[winner_pos])
    )
    displacements = current_relative - baseline_relative
    best_k = int(np.argmax(displacements))
    best_displacement = float(displacements[best_k])
    if best_displacement + _FLOAT_COMPARE_ATOL < margin:
        return None
    return int(candidate_positions[best_k]), best_displacement


def mcts_search_with_snapshots(
    model: nn.Module,
    boards: list[GomokuBoard],
    total_sims: int,
    action_sim: int,
    c_puct: float,
    gamma: float,
    fpu_multiplier: float,
    margin: float,
    device: torch.device,
) -> list[FlipResult]:
    """Run MCTS on each board, tracking snapshot-reset preference flips.

    All boards are searched simultaneously (one batched call), each to
    ``total_sims`` simulations. The visit distribution at ``action_sim`` is
    captured (for the caller to sample the played move from), and the flip
    trajectory over the whole [1, total_sims] range is recorded.

    Returns one FlipResult per input board.
    """
    assert 1 <= action_sim <= total_sims
    flip_tracking_start_sim = _flip_tracking_start_sim(margin)
    n_games = len(boards)

    # --- Root initialization (no Dirichlet, raw masked-softmax priors) ---
    obs_list = []
    legal_masks = []
    for board in boards:
        c0, c1, _ = board.GetBoardState()
        obs_list.append(encode_observation(c0, c1))
        legal_mask, _ = board.GetLegalMoves()
        legal_masks.append(legal_mask)

    priors, root_node_values = _evaluate_with_cache(model, obs_list, None, device)

    roots: list[MCTSNode] = []
    # Per-game committed flip state. Arrays are in each root's compact
    # child_actions ordering.
    alpha = np.empty(n_games, dtype=np.int64)
    child_actions_list: list[np.ndarray] = []
    winner_pos = np.empty(n_games, dtype=np.int64)
    baselines: list[np.ndarray] = []
    flip_sims: list[list[int]] = [[] for _ in range(n_games)]
    flip_events: list[list[dict[str, int | float]]] = [[] for _ in range(n_games)]

    for i in range(n_games):
        root = MCTSNode()
        root.visit_count = 1  # virtual visit so PUCT uses priors on first selection

        legal_flat = legal_masks[i].reshape(225).astype(bool)
        legal_indices = np.where(legal_flat)[0]
        final_priors = priors[i][legal_indices].astype(np.float32)
        root.expand(
            legal_indices.tolist(),
            final_priors.tolist(),
            float(root_node_values[i]) * fpu_multiplier,
        )
        roots.append(root)

        ca = np.asarray(root.child_actions, dtype=np.int64)
        child_actions_list.append(ca)
        a = int(np.argmax(priors[i]))
        alpha[i] = a
        winner_pos[i] = int(np.where(ca == a)[0][0])
        baselines.append(priors[i][ca].astype(np.float64))

    # Captured visit distribution at action_sim.
    dist_action: list[np.ndarray] = [np.zeros(225, np.float32) for _ in range(n_games)]

    # --- Simulation loop (mirrors mcts.mcts_search_batched) ---
    for sim in range(1, total_sims + 1):
        leaves: list[MCTSNode] = []
        action_paths: list[list[int]] = []
        for i in range(n_games):
            node = roots[i]
            path: list[int] = []
            while node.is_expanded and not node.is_terminal:
                node = node.select_child(c_puct)
                path.append(node.action)
            leaves.append(node)
            action_paths.append(path)

        eval_indices: list[int] = []
        eval_obs: list[np.ndarray] = []
        eval_legal: list[np.ndarray] = []
        for i, leaf in enumerate(leaves):
            if leaf.is_terminal:
                leaf.backup(-leaf.terminal_value, gamma)
                continue
            assert not leaf.is_expanded
            board_copy = copy_board(boards[i])
            terminal = False
            for action in action_paths[i]:
                row, col = idx_to_pos(action)
                outcome = board_copy.Move((row, col))
                if outcome != GameState.CONTINUE:
                    leaf.is_terminal = True
                    leaf.terminal_value = 0.0 if outcome == GameState.DRAW else 1.0
                    leaf.backup(-leaf.terminal_value, gamma)
                    terminal = True
                    break
            if terminal:
                continue
            c0, c1, _ = board_copy.GetBoardState()
            eval_obs.append(encode_observation(c0, c1))
            legal_mask, _ = board_copy.GetLegalMoves()
            eval_legal.append(legal_mask)
            eval_indices.append(i)

        if eval_indices:
            leaf_priors, leaf_values = _evaluate_with_cache(model, eval_obs, None, device)
            for j, i in enumerate(eval_indices):
                leaf = leaves[i]
                prior_j = leaf_priors[j]
                legal_actions = np.where(eval_legal[j].reshape(225))[0]
                leaf.expand(
                    legal_actions.tolist(),
                    prior_j[legal_actions].astype(np.float32).tolist(),
                    float(leaf_values[j]) * fpu_multiplier,
                )
                leaf.backup(leaf_values[j], gamma)

        # --- Snapshot: update each game's committed winner ---
        total = sim  # sum(child_n) == sim for every root
        for i in range(n_games):
            ns = np.asarray(roots[i].child_n, dtype=np.int64)
            ca = child_actions_list[i]
            if sim < flip_tracking_start_sim:
                continue
            qualified = _significant_top_candidate(
                ns, total, baselines[i], int(winner_pos[i]), margin
            )
            if qualified is not None:
                new_winner_pos, displacement = qualified
                old_winner = int(ca[winner_pos[i]])
                new_winner = int(ca[new_winner_pos])
                baseline_relative = float(
                    baselines[i][new_winner_pos] - baselines[i][winner_pos[i]]
                )
                current_relative = float(
                    (ns[new_winner_pos] - ns[winner_pos[i]]) / total
                )
                flip_sims[i].append(sim)
                flip_events[i].append({
                    "sim": sim,
                    "from": old_winner,
                    "to": new_winner,
                    "baseline_relative": baseline_relative,
                    "current_relative": current_relative,
                    "relative_shift": displacement,
                })
                winner_pos[i] = new_winner_pos
                # A confirmed flip consumes the accumulated movement. Every
                # candidate's new baseline is the full current visit dist.
                baselines[i] = ns.astype(np.float64) / total

        if sim == action_sim:
            for i in range(n_games):
                ns = np.asarray(roots[i].child_n, dtype=np.int64)
                dist_action[i] = _visit_dist_full(child_actions_list[i], ns, total)

    # --- Final extraction (at total_sims) ---
    results: list[FlipResult] = []
    for i in range(n_games):
        ns = np.asarray(roots[i].child_n, dtype=np.int64)
        qs = np.asarray(roots[i].child_q, dtype=np.float32)
        ca = child_actions_list[i]
        total = total_sims
        dist_final = _visit_dist_full(ca, ns, total)
        root_q = float((ns.astype(np.float32) * qs).sum() / total)

        beta_pos = int(np.argmax(ns))
        beta = int(ca[beta_pos])
        committed_winner = int(ca[winner_pos[i]])
        prior_i = priors[i]
        raw_entropy = float(-(prior_i * np.log(prior_i + 1e-30)).sum())

        results.append(FlipResult(
            alpha=int(alpha[i]),
            beta_final=beta,
            winner_final=committed_winner,
            flipped=(committed_winner != int(alpha[i])),
            flip_tracking_start_sim=flip_tracking_start_sim,
            flip_sims=flip_sims[i],
            flip_events=flip_events[i],
            p_raw_alpha=float(prior_i[int(alpha[i])]),
            p_raw_beta=float(prior_i[beta]),
            p_search_alpha=float(dist_final[int(alpha[i])]),
            p_search_beta=float(dist_final[beta]),
            p_raw_winner=float(prior_i[committed_winner]),
            p_search_winner=float(dist_final[committed_winner]),
            raw_entropy=raw_entropy,
            raw_value=float(root_node_values[i]),
            mcts_value=root_q,
            raw_mcts_kl_action=_kl_to_raw(dist_action[i], prior_i),
            raw_mcts_kl_final=_kl_to_raw(dist_final, prior_i),
            dist_action=dist_action[i],
        ))
    return results
