"""
MCTS (Monte Carlo Tree Search) with PUCT and Batched Leaf Evaluation

Implements PUCT-based tree search with neural network evaluation for Gomoku.
Supports batched search across multiple game positions simultaneously.
"""

import math
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from entropy_ops import rescale_to_entropy_np, softmax_np
from gomoku import GameState, GomokuBoard, encode_observation, idx_to_pos
from mcts_ext import MCTSNode

# ============================================================================
# D4 Canonicalization & NN Eval Cache
# ============================================================================

# Forward permutation table: _FORWARD_PERM[s, old_flat] = new_flat.
# Coordinate convention matches main/enhancement.py:268-269 (and the inverse
# table in mcts/training.py); both must stay in sync.
def _build_forward_perm() -> np.ndarray:
    coords = np.arange(225)
    r = coords // 15
    c = coords % 15
    new_rows = np.stack([r, c, 14 - r, 14 - c, r, 14 - r, c, 14 - c])
    new_cols = np.stack([c, 14 - r, 14 - c, r, 14 - c, c, r, 14 - r])
    return (new_rows * 15 + new_cols).astype(np.int64)  # [8, 225]


_FORWARD_PERM = _build_forward_perm()

# Spatial transforms on uint8 obs of shape [3, 15, 15], indexed by the same
# `s` ordering as _FORWARD_PERM and augment_batch_8fold.
_SPATIAL_TRANSFORMS: list[Callable[[np.ndarray], np.ndarray]] = [
    lambda o: o,                                 # s=0: identity
    lambda o: o.transpose(0, 2, 1)[:, :, ::-1],  # s=1: rot90
    lambda o: o[:, ::-1, ::-1],                  # s=2: rot180
    lambda o: o.transpose(0, 2, 1)[:, ::-1, :],  # s=3: rot270
    lambda o: o[:, :, ::-1],                     # s=4: flip-H
    lambda o: o[:, ::-1, :],                     # s=5: flip-V
    lambda o: o.transpose(0, 2, 1),              # s=6: transpose
    lambda o: o[:, ::-1, ::-1].transpose(0, 2, 1),  # s=7: anti-transpose
]

# Fixed chunk size with zero-padding: uniform shapes prevent allocator fragmentation and hit the throughput sweet spot (see CLAUDE.md).
_FIXED_FWD_BATCH = 64

# Cache maps canonical-obs bytes -> (canonical_priors_scaled raw bytes, value).
# Priors are stored as raw float32 bytes (900 B) rather than as an ndarray, to
# strip ~120 B of numpy header overhead per entry; reconstructed via
# `np.frombuffer` on hit (zero-copy, read-only view; permutation indexing
# produces a fresh writable array).
#
# `canonical_priors_scaled` is the post-mask, post-softmax, post-entropy-rescale
# distribution in canonical orientation (illegal squares are exactly 0). Why
# caching the scaled distribution is sound, and the per-stage cache-clearing
# rules, are in mcts/CLAUDE.md, "D4-canonical NN eval cache".
# LRU eviction caps memory; OrderedDict preserves insertion/access order.
_NN_EVAL_CACHE_MAX_ENTRIES = 1024 * 1024 * 8
_NN_EVAL_CACHE: "OrderedDict[bytes, tuple[bytes, float]]" = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0


def canonicalize_obs(obs: np.ndarray) -> tuple[bytes, int]:
    """Return (canonical_bytes, s) where s is the transform that produced it.

    Picks the lexicographically smallest of the 8 D4 transforms of `obs`.
    The key packs the two stone planes into bits (450 bits → 57 bytes);
    channel 2 is the all-1s board mask and is omitted as it carries no
    position information.
    """
    best_key: Optional[bytes] = None
    best_s = 0
    for s in range(8):
        transformed = _SPATIAL_TRANSFORMS[s](obs)
        key = np.packbits(np.ascontiguousarray(transformed[:2]).ravel()).tobytes()
        if best_key is None or key < best_key:
            best_key = key
            best_s = s
    assert best_key is not None
    return best_key, best_s


def get_nn_eval_cache_stats() -> tuple[int, int]:
    """Return (hits, misses) accumulated since the last clear."""
    return _CACHE_HITS, _CACHE_MISSES


def get_nn_eval_cache_size() -> int:
    return len(_NN_EVAL_CACHE)


def clear_nn_eval_cache() -> None:
    global _CACHE_HITS, _CACHE_MISSES
    _NN_EVAL_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


# MCTSNode is implemented in C++ (mcts_ext.MCTSNode) for fast PUCT selection
# and backup. Imported at the top of this file.

# ============================================================================
# Board Cloning
# ============================================================================


def copy_board(board: GomokuBoard) -> GomokuBoard:
    """Lightweight clone of a GomokuBoard (avoids deepcopy overhead)."""
    new = object.__new__(GomokuBoard)
    new.black_pieces = board.black_pieces.copy()
    new.white_pieces = board.white_pieces.copy()
    new.who_to_play = board.who_to_play
    new.occupied_count = board.occupied_count
    return new


# ============================================================================
# Cache-aware batched evaluation
# ============================================================================


def _evaluate_with_cache(
    model: nn.Module,
    obs_list: list[np.ndarray],
    entropy_multiplier: Optional[float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up `obs_list` in the NN cache, evaluate misses on GPU, post-process
    on CPU, and return per-obs scaled priors and values in original orientation.

    The cache stores fully-scaled canonical priors (mask + softmax + entropy
    rescale already applied), so cache hits collapse to a permutation lookup.
    Only misses pay for the model forward + entropy rescale.

    When `entropy_multiplier` is None, priors are the masked softmax of raw
    logits (no entropy rescale).

    Returns:
        priors: [N, 225] float32, illegal squares = 0, in original orientation.
        values: [N] float32, model value head (= side-to-move's perspective).
    """
    global _CACHE_HITS, _CACHE_MISSES
    n = len(obs_list)
    keys_and_s = [canonicalize_obs(o) for o in obs_list]
    miss_indices: list[int] = []
    miss_canonical_obs: list[np.ndarray] = []
    pending_keys: set[bytes] = set()
    for i, (key, s) in enumerate(keys_and_s):
        if key in _NN_EVAL_CACHE or key in pending_keys:
            continue
        pending_keys.add(key)
        miss_indices.append(i)
        miss_canonical_obs.append(
            np.ascontiguousarray(_SPATIAL_TRANSFORMS[s](obs_list[i]))
        )
    _CACHE_HITS += n - len(miss_indices)
    _CACHE_MISSES += len(miss_indices)

    if miss_indices:
        M = len(miss_indices)
        m_obs_np = np.stack(miss_canonical_obs)  # [M, 3, 15, 15] uint8, canonical

        # Forward in chunks of _FIXED_FWD_BATCH; the trailing chunk is padded
        # with zeros so every model call sees the same input shape. Padded
        # rows are sliced off after concat — they never enter the cache.
        logits_chunks: list[np.ndarray] = []
        values_chunks: list[np.ndarray] = []
        for s in range(0, M, _FIXED_FWD_BATCH):
            chunk_np = m_obs_np[s:s + _FIXED_FWD_BATCH]
            if chunk_np.shape[0] < _FIXED_FWD_BATCH:
                pad = np.zeros(
                    (_FIXED_FWD_BATCH - chunk_np.shape[0], 3, 15, 15),
                    dtype=chunk_np.dtype,
                )
                chunk_np = np.concatenate([chunk_np, pad], axis=0)
            chunk_t = torch.from_numpy(chunk_np).float().to(device)
            with torch.inference_mode():
                chunk_logits, chunk_values = model(chunk_t)
            logits_chunks.append(
                chunk_logits.squeeze(1).view(_FIXED_FWD_BATCH, 225).cpu().numpy()
            )
            values_chunks.append(chunk_values.squeeze(-1).cpu().numpy())
        m_logits_np = np.concatenate(logits_chunks, axis=0)[:M]
        m_values_np = np.concatenate(values_chunks, axis=0)[:M]

        # Canonical legal mask is implied by the canonical obs: any square
        # occupied in either stone plane is illegal.
        canonical_legal = ~(
            (m_obs_np[:, 0] | m_obs_np[:, 1]).reshape(-1, 225).astype(bool)
        )
        masked = np.where(canonical_legal, m_logits_np, -1e9).astype(np.float32)
        if entropy_multiplier is None:
            canonical_priors_scaled = softmax_np(masked, axis=-1).astype(np.float32)
        else:
            natural = softmax_np(masked, axis=-1)
            H_model = -(natural * np.log(natural + 1e-30)).sum(axis=-1)
            target_H = np.minimum(
                H_model * entropy_multiplier, math.log(225)
            ).astype(np.float32)
            canonical_priors_scaled = rescale_to_entropy_np(masked, target_H)

        for k, i in enumerate(miss_indices):
            key = keys_and_s[i][0]
            _NN_EVAL_CACHE[key] = (
                np.ascontiguousarray(canonical_priors_scaled[k], dtype=np.float32).tobytes(),
                float(m_values_np[k]),
            )

    priors = np.empty((n, 225), dtype=np.float32)
    values = np.empty(n, dtype=np.float32)
    for i, (key, s) in enumerate(keys_and_s):
        canonical_priors_bytes, value = _NN_EVAL_CACHE[key]
        _NN_EVAL_CACHE.move_to_end(key)
        canonical_priors = np.frombuffer(canonical_priors_bytes, dtype=np.float32)
        priors[i] = canonical_priors[_FORWARD_PERM[s]]
        values[i] = value

    # Evict only after every key in this batch has been marked recently-used.
    # Doing this before the lookup loop could pop hits at the LRU front that
    # this batch still needs to read.
    while len(_NN_EVAL_CACHE) > _NN_EVAL_CACHE_MAX_ENTRIES:
        _NN_EVAL_CACHE.popitem(last=False)

    return priors, values


# ============================================================================
# Root Dirichlet noise neighborhood
# ============================================================================

# Chebyshev radius around existing stones inside which root Dirichlet noise is
# applied. Sized to the tactical horizon of a single stone (a five-in-a-row uses
# neighbors up to 4 squares away in any direction — a board-fixed bound, so the
# radius should not exceed it). Why the support is narrowed at all: see
# mcts/CLAUDE.md, "Dirichlet noise".
_DIRICHLET_NEIGHBORHOOD_RADIUS = 4


def _stone_neighborhood_mask(c0: np.ndarray, c1: np.ndarray, radius: int) -> np.ndarray:
    """15x15 bool mask of squares within `radius` (Chebyshev) of any stone.

    Returns all-False on an empty board; the caller falls back to the legal
    mask in that case.
    """
    occupied = (c0 | c1).astype(bool)
    if not occupied.any():
        return np.zeros_like(occupied, dtype=bool)
    H, W = occupied.shape
    pad = np.pad(occupied, radius, mode="constant")
    out = np.zeros((H, W), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            out |= pad[dr : dr + H, dc : dc + W]
    return out


# ============================================================================
# Batched MCTS Search
# ============================================================================


def mcts_search_batched(
    model: nn.Module,
    boards: list[GomokuBoard],
    num_simulations: int,
    c_puct: float,
    entropy_multiplier: Optional[float],
    device: torch.device,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    gamma: float,
    fpu_multiplier: float,
    harvest_min_visits: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[tuple]]]:
    """
    Run batched MCTS search on multiple board positions.

    Args:
        model: Neural network for policy/value evaluation
        boards: List of N board positions to search from
        num_simulations: Number of MCTS simulations per position
        c_puct: PUCT exploration constant
        entropy_multiplier: When set, per-position prior entropy is rescaled to
            H(softmax(logits)) * entropy_multiplier. When None, priors are the
            masked softmax of the raw logits (vanilla AlphaZero).
        device: Torch device
        dirichlet_alpha: Dirichlet noise concentration (per-axis).
        dirichlet_epsilon: Dirichlet noise mixing weight (0 = no noise). When
            > 0, noise support is restricted to legal moves within
            _DIRICHLET_NEIGHBORHOOD_RADIUS (Chebyshev) of an existing stone;
            on an empty board it falls back to all legal moves.
        gamma: Per-ply discount in backup; backed-up Q carries a gamma^depth
            factor, so deeper terminals contribute less than shallow ones.
        fpu_multiplier: First Play Urgency scale. A not-yet-visited child is
            given Q = (node's NN value) * fpu_multiplier until its first backup.
            0 reproduces the legacy neutral-0 FPU (over-optimistic in losing
            positions, causes a uniform sweep of all legal moves); values near 1
            anchor untried moves to the node's own value so breadth stays
            prior-weighted.
        harvest_min_visits: When set, walk each search tree and emit every
            internal node (depth >= 1) whose visit statistic N = sum(child_n)
            clears this threshold as an additional training sample. When None,
            no harvesting is done and the returned per-game lists are empty.

    Returns:
        visit_distributions: [N, 225] normalized visit counts
        root_values: [N] weighted mean Q from root's perspective
        raw_entropies: [N] entropy of the masked-softmax prior at the root,
            pre-Dirichlet. For logging/diagnostics.
        raw_mcts_kls: [N] KL(visit_dist || raw prior) at the root, i.e. how far
            the search moved the policy away from the network's raw prior
            (pre-Dirichlet) — the policy-improvement gap. For logging/diagnostics.
        harvested: per-game list of (obs uint8[3,15,15], policy f32[225],
            value f32, N int) tuples for the internal nodes harvested from that
            game's search tree. Empty lists when harvest_min_visits is None.
    """
    n_games = len(boards)

    # --- Root initialization: batch forward pass ---
    obs_list = []
    legal_masks = []
    neighborhood_masks = []
    for board in boards:
        c0, c1, _ = board.GetBoardState()
        obs_list.append(encode_observation(c0, c1))
        legal_mask, _ = board.GetLegalMoves()
        legal_masks.append(legal_mask)
        neighborhood_masks.append(
            _stone_neighborhood_mask(c0, c1, _DIRICHLET_NEIGHBORHOOD_RADIUS)
        )

    priors, root_node_values = _evaluate_with_cache(
        model, obs_list, entropy_multiplier, device
    )

    # Create root nodes and expand
    roots: list[MCTSNode] = []
    for i in range(n_games):
        root = MCTSNode()
        root.visit_count = 1  # virtual visit so PUCT uses priors on first selection

        legal_flat = legal_masks[i].reshape(225).astype(bool)
        prior_i = priors[i]

        # Add Dirichlet noise (skipped when epsilon=0; alpha=0 would NaN out).
        # Noise is restricted to legal moves within _DIRICHLET_NEIGHBORHOOD_RADIUS
        # (Chebyshev) of an existing stone; on empty boards fall back to all
        # legal moves. Positions outside the noise support keep their plain
        # (1 - ε) · P mass; the ε mass is distributed over the noise support
        # only. Total over legal_indices still sums to 1.
        legal_indices = np.where(legal_flat)[0]
        if dirichlet_epsilon > 0:
            near_flat = neighborhood_masks[i].reshape(225) & legal_flat
            if not near_flat.any():
                near_flat = legal_flat
            noise_indices = np.where(near_flat)[0]
            noise = np.random.dirichlet([dirichlet_alpha] * len(noise_indices))
            noise_full = np.zeros(225, dtype=np.float32)
            noise_full[noise_indices] = noise.astype(np.float32)
            final_priors = (
                (1 - dirichlet_epsilon) * prior_i[legal_indices]
                + dirichlet_epsilon * noise_full[legal_indices]
            ).astype(np.float32)
        else:
            final_priors = prior_i[legal_indices].astype(np.float32)
        root.expand(
            legal_indices.tolist(),
            final_priors.tolist(),
            float(root_node_values[i]) * fpu_multiplier,
        )

        roots.append(root)

    # --- Simulation loop ---
    for _ in range(num_simulations):
        # Phase 1: Select leaves for all games
        leaves: list[MCTSNode] = []
        action_paths: list[list[int]] = []

        for i in range(n_games):
            node = roots[i]
            path = []

            # Traverse down tree using PUCT
            while node.is_expanded and not node.is_terminal:
                node = node.select_child(c_puct)
                path.append(node.action)

            leaves.append(node)
            action_paths.append(path)

        # Phase 2: Evaluate non-terminal, unexpanded leaves
        eval_indices = []  # indices into leaves that need NN eval
        eval_obs = []
        eval_legal = []

        for i, leaf in enumerate(leaves):
            if leaf.is_terminal:
                # Terminal node: backup cached value
                leaf.backup(-leaf.terminal_value, gamma)  # terminal_value is from parent's perspective
                continue

            assert not leaf.is_expanded, "PUCT would have descended through an expanded non-terminal node"

            # Need to expand: clone board and replay path
            board_copy = copy_board(boards[i])
            terminal = False
            for action in action_paths[i]:
                row, col = idx_to_pos(action)
                outcome = board_copy.Move((row, col))
                if outcome != GameState.CONTINUE:
                    # This leaf is terminal
                    leaf.is_terminal = True
                    # The player who just moved won (or draw)
                    if outcome == GameState.DRAW:
                        leaf.terminal_value = 0.0
                    else:
                        # The player who made the last action won.
                        # terminal_value is stored from parent's perspective.
                        # Parent chose this action, and the result is a win for
                        # the player who just moved = the parent's side.
                        leaf.terminal_value = 1.0
                    leaf.backup(-leaf.terminal_value, gamma)
                    terminal = True
                    break

            if terminal:
                continue

            # Get observation for NN eval
            c0, c1, _ = board_copy.GetBoardState()
            eval_obs.append(encode_observation(c0, c1))
            legal_mask, _ = board_copy.GetLegalMoves()
            eval_legal.append(legal_mask)
            eval_indices.append(i)

        # Phase 3: Batch NN evaluation (cache-aware)
        if eval_indices:
            leaf_priors, leaf_values = _evaluate_with_cache(
                model, eval_obs, entropy_multiplier, device
            )

            for j, i in enumerate(eval_indices):
                leaf = leaves[i]
                legal_flat = eval_legal[j].reshape(225)
                prior_j = leaf_priors[j]

                # Expand: store priors over legal actions; child nodes are
                # created on demand by select_child as PUCT visits them.
                legal_actions = np.where(legal_flat)[0]
                leaf.expand(
                    legal_actions.tolist(),
                    prior_j[legal_actions].astype(np.float32).tolist(),
                    float(leaf_values[j]) * fpu_multiplier,
                )

                # Backup: leaf_values[j] is from side-to-move at leaf
                leaf.backup(leaf_values[j], gamma)

    # --- Extract results ---
    visit_distributions = np.zeros((n_games, 225), dtype=np.float32)
    root_q_values = np.zeros(n_games, dtype=np.float32)

    for i, root in enumerate(roots):
        actions = root.child_actions
        ns = root.child_n
        qs = root.child_q
        total_child_visits = sum(ns)
        if total_child_visits > 0:
            ns_arr = np.asarray(ns, dtype=np.int32)
            qs_arr = np.asarray(qs, dtype=np.float32)
            visit_distributions[i, actions] = ns_arr.astype(np.float32) / total_child_visits
            root_q_values[i] = float((ns_arr * qs_arr).sum() / total_child_visits)

    raw_entropies = -(priors * np.log(priors + 1e-30)).sum(axis=-1)

    # KL(visit_dist || raw prior) per root: how far the search moved the policy
    # from the network's raw prior (pre-Dirichlet). Terms with visit mass 0
    # vanish; legal moves always carry positive prior, so log(P) is finite where
    # the visit distribution has support.
    raw_mcts_kls = (
        visit_distributions
        * (np.log(visit_distributions + 1e-30) - np.log(priors + 1e-30))
    ).sum(axis=-1)

    # --- Harvest internal nodes (optional) ---
    # Walk each tree for above-threshold internal nodes and reconstruct each
    # node's obs by replaying its action path on a copy of the root board (same
    # replay pattern used for leaf evaluation above).
    harvested: list[list[tuple]] = [[] for _ in range(n_games)]
    if harvest_min_visits is not None:
        for i, root in enumerate(roots):
            for action_path, value_target, policy, n in root.harvest(harvest_min_visits):
                board_copy = copy_board(boards[i])
                for action in action_path:
                    row, col = idx_to_pos(action)
                    board_copy.Move((row, col))
                c0, c1, _ = board_copy.GetBoardState()
                obs = encode_observation(c0, c1)
                harvested[i].append(
                    (obs, np.asarray(policy, dtype=np.float32), float(value_target), int(n))
                )

    return visit_distributions, root_q_values, raw_entropies, raw_mcts_kls, harvested
