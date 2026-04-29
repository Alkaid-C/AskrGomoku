"""
MCTS (Monte Carlo Tree Search) with PUCT and Batched Leaf Evaluation

Implements PUCT-based tree search with neural network evaluation for Gomoku.
Supports batched search across multiple game positions simultaneously.
"""

import math
from typing import Callable, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from entropy_ops import rescale_to_entropy_np, softmax_np
from gomoku import GameState, GomokuBoard, encode_observation, idx_to_pos

# ============================================================================
# D4 Canonicalization & NN Eval Cache
# ============================================================================

# Forward permutation table: _FORWARD_PERM[s, old_flat] = new_flat.
# Coordinate convention matches main/enhancement.py:268-269 (and the inverse
# table in mcts_post_train/training.py); both must stay in sync.
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

# Cache maps canonical-obs bytes -> (canonical_priors_scaled [225] float32, value).
# `canonical_priors_scaled` is the post-mask, post-softmax, post-entropy-rescale
# distribution in canonical orientation (illegal squares are exactly 0). Caching
# the scaled distribution rather than raw logits is sound because the entropy
# multiplier T is constant within a single training update, and the cache is
# cleared once per update at the optimizer.step() boundary.
_NN_EVAL_CACHE: dict[bytes, tuple[np.ndarray, float]] = {}
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


# ============================================================================
# MCTS Node
# ============================================================================


class MCTSNode:
    """Single node in the MCTS tree.

    On expansion, child stats are stored as parallel **Python lists** (not
    numpy arrays) so the PUCT inner loop in `select_child` reads native
    Python floats/ints — numpy scalar indexing is ~3x slower in tight
    loops because each `arr[k]` allocates a numpy scalar wrapper.
    `child_actions` is a list[int]; `child_priors`, `child_q`, `child_total`
    are list[float]; `child_n` is list[int]; `child_node` lazily holds
    materialized `MCTSNode`s indexed by the same `k`.
    """
    __slots__ = [
        'action',
        'child_actions',
        'child_n',
        'child_node',
        'child_priors',
        'child_q',
        'child_total',
        'is_expanded',
        'is_terminal',
        'parent',
        'parent_k',
        'prior',
        'terminal_value',
        'visit_count',
    ]

    def __init__(
        self,
        parent: Optional['MCTSNode'],
        parent_k: int,
        action: int,
        prior: float,
    ):
        self.parent = parent
        self.parent_k = parent_k       # index into parent's child lists (-1 for root)
        self.action = action           # flat index (-1 for root)
        self.child_actions: Optional[list[int]] = None
        self.child_priors: Optional[list[float]] = None
        self.child_q: Optional[list[float]] = None
        self.child_n: Optional[list[int]] = None
        self.child_total: Optional[list[float]] = None
        self.child_node: Optional[list[Optional[MCTSNode]]] = None
        self.is_expanded = False
        self.visit_count = 0
        self.prior = prior             # P: prior probability
        self.is_terminal = False
        self.terminal_value = 0.0      # from parent's perspective (+1 = parent wins)


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
# PUCT Selection and Backup
# ============================================================================


def select_child(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Select child with highest PUCT score, materializing it on demand.

    All child stats live in parallel arrays on `node`. Unvisited slots have
    `child_q[k] = 0` and `child_n[k] = 0`, so the unified PUCT formula
    `q + c_puct * prior * sqrt_parent / (1 + n)` collapses to the correct
    `c_puct * prior * sqrt_parent` for unvisited children (no branch).
    The chosen child's `MCTSNode` is created lazily on first selection.
    """
    actions = node.child_actions
    priors = node.child_priors
    qs = node.child_q
    ns = node.child_n
    nodes = node.child_node
    assert actions is not None and priors is not None
    assert qs is not None and ns is not None and nodes is not None

    c_sqrt = c_puct * math.sqrt(node.visit_count)
    K = len(actions)
    best_score = -float('inf')
    best_k = -1
    for k in range(K):
        score = qs[k] + c_sqrt * priors[k] / (1 + ns[k])
        if score > best_score:
            best_score = score
            best_k = k

    assert best_k >= 0
    child = nodes[best_k]
    if child is None:
        child = MCTSNode(
            parent=node,
            parent_k=best_k,
            action=actions[best_k],
            prior=priors[best_k],
        )
        nodes[best_k] = child
    return child


def backup(leaf: MCTSNode, value: float, gamma: float) -> None:
    """
    Backup value from leaf to root with per-ply discount.

    Updates each traversed node's `visit_count` and (when it has a parent)
    the parent's running stats for this child slot: `child_total[k]`,
    `child_n[k]`, `child_q[k] = child_total[k] / child_n[k]`. Root has no
    parent, so only its own `visit_count` is touched.

    Args:
        leaf: The leaf node where evaluation happened
        value: Value from the side-to-move's perspective at the leaf
        gamma: Per-ply discount factor. Applied at every level alongside
            the perspective flip, so a terminal +/-1 reaches the root with
            magnitude gamma^depth and alternating sign. This makes "lose
            later" strictly better than "lose sooner" in backed-up Q.
    """
    v = value
    node = leaf
    while node is not None:
        v = -v * gamma  # flip perspective AND discount
        node.visit_count += 1
        parent = node.parent
        if parent is not None:
            k = node.parent_k
            assert parent.child_total is not None and parent.child_n is not None and parent.child_q is not None
            parent.child_total[k] += v
            parent.child_n[k] += 1
            parent.child_q[k] = parent.child_total[k] / parent.child_n[k]
        node = parent


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
        m_obs_np = np.stack(miss_canonical_obs)  # [M, 3, 15, 15] uint8, canonical
        miss_obs_tensor = torch.from_numpy(m_obs_np).float().to(device)
        with torch.inference_mode():
            m_logits, m_values = model(miss_obs_tensor)
        m_logits_np = m_logits.squeeze(1).view(len(miss_indices), 225).cpu().numpy()
        m_values_np = m_values.squeeze(-1).cpu().numpy()

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
                canonical_priors_scaled[k].copy(),
                float(m_values_np[k]),
            )

    priors = np.empty((n, 225), dtype=np.float32)
    values = np.empty(n, dtype=np.float32)
    for i, (key, s) in enumerate(keys_and_s):
        canonical_priors, value = _NN_EVAL_CACHE[key]
        priors[i] = canonical_priors[_FORWARD_PERM[s]]
        values[i] = value
    return priors, values


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
) -> tuple[np.ndarray, np.ndarray]:
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
        dirichlet_alpha: Dirichlet noise parameter
        dirichlet_epsilon: Dirichlet noise weight (0 = no noise)
        gamma: Per-ply discount in backup; backed-up Q carries a gamma^depth
            factor, so deeper terminals contribute less than shallow ones.

    Returns:
        visit_distributions: [N, 225] normalized visit counts
        root_values: [N] weighted mean Q from root's perspective
    """
    n_games = len(boards)

    # --- Root initialization: batch forward pass ---
    obs_list = []
    legal_masks = []
    for board in boards:
        c0, c1, _ = board.GetBoardState()
        obs_list.append(encode_observation(c0, c1))
        legal_mask, _ = board.GetLegalMoves()
        legal_masks.append(legal_mask)

    priors, _root_values = _evaluate_with_cache(
        model, obs_list, entropy_multiplier, device
    )

    # Create root nodes and expand
    roots: list[MCTSNode] = []
    for i in range(n_games):
        root = MCTSNode(parent=None, parent_k=-1, action=-1, prior=0.0)
        root.visit_count = 1  # virtual visit so PUCT uses priors on first selection

        legal_flat = legal_masks[i].reshape(225)
        prior_i = priors[i]

        # Add Dirichlet noise (skipped when epsilon=0; alpha=0 would NaN out)
        legal_indices = np.where(legal_flat)[0]
        if dirichlet_epsilon > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_indices))
            final_priors = (
                (1 - dirichlet_epsilon) * prior_i[legal_indices]
                + dirichlet_epsilon * noise
            ).astype(np.float32)
        else:
            final_priors = prior_i[legal_indices].astype(np.float32)
        n_legal = len(legal_indices)
        root.child_actions = legal_indices.tolist()
        root.child_priors = final_priors.tolist()
        root.child_q = [0.0] * n_legal
        root.child_n = [0] * n_legal
        root.child_total = [0.0] * n_legal
        root.child_node = cast(list[Optional[MCTSNode]], [None] * n_legal)
        root.is_expanded = True

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
                node = select_child(node, c_puct)
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
                backup(leaf, -leaf.terminal_value, gamma)  # terminal_value is from parent's perspective
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
                    backup(leaf, -leaf.terminal_value, gamma)
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
                n_legal = len(legal_actions)
                leaf.child_actions = legal_actions.tolist()
                leaf.child_priors = prior_j[legal_actions].astype(np.float32).tolist()
                leaf.child_q = [0.0] * n_legal
                leaf.child_n = [0] * n_legal
                leaf.child_total = [0.0] * n_legal
                leaf.child_node = [None] * n_legal
                leaf.is_expanded = True

                # Backup: leaf_values[j] is from side-to-move at leaf
                backup(leaf, leaf_values[j], gamma)

    # --- Extract results ---
    visit_distributions = np.zeros((n_games, 225), dtype=np.float32)
    root_q_values = np.zeros(n_games, dtype=np.float32)

    for i, root in enumerate(roots):
        actions = root.child_actions
        ns = root.child_n
        qs = root.child_q
        assert actions is not None and ns is not None and qs is not None
        total_child_visits = sum(ns)
        if total_child_visits > 0:
            ns_arr = np.asarray(ns, dtype=np.int32)
            qs_arr = np.asarray(qs, dtype=np.float32)
            visit_distributions[i, actions] = ns_arr.astype(np.float32) / total_child_visits
            root_q_values[i] = float((ns_arr * qs_arr).sum() / total_child_visits)

    return visit_distributions, root_q_values
