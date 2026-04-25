"""
MCTS (Monte Carlo Tree Search) with PUCT and Batched Leaf Evaluation

Implements PUCT-based tree search with neural network evaluation for Gomoku.
Supports batched search across multiple game positions simultaneously.
"""

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entropy_ops import rescale_to_entropy
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

# Cache maps canonical-obs bytes -> (canonical_logits [225] float32, value).
# Cleared once per training update at the optimizer.step() boundary.
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

    Children are created lazily: `child_actions`/`child_priors` hold the
    full prior distribution over legal actions once the node is expanded,
    while `children` contains only the `MCTSNode` objects PUCT has actually
    selected at least once. Unvisited children's PUCT score is fully
    determined by `(prior, parent.visit_count)`, so no node is needed.
    """
    __slots__ = [
        'action',
        'child_actions',
        'child_priors',
        'children',
        'is_expanded',
        'is_terminal',
        'parent',
        'prior',
        'terminal_value',
        'total_value',
        'visit_count',
    ]

    def __init__(self, parent: Optional['MCTSNode'], action: int, prior: float):
        self.parent = parent
        self.action = action          # flat index (-1 for root)
        self.children: dict[int, MCTSNode] = {}
        self.child_actions: Optional[np.ndarray] = None  # int32 [num_legal]
        self.child_priors: Optional[np.ndarray] = None   # float32 [num_legal]
        self.is_expanded = False
        self.visit_count = 0
        self.total_value = 0.0        # W: accumulated from parent's perspective
        self.prior = prior            # P: prior probability
        self.is_terminal = False
        self.terminal_value = 0.0     # from parent's perspective (+1 = parent wins)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


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

    Iterates `child_actions`/`child_priors` (set when the node was expanded)
    rather than `children.values()`. Unvisited children contribute Q=0 and
    visit_count=0, making their score `c_puct * P * sqrt(N_parent)` — no
    `MCTSNode` instance is required to compute it. The chosen child is
    looked up in `node.children` and created lazily if absent.
    """
    assert node.child_actions is not None and node.child_priors is not None
    sqrt_parent = math.sqrt(node.visit_count)
    best_score = -float('inf')
    best_action = -1
    best_prior = 0.0

    for k in range(len(node.child_actions)):
        action = int(node.child_actions[k])
        prior = float(node.child_priors[k])
        existing = node.children.get(action)
        if existing is None:
            score = c_puct * prior * sqrt_parent
        else:
            score = existing.q_value + c_puct * prior * sqrt_parent / (1 + existing.visit_count)
        if score > best_score:
            best_score = score
            best_action = action
            best_prior = prior

    assert best_action >= 0
    child = node.children.get(best_action)
    if child is None:
        child = MCTSNode(parent=node, action=best_action, prior=best_prior)
        node.children[best_action] = child
    return child


def backup(leaf: MCTSNode, value: float, gamma: float) -> None:
    """
    Backup value from leaf to root with per-ply discount.

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
        node.total_value += v
        node.visit_count += 1
        node = node.parent


# ============================================================================
# Batched MCTS Search
# ============================================================================


def mcts_search_batched(
    model: nn.Module,
    boards: list[GomokuBoard],
    num_simulations: int,
    c_puct: float,
    entropy_multiplier: float,
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
        entropy_multiplier: Per-position prior entropy is rescaled to
            H(softmax(logits)) * entropy_multiplier. Caller passes the EMA
            temperature T (no exponent — the asymmetry T vs T**0.99 between
            pre-search flatten and post-search sharpen drives convergence to T=1).
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

    # Partition into cache hits and misses on D4-canonical key. Dedup by key
    # within the batch — duplicate canonical inputs (e.g. all empty-board
    # seeded games) must evaluate only once, not once per occurrence.
    global _CACHE_HITS, _CACHE_MISSES
    keys_and_s = [canonicalize_obs(o) for o in obs_list]
    miss_indices: list[int] = []
    miss_canonical_obs: list[np.ndarray] = []
    pending_keys: set[bytes] = set()
    for i, (key, s) in enumerate(keys_and_s):
        if key in _NN_EVAL_CACHE or key in pending_keys:
            continue
        pending_keys.add(key)
        miss_indices.append(i)
        miss_canonical_obs.append(np.ascontiguousarray(_SPATIAL_TRANSFORMS[s](obs_list[i])))
    _CACHE_HITS += len(keys_and_s) - len(miss_indices)
    _CACHE_MISSES += len(miss_indices)

    if miss_indices:
        miss_obs_tensor = torch.from_numpy(np.stack(miss_canonical_obs)).float().to(device)
        with torch.inference_mode():
            m_logits, m_values = model(miss_obs_tensor)
        m_logits_np = m_logits.squeeze(1).view(len(miss_indices), 225).cpu().numpy()
        m_values_np = m_values.squeeze(-1).cpu().numpy()
        for k, i in enumerate(miss_indices):
            key = keys_and_s[i][0]
            _NN_EVAL_CACHE[key] = (m_logits_np[k].copy(), float(m_values_np[k]))

    # Reassemble per-game logits by inverting the canonical permutation
    raw_logits = np.empty((n_games, 225), dtype=np.float32)
    for i, (key, s) in enumerate(keys_and_s):
        canonical_logits, _ = _NN_EVAL_CACHE[key]
        raw_logits[i] = canonical_logits[_FORWARD_PERM[s]]

    logits = torch.from_numpy(raw_logits).to(device)
    mask_tensor = torch.from_numpy(np.stack(legal_masks)).bool().to(device)
    logits = logits.masked_fill(~mask_tensor.view(n_games, 225), -1e9)
    # Rescale prior to target entropy = H_model * T per position.
    natural = F.softmax(logits, dim=-1)
    H_model = -(natural * (natural + 1e-30).log()).sum(dim=-1)
    target_H = (H_model * entropy_multiplier).clamp(max=math.log(225))
    priors = rescale_to_entropy(logits, target_H).cpu().numpy()

    # Create root nodes and expand
    roots: list[MCTSNode] = []
    for i in range(n_games):
        root = MCTSNode(parent=None, action=-1, prior=0.0)
        root.visit_count = 1  # virtual visit so PUCT uses priors on first selection

        legal_flat = legal_masks[i].reshape(225)
        prior_i = priors[i]

        # Add Dirichlet noise
        legal_indices = np.where(legal_flat)[0]
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal_indices))

        final_priors = (
            (1 - dirichlet_epsilon) * prior_i[legal_indices]
            + dirichlet_epsilon * noise
        ).astype(np.float32)
        root.child_actions = legal_indices.astype(np.int32)
        root.child_priors = final_priors
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
            n_eval = len(eval_indices)
            keys_and_s_eval = [canonicalize_obs(o) for o in eval_obs]
            miss_idx_eval: list[int] = []
            miss_canonical_obs: list[np.ndarray] = []
            pending_keys_eval: set[bytes] = set()
            for j, (key, s) in enumerate(keys_and_s_eval):
                if key in _NN_EVAL_CACHE or key in pending_keys_eval:
                    continue
                pending_keys_eval.add(key)
                miss_idx_eval.append(j)
                miss_canonical_obs.append(
                    np.ascontiguousarray(_SPATIAL_TRANSFORMS[s](eval_obs[j]))
                )
            _CACHE_HITS += n_eval - len(miss_idx_eval)
            _CACHE_MISSES += len(miss_idx_eval)

            if miss_idx_eval:
                miss_obs_tensor = torch.from_numpy(np.stack(miss_canonical_obs)).float().to(device)
                with torch.inference_mode():
                    m_logits, m_values = model(miss_obs_tensor)
                m_logits_np = m_logits.squeeze(1).view(len(miss_idx_eval), 225).cpu().numpy()
                m_values_np = m_values.squeeze(-1).cpu().numpy()
                for k, j in enumerate(miss_idx_eval):
                    key = keys_and_s_eval[j][0]
                    _NN_EVAL_CACHE[key] = (m_logits_np[k].copy(), float(m_values_np[k]))

            raw_logits = np.empty((n_eval, 225), dtype=np.float32)
            leaf_values = np.empty(n_eval, dtype=np.float32)
            for j, (key, s) in enumerate(keys_and_s_eval):
                canonical_logits, value = _NN_EVAL_CACHE[key]
                raw_logits[j] = canonical_logits[_FORWARD_PERM[s]]
                leaf_values[j] = value

            logits_t = torch.from_numpy(raw_logits).to(device)
            mask_t = torch.from_numpy(np.stack(eval_legal)).bool().to(device).view(n_eval, 225)
            logits_t = logits_t.masked_fill(~mask_t, -1e9)
            leaf_natural = F.softmax(logits_t, dim=-1)
            leaf_H = -(leaf_natural * (leaf_natural + 1e-30).log()).sum(dim=-1)
            leaf_target_H = (leaf_H * entropy_multiplier).clamp(max=math.log(225))
            leaf_priors = rescale_to_entropy(logits_t, leaf_target_H).cpu().numpy()

            for j, i in enumerate(eval_indices):
                leaf = leaves[i]
                legal_flat = eval_legal[j].reshape(225)
                prior_j = leaf_priors[j]

                # Expand: store priors over legal actions; child nodes are
                # created on demand by select_child as PUCT visits them.
                legal_actions = np.where(legal_flat)[0]
                leaf.child_actions = legal_actions.astype(np.int32)
                leaf.child_priors = prior_j[legal_actions].astype(np.float32)
                leaf.is_expanded = True

                # Backup: leaf_values[j] is from side-to-move at leaf
                backup(leaf, leaf_values[j], gamma)

    # --- Extract results ---
    visit_distributions = np.zeros((n_games, 225), dtype=np.float32)
    root_q_values = np.zeros(n_games, dtype=np.float32)

    for i, root in enumerate(roots):
        total_child_visits = 0
        weighted_q_sum = 0.0

        for action, child in root.children.items():
            visit_distributions[i, action] = child.visit_count
            total_child_visits += child.visit_count
            # Q is from root's perspective (= side-to-move's perspective)
            weighted_q_sum += child.visit_count * child.q_value

        if total_child_visits > 0:
            visit_distributions[i] /= total_child_visits
            root_q_values[i] = weighted_q_sum / total_child_visits

    return visit_distributions, root_q_values
