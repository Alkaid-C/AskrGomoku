"""
MCTS (Monte Carlo Tree Search) with PUCT and Batched Leaf Evaluation

Implements PUCT-based tree search with neural network evaluation for Gomoku.
Supports batched search across multiple game positions simultaneously.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gomoku import GameState, GomokuBoard, encode_observation, idx_to_pos

# ============================================================================
# MCTS Node
# ============================================================================


class MCTSNode:
    """Single node in the MCTS tree."""
    __slots__ = [
        'action',
        'children',
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
    """Select child with highest PUCT score."""
    sqrt_parent = math.sqrt(node.visit_count)
    best_score = -float('inf')
    best_child = None

    for child in node.children.values():
        score = child.q_value + c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        if score > best_score:
            best_score = score
            best_child = child

    assert best_child is not None
    return best_child


def backup(leaf: MCTSNode, value: float) -> None:
    """
    Backup value from leaf to root.

    Args:
        leaf: The leaf node where evaluation happened
        value: Value from the side-to-move's perspective at the leaf
    """
    v = value
    node = leaf
    while node is not None:
        v = -v  # flip perspective at each level
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
    prior_temperature: float,
    device: torch.device,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batched MCTS search on multiple board positions.

    Args:
        model: Neural network for policy/value evaluation
        boards: List of N board positions to search from
        num_simulations: Number of MCTS simulations per position
        c_puct: PUCT exploration constant
        prior_temperature: Temperature for softening policy prior (>1 = flatter)
        device: Torch device
        dirichlet_alpha: Dirichlet noise parameter
        dirichlet_epsilon: Dirichlet noise weight (0 = no noise)

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

    obs_tensor = torch.from_numpy(np.stack(obs_list)).float().to(device)
    mask_tensor = torch.from_numpy(np.stack(legal_masks)).bool().to(device)

    with torch.inference_mode():
        logits = model.forward_policy_only(obs_tensor)
    logits = logits.squeeze(1)
    logits = logits.view(n_games, 225)
    logits = logits.masked_fill(~mask_tensor.view(n_games, 225), -1e9)
    # Apply temperature to prior
    priors = F.softmax(logits / prior_temperature, dim=-1).cpu().numpy()

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

        for k, action in enumerate(legal_indices):
            p = (1 - dirichlet_epsilon) * prior_i[action] + dirichlet_epsilon * noise[k]
            child = MCTSNode(parent=root, action=action, prior=p)
            root.children[action] = child

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
            while node.children and not node.is_terminal:
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
                backup(leaf, -leaf.terminal_value)  # terminal_value is from parent's perspective
                continue

            assert not leaf.children, "PUCT would have descended through an expanded non-terminal node"

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
                    backup(leaf, -leaf.terminal_value)
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

        # Phase 3: Batch NN evaluation
        if eval_indices:
            obs_t = torch.from_numpy(np.stack(eval_obs)).float().to(device)
            with torch.inference_mode():
                logits_t, values_t = model(obs_t)
            logits_t = logits_t.squeeze(1).view(len(eval_indices), 225)
            mask_t = torch.from_numpy(np.stack(eval_legal)).bool().to(device).view(len(eval_indices), 225)
            logits_t = logits_t.masked_fill(~mask_t, -1e9)
            leaf_priors = F.softmax(logits_t / prior_temperature, dim=-1).cpu().numpy()
            leaf_values = values_t.squeeze(-1).cpu().numpy()

            for j, i in enumerate(eval_indices):
                leaf = leaves[i]
                legal_flat = eval_legal[j].reshape(225)
                prior_j = leaf_priors[j]

                # Expand: create children
                legal_actions = np.where(legal_flat)[0]
                for action in legal_actions:
                    child = MCTSNode(parent=leaf, action=action, prior=prior_j[action])
                    leaf.children[action] = child

                # Backup: leaf_values[j] is from side-to-move at leaf
                backup(leaf, leaf_values[j])

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
