"""
Gomoku Inference Engine.

Handles batched self-play, evaluation games, and rollouts using PyTorch models.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional
from gomoku_rules import GomokuBoard, GameState, Player, encode_observation, idx_to_pos, board_from_observation

# ============================================================================
# Configuration
# ============================================================================

BATCH_INFERENCE_SIZE = 64
LOG_PROB_MIN = -10.0
LOGIT_MASK_VALUE = -1e9


# ============================================================================
# Data Structures
# ============================================================================

class Trajectory:
    """Stores a single episode trajectory."""

    def __init__(self):
        self.observations = []  # List of [3, 15, 15] numpy arrays
        self.actions = []       # List of action indices
        self.players = []       # List of Player enum values (absolute)
        self.legal_masks = []   # List of [15, 15] legal masks
        self.is_current_policy = []  # List of bools: True if current_policy moved
        self.log_probs = []     # List of log probabilities
        self.entropies = []     # List of policy entropies (nats) for CLER
        self.outcome = None     # GameState enum


class GameState_InProgress:
    """Tracks state of a game in progress."""
    def __init__(self, game_id: int, black_model, white_model,
                 current_is_black: bool, opening_id: int = -1):
        self.game_id = game_id
        self.board = GomokuBoard(opening_id=opening_id)
        self.black_model = black_model
        self.white_model = white_model
        self.current_is_black = current_is_black
        self.traj = Trajectory()
        self.done = False


class EvalGameState:
    """Minimal state for evaluation games."""
    __slots__ = ['board', 'black_model', 'white_model', 'current_is_black', 'done', 'outcome']
    def __init__(self, black_model, white_model, current_is_black: bool):
        self.board = GomokuBoard()
        self.black_model = black_model
        self.white_model = white_model
        self.current_is_black = current_is_black
        self.done = False
        self.outcome = None


class CLERRolloutState:
    """Minimal state for CLER rollout games."""
    __slots__ = ['board', 'black_model', 'white_model', 'first_player', 'done', 'won']
    def __init__(self, obs: np.ndarray, next_player: Player, first_action: int,
                 black_model, white_model):
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


# ============================================================================
# Helper Functions
# ============================================================================

def obs_batch_to_tensor(obs_list: List[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(obs_list)).float().to(device)


def mask_batch_to_tensor(mask_list: List[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack(mask_list)).bool().to(device)


def select_action_batch(model: nn.Module, obs_list: List[np.ndarray],
                        mask_list: List[np.ndarray],
                        temperature: float, device: torch.device,
                        deterministic: bool = False) -> Tuple[List[int], List[float], List[float]]:
    """Select actions for a batch (Training). Returns actions, log_probs, entropies."""
    if len(obs_list) == 0:
        return [], [], []

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
            actions = logits_flat.argmax(dim=1).cpu().numpy().tolist()
            dist = Categorical(logits=logits_flat)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
            log_probs = dist.log_prob(actions_tensor)
            log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN).cpu().numpy().tolist()
        else:
            dist = Categorical(logits=logits_flat)
            actions_tensor = dist.sample()
            actions = actions_tensor.cpu().numpy().tolist()
            log_probs = dist.log_prob(actions_tensor)
            log_probs = torch.clamp(log_probs, min=LOG_PROB_MIN).cpu().numpy().tolist()

        entropies = dist.entropy().cpu().numpy().tolist()

    return actions, log_probs, entropies


def select_action_batch_eval(model: nn.Module, obs_list: List[np.ndarray],
                              mask_list: List[np.ndarray],
                              temperature: float, device: torch.device,
                              deterministic: bool = False) -> List[int]:
    """Select actions for a batch (Evaluation/Rollout). Returns only actions."""
    if len(obs_list) == 0:
        return []

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
            probs = torch.softmax(logits_flat, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().tolist()

    return actions


# ============================================================================
# Main Loops
# ============================================================================

def play_episodes_batched(black_white_pairs: List[Tuple], 
                          current_is_black: List[bool],
                          temperature: float, device: torch.device,
                          batch_size: int = BATCH_INFERENCE_SIZE,
                          select_action_batch_fn = select_action_batch,
                          deterministic: bool = False,
                          opening_ids: List[int] = None) -> List[Trajectory]:
    """
    Play multiple episodes with batched inference.
    """
    if opening_ids is None:
        opening_ids = [-1] * len(black_white_pairs)

    games = [GameState_InProgress(i, black, white, is_black, opening_id)
             for i, ((black, white), is_black, opening_id) in enumerate(
                 zip(black_white_pairs, current_is_black, opening_ids))]

    while True:
        active_games = [g for g in games if not g.done]
        if len(active_games) == 0:
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
                model = game.black_model if next_player == Player.BLACK else game.white_model
                models_list.append(model)

            # Batched inference by model
            model_groups = {}
            for i, model in enumerate(models_list):
                model_id = id(model)
                if model_id not in model_groups:
                    model_groups[model_id] = {'model': model, 'indices': [], 'obs': [], 'masks': []}
                model_groups[model_id]['indices'].append(i)
                model_groups[model_id]['obs'].append(obs_list[i])
                model_groups[model_id]['masks'].append(mask_list[i])

            all_actions = [None] * len(batch_games)
            all_log_probs = [None] * len(batch_games)
            all_entropies = [None] * len(batch_games)

            for group in model_groups.values():
                actions, log_probs, entropies = select_action_batch_fn(
                    group['model'], group['obs'], group['masks'],
                    temperature, device, deterministic
                )
                for idx, action, log_prob, entropy in zip(group['indices'], actions, log_probs, entropies):
                    all_actions[idx] = action
                    all_log_probs[idx] = log_prob
                    all_entropies[idx] = entropy

            for game, action, log_prob, entropy in zip(batch_games, all_actions, all_log_probs, all_entropies):
                game.traj.actions.append(action)
                game.traj.log_probs.append(log_prob)
                game.traj.entropies.append(entropy)
                row, col = idx_to_pos(action)
                outcome = game.board.Move((row, col))

                if outcome != GameState.CONTINUE:
                    game.traj.outcome = outcome
                    game.done = True

    return [g.traj for g in games]


def play_eval_games(black_white_pairs: List[Tuple],
                    current_is_black: List[bool],
                    temperature: float, device: torch.device,
                    batch_size: int = BATCH_INFERENCE_SIZE,
                    select_action_fn = select_action_batch_eval) -> List[Tuple[GameState, bool]]:
    """
    Play evaluation games - returns only outcomes.
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

            model_groups = {}
            for i, model in enumerate(models_list):
                model_id = id(model)
                if model_id not in model_groups:
                    model_groups[model_id] = {'model': model, 'indices': [], 'obs': [], 'masks': []}
                model_groups[model_id]['indices'].append(i)
                model_groups[model_id]['obs'].append(obs_list[i])
                model_groups[model_id]['masks'].append(mask_list[i])

            all_actions = [None] * len(batch_games)
            for group in model_groups.values():
                actions = select_action_fn(group['model'], group['obs'], group['masks'], temperature, device)
                for idx, action in zip(group['indices'], actions):
                    all_actions[idx] = action

            for game, action in zip(batch_games, all_actions):
                row, col = idx_to_pos(action)
                outcome = game.board.Move((row, col))
                if outcome != GameState.CONTINUE:
                    game.outcome = outcome
                    game.done = True

    return [(g.outcome, g.current_is_black) for g in games]


def play_cler_rollouts_batched(rollout_configs: List[Tuple[np.ndarray, Player, int, object, object]],
                                temperature: float, device: torch.device,
                                batch_size: int = BATCH_INFERENCE_SIZE,
                                select_action_fn = select_action_batch_eval) -> List[bool]:
    """
    Play CLER rollout games in batches.
    Returns: List of bool indicating if first_player won.
    """
    if not rollout_configs:
        return []

    games = [CLERRolloutState(obs, player, action, black_model, white_model)
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

            model_groups = {}
            for i, model in enumerate(models_list):
                model_id = id(model)
                if model_id not in model_groups:
                    model_groups[model_id] = {'model': model, 'indices': [], 'obs': [], 'masks': []}
                model_groups[model_id]['indices'].append(i)
                model_groups[model_id]['obs'].append(obs_list[i])
                model_groups[model_id]['masks'].append(mask_list[i])

            all_actions = [None] * len(batch_games)
            for group in model_groups.values():
                actions = select_action_fn(group['model'], group['obs'], group['masks'], temperature, device)
                for idx, action in zip(group['indices'], actions):
                    all_actions[idx] = action

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
