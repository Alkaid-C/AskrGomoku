"""
Shard loading, target/input transforms, and D4 augmentation for kl_gate.

Everything is held in RAM as numpy arrays; the network input planes are assembled
per batch on the GPU rather than stored, which would cost several times more (the
prior alone would go from one `[225]` row per sample to a full float plane stack).

Split is by `game_id`, so no game contributes plies to both sides:
`game_id < TRAIN_GAMES` is train, the rest is validation.
"""

import os

import numpy as np
import torch
from config import (
    DATA_DIR,
    GAMES_PER_SHARD,
    LOG_EPSILON,
    PRIOR_LOG_EPSILON,
    SHARD_FILENAME,
    TRAIN_GAMES,
    VAL_GAMES,
)


def augment_8fold(x: torch.Tensor) -> torch.Tensor:
    """Apply all 8 dihedral transforms to a [B, C, 15, 15] batch -> [8B, C, 15, 15].

    Same transform expressions and ordering as `enhancement.augment_batch_8fold`.
    No permutation table is needed here because the prior travels as a 15x15
    plane rather than a [225] vector, so it transforms exactly like the
    observation planes; the scalar target is D4-invariant.
    """
    x_t = x.transpose(-2, -1)
    x_r180 = x.flip(-2, -1)
    return torch.cat([
        x, x_t.flip(-1), x_r180, x_t.flip(-2),
        x.flip(-1), x.flip(-2), x_t, x_r180.transpose(-2, -1),
    ])


class KLGateSplit:
    """One split's arrays plus batch assembly."""

    def __init__(self, obs: np.ndarray, log_prior: np.ndarray, value: np.ndarray, y: np.ndarray):
        self.obs = obs                # [N, 3, 15, 15] uint8
        self.log_prior = log_prior    # [N, 225] float32, log(prior + eps)
        self.value = value            # [N] float32
        self.y = y                    # [N] float32, log(KL + eps)

    def __len__(self) -> int:
        return len(self.y)

    def make_batch(
        self, indices: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the [B, 5, 15, 15] input and [B] target for `indices`.

        Planes 0-2 are the observation, plane 3 is the log-prior, plane 4 is the
        network's value broadcast over the board.
        """
        b = len(indices)
        obs = torch.from_numpy(self.obs[indices]).to(device).float()
        log_prior = torch.from_numpy(self.log_prior[indices]).to(device).view(b, 1, 15, 15)
        value = torch.from_numpy(self.value[indices]).to(device).view(b, 1, 1, 1).expand(b, 1, 15, 15)
        x = torch.cat([obs, log_prior, value], dim=1)
        y = torch.from_numpy(self.y[indices]).to(device)
        return x, y


def load_split() -> tuple[KLGateSplit, KLGateSplit]:
    """Load every shard and return (train, val).

    The target `log(KL + LOG_EPSILON)` and the log-prior input plane are computed
    once here rather than per batch.
    """
    num_games = TRAIN_GAMES + VAL_GAMES
    n_shards = (num_games + GAMES_PER_SHARD - 1) // GAMES_PER_SHARD

    obs_parts, prior_parts, value_parts, kl_parts, game_id_parts = [], [], [], [], []
    for shard_idx in range(n_shards):
        path = os.path.join(DATA_DIR, SHARD_FILENAME.format(shard_idx))
        with np.load(path) as z:
            obs_parts.append(z['obs'])
            prior_parts.append(z['prior'])
            value_parts.append(z['value'])
            kl_parts.append(z['raw_mcts_kl'])
            game_id_parts.append(z['game_id'])

    obs = np.concatenate(obs_parts)
    prior = np.concatenate(prior_parts)
    value = np.concatenate(value_parts)
    kl = np.concatenate(kl_parts)
    game_id = np.concatenate(game_id_parts)

    # Transformed in place — the raw prior / KL are not needed downstream, and
    # the prior array is the bulk of the dataset's memory.
    prior += PRIOR_LOG_EPSILON
    log_prior = np.log(prior, out=prior)
    kl += LOG_EPSILON
    y = np.log(kl, out=kl)

    is_train = game_id < TRAIN_GAMES
    splits = []
    for sel in (is_train, ~is_train):
        splits.append(KLGateSplit(obs[sel], log_prior[sel], value[sel], y[sel]))
    return splits[0], splits[1]
