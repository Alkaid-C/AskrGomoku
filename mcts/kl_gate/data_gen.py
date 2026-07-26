"""
kl_gate data generation: label positions with their MCTS improvement gap.

`final_policy.pt` self-plays with the deployment search settings (raw masked
softmax priors, no Dirichlet noise) at `LABEL_NUM_SIMULATIONS` simulations. Every
played ply is recorded as `(obs, raw_mcts_kl, prior, value, game_id)`, where
`raw_mcts_kl = KL(visit_dist || raw prior)` is the regression label produced by
`mcts.mcts_search_batched` itself.

Start positions are a three-way mix (Renju opening / three random stones / empty)
so the dataset is not dominated by the handful of positions reachable in the first
few plies of a deterministic search.

Resumable: any shard already on disk is skipped, and per-shard RNG seeding keeps
newly generated shards from duplicating skipped ones.

Run from mcts/:  python3 kl_gate/data_gen.py
"""

import os
import random
import time

import numpy as np
import torch
from config import (
    ACTION_TEMPERATURE,
    DATA_DIR,
    GAMES_PER_SHARD,
    LABEL_NUM_SIMULATIONS,
    P_RANDOM3,
    P_RENJU,
    POLICY_PATH,
    SEED,
    SHARD_FILENAME,
    TRAIN_GAMES,
    VAL_GAMES,
)
from gomoku import (
    RENJU_OPENING_SEQUENCES,
    GameState,
    GomokuBoard,
    encode_observation,
    idx_to_pos,
)
from model import GomokuPolicyNet

from main import C_PUCT, DEVICE, DISCOUNT_GAMMA, FPU_MULTIPLIER

# `_evaluate_with_cache` is private but deliberately reused: it is the exact
# function the search used to produce the priors the KL label was measured
# against (canonical-orientation forward + inverse permutation). A plain
# `model(obs)` forward would give a slightly different prior, since the network
# is not exactly D4-equivariant.
from mcts import (
    _evaluate_with_cache,
    get_nn_eval_cache_size,
    get_nn_eval_cache_stats,
    mcts_search_batched,
)

BACKFILL_CHUNK = 8192  # obs per _evaluate_with_cache call, to bound peak memory


def _shard_path(shard_idx: int) -> str:
    return os.path.join(DATA_DIR, SHARD_FILENAME.format(shard_idx))


def _seed_shard_rngs(shard_idx: int) -> None:
    """Re-seed Python/numpy/torch RNGs as a function of (SEED, shard_idx).

    Without this, skip-on-disk resume would have every newly generated shard
    replay the same RNG sequence the first never-skipped shard saw. Seeding
    per-shard makes each missing shard's content depend only on its own index.
    """
    s = (SEED * 1_000_003 + shard_idx) & 0x7FFFFFFF
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_start_board() -> GomokuBoard:
    """Draw one start position from the P_RENJU / P_RANDOM3 / P_EMPTY mix.

    - renju:   one of the 184 Renju openings with the class's own random offset.
    - random3: three distinct random squares played in order, so the colours come
               out Black / White / Black and White is to move. Three stones can
               never make five, so the start is never terminal.
    - empty:   empty board, Black to move.
    """
    u = random.random()
    if u < P_RENJU:
        return GomokuBoard(opening_id=random.randrange(len(RENJU_OPENING_SEQUENCES)))
    if u < P_RENJU + P_RANDOM3:
        board = GomokuBoard(opening_id=-1)
        for action in random.sample(range(225), 3):
            board.Move(idx_to_pos(action))
        return board
    return GomokuBoard(opening_id=-1)  # the remaining P_EMPTY


def play_and_label(
    model: GomokuPolicyNet, boards: list[GomokuBoard]
) -> tuple[list[np.ndarray], list[float], list[int], list[int]]:
    """Self-play every board to completion, recording each ply's obs and label.

    A trimmed-down `self_play.play_mcts_games`: no harvesting, and only the
    observation and `raw_mcts_kl` are kept (the visit distribution is used for
    action sampling and then discarded).

    Returns (obs_list, kl_list, game_index_list, game_lengths).
    """
    n = len(boards)
    all_obs: list[np.ndarray] = []
    all_kl: list[float] = []
    all_game_idx: list[int] = []
    lengths = [0] * n

    active = list(range(n))
    while active:
        visit_dists, _, _, raw_mcts_kls, _ = mcts_search_batched(
            model,
            [boards[i] for i in active],
            LABEL_NUM_SIMULATIONS,
            C_PUCT,
            entropy_multiplier=None,
            device=DEVICE,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            gamma=DISCOUNT_GAMMA,
            fpu_multiplier=FPU_MULTIPLIER,
            harvest_min_visits=None,
        )

        still_active: list[int] = []
        for j, i in enumerate(active):
            c0, c1, _ = boards[i].GetBoardState()
            all_obs.append(encode_observation(c0, c1))
            all_kl.append(float(raw_mcts_kls[j]))
            all_game_idx.append(i)
            lengths[i] += 1

            if ACTION_TEMPERATURE == 1.0:
                sample_dist = visit_dists[j]
            else:
                sample_dist = np.maximum(visit_dists[j], 1e-30) ** (1.0 / ACTION_TEMPERATURE)
                sample_dist = sample_dist / sample_dist.sum()
            action = int(np.random.choice(225, p=sample_dist))

            if boards[i].Move(idx_to_pos(action)) == GameState.CONTINUE:
                still_active.append(i)

        active = still_active

    return all_obs, all_kl, all_game_idx, lengths


def backfill_prior_value(
    model: GomokuPolicyNet, obs_list: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the raw prior and value for every recorded position.

    Runs through the search's own evaluation path, so the prior returned here is
    bit-identical to the one `raw_mcts_kl` was measured against. Every position
    was evaluated as a search root, so this is served almost entirely from the
    warm NN cache.
    """
    priors = np.empty((len(obs_list), 225), dtype=np.float32)
    values = np.empty(len(obs_list), dtype=np.float32)
    for start in range(0, len(obs_list), BACKFILL_CHUNK):
        chunk = obs_list[start:start + BACKFILL_CHUNK]
        p, v = _evaluate_with_cache(model, chunk, None, DEVICE)
        priors[start:start + len(chunk)] = p
        values[start:start + len(chunk)] = v
    return priors, values


def generate() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading policy: {POLICY_PATH}")
    checkpoint = torch.load(POLICY_PATH, map_location=DEVICE, weights_only=False)
    model = GomokuPolicyNet().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_games = TRAIN_GAMES + VAL_GAMES
    n_shards = (num_games + GAMES_PER_SHARD - 1) // GAMES_PER_SHARD
    print(f"Generating {num_games} games in {n_shards} shards "
          f"({GAMES_PER_SHARD}/shard) -> {DATA_DIR}")
    print(f"  sims={LABEL_NUM_SIMULATIONS}, action_T={ACTION_TEMPERATURE}, "
          f"c_puct={C_PUCT}, gamma={DISCOUNT_GAMMA}, fpu={FPU_MULTIPLIER}")
    print(f"  train: game_id < {TRAIN_GAMES} | val: game_id >= {TRAIN_GAMES}")

    total_start = time.time()
    for shard_idx in range(n_shards):
        path = _shard_path(shard_idx)
        if os.path.exists(path):
            print(f"  [{shard_idx+1}/{n_shards}] {os.path.basename(path)} exists - skipping")
            continue

        _seed_shard_rngs(shard_idx)

        base_game_id = shard_idx * GAMES_PER_SHARD
        n_games = min(GAMES_PER_SHARD, num_games - base_game_id)
        boards = [build_start_board() for _ in range(n_games)]

        hits_before, misses_before = get_nn_eval_cache_stats()
        t0 = time.time()
        obs_list, kl_list, game_idx_list, lengths = play_and_label(model, boards)
        priors, values = backfill_prior_value(model, obs_list)
        torch.cuda.empty_cache()
        t_elapsed = time.time() - t0

        hits_after, misses_after = get_nn_eval_cache_stats()
        shard_hits = hits_after - hits_before
        shard_lookups = shard_hits + (misses_after - misses_before)
        hit_rate = shard_hits / shard_lookups if shard_lookups > 0 else 0.0

        obs_arr = np.stack(obs_list).astype(np.uint8)                       # [N, 3, 15, 15]
        kl_arr = np.asarray(kl_list, dtype=np.float32)                      # [N]
        game_id_arr = (np.asarray(game_idx_list, dtype=np.int32) + base_game_id)

        # Write to a tmp basename then rename, so a partial file on crash cannot
        # fool the resume logic. `np.savez_compressed` appends ".npz" when the
        # path lacks it, so the tmp basename carries no extension.
        tmp_base = os.path.join(DATA_DIR, f".tmp_shard_{shard_idx:04d}")
        np.savez_compressed(
            tmp_base,
            obs=obs_arr,
            raw_mcts_kl=kl_arr,
            prior=priors,
            value=values,
            game_id=game_id_arr,
        )
        os.replace(tmp_base + ".npz", path)

        elapsed_total = time.time() - total_start
        eta = elapsed_total / (shard_idx + 1) * (n_shards - shard_idx - 1)
        print(
            f"  [{shard_idx+1}/{n_shards}] {os.path.basename(path)}: "
            f"{n_games} games, {len(obs_list)} samples, len {np.mean(lengths):.1f}, "
            f"KL mean {kl_arr.mean():.3f} med {np.median(kl_arr):.3f} | "
            f"{t_elapsed:.0f}s | "
            f"cache hit {hit_rate:.1%} ({shard_hits}/{shard_lookups}), "
            f"size {get_nn_eval_cache_size()} | "
            f"total {elapsed_total/3600:.1f}h, eta {eta/3600:.1f}h"
        )

    print("Data generation complete.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    generate()
