"""
Stage 1 data generation.

A frozen RL teacher runs MCTS self-play; each ply is recorded as a
`(obs, visit_dist, root_Q)` tuple and written to disk in sharded
`.npz` files. Two temperatures decouple supervision from coverage:

- `prior_temperature` softens the teacher's logits into the MCTS prior
  (entropy multiplier inside MCTS).
- `action_temperature` flattens MCTS visits at sampling time to broaden
  trajectory coverage. The recorded supervision target is the original
  visit distribution, untouched.

No Dirichlet noise is used during generation — that exploration source
is reserved for stage 2.

Resumable: any shard already on disk is skipped, so repeated invocations
or parallel processes can fill the dataset incrementally.
"""

import os
import random
import time

import numpy as np
import torch
from gomoku import RENJU_OPENING_SEQUENCES, SEED_PROBABILITY
from model import GomokuPolicyNet
from self_play import play_mcts_games

from mcts import get_nn_eval_cache_size, get_nn_eval_cache_stats

SHARD_FILENAME = "stage1_shard_{:04d}.npz"


def _shard_path(data_dir: str, shard_idx: int) -> str:
    return os.path.join(data_dir, SHARD_FILENAME.format(shard_idx))


def _seed_shard_rngs(seed: int, shard_idx: int) -> None:
    """Re-seed Python/numpy/torch RNGs as a function of (seed, shard_idx).

    Without this, skip-on-disk resume would have every newly generated shard
    replay the same RNG sequence the first never-skipped shard saw — i.e.
    duplicate openings/MCTS samples across runs. Seeding per-shard makes
    each missing shard's content depend only on its own index, so resumption
    and parallel filling produce a deterministic, non-redundant dataset.
    """
    s = (seed * 1_000_003 + shard_idx) & 0x7FFFFFFF
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def generate_stage1_data(
    teacher_path: str,
    data_dir: str,
    num_games: int,
    games_per_shard: int,
    num_simulations: int,
    c_puct: float,
    prior_temperature: float,
    action_temperature: float,
    gamma: float,
    seed: int,
    device: torch.device,
) -> None:
    os.makedirs(data_dir, exist_ok=True)

    print(f"Loading teacher: {teacher_path}")
    checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_shards = (num_games + games_per_shard - 1) // games_per_shard
    num_openings = len(RENJU_OPENING_SEQUENCES)
    print(f"Generating {num_games} games in {n_shards} shards "
          f"({games_per_shard}/shard) -> {data_dir}")
    print(f"  prior_T={prior_temperature}, action_T={action_temperature}, "
          f"sims={num_simulations}, c_puct={c_puct}, gamma={gamma}")

    total_start = time.time()
    for shard_idx in range(n_shards):
        path = _shard_path(data_dir, shard_idx)
        if os.path.exists(path):
            print(f"  [{shard_idx+1}/{n_shards}] {os.path.basename(path)} exists - skipping")
            continue

        _seed_shard_rngs(seed, shard_idx)

        # Last shard may be smaller if num_games is not a multiple
        games_remaining = num_games - shard_idx * games_per_shard
        n_games = min(games_per_shard, games_remaining)

        opening_ids: list[int] = []
        for _ in range(n_games):
            if random.random() < SEED_PROBABILITY:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        hits_before, misses_before = get_nn_eval_cache_stats()
        t0 = time.time()
        records = play_mcts_games(
            model=model,
            num_games=n_games,
            num_simulations=num_simulations,
            c_puct=c_puct,
            entropy_multiplier=prior_temperature,
            device=device,
            opening_ids=opening_ids,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            gamma=gamma,
            action_temperature=action_temperature,
        )
        torch.cuda.empty_cache()
        t_elapsed = time.time() - t0
        hits_after, misses_after = get_nn_eval_cache_stats()
        shard_hits = hits_after - hits_before
        shard_misses = misses_after - misses_before
        shard_lookups = shard_hits + shard_misses
        hit_rate = shard_hits / shard_lookups if shard_lookups > 0 else 0.0

        all_obs: list[np.ndarray] = []
        all_dists: list[np.ndarray] = []
        all_q: list[float] = []
        for record in records:
            for obs, dist, q in zip(
                record.observations,
                record.visit_distributions,
                record.root_values,
            ):
                all_obs.append(obs)
                all_dists.append(dist)
                all_q.append(q)

        obs_arr = np.stack(all_obs).astype(np.uint8)            # [N, 3, 15, 15]
        dist_arr = np.stack(all_dists).astype(np.float32)        # [N, 225]
        q_arr = np.asarray(all_q, dtype=np.float32)              # [N]

        # Atomic-ish: write to a tmp basename then rename so a partial file
        # on crash doesn't fool the resume logic. `np.savez_compressed` appends
        # ".npz" if the path lacks it, so the tmp basename has no extension —
        # the actual file written is `<tmp_base>.npz`, which we then rename.
        tmp_base = os.path.join(data_dir, f".tmp_shard_{shard_idx:04d}")
        tmp_path = tmp_base + ".npz"
        np.savez_compressed(tmp_base, obs=obs_arr, visit_dist=dist_arr, root_Q=q_arr)
        os.replace(tmp_path, path)

        elapsed_total = time.time() - total_start
        remaining = n_shards - shard_idx - 1
        eta = elapsed_total / (shard_idx + 1) * remaining
        print(
            f"  [{shard_idx+1}/{n_shards}] {os.path.basename(path)}: "
            f"{n_games} games, {len(all_obs)} samples, "
            f"{t_elapsed:.0f}s | "
            f"cache hit {hit_rate:.1%} ({shard_hits}/{shard_lookups}), "
            f"size {get_nn_eval_cache_size()} | "
            f"total {elapsed_total/60:.1f}m, eta {eta/60:.1f}m"
        )

    print("Data generation complete.")
