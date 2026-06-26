#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from gomoku import board_from_observation
from probe_common import (
    configure_torch,
    constants_snapshot,
    default_recent_sample_updates,
    ensure_output_dir,
    latest_checkpoint_update,
    load_model,
    percentile_summary,
    player_to_move_from_obs,
    resolve_device,
    sample_shards_by_update,
    seed_everything,
    train_main,
    write_json,
)

from mcts import clear_nn_eval_cache, mcts_search_batched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Stage 2 MCTS target drift against an anchor checkpoint."
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("codex_efficiency_probe/data/stage2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("codex_efficiency_probe/results/q1_signal_drift"),
    )
    parser.add_argument("--anchor-update", default="latest")
    parser.add_argument("--horizons", type=int, nargs="*")
    parser.add_argument("--sample-updates", type=int, nargs="*")
    parser.add_argument("--num-positions", type=int, default=2048)
    parser.add_argument("--num-simulations", type=int, default=train_main.NUM_SIMULATIONS_S2)
    parser.add_argument("--root-batch-size", type=int, default=256)
    parser.add_argument("--floor-runs", type=int, default=5)
    parser.add_argument("--history-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=train_main.SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-harvested", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_anchor(stage2_dir: Path, anchor_arg: str) -> int:
    if anchor_arg == "latest":
        return latest_checkpoint_update(stage2_dir)
    return int(anchor_arg)


def resolve_horizons(args: argparse.Namespace) -> list[int]:
    if args.horizons:
        horizons = args.horizons
    else:
        rounds = train_main.STAGE2_REPLAY_BUFFER_ROUNDS
        horizons = [1, max(1, rounds // 2), rounds, 2 * rounds]
    return sorted({h for h in horizons if h > 0})


def load_position_pool(
    stage2_dir: Path,
    sample_updates: list[int],
    *,
    include_harvested: bool,
) -> np.ndarray:
    obs_parts = []
    for shard in sample_shards_by_update(stage2_dir, sample_updates):
        with np.load(shard.path) as data:
            obs = data["obs"]
            if include_harvested:
                obs_parts.append(obs.copy())
                continue
            played = (data["policy_weight"] == 1.0) & (data["value_weight"] == 1.0)
            obs_parts.append(obs[played].copy())
    if not obs_parts:
        raise RuntimeError("No sample shards selected")
    pool = np.concatenate(obs_parts, axis=0)
    if pool.shape[0] == 0:
        raise RuntimeError("Selected sample shards contain no usable positions")
    return pool


def select_positions(
    stage2_dir: Path,
    sample_updates: list[int],
    num_positions: int,
    seed: int,
    include_harvested: bool,
) -> np.ndarray:
    pool = load_position_pool(
        stage2_dir,
        sample_updates,
        include_harvested=include_harvested,
    )
    rng = np.random.default_rng(seed)
    replace = pool.shape[0] < num_positions
    indices = rng.choice(pool.shape[0], size=num_positions, replace=replace)
    return np.ascontiguousarray(pool[indices])


def obs_hash(obs: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(obs).tobytes()).hexdigest()
    return digest[:16]


def run_mcts_targets(
    *,
    stage2_dir: Path,
    output_dir: Path,
    update: int,
    run_index: int,
    run_seed: int,
    obs: np.ndarray,
    obs_id: str,
    num_simulations: int,
    root_batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = output_dir / "cache"
    ensure_output_dir(cache_dir)
    cache_name = (
        f"targets_u{update}_r{run_index}_seed{run_seed}_n{len(obs)}"
        f"_sims{num_simulations}_{obs_id}.npz"
    )
    cache_path = cache_dir / cache_name
    if cache_path.exists() and not overwrite:
        with np.load(cache_path) as data:
            return data["dist"], data["value"]

    print(
        f"running MCTS: checkpoint={update} run={run_index} "
        f"positions={len(obs)} sims={num_simulations} seed={run_seed}",
        flush=True,
    )
    seed_everything(run_seed)
    model = load_model(stage2_dir, update, device)
    model.eval()
    clear_nn_eval_cache()

    dist_parts = []
    value_parts = []
    t0 = time.time()
    for start in range(0, len(obs), root_batch_size):
        end = min(start + root_batch_size, len(obs))
        boards = [
            board_from_observation(o, player_to_move_from_obs(o))
            for o in obs[start:end]
        ]
        visit_dists, root_values, _, _, _ = mcts_search_batched(
            model=model,
            boards=boards,
            num_simulations=num_simulations,
            c_puct=train_main.C_PUCT,
            entropy_multiplier=None,
            device=device,
            dirichlet_alpha=train_main.STAGE2_DIRICHLET_ALPHA,
            dirichlet_epsilon=train_main.STAGE2_DIRICHLET_EPSILON,
            gamma=train_main.DISCOUNT_GAMMA,
            fpu_multiplier=train_main.FPU_MULTIPLIER,
            harvest_min_visits=None,
        )
        dist_parts.append(visit_dists.astype(np.float32, copy=False))
        value_parts.append(root_values.astype(np.float32, copy=False))
        print(f"  roots {end}/{len(obs)}", flush=True)

    dists = np.concatenate(dist_parts, axis=0)
    values = np.concatenate(value_parts, axis=0)
    np.savez_compressed(cache_path, dist=dists, value=values)
    clear_nn_eval_cache()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    return dists, values


def js_divergence_per_position(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (np.log(p + 1e-30) - np.log(m + 1e-30))).sum(axis=-1)
        + (q * (np.log(q + 1e-30) - np.log(m + 1e-30))).sum(axis=-1)
    )
    return js.astype(np.float64, copy=False)


def pair_metrics(
    a_dist: list[np.ndarray],
    a_value: list[np.ndarray],
    b_dist: list[np.ndarray],
    b_value: list[np.ndarray],
    *,
    same_group: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if same_group:
        pairs = itertools.combinations(range(len(a_dist)), 2)
        js_parts = []
        q_parts = []
        for i, j in pairs:
            js_parts.append(js_divergence_per_position(a_dist[i], a_dist[j]))
            q_parts.append((a_value[i] - a_value[j]).astype(np.float64) ** 2)
    else:
        js_parts = []
        q_parts = []
        for i in range(len(a_dist)):
            for j in range(len(b_dist)):
                js_parts.append(js_divergence_per_position(a_dist[i], b_dist[j]))
                q_parts.append((a_value[i] - b_value[j]).astype(np.float64) ** 2)

    if not js_parts:
        raise RuntimeError("Need at least one run pair for drift metrics")
    return np.mean(np.stack(js_parts, axis=0), axis=0), np.mean(np.stack(q_parts, axis=0), axis=0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    configure_torch()
    stage2_dir = args.stage2_dir
    output_dir = args.output_dir
    ensure_output_dir(output_dir)

    device = resolve_device(args.device)
    anchor_update = resolve_anchor(stage2_dir, args.anchor_update)
    horizons = resolve_horizons(args)
    comparison_updates = [anchor_update - h for h in horizons]
    missing = [u for u in comparison_updates if u <= 0]
    if missing:
        raise ValueError(f"Horizons point before checkpoint 1: {missing}")

    if args.sample_updates:
        sample_updates = args.sample_updates
    else:
        sample_updates = default_recent_sample_updates(
            stage2_dir, train_main.STAGE2_REPLAY_BUFFER_ROUNDS
        )

    obs = select_positions(
        stage2_dir,
        sample_updates,
        args.num_positions,
        seed=args.seed,
        include_harvested=args.include_harvested,
    )
    selected_id = obs_hash(obs)
    np.savez_compressed(output_dir / f"selected_positions_{selected_id}.npz", obs=obs)

    metadata = {
        "anchor_update": anchor_update,
        "horizons": horizons,
        "comparison_updates": comparison_updates,
        "sample_updates": sample_updates,
        "num_positions": len(obs),
        "position_hash": selected_id,
        "num_simulations": args.num_simulations,
        "floor_runs": args.floor_runs,
        "history_runs": args.history_runs,
        "root_batch_size": args.root_batch_size,
        "include_harvested": args.include_harvested,
        "device": str(device),
        "constants": constants_snapshot(),
    }
    write_json(output_dir / "q1_metadata.json", metadata)

    runs_dist: dict[int, list[np.ndarray]] = {}
    runs_value: dict[int, list[np.ndarray]] = {}

    def collect(update: int, n_runs: int) -> None:
        if update in runs_dist:
            return
        runs_dist[update] = []
        runs_value[update] = []
        for run_index in range(n_runs):
            run_seed = args.seed + update * 100_000 + run_index * 9973
            dists, values = run_mcts_targets(
                stage2_dir=stage2_dir,
                output_dir=output_dir,
                update=update,
                run_index=run_index,
                run_seed=run_seed,
                obs=obs,
                obs_id=selected_id,
                num_simulations=args.num_simulations,
                root_batch_size=args.root_batch_size,
                device=device,
                overwrite=args.overwrite,
            )
            runs_dist[update].append(dists)
            runs_value[update].append(values)

    collect(anchor_update, args.floor_runs)
    for update in comparison_updates:
        collect(update, args.history_runs)

    floor_js, floor_q = pair_metrics(
        runs_dist[anchor_update],
        runs_value[anchor_update],
        runs_dist[anchor_update],
        runs_value[anchor_update],
        same_group=True,
    )
    floor_js_summary = percentile_summary(floor_js)
    floor_q_summary = percentile_summary(floor_q)

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "metadata": metadata,
        "floor": {
            "policy_js": floor_js_summary,
            "value_mse": floor_q_summary,
        },
        "horizons": {},
    }
    per_position_npz: dict[str, np.ndarray] = {
        "floor_policy_js": floor_js.astype(np.float32),
        "floor_value_mse": floor_q.astype(np.float32),
    }

    for horizon, update in zip(horizons, comparison_updates):
        drift_js, drift_q = pair_metrics(
            runs_dist[anchor_update],
            runs_value[anchor_update],
            runs_dist[update],
            runs_value[update],
            same_group=False,
        )
        excess_js = drift_js - floor_js
        excess_q = drift_q - floor_q
        horizon_summary = {
            "comparison_update": update,
            "policy_js": percentile_summary(drift_js),
            "value_mse": percentile_summary(drift_q),
            "excess_policy_js": percentile_summary(excess_js),
            "excess_value_mse": percentile_summary(excess_q),
            "mean_excess_policy_js_clamped": float(np.maximum(excess_js, 0.0).mean()),
            "mean_excess_value_mse_clamped": float(np.maximum(excess_q, 0.0).mean()),
        }
        summary["horizons"][str(horizon)] = horizon_summary
        per_position_npz[f"h{horizon}_policy_js"] = drift_js.astype(np.float32)
        per_position_npz[f"h{horizon}_value_mse"] = drift_q.astype(np.float32)
        per_position_npz[f"h{horizon}_excess_policy_js"] = excess_js.astype(np.float32)
        per_position_npz[f"h{horizon}_excess_value_mse"] = excess_q.astype(np.float32)

        rows.append({
            "horizon": horizon,
            "comparison_update": update,
            "floor_policy_js_mean": floor_js_summary["mean"],
            "policy_js_mean": horizon_summary["policy_js"]["mean"],
            "excess_policy_js_mean": horizon_summary["excess_policy_js"]["mean"],
            "excess_policy_js_p95": horizon_summary["excess_policy_js"]["p95"],
            "floor_value_mse_mean": floor_q_summary["mean"],
            "value_mse_mean": horizon_summary["value_mse"]["mean"],
            "excess_value_mse_mean": horizon_summary["excess_value_mse"]["mean"],
            "excess_value_mse_p95": horizon_summary["excess_value_mse"]["p95"],
        })

    write_json(output_dir / "q1_summary.json", summary)
    write_csv(output_dir / "q1_summary.csv", rows)
    np.savez_compressed(output_dir / "q1_per_position_metrics.npz", **per_position_npz)

    print(f"wrote {output_dir / 'q1_summary.json'}", flush=True)
    print(f"wrote {output_dir / 'q1_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
