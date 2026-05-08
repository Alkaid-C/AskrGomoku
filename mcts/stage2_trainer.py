"""
Stage 2 trainer: vanilla AlphaZero MCTS self-play.

The student loaded from stage 1 plays itself with raw priors (no entropy
multiplier, no sharpening); Dirichlet noise at the root is the sole
exploration source. Targets are the raw MCTS visit distributions.
"""

import json
import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
from csv_logger import Stage2CSVLogger
from gomoku import RENJU_OPENING_SEQUENCES, GameState
from mcts import clear_nn_eval_cache, get_nn_eval_cache_stats
from model import GomokuPolicyNet
from self_play import compute_block_rates, play_mcts_games
from training import train_on_mcts_batch

TRAINING_STATE_FILE = "training_state.json"
FINAL_NAME = "final_policy.pt"


def _save_checkpoint(
    path: str,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    torch.save({
        'update': update,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


def _save_state(output_dir: str, update: int) -> None:
    state = {'current_update': update}
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _fmt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{int(s % 60):02d}s"


def _fmt_rate(r: float) -> str:
    return "--" if r != r else f"{r:.0%}"


def _try_resume(
    output_dir: str,
    total_updates: int,
    learning_rate: float,
    min_lr: float,
    weight_decay: float,
    device: torch.device,
) -> Optional[Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    int,
]]:
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        state = json.load(f)
    current_update = state['current_update']
    print(f"Resuming stage 2 from update {current_update}")

    ckpt_path = os.path.join(output_dir, f"checkpoint_update_{current_update}.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True
    )
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=min_lr
    )
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    model.train()
    return model, optimizer, scheduler, current_update


def run_stage2(
    stage1_checkpoint: str,
    output_dir: str,
    *,
    total_updates: int,
    episodes_per_update: int,
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    seed_probability: float,
    gamma: float,
    learning_rate: float,
    min_lr: float,
    weight_decay: float,
    value_loss_coeff: float,
    checkpoint_interval: int,
    device: torch.device,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_logger = Stage2CSVLogger(output_dir)

    print(f"Output: {output_dir}")
    print(f"MCTS: {num_simulations} sims, c_puct={c_puct}, gamma={gamma}")
    print(f"Dirichlet: alpha={dirichlet_alpha}, epsilon={dirichlet_epsilon}")
    print(f"Training: {total_updates} updates, {episodes_per_update} games/update")
    print(f"LR: {learning_rate} -> {min_lr} (cosine)")
    print()

    resumed = _try_resume(
        output_dir, total_updates, learning_rate, min_lr, weight_decay, device
    )
    if resumed is not None:
        model, optimizer, scheduler, start_update = resumed
    else:
        if not os.path.exists(stage1_checkpoint):
            raise RuntimeError(
                f"Missing stage 1 final: {stage1_checkpoint} "
                f"(required for fresh start; resume needs {output_dir}/{TRAINING_STATE_FILE})"
            )
        print(f"Loading stage 1 checkpoint: {stage1_checkpoint}")
        ckpt = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
        model = GomokuPolicyNet().to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=min_lr
        )
        start_update = 0

    training_start = time.time()
    num_openings = len(RENJU_OPENING_SEQUENCES)

    for update in range(start_update, total_updates):
        t_start = time.time()

        opening_ids: list[int] = []
        for _ in range(episodes_per_update):
            if random.random() < seed_probability:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        t0 = time.time()
        model.eval()
        records = play_mcts_games(
            model=model,
            num_games=episodes_per_update,
            num_simulations=num_simulations,
            c_puct=c_puct,
            entropy_multiplier=None,
            device=device,
            opening_ids=opening_ids,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            gamma=gamma,
            action_temperature=1.0,
        )
        block_stats = compute_block_rates(records, model, device)
        model.train()
        t_selfplay = time.time() - t0

        game_lengths = []
        black_wins = 0
        draws = 0
        for record in records:
            game_lengths.append(len(record.observations))
            assert record.outcome is not None, "Game did not terminate"
            if record.outcome == GameState.DRAW:
                draws += 1
            elif record.outcome == GameState.BLACK_WIN:
                black_wins += 1

        black_win_rate = black_wins / episodes_per_update
        draw_rate = draws / episodes_per_update
        avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0

        cache_hits, cache_misses = get_nn_eval_cache_stats()
        cache_total = cache_hits + cache_misses
        cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']

        t0 = time.time()
        train_results = train_on_mcts_batch(
            model, records, optimizer, device, value_loss_coeff=value_loss_coeff,
        )
        clear_nn_eval_cache()
        torch.cuda.empty_cache()
        t_train = time.time() - t0

        scheduler.step()

        elapsed = time.time() - training_start
        eta = elapsed / (update - start_update + 1) * (total_updates - update - 1) \
            if update > start_update else 0.0
        blk_line = (
            f"Blk B:M{_fmt_rate(block_stats['black_block_mcts_rate'])}"
            f"/R{_fmt_rate(block_stats['black_block_raw_rate'])}"
            f"({block_stats['black_block_opps']}) "
            f"W:M{_fmt_rate(block_stats['white_block_mcts_rate'])}"
            f"/R{_fmt_rate(block_stats['white_block_raw_rate'])}"
            f"({block_stats['white_block_opps']})"
        )
        print(
            f"Update {update+1:4d}/{total_updates} | "
            f"BlackWR: {black_win_rate:.0%} D: {draw_rate:.0%} | "
            f"Len: {avg_game_length:.0f} | "
            f"PLoss: {train_results['policy_loss']:.4f} VLoss: {train_results['value_loss']:.4f} | "
            f"KL: {train_results['kl_target_student']:.4f} | "
            f"{blk_line} | "
            f"Cache: {cache_hit_rate:.0%} ({cache_hits}/{cache_total}) | "
            f"{(time.time() - t_start):.1f}s (sp:{t_selfplay:.1f} tr:{t_train:.1f}) | "
            f"{_fmt_time(elapsed)}/{_fmt_time(eta)}"
        )

        csv_logger.log(update + 1, {
            'policy_loss': train_results['policy_loss'],
            'value_loss': train_results['value_loss'],
            'kl_target_student': train_results['kl_target_student'],
            'lr': current_lr,
            'avg_game_length': avg_game_length,
            'black_win_rate': black_win_rate,
            'draw_rate': draw_rate,
            'time_selfplay': t_selfplay,
            'time_train': t_train,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            **block_stats,
        })

        if (update + 1) % checkpoint_interval == 0:
            ckpt_id = update + 1
            ckpt_path = os.path.join(output_dir, f"checkpoint_update_{ckpt_id}.pt")
            _save_checkpoint(ckpt_path, ckpt_id, model, optimizer, scheduler)
            _save_state(output_dir, ckpt_id)
            print(f"  Saved: {os.path.basename(ckpt_path)}")

    final_ckpt = os.path.join(output_dir, f"checkpoint_update_{total_updates}.pt")
    _save_checkpoint(final_ckpt, total_updates, model, optimizer, scheduler)
    final_path = os.path.join(output_dir, FINAL_NAME)
    torch.save({'model_state_dict': model.state_dict(), 'update': total_updates}, final_path)
    _save_state(output_dir, total_updates)
    print(f"\nStage 2 complete! Final model: {final_path}")
