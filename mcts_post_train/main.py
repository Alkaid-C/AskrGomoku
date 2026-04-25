"""
MCTS Post-Training Entry Point

Takes an RL-trained checkpoint and refines it via MCTS-guided distillation
with pure self-play. Uses entropy-preserving temperature to gradually adapt
the model to MCTS search distributions without corrupting the learned policy.

Usage:
    python main.py <output_dir> --checkpoint <path>
"""

import os
import sys

sys.path.insert(0, os.getcwd())

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
from csv_logger import MCTSCSVLogger
from gomoku import (
    RENJU_OPENING_SEQUENCES,
    SEED_PROBABILITY,
    GameState,
)
from mcts import clear_nn_eval_cache, get_nn_eval_cache_stats
from model import GomokuPolicyNet
from self_play import compute_block_rates, play_mcts_games
from training import train_on_mcts_batch

# ============================================================================
# PyTorch Performance Settings
# ============================================================================

DEVICE = torch.device("cuda")
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'

# ============================================================================
# MCTS Post-Training Constants
# ============================================================================

NUM_SIMULATIONS = 1024
C_PUCT = 1.25
DISCOUNT_GAMMA = 0.98
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25

TOTAL_UPDATES = 8192
EPISODES_PER_UPDATE = 96
LEARNING_RATE = 0.5 / 8192
MIN_LR = LEARNING_RATE / 2
WEIGHT_DECAY = 1.0 / 2 ** 24

TEMP_EMA_WINDOW = 64
TEMP_CONVERGENCE_EXPONENT = 0.99
INITIAL_TEMPERATURE = 1.4

CHECKPOINT_INTERVAL = 32
PRINT_INTERVAL = 1

# ============================================================================
# State Management
# ============================================================================

TRAINING_STATE_FILE = "training_state.json"
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fmt_time(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{int(s%60):02d}s"


def save_training_state(output_dir: str, update: int, temperature: float) -> None:
    state = {
        'current_update': update,
        'temperature': temperature,
    }
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def load_training_state(output_dir: str, device: torch.device) -> Optional[Tuple]:
    """
    Load training state for resume.

    Returns:
        (model, optimizer, scheduler, start_update, temperature) or None
    """
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(path):
        return None

    print(f"Found training state: {path}")
    with open(path) as f:
        state = json.load(f)

    current_update = state['current_update']
    temperature = state.get('temperature', INITIAL_TEMPERATURE)

    print(f"Resuming from update {current_update}, T={temperature:.4f}")

    checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{current_update}.pt")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GomokuPolicyNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_UPDATES, eta_min=MIN_LR
    )
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return (model, optimizer, scheduler, current_update, temperature)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description='MCTS post-training for Gomoku')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to RL checkpoint to refine (required for fresh start)')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_logger = MCTSCSVLogger(output_dir)
    print(f"Output: {output_dir}")
    print(f"MCTS: {NUM_SIMULATIONS} sims, c_puct={C_PUCT}, gamma={DISCOUNT_GAMMA}")
    print(f"Temperature: initial={INITIAL_TEMPERATURE}, convergence={TEMP_CONVERGENCE_EXPONENT}")
    print(f"Training: {TOTAL_UPDATES} updates, {EPISODES_PER_UPDATE} games/update (pure self-play)")
    print(f"LR: {LEARNING_RATE} -> {MIN_LR} (cosine)")
    print()

    resume_result = load_training_state(output_dir, DEVICE)

    if resume_result is not None:
        model, optimizer, scheduler, start_update, temperature = resume_result
        print(f"Resumed from update {start_update}")
        print()
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for fresh training")

        seed_everything(SEED)
        print(f"Seed: {SEED}")

        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model = GomokuPolicyNet().to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TOTAL_UPDATES, eta_min=MIN_LR
        )

        temperature = INITIAL_TEMPERATURE
        start_update = 0

    # Training loop
    training_start_time = time.time()
    num_openings = len(RENJU_OPENING_SEQUENCES)

    for update in range(start_update, TOTAL_UPDATES):
        t_start = time.time()

        opening_ids: list[int] = []
        for _ in range(EPISODES_PER_UPDATE):
            if random.random() < SEED_PROBABILITY:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        # Self-play with MCTS (current model plays both sides)
        t0 = time.time()
        model.eval()
        records = play_mcts_games(
            model=model,
            num_games=EPISODES_PER_UPDATE,
            num_simulations=NUM_SIMULATIONS,
            c_puct=C_PUCT,
            entropy_multiplier=temperature,
            device=DEVICE,
            opening_ids=opening_ids,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            gamma=DISCOUNT_GAMMA,
        )
        block_stats = compute_block_rates(records, model, DEVICE)
        model.train()
        t_selfplay = time.time() - t0

        # Outcome stats: black/draw rate + avg length (pure self-play ⇒ no
        # "current" player, but black-win % is a useful first-player-advantage signal)
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

        black_win_rate = black_wins / EPISODES_PER_UPDATE
        draw_rate = draws / EPISODES_PER_UPDATE
        avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0

        entropy_divisor = temperature ** TEMP_CONVERGENCE_EXPONENT

        cache_hits, cache_misses = get_nn_eval_cache_stats()
        cache_total = cache_hits + cache_misses
        cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0

        current_lr = optimizer.param_groups[0]['lr']

        # Train
        t0 = time.time()
        train_results = train_on_mcts_batch(
            model, records, optimizer, DEVICE,
            entropy_divisor=entropy_divisor,
        )
        # Weights changed; canonical-logit cache is stale.
        clear_nn_eval_cache()
        torch.cuda.empty_cache()
        t_train = time.time() - t0

        model_entropy = train_results['model_entropy']
        mcts_entropy = train_results['mcts_entropy']
        sharpened_entropy = train_results['sharpened_entropy']

        t_total = time.time() - t_start

        if (update + 1) % PRINT_INTERVAL == 0:
            elapsed = time.time() - training_start_time
            eta = elapsed / (update - start_update + 1) * (TOTAL_UPDATES - update - 1) if update > start_update else 0

            def _fmt_rate(r: float) -> str:
                return "--" if r != r else f"{r:.0%}"  # r != r catches NaN

            blk_line = (
                f"Blk B:M{_fmt_rate(block_stats['black_block_mcts_rate'])}"
                f"/R{_fmt_rate(block_stats['black_block_raw_rate'])}"
                f"({block_stats['black_block_opps']}) "
                f"W:M{_fmt_rate(block_stats['white_block_mcts_rate'])}"
                f"/R{_fmt_rate(block_stats['white_block_raw_rate'])}"
                f"({block_stats['white_block_opps']})"
            )

            print(
                f"Update {update+1:4d}/{TOTAL_UPDATES} | "
                f"BlackWR: {black_win_rate:.0%} D: {draw_rate:.0%} | "
                f"Len: {avg_game_length:.0f} | "
                f"Ent: {model_entropy:.3f}/{sharpened_entropy:.3f}/{mcts_entropy:.3f} | "
                f"T: {temperature:.4f} | "
                f"PLoss: {train_results['policy_loss']:.4f} VLoss: {train_results['value_loss']:.4f} | "
                f"{blk_line} | "
                f"Cache: {cache_hit_rate:.0%} ({cache_hits}/{cache_total}) | "
                f"{t_total:.1f}s (sp:{t_selfplay:.1f} tr:{t_train:.1f}) | "
                f"{fmt_time(elapsed)}/{fmt_time(eta)}"
            )

        csv_logger.log_training_update(update + 1, {
            'policy_loss': train_results['policy_loss'],
            'value_loss': train_results['value_loss'],
            'model_entropy': model_entropy,
            'mcts_entropy': mcts_entropy,
            'sharpened_entropy': sharpened_entropy,
            'temperature': temperature,
            'entropy_divisor': entropy_divisor,
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

        # Advance temperature (EMA of entropy ratio) and LR for the next update.
        # Done after logging so the row above is a snapshot of the state used
        # for this update's self-play and training. Floor on model_entropy guards
        # against a nearly-deterministic model blowing up `ratio = mcts / model`.
        if model_entropy > 1e-3:
            ratio = mcts_entropy / model_entropy
            ema_alpha = 1.0 / TEMP_EMA_WINDOW
            temperature = temperature + ema_alpha * (ratio - temperature)
        scheduler.step()

        if (update + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_id = update + 1
            checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{ckpt_id}.pt")
            torch.save({
                'update': ckpt_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            save_training_state(output_dir, ckpt_id, temperature)
            print(f"  Saved: {os.path.basename(checkpoint_path)}")

    # Final save
    final_checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{TOTAL_UPDATES}.pt")
    torch.save({
        'update': TOTAL_UPDATES,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, final_checkpoint_path)

    final_path = os.path.join(output_dir, "final_policy.pt")
    torch.save({'model_state_dict': model.state_dict(), 'update': TOTAL_UPDATES}, final_path)
    print(f"\nTraining complete! Final model: {final_path}")

    save_training_state(output_dir, TOTAL_UPDATES, temperature)


if __name__ == "__main__":
    main()
