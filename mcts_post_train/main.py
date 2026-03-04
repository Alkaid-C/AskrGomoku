"""
MCTS Post-Training Entry Point

Takes an RL-trained checkpoint and refines it via MCTS-guided distillation.
Uses entropy-preserving temperature to gradually adapt the model to MCTS
search distributions without corrupting the learned policy.

Usage:
    python main.py <output_dir> --checkpoint <path> [--opponent-pool-dir <path>]
"""

import os
import sys

sys.path.insert(0, os.getcwd())

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from csv_logger import MCTSCSVLogger
from eval import (
    EVAL_ROUNDS,
    OPPONENT_POOL_SIZE,
    WIN_RATE_THRESHOLD,
    add_opponent_to_pool,
    evaluate_policy,
    load_checkpoint_model,
    sample_opponent_weighted,
)
from gomoku import (
    RENJU_OPENING_SEQUENCES,
    SEED_PROBABILITY,
    GameState,
)
from model import GomokuPolicyNet
from self_play import play_mcts_games
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

NUM_SIMULATIONS = 400
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25
ACTION_TEMPERATURE = 1.0

TOTAL_UPDATES = 3000
EPISODES_PER_UPDATE = 96
LEARNING_RATE = 0.5 / 8192
MIN_LR = LEARNING_RATE / 8
WEIGHT_DECAY = 1.0 / 2 ** 24

TEMP_EMA_WINDOW = 64
TEMP_CONVERGENCE_EXPONENT = 0.99
INITIAL_TEMPERATURE = 1.72

EVAL_INTERVAL = 32
PRINT_INTERVAL = 1
UPDATE_ID_OFFSET = 65537  # post-train IDs start after RL (which ends at 65536)

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


def save_training_state(
    output_dir: str, update: int,
    opponent_pool_updates: List[int],
    per_opponent_win_rates: Dict[str, float],
    temperature: float,
    opponent_pool_dir: Optional[str] = None,
) -> None:
    state: dict = {
        'current_update': update,
        'opponent_pool_updates': opponent_pool_updates,
        'per_opponent_win_rates': per_opponent_win_rates,
        'temperature': temperature,
    }
    if opponent_pool_dir is not None:
        state['opponent_pool_dir'] = opponent_pool_dir
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def load_training_state(output_dir: str, device: torch.device) -> Optional[Tuple]:
    """
    Load training state for resume.

    Returns:
        (model, optimizer, scheduler, opponent_pool, opponent_pool_updates,
         start_update, per_opponent_win_rates, temperature,
         opponent_pool_dir) or None
    """
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(path):
        return None

    print(f"Found training state: {path}")
    with open(path) as f:
        state = json.load(f)

    current_update = state['current_update']
    opponent_pool_updates = state['opponent_pool_updates']
    per_opponent_win_rates = state.get('per_opponent_win_rates', {})
    temperature = state.get('temperature', INITIAL_TEMPERATURE)

    print(f"Resuming from update {current_update}, T={temperature:.4f}")

    ckpt_id = UPDATE_ID_OFFSET + current_update
    checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{ckpt_id}.pt")
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

    # Load opponent pool — try output_dir first, fall back to saved external dir
    opponent_pool_dir = state.get('opponent_pool_dir')
    search_dirs = [output_dir]
    if opponent_pool_dir and opponent_pool_dir != output_dir:
        search_dirs.append(opponent_pool_dir)

    opponent_pool: deque[torch.nn.Module] = deque()
    loaded_updates: list[int] = []
    for pool_update in opponent_pool_updates:
        opp = None
        for search_dir in search_dirs:
            pool_path = os.path.join(search_dir, f"checkpoint_update_{pool_update}.pt")
            opp = load_checkpoint_model(pool_path, device)
            if opp is not None:
                break
        if opp is not None:
            opponent_pool.append(opp)
            loaded_updates.append(pool_update)

    if not opponent_pool:
        raise RuntimeError("No opponent models could be loaded")

    print(f"Loaded {len(opponent_pool)} opponents: {loaded_updates}")

    return (model, optimizer, scheduler, opponent_pool, loaded_updates,
            current_update, per_opponent_win_rates, temperature, opponent_pool_dir)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description='MCTS post-training for Gomoku')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to RL checkpoint to refine')
    parser.add_argument('--opponent-pool-dir', type=str, required=False,
                        help='Directory with checkpoint_update_*.pt for initial opponent pool')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_logger = MCTSCSVLogger(output_dir)
    print(f"Output: {output_dir}")
    print(f"MCTS: {NUM_SIMULATIONS} sims, c_puct={C_PUCT}")
    print(f"Temperature: initial={INITIAL_TEMPERATURE}, convergence={TEMP_CONVERGENCE_EXPONENT}")
    print(f"Training: {TOTAL_UPDATES} updates, {EPISODES_PER_UPDATE} games/update")
    print(f"LR: {LEARNING_RATE} -> {MIN_LR} (cosine)")
    print()

    # Try resume
    resume_result = load_training_state(output_dir, DEVICE)

    if resume_result is not None:
        (model, optimizer, scheduler, opponent_pool, opponent_pool_updates,
         start_update, per_opponent_win_rates, temperature,
         opponent_pool_dir) = resume_result
        print(f"Resumed from update {start_update}")
        print()
    else:
        # Fresh start — both flags required
        if args.checkpoint is None:
            parser.error("--checkpoint is required for fresh training")
        if args.opponent_pool_dir is None:
            parser.error("--opponent-pool-dir is required for fresh training")

        seed_everything(SEED)
        print(f"Seed: {SEED}")

        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError(f"Checkpoint missing 'model_state_dict': {args.checkpoint}")
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
        per_opponent_win_rates: Dict[str, float] = {}

        # Load opponent pool from RL training directory
        import glob
        import re
        opponent_pool: deque[torch.nn.Module] = deque()
        opponent_pool_updates: List[int] = []
        opponent_pool_dir: Optional[str] = os.path.abspath(args.opponent_pool_dir)

        pattern = re.compile(r'checkpoint_update_(\d+)\.pt')
        files = glob.glob(os.path.join(args.opponent_pool_dir, "checkpoint_update_*.pt"))
        updates_found = []
        for f in files:
            m = pattern.match(os.path.basename(f))
            if m:
                updates_found.append(int(m.group(1)))
        updates_found.sort()

        if len(updates_found) > OPPONENT_POOL_SIZE:
            step = len(updates_found) / OPPONENT_POOL_SIZE
            selected = [updates_found[int(i * step)] for i in range(OPPONENT_POOL_SIZE)]
        else:
            selected = updates_found

        for upd in selected:
            cp_path = os.path.join(args.opponent_pool_dir, f"checkpoint_update_{upd}.pt")
            opp = load_checkpoint_model(cp_path, DEVICE)
            if opp is not None:
                opponent_pool.append(opp)
                opponent_pool_updates.append(upd)

        if not opponent_pool:
            raise RuntimeError(f"No valid checkpoints found in {args.opponent_pool_dir}")
        print(f"Loaded {len(opponent_pool)} opponents from {args.opponent_pool_dir}")
        print()

    # Training loop
    training_start_time = time.time()

    for update in range(start_update, TOTAL_UPDATES):
        t_start = time.time()

        # Sample opponents and prepare games
        pairs_current_is_black: list[bool] = []
        opponents: list[torch.nn.Module] = []
        opening_ids: list[int] = []
        num_openings = len(RENJU_OPENING_SEQUENCES)

        for _ in range(EPISODES_PER_UPDATE):
            opp = sample_opponent_weighted(opponent_pool, opponent_pool_updates, per_opponent_win_rates)
            opponents.append(opp)
            pairs_current_is_black.append(random.random() < 0.5)

            if random.random() < SEED_PROBABILITY:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        # Self-play with MCTS
        t0 = time.time()
        model.eval()
        records = play_mcts_games(
            current_model=model,
            opponent_models=opponents,
            current_is_black=pairs_current_is_black,
            num_simulations=NUM_SIMULATIONS,
            c_puct=C_PUCT,
            prior_temperature=temperature,
            device=DEVICE,
            opening_ids=opening_ids,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            action_temperature=ACTION_TEMPERATURE,
        )
        model.train()
        t_selfplay = time.time() - t0

        # Compute game outcome stats
        # Build pseudo-trajectories for compute_outcome_stats
        game_lengths = []
        current_wins = 0
        draws = 0
        for i, record in enumerate(records):
            game_lengths.append(len(record.actions))
            is_black = pairs_current_is_black[i]
            if record.outcome == GameState.DRAW:
                draws += 1
            elif (record.outcome == GameState.BLACK_WIN and is_black) or (record.outcome == GameState.WHITE_WIN and not is_black):
                current_wins += 1

        win_rate = current_wins / EPISODES_PER_UPDATE
        draw_rate = draws / EPISODES_PER_UPDATE
        avg_game_length = float(np.mean(game_lengths)) if game_lengths else 0.0

        # Compute sharpen exponent
        sharpen_exponent = temperature ** TEMP_CONVERGENCE_EXPONENT

        # Train
        t0 = time.time()
        train_results = train_on_mcts_batch(
            model, records, optimizer, DEVICE,
            sharpen_exponent=sharpen_exponent,
        )
        t_train = time.time() - t0

        # Update temperature via EMA of entropy ratio
        model_entropy = train_results['model_entropy']
        mcts_entropy = train_results['mcts_entropy']
        if model_entropy > 0:
            ratio = mcts_entropy / model_entropy
            ema_alpha = 1.0 / TEMP_EMA_WINDOW
            temperature = temperature + ema_alpha * (ratio - temperature)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        t_total = time.time() - t_start

        # Print
        if (update + 1) % PRINT_INTERVAL == 0:
            elapsed = time.time() - training_start_time
            eta = elapsed / (update - start_update + 1) * (TOTAL_UPDATES - update - 1) if update > start_update else 0

            def fmt_time(s: float) -> str:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                if h > 0:
                    return f"{h}h{m:02d}m"
                return f"{m}m{int(s%60):02d}s"

            print(
                f"Update {update+1:4d}/{TOTAL_UPDATES} | "
                f"WR: {win_rate:.0%} D: {draw_rate:.0%} | "
                f"Len: {avg_game_length:.0f} | "
                f"Ent: {model_entropy:.3f}/{mcts_entropy:.3f} | "
                f"T: {temperature:.4f} | "
                f"PLoss: {train_results['policy_loss']:.4f} VLoss: {train_results['value_loss']:.4f} | "
                f"{t_total:.1f}s (sp:{t_selfplay:.1f} tr:{t_train:.1f}) | "
                f"{fmt_time(elapsed)}/{fmt_time(eta)}"
            )

            csv_logger.log_training_update(update + 1, {
                'policy_loss': train_results['policy_loss'],
                'value_loss': train_results['value_loss'],
                'model_entropy': model_entropy,
                'mcts_entropy': mcts_entropy,
                'temperature': temperature,
                'sharpen_exponent': sharpen_exponent,
                'lr': current_lr,
                'avg_game_length': avg_game_length,
                'win_rate': win_rate,
                'draw_rate': draw_rate,
                'time_selfplay': t_selfplay,
                'time_train': t_train,
            })

        # Evaluation
        if (update + 1) % EVAL_INTERVAL == 0:
            print(f"\n--- Eval @ {update+1} ---")

            # Checkpoint ID uses offset to avoid collision with RL checkpoint IDs
            ckpt_id = UPDATE_ID_OFFSET + update + 1
            checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{ckpt_id}.pt")
            torch.save({
                'update': update + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"  Saved: {os.path.basename(checkpoint_path)}")

            eval_start = time.time()
            eval_win_rate, per_opp_stats = evaluate_policy(
                model, opponent_pool, DEVICE,
                opponent_pool_updates=opponent_pool_updates,
            )
            eval_time = time.time() - eval_start

            for opp_key, opp_stats in per_opp_stats.items():
                per_opponent_win_rates[opp_key] = opp_stats['win_rate']

            sorted_opps = sorted(per_opp_stats.items(), key=lambda x: x[1]['win_rate'])
            hardest_id = int(sorted_opps[0][0]) if sorted_opps else -1
            hardest_wr = sorted_opps[0][1]['win_rate'] if sorted_opps else 0.0
            easiest_id = int(sorted_opps[-1][0]) if sorted_opps else -1
            easiest_wr = sorted_opps[-1][1]['win_rate'] if sorted_opps else 0.0

            total_eval_games = EVAL_ROUNDS * 2 * len(opponent_pool)
            print(f"  WR: {eval_win_rate:.1%} ({total_eval_games} games, {eval_time:.1f}s) | "
                  f"Hard: {hardest_id}:{hardest_wr:.0%} Easy: {easiest_id}:{easiest_wr:.0%}")

            checkpoint_added = False
            evicted_id = -1
            if eval_win_rate >= WIN_RATE_THRESHOLD:
                checkpoint_added = True
                evicted = add_opponent_to_pool(
                    opponent_pool, opponent_pool_updates, model, ckpt_id,
                    per_opponent_win_rates, DEVICE,
                )
                if evicted is not None:
                    evicted_id = evicted
                    print(f"  Pool: +current -{evicted}")
                    per_opponent_win_rates.pop(str(evicted), None)
                else:
                    print("  Pool: +current")
            else:
                print(f"  Pool: no change (WR {eval_win_rate:.1%} < {WIN_RATE_THRESHOLD:.1%})")

            csv_logger.log_eval_summary(update + 1, {
                'overall_win_rate': eval_win_rate,
                'total_games': total_eval_games,
                'eval_time': eval_time,
                'hardest_opponent_id': hardest_id,
                'hardest_win_rate': hardest_wr,
                'easiest_opponent_id': easiest_id,
                'easiest_win_rate': easiest_wr,
                'pool_size': len(opponent_pool),
                'checkpoint_added': checkpoint_added,
                'evicted_opponent_id': evicted_id,
            })

            for opp_key, opp_stats in per_opp_stats.items():
                losses = opp_stats['games'] - opp_stats['wins'] - opp_stats['draws']
                csv_logger.log_eval_opponent_details(update + 1, int(opp_key), {
                    'wins': opp_stats['wins'],
                    'losses': losses,
                    'draws': opp_stats['draws'],
                    'games': opp_stats['games'],
                    'win_rate': opp_stats['win_rate'],
                })

            save_training_state(output_dir, update + 1, opponent_pool_updates,
                                per_opponent_win_rates, temperature,
                                opponent_pool_dir)
            print(f"  Next eval: {update + 1 + EVAL_INTERVAL}")
            torch.cuda.empty_cache()
            print()

    # Final save — checkpoint must exist for resume to work
    final_ckpt_id = UPDATE_ID_OFFSET + TOTAL_UPDATES
    final_checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{final_ckpt_id}.pt")
    torch.save({
        'update': TOTAL_UPDATES,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, final_checkpoint_path)

    final_path = os.path.join(output_dir, "final_policy.pt")
    torch.save({'model_state_dict': model.state_dict(), 'update': TOTAL_UPDATES}, final_path)
    print(f"\nTraining complete! Final model: {final_path}")

    save_training_state(output_dir, TOTAL_UPDATES, opponent_pool_updates,
                        per_opponent_win_rates, temperature, opponent_pool_dir)


if __name__ == "__main__":
    main()
