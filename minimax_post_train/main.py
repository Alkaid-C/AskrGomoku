"""
Main Training Script

Entry point for Gomoku self-play training. Contains:
- Training state management (save/load)
- Main training loop
- CLI argument parsing
"""

import os

# Enable expandable segments to reduce CUDA memory fragmentation
# This helps avoid OOM errors when there is reserved but unallocated memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from collections import deque
import numpy as np
import random
import time
import json
import argparse
from typing import Optional, Tuple, Dict, List

from model import GomokuPolicyNet, N_BLOCKS, zero_center_taps, WIDTH
from model import (
    STEM_3X3_CHANNELS, STEM_DIRECTIONAL_5X5_CHANNELS, STEM_FULL_5X5_CHANNELS,
    STEM_DIRECTIONAL_7X7_CHANNELS, STEM_FULL_7X7_CHANNELS,
    TRUNK_DILATION2_SCHEDULE, N_SHARED_BLOCKS, N_DUAL_SE_BLOCKS,
    POLICY_HEAD_D, VALUE_HEAD_C1, VALUE_HEAD_C2_SPLIT, VALUE_HEAD_HIDDEN
)
from gomoku import (
    play_episodes_batched, select_action_batch, compute_outcome_stats,
    play_episodes_with_search, GameState,
    TEMPERATURE_TRAIN, SEED_PROBABILITY, RENJU_OPENING_SEQUENCES,
    SEARCH_DEPTH, ROOT_TOP_K, ROOT_RANDOM_K, INTERNAL_TOP_K, INTERNAL_RANDOM_K, SAMPLING_TAU
)
from enhancement import probe_tactical_accuracy, probe_tactical_accuracy_search
from training import (
    train_on_batch, compute_entropy_schedule,
    train_on_search_samples, apply_freeze_schedule, create_optimizer_for_unfrozen, maybe_update_optimizer,
    TOTAL_UPDATES, LEARNING_RATE, MIN_LR, LR_DECAY_MIDPOINT_PERCENTAGE, LR_DECAY_STEEPNESS, WEIGHT_DECAY,
    EPISODES_PER_UPDATE, EPISODES_CHUNK_SIZE,
    ENTROPY_TARGET_START, ENTROPY_TARGET_END, ENTROPY_BONUS_COEFF,
    ENTROPY_DECAY_MIDPOINT_PERCENTAGE, ENTROPY_DECAY_STEEPNESS, EMA_WINDOW, EVAL_WIN_RATE_EMA_WINDOW,
    VALUE_LOSS_COEFF, GAE_LAMBDA, VALUE_BASELINE_START,
    PRINT_INTERVAL, PROBE_INTERVAL,
    HEADS_ONLY_UPDATES, BLOCK_UNFREEZE_INTERVAL, M_RANK, M_SEP, ALPHA_SEP, LAMBDA_V
)
from eval import (
    create_random_policy, load_checkpoint_model,
    get_eval_interval, evaluate_policy, sample_opponent_weighted,
    add_opponent_to_pool, scan_historical_exploiters,
    OPPONENT_POOL_SIZE, EVAL_ROUNDS, WIN_RATE_THRESHOLD,
    SCAN_START_UPDATE, SCAN_PERIOD, NUM_SCAN_BUCKETS
)

from csv_logger import CSVLogger


# ============================================================================
# PyTorch Performance Settings
# ============================================================================

DEVICE = torch.device("cuda")
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'

# Memory management: CUDA cache is cleared after eval, probing, and mining


# ============================================================================
# State Management Constants
# ============================================================================

TRAINING_STATE_FILE = "training_state.json"


# ============================================================================
# Training State Management
# ============================================================================

def save_training_state(output_dir: str, update: int, opponent_pool_updates: List[int],
                        per_opponent_win_rates: Dict[str, float],
                        scan_event_counter: int,
                        evals_since_last_scan: int,
                        win_rate_ema: float,
                        unfrozen_blocks: int) -> None:
    """Save training state to JSON for resume capability."""
    state = {
        'current_update': update,
        'opponent_pool_updates': opponent_pool_updates,
        'total_updates': TOTAL_UPDATES,
        'per_opponent_win_rates': per_opponent_win_rates,
        'scan_event_counter': scan_event_counter,
        'evals_since_last_scan': evals_since_last_scan,
        'win_rate_ema': win_rate_ema,
        'unfrozen_blocks': unfrozen_blocks
    }

    training_state_file = os.path.join(output_dir, TRAINING_STATE_FILE)
    temp_file = training_state_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(temp_file, training_state_file)


def load_training_state(output_dir: str, device: torch.device) -> Optional[Tuple]:
    """
    Load training state from checkpoint to resume training.

    Returns:
        Tuple of (model, optimizer, scheduler, opponent_pool, opponent_pool_updates, start_update,
                  next_eval_update, per_opponent_win_rates,
                  scan_event_counter, evals_since_last_scan, win_rate_ema, unfrozen_blocks) when loading succeeds.
        Returns None only when no training state file exists.
        Note: unfrozen_blocks may be None if resuming from old state (will be recalculated).

    Raises:
        Exception: If a training state file exists but is corrupt or references
                   missing/invalid checkpoint data.
    """
    training_state_file = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(training_state_file):
        print(f"No training state file found ({training_state_file})")
        return None

    print(f"Found training state file: {training_state_file}")

    # Let JSON errors propagate - corrupted state file should crash
    with open(training_state_file, 'r') as f:
        state = json.load(f)

    current_update = state['current_update']
    opponent_pool_updates = state['opponent_pool_updates']

    per_opponent_win_rates = state.get('per_opponent_win_rates', {})
    scan_event_counter = state.get('scan_event_counter', 0)
    evals_since_last_scan = state.get('evals_since_last_scan', 0)
    win_rate_ema = state.get('win_rate_ema', 0.5)
    unfrozen_blocks = state.get('unfrozen_blocks', None)  # None = recalculate from update

    print(f"Resuming from update {current_update}")
    print(f"Opponent pool has {len(opponent_pool_updates)} models: {opponent_pool_updates}")
    print(f"Win rate EMA: {win_rate_ema:.3f}")
    print(f"Scan state: event_counter={scan_event_counter}, evals_since_last_scan={evals_since_last_scan}")
    if per_opponent_win_rates:
        print(f"Per-opponent win rates: {len(per_opponent_win_rates)} entries")

    checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{current_update}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Main checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    # Let load errors propagate - corrupted checkpoint should crash
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GomokuPolicyNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    zero_center_taps(model)

    # Create optimizer with only unfrozen parameters (applies freeze schedule internally)
    # Note: We don't restore optimizer state dict because param groups may differ
    # after applying freeze schedule. Adam momentum will warm up quickly.
    optimizer = create_optimizer_for_unfrozen(model, current_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"Created optimizer for unfrozen params at update {current_update} (optimizer state reset)")

    def lr_lambda(epoch):
        # Tanh-based decay from LEARNING_RATE to MIN_LR
        midpoint = TOTAL_UPDATES * LR_DECAY_MIDPOINT_PERCENTAGE
        steepness_k = 3.0 / (TOTAL_UPDATES * LR_DECAY_STEEPNESS)
        decay_factor = 0.5 * (1.0 + torch.tanh(torch.tensor(steepness_k * (midpoint - epoch)))).item()
        decayed_lr = MIN_LR + (LEARNING_RATE - MIN_LR) * decay_factor
        return decayed_lr / LEARNING_RATE

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    opponent_pool = deque()

    for pool_update in opponent_pool_updates:
        pool_checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{pool_update}.pt")
        if not os.path.exists(pool_checkpoint_path):
            raise FileNotFoundError(f"Opponent pool checkpoint not found: {pool_checkpoint_path}")

        # Let load errors propagate - corrupted opponent checkpoint should crash
        pool_checkpoint = torch.load(pool_checkpoint_path, map_location=device, weights_only=False)
        pool_model = GomokuPolicyNet().to(device)
        pool_model.load_state_dict(pool_checkpoint['model_state_dict'])
        pool_model.eval()
        opponent_pool.append(pool_model)
        print(f"Loaded opponent from update {pool_update}")

    print(f"Successfully loaded {len(opponent_pool)} opponents")

    next_eval_update = current_update + get_eval_interval(current_update)

    print(f"Next evaluation scheduled at update {next_eval_update}")
    print(f"Resume training starting from update {current_update + 1}")
    print()

    return (model, optimizer, scheduler, opponent_pool, opponent_pool_updates, current_update - 1, next_eval_update,
            per_opponent_win_rates, scan_event_counter, evals_since_last_scan, win_rate_ema, unfrozen_blocks)


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training loop."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Gomoku policy network with self-play', add_help=False)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CSV logger
    csv_logger = CSVLogger(output_dir)
    print(f"Output directory: {output_dir}")
    print()

    effective_chunk_size = min(EPISODES_CHUNK_SIZE, EPISODES_PER_UPDATE)
    if EPISODES_PER_UPDATE < EPISODES_CHUNK_SIZE:
        print(f"NOTE: EPISODES_PER_UPDATE ({EPISODES_PER_UPDATE}) < EPISODES_CHUNK_SIZE ({EPISODES_CHUNK_SIZE})")
        print(f"  Using chunk size = {effective_chunk_size} (no gradient accumulation)")
        print()

    print(f"Using device: {DEVICE}")
    print(f"Model architecture:")
    print(f"  Stem (dilated design):")
    print(f"    - 3x3: {STEM_3X3_CHANNELS}ch")
    print(f"    - 5x5 directional (d1+d2): {STEM_DIRECTIONAL_5X5_CHANNELS}ch, 5x5 full: {STEM_FULL_5X5_CHANNELS}ch")
    print(f"    - 7x7 directional (d1+d2+d3): {STEM_DIRECTIONAL_7X7_CHANNELS}ch, 7x7 full: {STEM_FULL_7X7_CHANNELS}ch")
    print(f"    - Total: {WIDTH} channels (center taps zeroed for d>1)")
    print(f"  Residual blocks: {N_BLOCKS} total ({N_SHARED_BLOCKS} shared + {N_DUAL_SE_BLOCKS} dual-SE) x {WIDTH} channels")
    print(f"    - Dilation schedule (conv2): {TRUNK_DILATION2_SCHEDULE}")
    print(f"    - Shared blocks: no SE | Dual-SE blocks: independent policy/value SE gates")
    print(f"  Policy head: {WIDTH} -> {POLICY_HEAD_D} (+SiLU) -> 225")
    print(f"  Value head: {WIDTH} -> {VALUE_HEAD_C1} -> {VALUE_HEAD_C2_SPLIT*2} -> GAP -> fc{VALUE_HEAD_HIDDEN} -> 1")
    num_accumulation_steps = (EPISODES_PER_UPDATE + effective_chunk_size - 1) // effective_chunk_size

    print(f"Training configuration (POST-TRAINING MODE - Search Supervision):")
    print(f"  Learning rate: {LEARNING_RATE} (tanh decay: mid={LR_DECAY_MIDPOINT_PERCENTAGE:.0%}, steep={LR_DECAY_STEEPNESS:.0%}, min: {MIN_LR})")
    print(f"  Search parameters:")
    print(f"    - Depth: {SEARCH_DEPTH}")
    print(f"    - Root candidates: {ROOT_TOP_K} top + {ROOT_RANDOM_K} random = {ROOT_TOP_K + ROOT_RANDOM_K}")
    print(f"    - Internal candidates: {INTERNAL_TOP_K} top + {INTERNAL_RANDOM_K} random = {INTERNAL_TOP_K + INTERNAL_RANDOM_K}")
    print(f"    - Sampling temperature: {SAMPLING_TAU}")
    print(f"  Loss weights: ranking margin={M_RANK}, separation margin={M_SEP}, alpha_sep={ALPHA_SEP}, lambda_v={LAMBDA_V}")
    print(f"  Progressive unfreezing:")
    print(f"    - Heads only: updates [0, {HEADS_ONLY_UPDATES})")
    print(f"    - Unfreeze blocks: every {BLOCK_UNFREEZE_INTERVAL} updates (from trunk end toward stem)")
    print(f"    - Stem unfreezes after all blocks")
    print(f"  EMA windows: per-update={EMA_WINDOW}, eval win_rate={EVAL_WIN_RATE_EMA_WINDOW}")
    print(f"  Episodes per update: {EPISODES_PER_UPDATE}")
    print(f"  Data augmentation: 8-fold symmetry (rot + flip)")
    print(f"  Tactical probe: every {PROBE_INTERVAL} updates (metrics only)")
    print(f"  Opening seeding: {SEED_PROBABILITY:.0%} of games start from Renju opening ({len(RENJU_OPENING_SEQUENCES)} patterns)")
    print(f"  Opponent pool size: {OPPONENT_POOL_SIZE}")
    print(f"  Eval interval: {get_eval_interval(0)} (early) -> {get_eval_interval(512)} (mid) -> {get_eval_interval(8192)} (late)")
    print(f"  Pool eviction: evict-easiest (by current win rate)")
    print(f"  Historical exploiter scanning:")
    print(f"    - Starts at update {SCAN_START_UPDATE}, every {SCAN_PERIOD} evals")
    print(f"    - {NUM_SCAN_BUCKETS} buckets for round-robin coverage")
    print(f"  Total updates: {TOTAL_UPDATES}")
    print()

    # Try to resume from existing training state
    print("=" * 60)
    print("Checking for existing training state...")
    print("=" * 60)
    resume_result = load_training_state(output_dir, DEVICE)

    if resume_result is not None:
        (current_policy, optimizer, scheduler, opponent_pool, opponent_pool_updates, start_update,
         next_eval_update, per_opponent_win_rates, scan_event_counter,
         evals_since_last_scan, win_rate_ema, unfrozen_blocks_loaded) = resume_result
        print("=" * 60)
        print("Successfully resumed training!")
        print("=" * 60)
        print()
    else:
        print("Starting fresh training (no existing state found)")
        print("=" * 60)
        print()

        start_update = -1

        current_policy = GomokuPolicyNet().to(DEVICE)
        current_policy.train()
        zero_center_taps(current_policy)

        # Apply initial freeze schedule (heads only at start)
        unfrozen_blocks = apply_freeze_schedule(current_policy, update=0)
        print(f"Initial freeze: {unfrozen_blocks} blocks unfrozen (heads only)")

        # Create optimizer for unfrozen parameters only
        optimizer = create_optimizer_for_unfrozen(current_policy, update=0, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        def lr_lambda(epoch):
            # Tanh-based decay from LEARNING_RATE to MIN_LR
            midpoint = TOTAL_UPDATES * LR_DECAY_MIDPOINT_PERCENTAGE
            steepness_k = 3.0 / (TOTAL_UPDATES * LR_DECAY_STEEPNESS)
            decay_factor = 0.5 * (1.0 + torch.tanh(torch.tensor(steepness_k * (midpoint - epoch)))).item()
            decayed_lr = MIN_LR + (LEARNING_RATE - MIN_LR) * decay_factor
            return decayed_lr / LEARNING_RATE

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print("Initializing opponent pool with random policies...")
        opponent_pool = deque()
        opponent_pool_updates = []
        for i in range(OPPONENT_POOL_SIZE):
            opponent_pool.append(create_random_policy(DEVICE))
            opponent_pool[-1].eval()
            opponent_pool_updates.append(-OPPONENT_POOL_SIZE + i)
        print(f"Opponent pool initialized with {len(opponent_pool)} models")
        print()

        next_eval_update = get_eval_interval(0)

        per_opponent_win_rates = {}
        scan_event_counter = 0
        evals_since_last_scan = 0
        win_rate_ema = 0.5

    # Metrics tracking for search-based training
    metric_buffer = {
        'loss': [], 'policy_loss': [], 'ranking_inside_loss': [], 'separation_outside_loss': [],
        'value_loss': [], 'top1_acc': [], 'top3_acc': [], 'value_mse': [],
        'win_rate': [], 'win_rate_as_black': [], 'win_rate_as_white': [],
        'wins': [], 'losses': [], 'draws': [], 'avg_length': [],
        'time': [], 'selfplay_time': [], 'train_time': []
    }

    # Track unfrozen blocks for progressive unfreezing
    if resume_result is None:
        unfrozen_blocks = 0  # Fresh start: heads only
    else:
        # Always apply freeze schedule on resume to ensure correct requires_grad state
        # (load_state_dict doesn't preserve requires_grad, so all params are trainable after load)
        unfrozen_blocks = apply_freeze_schedule(current_policy, start_update + 1)
        if unfrozen_blocks_loaded is not None and unfrozen_blocks_loaded != unfrozen_blocks:
            print(f"Warning: loaded unfrozen_blocks ({unfrozen_blocks_loaded}) differs from calculated ({unfrozen_blocks})")
        print(f"Applied freeze schedule on resume: {unfrozen_blocks} blocks unfrozen")

    training_start_time = time.time()

    # Define lr_lambda once for scheduler creation/recreation
    def lr_lambda(epoch):
        midpoint = TOTAL_UPDATES * LR_DECAY_MIDPOINT_PERCENTAGE
        steepness_k = 3.0 / (TOTAL_UPDATES * LR_DECAY_STEEPNESS)
        decay_factor = 0.5 * (1.0 + torch.tanh(torch.tensor(steepness_k * (midpoint - epoch)))).item()
        decayed_lr = MIN_LR + (LEARNING_RATE - MIN_LR) * decay_factor
        return decayed_lr / LEARNING_RATE

    # Training loop
    for update in range(start_update + 1, TOTAL_UPDATES):
        t_start = time.time()

        # Check if we need to update optimizer due to unfreezing schedule change
        current_lr = optimizer.param_groups[0]['lr']
        prev_optimizer = optimizer
        optimizer, unfrozen_blocks = maybe_update_optimizer(
            current_policy, optimizer, update, unfrozen_blocks,
            lr=current_lr, weight_decay=WEIGHT_DECAY
        )

        # Recreate scheduler if optimizer was replaced (Issue #2 fix)
        if optimizer is not prev_optimizer:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=update - 1)
            print(f"Update {update}: Recreated scheduler for new optimizer (unfrozen_blocks={unfrozen_blocks})")

        # Prepare episode setup
        current_is_black = []
        opening_ids = []
        opponent_indices = []
        opponents_list = list(opponent_pool)
        num_openings = len(RENJU_OPENING_SEQUENCES)

        for _ in range(EPISODES_PER_UPDATE):
            # Sample opponent index
            opponent_idx = random.randint(0, len(opponents_list) - 1)
            opponent_indices.append(opponent_idx)

            if random.random() < 0.5:
                current_is_black.append(True)
            else:
                current_is_black.append(False)

            if random.random() < SEED_PROBABILITY:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        t0 = time.time()
        # Use search-based self-play
        search_samples, outcomes = play_episodes_with_search(
            num_episodes=EPISODES_PER_UPDATE,
            current_policy=current_policy,
            opponents=opponents_list,
            opponent_indices=opponent_indices,
            current_is_black=current_is_black,
            device=DEVICE,
            depth=SEARCH_DEPTH,
            opening_ids=opening_ids,
            tau=SAMPLING_TAU
        )
        t_selfplay = time.time() - t0

        # Compute outcome stats
        wins, losses, draws = 0, 0, 0
        wins_as_black, wins_as_white = 0, 0
        games_as_black, games_as_white = 0, 0
        game_lengths = []

        for outcome, is_black, samples in zip(outcomes, current_is_black, search_samples):
            game_lengths.append(len(samples) * 2)  # Approximate: samples are only for current policy's turns

            if is_black:
                games_as_black += 1
            else:
                games_as_white += 1

            if outcome == GameState.DRAW:
                draws += 1
            elif outcome == GameState.BLACK_WIN:
                if is_black:
                    wins += 1
                    wins_as_black += 1
                else:
                    losses += 1
            elif outcome == GameState.WHITE_WIN:
                if not is_black:
                    wins += 1
                    wins_as_white += 1
                else:
                    losses += 1

        total_games = len(outcomes)
        win_rate = wins / total_games if total_games > 0 else 0
        win_rate_as_black = wins_as_black / games_as_black if games_as_black > 0 else 0
        win_rate_as_white = wins_as_white / games_as_white if games_as_white > 0 else 0
        avg_length = np.mean(game_lengths) if game_lengths else 0

        stats = {
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': win_rate, 'win_rate_as_black': win_rate_as_black,
            'win_rate_as_white': win_rate_as_white, 'avg_length': avg_length
        }

        t0 = time.time()
        # Use search-based training
        train_results = train_on_search_samples(
            current_policy, search_samples, optimizer, DEVICE
        )
        t_train = time.time() - t0

        # Run tactical probe on search samples
        if PROBE_INTERVAL > 0 and (update + 1) % PROBE_INTERVAL == 0:
            probe_stats = probe_tactical_accuracy_search(search_samples)

            # Log to CSV (use update + 1 for consistency with other CSVs)
            csv_logger.log_tactical_probe(update + 1, {
                'win_opportunities': probe_stats.win_opportunities,
                'win_hits': probe_stats.win_hits,
                'win_misses': probe_stats.win_misses,
                'win_accuracy': probe_stats.win_accuracy,
                'block_opportunities': probe_stats.block_opportunities,
                'block_hits': probe_stats.block_hits,
                'block_misses': probe_stats.block_misses,
                'block_accuracy': probe_stats.block_accuracy
            })
            torch.cuda.empty_cache()

        t_total = time.time() - t_start

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        metric_buffer['loss'].append(train_results['loss'])
        metric_buffer['policy_loss'].append(train_results['policy_loss'])
        metric_buffer['ranking_inside_loss'].append(train_results['ranking_inside_loss'])
        metric_buffer['separation_outside_loss'].append(train_results['separation_outside_loss'])
        metric_buffer['value_loss'].append(train_results['value_loss'])
        metric_buffer['top1_acc'].append(train_results['top1_acc'])
        metric_buffer['top3_acc'].append(train_results['top3_acc'])
        metric_buffer['value_mse'].append(train_results['value_mse'])
        metric_buffer['win_rate'].append(stats['win_rate'])
        metric_buffer['win_rate_as_black'].append(stats['win_rate_as_black'])
        metric_buffer['win_rate_as_white'].append(stats['win_rate_as_white'])
        metric_buffer['wins'].append(stats['wins'])
        metric_buffer['losses'].append(stats['losses'])
        metric_buffer['draws'].append(stats['draws'])
        metric_buffer['avg_length'].append(stats['avg_length'])
        metric_buffer['time'].append(t_total)
        metric_buffer['selfplay_time'].append(t_selfplay)
        metric_buffer['train_time'].append(t_train)

        if (update + 1) % PRINT_INTERVAL == 0:
            avg_loss = np.mean(metric_buffer['loss'])
            avg_policy_loss = np.mean(metric_buffer['policy_loss'])
            avg_ranking_inside = np.mean(metric_buffer['ranking_inside_loss'])
            avg_separation_outside = np.mean(metric_buffer['separation_outside_loss'])
            avg_value_loss = np.mean(metric_buffer['value_loss'])
            avg_top1_acc = np.mean(metric_buffer['top1_acc'])
            avg_top3_acc = np.mean(metric_buffer['top3_acc'])
            avg_value_mse = np.mean(metric_buffer['value_mse'])
            avg_win_rate = np.mean(metric_buffer['win_rate'])
            avg_win_rate_black = np.mean(metric_buffer['win_rate_as_black'])
            avg_win_rate_white = np.mean(metric_buffer['win_rate_as_white'])
            avg_length = np.mean(metric_buffer['avg_length'])
            avg_time = np.mean(metric_buffer['time'])
            avg_selfplay_time = np.mean(metric_buffer['selfplay_time'])
            avg_train_time = np.mean(metric_buffer['train_time'])

            elapsed_time = time.time() - training_start_time
            updates_done = update + 1
            updates_remaining = TOTAL_UPDATES - updates_done
            time_per_update = elapsed_time / updates_done
            eta_seconds = updates_remaining * time_per_update

            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}h{minutes:02d}m"
                elif minutes > 0:
                    return f"{minutes}m{secs:02d}s"
                else:
                    return f"{secs}s"

            elapsed_str = format_time(elapsed_time)
            eta_str = format_time(eta_seconds)

            print(f"Update {update + 1:5d}/{TOTAL_UPDATES} | "
                  f"Top1: {avg_top1_acc:.1%} Top3: {avg_top3_acc:.1%} | "
                  f"ValMSE: {avg_value_mse:.4f} | "
                  f"WR: {avg_win_rate:.0%} | "
                  f"Blk: {unfrozen_blocks} | "
                  f"{avg_time:.2f}s | "
                  f"ETA: {eta_str}")

            # Log to search_training.csv
            csv_logger.log_search_training(update + 1, {
                'policy_loss': avg_policy_loss,
                'ranking_inside_loss': avg_ranking_inside,
                'separation_outside_loss': avg_separation_outside,
                'value_loss': avg_value_loss,
                'top1_acc': avg_top1_acc,
                'top3_acc': avg_top3_acc,
                'value_mse': avg_value_mse,
                'unfrozen_blocks': unfrozen_blocks,
                'learning_rate': current_lr,
                'time_total': avg_time,
                'time_selfplay': avg_selfplay_time,
                'time_train': avg_train_time
            })

            # Also log to legacy training_updates.csv for compatibility
            csv_logger.log_training_update(update + 1, {
                'loss': avg_loss,
                'win_rate': avg_win_rate,
                'win_rate_black': avg_win_rate_black,
                'win_rate_white': avg_win_rate_white,
                'avg_game_length': avg_length,
                'entropy': float('nan'),  # Not applicable in search training
                'value_loss': avg_value_loss,
                'raw_value_mse': avg_value_mse,
                'time_total': avg_time,
                'time_selfplay': avg_selfplay_time,
                'time_train': avg_train_time,
                'learning_rate': current_lr
            })

            for key in metric_buffer:
                metric_buffer[key] = []

        if update + 1 >= next_eval_update:
            print(f"\n--- Eval @ {update + 1} ---")

            # Save current checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{update + 1}.pt")
            torch.save({
                'update': update + 1,
                'model_state_dict': current_policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"  Saved: {os.path.basename(checkpoint_path)}")

            # Run evaluation
            eval_start_time = time.time()
            win_rate, per_opp_stats = evaluate_policy(
                current_policy, opponent_pool, DEVICE,
                opponent_pool_updates=opponent_pool_updates
            )
            eval_time = time.time() - eval_start_time

            # Update win rate EMA (based on evaluation, not self-play)
            eval_ema_alpha = 1.0 / EVAL_WIN_RATE_EMA_WINDOW
            win_rate_ema = eval_ema_alpha * win_rate + (1.0 - eval_ema_alpha) * win_rate_ema

            # Update per_opponent_win_rates
            for opp_key, opp_stats in per_opp_stats.items():
                per_opponent_win_rates[opp_key] = opp_stats['win_rate']

            # Find hardest and easiest opponents
            sorted_opps = sorted(per_opp_stats.items(), key=lambda x: x[1]['win_rate'])
            hardest_opp_id = int(sorted_opps[0][0]) if sorted_opps else -1
            hardest_win_rate = sorted_opps[0][1]['win_rate'] if sorted_opps else 0.0
            easiest_opp_id = int(sorted_opps[-1][0]) if sorted_opps else -1
            easiest_win_rate = sorted_opps[-1][1]['win_rate'] if sorted_opps else 0.0

            total_games = EVAL_ROUNDS * 2 * len(opponent_pool)
            print(f"  WinRate: {win_rate:.1%} ({total_games} games, {eval_time:.1f}s) | "
                  f"Hard: {hardest_opp_id}:{hardest_win_rate:.0%} Easy: {easiest_opp_id}:{easiest_win_rate:.0%}")

            # Conditionally add to pool
            checkpoint_added = False
            evicted_opponent_id = -1
            if win_rate >= WIN_RATE_THRESHOLD:
                checkpoint_added = True
                evicted = add_opponent_to_pool(
                    opponent_pool, opponent_pool_updates, current_policy, update + 1,
                    per_opponent_win_rates, DEVICE
                )
                if evicted is not None:
                    evicted_opponent_id = evicted
                    print(f"  Pool: +current -{evicted}")
                    per_opponent_win_rates.pop(str(evicted), None)
                else:
                    print(f"  Pool: +current")
            else:
                print(f"  Pool: no change (WR {win_rate:.1%} < {WIN_RATE_THRESHOLD:.1%})")

            # Log eval summary
            csv_logger.log_eval_summary(update + 1, {
                'overall_win_rate': win_rate,
                'total_games': total_games,
                'eval_time': eval_time,
                'hardest_opponent_id': hardest_opp_id,
                'hardest_win_rate': hardest_win_rate,
                'easiest_opponent_id': easiest_opp_id,
                'easiest_win_rate': easiest_win_rate,
                'pool_size': len(opponent_pool),
                'checkpoint_added': checkpoint_added,
                'evicted_opponent_id': evicted_opponent_id
            })

            # Log per-opponent details
            for opp_key, opp_stats in per_opp_stats.items():
                opp_id = int(opp_key)
                losses = opp_stats['games'] - opp_stats['wins'] - opp_stats['draws']
                csv_logger.log_eval_opponent_details(update + 1, opp_id, {
                    'wins': opp_stats['wins'],
                    'losses': losses,
                    'draws': opp_stats['draws'],
                    'games': opp_stats['games'],
                    'win_rate': opp_stats['win_rate']
                })

            # Check scan trigger
            evals_since_last_scan += 1

            should_scan = (
                (update + 1) >= SCAN_START_UPDATE and
                evals_since_last_scan >= SCAN_PERIOD
            )

            if should_scan:
                print(f"\n--- Mining @ {update + 1} | Bucket {scan_event_counter % NUM_SCAN_BUCKETS} ---")
                scan_start_time = time.time()

                mined_exploiters, total_candidates, candidates_after_filter = scan_historical_exploiters(
                    output_dir, current_policy, opponent_pool_updates, scan_event_counter, DEVICE
                )

                bucket_id = scan_event_counter % NUM_SCAN_BUCKETS
                for rank, (mined_update, mined_win_rate) in enumerate(mined_exploiters, start=1):
                    mined_checkpoint = os.path.join(output_dir, f"checkpoint_update_{mined_update}.pt")
                    mined_model = load_checkpoint_model(mined_checkpoint, DEVICE)
                    if mined_model is not None:
                        evicted = add_opponent_to_pool(
                            opponent_pool, opponent_pool_updates, mined_model, mined_update,
                            per_opponent_win_rates, DEVICE
                        )
                        per_opponent_win_rates[str(mined_update)] = mined_win_rate

                        evicted_id = evicted if evicted is not None else -1
                        print(f"  Found: {mined_update}:{mined_win_rate:.0%} | +{mined_update} -{evicted_id if evicted_id != -1 else 'none'}")

                        csv_logger.log_mining_event(
                            scan_update=update + 1,
                            scan_event_num=scan_event_counter,
                            bucket_id=bucket_id,
                            total_candidates=total_candidates,
                            candidates_after_filter=candidates_after_filter,
                            mined_opponent_id=mined_update,
                            mined_win_rate=mined_win_rate,
                            mined_rank=rank,
                            added_to_pool=True,
                            evicted_opponent_id=evicted_id,
                            scan_time=time.time() - scan_start_time
                        )

                        if evicted is not None:
                            per_opponent_win_rates.pop(str(evicted), None)
                        del mined_model

                scan_time = time.time() - scan_start_time
                if not mined_exploiters:
                    print(f"  No exploiters found ({scan_time:.1f}s)")

                scan_event_counter += 1
                evals_since_last_scan = 0
                torch.cuda.empty_cache()

            # Save training state
            save_training_state(
                output_dir, update + 1, opponent_pool_updates,
                per_opponent_win_rates, scan_event_counter, evals_since_last_scan, win_rate_ema,
                unfrozen_blocks
            )

            eval_interval = get_eval_interval(update + 1)
            next_eval_update = (update + 1) + eval_interval
            print(f"  Next eval: {next_eval_update}")

            torch.cuda.empty_cache()
            print()

    final_path = os.path.join(output_dir, "final_policy.pt")
    torch.save(current_policy.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    save_training_state(
        output_dir, TOTAL_UPDATES, opponent_pool_updates,
        per_opponent_win_rates, scan_event_counter, evals_since_last_scan, win_rate_ema,
        unfrozen_blocks
    )
    print(f"Final training state saved")


if __name__ == "__main__":
    main()
