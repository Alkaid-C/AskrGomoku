"""
Main Training Script

Entry point for Gomoku self-play training. Contains:
- Training state management (save/load)
- Main training loop
- CLI argument parsing
"""

import os
import sys

# Ensure imports resolve from cwd (needed when this file is symlinked from another directory)
sys.path.insert(0, os.getcwd())

# Enable expandable segments to reduce CUDA memory fragmentation
# This helps avoid OOM errors when there is reserved but unallocated memory
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from csv_logger import CSVLogger
from enhancement import IMITATION_MAX_WEIGHT, IMITATION_MIN_WEIGHT, IMITATION_START_UPDATE, compute_adaptive_boosts, generate_offpolicy_rollout_samples, update_miss_rate_ema
from eval import (
    EVAL_ROUNDS,
    NUM_SCAN_BUCKETS,
    OPPONENT_POOL_SIZE,
    SCAN_PERIOD,
    SCAN_START_UPDATE,
    WIN_RATE_THRESHOLD,
    add_opponent_to_pool,
    create_random_policy,
    evaluate_policy,
    get_eval_interval,
    load_checkpoint_model,
    sample_opponent_weighted,
    scan_historical_exploiters,
)
from gomoku import RENJU_OPENING_SEQUENCES, SEED_PROBABILITY, TEMPERATURE_TRAIN, compute_outcome_stats, play_episodes_batched, select_action_batch
from model import GomokuPolicyNet
from training import (
    BASELINE_RAMP_END,
    EMA_WINDOW,
    ENTROPY_BONUS_COEFF,
    ENTROPY_DECAY_MIDPOINT_PERCENTAGE,
    ENTROPY_DECAY_STEEPNESS,
    ENTROPY_TARGET_END,
    ENTROPY_TARGET_START,
    EPISODES_PER_UPDATE,
    EVAL_WIN_RATE_EMA_WINDOW,
    LEARNING_RATE,
    LR_DECAY_MIDPOINT_PERCENTAGE,
    LR_DECAY_STEEPNESS,
    MIN_LR,
    POLICY_GAE_LAMBDA,
    PRINT_INTERVAL,
    TOTAL_UPDATES,
    VALUE_GAE_LAMBDA,
    VALUE_LOSS_COEFF_END,
    VALUE_LOSS_COEFF_START,
    WEIGHT_DECAY,
    compute_entropy_schedule,
    train_on_batch,
)

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
RNG_STATE_FILE = "rng_state.pt"
SEED = 42


# ============================================================================
# Reproducibility
# ============================================================================

def seed_everything(seed: int) -> None:
    """Seed all RNGs and enable deterministic CUDA operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_rng_state(output_dir: str) -> None:
    """Save all RNG states alongside training state for resume reproducibility."""
    rng_path = os.path.join(output_dir, RNG_STATE_FILE)
    torch.save({
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }, rng_path)


def load_rng_state(output_dir: str) -> bool:
    """Restore all RNG states from saved file. Returns True if successful."""
    rng_path = os.path.join(output_dir, RNG_STATE_FILE)
    if not os.path.exists(rng_path):
        return False
    state = torch.load(rng_path, map_location='cpu', weights_only=False)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    return True


# ============================================================================
# Training State Management
# ============================================================================

def save_training_state(output_dir: str, update: int, opponent_pool_updates: List[int],
                        win_miss_ema: float,
                        block_miss_ema: float,
                        per_opponent_win_rates: Dict[str, float],
                        scan_event_counter: int,
                        evals_since_last_scan: int,
                        win_rate_ema: float,
                        seed: Optional[int] = None) -> None:
    """Save training state to JSON for resume capability."""
    state = {
        'current_update': update,
        'opponent_pool_updates': opponent_pool_updates,
        'total_updates': TOTAL_UPDATES,
        'win_miss_ema': win_miss_ema,
        'block_miss_ema': block_miss_ema,
        'per_opponent_win_rates': per_opponent_win_rates,
        'scan_event_counter': scan_event_counter,
        'evals_since_last_scan': evals_since_last_scan,
        'win_rate_ema': win_rate_ema,
        'seed': seed
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
                  next_eval_update, win_miss_ema, block_miss_ema, per_opponent_win_rates,
                  scan_event_counter, evals_since_last_scan, win_rate_ema) when loading succeeds.
        Returns None only when no training state file exists.
        Note: opponent_pool_updates is filtered to only include successfully loaded opponents.

    Raises:
        RuntimeError: If a training state file exists but is corrupt or references
                      missing/invalid checkpoint data.
    """
    training_state_file = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(training_state_file):
        print(f"No training state file found ({training_state_file})")
        return None

    print(f"Found training state file: {training_state_file}")

    try:
        with open(training_state_file) as f:
            state = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Corrupt training state JSON in {training_state_file}: {e}") from e

    current_update = state['current_update']
    opponent_pool_updates = state['opponent_pool_updates']

    win_miss_ema = state.get('win_miss_ema', 1.0)
    block_miss_ema = state.get('block_miss_ema', 1.0)
    per_opponent_win_rates = state.get('per_opponent_win_rates', {})
    scan_event_counter = state.get('scan_event_counter', 0)
    evals_since_last_scan = state.get('evals_since_last_scan', 0)
    win_rate_ema = state.get('win_rate_ema', 0.5)
    seed = state.get('seed', None)

    print(f"Resuming from update {current_update}")
    print(f"Opponent pool has {len(opponent_pool_updates)} models: {opponent_pool_updates}")
    print(f"EMAs: win_rate={win_rate_ema:.3f}, win_miss={win_miss_ema:.3f}, block_miss={block_miss_ema:.3f}")
    print(f"Scan state: event_counter={scan_event_counter}, evals_since_last_scan={evals_since_last_scan}")
    if per_opponent_win_rates:
        print(f"Per-opponent win rates: {len(per_opponent_win_rates)} entries")

    checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{current_update}.pt")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint referenced by training state not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e

    model = GomokuPolicyNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
    loaded_opponent_updates = []  # Track successfully loaded opponents

    for pool_update in opponent_pool_updates:
        pool_checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{pool_update}.pt")
        if not os.path.exists(pool_checkpoint_path):
            print(f"Warning: Opponent pool checkpoint not found: {pool_checkpoint_path}")
            print("Skipping this opponent")
            continue

        try:
            pool_checkpoint = torch.load(pool_checkpoint_path, map_location=device, weights_only=False)
            pool_model = GomokuPolicyNet().to(device)
            pool_model.load_state_dict(pool_checkpoint['model_state_dict'])
            pool_model.eval()
            opponent_pool.append(pool_model)
            loaded_opponent_updates.append(pool_update)  # Only add if successfully loaded
            print(f"Loaded opponent from update {pool_update}")
        except Exception as e:
            print(f"Error loading opponent pool checkpoint {pool_checkpoint_path}: {e}")
            continue

    if len(opponent_pool) == 0:
        raise RuntimeError("No opponent models could be loaded from pool")

    print(f"Successfully loaded {len(opponent_pool)} opponents")

    next_eval_update = current_update + get_eval_interval(current_update)

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if load_rng_state(output_dir):
            print(f"Restored RNG state (seed={seed})")
        else:
            print(f"Warning: RNG state file not found, re-seeding with seed={seed}+update={current_update}")
            seed_everything(seed + current_update)

    print(f"Next evaluation scheduled at update {next_eval_update}")
    print(f"Resume training starting from update {current_update + 1}")
    print()

    return (model, optimizer, scheduler, opponent_pool, loaded_opponent_updates, current_update - 1, next_eval_update,
            win_miss_ema, block_miss_ema, per_opponent_win_rates, scan_event_counter, evals_since_last_scan, win_rate_ema, seed)


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
    seed = SEED

    # Initialize CSV logger
    csv_logger = CSVLogger(output_dir)
    print(f"Output directory: {output_dir}")
    print()

    print(f"Using device: {DEVICE}")
    print("Model architecture:")
    GomokuPolicyNet.print_topology()
    print("Training configuration:")
    print(f"  Learning rate: {LEARNING_RATE} (tanh decay: mid={LR_DECAY_MIDPOINT_PERCENTAGE:.0%}, steep={LR_DECAY_STEEPNESS:.0%}, min: {MIN_LR})")
    print(f"  Exploration (hybrid): Temperature={TEMPERATURE_TRAIN} (behavior) + Entropy bonus (gradient)")
    print(f"    Target entropy: {ENTROPY_TARGET_START} -> {ENTROPY_TARGET_END} nats (sigmoid: mid={ENTROPY_DECAY_MIDPOINT_PERCENTAGE:.0%}, steep={ENTROPY_DECAY_STEEPNESS:.0%})")
    print(f"    Bonus: coeff={ENTROPY_BONUS_COEFF}")
    print(f"  EMA windows: per-update={EMA_WINDOW}, eval win_rate={EVAL_WIN_RATE_EMA_WINDOW}")
    print(f"  Episodes per update: {EPISODES_PER_UPDATE}")
    print("  Data augmentation: 8-fold symmetry (rot + flip)")
    print(f"  Imitation learning: Dynamic weight (win_rate=0: {IMITATION_MAX_WEIGHT}, win_rate=1: {IMITATION_MIN_WEIGHT}), start: update {IMITATION_START_UPDATE}")
    print(f"  Value head: ENABLED (loss coeff: {VALUE_LOSS_COEFF_START} -> {VALUE_LOSS_COEFF_END} via cosine ramp over [0, {BASELINE_RAMP_END}])")
    print(f"  GAE: policy_lambda={POLICY_GAE_LAMBDA}, value_lambda={VALUE_GAE_LAMBDA}")
    print(f"  Baseline transition: cosine ramp alpha over [0, {BASELINE_RAMP_END}]")
    print("    - Advantages: (1-alpha)*max(0,R) + alpha*max(0,GAE) + tactical boost")
    print(f"    - Value loss coeff: {VALUE_LOSS_COEFF_START} -> {VALUE_LOSS_COEFF_END}")
    print("    - Tactical bonuses prevent terminal state learning collapse")
    print(f"  Opening seeding: {SEED_PROBABILITY:.0%} of games start from Renju opening ({len(RENJU_OPENING_SEQUENCES)} patterns)")
    print(f"  Opponent pool size: {OPPONENT_POOL_SIZE}")
    print(f"  Eval interval: {get_eval_interval(0)} (early) -> {get_eval_interval(512)} (mid) -> {get_eval_interval(8192)} (late)")
    print("  Pool eviction: evict-easiest (by current win rate)")
    print("  Historical exploiter scanning:")
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
         next_eval_update, win_miss_ema, block_miss_ema, per_opponent_win_rates, scan_event_counter,
         evals_since_last_scan, win_rate_ema, saved_seed) = resume_result
        if saved_seed is not None:
            if seed is not None and seed != saved_seed:
                print(f"Warning: SEED={seed} differs from saved seed {saved_seed}, using saved seed")
            seed = saved_seed
        print("=" * 60)
        print("Successfully resumed training!")
        print("=" * 60)
        print()
    else:
        print("Starting fresh training (no existing state found)")
        print("=" * 60)
        print()

        if seed is not None:
            seed_everything(seed)
            print(f"Random seed: {seed} (deterministic mode)")
            print()

        start_update = -1

        current_policy = GomokuPolicyNet().to(DEVICE)
        current_policy.train()

        optimizer = torch.optim.AdamW(
            current_policy.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            fused=True
        )

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

        win_miss_ema = 1.0
        block_miss_ema = 1.0
        next_eval_update = get_eval_interval(0)

        per_opponent_win_rates = {}
        scan_event_counter = 0
        evals_since_last_scan = 0
        win_rate_ema = 0.5

    # Metrics tracking
    metric_buffer = {
        'loss': [], 'win_rate': [], 'win_rate_as_black': [], 'win_rate_as_white': [],
        'wins': [], 'losses': [], 'draws': [], 'entropy': [], 'value_loss': [],
        'raw_value_mse': [], 'avg_length': [], 'time': [], 'selfplay_time': [],
        'train_time': [], 'tactics_wins': [], 'tactics_blocks': [],
        'tactics_synthetic_wins_eq': [], 'tactics_synthetic_wins_missed': [],
        'tactics_synthetic_blocks': [], 'imitation_black': [], 'imitation_white': [],
        'win_miss_ema': [], 'block_miss_ema': [], 'win_boost': [], 'block_boost': [],
        'opr_attempted': [], 'opr_candidates': [], 'opr_added': [],
        'opr_winrate_sum': [], 'opr_orig_winrate_sum': [], 'opr_entropy_sum': []
    }

    # Entropy EMA for adaptive entropy bonus
    ema_entropy = np.log(225)

    training_start_time = time.time()

    # Training loop
    for update in range(start_update + 1, TOTAL_UPDATES):
        t_start = time.time()

        pairs = []
        current_is_black = []
        opening_ids = []
        episode_opponents = []
        num_openings = len(RENJU_OPENING_SEQUENCES)
        for _ in range(EPISODES_PER_UPDATE):
            opponent = sample_opponent_weighted(opponent_pool, opponent_pool_updates, per_opponent_win_rates)
            episode_opponents.append(opponent)
            if random.random() < 0.5:
                pairs.append((current_policy, opponent))
                current_is_black.append(True)
            else:
                pairs.append((opponent, current_policy))
                current_is_black.append(False)

            if random.random() < SEED_PROBABILITY:
                opening_ids.append(random.randint(0, num_openings - 1))
            else:
                opening_ids.append(-1)

        t0 = time.time()
        current_policy.eval()
        trajectories = play_episodes_batched(
            pairs, current_is_black, TEMPERATURE_TRAIN, DEVICE,
            select_action_batch_fn=select_action_batch,
            opening_ids=opening_ids
        )
        current_policy.train()
        t_selfplay = time.time() - t0

        stats = compute_outcome_stats(trajectories, current_is_black)

        # Compute adaptive boosts
        win_boost, block_boost = compute_adaptive_boosts(win_miss_ema, block_miss_ema)

        # Compute scheduled entropy value for OPR threshold
        entropy_schedule = compute_entropy_schedule(update)

        # Generate off-policy rollout samples from lost games
        opr_samples, opr_stats = generate_offpolicy_rollout_samples(
            trajectories, current_is_black, episode_opponents, current_policy, DEVICE, update,
            entropy_schedule=entropy_schedule
        )

        t0 = time.time()
        train_results = train_on_batch(
            current_policy, trajectories, optimizer, DEVICE, update=update,
            win_boost=win_boost, block_boost=block_boost, opr_samples=opr_samples, ema_entropy=ema_entropy,
            win_rate=win_rate_ema, output_dir=output_dir
        )
        t_train = time.time() - t0

        # Update entropy EMA
        ema_alpha = 1.0 / EMA_WINDOW
        ema_entropy = ema_alpha * train_results['entropy'] + (1.0 - ema_alpha) * ema_entropy

        # Free probe tensors after gradient probe ran
        if train_results['probe_ran']:
            print("  [Probe] Gradient vectors saved to .npz")
            torch.cuda.empty_cache()

        # Update miss rate EMAs
        tactical_stats = train_results['tactical_stats']
        this_win_miss_rate = tactical_stats.win_misses / tactical_stats.win_opportunities if tactical_stats.win_opportunities > 0 else 0.0
        this_block_miss_rate = (
            tactical_stats.block_misses / tactical_stats.block_opportunities
            if tactical_stats.block_opportunities > 0
            else None
        )
        win_miss_ema = update_miss_rate_ema(this_win_miss_rate, win_miss_ema, EMA_WINDOW)
        if this_block_miss_rate is not None:
            block_miss_ema = update_miss_rate_ema(this_block_miss_rate, block_miss_ema, EMA_WINDOW)

        t_total = time.time() - t_start

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        metric_buffer['loss'].append(train_results['loss'])
        metric_buffer['win_rate'].append(stats['win_rate'])
        metric_buffer['win_rate_as_black'].append(stats['win_rate_as_black'])
        metric_buffer['win_rate_as_white'].append(stats['win_rate_as_white'])
        metric_buffer['wins'].append(stats['wins'])
        metric_buffer['losses'].append(stats['losses'])
        metric_buffer['draws'].append(stats['draws'])
        metric_buffer['entropy'].append(train_results['entropy'])
        metric_buffer['value_loss'].append(train_results['value_loss'])
        metric_buffer['raw_value_mse'].append(train_results['raw_value_mse'])
        metric_buffer['avg_length'].append(stats['avg_length'])
        metric_buffer['time'].append(t_total)
        metric_buffer['selfplay_time'].append(t_selfplay)
        metric_buffer['train_time'].append(t_train)
        metric_buffer['tactics_wins'].append(tactical_stats.wins_found)
        metric_buffer['tactics_blocks'].append(tactical_stats.blocks_found)
        metric_buffer['tactics_synthetic_wins_eq'].append(tactical_stats.synthetic_wins_eq)
        metric_buffer['tactics_synthetic_wins_missed'].append(tactical_stats.synthetic_wins_missed)
        metric_buffer['tactics_synthetic_blocks'].append(tactical_stats.synthetic_blocks)
        metric_buffer['imitation_black'].append(train_results['imitation_black'])
        metric_buffer['imitation_white'].append(train_results['imitation_white'])
        metric_buffer['win_miss_ema'].append(win_miss_ema)
        metric_buffer['block_miss_ema'].append(block_miss_ema)
        metric_buffer['win_boost'].append(win_boost)
        metric_buffer['block_boost'].append(block_boost)
        metric_buffer['opr_attempted'].append(opr_stats.attempted_episodes)
        metric_buffer['opr_candidates'].append(opr_stats.candidates_total)
        metric_buffer['opr_added'].append(opr_stats.samples_added)
        metric_buffer['opr_winrate_sum'].append(opr_stats.best_winrate_sum)
        metric_buffer['opr_orig_winrate_sum'].append(opr_stats.orig_winrate_sum)
        metric_buffer['opr_entropy_sum'].append(opr_stats.entropy_selected_sum)

        if (update + 1) % PRINT_INTERVAL == 0:
            avg_loss = np.mean(metric_buffer['loss'])
            avg_win_rate = np.mean(metric_buffer['win_rate'])
            avg_win_rate_black = np.mean(metric_buffer['win_rate_as_black'])
            avg_win_rate_white = np.mean(metric_buffer['win_rate_as_white'])
            avg_entropy = np.mean(metric_buffer['entropy'])
            avg_value_loss = np.mean(metric_buffer['value_loss'])
            avg_raw_value_mse = np.mean(metric_buffer['raw_value_mse'])
            avg_length = np.mean(metric_buffer['avg_length'])
            avg_time = np.mean(metric_buffer['time'])
            avg_selfplay_time = np.mean(metric_buffer['selfplay_time'])
            avg_train_time = np.mean(metric_buffer['train_time'])
            total_wins_found = sum(metric_buffer['tactics_wins'])
            total_blocks_found = sum(metric_buffer['tactics_blocks'])
            total_synthetic_wins_eq = sum(metric_buffer['tactics_synthetic_wins_eq'])
            total_synthetic_wins_missed = sum(metric_buffer['tactics_synthetic_wins_missed'])
            total_synthetic_blocks = sum(metric_buffer['tactics_synthetic_blocks'])
            total_imitation_black = sum(metric_buffer['imitation_black'])
            total_imitation_white = sum(metric_buffer['imitation_white'])

            latest_win_miss_ema = metric_buffer['win_miss_ema'][-1] if metric_buffer['win_miss_ema'] else 0.0
            latest_block_miss_ema = metric_buffer['block_miss_ema'][-1] if metric_buffer['block_miss_ema'] else 0.0
            latest_win_boost = metric_buffer['win_boost'][-1] if metric_buffer['win_boost'] else 0.0
            latest_block_boost = metric_buffer['block_boost'][-1] if metric_buffer['block_boost'] else 0.0

            # Off-policy rollout metrics aggregation
            total_opr_attempted = sum(metric_buffer['opr_attempted'])
            total_opr_candidates = sum(metric_buffer['opr_candidates'])
            total_opr_added = sum(metric_buffer['opr_added'])
            total_opr_winrate_sum = sum(metric_buffer['opr_winrate_sum'])
            total_opr_orig_winrate_sum = sum(metric_buffer['opr_orig_winrate_sum'])
            total_opr_entropy_sum = sum(metric_buffer['opr_entropy_sum'])
            avg_opr_candidates = total_opr_candidates / max(total_opr_attempted, 1)
            avg_opr_winrate = total_opr_winrate_sum / max(total_opr_added, 1)
            avg_opr_orig_winrate = total_opr_orig_winrate_sum / max(total_opr_added, 1)
            avg_opr_entropy = total_opr_entropy_sum / max(total_opr_added, 1)

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
                  f"WinRate: {avg_win_rate:.0%}(W{avg_win_rate_white:.0%}-B{avg_win_rate_black:.0%}) | "
                  f"AvgLen: {avg_length:.1f} | "
                  f"Entropy: {avg_entropy:.3f} | "
                  f"MissEMA: W={latest_win_miss_ema:.0%} B={latest_block_miss_ema:.0%} | "
                  f"{avg_time:.2f}s/upd | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")

            csv_logger.log_training_update(update + 1, {
                'loss': avg_loss,
                'win_rate': avg_win_rate,
                'win_rate_black': avg_win_rate_black,
                'win_rate_white': avg_win_rate_white,
                'avg_game_length': avg_length,
                'entropy': avg_entropy,
                'value_loss': avg_value_loss,
                'raw_value_mse': avg_raw_value_mse,
                'tactics_wins': total_wins_found,
                'tactics_blocks': total_blocks_found,
                'tactics_synth_wins_eq': total_synthetic_wins_eq,
                'tactics_synth_wins_missed': total_synthetic_wins_missed,
                'tactics_synth_blocks': total_synthetic_blocks,
                'imitation_black': total_imitation_black,
                'imitation_white': total_imitation_white,
                'win_miss_ema': latest_win_miss_ema,
                'block_miss_ema': latest_block_miss_ema,
                'win_boost': latest_win_boost,
                'block_boost': latest_block_boost,
                'opr_attempted': total_opr_attempted,
                'opr_candidates_avg': avg_opr_candidates,
                'opr_added': total_opr_added,
                'opr_winrate_avg': avg_opr_winrate,
                'opr_orig_winrate_avg': avg_opr_orig_winrate,
                'opr_entropy_avg': avg_opr_entropy,
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
                    print("  Pool: +current")
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
                    output_dir, current_policy, opponent_pool_updates, scan_event_counter, DEVICE,
                    opponent_pool=opponent_pool, win_rate_ema=win_rate_ema
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
                output_dir, update + 1, opponent_pool_updates, win_miss_ema, block_miss_ema,
                per_opponent_win_rates, scan_event_counter, evals_since_last_scan, win_rate_ema,
                seed=seed
            )
            if seed is not None:
                save_rng_state(output_dir)

            eval_interval = get_eval_interval(update + 1)
            next_eval_update = (update + 1) + eval_interval
            print(f"  Next eval: {next_eval_update}")

            torch.cuda.empty_cache()
            print()

    final_path = os.path.join(output_dir, "final_policy.pt")
    torch.save(current_policy.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    save_training_state(
        output_dir, TOTAL_UPDATES, opponent_pool_updates, win_miss_ema, block_miss_ema,
        per_opponent_win_rates, scan_event_counter, evals_since_last_scan, win_rate_ema,
        seed=seed
    )
    if seed is not None:
        save_rng_state(output_dir)
    print("Final training state saved")


if __name__ == "__main__":
    main()
