"""
Evaluation and Opponent Pool Management

Contains:
- Opponent pool management (add, evict, sample)
- Model evaluation against opponent pool
- Historical exploiter scanning/mining
- Checkpoint loading utilities
"""

import torch
import torch.nn as nn
from collections import deque
import copy
import os
import glob
import re
import random
from typing import List, Tuple, Optional, Dict

from model import GomokuPolicyNet, N_BLOCKS
from gomoku import GameState, play_eval_games, select_action_batch_eval


# ============================================================================
# Evaluation Constants
# ============================================================================

# --- Opponent Pool ---
OPPONENT_POOL_SIZE = 16        # Number of opponents to maintain in pool
DEFAULT_WIN_RATE = 0.5         # Default win rate for new opponents

# --- Evaluation ---
EVAL_ROUNDS = 32               # Number of evaluation rounds per opponent
EVAL_TEMP = 1.0                # Temperature for evaluation
EVAL_INTERVAL_EARLY = 4        # Evaluation interval for early training
EVAL_INTERVAL_MID = 32         # Evaluation interval for mid training
EVAL_INTERVAL_LATE = 128       # Evaluation interval for late training
WIN_RATE_THRESHOLD = 19.0/32   # Minimum win rate to add to opponent pool

# --- Opponent Sampling ---
UNIFORM_SAMPLING_FRACTION = 0.5  # Fraction of samples that are uniform

# --- Historical Exploiter Scanning ---
SCAN_START_UPDATE = 8192       # Update at which to start scanning
SCAN_PERIOD = 16               # Scan every N evaluations
NUM_SCAN_BUCKETS = 4           # Number of buckets for round-robin scanning
QUICK_SCREEN_ROUNDS = 16       # Rounds for quick screen
TOP_K_QUICK_SCREEN = 16        # Keep top K from quick screen
FINAL_SCREEN_ROUNDS = 64       # Rounds for final screen
MAX_MINED_OPPONENTS_PER_EVENT = 1  # Max opponents to add per scan
MINING_WIN_RATE_THRESHOLD = 27.0/64    # Only mine opponents with win rate below this
MINING_MODEL_BATCH = 16            # Max models to load simultaneously during mining


# ============================================================================
# Model Utilities
# ============================================================================

def create_random_policy(device: torch.device) -> nn.Module:
    """Create a policy network with random weights."""
    model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    return model


def copy_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Create a deep copy of a model."""
    model_copy = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    model_copy.eval()
    return model_copy


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> Optional[nn.Module]:
    """Load a model from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
        return None


# ============================================================================
# Evaluation Helpers
# ============================================================================

def get_eval_interval(update: int) -> int:
    """Get adaptive evaluation interval based on training progress."""
    if update < 128:
        return EVAL_INTERVAL_EARLY
    elif update < 2048:
        return EVAL_INTERVAL_MID
    else:
        return EVAL_INTERVAL_LATE


def evaluate_policy(current_model: nn.Module, opponent_pool: deque,
                    device: torch.device,
                    opponent_pool_updates: List[int],
                    num_rounds: int = None) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Evaluate current policy against opponents from the pool.

    Returns:
        Tuple of (overall_win_rate, per_opponent_stats) where per_opponent_stats maps
        opponent update number (as string) to {'wins': int, 'draws': int, 'games': int, 'win_rate': float}
    """
    current_model.eval()

    if num_rounds is None:
        num_rounds = EVAL_ROUNDS

    total_wins = 0
    total_draws = 0
    total_games = 0

    num_opponents = len(opponent_pool)
    per_opponent_wins = [0] * num_opponents
    per_opponent_draws = [0] * num_opponents
    per_opponent_games = [0] * num_opponents

    # Build all game pairs upfront for a single batched play_eval_games call
    pairs = []
    current_is_black_list = []
    opponent_indices = []

    for _ in range(num_rounds):
        for opp_idx, opponent in enumerate(opponent_pool):
            # Current plays as black
            pairs.append((current_model, opponent))
            current_is_black_list.append(True)
            opponent_indices.append(opp_idx)
            # Current plays as white
            pairs.append((opponent, current_model))
            current_is_black_list.append(False)
            opponent_indices.append(opp_idx)

    results = play_eval_games(
        pairs, current_is_black_list, EVAL_TEMP, device,
        select_action_fn=select_action_batch_eval
    )

    for (outcome, current_is_black), opp_idx in zip(results, opponent_indices):
        total_games += 1
        per_opponent_games[opp_idx] += 1

        if outcome == GameState.DRAW:
            total_draws += 1
            per_opponent_draws[opp_idx] += 1
        elif (outcome == GameState.BLACK_WIN and current_is_black) or \
             (outcome == GameState.WHITE_WIN and not current_is_black):
            total_wins += 1
            per_opponent_wins[opp_idx] += 1

    current_model.train()

    overall_win_rate = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0

    per_opponent_stats = {}
    for opp_idx in range(num_opponents):
        if opp_idx < len(opponent_pool_updates):
            key = str(opponent_pool_updates[opp_idx])
        else:
            key = str(opp_idx)

        games = per_opponent_games[opp_idx]
        wins = per_opponent_wins[opp_idx]
        draws = per_opponent_draws[opp_idx]
        win_rate = (wins + 0.5 * draws) / games if games > 0 else DEFAULT_WIN_RATE

        per_opponent_stats[key] = {
            'wins': wins,
            'draws': draws,
            'games': games,
            'win_rate': win_rate
        }

    return overall_win_rate, per_opponent_stats


def evaluate_against_opponents(current_model: nn.Module, opponents: List[nn.Module],
                                device: torch.device, num_rounds: int) -> List[float]:
    """Evaluate current model against multiple opponents in one batched run.

    Returns list of win rates, one per opponent.
    """
    current_model.eval()

    num_opponents = len(opponents)
    per_opp_wins = [0] * num_opponents
    per_opp_draws = [0] * num_opponents
    per_opp_games = [0] * num_opponents

    # Build all pairs upfront
    pairs = []
    current_is_black_list = []
    opponent_indices = []

    for _ in range(num_rounds):
        for opp_idx, opponent in enumerate(opponents):
            pairs.append((current_model, opponent))
            current_is_black_list.append(True)
            opponent_indices.append(opp_idx)
            pairs.append((opponent, current_model))
            current_is_black_list.append(False)
            opponent_indices.append(opp_idx)

    results = play_eval_games(
        pairs, current_is_black_list, EVAL_TEMP, device,
        select_action_fn=select_action_batch_eval
    )

    for (outcome, current_is_black), opp_idx in zip(results, opponent_indices):
        per_opp_games[opp_idx] += 1
        if outcome == GameState.DRAW:
            per_opp_draws[opp_idx] += 1
        elif (outcome == GameState.BLACK_WIN and current_is_black) or \
             (outcome == GameState.WHITE_WIN and not current_is_black):
            per_opp_wins[opp_idx] += 1

    current_model.train()

    win_rates = []
    for i in range(num_opponents):
        if per_opp_games[i] > 0:
            win_rates.append((per_opp_wins[i] + 0.5 * per_opp_draws[i]) / per_opp_games[i])
        else:
            win_rates.append(DEFAULT_WIN_RATE)

    return win_rates


# ============================================================================
# Opponent Pool Management
# ============================================================================

def find_easiest_opponent_index(opponent_pool_updates: List[int],
                                 per_opponent_win_rates: Dict[str, float]) -> int:
    """Find the index of the easiest opponent in the pool (highest win rate for current)."""
    easiest_idx = 0
    easiest_win_rate = -1.0

    for idx, update_num in enumerate(opponent_pool_updates):
        key = str(update_num)
        win_rate = per_opponent_win_rates.get(key, DEFAULT_WIN_RATE)
        if win_rate > easiest_win_rate:
            easiest_win_rate = win_rate
            easiest_idx = idx

    return easiest_idx


def evict_easiest_opponent(opponent_pool: deque, opponent_pool_updates: List[int],
                           per_opponent_win_rates: Dict[str, float]) -> int:
    """Remove the easiest opponent from the pool."""
    evict_idx = find_easiest_opponent_index(opponent_pool_updates, per_opponent_win_rates)

    pool_list = list(opponent_pool)
    evicted_update = opponent_pool_updates[evict_idx]

    del pool_list[evict_idx]
    del opponent_pool_updates[evict_idx]

    opponent_pool.clear()
    opponent_pool.extend(pool_list)

    return evicted_update


def add_opponent_to_pool(opponent_pool: deque, opponent_pool_updates: List[int],
                         new_model: nn.Module, new_update: int,
                         per_opponent_win_rates: Dict[str, float],
                         device: torch.device) -> Optional[int]:
    """Add a new opponent to the pool, evicting the easiest if pool is full."""
    snapshot = copy_model(new_model, device)
    evicted_update = None

    if len(opponent_pool) >= OPPONENT_POOL_SIZE:
        evicted_update = evict_easiest_opponent(opponent_pool, opponent_pool_updates,
                                                 per_opponent_win_rates)

    opponent_pool.append(snapshot)
    opponent_pool_updates.append(new_update)

    return evicted_update


def sample_opponent_weighted(opponent_pool: deque, opponent_pool_updates: List[int],
                              per_opponent_win_rates: Dict[str, float]) -> nn.Module:
    """Sample an opponent using difficulty-weighted distribution."""
    pool_list = list(opponent_pool)

    if random.random() < UNIFORM_SAMPLING_FRACTION:
        return random.choice(pool_list)

    # Difficulty-weighted: weight = 1 - win_rate (clamped to >= 0.01)
    weights = []
    for update_num in opponent_pool_updates:
        key = str(update_num)
        win_rate = per_opponent_win_rates.get(key, DEFAULT_WIN_RATE)
        weight = max(1.0 - win_rate, 0.01)
        weights.append(weight)

    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    idx = random.choices(range(len(pool_list)), weights=probs, k=1)[0]
    return pool_list[idx]


# ============================================================================
# Historical Exploiter Scanning
# ============================================================================

def discover_historical_checkpoints(output_dir: str, min_update: Optional[int]) -> List[int]:
    """Discover all checkpoint files and return their update numbers."""
    if min_update is None:
        min_update = SCAN_START_UPDATE

    checkpoint_files = glob.glob(os.path.join(output_dir, "checkpoint_update_*.pt"))
    update_numbers = []

    pattern = re.compile(r'checkpoint_update_(\d+)\.pt')
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            update_num = int(match.group(1))
            if update_num >= min_update:
                update_numbers.append(update_num)

    return sorted(update_numbers)


def get_bucket_candidates(scan_event_num: int, all_checkpoints: List[int]) -> List[int]:
    """Get checkpoint candidates for a given scan event using round-robin bucketing."""
    target_bucket = scan_event_num % NUM_SCAN_BUCKETS
    candidates = []

    for update_num in all_checkpoints:
        checkpoint_bucket = (update_num // EVAL_INTERVAL_LATE) % NUM_SCAN_BUCKETS
        if checkpoint_bucket == target_bucket:
            candidates.append(update_num)

    return candidates


def scan_historical_exploiters(output_dir: str, current_model: nn.Module, opponent_pool_updates: List[int],
                                scan_event_num: int, device: torch.device) -> Tuple[List[Tuple[int, float]], int, int]:
    """
    Scan historical checkpoints to find exploiters (hard opponents for current policy).

    Returns:
        Tuple of (mined_exploiters, total_candidates, candidates_after_filter) where:
        - mined_exploiters: List of (update_number, win_rate) tuples, sorted by difficulty
        - total_candidates: Total number of candidates in this bucket
        - candidates_after_filter: Number of candidates after filtering pool duplicates
    """
    print(f"  Scanning historical checkpoints (scan event {scan_event_num}, bucket {scan_event_num % NUM_SCAN_BUCKETS})...")

    all_checkpoints = discover_historical_checkpoints(output_dir, min_update=0)
    if not all_checkpoints:
        print(f"  No historical checkpoints found")
        return [], 0, 0

    candidates = get_bucket_candidates(scan_event_num, all_checkpoints)
    total_candidates = len(candidates)
    print(f"  Found {total_candidates} checkpoints in bucket {scan_event_num % NUM_SCAN_BUCKETS}")

    pool_set = set(opponent_pool_updates)
    candidates = [c for c in candidates if c not in pool_set]
    candidates_after_filter = len(candidates)
    print(f"  {candidates_after_filter} candidates after filtering pool duplicates")

    if not candidates:
        return [], total_candidates, candidates_after_filter

    # Quick screen - load candidates in chunks for batched evaluation
    print(f"  Quick screen ({QUICK_SCREEN_ROUNDS} rounds per candidate)...")
    quick_results = []

    for chunk_start in range(0, len(candidates), MINING_MODEL_BATCH):
        chunk_updates = candidates[chunk_start:chunk_start + MINING_MODEL_BATCH]
        chunk_opponents = []
        chunk_valid_updates = []

        for update_num in chunk_updates:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{update_num}.pt")
            opponent = load_checkpoint_model(checkpoint_path, device)
            if opponent is not None:
                chunk_opponents.append(opponent)
                chunk_valid_updates.append(update_num)

        if chunk_opponents:
            win_rates = evaluate_against_opponents(current_model, chunk_opponents, device, QUICK_SCREEN_ROUNDS)
            for update_num, win_rate in zip(chunk_valid_updates, win_rates):
                quick_results.append((update_num, win_rate))

        del chunk_opponents
        torch.cuda.empty_cache()

    if not quick_results:
        return [], total_candidates, candidates_after_filter

    quick_results.sort(key=lambda x: x[1])

    hardest_candidates = quick_results[:TOP_K_QUICK_SCREEN]
    print(f"  Top {len(hardest_candidates)} hardest: {[(u, f'{wr:.2%}') for u, wr in hardest_candidates]}")

    # Final screen - load all top-K at once for single batched evaluation
    print(f"  Final screen ({FINAL_SCREEN_ROUNDS} rounds per candidate)...")
    final_opponents = []
    final_valid_updates = []

    for update_num, _ in hardest_candidates:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_update_{update_num}.pt")
        opponent = load_checkpoint_model(checkpoint_path, device)
        if opponent is not None:
            final_opponents.append(opponent)
            final_valid_updates.append(update_num)

    final_results = []
    if final_opponents:
        win_rates = evaluate_against_opponents(current_model, final_opponents, device, FINAL_SCREEN_ROUNDS)
        for update_num, win_rate in zip(final_valid_updates, win_rates):
            final_results.append((update_num, win_rate))

    del final_opponents
    torch.cuda.empty_cache()

    final_results.sort(key=lambda x: x[1])
    # Only mine opponents that are actually hard (win rate < threshold)
    hard_opponents = [(u, wr) for u, wr in final_results if wr < MINING_WIN_RATE_THRESHOLD]
    mined = hard_opponents[:MAX_MINED_OPPONENTS_PER_EVENT]

    if mined:
        print(f"  Mined exploiters: {[(u, f'{wr:.2%}') for u, wr in mined]}")
    elif final_results:
        # Log the best (lowest) win rate for context
        best_wr = final_results[0][1]
        print(f"  No exploiters mined (best win rate {best_wr:.2%} >= threshold {MINING_WIN_RATE_THRESHOLD:.0%})")

    return mined, total_candidates, candidates_after_filter
