"""
Evaluation and Opponent Pool Management

Contains:
- Opponent pool management (add, evict, sample)
- Model evaluation against opponent pool
- Historical exploiter scanning/mining
- Checkpoint loading utilities
"""

import copy
import glob
import os
import random
import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gomoku import (
    LOGIT_MASK_VALUE,
    RENJU_OPENING_SEQUENCES,
    GameState,
    play_episodes_batched,
    play_eval_games,
    select_action_batch,
    select_action_batch_eval,
)
from model import GomokuPolicyNet

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
EVAL_INTERVAL_MID = 8         # Evaluation interval for mid training
EVAL_INTERVAL_LATE = 32        # Evaluation interval for late training
WIN_RATE_THRESHOLD = 19.0/32   # Minimum win rate to add to opponent pool

# --- Opponent Sampling ---
UNIFORM_SAMPLING_FRACTION = 0.5  # Fraction of samples that are uniform

# --- Historical Exploiter Scanning ---
SCAN_START_UPDATE = 8192       # Update at which to start scanning
SCAN_PERIOD = 16               # Scan every N evaluations
NUM_SCAN_BUCKETS = 8           # Number of buckets for round-robin scanning
QUICK_SCREEN_ROUNDS = 16       # Rounds for quick screen
TOP_K_QUICK_SCREEN = 16        # Keep top K from quick screen
FINAL_SCREEN_ROUNDS = 64       # Rounds for final screen
MAX_MINED_OPPONENTS_PER_EVENT = 1  # Max opponents to add per scan
MINING_WIN_RATE_THRESHOLD = 27.0/64    # Only mine opponents with win rate below this
MINING_MODEL_BATCH = 16            # Max models to load simultaneously during mining

# --- KL-Aware Mining & Eviction ---
MINING_BASE_THRESHOLD = 0.4
MINING_KL_CAP = 0.4
EVICTION_KL_WEIGHT = 0.5
FINGERPRINT_NUM_TRAJECTORIES = 16


# ============================================================================
# Fingerprint Infrastructure
# ============================================================================

_fp_positions: Optional[torch.Tensor] = None   # [N, 3, 15, 15]
_fp_masks: Optional[torch.Tensor] = None        # [N, 225] bool
_fp_cache: Dict[int, np.ndarray] = {}            # update_num → [N, 225] log-probs


def generate_fingerprint_positions(model: nn.Module, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate fingerprint positions via self-play with Renju openings.

    Returns (positions [N,3,15,15], masks [N,225]) tensors on device.
    """
    opening_ids = random.sample(range(len(RENJU_OPENING_SEQUENCES)), FINGERPRINT_NUM_TRAJECTORIES)
    pairs = [(model, model)] * FINGERPRINT_NUM_TRAJECTORIES
    current_is_black = [True] * FINGERPRINT_NUM_TRAJECTORIES

    trajs = play_episodes_batched(
        pairs, current_is_black, 1.0, device,
        select_action_batch, opening_ids,
    )

    obs_list = []
    mask_list = []
    for traj in trajs:
        for obs, mask in zip(traj.observations, traj.legal_masks):
            obs_list.append(obs)
            mask_list.append(mask.flatten())

    positions = torch.from_numpy(np.stack(obs_list)).float().to(device)
    masks = torch.from_numpy(np.stack(mask_list)).bool().to(device)
    return positions, masks


def fingerprint_model(model: nn.Module, positions: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
    """Run model on fingerprint positions, return masked log-softmax vectors.

    Returns: numpy array [N, 225] of log-probabilities.
    """
    all_log_probs = []
    batch_size = 256
    with torch.inference_mode():
        for start in range(0, positions.shape[0], batch_size):
            end = min(start + batch_size, positions.shape[0])
            obs_batch = positions[start:end]
            mask_batch = masks[start:end]

            logits_grid = model.forward_policy_only(obs_batch)
            logits = logits_grid.squeeze(1).view(-1, 225)
            logits = logits.masked_fill(~mask_batch, LOGIT_MASK_VALUE)
            log_probs = F.log_softmax(logits, dim=1)
            all_log_probs.append(log_probs.cpu().numpy())

    return np.concatenate(all_log_probs, axis=0)


def compute_symmetric_kl(log_probs_a: np.ndarray, log_probs_b: np.ndarray) -> float:
    """Compute mean symmetric KL divergence across positions (in nats)."""
    probs_a = np.exp(log_probs_a)
    probs_b = np.exp(log_probs_b)
    kl_ab = np.sum(probs_a * (log_probs_a - log_probs_b), axis=1)
    kl_ba = np.sum(probs_b * (log_probs_b - log_probs_a), axis=1)
    sym_kl = (kl_ab + kl_ba) / 2.0
    return float(np.mean(sym_kl))


def compute_min_kl(target_fp: np.ndarray, all_fps: Dict[int, np.ndarray],
                   exclude_update: Optional[int] = None) -> float:
    """Min symmetric KL from target to any entry in all_fps (skip exclude_update)."""
    min_kl = float('inf')
    for update_num, fp in all_fps.items():
        if update_num == exclude_update:
            continue
        kl = compute_symmetric_kl(target_fp, fp)
        if kl < min_kl:
            min_kl = kl
    return min_kl


# ============================================================================
# Model Utilities
# ============================================================================

def create_random_policy(device: torch.device) -> nn.Module:
    """Create a policy network with random weights."""
    model = GomokuPolicyNet().to(device)
    return model


def copy_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Create a deep copy of a model."""
    model_copy = GomokuPolicyNet().to(device)
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    model_copy.eval()
    return model_copy


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> Optional[nn.Module]:
    """Load a model from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = GomokuPolicyNet().to(device)
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
                    opponent_pool_updates: List[int]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Evaluate current policy against opponents from the pool.

    Returns:
        Tuple of (overall_win_rate, per_opponent_stats) where per_opponent_stats maps
        opponent update number (as string) to {'wins': int, 'draws': int, 'games': int, 'win_rate': float}
    """
    current_model.eval()
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


def _evict_kl_aware(opponent_pool: deque, opponent_pool_updates: List[int],
                    per_opponent_win_rates: Dict[str, float],
                    new_update: int) -> int:
    """Evict the most redundant+easy opponent using KL-aware scoring.

    The new entrant (new_update) participates in KL distance computation
    but is NOT an eviction candidate.
    """
    # Compute min_kl for each pool member (including new entrant in _fp_cache)
    member_min_kls = []
    for update_num in opponent_pool_updates:
        fp = _fp_cache.get(update_num)
        if fp is None:
            member_min_kls.append(0.0)
            continue
        min_kl = compute_min_kl(fp, _fp_cache, exclude_update=update_num)
        member_min_kls.append(min_kl)

    max_min_kl = max(member_min_kls) if member_min_kls else 0.0

    # Get win rate range
    win_rates = [per_opponent_win_rates.get(str(u), DEFAULT_WIN_RATE) for u in opponent_pool_updates]
    win_rate_range = max(win_rates) - min(win_rates) if win_rates else 0.0

    # Compute eviction scores — higher = easier + more redundant → evict
    best_score = -float('inf')
    best_idx = 0

    for idx, update_num in enumerate(opponent_pool_updates):
        if update_num == new_update:
            continue
        wr = per_opponent_win_rates.get(str(update_num), DEFAULT_WIN_RATE)
        if max_min_kl > 1e-8:
            score = wr - (win_rate_range * EVICTION_KL_WEIGHT) * (member_min_kls[idx] / max_min_kl)
        else:
            score = wr
        if score > best_score:
            best_score = score
            best_idx = idx

    evicted_update = opponent_pool_updates[best_idx]
    pool_list = list(opponent_pool)
    del pool_list[best_idx]
    del opponent_pool_updates[best_idx]
    opponent_pool.clear()
    opponent_pool.extend(pool_list)

    # Clean cache
    _fp_cache.pop(evicted_update, None)

    return evicted_update


def add_opponent_to_pool(opponent_pool: deque, opponent_pool_updates: List[int],
                         new_model: nn.Module, new_update: int,
                         per_opponent_win_rates: Dict[str, float],
                         device: torch.device) -> Optional[int]:
    """Add a new opponent to the pool, evicting if pool is full.

    Uses KL-aware eviction when fingerprint state is available,
    otherwise falls back to pure win-rate eviction.
    """
    snapshot = copy_model(new_model, device)
    evicted_update = None

    if len(opponent_pool) >= OPPONENT_POOL_SIZE:
        if _fp_positions is not None:
            # Fingerprint new model before eviction decision
            fp = fingerprint_model(snapshot, _fp_positions, _fp_masks)
            _fp_cache[new_update] = fp
            evicted_update = _evict_kl_aware(opponent_pool, opponent_pool_updates,
                                             per_opponent_win_rates, new_update)
        else:
            evicted_update = evict_easiest_opponent(opponent_pool, opponent_pool_updates,
                                                     per_opponent_win_rates)
    else:
        # Pool not full — fingerprint and cache if positions available
        if _fp_positions is not None:
            fp = fingerprint_model(snapshot, _fp_positions, _fp_masks)
            _fp_cache[new_update] = fp

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
                                scan_event_num: int, device: torch.device,
                                opponent_pool: Optional[deque] = None,
                                win_rate_ema: Optional[float] = None) -> Tuple[List[Tuple[int, float]], int, int]:
    """
    Scan historical checkpoints to find exploiters (hard opponents for current policy).

    When opponent_pool is provided, refreshes fingerprint positions and caches
    all pool member fingerprints for KL-aware mining and eviction.

    Returns:
        Tuple of (mined_exploiters, total_candidates, candidates_after_filter) where:
        - mined_exploiters: List of (update_number, win_rate) tuples, sorted by difficulty
        - total_candidates: Total number of candidates in this bucket
        - candidates_after_filter: Number of candidates after filtering pool duplicates
    """
    global _fp_positions, _fp_masks

    # Refresh fingerprint state if pool is available
    if opponent_pool is not None:
        _fp_positions, _fp_masks = generate_fingerprint_positions(current_model, device)
        print(f"  Fingerprint positions: {_fp_positions.shape[0]}")

        # Fingerprint all current pool members
        _fp_cache.clear()
        for pool_model, update_num in zip(opponent_pool, opponent_pool_updates):
            pool_model.eval()
            _fp_cache[update_num] = fingerprint_model(pool_model, _fp_positions, _fp_masks)

    print(f"  Scanning historical checkpoints (scan event {scan_event_num}, bucket {scan_event_num % NUM_SCAN_BUCKETS})...")

    all_checkpoints = discover_historical_checkpoints(output_dir, min_update=0)
    if not all_checkpoints:
        print("  No historical checkpoints found")
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

    # Fingerprint candidates while models are still loaded
    candidate_fps: Dict[int, np.ndarray] = {}
    if _fp_positions is not None:
        for opponent, update_num in zip(final_opponents, final_valid_updates):
            candidate_fps[update_num] = fingerprint_model(opponent, _fp_positions, _fp_masks)

    del final_opponents
    torch.cuda.empty_cache()

    final_results.sort(key=lambda x: x[1])

    # Apply KL-aware threshold per candidate when fingerprint state is available
    use_kl_threshold = bool(candidate_fps) and _fp_cache and win_rate_ema is not None
    mined = []
    for u, wr in final_results:
        if len(mined) >= MAX_MINED_OPPONENTS_PER_EVENT:
            break
        if use_kl_threshold and u in candidate_fps:
            assert win_rate_ema is not None
            min_kl = compute_min_kl(candidate_fps[u], _fp_cache)
            threshold = MINING_BASE_THRESHOLD + (win_rate_ema - MINING_BASE_THRESHOLD) * min(MINING_KL_CAP, min_kl) / MINING_KL_CAP
            accepted = wr < threshold
            print(f"    candidate {u}: wr={wr:.2%}, min_KL={min_kl:.4f}, threshold={threshold:.2%} → {'MINE' if accepted else 'skip'}")
            if accepted:
                mined.append((u, wr))
        else:
            if wr < MINING_WIN_RATE_THRESHOLD:
                mined.append((u, wr))

    if mined:
        print(f"  Mined exploiters: {[(u, f'{wr:.2%}') for u, wr in mined]}")
    elif final_results:
        best_wr = final_results[0][1]
        print(f"  No exploiters mined (best win rate {best_wr:.2%})")

    return mined, total_candidates, candidates_after_filter
