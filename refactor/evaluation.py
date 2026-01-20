"""
Evaluation and Opponent Pool Management.

Handles:
- Opponent Pool (deque of models)
- Evaluation vs Pool
- Historical Exploiter Mining
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import os
import glob
import re
import json
import copy
from typing import List, Tuple, Dict, Optional

from model import GomokuPolicyNet, N_BLOCKS
from gameplay_loop import play_eval_games, select_action_batch_eval, GameState, Player, BATCH_INFERENCE_SIZE

# ============================================================================
# Configuration
# ============================================================================

OPPONENT_POOL_SIZE = 16
EVAL_ROUNDS = 32
EVAL_TEMP = 1.0
EVAL_INTERVAL_EARLY = 8
EVAL_INTERVAL_MID = 32
EVAL_INTERVAL_LATE = 128
WIN_RATE_THRESHOLD = 0.625

SCAN_START_UPDATE = 8192
SCAN_PERIOD = 16
NUM_SCAN_BUCKETS = 4
QUICK_SCREEN_ROUNDS = 16
TOP_K_QUICK_SCREEN = 16
FINAL_SCREEN_ROUNDS = 64
MAX_MINED_OPPONENTS_PER_EVENT = 2

UNIFORM_SAMPLING_FRACTION = 0.5
DEFAULT_WIN_RATE = 0.5

TRAINING_STATE_FILE = "training_state.json"


# ============================================================================
# Helper Functions
# ============================================================================

def create_random_policy(device: torch.device) -> nn.Module:
    model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    return model

def copy_model(model: nn.Module, device: torch.device) -> nn.Module:
    model_copy = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    model_copy.eval()
    return model_copy

def get_eval_interval(update: int) -> int:
    if update < 512: return EVAL_INTERVAL_EARLY
    elif update < 8192: return EVAL_INTERVAL_MID
    else: return EVAL_INTERVAL_LATE


# ============================================================================
# Opponent Pool
# ============================================================================

class OpponentPool:
    def __init__(self, device: torch.device, output_dir: str):
        self.device = device
        self.output_dir = output_dir
        self.models = deque()
        self.updates = []
        self.win_rates = {} # update_str -> win_rate (of current policy VS this opponent)

    def initialize_random(self):
        for _ in range(OPPONENT_POOL_SIZE):
            model = create_random_policy(self.device)
            model.eval()
            self.models.append(model)
            self.updates.append(0)

    def sample(self) -> nn.Module:
        """Sample opponent using difficulty-weighted distribution."""
        if random.random() < UNIFORM_SAMPLING_FRACTION:
            return random.choice(self.models)
        
        # Difficulty-weighted: weight = 1 - win_rate (Harder = Higher weight)
        weights = []
        for update_num in self.updates:
            wr = self.win_rates.get(str(update_num), DEFAULT_WIN_RATE)
            weights.append(max(1.0 - wr, 0.01))
            
        total = sum(weights)
        if total <= 0: return random.choice(self.models)
        probs = [w/total for w in weights]
        idx = random.choices(range(len(self.models)), weights=probs, k=1)[0]
        return self.models[idx]

    def add(self, model: nn.Module, update: int, win_rate_vs_current: float = None) -> Optional[int]:
        """Add model to pool, evicting easiest. Returns evicted update num."""
        evicted = None
        if len(self.models) >= OPPONENT_POOL_SIZE:
            evicted = self._evict_easiest()
            
        snapshot = copy_model(model, self.device)
        self.models.append(snapshot)
        self.updates.append(update)
        return evicted

    def _evict_easiest(self) -> int:
        best_wr = -1.0
        best_idx = 0
        for i, update_num in enumerate(self.updates):
            wr = self.win_rates.get(str(update_num), DEFAULT_WIN_RATE)
            if wr > best_wr:
                best_wr = wr
                best_idx = i
        
        # Remove
        evicted_update = self.updates[best_idx]
        del self.models[best_idx]
        del self.updates[best_idx]
        # Cleanup stats
        self.win_rates.pop(str(evicted_update), None)
        return evicted_update


# ============================================================================
# Evaluation Logic
# ============================================================================

def evaluate_policy(current_model: nn.Module, pool: OpponentPool) -> Tuple[float, Dict[str, dict]]:
    """Evaluate current policy against the pool."""
    current_model.eval()
    
    total_wins = 0
    total_draws = 0
    total_games = 0
    
    # Track stats per opponent
    # We need to map game_idx -> opponent_idx
    num_opponents = len(pool.models)
    opp_stats = {str(i): {'wins':0, 'draws':0, 'games':0} for i in range(num_opponents)}
    
    pairs = []
    is_black = []
    opp_indices = [] 
    
    for _ in range(EVAL_ROUNDS):
        for idx, (opp_model, opp_update) in enumerate(zip(pool.models, pool.updates)):
            # Current Black
            pairs.append((current_model, opp_model))
            is_black.append(True)
            opp_indices.append(idx)
            # Current White
            pairs.append((opp_model, current_model))
            is_black.append(False)
            opp_indices.append(idx)

    results = play_eval_games(pairs, is_black, EVAL_TEMP, pool.device, batch_size=64)
    
    for (outcome, curr_is_blk), opp_idx in zip(results, opp_indices):
        total_games += 1
        key = str(opp_idx)
        opp_stats[key]['games'] += 1
        
        won = False
        draw = False
        if outcome == GameState.DRAW:
            draw = True
            total_draws += 1
            opp_stats[key]['draws'] += 1
        elif (outcome == GameState.BLACK_WIN and curr_is_blk) or \
             (outcome == GameState.WHITE_WIN and not curr_is_blk):
            won = True
            total_wins += 1
            opp_stats[key]['wins'] += 1
            
    current_model.train()
    
    overall_wr = (total_wins + 0.5 * total_draws) / total_games if total_games else 0
    
    # Format per-opponent stats
    final_stats = {}
    for key, s in opp_stats.items():
        wr = (s['wins'] + 0.5 * s['draws']) / s['games'] if s['games'] else 0
        losses = s['games'] - s['wins'] - s['draws']
        final_stats[key] = {
            'wins': s['wins'], 'draws': s['draws'], 'losses': losses, 'games': s['games'], 'win_rate': wr
        }
        
    return overall_wr, final_stats


# ============================================================================
# Mining Logic
# ============================================================================

def load_checkpoint_model(path: str, device: torch.device) -> Optional[nn.Module]:
    try:
        ckpt = torch.load(path, map_location=device)
        model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model
    except: return None

def scan_historical_exploiters(current_model: nn.Module, pool: OpponentPool, 
                               scan_event_num: int) -> Tuple[List[Tuple[int, float]], int]:
    """Scan history for hard opponents."""
    output_dir = pool.output_dir
    device = pool.device
    
    # Discovery
    all_ckpts = []
    pattern = re.compile(r'checkpoint_update_(\d+)\.pt')
    for f in glob.glob(os.path.join(output_dir, "checkpoint_update_*.pt")):
        m = pattern.match(os.path.basename(f))
        if m:
            u = int(m.group(1))
            if u >= SCAN_START_UPDATE: all_ckpts.append(u)
    
    target_bucket = scan_event_num % NUM_SCAN_BUCKETS
    candidates = []
    for u in all_ckpts:
        bucket = (u // EVAL_INTERVAL_LATE) % NUM_SCAN_BUCKETS
        if bucket == target_bucket and u not in pool.updates:
            candidates.append(u)
            
    if not candidates: return [], len(candidates)
    
    # Quick Screen
    quick_res = []
    for u in candidates:
        model = load_checkpoint_model(os.path.join(output_dir, f"checkpoint_update_{u}.pt"), device)
        if not model: continue
        
        # Single opponent eval logic (simplified)
        pairs = [(current_model, model), (model, current_model)] * QUICK_SCREEN_ROUNDS
        is_blk = [True, False] * QUICK_SCREEN_ROUNDS
        res = play_eval_games(pairs, is_blk, EVAL_TEMP, device, batch_size=BATCH_INFERENCE_SIZE)
        wins = sum(1 for (o, b) in res if (o==GameState.BLACK_WIN and b) or (o==GameState.WHITE_WIN and not b))
        draws = sum(1 for (o, _) in res if o==GameState.DRAW)
        wr = (wins + 0.5*draws) / len(res)
        quick_res.append((u, wr))
        
    quick_res.sort(key=lambda x: x[1])
    hardest = quick_res[:TOP_K_QUICK_SCREEN]
    
    # Final Screen
    final_res = []
    for u, _ in hardest:
        model = load_checkpoint_model(os.path.join(output_dir, f"checkpoint_update_{u}.pt"), device)
        if not model: continue
        
        pairs = [(current_model, model), (model, current_model)] * FINAL_SCREEN_ROUNDS
        is_blk = [True, False] * FINAL_SCREEN_ROUNDS
        res = play_eval_games(pairs, is_blk, EVAL_TEMP, device, batch_size=BATCH_INFERENCE_SIZE)
        wins = sum(1 for (o, b) in res if (o==GameState.BLACK_WIN and b) or (o==GameState.WHITE_WIN and not b))
        draws = sum(1 for (o, _) in res if o==GameState.DRAW)
        wr = (wins + 0.5*draws) / len(res)
        final_res.append((u, wr))
        
    final_res.sort(key=lambda x: x[1])
    return final_res[:MAX_MINED_OPPONENTS_PER_EVENT], len(candidates)
