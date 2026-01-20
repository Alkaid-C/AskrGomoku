"""
Refactored Gomoku Training Entry Point.

Orchestrates the training loop using modular components:
- gameplay_loop (Inference)
- data_augmentation (Logic/Tricks)
- optimization (Training)
- evaluation (Strategy)
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import json
from collections import deque

from model import GomokuPolicyNet, N_BLOCKS, zero_center_taps, SEED_PROBABILITY
from gomoku_rules import RENJU_OPENING_SEQUENCES
from gameplay_loop import play_episodes_batched, select_action_batch, BATCH_INFERENCE_SIZE
from data_augmentation import generate_cler_samples, compute_outcome_stats
from optimization import train_on_batch, TOTAL_UPDATES, LEARNING_RATE, WEIGHT_DECAY, LR_DECAY, MIN_LR, EPISODES_PER_UPDATE, WIN_MIN_BOOST, WIN_MAX_BOOST, BLOCK_MIN_BOOST, BLOCK_MAX_BOOST, MISS_RATE_EMA_WINDOW, ENTROPY_TARGET_START, ENTROPY_EMA_LAMBDA, TEMPERATURE_TRAIN
from evaluation import OpponentPool, evaluate_policy, scan_historical_exploiters, get_eval_interval, WIN_RATE_THRESHOLD, SCAN_PERIOD, NUM_SCAN_BUCKETS, SCAN_START_UPDATE, load_checkpoint_model, EVAL_ROUNDS, TRAINING_STATE_FILE
from csv_logger import CSVLogger

# ============================================================================
# State Management
# ============================================================================

def save_training_state(output_dir: str, update: int, opponent_pool_updates: list,
                        win_miss_ema: float, block_miss_ema: float,
                        per_opponent_win_rates: dict, scan_event_counter: int,
                        evals_since_last_scan: int, ema_entropy: float):
    state = {
        'current_update': update,
        'opponent_pool_updates': opponent_pool_updates,
        'total_updates': TOTAL_UPDATES,
        'win_miss_ema': win_miss_ema,
        'block_miss_ema': block_miss_ema,
        'per_opponent_win_rates': per_opponent_win_rates,
        'scan_event_counter': scan_event_counter,
        'evals_since_last_scan': evals_since_last_scan,
        'ema_entropy': ema_entropy
    }
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    temp = path + '.tmp'
    with open(temp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(temp, path)

def load_training_state(output_dir: str, device: torch.device):
    path = os.path.join(output_dir, TRAINING_STATE_FILE)
    if not os.path.exists(path):
        return None
    
    print(f"Loading state from {path}")
    with open(path, 'r') as f:
        state = json.load(f)
        
    current_update = state['current_update']
    
    # Load Checkpoint
    ckpt_path = os.path.join(output_dir, f"checkpoint_update_{current_update}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return None
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    zero_center_taps(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: max(LEARNING_RATE * (LR_DECAY**ep), MIN_LR)/LEARNING_RATE)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Rebuild Pool
    pool = OpponentPool(device, output_dir)
    pool_updates = state['opponent_pool_updates']
    for u in pool_updates:
        p_path = os.path.join(output_dir, f"checkpoint_update_{u}.pt")
        try:
            p_ckpt = torch.load(p_path, map_location=device, weights_only=False)
            p_model = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
            p_model.load_state_dict(p_ckpt['model_state_dict'])
            p_model.eval()
            pool.models.append(p_model)
            pool.updates.append(u)
        except Exception as e:
            print(f"Failed to load opponent {u}: {e}")
            
    if not pool.models:
        print("No opponents loaded!")
        return None
        
    # State Vars
    win_miss_ema = state.get('win_miss_ema', 1.0)
    block_miss_ema = state.get('block_miss_ema', 1.0)
    pool.win_rates = state.get('per_opponent_win_rates', {})
    scan_event_counter = state.get('scan_event_counter', 0)
    evals_since_last_scan = state.get('evals_since_last_scan', 0)
    ema_entropy = state.get('ema_entropy', ENTROPY_TARGET_START)
    
    next_eval = current_update + get_eval_interval(current_update)
    
    return (model, optimizer, scheduler, pool, current_update, next_eval, 
            win_miss_ema, block_miss_ema, scan_event_counter, evals_since_last_scan, ema_entropy)

# ============================================================================
# Main
# ============================================================================

def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda")
    
    # Enable TF32 (PyTorch 2.9+ API)
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
    logger = CSVLogger(output_dir)
    
    # Try Resume
    resume = load_training_state(output_dir, device)
    
    if resume:
        (policy, optimizer, scheduler, pool, start_update, next_eval_update, 
         win_miss_ema, block_miss_ema, scan_counter, evals_since_scan, ema_entropy) = resume
        print(f"Resumed from update {start_update}")
    else:
        print("Starting fresh training")
        start_update = 0
        policy = GomokuPolicyNet(n_blocks=N_BLOCKS).to(device)
        zero_center_taps(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: max(LEARNING_RATE * (LR_DECAY**ep), MIN_LR)/LEARNING_RATE)
        
        pool = OpponentPool(device, output_dir)
        pool.initialize_random()
        
        win_miss_ema = 1.0
        block_miss_ema = 1.0
        scan_counter = 0
        evals_since_scan = 0
        next_eval_update = get_eval_interval(0)
        ema_entropy = ENTROPY_TARGET_START

    # Metrics tracking
    metric_buffer = {
        'loss': [], 'win_rate': [], 'win_rate_as_black': [], 'win_rate_as_white': [],
        'wins': [], 'losses': [], 'draws': [], 'entropy': [], 'value_loss': [],
        'raw_value_mse': [], 'avg_length': [], 'time': [], 'selfplay_time': [],
        'train_time': [], 'tactics_wins': [], 'tactics_blocks': [],
        'tactics_synthetic_wins_eq': [], 'tactics_synthetic_wins_missed': [],
        'tactics_synthetic_blocks': [], 'imitation_black': [], 'imitation_white': [],
        'win_miss_ema': [], 'block_miss_ema': [], 'win_boost': [], 'block_boost': [],
        'cler_attempted': [], 'cler_candidates': [], 'cler_added': [],
        'cler_winrate_sum': [], 'cler_orig_winrate_sum': [], 'cler_entropy_sum': []
    }

    # Training loop
    training_start_time = time.time()
    num_openings = len(RENJU_OPENING_SEQUENCES)
    
    from optimization import PRINT_INTERVAL
    
    for update in range(start_update, TOTAL_UPDATES):
        t0 = time.time()
        
        # A. Self Play
        opponents = [pool.sample() for _ in range(EPISODES_PER_UPDATE)]
        pairs = []
        is_black = []
        
        for opp in opponents:
            if random.random() < 0.5:
                pairs.append((policy, opp))
                is_black.append(True)
            else:
                pairs.append((opp, policy))
                is_black.append(False)
                
        # Opening Seeding
        opening_ids = [random.randint(0, num_openings - 1) if random.random() < SEED_PROBABILITY else -1 for _ in range(EPISODES_PER_UPDATE)]
        
        trajectories = play_episodes_batched(
            pairs, is_black, TEMPERATURE_TRAIN, device, batch_size=BATCH_INFERENCE_SIZE, 
            select_action_batch_fn=select_action_batch, opening_ids=opening_ids
        )
        t_selfplay = time.time() - t0
        
        # B. CLER
        cler_samples, cler_metrics = generate_cler_samples(
            trajectories, is_black, opponents, policy, device, update
        )
        
        # C. Boost
        win_hit = 1.0 - win_miss_ema
        block_hit = 1.0 - block_miss_ema
        win_boost = WIN_MIN_BOOST + (1.0 - win_hit**2) * (WIN_MAX_BOOST - WIN_MIN_BOOST)
        block_boost = BLOCK_MIN_BOOST + (1.0 - block_hit**2) * (BLOCK_MAX_BOOST - BLOCK_MIN_BOOST)
        
        # D. Training
        t1 = time.time()
        (loss, mean_ret, mean_ent, val_loss, mse, 
         n_wins, n_blocks, n_sw_eq, n_sw_miss, n_sb, 
         n_imi_b, n_imi_w, 
         w_opp, w_miss, b_opp, b_miss, n_cler, probe) = train_on_batch(
            policy, trajectories, optimizer, device, update, 
            win_boost, block_boost, cler_samples, ema_entropy
        )
        t_train = time.time() - t1
        
        scheduler.step()
        
        # E. Stats
        ema_entropy = ENTROPY_EMA_LAMBDA * mean_ent + (1.0 - ENTROPY_EMA_LAMBDA) * ema_entropy
        
        curr_w_miss = w_miss / w_opp if w_opp > 0 else 0.0
        curr_b_miss = b_miss / b_opp if b_opp > 0 else 0.0
        alpha = 1.0 / MISS_RATE_EMA_WINDOW
        win_miss_ema = alpha * curr_w_miss + (1.0 - alpha) * win_miss_ema
        block_miss_ema = alpha * curr_b_miss + (1.0 - alpha) * block_miss_ema
        
        stats = compute_outcome_stats(trajectories, is_black)
        
        # Buffer Updates
        metric_buffer['loss'].append(loss)
        metric_buffer['win_rate'].append(stats['win_rate'])
        metric_buffer['win_rate_as_black'].append(stats['win_rate_as_black'])
        metric_buffer['win_rate_as_white'].append(stats['win_rate_as_white'])
        metric_buffer['wins'].append(stats['wins'])
        metric_buffer['losses'].append(stats['losses'])
        metric_buffer['draws'].append(stats['draws'])
        metric_buffer['entropy'].append(mean_ent)
        metric_buffer['value_loss'].append(val_loss)
        metric_buffer['raw_value_mse'].append(mse)
        metric_buffer['avg_length'].append(stats['avg_length'])
        metric_buffer['time'].append(time.time()-t0)
        metric_buffer['selfplay_time'].append(t_selfplay)
        metric_buffer['train_time'].append(t_train)
        metric_buffer['tactics_wins'].append(n_wins)
        metric_buffer['tactics_blocks'].append(n_blocks)
        metric_buffer['tactics_synthetic_wins_eq'].append(n_sw_eq)
        metric_buffer['tactics_synthetic_wins_missed'].append(n_sw_miss)
        metric_buffer['tactics_synthetic_blocks'].append(n_sb)
        metric_buffer['imitation_black'].append(n_imi_b)
        metric_buffer['imitation_white'].append(n_imi_w)
        metric_buffer['win_miss_ema'].append(win_miss_ema)
        metric_buffer['block_miss_ema'].append(block_miss_ema)
        metric_buffer['win_boost'].append(win_boost)
        metric_buffer['block_boost'].append(block_boost)
        metric_buffer['cler_attempted'].append(cler_metrics['cf_attempted_episodes'])
        metric_buffer['cler_candidates'].append(cler_metrics['cf_steps_candidates_total'])
        metric_buffer['cler_added'].append(cler_metrics['cf_added_samples'])
        metric_buffer['cler_winrate_sum'].append(cler_metrics['cf_best_winrate_sum'])
        metric_buffer['cler_orig_winrate_sum'].append(cler_metrics['cf_orig_winrate_sum'])
        metric_buffer['cler_entropy_sum'].append(cler_metrics['cf_entropy_selected_sum'])

        # F. Logging
        if (update + 1) % PRINT_INTERVAL == 0:
            avg_loss = np.mean(metric_buffer['loss'])
            avg_win_rate = np.mean(metric_buffer['win_rate'])
            avg_win_rate_black = np.mean(metric_buffer['win_rate_as_black'])
            avg_win_rate_white = np.mean(metric_buffer['win_rate_as_white'])
            total_wins = sum(metric_buffer['wins'])
            total_losses = sum(metric_buffer['losses'])
            total_draws = sum(metric_buffer['draws'])
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
            
            # CLER Aggregation
            total_cler_attempted = sum(metric_buffer['cler_attempted'])
            total_cler_candidates = sum(metric_buffer['cler_candidates'])
            total_cler_added = sum(metric_buffer['cler_added'])
            avg_cler_winrate = sum(metric_buffer['cler_winrate_sum']) / max(total_cler_added, 1)
            avg_cler_orig_winrate = sum(metric_buffer['cler_orig_winrate_sum']) / max(total_cler_added, 1)
            avg_cler_entropy = sum(metric_buffer['cler_entropy_sum']) / max(total_cler_added, 1)
            avg_cler_candidates_per_att = total_cler_candidates / max(total_cler_attempted, 1)

            print(f"Upd {update+1} | Win: {avg_win_rate:.2f} | Ent: {avg_entropy:.3f} | Loss: {avg_loss:.3f} | Time: {avg_time:.1f}s")
            
            logger.log_training_update(update+1, {
                'loss': avg_loss, 'win_rate': avg_win_rate, 
                'win_rate_black': avg_win_rate_black, 'win_rate_white': avg_win_rate_white,
                'avg_game_length': avg_length, 'entropy': avg_entropy,
                'value_loss': avg_value_loss, 'raw_value_mse': avg_raw_value_mse,
                'tactics_wins': total_wins_found, 'tactics_blocks': total_blocks_found,
                'tactics_synth_wins_eq': total_synthetic_wins_eq, 'tactics_synth_wins_missed': total_synthetic_wins_missed, 'tactics_synth_blocks': total_synthetic_blocks,
                'imitation_black': total_imitation_black, 'imitation_white': total_imitation_white,
                'win_miss_ema': win_miss_ema, 'block_miss_ema': block_miss_ema,
                'win_boost': win_boost, 'block_boost': block_boost,
                'cf_attempted': total_cler_attempted, 'cf_candidates_avg': avg_cler_candidates_per_att,
                'cf_added': total_cler_added, 'cf_winrate_avg': avg_cler_winrate, 'cf_entropy_avg': avg_cler_entropy,
                'time_total': avg_time, 'time_selfplay': avg_selfplay_time, 'time_train': avg_train_time,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            if probe: logger.log_gradient_probe(update+1, probe)
            
            # Reset buffer
            for k in metric_buffer: metric_buffer[k] = []

        # G. Evaluation
        if (update + 1) >= next_eval_update:
            # Checkpoint
            ckpt_path = os.path.join(output_dir, f"checkpoint_update_{update+1}.pt")
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'update': update+1
            }, ckpt_path)
            
            # Eval
            eval_start = time.time()
            wr, opp_stats = evaluate_policy(policy, pool)
            print(f"Eval @ {update+1}: WR {wr:.2f}")
            
            # Log Opponents
            for idx_str, s in opp_stats.items():
                opp_idx = int(idx_str)
                opp_id = pool.updates[opp_idx]
                # Update win rate in pool for this specific opponent instance
                pool.win_rates[str(opp_id)] = s['win_rate']
                logger.log_eval_opponent_details(update+1, int(opp_id), s)
            
            # Update Pool
            added = False
            evicted = None
            if wr >= WIN_RATE_THRESHOLD:
                evicted = pool.add(policy, update+1)
                added = True
            
            # Log Summary
            sorted_opps = sorted(opp_stats.items(), key=lambda x: x[1]['win_rate'])
            hardest_id = int(sorted_opps[0][0]) if sorted_opps else -1
            hardest_wr = sorted_opps[0][1]['win_rate'] if sorted_opps else 0.0
            easiest_id = int(sorted_opps[-1][0]) if sorted_opps else -1
            easiest_wr = sorted_opps[-1][1]['win_rate'] if sorted_opps else 0.0

            logger.log_eval_summary(update+1, {
                'overall_win_rate': wr, 'total_games': len(opp_stats)*EVAL_ROUNDS*2, 'eval_time': time.time()-eval_start,
                'hardest_opponent_id': hardest_id, 'hardest_win_rate': hardest_wr,
                'easiest_opponent_id': easiest_id, 'easiest_win_rate': easiest_wr,
                'pool_size': len(pool.models), 'checkpoint_added': added, 'evicted_opponent_id': evicted if evicted is not None else -1
            })
            
            # Mining
            evals_since_scan += 1
            if (update + 1) >= SCAN_START_UPDATE and evals_since_scan >= SCAN_PERIOD:
                t_scan = time.time()
                mined, cand_count = scan_historical_exploiters(policy, pool, scan_counter)
                
                # Log Mined
                for rank, (u, r) in enumerate(mined, 1):
                    mined_model = load_checkpoint_model(os.path.join(output_dir, f"checkpoint_update_{u}.pt"), device)
                    if mined_model is None:
                        continue
                        
                    evicted_id = pool.add(mined_model, u)
                    pool.win_rates[str(u)] = r
                    print(f"Mined: {u} (WR {r:.2f})")
                    
                    logger.log_mining_event(
                        update+1, scan_counter, scan_counter % NUM_SCAN_BUCKETS,
                        cand_count, cand_count, u, r, rank, True, evicted_id if evicted_id else -1, time.time()-t_scan
                    )
                    
                scan_counter += 1
                evals_since_scan = 0
                
            # Schedule Next
            next_eval_update = (update + 1) + get_eval_interval(update+1)
            
            # Save State
            save_training_state(output_dir, update+1, pool.updates, win_miss_ema, block_miss_ema, 
                                pool.win_rates, scan_counter, evals_since_scan, ema_entropy)

    # Final Save
    final_path = os.path.join(output_dir, "final_policy.pt")
    torch.save(policy.state_dict(), final_path)
    save_training_state(output_dir, TOTAL_UPDATES, pool.updates, win_miss_ema, block_miss_ema, 
                        pool.win_rates, scan_counter, evals_since_scan, ema_entropy)
    print("Training Complete.")

if __name__ == "__main__":
    main()
