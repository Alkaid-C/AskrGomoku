"""
CSV Logging Module for Training Metrics

Manages CSV logging for training updates, evaluation summaries, opponent details,
historical exploiter mining, gradient conflict probing, and tactical accuracy probing.
"""

import os
import csv


class CSVLogger:
    """Manages CSV logging for training metrics."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define CSV file paths
        self.training_updates_path = os.path.join(output_dir, "training_updates.csv")
        self.eval_summary_path = os.path.join(output_dir, "eval_summary.csv")
        self.eval_opponent_details_path = os.path.join(output_dir, "eval_opponent_details.csv")
        self.mining_log_path = os.path.join(output_dir, "mining_log.csv")
        self.gradient_probe_path = os.path.join(output_dir, "gradient_probe.csv")
        self.tactical_probe_path = os.path.join(output_dir, "tactical_probe.csv")
        self.search_training_path = os.path.join(output_dir, "search_training.csv")

        # Initialize CSV files with headers if they don't exist
        self._init_training_updates_csv()
        self._init_eval_summary_csv()
        self._init_eval_opponent_details_csv()
        self._init_mining_log_csv()
        self._init_gradient_probe_csv()
        self._init_tactical_probe_csv()
        self._init_search_training_csv()

    def _init_training_updates_csv(self):
        if not os.path.exists(self.training_updates_path):
            with open(self.training_updates_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'loss', 'win_rate', 'win_rate_black', 'win_rate_white',
                    'avg_game_length', 'entropy', 'value_loss', 'raw_value_mse',
                    'time_total', 'time_selfplay', 'time_train', 'learning_rate'
                ])

    def _init_eval_summary_csv(self):
        if not os.path.exists(self.eval_summary_path):
            with open(self.eval_summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'overall_win_rate', 'total_games', 'eval_time',
                    'hardest_opponent_id', 'hardest_win_rate',
                    'easiest_opponent_id', 'easiest_win_rate',
                    'pool_size', 'checkpoint_added', 'evicted_opponent_id'
                ])

    def _init_eval_opponent_details_csv(self):
        if not os.path.exists(self.eval_opponent_details_path):
            with open(self.eval_opponent_details_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'opponent_id', 'wins', 'losses', 'draws', 'total_games', 'win_rate'
                ])

    def _init_mining_log_csv(self):
        if not os.path.exists(self.mining_log_path):
            with open(self.mining_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'scan_update', 'scan_event_num', 'bucket_id',
                    'total_candidates', 'candidates_after_filter',
                    'mined_opponent_id', 'mined_win_rate', 'mined_rank',
                    'added_to_pool', 'evicted_opponent_id',
                    'scan_time'
                ])

    def _init_gradient_probe_csv(self):
        if not os.path.exists(self.gradient_probe_path):
            with open(self.gradient_probe_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update',
                    'overall_cos_sim', 'overall_policy_norm', 'overall_value_norm',
                    'stem_cos_sim', 'stem_policy_norm', 'stem_value_norm',
                    'blocks_0_3_cos_sim', 'blocks_0_3_policy_norm', 'blocks_0_3_value_norm',
                    'blocks_4_7_cos_sim', 'blocks_4_7_policy_norm', 'blocks_4_7_value_norm',
                    'blocks_8_11_cos_sim', 'blocks_8_11_policy_norm', 'blocks_8_11_value_norm',
                    'blocks_12_15_cos_sim', 'blocks_12_15_policy_norm', 'blocks_12_15_value_norm'
                ])

    def _init_tactical_probe_csv(self):
        if not os.path.exists(self.tactical_probe_path):
            with open(self.tactical_probe_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update',
                    'win_opportunities', 'win_hits', 'win_misses', 'win_accuracy',
                    'block_opportunities', 'block_hits', 'block_misses', 'block_accuracy'
                ])

    def _init_search_training_csv(self):
        if not os.path.exists(self.search_training_path):
            with open(self.search_training_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'policy_loss', 'ranking_inside_loss', 'separation_outside_loss',
                    'value_loss', 'top1_acc', 'top3_acc', 'value_mse',
                    'unfrozen_blocks', 'learning_rate', 'time_total', 'time_selfplay', 'time_train'
                ])

    def log_training_update(self, update: int, metrics: dict):
        with open(self.training_updates_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update, metrics['loss'], metrics['win_rate'], metrics['win_rate_black'], metrics['win_rate_white'],
                metrics['avg_game_length'], metrics['entropy'], metrics['value_loss'], metrics['raw_value_mse'],
                metrics['time_total'], metrics['time_selfplay'], metrics['time_train'], metrics['learning_rate']
            ])

    def log_eval_summary(self, update: int, metrics: dict):
        with open(self.eval_summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update, metrics['overall_win_rate'], metrics['total_games'], metrics['eval_time'],
                metrics['hardest_opponent_id'], metrics['hardest_win_rate'],
                metrics['easiest_opponent_id'], metrics['easiest_win_rate'],
                metrics['pool_size'], metrics['checkpoint_added'], metrics['evicted_opponent_id']
            ])

    def log_eval_opponent_details(self, update: int, opponent_id: int, metrics: dict):
        with open(self.eval_opponent_details_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update, opponent_id, metrics['wins'], metrics['losses'],
                metrics['draws'], metrics['games'], metrics['win_rate']
            ])

    def log_mining_event(self, scan_update: int, scan_event_num: int, bucket_id: int,
                         total_candidates: int, candidates_after_filter: int,
                         mined_opponent_id: int, mined_win_rate: float, mined_rank: int,
                         added_to_pool: bool, evicted_opponent_id: int, scan_time: float):
        with open(self.mining_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                scan_update, scan_event_num, bucket_id,
                total_candidates, candidates_after_filter,
                mined_opponent_id, mined_win_rate, mined_rank,
                added_to_pool, evicted_opponent_id,
                scan_time
            ])

    def log_gradient_probe(self, update: int, metrics: dict):
        with open(self.gradient_probe_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update,
                metrics['overall_cos_sim'], metrics['overall_policy_norm'], metrics['overall_value_norm'],
                metrics['stem_cos_sim'], metrics['stem_policy_norm'], metrics['stem_value_norm'],
                metrics['blocks_0_3_cos_sim'], metrics['blocks_0_3_policy_norm'], metrics['blocks_0_3_value_norm'],
                metrics['blocks_4_7_cos_sim'], metrics['blocks_4_7_policy_norm'], metrics['blocks_4_7_value_norm'],
                metrics['blocks_8_11_cos_sim'], metrics['blocks_8_11_policy_norm'], metrics['blocks_8_11_value_norm'],
                metrics['blocks_12_15_cos_sim'], metrics['blocks_12_15_policy_norm'], metrics['blocks_12_15_value_norm']
            ])

    def log_tactical_probe(self, update: int, metrics: dict):
        with open(self.tactical_probe_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update,
                metrics['win_opportunities'], metrics['win_hits'], metrics['win_misses'], metrics['win_accuracy'],
                metrics['block_opportunities'], metrics['block_hits'], metrics['block_misses'], metrics['block_accuracy']
            ])

    def log_search_training(self, update: int, metrics: dict):
        with open(self.search_training_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update, metrics['policy_loss'], metrics['ranking_inside_loss'], metrics['separation_outside_loss'],
                metrics['value_loss'], metrics['top1_acc'], metrics['top3_acc'], metrics['value_mse'],
                metrics['unfrozen_blocks'], metrics['learning_rate'], metrics['time_total'],
                metrics['time_selfplay'], metrics['time_train']
            ])
