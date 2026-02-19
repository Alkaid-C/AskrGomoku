"""
CSV Logging Module for Training Metrics

Manages CSV logging for training updates, evaluation summaries, opponent details,
historical exploiter mining, and gradient conflict probing.
"""

import csv
import os


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

        # Initialize CSV files with headers if they don't exist
        self._init_training_updates_csv()
        self._init_eval_summary_csv()
        self._init_eval_opponent_details_csv()
        self._init_mining_log_csv()
        self._init_gradient_probe_csv()

    def _init_training_updates_csv(self):
        if not os.path.exists(self.training_updates_path):
            with open(self.training_updates_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'loss', 'win_rate', 'win_rate_black', 'win_rate_white',
                    'avg_game_length', 'entropy', 'value_loss', 'raw_value_mse',
                    'tactics_wins', 'tactics_blocks', 'tactics_synth_wins_eq',
                    'tactics_synth_wins_missed', 'tactics_synth_blocks',
                    'imitation_black', 'imitation_white',
                    'win_miss_ema', 'block_miss_ema', 'win_boost', 'block_boost',
                    'opr_attempted', 'opr_candidates_avg', 'opr_added', 'opr_winrate_avg', 'opr_orig_winrate_avg', 'opr_entropy_avg',
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
                    'blocks_0_2_cos_sim', 'blocks_0_2_policy_norm', 'blocks_0_2_value_norm',
                    'blocks_3_5_cos_sim', 'blocks_3_5_policy_norm', 'blocks_3_5_value_norm',
                    'blocks_6_8_cos_sim', 'blocks_6_8_policy_norm', 'blocks_6_8_value_norm',
                    'blocks_9_11_cos_sim', 'blocks_9_11_policy_norm', 'blocks_9_11_value_norm',
                    'blocks_12_14_cos_sim', 'blocks_12_14_policy_norm', 'blocks_12_14_value_norm',
                    'blocks_15_17_cos_sim', 'blocks_15_17_policy_norm', 'blocks_15_17_value_norm',
                    'overall_entropy_norm', 'stem_entropy_norm',
                    'blocks_0_2_entropy_norm', 'blocks_3_5_entropy_norm',
                    'blocks_6_8_entropy_norm', 'blocks_9_11_entropy_norm',
                    'blocks_12_14_entropy_norm', 'blocks_15_17_entropy_norm'
                ])

    def log_training_update(self, update: int, metrics: dict):
        with open(self.training_updates_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update, metrics['loss'], metrics['win_rate'], metrics['win_rate_black'], metrics['win_rate_white'],
                metrics['avg_game_length'], metrics['entropy'], metrics['value_loss'], metrics['raw_value_mse'],
                metrics['tactics_wins'], metrics['tactics_blocks'], metrics['tactics_synth_wins_eq'],
                metrics['tactics_synth_wins_missed'], metrics['tactics_synth_blocks'],
                metrics['imitation_black'], metrics['imitation_white'],
                metrics['win_miss_ema'], metrics['block_miss_ema'], metrics['win_boost'], metrics['block_boost'],
                metrics.get('opr_attempted', 0), metrics.get('opr_candidates_avg', 0.0),
                metrics.get('opr_added', 0), metrics.get('opr_winrate_avg', 0.0), metrics.get('opr_orig_winrate_avg', 0.0), metrics.get('opr_entropy_avg', 0.0),
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
                metrics['blocks_0_2_cos_sim'], metrics['blocks_0_2_policy_norm'], metrics['blocks_0_2_value_norm'],
                metrics['blocks_3_5_cos_sim'], metrics['blocks_3_5_policy_norm'], metrics['blocks_3_5_value_norm'],
                metrics['blocks_6_8_cos_sim'], metrics['blocks_6_8_policy_norm'], metrics['blocks_6_8_value_norm'],
                metrics['blocks_9_11_cos_sim'], metrics['blocks_9_11_policy_norm'], metrics['blocks_9_11_value_norm'],
                metrics['blocks_12_14_cos_sim'], metrics['blocks_12_14_policy_norm'], metrics['blocks_12_14_value_norm'],
                metrics['blocks_15_17_cos_sim'], metrics['blocks_15_17_policy_norm'], metrics['blocks_15_17_value_norm'],
                metrics.get('overall_entropy_norm', 0.0), metrics.get('stem_entropy_norm', 0.0),
                metrics.get('blocks_0_2_entropy_norm', 0.0), metrics.get('blocks_3_5_entropy_norm', 0.0),
                metrics.get('blocks_6_8_entropy_norm', 0.0), metrics.get('blocks_9_11_entropy_norm', 0.0),
                metrics.get('blocks_12_14_entropy_norm', 0.0), metrics.get('blocks_15_17_entropy_norm', 0.0)
            ])
