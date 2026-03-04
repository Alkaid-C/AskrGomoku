"""
CSV Logging for MCTS Post-Training Metrics
"""

import csv
import os


class MCTSCSVLogger:
    """Manages CSV logging for MCTS post-training."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.training_path = os.path.join(output_dir, "training_updates.csv")
        self.eval_summary_path = os.path.join(output_dir, "eval_summary.csv")
        self.eval_details_path = os.path.join(output_dir, "eval_opponent_details.csv")

        self._init_training_csv()
        self._init_eval_summary_csv()
        self._init_eval_details_csv()

    def _init_training_csv(self) -> None:
        if not os.path.exists(self.training_path):
            with open(self.training_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'policy_loss', 'value_loss',
                    'model_entropy', 'mcts_entropy',
                    'temperature', 'sharpen_exponent',
                    'lr', 'avg_game_length',
                    'win_rate', 'draw_rate',
                    'time_selfplay', 'time_train',
                ])

    def _init_eval_summary_csv(self) -> None:
        if not os.path.exists(self.eval_summary_path):
            with open(self.eval_summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'overall_win_rate', 'total_games', 'eval_time',
                    'hardest_opponent_id', 'hardest_win_rate',
                    'easiest_opponent_id', 'easiest_win_rate',
                    'pool_size', 'checkpoint_added', 'evicted_opponent_id',
                ])

    def _init_eval_details_csv(self) -> None:
        if not os.path.exists(self.eval_details_path):
            with open(self.eval_details_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'opponent_id', 'wins', 'losses',
                    'draws', 'games', 'win_rate',
                ])

    def log_training_update(self, update: int, metrics: dict) -> None:
        with open(self.training_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update,
                metrics['policy_loss'], metrics['value_loss'],
                metrics['model_entropy'], metrics['mcts_entropy'],
                metrics['temperature'], metrics['sharpen_exponent'],
                metrics['lr'], metrics['avg_game_length'],
                metrics['win_rate'], metrics['draw_rate'],
                metrics['time_selfplay'], metrics['time_train'],
            ])

    def log_eval_summary(self, update: int, metrics: dict) -> None:
        with open(self.eval_summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update, metrics['overall_win_rate'], metrics['total_games'],
                metrics['eval_time'],
                metrics['hardest_opponent_id'], metrics['hardest_win_rate'],
                metrics['easiest_opponent_id'], metrics['easiest_win_rate'],
                metrics['pool_size'], metrics['checkpoint_added'],
                metrics['evicted_opponent_id'],
            ])

    def log_eval_opponent_details(self, update: int, opponent_id: int, metrics: dict) -> None:
        with open(self.eval_details_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update, opponent_id, metrics['wins'], metrics['losses'],
                metrics['draws'], metrics['games'], metrics['win_rate'],
            ])
