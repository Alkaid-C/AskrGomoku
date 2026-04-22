"""
CSV Logging for MCTS Post-Training Metrics
"""

import csv
import os


class MCTSCSVLogger:
    """Manages CSV logging for MCTS post-training."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.training_path = os.path.join(output_dir, "training_updates.csv")
        self._init_training_csv()

    def _init_training_csv(self) -> None:
        if not os.path.exists(self.training_path):
            with open(self.training_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'update', 'policy_loss', 'value_loss',
                    'model_entropy', 'mcts_entropy',
                    'temperature', 'sharpen_exponent',
                    'lr', 'avg_game_length',
                    'black_win_rate', 'draw_rate',
                    'black_block_opps', 'black_block_mcts_rate', 'black_block_raw_rate',
                    'white_block_opps', 'white_block_mcts_rate', 'white_block_raw_rate',
                    'time_selfplay', 'time_train',
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
                metrics['black_win_rate'], metrics['draw_rate'],
                metrics['black_block_opps'], metrics['black_block_mcts_rate'], metrics['black_block_raw_rate'],
                metrics['white_block_opps'], metrics['white_block_mcts_rate'], metrics['white_block_raw_rate'],
                metrics['time_selfplay'], metrics['time_train'],
            ])
