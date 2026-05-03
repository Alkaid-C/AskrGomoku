"""
CSV Logging for two-stage MCTS distillation.
"""

import csv
import os
from typing import ClassVar


class Stage1CSVLogger:
    """CSV logger for stage 1 (offline distillation from teacher data)."""

    COLUMNS: ClassVar[list[str]] = [
        'update', 'policy_loss', 'value_loss',
        'kl_target_student', 'kl_ema', 'lr', 'time_train',
    ]

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, "training_updates.csv")
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, update: int, metrics: dict) -> None:
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([update] + [metrics[c] for c in self.COLUMNS[1:]])


class Stage2CSVLogger:
    """CSV logger for stage 2 (vanilla MCTS self-play training)."""

    COLUMNS: ClassVar[list[str]] = [
        'update', 'policy_loss', 'value_loss', 'kl_target_student',
        'lr', 'avg_game_length', 'black_win_rate', 'draw_rate',
        'black_block_opps', 'black_block_mcts_rate', 'black_block_raw_rate',
        'white_block_opps', 'white_block_mcts_rate', 'white_block_raw_rate',
        'time_selfplay', 'time_train',
        'cache_hit_rate', 'cache_hits', 'cache_misses',
    ]

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, "training_updates.csv")
        if not os.path.exists(self.path):
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, update: int, metrics: dict) -> None:
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([update] + [metrics[c] for c in self.COLUMNS[1:]])
