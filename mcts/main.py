"""
Two-stage MCTS distillation entry point.

    python3 main.py <stage> <working_dir>

stage ∈ {generate_data, stage1, stage2}. The working_dir is expected to contain
`teacher.pt` upfront. Each stage writes into its own subdirectory and is
resumable from on-disk state.

All hyperparameters are module-level constants below — single source of truth,
no CLI tuning surface.
"""

import os
import sys

sys.path.insert(0, os.getcwd())

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import random
from typing import Optional

import numpy as np
import torch

# ============================================================================
# Hyperparameters
# ============================================================================

# === Data generation ===
NUM_GAMES = 16384
NUM_SIMULATIONS_GEN = 2048
GAMES_PER_SHARD = 1024
PRIOR_TEMPERATURE = 1.28        # teacher logits → MCTS prior (entropy multiplier)
ACTION_TEMPERATURE = 1.0        # MCTS visits → move sampling; broadens trajectories
SEED_PROBABILITY = 0.5          # fraction of games started from a Renju opening (overrides gomoku.SEED_PROBABILITY for MCTS)

# === Stage 1 ===
STAGE1_EPOCHS = 8
RAW_BATCH_PER_UPDATE = 4096
STAGE1_LR = 1.0/1024
STAGE1_MIN_LR = 1.0/1024
STAGE1_KL_EMA_WINDOW = 32
STAGE1_KL_EMA_THRESHOLD: Optional[float] = None  # None → run all epochs
STAGE1_CHECKPOINT_INTERVAL = 128

# === Stage 2 ===
STAGE2_TOTAL_UPDATES = 2048
STAGE2_EPISODES_PER_UPDATE = 256
NUM_SIMULATIONS_S2 = 2048
STAGE2_OPTIMIZE_STEPS_PER_UPDATE = 4  # train K times on each self-play batch; LR is divided by K
STAGE2_LR = 1.0 / 1024 / STAGE2_OPTIMIZE_STEPS_PER_UPDATE
STAGE2_MIN_LR = STAGE2_LR / 8
STAGE2_DIRICHLET_ALPHA = 0.15
STAGE2_DIRICHLET_EPSILON = 0.25
STAGE2_CHECKPOINT_INTERVAL = 32
STAGE2_REPLAY_BUFFER_ROUNDS = 8      # number of past self-play rounds to retain for training
STAGE2_SAMPLE_RATIO = 0.5           # k_0 = SAMPLE_RATIO * len(most_recent_round); per-round draw budget
STAGE2_DECAY_RATIO = 0.5 ** 0.5     # k_i = k_0 * DECAY_RATIO**i (i=0 most recent); recency-weighted replay

# === Shared ===
TRAIN_BATCH_SIZE = 512          # GPU micro-batch cap
VALUE_LOSS_COEFF = 1.0
GRAD_CLIP_NORM = 16.0
WEIGHT_DECAY = 1.0 / 2 ** 24
C_PUCT = 1.25
DISCOUNT_GAMMA = 63.0/64
SEED = 42

# ============================================================================
# PyTorch performance settings
# ============================================================================

DEVICE = torch.device("cuda")
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Two-stage MCTS distillation")
    parser.add_argument('stage', choices=['generate_data', 'stage1', 'stage2'])
    parser.add_argument('working_dir', type=str)
    args = parser.parse_args()

    working_dir = args.working_dir
    os.makedirs(working_dir, exist_ok=True)
    seed_everything(SEED)

    if args.stage == 'generate_data':
        from data_generator import generate_stage1_data
        teacher_path = os.path.join(working_dir, "teacher.pt")
        if not os.path.exists(teacher_path):
            parser.error(f"Missing teacher: {teacher_path}")
        generate_stage1_data(
            teacher_path=teacher_path,
            data_dir=os.path.join(working_dir, "stage1_data"),
            num_games=NUM_GAMES,
            games_per_shard=GAMES_PER_SHARD,
            num_simulations=NUM_SIMULATIONS_GEN,
            c_puct=C_PUCT,
            prior_temperature=PRIOR_TEMPERATURE,
            action_temperature=ACTION_TEMPERATURE,
            seed_probability=SEED_PROBABILITY,
            gamma=DISCOUNT_GAMMA,
            seed=SEED,
            device=DEVICE,
        )
    elif args.stage == 'stage1':
        from stage1_trainer import run_stage1
        run_stage1(
            data_dir=os.path.join(working_dir, "stage1_data"),
            output_dir=os.path.join(working_dir, "stage1"),
            epochs=STAGE1_EPOCHS,
            raw_batch_per_update=RAW_BATCH_PER_UPDATE,
            train_batch_size=TRAIN_BATCH_SIZE,
            learning_rate=STAGE1_LR,
            min_lr=STAGE1_MIN_LR,
            weight_decay=WEIGHT_DECAY,
            value_loss_coeff=VALUE_LOSS_COEFF,
            grad_clip_norm=GRAD_CLIP_NORM,
            kl_ema_window=STAGE1_KL_EMA_WINDOW,
            kl_ema_threshold=STAGE1_KL_EMA_THRESHOLD,
            checkpoint_interval=STAGE1_CHECKPOINT_INTERVAL,
            seed=SEED,
            device=DEVICE,
        )
    elif args.stage == 'stage2':
        from stage2_trainer import run_stage2
        # stage1_final.pt is only needed for a fresh start; if stage 2 has its
        # own training_state.json, it can resume independently of stage 1's
        # artifacts (existence check is deferred into run_stage2).
        stage1_final = os.path.join(working_dir, "stage1", "stage1_final.pt")
        run_stage2(
            stage1_checkpoint=stage1_final,
            output_dir=os.path.join(working_dir, "stage2"),
            total_updates=STAGE2_TOTAL_UPDATES,
            episodes_per_update=STAGE2_EPISODES_PER_UPDATE,
            num_simulations=NUM_SIMULATIONS_S2,
            c_puct=C_PUCT,
            dirichlet_alpha=STAGE2_DIRICHLET_ALPHA,
            dirichlet_epsilon=STAGE2_DIRICHLET_EPSILON,
            seed_probability=SEED_PROBABILITY,
            gamma=DISCOUNT_GAMMA,
            learning_rate=STAGE2_LR,
            min_lr=STAGE2_MIN_LR,
            weight_decay=WEIGHT_DECAY,
            value_loss_coeff=VALUE_LOSS_COEFF,
            optimize_steps_per_update=STAGE2_OPTIMIZE_STEPS_PER_UPDATE,
            replay_buffer_rounds=STAGE2_REPLAY_BUFFER_ROUNDS,
            sample_ratio=STAGE2_SAMPLE_RATIO,
            decay_ratio=STAGE2_DECAY_RATIO,
            checkpoint_interval=STAGE2_CHECKPOINT_INTERVAL,
            device=DEVICE,
        )


if __name__ == "__main__":
    main()
