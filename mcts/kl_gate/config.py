"""
Constants for the kl_gate study — single source of truth, no CLI tuning surface.

Also puts `mcts/` on `sys.path`, so importing this module first is what lets the
others reach `main`, `mcts`, `gomoku` and `model`. Search-side constants
(`C_PUCT`, `DISCOUNT_GAMMA`, `FPU_MULTIPLIER`) and `WEIGHT_DECAY` are taken from
`main.py` at their point of use rather than redefined here, so the labelling
search matches deployment exactly. Importing `main` is side-effect-safe: `main()`
is guarded by `__name__`, and its only module-level effects are
`PYTORCH_ALLOC_CONF` and the tf32 flags, both of which we want.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# === Labelling search ===
LABEL_NUM_SIMULATIONS = 384
ACTION_TEMPERATURE = 0.75       # MCTS visits -> move sampling; broadens trajectories

# === Dataset ===
TRAIN_GAMES = 32768             # the leading whole shards
VAL_GAMES = 1024                # the trailing shard
GAMES_PER_SHARD = 1024
# Start-position mix (must sum to 1). Renju openings carry a random offset and
# 184 variants, random-3 starts are drawn fresh every game; only the empty start
# is a single fixed position.
P_RENJU = 0.5
P_RANDOM3 = 0.3
P_EMPTY = 0.2
assert abs(P_RENJU + P_RANDOM3 + P_EMPTY - 1.0) < 1e-9

# Regression target is log(KL + LOG_EPSILON). The epsilon sits at the finite-
# simulation resolution floor of a LABEL_NUM_SIMULATIONS-sim search,
# ~(k_eff - 1) / (2 * LABEL_NUM_SIMULATIONS), so the log cannot stretch the
# unresolvable low end into large negative values that would dominate the
# gradient. Retune alongside LABEL_NUM_SIMULATIONS.
LOG_EPSILON = 0.01
PRIOR_LOG_EPSILON = 1e-9        # floor for the log-prior input plane

# === Network ===
NET_WIDTH = 64                  # must equal the sum of all stem branch channels
NET_BLOCKS = 8
NET_DILATION_SCHEDULE = [1, 2, 1, 3, 1, 2, 1, 1]  # length must equal NET_BLOCKS
NET_GN_GROUPS = 8               # must divide NET_WIDTH
NET_SE_REDUCTION = 4
NET_STEM_3X3_CHANNELS = 32
NET_STEM_DIRECTIONAL_5X5_CHANNELS = 16
NET_STEM_DIRECTIONAL_7X7_CHANNELS = 16
NET_HEAD_HIDDEN = 128

# === Training ===
LR = 1e-3
MIN_LR = 1e-5
EPOCHS = 30
BATCH_SIZE = 512                # base samples per step; 8x that many rows after augmentation
GRAD_CLIP_NORM = 16.0

SEED = 4242                     # distinct from main.SEED

# === Paths (relative to mcts/) ===
# The stage-2 output (update 216), NOT mcts/final_policy.pt — that file is a
# byte-identical copy of the RL teacher.pt (update 0), whose policy was never
# distilled onto MCTS-shaped targets and whose prior is therefore far sharper
# (33% of positions with max prior > 0.95, vs 8% here).
POLICY_PATH = "release/stage2/final_policy.pt"
DATA_DIR = "kl_gate/data"
OUT_DIR = "kl_gate/out"
SHARD_FILENAME = "kl_gate_shard_{:04d}.npz"
