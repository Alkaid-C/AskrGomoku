# vanilla/ — Training Pipeline (Simple Model)

## File Responsibilities

- `main.py` — Entry point. Training loop, state save/load, CLI.
- `model.py` — `GomokuPolicyNet` architecture and constants. Rarely changed.
- `training.py` — Loss computation, GAE, gradient accumulation, gradient probe.
- `enhancement.py` — Sample augmentation/generation: 8-fold symmetry, tactical boost, OPR, imitation.
- `gomoku.py` — Game engine, batched self-play, action selection, Renju openings. Rarely changed.
- `eval.py` — Opponent pool management, evaluation, historical mining.
- `csv_logger.py` — CSV logging for all metrics.
- `play_web.py` — Flask server for interactive play against a checkpoint.

## Model Architecture (`model.py`)

Same three-method API as `main/model.py`: `forward()` (both heads), `forward_policy_only()` (self-play/eval), `forward_value_only()` (GAE). Block count is read from the `N_BLOCKS` constant — not a constructor argument.

### Stem

Single 3×3 conv (`3` → `WIDTH`) → GroupNorm → SiLU.

### Trunk

`N_BLOCKS` standard `ResidualBlock` (pre-activation, GroupNorm, no dilation, no SE) → GroupNorm → SiLU.

### Policy Head

3× Conv3×3 with GroupNorm + SiLU → Conv1×1 → 225 logits.

### Value Head

Conv1×1 (`WIDTH` → `VALUE_HEAD_CHANNELS`) → SiLU → flatten → FC (`VALUE_HEAD_CHANNELS * 225` → `VALUE_HEAD_HIDDEN`) → SiLU → FC → tanh.

## Training Loop (`main.py`)

Each update:
1. Sample opponents from pool (`UNIFORM_SAMPLING_FRACTION` uniform, rest difficulty-weighted)
2. Play `EPISODES_PER_UPDATE` games via `play_episodes_batched` (`SEED_PROBABILITY` start from Renju openings)
3. Generate OPR samples from lost games (`enhancement.py`)
4. `train_on_batch`: tactical enhancement → 8-fold augmentation → GAE → loss → gradient probe → optimizer step
5. Every `get_eval_interval(update)` updates: save checkpoint, evaluate vs pool, conditionally add to pool, possibly mine historical exploiters

## Loss Function (`training.py`)

```
loss = policy_loss + value_coeff * value_loss + ENTROPY_BONUS_COEFF * entropy_bonus_scale * entropy_loss
```

- **Policy loss**: `-Σ(weight * advantage * log_prob) / Σ(weight)`. Advantage = `(1-α)*max(0, return) + α*max(0, GAE) + tactical_boost`. The `max(0, ...)` means only positive advantages reinforce — bad moves are never explicitly pushed down.
- **Value loss**: weighted MSE between predicted value and target. Target = return (terminal) or `-V(next)` (non-terminal, negamax convention). Only computed on real samples (not synthetic). Coefficient ramps from `VALUE_LOSS_COEFF_START` to `VALUE_LOSS_COEFF_END` via α.
- **Entropy loss**: `-Σ(weight * entropy) / Σ(weight)`, scaled by `entropy_bonus_scale = entropy_schedule / ema_entropy` (adaptive ratio).

## Schedules and Ramps

- **LR**: tanh decay from `LEARNING_RATE` to `MIN_LR`. Midpoint at `LR_DECAY_MIDPOINT_PERCENTAGE` of training, transition width `LR_DECAY_STEEPNESS`.
- **Entropy target**: sigmoid (tanh-based) decay from `ENTROPY_TARGET_START` to `ENTROPY_TARGET_END`. Midpoint at `ENTROPY_DECAY_MIDPOINT_PERCENTAGE`, width `ENTROPY_DECAY_STEEPNESS`.
- **Baseline α** (GAE ramp): cosine ramp from 0→1 over `[0, BASELINE_RAMP_END]`. At α=0 uses raw returns; at α=1 uses GAE. Also controls value loss coefficient interpolation.
- **Eval interval**: `EVAL_INTERVAL_EARLY` (updates < 128) → `EVAL_INTERVAL_MID` (< 2048) → `EVAL_INTERVAL_LATE`.

## Enhancement Details (`enhancement.py`)

### Tactical Boost

Scans every training sample for win-in-1 and block-win-in-1. If detected:
- **Correct move taken**: adds a boost to the sample's advantage (adaptive, based on miss rate EMA)
- **Correct move missed**: generates synthetic samples with the correct move and a fixed high advantage

### OPR (Off-Policy Rollout)

Starts at `OPR_START_UPDATE`. For lost games, with probability `OPR_TRIGGER_PROB`:
1. Find steps where policy entropy < `entropy_schedule * OPR_ENTROPY_TH_MULTIPLIER` and far from terminal
2. Weighted-sample one such step (weight = threshold − entropy)
3. Try `OPR_NUM_ACTIONS` local alternative actions, each with `OPR_NUM_ROLLOUTS` rollouts
4. Also rollout the original action for fair comparison
5. If best alternative beats original by ≥ `OPR_WIN_MARGIN`, add as synthetic training sample

### Imitation Learning

From `IMITATION_START_UPDATE`: opponent moves that led to opponent winning are added as training samples. Weight = `(1 - win_rate) * (IMITATION_MAX_WEIGHT - IMITATION_MIN_WEIGHT) + IMITATION_MIN_WEIGHT`.

## Gradient Probe (`training.py`)

Every `PROBE_INTERVAL` updates, computes 5 separate gradient vectors (policy_real, policy_synthetic, value_real, entropy_real, entropy_synthetic) by running 5 backward passes. Saves full vectors to `.npz` files (`gradient_probe_NNNNNN.npz`) containing param names, offsets, and all gradient vectors for post-hoc analysis (e.g., per-layer cosine similarity).
