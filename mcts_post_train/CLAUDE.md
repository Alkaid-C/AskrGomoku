# mcts_post_train/ — MCTS-Guided Distillation Pipeline

Pure self-play MCTS distillation: the current model plays both sides, every position becomes a training sample.

## File Responsibilities

- `main.py` — Entry point. Training loop, temperature calibration, state save/load, CLI.
- `mcts.py` — PUCT tree search with batched neural network leaf evaluation.
- `self_play.py` — MCTS self-play (current model plays both sides); every ply is recorded as training data. Also exposes `compute_block_rates` for the block-win-in-1 hit-rate diagnostic.
- `training.py` — Supervised distillation loss (CE + MSE), 8-fold distribution augmentation.
- `csv_logger.py` — CSV logging for training metrics.
- `gomoku.py` → symlink to `main/gomoku.py`
- `model.py` → symlink to `main/model.py`
- `enhancement.py` → symlink to `main/enhancement.py` (only `find_blocking_moves` is used, for the block-rate diagnostic)

## MCTS Search (`mcts.py`)

PUCT selection: `Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)`.

### Value sign convention

- The model outputs value from the **side-to-move's perspective**.
- Each node stores Q (= `W/N`) from the **parent's perspective** (how good this action was for the parent).
- `backup(leaf, value)` negates at every level: first negation converts side-to-move → parent's perspective, subsequent negations alternate correctly up the tree.
- Terminal nodes: the player who just moved won, so `terminal_value = +1` from parent's perspective.
- Root Q is meaningless (no parent). Root gets `visit_count = 1` as a virtual visit so PUCT uses priors on the first selection. The extracted root value is the visit-weighted mean Q of children (= side-to-move's perspective, suitable as training value target).

### Batched search

For N concurrent positions, each simulation:
1. PUCT-select to a leaf in each tree
2. Terminal leaves → backup cached value, skip eval
3. Non-terminal leaves → clone board, replay action path, collect observations
4. Single batched forward pass on all non-terminal leaves
5. Expand (create children with masked softmax priors), backup values

Dirichlet noise (α, ε) is added to root priors only.

## Temperature Mechanism (`main.py`, `training.py`)

MCTS visit distributions are structurally different from raw policy outputs — flatter, differently shaped. Direct distillation would corrupt the model. The temperature T mediates this:

- **Before search**: prior logits are divided by T (softens the prior, encouraging broader exploration)
- **After search**: visit distribution is raised to exponent T^0.99 (sharpens targets back toward the model's natural entropy)
- **Calibration**: T is updated each step via EMA (α = 1/`TEMP_EMA_WINDOW`) of `H_mcts / H_model`
- **Convergence**: the 0.99 exponent means each update nudges the model slightly toward the MCTS distribution rather than matching it exactly; T drifts toward 1.0 as the model adapts

Initial T = `INITIAL_TEMPERATURE` (empirically determined).

## Training (`training.py`)

### Loss

```
loss = policy_loss + VALUE_LOSS_COEFF * value_loss
```

- **Policy loss**: cross-entropy against sharpened MCTS visit distribution. `-(sharpened * log_softmax(logits)).sum(dim=-1).mean()`
- **Value loss**: MSE against MCTS root Q-value.

No entropy bonus, no GAE, no tactical boost, no imitation, no OPR. The search itself provides exploration.

### Distribution augmentation

The RL pipeline's `augment_batch_8fold` transforms single action indices via coordinate math. For full [225] distributions, `training.py` precomputes 8 inverse permutation tables from the same coordinate transforms (`new_rows`/`new_cols` from `enhancement.py`). Each permutation maps `new_dist[new_idx] = old_dist[inv_perm[new_idx]]`.

Gradient accumulation across micro-batches of `TRAIN_BATCH_SIZE`, then `clip_grad_norm_` + `optimizer.step()`.

## Training Loop (`main.py`)

Each update:
1. Sample opening IDs (`SEED_PROBABILITY` fraction get a Renju opening; rest start empty)
2. `play_mcts_games`: the current model plays both sides in `EPISODES_PER_UPDATE` games. All active games batch into a single `mcts_search_batched` call per ply; every recorded position (obs, visit_dist, root_Q) becomes a training sample
3. Compute sharpen exponent = T^`TEMP_CONVERGENCE_EXPONENT`
4. `train_on_mcts_batch`: augment 8-fold, sharpen targets, CE + MSE loss
5. Update T via EMA of entropy ratio
6. Every `CHECKPOINT_INTERVAL` updates: save checkpoint and persist training state

### Schedules

- **LR**: cosine annealing from `LEARNING_RATE` to `MIN_LR` over `TOTAL_UPDATES`.
- **Temperature**: self-calibrating via EMA (no schedule).
- **Checkpoint interval**: fixed at `CHECKPOINT_INTERVAL`.
